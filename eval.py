import argparse
import ast
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer

from utils.generate import generate
from utils.grader import math_equal
from utils.parser import extract_answer
from data.math import extract_answer_gsm8k, collate_fn_gsm8k, collate_fn_math


# ---- Initialization ----

def init_dist():
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def save_log(args, metrics):
    out_dir = os.path.join(
        args.output,
        f"{args.task}-len{args.gen_length}-blk{args.block_length}-step{args.steps}",
    )
    os.makedirs(out_dir, exist_ok=True)
    log = {
        "task": args.task, "ckpt_path": args.ckpt_path,
        "steps": args.steps, "gen_length": args.gen_length,
        "block_length": args.block_length, "seed": args.seed,
        **metrics,
    }
    with open(os.path.join(out_dir, "evaluation_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {out_dir}/evaluation_log.json")


# ---- Math evaluation (GSM8K / MATH500) ----

def eval_math(model, tokenizer, device, args):
    if args.task == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test").with_format("torch")
        collate_fn = collate_fn_gsm8k
    else:
        ds = load_dataset("ankner/math-500", split="test").with_format("torch")
        collate_fn = collate_fn_math

    sampler = DistributedSampler(
        ds, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=False,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=1, pin_memory=True, sampler=sampler,
    )

    counts = torch.tensor([0, 0], device=device)
    pbar = tqdm(dl, disable=dist.get_rank() != 0)

    for batch in pbar:
        msgs = [[{"role": "user", "content": p}] for p in batch["problems"]]
        prompts = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(
            prompts, return_tensors="pt", padding=True,
        )["input_ids"].to(device)

        gen_ids = generate(
            model=model, prompt=prompt_ids, steps=args.steps,
            gen_length=args.gen_length, block_length=args.block_length,
        )
        responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for ans, res in zip(batch["answers"], responses):
            counts[1] += 1
            if args.task == "gsm8k":
                correct = math_equal(extract_answer_gsm8k(ans), extract_answer(res))
            else:
                correct = math_equal(
                    extract_answer(ans), extract_answer(res), timeout=True,
                )
            if correct:
                counts[0] += 1

        if dist.get_rank() == 0:
            acc = counts[0] / max(counts[1], 1)
            pbar.set_description(f"acc: {acc.item() * 100:.2f}%")

    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        acc = (counts[0] / counts[1]).item()
        print(f"\n{args.task} Accuracy: {counts[0].item()}/{counts[1].item()} = {acc * 100:.2f}%")
        save_log(args, {
            "accuracy": acc,
            "correct": int(counts[0].item()),
            "total": int(counts[1].item()),
        })


# ---- Code evaluation (HumanEval / MBPP) ----

def _format_humaneval(problems):
    formatted = {}
    for tid, p in problems.items():
        formatted[tid] = dict(p)
        formatted[tid]["prompt"] = (
            "You are an expert Python programmer. Your task is to complete the "
            f"implementation of a function named `{p['entry_point']}`.\n\n"
            f"Here is the function to complete:\n```python\n{p['prompt'].rstrip()}\n```\n"
        )
    return formatted


def _format_mbpp_prompt(ex):
    func_name = ex["test_list"][0].split(" ")[1].split("(")[0]
    tests_str = "\n".join(f"    {t}" for t in ex["test_list"])
    try:
        tree = ast.parse(ex["test_list"][0].strip())
        n_args = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == func_name:
                n_args = len(node.args)
                break
    except Exception:
        n_args = 2
    params = ", ".join(f"input_param_{i + 1}" for i in range(n_args))
    return (
        "You are an expert Python programmer. Your task is to complete the "
        f"implementation of a function named `{func_name}`.\n\n"
        f"** TARGET FUNCTION **\n{ex['text']}\n\n"
        "** UNIT TESTS **\n"
        f"Your code should pass unit tests like:\n{tests_str}\n\n"
        "Here is the function to complete:\n"
        f"```python\ndef {func_name}({params}):\n"
        f"    \"\"\"{ex['text']}\n    \"\"\"\n```\n"
    )


def _load_code_tasks(task):
    from utils.code_exec import read_problems, stream_jsonl
    if task == "humaneval":
        return _format_humaneval(read_problems("datasets/HumanEval.jsonl.gz"))
    examples = list(stream_jsonl("datasets/mbpp.jsonl"))
    return {
        ex["task_id"]: {"task_id": ex["task_id"], "prompt": _format_mbpp_prompt(ex)}
        for ex in examples[10:510]
    }


def eval_code(model, tokenizer, device, args):
    from utils.code_exec import write_jsonl, evaluate_functional_correctness

    problems = _load_code_tasks(args.task)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    task_ids = sorted(problems.keys())
    while len(task_ids) % world_size != 0:
        task_ids.append("[PAD]")
    task_ids = task_ids[rank::world_size]

    out_dir = os.path.join(
        args.output,
        f"{args.task}-len{args.gen_length}-blk{args.block_length}-step{args.steps}",
    )
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for tid in tqdm(task_ids, disable=rank != 0):
        prompt = (
            "Please write a valid Python function."
            if tid == "[PAD]"
            else problems[tid]["prompt"]
        )
        msgs = [[{"role": "user", "content": prompt}]]
        prompt_text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

        gen_ids = generate(
            model=model, prompt=prompt_ids, steps=args.steps,
            gen_length=args.gen_length, block_length=args.block_length,
            temperature=0.0,
        )
        completion = tokenizer.batch_decode(
            gen_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True,
        )[0]

        if tid == "[PAD]":
            continue
        row = {"task_id": tid, "completion": completion}
        if args.task == "mbpp":
            row["prompt"] = prompt
        samples.append(row)

    dist.barrier(device_ids=[torch.cuda.current_device()])
    gathered = [None] * world_size
    dist.all_gather_object(gathered, samples)

    if rank == 0:
        merged = [s for part in gathered for s in part]
        merged_path = os.path.join(out_dir, f"{args.task}_samples_merged.jsonl")
        write_jsonl(merged_path, merged)

        problem_file = (
            "datasets/HumanEval.jsonl.gz"
            if args.task == "humaneval"
            else "datasets/mbpp_test.jsonl"
        )
        result = evaluate_functional_correctness(
            input_file=merged_path, problem_file=problem_file,
            is_mbpp=(args.task == "mbpp"), n_workers=8, timeout=3.0, k=(1,),
        )
        print(f"\n{args.task}: {result}")
        save_log(args, {"metrics": result, "pass@1": result.get("pass@1")})


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="JustGRPO Evaluation")
    parser.add_argument("--task", type=str, default="gsm8k",
                        choices=["gsm8k", "math500", "humaneval", "mbpp"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="EvalResult")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=113)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("JustGRPO evaluation requires --batch_size 1.")

    torch.manual_seed(args.seed)
    device = init_dist()

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = AutoModel.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False).to(device)

    if args.task in ("gsm8k", "math500"):
        eval_math(model, tokenizer, device, args)
    else:
        eval_code(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
