import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer

from utils.generate import generate
from utils.grader import math_equal
from utils.parser import extract_answer
from data.math import extract_answer_gsm8k, collate_fn_gsm8k


def evaluate(args):
    # --- Initialize distributed ---
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = f'cuda:{rank % torch.cuda.device_count()}'
    torch.cuda.set_device(device)

    # --- Load model & tokenizer ---
    print(f"[Rank {rank}] Loading model from {args.ckpt_path}...")
    model = AutoModel.from_pretrained(
        args.ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().requires_grad_(False).to(device)

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')
    tokenizer.pad_token_id = 126081

    # --- Load GSM8K test set ---
    ds = load_dataset("gsm8k", "main", split="test").with_format('torch')
    sampler = DistributedSampler(ds, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn_gsm8k, sampler=sampler)

    # --- Evaluate ---
    counts = torch.tensor([0, 0], device=device)  # [correct, total]
    pbar = tqdm(dataloader, disable=rank != 0)

    for batch in pbar:
        # Tokenize
        prompts = [[{"role": "user", "content": p}] for p in batch['problems']]
        prompts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

        # Generate
        generated_ids = generate(
            model=model,
            prompt=prompt_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
        )
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Judge
        for ans, res in zip(batch['answers'], responses):
            counts[1] += 1
            if math_equal(extract_answer_gsm8k(ans), extract_answer(res)):
                counts[0] += 1

        # Gather & display
        gathered = [counts.clone() for _ in range(world_size)] if rank == 0 else None
        dist.gather(counts, gathered, dst=0)
        if rank == 0:
            total = torch.stack(gathered).sum(dim=0)
            pbar.set_description(f"Acc: {total[0]/total[1]*100:.2f}%")

    # --- Final result ---
    if rank == 0:
        print(f"\nGSM8K Accuracy: {total[0]}/{total[1]} = {total[0]/total[1]*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    evaluate(parser.parse_args())
