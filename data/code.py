import re
import ast
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.sampler import InfiniteSampler
from data.local_sandbox import local_execute
from utils.distributed import get_rank, get_world_size


def extract_code(completion, language="python"):
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    return matches[0] if matches else ""


def get_code_format_reward(language="python"):
    """Return a function that scores format compliance for code responses."""
    pattern = re.compile(
        rf"^(?:(?!```)[\s\S])*?```{language}\n(?:(?!```)[\s\S])*?```\n?$",
        re.DOTALL,
    )

    def code_format_reward(responses):
        rewards = []
        for content in responses:
            if not pattern.fullmatch(content):
                rewards.append(0.0)
                continue
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue
            code = code_blocks[0].strip()
            if language == "python":
                try:
                    ast.parse(code)
                    rewards.append(1.0)
                except Exception:
                    rewards.append(0.5)
            else:
                rewards.append(1.0)
        return rewards

    return code_format_reward


def collate_fn_code(batch):
    problems, test_cases = [], []
    for item in batch:
        question = item["question"]
        tests = item["test_cases"]
        if "\n```\n" not in question and tests:
            question = question + "\nTest cases: " + tests[0]
        problems.append(question)
        test_cases.append(tests)
    return {"problems": problems, "test_cases": test_cases}


def reward_code(batch, responses, num_generations, device):
    """Format reward + code pass rate"""
    test_cases_expanded = batch["test_cases"] * num_generations
    format_rewards = get_code_format_reward()(responses)

    pairs = []
    valid_indices = []
    for i, (fmt, response) in enumerate(zip(format_rewards, responses)):
        if fmt < 1.0:
            continue
        pairs.append((extract_code(response), test_cases_expanded[i]))
        valid_indices.append(i)

    pass_rates = local_execute(pairs) if pairs else []

    rewards = torch.zeros(len(responses), device=device, dtype=torch.float32)
    for i in range(len(responses)):
        rewards[i] = format_rewards[i]
    for idx, rate in zip(valid_indices, pass_rates):
        rewards[idx] += rate
    return rewards


def load_code_dataset_and_reward(
    local_path, batch_size, num_workers=4,
    rank=None, num_replicas=None, seed=112,
):
    if local_path.endswith(".jsonl") or local_path.endswith(".json"):
        ds = load_dataset("json", data_files=local_path, split="train")
    else:
        ds = load_dataset(local_path, split="train")

    ds = ds.with_format("torch")
    ds = ds.shuffle(seed=seed)

    if rank is None:
        try: rank = get_rank()
        except Exception: rank = 0
    if num_replicas is None:
        try: num_replicas = get_world_size()
        except Exception: num_replicas = 1

    sampler = InfiniteSampler(ds, rank=rank, num_replicas=num_replicas)
    dataloader = DataLoader(
        ds, collate_fn=collate_fn_code, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
    )
    return dataloader, reward_code
