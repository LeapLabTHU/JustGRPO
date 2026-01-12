"""
GSM8K Dataset Loader for LLaDOU Training.

Provides dataloader and reward function for GSM8K math problems.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.distributed import get_rank, get_world_size
from data.sampler import InfiniteSampler
from utils.grader import math_equal
from utils.parser import extract_answer


def collate_fn_gsm8k(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    return {"problems": problems, "answers": answers}


def extract_answer_gsm8k(answer: str):
    """Extract the final answer from GSM8K format (after ####)."""
    return answer.split('####')[-1].strip()


def reward_gsm8k(batch, responses, num_generations, device):
    """
    Compute reward for GSM8K responses.
    
    Args:
        batch: Batch containing ground truth answers
        responses: Model generated responses
        num_generations: Number of generations per problem
        device: Torch device
    
    Returns:
        Tensor of rewards (+1 for correct, -1 for incorrect)
    """
    answers = batch['answers'] * num_generations
    
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]
    
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res):
            rewards[i] = 1.0
        else:
            rewards[i] = -1.0
    
    return rewards


def load_gsm8k_dataset_and_reward(
    local_path: str = "gsm8k",
    batch_size: int = 1,
    split: str = 'train',
    num_workers: int = 4,
    seed: int = 112,
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    ds = load_dataset(local_path, "main", split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    
    sampler = InfiniteSampler(
        ds, 
        rank=get_rank(), 
        num_replicas=get_world_size(),
    )
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_gsm8k,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, reward_gsm8k
