import os
import re
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

import utils.distributed as dist
from grpo import sample, logprob_loss, compute_group_advantages


@dataclass
class TrainConfig:
    """Training hyperparameters for GRPO."""
    
    # --- Model ---
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    
    # --- Training ---
    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 125
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 1234
    num_generations: int = 8
    repeat_times: int = 2
    gen_steps: int = 256
    gen_length: int = 256

    # --- Misc ---
    output_dir: str = "./checkpoints"
    log_every: int = 1
    save_every: int = 10
    resume_ckpt: Optional[str] = None


def train(config: TrainConfig):
    """
    Main GRPO training loop.
    
    Args:
        config: TrainConfig with model path, learning rate, batch size, etc.
    """
    
    # --- Initialize distributed ---
    dist.init()
    rank = dist.get_rank()
    device = torch.device('cuda')
    
    print("=" * 60)
    print("JustGRPO Training")
    print("=" * 60)

    # --- Random seeds ---
    np.random.seed((config.seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    # --- Load model ---
    print(f"Loading model from {config.model_path}...")
    from transformers import AutoTokenizer, AutoModel
    
    model = AutoModel.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval().to(device)
    
    # Activation checkpointing
    if hasattr(model, 'model') and hasattr(model.model, 'set_activation_checkpointing'):
        model.model.set_activation_checkpointing('whole_layer')
    
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token_id = 126336  # LLaDA mask token
    
    # --- Load dataset (GSM8K) ---
    print("Loading GSM8K dataset...")
    from data.math import load_gsm8k_dataset_and_reward
    
    dataloader, reward_fn = load_gsm8k_dataset_and_reward(
        local_path="gsm8k",
        batch_size=config.batch_size_per_device,
        num_workers=4,
    )
    
    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    
    # --- Accelerator setup ---
    accelerator = dist.get_accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # --- Resume ---
    start_step = 0
    if config.resume_ckpt is not None:
        local_resume_path = config.resume_ckpt.split('/')[-1]
        if os.path.exists(local_resume_path):
            print(f"Resuming from {local_resume_path}")
            accelerator.load_state(local_resume_path)
        match = re.search(r'(\d+)$', config.resume_ckpt.rstrip('/'))
        if match:
            start_step = int(match.group(1))
    
    dataloader_iter = iter(dataloader)
    
    if start_step > 0:
        print(f"Skipping {start_step} batches...")
        for _ in range(start_step):
            next(dataloader_iter)
    
    # --- Output directory ---
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
    
    # --- Training loop ---
    print(f"Starting training for {config.total_steps} steps...")
    print(f"Group size: {config.num_generations * config.repeat_times}")
    print(f"Grad accumulation: {config.grad_accumulation}")
    print(f"Effective batch: {config.batch_size_per_device * dist.get_world_size() * config.grad_accumulation}")
    print(f"Learning rate: {config.learning_rate}")

    for step in range(start_step, config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        
        all_rewards = []
        
        for accum_idx in range(config.grad_accumulation):
            print(f"[Step {step+1}/{config.total_steps}] [Accum {accum_idx+1}/{config.grad_accumulation}] Sampling...")
            with dist.ddp_sync(model, sync=(accum_idx == config.grad_accumulation - 1)):
                model.eval()
                
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    # --- Rollout ---
                    batch = next(dataloader_iter)
                    inputs_chunks = []
                    
                    for _ in range(config.repeat_times):
                        inputs = sample(
                            model=accelerator.unwrap_model(model),
                            batch=batch,
                            tokenizer=tokenizer,
                            device=device,
                            reward_fn=reward_fn,
                            num_generations=config.num_generations,
                            steps=config.gen_steps,
                            gen_length=config.gen_length,
                        )
                        inputs_chunks.append(inputs)

                    # --- Compute Advantages ---
                    rewards = torch.cat([chunk['rewards'] for chunk in inputs_chunks], dim=0)
                    advantages = compute_group_advantages(rewards, config.num_generations * config.repeat_times)
                    
                    valid_samples = (advantages != 0).sum()
                    split_advantages = advantages.split(config.num_generations, dim=0)
                    for chunk, adv in zip(inputs_chunks, split_advantages):
                        chunk["advantages"] = adv
                    
                    accelerator.wait_for_everyone()

                    # --- Compute Loss ---
                    print(f"[Step {step+1}/{config.total_steps}] [Accum {accum_idx+1}/{config.grad_accumulation}] Computing loss...")
                    model.train()
                    for inputs in inputs_chunks:
                        logprob_loss(
                            model=model,
                            inputs=inputs,
                            valid_samples=valid_samples,
                            gain=1.0,
                            accelerator=accelerator,
                            gen_length=config.gen_length,
                        )
                        all_rewards.append(inputs['rewards'].detach())
                
                accelerator.wait_for_everyone()
                
                for key in list(inputs.keys()):
                    del inputs[key]

        # --- Grad Clip & Optimizer Step ---
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        
        optimizer.step()
        
        # --- Logging ---
        if (step + 1) % config.log_every == 0:
            all_rewards_tensor = torch.cat(all_rewards, dim=0)
            gathered_rewards = accelerator.gather(all_rewards_tensor)
            mean_reward = gathered_rewards.mean().item()
            print(f"[Step {step+1}/{config.total_steps}] reward={mean_reward:.4f}, grad={grad_norm:.4f}")
        
        # --- Save checkpoint ---
        if (step + 1) % config.save_every == 0:
            state_dict = accelerator.get_state_dict(model)
            save_path = os.path.join(config.output_dir, f'training-state-{step+1:06d}')
            accelerator.save_state(save_path)
            if rank == 0:
                save_path = os.path.join(config.output_dir, f'ckpt-{step+1:06d}')
                accelerator.unwrap_model(model).save_pretrained(
                    save_path, state_dict=state_dict, safe_serialization=True
                )
            print(f"Saved checkpoint to {save_path}")
        accelerator.wait_for_everyone()
    
    print("\nTraining complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="JustGRPO Training")
    
    parser.add_argument("--run_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume checkpoint path")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create config from CLI args
    config = TrainConfig(
        output_dir=args.run_dir,
        grad_accumulation=args.grad_accum,
        resume_ckpt=args.resume_ckpt,
    )

    train(config)
