import torch
import torch.nn.functional as F

from utils.generate import generate


@torch.no_grad()
def sample(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1., steps=256, gen_length=256):
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Rollout with AR order (block_length=1)
    generated_ids = generate(model=model, prompt=prompt_ids.repeat(num_generations, 1),
                             steps=steps, gen_length=gen_length, temperature=temperature, block_length=1)

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {
        'generated_ids': generated_ids,
        'prompt_len': prompt_ids.shape[1],
        'rewards': reward_fn(batch, responses, num_generations, device).float(),
    }


def logprob_loss(model, inputs, valid_samples, eps=0.2, gain=1.0, temperature=1., accelerator=None,
                 gen_length=256, mask_id=126336):
    advantages, generated_ids, prompt_len = inputs['advantages'], inputs['generated_ids'], inputs['prompt_len']
    batch_size, device = advantages.shape[0], generated_ids.device
    prompt_ids, completion_ids = generated_ids[:, :prompt_len], generated_ids[:, prompt_len:]

    valid_samples = accelerator.gather(valid_samples).float().mean().item()
    scale = gain / gen_length / (valid_samples + 1e-5)

    for t in range(gen_length):
        # Construct input with AR masking (Past=Observed, Future=Masked)
        x = torch.cat([prompt_ids, completion_ids[:, :t],
                       torch.full((batch_size, gen_length - t), mask_id, device=device, dtype=generated_ids.dtype)], dim=1)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            logits = model(x).logits / temperature

        # Compute log probability of next token
        log_prob = F.log_softmax(logits[:, prompt_len + t, :].float(), dim=-1)
        token_log_prob = log_prob.gather(-1, completion_ids[:, t:t+1]).squeeze(-1)

        ratio = (token_log_prob - token_log_prob.detach()).exp()
        clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        accelerator.backward(loss.mul(scale).sum())

    return {
        "reward": accelerator.gather(inputs['rewards'].detach()).mean().item(),
        "valid_samples": valid_samples,
    }


def compute_group_advantages(rewards, group_size):
    mean = rewards.view(group_size, -1).mean(dim=0).repeat(group_size)
    std = rewards.view(group_size, -1).std(dim=0).repeat(group_size)
    return (rewards - mean) / (std + 1e-4)
