<div align="center">

# JustGRPO

**The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models**

<p align="center">
    <a href="https://nzl-thu.github.io/">Zanlin Ni<sup>1</sup></a> &emsp;
    <a href="https://scholar.google.com/citations?user=Xgt7njgAAAAJ&hl=zh-CN">Shenzhi Wang<sup>1</sup></a> &emsp;
    <a href="https://yueyang130.github.io/">Yang Yue<sup>1</sup></a> &emsp;
    <a href="https://scholar.google.com/citations?user=e-FRHr4AAAAJ&hl=zh-TW">Tianyu Yu<sup>2</sup></a> &emsp;
    <a href="https://brawny-college-5b2.notion.site/Weilin-Zhao-11d20b7deb8280388213d5f5ed072992">Weilin Zhao<sup>2</sup></a> &emsp;
    <a href="https://dblp.uni-trier.de/pid/402/2123.html">Yeguo Hua<sup>3</sup></a> &emsp;
</p>
<p align="center">
    Tianyi Chen<sup>3</sup> &emsp;
    Jun Song<sup>4</sup> &emsp;
    Cheng Yu<sup>4</sup> &emsp;
    Bo Zheng<sup>4</sup> &emsp;
    <a href="https://gaohuang-net.github.io/">Gao Huang<sup>1✉</sup></a>
</p>

<p align="center">
    <sup>1</sup>LeapLab, Tsinghua University &emsp;
    <sup>2</sup>NLPLab, Tsinghua University &emsp;
    <sup>3</sup>Tsinghua University &emsp;
    <sup>4</sup>Alibaba Group
</p>

[![Project](https://img.shields.io/badge/🌐%20Project-Page-green)](https://nzl-thu.github.io/the-flexibility-trap/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.15165-b31b1b.svg)](https://arxiv.org/abs/2601.15165)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/🤗%20Model-JustGRPO-yellow)](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO)

*No combinatorial trajectories. No ELBO approximations. No diffusion-specific adaptations.*

**Just GRPO.**

</div>

## 📢 News


- **[2026.03]** 🎉 Training code, evaluation scripts, and model checkpoints for MATH-500, HumanEval and MBPP datasets released!
- **[2026.01]** 📄 Paper available on [arXiv](https://arxiv.org/abs/2601.15165)!
- **[2026.01]** 🎉 Training code, evaluation scripts, and [model checkpoint](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO) on GSM8K released!

## Why JustGRPO?

Diffusion LLMs (dLLMs) can generate tokens in **arbitrary order**, which theoretically offers more flexibility than standard left-to-right generation. But does this flexibility actually unlocks unique reasoning capabilities inaccessible to standard AR models?

<div align="center">
  <img src="assets/mechanism_to_passk.png" width="90%" alt="Mechanism to Pass@k"/>
</div>

**We found the opposite.** Arbitrary-order generation allows models to *bypass* high-uncertainty tokens (e.g., "Therefore", "Since") — the very tokens that create branching points in reasoning. This premature bypass collapses the solution space, leading to *lower* reasoning potential (Pass@k).

**Our solution is simple:** Since AR order preserves better reasoning potential, we just train dLLMs with standard GRPO in AR mode. No bells and whistles.


## Results

JustGRPO achieves state-of-the-art performance across reasoning and coding benchmarks:

<div align="center">
  <img src="assets/acc_compare.png" width="90%" alt="Accuracy Comparison"/>
</div>

| Benchmark | Gen Length 128 | Gen Length 256 | Gen Length 512 |
|:---:|:---:|:---:|:---:|
| **GSM8K** | 83.8 | 89.1 | 89.8 |
| **MATH-500** | 39.0 | 45.1 | 45.2 |
| **HumanEval** | 37.8 | 49.4 | 48.7 |
| **MBPP** | 50.6 | 52.4 | 49.0 |


## Simplicity

Existing RL methods for dLLMs often require handling the complexity of arbitrary-order generation:

| Challenge | Description |
|:---|:---|
| Combinatorial trajectories | Optimizing over factorial-sized denoising paths |
| Intractable likelihoods | ELBO-based surrogates instead of true objectives |
| Sampler-learner mismatch | Confidence-based samplers vs. original diffusion prior |

- **JustGRPO sidesteps all of this** by treating dLLMs as autoregressive models during RL training. The result? Standard GRPO, directly applicable, with exact likelihood computation.
- **The core logic of JustGRPO (`grpo.py`) fits in ~60 lines**: rollout sampling and log-probability loss computation. That's it.

> 💡 The model still retains **parallel decoding** at inference time — we only use AR order during training. See our paper for more details.



## Installation

JustGRPO is designed to be lightweight and dependency-minimal.

```bash
git clone https://github.com/LeapLabTHU/JustGRPO.git
cd JustGRPO
pip install -r requirements.txt
```

**Dependencies:**
- `accelerate`
- `transformers`
- `datasets`
- Standard evaluation utilities (`sympy`, `latex2sympy2`, etc.)


## Usage

We provide evaluation and training code for **GSM8K**, **MATH-500**, **HumanEval**, and **MBPP**.

### Evaluation

Model checkpoints:
- [LLaDA-Instruct-JustGRPO-GSM8K](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO-GSM8K) (GSM8K)
- [LLaDA-Instruct-JustGRPO-Math500](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO-Math500) (MATH-500)
- [LLaDA-Instruct-JustGRPO-Code](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO-Code) (HumanEval & MBPP)

```bash
torchrun --nproc-per-node=8 eval.py \
  --task gsm8k \  # math500/humaneval/mbpp
  --ckpt_path /path/to/ckpt \
  --gen_length 256 --steps 256 --block_length 32
```

### Training

**Math (GSM8K / MATH-500):**

```bash
accelerate launch --num_processes 8 --config_file configs/fsdp.yaml train.py \
  --dataset gsm8k \
  --grad_accum 8
```

```bash
accelerate launch --num_processes 8 --config_file configs/fsdp.yaml train.py \
  --dataset math \
  --grad_accum 8
```

**Code (MBPP / HumanEval):**

Code training uses the **AceCode-Hard** subset, following [ml-diffucoder](https://github.com/apple/ml-diffucoder). You can download the dataset here: [AceCode-Hard (Google Drive)](https://drive.google.com/file/d/1eyxdcLRiEI0Km9ohaGxah0hj53iUWuMA/view?usp=sharing). Place the downloaded file at `datasets/acecode_hard.jsonl`.

```bash
accelerate launch --num_processes 8 --config_file configs/fsdp.yaml train.py \
  --dataset code \
  --code_data_path datasets/acecode_hard.jsonl \
  --grad_accum 8
```

> **Note:** Keep global batch size = `num_gpus` × `grad_accum` = **64**.


## Citation

If you find this work useful, please consider citing our paper.

```bibtex
@article{ni2026flexibility,
  title={The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models},
  author={Ni, Zanlin and Wang, Shenzhi and Yue, Yang and Yu, Tianyu and Zhao, Weilin and Hua, Yeguo and Chen, Tianyi and Song, Jun and Yu, Cheng and Zheng, Bo and Huang, Gao},
  journal={arXiv preprint arXiv:2601.15165},
  year={2026}
}
```

## Acknowledgments

This project builds upon the following excellent works:

- [LLaDOU](https://github.com/maple-research-lab/LLaDOU)
- [ml-diffucoder](https://github.com/apple/ml-diffucoder)
- [ESPO](https://github.com/ML-GSAI/ESPO)
- [LLaDA](https://github.com/ML-GSAI/LLaDA)
- [d1](https://github.com/dllm-reasoning/d1)

We sincerely appreciate the authors for making their work open source.
