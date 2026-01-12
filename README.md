<div align="center">

# JustGRPO

**The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models**

<p align="center">
    <a href="https://nzl-thu.github.io/">Zanlin Ni<sup>1</sup></a> &emsp;
    <a href="https://scholar.google.com/citations?user=Xgt7njgAAAAJ&hl=zh-CN">Shenzhi Wang<sup>1</sup></a> &emsp;
    <a href="https://yueyang130.github.io/">Yang Yue<sup>1</sup></a> &emsp;
    <a href="https://scholar.google.com/citations?user=e-FRHr4AAAAJ&hl=zh-TW">Tianyu Yu<sup>2</sup></a> &emsp;
    <a href="https://brawny-college-5b2.notion.site/Weilin-Zhao-11d20b7deb8280388213d5f5ed072992">Weilin Zhao<sup>2</sup></a> &emsp;
</p>
<p align="center">
    <a href="https://dblp.uni-trier.de/pid/402/2123.html">Yeguo Hua<sup>2</sup></a> &emsp;
    Tianyi Chen<sup>2</sup> &emsp;
    Jun Song<sup>3</sup> &emsp;
    Cheng Yu<sup>3</sup> &emsp;
    Bo Zheng<sup>3</sup> &emsp;
    <a href="https://gaohuang-net.github.io/">Gao Huang<sup>1✉</sup></a>
</p>

<p align="center">
    <sup>1</sup>LeapLab, Tsinghua University &emsp;
    <sup>2</sup>Tsinghua University &emsp;
    <sup>3</sup>Alibaba Group
</p>

[![Project](https://img.shields.io/badge/🌐%20Project-Page-green)](https://nzl-thu.github.io/the-flexibility-trap/)
[![Paper](https://img.shields.io/badge/📄%20Paper-PDF-b31b1b.svg)](https://nzl-thu.github.io/the-flexibility-trap/static/paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/🤗%20Model-JustGRPO-yellow)](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO)

*No combinatorial trajectories. No ELBO approximations. No diffusion-specific adaptations.*

**Just GRPO.**

</div>


## 📢 News

- **[2026.01]** 🎉 Training code, evaluation scripts, and [model checkpoint](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO) on GSM8K released!

## 📋 TODO

- [ ] Add support for MATH-500
- [ ] Add support for coding tasks (HumanEval, MBPP)


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

We provide training and evaluation code on **GSM8K**.
The RL-trained model is available at [Huggingface](https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO).

### Training

```bash
accelerate launch --num_processes 8 --main_process_ip localhost --config_file configs/fsdp.yaml train.py \
  --run_dir ./checkpoints \
  --grad_accum 8
```
- Note: The global batch size = num_gpus * grad_accum, keep it equal to 64.

### Evaluation

```bash
torchrun --standalone --nproc-per-node=8 eval.py \
  --ckpt_path /path/to/ckpt \
  --steps 256 \
  --gen_length 256 \
  --block_length 32
```


## Citation

If you find this work useful, please cite:

```bibtex
@misc{ni2026flexibility,
  author    = {Ni, Zanlin and Wang, Shenzhi and Yue, Yang and Yu, Tianyu and Zhao, Weilin and Hua, Yeguo and Chen, Tianyi and Song, Jun and Yu, Cheng and Zheng, Bo and Huang, Gao},
  title     = {The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models},
  year      = {2026},
  howpublished = {\url{https://nzl-thu.github.io/the-flexibility-trap}},
}
```

## Acknowledgments

This project builds upon the following excellent works:

- [LLaDA](https://github.com/ML-GSAI/LLaDA)
- [d1](https://github.com/dllm-reasoning/d1)
- [LLaDOU](https://github.com/maple-research-lab/LLaDOU)

We thank the authors for their open-source contributions to the community.
