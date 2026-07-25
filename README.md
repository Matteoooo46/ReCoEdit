# ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing

<div align="center">

**ReCoEdit** improves product consistency in AI image editing through a two-stage alignment pipeline: a prompt rewriter that generates editing instructions from product data, and RL-based consistency alignment (Flow-GRPO) that optimizes the diffusion model to faithfully preserve product identity during editing.

[📄 Paper (coming soon)]() | [🌐 Blog](https://matteoooo46.github.io/ReCoEdit) | [🤗 Models]()

</div>

---

## Overview

E-commerce product image editing requires two capabilities: (1) generating accurate editing prompts from raw product metadata, and (2) preserving product identity (color, shape, logos) after editing. ReCoEdit addresses both:

![Method Overview](assets/method_overview.png)

1. **Prompt Rewriter** (Qwen3-VL-8B + LoRA): An SFT-trained VLM that takes a product reference image and raw prompt, then outputs a structured editing prompt (120-150 chars, Chinese) optimized for Qwen-Image-Edit.

2. **Consistency Alignment** (Flow-GRPO + Product Reward): Online RL training of Qwen-Image-Edit-2511 using GRPO (Group Relative Policy Optimization). A custom *product consistency reward* — a Qwen3-VL-30B judge with chain-of-thought scoring (1-5 scale) — compares generated images against reference product images to guide the policy.

3. **APG Inference**: At inference time, Adaptive Projected Guidance (APG) is applied to prevent over-saturation, combined with the trained rewriter for a complete editing pipeline.

---

## Key Results

| Metric | Before RL | After RL | Improvement |
|--------|-----------|----------|-------------|
| Product Consistency Reward | 0.022 | **0.480** | **21.8×** |
| Training Epochs | — | 28 | Converged at epoch 25 |

- Base model: Qwen-Image-Edit-2511 (flow-matching diffusion)
- RL framework: Flow-GRPO with LoRA (rank=64)
- Reward model: Qwen3-VL-30B-A3B-Instruct (MoE, 3B active)
- Training: 8× GPU, bf16, FSDP with optimizer offload

See [RL Training Report](flow_grpo/docs/RL_training_report.md) for full details.

---

## Architecture

```
Product Data ──→ Rewriter (Qwen3-VL-8B) ──→ Structured Prompt
                                                  │
                                                  ▼
                      Qwen-Image-Edit-2511 ──→ Edited Image ──→ VLM Judge (Qwen3-VL-30B)
                            ▲                                         │
                            │                                         ▼
                            └──── GRPO Update ◄──── Product Consistency Reward (1-5)
```

---

## Directory Structure

```
ReCoEdit/
├── flow_grpo/              # RL training framework (Flow-GRPO)
│   ├── flow_grpo/          # Core package: rewards, prompts, diffusers patches
│   ├── config/             # Training configs (GRPO, product consistency)
│   ├── scripts/            # Training entry points & launchers
│   └── docs/               # Training reports
├── rewriter/               # Prompt rewriter SFT configs (LLaMA-Factory)
├── inference/              # Inference scripts (APG, rewriter, batch generation)
├── scripts/                # Shell launchers for training & inference
├── reward-server/          # GenEval reward server
└── assets/                 # Figures, diagrams
```

---

## Installation

```bash
# Clone
git clone https://github.com/Matteoooo46/ReCoEdit.git
cd ReCoEdit

# Install Flow-GRPO package
cd flow_grpo && pip install -e . && cd ..

# Install other dependencies
pip install torch>=2.6 transformers>=4.40 diffusers>=0.33
pip install peft accelerate datasets ml_collections
pip install openai httpx Pillow  # for VLM reward calls
```

### Model Weights

| Model | HuggingFace Path |
|-------|-----------------|
| Base diffusion model | `Qwen/Qwen-Image-Edit-2511` |
| Rewriter base | `Qwen/Qwen3-VL-8B-Instruct` |
| Reward judge (VLM) | `Qwen/Qwen3-VL-30B-A3B-Instruct` |

Download models to a local `models/` directory before training.

---

## Usage

### 1. SFT Training (Prompt Rewriter)

```bash
# Using LLaMA-Factory with config in rewriter/
llamafactory-cli train rewriter/qwen3-vl-8b-sft.yaml
```

### 2. SFT Training (Qwen-Image-Edit)

```bash
# Full fine-tuning
bash Qwen-Image-Edit-2511-full.sh

# LoRA fine-tuning
bash scripts/Qwen-Image-Edit-2511.sh
```

### 3. RL Training (Consistency Alignment)

```bash
# Start VLM reward server first
bash start_vllm_reward.sh

# Launch GRPO training with product consistency reward
bash scripts/start_training_product_consistency.sh
```

### 4. Inference

```bash
# APG + Rewriter inference (recommended)
python inference/qwen_image_edit_2511_inference_apg_rewriter.py

# Batch generation on 8 GPUs
bash scripts/run_new_rl_inference_8gpu.sh
```

---

## Key Design Decisions

### Product Consistency Reward

The reward uses a CoT (Chain-of-Thought) scoring prompt design:
- Step 1: Describe key features of the reference product (color, style, pattern, logo)
- Step 2: Check if the product appears in the generated image
- Step 3: Compare each key feature between reference and generated
- Step 4: Conclude with a 1-5 score

Score is normalized to [0, 1] via `(score - 1) / 4`. Pure scoring mode is used (no classification/inversion), as verified by ablation experiments.

### GRPO Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `noise_level` | 1.2 | Higher exploration noise for SDE sampling |
| `clip_range` | 1e-3 | Conservative clipping for stable training |
| `beta` | 0 | No KL penalty (empirically better) |
| `k` | 16 | 16 images per prompt for group advantage |
| LoRA rank | 64 | Balance between capacity and efficiency |

---

## Citation

```bibtex
@misc{recoedit2025,
  title={ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/Matteoooo46/ReCoEdit}},
}
```

## Acknowledgments

This project builds on top of:
- [Flow-GRPO](https://github.com/yifan123/flow_grpo) — Flow matching RL training framework
- [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) — Base image editing model
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language model family
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT training framework
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Diffusion model training

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
