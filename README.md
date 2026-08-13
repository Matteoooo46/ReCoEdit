# ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing

<div align="center">

**ReCoEdit** improves product consistency in AI image editing through a two-stage alignment pipeline: a prompt rewriter that generates editing instructions from product data, and RL-based consistency alignment (Flow-GRPO) that optimizes the diffusion model to faithfully preserve product identity during editing.

[🌐 Blog](https://matteoooo46.github.io/ReCoEdit) | [🤗 RL](https://huggingface.co/Matteoooo46/ReCoEdit-RL) | [🤗 Rewriter](https://huggingface.co/Matteoooo46/ReCoEdit-rewriter)

</div>

---

##概述

E-commerce product image editing requires two capabilities: (1) generating accurate editing prompts from raw product metadata, and (2) preserving product identity (color, shape, logos) after editing. ReCoEdit addresses both:

![Method Overview]

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
├── inference/              # Two clean CLI inference scripts (APG, APG+Rewriter)
├── scripts/                # SFT launchers + rewriter LLaMA-Factory config
└── docs/                   # GitHub Pages blog
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
# Fill in the paths marked "EDIT" in the config first, then:
llamafactory-cli train scripts/rewriter_sft_qwen3vl_8b.yaml
```

### 2. SFT Training (Qwen-Image-Edit)

```bash
# Full fine-tuning (path variables are required)
export DIFFSYNTH_DIR=/path/to/DiffSynth-Studio
export TRAIN_METADATA=/path/to/metadata.json
export OUTPUT_DIR=/path/to/output
bash scripts/Qwen-Image-Edit-2511-full.sh

# LoRA fine-tuning
bash scripts/Qwen-Image-Edit-2511.sh
```

### 3. RL Training (Consistency Alignment)

```bash
# Point the reward function at your VLM server (Qwen3-VL-30B via vLLM):
export RECOEDIT_VLM_URL=http://your-vlm-server:8080/v1

# Launch GRPO training with the product-consistency reward
cd flow_grpo
torchrun --standalone --nproc_per_node=8 scripts/train_qwenimage_edit.py \
    --config config/grpo.py:counting_qwenimage_edit_8gpu_product_consistency
```

### 4. Inference

```bash
# APG only (faster, no rewriter)
python inference/qwen_image_edit_2511_inference_apg.py \
    --input_image product.jpg \
    --prompt "把产品放到户外花园场景中" \
    --output result.png

# APG + Rewriter (recommended, better product consistency)
python inference/qwen_image_edit_2511_inference_apg_rewriter.py \
    --input_image product.jpg \
    --prompt "把产品放到户外花园场景中" \
    --output result.png

# Multiple reference images
python inference/qwen_image_edit_2511_inference_apg_rewriter.py \
    --input_image ref1.jpg ref2.jpg \
    --prompt "..." \
    --output result.png
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
