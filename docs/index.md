---
layout: default
title: ReCoEdit
---

# ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing

**TL;DR**: We built a two-stage pipeline that makes AI image editing models faithfully preserve product identity. First, a prompt rewriter generates structured editing instructions from raw e-commerce data. Then, RL training with a VLM judge teaches the model what "product consistency" means. The result: 21.8× improvement in product consistency scores.

---

## Motivation

E-commerce platforms need to generate high-quality product advertisement images at scale — placing products into lifestyle scenes, changing backgrounds, or adding human models. Modern image editing models like Qwen-Image-Edit-2511 can do this, but they suffer from a critical problem: **the product often changes during editing**. Colors shift, logos disappear, shapes distort.

We identified two root causes:
1. **Prompt quality**: Raw e-commerce prompts are unstructured and miss key product details
2. **No consistency signal**: The base model was never explicitly trained to preserve product identity

ReCoEdit addresses both.

---

## Method

### Phase 1: Prompt Rewriter

The first challenge is generating good editing prompts. Raw product data (titles, descriptions) doesn't directly translate to effective image editing instructions. We need prompts that:
- Describe the desired scene clearly
- Specify what to keep from the original product
- Follow Qwen-Image-Edit's expected format (Chinese, 120-150 characters)

**Our approach**: Fine-tune Qwen3-VL-8B-Instruct with LoRA (rank=64) on paired data of (product image, raw prompt → structured editing prompt). Training uses LLaMA-Factory with the config in `rewriter/qwen3-vl-8b-sft.yaml`.

```
Input:  Product reference image + "把这个产品放到户外场景" (Put this product in an outdoor scene)
Output: "将图中的[产品名]放置于自然光充足的户外花园场景中，产品保持原有颜色和材质，
        背景为绿色植被和木质桌面，产品居中摆放，光线柔和，保持产品logo清晰可见..."
```

### Phase 2: SFT Training of the Editing Model

Before RL, we supervised fine-tune Qwen-Image-Edit-2511 on product editing pairs:
- Input: (source image, editing prompt) pairs from e-commerce data
- Training: Full-parameter and LoRA variants using DiffSynth-Studio
- Dataset: Multi-version curated metadata with captions (`metadata_all_products_expanded_ultimate_with_ultraedit500k_captioned_final.json`)

The SFT model learns basic editing capabilities but still lacks strong product consistency — that's where RL comes in.

### Phase 3: Consistency Alignment via RL (GRPO)

This is the core innovation. We apply **Group Relative Policy Optimization (GRPO)** from the Flow-GRPO framework to directly optimize for product consistency.

#### How GRPO works for image editing

GRPO is an online RL algorithm for diffusion models:
1. **Sample**: For each prompt, generate k=16 images from the current policy
2. **Score**: Each image gets a reward (product consistency score from VLM judge)
3. **Compare**: Compute advantage = (score - group_mean) / group_std
4. **Update**: Policy gradient step — push up high-reward generations, push down low-reward ones

```
For each training step:
  prompt → generate 16 images → VLM judge scores each → compute advantages → update LoRA weights
```

#### Product Consistency Reward

The key to successful RL is the reward function. We designed a **VLM-as-judge** approach:

- **Model**: Qwen3-VL-30B-A3B-Instruct (MoE, 3B active params), served via vLLM
- **Scoring**: Chain-of-thought prompt in Chinese, 1-5 scale
  - Step 1: Describe reference product features (color, style, pattern, logo)
  - Step 2: Check if the product appears in the generated image
  - Step 3: Compare each feature point-by-point
  - Step 4: Output `Score: X` where X ∈ {1, 2, 3, 4, 5}
- **Normalization**: `(score - 1) / 4` → reward in [0, 1]

We experimented with a classification+inversion approach (detect if product *should* appear, invert score if not) but found **pure 1-5 scoring works best** — simpler and more stable.

#### RL Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | Qwen-Image-Edit-2511 (SFT checkpoint) | Starting from a capable editor |
| LoRA rank / alpha | 64 / 128 | Trainable but efficient (8 GPUs) |
| noise_level | 1.2 | Higher exploration in SDE sampling |
| clip_range | 1e-3 | Prevent destructive updates |
| beta (KL penalty) | 0 | Empirically better without KL constraint |
| k (samples per prompt) | 16 | More comparisons = better advantage estimation |
| Batch size | 4 per GPU × 8 GPUs | Fit within memory with FSDP |
| Mixed precision | bf16 | Memory efficient |
| FSDP + activation ckpt + optimizer offload | On | Required for 8-GPU training |

---

## Results

### Training Dynamics

The reward curve shows a clear learning signal:

| Epoch | Reward | Notes |
|-------|--------|-------|
| 0 | 0.022 | Cold start (SFT weights) |
| 5 | **0.169** | First breakthrough |
| 10 | 0.318 | Steady improvement |
| 15 | 0.434 | Approaching convergence |
| 20 | 0.470 | Near peak |
| **25** | **0.480** | 🏆 Peak performance |
| 28 | 0.450 | Plateau, training stopped |

**21.8× improvement** from the initial SFT checkpoint, converging to a stable reward range of 0.42-0.48.

### Qualitative Results

*(Insert comparison images here: SFT-only vs. ReCoEdit on the same prompts)*

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
  <!-- Replace with actual result images -->
  <div style="flex: 1; min-width: 200px;">
    <p><strong>Input</strong></p>
    <img src="../assets/demo_input.png" alt="Input" style="width:100%">
  </div>
  <div style="flex: 1; min-width: 200px;">
    <p><strong>SFT Only</strong></p>
    <img src="../assets/demo_sft.png" alt="SFT" style="width:100%">
  </div>
  <div style="flex: 1; min-width: 200px;">
    <p><strong>ReCoEdit (Ours)</strong></p>
    <img src="../assets/demo_rl.png" alt="ReCoEdit" style="width:100%">
  </div>
</div>

### Key Findings

1. **VLM judges work for product consistency**: A 30B MoE VLM with CoT prompting provides reliable 1-5 product consistency scores that serve as effective RL rewards
2. **GRPO is sample-efficient**: With k=16 and 8 GPUs, meaningful improvements appear within 5 epochs
3. **Pure scoring beats classification+inversion**: Simpler reward design leads to more stable training
4. **No KL penalty needed**: Setting beta=0 didn't cause reward hacking — the LoRA constraint + clipping was sufficient
5. **Prompt rewriter matters**: Better prompts lead to better edits before RL; the two components are complementary

---

## Inference Pipeline

The final inference pipeline combines all components:

```
Product Metadata ──→ Rewriter (Qwen3-VL-8B) ──→ Structured Prompt
                                                      │
                                                      ▼
Reference Image ──→ Qwen-Image-Edit-2511 ──→ Edited Image
                         │                      (+ APG guidance)
                         │
                    RL LoRA weights
                (consistency aligned)
```

**APG (Adaptive Projected Guidance)** is applied at inference to prevent the oversaturation/artifacts that can occur with high CFG scales.

---

## Repository Structure

```
ReCoEdit/
├── README.md                    # Quick overview
├── docs/                        # This blog page (GitHub Pages)
├── assets/                      # Figures and demo images
├── flow_grpo/                   # RL training (GRPO)
│   ├── flow_grpo/rewards.py     # Product consistency reward implementation
│   ├── config/grpo.py           # RL training configurations
│   ├── scripts/train_qwenimage_edit.py  # Main training script
│   └── docs/RL_training_report.md       # Detailed training log
├── rewriter/                    # Rewriter SFT configs
├── inference/                   # All inference scripts
└── scripts/                     # Shell launchers
```

---

## Getting Started

### Prerequisites

- 8× GPU with 40GB+ VRAM (A100/H200 recommended)
- vLLM server for reward model (Qwen3-VL-30B)
- Python 3.10+, PyTorch 2.6+

### Quick Start

```bash
git clone https://github.com/Matteoooo46/ReCoEdit.git
cd ReCoEdit

# Install
cd flow_grpo && pip install -e . && cd ..

# Download models
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir models/Qwen3-VL-30B-A3B-Instruct

# 1. Train rewriter
llamafactory-cli train rewriter/qwen3-vl-8b-sft.yaml

# 2. SFT train editing model
bash Qwen-Image-Edit-2511-full.sh

# 3. Start reward server
bash start_vllm_reward.sh

# 4. RL training
bash scripts/start_training_product_consistency.sh

# 5. Inference with trained model
python inference/qwen_image_edit_2511_inference_apg_rewriter.py
```

---

## Citation

```bibtex
@misc{recoedit2025,
  title={ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing},
  author={},
  year={2025},
  howpublished={\url{https://github.com/Matteoooo46/ReCoEdit}},
}
```

## License

MIT License.

---

*Last updated: July 2025*
