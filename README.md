# ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing

<div align="center">

**ReCoEdit helps image editing models change the scene without changing the product.**

<p>
  <a href="https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo"><img src="https://img.shields.io/badge/🤗-Online%20Demo-blue"></a>
  <a href="https://matteoooo46.github.io/ReCoEdit"><img src="https://img.shields.io/badge/📄-Project%20Blog-black"></a>
  <a href="https://huggingface.co/Matteoooo46/ReCoEdit-RL"><img src="https://img.shields.io/badge/🤗-RL%20Checkpoint-yellow"></a>
  <a href="https://huggingface.co/Matteoooo46/ReCoEdit-rewriter"><img src="https://img.shields.io/badge/🤗-Rewriter%20Checkpoint-yellow"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

</div>

<!--
TODO: Add a qualitative teaser to assets/recoedit_teaser.png, then uncomment:

<p align="center">
  <img src="assets/recoedit_teaser.png" alt="ReCoEdit qualitative results" width="100%">
</p>
-->

## 📖 Abstract

Product image editing must satisfy two goals at the same time: follow the requested edit and preserve the identity of the reference product. In practice, colors shift, logos disappear, patterns change, and shapes deform when the model focuses too heavily on the new scene.

ReCoEdit addresses this problem with two learned components and an inference-time guidance method:

1. **Prompt Rewriter — Qwen3-VL-8B + LoRA.** Given one or more product reference images and a raw instruction, the rewriter produces a structured Chinese editing prompt for Qwen-Image-Edit.
2. **Consistency Alignment — Flow-GRPO + Product Reward.** Qwen-Image-Edit-2511 is optimized with online reinforcement learning. A Qwen3-VL-30B judge compares each edited image with its reference product and provides a normalized consistency reward.
3. **APG Inference.** Adaptive Projected Guidance is applied at inference time to reduce oversaturation and unstable CFG behavior.

```mermaid
flowchart LR
    A["Reference image(s) + raw instruction"] --> B["Prompt Rewriter"]
    B --> C["Structured editing prompt"]
    A --> D["Qwen-Image-Edit-2511"]
    C --> D
    D --> E["Edited image"]
    A --> F["VLM product judge"]
    E --> F
    F --> G["Product consistency reward"]
    G -->|"GRPO update"| D
```

## 📢 News

- **August 2026:** The [ReCoEdit online demo](https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo) is now available on Hugging Face Spaces.

## 🌐 Online Demo

Try the complete ReCoEdit workflow without setting up a local environment:

### [Launch the ReCoEdit Demo →](https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo)

The demo provides an accessible entry point for product-image editing with the released ReCoEdit components.

## 📊 Key Results

The internal product-consistency reward improves substantially during RL training:

| Model stage | Product consistency reward ↑ | Change |
|---|---:|---:|
| SFT initialization | 0.022 | — |
| ReCoEdit RL, peak at epoch 25 | **0.480** | **+0.458 (21.8×)** |

- Base model: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- RL framework: Flow-GRPO with LoRA rank 64
- Reward judge: [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- Training configuration: 8 GPUs, bf16, FSDP, activation checkpointing, and optimizer offload

Full epoch-by-epoch reward curve is in [**TRAIN.md**](TRAIN.md).

> [!NOTE]
> The value above is the reward used by the RL pipeline. Dataset statistics, independent benchmarks, ablations, and qualitative comparisons will be added after they are ready.

## 🔥 Quick Start

### Option 1: Use the online demo

Open the [Hugging Face Space](https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo), upload a product image, enter an editing instruction, and run the edit.

### Option 2: Run locally

#### 1. Clone and create the environment

```bash
git clone https://github.com/Matteoooo46/ReCoEdit.git
cd ReCoEdit

conda create -n recoedit python=3.10 -y
conda activate recoedit

pip install -r requirements.txt
```

The inference implementation also uses [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and an APG guidance helper from the upstream Flow-GRPO fork of the pipeline:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
pip install -e third_party/DiffSynth-Studio

# APG helper — expose whichever apg_guidance.py you use on PYTHONPATH.
```

#### 2. Run APG inference

The base model and ReCoEdit RL checkpoint are downloaded from Hugging Face on the first run and then reused from the local cache.

```bash
python inference/qwen_image_edit_2511_inference_apg.py \
    --input_image path/to/product.jpg \
    --prompt "把产品放到户外花园场景中" \
    --output outputs/result.png
```

#### 3. Run APG + Prompt Rewriter

```bash
python inference/qwen_image_edit_2511_inference_apg_rewriter.py \
    --input_image path/to/product.jpg \
    --prompt "把产品放到户外花园场景中" \
    --output outputs/result.png
```

Multiple reference images are supported:

```bash
python inference/qwen_image_edit_2511_inference_apg_rewriter.py \
    --input_image ref1.jpg ref2.jpg \
    --prompt "把产品放到户外花园场景中" \
    --output outputs/result.png
```

## 🤗 Model Zoo

| Component | Base model | Released weights | Role |
|---|---|---|---|
| Image editor | [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | [ReCoEdit-RL](https://huggingface.co/Matteoooo46/ReCoEdit-RL) | Product-consistent image editing |
| Prompt rewriter | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [ReCoEdit-rewriter](https://huggingface.co/Matteoooo46/ReCoEdit-rewriter) | Structured editing-prompt generation |
| Reward judge | [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | — | Product-consistency scoring during RL |

## 🔥 Train & Eval

- **Training** — full three-stage pipeline (rewriter SFT → edit-model SFT → consistency RL), including data format, hyperparameters, and launch commands: [**TRAIN.md**](TRAIN.md).
- **Evaluation** — evaluation protocol and benchmarks: [**EVAL.md**](EVAL.md) *(coming soon)*.

## 🙏 Acknowledgments

ReCoEdit builds on the following open-source projects and models. **We do not vendor their code into this repository** — you install them separately when training. Please cite them if you use ReCoEdit:

- [Flow-GRPO](https://github.com/yifan123/flow_grpo) — reinforcement learning for flow-matching models (used in stage 3)
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image) — base image-editing model family
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — prompt rewriter and product-consistency judge
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — prompt-rewriter SFT
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — image-model training and inference
- [vLLM](https://github.com/vllm-project/vllm) — serves the reward VLM during RL

## ✍️ Citation

If ReCoEdit is useful for your work, please cite the repository:

```bibtex
@software{recoedit2025,
  title  = {ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing},
  author = {Matteoooo46},
  year   = {2026},
  url    = {https://github.com/Matteoooo46/ReCoEdit}
}
```

## 📜 License

This project is released under the [MIT License](LICENSE).
