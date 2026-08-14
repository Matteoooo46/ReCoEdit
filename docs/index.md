---
layout: page
title: ReCoEdit
description: Rewriter-Guided Consistency Alignment for Product Image Editing
---

<section class="project-hero">
  <p class="project-eyebrow">Product-aware image editing</p>
  <h1>ReCoEdit</h1>
  <p class="project-subtitle">Rewriter-Guided Consistency Alignment for Product Image Editing</p>
  <p class="project-kicker">Change the scene. Preserve the product.</p>
  <div class="project-actions">
    <a href="https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo">Try the Demo</a>
    <a href="https://github.com/Matteoooo46/ReCoEdit">GitHub</a>
    <a href="https://huggingface.co/Matteoooo46/ReCoEdit-RL">RL Checkpoint</a>
    <a href="https://huggingface.co/Matteoooo46/ReCoEdit-rewriter">Rewriter Checkpoint</a>
  </div>
</section>

<!--
TODO: Add a project-page hero image to docs/assets/recoedit_hero.png, then uncomment:

<p align="center">
  <img src="{{ '/assets/recoedit_hero.png' | relative_url }}" alt="ReCoEdit examples" width="100%">
</p>
-->

## Introducing ReCoEdit

Product-image editing looks simple: keep the product and change everything around it. In practice, editing models often alter the product itself. Colors drift, logos disappear, materials change, and fine structures are lost when a new scene is generated.

ReCoEdit is a practical alignment pipeline built on Qwen-Image-Edit-2511. It combines a vision-language prompt rewriter, product-aware reinforcement learning, and APG inference so that an edit can follow the requested scene while preserving the identity of the reference product.

## Try It Online

The fastest way to experience ReCoEdit is through the hosted Hugging Face Space. Upload a reference product image, describe the edit, and run the complete pipeline in your browser.

### [Launch ReCoEdit Demo →](https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo)

## What ReCoEdit Adds

### Prompt Rewriting

A Qwen3-VL-8B rewriter examines the reference product and expands a raw request into a structured editing prompt. It emphasizes the target scene while retaining the product's defining color, shape, material, pattern, logo, and text.

### Product-Consistency Alignment

ReCoEdit applies Flow-GRPO to Qwen-Image-Edit-2511. During training, the policy generates a group of candidate edits. A Qwen3-VL-30B judge compares every candidate with the reference product and assigns a consistency score. Group-relative advantages then update the editor toward images that preserve product identity more faithfully.

### APG Inference

Adaptive Projected Guidance helps control oversaturation and unstable classifier-free guidance behavior. At inference time, APG can be used by itself or together with the learned prompt rewriter.

## How It Works

```text
Reference image(s) + raw instruction
                  │
                  ▼
       Qwen3-VL Prompt Rewriter
                  │
                  ▼
      Structured editing prompt
                  │
                  ▼
      Qwen-Image-Edit-2511 + APG ──────────► Edited image
                  ▲                              │
                  │                              ▼
            GRPO update ◄──────── Product consistency judge
```

The prompt rewriter improves the instruction before editing. The RL stage improves the editing policy itself. APG is then applied during inference as an additional guidance mechanism.

## Training Signal

The product-consistency reward rises from the SFT initialization to its peak during GRPO training:

| Model stage | Product consistency reward ↑ |
|---|---:|
| SFT initialization | 0.022 |
| ReCoEdit RL, epoch 25 | **0.480** |

This is an absolute gain of **0.458** on the internal normalized reward, corresponding to a **21.8×** increase over the initialization value.

<div class="project-note">
The value above is the reward optimized by the RL pipeline. Dataset statistics, independent benchmarks, ablations, and sample counts will be added when the evaluation release is ready.
</div>

## Qualitative Showcase

<!--
TODO: Add an interactive or side-by-side gallery here.
Suggested groups: Input / SFT / ReCoEdit for scene replacement, logo preservation,
material preservation, and multiple-reference editing.
-->

## Released Components

| Component | Link | Purpose |
|---|---|---|
| Online demo | [ReCoEdit-Demo](https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo) | Run ReCoEdit in the browser |
| RL checkpoint | [ReCoEdit-RL](https://huggingface.co/Matteoooo46/ReCoEdit-RL) | Consistency-aligned image editor |
| Rewriter checkpoint | [ReCoEdit-rewriter](https://huggingface.co/Matteoooo46/ReCoEdit-rewriter) | Product-aware prompt rewriting |
| Source code | [GitHub](https://github.com/Matteoooo46/ReCoEdit) | Training and inference implementation |

## Quick Start

```bash
git clone https://github.com/Matteoooo46/ReCoEdit.git
cd ReCoEdit

conda create -n recoedit python=3.10 -y
conda activate recoedit

pip install -r requirements.txt
pip install -e ./flow_grpo
```

For the shortest path, use the [online demo](https://huggingface.co/spaces/Matteoooo46/ReCoEdit-Demo). For local inference and training instructions, see the [repository README](https://github.com/Matteoooo46/ReCoEdit#quick-start).

## Data and Evaluation

<!-- TODO: Add dataset access, data format, licenses, evaluation protocol, and benchmark results. -->

Dataset and evaluation documentation will be published here when ready.

## Citation

```bibtex
@software{recoedit2025,
  title  = {ReCoEdit: Rewriter-Guided Consistency Alignment for Product Image Editing},
  author = {Matteoooo46},
  year   = {2025},
  url    = {https://github.com/Matteoooo46/ReCoEdit}
}
```

## License

ReCoEdit is released under the [MIT License](https://github.com/Matteoooo46/ReCoEdit/blob/master/LICENSE).

---

<div align="center">

**ReCoEdit — edit the scene, not the product.**

</div>
