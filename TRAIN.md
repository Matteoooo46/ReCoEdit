# Training

ReCoEdit is trained in three stages. Each stage relies on an external open-source
training framework — we do **not** vendor those frameworks into this repository;
you install them separately.

| Stage | What it produces | Framework used |
|-------|------------------|-----------------|
| 1. Rewriter SFT | Qwen3-VL-8B LoRA that rewrites raw prompts | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| 2. Edit Model SFT | Qwen-Image-Edit-2511 fine-tuned on product editing data | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) |
| 3. Consistency RL | LoRA adapter that maximizes a product-consistency reward | [Flow-GRPO](https://github.com/yifan123/flow_grpo) |

The three stages are independent: stage 2 does not require stage 1, and stage 3
consumes only the checkpoint from stage 2 (the rewriter is only used at inference
time). Each section below documents what we did — data, config, launch command.

---

## Stage 1 — Rewriter SFT (Qwen3-VL-8B)

**Framework**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — install per its README.

**Goal**: teach Qwen3-VL-8B-Instruct to rewrite a raw editing prompt (Chinese) into
a structured, product-aware prompt suitable for Qwen-Image-Edit.

**Data format.** We used LLaMA-Factory's `sharegpt` format. Each example is
`(product reference image, raw prompt) → structured prompt`, e.g.:

```jsonc
{
  "messages": [
    {"role": "user", "content": "<image>\n<system-prompt>\n\nOriginal prompt: 把这个产品放到户外场景"},
    {"role": "assistant", "content": "i2v描述: 将图中的[产品名]置于自然光的户外花园场景中，产品保持原有颜色和材质..."}
  ],
  "images": ["path/to/product.jpg"]
}
```

Register your dataset in `LLaMA-Factory/data/dataset_info.json`, then point the
config's `dataset:` field at the name you registered (not a file path).

**Config**: [`scripts/rewriter_sft_qwen3vl_8b.yaml`](scripts/rewriter_sft_qwen3vl_8b.yaml)

Key hyperparameters:
- Base model: `Qwen/Qwen3-VL-8B-Instruct`
- Template: `qwen3_vl_nothink` (required by Qwen3-VL)
- Fine-tuning: LoRA, `lora_target: all`
- LR: `1e-4`, cosine schedule, 10% warmup
- Precision: bf16
- Batch: 1 per device × 8 accumulation
- 10 epochs; save every 500 steps
- Eval on a 10% held-out split

**Launch**:

```bash
# Replace the placeholder paths in the config first, then:
llamafactory-cli train scripts/rewriter_sft_qwen3vl_8b.yaml
```

We selected `checkpoint-10000` for downstream use. The released weights are
[🤗 Matteoooo46/ReCoEdit-rewriter](https://huggingface.co/Matteoooo46/ReCoEdit-rewriter).

---

## Stage 2 — Edit Model SFT (Qwen-Image-Edit-2511)

**Framework**: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) —
clone locally and set `DIFFSYNTH_DIR` to its path.

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
export DIFFSYNTH_DIR=$PWD/DiffSynth-Studio
```

**Goal**: adapt Qwen-Image-Edit-2511 to the domain of e-commerce product editing
(product-consistent transformations of a reference image).

**Data format.** DiffSynth-Studio expects a metadata JSON where each entry has
`image` (source), `edit_image` (target), and a prompt:

```json
[
  {
    "image": "path/to/source.jpg",
    "edit_image": "path/to/target.jpg",
    "prompt": "structured editing prompt..."
  }
]
```

We trained on a curated internal dataset of e-commerce product editing pairs,
augmented with a subset of UltraEdit. Prompts were pre-captioned with a
Qwen3-VL-8B captioner. You can use any editing-pair dataset with the schema
above.

**Scripts** (both use env-var paths — nothing is hardcoded):

- Full-parameter SFT: [`scripts/Qwen-Image-Edit-2511-full.sh`](scripts/Qwen-Image-Edit-2511-full.sh)
- LoRA SFT: [`scripts/Qwen-Image-Edit-2511.sh`](scripts/Qwen-Image-Edit-2511.sh)

Key hyperparameters (full SFT):
- Base: `Qwen/Qwen-Image-Edit-2511` transformer + `Qwen/Qwen-Image` VAE / text encoder
- LR: `1e-5` (full) / `1e-4` (LoRA, rank 128)
- Gradient checkpointing: on
- `zero_cond_t: true` (empty conditioning for the first sampling step)
- 50 epochs, save every 1000 steps, keep last 2 checkpoints
- 8× GPU via `accelerate launch`

**Launch (full SFT)**:

```bash
export DIFFSYNTH_DIR=/path/to/DiffSynth-Studio
export TRAIN_METADATA=/path/to/metadata.json
export OUTPUT_DIR=/path/to/sft_output
# Optional:
# export RESUME_CKPT=/path/to/prev_step.safetensors

bash scripts/Qwen-Image-Edit-2511-full.sh
```

**Launch (LoRA SFT)**:

```bash
export DIFFSYNTH_DIR=/path/to/DiffSynth-Studio
export TRAIN_METADATA=/path/to/metadata.json
export OUTPUT_DIR=/path/to/lora_output
# Optional:
# export RESUME_LORA=/path/to/prev_lora.safetensors
# export NUM_PROCESSES=8
# export LORA_RANK=128

bash scripts/Qwen-Image-Edit-2511.sh
```

The output `step-N.safetensors` from this stage becomes the starting point for
stage 3.

---

## Stage 3 — Consistency RL (Flow-GRPO + Product Reward)

**Framework**: [Flow-GRPO](https://github.com/yifan123/flow_grpo) — install per
its README:

```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo && pip install -e . && cd ..
```

We modified Flow-GRPO with (a) a Qwen-Image-Edit-2511 training script and (b) a
`product_consistency` reward. Those modifications are **not** in this
repository — we describe below what to change so you can reproduce them on top
of the upstream Flow-GRPO checkout.

**Goal**: optimize the SFT model so that generated edits preserve product
identity (color, shape, logo) as scored by a VLM judge.

### 3.1 Reward server

The reward is a VLM-as-judge call. You need a Qwen3-VL-30B-A3B-Instruct
endpoint reachable over HTTP. We served the model with
[vLLM](https://github.com/vllm-project/vllm):

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8080 \
    --tensor-parallel-size 4
```

Then export the endpoint URL that our reward function reads:

```bash
export RECOEDIT_VLM_URL=http://your-vlm-server:8080/v1
```

### 3.2 Product consistency reward

We added a `product_consistency` reward that sends the generated image + the
product reference image + a Chinese chain-of-thought scoring prompt to the VLM,
then parses `Score: X` where X ∈ {1..5}, normalized to `[0, 1]` via `(X-1)/4`.

The scoring prompt asks the judge to:
1. Describe key features of the reference product (color, style, pattern, logo).
2. Check if the product appears in the generated image.
3. Compare each feature point-by-point.
4. Output `Score: X`.

We ablated a variant that first classifies whether the prompt should contain
the product and inverts the reward for negative cases; pure 1–5 scoring proved
more stable and is what we use in the released checkpoint.

### 3.3 Dataset

Each training entry has a prompt, `product_name`, and
`product_ref_for_reward` (path to the reference product image). Use a JSONL:

```jsonl
{"prompt": "...", "product_name": "...", "product_ref_for_reward": "/path/to/ref.jpg"}
{"prompt": "...", "product_name": "...", "product_ref_for_reward": "/path/to/ref.jpg"}
```

Split into `train_metadata.jsonl` and `val_metadata.jsonl`, place them in a
directory, and point the RL config's `config.dataset` at that directory.

### 3.4 Config

We used the following hyperparameters in the RL config (Flow-GRPO
`counting_qwenimage_edit_8gpu` base, adapted to our reward):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `pretrained.model` | `Qwen/Qwen-Image-Edit-2511` | Base transformer |
| `train.transformer_path` | your stage-2 SFT `.safetensors` | Starting point |
| `sample.num_steps` | 10 | Sampling steps for policy rollouts |
| `sample.num_image_per_prompt` | 16 (k=16) | Group size for advantage |
| `sample.noise_level` | 1.0 | SDE exploration noise |
| `sample.guidance_scale` | 4.0 | CFG during sampling |
| `train.batch_size` | 4 / GPU | Per-GPU batch |
| `train.beta` | 0 | No KL penalty |
| `train.clip_range` | 1e-3 | PPO clip |
| `use_lora` | True | LoRA rank 64 |
| `mixed_precision` | bf16 | |
| `activation_checkpointing` | True | |
| `fsdp_optimizer_offload` | True | Fit into 8× 80GB VRAM |
| `reward_fn` | `{"product_consistency": 1.0}` | Sole reward |

Point `train.transformer_path` at the checkpoint you produced in stage 2 before
launching.

### 3.5 Launch

```bash
# Optional — WandB logging
export WANDB_API_KEY=your_wandb_api_key

# VLM reward endpoint (see 3.1)
export RECOEDIT_VLM_URL=http://your-vlm-server:8080/v1

cd flow_grpo  # your upstream Flow-GRPO checkout with our reward + config added
torchrun --standalone --nproc_per_node=8 --master_port=19502 \
    scripts/train_qwenimage_edit.py \
    --config config/grpo.py:counting_qwenimage_edit_8gpu_product_consistency
```

### 3.6 Training dynamics

Our production run (28 epochs, 8× GPU):

| Epoch | Reward | Notes |
|-------|--------|-------|
| 0 | 0.022 | Cold start from stage-2 SFT |
| 5 | 0.169 | First breakthrough |
| 10 | 0.318 | Steady improvement |
| 15 | 0.434 | Approaching convergence |
| 20 | 0.470 | Near peak |
| **25** | **0.480** | 🏆 Peak — released checkpoint |
| 28 | 0.450 | Plateau, training stopped |

**21.8× improvement** over the stage-2 starting point. The best LoRA adapter is
released as [🤗 Matteoooo46/ReCoEdit-RL](https://huggingface.co/Matteoooo46/ReCoEdit-RL).

---

## Reproducibility Notes

- Stages 1 and 2 depend on **your own dataset** — the released ReCoEdit
  checkpoints were trained on internal e-commerce product data that we cannot
  redistribute. Any editing-pair dataset with the schema above will work.
- Stage 3 will consume the stage-2 output; you can also start it directly from
  the base Qwen-Image-Edit-2511, but expect a slower cold start.
- All three stages log to WandB by default; unset `report_to` /
  `WANDB_API_KEY` to disable.
