# Evaluation

## Product-region validation: 22 products, three configurations

On **117 matched frames valid in all three configurations**, the complete RL + trained rewriter + APG pipeline improves Seg CLIP-I from **0.709493 to 0.746724** (+0.037231, +5.25% relative), and Seg DINO-I from **0.345001 to 0.421937** (+0.076935, +22.30% relative), compared with the original-model CFG baseline.

[Open the 22-product image and score viewer](https://js-ad.a.yximgs.com/bs2/ad-nieuwland-material/qwen_inference/comparison/random22_three_way_20260916/12120fd339b23558/index.html)

### Data and inference

From the existing random35 product list, select the **22 products with nonempty Hive second-level categories** in the exported lookup; exclude 13 with missing categories. Each product has six storyboard instructions: **132 frames per configuration, 396 images total**. Align samples by product ID and frame index. This is a category-available subset; training/evaluation overlap has not been audited.

| Configuration | Image weights | Prompt / guidance |
|---|---|---|
| Base + Original + CFG | Qwen-Image-Edit-2511, unmodified | Original prompt, empty negative prompt, CFG |
| RL + Original + APG | Base + RL LoRA | Original prompt, empty negative prompt, APG |
| RL + Rewriter + APG | Base + same RL LoRA | Trained rewriter plus anti-text suffix and negative prompt, APG |

- RL adapter: `flow_grpo/best_checkpoints/grpo_round2_epoch28_for_inference.safetensors` in the experiment workspace. These scores do not correspond to the newer Seg-reward or recoedit-r10 checkpoints.
- Rewriter: Qwen3-VL-8B-Instruct with `qwen3-vl-8b-sft-v2/checkpoint-10000`.
- Inference: 40 steps, guidance scale 4.0, 896 × 1152, seed = frame index (0–5), `zero_cond_t=True`. APG: `eta=0.0`, `beta=-0.5`, `norm_threshold=None`.
- Inference conditions on product reference images and the lifestyle/character image where present. Scoring uses the **first sorted product reference image**, identical across methods.

### Seg CLIP-I / Seg DINO-I protocol

1. Use the **literal English second-level category translation** as the GroundingDINO query, lowercased with a terminal period. Do not replace broad categories with title-derived object names. Exact mappings: [categories.json](evaluation/random22/categories.json).
2. GroundingDINO independently detects the highest-confidence product box in the reference and generated image.
3. SAM2 uses each box to segment the product. Extract the masked product crop with padding and replace background pixels with a fixed color.
4. Compute normalized CLIP image embeddings and DINOv2 CLS embeddings. Their reference/generated **cosine similarities** are Seg CLIP-I and Seg DINO-I (higher is better).
5. Detection/segmentation failures are invalid, **not zero scores**. Report each method's success rate and compare methods on the common valid intersection.

Evaluator defaults: `IDEA-Research/grounding-dino-base` (box threshold 0.30, text threshold 0.25), `facebook/sam2.1-hiera-large` (multimask enabled), `openai/clip-vit-large-patch14`, `facebook/dinov2-base`; 5% crop padding, white background (255), float32. The historical run did not save a complete command/environment manifest, so package versions and any unrecorded command-line overrides cannot be independently recovered from the score files. Queries, statuses and scores are preserved.

### Aligned results: 117 common valid frames

| Method | Seg CLIP-I ↑ | Seg DINO-I ↑ | Δ CLIP vs base | Δ DINO vs base |
|---|---:|---:|---:|---:|
| Base + Original + CFG | 0.709493 | 0.345001 | — | — |
| RL + Original + APG | 0.716729 | 0.351560 | +0.007236 | +0.006559 |
| RL + Rewriter + APG | **0.746724** | **0.421937** | **+0.037231** | **+0.076935** |

Relative to RL + Original + APG, the rewriter configuration adds **+0.029995 CLIP-I** and **+0.070376 DINO-I** on the common frames.

### Coverage and individual valid-set means

| Method | Valid / attempted | Success rate | Seg CLIP-I ↑ | Seg DINO-I ↑ |
|---|---:|---:|---:|---:|
| Base + Original + CFG | 124 / 132 | 93.94% | 0.710816 | 0.340537 |
| RL + Original + APG | 124 / 132 | 93.94% | 0.716548 | 0.350218 |
| RL + Rewriter + APG | 120 / 132 | 90.91% | 0.747456 | 0.417182 |

All 28 invalid image evaluations have status `generated_detection_failed`. The common valid intersection is **117/132 frames (88.64%)**. Individual means use different valid subsets; use the aligned table for matched comparisons.

### Interpretation

The results support improved localized product similarity for the **complete trained inference pipeline** on this subset. The RL + Original comparison changes both weights and guidance (CFG → APG), so it does not isolate RL's causal effect; a base + original + APG control would be needed. The rewriter comparison includes its anti-text suffix and negative prompt as well as rewriting.

These metrics do not measure instruction following or overall image quality. Literal broad-category queries may select unintended regions; successful detection is not human verification of mask accuracy. Consider the rewriter configuration's lower success rate alongside its higher similarities. This small subset and single seed per frame do not establish statistical significance or broad generalization. These metrics are distinct from the training reward in TRAIN.md.

### Artifacts and reproduction

- [summary.json](evaluation/random22/summary.json): aggregate scores on individual and common valid sets.
- [scores.jsonl](evaluation/random22/scores.jsonl): all 396 per-image scores/statuses and public output image URLs, without machine-local paths.
- [three_way_metrics.csv](evaluation/random22/three_way_metrics.csv): 132 matched frames and pairwise differences.
- [categories.json](evaluation/random22/categories.json): 22 literal category translations and queries.
- [Viewer HTML](evaluation/random22/index.html): all 396 images, references, prompts and scores; images load from public asset URLs.

The [generic evaluator](scripts/seg_clip_dino_eval.py) reads JSONL with `id`, `product_name`, `query`, `reference_image`, and `generated_image`. Download/reconstruct local image pairs and supply their paths:

```bash
python scripts/seg_clip_dino_eval.py \
  --input /path/to/local_pairs.jsonl --output evaluation/rerun \
  --device cuda:0 --no-visuals
```

The [experiment orchestrator](scripts/eval_random22_three_way_seg.py) expects the original workspace layout and local inference outputs (large image directories are not included in Git). In that workspace:

```bash
CUDA_VISIBLE_DEVICES=0 verl-omni/.venv/bin/python \
  scripts/eval_random22_three_way_seg.py \
  --device cuda:0 --output eval/random22_three_way_seg --no-visuals
```

Successful samples resume by ID. Use a fresh output directory when changing queries, images or evaluator settings.
