# Evaluation protocols and archived results

These are measured results from existing evaluation artifacts, not new inference.
Numerical snapshots and source-file SHA-256 hashes are in
[docs/results/archived_metrics.json](docs/results/archived_metrics.json).
`python scripts/export_documented_results.py` re-exports them in the research
workspace. The public snapshot needs no private images to read, but aggregate
statistics are not a substitute for independently reproducing per-image scores.

## Cohorts and checkpoint identity

| Evaluation | Products | Frames per configuration |
|---|---:|---:|
| Random22 | 22 | 132 |
| Selected20 | 20 | 120 |

The historical viewer covers 21 products and 212 frames in a separate comparison.
The Selected20 results are exploratory. Generalization to unseen products remains unverified.

The public [RL adapter](https://huggingface.co/Matteoooo46/ReCoEdit-RL) and
[rewriter adapter](https://huggingface.co/Matteoooo46/ReCoEdit-rewriter) exist.
However, their availability does not establish that every historical experiment
used those exact bytes. Random22 uses the archived
`grpo_round2_epoch28_for_inference` adapter; its SFT comparison uses step 22,000
from the resume-66,000 branch, not the later resume-132,000 branch. The public RL
model card identifies epoch 24. Keep these distinct until hashes and SFT-base
provenance have been matched.

Random22 inference used 40 steps, guidance 4.0, 896×1152 images and frame-index
seeds. APG used eta=0, beta=-0.5 and no explicit norm threshold. Rewriter
inference also uses historical anti-text post-processing and negative prompts.
These are configuration comparisons, not isolated single-factor interventions.
Selected20 combines historical generation settings, including differences in how the RL adapter was loaded.

## 1. Seg CLIP-I / Seg DINO-I: product-region consistency

Detect products independently in reference/generated images using an English
category query and GroundingDINO; segment with SAM2; crop the masked regions
onto white backgrounds; compute cosine similarity between L2-normalized CLIP
or DINOv2 features. Scores measure representation similarity, not a calibrated
probability of identical products.

Models: `IDEA-Research/grounding-dino-base`, `facebook/sam2.1-hiera-large`,
`openai/clip-vit-large-patch14`, and `facebook/dinov2-base`. Default box/text
thresholds: 0.3/0.25. Historical random22 uses literal English L2 translations,
not the later strict title/L3 query policy. Early library/model revisions were
not fully frozen, so bitwise reproducibility is not claimed.

Failed detection/segmentation produces a missing measurement, not zero.
Report coverage and compare the **same successful-frame intersection** across
methods. This is still conditional on detection success and may exclude hard
cases. A detector failure alone does not establish product absence.

### Random22: historical all-frame protocol

132 candidate frames; **117 common successful frames** across these methods.
No prompt-visibility mask was applied to this historical table.

| Configuration | Seg CLIP-I ↑ | Seg DINO-I ↑ |
|---|---:|---:|
| Base + Original + CFG | 0.709493 | 0.345001 |
| RL + Original + APG | 0.716729 | 0.351560 |
| RL + Trained rewriter + APG | 0.746724 | 0.421937 |

The complete configuration exceeds Base by **0.037231 CLIP / 0.076935 DINO**
on that intersection, while also changing guidance and prompt processing.

### Selected20: prompt-eligible region protocol

Eligibility is determined from the **original prompt**, shared across methods,
not generated images or metric values. The manual text audit distinguishes
required object/package/components, required contents/transformed products,
not-required and uncertain frames. Required cases are included; uncertain ones
are withheld. This is a text rule, not an image-based product-presence detector.

82 of 120 frames are eligible. Valid counts: Base 77, SFT 79, SFT+RL 79,
SFT+RL+rewriter 81. **75 frames succeed in all four methods**. Seg columns below
use those same 75; MUSIQ uses all 120.

| Archived configuration | MUSIQ ↑ (120) | Seg CLIP-I ↑ (75) | Seg DINO-I ↑ (75) |
|---|---:|---:|---:|
| Base + Original + APG | 66.324930 | 0.746347 | 0.425366 |
| SFT + Original + APG | 67.420095 | 0.746075 | 0.401423 |
| SFT+RL + Original + APG | 66.409286 | 0.761344 | 0.444113 |
| SFT+RL + Trained rewriter + APG | 67.583915 | 0.770416 | 0.489846 |

Within this exploratory set, SFT raises MUSIQ but lowers DINO; RL recovers
region consistency. Older Qwen Product/Detail scores remain explicitly labeled
`selected20_historical` in the snapshot, not relabeled as InternVL scores.

## 2. Whole-image quality

TOPIQ-NR and MUSIQ operate on generated images without a reference or prompt.
They do not measure identity or instruction following. Q-Align below uses its
**aesthetic** task, not an interchangeable quality scale. Do not add raw scores.
Random22 has **132 scored images per method for each metric**:

| Configuration | TOPIQ-NR ↑ | MUSIQ ↑ | Q-Align aesthetic ↑ |
|---|---:|---:|---:|
| Base + Original + CFG | 0.501130 | 66.833690 | 3.706025 |
| Base + Original + APG | 0.489005 | 66.833409 | 3.718550 |
| SFT + Original + APG | 0.517078 | 67.677882 | 3.687678 |
| RL + Original + APG | 0.481191 | 66.600368 | 3.677668 |
| RL + Trained rewriter + APG | 0.484860 | 67.757875 | 3.598041 |

SFT improves TOPIQ-NR by 0.028072 and MUSIQ by 0.844473 relative to Base+APG,
but lowers Q-Align aesthetic. This supports improvement in those two NR-IQA
metrics, **not universal perceptual improvement**. No human calibration or
confidence intervals are claimed.

## 3. InternVL: trained vs. untrained rewriting

Judge: **InternVL3.5-8B-Instruct**, local BF16, deterministic decoding
(`do_sample=false`), four independent workers. Intent evaluation uses one
448-pixel tile; Product/Detail uses up to two tiles plus a thumbnail when
applicable. Vision tiles are processed serially for memory control.

### Text intent and image edit completion

Extract 4–10 visible requirements from the original instruction with exact
supporting quotations. Weight main subjects/actions/scenes by 2 and secondary
lighting/style/composition by 1. Freeze the checklist across methods. The text
judge checks the rewrite; the image judge uses the original instruction and
checklist, not the rewrite. Method labels are hidden from the judge.

Per-frame score: `100 × sum(weight × status_value) / sum(weight)`, with
`met=1`, `partial=0.5`, `missing=0`, `contradicted=0`. Average across frames.
These are judge-derived scores, not human ratings or embedding cosine scores.

21/120 original checklists failed validation and were excluded from all methods,
not scored zero. **99 matched frames** remain.
Product/Detail is independent, with **120 frames per method**.

| Configuration | Text intent ↑ /100 (99) | Image completion ↑ /100 (99) | Product ↑ /5 (120) | Detail ↑ /5 (120) |
|---|---:|---:|---:|---:|
| Base + Original | 100.000 | 97.868 | 1.567 | 1.567 |
| SFT + Original | 100.000 | 95.264 | 1.533 | 1.533 |
| SFT+RL + Original | 100.000 | 98.097 | 1.567 | 1.567 |
| SFT+RL + Trained rewriter | 85.778 | 88.794 | 1.892 | 1.900 |
| SFT+RL + Untrained rewriter | 43.215 | 50.763 | 3.433 | 3.433 |

Original text scores of 100 are identity comparisons, not a learned gain.
Trained rewriting exceeds untrained rewriting by **42.563 text-intent points**
and **38.031 image-completion points**, but does not exceed SFT+RL with the
original prompt in image completion. “Untrained” means the pretrained Qwen
rewriter without task-specific SFT, not random initialization. No weighted
total is reported.

### Product/Detail rubric and validity warning

Product covers instance geometry, structure, components, branding, patterns and
materials; Detail covers texture, seams, edges, labels and small parts. Integer
anchors range from 1 (missing/destroyed) to 5 (faithfully preserved). Requested
attribute changes should not be penalized. Inputs are the reference, generated
image and instruction. In this archived all-frame protocol, an absent or
unmatchable product receives **Product=1 and Detail=1**, even if the prompt did
not require its appearance.

**599/600 Product/Detail pairs are identical**, with strong endpoint concentration.
The protocol conflates visibility with fine-detail preservation; limited image
resolution further constrains detail judgments. These columns are diagnostics,
not validated independent dimensions. An untrained rewrite that merely describes
the product may score well here while failing the intended scene/edit. That is
not evidence of better overall editing.

## Reproduction boundaries

Research-workspace entry points: `scripts/seg_clip_dino_eval.py`,
`scripts/score_random22_image_quality.py`,
`scripts/filter_random22_product_visibility.py`, and
`scripts/eval_selected20_all_internvl.py`. They require local manifests/assets;
they are not claimed to run from a clean public clone without those inputs.
Full InternVL prompts/settings are included in
[docs/results/internvl_protocol.json](docs/results/internvl_protocol.json),
with the private model path replaced by its name and the original protocol hash
retained. This publishes the rubric used, including its limitations, not a
retroactively improved rubric applied to unchanged scores.

For new benchmarks, pre-register held-out products; freeze weight hashes,
original prompts, seeds, guidance, resolution and negative prompts; calibrate
judges against blinded humans; report coverage/failures; bootstrap paired
differences by **product**, not independent frames. Keep internal RL reward
curves separate from external metrics.

## Original Random22 evaluation archive

The earlier detailed coverage tables, per-product results, viewer links and
reproduction commands are preserved in [the original protocol](docs/RANDOM22_ARCHIVE.md).
