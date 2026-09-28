# Training data construction

This document separates the historical data-generation workflow from a portable
annotation-export interface. The two released adapters are already available:
[ReCoEdit-rewriter](https://huggingface.co/Matteoooo46/ReCoEdit-rewriter) and
[ReCoEdit-RL](https://huggingface.co/Matteoooo46/ReCoEdit-RL).
An adapter release does not by itself identify every training-data revision or
the exact SFT initialization used by every later experiment.

## 1. Product-grounded synthesis

```text
Catalog metadata + reference images
  -> reference selection -> character design + six-shot storyboard
  -> character portrait + product-conditioned target scenes
                                                        |
Imported product-transfer / general editing pairs -------+
                                                        v
                       references X, original instruction u, target y
                           /                             \
          stored editor prompt p                  target caption a = h(y)
                   |                                      |
          editor SFT: (X, p) -> y              rewriter SFT: (X, u) -> a
```

The historical catalog branch is implemented in the workspace's
`inference/batch_generate_images_all_refs.py`:

1. Fetch product titles, selling points, categories and images by product ID.
2. Use Qwen3-VL to select overall/detail references: first select candidates in
   batches of six images, then choose the final reference set from the candidates.
3. Use Gemini 2.5 Pro to design a character and a six-shot advertising storyboard.
4. Use Gemini 2.5 Flash Image to generate a character portrait and scene targets.
   Scene generation conditions on the main product reference, the character
   portrait and up to two preceding generated frames.
5. Retain references, targets, original instructions and source identifiers.
   Export frames in numeric order, not lexicographic filename order.

Six frames describes this catalog branch, not every imported training example.
Product-transfer scenes and general editing pairs (including Doubao/NanoBanana
and UltraEdit assets) are separate inputs to the mixed corpus. Source names do
not establish redistribution rights; retain and check each source's terms.
The historical catalog fetcher and generation gateways depend on internal
services. They are not a public, turnkey synthesis API. Do not publish service
credentials or assume public users can access the original endpoints.

## 2. Separate editor and rewriter supervision

The offline Qwen3-VL-8B captioner (`qwenvl_rewrite.py` in the research workspace)
observes the **target image**, describing appearance, composition, scene, style
and visible details. This is outcome-conditioned supervision, not just a
language-only paraphrase of the original prompt.

| Field | Editor metadata | Rewriter metadata |
|---|---|---|
| Target image y | `image` | Not supplied as model input |
| Conditioning references X | `edit_image` (list) | `images` (list) |
| Original instruction u | `original_prompt` retained for provenance | `input` |
| Editor text condition p | `prompt` | Not assumed identical to output |
| Target-grounded caption a | `caption` retained separately | `output` |
| Rewriting task specification | — | `instruction` |

**`image` is the target; `edit_image` contains the reference/input images.**
The editor consumes `prompt`, not automatically `caption`. Do not silently
replace one with the other. The rewriter never sees the target at inference.
Editor and rewriter SFT are independent branches; editor SFT does not require
running a trained rewriter first.

## 3. Audited annotation snapshot

| Snapshot | Annotation rows | Unique referenced image paths |
|---|---:|---:|
| Editor: `metadata_all_products_expanded_ultimate_with_ultraedit500k_captioned_final.json` | 1,118,329 | 1,803,953 |
| Rewriter: `llama_factory_rewriter_train.jsonl` | 1,118,003 | 685,652 |

These counts come from the local `hf_dataset_upload/*-audit.json` reports.
They count rows/image paths, **not unique products**, and describe the audited
snapshot rather than certifying the dataset consumed by every historical checkpoint.
The reports recorded no missing files at audit time, not a semantic quality audit.
The editor and rewriter counts differ; they must not be described as identical
training sets without a row-level lineage join.

The separately packaged 5,000-product archive is the first 5,000 `item_*`
directories in sorted order, not a random sample or the complete SFT annotation
set. Its local card records 201,435 images and 219,845 files. Dataset packages
and model adapters are different artifacts. Availability, access permissions
and asset-license terms must be checked separately before distribution.

## 4. Portable annotation construction

[`scripts/build_training_annotations.py`](scripts/build_training_annotations.py)
provides a dependency-free export boundary after target generation and captioning.
It does **not** call paid services, regenerate the historical million-row corpus,
or claim to recreate a released checkpoint exactly.

Create JSONL records with explicit fields (illustrative paths/text):

```json
{"sample_id":"p001_f0","product_id":"p001","source":"catalog","references":["images/p001_ref.jpg"],"target":"images/p001_scene0.jpg","original_prompt":"把商品放在木桌上","editor_prompt":"商品放在明亮房间的木桌上","caption":"木桌上的商品，背景是明亮的室内场景","rewriter_instruction":"根据参考图片和原始指令生成编辑提示词，保留编辑意图。"}
```

Run on your own licensed images; the output directory must not already exist:

```bash
python scripts/build_training_annotations.py \
  --input data/records.jsonl --image-root data --output data/export_v1
```

Outputs are `editor.jsonl`, `rewriter.jsonl`, `lineage.jsonl`, and `manifest.json`.
The exporter checks required fields, unique sample IDs and existing relative
image paths, rejects target/reference path leakage, hashes the input manifest
and never substitutes the caption for the editor prompt. Image paths remain
relative to `--image-root`; images are not copied. Use that directory as the
editor dataset base path / LLaMA-Factory media directory. Register the rewriter
columns as `instruction`, `input`, `output`, `images`. If a particular editor
loader requires a JSON array, convert JSONL explicitly for that loader.

The supplied rewriting instruction is explicit per record: do not substitute
this illustrative instruction for a historical system prompt and call it the
same experiment. Captioning mistakes can teach omission of the original edit;
review original-instruction coverage before admitting examples to training.

## 5. Quality control and split boundaries

Before a new benchmark, freeze product-disjoint train/validation/test manifests
**before** measuring model gains. Audit product IDs and image content hashes
across splits, decode images, inspect reference/target pairing, review caption
faithfulness and retain rejection reasons. File existence alone cannot verify
these properties. Neither the new exporter nor the old audit certifies them.

RL consumes references/instructions and generates its own online samples; SFT
uses target images. Shared products/prompts are possible, but the two learning
objectives and sample formats are not identical. Later category-strict Seg RL
subsets and fixed-100 validation sets are not interchangeable with the original
Flow-GRPO data. See [EVAL.md](EVAL.md) for the evaluated cohorts and limitations.
