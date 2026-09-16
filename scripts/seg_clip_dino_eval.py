#!/usr/bin/env python3
"""
Seg CLIP-I / Seg DINO-I 计算脚本（严格按 Seg_CLIP_I_Seg_DINO_I_产品名流程.md 实现）

流水线（每个样本）：
  1. GroundingDINO 用 product_name 在 reference_image 上定位商品 → 最高分 bbox
  2. GroundingDINO 用同一 product_name 在 generated_image 上定位商品 → 最高分 bbox
  3. SAM2 以 bbox 为 box prompt 分割 reference / generated 商品 → 布尔 mask
  4. mask 最小外接 bbox + padding 裁出 product crop（mask 外像素用固定背景值）
  5. CLIP image encoder 提取 ref/gen crop embedding → L2 归一化 → cosine = seg_clip_i
  6. DINOv2 CLS token 提取 ref/gen crop embedding → L2 归一化 → cosine = seg_dino_i
  7. 逐样本写入 results.jsonl（断点续跑）、per_sample_metrics.csv、summary.json

模型（全部由 transformers 原生加载，均可通过命令行覆盖为 HF repo ID 或本地目录）：
  - GroundingDINO 默认: IDEA-Research/grounding-dino-base
  - SAM2 默认: facebook/sam2.1-hiera-large
  - CLIP 默认: openai/clip-vit-large-patch14
  - DINOv2 默认: facebook/dinov2-base

输入 JSONL（每行一条）：
  {"id":"000001","product_name":"purple V-neck knitted sweater",
   "reference_image":"/data/ref/000001.jpg","generated_image":"/data/gen/000001.png",
   "edit_instruction":"Change the background to a cafe.",
   "human_score": 2}                      # human_score 可选（0/1/2），有则计算 Spearman

用法：
  python scripts/seg_clip_dino_eval.py --input eval_data.jsonl --output evaluation
  python scripts/seg_clip_dino_eval.py --input eval_data.jsonl --output evaluation --limit 100
  # 产品名定位差时补充通用类别（流程第 6 节）：
  python scripts/seg_clip_dino_eval.py --input eval_data.jsonl --output evaluation --query_suffix handbag

依赖：wzt_new 环境已齐（torch/transformers/numpy/pandas/scipy/PIL），无需额外安装。
首次运行自动从 HF 下载 4 个 checkpoint（约 5GB），需要能访问 huggingface.co。

输出目录结构（流程第 34 节）：
  evaluation/
  ├── per_sample_metrics.csv
  ├── summary.json
  ├── results.jsonl          # 断点续跑：已有 id 自动跳过
  ├── bbox_visualizations/
  ├── mask_visualizations/
  ├── masks/                 # 布尔 mask 的 npz
  └── product_crops/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

DEVICE_AUTO = "auto"
FAIL_REASON_REF_DETECT = "reference_detection_failed"
FAIL_REASON_GEN_DETECT = "generated_detection_failed"
FAIL_REASON_REF_SEG = "reference_segmentation_failed"
FAIL_REASON_GEN_SEG = "generated_segmentation_failed"

# ─────────────────────────────────────────────────────────────────────────────
# 模型加载（程序启动时各加载一次，全部 eval）
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> str:
    if device_arg != DEVICE_AUTO:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(args, device: str):
    """加载 GroundingDINO / SAM2 / CLIP / DINOv2，各一次。"""
    import transformers
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPModel,
        CLIPProcessor,
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    dtype = getattr(torch, args.dtype)
    if device == "cpu" and dtype != torch.float32:
        print(f"[warn] CPU 下将 dtype 从 {args.dtype} 强制改为 float32")
        dtype = torch.float32

    print(f"[load] GroundingDINO: {args.gdino_model}")
    gdino = GroundingDinoForObjectDetection.from_pretrained(
        args.gdino_model, dtype=dtype
    ).to(device).eval()
    gdino_proc = GroundingDinoProcessor.from_pretrained(args.gdino_model)

    print(f"[load] SAM2: {args.sam2_model}")
    sam2 = Sam2Model.from_pretrained(args.sam2_model, dtype=dtype).to(device).eval()
    sam2_proc = Sam2Processor.from_pretrained(args.sam2_model)

    print(f"[load] CLIP: {args.clip_model}")
    clip = CLIPModel.from_pretrained(args.clip_model, dtype=dtype).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model)

    print(f"[load] DINOv2: {args.dinov2_model}")
    dino = AutoModel.from_pretrained(args.dinov2_model, dtype=dtype).to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained(args.dinov2_model)

    return {
        "gdino": gdino, "gdino_proc": gdino_proc,
        "sam2": sam2, "sam2_proc": sam2_proc,
        "clip": clip, "clip_proc": clip_proc,
        "dino": dino, "dino_proc": dino_proc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 第 4/5 节：GroundingDINO 定位（同一 product_name、同一阈值）
# ─────────────────────────────────────────────────────────────────────────────

def build_query(product_name: str, suffix: str = "") -> str:
    """query 规则：product_name[ + 通用类别] 转小写并以句号结尾（GroundingDINO 官方约定）。"""
    name = f"{product_name} {suffix}".strip() if suffix else product_name
    return name.lower().strip() + "."


def detect_product(models, image: Image.Image, query: str, device: str,
                   box_threshold: float, text_threshold: float):
    """返回 (bbox_xyxy, score)，无候选返回 (None, None)。bbox 已限制在图像范围内。"""
    gdino, proc = models["gdino"], models["gdino_proc"]
    w, h = image.size
    inputs = proc(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]
    scores = results.get("scores", [])
    boxes = results.get("boxes", [])
    if len(scores) == 0 or len(boxes) == 0:
        return None, None

    # 多个候选：取分数最高者。post_process 后的 box 已经是原图绝对坐标 xyxy。
    best = int(torch.argmax(scores).item()) if torch.is_tensor(scores) else int(np.argmax(scores))
    x1, y1, x2, y2 = [float(v) for v in boxes[best]]

    # 限制在图像有效范围内
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, None
    return (x1, y1, x2, y2), float(scores[best])


# ─────────────────────────────────────────────────────────────────────────────
# 第 8/9 节：SAM2 分割（bbox 作为 box prompt，固定 multimask 策略）
# ─────────────────────────────────────────────────────────────────────────────

def segment_product(models, image: Image.Image, bbox, device: str,
                    multimask_output: bool):
    """返回与原图同尺寸的二维布尔 mask（True=商品像素），失败返回 None。"""
    sam2, proc = models["sam2"], models["sam2_proc"]
    x1, y1, x2, y2 = bbox

    # SAM2 的 input_boxes 使用原图绝对像素坐标 xyxy，不做 0~1 归一化。
    inputs = proc(
        images=image,
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam2(**inputs, multimask_output=multimask_output)

    # SAM2 官方 HF API：只需要 pred_masks + original_sizes。
    masks = proc.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )[0]  # 单图: [num_objects, num_masks, H, W]

    if masks.ndim != 4 or masks.shape[0] < 1:
        return None

    object_masks = masks[0]  # 单个 bbox 对应的 object: [num_masks, H, W]
    if object_masks.shape[0] < 1:
        return None

    if multimask_output and object_masks.shape[0] > 1:
        # 单图、单 bbox 时 iou_scores 通常可展平为 num_masks 个分数。
        iou_scores = outputs.iou_scores[0].detach().float().cpu().reshape(-1)
        best = int(torch.argmax(iou_scores).item())
        best = min(best, object_masks.shape[0] - 1)
    else:
        best = 0

    mask_tensor = object_masks[best]
    # post_process_masks 通常已二值化；为兼容不同版本，这里统一阈值化。
    if mask_tensor.dtype == torch.bool:
        mask = mask_tensor.numpy()
    else:
        mask = mask_tensor.numpy() > 0.5

    if not mask.any():
        return None
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 第 10/11 节：商品 crop（mask 外接框 + 5% padding，mask 外用固定背景值）
# ─────────────────────────────────────────────────────────────────────────────

def extract_crop(image: Image.Image, mask: np.ndarray,
                 padding_ratio: float, bg_value: int) -> Image.Image:
    img = np.asarray(image.convert("RGB")).copy()
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pad_y = int((y2 - y1) * padding_ratio)
    pad_x = int((x2 - x1) * padding_ratio)
    h, w = img.shape[:2]
    y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    # mask 外像素 → 固定背景值（流程第 10.5 / 11.5 节）
    crop = img[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = bg_value
    return Image.fromarray(crop)


# ─────────────────────────────────────────────────────────────────────────────
# 第 13-16 节：CLIP embedding + cosine
# ─────────────────────────────────────────────────────────────────────────────

def clip_embed(models, crop: Image.Image, device: str) -> torch.Tensor:
    proc, model = models["clip_proc"], models["clip"]
    inputs = proc(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)  # [1, D]
    # Depending on transformers version, this returns a tensor or an output
    # object. CLIPModel.get_image_features() in 5.14.1 stores the already
    # projected embedding in pooler_output; do not apply visual_projection again.
    if not torch.is_tensor(feat):
        output = feat
        feat = getattr(output, "image_embeds", None)
        if not torch.is_tensor(feat):
            feat = getattr(output, "pooler_output", None)
    if not torch.is_tensor(feat):
        raise TypeError("CLIP get_image_features() did not return image embeddings")
    return torch.nn.functional.normalize(feat, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 第 17-20 节：DINOv2 embedding（CLS token）+ cosine
# ─────────────────────────────────────────────────────────────────────────────

def dino_embed(models, crop: Image.Image, device: str) -> torch.Tensor:
    proc, model = models["dino_proc"], models["dino"]
    inputs = proc(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    cls = out.last_hidden_state[:, 0, :]  # [1, D]
    return torch.nn.functional.normalize(cls, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 可视化（流程第 4.11 / 8.12 节）
# ─────────────────────────────────────────────────────────────────────────────

def draw_bbox(image: Image.Image, bbox, label: str, score: float,
              color: str = "#00ff00") -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    d.rectangle([x1, y1, x2, y2], outline=color, width=4)
    d.text((x1, max(0, y1 - 18)), f"{label} {score:.2f}", fill=color)
    return img


def draw_mask_overlay(image: Image.Image, mask: np.ndarray,
                      color: tuple = (255, 0, 0), alpha: int = 90) -> Image.Image:
    img = image.convert("RGB").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    overlay.paste(color + (alpha,), mask=mask_img)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# 单样本
# ─────────────────────────────────────────────────────────────────────────────

def process_sample(rec: dict, models, device: str, args, out_dir: Path):
    sid = str(rec["id"])
    product_name = str(rec["product_name"])
    result = {
        "id": sid,
        "product_name": product_name,
        "reference_image": rec.get("reference_image", ""),
        "generated_image": rec.get("generated_image", ""),
        "edit_instruction": rec.get("edit_instruction", ""),
        "human_score": rec.get("human_score", None),
        "status": "ok",
        "ref_detection_score": None,
        "gen_detection_score": None,
        "ref_mask_area_ratio": None,
        "gen_mask_area_ratio": None,
        "seg_clip_i": np.nan,
        "seg_dino_i": np.nan,
    }
    # 每条记录可自带完整 query（优先），否则 product_name + query_suffix（覆盖全局 --query_suffix）
    query = str(rec.get("query") or "").strip()
    if not query:
        suffix = str(rec.get("query_suffix") or "").strip() or args.query_suffix
        query = build_query(product_name, suffix)
    result["query"] = query
    fname = sid.replace("/", "_")

    # ── 读取图片（流程第 2 节：统一转 RGB + 有效性检查）──
    try:
        ref_img = Image.open(rec["reference_image"]).convert("RGB")
        gen_img = Image.open(rec["generated_image"]).convert("RGB")
    except Exception as e:
        result["status"] = f"image_load_failed:{type(e).__name__}"
        return result

    # ── GroundingDINO 定位（流程第 4/5 节）──
    ref_bbox, ref_score = detect_product(
        models, ref_img, query, device, args.box_threshold, args.text_threshold)
    if ref_bbox is None:
        result["status"] = FAIL_REASON_REF_DETECT
        return result
    result["ref_detection_score"] = round(ref_score, 4)
    if args.save_visuals:
        (out_dir / "bbox_visualizations").mkdir(parents=True, exist_ok=True)
        draw_bbox(ref_img, ref_bbox, product_name, ref_score).save(
            out_dir / "bbox_visualizations" / f"{fname}_ref.png")

    gen_bbox, gen_score = detect_product(
        models, gen_img, query, device, args.box_threshold, args.text_threshold)
    if gen_bbox is None:
        result["status"] = FAIL_REASON_GEN_DETECT
        return result
    result["gen_detection_score"] = round(gen_score, 4)
    if args.save_visuals:
        draw_bbox(gen_img, gen_bbox, product_name, gen_score).save(
            out_dir / "bbox_visualizations" / f"{fname}_gen.png")

    # ── SAM2 分割（流程第 8/9 节）──
    ref_mask = segment_product(models, ref_img, ref_bbox, device, args.sam2_multimask)
    if ref_mask is None:
        result["status"] = FAIL_REASON_REF_SEG
        return result
    result["ref_mask_area_ratio"] = round(float(ref_mask.mean()), 6)

    gen_mask = segment_product(models, gen_img, gen_bbox, device, args.sam2_multimask)
    if gen_mask is None:
        result["status"] = FAIL_REASON_GEN_SEG
        return result
    result["gen_mask_area_ratio"] = round(float(gen_mask.mean()), 6)

    # ── 商品 crop（流程第 10/11 节）──
    ref_crop = extract_crop(ref_img, ref_mask, args.crop_padding, args.crop_bg)
    gen_crop = extract_crop(gen_img, gen_mask, args.crop_padding, args.crop_bg)

    # ── CLIP / DINOv2 embedding + cosine（流程第 13-20 节）──
    ref_clip = clip_embed(models, ref_crop, device)
    gen_clip = clip_embed(models, gen_crop, device)
    result["seg_clip_i"] = round(
        float(torch.nn.functional.cosine_similarity(ref_clip, gen_clip)[0]), 6)

    ref_dino = dino_embed(models, ref_crop, device)
    gen_dino = dino_embed(models, gen_crop, device)
    result["seg_dino_i"] = round(
        float(torch.nn.functional.cosine_similarity(ref_dino, gen_dino)[0]), 6)

    # ── 可视化与中间产物（bbox 可视化已在检测成功后立即保存）──
    if args.save_visuals:
        (out_dir / "mask_visualizations").mkdir(parents=True, exist_ok=True)
        draw_mask_overlay(ref_img, ref_mask).save(
            out_dir / "mask_visualizations" / f"{fname}_ref.png")
        draw_mask_overlay(gen_img, gen_mask).save(
            out_dir / "mask_visualizations" / f"{fname}_gen.png")
        (out_dir / "product_crops").mkdir(parents=True, exist_ok=True)
        ref_crop.save(out_dir / "product_crops" / f"{fname}_ref.png")
        gen_crop.save(out_dir / "product_crops" / f"{fname}_gen.png")
        np.savez_compressed(
            out_dir / "masks" / f"{fname}.npz",
            ref_mask=ref_mask, gen_mask=gen_mask)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总（流程第 23-25 节）+ 人工相关性（第 31-33 节）
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame, out_dir: Path) -> dict:
    ok = df[df["status"] == "ok"].copy()
    num_valid, num_total = len(ok), len(df)

    def stat_block(series: pd.Series) -> dict:
        if len(series) == 0:
            return {"mean": None, "median": None, "std": None, "n": 0}
        mean = float(series.mean())
        median = float(series.median())
        std = float(series.std()) if len(series) >= 2 else 0.0
        return {
            "mean": round(mean, 6),
            "median": round(median, 6),
            "std": round(std, 6),
            "n": int(len(series)),
        }

    summary = {
        "num_valid": num_valid,
        "num_total": num_total,
        "evaluation_success_rate": round(num_valid / num_total, 6) if num_total else 0.0,
        "failure_counts": df["status"].value_counts().to_dict() if len(df) else {},
        "seg_clip_i": stat_block(ok["seg_clip_i"]),
        "seg_dino_i": stat_block(ok["seg_dino_i"]),
    }

    # 人工评分相关性：至少 5 条，且人工分和对应 metric 都不能是常数。
    human = df.dropna(subset=["human_score", "seg_clip_i", "seg_dino_i"])
    if (
        len(human) >= 5
        and human["human_score"].nunique() > 1
        and human["seg_clip_i"].nunique() > 1
        and human["seg_dino_i"].nunique() > 1
    ):
        from scipy.stats import spearmanr
        rc, _ = spearmanr(human["human_score"], human["seg_clip_i"])
        rd, _ = spearmanr(human["human_score"], human["seg_dino_i"])
        if np.isfinite(rc):
            summary["clip_human_spearman"] = round(float(rc), 6)
        if np.isfinite(rd):
            summary["dino_human_spearman"] = round(float(rd), 6)
        summary["human_corr_n"] = int(len(human))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Seg CLIP-I / Seg DINO-I 评估")
    ap.add_argument("--input", required=True, help="输入 JSONL")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam2_model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--dinov2_model", default="facebook/dinov2-base")
    ap.add_argument("--box_threshold", type=float, default=0.3)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--query_suffix", default="", help="产品名后补充的通用类别，如 handbag")
    ap.add_argument("--device", default=DEVICE_AUTO, help="cuda/cpu/auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sam2_multimask", action=argparse.BooleanOptionalAction, default=True,
                    help="multimask_output=True 并取 IoU 最高 mask（固定策略，--no-sam2_multimask 关闭）")
    ap.add_argument("--crop_padding", type=float, default=0.05, help="crop padding 比例")
    ap.add_argument("--crop_bg", type=int, default=255, help="crop 中 mask 外像素的固定背景值")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（测试用）")
    ap.add_argument("--no-resume", action="store_true", help="不跳过已有结果")
    ap.add_argument("--no-visuals", action="store_true", help="不保存可视化/中间产物")
    args = ap.parse_args()
    args.save_visuals = not args.no_visuals

    if args.query_suffix:
        print(f"query 规则: product_name + '{args.query_suffix}'（小写+句号）")

    device = resolve_device(args.device)
    print(f"device: {device} (torch {torch.__version__}, cuda={torch.cuda.is_available()})")

    # 读输入
    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print(f"输入样本: {len(records)}")

    out_dir = Path(args.output)
    for sub in ("bbox_visualizations", "mask_visualizations", "product_crops", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # 断点续跑。--no-resume 表示从头重跑，因此先清空旧 results.jsonl。
    results_path = out_dir / "results.jsonl"
    if args.no_resume and results_path.exists():
        results_path.unlink()

    done_ids = set()
    if results_path.exists() and not args.no_resume:
        for line in open(results_path, encoding="utf-8"):
            if not line.strip():
                continue
            old = json.loads(line)
            # 只跳过已经成功完成的样本；失败样本下次自动重试。
            if old.get("status") == "ok":
                done_ids.add(str(old["id"]))
        print(f"已跳过 {len(done_ids)} 条成功完成样本（失败样本会重试）")

    todo = [r for r in records if str(r["id"]) not in done_ids]
    if args.limit:
        todo = todo[: args.limit]

    # 加载模型（一次）
    models = load_models(args, device)

    # 逐样本处理
    n_ok = n_fail = 0
    with open(results_path, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(todo, 1):
            res = process_sample(rec, models, device, args, out_dir)
            # 把 NaN 转成 None 以便 JSON 序列化
            for k in ("seg_clip_i", "seg_dino_i"):
                if res[k] is None or (isinstance(res[k], float) and np.isnan(res[k])):
                    res[k] = None
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            if res["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            print(f"[{i}/{len(todo)}] {res['id'][:40]:<40} {res['status']:<32} "
                  f"clip={res['seg_clip_i']} dino={res['seg_dino_i']}")

    # 汇总：若某个 id 曾失败后又重试成功，只保留最后一次结果。
    latest_by_id = {}
    for line in open(results_path, encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        latest_by_id[str(item["id"])] = item
    all_res = list(latest_by_id.values())
    df = pd.DataFrame(all_res)
    # 失败样本 seg 指标写成 NaN（流程第 22 节）；human_score 统一为数值
    df[["seg_clip_i", "seg_dino_i"]] = df[["seg_clip_i", "seg_dino_i"]].astype(float)
    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
    csv_path = out_dir / "per_sample_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary = build_summary(df, out_dir)
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"完成: 有效 {summary['num_valid']}/{summary['num_total']} "
          f"(成功率 {summary['evaluation_success_rate']:.1%})")
    print(f"Seg CLIP-I: mean={summary['seg_clip_i']['mean']} "
          f"median={summary['seg_clip_i']['median']} std={summary['seg_clip_i']['std']}")
    print(f"Seg DINO-I: mean={summary['seg_dino_i']['mean']} "
          f"median={summary['seg_dino_i']['median']} std={summary['seg_dino_i']['std']}")
    if "clip_human_spearman" in summary:
        print(f"Spearman(人工, CLIP) = {summary['clip_human_spearman']}")
        print(f"Spearman(人工, DINO) = {summary['dino_human_spearman']}")
    print(f"失败分布: {summary['failure_counts']}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
