"""
把 metadata_all_products_expanded_ultimate.json 转成 RL 训练用的 jsonl。

输出字段（每行一条样本）：
  - item_dir:                产品的唯一 key 目录
  - prompt:                  DiT 用的长 prompt
  - original_prompt:         原始短指令（可选保留）
  - images:                  edit_image 原序，模型输入
  - product_ref_for_reward:  给 reward 用的参考图（VLM 根据产品名识别图中产品后对比）
  - product_name:            产品名称，给 VLM judge 用
  - gt_image:                nano banana / 流水线 GT，仅 eval 可视化用

三种数据源：
  1. kousiqi / lijiahui / zhaoguiqin 2-img: edit_image=[产品图,人物图]
     产品名 → extracted_item_info.json 的 product_info
     产品参考图 → reference_img_0.png
  2. zhaoguiqin 1-img: edit_image=[产品参考图]
     产品名 → extracted_item_info.json 的 product_info
     产品参考图 → edit_image 本身（就是产品参考图）
  3. nanoBanana 1-img: edit_image=[condition.jpg（人物穿着产品）]
     产品名 → info.json 的 item 字段
     产品参考图 → edit_image 本身（condition.jpg，VLM 能根据产品名识别其中的产品）

按 item_dir 切 train/val，避免商品维度的数据泄露。
"""
import argparse
import json
import os
import random
from collections import defaultdict

from tqdm import tqdm

INPUT_METADATA = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate.json"
OUTPUT_DIR = "/data/phd/jinjiachun/zzt/dual_grpo/flow_grpo/dataset/product_consistency"


def find_item_dir(image_path: str, cache: dict, max_levels: int = 8):
    """从图片路径向上回溯，找到第一个含 extracted_item_info.json 的目录。"""
    d = os.path.dirname(image_path)
    for _ in range(max_levels):
        if d in cache:
            return cache[d]
        candidate = os.path.join(d, "extracted_item_info.json")
        if os.path.exists(candidate):
            cache[d] = d
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    cache[os.path.dirname(image_path)] = None
    return None


def find_nanoBanana_asset_dir(image_path: str):
    """从 nanoBanana 图片路径提取 asset 目录 (category_X/asset_Y)。"""
    parts = image_path.split("/")
    for i, p in enumerate(parts):
        if p.startswith("category_") and i + 1 < len(parts) and parts[i + 1].startswith("asset_"):
            return "/".join(parts[: i + 2])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_METADATA)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--val-ratio", type=float, default=0.02,
                    help="按 item 划分验证集的比例")
    ap.add_argument("--min-val-items", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-files", action="store_true",
                    help="逐条 os.path.exists 校验所有输入路径（慢但稳）")
    ap.add_argument("--max-train", type=int, default=0,
                    help="保留训练样本上限（0=不限制，方便先小规模 sanity check）")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Total samples in metadata: {len(data)}")

    # 不再过滤 2-image，所有样本都用
    n_1img = sum(1 for s in data if len(s.get("edit_image", [])) == 1)
    n_2img = sum(1 for s in data if len(s.get("edit_image", [])) == 2)
    n_other = len(data) - n_1img - n_2img
    print(f"1-image samples: {n_1img}")
    print(f"2-image samples: {n_2img}")
    print(f"Other samples: {n_other}")

    item_dir_cache: dict = {}
    item_info_cache: dict = {}
    nanoBanana_info_cache: dict = {}
    ref_exists_cache: dict = {}

    def get_item_product_info(item_dir: str):
        if item_dir in item_info_cache:
            return item_info_cache[item_dir]
        p = os.path.join(item_dir, "extracted_item_info.json")
        info = None
        try:
            with open(p) as f:
                info = json.load(f)
        except Exception as e:
            print(f"[warn] failed to parse {p}: {e}")
        item_info_cache[item_dir] = info
        return info

    def get_nanoBanana_product_info(asset_dir: str):
        if asset_dir in nanoBanana_info_cache:
            return nanoBanana_info_cache[asset_dir]
        p = os.path.join(asset_dir, "info.json")
        info = None
        try:
            with open(p) as f:
                info = json.load(f)
        except Exception as e:
            print(f"[warn] failed to parse {p}: {e}")
        nanoBanana_info_cache[asset_dir] = info
        return info

    def get_product_ref(item_dir: str):
        if item_dir in ref_exists_cache:
            return ref_exists_cache[item_dir]
        p = os.path.join(item_dir, "reference_imgs/reference_img_0.png")
        result = p if os.path.exists(p) else None
        ref_exists_cache[item_dir] = result
        return result

    samples = []
    skipped = defaultdict(int)

    for s in tqdm(data, desc="Building samples"):
        edit_images = s.get("edit_image", [])
        n_img = len(edit_images)
        if n_img == 0:
            skipped["no_edit_image"] += 1
            continue

        is_nanoBanana = any("yaozhengjian" in img for img in edit_images)

        if is_nanoBanana:
            # nanoBanana: product name from info.json, product_ref = edit_image itself
            asset_dir = find_nanoBanana_asset_dir(edit_images[0])
            if not asset_dir:
                skipped["no_asset_dir"] += 1
                continue

            info = get_nanoBanana_product_info(asset_dir)
            if not info or not info.get("item"):
                skipped["no_product_info"] += 1
                continue

            # item_dir 用 asset_dir 作为唯一 key
            item_dir = asset_dir
            product_name = info["item"]
            product_ref = edit_images[0]  # condition.jpg 就是产品参考（VLM 据名称识别）

        else:
            # kousiqi / lijiahui / zhaoguiqin: product info from extracted_item_info.json
            item_dir = find_item_dir(s["image"], item_dir_cache)
            if not item_dir:
                skipped["no_item_dir"] += 1
                continue

            info = get_item_product_info(item_dir)
            if not info or not info.get("product_info"):
                skipped["no_product_info"] += 1
                continue

            product_name = info["product_info"]

            if n_img >= 2:
                # 2-image: 有独立产品参考图
                product_ref = get_product_ref(item_dir)
                if not product_ref:
                    # fallback: edit_image[0] 也可能是产品参考图
                    product_ref = edit_images[0]
            else:
                # 1-image zhaoguiqin: edit_image 本身就是产品参考图
                product_ref = edit_images[0]

        gt = s["image"]

        if args.check_files:
            all_exist = all(os.path.exists(img) for img in edit_images)
            if not all_exist:
                skipped["missing_edit_image"] += 1
                continue
            if not os.path.exists(gt):
                skipped["missing_gt"] += 1
                continue
            if not os.path.exists(product_ref):
                skipped["missing_product_ref"] += 1
                continue

        samples.append({
            "item_dir": item_dir,
            "prompt": s["prompt"],
            "original_prompt": s.get("original_prompt", ""),
            "images": edit_images,
            "product_ref_for_reward": product_ref,
            "product_name": product_name,
            "gt_image": gt,
        })

    print(f"Valid samples: {len(samples)}")
    if skipped:
        print(f"Skipped reasons: {dict(skipped)}")

    # split by item_dir
    items_to_samples = defaultdict(list)
    for s in samples:
        items_to_samples[s["item_dir"]].append(s)

    all_items = sorted(items_to_samples.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_items)

    n_val = max(args.min_val_items, int(len(all_items) * args.val_ratio))
    n_val = min(n_val, len(all_items) - 1)
    val_items = set(all_items[:n_val])
    train_items = set(all_items[n_val:])

    train_samples = [s for it in train_items for s in items_to_samples[it]]
    val_samples = [s for it in val_items for s in items_to_samples[it]]
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    if args.max_train and args.max_train > 0:
        train_samples = train_samples[: args.max_train]

    print(f"Train: {len(train_samples)} samples / {len(train_items)} items")
    print(f"Val:   {len(val_samples)} samples / {len(val_items)} items")

    os.makedirs(args.output_dir, exist_ok=True)
    for name, ss in [("train", train_samples), ("val", val_samples)]:
        out_path = os.path.join(args.output_dir, f"{name}_metadata.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in ss:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}  ({len(ss)} lines)")


if __name__ == "__main__":
    main()
