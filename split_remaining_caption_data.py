#!/usr/bin/env python3
"""
筛选原始 metadata 中尚未被训练数据覆盖的记录（即还没做完 caption 的部分），
均分为6份，供 2×H200 + 4×4090 重跑。
匹配逻辑：训练数据 images[0] ∈ 原始 metadata edit_image
"""

import json
import os

METADATA_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate_with_ultraedit500k.json"
TRAINING_JSONL = "/data/phd/kousiqi/zhitao/llama_factory_rewriter_train.jsonl"
OUTPUT_DIR = "/data/phd/kousiqi/zhitao"

NUM_SPLITS = 6

def main():
    # 1. 从训练数据构建已覆盖集合
    # 匹配 key: (edit_image[0], original_prompt), 避免同产品多帧被误判覆盖
    print("[1/4] 读取训练数据，构建已覆盖集合...")
    covered = set()
    with open(TRAINING_JSONL) as f:
        for line in f:
            try:
                rec = json.loads(line)
                imgs = rec.get("images", [])
                inp = rec.get("input", "")
                if imgs:
                    covered.add((imgs[0], inp))
            except Exception:
                pass
    print(f"  训练数据: {len(covered):,} 条已覆盖")

    # 2. 读取原始 metadata，筛选未覆盖的记录
    print("[2/4] 读取原始 metadata，筛选未覆盖记录...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    print(f"  原始数据: {len(metadata):,} 条")

    remaining = []
    for rec in metadata:
        edit_images = rec.get("edit_image", [])
        prompt_text = rec.get("prompt", "") or rec.get("original_prompt", "")
        # 与训练数据同 key: (edit_image[0], prompt) — 注意用 prompt 非 original_prompt
        key = (edit_images[0] if edit_images else "", prompt_text)
        if key not in covered:
            remaining.append(rec)

    print(f"  未覆盖: {len(remaining):,} 条 ({len(remaining)/len(metadata)*100:.1f}%)")

    if not remaining:
        print("✅ 所有数据已覆盖，无需重跑!")
        return

    # 3. 均分为6份
    print(f"[3/4] 均分为 {NUM_SPLITS} 份...")
    chunk_size = (len(remaining) + NUM_SPLITS - 1) // NUM_SPLITS
    splits = []
    for i in range(NUM_SPLITS):
        start = i * chunk_size
        end = min(len(remaining), (i + 1) * chunk_size)
        splits.append(remaining[start:end])
        print(f"  split_{i}: [{start:,}..{end:,}) = {len(splits[-1]):,} 条")

    # 4. 写出6个输入文件
    print("[4/4] 写出输入文件...")
    names = ["h200_0", "h200_1", "4090_m0", "4090_m1", "4090_m2", "4090_m3"]
    for i, name in enumerate(names):
        path = os.path.join(OUTPUT_DIR, f"metadata_caption_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits[i], f, ensure_ascii=False)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {name}: {len(splits[i]):,} 条 -> {path} ({size_mb:.1f}MB)")

    # 打印启动命令
    print("\n" + "=" * 60)
    print("🚀 各机器启动命令:")
    print("=" * 60)
    print()
    print("# --- 在 H200 机器 0 上执行 ---")
    print("bash run_caption_all.sh h200 0")
    print()
    print("# --- 在 H200 机器 1 上执行 ---")
    print("bash run_caption_all.sh h200 1")
    print()
    print("# --- 在 4090 机器 0 上执行 ---")
    print("bash run_caption_all.sh 4090 0")
    print()
    print("# --- 在 4090 机器 1 上执行 ---")
    print("bash run_caption_all.sh 4090 1")
    print()
    print("# --- 在 4090 机器 2 上执行 ---")
    print("bash run_caption_all.sh 4090 2")
    print()
    print("# --- 在 4090 机器 3 上执行 ---")
    print("bash run_caption_all.sh 4090 3")

    # 运行时注意事项
    print()
    print("⚠️  注意:")
    print("  1. 启动前确认各机器的 caption_checkpoints/ 目录中没有旧进度文件")
    print("     删除命令: rm /data/phd/kousiqi/zhitao/caption_checkpoints/caption_ckpt_<TAG>_gpu*.txt")
    print("  2. 全部完成后，执行合并:")
    print("     cd /data/phd/kousiqi/zhitao && python3 qwenvl_rewrite.py --merge-only \\")
    print("       --tags h200_0 h200_1 4090_m0 4090_m1 4090_m2 4090_m3")


if __name__ == "__main__":
    main()
