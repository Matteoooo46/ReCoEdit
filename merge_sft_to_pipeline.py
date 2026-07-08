"""
将 SFT 单文件 checkpoint 合并到 QwenImageEdit pipeline 目录。

读取 step-40000.safetensors，按原 pipeline 的 weight_map 拆分为 5 个分片，
输出到新的 pipeline 目录供 RL 训练使用。

用法：
  python merge_sft_to_pipeline.py
"""

import json
import os
import shutil
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file

SFT_CHECKPOINT = "/data/phd/kousiqi/zhitao/full_all_products_rewrite_expanded_ultimate_resume_132000/step-40000.safetensors"
SRC_PIPELINE = "/data/phd/jinjiachun/zzt/dual_grpo/qwen-image-edit-grpo"
DST_PIPELINE = "/data/phd/jinjiachun/zzt/dual_grpo/qwen-image-edit-grpo-sft-40k"


def main():
    print(f"SFT checkpoint: {SFT_CHECKPOINT}")
    print(f"Source pipeline: {SRC_PIPELINE}")
    print(f"Output pipeline: {DST_PIPELINE}")

    # 1. 复制整个 pipeline 目录（text_encoder, vae, tokenizer 等不变）
    if os.path.exists(DST_PIPELINE):
        print(f"输出目录已存在: {DST_PIPELINE}")
        resp = input("是否覆盖 transformer 权重？(y/n): ").strip().lower()
        if resp != "y":
            print("取消")
            return
    else:
        print("复制 pipeline 目录...")
        shutil.copytree(SRC_PIPELINE, DST_PIPELINE)

    # 2. 读取原 transformer 的 weight_map（确定每个 key 在哪个分片）
    index_path = os.path.join(SRC_PIPELINE, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]  # key -> shard_filename

    # 按分片文件名分组 key
    shard_keys = defaultdict(list)
    for key, shard_file in weight_map.items():
        shard_keys[shard_file].append(key)

    print(f"原分片数: {len(shard_keys)}")
    for shard_file, keys in sorted(shard_keys.items()):
        print(f"  {shard_file}: {len(keys)} keys")

    # 3. 读取 SFT checkpoint 并拆分写入
    print("读取 SFT checkpoint...")
    sft_reader = safe_open(SFT_CHECKPOINT, framework="pt")

    transformer_dir = os.path.join(DST_PIPELINE, "transformer")
    total_size = 0

    for shard_file, keys in sorted(shard_keys.items()):
        print(f"写入 {shard_file} ({len(keys)} keys)...", end=" ", flush=True)
        shard_tensors = {}
        for key in keys:
            shard_tensors[key] = sft_reader.get_tensor(key)

        shard_size = sum(t.nelement() * t.element_size() for t in shard_tensors.values())
        total_size += shard_size

        out_path = os.path.join(transformer_dir, shard_file)
        save_file(shard_tensors, out_path)
        del shard_tensors
        print(f"done ({shard_size / 1e9:.2f} GB)")

    # 4. 更新 index.json 的 metadata
    index_data["metadata"]["total_size"] = total_size
    with open(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\n完成！总大小: {total_size / 1e9:.2f} GB")
    print(f"Pipeline 目录: {DST_PIPELINE}")
    print(f"RL config 中设置 pretrained.model = {DST_PIPELINE}")


if __name__ == "__main__":
    main()
