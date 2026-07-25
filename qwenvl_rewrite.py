#!/usr/bin/env python3
"""
使用 Qwen3-VL-8B-Instruct (transformers 直接加载) 对 metadata JSON 中
每一条数据的 image 进行 Caption 描述。

无需 vLLM，直接用 transformers + multiprocessing 多 GPU 并行。
支持自动断点续跑、多机分片、非均匀数据切分。

==== 用法 ====

  单机 (8 GPU), 处理全部:
    python qwenvl_rewrite.py

  均匀分片 (N 台机器):
    python qwenvl_rewrite.py --shard 0 --num-shards 4

  非均匀切分 (指定数据范围 + 自定义标签):
    python qwenvl_rewrite.py --data-start 0 --data-end 571886 --tag h200
    python qwenvl_rewrite.py --data-start 571886 --data-end 714857 --tag 4090_m0

  完成后合并:
    python qwenvl_rewrite.py --merge-only --tags h200 4090_m0 4090_m1 4090_m2 4090_m3

  测试少量:
    python qwenvl_rewrite.py --limit 100 --num-gpus 1
"""

import os
import json
import time
import argparse
import traceback
from io import BytesIO
from typing import Optional, List, Tuple
from functools import partial

import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm


# ==========================================
# 默认配置
# ==========================================
MODEL_PATH = "/data/phd/kousiqi/zhitao/models/Qwen3-VL-8B-Instruct"
INPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate_with_ultraedit500k.json"
OUTPUT_DIR = "/data/phd/kousiqi/zhitao"
OUTPUT_BASENAME = "metadata_all_products_expanded_ultimate_with_ultraedit500k_captioned"

NUM_GPUS_DEFAULT = 8
MAX_NEW_TOKENS = 512
IMAGE_MAX_SIZE = (1024, 1024)

# ==========================================
# Caption Prompt
# ==========================================
CAPTION_PROMPT = """你是一个专业的图像视觉描述专家。请仔细观察提供的图片，并严格按照以下维度和顺序，输出一段高质量的图像描述（Caption）：

1. 主体描述：明确谁/什么是画面主体，写清外观特征、服装、材质、姿态以及其在画面中的位置关系。
2. 场景描述：明确主体所处环境，包括背景、前景、空间结构、相关道具以及光线环境。
3. 风格/视觉特征：总结真实画面的视觉特征（如写实、广告质感、电影感、未来感、极简、工业风、复古等）。注意：除非画面本身明显呈现虚构艺术风格，否则请客观描述真实质感，不要虚构艺术流派。
4. 镜头语言：说明景别（如特写、中近景、远景）、视角（俯视、平视、仰视）以及构图方式（居中构图、对称构图、三分法等）。
5. 氛围词：精准提取画面传达的情绪与氛围（如冷峻、温暖、神秘、梦幻、克制、高级、紧张、宁静等）。
6. 细节修饰：补充可见的微观细节，如材质纹理、高光、反射、景深、边缘虚化、烟雾、水波、光晕等。

要求：
请将以上元素有机融合，写成一段连贯、丰富且结构清晰的自然语言描述。不要输出"主体："、"场景："这样的维度标题，直接输出最终的整段描述文本即可。"""


# ==========================================
# 命令行参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Batch image captioning with Qwen3-VL-8B-Instruct (transformers)")
    parser.add_argument("--input", type=str, default=INPUT_JSON_PATH)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output-basename", type=str, default=OUTPUT_BASENAME)
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS_DEFAULT,
                        help="本机使用的 GPU 数量")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)

    # 数据范围：两种方式二选一
    # 方式 1: 非均匀切分
    parser.add_argument("--data-start", type=int, default=None,
                        help="全局起始索引 (闭区间), 与 --data-end 配合")
    parser.add_argument("--data-end", type=int, default=None,
                        help="全局结束索引 (开区间), 与 --data-start 配合")
    # 方式 2: 均匀分片
    parser.add_argument("--shard", type=int, default=None,
                        help="当前分片编号 (0-based)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="总分片数")

    # 输出标签：自定义文件名后缀
    parser.add_argument("--tag", type=str, default=None,
                        help="输出文件标签，如 h200 / 4090_m0。用于命名输出文件和断点")

    # 合并模式
    parser.add_argument("--merge-only", action="store_true",
                        help="只合并已有结果")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                        help="合并时使用的标签列表，如: --tags h200 4090_m0 4090_m1")

    # 测试
    parser.add_argument("--limit", type=int, default=0,
                        help="最多处理 N 条 (0 = 全部), 在 data-start 基础上生效")

    return parser.parse_args()


# ==========================================
# 单个 GPU 上的 worker 函数
# ==========================================
def _worker_fn(gpu_id: int, task_data: List[dict],
               model_path: str, output_dir: str, output_basename: str,
               tag: str, max_new_tokens: int):
    """
    在一个 GPU 上加载模型，处理分配给它的数据。
    task_data: 完整的 metadata 子集，每条包含原始 JSON 的所有字段
    """
    worker_name = f"[GPU {gpu_id}]"
    suffix = f"_{tag}" if tag else ""
    ckpt_dir = os.path.join(output_dir, "caption_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_basename}{suffix}_gpu{gpu_id}.json")
    checkpoint_path = os.path.join(ckpt_dir, f"caption_ckpt{suffix}_gpu{gpu_id}.txt")

    # ---- 加载断点 ----
    done_count = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            done_count = int(f.read().strip())

    total = len(task_data)

    if done_count >= total:
        print(f"{worker_name} 全部 {total} 条已完成，跳过")
        return output_path, total

    print(f"{worker_name} 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"{worker_name} 模型加载完成，处理 {done_count}/{total} -> {total}")

    # ---- 加载已有结果 ----
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    while len(results) < done_count:
        results.append(None)

    # ---- 逐条处理 ----
    for idx in range(done_count, total):
        item = task_data[idx]
        img_path = item.get("image", "")

        if not img_path or not os.path.exists(img_path):
            new_item = dict(item)
            new_item["caption"] = f"ERROR: Image not found: {img_path}"
            results.append(new_item)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(IMAGE_MAX_SIZE)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(f"cuda:{gpu_id}")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0][input_len:]
            caption = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        except Exception as e:
            caption = f"ERROR: {e}"

        new_item = dict(item)
        new_item["caption"] = caption
        new_item["caption_model"] = os.path.basename(model_path)
        results.append(new_item)

        # 增量保存（每 100 条）
        if (idx + 1) % 100 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            with open(checkpoint_path, "w") as f:
                f.write(str(idx + 1))

            done = idx + 1
            pct = done / total * 100
            print(f"{worker_name} {done}/{total} ({pct:.1f}%) — 已保存")

    # ---- 最终保存 ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    with open(checkpoint_path, "w") as f:
        f.write(str(total))
    print(f"{worker_name} 完成! {total}/{total}")

    return output_path, total


# ==========================================
# 合并函数
# ==========================================
def merge_gpu_results(output_dir: str, output_basename: str, tag: str, num_gpus: int):
    """合并本机各 GPU 的分片结果成一个文件"""
    suffix = f"_{tag}" if tag else ""
    merged = []
    for gpu_id in range(num_gpus):
        gpu_file = os.path.join(output_dir, f"{output_basename}{suffix}_gpu{gpu_id}.json")
        if os.path.exists(gpu_file):
            with open(gpu_file) as f:
                data = json.load(f)
            merged.extend(data)
            print(f"  GPU {gpu_id}: {len(data):,} 条 <- {gpu_file}")
        else:
            print(f"  GPU {gpu_id}: 文件不存在，跳过")

    shard_file = os.path.join(output_dir, f"{output_basename}{suffix}.json")
    with open(shard_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  合并完成: {len(merged):,} 条 -> {shard_file}")
    return merged


def merge_all_tags(args):
    """合并所有 tag 的结果为一个完整 JSON"""
    merged = []
    for tag in args.tags:
        suffix = f"_{tag}"
        tag_file = os.path.join(args.output_dir, f"{args.output_basename}{suffix}.json")
        if not os.path.exists(tag_file):
            print(f"  ⚠️ tag 文件不存在，跳过: {tag_file}")
            continue
        with open(tag_file) as f:
            data = json.load(f)
        merged.extend(data)
        print(f"  {tag}: {len(data):,} 条 <- {tag_file}")

    merged_file = os.path.join(args.output_dir, f"{args.output_basename}.json")
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n合并完成: {len(merged):,} 条 -> {merged_file}")

    with open(args.input) as f:
        original = json.load(f)
    if len(merged) == len(original):
        print("✓ 合并结果与原始数据条数一致")
    else:
        print(f"⚠️ 条数不匹配: 合并 {len(merged):,} vs 原始 {len(original):,}")


# ==========================================
# 主函数
# ==========================================
def main():
    args = parse_args()

    # ---- 合并模式 ----
    if args.merge_only:
        if args.tags:
            print(f"合并 {len(args.tags)} 个 tag: {args.tags}")
            merge_all_tags(args)
        else:
            print("请用 --tags 指定要合并的标签列表")
            print("示例: --merge-only --tags h200 4090_m0 4090_m1 4090_m2 4090_m3")
        return

    # ---- 推理模式 ----
    mp.set_start_method("spawn", force=True)

    # 确定数据范围
    # 优先级: --shard/--num-shards (均匀分片) < --data-start/--data-end (显式范围)
    metadata = None  # 延迟加载

    # 1. 先只加载获取 total
    print(f"[1/3] 加载数据: {args.input}")
    t0 = time.time()
    with open(args.input, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    total = len(metadata)
    print(f"      总记录数: {total:,}, 耗时: {time.time() - t0:.1f}s")

    # 2. 确定范围
    if args.data_start is not None and args.data_end is not None:
        s_start = args.data_start
        s_end = args.data_end
    elif args.shard is not None and args.num_shards is not None:
        shard_size = (total + args.num_shards - 1) // args.num_shards
        s_start = args.shard * shard_size
        s_end = min(total, (args.shard + 1) * shard_size)
    else:
        s_start = 0
        s_end = total

    if args.limit > 0:
        s_end = min(s_end, s_start + args.limit)

    my_data = metadata[s_start:s_end]
    my_total = len(my_data)

    # 确定 tag（文件名后缀）
    if args.tag:
        tag = args.tag
    elif args.shard is not None and args.num_shards is not None:
        tag = f"shard{args.shard}of{args.num_shards}"
    else:
        tag = ""

    print(f"[2/3] 数据范围: [{s_start:,} ~ {s_end:,}) 共 {my_total:,} 条")
    print(f"      标签: {tag or '(无)'}")
    print(f"      模型: {args.model_path}")
    print(f"      GPU 数: {args.num_gpus}")

    # 3. 按 GPU 数量均匀分配数据
    num_gpus = args.num_gpus
    chunk_size = (my_total + num_gpus - 1) // num_gpus
    chunks = []
    for g in range(num_gpus):
        c_start = g * chunk_size
        c_end = min(my_total, (g + 1) * chunk_size)
        if c_start < c_end:
            chunks.append((g, my_data[c_start:c_end]))

    print(f"[3/3] 启动 {len(chunks)} 个 GPU worker, 各自处理 ~{chunk_size:,} 条")
    print()

    # 4. 启动 worker
    processes = []
    for gpu_id, chunk_data in chunks:
        if len(chunk_data) == 0:
            continue
        p = mp.Process(
            target=_worker_fn,
            args=(gpu_id, chunk_data, args.model_path,
                  args.output_dir, args.output_basename,
                  tag, args.max_new_tokens),
            name=f"GPU-{gpu_id}",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        print(f"\n⚠️ {len(failed)} 个 GPU worker 异常退出!")
        for p in failed:
            print(f"  {p.name}: exitcode={p.exitcode}")
        print("修复问题后重新运行相同命令即可断点续跑。")
        return

    # 5. 合并本机各 GPU 结果
    print("\n合并本机各 GPU 结果...")
    merged = merge_gpu_results(args.output_dir, args.output_basename, tag, num_gpus)

    success = sum(1 for r in merged
                  if isinstance(r.get("caption", ""), str)
                  and not r["caption"].startswith("ERROR"))
    failed_count = len(merged) - success

    print(f"\n✅ 本机完成! 成功: {success:,}, 失败: {failed_count:,}")
    print(f"输出: {args.output_dir}/{args.output_basename}{'_' + tag if tag else ''}.json")


if __name__ == "__main__":
    main()
