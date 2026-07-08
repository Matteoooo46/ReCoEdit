#!/usr/bin/env python3
"""
对 merged_scored_v4.html 中的 21 个产品做 prompt rewrite (caption)。
使用 Qwen3-VL-8B-Instruct + CAPTION_PROMPT 对每张生成图片进行描述，
输出 LlamaFactory 训练格式。

每台机器 8 GPU 并行，中断自动续跑。
"""

import os, json, re, time, argparse, traceback
from io import BytesIO
from typing import Optional, List
from PIL import Image
import torch
import torch.multiprocessing as mp
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================
# 配置
# ============================================
MODEL_PATH = "/data/phd/kousiqi/zhitao/models/Qwen3-VL-8B-Instruct"
RESULTS_BASE = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
OUTPUT_DIR = "/data/phd/kousiqi/zhitao"
OUTPUT_FILE = "llama_factory_21products_rewriter_train.json"
CHECKPOINT_DIR = "/data/phd/kousiqi/zhitao/caption_checkpoints"

NUM_GPUS = 8
IMAGE_MAX_SIZE = (1024, 1024)
SAVE_INTERVAL = 20

# 从 HTML 中提取的 21 个产品 ID
PRODUCT_IDS = [
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae", "009785b8-36c6-4775-ada0-c9497e7072c2",
    "010ccf98-82c3-40f9-8284-65518eeff3a0", "024419f0-d5c4-482e-9572-ba7885cdf4e4",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8", "02cdc727-ab85-4692-8ef6-00b725c64141",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd", "03641bdb-7a11-4c05-83a3-347d535e8c91",
    "03d7f47e-bb37-4a19-8685-0e231f933627", "03dac08d-8b1e-4a3b-a2d7-94e0ae9ee787",
    "056924c6-1454-4e99-a40c-b8e0b362529c", "064eb1bd-65cd-4fc5-8316-a172aa2f8f2f",
    "07b28d8b-d588-4683-b55d-83d59a89a9b0", "07b41236-33e3-4fec-bf34-c500ef7fb220",
    "08936a21-6a4a-40c1-828f-f30763afdf02",
    "workspace_25833632310597_1772644206", "workspace_4959165917841_1772653742",
    "workspace_item_21825264046857", "workspace_item_23397479040558",
    "workspace_item_25917400924058", "workspace_item_25936355083926",
]

# 与训练数据一致的 system prompt
PROMPT_REWRITE_SYSTEM = """
你是一个电商商品图像编辑 prompt 改写助手。你的任务是根据商品参考图和 original prompt，生成一段更清晰、更具体、更适合图像编辑模型执行的中文 prompt。

请遵守以下原则：

1. 以 original prompt 的编辑意图为主。
保留并扩写 original prompt 中的场景、背景、光线、氛围、构图、人物、动作、道具和风格要求。不要改变原始编辑目标，不要把不同 prompt 都改写成相似的商品展示图。

2. 适度补充商品细节。
根据参考图描述商品本体的关键视觉特征，包括商品类别、整体形状、主色/辅色、主要材质、明显 logo/文字/图案、最重要的结构细节。商品细节要足够帮助模型保持商品一致性，但不要写成参考图长 caption。

3. 不要过度描述参考图。
参考图只用于识别商品本体。不要描述参考图里的原始背景、桌面、墙面、光照、阴影、拍摄角度、摆放方式、手、模特或装饰物，除非 original prompt 明确要求保留。

4. 控制内容比例和长度。
最终 prompt 中，编辑目标约占 50%，商品细节约占 40%，质量约束约占 10%。总长度控制在 120～150 个中文字符。不要列小标题，不要分点输出。

5. 质量和约束。
保持商品颜色、形状、logo、文字、图案和材质尽量与参考图一致；不要新增无关文字、水印、乱码；不要让商品变形、镜像、错色或丢失关键标识。只描述单帧静态画面。

只输出一条最终 prompt，格式如下：
i2v描述：<优化后的 prompt>"""

CAPTION_PROMPT = """你是一个专业的图像视觉描述专家。请仔细观察提供的图片，并严格按照以下维度和顺序，输出一段高质量的图像描述（Caption）：

1. 主体描述：明确谁/什么是画面主体，写清外观特征、服装、材质、姿态以及其在画面中的位置关系。
2. 场景描述：明确主体所处环境，包括背景、前景、空间结构、相关道具以及光线环境。
3. 风格/视觉特征：总结真实画面的视觉特征（如写实、广告质感、电影感、未来感、极简、工业风、复古等）。注意：除非画面本身明显呈现虚构艺术风格，否则请客观描述真实质感，不要虚构艺术流派。
4. 镜头语言：说明景别（如特写、中近景、远景）、视角（俯视、平视、仰视）以及构图方式（居中构图、对称构图、三分法等）。
5. 氛围词：精准提取画面传达的情绪与氛围（如冷峻、温暖、神秘、梦幻、克制、高级、紧张、宁静等）。
6. 细节修饰：补充可见的微观细节，如材质纹理、高光、反射、景深、边缘虚化、烟雾、水波、光晕等。

要求：
请将以上元素有机融合，写成一段连贯、丰富且结构清晰的自然语言描述。不要输出"主体："、"场景："这样的维度标题，直接输出最终的整段描述文本即可。"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_task_list():
    """扫描 21 个产品目录，构建待处理任务列表。
    每个任务: {pid, frame_index, nano_img, ref_img, original_prompt}
    """
    tasks = []
    for pid in PRODUCT_IDS:
        # 找 rewrite_optimized 目录
        variant_dirs = []
        for suffix in ["_rewrite_optimized", "_rewrite_optimized_apg",
                        "_rewrite_optimized_grpo_round2", "_rewrite_optimized_initial"]:
            d = os.path.join(RESULTS_BASE, f"{pid}{suffix}")
            if os.path.isdir(d):
                variant_dirs.append(d)
        if not variant_dirs:
            print(f"  ⚠️ {pid}: 无 rewrite 目录，跳过")
            continue
        variant_dir = variant_dirs[0]

        # 读 prompt_rewrite_log
        log_path = os.path.join(variant_dir, f"{pid}_prompt_rewrite_log.json")
        if not os.path.exists(log_path):
            # 可能的命名变体
            for f in os.listdir(variant_dir):
                if f.endswith("_prompt_rewrite_log.json"):
                    log_path = os.path.join(variant_dir, f)
                    break
        if not os.path.exists(log_path):
            print(f"  ⚠️ {pid}: 无 prompt_rewrite_log.json，跳过")
            continue

        with open(log_path) as f:
            log = json.load(f)

        # 找 ref 图
        ref_img = None
        for f in sorted(os.listdir(variant_dir)):
            if 'ref' in f.lower() and f.endswith(('.png', '.jpg', '.jpeg')):
                ref_img = os.path.join(variant_dir, f)
                break

        for entry in log:
            frame_idx = entry.get("frame_index", 0)
            original_prompt = entry.get("original_prompt", "")

            # 清理 prompt 中的后缀标记
            original_prompt = re.sub(r'[,，]\s*Cinematic.*$', '', original_prompt).strip()

            # 找对应的 nano 图片
            nano_img = None
            for f in sorted(os.listdir(variant_dir)):
                if f"nano_{frame_idx+1}" in f or f == f"p1_nano_{frame_idx+1}.png":
                    nano_img = os.path.join(variant_dir, f)
                    break

            if not nano_img:
                # fallback: 找任意 p1_nano_ 图片
                nano_imgs = sorted([f for f in os.listdir(variant_dir)
                                    if 'nano' in f and f.endswith('.png')])
                if frame_idx < len(nano_imgs):
                    nano_img = os.path.join(variant_dir, nano_imgs[frame_idx])

            if not nano_img or not os.path.exists(nano_img):
                continue

            tasks.append({
                "pid": pid,
                "frame_index": frame_idx,
                "nano_img": nano_img,
                "ref_img": ref_img,
                "original_prompt": original_prompt,
            })

    return tasks


def _worker_fn(gpu_id: int, tasks: List[dict], tag: str):
    """GPU worker：加载模型，逐张图片生成 caption"""
    worker_name = f"[GPU {gpu_id}]"
    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE.replace('.json','')}{tag}_gpu{gpu_id}.json")
    ckpt_dir = CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_21products{tag}_gpu{gpu_id}.txt")

    done_count = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            done_count = int(f.read().strip())

    total = len(tasks)
    if done_count >= total:
        print(f"{worker_name} 全部 {total} 条已完成，跳过")
        return

    print(f"{worker_name} 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        device_map={"": gpu_id}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载已有结果
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    while len(results) < done_count:
        results.append(None)

    print(f"{worker_name} 处理 {done_count}/{total} -> {total}")

    for idx in range(done_count, total):
        task = tasks[idx]
        nano_img = task["nano_img"]
        ref_img = task["ref_img"]
        original_prompt = task["original_prompt"]

        caption = ""
        try:
            img = Image.open(nano_img).convert("RGB")
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
                generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0][input_len:]
            caption = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception as e:
            caption = f"ERROR: {e}"

        # LlamaFactory 格式
        n_images = 1 if ref_img and os.path.exists(ref_img) else 0
        image_tokens = "\n".join(["<image>"] * n_images) if n_images > 0 else ""
        instruction = f"{image_tokens}\n{PROMPT_REWRITE_SYSTEM}" if image_tokens else PROMPT_REWRITE_SYSTEM

        result = {
            "instruction": instruction,
            "input": original_prompt,
            "output": caption,
            "images": [ref_img] if n_images > 0 else [],
            "_pid": task["pid"],
            "_frame": task["frame_index"],
            "_nano_img": nano_img,
        }
        results.append(result)

        if (idx + 1) % SAVE_INTERVAL == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            with open(ckpt_path, "w") as f:
                f.write(str(idx + 1))
            print(f"{worker_name} {idx+1}/{total} ({(idx+1)/total*100:.1f}%) — 已保存")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    with open(ckpt_path, "w") as f:
        f.write(str(total))
    print(f"{worker_name} 完成! {total}/{total}")


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    # 1. 构建任务列表
    print("扫描 21 个产品目录...")
    tasks = build_task_list()
    print(f"共 {len(tasks)} 条任务")
    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"限制为 {len(tasks)} 条")

    # 统计各产品
    from collections import Counter
    pid_counts = Counter(t["pid"] for t in tasks)
    for pid, n in sorted(pid_counts.items()):
        print(f"  {pid}: {n} 条")

    # 2. 按 GPU 分配
    num_gpus = args.num_gpus
    chunk_size = (len(tasks) + num_gpus - 1) // num_gpus
    processes = []
    tag = ""  # 不区分多机

    for g in range(num_gpus):
        c_start = g * chunk_size
        c_end = min(len(tasks), (g + 1) * chunk_size)
        if c_start >= c_end:
            continue
        chunk = tasks[c_start:c_end]
        p = mp.Process(target=_worker_fn, args=(g, chunk, tag), name=f"GPU-{g}")
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        print(f"\n⚠️ {len(failed)} 个 GPU worker 异常退出!")
        return

    # 3. 合并
    print("\n合并 GPU 结果...")
    merged = []
    for g in range(num_gpus):
        f = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE.replace('.json','')}_gpu{g}.json")
        if os.path.exists(f):
            with open(f) as fh:
                merged.extend(json.load(fh))
            print(f"  GPU {g}: 累计 {len(merged):,} 条")

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in merged if not r["output"].startswith("ERROR"))
    print(f"\n✅ 完成! 成功 {success}/{len(merged)}, 输出: {out_path}")


if __name__ == "__main__":
    main()
