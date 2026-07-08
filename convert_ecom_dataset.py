#!/usr/bin/env python3
"""
转换旧数据集 (llama_factory_rewriter_train_expanded.json) → 新数据集
复用方案: 替换 instruction (system prompt) + 修改 input/output 前缀
"""
import json
import sys
import time

INPUT_PATH = "/data/phd/kousiqi/zhitao/llama_factory_rewriter_train_expanded.json"
OUTPUT_PATH = "/data/phd/kousiqi/zhitao/llama_factory_ecom_rewriter_train.json"

# 来自 qwen_image_edit_2511_inference_apg.py:66-139
# 来自 qwen_image_edit_2511_inference_apg.py:66-88（精简版）
NEW_SYSTEM_PROMPT = """\
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


def transform_item(item: dict) -> dict:
    """将旧数据条目转换为新格式。"""
    # 1. 替换 instruction → 新的 system prompt (保留 <image> token)
    item["instruction"] = "<image>\n" + NEW_SYSTEM_PROMPT

    # 2. input: 【基础提示词】：xxx → 原始 prompt：xxx
    old_input = item.get("input", "")
    if old_input.startswith("【基础提示词】："):
        item["input"] = "原始 prompt：" + old_input[len("【基础提示词】："):]
    elif old_input.startswith("【基础提示词】:"):
        item["input"] = "原始 prompt：" + old_input[len("【基础提示词】:"):]
    else:
        # 没有前缀的，直接加
        item["input"] = "原始 prompt：" + old_input

    # 3. output: 加 i2v描述： 前缀（如果没有）
    old_output = item.get("output", "")
    if not old_output.startswith("i2v描述："):
        item["output"] = "i2v描述：" + old_output

    # 4. images: 只保留第一张图（匹配 instruction 中的单个 <image> token）
    images = item.get("images", [])
    if isinstance(images, list) and len(images) > 1:
        item["images"] = images[:1]

    return item


def main():
    print(f"读取旧数据集: {INPUT_PATH}")
    t0 = time.time()
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载完成: {len(data)} 条, 耗时 {time.time() - t0:.1f}s")

    print("开始转换...")
    t1 = time.time()
    for i, item in enumerate(data):
        transform_item(item)
        if (i + 1) % 50000 == 0:
            print(f"  已处理 {i + 1}/{len(data)} ({time.time() - t1:.1f}s)")

    print(f"转换完成, 耗时 {time.time() - t1:.1f}s")

    print(f"写入新数据集: {OUTPUT_PATH}")
    t2 = time.time()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"写入完成, 耗时 {time.time() - t2:.1f}s")

    # 验证前 3 条
    print("\n=== 验证 (前2条) ===")
    for j in range(min(2, len(data))):
        item = data[j]
        print(f"[{j}] instruction[:80]: {item['instruction'][:80]}")
        print(f"[{j}] input[:80]:      {item['input'][:80]}")
        print(f"[{j}] output[:80]:     {item['output'][:80]}")
        print(f"[{j}] images:          {item.get('images', [])}")
        print()

    in_size = __import__("os").path.getsize(INPUT_PATH)
    out_size = __import__("os").path.getsize(OUTPUT_PATH)
    print(f"文件大小: {in_size / 1e9:.1f}GB → {out_size / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
