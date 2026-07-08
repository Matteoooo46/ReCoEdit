"""
产品一致性 VLM 检测脚本：先检测图中产品，再评分一致性。

用法：
  python test_product_consistency_reward.py \
      --ref-image /data/phd/kousiqi/zhitao/new_validation_set/009785b8-36c6-4775-ada0-c9497e7072c2/reference_imgs/reference_img_0.png \
      --gen-image /data/phd/kousiqi/zhitao/new_validation_set/009785b8-36c6-4775-ada0-c9497e7072c2/augmented_generated_images/0_0.png \
      --product-name "春夏季松紧腰直筒棉麻长裤男士宽松薄款透气苎麻休闲裤子百搭复古"

  # 用训练集中的样本测试
  python test_product_consistency_reward.py --from-dataset --idx 0
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO

from openai import OpenAI
from PIL import Image

VLM_BASE_URL = "http://10.15.2.90:8080/v1"
VLM_API_KEY = "flowgrpo"
VLM_MODEL = "Qwen3-VL-30B-A3B-Instruct"
DATASET_DIR = "/data/phd/jinjiachun/zzt/dual_grpo/flow_grpo/dataset/product_consistency"


def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def detect_product(image: Image.Image, product_name: str) -> str:
    """让 VLM 在图中定位并描述产品。"""
    prompt = f"""请仔细观察这张图片，找到与以下产品名称对应的产品：「{product_name}」

请描述你在图中看到的产品外观，包括：
1. 产品是否出现在图中
2. 产品的颜色、款式、图案等外观特征
3. 产品在图中的位置（如人物穿着、手持、放置等）

请简洁回答。"""

    client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=60.0)
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": pil_to_base64(image)}},
                ],
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def score_consistency(ref_image: Image.Image, gen_image: Image.Image, product_name: str) -> dict:
    """评估两张图中产品的一致性。"""
    scoring_prompt = f"""你是一位专业的商品图像质量审核专家。你的任务是比较两张图片中的同一件产品，评估它们的外观一致性。

## 输入
- 产品名称：{product_name}
- 图片1：产品参考图
- 图片2：生成的图片

## 评分标准（1-5分）
请仔细观察图片2中与「{product_name}」对应的产品，与图片1中的产品进行外观对比，判断它们是否是同一件产品：

- **1分**：完全不一致。图片2中找不到该产品，或产品外观完全不同（颜色、形状、款式都不同）。
- **2分**：大部分不一致。产品有轻微相似（如同一类商品），但颜色、款式、细节明显不同。
- **3分**：部分一致。产品的大致类型和主要颜色正确，但款式细节有明显差异（如领口不同、图案不同、材质不同）。
- **4分**：基本一致。产品外观高度相似，仅有细微差异（如细微色差、小细节不同）。
- **5分**：完全一致。产品外观几乎完全匹配，颜色、款式、图案、细节都高度一致。

## 重要规则
- 你必须根据产品名称在两张图中定位产品，而不是整图对比
- 如果图片2中人物穿戴了该产品，重点对比穿戴的产品而非人物
- 如果图片中产品被遮挡，基于可见部分评分
- 严格评分，5分应非常罕见

## 输出格式
仅输出一行，格式如下：
Score: X
（X为1-5的整数）"""

    client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=60.0)
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": scoring_prompt},
                    {"type": "image_url", "image_url": {"url": pil_to_base64(ref_image)}},
                    {"type": "image_url", "image_url": {"url": pil_to_base64(gen_image)}},
                ],
            },
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"Score:\s*(\d)", raw)
    score = int(match.group(1)) if match else -1
    return {"raw_text": raw, "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-image", help="产品参考图路径")
    ap.add_argument("--gen-image", help="生成图路径")
    ap.add_argument("--product-name", help="产品名称")
    ap.add_argument("--from-dataset", action="store_true", help="从训练集 val 中取样本测试")
    ap.add_argument("--idx", type=int, default=0, help="从数据集取第几条（配合 --from-dataset）")
    ap.add_argument("--skip-detect", action="store_true", help="跳过产品检测步骤，只评分")
    args = ap.parse_args()

    if args.from_dataset:
        jsonl_path = os.path.join(DATASET_DIR, "val_metadata.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if args.idx >= len(lines):
            print(f"idx={args.idx} 超出范围（共 {len(lines)} 条）")
            return
        sample = json.loads(lines[args.idx])
        ref_path = sample.get("product_ref_for_reward", "")
        product_name = sample.get("product_name", "")
        gt_path = sample.get("gt_image", "")

        print(f"=== 数据集样本 #{args.idx} ===")
        print(f"  product_name: {product_name}")
        print(f"  product_ref:  {ref_path}")
        print(f"  gt_image:     {gt_path}")
        print(f"  prompt:       {sample.get('prompt', '')[:100]}...")
        print()

        if not ref_path or not os.path.exists(ref_path):
            # fallback: 用 edit_image 第一张
            edit_images = sample.get("images", [])
            if edit_images and os.path.exists(edit_images[0]):
                ref_path = edit_images[0]
                print(f"  [fallback] 用 edit_image[0] 作为参考图: {ref_path}")
            else:
                print("  错误：找不到产品参考图")
                return

        if not gt_path or not os.path.exists(gt_path):
            print("  错误：找不到 GT 图")
            return

        ref_image = Image.open(ref_path).convert("RGB")
        gen_image = Image.open(gt_path).convert("RGB")
        product_name = product_name

    elif args.ref_image and args.gen_image and args.product_name:
        ref_image = Image.open(args.ref_image).convert("RGB")
        gen_image = Image.open(args.gen_image).convert("RGB")
        product_name = args.product_name
    else:
        print("请指定 --ref-image/--gen-image/--product-name 或 --from-dataset")
        return

    print(f"产品名称: {product_name}")
    print(f"参考图尺寸: {ref_image.size}")
    print(f"生成图尺寸: {gen_image.size}")
    print()

    # Step 1: 检测产品
    if not args.skip_detect:
        print("=" * 50)
        print("Step 1: 检测参考图中的产品")
        print("=" * 50)
        ref_desc = detect_product(ref_image, product_name)
        print(ref_desc)
        print()

        print("=" * 50)
        print("Step 2: 检测生成图中的产品")
        print("=" * 50)
        gen_desc = detect_product(gen_image, product_name)
        print(gen_desc)
        print()

    # Step 2: 评分一致性
    print("=" * 50)
    print("Step 3: 产品一致性评分")
    print("=" * 50)
    result = score_consistency(ref_image, gen_image, product_name)
    print(f"VLM 原始输出: {result['raw_text']}")
    print(f"一致性分数: {result['score']}/5  (归一化: {(result['score'] - 1) / 4.0:.2f})")


if __name__ == "__main__":
    main()
