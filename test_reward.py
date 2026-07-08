"""
测试新版 product_consistency reward model：prompt 预分类 + CoT 图像评分。

测试内容：
  1. prompt 分类 — 判断 prompt 是否要求图片中出现具体产品
  2. 图像评分 (CoT) — 对产品图做一致性二分类 (Yes/No)
  3. 组合测试 — 真实产品数据上的完整评测

用法：
  python test_reward.py              # 全部测试
  python test_reward.py --skip-cls   # 跳过 prompt 分类测试
  python test_reward.py --skip-img   # 跳过图像评分测试
  python test_reward.py --dry-run 2  # 图像评分只跑前2个产品
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO

# 绕过代理，直连 VLM 服务
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_var, None)

from openai import OpenAI
from PIL import Image

VLM_BASE_URL = "http://10.15.2.90:8080/v1"
VLM_API_KEY  = "flowgrpo"
VLM_MODEL    = "Qwen3-VL-30B-A3B-Instruct"

# ─── Prompt 分类 (与 rewards.py 中 CLASSIFY_PROMPT 一致) ───
CLASSIFY_PROMPT = (
    "判断以下图像生成提示词是否要求生成的图片中出现具体的产品（如服装、鞋帽、配饰、"
    "箱包、电子产品、化妆品、食品饮料、家居用品等具体商品）。\n"
    "- 如果提示词要求图片中出现某件具体产品，回复 Yes。\n"
    "- 如果提示词仅描述人物姿势、背景、场景、艺术风格等，没有要求出现具体产品，回复 No。\n\n"
    "提示词：{prompt}\n\n"
    "仅回复 Yes 或 No。"
)

# ─── 图像评分 CoT (与 rewards.py 中 SCORING_PROMPT 一致) ───
SCORING_PROMPT = (
    "你是一位专业的商品图像质量审核专家。请逐步分析两张图片中的产品外观一致性。\n\n"
    "产品名称：{product_name}\n"
    "图片1：产品参考图\n"
    "图片2：生成的图片\n\n"
    "请按以下步骤分析（在最终输出 Yes 或 No 之前，先写出分析过程）：\n"
    "Step 1 — 定位产品：在图片1中描述该产品的关键外观特征（颜色、款式、材质、图案、logo等）。\n"
    "Step 2 — 检查存在性：在图片2中是否能找到该产品？\n"
    "Step 3 — 逐项对比：将图片2中产品的外观与Step 1的关键特征逐项对比（颜色、款式、图案、细节）。\n"
    "Step 4 — 结论：基于以上对比，判断是否一致。\n\n"
    "规则：\n"
    "- 若人物穿戴该产品，重点对比产品本身而非人物。\n"
    "- 严格判断：有明显颜色/款式/图案差异即为 No。\n"
    "- 分析过程的最后一行必须是：Yes 或 No"
)

# ─── 图片数据路径 ───
RESULTS_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
VAL_DIR     = "/data/phd/kousiqi/zhitao/new_validation_set"

# ─── 测试用 prompt 样本 ───
PRODUCT_PROMPTS = [
    "A model wearing a red leather jacket with gold zippers, standing in a studio",
    "A close-up shot of Nike Air Max 270 sneakers on a wooden floor",
    "A woman holding a Louis Vuitton handbag in front of a mirror",
    "An iPhone 15 Pro laying on a marble table next to a coffee cup",
    "A person wearing Ray-Ban Aviator sunglasses on a sunny beach",
]

NON_PRODUCT_PROMPTS = [
    "A person standing with hands on hips, neutral expression, white background",
    "A serene mountain landscape at sunset with pine trees in the foreground",
    "Abstract geometric shapes in pastel colors floating in space",
    "A person walking naturally down a city street, candid shot",
    "A gradient background transitioning from blue to purple, studio lighting",
]


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"


def extract_score(text: str) -> float:
    """与 rewards.py 中 _extract_score 一致"""
    text = text.strip()
    # With CoT, the last non-empty line should be the verdict
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if last_line in ("Yes", "yes", "YES", "No", "no", "NO"):
            return 1.0 if last_line.lower() == "yes" else 0.0
    # Fallback: check if the whole text starts with Yes/No
    if text.startswith("Yes") or text.startswith("yes") or text.startswith("YES"):
        return 1.0
    elif text.startswith("No") or text.startswith("no") or text.startswith("NO"):
        return 0.0
    # Last resort: regex search
    m_yes = re.search(r'\b[Yy]es\b', text)
    m_no = re.search(r'\b[Nn]o\b', text)
    if m_yes and not m_no:
        return 1.0
    elif m_no and not m_yes:
        return 0.0
    return 0.0


# ═══════════════════════════════════════════════════════════════
# 测试 1: Prompt 分类
# ═══════════════════════════════════════════════════════════════

def test_prompt_classification(client):
    print(f"\n{'='*65}")
    print("测试 1: Prompt 预分类（是否涉及产品）")
    print(f"{'='*65}\n")

    all_prompts = PRODUCT_PROMPTS + NON_PRODUCT_PROMPTS
    expected = [True]*5 + [False]*5

    correct = 0
    for i, prompt in enumerate(all_prompts):
        label = "产品 ✓" if expected[i] else "非产品 ✗"
        try:
            resp = client.chat.completions.create(
                model=VLM_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": CLASSIFY_PROMPT.format(prompt=prompt)},
                ]}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            predicted = raw.startswith("Yes") or raw.startswith("yes")
            ok = "✓" if predicted == expected[i] else "✗ MISMATCH"
            if predicted == expected[i]:
                correct += 1
            print(f"  [{ok}] {label}  |  {prompt[:70]}...")
            print(f"         VLM → {raw}")
        except Exception as e:
            print(f"  [ERROR] {prompt[:70]}... → {e}")

    print(f"\n  分类准确率: {correct}/{len(all_prompts)} ({100*correct/len(all_prompts):.0f}%)")


# ═══════════════════════════════════════════════════════════════
# 测试 2: 图像评分 (CoT)
# ═══════════════════════════════════════════════════════════════

def get_product_name(prod_id):
    info_path = os.path.join(VAL_DIR, prod_id, "item_info.json")
    if os.path.exists(info_path):
        return json.load(open(info_path)).get("itemTitle", prod_id)
    return prod_id


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


def test_image_scoring(client, dry_run=None):
    print(f"\n{'='*65}")
    print("测试 2: 图像评分 (CoT)")
    print(f"{'='*65}\n")

    # 收集评测目录
    all_dirs = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if d.endswith("_data_enhanced_prompt_enhanced")
        and not os.path.basename(d).startswith("raw")
        and "_raw_" not in d
    ])
    valid_dirs = []
    for d in all_dirs:
        prod_path = os.path.join(RESULTS_DIR, d)
        has_ref  = any(f.startswith("p1_ref_") and f.endswith(".png") for f in os.listdir(prod_path))
        has_imgs = any(f.startswith("p1_my_")  and f.endswith(".png") for f in os.listdir(prod_path))
        if has_ref and has_imgs:
            valid_dirs.append(d)

    if dry_run:
        valid_dirs = valid_dirs[:dry_run]

    print(f"共 {len(valid_dirs)} 个产品待评测\n")

    all_scores = []
    cot_examples = []  # 保存前几个 CoT 输出供展示

    for idx, dirname in enumerate(valid_dirs, 1):
        prod_id   = dirname.replace("_data_enhanced_prompt_enhanced", "")
        prod_path = os.path.join(RESULTS_DIR, dirname)
        prod_name = get_product_name(prod_id)

        files     = os.listdir(prod_path)
        my_imgs   = sorted([f for f in files if f.startswith("p1_my_")   and f.endswith(".png")], key=natural_sort_key)
        ref_imgs  = sorted([f for f in files if f.startswith("p1_ref_")  and f.endswith(".png")], key=natural_sort_key)
        ref_image = Image.open(os.path.join(prod_path, ref_imgs[0])).convert("RGB")

        print(f"[{idx}/{len(valid_dirs)}] {prod_name[:50]}  (my={len(my_imgs)}张)")

        scores = []
        for fname in my_imgs:
            gen_img = Image.open(os.path.join(prod_path, fname)).convert("RGB")
            prompt = SCORING_PROMPT.format(product_name=prod_name)
            try:
                resp = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": pil_to_base64(ref_image)}},
                        {"type": "image_url", "image_url": {"url": pil_to_base64(gen_img)}},
                    ]}],
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content.strip()
                score = extract_score(raw)
                scores.append(score)

                # 保存前 2 个产品的第 1 张图的完整 CoT 输出
                if idx <= 2 and len(cot_examples) < 4:
                    cot_examples.append({
                        "product": prod_name,
                        "image": fname,
                        "raw_response": raw,
                        "score": score,
                    })
                print(f"    {fname:<20}  Score: {score:.0f}")
            except Exception as e:
                print(f"    {fname:<20}  [ERROR] {e}")
                scores.append(0.0)

        if scores:
            valid = [s for s in scores if s >= 0]
            n_yes = sum(1 for s in valid if s == 1.0)
            n_no  = sum(1 for s in valid if s == 0.0)
            print(f"    均分={sum(valid)/len(valid):.2f}  Yes={n_yes}  No={n_no}")
        all_scores.extend(scores)

    # 总体统计
    valid_all = [s for s in all_scores if s >= 0]
    n_yes = sum(1 for s in valid_all if s == 1.0)
    n_no  = sum(1 for s in valid_all if s == 0.0)
    print(f"\n{'─'*65}")
    print(f"图像评分汇总: {len(valid_all)} 张, Yes={n_yes}, No={n_no}, Yes率={n_yes/len(valid_all):.1%}")

    # 展示 CoT 输出示例
    if cot_examples:
        print(f"\n{'='*65}")
        print("CoT 推理示例（前几组）")
        print(f"{'='*65}")
        for ex in cot_examples:
            print(f"\n  ▸ 产品: {ex['product'][:40]}")
            print(f"  ▸ 图片: {ex['image']}")
            print(f"  ▸ Score: {ex['score']:.0f}")
            print(f"  ▸ VLM 完整输出:")
            for line in ex['raw_response'].split('\n'):
                print(f"      {line}")
            print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cls", action="store_true", help="跳过 prompt 分类测试")
    ap.add_argument("--skip-img", action="store_true", help="跳过图像评分测试")
    ap.add_argument("--dry-run", type=int, default=None, metavar="N",
                    help="图像评分只跑前 N 个产品")
    args = ap.parse_args()

    # 检查 VLM 连通性
    print("检查 VLM 服务连通性...", end=" ")
    try:
        client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=20.0)
        models = client.models.list()
        print(f"✓ (模型: {models.data[0].id})")
    except Exception as e:
        print(f"✗ 失败: {e}")
        print("请确认 10.15.2.90:8080 上的 vLLM 服务已启动")
        sys.exit(1)

    if not args.skip_cls:
        test_prompt_classification(client)

    if not args.skip_img:
        test_image_scoring(client, dry_run=args.dry_run)

    print(f"\n{'='*65}")
    print("全部测试完成")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
