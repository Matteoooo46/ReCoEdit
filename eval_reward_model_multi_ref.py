"""
批量评测产品一致性 Reward Model（多参考图版本）。

与 eval_reward_model.py 的区别：
  - 加载目录下所有 p1_ref_*.png 作为参考图（而非只用 p1_ref_0.png）
  - 每张生成图与所有参考图逐一对比打分，取最高分作为最终得分
  - 解决"不确定哪张参考图中的产品是生成目标"的问题

数据来源：qwen_inference_results_single 下的 *_data_enhanced_prompt_enhanced 目录
- p1_ref_*.png   : 参考图（全部加载，取最高匹配分）
- p1_my_*.png    : 自己模型生成的图（预期分数低）
- p1_nano_*.png  : nano banana 生成的图（预期分数高）

用法：
  python eval_reward_model_multi_ref.py              # 全部产品
  python eval_reward_model_multi_ref.py --dry-run 2  # 只跑前2个产品快速验证
"""

import argparse
import base64
import json
import os
import re
from io import BytesIO
from collections import defaultdict

# 绕过代理，直连本机 vllm 服务
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_var, None)

from openai import OpenAI
from PIL import Image

VLM_BASE_URL = "http://10.15.2.90:8080/v1"
VLM_API_KEY  = "flowgrpo"
VLM_MODEL    = "Qwen3-VL-30B-A3B-Instruct"
RESULTS_DIR  = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
VAL_DIR      = "/data/phd/kousiqi/zhitao/new_validation_set"

SCORING_PROMPT = """你是一位专业的商品图像质量审核专家。比较两张图片中同一件产品的外观一致性。

产品名称：{product_name}
图片1：产品参考图
图片2：生成的图片

评分标准（1-5分）：
- 1分：完全不一致。图片2中找不到该产品，或外观完全不同。
- 2分：大部分不一致。有轻微相似，但颜色、款式、细节明显不同。
- 3分：部分一致。大致类型和主要颜色正确，但款式细节有明显差异。
- 4分：基本一致。外观高度相似，仅有细微差异。
- 5分：完全一致。颜色、款式、图案、细节都高度一致。

规则：根据产品名称定位产品；若人物穿戴该产品，重点对比产品本身；严格评分，5分极罕见。

仅输出一行：Score: X（X为1-5整数）"""


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"


def score_pair(client, product_name, ref_img, gen_img) -> int:
    """单张参考图 vs 单张生成图，返回 1-5 分或 -1（出错）"""
    prompt = SCORING_PROMPT.format(product_name=product_name)
    try:
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": pil_to_base64(ref_img)}},
                {"type": "image_url", "image_url": {"url": pil_to_base64(gen_img)}},
            ]}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"Score:\s*(\d)", raw)
        return int(m.group(1)) if m else -1
    except Exception as e:
        print(f"        [ERROR] {e}")
        return -1


def score_against_refs(client, product_name, ref_imgs, gen_img, ref_names):
    """多张参考图逐一对比，取最高分。

    参数:
        client:       OpenAI client
        product_name: 产品名称
        ref_imgs:     参考图 PIL Image 列表
        gen_img:      待评分生成图
        ref_names:    参考图文件名列表（用于输出）

    返回:
        (best_score, best_ref_name, all_scores)
        - best_score:     最高分 (1-5 或 -1)
        - best_ref_name:  最高分对应的参考图文件名
        - all_scores:     [(ref_name, score), ...] 每张参考图的分数
    """
    all_scores = []
    best_score = -1
    best_ref_name = "N/A"

    for i, ref_img in enumerate(ref_imgs):
        score = score_pair(client, product_name, ref_img, gen_img)
        all_scores.append((ref_names[i], score))
        if score > best_score:
            best_score = score
            best_ref_name = ref_names[i]

    return best_score, best_ref_name, all_scores


def get_product_name(prod_id):
    info_path = os.path.join(VAL_DIR, prod_id, "item_info.json")
    if os.path.exists(info_path):
        return json.load(open(info_path)).get("itemTitle", prod_id)
    return prod_id


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


def fmt_scores(scores):
    valid = [s for s in scores if s > 0]
    if not valid:
        return "无有效分数"
    avg = sum(valid) / len(valid)
    dist = {i: valid.count(i) for i in range(1, 6) if valid.count(i) > 0}
    return f"均分={avg:.2f}  分布={dist}  (共{len(valid)}张)"


def main():
    ap = argparse.ArgumentParser(
        description="多参考图版本：所有 p1_ref_*.png 逐一对比，取最高分")
    ap.add_argument("--dry-run", type=int, default=None, metavar="N",
                    help="只跑前 N 个产品（快速验证）")
    args = ap.parse_args()

    client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=90.0)

    # 收集所有有效目录
    all_dirs = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if d.endswith("_data_enhanced_prompt_enhanced")
        and not os.path.basename(d).startswith("raw")
        and "_raw_" not in d
    ])

    # 过滤：至少有一张参考图和一张 my 生成图
    valid_dirs = []
    for d in all_dirs:
        prod_path = os.path.join(RESULTS_DIR, d)
        try:
            files = os.listdir(prod_path)
        except Exception:
            continue
        has_ref  = any(f.startswith("p1_ref_") and f.endswith(".png") for f in files)
        has_imgs = any(f.startswith("p1_my_")  and f.endswith(".png") for f in files)
        if has_ref and has_imgs:
            valid_dirs.append(d)

    if args.dry_run:
        valid_dirs = valid_dirs[:args.dry_run]

    print(f"共 {len(valid_dirs)} 个产品待评测\n")

    # 汇总用
    all_my_scores   = []
    all_nano_scores = []
    product_results = []

    for idx, dirname in enumerate(valid_dirs, 1):
        prod_id   = dirname.replace("_data_enhanced_prompt_enhanced", "")
        prod_path = os.path.join(RESULTS_DIR, dirname)
        prod_name = get_product_name(prod_id)

        # 图片列表
        files     = os.listdir(prod_path)
        my_imgs   = sorted([f for f in files if f.startswith("p1_my_")   and f.endswith(".png")], key=natural_sort_key)
        nano_imgs = sorted([f for f in files if f.startswith("p1_nano_") and f.endswith(".png")], key=natural_sort_key)
        ref_imgs  = sorted([f for f in files if f.startswith("p1_ref_")  and f.endswith(".png")], key=natural_sort_key)

        # 加载所有参考图
        ref_images = [Image.open(os.path.join(prod_path, f)).convert("RGB") for f in ref_imgs]

        # ── 产品头部 ──
        print(f"{'='*65}")
        print(f"[{idx}/{len(valid_dirs)}] {prod_name[:50]}")
        print(f"  参考图: {len(ref_imgs)} 张  ({', '.join(ref_imgs)})")
        print(f"  待评分: my={len(my_imgs)}张  nano={len(nano_imgs)}张")
        print(f"  预计 VLM 调用: {(len(my_imgs)+len(nano_imgs))*len(ref_imgs)} 次")
        print(f"{'='*65}")

        my_scores   = []
        nano_scores = []

        # ── 评分 p1_my ──
        print("  【p1_my — 自己模型生成】")
        for fname in my_imgs:
            gen_img = Image.open(os.path.join(prod_path, fname)).convert("RGB")
            best_score, best_ref, all_scores = score_against_refs(
                client, prod_name, ref_images, gen_img, ref_names=ref_imgs)
            norm = f"{(best_score-1)/4:.2f}" if best_score > 0 else " N/A"
            my_scores.append(best_score)

            print(f"    {fname:<24}  Best: {best_score}/5  (via {best_ref})  归一化: {norm}")
            if len(ref_imgs) > 1:
                details = "  ".join(f"{name}:{s}" for name, s in all_scores if s > 0)
                print(f"      → 各参考图分数: {details}")

        # ── 评分 p1_nano ──
        print("  【p1_nano — nano banana 生成】")
        for fname in nano_imgs:
            gen_img = Image.open(os.path.join(prod_path, fname)).convert("RGB")
            best_score, best_ref, all_scores = score_against_refs(
                client, prod_name, ref_images, gen_img, ref_names=ref_imgs)
            norm = f"{(best_score-1)/4:.2f}" if best_score > 0 else " N/A"
            nano_scores.append(best_score)

            print(f"    {fname:<24}  Best: {best_score}/5  (via {best_ref})  归一化: {norm}")
            if len(ref_imgs) > 1:
                details = "  ".join(f"{name}:{s}" for name, s in all_scores if s > 0)
                print(f"      → 各参考图分数: {details}")

        # ── 本产品汇总 ──
        print(f"\n  ── 本产品汇总 ──")
        print(f"  p1_my  : {fmt_scores(my_scores)}")
        print(f"  p1_nano: {fmt_scores(nano_scores)}")

        all_my_scores.extend(my_scores)
        all_nano_scores.extend(nano_scores)
        product_results.append({
            "name":       prod_name[:50],
            "my_scores":  my_scores,
            "nano_scores": nano_scores,
        })

    # ═══════════════════════════════════════════════
    # 总体统计
    # ═══════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("总体统计")
    print(f"{'='*65}")

    valid_my   = [s for s in all_my_scores   if s > 0]
    valid_nano = [s for s in all_nano_scores if s > 0]

    print(f"\n【p1_my（自己模型）】")
    print(f"  有效评分: {len(valid_my)} 张")
    if valid_my:
        avg = sum(valid_my) / len(valid_my)
        dist = {i: valid_my.count(i) for i in range(1, 6)}
        print(f"  全局均分: {avg:.3f}/5  (归一化: {(avg-1)/4:.3f})")
        print(f"  分布: 1分={dist.get(1,0)}  2分={dist.get(2,0)}  3分={dist.get(3,0)}  4分={dist.get(4,0)}  5分={dist.get(5,0)}")

    print(f"\n【p1_nano（nano banana）】")
    print(f"  有效评分: {len(valid_nano)} 张")
    if valid_nano:
        avg = sum(valid_nano) / len(valid_nano)
        dist = {i: valid_nano.count(i) for i in range(1, 6)}
        print(f"  全局均分: {avg:.3f}/5  (归一化: {(avg-1)/4:.3f})")
        print(f"  分布: 1分={dist.get(1,0)}  2分={dist.get(2,0)}  3分={dist.get(3,0)}  4分={dist.get(4,0)}  5分={dist.get(5,0)}")

    if valid_my and valid_nano:
        diff = sum(valid_nano)/len(valid_nano) - sum(valid_my)/len(valid_my)
        print(f"\n  nano均分 - my均分 = {diff:+.3f}  {'✓ nano更高（符合预期）' if diff > 0 else '✗ 与预期相反'}")

    # ── 各产品均分对比 ──
    print(f"\n{'='*65}")
    print("各产品均分对比")
    print(f"{'='*65}")
    print(f"  {'产品名称':<45}  {'my均分':>6}  {'nano均分':>8}  {'差值':>6}")
    print(f"  {'-'*45}  {'------':>6}  {'--------':>8}  {'------':>6}")
    for r in product_results:
        vm = [s for s in r["my_scores"]   if s > 0]
        vn = [s for s in r["nano_scores"] if s > 0]
        am = sum(vm)/len(vm) if vm else float("nan")
        an = sum(vn)/len(vn) if vn else float("nan")
        diff_str = f"{an-am:+.2f}" if vm and vn else "  N/A"
        am_str = f"{am:.2f}" if vm else "  N/A"
        an_str = f"{an:.2f}" if vn else "  N/A"
        print(f"  {r['name']:<45}  {am_str:>6}  {an_str:>8}  {diff_str:>6}")


if __name__ == "__main__":
    main()
