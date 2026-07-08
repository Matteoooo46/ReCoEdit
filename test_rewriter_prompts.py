#!/usr/bin/env python3
"""从训练数据挑 20 条样本，用推理脚本的 VLM 做 prompt 改写，生成对比 HTML"""
import json, os, base64, time, re
from io import BytesIO
from PIL import Image
from openai import OpenAI
import httpx

# ─── 配置 ───
VLM_URL = "http://10.80.243.156:8080/v1"
VLM_MODEL = "Qwen3-VL-30B-A3B-Instruct"
DATA_PATH = "/data/phd/kousiqi/zhitao/llama_factory_ecom_rewriter_train.json"
OUT_HTML = "/data/phd/kousiqi/zhitao/rewriter_test_results.html"
NUM_SAMPLES = 20

# 与推理脚本一致
PROMPT_REWRITE_SYSTEM = """你是一个电商商品图像编辑 prompt 改写助手。你的任务是根据商品参考图和 original prompt，生成一段更清晰、更具体、更适合图像编辑模型执行的中文 prompt。

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

client = OpenAI(base_url=VLM_URL, api_key="EMPTY")
client._client.timeout = httpx.Timeout(120.0)


def img_to_base64(img_path):
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    # Resize to max 512px for HTML display
    img.thumbnail((512, 512))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def rewrite_prompt(ref_img_path, original_prompt):
    """Call VLM to rewrite prompt"""
    try:
        if not os.path.exists(ref_img_path):
            return f"[图片不存在: {ref_img_path}]"

        img_b64 = img_to_base64(ref_img_path)
        if not img_b64:
            return "[图片读取失败]"

        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": f"上面是商品的参考图。\n原始 prompt：{original_prompt}"},
        ]

        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_REWRITE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()
        # Extract "i2v描述：" part
        m = re.search(r"i2v描述\s*[:：]\s*(.+)", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        return text
    except Exception as e:
        return f"[改写失败: {e}]"


# ─── 选取样本 ───
print("Loading data...")
with open(DATA_PATH) as f:
    data = json.load(f)

# Pick diverse samples: spread across the dataset
import random
random.seed(42)
indices = sorted(random.sample(range(len(data)), NUM_SAMPLES))

samples = []
for i in indices:
    d = data[i]
    orig_prompt = d["input"]
    if orig_prompt.startswith("原始 prompt："):
        orig_prompt = orig_prompt[len("原始 prompt："):]
    ref_img = d.get("images", [""])[0] if d.get("images") else ""
    samples.append({
        "idx": i,
        "ref_img": ref_img,
        "original_prompt": orig_prompt,
        "training_output": d["output"].replace("i2v描述：", ""),
    })

# ─── 调用 VLM 改写 ───
print(f"Testing {len(samples)} samples...")
for j, s in enumerate(samples):
    print(f"  [{j+1}/{len(samples)}] idx={s['idx']} ...")
    s["rewritten_prompt"] = rewrite_prompt(s["ref_img"], s["original_prompt"])
    s["ref_img_b64"] = img_to_base64(s["ref_img"])
    time.sleep(0.5)  # rate limit

# ─── 生成 HTML ───
print("Generating HTML...")
rows_html = ""
for j, s in enumerate(samples):
    img_tag = ""
    if s["ref_img_b64"]:
        img_tag = f'<img src="data:image/jpeg;base64,{s["ref_img_b64"]}" class="ref-img">'
    else:
        img_tag = f'<div class="no-img">图片不可用<br><small>{s["ref_img"]}</small></div>'

    rows_html += f"""
    <div class="sample-card">
      <div class="sample-header">#{j+1} — idx={s['idx']}</div>
      <div class="sample-body">
        <div class="ref-col">{img_tag}</div>
        <div class="prompt-col">
          <div class="section">
            <div class="label original-label">Original Prompt</div>
            <div class="content">{s['original_prompt']}</div>
          </div>
          <div class="section">
            <div class="label rewrite-label">Rewritten Prompt (VLM)</div>
            <div class="content highlight">{s['rewritten_prompt']}</div>
          </div>
          <div class="section">
            <div class="label training-label">Training Data Output</div>
            <div class="content dim">{s['training_output']}</div>
          </div>
        </div>
      </div>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rewriter Prompt 改写测试</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 8px; color: #16213e; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }}
.sample-card {{ background: #fff; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); overflow: hidden; }}
.sample-header {{ background: #16213e; color: #fff; padding: 10px 20px; font-weight: 600; font-size: 14px; }}
.sample-body {{ display: flex; gap: 20px; padding: 20px; }}
.ref-col {{ flex: 0 0 280px; }}
.ref-col img {{ width: 100%; border-radius: 8px; border: 1px solid #e0e0e0; }}
.no-img {{ width: 100%; height: 280px; background: #f5f5f5; border: 1px dashed #ccc; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #999; font-size: 13px; }}
.prompt-col {{ flex: 1; display: flex; flex-direction: column; gap: 14px; }}
.section {{ }}
.label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; padding: 3px 10px; border-radius: 4px; display: inline-block; margin-bottom: 6px; }}
.original-label {{ background: #e3f2fd; color: #1565c0; }}
.rewrite-label {{ background: #e8f5e9; color: #2e7d32; }}
.training-label {{ background: #fff3e0; color: #e65100; }}
.content {{ font-size: 14px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }}
.content.highlight {{ background: #f9fdf9; border-left: 3px solid #2e7d32; padding: 10px 14px; border-radius: 0 6px 6px 0; }}
.content.dim {{ color: #999; font-size: 13px; }}
</style>
</head>
<body>
<h1>Rewriter Prompt 改写测试</h1>
<div class="subtitle">{NUM_SAMPLES} 条训练数据 + VLM ({VLM_MODEL}) 改写结果 | 随机采样 | 2026-07-02</div>
{rows_html}
</body>
</html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nDone! HTML saved to: {OUT_HTML}")
print(f"File size: {os.path.getsize(OUT_HTML) / 1e6:.1f} MB")
