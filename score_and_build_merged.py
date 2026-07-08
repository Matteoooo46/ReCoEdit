#!/usr/bin/env python3
"""
对 21 产品的 rewrite_optimized 目录中所有 nano/my 图片打分，
然后将分数嵌入合并 HTML 中，在每张图片下方标注得分。
"""
import os, json, re, time, base64, argparse
from io import BytesIO
from collections import defaultdict

# 绕过代理
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_var, None)

from openai import OpenAI
from PIL import Image
from blobstore import BlobStoreClient

VLM_BASE_URL = "http://10.15.2.90:8080/v1"
VLM_API_KEY = "flowgrpo"
VLM_MODEL = "Qwen3-VL-30B-A3B-Instruct"
RESULTS_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
OUTPUT_BASE_DIR = RESULTS_DIR
RUN_TAG = "rewrite_optimized"
NEW_RUN_TAG = "scored_v4"  # 多参考图评分版本

BLOBSTORE_BUCKET = "ad-nieuwland-material"
BLOBSTORE_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

PRODUCTS = [
    "workspace_item_23397479040558", "workspace_item_25936355083926",
    "workspace_item_25917400924058", "workspace_25833632310597_1772644206",
    "workspace_4959165917841_1772653742", "workspace_item_21825264046857",
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae", "03d7f47e-bb37-4a19-8685-0e231f933627",
    "009785b8-36c6-4775-ada0-c9497e7072c2", "03dac08d-8b1e-4a3b-a2d7-94e0ae9ee787",
    "010ccf98-82c3-40f9-8284-65518eeff3a0", "056924c6-1454-4e99-a40c-b8e0b362529c",
    "024419f0-d5c4-482e-9572-ba7885cdf4e4", "064eb1bd-65cd-4fc5-8316-a172aa2f8f2f",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8", "07b28d8b-d588-4683-b55d-83d59a89a9b0",
    "02cdc727-ab85-4692-8ef6-00b725c64141", "07b41236-33e3-4fec-bf34-c500ef7fb220",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd", "08936a21-6a4a-40c1-828f-f30763afdf02",
    "03641bdb-7a11-4c05-83a3-347d535e8c91",
]

SCORES_FILE = os.path.join(OUTPUT_BASE_DIR, "all_scores_rewrite_optimized_merged_scored_v1.json")  # 固定文件名
MERGED_HTML_PATH = os.path.join(OUTPUT_BASE_DIR, f"merged_{NEW_RUN_TAG}.html")

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


def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"


def score_pair(client, product_name, ref_img, gen_img):
    try:
        prompt = SCORING_PROMPT.format(product_name=product_name)
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
    """多张参考图逐一对比，取最高分"""
    best_score = -1
    best_ref_name = "N/A"
    for i, ref_img in enumerate(ref_imgs):
        s = score_pair(client, product_name, ref_img, gen_img)
        if s > best_score:
            best_score = s
            best_ref_name = ref_names[i]
    return best_score, best_ref_name


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


def upload_files(file_paths, blob_prefix):
    blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
    path_to_url = {}
    valid_paths = list(set(p for p in file_paths if p and os.path.exists(p)))
    print(f"☁️  正在上传 {len(valid_paths)} 个文件到 {blob_prefix} ...")
    for idx, file_path in enumerate(valid_paths):
        try:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            bs_key = f"{blob_prefix}/{parent_dir}/{file_name}"
            blobstore.upload_binary_to_s3(file_path, bs_key)
            path_to_url[file_path] = f"{BLOBSTORE_CDN}/{bs_key}"
            if (idx + 1) % 50 == 0:
                print(f"  已上传 {idx + 1}/{len(valid_paths)} ...")
        except Exception as e:
            print(f"  [上传失败] {file_path}: {e}")
            path_to_url[file_path] = ""
    print(f"✅ 上传完成: {len(valid_paths)} 个文件")
    return path_to_url


def build_scored_html(products_config):
    """生成带评分的 HTML，JSON 直写为 JS 变量"""
    config_json = json.dumps(products_config, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电商图片生成效果对比 - 21产品（含评分）</title>
    <style>
        :root {{
            --bg-color: #f5f6f7; --card-bg: #ffffff; --text-main: #1f2329; --text-secondary: #8f959e;
            --border-color: #dee0e3; --primary-color: #3370ff; --rewrite-color: #00b96b;
            --score-high: #00b96b; --score-mid: #faad14; --score-low: #ff4d4f;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }}
        .summary {{ background: var(--card-bg); padding: 15px 25px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
        .summary table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        .summary th, .summary td {{ padding: 6px 12px; text-align: center; border-bottom: 1px solid var(--border-color); }}
        .summary th {{ background: #fafafa; font-weight: 600; }}
        .product-section {{ background: var(--card-bg); border-radius: 12px; padding: 25px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .product-title {{ margin-top: 0; color: var(--primary-color); border-bottom: 1px dashed var(--border-color); padding-bottom: 10px; }}
        .section-subtitle {{ font-size: 16px; color: #555; margin: 20px 0 10px 0; }}
        .global-inputs {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .global-img-box {{ width: 140px; text-align: center; font-size: 13px; color: var(--text-secondary); }}
        .global-img-box img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color); margin-bottom: 5px; }}
        .grid-container {{ display: flex; gap: 20px; overflow-x: auto; padding-bottom: 15px; }}
        .frame-column {{ flex: 0 0 280px; background: #fafafa; border-radius: 8px; border: 1px solid var(--border-color); padding: 15px; display: flex; flex-direction: column; }}
        .prompt-text {{ font-size: 14px; color: #d83931; line-height: 1.5; margin-bottom: 15px; min-height: 80px; white-space: pre-wrap; }}
        .original-prompt {{ font-size: 12px; color: #646a73; line-height: 1.4; margin-bottom: 10px; white-space: pre-wrap; }}
        .image-compare-box {{ margin-bottom: 15px; }}
        .image-compare-box h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--text-main); }}
        .image-compare-box img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border-color); cursor: pointer; transition: transform 0.2s; }}
        .image-compare-box img:hover {{ transform: scale(1.02); }}
        .my-model-title {{ color: var(--primary-color) !important; }}
        .score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 13px; margin-top: 4px; }}
        .score-1, .score-2 {{ background: #fff2f0; color: var(--score-low); }}
        .score-3 {{ background: #fffbe6; color: var(--score-mid); }}
        .score-4, .score-5 {{ background: #f6ffed; color: var(--score-high); }}
        .product-avg {{ font-size: 13px; color: var(--text-secondary); margin-top: 4px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>🛍️ 电商图片编辑 - 21产品对比（含 Reward 评分）</h2>
    </div>
    <div class="summary" id="summaryBox"></div>
    <div id="mainContent">正在加载...</div>
</div>

<script>
var PRODUCTS_DATA = {config_json};

(function() {{
    function escapeHtml(text) {{
        if (!text) return '';
        return String(text).replace(/[&<>"]/g, function(ch) {{
            return {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}}[ch];
        }});
    }}

    function scoreBadge(score) {{
        if (score == null || score < 0) return '';
        var cls = 'score-badge score-' + score;
        return '<span class="' + cls + '">Score: ' + score + '/5</span>';
    }}

    function render() {{
        var container = document.getElementById('mainContent');
        var summaryHtml = '<table><tr><th>产品</th><th>帧数</th><th>My 均分</th><th>Nano 均分</th><th>差值</th></tr>';
        container.innerHTML = '';

        PRODUCTS_DATA.forEach(function(product) {{
            // Product section
            var html = '<div class="product-section"><h2 class="product-title">' + escapeHtml(product.title) + '</h2><h3 class="section-subtitle">Global input:</h3><div class="global-inputs">';
            product.globals.forEach(function(g) {{
                html += '<div class="global-img-box"><img src="' + g.path + '" alt="' + escapeHtml(g.name) + '" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=图片丢失\\'"><div>' + escapeHtml(g.name) + '</div></div>';
            }});
            html += '</div><h3 class="section-subtitle">Performance comparison:</h3><div class="grid-container">';

            var myScores = [], nanoScores = [];

            product.frames.forEach(function(frame) {{
                var originalHtml = frame.original_prompt ? '<div class="original-prompt"><b>Original:</b> ' + escapeHtml(frame.original_prompt) + '</div>' : '';
                var myScoreHtml = scoreBadge(frame.my_score);
                var nanoScoreHtml = scoreBadge(frame.nano_score);
                if (frame.my_score != null && frame.my_score >= 0) myScores.push(frame.my_score);
                if (frame.nano_score != null && frame.nano_score >= 0) nanoScores.push(frame.nano_score);

                html += '<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> ' + escapeHtml(frame.prompt) + '</div>' + originalHtml
                    + '<div class="image-compare-box"><h4>nano banana:</h4><img src="' + frame.nano_path + '" alt="Nano" onerror="this.src=\\'https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失\\'">' + nanoScoreHtml + '</div>'
                    + '<div class="image-compare-box"><h4 class="my-model-title">My Model:</h4><img src="' + frame.my_path + '" alt="My Model" onerror="this.src=\\'https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失\\'">' + myScoreHtml + '</div>'
                    + '</div>';
            }});
            html += '</div></div>';
            container.innerHTML += html;

            // Summary row
            var myAvg = myScores.length ? (myScores.reduce(function(a,b){{return a+b;}},0)/myScores.length).toFixed(2) : 'N/A';
            var nanoAvg = nanoScores.length ? (nanoScores.reduce(function(a,b){{return a+b;}},0)/nanoScores.length).toFixed(2) : 'N/A';
            var diff = (myScores.length && nanoScores.length) ? (nanoAvg - myAvg).toFixed(2) : 'N/A';
            var diffColor = diff > 0 ? 'color:var(--score-high)' : 'color:var(--score-low)';
            summaryHtml += '<tr><td style="text-align:left">' + escapeHtml(product.title) + '</td><td>' + product.frames.length + '</td><td>' + myAvg + '</td><td>' + nanoAvg + '</td><td style="' + diffColor + '">' + (diff >= 0 ? '+' : '') + diff + '</td></tr>';
        }});

        summaryHtml += '</table>';
        document.getElementById('summaryBox').innerHTML = summaryHtml;
    }}

    render();
}})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-score", action="store_true", help="跳过打分，只用已有分数重建 HTML")
    args = ap.parse_args()

    all_scores = {}

    if not args.skip_score:
        # ========================================
        # Phase 1: 打分
        # ========================================
        client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=120.0)
        total_pairs = 0

        for pi, product_name in enumerate(PRODUCTS):
            report_dir = os.path.join(RESULTS_DIR, f"{product_name}_{RUN_TAG}")
            if not os.path.isdir(report_dir):
                print(f"❌ {product_name}: 目录不存在")
                continue

            # 找所有参考图（多参考图：加载全部 p1_ref_*.png）
            files = os.listdir(report_dir)
            ref_files = sorted([f for f in files if f.startswith("p1_ref_") and f.endswith(".png")])
            my_files = sorted([f for f in files if f.startswith("p1_my_") and f.endswith(".png")], key=natural_sort_key)
            nano_files = sorted([f for f in files if f.startswith("p1_nano_") and f.endswith(".png")], key=natural_sort_key)

            if not ref_files:
                print(f"⚠️ {product_name}: 无参考图，跳过")
                continue

            ref_images = [Image.open(os.path.join(report_dir, f)).convert("RGB") for f in ref_files]
            prod_name_short = product_name[:40]
            prod_scores = {"my": {}, "nano": {}}

            est_calls = (len(my_files) + len(nano_files)) * len(ref_files)
            print(f"\n[{pi+1}/{len(PRODUCTS)}] {product_name}  (refs={len(ref_files)}, my={len(my_files)}, nano={len(nano_files)}, VLM calls≈{est_calls})")

            # 打分 my（多参考图取最高）
            for fname in my_files:
                gen_img = Image.open(os.path.join(report_dir, fname)).convert("RGB")
                best_score, best_ref = score_against_refs(client, prod_name_short, ref_images, gen_img, ref_files)
                prod_scores["my"][fname] = best_score
                total_pairs += len(ref_files)
                print(f"  my  {fname}: {best_score}/5  (via {best_ref})")

            # 打分 nano（多参考图取最高）
            for fname in nano_files:
                gen_img = Image.open(os.path.join(report_dir, fname)).convert("RGB")
                best_score, best_ref = score_against_refs(client, prod_name_short, ref_images, gen_img, ref_files)
                prod_scores["nano"][fname] = best_score
                total_pairs += len(ref_files)
                print(f"  nano {fname}: {best_score}/5  (via {best_ref})")

            all_scores[product_name] = prod_scores

        # 保存分数
        with open(SCORES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 打分完成: {total_pairs} 对, 保存至 {SCORES_FILE}")
    else:
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            all_scores = json.load(f)
        print(f"✅ 从 {SCORES_FILE} 加载已有分数")

    # ========================================
    # Phase 2: 构建带分数的 HTML
    # ========================================
    all_products_config = []
    all_upload_paths = []

    for product_name in PRODUCTS:
        report_dir = os.path.join(RESULTS_DIR, f"{product_name}_{RUN_TAG}")
        viewer_config_path = os.path.join(report_dir, f"{product_name}_viewer_config.json")

        if not os.path.exists(viewer_config_path):
            print(f"❌ {product_name}: viewer_config 不存在")
            continue

        with open(viewer_config_path, "r", encoding="utf-8") as f:
            product_config = json.load(f)

        prod_scores = all_scores.get(product_name, {"my": {}, "nano": {}})
        my_scores = prod_scores.get("my", {})
        nano_scores = prod_scores.get("nano", {})

        # Globals
        ref_files = sorted([fn for fn in os.listdir(report_dir) if fn.startswith("p1_ref_") and fn.endswith(".png")])
        char_files = sorted([fn for fn in os.listdir(report_dir) if fn.startswith("p1_char") and fn.endswith((".png", ".jpg"))])
        new_globals = []
        for idx, rf in enumerate(ref_files):
            local = os.path.join(report_dir, rf)
            all_upload_paths.append(local)
            new_globals.append({"name": f"产品参考{idx+1}", "path": local})
        for cf in char_files:
            local = os.path.join(report_dir, cf)
            all_upload_paths.append(local)
            new_globals.append({"name": "人物参考", "path": local})

        # Frames with scores
        new_frames = []
        raw_dir = os.path.join(RESULTS_DIR, f"{product_name}_raw_{RUN_TAG}")
        frames_from_config = product_config.get("frames", [])

        for fi, frame in enumerate(frames_from_config):
            # Find local nano/my files
            frame_num = fi + 1  # p1_nano_1.png, p1_my_1.png
            nano_local = os.path.join(report_dir, f"p1_nano_{frame_num}.png")
            my_local = os.path.join(report_dir, f"p1_my_{frame_num}.png")

            if not os.path.exists(nano_local):
                nano_local = None
            if not os.path.exists(my_local):
                my_local = None

            # Try raw_dir too
            if not my_local:
                raw_my = os.path.join(raw_dir, f"frame_{fi}.jpg")
                if os.path.exists(raw_my):
                    my_local = raw_my

            if nano_local:
                all_upload_paths.append(nano_local)
            if my_local:
                all_upload_paths.append(my_local)

            # Look up scores
            nano_fname = f"p1_nano_{frame_num}.png"
            my_fname = f"p1_my_{frame_num}.png"
            my_score = my_scores.get(my_fname, -1)
            nano_score = nano_scores.get(nano_fname, -1)

            new_frames.append({
                "prompt": frame.get("prompt", ""),
                "original_prompt": frame.get("original_prompt", ""),
                "nano_path": nano_local if nano_local else "",
                "my_path": my_local if my_local else "",
                "nano_score": nano_score if nano_score > 0 else None,
                "my_score": my_score if my_score > 0 else None,
            })

        all_products_config.append({
            "title": f"{product_name} [{NEW_RUN_TAG}]",
            "globals": new_globals,
            "frames": new_frames,
        })
        print(f"✅ {product_name}: {len(new_globals)} refs, {len(new_frames)} frames")

    # Upload
    print(f"\n📤 共 {len(all_upload_paths)} 个文件待上传")
    merged_blob_prefix = f"qwen_inference/merged/{NEW_RUN_TAG}"
    url_map = upload_files(all_upload_paths, merged_blob_prefix)

    # Replace paths with CDN URLs
    for pc in all_products_config:
        for g in pc["globals"]:
            if g["path"] in url_map:
                g["path"] = url_map[g["path"]]
        for f in pc["frames"]:
            if f["nano_path"] in url_map:
                f["nano_path"] = url_map[f["nano_path"]]
            if f["my_path"] in url_map:
                f["my_path"] = url_map[f["my_path"]]

    # Build HTML
    html_text = build_scored_html(all_products_config)
    with open(MERGED_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"✅ HTML 已保存: {MERGED_HTML_PATH}")

    # Upload HTML
    try:
        blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
        html_bs_key = f"{merged_blob_prefix}/viewer/{os.path.basename(MERGED_HTML_PATH)}"
        blobstore.upload_binary_to_s3(MERGED_HTML_PATH, html_bs_key)
        html_url = f"{BLOBSTORE_CDN}/{html_bs_key}"
        tf = sum(len(p["frames"]) for p in all_products_config)
        print(f"\n{'='*60}")
        print(f"🎉 完成！{len(all_products_config)} 产品, {tf} 帧（含评分）")
        print(f"🌐 {html_url}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"❌ HTML 上传失败: {e}")


if __name__ == "__main__":
    main()
