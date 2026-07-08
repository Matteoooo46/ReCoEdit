#!/usr/bin/env python3
"""对比网页：nano(共用) + Old My + New My，三组都打分"""
import os, json, re, time, base64, argparse
from io import BytesIO

for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"):
    os.environ.pop(v, None)

from openai import OpenAI
from PIL import Image
from blobstore import BlobStoreClient

VLM_URL = "http://10.15.2.90:8080/v1"
VLM_KEY = "flowgrpo"
VLM_MODEL = "Qwen3-VL-30B-A3B-Instruct"
BASE = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
OLD_TAG = "rewrite_optimized_grpo_round2_old"
NEW_TAG = "rewrite_optimized"
CMP_TAG = "cmp_scored_v1"

BLOB_BUCKET = "ad-nieuwland-material"
BLOB_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

PRODUCTS = [
    "workspace_item_23397479040558", "workspace_item_25936355083926",
    "workspace_item_25917400924058", "workspace_25833632310597_1772644206",
    "workspace_4959165917841_1772653742", "workspace_item_21825264046857",
]

SCORES_FILE = os.path.join(BASE, f"scores_{CMP_TAG}.json")
HTML_PATH = os.path.join(BASE, f"comparison_{CMP_TAG}.html")

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


def pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"


def score_pair(client, product_name, ref_img, gen_img):
    prompt = SCORING_PROMPT.format(product_name=product_name)
    try:
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": pil_to_b64(ref_img)}},
                {"type": "image_url", "image_url": {"url": pil_to_b64(gen_img)}},
            ]}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"Score:\s*(\d)", raw)
        return int(m.group(1)) if m else -1
    except Exception as e:
        print(f"      [ERROR] {e}")
        return -1


def upload_files(file_paths, blob_prefix):
    blobstore = BlobStoreClient(BLOB_BUCKET)
    path_to_url = {}
    valid = list(set(p for p in file_paths if p and os.path.exists(p)))
    print(f"☁️  上传 {len(valid)} 文件到 {blob_prefix} ...")
    for idx, fp in enumerate(valid):
        try:
            fn = os.path.basename(fp)
            pd = os.path.basename(os.path.dirname(fp))
            blobstore.upload_binary_to_s3(fp, f"{blob_prefix}/{pd}/{fn}")
            path_to_url[fp] = f"{BLOB_CDN}/{blob_prefix}/{pd}/{fn}"
            if (idx + 1) % 50 == 0:
                print(f"  已上传 {idx + 1}/{len(valid)} ...")
        except Exception as e:
            print(f"  [上传失败] {fp}: {e}")
            path_to_url[fp] = ""
    print(f"✅ 上传完成: {len(valid)}")
    return path_to_url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args()

    all_scores = {}

    if not args.skip_score:
        client = OpenAI(base_url=VLM_URL, api_key=VLM_KEY, timeout=120.0)
        total = 0

        for pi, pn in enumerate(PRODUCTS):
            new_report = os.path.join(BASE, f"{pn}_{NEW_TAG}")
            old_report = os.path.join(BASE, f"{pn}_{OLD_TAG}")

            # 参考图（新旧 report 中 ref 应该一样，用新的）
            refs = sorted([f for f in os.listdir(new_report) if f.startswith("p1_ref_") and f.endswith(".png")])
            if not refs:
                continue
            ref_img = Image.open(os.path.join(new_report, refs[0])).convert("RGB")

            def num_sort(fname, prefix):
                try:
                    return int(fname.replace(prefix, "").replace(".png", ""))
                except:
                    return 0

            # 收集三个组的图片
            nano_files = sorted([f for f in os.listdir(new_report) if f.startswith("p1_nano_") and f.endswith(".png")],
                                key=lambda s: num_sort(s, "p1_nano_"))
            old_my_files = sorted([f for f in os.listdir(old_report) if f.startswith("p1_my_") and f.endswith(".png")],
                                   key=lambda s: num_sort(s, "p1_my_"))
            new_my_files = sorted([f for f in os.listdir(new_report) if f.startswith("p1_my_") and f.endswith(".png")],
                                   key=lambda s: num_sort(s, "p1_my_"))

            prod_scores = {"nano": {}, "old_my": {}, "new_my": {}}
            est = len(nano_files) + len(old_my_files) + len(new_my_files)
            print(f"\n[{pi+1}/{len(PRODUCTS)}] {pn}  (nano={len(nano_files)}, old_my={len(old_my_files)}, new_my={len(new_my_files)}, calls={est})")

            # 评分 nano
            for fn in nano_files:
                img = Image.open(os.path.join(new_report, fn)).convert("RGB")
                s = score_pair(client, pn[:40], ref_img, img)
                prod_scores["nano"][fn] = s
                total += 1
                print(f"  nano {fn}: {s}/5")

            # 评分 old_my
            for fn in old_my_files:
                img = Image.open(os.path.join(old_report, fn)).convert("RGB")
                s = score_pair(client, pn[:40], ref_img, img)
                prod_scores["old_my"][fn] = s
                total += 1
                print(f"  old_my {fn}: {s}/5")

            # 评分 new_my
            for fn in new_my_files:
                img = Image.open(os.path.join(new_report, fn)).convert("RGB")
                s = score_pair(client, pn[:40], ref_img, img)
                prod_scores["new_my"][fn] = s
                total += 1
                print(f"  new_my {fn}: {s}/5")

            all_scores[pn] = prod_scores

        with open(SCORES_FILE, "w") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 打分完成: {total} 对, 保存至 {SCORES_FILE}")
    else:
        with open(SCORES_FILE) as f:
            all_scores = json.load(f)
        print(f"✅ 从 {SCORES_FILE} 加载已有分数")

    # ========================================
    # 构建 HTML
    # ========================================
    all_products = []
    all_upload_paths = []

    for pn in PRODUCTS:
        old_report = os.path.join(BASE, f"{pn}_{OLD_TAG}")
        new_report = os.path.join(BASE, f"{pn}_{NEW_TAG}")
        prompt_log = os.path.join(new_report, f"{pn}_prompt_rewrite_log.json")
        prod_scores = all_scores.get(pn, {})

        # 读取 prompt
        prompts = {}
        if os.path.exists(prompt_log):
            for e in json.load(open(prompt_log)):
                prompts[e.get("frame_index", -1)] = e

        # 参考图（从 new report 取，新旧应该一样）
        ref_imgs = sorted([f for f in os.listdir(new_report) if f.startswith("p1_ref_") and f.endswith(".png")])
        char_imgs = sorted([f for f in os.listdir(new_report) if f.startswith("p1_char") and f.endswith((".png", ".jpg"))])
        globals_list = []
        for i, r in enumerate(ref_imgs):
            globals_list.append((f"产品参考{i+1}", os.path.join(new_report, r)))
        for c in char_imgs:
            globals_list.append(("人物参考", os.path.join(new_report, c)))
        for _, p in globals_list:
            all_upload_paths.append(p)

        # nano 文件列表（从 new report）
        nano_files = sorted([f for f in os.listdir(new_report) if f.startswith("p1_nano_") and f.endswith(".png")],
                            key=lambda s: int(s.replace("p1_nano_","").replace(".png","")) if s.replace("p1_nano_","").replace(".png","").isdigit() else 0)

        # 帧数取最大
        max_f = max(len(nano_files),
                     len([f for f in os.listdir(old_report) if f.startswith("p1_my_")]),
                     len([f for f in os.listdir(new_report) if f.startswith("p1_my_")]))

        frames = []
        for i in range(max_f):
            num = i + 1
            nano_fn = f"p1_nano_{num}.png"
            my_fn = f"p1_my_{num}.png"

            nano_local = os.path.join(new_report, nano_fn) if os.path.exists(os.path.join(new_report, nano_fn)) else ""
            old_my_local = os.path.join(old_report, my_fn) if os.path.exists(os.path.join(old_report, my_fn)) else ""
            new_my_local = os.path.join(new_report, my_fn) if os.path.exists(os.path.join(new_report, my_fn)) else ""

            for p in [nano_local, old_my_local, new_my_local]:
                if p: all_upload_paths.append(p)

            nano_score = prod_scores.get("nano", {}).get(nano_fn, -1)
            old_score = prod_scores.get("old_my", {}).get(my_fn, -1)
            new_score = prod_scores.get("new_my", {}).get(my_fn, -1)

            pi_info = prompts.get(i, {})
            frames.append({
                "prompt": pi_info.get("rewritten_prompt", ""),
                "original_prompt": pi_info.get("original_prompt", ""),
                "nano": nano_local, "old_my": old_my_local, "new_my": new_my_local,
                "nano_score": nano_score if nano_score > 0 else None,
                "old_score": old_score if old_score > 0 else None,
                "new_score": new_score if new_score > 0 else None,
            })

        all_products.append({
            "title": pn,
            "globals": [{"name": n, "path": p} for n, p in globals_list],
            "frames": frames,
        })
        # compute avg
        ns = [f["nano_score"] for f in frames if f["nano_score"]]
        os_ = [f["old_score"] for f in frames if f["old_score"]]
        ns_ = [f["new_score"] for f in frames if f["new_score"]]
        na = sum(ns)/len(ns) if ns else 0
        oa = sum(os_)/len(os_) if os_ else 0
        na2 = sum(ns_)/len(ns_) if ns_ else 0
        print(f"✅ {pn}: nano_avg={na:.2f} old_avg={oa:.2f} new_avg={na2:.2f}")

    # 上传
    print(f"\n📤 共 {len(all_upload_paths)} 文件")
    blob_prefix = f"qwen_inference/comparison/{CMP_TAG}"
    url_map = upload_files(all_upload_paths, blob_prefix)

    # 替换路径
    for prod in all_products:
        for g in prod["globals"]:
            if g["path"] in url_map: g["path"] = url_map[g["path"]]
        for f in prod["frames"]:
            for k in ["nano", "old_my", "new_my"]:
                if f[k] in url_map: f[k] = url_map[f[k]]

    # 生成 HTML
    config_json = json.dumps(all_products, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GRPO Round2 对比（含评分）</title>
<style>
:root {{ --bg:#f5f6f7; --card:#fff; --text:#1f2329; --sub:#8f959e; --border:#dee0e3; --pri:#3370ff; --old:#faad14; --new:#00b96b; --score-hi:#00b96b; --score-mid:#faad14; --score-lo:#ff4d4f; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg); color:var(--text); margin:0; padding:20px; }}
.container {{ max-width:1600px; margin:0 auto; }}
h2 {{ color:var(--pri); }}
.product-section {{ background:var(--card); border-radius:12px; padding:25px; margin-bottom:40px; box-shadow:0 4px 12px rgba(0,0,0,.08); }}
.product-title {{ margin-top:0; color:var(--pri); border-bottom:1px dashed var(--border); padding-bottom:10px; }}
.global-inputs {{ display:flex; gap:15px; margin-bottom:20px; flex-wrap:wrap; }}
.global-img-box {{ width:140px; text-align:center; font-size:13px; color:var(--sub); }}
.global-img-box img {{ width:100%; height:140px; object-fit:cover; border-radius:6px; border:1px solid var(--border); }}
.grid-container {{ display:flex; gap:15px; overflow-x:auto; padding-bottom:15px; }}
.frame-column {{ flex:0 0 800px; background:#fafafa; border-radius:8px; border:1px solid var(--border); padding:15px; }}
.prompt-text {{ font-size:14px; color:#d83931; line-height:1.5; margin-bottom:15px; white-space:pre-wrap; }}
.original-prompt {{ font-size:12px; color:#646a73; line-height:1.4; margin-bottom:10px; white-space:pre-wrap; }}
.compare-row {{ display:flex; gap:12px; }}
.compare-box {{ flex:1; }}
.compare-box h4 {{ font-size:13px; margin:0 0 4px 0; padding:3px 6px; border-radius:4px; }}
.compare-box img {{ width:100%; border-radius:6px; border:1px solid var(--border); }}
.nano-header {{ background:#e6f7ff; color:#1890ff; }}
.old-header {{ background:#fff7e6; color:var(--old); }}
.new-header {{ background:#f6ffed; color:var(--new); }}
.score {{ display:inline-block; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:12px; margin-top:3px; }}
.s-1,.s-2 {{ background:#fff2f0; color:var(--score-lo); }}
.s-3 {{ background:#fffbe6; color:var(--score-mid); }}
.s-4,.s-5 {{ background:#f6ffed; color:var(--score-hi); }}
.summary {{ background:var(--card); padding:15px 25px; border-radius:8px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,.05); }}
.summary table {{ border-collapse:collapse; width:100%; font-size:14px; }}
.summary th,.summary td {{ padding:6px 12px; text-align:center; border-bottom:1px solid var(--border); }}
.summary th {{ background:#fafafa; }}
</style></head>
<body><div class="container">
<h2>🛍️ GRPO Round2 对比 — Nano(共用) vs Old My vs New My（含评分）</h2>
<div class="summary" id="summaryBox"></div>
<div id="mainContent">加载中...</div>
</div>
<script>
var DATA = {config_json};
(function(){{
    function esc(t){{ if(!t)return''; return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}}); }}
    function badge(s){{ if(s==null)return''; var c='s-'+Math.min(s,5); return '<span class="score '+c+'">Score: '+s+'/5</span>'; }}
    var c=document.getElementById('mainContent'),s='<table><tr><th>产品</th><th>帧数</th><th>Nano均分</th><th>Old My均分</th><th>New My均分</th></tr>';
    c.innerHTML='';
    DATA.forEach(function(p){{
        var ns=[],os=[],ms=[];
        p.frames.forEach(function(f){{ if(f.nano_score)ns.push(f.nano_score); if(f.old_score)os.push(f.old_score); if(f.new_score)ms.push(f.new_score); }});
        var na=ns.length?(ns.reduce(function(a,b){{return a+b;}},0)/ns.length).toFixed(2):'N/A';
        var oa=os.length?(os.reduce(function(a,b){{return a+b;}},0)/os.length).toFixed(2):'N/A';
        var ma=ms.length?(ms.reduce(function(a,b){{return a+b;}},0)/ms.length).toFixed(2):'N/A';
        s+='<tr><td style="text-align:left">'+esc(p.title)+'</td><td>'+p.frames.length+'</td><td>'+na+'</td><td>'+oa+'</td><td>'+ma+'</td></tr>';
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px 0;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{ h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=丢失\\'"><div>'+esc(g.name)+'</div></div>'; }});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px 0;">对比 (Nano | Old My | New My):</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var orig=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h+='<div class="compare-row">';
            h+='<div class="compare-box"><h4 class="nano-header">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff&text=Nano\\'">'+badge(f.nano_score)+'</div>';
            h+='<div class="compare-box"><h4 class="old-header">Old My Model</h4><img src="'+f.old_my+'" onerror="this.src=\\'https://dummyimage.com/250x350/fff7e6/faad14&text=Old+My\\'">'+badge(f.old_score)+'</div>';
            h+='<div class="compare-box"><h4 class="new-header">New My Model</h4><img src="'+f.new_my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b&text=New+My\\'">'+badge(f.new_score)+'</div>';
            h+='</div></div>';
        }});
        h+='</div></div>'; c.innerHTML+=h;
    }});
    s+='</table>'; document.getElementById('summaryBox').innerHTML=s;
}})();
</script></body></html>"""

    with open(HTML_PATH, "w") as f:
        f.write(html)
    print(f"✅ HTML: {HTML_PATH}")

    try:
        blobstore = BlobStoreClient(BLOB_BUCKET)
        html_key = f"{blob_prefix}/viewer/comparison_{CMP_TAG}.html"
        blobstore.upload_binary_to_s3(HTML_PATH, html_key)
        url = f"{BLOB_CDN}/{html_key}"
        tf = sum(len(p["frames"]) for p in all_products)
        print(f"\n🎉 完成！{len(all_products)}产品, {tf}帧, 三组评分")
        print(f"🌐 {url}")
    except Exception as e:
        print(f"❌ 上传失败: {e}")


if __name__ == "__main__":
    main()
