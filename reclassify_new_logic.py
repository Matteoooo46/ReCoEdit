#!/usr/bin/env python3
"""用 rewards.py 新 CLASSIFY_PROMPT（参考图+prompt）重新分类 21 产品，评分沿用已有数据"""
import os, json, base64, asyncio
from io import BytesIO
for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(v,None)
from openai import AsyncOpenAI
from PIL import Image
import httpx
from blobstore import BlobStoreClient

VLM_URL="http://10.15.2.90:8080/v1"; VLM_KEY="flowgrpo"; VLM_MODEL="Qwen3-VL-30B-A3B-Instruct"
BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
TAG="rewrite_optimized"
OLD_SCORES_FILE=os.path.join(BASE,"scores_cot_full_v1.json")
CMP_TAG="reclassify_v3"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"
HTML_PATH=os.path.join(BASE,f"reclassify_{CMP_TAG}.html")

# ─── 新版 CLASSIFY_PROMPT（与 rewards.py 一致）───
CLASSIFY_PROMPT="""你是一个产品一致性检测系统。图片1是产品参考图，下面是一条图像生成提示词。
请逐步分析：该提示词描述的生成图片中，是否应该出现图片1中的产品？

判断标准：
- 如果提示词描述的人物穿着或使用了图片1中的同类产品（如都涉及服装、鞋帽等），回复 Yes。
- 如果提示词描述的场景与产品类别无关，或完全没有提到任何产品，回复 No。

请按以下步骤分析（每步1-2句话）：
Step 1 — 识别参考图产品类别：观察图片1，判断产品属于哪个类别（服装/鞋帽/化妆品/食品/电子产品等），描述其关键特征。
Step 2 — 分析提示词需求：提示词描述的场景中，是否需要出现该品类或具体产品？
Step 3 — 得出结论：基于以上分析，判断是否应该出现该产品。

提示词：{prompt}

最后一行仅输出：Yes 或 No"""

PRODUCTS=[
    "workspace_item_23397479040558","workspace_item_25936355083926",
    "workspace_item_25917400924058","workspace_25833632310597_1772644206",
    "workspace_4959165917841_1772653742","workspace_item_21825264046857",
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae","009785b8-36c6-4775-ada0-c9497e7072c2",
    "010ccf98-82c3-40f9-8284-65518eeff3a0","024419f0-d5c4-482e-9572-ba7885cdf4e4",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8","02cdc727-ab85-4692-8ef6-00b725c64141",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd","03641bdb-7a11-4c05-83a3-347d535e8c91",
    "03d7f47e-bb37-4a19-8685-0e231f933627","03dac08d-8b1e-4a3b-a2d7-94e0ae9ee787",
    "056924c6-1454-4e99-a40c-b8e0b362529c","064eb1bd-65cd-4fc5-8316-a172aa2f8f2f",
    "07b28d8b-d588-4683-b55d-83d59a89a9b0","07b41236-33e3-4fec-bf34-c500ef7fb220",
    "08936a21-6a4a-40c1-828f-f30763afdf02",
]

def pil_to_base64(img):
    buf=BytesIO(); img.save(buf,format="JPEG")
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"

def upload(file_paths, prefix):
    bs=BlobStoreClient(BLOB_BUCKET); m={}
    v=[p for p in set(file_paths) if p and os.path.exists(p)]
    print(f"\nUploading {len(v)} files...")
    for i,fp in enumerate(v):
        try:
            fn=os.path.basename(fp); pd=os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp,f"{prefix}/{pd}/{fn}")
            m[fp]=f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1)%100==0: print(f"  {i+1}/{len(v)}")
        except Exception as e: print(f"  [FAIL] {fp}: {e}"); m[fp]=""
    print(f"Upload done: {len(v)}")
    return m

def num_sort(fname, prefix):
    try: return int(fname.replace(prefix,"").replace(".png",""))
    except: return 0

async def reclassify():
    client = AsyncOpenAI(
        base_url=VLM_URL, api_key=VLM_KEY,
        http_client=httpx.AsyncClient(proxy=None, timeout=90.0),
    )

    # 加载旧分数
    with open(OLD_SCORES_FILE) as f:
        old_scores = json.load(f)

    new_cls = {}
    total = 0

    for pi, pn in enumerate(PRODUCTS):
        d = os.path.join(BASE, f"{pn}_{TAG}")
        if not os.path.isdir(d): continue
        refs = sorted([f for f in os.listdir(d) if f.startswith("p1_ref_") and f.endswith(".png")])
        if not refs: continue
        ref_img = Image.open(os.path.join(d, refs[0])).convert("RGB")
        ref_b64 = pil_to_base64(ref_img)

        plog = os.path.join(d, f"{pn}_prompt_rewrite_log.json")
        prompts = {}
        if os.path.exists(plog):
            for e in json.load(open(plog)): prompts[e.get("frame_index", -1)] = e

        myf = sorted([f for f in os.listdir(d) if f.startswith("p1_my_") and f.endswith(".png")],
                      key=lambda s: num_sort(s, "p1_my_"))

        pd = {"results": {}, "raw": {}}
        tasks = []
        prompt_texts = []

        for i in range(len(myf)):
            pi_dict = prompts.get(str(i), prompts.get(i, {}))
            orig_prompt = pi_dict.get("original_prompt", "")
            # fallback to rewritten if original is empty
            if not orig_prompt or len(orig_prompt.strip()) < 5:
                orig_prompt = pi_dict.get("rewritten_prompt", "")
            prompt_texts.append(orig_prompt)
            if orig_prompt:
                msg = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": ref_b64}},
                    {"type": "text", "text": CLASSIFY_PROMPT.format(prompt=orig_prompt)},
                ]}]
                tasks.append(client.chat.completions.create(
                    model=VLM_MODEL, messages=msg, max_tokens=500, temperature=0.0))
            else:
                tasks.append(None)

        # 批量执行
        results = []
        for t in tasks:
            if t:
                try:
                    r = await t
                    results.append(r.choices[0].message.content.strip())
                except Exception as e:
                    results.append(f"[ERROR] {e}")
            else:
                results.append("[No prompt]")
            total += 1

        def extract_cls(text):
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if lines:
                last = lines[-1]
                if last.lower() in ("yes", "no"):
                    return last.lower() == "yes"
            if text.lower().startswith("yes"): return True
            if text.lower().startswith("no"): return False
            return True
        cls_yes = 0
        for i, raw in enumerate(results):
            is_prod = extract_cls(raw)
            pd["results"][str(i)] = is_prod
            pd["raw"][str(i)] = raw
            if is_prod: cls_yes += 1

        print(f"[{pi+1}/21] {pn}: {cls_yes}/{len(myf)} has_product (new logic with CoT)")

        # Merge with old scores
        if pn in old_scores:
            old = old_scores[pn]
            old["cls_result"] = pd["results"]
            old["cls_raw"] = pd["raw"]
        new_cls[pn] = pd

    print(f"\nDone: {total} classifications")

    # Save merged data
    merged = {}
    for pn in PRODUCTS:
        if pn in old_scores:
            merged[pn] = old_scores[pn]
            if pn in new_cls:
                merged[pn]["cls_result"] = new_cls[pn]["results"]
                merged[pn]["cls_raw"] = new_cls[pn]["raw"]
        elif pn in new_cls:
            merged[pn] = {"cls_result": new_cls[pn]["results"], "cls_raw": new_cls[pn]["raw"]}

    # Build HTML
    products = []; uploads = []
    for pn in PRODUCTS:
        d = os.path.join(BASE, f"{pn}_{TAG}")
        if pn not in merged: continue
        pd = merged[pn]
        refs = sorted([f for f in os.listdir(d) if f.startswith("p1_ref_") and f.endswith(".png")])
        globals_list = []
        for i, r in enumerate(refs):
            globals_list.append((f"Product Ref {i+1}", os.path.join(d, r)))
        for _, p in globals_list: uploads.append(p)

        myf = sorted([f for f in os.listdir(d) if f.startswith("p1_my_") and f.endswith(".png")],
                      key=lambda s: num_sort(s, "p1_my_"))
        nano = sorted([f for f in os.listdir(d) if f.startswith("p1_nano_") and f.endswith(".png")],
                       key=lambda s: num_sort(s, "p1_nano_"))

        plog = os.path.join(d, f"{pn}_prompt_rewrite_log.json")
        prompts = {}
        if os.path.exists(plog):
            for e in json.load(open(plog)): prompts[e.get("frame_index", -1)] = e

        frames = []
        for i in range(len(myf)):
            num = i + 1
            nf = f"p1_nano_{num}.png"; mf = f"p1_my_{num}.png"
            nl = os.path.join(d, nf) if os.path.exists(os.path.join(d, nf)) else ""
            ml = os.path.join(d, mf) if os.path.exists(os.path.join(d, mf)) else ""
            for p in [nl, ml]:
                if p: uploads.append(p)
            pi = prompts.get(str(i), prompts.get(i, {}))

            # Old scores
            ns = pd.get("nano", {}).get(nf, 0)
            ms = pd.get("my", {}).get(mf, 0)
            skipped = ms == -2

            # New classification
            is_prod = pd.get("cls_result", {}).get(str(i), True)
            if isinstance(is_prod, str): is_prod = (is_prod == "True")

            frames.append({
                "prompt": pi.get("rewritten_prompt", ""),
                "original_prompt": pi.get("original_prompt", ""),
                "nano": nl, "my": ml,
                "nano_score": ns if ns > 0 else 0.0,
                "my_score": ms if ms >= 0 else 0.0,
                "nano_cot": pd.get("cot_nano", {}).get(nf, ""),
                "my_cot": pd.get("cot_my", {}).get(mf, ""),
                "cls_is_product": bool(is_prod),
                "skipped": skipped or not bool(is_prod),
                "cls_raw": pd.get("cls_raw", {}).get(str(i), ""),
            })

        ns = [f["nano_score"] for f in frames]
        ms = [f["my_score"] for f in frames if f["my_score"] >= 0]
        na = sum(ns)/len(ns)*100 if ns else 0
        ma = sum(ms)/len(ms)*100 if ms else 0
        cls_yes = sum(1 for f in frames if f["cls_is_product"])
        print(f"  {pn}: nano Yes={na:.0f}% my Yes={ma:.0f}% cls(new)={cls_yes}/{len(frames)}")
        products.append({"title": pn, "globals": [{"name": n, "path": p} for n, p in globals_list], "frames": frames})

    # 上传图片
    prefix = f"qwen_inference/comparison/{CMP_TAG}"
    url_map = upload(uploads, prefix)
    for pr in products:
        for g in pr["globals"]:
            if g["path"] in url_map: g["path"] = url_map[g["path"]]
        for f in pr["frames"]:
            for k in ["nano", "my"]:
                if f[k] in url_map: f[k] = url_map[f[k]]

    # HTML
    js_data = json.dumps(products, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>New Classification Logic (ref+prompt) - 21 products</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--yes:#00b96b;--no:#ff4d4f;--skip:#8f959e}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1600px;margin:0 auto}}
h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box img{{width:140px;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border);margin-bottom:5px}}
.grid-container{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 560px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:14px;color:#d83931;line-height:1.5;margin-bottom:4px;white-space:pre-wrap}}
.original-prompt{{font-size:12px;color:#646a73;margin-bottom:6px;white-space:pre-wrap}}
.cls-line{{font-size:12px;margin-bottom:8px;padding:4px 8px;border-radius:4px;background:#f9f9f9}}
.compare-row{{display:flex;gap:12px}}
.compare-box{{flex:1}}
.compare-box h4{{font-size:13px;margin:0 0 4px 0;padding:3px 6px;border-radius:4px}}
.compare-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.nano-h{{background:#e6f7ff;color:#1890ff}}.my-h{{background:#f6ffed;color:var(--yes)}}
.verdict{{display:inline-block;padding:3px 10px;border-radius:4px;font-weight:bold;font-size:13px;margin-top:4px}}
.v-yes{{background:#f6ffed;color:var(--yes);border:1px solid var(--yes)}}
.v-no{{background:#fff2f0;color:var(--no);border:1px solid var(--no)}}
.v-skip{{background:#f0f0f0;color:var(--skip);border:1px solid var(--skip)}}
.cot-box{{font-size:12px;color:#555;background:#f0f0f0;border-radius:6px;padding:10px;margin-top:6px;max-height:200px;overflow-y:auto;white-space:pre-wrap;line-height:1.5}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
.summary table{{border-collapse:collapse;width:100%;font-size:14px}}
.summary th,.summary td{{padding:6px 12px;text-align:center;border-bottom:1px solid var(--border)}}
.summary th{{background:#fafafa}}
.product-links{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}
.product-link{{display:inline-block;padding:6px 14px;background:var(--pri);color:#fff;border-radius:6px;font-size:13px;cursor:pointer;text-decoration:none}}
.product-link:hover{{opacity:0.85}}
.product-link.active{{background:#faad14;color:#000}}
</style></head>
<body><div class="container">
<h2>New Classification Logic: ref image + prompt (21 products)</h2>
<div class="summary" id="summaryBox"></div>
<div class="product-links" id="productIndex"></div>
<div id="mainContent"><p style="color:#999">Click a product above to view details</p></div>
</div>
<script>
var DATA = {js_data};
(function() {{
    function esc(t) {{
        if(!t) return '';
        return String(t).replace(/[&<>"]/g, function(c) {{
            return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];
        }});
    }}
    function vd(s, cot, skipped) {{
        if(skipped) return '<span class="verdict v-skip">Skipped</span>';
        var cls = s>0?'v-yes':'v-no'; var txt = s>0?'Yes':'No';
        var cotHtml = cot?'<details style="margin-top:3px"><summary style="cursor:pointer;font-size:11px;color:#999">Show CoT</summary><div class="cot-box">'+esc(cot)+'</div></details>':'';
        return '<span class="verdict '+cls+'">'+txt+'</span>'+cotHtml;
    }}
    var s = '<table><tr><th>Product</th><th>Frames</th><th>New cls: Has Product</th><th>Nano Yes%</th><th>My Yes%</th></tr>';
    var idxHtml = '<b>Jump to:</b> ';
    DATA.forEach(function(p, pi) {{
        var ns=[], ss=[], skipped=0;
        p.frames.forEach(function(f) {{ns.push(f.nano_score); if(f.skipped) skipped++; else if(f.my_score>=0) ss.push(f.my_score)}});
        var na=ns.length?(ns.reduce(function(a,b){{return a+b}},0)/ns.length*100).toFixed(0)+'%':'N/A';
        var ma=ss.length?(ss.reduce(function(a,b){{return a+b}},0)/ss.length*100).toFixed(0)+'%':'N/A';
        var clsCnt=p.frames.filter(function(f){{return f.cls_is_product}}).length;
        s+='<tr><td style="text-align:left;cursor:pointer;color:var(--pri);text-decoration:underline" onclick="show('+pi+')">'+esc(p.title)+'</td><td>'+p.frames.length+'</td><td>'+clsCnt+'/'+p.frames.length+'</td><td>'+na+'</td><td>'+ma+'</td></tr>';
        idxHtml += '<span class="product-link" id="link'+pi+'" onclick="show('+pi+')">#'+(pi+1)+' '+esc(p.title)+'</span> ';
    }});
    s+='</table>'; document.getElementById('summaryBox').innerHTML=s;
    document.getElementById('productIndex').innerHTML=idxHtml;

    function show(pi) {{
        var p = DATA[pi];
        for(var i=0; i<DATA.length; i++) {{
            var el = document.getElementById('link'+i);
            if(el) el.className = (i===pi) ? 'product-link active' : 'product-link';
        }}
        var h = '<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g) {{
            h += '<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00\\'"><div>'+esc(g.name)+'</div></div>';
        }});
        h += '</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">CoT Scoring (Nano | My Model):</h3><div class="grid-container">';
        p.frames.forEach(function(f) {{
            var orig = f.original_prompt ? '<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>' : '';
            var clsClr = f.cls_is_product ? 'color:var(--yes)' : 'color:var(--no)';
            var clsTxt = f.cls_is_product ? 'Has product' : 'No product (skip score)';
            h += '<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            var clsCotHtml = f.cls_raw ? '<details style="margin-top:3px"><summary style="cursor:pointer;font-size:11px;color:#999">Show cls CoT</summary><div class="cot-box">'+esc(f.cls_raw)+'</div></details>' : '';
            h += '<div class="cls-line">[New logic] Prompt cls: <span style="'+clsClr+'"><b>'+clsTxt+'</b></span></div>'+clsCotHtml;
            h += '<div class="compare-row">';
            h += '<div class="compare-box"><h4 class="nano-h">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff\\'">'+vd(f.nano_score, f.nano_cot, false)+'</div>';
            h += '<div class="compare-box"><h4 class="my-h">My Model</h4><img src="'+f.my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b\\'">'+vd(f.my_score, f.my_cot, f.skipped)+'</div>';
            h += '</div></div>';
        }});
        h += '</div></div>';
        document.getElementById('mainContent').innerHTML = h;
    }}
    window.show = show;
    show(0);
}})();
</script></body></html>"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML: {HTML_PATH} ({len(html)} bytes)")

    try:
        bs = BlobStoreClient(BLOB_BUCKET)
        hk = f"{prefix}/viewer/reclassify_{CMP_TAG}.html"
        bs.upload_binary_to_s3(HTML_PATH, hk)
        tf = sum(len(p["frames"]) for p in products)
        print(f"\nDone! {len(products)} products, {tf} frames")
        print(f"URL: {BLOB_CDN}/{hk}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    asyncio.run(reclassify())
