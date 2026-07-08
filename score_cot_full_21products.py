#!/usr/bin/env python3
"""完整 CoT Reward：Prompt 分类 + 图像评分，21 产品。点击式渲染避免大页面卡死。"""
import os, json, re, base64, argparse
from io import BytesIO
for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(v,None)
from openai import OpenAI
from PIL import Image
from blobstore import BlobStoreClient

VLM_URL="http://10.15.2.90:8080/v1"; VLM_KEY="flowgrpo"; VLM_MODEL="Qwen3-VL-30B-A3B-Instruct"
BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
TAG="rewrite_optimized"; CMP_TAG="cot_full_v4"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

SCORES_FILE=os.path.join(BASE,"scores_cot_full_v1.json")
HTML_PATH=os.path.join(BASE,f"cot_full_{CMP_TAG}.html")

# Prompt 分类
CLASSIFY_PROMPT="""判断以下图像生成提示词是否要求生成的图片中出现具体的产品（如服装、鞋帽、配饰、箱包、电子产品、化妆品、食品饮料、家居用品等具体商品）。
- 如果提示词要求图片中出现某件具体产品，回复 Yes。
- 如果提示词仅描述人物姿势、背景、场景、艺术风格等，没有要求出现具体产品，回复 No。

提示词：{prompt}

仅回复 Yes 或 No。"""

# 图像评分 CoT
SCORING_PROMPT="""你是一位专业的商品图像质量审核专家。请逐步分析两张图片中的产品外观一致性。

产品名称：{product_name}
图片1：产品参考图
图片2：生成的图片

请按以下步骤分析（在最终输出 Yes 或 No 之前，先写出分析过程）：
Step 1 — 定位产品：在图片1中描述该产品的关键外观特征（颜色、款式、材质、图案、logo等）。
Step 2 — 检查存在性：在图片2中是否能找到该产品？
Step 3 — 逐项对比：将图片2中产品的外观与Step 1的关键特征逐项对比（颜色、款式、图案、细节）。
Step 4 — 结论：基于以上对比，判断是否一致。

规则：
- 若人物穿戴该产品，重点对比产品本身而非人物。
- 严格判断：有明显颜色/款式/图案差异即为 No。
- 分析过程的最后一行必须是：Yes 或 No"""

def p2b(img):
    b=BytesIO(); img.save(b,format="JPEG")
    return f"data:image;base64,{base64.b64encode(b.getvalue()).decode()}"

def classify_prompt(client, prompt_text):
    if not prompt_text or len(prompt_text.strip()) < 5:
        return False, ""
    try:
        r=client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role":"user","content":CLASSIFY_PROMPT.format(prompt=prompt_text)}],
            max_tokens=10, temperature=0.0)
        raw=r.choices[0].message.content.strip()
        return raw.lower().startswith("yes"), raw
    except:
        return True, "[ERROR]"

def extract_score(text):
    text=text.strip()
    lines=[l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last=lines[-1]
        if last in("Yes","yes","YES","No","no","NO"):
            return 1.0 if last.lower()=="yes" else 0.0, text
    if text.startswith("Yes")or text.startswith("yes"): return 1.0, text
    if text.startswith("No")or text.startswith("no"): return 0.0, text
    return 0.0, text

def score_cot(client, name, ref, gen):
    try:
        r=client.chat.completions.create(model=VLM_MODEL,messages=[{"role":"user","content":[
            {"type":"text","text":SCORING_PROMPT.format(product_name=name)},
            {"type":"image_url","image_url":{"url":p2b(ref)}},
            {"type":"image_url","image_url":{"url":p2b(gen)}}]}],temperature=0.2)
        raw=r.choices[0].message.content.strip()
        s,cot=extract_score(raw)
        return s,cot,raw
    except Exception as e:
        return 0.0,"",f"[ERROR] {e}"

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

# --- 评分 ---
ap=argparse.ArgumentParser(); ap.add_argument("--skip-score",action="store_true"); args=ap.parse_args()
all_data={}

if not args.skip_score:
    client=OpenAI(base_url=VLM_URL,api_key=VLM_KEY,timeout=120)
    total_img=0; total_cls=0
    for pi,pn in enumerate(PRODUCTS):
        d=os.path.join(BASE,f"{pn}_{TAG}")
        if not os.path.isdir(d): continue
        refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
        if not refs: continue
        ref=Image.open(os.path.join(d,refs[0])).convert("RGB")
        nano=sorted([f for f in os.listdir(d) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
        myf=sorted([f for f in os.listdir(d) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))
        plog=os.path.join(d,f"{pn}_prompt_rewrite_log.json")
        prompts={}
        if os.path.exists(plog):
            for e in json.load(open(plog)): prompts[e.get("frame_index",-1)]=e
        pd={"nano":{},"my":{},"cot_nano":{},"cot_my":{},"cls_result":{},"cls_raw":{}}
        print(f"\n[{pi+1}/21] {pn} (nano={len(nano)} my={len(myf)})")
        cls_cache={}
        for i in range(len(myf)):
            pi_dict=prompts.get(str(i),prompts.get(i,{}))
            rw_prompt=pi_dict.get("rewritten_prompt","")
            if rw_prompt and rw_prompt not in cls_cache:
                is_prod,raw=classify_prompt(client,rw_prompt)
                cls_cache[rw_prompt]=(is_prod,raw); total_cls+=1
            elif rw_prompt:
                is_prod,raw=cls_cache[rw_prompt]
            else:
                is_prod,raw=True,"[No prompt]"
            pd["cls_result"][str(i)]=is_prod; pd["cls_raw"][str(i)]=raw
        cls_yes=sum(1 for v in pd["cls_result"].values() if v)
        print(f"  Prompt cls: {cls_yes}/{len(myf)} product-related")
        for fn in nano:
            img=Image.open(os.path.join(d,fn)).convert("RGB")
            s,cot,raw=score_cot(client,pn[:40],ref,img)
            pd["nano"][fn]=s; pd["cot_nano"][fn]=raw; total_img+=1
            print(f"  nano {fn}: {'Yes' if s>0 else 'No'}")
        for i,fn in enumerate(myf):
            is_prod=pd["cls_result"].get(str(i),True)
            if is_prod:
                img=Image.open(os.path.join(d,fn)).convert("RGB")
                s,cot,raw=score_cot(client,pn[:40],ref,img)
                pd["my"][fn]=s; pd["cot_my"][fn]=raw; total_img+=1
                print(f"  my   {fn}: {'Yes' if s>0 else 'No'}")
            else:
                pd["my"][fn]=-2; pd["cot_my"][fn]="[Prompt not product-related, skipped]"
                print(f"  my   {fn}: skip")
        pd["prompts"]=prompts; pd["refs"]=refs
        all_data[pn]=pd
    with open(SCORES_FILE,"w")as f: json.dump(all_data,f,ensure_ascii=False,indent=2)
    print(f"\nDone: {total_cls} cls + {total_img} img -> {SCORES_FILE}")
else:
    with open(SCORES_FILE)as f: all_data=json.load(f)
    print(f"Loaded from {SCORES_FILE}")

# --- 构建 HTML ---
products=[]; uploads=[]
for pn in PRODUCTS:
    d=os.path.join(BASE,f"{pn}_{TAG}")
    if pn not in all_data: continue
    pd=all_data[pn]
    refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
    globals_list=[]
    for i,r in enumerate(refs):
        globals_list.append((f"Product Ref {i+1}",os.path.join(d,r)))
    for _,p in globals_list: uploads.append(p)
    nano=sorted([f for f in os.listdir(d) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
    myf=sorted([f for f in os.listdir(d) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))
    prompts=pd.get("prompts",{})
    frames=[]
    for i in range(len(myf)):
        num=i+1
        nf=f"p1_nano_{num}.png"; mf=f"p1_my_{num}.png"
        nl=os.path.join(d,nf) if os.path.exists(os.path.join(d,nf)) else ""
        ml=os.path.join(d,mf) if os.path.exists(os.path.join(d,mf)) else ""
        for p in [nl,ml]:
            if p: uploads.append(p)
        pi=prompts.get(str(i),prompts.get(i,{}))
        ns=pd["nano"].get(nf,0); ms=pd["my"].get(mf,0)
        frames.append({
            "prompt":pi.get("rewritten_prompt",""),
            "original_prompt":pi.get("original_prompt",""),
            "nano":nl,"my":ml,
            "nano_score":ns if ns>0 else 0.0,
            "my_score":ms if ms>0 else (-2 if ms==-2 else 0.0),
            "nano_cot":pd["cot_nano"].get(nf,""),
            "my_cot":pd["cot_my"].get(mf,""),
            "cls_is_product":pd["cls_result"].get(str(i),True),
            "cls_raw":pd["cls_raw"].get(str(i),""),
            "skipped":ms==-2,
        })
    ns=[f["nano_score"]for f in frames]
    ms=[f["my_score"]for f in frames if f["my_score"]>=0]
    na=sum(ns)/len(ns)*100 if ns else 0
    ma=sum(ms)/len(ms)*100 if ms else 0
    cls_yes=sum(1 for f in frames if f["cls_is_product"])
    print(f"  {pn}: nano Yes={na:.0f}% my Yes={ma:.0f}% cls={cls_yes}/{len(frames)}")
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in globals_list],"frames":frames})

# 上传图片
prefix=f"qwen_inference/comparison/{CMP_TAG}"
url_map=upload(uploads,prefix)
for pr in products:
    for g in pr["globals"]:
        if g["path"]in url_map: g["path"]=url_map[g["path"]]
    for f in pr["frames"]:
        for k in["nano","my"]:
            if f[k]in url_map: f[k]=url_map[f[k]]

# ===== 点击式 HTML（一次只渲染一个产品）=====
js_data = json.dumps(products, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CoT Reward - Prompt Classification + Image Scoring (21 products)</title>
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
.cls-line{{font-size:12px;margin-bottom:8px}}
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
<h2>CoT Reward - Prompt Classification + Image Scoring (21 products)</h2>
<div class="summary" id="summaryBox"></div>
<div class="product-links" id="productIndex"></div>
<div id="mainContent"><p style="color:#999">Click a product name above to view details</p></div>
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
        var cls = s>0?'v-yes':'v-no';
        var txt = s>0?'Yes':'No';
        var cotHtml = cot?'<details style="margin-top:3px"><summary style="cursor:pointer;font-size:11px;color:#999">Show reasoning (CoT)</summary><div class="cot-box">'+esc(cot)+'</div></details>':'';
        return '<span class="verdict '+cls+'">'+txt+'</span>'+cotHtml;
    }}
    // Build summary table + index
    var s = '<table><tr><th>Product</th><th>Frames</th><th>Has Product</th><th>Nano Yes%</th><th>My Yes%</th></tr>';
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
    s+='</table>';
    document.getElementById('summaryBox').innerHTML = s;
    document.getElementById('productIndex').innerHTML = idxHtml;

    // Render one product
    function show(pi) {{
        var p = DATA[pi];
        // Highlight active link
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
            var clsTxt = f.cls_is_product ? 'Has product' : 'No product';
            h += '<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h += '<div class="cls-line">Prompt classification: <span style="'+clsClr+'">'+clsTxt+'</span></div>';
            h += '<div class="compare-row">';
            h += '<div class="compare-box"><h4 class="nano-h">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff\\'">'+vd(f.nano_score, f.nano_cot, false)+'</div>';
            h += '<div class="compare-box"><h4 class="my-h">My Model</h4><img src="'+f.my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b\\'">'+vd(f.my_score, f.my_cot, f.skipped)+'</div>';
            h += '</div></div>';
        }});
        h += '</div></div>';
        document.getElementById('mainContent').innerHTML = h;
    }}
    // Show first product by default
    window.show = show;
    show(0);
}})();
</script></body></html>"""

with open(HTML_PATH,"w",encoding="utf-8") as f: f.write(html)
print(f"HTML saved: {HTML_PATH} ({len(html)} bytes)")

try:
    bs=BlobStoreClient(BLOB_BUCKET)
    hk=f"{prefix}/viewer/cot_full_{CMP_TAG}.html"
    bs.upload_binary_to_s3(HTML_PATH,hk)
    tf=sum(len(p["frames"])for p in products)
    print(f"\nDone! {len(products)} products, {tf} frames")
    print(f"URL: {BLOB_CDN}/{hk}")
except Exception as e: print(f"Upload failed: {e}")
