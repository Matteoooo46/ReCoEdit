#!/usr/bin/env python3
"""CoT Reward 评分 21 产品：输出判断原因 + Yes/No，生成可访问网页"""
import os, json, re, base64, argparse
from io import BytesIO
for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(v,None)
from openai import OpenAI
from PIL import Image
from blobstore import BlobStoreClient

VLM_URL="http://10.15.2.90:8080/v1"; VLM_KEY="flowgrpo"; VLM_MODEL="Qwen3-VL-30B-A3B-Instruct"
BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
VAL_DIR="/data/phd/kousiqi/zhitao/new_validation_set"
TAG="rewrite_optimized"
CMP_TAG="cot_scored_v1"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

SCORES_FILE=os.path.join(BASE,f"scores_{CMP_TAG}.json")
HTML_PATH=os.path.join(BASE,f"cot_comparison_{CMP_TAG}.html")

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

def get_product_name(prod_id):
    info_path=os.path.join(VAL_DIR,prod_id,"item_info.json")
    if os.path.exists(info_path):
        return json.load(open(info_path)).get("itemTitle",prod_id)
    return prod_id

def p2b(img):
    b=BytesIO(); img.save(b,format="JPEG")
    return f"data:image;base64,{base64.b64encode(b.getvalue()).decode()}"

def extract_score(text):
    text=text.strip()
    lines=[l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last=lines[-1]
        if last in("Yes","yes","YES","No","no","NO"):
            return 1.0 if last.lower()=="yes" else 0.0, text
    if text.startswith("Yes")or text.startswith("yes"): return 1.0, text
    if text.startswith("No")or text.startswith("no"): return 0.0, text
    m_yes=re.search(r'\b[Yy]es\b',text); m_no=re.search(r'\b[Nn]o\b',text)
    if m_yes and not m_no: return 1.0, text
    if m_no and not m_yes: return 0.0, text
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
    print(f"\n☁️ 上传 {len(v)} 文件...")
    for i,fp in enumerate(v):
        try:
            fn=os.path.basename(fp); pd=os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp,f"{prefix}/{pd}/{fn}")
            m[fp]=f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1)%100==0: print(f"  {i+1}/{len(v)}")
        except Exception as e: print(f"  [FAIL] {fp}: {e}"); m[fp]=""
    print(f"✅ 上传完成: {len(v)}")
    return m

def num_sort(fname, prefix):
    try: return int(fname.replace(prefix,"").replace(".png",""))
    except: return 0

# --- 产品列表 ---
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
    total=0
    for pi,pn in enumerate(PRODUCTS):
        d=os.path.join(BASE,f"{pn}_{TAG}")
        if not os.path.isdir(d): print(f"❌ {pn}: 目录不存在"); continue
        refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
        if not refs: continue
        ref=Image.open(os.path.join(d,refs[0])).convert("RGB")
        pname=get_product_name(pn)[:40]

        nano=sorted([f for f in os.listdir(d) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
        myf=sorted([f for f in os.listdir(d) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))

        plog=os.path.join(d,f"{pn}_prompt_rewrite_log.json")
        prompts={}
        if os.path.exists(plog):
            for e in json.load(open(plog)): prompts[e.get("frame_index",-1)]=e

        pd={"nano":{},"my":{},"cot_nano":{},"cot_my":{}}
        est=len(nano)+len(myf)
        print(f"\n[{pi+1}/21] {pn} (nano={len(nano)} my={len(myf)} calls={est})")

        for fn in nano:
            img=Image.open(os.path.join(d,fn)).convert("RGB")
            s,cot,raw=score_cot(client,pname,ref,img)
            pd["nano"][fn]=s; pd["cot_nano"][fn]=raw; total+=1
            print(f"  nano {fn}: {'Yes' if s>0 else 'No'}")

        for fn in myf:
            img=Image.open(os.path.join(d,fn)).convert("RGB")
            s,cot,raw=score_cot(client,pname,ref,img)
            pd["my"][fn]=s; pd["cot_my"][fn]=raw; total+=1
            print(f"  my   {fn}: {'Yes' if s>0 else 'No'}")

        pd["prompts"]=prompts; pd["refs"]=refs; pd["product_name"]=pname
        all_data[pn]=pd
    with open(SCORES_FILE,"w")as f: json.dump(all_data,f,ensure_ascii=False,indent=2)
    print(f"\n✅ 打分: {total} 对 -> {SCORES_FILE}")
else:
    with open(SCORES_FILE)as f: all_data=json.load(f)
    print(f"✅ 从 {SCORES_FILE} 加载")

# --- 构建 HTML ---
products=[]; uploads=[]
for pn in PRODUCTS:
    d=os.path.join(BASE,f"{pn}_{TAG}")
    if pn not in all_data: continue
    pd=all_data[pn]
    refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
    globals_list=[]
    for i,r in enumerate(refs):
        globals_list.append((f"产品参考{i+1}",os.path.join(d,r)))
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
        frames.append({
            "prompt":pi.get("rewritten_prompt",""),
            "original_prompt":pi.get("original_prompt",""),
            "nano":nl,"my":ml,
            "nano_score":pd["nano"].get(nf,0),
            "my_score":pd["my"].get(mf,0),
            "nano_cot":pd["cot_nano"].get(nf,""),
            "my_cot":pd["cot_my"].get(mf,""),
        })

    ns=[f["nano_score"]for f in frames]; ms=[f["my_score"]for f in frames]
    na=sum(ns)/len(ns)*100 if ns else 0; ma=sum(ms)/len(ms)*100 if ms else 0
    print(f"✅ {pn}: nano Yes率={na:.0f}% my Yes率={ma:.0f}%")
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in globals_list],"frames":frames})

# 上传
prefix=f"qwen_inference/comparison/{CMP_TAG}"
url_map=upload(uploads,prefix)

for pr in products:
    for g in pr["globals"]:
        if g["path"]in url_map: g["path"]=url_map[g["path"]]
    for f in pr["frames"]:
        for k in["nano","my"]:
            if f[k]in url_map: f[k]=url_map[f[k]]

# HTML
cfg=json.dumps(products,ensure_ascii=False)
html=f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CoT Reward 评分 — 21产品（含判断原因）</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--yes:#00b96b;--no:#ff4d4f}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1600px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box{{width:140px;text-align:center;font-size:13px;color:var(--sub)}}
.global-img-box img{{width:100%;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border)}}
.grid-container{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 560px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:14px;color:#d83931;line-height:1.5;margin-bottom:8px;white-space:pre-wrap}}
.original-prompt{{font-size:12px;color:#646a73;margin-bottom:8px;white-space:pre-wrap}}
.compare-row{{display:flex;gap:12px}}
.compare-box{{flex:1}}
.compare-box h4{{font-size:13px;margin:0 0 4px 0;padding:3px 6px;border-radius:4px}}
.compare-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.nano-h{{background:#e6f7ff;color:#1890ff}}.my-h{{background:#f6ffed;color:var(--yes)}}
.verdict{{display:inline-block;padding:3px 10px;border-radius:4px;font-weight:bold;font-size:13px;margin-top:4px}}
.v-yes{{background:#f6ffed;color:var(--yes);border:1px solid var(--yes)}}
.v-no{{background:#fff2f0;color:var(--no);border:1px solid var(--no)}}
.cot-box{{font-size:12px;color:#555;background:#f0f0f0;border-radius:6px;padding:10px;margin-top:6px;max-height:200px;overflow-y:auto;white-space:pre-wrap;line-height:1.5}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
.summary table{{border-collapse:collapse;width:100%;font-size:14px}}
.summary th,.summary td{{padding:6px 12px;text-align:center;border-bottom:1px solid var(--border)}}
.summary th{{background:#fafafa}}
</style></head>
<body><div class="container">
<h2>🛍️ CoT Reward 评分 — 21产品（含判断原因 + Yes/No）</h2>
<div class="summary" id="summaryBox"></div>
<div id="mainContent">加载中...</div>
</div>
<script>
var DATA={cfg};
(function(){{
    function esc(t){{if(!t)return'';return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}})}}
    function verdict(s, cot){{var c=s>0?'v-yes':'v-no';var t=s>0?'Yes':'No';var cotHtml=cot?'<details style="margin-top:3px"><summary style="cursor:pointer;font-size:11px;color:#999">查看判断原因</summary><div class="cot-box">'+esc(cot)+'</div></details>':'';return'<span class="verdict '+c+'">'+t+'</span>'+cotHtml;}}
    var c=document.getElementById('mainContent'),s='<table><tr><th>产品</th><th>帧数</th><th>Nano Yes率</th><th>My Yes率</th></tr>';
    c.innerHTML='';
    DATA.forEach(function(p){{
        var ns=[],ms=[];
        p.frames.forEach(function(f){{ns.push(f.nano_score);ms.push(f.my_score)}});
        var na=ns.length?(ns.reduce(function(a,b){{return a+b}},0)/ns.length*100).toFixed(0)+'%':'N/A';
        var ma=ms.length?(ms.reduce(function(a,b){{return a+b}},0)/ms.length*100).toFixed(0)+'%':'N/A';
        s+='<tr><td style="text-align:left">'+esc(p.title)+'</td><td>'+p.frames.length+'</td><td>'+na+'</td><td>'+ma+'</td></tr>';
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=丢失\\'"><div>'+esc(g.name)+'</div></div>'}});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">评分对比（Nano | My Model + CoT原因）:</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var orig=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h+='<div class="compare-row">';
            h+='<div class="compare-box"><h4 class="nano-h">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff&text=Nano\\'">'+verdict(f.nano_score,f.nano_cot)+'</div>';
            h+='<div class="compare-box"><h4 class="my-h">My Model</h4><img src="'+f.my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b&text=My\\'">'+verdict(f.my_score,f.my_cot)+'</div>';
            h+='</div></div>';
        }});
        h+='</div></div>';c.innerHTML+=h;
    }});
    s+='</table>';document.getElementById('summaryBox').innerHTML=s;
}})();
</script></body></html>"""

with open(HTML_PATH,"w")as f: f.write(html)
print(f"✅ HTML: {HTML_PATH}")
try:
    bs=BlobStoreClient(BLOB_BUCKET)
    hk=f"{prefix}/viewer/cot_{CMP_TAG}.html"
    bs.upload_binary_to_s3(HTML_PATH,hk)
    tf=sum(len(p["frames"])for p in products)
    print(f"\n🎉 完成！{len(products)}产品, {tf}帧, CoT评分")
    print(f"🌐 {BLOB_CDN}/{hk}")
except Exception as e: print(f"❌ 上传失败: {e}")
