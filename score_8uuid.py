#!/usr/bin/env python3
"""评分 8 个 UUID 新产品，生成 Old vs New 对比网页"""
import os, json, re, base64
from io import BytesIO
for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(v,None)
from openai import OpenAI
from PIL import Image
from blobstore import BlobStoreClient

VLM_URL="http://10.15.2.90:8080/v1"; VLM_KEY="flowgrpo"; VLM_MODEL="Qwen3-VL-30B-A3B-Instruct"
BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
OLD_SCORES_FILE=os.path.join(BASE,"all_scores_rewrite_optimized_merged_scored_v1.json")
TAG="rewrite_optimized"; CMP_TAG="cmp_8uuid_v1"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

PRODUCTS=[
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae","009785b8-36c6-4775-ada0-c9497e7072c2",
    "010ccf98-82c3-40f9-8284-65518eeff3a0","024419f0-d5c4-482e-9572-ba7885cdf4e4",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8","02cdc727-ab85-4692-8ef6-00b725c64141",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd","03641bdb-7a11-4c05-83a3-347d535e8c91",
]

SCORES_FILE=os.path.join(BASE,f"scores_{CMP_TAG}.json")
HTML_PATH=os.path.join(BASE,f"comparison_{CMP_TAG}.html")

PROMPT="""你是一位专业的商品图像质量审核专家。比较两张图片中同一件产品的外观一致性。

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

def p2b(img):
    b=BytesIO(); img.save(b,format="PNG")
    return f"data:image;base64,{base64.b64encode(b.getvalue()).decode()}"

def score_pair(client, ref, gen):
    try:
        r=client.chat.completions.create(model=VLM_MODEL,messages=[{"role":"user","content":[
            {"type":"text","text":PROMPT},{"type":"image_url","image_url":{"url":p2b(ref)}},
            {"type":"image_url","image_url":{"url":p2b(gen)}}]}],temperature=0.2)
        m=re.search(r"Score:\s*(\d)",r.choices[0].message.content.strip())
        return int(m.group(1))if m else -1
    except: return -1

def upload(file_paths, prefix):
    bs=BlobStoreClient(BLOB_BUCKET); m={}
    v=[p for p in set(file_paths) if p and os.path.exists(p)]
    print(f"☁️ 上传 {len(v)} 文件...")
    for i,fp in enumerate(v):
        try:
            fn=os.path.basename(fp); pd=os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp,f"{prefix}/{pd}/{fn}")
            m[fp]=f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1)%50==0: print(f"  {i+1}/{len(v)}")
        except Exception as e: print(f"  [FAIL] {fp}: {e}"); m[fp]=""
    print(f"✅ 上传完成: {len(v)}")
    return m

def num_sort(fname, prefix):
    try: return int(fname.replace(prefix,"").replace(".png",""))
    except: return 0

# --- 评分 ---
import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--skip-score",action="store_true"); args=ap.parse_args()
new_scores={}

if not args.skip_score:
    client=OpenAI(base_url=VLM_URL,api_key=VLM_KEY,timeout=120)
    total=0
    for pi,pn in enumerate(PRODUCTS):
        d=os.path.join(BASE,f"{pn}_{TAG}")
        refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
        if not refs: continue
        ref=Image.open(os.path.join(d,refs[0])).convert("RGB")
        nano=sorted([f for f in os.listdir(d) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
        myf=sorted([f for f in os.listdir(d) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))
        ps={"nano":{},"my":{}}
        print(f"\n[{pi+1}/8] {pn} (nano={len(nano)} my={len(myf)})")
        for fn in nano:
            s=score_pair(client,ref,Image.open(os.path.join(d,fn)).convert("RGB"))
            ps["nano"][fn]=s; total+=1; print(f"  nano {fn}: {s}/5")
        for fn in myf:
            s=score_pair(client,ref,Image.open(os.path.join(d,fn)).convert("RGB"))
            ps["my"][fn]=s; total+=1; print(f"  my   {fn}: {s}/5")
        new_scores[pn]=ps
    with open(SCORES_FILE,"w")as f: json.dump(new_scores,f,ensure_ascii=False,indent=2)
    print(f"\n✅ 打分: {total} 对 -> {SCORES_FILE}")
else:
    with open(SCORES_FILE)as f: new_scores=json.load(f)

# --- 读取旧分数 ---
old_scores={}
if os.path.exists(OLD_SCORES_FILE):
    with open(OLD_SCORES_FILE)as f: old_scores=json.load(f)

# --- 构建 HTML ---
products=[]; uploads=[]
for pn in PRODUCTS:
    d=os.path.join(BASE,f"{pn}_{TAG}")
    refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
    globals_list=[]
    for i,r in enumerate(refs):
        globals_list.append((f"产品参考{i+1}",os.path.join(d,r)))
    for _,p in globals_list: uploads.append(p)

    nano=sorted([f for f in os.listdir(d) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
    myf=sorted([f for f in os.listdir(d) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))

    # 读取 prompt log
    plog=os.path.join(d,f"{pn}_prompt_rewrite_log.json")
    prompts={}
    if os.path.exists(plog):
        for e in json.load(open(plog)): prompts[e.get("frame_index",-1)]=e

    ns=new_scores.get(pn,{}); os_=old_scores.get(pn,{})
    ns_n=ns.get("nano",{}); ns_m=ns.get("my",{})
    os_m=os_.get("my",{}); os_n=os_.get("nano",{})

    frames=[]
    for i in range(len(myf)):
        num=i+1
        nf=f"p1_nano_{num}.png"; mf=f"p1_my_{num}.png"
        nl=os.path.join(d,nf) if os.path.exists(os.path.join(d,nf)) else ""
        ml=os.path.join(d,mf) if os.path.exists(os.path.join(d,mf)) else ""
        for p in [nl,ml]:
            if p: uploads.append(p)
        pi=prompts.get(i,{})
        frames.append({
            "prompt":pi.get("rewritten_prompt",""),
            "original_prompt":pi.get("original_prompt",""),
            "nano":nl,"my":ml,
            "new_nano":ns_n.get(nf,-1)if ns_n.get(nf,-1)>0 else None,
            "new_my":ns_m.get(mf,-1)if ns_m.get(mf,-1)>0 else None,
            "old_my":os_m.get(mf,-1)if os_m.get(mf,-1)>0 else None,
            "old_nano":os_n.get(nf,-1)if os_n.get(nf,-1)>0 else None,
        })

    # calc avgs
    nna=[f["new_nano"]for f in frames if f["new_nano"]]
    nma=[f["new_my"]for f in frames if f["new_my"]]
    oma=[f["old_my"]for f in frames if f["old_my"]]
    ona=[f["old_nano"]for f in frames if f["old_nano"]]
    print(f"✅ {pn}: old_my={sum(oma)/len(oma):.2f} new_my={sum(nma)/len(nma):.2f}" if nma else f"✅ {pn}")
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in globals_list],"frames":frames})

# 上传
print(f"\n📤 {len(uploads)} 文件")
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
<title>GRPO Round2 — 8产品 Old vs New 对比</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--old:#faad14;--new:#00b96b;--hi:#00b96b;--mid:#faad14;--lo:#ff4d4f}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1500px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box{{width:140px;text-align:center;font-size:13px;color:var(--sub)}}
.global-img-box img{{width:100%;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border)}}
.grid-container{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 540px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:14px;color:#d83931;line-height:1.5;margin-bottom:10px;white-space:pre-wrap}}
.original-prompt{{font-size:12px;color:#646a73;margin-bottom:10px;white-space:pre-wrap}}
.compare-row{{display:flex;gap:12px}}
.compare-box{{flex:1}}
.compare-box h4{{font-size:13px;margin:0 0 4px 0;padding:3px 6px;border-radius:4px}}
.compare-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.nano-h{{background:#e6f7ff;color:#1890ff}}.old-h{{background:#fff7e6;color:var(--old)}}.new-h{{background:#f6ffed;color:var(--new)}}
.score{{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px;margin-top:3px}}
.s-1,.s-2{{background:#fff2f0;color:var(--lo)}}.s-3{{background:#fffbe6;color:var(--mid)}}.s-4,.s-5{{background:#f6ffed;color:var(--hi)}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
.summary table{{border-collapse:collapse;width:100%;font-size:14px}}
.summary th,.summary td{{padding:6px 12px;text-align:center;border-bottom:1px solid var(--border)}}
.summary th{{background:#fafafa}}
</style></head>
<body><div class="container">
<h2>🛍️ GRPO Round2 对比 — 8 产品（Old vs New My Model，含评分）</h2>
<div class="summary" id="summaryBox"></div>
<div id="mainContent">加载中...</div>
</div>
<script>
var DATA={cfg};
(function(){{
    function esc(t){{if(!t)return'';return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}})}}
    function b(s){{if(s==null)return'';var c='s-'+Math.min(s,5);return'<span class="score '+c+'">Score: '+s+'/5</span>'}}
    var c=document.getElementById('mainContent'),s='<table><tr><th>产品</th><th>帧数</th><th>Old My</th><th>New My</th></tr>';
    c.innerHTML='';
    DATA.forEach(function(p){{
        var om=[],nm=[];
        p.frames.forEach(function(f){{if(f.old_my)om.push(f.old_my);if(f.new_my)nm.push(f.new_my)}});
        var oa=om.length?(om.reduce(function(a,b){{return a+b}},0)/om.length).toFixed(2):'N/A';
        var na=nm.length?(nm.reduce(function(a,b){{return a+b}},0)/nm.length).toFixed(2):'N/A';
        s+='<tr><td style="text-align:left">'+esc(p.title)+'</td><td>'+p.frames.length+'</td><td>'+oa+'</td><td>'+na+'</td></tr>';
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px 0;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=丢失\\'"><div>'+esc(g.name)+'</div></div>'}});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px 0;">对比（Nano | Old My | New My）:</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var orig=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h+='<div class="compare-row">';
            h+='<div class="compare-box"><h4 class="nano-h">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff&text=Nano\\'">'+b(f.new_nano)+'</div>';
            h+='<div class="compare-box"><h4 class="old-h">Old My Model</h4><div style="width:100%;height:200px;background:#fff7e6;border-radius:6px;border:1px dashed #faad14;display:flex;align-items:center;justify-content:center;font-size:14px;color:#999">图片已覆盖<br>仅展示分数</div>'+b(f.old_my)+'</div>';
            h+='<div class="compare-box"><h4 class="new-h">New My Model</h4><img src="'+f.my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b&text=New+My\\'">'+b(f.new_my)+'</div>';
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
    hk=f"{prefix}/viewer/comparison_{CMP_TAG}.html"
    bs.upload_binary_to_s3(HTML_PATH,hk)
    print(f"\n🎉 完成！8产品, {sum(len(p['frames'])for p in products)}帧")
    print(f"🌐 {BLOB_CDN}/{hk}")
except Exception as e: print(f"❌ 上传失败: {e}")
