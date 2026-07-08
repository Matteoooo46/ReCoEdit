#!/usr/bin/env python3
"""15 UUID 产品对比：Pre-RL（新 rewrite_optimized）vs GRPO Round2（_grpo_round2）"""
import os, json, re, base64, argparse
from io import BytesIO
for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(v,None)
from openai import OpenAI
from PIL import Image
from blobstore import BlobStoreClient

VLM_URL="http://10.15.2.90:8080/v1"; VLM_KEY="flowgrpo"; VLM_MODEL="Qwen3-VL-30B-A3B-Instruct"
BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
OLD_TAG="rewrite_optimized_grpo_round2"  # GRPO round2 (RL)
NEW_TAG="rewrite_optimized"              # Pre-RL original
CMP_TAG="cmp_15uuid_rl_vs_prerl_v1"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

PRODUCTS=[
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae","009785b8-36c6-4775-ada0-c9497e7072c2",
    "010ccf98-82c3-40f9-8284-65518eeff3a0","024419f0-d5c4-482e-9572-ba7885cdf4e4",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8","02cdc727-ab85-4692-8ef6-00b725c64141",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd","03641bdb-7a11-4c05-83a3-347d535e8c91",
    "03d7f47e-bb37-4a19-8685-0e231f933627","03dac08d-8b1e-4a3b-a2d7-94e0ae9ee787",
    "056924c6-1454-4e99-a40c-b8e0b362529c","064eb1bd-65cd-4fc5-8316-a172aa2f8f2f",
    "07b28d8b-d588-4683-b55d-83d59a89a9b0","07b41236-33e3-4fec-bf34-c500ef7fb220",
    "08936a21-6a4a-40c1-828f-f30763afdf02",
]

SCORES_FILE=os.path.join(BASE,f"scores_{CMP_TAG}.json")
HTML_PATH=os.path.join(BASE,f"comparison_{CMP_TAG}.html")

SCORING_PROMPT="""你是一位专业的商品图像质量审核专家。比较两张图片中同一件产品的外观一致性。

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

def p2b(img):
    b=BytesIO(); img.save(b,format="PNG")
    return f"data:image;base64,{base64.b64encode(b.getvalue()).decode()}"

def score_pair(client, name, ref, gen):
    try:
        r=client.chat.completions.create(model=VLM_MODEL,messages=[{"role":"user","content":[
            {"type":"text","text":SCORING_PROMPT.format(product_name=name)},
            {"type":"image_url","image_url":{"url":p2b(ref)}},
            {"type":"image_url","image_url":{"url":p2b(gen)}}]}],temperature=0.2)
        m=re.search(r"Score:\s*(\d)",r.choices[0].message.content.strip())
        return int(m.group(1))if m else -1
    except: return -1

def upload(file_paths, prefix):
    bs=BlobStoreClient(BLOB_BUCKET); m={}
    v=[p for p in set(file_paths) if p and os.path.exists(p)]
    print(f"\n☁️ 上传 {len(v)} 文件到 {prefix} ...")
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

# --- 评分 ---
ap=argparse.ArgumentParser(); ap.add_argument("--skip-score",action="store_true"); args=ap.parse_args()
all_scores={}

if not args.skip_score:
    client=OpenAI(base_url=VLM_URL,api_key=VLM_KEY,timeout=120)
    total=0
    for pi,pn in enumerate(PRODUCTS):
        new_dir=os.path.join(BASE,f"{pn}_{NEW_TAG}")
        old_dir=os.path.join(BASE,f"{pn}_{OLD_TAG}")
        refs=sorted([f for f in os.listdir(new_dir) if f.startswith("p1_ref_")and f.endswith(".png")])
        if not refs: continue
        ref=Image.open(os.path.join(new_dir,refs[0])).convert("RGB")
        nm=pn[:40]

        nano=sorted([f for f in os.listdir(new_dir) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
        old_my=sorted([f for f in os.listdir(old_dir) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))
        new_my=sorted([f for f in os.listdir(new_dir) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))

        ps={"nano":{},"old_my":{},"new_my":{}}
        est=len(nano)+len(old_my)+len(new_my)
        print(f"\n[{pi+1}/15] {pn} (nano={len(nano)} old_my={len(old_my)} new_my={len(new_my)} calls={est})")

        for fn in nano:
            s=score_pair(client,nm,ref,Image.open(os.path.join(new_dir,fn)).convert("RGB"))
            ps["nano"][fn]=s; total+=1; print(f"  nano {fn}: {s}/5")
        for fn in old_my:
            s=score_pair(client,nm,ref,Image.open(os.path.join(old_dir,fn)).convert("RGB"))
            ps["old_my"][fn]=s; total+=1; print(f"  old  {fn}: {s}/5")
        for fn in new_my:
            s=score_pair(client,nm,ref,Image.open(os.path.join(new_dir,fn)).convert("RGB"))
            ps["new_my"][fn]=s; total+=1; print(f"  new  {fn}: {s}/5")

        all_scores[pn]=ps
    with open(SCORES_FILE,"w")as f: json.dump(all_scores,f,ensure_ascii=False,indent=2)
    print(f"\n✅ 打分: {total} 对 -> {SCORES_FILE}")
else:
    with open(SCORES_FILE)as f: all_scores=json.load(f)
    print(f"✅ 从 {SCORES_FILE} 加载")

# --- 构建 HTML ---
products=[]; uploads=[]
for pn in PRODUCTS:
    new_dir=os.path.join(BASE,f"{pn}_{NEW_TAG}")
    old_dir=os.path.join(BASE,f"{pn}_{OLD_TAG}")
    ps=all_scores.get(pn,{})

    # globals
    refs=sorted([f for f in os.listdir(new_dir) if f.startswith("p1_ref_")and f.endswith(".png")])
    globals_list=[]
    for i,r in enumerate(refs):
        globals_list.append((f"产品参考{i+1}",os.path.join(new_dir,r)))
    for _,p in globals_list: uploads.append(p)

    # frames
    nano=sorted([f for f in os.listdir(new_dir) if f.startswith("p1_nano_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_nano_"))
    old_my=sorted([f for f in os.listdir(old_dir) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))
    new_my=sorted([f for f in os.listdir(new_dir) if f.startswith("p1_my_")and f.endswith(".png")],key=lambda s:num_sort(s,"p1_my_"))

    # prompt log from new
    plog=os.path.join(new_dir,f"{pn}_prompt_rewrite_log.json")
    prompts={}
    if os.path.exists(plog):
        for e in json.load(open(plog)): prompts[e.get("frame_index",-1)]=e

    frames=[]
    for i in range(len(new_my)):
        num=i+1
        nf=f"p1_nano_{num}.png"; mf=f"p1_my_{num}.png"
        nl=os.path.join(new_dir,nf) if os.path.exists(os.path.join(new_dir,nf)) else ""
        ol_=os.path.join(old_dir,mf) if os.path.exists(os.path.join(old_dir,mf)) else ""
        ml=os.path.join(new_dir,mf) if os.path.exists(os.path.join(new_dir,mf)) else ""
        for p in [nl,ol_,ml]:
            if p: uploads.append(p)
        pi=prompts.get(i,{})
        frames.append({
            "prompt":pi.get("rewritten_prompt",""),
            "original_prompt":pi.get("original_prompt",""),
            "nano":nl,"old_my":ol_,"new_my":ml,
            "nano_score":ps.get("nano",{}).get(nf,-1)if ps.get("nano",{}).get(nf,-1)>0 else None,
            "old_score":ps.get("old_my",{}).get(mf,-1)if ps.get("old_my",{}).get(mf,-1)>0 else None,
            "new_score":ps.get("new_my",{}).get(mf,-1)if ps.get("new_my",{}).get(mf,-1)>0 else None,
        })

    ns=[f["nano_score"]for f in frames if f["nano_score"]]
    os_=[f["old_score"]for f in frames if f["old_score"]]
    ms=[f["new_score"]for f in frames if f["new_score"]]
    na=sum(ns)/len(ns)if ns else 0; oa=sum(os_)/len(os_)if os_ else 0; ma=sum(ms)/len(ms)if ms else 0
    print(f"✅ {pn}: nano={na:.2f} pre-RL={ma:.2f} RL={oa:.2f}")
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in globals_list],"frames":frames})

# 上传
prefix=f"qwen_inference/comparison/{CMP_TAG}"
url_map=upload(uploads,prefix)

for pr in products:
    for g in pr["globals"]:
        if g["path"]in url_map: g["path"]=url_map[g["path"]]
    for f in pr["frames"]:
        for k in["nano","old_my","new_my"]:
            if f[k]in url_map: f[k]=url_map[f[k]]

# HTML
cfg=json.dumps(products,ensure_ascii=False)
html=f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>15产品对比 — Pre-RL vs GRPO Round2（含评分）</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--pre:#faad14;--rl:#00b96b;--hi:#00b96b;--mid:#faad14;--lo:#ff4d4f}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1600px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box{{width:140px;text-align:center;font-size:13px;color:var(--sub)}}
.global-img-box img{{width:100%;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border)}}
.grid-container{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 800px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:14px;color:#d83931;line-height:1.5;margin-bottom:10px;white-space:pre-wrap}}
.original-prompt{{font-size:12px;color:#646a73;margin-bottom:10px;white-space:pre-wrap}}
.compare-row{{display:flex;gap:12px}}
.compare-box{{flex:1}}
.compare-box h4{{font-size:13px;margin:0 0 4px 0;padding:3px 6px;border-radius:4px}}
.compare-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.nano-h{{background:#e6f7ff;color:#1890ff}}.pre-h{{background:#fff7e6;color:var(--pre)}}.rl-h{{background:#f6ffed;color:var(--rl)}}
.score{{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px;margin-top:3px}}
.s-1,.s-2{{background:#fff2f0;color:var(--lo)}}.s-3{{background:#fffbe6;color:var(--mid)}}.s-4,.s-5{{background:#f6ffed;color:var(--hi)}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
.summary table{{border-collapse:collapse;width:100%;font-size:14px}}
.summary th,.summary td{{padding:6px 12px;text-align:center;border-bottom:1px solid var(--border)}}
.summary th{{background:#fafafa}}
</style></head>
<body><div class="container">
<h2>🛍️ 15产品对比 — Pre-RL（原始checkpoint）vs GRPO Round2（RL后）（含评分）</h2>
<div class="summary" id="summaryBox"></div>
<div id="mainContent">加载中...</div>
</div>
<script>
var DATA={cfg};
(function(){{
    function esc(t){{if(!t)return'';return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}})}}
    function b(s){{if(s==null)return'';var c='s-'+Math.min(s,5);return'<span class="score '+c+'">Score: '+s+'/5</span>'}}
    var c=document.getElementById('mainContent'),s='<table><tr><th>产品</th><th>帧数</th><th>Pre-RL</th><th>GRPO RL</th><th>变化</th></tr>';
    c.innerHTML='';
    DATA.forEach(function(p){{
        var pm=[],rm=[];
        p.frames.forEach(function(f){{if(f.new_score)pm.push(f.new_score);if(f.old_score)rm.push(f.old_score)}});
        var pa=pm.length?(pm.reduce(function(a,b){{return a+b}},0)/pm.length).toFixed(2):'N/A';
        var ra=rm.length?(rm.reduce(function(a,b){{return a+b}},0)/rm.length).toFixed(2):'N/A';
        var diff=(pm.length&&rm.length)?(ra-pa).toFixed(2):'N/A';
        var dc=diff>0?'color:var(--rl)':(diff<0?'color:var(--lo)':'');
        s+='<tr><td style="text-align:left">'+esc(p.title)+'</td><td>'+p.frames.length+'</td><td>'+pa+'</td><td>'+ra+'</td><td style="'+dc+'">'+(diff>=0?'+':'')+diff+'</td></tr>';
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=丢失\\'"><div>'+esc(g.name)+'</div></div>'}});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">对比（Nano | Pre-RL My | GRPO RL My）:</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var orig=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h+='<div class="compare-row">';
            h+='<div class="compare-box"><h4 class="nano-h">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff&text=Nano\\'">'+b(f.nano_score)+'</div>';
            h+='<div class="compare-box"><h4 class="pre-h">Pre-RL My Model</h4><img src="'+f.new_my+'" onerror="this.src=\\'https://dummyimage.com/250x350/fff7e6/faad14&text=PreRL\\'">'+b(f.new_score)+'</div>';
            h+='<div class="compare-box"><h4 class="rl-h">GRPO RL My Model</h4><img src="'+f.old_my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b&text=RL\\'">'+b(f.old_score)+'</div>';
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
    tf=sum(len(p["frames"])for p in products)
    print(f"\n🎉 完成！{len(products)}产品, {tf}帧")
    print(f"🌐 {BLOB_CDN}/{hk}")
except Exception as e: print(f"❌ 上传失败: {e}")
