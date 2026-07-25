#!/usr/bin/env python3
"""35产品 random35_apg_rewrite 合并网页"""
import os, json, glob
from blobstore import BlobStoreClient

BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
TAG="random35_apg_rewrite"
CMP_TAG="random35_v1"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

# Get all products (exclude _raw_)
prods=sorted(set(os.path.basename(d).replace(f"_{TAG}","") for d in glob.glob(f"{BASE}/*_{TAG}") if os.path.isdir(d) and f"_raw_{TAG}" not in d))
print(f"Products: {len(prods)}")

def upload(file_paths, prefix):
    bs=BlobStoreClient(BLOB_BUCKET); m={}
    v=[p for p in set(file_paths) if p and os.path.exists(p)]
    print(f"Uploading {len(v)} files...")
    for i,fp in enumerate(v):
        try:
            fn=os.path.basename(fp); pd=os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp,f"{prefix}/{pd}/{fn}")
            m[fp]=f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1)%200==0: print(f"  {i+1}/{len(v)}")
        except Exception as e: print(f"  [FAIL] {fp}: {e}"); m[fp]=""
    print(f"Upload done: {len(v)}")
    return m

products=[]; uploads=[]
for pn in prods:
    d=os.path.join(BASE,f"{pn}_{TAG}")
    if not os.path.isdir(d): continue

    # Globals
    refs=sorted([f for f in os.listdir(d) if f.startswith("p1_ref_")and f.endswith(".png")])
    gl=[]
    for i,r in enumerate(refs):
        gl.append((f"Ref{i+1}",os.path.join(d,r))); uploads.append(os.path.join(d,r))

    # Prompt log
    plog=os.path.join(d,f"{pn}_prompt_log.json")
    prompts={}
    if os.path.exists(plog):
        for e in json.load(open(plog)): prompts[e.get("frame_index",-1)]=e

    # My images
    myf=sorted([f for f in os.listdir(d) if f.startswith("p1_my_")and f.endswith(".png")],
                key=lambda s:int(s.replace("p1_my_","").replace(".png","")))

    frames=[]
    for i in range(len(myf)):
        num=i+1
        ml=os.path.join(d,f"p1_my_{num}.png")
        if not os.path.exists(ml): ml=""
        if ml: uploads.append(ml)
        pi=prompts.get(str(i),prompts.get(i,{}))
        frames.append({
            "prompt":pi.get("rewritten_prompt",""),
            "original_prompt":pi.get("original_prompt",""),
            "my":ml,
        })
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in gl],"frames":frames})
    print(f"  {pn}: {len(gl)} refs, {len(frames)} frames")

# Upload
prefix=f"qwen_inference/comparison/{CMP_TAG}"
url_map=upload(uploads,prefix)
for pr in products:
    for g in pr["globals"]:
        if g["path"]in url_map: g["path"]=url_map[g["path"]]
    for f in pr["frames"]:
        if f["my"]in url_map: f["my"]=url_map[f["my"]]

# HTML
js=json.dumps(products,ensure_ascii=False)
html=f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Random35 APG+Rewrite ({len(products)} products)</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff}}
body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1500px;margin:0 auto}}h2{{color:var(--pri)}}
.section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box img{{width:140px;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border);margin-bottom:5px}}
.grid{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.col{{flex:0 0 480px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt{{font-size:14px;color:#1f2329;line-height:1.5;margin-bottom:6px;white-space:pre-wrap;background:#f0f8ff;padding:8px;border-radius:4px;border-left:3px solid var(--pri)}}
.orig{{font-size:12px;color:#646a73;margin-bottom:8px;white-space:pre-wrap}}
.imgbox img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.links{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}
.link{{display:inline-block;padding:6px 14px;background:var(--pri);color:#fff;border-radius:6px;font-size:13px;cursor:pointer}}
.link:hover{{opacity:0.85}}.link.active{{background:#faad14;color:#000}}
</style></head>
<body><div class="container">
<h2>Random35 — APG + Rewriter ({len(products)} products)</h2>
<div class="links" id="nav"></div>
<div id="content"><p style="color:#999">Click a product above</p></div>
</div>
<script>
var DATA={js};
(function(){{
    function e(t){{return t?String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}}):'';}}
    var h='<b>Products:</b> ';
    DATA.forEach(function(p,i){{h+='<span class="link" id="l'+i+'" onclick="s('+i+')">#'+(i+1)+' '+e(p.title)+' ('+p.frames.length+')</span> ';}});
    document.getElementById('nav').innerHTML=h;
    function s(i){{
        var p=DATA[i];
        for(var j=0;j<DATA.length;j++){{var el=document.getElementById('l'+j);if(el)el.className=(j===i)?'link active':'link';}}
        var o='<div class="section"><h2 class="title">'+e(p.title)+'</h2>';
        if(p.globals.length){{o+='<h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Reference images:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{o+='<div class="global-img-box"><img src="'+g.path+'" alt="'+e(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00\\'"><div>'+e(g.name)+'</div></div>';}});
        o+='</div>';}}
        o+='<h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Generated Images:</h3><div class="grid">';
        p.frames.forEach(function(f){{
            var or=f.original_prompt?'<div class="orig"><b>Original:</b> '+e(f.original_prompt)+'</div>':'';
            o+='<div class="col"><div class="prompt"><b>Rewritten:</b> '+e(f.prompt)+'</div>'+or+'<div class="imgbox"><img src="'+f.my+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff\\'"></div></div>';
        }});
        o+='</div></div>';document.getElementById('content').innerHTML=o;
    }}
    window.s=s;s(0);
}})();
</script></body></html>"""

HTML_PATH=os.path.join(BASE,f"random35_{CMP_TAG}.html")
with open(HTML_PATH,"w")as f: f.write(html)
print(f"\nHTML: {len(html)} bytes")
bs=BlobStoreClient(BLOB_BUCKET)
hk=f"{prefix}/viewer/random35_{CMP_TAG}.html"
bs.upload_binary_to_s3(HTML_PATH,hk)
print(f"\nDone! URL: {BLOB_CDN}/{hk}")
