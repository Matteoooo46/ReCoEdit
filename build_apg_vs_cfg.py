#!/usr/bin/env python3
"""APG vs CFG 对比网页：21产品，Nano | My(CFG) | My(APG)"""
import os, json, glob
from blobstore import BlobStoreClient

BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
CFG_TAG="rewrite_optimized"
APG_TAG="rewrite_optimized_apg"
CMP_TAG="apg_vs_cfg_v2"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

# Products: all that have both CFG and APG data
apg_dirs=set(os.path.basename(d).replace(f"_{APG_TAG}","") for d in glob.glob(f"{BASE}/*_{APG_TAG}") if os.path.isdir(d) and f"_raw_{APG_TAG}" not in d)
cfg_dirs=set(os.path.basename(d).replace(f"_{CFG_TAG}","") for d in glob.glob(f"{BASE}/*_{CFG_TAG}") if os.path.isdir(d) and f"_raw_{CFG_TAG}" not in d)
PRODUCTS=sorted(apg_dirs & cfg_dirs)
print(f"Products with both CFG & APG: {len(PRODUCTS)}")

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
for pn in PRODUCTS:
    cfg_dir=os.path.join(BASE,f"{pn}_{CFG_TAG}")
    apg_dir=os.path.join(BASE,f"{pn}_{APG_TAG}")

    # Globals from APG dir (should be same as CFG for workspace, APG for UUID)
    refs=sorted([f for f in os.listdir(apg_dir) if f.startswith("p1_ref_")and f.endswith(".png")])
    chars=sorted([f for f in os.listdir(apg_dir) if f.startswith("p1_char")and f.endswith((".png",".jpg"))])
    globals_list=[]
    for i,r in enumerate(refs):
        globals_list.append((f"Ref {i+1}",os.path.join(apg_dir,r)))
        uploads.append(os.path.join(apg_dir,r))
    for c in chars:
        globals_list.append(("Char",os.path.join(apg_dir,c)))
        uploads.append(os.path.join(apg_dir,c))

    # Frames: match CFG and APG by p1_my numbering
    cfg_my=sorted([f for f in os.listdir(cfg_dir) if f.startswith("p1_my_")and f.endswith(".png")],
                   key=lambda s:int(s.replace("p1_my_","").replace(".png","")))
    apg_my=sorted([f for f in os.listdir(apg_dir) if f.startswith("p1_my_")and f.endswith(".png")],
                   key=lambda s:int(s.replace("p1_my_","").replace(".png","")))

    # Nano from APG dir
    nano_fs=sorted([f for f in os.listdir(apg_dir) if f.startswith("p1_nano_")and f.endswith(".png")],
                    key=lambda s:int(s.replace("p1_nano_","").replace(".png","")))

    # Prompt log from CFG dir
    plog=os.path.join(cfg_dir,f"{pn}_prompt_rewrite_log.json")
    prompts={}
    if os.path.exists(plog):
        for e in json.load(open(plog)): prompts[e.get("frame_index",-1)]=e

    max_f=max(len(cfg_my),len(apg_my))
    frames=[]
    for i in range(max_f):
        num=i+1
        nl=os.path.join(apg_dir,f"p1_nano_{num}.png")
        cl=os.path.join(cfg_dir,f"p1_my_{num}.png")
        al=os.path.join(apg_dir,f"p1_my_{num}.png")
        if not os.path.exists(nl): nl=""
        if not os.path.exists(cl): cl=""
        if not os.path.exists(al): al=""
        for p in [nl,cl,al]:
            if p: uploads.append(p)
        pi=prompts.get(str(i),prompts.get(i,{}))
        frames.append({
            "prompt":pi.get("rewritten_prompt",""),
            "original_prompt":pi.get("original_prompt",""),
            "nano":nl,"cfg_my":cl,"apg_my":al,
        })
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in globals_list],"frames":frames})
    print(f"  {pn}: {len(globals_list)} refs, {len(frames)} frames")

# Upload
prefix=f"qwen_inference/comparison/{CMP_TAG}"
url_map=upload(uploads,prefix)
for pr in products:
    for g in pr["globals"]:
        if g["path"]in url_map: g["path"]=url_map[g["path"]]
    for f in pr["frames"]:
        for k in["nano","cfg_my","apg_my"]:
            if f[k]in url_map: f[k]=url_map[f[k]]

# HTML
js_data=json.dumps(products,ensure_ascii=False)
html=f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>APG vs CFG Comparison - 21 products</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--cfg:#faad14;--apg:#00b96b}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1600px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box img{{width:140px;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border);margin-bottom:5px}}
.grid-container{{display:flex;gap:12px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 780px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:14px;color:#d83931;line-height:1.5;margin-bottom:4px;white-space:pre-wrap}}
.original-prompt{{font-size:12px;color:#646a73;margin-bottom:6px;white-space:pre-wrap}}
.compare-row{{display:flex;gap:10px}}
.compare-box{{flex:1}}
.compare-box h4{{font-size:13px;margin:0 0 4px 0;padding:3px 6px;border-radius:4px}}
.compare-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.nano-h{{background:#e6f7ff;color:#1890ff}}.cfg-h{{background:#fff7e6;color:var(--cfg)}}.apg-h{{background:#f6ffed;color:var(--apg)}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
.summary table{{border-collapse:collapse;width:100%;font-size:14px}}
.summary th,.summary td{{padding:6px 12px;text-align:center;border-bottom:1px solid var(--border)}}
.summary th{{background:#fafafa}}
.product-links{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}
.product-link{{display:inline-block;padding:6px 14px;background:var(--pri);color:#fff;border-radius:6px;font-size:13px;cursor:pointer}}
.product-link:hover{{opacity:0.85}}.product-link.active{{background:#faad14;color:#000}}
</style></head>
<body><div class="container">
<h2>APG vs CFG Comparison - {len(products)} products</h2>
<div class="summary" id="summaryBox"></div>
<div class="product-links" id="productIndex"></div>
<div id="mainContent"><p style="color:#999">Click a product above</p></div>
</div>
<script>
var DATA={js_data};
(function(){{
    function esc(t){{if(!t)return'';return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}})}}
    var s='<table><tr><th>Product</th><th>Frames</th></tr>';
    var idxHtml='<b>Jump:</b> ';
    DATA.forEach(function(p,pi){{
        s+='<tr><td style="text-align:left;cursor:pointer;color:var(--pri);text-decoration:underline" onclick="show('+pi+')">'+esc(p.title)+'</td><td>'+p.frames.length+'</td></tr>';
        idxHtml+='<span class="product-link" id="link'+pi+'" onclick="show('+pi+')">#'+(pi+1)+' '+esc(p.title)+'</span> ';
    }});
    s+='</table>';document.getElementById('summaryBox').innerHTML=s;
    document.getElementById('productIndex').innerHTML=idxHtml;
    function show(pi){{
        var p=DATA[pi];
        for(var i=0;i<DATA.length;i++){{var el=document.getElementById('link'+i);if(el)el.className=(i===pi)?'product-link active':'product-link';}}
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00\\'"><div>'+esc(g.name)+'</div></div>';}});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Comparison: Nano | CFG | APG</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var orig=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h+='<div class="compare-row">';
            h+='<div class="compare-box"><h4 class="nano-h">Nano banana</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff\\'"></div>';
            h+='<div class="compare-box"><h4 class="cfg-h">My Model (CFG)</h4><img src="'+f.cfg_my+'" onerror="this.src=\\'https://dummyimage.com/250x350/fff7e6/faad14\\'"></div>';
            h+='<div class="compare-box"><h4 class="apg-h">My Model (APG)</h4><img src="'+f.apg_my+'" onerror="this.src=\\'https://dummyimage.com/250x350/f6ffed/00b96b\\'"></div>';
            h+='</div></div>';
        }});
        h+='</div></div>';document.getElementById('mainContent').innerHTML=h;
    }}
    window.show=show;show(0);
}})();
</script></body></html>"""

HTML_PATH=os.path.join(BASE,f"apg_vs_cfg_{CMP_TAG}.html")
with open(HTML_PATH,"w")as f: f.write(html)
print(f"\nHTML: {HTML_PATH} ({len(html)} bytes)")

try:
    bs=BlobStoreClient(BLOB_BUCKET)
    hk=f"{prefix}/viewer/apg_vs_cfg_{CMP_TAG}.html"
    bs.upload_binary_to_s3(HTML_PATH,hk)
    tf=sum(len(p["frames"])for p in products)
    print(f"\nDone! {len(products)} products, {tf} frames")
    print(f"URL: {BLOB_CDN}/{hk}")
except Exception as e: print(f"Upload failed: {e}")
