#!/usr/bin/env python3
"""LoRA vs Raw 对比网页"""
import os, json, glob, re, html as html_lib
import urllib.request
from blobstore import BlobStoreClient

BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"
CMP_TAG="lora_vs_raw"
LORA_PAGE_URL="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material/qwen_inference/all_21_lora_vs_fullft/viewer/lora_vs_fullft_v4.html"

# Get LoRA data from CDN
print("Downloading LoRA data...")
req=urllib.request.Request(LORA_PAGE_URL,headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=30)as r: h=r.read().decode()
m=re.search(r'<textarea[^>]*>(.*?)</textarea>', h, re.DOTALL)
lora_data=json.loads(html_lib.unescape(m.group(1)))
print(f"LoRA products: {len(lora_data)}")

def upload(file_paths, prefix):
    bs=BlobStoreClient(BLOB_BUCKET); m={}
    v=[p for p in set(file_paths) if p and os.path.exists(p)]
    print(f"Uploading {len(v)} files...")
    for i,fp in enumerate(v):
        try:
            fn=os.path.basename(fp); pd=os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp,f"{prefix}/{pd}/{fn}")
            m[fp]=f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1)%100==0: print(f"  {i+1}/{len(v)}")
        except Exception as e: print(f"  [FAIL] {fp}: {e}"); m[fp]=""
    print(f"Upload done: {len(v)}")
    return m

products=[]; uploads=[]
for lp in lora_data:
    pn=lp['title']
    raw_dir=os.path.join(BASE,f"{pn}_raw_rewrite_optimized")
    report_dir=os.path.join(BASE,f"{pn}_rewrite_optimized")
    if not os.path.isdir(raw_dir): continue

    # Collect ref images from report dir
    refs=sorted([f for f in os.listdir(report_dir) if f.startswith("p1_ref_")and f.endswith(".png")])
    chars=sorted([f for f in os.listdir(report_dir) if f.startswith("p1_char")and f.endswith((".png",".jpg"))])
    gl=[]
    for i,r in enumerate(refs):
        gl.append((f"Ref{i+1}",os.path.join(report_dir,r))); uploads.append(os.path.join(report_dir,r))
    for c in chars:
        gl.append(("Char",os.path.join(report_dir,c))); uploads.append(os.path.join(report_dir,c))

    # Raw frames
    raw_frames=sorted(glob.glob(f"{raw_dir}/frame_*.jpg"),key=lambda x:int(re.search(r'frame_(\d+)',x).group(1)))

    frames=[]
    for i in range(len(lp['frames'])):
        lora_frame=lp['frames'][i]
        # Raw image
        raw_path=os.path.join(raw_dir,f"frame_{i}.jpg")
        if not os.path.exists(raw_path): raw_path=""
        if raw_path: uploads.append(raw_path)

        frames.append({
            "prompt":lora_frame.get("original_prompt","") or lora_frame.get("prompt",""),
            "raw":raw_path,
            "lora":lora_frame.get("lora_path",""),
        })
    products.append({"title":pn,"globals":[{"name":n,"path":p}for n,p in gl],"frames":frames})
    print(f"  {pn}: {len(gl)} refs, {len(frames)} frames")

# Upload raw images
prefix=f"qwen_inference/comparison/{CMP_TAG}"
url_map=upload(uploads,prefix)
for pr in products:
    for g in pr["globals"]:
        if g["path"]in url_map: g["path"]=url_map[g["path"]]
    for f in pr["frames"]:
        if f["raw"]in url_map: f["raw"]=url_map[f["raw"]]

# HTML
js=json.dumps(products,ensure_ascii=False)
html=f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LoRA微调效果</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--lora:#00b96b;--raw:#faad14}}
body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1500px;margin:0 auto}}h2{{color:var(--pri)}}
.section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.globals{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.gbox{{width:140px;text-align:center;font-size:13px;color:var(--sub)}}
.gbox img{{width:100%;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border);margin-bottom:5px}}
.grid{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.col{{flex:0 0 560px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.ptext{{font-size:14px;line-height:1.5;margin-bottom:8px;white-space:pre-wrap;background:#f0f8ff;padding:8px;border-radius:4px;border-left:3px solid var(--pri)}}
.row{{display:flex;gap:12px}}
.box{{flex:1}}
.box h4{{font-size:13px;margin:0 0 4px 0;padding:3px 6px;border-radius:4px}}
.box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.raw-h{{background:#fff7e6;color:var(--raw)}}.lora-h{{background:#f6ffed;color:var(--lora)}}
.links{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}
.link{{display:inline-block;padding:6px 14px;background:var(--pri);color:#fff;border-radius:6px;font-size:13px;cursor:pointer}}
.link:hover{{opacity:0.85}}.link.active{{background:#faad14;color:#000}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px}}
.summary table{{border-collapse:collapse;width:100%;font-size:14px}}
.summary th,.summary td{{padding:6px 12px;text-align:center;border-bottom:1px solid var(--border)}}
.summary th{{background:#fafafa}}
</style></head>
<body><div class="container">
<h2>🛍️ LoRA微调效果 — Raw vs LoRA ({len(products)} products)</h2>
<div class="summary" id="sum"></div>
<div class="links" id="nav"></div>
<div id="content"><p style="color:#999">Click a product above</p></div>
</div>
<script>
var DATA={js};
(function(){{
    function e(t){{return t?String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}}):'';}}
    function img(url,alt){{return url?'<img src="'+url+'" alt="'+e(alt)+'" onerror="this.src=\\'https://dummyimage.com/250x350/ffcccc/f00\\'">':'<div style="width:100%;height:200px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;color:#999">N/A</div>';}}
    var s='<table><tr><th>Product</th><th>Frames</th></tr>';
    var idx='<b>Jump:</b> ';
    DATA.forEach(function(p,i){{
        s+='<tr><td style="text-align:left;cursor:pointer;color:var(--pri);text-decoration:underline" onclick="show('+i+')">'+e(p.title)+'</td><td>'+p.frames.length+'</td></tr>';
        idx+='<span class="link" id="l'+i+'" onclick="show('+i+')">#'+(i+1)+' '+e(p.title)+'</span> ';
    }});
    s+='</table>';document.getElementById('sum').innerHTML=s;
    document.getElementById('nav').innerHTML=idx;
    function show(i){{
        var p=DATA[i];
        for(var j=0;j<DATA.length;j++){{var el=document.getElementById('l'+j);if(el)el.className=(j===i)?'link active':'link';}}
        var o='<div class="section"><h2 class="title">'+e(p.title)+'</h2>';
        if(p.globals.length){{o+='<h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Reference:</h3><div class="globals">';
        p.globals.forEach(function(g){{o+='<div class="gbox">'+img(g.path,g.name)+'<div>'+e(g.name)+'</div></div>';}});
        o+='</div>';}}
        o+='<h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Raw vs LoRA:</h3><div class="grid">';
        p.frames.forEach(function(f){{
            o+='<div class="col"><div class="ptext"><b>Prompt:</b> '+e(f.prompt)+'</div>';
            o+='<div class="row">';
            o+='<div class="box"><h4 class="raw-h">Raw (标准推理)</h4>'+img(f.raw,'Raw')+'</div>';
            o+='<div class="box"><h4 class="lora-h">LoRA</h4>'+img(f.lora,'LoRA')+'</div>';
            o+='</div></div>';
        }});
        o+='</div></div>';document.getElementById('content').innerHTML=o;
    }}
    window.show=show;show(0);
}})();
</script></body></html>"""

HTML_PATH=os.path.join(BASE,f"lora_vs_raw_{CMP_TAG}.html")
with open(HTML_PATH,"w")as f: f.write(html)
print(f"\nHTML: {len(html)} bytes")
bs=BlobStoreClient(BLOB_BUCKET)
hk=f"{prefix}/viewer/index.html"
bs.upload_binary_to_s3(HTML_PATH,hk)
print(f"URL: {BLOB_CDN}/{hk}")
