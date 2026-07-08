#!/usr/bin/env python3
"""展示 Llama rewriter 输出：nano图片 + 改写prompt"""
import os, json, re
from blobstore import BlobStoreClient

JSON_PATH="/data/phd/kousiqi/zhitao/llama_factory_21products_rewriter_train.json"
BASE="/data/phd/kousiqi/zhitao/qwen_inference_results_single"
CMP_TAG="rewriter_v1"
BLOB_BUCKET="ad-nieuwland-material"; BLOB_CDN="https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

with open(JSON_PATH) as f:
    data = json.load(f)

# Group by product
from collections import defaultdict
products = defaultdict(list)
for e in data:
    pid = e["_pid"]
    products[pid].append(e)

print(f"Products: {len(products)}, Total entries: {len(data)}")

def upload(file_paths, prefix):
    bs = BlobStoreClient(BLOB_BUCKET); m = {}
    v = [p for p in set(file_paths) if p and os.path.exists(p)]
    print(f"Uploading {len(v)} files...")
    for i, fp in enumerate(v):
        try:
            fn = os.path.basename(fp); pd = os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp, f"{prefix}/{pd}/{fn}")
            m[fp] = f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1)%100 == 0: print(f"  {i+1}/{len(v)}")
        except Exception as e:
            print(f"  [FAIL] {fp}: {e}"); m[fp] = ""
    print(f"Upload done: {len(v)}")
    return m

all_products = []
uploads = []

for pid, entries in sorted(products.items()):
    entries.sort(key=lambda e: e["_frame"])
    # Get reference images from the first entry's images list
    ref_imgs = entries[0].get("images", [])
    globals_list = []
    for ri, img_path in enumerate(ref_imgs):
        if os.path.exists(img_path):
            globals_list.append((f"Ref {ri+1}", img_path))
            uploads.append(img_path)

    frames = []
    for e in entries:
        nano_path = e.get("_nano_img", "")
        if nano_path and os.path.exists(nano_path):
            uploads.append(nano_path)
        frames.append({
            "frame": e["_frame"],
            "input": e["input"],
            "output": e["output"],
            "nano": nano_path if os.path.exists(nano_path) else "",
        })

    all_products.append({
        "title": pid,
        "globals": [{"name": n, "path": p} for n, p in globals_list],
        "frames": frames,
    })
    print(f"  {pid}: {len(globals_list)} refs, {len(frames)} frames")

# Upload
prefix = f"qwen_inference/comparison/{CMP_TAG}"
url_map = upload(uploads, prefix)
for pr in all_products:
    for g in pr["globals"]:
        if g["path"] in url_map: g["path"] = url_map[g["path"]]
    for f in pr["frames"]:
        if f["nano"] and f["nano"] in url_map: f["nano"] = url_map[f["nano"]]

# HTML
js_data = json.dumps(all_products, ensure_ascii=False)
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Llama Rewriter Output - 21 products</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--yes:#00b96b}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1500px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box img{{width:140px;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border);margin-bottom:5px}}
.grid-container{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 520px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:13px;color:#646a73;line-height:1.5;margin-bottom:6px;white-space:pre-wrap}}
.output-text{{font-size:14px;color:#1f2329;line-height:1.5;margin-bottom:10px;white-space:pre-wrap;background:#f0f8ff;padding:8px;border-radius:4px;border-left:3px solid var(--pri)}}
.image-box{{}}
.image-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.image-box h4{{font-size:13px;margin:0 0 4px 0;color:var(--sub)}}
.summary{{background:var(--card);padding:15px 25px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}}
.product-links{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}
.product-link{{display:inline-block;padding:6px 14px;background:var(--pri);color:#fff;border-radius:6px;font-size:13px;cursor:pointer}}
.product-link:hover{{opacity:0.85}}.product-link.active{{background:#faad14;color:#000}}
</style></head>
<body><div class="container">
<h2>Llama Rewriter - Nano image + Rewritten prompt (21 products)</h2>
<div class="summary" id="summaryBox"></div>
<div class="product-links" id="productIndex"></div>
<div id="mainContent"><p style="color:#999">Click a product above</p></div>
</div>
<script>
var DATA = {js_data};
(function(){{
    function esc(t){{if(!t)return'';return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}})}}
    var idxHtml='<b>Jump:</b> ';
    DATA.forEach(function(p,pi){{
        idxHtml+='<span class="product-link" id="link'+pi+'" onclick="show('+pi+')">#'+(pi+1)+' '+esc(p.title)+'</span> ';
    }});
    document.getElementById('productIndex').innerHTML=idxHtml;

    function show(pi){{
        var p=DATA[pi];
        for(var i=0;i<DATA.length;i++){{var el=document.getElementById('link'+i);if(el)el.className=(i===pi)?'product-link active':'product-link';}}
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+' ('+p.frames.length+' frames)</h2>';
        if(p.globals.length){{h+='<h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Reference images:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00\\'"><div>'+esc(g.name)+'</div></div>';}});
        h+='</div>';}}
        h+='<h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Nano Image + Rewritten Prompt:</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            h+='<div class="frame-column">';
            h+='<div class="output-text"><b>Rewritten (output):</b> '+esc(f.output)+'</div>';
            h+='<div class="prompt-text"><b>Original:</b> '+esc(f.input)+'</div>';
            h+='<div class="image-box"><h4>Nano banana image</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff\\'"></div>';
            h+='</div>';
        }});
        h+='</div></div>';document.getElementById('mainContent').innerHTML=h;
    }}
    window.show=show;show(0);
}})();
</script></body></html>"""

HTML_PATH = os.path.join(BASE, f"rewriter_{CMP_TAG}.html")
with open(HTML_PATH, "w") as f: f.write(html)
print(f"\nHTML: {HTML_PATH} ({len(html)} bytes)")

try:
    bs = BlobStoreClient(BLOB_BUCKET)
    hk = f"{prefix}/viewer/rewriter_{CMP_TAG}.html"
    bs.upload_binary_to_s3(HTML_PATH, hk)
    print(f"\nDone! {len(all_products)} products, {sum(len(p['frames'])for p in all_products)} frames")
    print(f"URL: {BLOB_CDN}/{hk}")
except Exception as e: print(f"Upload failed: {e}")
