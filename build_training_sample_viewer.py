#!/usr/bin/env python3
"""从训练数据中抽取10个产品，展示 prompt + original_prompt + nano 图片"""
import json, os, random
from collections import defaultdict
from blobstore import BlobStoreClient

META = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate_with_ultraedit500k.json"
BASE = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
BLOB_BUCKET = "ad-nieuwland-material"
BLOB_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"
CMP_TAG = "training_sample_v2"

with open(META) as f:
    data = json.load(f)

# Group by edit_image paths to find products
from collections import Counter
products = defaultdict(list)
for e in data:
    # Use the directory containing the condition image as product key
    ei = e.get("edit_image", "")
    if isinstance(ei, list):
        ei = ei[0] if ei else ""
    product_key = os.path.dirname(ei).split("/")[-2] if "/" in ei else "unknown"
    products[product_key].append(e)

# Select 10 diverse products (pick ones with most entries)
top = sorted(products.items(), key=lambda x: -len(x[1]))[:10]
print(f"Selected {len(top)} products")

def upload(file_paths, prefix):
    bs = BlobStoreClient(BLOB_BUCKET); m = {}
    v = list(set(p for p in file_paths if p and os.path.exists(p)))
    print(f"Uploading {len(v)} files...")
    for i, fp in enumerate(v):
        try:
            fn = os.path.basename(fp); pd = os.path.basename(os.path.dirname(fp))
            bs.upload_binary_to_s3(fp, f"{prefix}/{pd}/{fn}")
            m[fp] = f"{BLOB_CDN}/{prefix}/{pd}/{fn}"
            if (i+1) % 50 == 0: print(f"  {i+1}/{len(v)}")
        except: m[fp] = ""
    print(f"Upload done: {len(v)}")
    return m

all_products = []; uploads = []
for product_key, entries in top:
    # Take up to 3 entries per product for brevity
    samples = entries[:3]
    frames = []
    for e in samples:
        # Use the 'image' field directly - it IS the nano banana generated image
        nano_path = e.get("image", "")
        if nano_path and os.path.exists(nano_path):
            uploads.append(nano_path)
        frames.append({
            "prompt": e.get("prompt", ""),
            "original_prompt": e.get("original_prompt", ""),
            "nano": nano_path if os.path.exists(nano_path) else "",
        })
    all_products.append({"title": product_key, "frames": frames, "total_entries": len(entries)})
    print(f"  {product_key}: {len(entries)} total, showing {len(frames)}")

# Upload
prefix = f"qwen_inference/comparison/{CMP_TAG}"
url_map = upload(uploads, prefix)
for pr in all_products:
    for f in pr["frames"]:
        if f["nano"] in url_map: f["nano"] = url_map[f["nano"]]

# HTML
js_data = json.dumps(all_products, ensure_ascii=False)
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Training Data Sample - 10 products</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff;--yes:#00b96b}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1400px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.grid-container{{display:flex;gap:15px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 480px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-box{{font-size:14px;color:#1f2329;line-height:1.5;margin-bottom:8px;white-space:pre-wrap;background:#f0f8ff;padding:8px;border-radius:4px;border-left:3px solid var(--pri)}}
.original-box{{font-size:12px;color:#646a73;line-height:1.4;margin-bottom:10px;white-space:pre-wrap;background:#fff7e6;padding:8px;border-radius:4px;border-left:3px solid #faad14}}
.image-box{{}}
.image-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.image-box h4{{font-size:12px;margin:4px 0;color:var(--sub)}}
.product-links{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}
.product-link{{display:inline-block;padding:6px 14px;background:var(--pri);color:#fff;border-radius:6px;font-size:13px;cursor:pointer}}
.product-link:hover{{opacity:0.85}}.product-link.active{{background:#faad14;color:#000}}
</style></head>
<body><div class="container">
<h2>Training Data Sample (prompt vs original_prompt)</h2>
<div class="product-links" id="productIndex"></div>
<div id="mainContent"><p style="color:#999">Click a product above</p></div>
</div>
<script>
var DATA = {js_data};
(function(){{
    function esc(t){{if(!t)return'';return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}})}}
    var idxHtml='<b>Jump:</b> ';
    DATA.forEach(function(p,pi){{
        idxHtml+='<span class="product-link" id="link'+pi+'" onclick="show('+pi+')">#'+(pi+1)+' '+esc(p.title)+' ('+p.total_entries+' entries)</span> ';
    }});
    document.getElementById('productIndex').innerHTML=idxHtml;
    function show(pi){{
        var p=DATA[pi];
        for(var i=0;i<DATA.length;i++){{var el=document.getElementById('link'+i);if(el)el.className=(i===pi)?'product-link active':'product-link';}}
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+' ('+p.total_entries+' total entries, showing '+p.frames.length+')</h2><div class="grid-container">';
        p.frames.forEach(function(f){{
            h+='<div class="frame-column">';
            h+='<div class="prompt-box"><b>Prompt (rewritten):</b><br>'+esc(f.prompt)+'</div>';
            h+='<div class="original-box"><b>Original prompt:</b><br>'+esc(f.original_prompt)+'</div>';
            h+='<div class="image-box"><h4>Nano banana image:</h4><img src="'+f.nano+'" onerror="this.src=\\'https://dummyimage.com/250x350/e6f7ff/1890ff\\'"></div>';
            h+='</div>';
        }});
        h+='</div></div>';document.getElementById('mainContent').innerHTML=h;
    }}
    window.show=show;show(0);
}})();
</script></body></html>"""

HTML_PATH = os.path.join(BASE, f"training_sample_{CMP_TAG}.html")
with open(HTML_PATH, "w") as f: f.write(html)
print(f"\nHTML: {HTML_PATH} ({len(html)} bytes)")
try:
    bs = BlobStoreClient(BLOB_BUCKET)
    hk = f"{prefix}/viewer/training_sample_{CMP_TAG}.html"
    bs.upload_binary_to_s3(HTML_PATH, hk)
    print(f"\nDone! URL: {BLOB_CDN}/{hk}")
except Exception as e: print(f"Upload failed: {e}")
