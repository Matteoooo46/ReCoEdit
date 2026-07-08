#!/usr/bin/env python3
"""Old vs New 对比网页：grpo_round2_old vs rewrite_optimized（GRPO round2）"""
import os, json
from blobstore import BlobStoreClient

BASE = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
OLD_TAG = "rewrite_optimized_grpo_round2_old"
NEW_TAG = "rewrite_optimized"
CMP_TAG = "cmp_grpo_round2_vs_old"

BLOB_BUCKET = "ad-nieuwland-material"
BLOB_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

PRODUCTS = [
    "workspace_item_23397479040558", "workspace_item_25936355083926",
    "workspace_item_25917400924058", "workspace_25833632310597_1772644206",
    "workspace_4959165917841_1772653742", "workspace_item_21825264046857",
]

HTML_PATH = os.path.join(BASE, f"comparison_{CMP_TAG}.html")


def upload_files(file_paths, blob_prefix):
    blobstore = BlobStoreClient(BLOB_BUCKET)
    path_to_url = {}
    valid_paths = list(set(p for p in file_paths if p and os.path.exists(p)))
    print(f"☁️  正在上传 {len(valid_paths)} 个文件到 {blob_prefix} ...")
    for idx, file_path in enumerate(valid_paths):
        try:
            fn = os.path.basename(file_path)
            pd = os.path.basename(os.path.dirname(file_path))
            bs_key = f"{blob_prefix}/{pd}/{fn}"
            blobstore.upload_binary_to_s3(file_path, bs_key)
            path_to_url[file_path] = f"{BLOB_CDN}/{bs_key}"
            if (idx + 1) % 50 == 0:
                print(f"  已上传 {idx + 1}/{len(valid_paths)} ...")
        except Exception as e:
            print(f"  [上传失败] {file_path}: {e}")
            path_to_url[file_path] = ""
    print(f"✅ 上传完成: {len(valid_paths)} 个文件")
    return path_to_url


def collect_frames(report_dir, raw_dir):
    """返回 [(nano_local, my_local)], 按 p1_nano_X 排序，X=1,2,..."""
    frames = []
    if not os.path.isdir(report_dir):
        return frames
    for fname in sorted(os.listdir(report_dir)):
        if fname.startswith("p1_nano_") and fname.endswith(".png"):
            num = fname.replace("p1_nano_", "").replace(".png", "")
            nano = os.path.join(report_dir, fname)
            my = os.path.join(report_dir, f"p1_my_{num}.png")
            my_raw = os.path.join(raw_dir, f"frame_{int(num)-1}.jpg") if raw_dir else None
            if not os.path.exists(my) and my_raw and os.path.exists(my_raw):
                my = my_raw
            frames.append((nano, my))
    return frames


def collect_globals(report_dir):
    """收集参考图"""
    refs = sorted([f for f in os.listdir(report_dir) if f.startswith("p1_ref_") and f.endswith(".png")])
    chars = sorted([f for f in os.listdir(report_dir) if f.startswith("p1_char") and f.endswith((".png", ".jpg"))])
    globals_list = []
    for i, r in enumerate(refs):
        globals_list.append((f"产品参考{i+1}", os.path.join(report_dir, r)))
    for c in chars:
        globals_list.append(("人物参考", os.path.join(report_dir, c)))
    return globals_list


def main():
    all_products = []
    all_upload_paths = []

    for pn in PRODUCTS:
        old_report = os.path.join(BASE, f"{pn}_{OLD_TAG}")
        old_raw = os.path.join(BASE, f"{pn}_raw_{OLD_TAG}")
        new_report = os.path.join(BASE, f"{pn}_{NEW_TAG}")
        new_raw = os.path.join(BASE, f"{pn}_raw_{NEW_TAG}")

        # 读取 new 的 prompt log 获取 prompt 文本
        prompt_log_path = os.path.join(new_report, f"{pn}_prompt_rewrite_log.json")
        prompts = {}
        if os.path.exists(prompt_log_path):
            with open(prompt_log_path) as f:
                for entry in json.load(f):
                    prompts[entry.get("frame_index", -1)] = entry

        # 收集图片
        old_frames = collect_frames(old_report, old_raw)
        new_frames = collect_frames(new_report, new_raw)
        globals_list = collect_globals(new_report)  # 用新的参考图

        # 上传 globals
        for _, gpath in globals_list:
            all_upload_paths.append(gpath)

        # 对齐 old 和 new frame
        max_frames = max(len(old_frames), len(new_frames))
        product_frames = []
        for i in range(max_frames):
            old_nano = old_frames[i][0] if i < len(old_frames) and os.path.exists(old_frames[i][0]) else ""
            old_my   = old_frames[i][1] if i < len(old_frames) and os.path.exists(old_frames[i][1]) else ""
            new_nano = new_frames[i][0] if i < len(new_frames) and os.path.exists(new_frames[i][0]) else ""
            new_my   = new_frames[i][1] if i < len(new_frames) and os.path.exists(new_frames[i][1]) else ""

            for p in [old_nano, old_my, new_nano, new_my]:
                if p:
                    all_upload_paths.append(p)

            prompt_info = prompts.get(i, {})
            product_frames.append({
                "prompt": prompt_info.get("rewritten_prompt", ""),
                "original_prompt": prompt_info.get("original_prompt", ""),
                "old_nano": old_nano,
                "old_my": old_my,
                "new_nano": new_nano,
                "new_my": new_my,
            })

        all_products.append({
            "title": pn,
            "globals": [{"name": n, "path": p} for n, p in globals_list],
            "frames": product_frames,
            "old_frames": len(old_frames),
            "new_frames": len(new_frames),
        })
        print(f"✅ {pn}: old={len(old_frames)} new={len(new_frames)}")

    # 上传
    print(f"\n📤 共 {len(all_upload_paths)} 个文件")
    blob_prefix = f"qwen_inference/comparison/{CMP_TAG}"
    url_map = upload_files(all_upload_paths, blob_prefix)

    # 替换路径
    for prod in all_products:
        for g in prod["globals"]:
            if g["path"] in url_map: g["path"] = url_map[g["path"]]
        for f in prod["frames"]:
            for k in ["old_nano", "old_my", "new_nano", "new_my"]:
                if f[k] in url_map: f[k] = url_map[f[k]]

    # 生成 HTML
    config_json = json.dumps(all_products, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GRPO Round2 对比 — Old vs New</title>
<style>
:root {{ --bg:#f5f6f7; --card:#fff; --text:#1f2329; --sub:#8f959e; --border:#dee0e3; --pri:#3370ff; --old:#faad14; --new:#00b96b; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg); color:var(--text); margin:0; padding:20px; }}
.container {{ max-width:1500px; margin:0 auto; }}
h2 {{ color:var(--pri); }}
.product-section {{ background:var(--card); border-radius:12px; padding:25px; margin-bottom:40px; box-shadow:0 4px 12px rgba(0,0,0,.08); }}
.product-title {{ margin-top:0; color:var(--pri); border-bottom:1px dashed var(--border); padding-bottom:10px; }}
.global-inputs {{ display:flex; gap:15px; margin-bottom:20px; flex-wrap:wrap; }}
.global-img-box {{ width:140px; text-align:center; font-size:13px; color:var(--sub); }}
.global-img-box img {{ width:100%; height:140px; object-fit:cover; border-radius:6px; border:1px solid var(--border); }}
.grid-container {{ display:flex; gap:15px; overflow-x:auto; padding-bottom:15px; }}
.frame-column {{ flex:0 0 540px; background:#fafafa; border-radius:8px; border:1px solid var(--border); padding:15px; }}
.prompt-text {{ font-size:14px; color:#d83931; line-height:1.5; margin-bottom:15px; white-space:pre-wrap; }}
.original-prompt {{ font-size:12px; color:#646a73; line-height:1.4; margin-bottom:10px; white-space:pre-wrap; }}
.compare-row {{ display:flex; gap:15px; }}
.compare-box {{ flex:1; }}
.compare-box h4 {{ font-size:13px; margin:0 0 6px 0; padding:3px 6px; border-radius:4px; }}
.compare-box img {{ width:100%; border-radius:6px; border:1px solid var(--border); }}
.old-header {{ background:#fff7e6; color:var(--old); }}
.new-header {{ background:#f6ffed; color:var(--new); }}
.summary {{ background:var(--card); padding:15px 25px; border-radius:8px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,.05); }}
.summary table {{ border-collapse:collapse; width:100%; font-size:14px; }}
.summary th, .summary td {{ padding:6px 12px; text-align:center; border-bottom:1px solid var(--border); }}
.summary th {{ background:#fafafa; }}
</style></head>
<body><div class="container">
<h2>🛍️ GRPO Round2 对比 — Old（之前LoRA）vs New（GRPO checkpoint）</h2>
<div class="summary" id="summaryBox"></div>
<div id="mainContent">加载中...</div>
</div>
<script>
var DATA = {config_json};
(function(){{
    function esc(t){{ if(!t)return''; return String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}}); }}
    var c=document.getElementById('mainContent'),s='<table><tr><th>产品</th><th>Old帧</th><th>New帧</th></tr>';
    c.innerHTML='';
    DATA.forEach(function(p){{
        s+='<tr><td style="text-align:left">'+esc(p.title)+'</td><td>'+p.old_frames+'</td><td>'+p.new_frames+'</td></tr>';
        var h='<div class="product-section"><h2 class="product-title">'+esc(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px 0;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{ h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+esc(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=丢失\\'"><div>'+esc(g.name)+'</div></div>'; }});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px 0;">Performance comparison (Left=Old, Right=New):</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var orig=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+esc(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> '+esc(f.prompt)+'</div>'+orig;
            h+='<div class="compare-row">';
            // Old column
            h+='<div class="compare-box"><h4 class="old-header">Old — nano banana</h4><img src="'+f.old_nano+'" onerror="this.src=\\'https://dummyimage.com/260x350/fff7e6/faad14&text=Old+Nano\\'"></div>';
            h+='<div class="compare-box"><h4 class="new-header">New — nano banana</h4><img src="'+f.new_nano+'" onerror="this.src=\\'https://dummyimage.com/260x350/f6ffed/00b96b&text=New+Nano\\'"></div>';
            h+='</div><div class="compare-row" style="margin-top:10px;">';
            h+='<div class="compare-box"><h4 class="old-header">Old — My Model</h4><img src="'+f.old_my+'" onerror="this.src=\\'https://dummyimage.com/260x350/fff7e6/faad14&text=Old+My\\'"></div>';
            h+='<div class="compare-box"><h4 class="new-header">New — My Model</h4><img src="'+f.new_my+'" onerror="this.src=\\'https://dummyimage.com/260x350/f6ffed/00b96b&text=New+My\\'"></div>';
            h+='</div></div>';
        }});
        h+='</div></div>'; c.innerHTML+=h;
    }});
    s+='</table>'; document.getElementById('summaryBox').innerHTML=s;
}})();
</script></body></html>"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML: {HTML_PATH}")

    # 上传
    try:
        blobstore = BlobStoreClient(BLOB_BUCKET)
        html_key = f"{blob_prefix}/viewer/comparison_{CMP_TAG}.html"
        blobstore.upload_binary_to_s3(HTML_PATH, html_key)
        url = f"{BLOB_CDN}/{html_key}"
        tf = sum(p["old_frames"] + p["new_frames"] for p in all_products)
        print(f"\n🎉 完成！{len(all_products)}产品, 共{tf}帧对比")
        print(f"🌐 {url}")
    except Exception as e:
        print(f"❌ 上传失败: {e}")


if __name__ == "__main__":
    main()
