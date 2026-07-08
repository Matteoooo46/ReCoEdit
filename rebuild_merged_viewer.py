#!/usr/bin/env python3
"""
直接使用每个产品 rewrite_optimized 文件夹中推理脚本生成的 viewer_config.json，
只替换图片路径为全新 CDN URL，生成 21 产品合并网页。
"""
import os
import json
import html as html_lib
from blobstore import BlobStoreClient

OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"
RUN_TAG = "rewrite_optimized"
NEW_RUN_TAG = "rewrite_optimized_merged_v5"  # 全新 prefix

BLOBSTORE_BUCKET = "ad-nieuwland-material"
BLOBSTORE_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"

PRODUCTS = [
    # 老 6 产品
    "workspace_item_23397479040558",
    "workspace_item_25936355083926",
    "workspace_item_25917400924058",
    "workspace_25833632310597_1772644206",
    "workspace_4959165917841_1772653742",
    "workspace_item_21825264046857",
    # 新 15 产品
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae",
    "03d7f47e-bb37-4a19-8685-0e231f933627",
    "009785b8-36c6-4775-ada0-c9497e7072c2",
    "03dac08d-8b1e-4a3b-a2d7-94e0ae9ee787",
    "010ccf98-82c3-40f9-8284-65518eeff3a0",
    "056924c6-1454-4e99-a40c-b8e0b362529c",
    "024419f0-d5c4-482e-9572-ba7885cdf4e4",
    "064eb1bd-65cd-4fc5-8316-a172aa2f8f2f",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8",
    "07b28d8b-d588-4683-b55d-83d59a89a9b0",
    "02cdc727-ab85-4692-8ef6-00b725c64141",
    "07b41236-33e3-4fec-bf34-c500ef7fb220",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd",
    "08936a21-6a4a-40c1-828f-f30763afdf02",
    "03641bdb-7a11-4c05-83a3-347d535e8c91",
]

MERGED_HTML_PATH = os.path.join(OUTPUT_BASE_DIR, f"merged_21products_{NEW_RUN_TAG}.html")


def build_viewer_html(products_config):
    """生成展示页面。JSON 数据直接写入 JS，避免 textarea 逃逸问题。"""
    config_json = json.dumps(products_config, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电商图片生成效果对比 - 21产品</title>
    <style>
        :root {{
            --bg-color: #f5f6f7;
            --card-bg: #ffffff;
            --text-main: #1f2329;
            --text-secondary: #8f959e;
            --border-color: #dee0e3;
            --primary-color: #3370ff;
            --rewrite-color: #00b96b;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }}
        .btn {{ background-color: var(--primary-color); color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; font-size: 14px; }}
        .btn:hover {{ background-color: #2458db; }}
        .product-section {{ background: var(--card-bg); border-radius: 12px; padding: 25px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .product-title {{ margin-top: 0; color: var(--primary-color); border-bottom: 1px dashed var(--border-color); padding-bottom: 10px; }}
        .section-subtitle {{ font-size: 16px; color: #555; margin: 20px 0 10px 0; }}
        .global-inputs {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .global-img-box {{ width: 140px; text-align: center; font-size: 13px; color: var(--text-secondary); }}
        .global-img-box img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color); margin-bottom: 5px; }}
        .grid-container {{ display: flex; gap: 20px; overflow-x: auto; padding-bottom: 15px; }}
        .frame-column {{ flex: 0 0 280px; background: #fafafa; border-radius: 8px; border: 1px solid var(--border-color); padding: 15px; display: flex; flex-direction: column; }}
        .prompt-text {{ font-size: 14px; color: #d83931; line-height: 1.5; margin-bottom: 15px; min-height: 80px; white-space: pre-wrap; }}
        .original-prompt {{ font-size: 12px; color: #646a73; line-height: 1.4; margin-bottom: 10px; white-space: pre-wrap; }}
        .image-compare-box {{ margin-bottom: 15px; }}
        .image-compare-box h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--text-main); }}
        .image-compare-box img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border-color); cursor: pointer; transition: transform 0.2s; }}
        .image-compare-box img:hover {{ transform: scale(1.02); }}
        .my-model-title {{ color: var(--primary-color) !important; }}
        .my-rewrite-title {{ color: var(--rewrite-color) !important; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>🛍️ 电商图片编辑 - 21产品对比</h2>
    </div>
    <div id="mainContent">正在加载...</div>
</div>

<script>
var PRODUCTS_DATA = {config_json};

(function() {{
    function escapeHtml(text) {{
        if (!text) return '';
        return String(text).replace(/[&<>"]/g, function(ch) {{
            return {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}}[ch];
        }});
    }}

    function render() {{
        var container = document.getElementById('mainContent');
        container.innerHTML = '';
        PRODUCTS_DATA.forEach(function(product) {{
            var html = '<div class="product-section"><h2 class="product-title">' + escapeHtml(product.title) + '</h2><h3 class="section-subtitle">Global input:</h3><div class="global-inputs">';
            product.globals.forEach(function(g) {{
                html += '<div class="global-img-box"><img src="' + g.path + '" alt="' + escapeHtml(g.name) + '" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00&text=图片丢失\\'"><div>' + escapeHtml(g.name) + '</div></div>';
            }});
            html += '</div><h3 class="section-subtitle">Performance comparison:</h3><div class="grid-container">';
            product.frames.forEach(function(frame) {{
                var originalHtml = frame.original_prompt ? '<div class="original-prompt"><b>Original:</b> ' + escapeHtml(frame.original_prompt) + '</div>' : '';
                html += '<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> ' + escapeHtml(frame.prompt) + '</div>' + originalHtml + '<div class="image-compare-box"><h4>nano banana:</h4><img src="' + frame.nano_path + '" alt="Nano" onerror="this.src=\\'https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失\\'"></div><div class="image-compare-box"><h4 class="my-model-title">My Model:</h4><img src="' + frame.my_path + '" alt="My Model" onerror="this.src=\\'https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失\\'"></div></div>';
            }});
            html += '</div></div>';
            container.innerHTML += html;
        }});
    }}

    render();
}})();
</script>
</body>
</html>
"""


def upload_files(file_paths, blob_prefix):
    blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
    path_to_url = {}
    valid_paths = list(set(p for p in file_paths if p and os.path.exists(p)))
    print(f"☁️  正在上传 {len(valid_paths)} 个文件到 {blob_prefix} ...")
    for idx, file_path in enumerate(valid_paths):
        try:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            bs_key = f"{blob_prefix}/{parent_dir}/{file_name}"
            blobstore.upload_binary_to_s3(file_path, bs_key)
            path_to_url[file_path] = f"{BLOBSTORE_CDN}/{bs_key}"
            if (idx + 1) % 50 == 0:
                print(f"  已上传 {idx + 1}/{len(valid_paths)} ...")
        except Exception as e:
            print(f"  [上传失败] {file_path}: {e}")
            path_to_url[file_path] = ""
    print(f"✅ 上传完成: {len(valid_paths)} 个文件")
    return path_to_url


def main():
    all_products_config = []
    all_upload_paths = []

    for product_name in PRODUCTS:
        report_dir = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_{RUN_TAG}")
        viewer_config_path = os.path.join(report_dir, f"{product_name}_viewer_config.json")

        if not os.path.exists(viewer_config_path):
            print(f"❌ {product_name}: viewer_config 不存在")
            continue

        # 直接读取推理脚本生成的 viewer_config.json
        with open(viewer_config_path, "r", encoding="utf-8") as f:
            product_config = json.load(f)

        frames = product_config.get("frames", [])
        globals_list = product_config.get("globals", [])
        title = product_config.get("title", f"{product_name} [{RUN_TAG}]")

        # 收集需要上传的本地图片文件
        # globals: 从相对路径构建本地路径
        new_globals = []
        for g in globals_list:
            gpath = g.get("path", "")
            if gpath.startswith("http"):
                # 旧 CDN URL → 在 report_dir 找对应文件
                # 尝试匹配 p1_ref / p1_char
                for prefix in ["p1_ref_", "p1_char"]:
                    for fname in sorted(os.listdir(report_dir)):
                        if fname.startswith(prefix) and fname.endswith((".png", ".jpg")):
                            local = os.path.join(report_dir, fname)
                            if local not in [x[0] for x in all_upload_paths if isinstance(x, tuple)]:
                                pass  # 下面统一处理
            else:
                # 相对路径如 "folder/p1_ref_0.png" → 提取 basename
                fname = os.path.basename(gpath)
                local = os.path.join(report_dir, fname)
                if os.path.exists(local):
                    all_upload_paths.append(local)
                    new_globals.append({"name": g.get("name", ""), "path": local})
                    continue

        # 如果 globals 没收集到，从 report_dir 直接扫描
        if not new_globals:
            ref_files = sorted([f for f in os.listdir(report_dir) if f.startswith("p1_ref_")])
            char_files = sorted([f for f in os.listdir(report_dir) if f.startswith("p1_char")])
            for idx, rf in enumerate(ref_files):
                local = os.path.join(report_dir, rf)
                all_upload_paths.append(local)
                new_globals.append({"name": f"产品参考{idx + 1}", "path": local})
            for cf in char_files:
                local = os.path.join(report_dir, cf)
                all_upload_paths.append(local)
                new_globals.append({"name": "人物参考", "path": local})

        # frames: 从 viewer_config 中的路径提取本地文件
        new_frames = []
        for f in frames:
            my_url = f.get("my_path", "")
            nano_url = f.get("nano_path", "")

            # 从 URL/相对路径提取文件名，在 report_dir 和 raw_dir 中查找
            raw_dir = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw_{RUN_TAG}")

            # my image: 在 raw_dir 中找 frame_X.jpg
            my_local = None
            my_fname = os.path.basename(my_url)
            for candidate_dir in [raw_dir, report_dir]:
                candidate = os.path.join(candidate_dir, my_fname)
                if os.path.exists(candidate):
                    my_local = candidate
                    break
            if not my_local:
                # 尝试从 my_url 模式匹配
                for candidate_dir in [raw_dir, report_dir]:
                    for img in sorted(os.listdir(candidate_dir)):
                        if my_fname in img or img.endswith(my_fname.split(".")[-1]):
                            candidate = os.path.join(candidate_dir, img)
                            if my_fname in candidate:
                                my_local = candidate
                                break
                    if my_local:
                        break

            # nano image: 在 report_dir 中找 p1_nano_X.png
            nano_local = None
            nano_fname = os.path.basename(nano_url)
            for candidate_dir in [report_dir, raw_dir]:
                candidate = os.path.join(candidate_dir, nano_fname)
                if os.path.exists(candidate):
                    nano_local = candidate
                    break
            if not nano_local:
                for img in sorted(os.listdir(report_dir)):
                    if img.startswith("p1_nano_") and nano_fname in img:
                        nano_local = os.path.join(report_dir, img)
                        break
                if not nano_local:
                    # 尝试匹配 raw dir 中的文件
                    for img in sorted(os.listdir(raw_dir) if os.path.exists(raw_dir) else []):
                        if nano_fname in img:
                            nano_local = os.path.join(raw_dir, img)
                            break

            if my_local:
                all_upload_paths.append(my_local)
            if nano_local:
                all_upload_paths.append(nano_local)

            new_frames.append({
                "prompt": f.get("prompt", ""),
                "original_prompt": f.get("original_prompt", ""),
                "nano_path": nano_local if nano_local else nano_url,
                "my_path": my_local if my_local else my_url,
            })

        all_products_config.append({
            "title": f"{product_name} [{NEW_RUN_TAG}]",
            "globals": new_globals,
            "frames": new_frames,
        })
        print(f"✅ {product_name}: {len(new_globals)} refs, {len(new_frames)} frames")

    # 上传所有本地图片
    total_files = len(all_upload_paths)
    print(f"\n📤 共 {total_files} 个文件待上传")
    merged_blob_prefix = f"qwen_inference/merged/{NEW_RUN_TAG}"
    url_map = upload_files(all_upload_paths, merged_blob_prefix)

    # 用 CDN URL 替换所有本地路径
    for product_config in all_products_config:
        for g in product_config["globals"]:
            if g["path"] in url_map:
                g["path"] = url_map[g["path"]]
        for f in product_config["frames"]:
            if f["nano_path"] in url_map:
                f["nano_path"] = url_map[f["nano_path"]]
            if f["my_path"] in url_map:
                f["my_path"] = url_map[f["my_path"]]

    # 生成 HTML
    html_text = build_viewer_html(all_products_config)
    with open(MERGED_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"✅ HTML 已保存: {MERGED_HTML_PATH}")

    # 上传 HTML
    try:
        blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
        html_bs_key = f"{merged_blob_prefix}/viewer/{os.path.basename(MERGED_HTML_PATH)}"
        blobstore.upload_binary_to_s3(MERGED_HTML_PATH, html_bs_key)
        html_url = f"{BLOBSTORE_CDN}/{html_bs_key}"
        total_frames = sum(len(p["frames"]) for p in all_products_config)
        print(f"\n{'='*60}")
        print(f"🎉 完成！{len(all_products_config)} 产品, {total_frames} 帧")
        print(f"🌐 {html_url}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"❌ HTML 上传失败: {e}")


if __name__ == "__main__":
    main()
