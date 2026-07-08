#!/usr/bin/env python3
"""Merge all *_viewer.html into one combined HTML and upload to CDN."""

import os, json, re, html as html_mod, glob, sys
from blobstore import BlobStoreClient

SRC_DIR = '/data/phd/kousiqi/zhitao/qwen_inference_results_single'
BLOBSTORE_BUCKET = 'ad-nieuwland-material'
BLOBSTORE_CDN    = 'https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material'

OUTPUT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'merged_viewer'
BLOB_PREFIX = f'qwen_inference/all_21_products/viewer/{OUTPUT_NAME}.html'
OUTPUT_LOCAL = os.path.join(SRC_DIR, f'{OUTPUT_NAME}.html')


def build_html(all_configs, title_suffix=''):
    n = len(all_configs)
    json_str   = json.dumps(all_configs, ensure_ascii=False, indent=4)
    json_esc   = html_mod.escape(json_str)
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电商图片生成效果对比 - {n}个产品{title_suffix}</title>
    <style>
        :root {{
            --bg-color: #f5f6f7; --card-bg: #ffffff; --text-main: #1f2329;
            --text-secondary: #8f959e; --border-color: #dee0e3;
            --primary-color: #3370ff; --rewrite-color: #00b96b;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
               background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center;
                  margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }}
        .control-panel {{ background: var(--card-bg); padding: 15px; border-radius: 8px;
                          box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .control-panel textarea {{ width: 100%; height: 250px; font-family: monospace; font-size: 13px;
                                   padding: 10px; border: 1px solid var(--border-color); border-radius: 4px;
                                   box-sizing: border-box; resize: vertical; }}
        .btn {{ background-color: var(--primary-color); color: white; border: none; padding: 8px 16px;
               border-radius: 4px; cursor: pointer; margin-top: 10px; font-size: 14px; }}
        .btn:hover {{ background-color: #2458db; }}
        .product-section {{ background: var(--card-bg); border-radius: 12px; padding: 25px;
                            margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .product-title {{ margin-top: 0; color: var(--primary-color);
                          border-bottom: 1px dashed var(--border-color); padding-bottom: 10px; }}
        .section-subtitle {{ font-size: 16px; color: #555; margin: 20px 0 10px 0; }}
        .global-inputs {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .global-img-box {{ width: 140px; text-align: center; font-size: 13px; color: var(--text-secondary); }}
        .global-img-box img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 6px;
                               border: 1px solid var(--border-color); margin-bottom: 5px; }}
        .grid-container {{ display: flex; gap: 20px; overflow-x: auto; padding-bottom: 15px; }}
        .frame-column {{ flex: 0 0 280px; background: #fafafa; border-radius: 8px;
                         border: 1px solid var(--border-color); padding: 15px; display: flex; flex-direction: column; }}
        .prompt-text {{ font-size: 14px; color: #d83931; line-height: 1.5; margin-bottom: 15px;
                        min-height: 80px; white-space: pre-wrap; }}
        .original-prompt {{ font-size: 12px; color: #646a73; line-height: 1.4; margin-bottom: 10px; white-space: pre-wrap; }}
        .image-compare-box {{ margin-bottom: 15px; }}
        .image-compare-box h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--text-main); }}
        .image-compare-box img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border-color);
                                  cursor: pointer; transition: transform 0.2s; }}
        .image-compare-box img:hover {{ transform: scale(1.02); }}
        .my-model-title {{ color: var(--primary-color) !important; }}
        .my-rewrite-title {{ color: var(--rewrite-color) !important; }}
        .product-nav {{ background: var(--card-bg); border-radius: 8px; padding: 12px 15px;
                        margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                        position: sticky; top: 0; z-index: 100; }}
        .product-nav a {{ display: inline-block; margin: 3px 4px; padding: 3px 9px;
                          background: #f0f2f5; border-radius: 4px; text-decoration: none;
                          color: var(--text-main); font-size: 12px; }}
        .product-nav a:hover {{ background: var(--primary-color); color: white; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>&#x1f6cd;&#xfe0f; 电商图片编辑 - {n}个产品{title_suffix}</h2>
        <button class="btn" onclick="togglePanel()">&#x2699;&#xfe0f; 展开/隐藏配置面板</button>
    </div>
    <div class="control-panel" id="controlPanel" style="display:none;">
        <h3>配置数据 (JSON Array)</h3>
        <textarea id="mainConfig">{json_esc}</textarea>
        <button class="btn" onclick="renderPage()">&#x1f680; 渲染网页</button>
    </div>
    <div id="productNav" class="product-nav"></div>
    <div id="mainContent"></div>
</div>
<script>
    function togglePanel() {{
        const p = document.getElementById('controlPanel');
        p.style.display = p.style.display === 'none' ? 'block' : 'none';
    }}
    function escapeHtml(t) {{
        if (!t) return '';
        return String(t).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function renderPage() {{
        try {{
            const data = JSON.parse(document.getElementById('mainConfig').value);
            const container = document.getElementById('mainContent');
            container.innerHTML = '';
            const nav = document.getElementById('productNav');
            nav.innerHTML = '<b>产品导航：</b> ';
            data.forEach((p, i) => nav.innerHTML += `<a href="#p${{i}}">${{escapeHtml(p.title)}}</a>`);
            data.forEach((product, idx) => {{
                let html = `<a id="p${{idx}}"></a>
                <div class="product-section">
                    <h2 class="product-title">第 ${{idx+1}}/${{data.length}} 个: ${{escapeHtml(product.title)}}</h2>
                    <h3 class="section-subtitle">Global input:</h3>
                    <div class="global-inputs">`;
                product.globals.forEach(g => {{
                    html += `<div class="global-img-box">
                        <img src="${{g.path}}" loading="lazy" onerror="this.src='https://dummyimage.com/150x150/ffcccc/f00&text=丢失'">
                        <div>${{escapeHtml(g.name)}}</div></div>`;
                }});
                html += `</div><h3 class="section-subtitle">Performance comparison:</h3><div class="grid-container">`;
                product.frames.forEach(frame => {{
                    const origHtml = frame.original_prompt
                        ? `<div class="original-prompt"><b>Original:</b> ${{escapeHtml(frame.original_prompt)}}</div>` : '';
                    const rewriteHtml = frame.my_rewrite_path
                        ? `<div class="image-compare-box"><h4 class="my-rewrite-title">My Model (Rewrite):</h4>
                           <img src="${{frame.my_rewrite_path}}" loading="lazy"></div>` : '';
                    html += `<div class="frame-column">
                        <div class="prompt-text"><b>Rewrite Prompt:</b> ${{escapeHtml(frame.prompt)}}</div>
                        ${{origHtml}}
                        <div class="image-compare-box"><h4>nano banana:</h4>
                            <img src="${{frame.nano_path}}" loading="lazy" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=丢失'">
                        </div>
                        <div class="image-compare-box"><h4 class="my-model-title">My Model:</h4>
                            <img src="${{frame.my_path}}" loading="lazy" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=丢失'">
                        </div>
                        ${{rewriteHtml}}</div>`;
                }});
                html += `</div></div>`;
                container.innerHTML += html;
            }});
            document.getElementById('controlPanel').style.display = 'none';
        }} catch(e) {{ alert('JSON格式有误: ' + e.message); }}
    }}
    renderPage();
</script>
</body>
</html>'''


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, '*_viewer.html')))
    files = [f for f in files if 'all_21' not in os.path.basename(f) and 'merged' not in os.path.basename(f)]

    all_configs = []
    for f in files:
        with open(f) as fh:
            content = fh.read()
        m = re.search(r'<textarea id="mainConfig">(.*?)</textarea>', content, re.DOTALL)
        if m:
            cfg = json.loads(html_mod.unescape(m.group(1)))
            all_configs.extend(cfg)
        else:
            print(f'WARNING: no config in {os.path.basename(f)}')

    print(f'Loaded {len(all_configs)} products from {len(files)} files')

    title_suffix = f' ({OUTPUT_NAME})'
    html_text = build_html(all_configs, title_suffix)
    with open(OUTPUT_LOCAL, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print(f'Local HTML: {OUTPUT_LOCAL} ({os.path.getsize(OUTPUT_LOCAL)} bytes)')

    client = BlobStoreClient(BLOBSTORE_BUCKET)
    client.upload_binary_to_s3(OUTPUT_LOCAL, BLOB_PREFIX)
    url = f'{BLOBSTORE_CDN}/{BLOB_PREFIX}'
    print('=' * 50)
    print(f'HTML 公网访问地址: {url}')
    print('=' * 50)


if __name__ == '__main__':
    main()
