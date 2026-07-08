#!/usr/bin/env python3
"""Merge all *_rewrite_optimized_viewer.html into one combined HTML and upload to CDN."""

import os, re, json, html as html_lib, glob
from blobstore import BlobStoreClient

SRC_DIR = '/data/phd/kousiqi/zhitao/qwen_inference_results_single'
OUT_LOCAL = '/data/phd/kousiqi/zhitao/rewrite_optimized_merged.html'
BLOBSTORE_BUCKET = 'ad-nieuwland-material'
BLOBSTORE_CDN = 'https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material'
BLOB_KEY = 'qwen_inference/merged/rewrite_optimized/viewer/rewrite_optimized.html'


def load_configs():
    files = sorted(glob.glob(os.path.join(SRC_DIR, '*_rewrite_optimized_viewer.html')))
    files = [f for f in files if 'merged' not in os.path.basename(f)]
    print(f'Found {len(files)} individual viewer files')
    configs = []
    seen = set()
    for f in files:
        with open(f, encoding='utf-8') as fh:
            content = fh.read()
        m = re.search(r'<textarea[^>]*id="mainConfig"[^>]*>(.*?)</textarea>', content, re.DOTALL)
        if not m:
            print(f'  WARNING: no config in {os.path.basename(f)}')
            continue
        items = json.loads(html_lib.unescape(m.group(1)))
        for item in items:
            title = item.get('title', '')
            if title in seen:
                print(f'  SKIP duplicate: {title}')
                continue
            seen.add(title)
            configs.append(item)
    print(f'Loaded {len(configs)} unique products')
    return configs


def build_html(products):
    config_json = json.dumps(products, ensure_ascii=False, indent=4)
    escaped = html_lib.escape(config_json)
    n = len(products)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rewrite Optimized 推理结果 ({n} 个产品)</title>
    <style>
        :root {{
            --bg-color: #f5f6f7;
            --card-bg: #ffffff;
            --text-main: #1f2329;
            --text-secondary: #8f959e;
            --border-color: #dee0e3;
            --primary-color: #3370ff;
            --nano-color: #6b7280;
            --mymodel-color: #16a34a;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
        .container {{ max-width: 1500px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }}
        .legend {{ font-size: 13px; color: var(--text-secondary); margin-top: 6px; }}
        .legend .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: white; margin-right: 6px; }}
        .legend .nano {{ background: var(--nano-color); }}
        .legend .mymodel {{ background: var(--mymodel-color); }}
        .control-panel {{ background: var(--card-bg); padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .control-panel textarea {{ width: 100%; height: 250px; font-family: monospace; font-size: 13px; padding: 10px; border: 1px solid var(--border-color); border-radius: 4px; box-sizing: border-box; resize: vertical; }}
        .btn {{ background-color: var(--primary-color); color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; font-size: 14px; }}
        .btn:hover {{ background-color: #2458db; }}
        .product-nav {{ background: var(--card-bg); border-radius: 8px; padding: 12px 15px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100; }}
        .product-nav a {{ display: inline-block; margin: 3px 4px; padding: 3px 9px; background: #f0f2f5; border-radius: 4px; text-decoration: none; color: var(--text-main); font-size: 12px; }}
        .product-nav a:hover {{ background: var(--primary-color); color: white; }}
        .product-section {{ background: var(--card-bg); border-radius: 12px; padding: 25px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .product-title {{ margin-top: 0; color: var(--primary-color); border-bottom: 1px dashed var(--border-color); padding-bottom: 10px; }}
        .section-subtitle {{ font-size: 16px; color: #555; margin: 20px 0 10px 0; }}
        .global-inputs {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .global-img-box {{ width: 140px; text-align: center; font-size: 13px; color: var(--text-secondary); }}
        .global-img-box img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color); margin-bottom: 5px; }}
        .grid-container {{ display: flex; gap: 20px; overflow-x: auto; padding-bottom: 15px; }}
        .frame-column {{ flex: 0 0 300px; background: #fafafa; border-radius: 8px; border: 1px solid var(--border-color); padding: 15px; display: flex; flex-direction: column; }}
        .prompt-text {{ font-size: 14px; color: #d83931; line-height: 1.5; margin-bottom: 10px; white-space: pre-wrap; }}
        .original-prompt {{ font-size: 12px; color: #646a73; line-height: 1.4; margin-bottom: 10px; white-space: pre-wrap; }}
        .image-compare-box {{ margin-bottom: 15px; }}
        .image-compare-box h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--text-main); }}
        .image-compare-box img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border-color); cursor: pointer; transition: transform 0.2s; }}
        .image-compare-box img:hover {{ transform: scale(1.02); }}
        .nano-title {{ color: var(--nano-color) !important; }}
        .mymodel-title {{ color: var(--mymodel-color) !important; }}
        .missing {{ width: 100%; aspect-ratio: 4/5; display: flex; align-items: center; justify-content: center; border: 1px dashed var(--border-color); border-radius: 6px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h2 style="margin:0;">&#x1f6cd;&#xfe0f; Rewrite Optimized 推理结果 &mdash; {n} 个产品</h2>
            <div class="legend">
                <span class="tag nano">Nano</span>nano banana 基线输出
                <span class="tag mymodel">My Model</span>Qwen-Image-Edit rewrite_optimized 输出
            </div>
        </div>
        <button class="btn" onclick="togglePanel()">&#x2699;&#xfe0f; 展开/隐藏配置面板</button>
    </div>

    <div class="control-panel" id="controlPanel" style="display:none">
        <h3>配置数据 (JSON Array)</h3>
        <textarea id="mainConfig">{escaped}</textarea>
        <button class="btn" onclick="renderPage()">&#x1f680; 重新渲染</button>
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

    function imgOrMissing(url, alt) {{
        if (!url) return `<div class="missing">${{escapeHtml(alt)}} 缺失</div>`;
        return `<img src="${{url}}" alt="${{escapeHtml(alt)}}" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'),{{className:'missing',textContent:'${{escapeHtml(alt)}} 加载失败'}}))">`;
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
                        ${{imgOrMissing(g.path, g.name)}}
                        <div>${{escapeHtml(g.name)}}</div>
                    </div>`;
                }});
                html += `</div>
                    <h3 class="section-subtitle">Performance comparison (Nano vs My Model):</h3>
                    <div class="grid-container">`;

                product.frames.forEach(frame => {{
                    const origHtml = frame.original_prompt
                        ? `<div class="original-prompt"><b>Original:</b> ${{escapeHtml(frame.original_prompt)}}</div>` : '';
                    html += `<div class="frame-column">
                        <div class="prompt-text"><b>Rewrite Prompt:</b> ${{escapeHtml(frame.prompt)}}</div>
                        ${{origHtml}}
                        <div class="image-compare-box">
                            <h4 class="nano-title">nano banana:</h4>
                            ${{imgOrMissing(frame.nano_path, 'Nano')}}
                        </div>
                        <div class="image-compare-box">
                            <h4 class="mymodel-title">My Model:</h4>
                            ${{imgOrMissing(frame.my_path, 'My Model')}}
                        </div>
                    </div>`;
                }});
                html += `</div></div>`;
                container.innerHTML += html;
            }});
        }} catch(e) {{ alert('JSON 格式有误: ' + e.message); }}
    }}

    renderPage();
</script>
</body>
</html>"""


def main():
    products = load_configs()
    html = build_html(products)
    with open(OUT_LOCAL, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Local HTML saved: {OUT_LOCAL} ({os.path.getsize(OUT_LOCAL)} bytes)')

    client = BlobStoreClient(BLOBSTORE_BUCKET)
    client.upload_binary_to_s3(OUT_LOCAL, BLOB_KEY)
    url = f'{BLOBSTORE_CDN}/{BLOB_KEY}'
    print('=' * 60)
    print(f'公网访问地址: {url}')
    print('=' * 60)


if __name__ == '__main__':
    main()
