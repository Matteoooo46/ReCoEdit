import argparse
import glob
import html as html_lib
import json
import os


def build_viewer_html(products_config):
    config_json = json.dumps(products_config, ensure_ascii=False, indent=4)
    escaped_config_json = html_lib.escape(config_json)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电商图片生成效果对比（合并版）</title>
    <style>
        :root {{ --bg-color:#f5f6f7; --card-bg:#fff; --text-main:#1f2329; --text-secondary:#8f959e; --border-color:#dee0e3; --primary-color:#3370ff; --rewrite-color:#00b96b; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg-color); color:var(--text-main); margin:0; padding:20px; }}
        .container {{ max-width:1400px; margin:0 auto; }}
        .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; padding-bottom:10px; border-bottom:2px solid var(--border-color); }}
        .control-panel {{ background:var(--card-bg); padding:15px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.05); margin-bottom:20px; }}
        .control-panel textarea {{ width:100%; height:250px; font-family:monospace; font-size:13px; padding:10px; border:1px solid var(--border-color); border-radius:4px; box-sizing:border-box; resize:vertical; }}
        .btn {{ background:var(--primary-color); color:white; border:none; padding:8px 16px; border-radius:4px; cursor:pointer; margin-top:10px; font-size:14px; }}
        .btn:hover {{ background:#2458db; }}
        .product-section {{ background:var(--card-bg); border-radius:12px; padding:25px; margin-bottom:40px; box-shadow:0 4px 12px rgba(0,0,0,.08); }}
        .product-title {{ margin-top:0; color:var(--primary-color); border-bottom:1px dashed var(--border-color); padding-bottom:10px; }}
        .section-subtitle {{ font-size:16px; color:#555; margin:20px 0 10px; }}
        .global-inputs {{ display:flex; gap:15px; margin-bottom:20px; flex-wrap:wrap; }}
        .global-img-box {{ width:140px; text-align:center; font-size:13px; color:var(--text-secondary); }}
        .global-img-box img {{ width:100%; height:140px; object-fit:cover; border-radius:6px; border:1px solid var(--border-color); margin-bottom:5px; }}
        .grid-container {{ display:flex; gap:20px; overflow-x:auto; padding-bottom:15px; }}
        .frame-column {{ flex:0 0 280px; background:#fafafa; border-radius:8px; border:1px solid var(--border-color); padding:15px; display:flex; flex-direction:column; }}
        .prompt-text {{ font-size:14px; color:#d83931; line-height:1.5; margin-bottom:15px; min-height:80px; white-space:pre-wrap; }}
        .original-prompt {{ font-size:12px; color:#646a73; line-height:1.4; margin-bottom:10px; white-space:pre-wrap; }}
        .image-compare-box {{ margin-bottom:15px; }}
        .image-compare-box h4 {{ margin:0 0 8px; font-size:14px; color:var(--text-main); }}
        .image-compare-box img {{ width:100%; border-radius:6px; border:1px solid var(--border-color); cursor:pointer; transition:transform .2s; }}
        .image-compare-box img:hover {{ transform:scale(1.02); }}
        .my-model-title {{ color:var(--primary-color)!important; }}
        .my-rewrite-title {{ color:var(--rewrite-color)!important; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header"><h2>🛍️ 电商图片编辑</h2><button class="btn" onclick="togglePanel()">⚙️ 展开/隐藏配置面板</button></div>
    <div class="control-panel" id="controlPanel">
        <h3>配置数据 (JSON Array)</h3>
        <textarea id="mainConfig">{escaped_config_json}</textarea>
        <button class="btn" onclick="renderPage()">🚀 渲染网页</button>
    </div>
    <div id="mainContent"></div>
</div>
<script>
function togglePanel(){{const panel=document.getElementById('controlPanel');panel.style.display=panel.style.display==='none'?'block':'none';}}
function escapeHtml(text){{if(!text)return '';return String(text).replace(/[&<>\"]/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[ch]));}}
function renderPage(){{
    try{{
        const data=JSON.parse(document.getElementById('mainConfig').value);
        const container=document.getElementById('mainContent'); container.innerHTML='';
        data.forEach(product=>{{
            let html=`<div class="product-section"><h2 class="product-title">${{escapeHtml(product.title)}}</h2><h3 class="section-subtitle">Global input:</h3><div class="global-inputs">`;
            product.globals.forEach(g=>{{html+=`<div class="global-img-box"><img src="${{g.path}}" alt="${{escapeHtml(g.name)}}" onerror="this.src='https://dummyimage.com/150x150/ffcccc/f00&text=图片丢失'"><div>${{escapeHtml(g.name)}}</div></div>`;}});
            html+=`</div><h3 class="section-subtitle">Performance comparison:</h3><div class="grid-container">`;
            product.frames.forEach(frame=>{{
                let originalPromptHtml=frame.original_prompt?`<div class="original-prompt"><b>Original:</b> ${{escapeHtml(frame.original_prompt)}}</div>`:'';
                let rewriteHtml=frame.my_rewrite_path?`<div class="image-compare-box"><h4 class="my-rewrite-title">My Model (Rewrite):</h4><img src="${{frame.my_rewrite_path}}" alt="My Rewrite Model" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失'"></div>`:'';
                html+=`<div class="frame-column"><div class="prompt-text"><b>Rewrite Prompt:</b> ${{escapeHtml(frame.prompt)}}</div>${{originalPromptHtml}}<div class="image-compare-box"><h4>nano banana:</h4><img src="${{frame.nano_path}}" alt="Nano" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失'"></div><div class="image-compare-box"><h4 class="my-model-title">My Model:</h4><img src="${{frame.my_path}}" alt="My Model" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失'"></div>${{rewriteHtml}}</div>`;
            }});
            html+=`</div></div>`; container.innerHTML+=html;
        }});
        document.getElementById('controlPanel').style.display='none';
    }}catch(error){{alert('JSON 格式有误，请检查语法！\n\n错误信息: '+error.message);}}
}}
renderPage();
</script>
</body>
</html>
"""


def load_one_config(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    return [obj]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_dir", required=True, help="所有产品结果所在的父目录，例如 /data/.../qwen_inference_results_single")
    parser.add_argument("--output_html", default="merged_viewer.html", help="合并后的 html 文件名或绝对路径")
    parser.add_argument("--pattern", default="*_rewrite_expanded/*_viewer_config.json", help="相对 output_base_dir 的配置文件匹配模式")
    args = parser.parse_args()

    config_paths = sorted(glob.glob(os.path.join(args.output_base_dir, args.pattern)))
    if not config_paths:
        raise FileNotFoundError(f"没有找到配置文件：{os.path.join(args.output_base_dir, args.pattern)}")

    products = []
    for path in config_paths:
        products.extend(load_one_config(path))

    output_html = args.output_html
    if not os.path.isabs(output_html):
        output_html = os.path.join(args.output_base_dir, output_html)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(build_viewer_html(products))

    merged_json = os.path.splitext(output_html)[0] + ".json"
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=4)

    print(f"已合并 {len(products)} 个产品")
    print(f"HTML: {output_html}")
    print(f"JSON: {merged_json}")


if __name__ == "__main__":
    main()
