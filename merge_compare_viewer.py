#!/usr/bin/env python3
"""
合并两批推理结果 HTML，按产品配对做上下对比：
  - 上：未加入约束 (test_original)
  - 下：训练数据增强 + 约束 (data_enhanced)
"""

import os, sys, json, re, html as html_mod, glob
from blobstore import BlobStoreClient

SRC_DIR         = '/data/phd/kousiqi/zhitao/qwen_inference_results_single'
BLOBSTORE_BUCKET = 'ad-nieuwland-material'
BLOBSTORE_CDN    = 'https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material'

FILE_A = os.path.join(SRC_DIR, 'all_21_test_original_viewer.html')   # 未加入约束（已合并）
# B: 扫描所有 *data_enhanced*_viewer.html 个人产品文件
DATA_ENHANCED_PATTERN = os.path.join(SRC_DIR, '*data_enhanced*viewer.html')

OUTPUT_NAME  = sys.argv[1] if len(sys.argv) > 1 else 'compare_test_vs_data_enhanced'
OUTPUT_LOCAL = os.path.join(SRC_DIR, f'{OUTPUT_NAME}.html')
BLOB_KEY     = f'qwen_inference/all_21_products/viewer/{OUTPUT_NAME}.html'


def extract_configs(html_path):
    with open(html_path, encoding='utf-8') as f:
        content = f.read()
    m = re.search(r'<textarea id="mainConfig">(.*?)</textarea>', content, re.DOTALL)
    if not m:
        raise ValueError(f'No config found in {html_path}')
    return json.loads(html_mod.unescape(m.group(1)))


def build_html(pairs, title='对比：未加入约束 vs 数据增强+约束'):
    n = len(pairs)
    all_data = [{"a": a, "b": b} for a, b in pairs]
    json_str  = json.dumps(all_data, ensure_ascii=False, indent=2)
    json_esc  = html_mod.escape(json_str)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg:#f5f6f7; --card:#fff; --text:#1f2329; --sub:#8f959e;
    --border:#dee0e3; --primary:#3370ff; --green:#00b96b; --red:#d83931;
  }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
         background:var(--bg);color:var(--text);margin:0;padding:20px; }}
  .container {{ max-width:1800px;margin:0 auto; }}
  .header {{ display:flex;justify-content:space-between;align-items:center;
             margin-bottom:20px;padding-bottom:10px;border-bottom:2px solid var(--border); }}
  .btn {{ background:var(--primary);color:#fff;border:none;padding:8px 16px;
          border-radius:4px;cursor:pointer;font-size:14px; }}
  .btn:hover {{ background:#2458db; }}
  .control-panel {{ background:var(--card);padding:15px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:20px; }}
  .control-panel textarea {{ width:100%;height:200px;font-family:monospace;font-size:12px;
                              padding:10px;border:1px solid var(--border);border-radius:4px;
                              box-sizing:border-box;resize:vertical; }}
  .product-nav {{ background:var(--card);border-radius:8px;padding:10px 15px;
                  margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.05);
                  position:sticky;top:0;z-index:100; }}
  .product-nav a {{ display:inline-block;margin:3px 4px;padding:3px 9px;
                    background:#f0f2f5;border-radius:4px;text-decoration:none;
                    color:var(--text);font-size:12px; }}
  .product-nav a:hover {{ background:var(--primary);color:#fff; }}
  .product-section {{ background:var(--card);border-radius:12px;padding:25px;
                      margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08); }}
  .product-title {{ margin-top:0;color:var(--primary);border-bottom:1px dashed var(--border);padding-bottom:10px; }}
  .global-inputs {{ display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap; }}
  .global-img-box {{ width:120px;text-align:center;font-size:12px;color:var(--sub); }}
  .global-img-box img {{ width:100%;height:120px;object-fit:cover;border-radius:6px;
                         border:1px solid var(--border);margin-bottom:4px; }}
  .grid-container {{ display:flex;gap:16px;overflow-x:auto;padding-bottom:10px; }}
  .frame-column {{ flex:0 0 240px;background:#fafafa;border-radius:8px;
                   border:1px solid var(--border);padding:12px;display:flex;flex-direction:column; }}
  .prompt-text {{ font-size:13px;color:var(--red);line-height:1.5;margin-bottom:6px;
                  min-height:60px;white-space:pre-wrap; }}
  .original-prompt {{ font-size:11px;color:#646a73;line-height:1.4;margin-bottom:8px;white-space:pre-wrap; }}
  .image-box {{ margin-bottom:10px; }}
  .image-box h4 {{ margin:0 0 4px 0;font-size:12px;color:var(--sub); }}
  .image-box img {{ width:100%;border-radius:6px;border:1px solid var(--border);
                   cursor:pointer;transition:transform .2s; }}
  .image-box img:hover {{ transform:scale(1.02); }}
  .row-divider {{ border:none;border-top:1px dashed var(--border);margin:8px 0; }}
  .label-orig {{ font-size:11px;font-weight:bold;color:var(--red);margin:4px 0; }}
  .label-enh  {{ font-size:11px;font-weight:bold;color:var(--green);margin:4px 0; }}
  .my-title {{ color:var(--primary)!important; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h2>&#x1f50d; 对比：未加入约束 vs 数据增强+约束（{n} 个产品）</h2>
    <button class="btn" onclick="togglePanel()">&#x2699;&#xfe0f; 配置</button>
  </div>
  <div class="control-panel" id="controlPanel" style="display:none">
    <textarea id="mainConfig">{json_esc}</textarea>
    <button class="btn" style="margin-top:8px" onclick="renderPage()">&#x1f680; 渲染</button>
  </div>
  <div id="productNav" class="product-nav"></div>
  <div id="mainContent"></div>
</div>
<script>
function togglePanel(){{
  const p=document.getElementById('controlPanel');
  p.style.display=p.style.display==='none'?'block':'none';
}}
function esc(t){{
  if(!t)return'';
  return String(t).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function globalHtml(globals){{
  if(!globals||!globals.length)return'';
  let h='<div class="global-inputs">';
  globals.forEach(g=>{{
    h+=`<div class="global-img-box">
      <img src="${{g.path}}" loading="lazy" onerror="this.src='https://dummyimage.com/120x120/ffcccc/f00&text=丢失'">
      <div>${{esc(g.name)}}</div></div>`;
  }});
  return h+'</div>';
}}
function renderPage(){{
  const data=JSON.parse(document.getElementById('mainConfig').value);
  const nav=document.getElementById('productNav');
  nav.innerHTML='<b>产品导航：</b> ';
  data.forEach((p,i)=>nav.innerHTML+=`<a href="#p${{i}}">${{esc((p.a||p.b).title.replace(/\s*\[.*?\]\s*$/,''))}}</a>`);
  const container=document.getElementById('mainContent');
  container.innerHTML='';
  data.forEach((pair,idx)=>{{
    const a=pair.a||{{}}, b=pair.b||{{}};
    const title=(a.title||b.title||'').replace(/\s*\[.*?\]\s*$/,'');
    const framesA=a.frames||[];
    const framesB=b.frames||[];
    const count=Math.max(framesA.length,framesB.length);
    let framesHtml='<div class="grid-container">';
    for(let i=0;i<count;i++){{
      const fa=framesA[i]||{{}};
      const fb=framesB[i]||{{}};
      const orig=fa.original_prompt
        ?`<div class="original-prompt"><b>Original:</b> ${{esc(fa.original_prompt)}}</div>`:'';
      const enhPath=fb.my_path||'';
      framesHtml+=`<div class="frame-column">
        <div class="prompt-text"><b>Prompt:</b> ${{esc(fa.prompt||fb.prompt)}}</div>
        ${{orig}}
        <div class="image-box"><h4>nano banana</h4>
          <img src="${{fa.nano_path||''}}" loading="lazy" onerror="this.src='https://dummyimage.com/240x300/ffcccc/f00&text=丢失'">
        </div>
        <div class="label-orig">&#x1f534; 未加入约束</div>
        <div class="image-box">
          <img src="${{fa.my_path||''}}" loading="lazy" onerror="this.src='https://dummyimage.com/240x300/ffcccc/f00&text=丢失'">
        </div>
        <div class="label-enh">&#x1f7e2; 数据增强+约束</div>
        <div class="image-box">
          <img src="${{enhPath}}" loading="lazy" onerror="this.src='https://dummyimage.com/240x300/ffcccc/f00&text=丢失'">
        </div>
      </div>`;
    }}
    framesHtml+='</div>';
    container.innerHTML+=`
    <a id="p${{idx}}"></a>
    <div class="product-section">
      <h2 class="product-title">第 ${{idx+1}}/${{data.length}} 个：${{esc(title)}}</h2>
      ${{globalHtml(a.globals||b.globals)}}
      ${{framesHtml}}
    </div>`;
  }});
  document.getElementById('controlPanel').style.display='none';
}}
renderPage();
</script>
</body>
</html>'''


def main():
    configs_a = extract_configs(FILE_A)
    print(f'A (test_original): {len(configs_a)} 产品')

    # 加载所有 data_enhanced 个产品 HTML，按 base_name 建立索引
    def base_name(title):
        return re.sub(r'\s*\[.*?\]\s*$', '', title).strip()

    enhanced_files = sorted(glob.glob(DATA_ENHANCED_PATTERN))
    enhanced_files = [f for f in enhanced_files if 'all_21' not in os.path.basename(f)
                      and 'compare' not in os.path.basename(f)]
    index_b = {}
    for fp in enhanced_files:
        try:
            cfgs = extract_configs(fp)
            for c in cfgs:
                index_b[base_name(c['title'])] = c
        except Exception as e:
            print(f'WARNING: skip {os.path.basename(fp)}: {e}')
    print(f'B (data_enhanced): {len(index_b)} 产品 (来自 {len(enhanced_files)} 个文件)')

    pairs = []
    for ca in configs_a:
        key = base_name(ca['title'])
        cb = index_b.get(key)
        pairs.append((ca, cb))
        if cb is None:
            print(f'  (无 data_enhanced) {key}')

    matched = sum(1 for _, cb in pairs if cb is not None)
    print(f'共 {len(pairs)} 个产品，其中 {matched} 个有对应 data_enhanced 结果')

    html_text = build_html(pairs)
    with open(OUTPUT_LOCAL, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print(f'本地 HTML: {OUTPUT_LOCAL} ({os.path.getsize(OUTPUT_LOCAL)} bytes)')

    client = BlobStoreClient(BLOBSTORE_BUCKET)
    client.upload_binary_to_s3(OUTPUT_LOCAL, BLOB_KEY)
    url = f'{BLOBSTORE_CDN}/{BLOB_KEY}'
    print('=' * 60)
    print(f'HTML 公网访问地址: {url}')
    print('=' * 60)


if __name__ == '__main__':
    main()
