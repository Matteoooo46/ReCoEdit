"""Build an auditable three-way viewer and portable score artifacts; --publish uploads."""
import argparse
import csv
import hashlib
import html
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ['random35_base_original_cfg', 'random22_rl_original_apg', 'random35_apg_rewrite']
LABELS = ['Base + Original + CFG', 'RL + Original + APG', 'RL + Rewriter + APG']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'eval/random22_three_way_seg'
    output = ROOT / 'artifacts/random22_validation'
    output.mkdir(parents=True, exist_ok=True)
    records = {r['id']: r for r in map(json.loads, (source / 'results.jsonl').read_text().splitlines())}
    categories = json.loads((source / 'category_queries.json').read_text())
    summary = json.loads((source / 'three_way_summary.json').read_text())
    assert len(categories) == 22 and len(records) == 396
    assets, portable, sections = {}, [], []
    prefix = 'qwen_inference/comparison/random22_three_way_20260916'
    cdn = 'https://js-ad.a.yximgs.com/bs2/ad-nieuwland-material/'
    def asset(path):
        path = Path(path)
        data = path.read_bytes()
        key = prefix + '/assets/' + hashlib.sha256(data).hexdigest() + path.suffix
        assets[key] = (data, 'image/jpeg' if data[:2] == b'\xff\xd8' else 'image/png')
        return cdn + key
    e = html.escape
    for p in categories:
        pid = p['product_id']
        inputs = ROOT / 'batch_product_images_all_refs' / pid
        name = json.loads((inputs / 'extracted_item_info.json').read_text())['product_info'].splitlines()[0]
        refs = sorted(x for x in (inputs / 'reference_imgs').iterdir() if x.suffix.lower() in ['.png','.jpg','.jpeg','.webp'])
        character = inputs / 'final_lifestyle_image.png'
        if character.exists():
            refs.append(character)
        section = f'<section id="{pid}"><h2>{e(name)}</h2><p>{pid} · {e(p["category_l2"])} → {e(p["query"])}</p><div class="refs">'
        for ref in refs:
            label = 'Character input' if ref == character else ('Scoring reference' if str(ref) == p['reference_image'] else 'Additional input')
            section += f'<figure><img loading="lazy" src="{asset(ref)}"><figcaption>{label}</figcaption></figure>'
        section += '</div>'
        for fi in range(6):
            section += f'<article><h3>Frame {fi+1}</h3><div class="grid">'
            for mode,label in zip(MODES,LABELS):
                r = records[f'{mode}__{pid}_f{fi:03d}']
                url = asset(r['generated_image'])
                scores = ' · '.join(f'{m}: {r[m]:.4f}' for m in ['seg_clip_i','seg_dino_i']) if r['status']=='ok' else r['status']
                section += f'<div><h4>{label}</h4><a href="{url}" target="_blank" rel="noopener"><img loading="lazy" src="{url}"></a><p>{scores}</p><details><summary>Prompt</summary><p>{e(r["edit_instruction"])}</p></details></div>'
                portable.append({k:r.get(k) for k in ['id','status','query','seg_clip_i','seg_dino_i','ref_detection_score','gen_detection_score']} | {'product_id':pid,'frame':fi,'method':mode,'image_url':url})
            section += '</div></article>'
        sections.append(section+'</section>')
    common = summary['common_valid']['means_and_deltas']
    table = '<table><tr><th>Method</th><th>Valid / 132</th><th>Seg CLIP-I (common 117)</th><th>Seg DINO-I (common 117)</th></tr>'
    for mode,label in zip(MODES,LABELS):
        table += f'<tr><td>{label}</td><td>{summary["methods"][mode]["valid"]}</td><td>{common[mode+"_seg_clip_i"]:.6f}</td><td>{common[mode+"_seg_dino_i"]:.6f}</td></tr>'
    table += '</table>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ReCoEdit · 22商品三组推理与评分</title><style>body{font-family:system-ui;margin:24px;background:#f4f6f8;color:#243047}main{max-width:1600px;margin:auto}section,header{background:white;padding:20px;border-radius:12px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.grid img{width:100%;max-height:650px;object-fit:contain;background:#eee}.refs{display:flex;overflow:auto}.refs img{height:160px;max-width:200px;object-fit:contain}figure{margin:8px}td,th{padding:10px;border-bottom:1px solid #ddd}nav{display:flex;flex-wrap:wrap;gap:10px}article{border-top:1px solid #ddd;padding:14px 0}details p{white-space:pre-wrap}a{color:#2864dc}@media(max-width:700px){.grid{grid-template-columns:1fr}table{font-size:12px}}</style><main><header><h1>ReCoEdit · 22商品三组推理与评分</h1><p>22 products · 132 frames · 396 images. Literal English L2 queries; GroundingDINO → SAM2 → CLIP/DINOv2. Failed detections are excluded, not scored as zero.</p>'+table+'<p>The RL+original comparison changes both checkpoint and guidance (CFG→APG); it does not isolate the training effect. Rewriter mode also adds anti-text prompts. Scores measure localized similarity, not instruction following.</p><nav>'+''.join(f'<a href="#{p["product_id"]}">#{i+1} {e(p["category_l2"])}</a>' for i,p in enumerate(categories))+'</nav></header>'+''.join(sections)+'</main></html>'
    key = prefix + '/' + hashlib.sha256(page.encode()).hexdigest()[:16] + '/index.html'
    (output/'index.html').write_text(page)
    (output/'scores.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in portable))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'categories.json').write_text(json.dumps([{k:v for k,v in p.items() if k!='reference_image'} for p in categories],ensure_ascii=False,indent=2)+'\n')
    (output/'three_way_metrics.csv').write_bytes((source/'three_way_metrics.csv').read_bytes())
    (output/'viewer_url.txt').write_text(cdn+key+'\n')
    if args.publish:
        sys.path.insert(0,str(ROOT))
        from blobstore import BlobStore
        store=BlobStore(bucket='ad-nieuwland-material')
        for i,(asset_key,(body,mime)) in enumerate(assets.items(),1):
            store.client.put_object(Bucket=store.bucket,Key=asset_key,Body=body,ContentType=mime)
            if i%50==0: print(f'Uploaded {i}/{len(assets)}',flush=True)
        store.client.put_object(Bucket=store.bucket,Key=key,Body=page.encode(),ContentType='text/html; charset=utf-8')
        assert store.client.head_object(Bucket=store.bucket,Key=key)['ContentLength']==len(page.encode())
    print(f'22 products; 396 scores; {len(assets)} assets\n{cdn+key}',flush=True)


if __name__ == '__main__':
    main()
