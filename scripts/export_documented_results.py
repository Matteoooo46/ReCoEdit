"""Export small, path-free documentation snapshots from existing scores (no inference)."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'docs/results'
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        'random22_seg': 'artifacts/random22_validation/summary.json',
        'random22_iqa': 'eval/random22_image_quality/summary.json',
        'selected20_internvl': 'eval/selected20_five_way_internvl_v2/summary.json',
        'selected20_historical': 'eval/exploratory20_old15_train5/current_four_way_summary.csv',
    }
    manifest = {}
    tables = {}
    for name, relative in sources.items():
        path = ROOT / relative
        raw = path.read_bytes()
        manifest[name] = {'source': relative, 'sha256': hashlib.sha256(raw).hexdigest()}
        if name == 'random22_seg':
            data = json.loads(raw)
            tables[name] = {k: data[k] for k in ('methods', 'common_valid')}
        elif name == 'random22_iqa':
            data = json.loads(raw)
            tables[name] = {k: v['methods'] for k, v in data['metrics'].items()}
        elif name == 'selected20_internvl':
            data = json.loads(raw)
            # Whitelist aggregate numeric results; never export image paths or prompts.
            tables[name] = data['cohorts']
        else:
            with path.open(encoding='utf-8-sig', newline='') as f:
                tables[name] = list(csv.DictReader(f))
    payload = {'note': 'Archived configuration comparisons; see EVAL.md for cohort and validity limitations.',
               'sources': manifest, 'results': tables}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    if '/data/' in text or 'http://' in text:
        raise ValueError('Unexpected local path or endpoint in public snapshot')
    (out / 'archived_metrics.json').write_text(text, encoding='utf-8')
    protocol_path = ROOT/'eval/selected20_five_way_internvl_v2/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['model'] = 'InternVL3.5-8B-Instruct'
    protocol['source_protocol_sha256'] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    public_protocol = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
    if '/data/' in public_protocol:
        raise ValueError('Unexpected local path in public protocol')
    (out/'internvl_protocol.json').write_text(public_protocol, encoding='utf-8')
    print(out / 'archived_metrics.json')


if __name__ == '__main__':
    main()
