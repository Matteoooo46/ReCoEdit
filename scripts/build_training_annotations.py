"""Export editor/rewriter supervision from explicit, captioned JSONL records."""
import argparse
import hashlib
import json
from pathlib import Path


def build(source, image_root, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    root = image_root.resolve(strict=True)
    seen = set()
    records = []

    def image_path(value):
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError('Image paths must be nonempty relative strings')
        resolved = (root / value).resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            raise ValueError(f'Not an image file: {value}')
        return resolved.relative_to(root).as_posix()

    raw = source.read_bytes()
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for key in ('sample_id', 'product_id', 'source', 'original_prompt',
                    'editor_prompt', 'caption', 'rewriter_instruction'):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f'Missing or empty {key}')
        if r['sample_id'] in seen:
            raise ValueError(f'Duplicate sample_id: {r["sample_id"]}')
        seen.add(r['sample_id'])
        if r['caption'].lstrip().startswith('ERROR:'):
            raise ValueError(f'Failed caption: {r["sample_id"]}')
        if not isinstance(r.get('references'), list) or not r['references']:
            raise ValueError('references must be a nonempty list')
        refs = [image_path(p) for p in r['references']]
        target = image_path(r['target'])
        if target in refs or len(refs) != len(set(refs)):
            raise ValueError('Target/reference overlap or duplicate reference paths')
        records.append((r, refs, target))
    if not records:
        raise ValueError('Empty dataset')
    output.mkdir(parents=True, exist_ok=False)
    with (output/'editor.jsonl').open('w', encoding='utf-8') as editor, \
         (output/'rewriter.jsonl').open('w', encoding='utf-8') as rewriter, \
         (output/'lineage.jsonl').open('w', encoding='utf-8') as lineage:
        for r, refs, target in records:
            rows = [
                (editor, dict(image=target, edit_image=refs, prompt=r['editor_prompt'],
                              original_prompt=r['original_prompt'], caption=r['caption'])),
                (rewriter, dict(images=refs, instruction=r['rewriter_instruction'],
                                input=r['original_prompt'], output=r['caption'])),
                (lineage, {k: r[k] for k in ('sample_id', 'product_id', 'source')}),
            ]
            for stream, row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = dict(schema_version=1, records=len(records),
                    products=len({r['product_id'] for r, _, _ in records}),
                    input_sha256=hashlib.sha256(raw).hexdigest(),
                    image_paths='relative to supplied image-root; files not copied',
                    split='unsplit; product/content overlap audit required')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.image_root, args.output), indent=2))


if __name__ == '__main__':
    main()
