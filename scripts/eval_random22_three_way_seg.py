#!/usr/bin/env python3
"""Three-way Seg CLIP/DINO evaluation using literal English Hive L2 translations."""
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("random22_rl_original_apg", "random35_apg_rewrite", "random35_base_original_cfg")
METRICS = ("seg_clip_i", "seg_dino_i")


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args, out):
    with Path(args.categories).open(encoding="utf-8-sig", newline="") as f:
        categories = {r["product_id"]: r for r in csv.DictReader(f)}
    ids = [x.strip() for x in Path(args.products).read_text().splitlines() if x.strip()]
    if len(ids) != 22 or len(set(ids)) != 22:
        raise ValueError("Expected 22 unique products")
    records, metadata = [], []
    for pid in ids:
        category = categories[pid]
        english = category["category_l2_en"].strip()
        if not category["category_l2"].strip() or not english or not english.isascii():
            raise ValueError(f"Missing literal English L2 translation: {pid}")
        folder = ROOT / "batch_product_images_all_refs" / pid
        refs = sorted(p for p in (folder / "reference_imgs").iterdir()
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if not refs:
            raise ValueError(f"Missing reference: {pid}")
        query = english.lower().rstrip(".") + "."
        logs = {}
        for mode in MODES:
            path = ROOT / "qwen_inference_results_single" / f"{pid}_{mode}" / f"{pid}_prompt_log.json"
            logs[mode] = {int(r["frame_index"]): r for r in json.loads(path.read_text())}
            if set(logs[mode]) != set(range(6)):
                raise ValueError(f"Expected frames 0-5: {pid} {mode}")
        metadata.append({"product_id": pid, "item_id": category["item_id"],
                         "category_l2": category["category_l2"], "category_l2_en": english,
                         "query": query, "reference_image": str(refs[0])})
        for fi in range(6):
            for mode in MODES:
                path = ROOT / "qwen_inference_results_single" / f"{pid}_raw_{mode}" / f"frame_{fi}.jpg"
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(path)
                records.append({"id": f"{mode}__{pid}_f{fi:03d}",
                                "product_name": category["product_name"], "query": query,
                                "reference_image": str(refs[0]), "generated_image": str(path),
                                "edit_instruction": logs[mode][fi]["rewritten_prompt"]})
    (out / "input.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    dump(out / "category_queries.json", metadata)
    print(f"Prepared 22 products, {len(records)} images, 132 three-way comparisons", flush=True)
    return records


def summarize(out, records):
    expected = {r["id"] for r in records}
    latest = {}
    for line in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] in expected:
            latest[r["id"]] = r
    def valid(r):
        return r.get("status") == "ok" and all(isinstance(r.get(m), (int, float))
                                                and math.isfinite(r[m]) for m in METRICS)
    def mean(values):
        return sum(values) / len(values) if values else None
    summary = {"query_policy": "literal English L2 translation only", "methods": {}}
    for mode in MODES:
        rows = [r for k, r in latest.items() if k.startswith(mode + "__")]
        good = [r for r in rows if valid(r)]
        summary["methods"][mode] = {"expected": 132, "evaluated": len(rows), "valid": len(good),
                                   "status_counts": dict(Counter(r["status"] for r in rows)),
                                   **{m: mean([r[m] for r in good]) for m in METRICS}}
    comparisons, common = [], []
    for record in records:
        if not record["id"].startswith(MODES[0] + "__"):
            continue
        sample = record["id"].split("__", 1)[1]
        group = [latest.get(mode + "__" + sample, {}) for mode in MODES]
        row = {"sample": sample, "all_three_valid": all(valid(r) for r in group)}
        for mode, result in zip(MODES, group):
            row[mode + "_status"] = result.get("status", "not_evaluated")
            for m in METRICS:
                row[mode + "_" + m] = result.get(m) if valid(result) else None
        for label, a, b in (("rl_original_apg_minus_base_cfg", 0, 2),
                            ("apg_rewrite_minus_base_cfg", 1, 2),
                            ("apg_rewrite_minus_rl_original_apg", 1, 0)):
            for m in METRICS:
                row[label + "_" + m] = group[a][m] - group[b][m] if valid(group[a]) and valid(group[b]) else None
        comparisons.append(row)
        if row["all_three_valid"]:
            common.append(row)
    summary["common_valid"] = {"expected": 132, "valid": len(common),
                               "means_and_deltas": {k: mean([r[k] for r in common])
                                for k in comparisons[0] if any(k.endswith("_" + m) for m in METRICS)}}
    with (out / "three_way_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    dump(out / "three_way_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", default=str(ROOT / "random22_l2_products.txt"))
    parser.add_argument("--categories", default=str(ROOT / "dataset/item_categories/random35/category_translation_audit.csv"))
    parser.add_argument("--output", default=str(ROOT / "eval/random22_three_way_seg"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Max images this invocation; use 3 for one three-way sample")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args, extra = parser.parse_known_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = prepare(args, out)
    if args.prepare_only:
        return
    if not args.summary_only:
        cmd = [sys.executable, str(ROOT / "scripts/seg_clip_dino_eval.py"),
               "--input", str(out / "input.jsonl"), "--output", str(out), "--device", args.device]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd + extra, check=True)
    summarize(out, records)


if __name__ == "__main__":
    main()
