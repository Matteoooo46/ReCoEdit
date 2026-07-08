"""
为 fuke_weigai 缺少产品名称的目录批量调用 KwaishopProductItemCenterClient 提取产品信息。

目录名不是 item_id，真实 item_id 来源（按优先级）：
  1. kafka_msg_final.json 的 item_id 字段
  2. *_final_output.mp4 文件名前缀
  3. 目录名本身（兜底）

调用成功后在目录下生成 extracted_item_info.json，格式与 kousiqi 的一致：
  { "product_info": "产品标题", "item_id": "...", ... }

用法：
  python fetch_fuke_weigai_product_names.py [--base-dir ...] [--workers 8] [--dry-run]
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.protobuf.json_format import MessageToDict
from video_graph.common.client.kwaishop_product_item_center_client import (
    KwaishopProductItemCenterClient,
)


def extract_item_id(dir_path: str):
    """从目录中提取真实的产品 item_id，返回 int 或 None。"""
    # 优先级1: kafka_msg_final.json
    kafka_path = os.path.join(dir_path, "kafka_msg_final.json")
    if os.path.exists(kafka_path):
        try:
            with open(kafka_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            item_id = d.get("item_id")
            if item_id and str(item_id).isdigit():
                return int(item_id)
        except Exception:
            pass

    # 优先级2: *_final_output.mp4 文件名前缀
    for f in os.listdir(dir_path):
        m = re.match(r"(\d+)_final_output\.mp4", f)
        if m:
            return int(m.group(1))

    # 优先级3: 目录名本身
    dir_name = os.path.basename(dir_path)
    if dir_name.isdigit():
        return int(dir_name)

    return None


def fetch_one(client: KwaishopProductItemCenterClient, dir_path: str, item_id: int):
    """对单个 item_id 调用 API，成功则写 extracted_item_info.json。"""
    out_path = os.path.join(dir_path, "extracted_item_info.json")
    if os.path.exists(out_path):
        return "skip"

    try:
        resp = client._sync_run(item_id=item_id)
        resp_dict = MessageToDict(resp)
        item_id_str = str(item_id)
        item_info = resp_dict.get("itemInfo", {}).get(item_id_str)
        if not item_info:
            return f"no_info(item_id={item_id})"

        result = {
            "product_info": item_info.get("itemTitle", ""),
            "item_id": item_id_str,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return "ok"

    except Exception as e:
        return f"error({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="/data/zgq/nieuwland/src/t2i2v/t2i2v_project/t2i2v_pipeline/scene_plot_fuke_weigai_produce",
    )
    ap.add_argument("--workers", type=int, default=8, help="并发线程数")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不调用 API")
    ap.add_argument(
        "--only-missing", action="store_true", help="跳过已有 extracted_item_info.json 的目录"
    )
    args = ap.parse_args()

    # 1. 扫描所有产品目录，目录名即 item_id
    print(f"扫描 {args.base_dir} ...")
    tasks = []  # (dir_path, item_id)
    already_has = 0

    for date_dir in sorted(os.listdir(args.base_dir)):
        date_path = os.path.join(args.base_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        for item_dir in os.listdir(date_path):
            item_path = os.path.join(date_path, item_dir)
            if not os.path.isdir(item_path):
                continue

            if args.only_missing and os.path.exists(
                os.path.join(item_path, "extracted_item_info.json")
            ):
                already_has += 1
                continue

            item_id = extract_item_id(item_path)
            if item_id is None:
                continue
            tasks.append((item_path, item_id))

    print(f"需处理: {len(tasks)} 个目录")
    if already_has:
        print(f"已有 extracted_item_info.json 跳过: {already_has}")

    if args.dry_run:
        for dir_path, item_id in tasks[:10]:
            print(f"  {dir_path} -> item_id={item_id}")
        return

    if not tasks:
        print("没有需要处理的目录")
        return

    # 2. 并发调用 API
    client = KwaishopProductItemCenterClient()
    ok = 0
    skip = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, client, dir_path, item_id): dir_path
            for dir_path, item_id in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            dir_path = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = f"exception({e})"

            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 20:
                    print(f"  [FAIL] {dir_path}: {result}")

            if i % 500 == 0:
                print(f"进度: {i}/{len(tasks)} (ok={ok} skip={skip} fail={fail})")

    print(f"\n完成: ok={ok} skip={skip} fail={fail} / 总计 {len(tasks)}")


if __name__ == "__main__":
    main()
