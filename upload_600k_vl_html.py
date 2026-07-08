"""
把 600k_vl.zip 中的 all_products_viewer.html 转成公网可访问链接。

流程：
1. 从 zip 中读出 all_products_viewer.html 嵌入的 mainConfig JSON
2. 逐张图片直接从 zip 内读出 bytes，上传到 BlobStore
3. 将 config 里所有相对路径替换为公网 URL
4. 用新 config 重写 textarea，整个 HTML 上传到 BlobStore，打印访问链接
"""

import zipfile
import re
import json
import html as html_lib
import io
import os
import tempfile
from blobstore import BlobStoreClient

ZIP_PATH = "/data/phd/kousiqi/zhitao/600k_vl.zip"
ZIP_ROOT = "600k_vl/"
HTML_INSIDE_ZIP = "600k_vl/all_products_viewer.html"

BLOB_PREFIX = "qwen_inference/600k_vl"
BLOBSTORE_BUCKET = "ad-nieuwland-material"
BLOBSTORE_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"


def main():
    blobstore = BlobStoreClient(BLOBSTORE_BUCKET)

    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(HTML_INSIDE_ZIP) as f:
            html = f.read().decode("utf-8")

        m = re.search(r'(<textarea[^>]*id="mainConfig"[^>]*>)(.*?)(</textarea>)', html, re.DOTALL)
        if not m:
            raise RuntimeError("没在 HTML 中找到 mainConfig textarea")
        config = json.loads(html_lib.unescape(m.group(2)))
        print(f"✅ 读取 config，共 {len(config)} 个产品")

        # 收集所有相对路径
        all_paths = set()
        for p in config:
            for g in p["globals"]:
                all_paths.add(g["path"])
            for fr in p["frames"]:
                all_paths.add(fr["nano_path"])
                all_paths.add(fr["my_path"])
        all_paths = sorted(all_paths)
        print(f"☁️  准备上传 {len(all_paths)} 张图片...")

        path_to_url = {}
        for i, rel in enumerate(all_paths, 1):
            zip_member = f"{ZIP_ROOT}{rel}"
            try:
                with z.open(zip_member) as zf:
                    data = zf.read()
                # blobstore SDK 接收本地文件路径，写一个临时文件再上传
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(rel)[1], delete=False) as tf:
                    tf.write(data)
                    tmp_path = tf.name
                try:
                    bs_key = f"{BLOB_PREFIX}/{rel}"
                    blobstore.upload_binary_to_s3(tmp_path, bs_key)
                    path_to_url[rel] = f"{BLOBSTORE_CDN}/{bs_key}"
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                print(f"  [上传失败] {rel}: {e}")
                path_to_url[rel] = ""
            if i % 50 == 0:
                print(f"  进度 {i}/{len(all_paths)}")

    # 替换 config 里的路径为公网 URL
    for p in config:
        for g in p["globals"]:
            g["path"] = path_to_url.get(g["path"], g["path"])
        for fr in p["frames"]:
            fr["nano_path"] = path_to_url.get(fr["nano_path"], fr["nano_path"])
            fr["my_path"] = path_to_url.get(fr["my_path"], fr["my_path"])

    # 重写 textarea
    new_config_json = json.dumps(config, ensure_ascii=False, indent=4)
    new_textarea = m.group(1) + html_lib.escape(new_config_json) + m.group(3)
    new_html = html[:m.start()] + new_textarea + html[m.end():]

    # 保存 HTML 到本地后上传
    out_local = "/data/phd/kousiqi/zhitao/600k_vl_all_products_viewer_remote.html"
    with open(out_local, "w", encoding="utf-8") as f:
        f.write(new_html)
    print(f"✅ 本地 HTML 已保存: {out_local}")

    html_bs_key = f"{BLOB_PREFIX}/viewer/all_products_viewer.html"
    blobstore.upload_binary_to_s3(out_local, html_bs_key)
    html_url = f"{BLOBSTORE_CDN}/{html_bs_key}"
    print("==================================================")
    print(f"🌐 HTML 公网访问地址: {html_url}")
    print("==================================================")


if __name__ == "__main__":
    main()
