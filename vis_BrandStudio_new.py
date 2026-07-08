import os
import json
import traceback
import glob
from blobstore import BlobStoreClient  # 你的上传依赖

# ================= 配置区域 =================
WORKSPACE_DIR = "/data/lijiahui/code/test_gemini/1218_manju/0404_batch_replica"
OUTPUT_HTML = "0404_batch_replicax.html"
BLOB_PREFIX = "0404_batch_replicax"

def upload_files(file_paths, blob_prefix):
    """
    批量上传文件
    """
    blobstore = BlobStoreClient("ad-nieuwland-material")
    uploaded_urls = []
    path_to_url = {}

    valid_paths = [p for p in file_paths if p and os.path.exists(p)]
    unique_files = list(set(valid_paths))

    print(f"☁️  正在上传 {len(unique_files)} 个文件...")

    for file_path in unique_files:
        try:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            # 避免父目录名太普通导致冲突
            if parent_dir in ["reference_imgs", "processed_videos", "videos", "product_images"]:
                # 再往上一级找，拿到 seg_xxx 或 item_ID
                parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

            bs_key = f"{blob_prefix}/{parent_dir}/{file_name}"
            # 这里请根据你的实际 bucket 域名调整
            bs_url = f"https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material/{bs_key}"

            blobstore.upload_binary_to_s3(file_path, bs_key)
            path_to_url[file_path] = bs_url
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
            path_to_url[file_path] = ""

    for file_path in file_paths:
        if not file_path:
            uploaded_urls.append("")
        else:
            uploaded_urls.append(path_to_url.get(file_path, ""))

    return uploaded_urls

def find_best_video(base_path):
    """
    辅助函数：在指定目录下寻找最佳视频
    逻辑：
    1. 寻找 final*.mp4
    2. 必须排除 visual_only
    3. 优先选择 final_output_ 开头的文件
    """
    if not os.path.exists(base_path):
        return None
        
    # 获取所有 final 开头的 mp4
    candidates = glob.glob(os.path.join(base_path, "final*.mp4"))
    
    # 【关键修改】过滤掉 visual_only
    real_candidates = [p for p in candidates if "visual_only" not in os.path.basename(p)]
    
    if not real_candidates:
        return None
    
    # 排序优化：让 final_output_xxx.mp4 排在 final_xxx.mp4 前面 (如果有多种命名混杂的话)
    # 同时确保最短的文件名通常是我们要的 (防止选中 final_output_xxx_debug.mp4 等)
    real_candidates.sort(key=lambda x: (len(x), x))
    
    return real_candidates[0]

def get_media_pairs(workspace_dir):
    all_pairs = []
    missing_items = [] 
    
    if not os.path.exists(workspace_dir):
        print(f"❌ 错误: 根目录不存在 -> {workspace_dir}")
        return [], []

    # 扫描 workspace_ 开头的文件夹
    items = [d for d in os.listdir(workspace_dir) if d.startswith("workspace_")]
    items.sort()
    
    print(f"📂 扫描中: 找到 {len(items)} 个 workspace 文件夹")

    for item_name in items:
        item_path = os.path.join(workspace_dir, item_name)
        
        if not os.path.isdir(item_path):
            continue
        
        # 提取 ID (用于展示)
        item_id_display = item_name.replace("workspace_item_", "").replace("workspace_", "")
        
        # 寻找子文件夹 (proj_item...seg_0)
        all_subs = os.listdir(item_path)
        valid_seg_dirs = []
        for sub in all_subs:
            # 只要包含 seg_ 且是目录
            if "seg_" in sub and os.path.isdir(os.path.join(item_path, sub)):
                valid_seg_dirs.append(sub)
        valid_seg_dirs.sort() # 优先 seg_0

        # video_path = os.path.join(item_path, "final_full_manju_video.mp4")

        # ==========================================
        # 1. 寻找最终视频 (排除 visual_only)
        # ==========================================
        video_path = None
        video_path = os.path.join(item_path, "final_full_manju_video.mp4")
        
        # A. 优先在 seg 子目录下找
        if not video_path and valid_seg_dirs:
            for sub in valid_seg_dirs:
                video_path = find_best_video(os.path.join(item_path, sub))
                if video_path:
                    break
        
        # B. 如果子目录没找到，去父目录找
        if not video_path:
            video_path = find_best_video(item_path)
        
        if not video_path:
            missing_items.append(f"{item_name} -> No clean final*.mp4 found")
            continue 

        # ==========================================
        # 2. 寻找商品参考图片
        # ==========================================
        product_img_path = None
        search_dirs = []
        
        # 定义搜索优先级
        if valid_seg_dirs:
            # 1. seg/reference_imgs
            search_dirs.append(os.path.join(item_path, valid_seg_dirs[0], "reference_imgs"))
            # 2. seg/product_images (截图里有)
            search_dirs.append(os.path.join(item_path, valid_seg_dirs[0], "product_images"))
        # 3. 父目录/reference_imgs
        search_dirs.append(os.path.join(item_path, "reference_imgs"))
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                found_imgs = []
                # 支持常见图片格式
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG']:
                    found_imgs.extend(glob.glob(os.path.join(search_dir, ext)))
                if found_imgs:
                    found_imgs.sort() # 默认取第一张
                    product_img_path = found_imgs[0]
                    break
        
        # ==========================================
        # 3. 寻找参考视频 (Reference)
        # ==========================================
        reference_video_path = None

        
        
        # A. 尝试从 JSON 读取
        if not reference_video_path and valid_seg_dirs:
            json_path = os.path.join(item_path, valid_seg_dirs[0], "selected_reference_info.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        rp = data.get("selected_reference_video", "")
                        if rp and os.path.exists(rp):
                            reference_video_path = rp
                except: pass
        
        # B. 暴力搜索 manju_video 或 reference 开头的视频
        if not reference_video_path and valid_seg_dirs:
            sub_p = os.path.join(item_path, valid_seg_dirs[0])
            for pat in ["manju_video*.mp4", "reference*.mp4"]:
                fv = glob.glob(os.path.join(sub_p, pat))
                if fv:
                    reference_video_path = fv[0]
                    break

        # ==========================================
        # 4. 获取商品名称 & 剧本
        # ==========================================
        product_name = item_id_display.split('_')[0]
        scene_plot = []
        
        if valid_seg_dirs:
            sub_dir_path = os.path.join(item_path, valid_seg_dirs[0])
            
            # 读取商品名
            for inf in ["item_info.json", "extracted_item_info.json"]:
                p = os.path.join(sub_dir_path, inf)
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            title = d.get("itemTitle") or d.get("title") or d.get("item_name")
                            if title:
                                product_name = title[:50] # 截断
                                break
                    except: pass
            
            # 读取剧本
            sp_path = os.path.join(sub_dir_path, "scene_plot_optimized.json")
            if os.path.exists(sp_path):
                try:
                    with open(sp_path, 'r', encoding='utf-8') as f:
                        scene_plot = json.load(f)
                except: pass

        all_pairs.append({
            "id": item_name,
            "display_id": item_id_display,
            "product_name": product_name,
            "product_img": product_img_path,
            "reference_video": reference_video_path,
            "scene_plot": scene_plot,
            "gen_video": video_path
        })

    return all_pairs, missing_items

def generate_html(pairs, blob_prefix):
    html_header = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>生成结果 - {blob_prefix}</title>
<style>
    body {{ background: #121212; color: #e0e0e0; font-family: sans-serif; padding: 20px; }}
    .container {{ max-width: 1600px; margin: 0 auto; }}
    .card {{ 
        background: #1e1e1e; border: 1px solid #333; border-radius: 8px; 
        padding: 20px; margin-bottom: 25px; display: flex; flex-direction: column; gap: 15px;
    }}
    .header {{ 
        display: flex; justify-content: space-between; border-bottom: 1px solid #333; 
        padding-bottom: 10px; font-size: 14px; color: #aaa;
    }}
    .header strong {{ color: #fff; font-size: 16px; }}
    
    .media-row {{ display: grid; grid-template-columns: 1fr 1fr 1.5fr 1.5fr; gap: 15px; }}
    .col {{ text-align: center; }}
    .col-label {{ font-size: 12px; color: #888; margin-top: 5px; text-transform: uppercase; }}
    
    video, img {{ width: 100%; max-height: 320px; background: #000; border-radius: 4px; border: 1px solid #333; }}
    img {{ object-fit: contain; cursor: pointer; }}
    
    .final-video {{ border: 2px solid #4caf50; box-shadow: 0 0 10px rgba(76,175,80,0.3); }}
    
    .script {{ 
        background: #121212; padding: 10px; border-radius: 4px; text-align: left; 
        height: 320px; overflow-y: auto; font-size: 13px; line-height: 1.4;
    }}
    .shot {{ margin-bottom: 8px; padding-left: 8px; border-left: 2px solid #2196f3; }}
    .missing {{ display: flex; align-items: center; justify-content: center; height: 100px; border: 1px dashed #444; color: #666; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 视频生成报告 ({len(pairs)})</h1>
"""
    html_body = []
    
    BATCH_SIZE = 10
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i+BATCH_SIZE]
        print(f"Generating HTML batch {i//BATCH_SIZE + 1}...")
        
        # 上传逻辑
        files = []
        for p in batch:
            files.extend([p['product_img'], p['reference_video'], p['gen_video']])
        urls = upload_files(files, blob_prefix)
        
        idx = 0
        for p in batch:
            u_img, u_ref, u_gen = urls[idx], urls[idx+1], urls[idx+2]
            idx += 3
            
            # 构建剧本HTML
            script_html = ""
            if p['scene_plot']:
                for si, s in enumerate(p['scene_plot']):
                    desc = s.get('scene_description') or s.get('description', '')
                    script_html += f'<div class="shot"><b>#{si+1}</b> {desc}</div>'
            else:
                script_html = '<div class="missing">无剧本</div>'
            
            # 构建媒体HTML
            img_tag = f'<img src="{u_img}" onclick="window.open(this.src)">' if u_img else '<div class="missing">无图片</div>'
            ref_tag = f'<video src="{u_ref}" controls muted playsinline></video>' if u_ref else '<div class="missing">无参考视频</div>'
            
            # 最终视频展示
            video_info_html = ""
            if p['gen_video']:
                # 显示实际使用的文件名，方便核对
                fname = os.path.basename(p['gen_video'])
                video_info_html = f'<div style="font-size:10px;color:#4caf50;margin-top:2px;">{fname}</div>'
            
            gen_tag = f"""
            <video class="final-video" src="{u_gen}" controls autoplay muted loop playsinline></video>
            {video_info_html}
            """

            html_body.append(f"""
            <div class="card">
                <div class="header">
                    <div>ID: <span style="color:#64b5f6">{p['display_id']}</span></div>
                    <strong>{p['product_name']}</strong>
                </div>
                <div class="media-row">
                    <div class="col">{img_tag}<div class="col-label">商品图</div></div>
                    <div class="col">{ref_tag}<div class="col-label">参考视频</div></div>
                    <div class="col"><div class="script">{script_html}</div><div class="col-label">分镜脚本</div></div>
                    <div class="col">{gen_tag}<div class="col-label" style="color:#4caf50">✅ 最终成片</div></div>
                </div>
            </div>
            """)

    return html_header + "\n".join(html_body) + "</div></body></html>"

def main():
    print("🚀 开始处理...")
    pairs, missing = get_media_pairs(WORKSPACE_DIR)
    
    if missing:
        print(f"\n⚠️  {len(missing)} 个任务有问题 (Sample):")
        for m in missing[:5]: print(f" - {m}")
        
    if not pairs:
        print("❌ 未找到有效数据")
        return

    print(f"\n✅ 准备生成HTML: 共 {len(pairs)} 条数据")
    html = generate_html(pairs, BLOB_PREFIX)
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🎉 成功生成: {os.path.abspath(OUTPUT_HTML)}")

if __name__ == "__main__":
    main()