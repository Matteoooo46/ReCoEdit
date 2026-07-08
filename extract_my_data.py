import os
import json

# ==========================================
# 1. 基础路径配置
# ==========================================
# 你存放所有造好数据的总目录
ROOT_DIR = "/data/phd/kousiqi/zhitao/batch_product_images_all_refs"

# 最终提取出来的训练账本保存路径
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_novita_generated.json"

print("🔍 开始扫描造数据工厂，准备提取高质量(满6帧)的训练集...")

all_training_data = []
project_count = 0
skipped_count = 0 # 记录因为不满 6 张被淘汰的商品数量

# ==========================================
# 2. 遍历所有商品文件夹
# ==========================================
for folder_name in os.listdir(ROOT_DIR):
    workspace = os.path.join(ROOT_DIR, folder_name)
    if not os.path.isdir(workspace):
        continue

    # 定义我们要找的关键文件和文件夹路径
    script_path = os.path.join(workspace, "final_storyboard_script.json")
    char_img_path = os.path.join(workspace, "final_lifestyle_image.png")
    ref_img_dir = os.path.join(workspace, "reference_imgs")
    storyboard_dir = os.path.join(workspace, "storyboard_images_all_refs")

    # 检查核心文件是否齐全
    if not all([os.path.exists(script_path), os.path.exists(char_img_path), 
                os.path.exists(ref_img_dir), os.path.exists(storyboard_dir)]):
        continue

    # 提取商品原图
    ref_imgs = sorted([img for img in os.listdir(ref_img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not ref_imgs:
        continue
    product_ref_path = os.path.join(ref_img_dir, ref_imgs[0])

    # 解析剧本获取 Prompt
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取剧本失败 {folder_name}: {e}")
        continue

    # 提取分镜列表
    scenes_list = []
    if isinstance(script_data, dict):
        if "script" in script_data and isinstance(script_data["script"], list):
            scenes_list = script_data["script"]
        else:
            for k, v in script_data.items():
                if isinstance(v, list) and len(v) > len(scenes_list):
                    scenes_list = v
    elif isinstance(script_data, list):
        scenes_list = script_data
    
    if not scenes_list:
        scenes_list = [script_data]

    # ==========================================
    # 3. ★★★ 核心逻辑：只保留参考图，剔除历史镜头 ★★★
    # ==========================================
    temp_workspace_data = [] # 临时购物车：只存放当前这个商品的有效帧

    for idx, scene in enumerate(scenes_list):
        if not isinstance(scene, dict): continue

        scene_num = idx + 1
        scene_prompt = scene.get('description', '')
        if not scene_prompt: continue

        # 找到目标分镜图
        target_img_path = os.path.join(storyboard_dir, f"scene_{scene_num}.png")
        if not os.path.exists(target_img_path):
            continue

        # 💥 核心修改点：丢弃 history_frames，固定只塞入这 2 张图！
        edit_images = [
            char_img_path,     # 条件图 1：人物定妆照
            product_ref_path   # 条件图 2：商品原图
        ]

        # 组装一条数据
        json_line = {
            "image": target_img_path,
            "edit_image": edit_images,
            "prompt": scene_prompt
        }
        
        # 先装进当前商品的临时购物车
        temp_workspace_data.append(json_line)

    # ==========================================
    # 💥 拦截网：判断这个购物车里够不够 6 张图！
    # ==========================================
    if len(temp_workspace_data) >= 5:
        # 如果满 6 张（或以上），将购物车里的数据一口气倒进总账本
        all_training_data.extend(temp_workspace_data)
        project_count += 1
    else:
        # 如果不到 6 张，购物车直接被丢弃
        skipped_count += 1
        print(f"⚠️ 拦截残次品: [{folder_name}] 仅生成了 {len(temp_workspace_data)} 张图，已舍弃。")

# ==========================================
# 4. 导出最终的标准 JSON 账本
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_training_data, f, ensure_ascii=False, indent=4)

print("\n==================================================")
print(f"🎉 严格的提取大业圆满完成！(已彻底移除历史镜头作为训练条件)")
print(f"📦 共成功收录了 {project_count} 个完美的商品项目。")
print(f"🗑️  共拦截并丢弃了 {skipped_count} 个分镜不足的残次品项目。")
print(f"✅ 共为你提取出了 {len(all_training_data)} 条高质量的训练数据！")
print(f"💾 训练集 JSON 已安全保存至: {OUTPUT_JSON_PATH}")
print("==================================================")