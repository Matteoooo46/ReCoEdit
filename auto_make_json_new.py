import os
import json

# ==========================================
# 1. 基础路径配置
# ==========================================
# 修改为你新的数据集根目录
ROOT_DIR = "/data/phd/kousiqi/zhitao/new_training_data/scene_plot_transfer_online_produce"

# 输出的训练元数据 JSON 文件路径 (建议换个名字以免覆盖之前的)
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/new_training_data/metadata_transfer_online.json"

project_count = 0
all_data = []

print(f"开始扫描数据集：{ROOT_DIR}")

# ==========================================
# 2. 动态寻找所有有效的产品文件夹
# ==========================================
product_paths = []
# 自动遍历 20260210 等日期文件夹下的所有子目录
for root_path, dirs, files in os.walk(ROOT_DIR):
    # 只要一个文件夹底下同时包含这两个核心文件夹，它就是一个我们需要处理的产品包！
    if "reference_imgs" in dirs and "augmented_generated_images" in dirs:
        product_paths.append(root_path)

print(f"在日期目录下共找到了 {len(product_paths)} 个有效的产品文件夹。")

# ==========================================
# 3. 遍历提取数据
# ==========================================
for product_path in product_paths:
    product_name = os.path.basename(product_path) 
    
    found_reference_dir = os.path.join(product_path, "reference_imgs")
    found_target_dir = os.path.join(product_path, "augmented_generated_images") # 修改了名字
    
    # 为了兼容旧数据，如果偶尔有 character_refs 也能自动带上，没有就不带
    found_character_dir = os.path.join(product_path, "character_refs") if os.path.exists(os.path.join(product_path, "character_refs")) else None

    # ---------------------------------------------------
    # 第一步：获取【参考图】(组装 edit_image 列表)
    # ---------------------------------------------------
    edit_image_list = []
    
    # 获取产品参考图 (支持读取里面所有的参考图)
    ref_imgs = sorted([img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not ref_imgs:
        print(f"[跳过] {product_name}: 找不到产品参考图片")
        continue
    
    for ref_img in ref_imgs:
        edit_image_list.append(os.path.join(found_reference_dir, ref_img))
        
    # 如果该产品碰巧有角色图，也加进去
    if found_character_dir:
        char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
        if char_imgs:
            edit_image_list.append(os.path.join(found_character_dir, char_imgs[0]))

    # ---------------------------------------------------
    # 第二步：遍历【目标帧】并读取对应的 json Prompt
    # ---------------------------------------------------
    target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]
    
    # 严格按 0, 1, 2, 3... 排序
    def get_frame_num(filename):
        try: return int(filename.split('_')[0])
        except: return 999999 
            
    target_files.sort(key=get_frame_num) 

    has_valid_frame = False 
    
    for target_file in target_files:
        target_img_path = os.path.join(found_target_dir, target_file)
        
        # 寻找同名的 json 文件 (如 0_0.json)
        json_file = target_file.replace(".png", ".json")
        json_path = os.path.join(found_target_dir, json_file)
        if not os.path.exists(json_path): continue 
            
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                prompt_data = json.load(jf)
                # 处理 JSON 是个列表的情况
                real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
                    
                prompt_text = real_data.get("prompt", "")
                if not prompt_text:
                    prompt_text = real_data.get("product_info", "")
        except Exception as e:
            print(f"[读取JSON失败] 文件: {json_path}, 报错: {e}")
            continue

        # ---------------------------------------------------
        # 第三步：组装这一帧的训练数据
        # ---------------------------------------------------
        json_line = {
            "image": target_img_path,                             
            "edit_image": edit_image_list, 
            "prompt": prompt_text                                 
        }
        
        all_data.append(json_line) 
        has_valid_frame = True
        
    if has_valid_frame:
        project_count += 1

# ==========================================
# 4. 一次性导出为标准 JSON 数组
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("\n==========================================")
print(f"🎉 新数据集扫描并串联完毕！")
print(f"📦 共成功处理了 {project_count} 个产品目录")
print(f"✅ 共生成了 {len(all_data)} 条训练数据")
print(f"💾 标准格式数据集已保存在: {OUTPUT_JSON_PATH}")
print("==========================================")