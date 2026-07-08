import os
import json

# ==========================================
# 1. 基础路径配置
# ==========================================
# 修改为你新的数据集根目录
ROOT_DIR = "/data/zhaoguiqin/nieuwland/src/t2i2v/t2i2v_project/t2i2v_pipeline/scene_plot_fuke_weigai_produce"

# 输出的训练元数据 JSON 文件路径
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_fuke_weigai.json"

project_count = 0
all_data = []

print(f"开始扫描数据集：{ROOT_DIR}")

# ==========================================
# 2. 动态寻找所有有效的产品文件夹
# ==========================================
product_paths = []
for root_path, dirs, files in os.walk(ROOT_DIR):
    if "reference_imgs" in dirs and "augmented_generated_images" in dirs:
        product_paths.append(root_path)

print(f"在目录下共找到了 {len(product_paths)} 个有效的产品文件夹。")

# ==========================================
# 3. 遍历提取数据
# ==========================================
for product_path in product_paths:
    product_name = os.path.basename(product_path) 
    
    found_reference_dir = os.path.join(product_path, "reference_imgs")
    found_target_dir = os.path.join(product_path, "augmented_generated_images")
    found_character_dir = os.path.join(product_path, "character_refs") if os.path.exists(os.path.join(product_path, "character_refs")) else None

    # ---------------------------------------------------
    # 第一步：获取并严格排序【参考图】与【角色图】
    # ---------------------------------------------------
    # 【核心修改点】：适配 "reference_img_12_0.png" 格式
    def get_ref_num(filename):
        # 先去掉前缀，变成 "12_0.png"
        clean_name = filename.replace("reference_img_", "")
        try:
            # 提取下划线前面的主序号 "12"
            return int(clean_name.split('_')[0])
        except:
            return 999999

    ref_imgs = [img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not ref_imgs:
        print(f"[跳过] {product_name}: 找不到产品参考图片")
        continue
    
    # 使用专门的提取函数进行精确排序
    ref_imgs.sort(key=get_ref_num)
    
    char_imgs = []
    if found_character_dir:
        char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])

    # ---------------------------------------------------
    # 第二步：获取并排序【目标帧】
    # ---------------------------------------------------
    target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]
    
    # 目标帧也是 "12_0.png" 格式，提取前面的 "12"
    def get_frame_num(filename):
        try: return int(filename.split('_')[0])
        except: return 999999 
            
    target_files.sort(key=get_frame_num) 

    has_valid_frame = False 
    
    # ---------------------------------------------------
    # 第三步：遍历目标帧，【一对一】组装数据
    # ---------------------------------------------------
    for i, target_file in enumerate(target_files):
        target_img_path = os.path.join(found_target_dir, target_file)
        
        # 寻找同名的 json 文件
        json_file = target_file.replace(".png", ".json")
        json_path = os.path.join(found_target_dir, json_file)
        if not os.path.exists(json_path): continue 
            
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                prompt_data = json.load(jf)
                real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
                    
                prompt_text = real_data.get("prompt", "")
                if not prompt_text:
                    prompt_text = real_data.get("product_info", "")
        except Exception as e:
            print(f"[读取JSON失败] 文件: {json_path}, 报错: {e}")
            continue

        current_edit_image_list = []
        
        # 精准匹配：取第 i 个目标帧对应的第 i 个参考图
        if i < len(ref_imgs):
            current_edit_image_list.append(os.path.join(found_reference_dir, ref_imgs[i]))
        else:
            # 防御机制：目标帧多于参考图时，用最后一张参考图兜底
            current_edit_image_list.append(os.path.join(found_reference_dir, ref_imgs[-1]))
            
        # 角色图固定拼在最后
        if char_imgs:
            current_edit_image_list.append(os.path.join(found_character_dir, char_imgs[0]))

        json_line = {
            "image": target_img_path,                             
            "edit_image": current_edit_image_list, 
            "prompt": prompt_text                                 
        }
        
        all_data.append(json_line) 
        has_valid_frame = True
        
    if has_valid_frame:
        project_count += 1

# ==========================================
# 4. 导出
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("\n==========================================")
print(f"🎉 新数据集扫描并串联完毕 (1对1参考图对齐)！")
print(f"📦 共成功处理了 {project_count} 个产品")
print(f"✅ 共生成了 {len(all_data)} 条训练数据")
print("==========================================")