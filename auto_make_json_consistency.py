import os
import json

# ==========================================
# 1. 基础路径配置
# ==========================================
ROOT_DIR = "/data/phd/lijiahui/data"
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products_consistency.json"

project_count = 0
all_data = [] # 【修改点 1】：用一个大列表来装所有数据，最后直接导出标准 JSON

print("开始扫描数据集并串联关键帧")

# 遍历 ROOT_DIR 下的每一个 workspace_item 文件夹
workspace_paths = []
# root是当前路径，dirs是当前路径下的文件夹列表
for root_path, dirs, files in os.walk(ROOT_DIR):
    for dir_name in dirs:
        # 只要文件夹名字是以 workspace_ 开头，就把它加入名单
        if dir_name.startswith("workspace_"):
            workspace_paths.append(os.path.join(root_path, dir_name))

print(f"在 data 目录下共找到了 {len(workspace_paths)} 个 workspace 文件夹。")

# 遍历我们揪出来的所有 workspace 文件夹
for workspace_path in workspace_paths:
    workspace_name = os.path.basename(workspace_path) # 提取出文件夹名字

    # 1. 使用 os.walk 动态寻找三个核心文件夹
    found_character_dir = None
    found_reference_dir = None
    found_target_dir = None
    
    for root, dirs, files in os.walk(workspace_path):
        if "character_refs" in dirs:
            found_character_dir = os.path.join(root, "character_refs")
        if "reference_imgs" in dirs:
            found_reference_dir = os.path.join(root, "reference_imgs")
        if "augmented_generated_images_v8_v6_context_pro" in dirs:
            found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")
    
    # 检查是否找齐了文件夹
    if not found_character_dir or not found_reference_dir or not found_target_dir:
        print(f"[跳过] {workspace_name}: 未能找到所有必须的子文件夹")
        continue

    # ---------------------------------------------------
    # 第一步：获取【人物图】
    # ---------------------------------------------------
    char_imgs = [img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg'))]
    if not char_imgs: continue
    character_img_path = os.path.join(found_character_dir, char_imgs[0])

    # ---------------------------------------------------
    # 第二步：获取【产品图】
    # ---------------------------------------------------
    ref_imgs = [img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg'))]
    if not ref_imgs: continue
    reference_img_path = os.path.join(found_reference_dir, ref_imgs[0])

    # ---------------------------------------------------
    # 第三步：遍历【目标帧】并按顺序串联
    # ---------------------------------------------------
    # 获取所有的 png 帧
    target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]
    
    # 【修改点 2：极其重要！】必须对文件进行强行排序，否则 10_0.png 会排在 2_0.png 前面！
    # 提取文件名里的第一个数字进行排序 (例如把 "12_0.png" 提取出数字 12)
    def get_frame_num(filename):
        try:
            return int(filename.split('_')[0])
        except:
            return 999999 # 如果解析失败扔到最后
            
    target_files.sort(key=get_frame_num) # 严格按 0, 1, 2, 3... 排序

    has_valid_frame = False 
    prev_frame_path = None # 【修改点 3】：初始化“上一帧”的变量
    
    for target_file in target_files:
        target_img_path = os.path.join(found_target_dir, target_file)
        
        # 寻找对应的 json
        json_file = target_file.replace(".png", ".json")
        json_path = os.path.join(found_target_dir, json_file)
        if not os.path.exists(json_path): continue 
            
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                prompt_data = json.load(jf)
                if isinstance(prompt_data, list):
                    real_data = prompt_data[0]
                else:
                    real_data = prompt_data
                    
                prompt_text = real_data.get("prompt", "")
                if not prompt_text:
                    prompt_text = real_data.get("product_info", "")
        except Exception as e:
            print(f"[读取JSON失败] 文件: {json_path}, 报错: {e}")
            continue

        # ---------------------------------------------------
        # 第四步：组装包含“上一帧”的条件列表
        # ---------------------------------------------------
        # 基础条件：产品图 + 人物图
        edit_image_list = [reference_img_path, character_img_path]
        
        # 如果不是第一帧 (即 prev_frame_path 有内容)，就把上一帧的图片也加进去！
        if prev_frame_path is not None:
            edit_image_list.append(prev_frame_path)
            
        json_line = {
            "image": target_img_path,                             
            "edit_image": edit_image_list, # 现在的列表动态包含了 2 张或 3 张图
            "prompt": prompt_text                                 
        }
        
        all_data.append(json_line) # 装进大列表里
        
        has_valid_frame = True
        
        # 【修改点 4】：当前帧处理完毕，把它变成“上一帧”，留给下一轮循环使用！
        prev_frame_path = target_img_path
        
    if has_valid_frame:
        project_count += 1

# ==========================================
# 3. 一次性导出为标准 JSON 数组 (取代 fix_json)
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("==========================================")
print(f"🎉 扫描并串联完毕！")
print(f"📦 共成功处理了 {project_count} 个产品")
print(f"✅ 共生成了 {len(all_data)} 条带时间线关联的训练数据")
print(f"💾 标准格式数据集已保存在: {OUTPUT_JSON_PATH}")
print("==========================================")