import os
import json

# ==========================================
# 1. 基础路径配置
# ==========================================
# 新的数据集根目录
ROOT_DIR = "/data/phd/yaozhengjian/zjYao_Datasets/Doubao/nanoBanana_images"
# 生成的训练 JSON 存放位置
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_doubao_nanobanana.json"

# 💥 核心修改：设置你要提取的目标数据条数
TARGET_COUNT = 500000

project_count = 0
all_data = [] 

print(f"🚀 开始扫描 Doubao NanoBanana 数据集，目标提取 {TARGET_COUNT} 条数据...")

# ==========================================
# 2. 暴力且精准的文件夹遍历逻辑
# ==========================================
# root 是当前所在的绝对路径，files 是该路径下所有的文件
for root, dirs, files in os.walk(ROOT_DIR):
    
    # 💥 拦截器 1：如果总数据量已经达标，直接终止整个外层扫描，节省时间
    if len(all_data) >= TARGET_COUNT:
        break
    
    # 🌟 核心判断：只要这个文件夹里有这个总的 JSON 文件，说明我们找对地方了
    if "scene_prompts_condition.json" in files:
        
        # 1. 获取参考图和 JSON 的绝对路径
        condition_img_path = os.path.join(root, "condition.jpg")
        json_path = os.path.join(root, "scene_prompts_condition.json")
        
        # 如果连参考图都没有，直接跳过这个损坏的数据
        if not os.path.exists(condition_img_path):
            continue
            
        # 2. 读取那个包含了 8 条 prompt 的数组列表
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                prompts_list = json.load(f)
        except Exception as e:
            print(f"❌ [读取JSON失败] 文件: {json_path}, 报错: {e}")
            continue

        has_valid_frame = False
        
        # 3. 遍历这个 prompt 列表，并和对应的分镜图片拼对
        for i, prompt_text in enumerate(prompts_list):
            
            # 💥 拦截器 2：在添加每一帧之前检查，一旦达标立刻退出内层循环
            if len(all_data) >= TARGET_COUNT:
                break
            
            target_img_name = f"output_img_condition_prompt_{i+1}.png"
            target_img_path = os.path.join(root, target_img_name)
            
            # 检查这张分镜图在不在硬盘上
            if not os.path.exists(target_img_path):
                continue
                
            # 4. 组装最终供大模型训练用的格式
            edit_image_list = [condition_img_path]
            
            json_line = {
                "image": target_img_path,                             
                "edit_image": edit_image_list, 
                "prompt": prompt_text                                 
            }
            
            all_data.append(json_line)
            has_valid_frame = True
            
        # 只要成功装载了至少一帧，这个 asset 就算有效
        if has_valid_frame:
            project_count += 1

# ==========================================
# 3. 导出标准 JSON 数组
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("==========================================")
print(f"🎉 数据集转换完毕！")
print(f"📦 共扫描并涉及了 {project_count} 个 Asset 文件夹")
print(f"✅ 完美！共精准生成了 {len(all_data)} 条训练数据")
print(f"💾 标准格式数据集已保存在: {OUTPUT_JSON_PATH}")
print("==========================================")