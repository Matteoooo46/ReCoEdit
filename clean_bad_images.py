import json
import os
from PIL import Image
from tqdm import tqdm

# 你的训练账本路径
JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded.json"
# 建议换个新名字，防止覆盖
CLEAN_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_clean.json"

print("🔍 开始严格排雷：一旦发现坏图，连坐剔除整个产品的所有数据...")

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    all_data = json.load(f)

# ==========================================
# 1. 先把散装的数据按“产品文件夹”进行分组
# ==========================================
print("📦 正在按产品归类数据...")
grouped_data = {}
for item in all_data:
    img_path = item.get("image")
    # 用图片所在的父文件夹路径作为该产品的唯一标识
    product_dir = os.path.dirname(img_path)
    
    if product_dir not in grouped_data:
        grouped_data[product_dir] = []
    grouped_data[product_dir].append(item)

clean_data = []
bad_product_count = 0
good_product_count = 0

print(f"🔎 共发现 {len(grouped_data)} 个产品，开始逐个进行严格质检...")

# ==========================================
# 2. 以“产品”为单位进行连坐质检
# ==========================================
for product_dir, items in tqdm(grouped_data.items()):
    product_is_good = True
    
    # 检查该产品下的每一帧数据
    for item in items:
        img_path = item.get("image")
        
        # 检查主图
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            product_is_good = False
            break # 💥 发现一张坏图，直接判定整个产品死刑，跳出内循环
            
        # 检查条件图 (edit_image)
        for edit_img_path in item.get("edit_image", []):
            try:
                with Image.open(edit_img_path) as img:
                    img.verify()
            except Exception:
                product_is_good = False
                break # 💥 条件图坏了也一样判定死刑
        
        # 如果内层循环(条件图)发现了坏图，外层也要马上打断
        if not product_is_good:
            break
            
    # ==========================================
    # 3. 宣判结果
    # ==========================================
    if product_is_good:
        # 全员健康，将该产品的所有帧(整个 items 列表)一口气加入白名单
        clean_data.extend(items)
        good_product_count += 1
    else:
        # 连坐淘汰
        bad_product_count += 1
        print(f"\n💣 发现坏图，整个产品被连坐剔除: {product_dir}")

# ==========================================
# 4. 保存最纯净的账本
# ==========================================
with open(CLEAN_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=4)

print("\n==========================================")
print(f"🎉 严格扫雷完毕！")
print(f"🗑️ 共剔除了 {bad_product_count} 个包含坏图的产品文件夹。")
print(f"📦 完美保留了 {good_product_count} 个绝对健康的完整产品。")
print(f"✅ 最终剩余 {len(clean_data)} 条极高质量的训练数据！")
print(f"💾 干净的账本已保存至: {CLEAN_JSON_PATH}")
print("==========================================")