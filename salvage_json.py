import os
import glob
import json

# ==========================================
# 1. 配置路径
# ==========================================
# 你存放这 40 个 json 文件的目录
BASE_DIR = "/data/phd/kousiqi/zhitao"

# 使用通配符 (*) 精准匹配所有切片文件
# 根据你的截图，文件名特征是 metadata_all_products_autoregressive_chunk_...
SEARCH_PATTERN = os.path.join(BASE_DIR, "metadata_all_products_autoregressive_chunk_*.json")

# 合并后的终极版 JSON 文件名
OUTPUT_FILE = os.path.join(BASE_DIR, "metadata_all_products_autoregressive_merged_added.json")

# ==========================================
# 2. 扫描并合并数据
# ==========================================
all_merged_data = []
file_list = glob.glob(SEARCH_PATTERN)

print(f"🔍 自动扫描到 {len(file_list)} 个 JSON 切片文件，开始执行合并术...")

for file_path in file_list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 因为每个 json 都是一个列表，所以用 extend 把它们拆包后塞进大列表里
            if isinstance(data, list):
                all_merged_data.extend(data)
            else:
                all_merged_data.append(data)
    except Exception as e:
        print(f"❌ 读取文件失败跳过 {os.path.basename(file_path)}: {e}")

# ==========================================
# 3. 导出终极文件
# ==========================================
print(f"💾 合并计算完毕！总共收集到了 {len(all_merged_data)} 条珍贵的训练数据。")
print(f"⏳ 正在将巨大的数据块写入硬盘，请稍候...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    # 保持格式化输出，方便以后人类肉眼检查
    json.dump(all_merged_data, f, ensure_ascii=False, indent=2)

print("==========================================")
print(f"🎉 终极合并大业完成！")
print(f"📁 你的最终训练集在这里: {OUTPUT_FILE}")
print("==========================================")