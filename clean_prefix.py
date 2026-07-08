import json
import os

# ==========================================
# 1. 配置文件路径
# ==========================================
# 填入你刚才生成的那个 json 的路径
INPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products.json"

# 自动生成一个带 _clean 后缀的新文件路径
OUTPUT_JSON_PATH = INPUT_JSON_PATH.replace(".json", "_clean.json")

# 定义要干掉的前缀（注意末尾带一个空格，这样切完之后首字母就不会多出一个空格了）
PREFIX_TO_REMOVE = "IMG_1018.CR2 "

print(f"🔍 开始读取数据集: {INPUT_JSON_PATH}")

# ==========================================
# 2. 读取数据
# ==========================================
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ 读取文件失败: {e}")
    exit()

# ==========================================
# 3. 遍历清洗 Prompt
# ==========================================
cleaned_count = 0

for item in data:
    original_prompt = item.get("prompt", "")
    
    # 判断 prompt 是不是以这个前缀开头
    if original_prompt.startswith(PREFIX_TO_REMOVE):
        # 核心逻辑：利用 Python 切片，直接把前缀长度的部分砍掉，只保留后面的内容
        item["prompt"] = original_prompt[len(PREFIX_TO_REMOVE):]
        cleaned_count += 1

# ==========================================
# 4. 保存纯净版数据
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    # indent=4 保证输出的 json 依然是格式化好的漂亮格式
    json.dump(data, f, ensure_ascii=False, indent=4)

print("==========================================")
print(f"🎉 清洗手术成功！")
print(f"✂️ 共扫描了 {len(data)} 条数据，成功切除了 {cleaned_count} 个烦人的前缀。")
print(f"💾 纯净版 JSON 已安全保存至: {OUTPUT_JSON_PATH}")
print("==========================================")