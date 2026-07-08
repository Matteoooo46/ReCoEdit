import json
import os

# ==========================================
# 1. 配置参数
# ==========================================
# 输入的原始 JSON 文件路径 (如果在当前目录，直接写文件名即可)
INPUT_JSON_PATH = "metadata_all_products_expanded_ultimate.json"

# 替换后输出的新 JSON 文件路径 (建议先存为新文件，确认无误后再覆盖)
OUTPUT_JSON_PATH = "metadata_all_products_expanded_ultimate_updated.json"

# 需要被替换的旧路径
OLD_PATH = "/data/phd/kousiqi/zhitao/new_training_data"
# 要替换成的新路径
NEW_PATH = "/data/zhaoguiqin/nieuwland/src/t2i2v/t2i2v_project/t2i2v_pipeline"

# ==========================================
# 2. 执行替换逻辑
# ==========================================
def main():
    print(f"📂 正在读取文件: {INPUT_JSON_PATH}")
    
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ 找不到文件 {INPUT_JSON_PATH}，请确认路径是否正确！")
        return

    # 将整个 JSON 文件作为纯文本读取
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        raw_json_string = f.read()

    # 统计一下里面到底有多少个旧路径
    match_count = raw_json_string.count(OLD_PATH)
    print(f"🔍 扫描完毕，共发现 {match_count} 处需要替换的旧路径。")

    if match_count == 0:
        print("⚠️ 未找到匹配的旧路径，无需替换。")
        return

    print("⏳ 正在执行全局路径替换...")
    # 执行字符串替换
    updated_json_string = raw_json_string.replace(OLD_PATH, NEW_PATH)

    # ==========================================
    # 3. 格式安全校验与保存
    # ==========================================
    try:
        # 为了防止破坏 JSON 结构，我们强行把它当做 JSON 重新解析一次
        # 如果解析不报错，说明替换非常安全
        json_data = json.loads(updated_json_string)
        
        # 将验证通过的数据以标准的、带缩进的格式保存
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
            
        print("\n==================================================")
        print("🎉 路径替换完美成功！")
        print(f"📦 成功修改了 {match_count} 处路径")
        print(f"💾 新的配置文件已安全保存至: {OUTPUT_JSON_PATH}")
        print("==================================================")
        
    except json.JSONDecodeError as e:
        print(f"❌ 致命错误：替换后破坏了 JSON 的内部格式，保存中止！报错详情: {e}")

if __name__ == "__main__":
    main()