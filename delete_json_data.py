import json

# ==========================================
# 1. 配置参数
# ==========================================
INPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_transfer_online_optimized.json"     # 这里填你原始 JSON 文件的路径
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_transfer_online_optimized_removed.json"   # 这里填你想保存的新 JSON 文件路径

# 你想要作为判断依据的键名（通常是 "image" 或 "prompt"）
CHECK_KEY = "image"  

# 💥 【修改点 1】：改成了一个列表，你可以随意往里面添加多个要删除的名字/UUID
# 注意：每个名字用双引号包起来，后面加逗号分隔
TARGET_STRINGS = [
    "003a23da-dfc1-49fb-8b3a-495a3e9d99ae",
    "009785b8-36c6-4775-ada0-c9497e7072c2",
    "010ccf98-82c3-40f9-8284-65518eeff3a0",
    "024419f0-d5c4-482e-9572-ba7885cdf4e4",
    "02b9ab76-6cbb-4d03-a969-9f3d5050c4d8",
    "02cdc727-ab85-4692-8ef6-00b725c64141",
    "03278e2e-dab1-4b4d-a3ef-e1e5337549dd",
    "03641bdb-7a11-4c05-83a3-347d535e8c91",
    "03d7f47e-bb37-4a19-8685-0e231f933627",
    "03dac08d-8b1e-4a3b-a2d7-94e0ae9ee787",
    "056924c6-1454-4e99-a40c-b8e0b362529c",
    "064eb1bd-65cd-4fc5-8316-a172aa2f8f2f",
    "07b28d8b-d588-4683-b55d-83d59a89a9b0",
    "07b41236-33e3-4fec-bf34-c500ef7fb220",
    "08936a21-6a4a-40c1-828f-f30763afdf02"
    # 你可以继续在下面添加...
]

# ==========================================
# 2. 执行删除逻辑
# ==========================================
def main():
    print(f"📂 正在读取文件: {INPUT_JSON_PATH}")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {INPUT_JSON_PATH}，请检查路径是否正确！")
        return

    original_count = len(dataset)
    
    # 💥 【修改点 2】：核心批量过滤逻辑
    # 只要这个数据的 CHECK_KEY 字段里，包含了 TARGET_STRINGS 列表中的【任意一个】词，就不要它了
    filtered_dataset = [
        item for item in dataset 
        if not any(target in item.get(CHECK_KEY, "") for target in TARGET_STRINGS)
    ]
    
    deleted_count = original_count - len(filtered_dataset)

    # ==========================================
    # 3. 保存新文件
    # ==========================================
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, ensure_ascii=False, indent=4)

    print("\n==================================================")
    print(f"📊 批量数据删除报告:")
    print(f"  👉 过滤名单数量 : {len(TARGET_STRINGS)} 个目标")
    print(f"  👉 原始数据量   : {original_count} 条")
    print(f"  👉 成功删除     : {deleted_count} 条")
    print(f"  👉 最终保留量   : {len(filtered_dataset)} 条")
    print("==================================================")
    print(f"💾 清理后的数据已保存至: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()