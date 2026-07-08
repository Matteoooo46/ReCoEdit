import json
import os

def merge_json_files(directory_path, file1_name, file2_name, output_name="merged_output.json"):
    """
    合并指定路径下的两个 JSON 文件
    """
    # 1. 拼接完整的绝对路径
    path1 = os.path.join(directory_path, file1_name)
    path2 = os.path.join(directory_path, file2_name)
    out_path = os.path.join(directory_path, output_name)

    # 2. 检查文件是否存在
    if not os.path.exists(path1):
        raise FileNotFoundError(f"❌ 找不到文件: {path1}")
    if not os.path.exists(path2):
        raise FileNotFoundError(f"❌ 找不到文件: {path2}")

    print(f"📂 正在读取文件 1: {file1_name}")
    with open(path1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
        
    print(f"📂 正在读取文件 2: {file2_name}")
    with open(path2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    # 3. 智能判断数据类型并合并
    print("🔄 正在智能合并数据...")
    
    # 情况 A：如果两个 JSON 都是列表 (List) 格式 [ {}, {} ] -> 拼接列表
    if isinstance(data1, list) and isinstance(data2, list):
        merged_data = data1 + data2
        print(f"   -> 检测到列表格式，文件1有 {len(data1)} 条，文件2有 {len(data2)} 条，合并后共 {len(merged_data)} 条。")
        
    # 情况 B：如果两个 JSON 都是字典 (Dict) 格式 { "a": 1 }, { "b": 2 } -> 合并键值对
    elif isinstance(data1, dict) and isinstance(data2, dict):
        merged_data = data1.copy()
        merged_data.update(data2) # 注意：如果遇到相同的键，文件2的值会覆盖文件1的值
        print(f"   -> 检测到字典格式，已合并所有键值对。")
        
    else:
        raise ValueError("❌ JSON 数据结构不一致或不支持直接合并（必须同为列表或同为字典）！")

    # 4. 导出合并后的新 JSON 文件
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(merged_data, fout, ensure_ascii=False, indent=4)

    print("\n==========================================")
    print(f"🎉 任务完美结束！")
    print(f"💾 合并后的文件已保存至: {out_path}")
    print("==========================================")

# ==========================================
# 🚀 主程序入口：在这里修改你的路径和文件名
# ==========================================
if __name__ == "__main__":
    # 你的目标文件夹路径 (请替换为你自己的路径)
    TARGET_DIR = "/data/phd/kousiqi/zhitao" 
    
    # 需要合并的两个文件名
    FILE_1 = "metadata_all_products_expanded1.json" 
    FILE_2 = "metadata_fuke_weigai_optimized.json"
    
    # 合并后生成的新文件名
    OUTPUT_FILE = "metadata_all_products_expanded_ultimate.json"
    
    # 执行合并
    merge_json_files(TARGET_DIR, FILE_1, FILE_2, OUTPUT_FILE)