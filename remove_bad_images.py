import json
import os
from PIL import Image

# ==========================================
# 1. 配置参数
# ==========================================
INPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_fuke_weigai.json" # 你的原始JSON路径
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_fuke_weigai_removed.json" # 清洗后的JSON路径

# ==========================================
# 2. 图像安全校验函数
# ==========================================
def is_valid_image(img_path):
    """
    全面校验图片是否可用
    1. 检查路径是否为空
    2. 检查文件是否存在于磁盘
    3. 检查文件大小是否为 0 字节 (通常发生在磁盘满时)
    4. 检查文件头部信息是否完整 (使用 PIL.Image.verify 快速校验)
    """
    if not img_path:
        return False, "路径为空"
        
    if not os.path.exists(img_path):
        return False, "文件不存在"
        
    if os.path.getsize(img_path) == 0:
        return False, "0字节空文件"
        
    try:
        # verify() 是轻量级检查，只读取文件头，速度极快，专门用于检测图片是否损坏
        with Image.open(img_path) as img:
            img.verify() 
        return True, "正常"
    except Exception as e:
        return False, f"图片损坏 ({e})"

# ==========================================
# 3. 执行清洗逻辑
# ==========================================
def main():
    print(f"📂 正在读取数据集: {INPUT_JSON_PATH}")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {INPUT_JSON_PATH}")
        return

    original_count = len(dataset)
    valid_dataset = []
    error_log = []

    print(f"🔍 共加载了 {original_count} 条数据，开始深度逐图校验 (这可能需要十几秒)...\n")

    for idx, item in enumerate(dataset):
        is_item_valid = True
        
        # 1. 检查目标图 (image)
        target_img = item.get("image", "")
        valid, reason = is_valid_image(target_img)
        if not valid:
            error_log.append(f"数据 [{idx}]: 目标图异常 -> {reason} ({target_img})")
            continue # 直接跳过这条数据

        # 2. 检查所有参考图 (edit_image 列表)
        ref_imgs = item.get("edit_image", [])
        for ref_img in ref_imgs:
            valid, reason = is_valid_image(ref_img)
            if not valid:
                error_log.append(f"数据 [{idx}]: 参考图异常 -> {reason} ({ref_img})")
                is_item_valid = False
                break # 发现一张坏图，跳出内部循环
                
        # 3. 如果所有图片都完好，则保留该数据
        if is_item_valid:
            valid_dataset.append(item)

    # ==========================================
    # 4. 保存与报告
    # ==========================================
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f, ensure_ascii=False, indent=4)

    deleted_count = original_count - len(valid_dataset)

    print("==================================================")
    print(f"📊 坏图清洗报告:")
    print(f"  👉 原始数据量 : {original_count} 条")
    print(f"  👉 发现坏数据 : {deleted_count} 条 (已剔除)")
    print(f"  👉 最终保留量 : {len(valid_dataset)} 条")
    print("==================================================")
    print(f"💾 健康数据集已保存至: {OUTPUT_JSON_PATH}")
    
    # 打印前 5 条具体报错信息，方便你排查问题原因
    if error_log:
        print("\n🔎 典型错误样例 (前5条):")
        for err in error_log[:5]:
            print("  - " + err)

if __name__ == "__main__":
    main()