import json
import torch
import time
import os
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. 配置参数
# ==========================================
INPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_transfer_online.json"
OUTPUT_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_transfer_online_clean.json"

# 💡 注意：CLIP 模型的图像相似度通常极高
# 0.98 以上几乎完全一致，0.95 可能是非常相似的构图和颜色
SIMILARITY_THRESHOLD = 0.95  

# ==========================================
# 2. 加载 CLIP 图像 Embedding 模型
# ==========================================
print("🌟 正在加载 CLIP 视觉大模型...")
model = SentenceTransformer('clip-ViT-B-32', device='cuda' if torch.cuda.is_available() else 'cpu')
print("✅ CLIP 模型加载完毕！")

# ==========================================
# 3. 读取数据并打开图片
# ==========================================
print(f"📂 正在读取数据集: {INPUT_JSON_PATH}")
with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"🔍 共加载了 {len(dataset)} 条数据，准备提取图片...")

valid_dataset = []
images_to_encode = []

for i, item in enumerate(dataset):
    # 【核心修改点】：读取 JSON 里的 "image" 字段 (即生成的最终目标帧)
    # 用目标帧去重，能避免同一个产品的不同帧被误杀
    main_img_path = item.get("image", "")
    
    if not main_img_path or not os.path.exists(main_img_path):
        print(f"⚠️ 第 {i} 条数据图片路径无效，跳过。")
        continue
        
    try:
        # 打开图片 (必须转换为 RGB 模式交由 CLIP 处理)
        img = Image.open(main_img_path).convert("RGB")
        images_to_encode.append(img)
        valid_dataset.append(item)
    except Exception as e:
        print(f"❌ 无法读取图片 {main_img_path}: {e}")

print(f"🖼️ 成功加载 {len(images_to_encode)} 张有效图片。")

# ==========================================
# 4. 批量计算图像 Embedding 向量
# ==========================================
print("🧠 正在使用 CLIP 将图片转换为高维特征向量 (Embedding)...")
start_time = time.time()
image_embeddings = model.encode(images_to_encode, convert_to_tensor=True, show_progress_bar=True)
print(f"⏱️ 向量计算完成，耗时 {time.time() - start_time:.2f} 秒。")

# ==========================================
# 5. 计算相似度矩阵并进行筛选
# ==========================================
print(f"⚖️ 正在进行图像余弦相似度对比 (阈值: {SIMILARITY_THRESHOLD})...")
cosine_scores = util.cos_sim(image_embeddings, image_embeddings)

keep_indices = []
dropped_indices = set()

for i in range(len(valid_dataset)):
    if i in dropped_indices:
        continue 
    
    keep_indices.append(i) # 保留当前数据
    
    # 查找后面的图片是否与当前图片撞车
    for j in range(i + 1, len(valid_dataset)):
        if j not in dropped_indices and cosine_scores[i][j] > SIMILARITY_THRESHOLD:
            dropped_indices.add(j)
            # 取消注释可以在终端看到到底删掉了哪些相似的图
            # print(f"  [✂️ 丢弃] 第 {j} 张图与第 {i} 张图高度相似 (相似度: {cosine_scores[i][j]:.4f})")

# ==========================================
# 6. 保存清洗后的数据
# ==========================================
filtered_dataset = [valid_dataset[i] for i in keep_indices]

print("\n==================================================")
print(f"📊 视觉清洗报告:")
print(f"  👉 原始有效数据 : {len(valid_dataset)} 条")
print(f"  👉 视觉雷同丢弃 : {len(dropped_indices)} 条")
print(f"  👉 最终保留数据 : {len(filtered_dataset)} 条")
print("==================================================")

with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(filtered_dataset, f, ensure_ascii=False, indent=4)
print(f"💾 清洗后的干净数据已保存至: {OUTPUT_JSON_PATH}")