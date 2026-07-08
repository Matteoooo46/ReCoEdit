import os
import json
import shutil
import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# ======================================================================
# 🌟 唯一需要你修改的地方：把你要攻克的那个“失败产品”的路径贴在这里
# ======================================================================
TARGET_WORKSPACE_DIR = "/data/phd/lijiahui/data/0311_gen_2_eyemax_pro/workspace_25921759763142_1773237018"

# 结果统一存放到这个安全区域
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/hard_products_rescue"
product_name = os.path.basename(TARGET_WORKSPACE_DIR)

RAW_GEN_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw_gen")
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_report")
TRAIN_JSON_PATH = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_training_data.json")

os.makedirs(RAW_GEN_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 防显存爆炸黑科技
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 初始化模型 (带显存优化)
# ==========================================
print(f"🚀 开始全力攻克产品: {product_name}")
print("正在加载模型...")

MODEL_CONFIG_PATHS = {
    "dit": [
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"
    ],
    "text_encoder": [
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/text_encoder/model-00001-of-00004.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/text_encoder/model-00002-of-00004.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/text_encoder/model-00003-of-00004.safetensors",
        "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/text_encoder/model-00004-of-00004.safetensors"
    ],
    "vae": "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors",
    "processor": "/data/phd/kousiqi/kousiqi/models--Qwen--Qwen-Image-Edit-2511/processor",
}

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=MODEL_CONFIG_PATHS["dit"]),
        ModelConfig(path=MODEL_CONFIG_PATHS["text_encoder"]),
        ModelConfig(path=MODEL_CONFIG_PATHS["vae"])
    ],
    processor_config=ModelConfig(path=MODEL_CONFIG_PATHS["processor"]),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2 # 自动留出显存余量防崩溃
)

# ==========================================
# 2. 全自动雷达：扫描所有子文件夹与图片
# ==========================================
found_char_dir, found_ref_dir, found_target_dir = None, None, None

for root, dirs, files in os.walk(TARGET_WORKSPACE_DIR):
    if "character_refs" in dirs: found_char_dir = os.path.join(root, "character_refs")
    if "reference_imgs" in dirs: found_ref_dir = os.path.join(root, "reference_imgs")
    if "augmented_generated_images_v8_v6_context_pro" in dirs: found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")

if not found_char_dir or not found_ref_dir or not found_target_dir:
    raise ValueError(f"❌ 文件夹残缺，这不是一个完整的有效数据！")

# 自动抓取所有参考图和人物图
char_imgs = sorted([img for img in os.listdir(found_char_dir) if img.endswith(('.png', '.jpg'))])
ref_imgs = sorted([img for img in os.listdir(found_ref_dir) if img.endswith(('.png', '.jpg'))])

if not char_imgs or not ref_imgs:
    raise ValueError(f"❌ 找不到图片，该数据为空壳！")

char_img_path = os.path.join(found_char_dir, char_imgs[0])
ref_img_paths = [os.path.join(found_ref_dir, img) for img in ref_imgs]

# 提取目标帧并严格排序
target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]
target_files.sort(key=lambda x: int(x.split('_')[0]) if '_' in x else 999)

print(f"✅ 自动识别成功：找到 {len(ref_img_paths)} 张产品参考图，1 张人物图，共需生成 {len(target_files)} 帧。")

# 准备展示网页的参考图 (拷贝到 report)
shutil.copy(char_img_path, os.path.join(REPORT_DIR, "p1_char.png"))
for idx, p in enumerate(ref_img_paths):
    shutil.copy(p, os.path.join(REPORT_DIR, f"p1_ref_{idx}.png"))

# ==========================================
# 3. 核心推理与数据组装循环
# ==========================================
all_training_data = []
prev_generated_img_path = None
prev_nano_img_path = None

for i, target_file in enumerate(target_files):
    target_img_path = os.path.join(found_target_dir, target_file)
    json_path = os.path.join(found_target_dir, target_file.replace(".png", ".json"))
    
    if not os.path.exists(json_path): continue
        
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            p_data = json.load(jf)
            real_data = p_data[0] if isinstance(p_data, list) else p_data
            prompt_text = real_data.get("prompt", "") or real_data.get("product_info", "")
    except Exception: continue
    if not prompt_text: continue

    print(f"\n---> 正在生成第 {i} 帧...")
    
    # 【自动隔离】：装载当前帧所需的图像列表
    pil_edit_images = [Image.open(p).convert("RGB") for p in ref_img_paths] + [Image.open(char_img_path).convert("RGB")]
    json_edit_images = ref_img_paths + [char_img_path]

    # 🧨 推理时：加入上一帧 Nano 图
    if prev_nano_img_path and os.path.exists(prev_nano_img_path):
        pil_edit_images.append(Image.open(prev_nano_img_path).convert("RGB"))
        print(f"   👁️ 已加入上一帧 Nano 图辅助推理")

    # 🧨 训练时：JSON 里记录上一帧生成的图
    if prev_generated_img_path:
        json_edit_images.append(prev_generated_img_path)
        print(f"   📝 已将上一帧生成图记入训练 JSON")

    # 执行大模型推理
    try:
        generated_image = pipe(
            prompt=prompt_text,
            edit_image=pil_edit_images,
            num_inference_steps=20,
            seed=42
        )
        
        # 1. 保存生成的原始图
        current_gen_path = os.path.join(RAW_GEN_DIR, f"frame_{i}.jpg")
        generated_image.save(current_gen_path)
        
        # 2. 拷贝生成图和Nano图供网页对比
        shutil.copy(current_gen_path, os.path.join(REPORT_DIR, f"p1_my_{i+1}.png"))
        if os.path.exists(target_img_path):
            shutil.copy(target_img_path, os.path.join(REPORT_DIR, f"p1_nano_{i+1}.png"))

        # 3. 组装这完美的一条训练数据！
        json_line = {
            "image": target_img_path,
            "edit_image": json_edit_images,
            "prompt": prompt_text
        }
        all_training_data.append(json_line)
        
        # 4. 更新指针，留给下一帧用
        prev_generated_img_path = current_gen_path
        prev_nano_img_path = target_img_path
        
    except Exception as e:
        print(f"❌ [推理失败] 帧: {target_file}, 报错: {e}")
        continue

# ==========================================
# 4. 收尾：保存专属训练数据并打包
# ==========================================
with open(TRAIN_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_training_data, f, ensure_ascii=False, indent=2)

print(f"\n==================================================")
print(f"🎉 成功攻克产品！")
print(f"💾 该产品专属的训练数据集已生成: {TRAIN_JSON_PATH} (共 {len(all_training_data)} 条数据)")

folder_name = os.path.basename(REPORT_DIR)
try:
    os.system(f"cd {OUTPUT_BASE_DIR} && zip -rq {folder_name}.zip {folder_name}/")
    print(f"📦 展示网页所需图片已打包: {os.path.join(OUTPUT_BASE_DIR, folder_name + '.zip')}")
except Exception: pass
print(f"==================================================")