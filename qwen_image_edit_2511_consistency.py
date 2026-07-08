import os
import torch
import json
import shutil
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image

# ======================================================================
# 💥 核心控制台：你想用第几号显卡？在这里填数字！
# ==========================================
TARGET_GPU_ID = "0"  # <--- 每次换卡，只改这一个数字！！！
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID

print(f"🚀 [物理隔离生效] 当前进程已死死锁定在物理显卡 GPU: {TARGET_GPU_ID}")

# ==========================================
# 1. 基础路径配置
# ==========================================
# ⚠️ 填入你要推理的那个产品的文件夹路径 (支持动态扫描)
TARGET_WORKSPACE_DIR = "/data/phd/lijiahui/data/batch_no_voice_gen_shipin/workspace_item_21825264046857"

# 结果保存的父目录
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_autoregressive"

# 获取最末级的文件夹名称，为了避免重复，这里用上一级和本级的组合名作为标识
product_name = os.path.basename(TARGET_WORKSPACE_DIR)

# 存放生成的原始图 (frame_0.jpg, frame_1.jpg)
RAW_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw_autoregressive_initial")
# 存放按网页规则重命名好的展示图 (p1_nano_1.png, p1_my_1.png)
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_report_autoregressive_initial")

os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================
# 2. 初始化模型
# ==========================================
print("正在加载模型和LoRA，请稍候...")
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

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=MODEL_CONFIG_PATHS["dit"], **vram_config),
        ModelConfig(path=MODEL_CONFIG_PATHS["text_encoder"], **vram_config),
        ModelConfig(path=MODEL_CONFIG_PATHS["vae"], **vram_config)
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(path=MODEL_CONFIG_PATHS["processor"]),
    vram_limit=75
)

# ⚠️ 注意这里换成了你第二段代码里用的自回归特化版 LoRA
lora_path = "/data/phd/kousiqi/zhitao/lora_all_products/step-21000.safetensors"
pipe.load_lora(pipe.dit, lora_path)

print("模型加载完毕！")


# ==========================================
# 3. 定位核心文件夹及准备基础参考图
# ==========================================
print(f"\n📂 开始扫描产品: {TARGET_WORKSPACE_DIR}")

found_character_dir = None
found_reference_dir = None
found_target_dir = None

# 使用精准拼接路径，防止找不到
found_character_dir = os.path.join(TARGET_WORKSPACE_DIR, "character_refs")
found_reference_dir = os.path.join(TARGET_WORKSPACE_DIR, "reference_imgs")
found_target_dir = os.path.join(TARGET_WORKSPACE_DIR, "augmented_generated_images_v8_v6_context_pro")

for root, dirs, files in os.walk(TARGET_WORKSPACE_DIR):
    if "character_refs" in dirs: found_character_dir = os.path.join(root, "character_refs")
    if "reference_imgs" in dirs: found_reference_dir = os.path.join(root, "reference_imgs")
    if "augmented_generated_images_v8_v6_context_pro" in dirs: found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")
        
if not found_character_dir or not found_reference_dir or not found_target_dir:
    raise ValueError(f"❌ 未能在 {TARGET_WORKSPACE_DIR} 中找齐所有必须的子文件夹！")

# 获取并加载【人物图】
char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg'))])
if not char_imgs: raise FileNotFoundError("找不到人物图！")
char_img_path = os.path.join(found_character_dir, char_imgs[0])

# 获取并加载所有的【产品图】
ref_imgs = sorted([img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg'))])
if not ref_imgs: raise FileNotFoundError("找不到产品图！")

# 准备基础参考图列表（每一帧的固定根底）
base_edit_images = []
for idx, ref_img in enumerate(ref_imgs):
    ref_path = os.path.join(found_reference_dir, ref_img)
    base_edit_images.append(Image.open(ref_path).convert("RGB"))
    # 拷贝产品参考图到 Report 目录
    shutil.copy(ref_path, os.path.join(REPORT_DIR, f"p1_ref_{idx}.png"))

# 追加人物图
base_edit_images.append(Image.open(char_img_path).convert("RGB"))
# 拷贝人物参考图到 Report 目录
shutil.copy(char_img_path, os.path.join(REPORT_DIR, "p1_char.png"))

print(f"✅ 基础参考图片整理完毕。固定不变的基础图共 {len(base_edit_images)} 张。")


# ==========================================
# 4. 自动读取目标帧并进行自回归生成
# ==========================================
target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]

def get_frame_num(filename):
    try: return int(filename.split('_')[0])
    except: return 999999
        
target_files.sort(key=get_frame_num)
print(f"🔍 共扫描到 {len(target_files)} 帧连续画面需要生成。")

prev_generated_img = None # 核心变量：用来记住“上一帧”

for i, target_file in enumerate(target_files):
    json_file = target_file.replace(".png", ".json")
    json_path = os.path.join(found_target_dir, json_file)
    
    if not os.path.exists(json_path): continue 
        
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            prompt_data = json.load(jf)
            real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
            prompt_text = real_data.get("prompt", "") or real_data.get("product_info", "")
    except Exception as e:
        print(f"  [读取JSON失败] 文件: {json_file}, 跳过。")
        continue
        
    if not prompt_text: continue

    print(f"\n---> 正在自回归生成第 {i} 帧...")
    print(f"Prompt: {prompt_text[:50]}...")
    
    # 💥 【核心自回归逻辑】
    # 复制一份基础参考图列表（防止污染下一帧的基础池）
    current_edit_images = base_edit_images.copy()
    
    if prev_generated_img is not None:
        print(f"   🔄 已将第 {i-1} 帧生成的画面作为垫图追加进入参考队列！(当前队列共 {len(current_edit_images) + 1} 张图)")
        current_edit_images.append(prev_generated_img)
    else:
        print("   ✅ 这是第 0 帧（起始帧），使用纯净的基础参考图启动。")

    # 推理生成 
    try:
        generated_image = pipe(
            prompt=prompt_text,
            edit_image=current_edit_images, 
            seed=i, # 让每一帧的种子稍微变一变，增加一点画面动态
            num_inference_steps=40,
            height=1152,
            width=896,
            zero_cond_t=True, 
        )
        
        # 1. 保存原始推理结果 (frame_0.jpg ...)
        my_img_path = os.path.join(RAW_OUTPUT_DIR, f"frame_{i}.jpg")
        generated_image.save(my_img_path)
        print(f"   💾 第 {i} 帧已保存至: {my_img_path}")
        
        # 【灵魂一步】：把当前生成的这幅新画，变成对象存起来，留给下一帧吃掉！
        prev_generated_img = Image.open(my_img_path).convert("RGB")
        
        # 2. 整理展示对比图
        nano_img_path = os.path.join(found_target_dir, target_file)
        if os.path.exists(nano_img_path):
            shutil.copy(nano_img_path, os.path.join(REPORT_DIR, f"p1_nano_{i+1}.png"))
        if os.path.exists(my_img_path):
            shutil.copy(my_img_path, os.path.join(REPORT_DIR, f"p1_my_{i+1}.png"))
            
    except Exception as e:
        print(f"  ❌ [推理失败] 帧: {target_file}, 报错: {e}")

# ==========================================
# 5. 打包输出
# ==========================================
print("\n==================================================")
print("开始自动整理与打包...")
folder_name = os.path.basename(REPORT_DIR)
zip_file_path = f"{REPORT_DIR}.zip" 

try:
    print(f"正在压缩 {folder_name} ...")
    os.system(f"cd {OUTPUT_BASE_DIR} && zip -r {folder_name}.zip {folder_name}/")
    print(f"🎉 任务完美结束！所有自回归连贯帧生成完毕。")
    print(f"📦 压缩包下载路径: {zip_file_path}")
    print("==================================================")
except Exception as e:
    print(f"压缩失败，请手动在终端执行 zip 命令。报错：{e}")