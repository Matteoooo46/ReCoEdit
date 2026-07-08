from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
import os
import json
import shutil
import wandb

# ==========================================
# 1. 基础路径配置
# ==========================================
# ⚠️ 填入你要推理的那个产品的文件夹路径
TARGET_WORKSPACE_DIR = "/data/phd/kousiqi/zhitao/new_validation_set/08936a21-6a4a-40c1-828f-f30763afdf02"

# 结果保存的父目录
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"

product_name = os.path.basename(TARGET_WORKSPACE_DIR)

# 存放生成的原始图 (frame_0.jpg, frame_1.jpg)
RAW_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw")
# 存放按网页规则重命名好的展示图 (p1_nano_1.png, p1_my_1.png)
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_rewrite_expanded")

os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================
# 1.5 初始化 wandb
# ==========================================
wandb.init(
    entity="matteo46-sjtu",
    project="qwen-inference-results",
    name=product_name,
    config={
        "num_inference_steps": 40,
        "height": 1152,
        "width": 896,
        "product": product_name,
        "use_lora": False,
    },
)
result_table = wandb.Table(columns=["frame", "prompt", "nano", "my"])
ref_table = wandb.Table(columns=["type", "image"])

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
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
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
    vram_limit=135
)

# lora_path = "/data/phd/kousiqi/zhitao/lora_all_products_rewrite_expanded_ultimate_resume_137000/step-3000.safetensors"
# pipe.load_lora(pipe.dit, lora_path)

# print("模型加载完毕！")

# ==========================================
# 3. 定位核心文件夹及准备参考图
# ==========================================
print(f"\n开始处理产品: {product_name}")

# found_character_dir = None
found_reference_dir = None
found_target_dir = None

for root, dirs, files in os.walk(TARGET_WORKSPACE_DIR):
    # if "character_refs" in dirs: found_character_dir = os.path.join(root, "character_refs")
    if "reference_imgs" in dirs: found_reference_dir = os.path.join(root, "reference_imgs")
    if "augmented_generated_images" in dirs: found_target_dir = os.path.join(root, "augmented_generated_images")
        
if not found_reference_dir or not found_target_dir:
    raise ValueError(f"❌ 未能在 {TARGET_WORKSPACE_DIR} 中找齐所有必须的子文件夹！")

# 获取并加载【人物图】
# char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg'))])
# if not char_imgs: raise FileNotFoundError("找不到人物图！")
# char_img_path = os.path.join(found_character_dir, char_imgs[0])

# 获取并加载所有的【产品图】
ref_imgs = sorted([img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg'))])
if not ref_imgs: raise FileNotFoundError("找不到产品图！")

# 准备用于推理的列表：动态加入所有产品图 + 1张人物图（这里固定不变，不加生成帧）
edit_image_list = []
for idx, ref_img in enumerate(ref_imgs):
    ref_path = os.path.join(found_reference_dir, ref_img)
    edit_image_list.append(Image.open(ref_path).convert("RGB"))
    # 按照规则：拷贝产品参考图到 Report 目录 (p1_ref_0.png, p1_ref_1.png...)
    shutil.copy(ref_path, os.path.join(REPORT_DIR, f"p1_ref_{idx}.png"))
    ref_table.add_data(f"product_ref_{idx}", wandb.Image(ref_path))

# edit_image_list.append(Image.open(char_img_path).convert("RGB"))
# 按照规则：拷贝人物参考图到 Report 目录 (p1_char.png)
# shutil.copy(char_img_path, os.path.join(REPORT_DIR, "p1_char.png"))

print(f"✅ 参考图片已整理至 Report 文件夹。用于推理的图片共 {len(edit_image_list)} 张。")

# ==========================================
# 4. 获取帧列表并排序
# ==========================================
target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]

def get_frame_num(filename):
    try: return int(filename.split('_')[0])
    except: return 999999
        
target_files.sort(key=get_frame_num)
print(f"共发现 {len(target_files)} 帧需要生成。")

# ==========================================
# 5. 开始独立逐帧推理与重命名整理
# ==========================================
for i, target_file in enumerate(target_files):
    json_file = target_file.replace(".png", ".json")
    json_path = os.path.join(found_target_dir, json_file)
    
    if not os.path.exists(json_path): continue 
        
    # 读取 Prompt
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            prompt_data = json.load(jf)
            real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
            prompt_text = real_data.get("prompt", "") or real_data.get("product_info", "")
    except Exception as e:
        print(f"  [读取JSON失败] 文件: {json_file}, 报错: {e}")
        continue
        
    if not prompt_text: continue

    print(f"\n---> 正在生成第 {i} 帧...")
    
    # 💥 魔法 1：在正向提示词末尾，强行追加“不要文字”的指令
    prompt_text = prompt_text + ", pure visual, clean background, strictly no text, no watermark"
    print(f"Prompt: {prompt_text[:50]}...")

    # 💥 魔法 2：反向提示词 (Negative Prompt) 黑名单！
    anti_text_negative_prompt = "text, watermark, signature, letters, writing, words, typography, logo, brand name, garbled characters, subtitles"

    # 推理生成 
    try:
        generated_image = pipe(
            prompt=prompt_text,
            negative_prompt=anti_text_negative_prompt, # 传入黑名单
            edit_image=edit_image_list, 
            seed=1,
            num_inference_steps=40,
            height=1152,
            width=896,
            edit_image_auto_resize=True,
            zero_cond_t=True, 
        )
        
        # 1. 保存原始推理结果 (frame_0.jpg, frame_1.jpg ...)
        my_img_path = os.path.join(RAW_OUTPUT_DIR, f"frame_{i}.jpg")
        generated_image.save(my_img_path)
        print(f"第 {i} 帧已保存至: {my_img_path}")
        
        # 2. 拷贝 Nano 的原始图，重命名为 p1_nano_1.png ... (注意序号 i+1)
        nano_img_path = os.path.join(found_target_dir, target_file)
        if os.path.exists(nano_img_path):
            shutil.copy(nano_img_path, os.path.join(REPORT_DIR, f"p1_nano_{i+1}.png"))

        # 3. 拷贝你刚生成的图，重命名为 p1_my_1.png ... (注意序号 i+1)
        if os.path.exists(my_img_path):
            shutil.copy(my_img_path, os.path.join(REPORT_DIR, f"p1_my_{i+1}.png"))

        # 4. 写入 wandb 对比表格
        nano_wandb = wandb.Image(nano_img_path) if os.path.exists(nano_img_path) else None
        result_table.add_data(i + 1, prompt_text, nano_wandb, wandb.Image(my_img_path))
            
    # 【就是这里！】你刚才不小心丢掉了这个 except，导致 Python 疯了
    except Exception as e:
        print(f"  [推理失败] 帧: {target_file}, 报错: {e}")

# ==========================================
# 6. 使用终端命令 ZIP 打包
# ==========================================
print("\n开始自动整理推理结果...")
print(f"整理完毕！所有重命名好的展示图都在: {REPORT_DIR}")

folder_name = os.path.basename(REPORT_DIR)
zip_file_path = f"{REPORT_DIR}.zip" 

try:
    print(f"正在把 {folder_name} 文件夹压缩成 ZIP 包...")
    # 切换到父目录然后使用系统 zip 命令打包
    os.system(f"cd {OUTPUT_BASE_DIR} && zip -r {folder_name}.zip {folder_name}/")
    print("==================================================")
    print(f"🎉 任务完美结束！")
    print(f"📦 压缩包下载路径: {zip_file_path}")
    print("==================================================")
except Exception as e:
    print(f"压缩失败，请手动在终端执行 zip 命令。报错：{e}")

# ==========================================
# 7. 上传 wandb 表格并结束 run
# ==========================================
wandb.log({"references": ref_table, "results": result_table})
print(f"\n🔗 wandb 链接: {wandb.run.url}")
wandb.finish()