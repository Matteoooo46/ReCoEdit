import os
import argparse

# ==========================================
# 1. 绝对防御：在任何库加载前先设置环境
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, required=True, help="使用的物理显卡ID")
parser.add_argument("--chunk_index", type=int, required=True, help="当前分配的数据块序号")
parser.add_argument("--total_chunks", type=int, required=True, help="总共分了多少块")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 2. 现在可以安全导包了
# ==========================================
import json
import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# ==========================================
# 3. 基础路径配置
# ==========================================
ROOT_DIR = "/data/phd/lijiahui/data"
OUTPUT_JSON_PATH = f"/data/phd/kousiqi/zhitao/metadata_all_products_autoregressive_chunk_{args.chunk_index}.json"
GEN_OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_generated_autoregressive_frames"
os.makedirs(GEN_OUTPUT_BASE_DIR, exist_ok=True)

# ==========================================
# 4. 加载模型 (💥 终极内存组装 + 动态显存管家)
# ==========================================
print(f"🚀 进程 {args.chunk_index} 正在使用 GPU {args.gpu_id} 加载模型，请稍候...")

# 💥 必须用你自己的 5 个文件的 Edit 模型！(绝不能用 9 个文件的 Base 模型)
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

# 💥 官方黑科技 1：定义全生命周期的显存流转规则
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
        # 💥 官方黑科技 2：将规则批量注入到你本地的 5 份文件模型中
        ModelConfig(path=MODEL_CONFIG_PATHS["dit"], **vram_config),
        ModelConfig(path=MODEL_CONFIG_PATHS["text_encoder"], **vram_config),
        ModelConfig(path=MODEL_CONFIG_PATHS["vae"], **vram_config)
    ],
    tokenizer_config=None,
    # 你的图生图任务必须要 processor，同样注入规则
    processor_config=ModelConfig(path=MODEL_CONFIG_PATHS["processor"], **vram_config),
    
    # 💥 官方黑科技 3：动态获取真实剩余显存，并留出 0.5G 余量！
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2
)

print("✅ 模型按官方最新规范加载完毕，原生动态卸载模式启动！准备狂飙生图！")
# ==========================================
# 5. 扫描与切片分发
# ==========================================
workspace_paths = []
with open("failed_workspaces.txt", "r", encoding="utf-8") as f:
    for line in f:
        path = line.strip()
        if path: 
            workspace_paths.append(path)

workspace_paths.sort()
total_workspaces = len(workspace_paths)
chunk_size = total_workspaces // args.total_chunks
start_idx = args.chunk_index * chunk_size
end_idx = (args.chunk_index + 1) * chunk_size if args.chunk_index != args.total_chunks - 1 else total_workspaces

my_assigned_workspaces = workspace_paths[start_idx:end_idx]
print(f"✅ 进程 {args.chunk_index}/{args.total_chunks} 启动，负责处理 {len(my_assigned_workspaces)} 个商品。")

# ==========================================
# 6. 核心逻辑 (自回归 + Teacher Forcing)
# ==========================================
project_count = 0
all_data = []
current_part = 1
SAVE_INTERVAL = 1000

for workspace_path in my_assigned_workspaces:
    workspace_name = os.path.basename(workspace_path)
    
    found_character_dir = None
    found_reference_dir = None
    found_target_dir = None
    
    for root, dirs, files in os.walk(workspace_path):
        if "character_refs" in dirs:
            found_character_dir = os.path.join(root, "character_refs")
        if "reference_imgs" in dirs:
            found_reference_dir = os.path.join(root, "reference_imgs")
        if "augmented_generated_images_v8_v6_context_pro" in dirs:
            found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")
            
    if not found_character_dir or not found_reference_dir or not found_target_dir:
        continue

    char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg'))])
    ref_imgs = sorted([img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg'))])
    
    if not char_imgs or not ref_imgs:
        continue
        
    character_img_path = os.path.join(found_character_dir, char_imgs[0])
    reference_img_path = os.path.join(found_reference_dir, ref_imgs[0])

    target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]
    def get_frame_num(filename):
        try: return int(filename.split('_')[0])
        except: return 999999
    target_files.sort(key=get_frame_num)

    workspace_gen_dir = os.path.join(GEN_OUTPUT_BASE_DIR, workspace_name)
    os.makedirs(workspace_gen_dir, exist_ok=True)

    prev_generated_img_path = None
    prev_target_img_path = None
    has_valid_frame = False
    
    print(f"🎬 开始处理: {workspace_name}")
    
    for i, target_file in enumerate(target_files):
        target_img_path = os.path.join(found_target_dir, target_file)
        
        json_file = target_file.replace(".png", ".json")
        json_path = os.path.join(found_target_dir, json_file)
        if not os.path.exists(json_path): continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                prompt_data = json.load(jf)
                real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
                prompt_text = real_data.get("prompt", "")
                if not prompt_text:
                    prompt_text = real_data.get("product_info", "")
        except Exception:
            continue
            
        if not prompt_text: continue
                
        pil_edit_images = [
            Image.open(reference_img_path).convert("RGB"),
            Image.open(character_img_path).convert("RGB")
        ]
        json_edit_images = [reference_img_path, character_img_path]

        # ==========================================
        # 💥 1. 训练时 (写进 JSON 账本)：只加入上一帧推理生成的图像
        # ==========================================
        if prev_generated_img_path is not None:
            json_edit_images.append(prev_generated_img_path)

        # ==========================================
        # 💥 2. 推理时 (喂给 pipe 画图)：只加入上一帧 nano banana 的图像
        # ==========================================
        if prev_target_img_path is not None:
            pil_edit_images.append(Image.open(prev_target_img_path).convert("RGB"))
            
        try:
            generated_image = pipe(
                prompt=prompt_text,
                edit_image=pil_edit_images,
                num_inference_steps=20,
                seed=42
                # use_image_prompts=True
            )
            
            current_generated_img_path = os.path.join(workspace_gen_dir, f"gen_frame_{i}.jpg")
            generated_image.save(current_generated_img_path)
            
            json_line = {
                "image": target_img_path,
                "edit_image": json_edit_images,
                "prompt": prompt_text
            }
            all_data.append(json_line)
            has_valid_frame = True
            
            prev_generated_img_path = current_generated_img_path
            prev_target_img_path = target_img_path
            
        except Exception as e:
            print(f"❌ [推理或记录失败] 文件: {target_file}, 报错: {e}")
            continue

    if has_valid_frame:
        project_count += 1

    if len(all_data) >= SAVE_INTERVAL:
        part_path = OUTPUT_JSON_PATH.replace(".json", f"_part{current_part}.json")
        with open(part_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        all_data = []
        current_part += 1

if len(all_data) > 0:
    part_path = OUTPUT_JSON_PATH.replace(".json", f"_part{current_part}.json")
    with open(part_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"🎉 进程 {args.chunk_index} 跑通啦！成功处理 {project_count} 个 projects。")