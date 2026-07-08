import os
import json
import torch
import argparse
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# ==========================================
# 1. 接收启动脚本传来的“工号”和“显卡号”
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, required=True, help="使用的物理显卡ID")
parser.add_argument("--chunk_index", type=int, required=True, help="当前分配的数据块序号")
parser.add_argument("--total_chunks", type=int, required=True, help="总共分了多少块")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# ==========================================
# 2. 基础路径配置 (💥 修改为你的新数据集路径)
# ==========================================
ROOT_DIR = "/data/phd/yaozhengjian/zjYao_Datasets/Doubao/nanoBanana_images"
OUTPUT_JSON_PATH = f"/data/phd/yaozhengjian/zjYao_Datasets/Doubao/metadata_doubao_autoregressive_chunk_{args.chunk_index}.json"
GEN_OUTPUT_BASE_DIR = "/data/phd/yaozhengjian/zjYao_Datasets/Doubao/qwen_generated_autoregressive_frames"
os.makedirs(GEN_OUTPUT_BASE_DIR, exist_ok=True)

# ==========================================
# 3. 初始化 Qwen 模型
# ==========================================
print(f"🚀 进程 {args.chunk_index} 正在使用 GPU {args.gpu_id} 加载模型，请稍候...")
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
    tokenizer_config=None,
    processor_config=ModelConfig(path=MODEL_CONFIG_PATHS["processor"]),
)

# ==========================================
# 4. 扫描目录并进行切片 (💥 全新寻址逻辑)
# ==========================================
workspace_paths = []
# 只要某个文件夹下同时包含 condition.jpg 和 json，就认为它是一个合法的任务包
for root_path, dirs, files in os.walk(ROOT_DIR):
    if "condition.jpg" in files and "scene_prompts_condition.json" in files:
        workspace_paths.append(root_path)

workspace_paths.sort() # 必须排序保证多卡切片一致

total_workspaces = len(workspace_paths)
chunk_size = total_workspaces // args.total_chunks
start_idx = args.chunk_index * chunk_size
end_idx = (args.chunk_index + 1) * chunk_size if args.chunk_index != args.total_chunks - 1 else total_workspaces

my_assigned_workspaces = workspace_paths[start_idx:end_idx]
print(f"总计找到 {total_workspaces} 个有效 Asset 文件夹。本进程分配到 {len(my_assigned_workspaces)} 个。")

# ==========================================
# 5. 核心逻辑：自回归边推理边造数据
# ==========================================
project_count = 0
all_data = [] 
current_part = 1  
SAVE_INTERVAL = 1000  

for workspace_path in my_assigned_workspaces:
    # 提取类似 category_x_asset_y 的名字用作保存文件夹
    path_parts = workspace_path.split(os.sep)
    try:
        # 尝试拼凑具有辨识度的名字，例如 "category_1_asset_10001"
        asset_name = f"{path_parts[-3]}_{path_parts[-2]}" 
    except:
        asset_name = os.path.basename(workspace_path)
    
    # 💥 读取唯一的全局参考图
    condition_img_path = os.path.join(workspace_path, "condition.jpg")
    
    # 💥 集中读取该分镜的所有 Prompt 数组
    json_path = os.path.join(workspace_path, "scene_prompts_condition.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            prompts_list = json.load(jf)
    except Exception as e:
        print(f"⚠️ [读取JSON失败] 跳过 {asset_name}: {e}")
        continue
        
    # 💥 扫描该目录下的所有目标分镜图 (output_img_condition_prompt_x.png)
    target_files = [f for f in os.listdir(workspace_path) if f.startswith("output_img_condition_prompt_") and f.endswith(".png")]
    def get_frame_num(filename):
        try: return int(filename.split('_')[-1].split('.')[0])
        except: return 999999
    target_files.sort(key=get_frame_num)

    # 检查图片数量和提示词数量是否匹配，以最小的为准防止越界报错
    num_frames = min(len(target_files), len(prompts_list))
    if num_frames == 0: continue

    workspace_gen_dir = os.path.join(GEN_OUTPUT_BASE_DIR, asset_name)
    os.makedirs(workspace_gen_dir, exist_ok=True)

    # 💥 修改 1：新增 prev_target_img_path 变量
    prev_generated_img_path = None 
    prev_target_img_path = None 
    has_valid_frame = False
    
    for i in range(num_frames):
        target_file = target_files[i]
        target_img_path = os.path.join(workspace_path, target_file)
        prompt_text = prompts_list[i] 
        
        # --- 基础条件图 (单张) ---
        pil_edit_images = [Image.open(condition_img_path).convert("RGB")]
        json_edit_images = [condition_img_path]
        
        # --- 加上一帧的 Qwen 生成图 (推理和 JSON 都要) ---
        if prev_generated_img_path is not None:
            pil_edit_images.append(Image.open(prev_generated_img_path).convert("RGB"))
            json_edit_images.append(prev_generated_img_path)

        # ==========================================
        # 💥 修改 2：加上一帧的 Nano Banana 原图 (只给推理用！)
        # ==========================================
        if prev_target_img_path is not None:
            pil_edit_images.append(Image.open(prev_target_img_path).convert("RGB"))
            
        try:
            generated_image = pipe(
                prompt=prompt_text,
                edit_image=pil_edit_images, # 包含 3 张图 (全局参考, 上帧Qwen, 上帧Nano)
                num_inference_steps=20,     
                seed=42,                    
                use_image_prompts=True
            )
            
            current_generated_img_path = os.path.join(workspace_gen_dir, f"gen_frame_{i}.jpg")
            generated_image.save(current_generated_img_path)
            
            # 写入 JSON 时，json_edit_images 里只有 2 张图 (全局参考, 上帧Qwen)
            json_line = {
                "image": target_img_path,          
                "edit_image": json_edit_images,    
                "prompt": prompt_text                                 
            }
            all_data.append(json_line)
            has_valid_frame = True
            
            # 💥 修改 3：同步更新两个“上一帧”变量
            prev_generated_img_path = current_generated_img_path
            prev_target_img_path = target_img_path
            
        except Exception as e:
            print(f"❌ [推理或记录失败] 帧: {i}, 报错: {e}")
            continue

    if has_valid_frame:
        project_count += 1
        
    # 阶段性分段保存 (冲水阀)
    if len(all_data) >= SAVE_INTERVAL:
        part_path = OUTPUT_JSON_PATH.replace(".json", f"_part{current_part}.json")
        with open(part_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            
        print(f"📦 [阶段保存] 进程 {args.chunk_index} 成功保存 {len(all_data)} 条数据至: {part_path}")
        all_data = [] 
        current_part += 1

# ==========================================
# 6. 导出尾部残余数据
# ==========================================
if len(all_data) > 0:
    part_path = OUTPUT_JSON_PATH.replace(".json", f"_part{current_part}.json")
    with open(part_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"📜 尾部剩余的 {len(all_data)} 条数据已保存至: {part_path}")

print(f"🎉 进程 {args.chunk_index} 全部跑通！成功处理 {project_count} 个 Asset。")