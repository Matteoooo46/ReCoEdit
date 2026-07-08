from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import base64
import torch
import os
import json
import shutil
import io
import requests
import time

# ======================================================================
# 💥 1. 核心控制台配置
# ======================================================================
TARGET_GPU_ID = "4"  
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID

# 工作区路径配置
TARGET_WORKSPACE_DIR = "/data/phd/lijiahui/data/batch_no_voice_gen_shipin/workspace_item_23397479040558"
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"

product_name = os.path.basename(TARGET_WORKSPACE_DIR)

RAW_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw_gemini")
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_rewrite_gemini_expanded")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_optimized_prompts_gemini.json")

os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

PROMPT_OPTIMIZATION_TEMPLATE = """你是一个专业的 AI 绘画提示词（Prompt）扩写与视觉优化专家。
我会为你提供多张【参考图片】（包含产品细节和人物特征）以及一段简短的【基础提示词】。
请你仔细观察参考图片，并结合基础提示词的核心意图，将其扩写为一段极其丰富、具备电影感和高质感的最终生图提示词。

【撰写维度与细节要求】（请在扩写时严格包含以下元素）：
1. 核心不可变：严格保留【基础提示词】中的核心动作、场景和意图，绝不能偏离原本的故事线。
2. 主体描述：根据参考图，精准补充主体的外观特征、服装、材质、姿态以及其在画面中的位置关系。
3. 场景描述：明确主体所处环境，包括背景、前景、空间结构、相关道具以及光线环境。
4. 风格/视觉特征：补充专业的画面修饰词（如写实、广告质感、电影感、未来感、极简、工业风、复古等）。
5. 镜头语言：说明景别（如特写、中近景、远景）、视角（俯视、平视、仰视）以及构图方式（居中、三分法等）。
6. 氛围词与细节修饰：精准提取画面传达的情绪与氛围，并补充可见的微观细节（如材质纹理、高光、反射、景深、边缘虚化、烟雾、光晕等）。

【参考示例】（你的输出必须达到与下方示例相同的细节密度、画面感与行文连贯度）：
"画面中央，一个由纤细黑色线条勾勒出的极简风格泰迪熊轮廓悬浮于一座洁白立方体展台之上，其造型高度抽象，仅以流畅的曲线勾勒出头部、圆耳、躯干与环形四肢，无任何面部细节或填充色彩，材质看似光滑的金属或亚克力，呈现出轻盈而现代的雕塑感。背景为一间明亮宽敞的艺术画廊，暖调自然光从右侧斜射入内，在米白色墙壁与浅木色地板上投下柔和的光影，远处可见模糊的白色展架与挂画，营造出纵深空间感；前景的展台棱角分明，表面哑光质感，与主体形成鲜明对比。整体视觉风格干净、极简且富有设计感，带有强烈的当代艺术装置气息与广告级的精致质感。镜头采用中景平视构图，主体居中突出，利用浅景深使背景虚化，强化了泰迪熊符号的视觉焦点地位。氛围宁静、空灵而略带哲思，仿佛在探讨童年记忆与现代美学之间的抽象对话。微观细节上，泰迪熊边缘锐利无毛刺，展台顶部有微妙的高光反射，背景墙面纹理隐约可见，光影过渡自然柔和，增强了画面的空间层次与真实触感。"

【格式要求】：
请将以上元素有机融合，写成一段连贯、丰富且结构清晰的自然语言描述。不要输出"主体："、"场景："等任何维度标题，也不要包含“好的”、“这是优化后的提示词”等解释性废话，直接输出最终的整段提示词文本。

---
【基础提示词】：{original_prompt}
"""

# ======================================================================
# 🎨 2. 初始化 Qwen-Image 生图模型 (DiffSynth 画师)
# ======================================================================
print("正在加载 DiT 模型，请稍候...")
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
    vram_limit=75
)

lora_path = "/data/phd/kousiqi/zhitao/lora_all_products_rewrite_expanded_resume_149000/step-34000.safetensors"
pipe.load_lora(pipe.dit, lora_path)
print("✅ 模型加载完毕！")

# ======================================================================
# 🛠️ 3. 辅助函数：内存级图片压缩与 Base64 编码 (防 4MB 撑爆)
# ======================================================================
def get_compressed_base64_image(image_path, max_size=512):
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ 图片读取或压缩失败 {image_path}: {e}")
        return None

# ======================================================================
# 🛠️ 4. 定义内网 Gemini 扩写推理函数
# ======================================================================
def get_optimized_prompt_gemini(original_prompt, image_paths):
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-pro:generateContent"
    API_KEY = "REDACTED" 
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    
    final_text = PROMPT_OPTIMIZATION_TEMPLATE.replace("{original_prompt}", original_prompt)
    parts = [{"text": final_text}]
    
    if isinstance(image_paths, str):
        image_paths = [image_paths]
        
    for img_path in image_paths:
        img_base64 = get_compressed_base64_image(img_path)
        if img_base64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_base64}})
            
    payload = {"contents": [{"parts": parts}]}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), proxies={"http": None, "https": None}, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                for part in response_data["candidates"][0].get("content", {}).get("parts", []):
                    if "text" in part: 
                        raw_answer = part["text"].strip()
                        import re
                        clean_caption = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL)
                        return clean_caption.strip()
            
            print(f"      [Warning] API 返回格式异常或为空 (尝试 {attempt + 1})")
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      [Warning] API 调用失败 (尝试 {attempt + 1})，等 2 秒重试... 报错: {e}")
                time.sleep(2)
            else:
                print(f"      ❌ API 请求彻底失败: {e}")
                return original_prompt 
                
    return original_prompt


# ======================================================================
# 🚀 5. 核心处理流程
# ======================================================================
print(f"\n📂 开始处理产品: {product_name}")

# 自适应扫描文件夹命名
found_character_dir = None
found_reference_dir = None
found_target_dir = None

for root, dirs, files in os.walk(TARGET_WORKSPACE_DIR):
    if "augmented_generated_images" in dirs: 
        found_target_dir = os.path.join(root, "augmented_generated_images")
    elif "augmented_generated_images_v8_v6_context_pro" in dirs: 
        found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")
        
    if "reference_imgs" in dirs: 
        found_reference_dir = os.path.join(root, "reference_imgs")
    if "character_refs" in dirs: 
        found_character_dir = os.path.join(root, "character_refs")

if not found_target_dir or not found_reference_dir:
    raise ValueError("❌ 找不到目标图片文件夹或参考图文件夹，请检查路径！")

vl_inference_image_paths = []
edit_image_list = []

# 1. 载入产品参考图 (必须有)
ref_imgs = sorted([img for img in os.listdir(found_reference_dir) if img.endswith(('.png', '.jpg'))])
for idx, ref_img in enumerate(ref_imgs):
    ref_path = os.path.join(found_reference_dir, ref_img)
    vl_inference_image_paths.append(ref_path)
    edit_image_list.append(Image.open(ref_path).convert("RGB"))
    shutil.copy(ref_path, os.path.join(REPORT_DIR, f"p1_ref_{idx}.png"))

# 2. 动态载入人物图 (如果文件夹存在且里面有图，才加载)
has_character = False
if found_character_dir:
    char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.endswith(('.png', '.jpg'))])
    if char_imgs:
        char_img_path = os.path.join(found_character_dir, char_imgs[0])
        vl_inference_image_paths.append(char_img_path)
        edit_image_list.append(Image.open(char_img_path).convert("RGB"))
        shutil.copy(char_img_path, os.path.join(REPORT_DIR, "p1_char.png"))
        has_character = True

print(f"✅ 载入完成，包含 {len(ref_imgs)} 张产品图，{'1 张人物图' if has_character else '无人物图'}。共 {len(vl_inference_image_paths)} 张参考图用于推理和生图。\n")

target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]
target_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 999999)

all_optimized_data = [] # 记录账本
viewer_frames = []      # 收集供 HTML 页面的数据

for i, target_file in enumerate(target_files):
    json_file = target_file.replace(".png", ".json")
    json_path = os.path.join(found_target_dir, json_file)
    
    if not os.path.exists(json_path): continue 
        
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            prompt_data = json.load(jf)
            real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
            original_prompt = real_data.get("prompt", "") or real_data.get("product_info", "")
    except Exception as e:
        continue
        
    if not original_prompt: continue

    print(f"---> 正在处理第 {i} 帧...")
    print(f"原 Prompt: {original_prompt}")

    # 第一步：利用 Gemini 大模型进行 Prompt 扩写推理
    try:
        optimized_prompt = get_optimized_prompt_gemini(original_prompt, vl_inference_image_paths)
        print(f"【优化后】 {optimized_prompt[:80]}...")
    except Exception as e:
        print(f"  ❌ Gemini 推理失败: {e}")
        continue
    
    all_optimized_data.append({
        "frame_index": i,
        "target_file": target_file,
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt
    })

    # 第二步：利用 Qwen-Image 扩散模型画图
    try:
        generated_image = pipe(
            optimized_prompt,
            edit_image=edit_image_list, 
            seed=1,
            num_inference_steps=40,
            height=1152,
            width=896,
            edit_image_auto_resize=True,
            zero_cond_t=True, 
        )
        
        my_img_path = os.path.join(RAW_OUTPUT_DIR, f"frame_{i}.jpg")
        generated_image.save(my_img_path)
        print(f"   ✅ 第 {i} 帧已画完并保存至: {my_img_path}")
        
        nano_name = f"p1_nano_{i+1}.png"
        my_name = f"p1_my_{i+1}.png"
        
        nano_img_path = os.path.join(found_target_dir, target_file)
        if os.path.exists(nano_img_path):
            shutil.copy(nano_img_path, os.path.join(REPORT_DIR, nano_name))
        if os.path.exists(my_img_path):
            shutil.copy(my_img_path, os.path.join(REPORT_DIR, my_name))
            
        # 收集 HTML 所需的关联数据
        viewer_frames.append({
            "prompt": original_prompt,
            "rewrite_prompt": optimized_prompt,
            "nano_path": nano_name,
            "my_path": my_name
        })
            
    except Exception as e:
        print(f"  ❌ 扩散模型画图失败 帧: {target_file}, 报错: {e}")

# ======================================================================
# 📦 6. 原生打包与账本存储、自动生成 HTML
# ======================================================================
print("\n==================================================")
print("💾 正在保存优化后的 Prompt JSON 账本...")
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_optimized_data, f, ensure_ascii=False, indent=4)
print(f"✅ JSON 账本已保存至: {OUTPUT_JSON_PATH}")

# ===========================
# 🌐 HTML 自动生成注入 (包含优化 Prompt 栏位)
# ===========================
print("📝 正在自动生成全景展示 HTML...")
html_data_payload = [{
    "product_name": product_name,
    "ref_path": "p1_ref_0.png",
    "char_path": "p1_char.png" if has_character else "", 
    "frames": viewer_frames
}]
json_string = json.dumps(html_data_payload, ensure_ascii=False)

# 内置带高亮颜色配置的 HTML 模板
html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{product_name} 生成对比报告</title>
    <style>
        :root {{ --bg-color: #f5f6f7; --card-bg: #ffffff; --text-main: #1f2329; --primary-color: #3370ff; --rewrite-color: #00b96b; }}
        body {{ font-family: -apple-system, sans-serif; background: var(--bg-color); color: var(--text-main); padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ border-bottom: 2px solid #dee0e3; margin-bottom: 20px; padding-bottom: 10px; }}
        .product-section {{ background: var(--card-bg); border-radius: 12px; padding: 24px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        .refs-container {{ display: flex; gap: 15px; margin-bottom: 20px; background: #fafafa; padding: 15px; border-radius: 8px; }}
        .refs-container img {{ height: 150px; border-radius: 6px; object-fit: cover; border: 1px solid #ddd; }}
        .frames-grid {{ display: flex; overflow-x: auto; gap: 20px; padding-bottom: 15px; }}
        .frame-column {{ min-width: 300px; flex: 0 0 300px; display: flex; flex-direction: column; gap: 15px; }}
        .prompt-text {{ background: #f0f4ff; border-left: 4px solid var(--primary-color); padding: 12px; font-size: 13px; border-radius: 4px; line-height: 1.5; }}
        .rewrite-text {{ background: #e8f9f0; border-left: 4px solid var(--rewrite-color); padding: 12px; font-size: 13px; border-radius: 4px; line-height: 1.5; color: #1e5c3e; }}
        .image-compare-box img {{ width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
        .image-compare-box h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #555; }}
    </style>
</head>
<body>
    <div class="container" id="app"></div>
    <script>
        const data = {json_string};
        const app = document.getElementById('app');
        
        data.forEach(product => {{
            let html = `
                <div class="product-section">
                    <div class="header"><h2>📦 ${{product.product_name}}</h2></div>
                    <div class="refs-container">
                        <div><p>参考图</p><img src="${{product.ref_path}}"></div>
                        ${{product.char_path ? `<div><p>人物图</p><img src="${{product.char_path}}"></div>` : ''}}
                    </div>
                    <div class="frames-grid">
            `;
            
            product.frames.forEach((frame, idx) => {{
                html += `
                    <div class="frame-column">
                        <h3>Frame ${{idx}}</h3>
                        <div class="prompt-text"><b>原 Prompt:</b><br>${{frame.prompt}}</div>
                        <div class="rewrite-text"><b>优化 Prompt:</b><br>${{frame.rewrite_prompt}}</div>
                        
                        <div class="image-compare-box">
                            <h4>Nano Banana:</h4>
                            <img src="${{frame.nano_path}}" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=无图'">
                        </div>
                        <div class="image-compare-box">
                            <h4>My Model:</h4>
                            <img src="${{frame.my_path}}" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=无图'">
                        </div>
                    </div>
                `;
            }});
            html += `</div></div>`;
            app.innerHTML += html;
        }});
    </script>
</body>
</html>
"""
html_file_path = os.path.join(REPORT_DIR, f"{product_name}_viewer.html")
with open(html_file_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"✅ HTML 展示页已生成至: {html_file_path}")
print("==================================================")

print("\n开始自动打包结果...")
folder_name = os.path.basename(REPORT_DIR)
zip_file_path = f"{REPORT_DIR}.zip" 

try:
    archive_base = os.path.join(OUTPUT_BASE_DIR, folder_name)
    shutil.make_archive(archive_base, 'zip', archive_base)
    print("==================================================")
    print(f"🎉 任务完美结束！")
    print(f"📦 压缩包下载路径: {zip_file_path}")
    print("==================================================")
except Exception as e:
    print(f"压缩失败，报错：{e}")