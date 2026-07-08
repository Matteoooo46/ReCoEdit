from PIL import Image
import os
import json
import base64
import random
import time
from io import BytesIO
from openai import OpenAI

# ==========================================
# 🌟 Qwen-VL 视觉打标大脑初始化
# ==========================================
# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="http://wanqing.internal/api/gateway/v1/endpoints",
    api_key="ta5zooh2lx89jxi3tfbvjcj36kdibr89ef95",
    timeout=120.0,
    max_retries=3,
)

MODELS = [
    "ep-e650ybb-1765370705290919970", "ep-rhnpn7-1765370642260722106", "ep-n7to58-1765370748164703856",
    "ep-lu2m2z-1765370777929428419", "ep-6o7c47-1765370937955493211", "ep-iastx4-1765370964224510371",
    "ep-4lgyii-1765370993248863264", "ep-d6djrs-1765371024830814619", "ep-9ez3xy-1765375805160294747",
    "ep-lkgnu7-1765375768345785170", "ep-0smb53-1765375834030713608", "ep-xffpxq-1765375858948778299",
    "ep-73icey-1765375886534547850", "ep-ar4rzz-1765375919526030658", "ep-eln38m-1765375940772709889",
    "ep-j8rghf-1765375963749909186",
]

CAPTION_PROMPT = """你是一个专业的图像视觉描述专家。请仔细观察提供的图片，并严格按照以下维度和顺序，输出一段高质量的图像描述（Caption）：
1. 主体描述：明确谁/什么是画面主体，写清外观特征、服装、材质、姿态以及其在画面中的位置关系。
2. 场景描述：明确主体所处环境，包括背景、前景、空间结构、相关道具以及光线环境。
3. 风格/视觉特征：总结真实画面的视觉特征（如写实、广告质感、电影感、未来感、极简、工业风、复古等）。
4. 镜头语言：说明景别、视角以及构图方式。
5. 氛围词：精准提取画面传达的情绪与氛围。
6. 细节修饰：补充可见的微观细节，如材质纹理、高光、反射、景深等。
要求：请将以上元素有机融合，写成一段连贯、丰富且结构清晰的自然语言描述。不要输出维度标题，直接输出最终的整段描述文本即可。"""

def pil_image_to_base64(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_image_text}"

def generate_image_caption(image_path, retry_count=3):
    expanded_path = os.path.expanduser(image_path)
    if not os.path.exists(expanded_path):
        return f"Error: Image not found at {expanded_path}"
    try:
        with Image.open(expanded_path) as pil_img:
            pil_img.thumbnail((1024, 1024))
            image_base64 = pil_image_to_base64(pil_img)
    except Exception as e:
        return f"Error: Failed to process image {image_path}. {e}"

    for attempt in range(retry_count):
        try:
            model_id = random.choice(MODELS)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_base64}},
                        ],
                    },
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2
                print(f"      [Warning] Qwen-VL 调用失败 (尝试 {attempt + 1})，等待 {wait_time}s 重试: {e}")
                time.sleep(wait_time)
            else:
                return f"Error: All retry attempts failed. {e}"

# ==========================================
# 1. 基础路径配置
# ==========================================
TARGET_WORKSPACE_DIR = "/data/phd/lijiahui/data/batch_no_voice_gen_shipin/workspace_item_21825264046857"
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_optimized_prompts"

product_name = os.path.basename(TARGET_WORKSPACE_DIR)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 最终生成的 JSON 账本存放路径
OUTPUT_JSON_PATH = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_optimized_prompts.json")

# ==========================================
# 2. 定位核心文件夹
# ==========================================
print(f"\n🚀 开始处理产品，目标生成优化版 JSON 账本: {product_name}")

found_target_dir = None
for root, dirs, files in os.walk(TARGET_WORKSPACE_DIR):
    if "augmented_generated_images_v8_v6_context_pro" in dirs: 
        found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")
        break
        
if not found_target_dir:
    raise ValueError(f"❌ 未能在 {TARGET_WORKSPACE_DIR} 中找到 augmented_generated_images_v8_v6_context_pro 文件夹！")

# ==========================================
# 3. 获取帧列表并排序
# ==========================================
target_files = [f for f in os.listdir(found_target_dir) if f.endswith(".png")]

def get_frame_num(filename):
    try: return int(filename.split('_')[0])
    except: return 999999
        
target_files.sort(key=get_frame_num)
print(f"共发现 {len(target_files)} 帧图片数据需要提词。")

# ==========================================
# 4. 开始提取与优化 Prompt
# ==========================================
all_optimized_data = []

for i, target_file in enumerate(target_files):
    json_file = target_file.replace(".png", ".json")
    json_path = os.path.join(found_target_dir, json_file)
    nano_img_path = os.path.join(found_target_dir, target_file) # Nano Banana 生成的参考帧
    
    if not os.path.exists(json_path): continue 
        
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            prompt_data = json.load(jf)
            real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
            original_prompt = real_data.get("prompt", "") or real_data.get("product_info", "")
    except Exception as e:
        print(f"  [读取JSON失败] 文件: {json_file}, 报错: {e}")
        continue
        
    if not original_prompt: continue

    print(f"\n---> 正在处理第 {i} 帧: {target_file}")
    
    optimized_prompt = original_prompt # 默认使用旧的，防出错
    
    # 召唤 Qwen-VL 反推高精度 Prompt
    if os.path.exists(nano_img_path):
        print("   🔄 正在召唤 Qwen-VL 反推高精度 Prompt...")
        vl_caption = generate_image_caption(nano_img_path)
        
        if vl_caption and not vl_caption.startswith("Error:"):
            print(f"   ✨ 【原生 Prompt】: {original_prompt[:50]}...")
            print(f"   ✨ 【Qwen-VL 视觉重构 Prompt】: {vl_caption[:100]}...")
            optimized_prompt = vl_caption
        else:
            print(f"   ⚠️ Qwen-VL 提取失败，退回使用原始 JSON Prompt。报错信息: {vl_caption}")
            
    # 将结果装载进字典
    json_line = {
        "frame_index": i,
        "image_file": nano_img_path,
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt
    }
    all_optimized_data.append(json_line)

# ==========================================
# 5. 导出统一 JSON 账本
# ==========================================
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_optimized_data, f, ensure_ascii=False, indent=4)

print("\n==================================================")
print(f"🎉 任务完美结束！")
print(f"📦 共成功处理 {len(all_optimized_data)} 条数据。")
print(f"💾 优化后的 Prompt 账本已保存至: {OUTPUT_JSON_PATH}")
print("==================================================")