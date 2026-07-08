import os
import json
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
import time
import random
import argparse
import multiprocessing as mp

# ==========================================
# 🌟 Qwen-VL 视觉打标大脑相关配置与函数
# ==========================================
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
    if image.mode != "RGB": image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return f"data:image;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def generate_image_caption(image_path, retry_count=3):
    expanded_path = os.path.expanduser(image_path)
    if not os.path.exists(expanded_path): return f"Error: Image not found at {expanded_path}"
    
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
                messages=[{"role": "user", "content": [{"type": "text", "text": CAPTION_PROMPT}, {"type": "image_url", "image_url": {"url": image_base64}}]}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1: time.sleep((attempt + 1) * 2)
            else: return f"Error: All retry attempts failed. {e}"

# ==========================================
# 🚀 核心：单进程提词函数 (支持断点续传 & 分块保存)
# ==========================================
def run_optimized_on_single_process(global_rank, total_world_size, original_json_path, temp_output_dir):
    print(f"🚀 [Global Rank {global_rank}] 进程启动...")
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        
    # 💥 全局大饼切分：这台机器的这个进程，只拿属于自己的那一块
    my_assigned_data = all_data[global_rank::total_world_size]
    assigned_count = len(my_assigned_data)
    print(f"📦 [Global Rank {global_rank}] 我被分到了 {assigned_count} 条数据。")
    
    # 断点续传核心逻辑
    my_temp_output_json = os.path.join(temp_output_dir, f"optimized_rank_{global_rank}.json")
    my_clean_workspace_data = [] 
    processed_image_paths = set() 
    
    if os.path.exists(my_temp_output_json):
        try:
            with open(my_temp_output_json, 'r', encoding='utf-8') as f:
                my_clean_workspace_data = json.load(f)
            for item in my_clean_workspace_data:
                processed_image_paths.add(item.get("image"))
            print(f"🔄 [Global Rank {global_rank}] 发现断点！已恢复了 {len(my_clean_workspace_data)} 条数据。")
        except Exception as e:
            print(f"⚠️ [Global Rank {global_rank}] 历史文件损坏，将重头开始: {e}")
            my_clean_workspace_data = []

    processed_count = 0
    skipped_count = 0

    for idx, item in enumerate(my_assigned_data):
        img_path = item.get("image")
        
        # 已处理或空路径直接跳过
        if not img_path or img_path in processed_image_paths:
            if not img_path: skipped_count += 1
            continue

        if idx % 50 == 0:
            print(f"   ⌛ [Global Rank {global_rank}] 进度: {idx}/{assigned_count}")
        
        if os.path.exists(img_path):
            vl_caption = generate_image_caption(img_path)
            if vl_caption and not vl_caption.startswith("Error:"):
                item["original_prompt"] = item.get("prompt") 
                item["prompt"] = vl_caption
                my_clean_workspace_data.append(item)
                processed_count += 1
                processed_image_paths.add(img_path)
            else:
                my_clean_workspace_data.append(item) 
                skipped_count += 1
                processed_image_paths.add(img_path)
        else:
            skipped_count += 1

        # 💥 核心：每处理 2000 条，立即强制保存一次！防爆机！
        if len(my_clean_workspace_data) > 0 and len(my_clean_workspace_data) % 2000 == 0:
            with open(my_temp_output_json, 'w', encoding='utf-8') as f:
                json.dump(my_clean_workspace_data, f, ensure_ascii=False, indent=2)
            print(f"💾 [Global Rank {global_rank}] 已达到 2000 条，完成阶段性存盘！")

    # 循环彻底结束后，最后再存一次底
    with open(my_temp_output_json, 'w', encoding='utf-8') as f:
        json.dump(my_clean_workspace_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ [Global Rank {global_rank}] 全部任务完成！")

# ==========================================
# 📦 结果聚合
# ==========================================
def reconstruct_final_json(total_world_size, temp_output_dir, output_final_json):
    print(f"\n==========================================")
    print(f"🎉 开始归位与重建 32 个节点的大业...")
    
    final_optimized_data = []
    
    for i in range(total_world_size):
        temp_json_path = os.path.join(temp_output_dir, f"optimized_rank_{i}.json")
        if not os.path.exists(temp_json_path):
            print(f"❌ 警告：未能在 {temp_json_path} 找到结果文件。(该节点可能还没跑完)")
            continue
            
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            temp_json_data = json.load(f)
            final_optimized_data.extend(temp_json_data)
            
    with open(output_final_json, 'w', encoding='utf-8') as f:
        json.dump(final_optimized_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 重建完成！共融合了 {len(final_optimized_data)} 条数据。")
    print(f"💾 终极数据集保存在: {output_final_json}")


if __name__ == "__main__":
    # ==========================================
    # 接收外部超级集群指令
    # ==========================================
    parser = argparse.ArgumentParser(description="分布式打标集群版")
    parser.add_argument("--node_rank", type=int, required=True, help="当前机器编号 (0 到 3)")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="每台机器的 GPU/进程数 (默认为 8)")
    parser.add_argument("--total_nodes", type=int, default=4, help="集群总机器数 (默认为 4)")
    parser.add_argument("--merge_only", action="store_true", help="是否仅执行最终合并逻辑")
    args = parser.parse_args()

    ORIGINAL_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_fuke_weigai.json"
    OUTPUT_FINAL_JSON = "/data/phd/kousiqi/zhitao/metadata_fuke_weigai_optimized.json"
    
    # 💥 极度重要：因为你要用 4 台机器，临时文件夹必须挂载在 4 台机器都能访问的共享存储(NAS)上！
    # 否则机器 A 的结果，机器 B 是合并不到的！(假设你的 /data 目录是互通的)
    TEMP_OUTPUT_DIR = "/data/phd/kousiqi/zhitao/hard_products_rescue_temp_cluster"
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    
    total_world_size = args.total_nodes * args.gpus_per_node # 4 * 8 = 32

    if args.merge_only:
        # 如果你加了 --merge_only 参数，脚本就什么都不跑，只负责把 32 个碎文件拼起来
        reconstruct_final_json(total_world_size, TEMP_OUTPUT_DIR, OUTPUT_FINAL_JSON)
    else:
        print(f"\n==========================================")
        print(f"🚀 集群点火！当前机器 Node: {args.node_rank}/{args.total_nodes - 1}")
        print(f"👥 全局总进程数: {total_world_size}")
        print("==========================================")
        
        mp.set_start_method('spawn') 
        processes = []
        
        # 启动本机上的 8 个进程
        for local_rank in range(args.gpus_per_node):
            # 计算全局唯一编号 (Global Rank)
            # 例如 node_rank=1 时，它的 8 个进程全局编号是 8, 9, 10, 11, 12, 13, 14, 15
            global_rank = (args.node_rank * args.gpus_per_node) + local_rank
            
            p = mp.Process(
                target=run_optimized_on_single_process, 
                args=(global_rank, total_world_size, ORIGINAL_JSON_PATH, TEMP_OUTPUT_DIR)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        print(f"🎉 当前机器 (Node {args.node_rank}) 的 8 个进程已全部执行完毕！")