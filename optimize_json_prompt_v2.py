import os
import json
import base64
import httpx
from openai import OpenAI
import time
import argparse
import multiprocessing as mp

# ==========================================
# 🌟 Qwen-VL 视觉打标大脑相关配置与函数
# ==========================================
QWEN_API_KEY = "EMPTY"
QWEN_API_BASE = "http://10.80.243.156:8080/v1"
QWEN_MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)
client._client.timeout = httpx.Timeout(360.0)

# ==========================================
# 📝 细节增强版 Caption 生成提示词
# ==========================================
CAPTION_PROMPT = """你是一个专业的电商图像视觉描述专家，专门为"图像编辑/商品广告生成"任务生成高保真目标图描述。本任务的最大挑战是**商品一致性**——你的描述必须详尽到让模型即使看不到商品参考图，也能凭描述复现出与参考图完全相同的商品。请仔细观察图片，输出一段连贯的中文画面描述。

【最高指导原则】
"商品本体描述"是本任务的绝对核心，必须占整段输出的 60%-70% 篇幅。其他维度（场景/镜头/风格/氛围）作为辅助上下文，简洁带过即可。商品本身每一个可观察的细节都必须被精确写出，不允许遗漏任何文字、logo、颜色或材质特征。

============================
A. 商品本体描述（核心，必须详尽穷举）
============================

1. 商品类别与整体形态
   - 商品类别（瓶装/罐装/盒装/管装/袋装/泵装/喷雾装/裸品等）；
   - 整体轮廓（圆柱/方柱/锥形/不规则/双层/带提手/带喷头等）；
   - 比例关系（高度/直径比、瓶身/瓶盖比例、各部件相对占比）；
   - 在画面中的姿态（正立/侧倒/倾斜角度，正面/侧面/45度朝向）。

2. 商品分部描述（自上而下，逐个部件展开）
   将商品拆解为各组成部分（如瓶盖、压头、瓶颈、瓶身上段/中段/下段、瓶底、外盒、内瓶、手柄、刷头），逐一描述每个部件的：
   - 形状与尺寸特征；
   - 颜色（具体色名 + 明度/饱和度倾向，如"哑光珠光白""深邃午夜蓝带紫调""暖调香槟金"），有多种颜色需分别说明；
   - 材质（金属/玻璃/塑料/纸质/陶瓷/木质等），并明确表面工艺（哑光/亮光/磨砂/拉丝/电镀/烫金/UV凸印/喷涂）；
   - 是否透明/半透明，透明时可见的内部内容物颜色与状态（液体/膏体/粉末/颗粒）。

3. 所有可见的文字、logo、图案（极其重要，必须精确转录）
   - **完整列出画面中商品上每一处可识别的文字内容**（品牌名、产品名、规格、容量、slogan、成分、警示语、批号、产地等），逐字转录，包括中文、英文、数字、符号；
   - 每处文字必须标注：
     · 文字内容（尽可能逐字精确，对中英混排要保留原顺序）；
     · 所在位置（瓶身正面中部/瓶颈环带/盒侧/盒顶/盒底/标签上方等）；
     · 字体风格（无衬线/衬线/手写/书法/印刷体）、字号相对大小（主标题/副标题/小字）、颜色、是否凸印/烫金/普通印刷；
   - 所有 logo 与图标的形状、颜色、位置、相对大小（占商品立面的比例）；
   - 装饰图案：条纹、色块、渐变、纹理、插画、二维码、防伪标、产品图示。

4. 表面细节与做工
   - 高光分布（位置、形状、强度）、镜面反射区域、漫反射区域；
   - 边缘工艺（尖锐边/圆角/倒角）、接缝/封口/拉环/吸管孔/泵头按压结构；
   - 是否有标签贴纸（边缘是否翘起、是否带胶印）、塑封膜反光、防伪封条。

5. 商品的物理状态
   - 是否打开/未打开、压头是否按下；
   - 是否有内容物外溢、滴落、起泡、雾气、冷凝水滴等动态特征；
   - 是否与其他物件接触/堆叠/重叠。

============================
B. 辅助上下文（简短，每项 1-2 句即可，合计不超过整段 30%）
============================

6. 场景与背景：背景颜色/材质（如纯色背景、大理石台面、木纹桌面、织物背景）、前景道具及其与商品的空间关系。
7. 镜头语言：景别（特写/近景/中景）、视角（正面/45度/侧面/俯视/仰视）、构图方式。
8. 光影氛围：主光方向与色温（暖/冷）、阴影硬度、整体氛围（明亮/沉稳/温暖/冷峻/戏剧性）。

============================
C. 输出要求
============================

- 输出一段连贯的中文自然语言描述，不分行、不带维度标题；
- 长度约 350-600 字，**商品描述部分必须占 60-70%**；
- 描述顺序：商品整体形态 → 商品各部件（自上而下）→ 商品文字与 logo（逐处转录）→ 商品表面细节与状态 → 场景背景 → 镜头与光影氛围；
- **禁止使用模糊主观词**（如"精美/漂亮/高级/有质感/精致/优雅/吸引人"），必须用具体可视化特征替代；
- 仅描述画面中可见的内容，不做情感推断、不虚构未呈现的细节；
- 对于看不清的文字，标注为"此处有一行XX颜色印刷文字（具体内容不清晰）"，但**绝不允许直接省略不写**；
- 对于完全相同的重复元素（如多瓶并列），只描述一遍并注明数量与排列方式。"""

# ==========================================
# 🧷 编辑任务前缀（关注输入条件图 + 保持商品一致性）
# ==========================================
PROMPT_PREFIX = """任务：根据下方"目标画面描述"对参考输入图进行编辑。

【硬性约束】（最高优先级，与目标描述冲突时一律以本节为准）
- 商品本体：完全保留输入图中商品的外观、形状、比例、姿态、材质纹理、颜色与表面细节；
- 可识别信息：完整保留商品上的所有 logo、品牌名、包装文字、图案与印刷内容，不得替换、变形、遮挡或改变排版；
- 未要求修改的区域：保持与输入图一致，不引入额外变化。

【编辑目标】
按下方"目标画面描述"重构除商品主体之外的画面要素（背景、场景、光照、氛围、构图、其他物体等），使最终结果在视觉风格、构图、光线与氛围上与描述一致。

【目标画面描述】
"""

def format_final_prompt(caption: str) -> str:
    return PROMPT_PREFIX + caption

def encode_img_to_base64(img_path):
    """把本地图片编码成 OpenAI 兼容的 data URL（直接读原始字节，与 qwen_image_edit_2511 脚本一致）。"""
    ext = os.path.splitext(img_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/png")

    with open(img_path, "rb") as f:
        data = f.read()
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")

def generate_image_caption(image_path, retry_count=3):
    expanded_path = os.path.expanduser(image_path)
    if not os.path.exists(expanded_path): return f"Error: Image not found at {expanded_path}"

    try:
        image_base64 = encode_img_to_base64(expanded_path)
    except Exception as e:
        return f"Error: Failed to process image {image_path}. {e}"

    messages = [
        {"role": "system", "content": CAPTION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": "请严格按照系统提示的维度与要求，为这张图片生成一段连贯的中文画面描述。只输出最终描述文本，不要其他解释。"},
            ],
        },
    ]

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL_NAME,
                messages=messages,
                temperature=0.1,  # 描述类任务调低温度，减少幻觉、保持稳定
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
                # 字段语义：
                #   original_prompt = 原始 prompt（备份）
                #   caption         = VL 生成的纯 GT 描述（不含前缀，便于下次换前缀重生成）
                #   prompt          = 最终训练用 prompt = 前缀 + caption
                item["original_prompt"] = item.get("prompt")
                item["caption"] = vl_caption
                item["prompt"] = format_final_prompt(vl_caption)
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
    print(f"🎉 开始归位与重建 {total_world_size} 个节点的大业...")

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
    parser = argparse.ArgumentParser(description="分布式打标集群版 v2（细节增强 + 编辑前缀格式化）")
    parser.add_argument("--node_rank", type=int, required=True, help="当前机器编号 (0 到 3)")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="每台机器的 GPU/进程数 (默认为 8)")
    parser.add_argument("--total_nodes", type=int, default=4, help="集群总机器数 (默认为 4)")
    parser.add_argument("--merge_only", action="store_true", help="是否仅执行最终合并逻辑")
    args = parser.parse_args()

    ORIGINAL_JSON_PATH = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate.json"
    OUTPUT_FINAL_JSON = "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate_optimized.json"

    # 💥 极度重要：因为你要用 4 台机器，临时文件夹必须挂载在 4 台机器都能访问的共享存储(NAS)上！
    # 否则机器 A 的结果，机器 B 是合并不到的！(假设你的 /data 目录是互通的)
    # 注意：v2 用单独的临时目录，避免与旧版结果混在一起
    TEMP_OUTPUT_DIR = "/data/phd/kousiqi/zhitao/hard_products_rescue_temp_cluster_v2"
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
