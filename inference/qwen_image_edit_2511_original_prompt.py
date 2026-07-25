from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
import os
import json
import shutil
import base64
import re
import time
import html as html_lib
import argparse
from blobstore import BlobStoreClient


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit 单产品推理脚本")
    parser.add_argument("--workspace_dir", type=str, default=None,
                        help="产品 workspace 文件夹路径（覆盖下方硬编码值）")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定使用的 GPU 编号（0-7），会自动设置 CUDA_VISIBLE_DEVICES")
    return parser.parse_args()


cmd_args = parse_args()
if cmd_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd_args.gpu)

# ==========================================
# 1. 基础路径配置
# ==========================================
TARGET_WORKSPACE_DIR = cmd_args.workspace_dir or "/data/phd/lijiahui/data/batch_no_voice_gen_fuzhuang/workspace_item_25936355083926"

# 结果保存的父目录
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"

product_name = os.path.basename(TARGET_WORKSPACE_DIR)

# ⚠️ 每次换 ckpt 或换 prompt 策略时修改这一个，其余路径自动隔离
RUN_TAG = "original_prompt_full_added"

# 存放生成的原始图 (frame_0.jpg, frame_1.jpg)
RAW_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw_{RUN_TAG}")
# 存放按网页规则重命名好的展示图 (p1_nano_1.png, p1_my_1.png)
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_{RUN_TAG}")

os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 展示 HTML / 可合并 JSON 配置
HTML_REPORT_PATH = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_{RUN_TAG}_viewer.html")
VIEWER_CONFIG_PATH = os.path.join(REPORT_DIR, f"{product_name}_viewer_config.json")
PROMPT_LOG_PATH = os.path.join(REPORT_DIR, f"{product_name}_prompt_rewrite_log.json")





BLOB_PREFIX = f"qwen_inference/{product_name}/{RUN_TAG}"
BLOBSTORE_BUCKET = "ad-nieuwland-material"
BLOBSTORE_CDN = "https://s1-11661.kwimgs.com/bs2/ad-nieuwland_material"


def upload_files(file_paths, blob_prefix):
    blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
    path_to_url = {}
    valid_paths = list(set(p for p in file_paths if p and os.path.exists(p)))
    print(f"☁️  正在上传 {len(valid_paths)} 个文件...")
    for file_path in valid_paths:
        try:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            bs_key = f"{blob_prefix}/{parent_dir}/{file_name}"
            blobstore.upload_binary_to_s3(file_path, bs_key)
            path_to_url[file_path] = f"{BLOBSTORE_CDN}/{bs_key}"
        except Exception as e:
            print(f"  [上传失败] {file_path}: {e}")
            path_to_url[file_path] = ""
    return [path_to_url.get(p, "") if p else "" for p in file_paths]

NO_TEXT_SUFFIX = "，画面中不要出现任何多余的文字、字幕、水印、签名、乱码或无关logo，仅保留商品原本的品牌文字且保持清晰一致。"
ANTI_TEXT_NEGATIVE_PROMPT = "text, watermark, signature, letters, writing, words, typography, logo, brand name, garbled characters, subtitles"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def safe_load_prompt(json_path):
    """读取原始 prompt，兼容 list[dict] 和 dict。"""
    with open(json_path, "r", encoding="utf-8") as jf:
        prompt_data = json.load(jf)
        real_data = prompt_data[0] if isinstance(prompt_data, list) else prompt_data
        return real_data.get("prompt", "") or real_data.get("product_info", "")


def get_frame_num(filename):
    try:
        return int(filename.split("_")[0])
    except Exception:
        return 999999


def build_viewer_html(products_config):
    """生成和 image_viewer.html 同结构的展示页面。"""
    config_json = json.dumps(products_config, ensure_ascii=False, indent=4)
    escaped_config_json = html_lib.escape(config_json)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电商图片生成效果对比</title>
    <style>
        :root {{
            --bg-color: #f5f6f7;
            --card-bg: #ffffff;
            --text-main: #1f2329;
            --text-secondary: #8f959e;
            --border-color: #dee0e3;
            --primary-color: #3370ff;
            --rewrite-color: #00b96b;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }}
        .control-panel {{ background: var(--card-bg); padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .control-panel textarea {{ width: 100%; height: 250px; font-family: monospace; font-size: 13px; padding: 10px; border: 1px solid var(--border-color); border-radius: 4px; box-sizing: border-box; resize: vertical; }}
        .btn {{ background-color: var(--primary-color); color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; font-size: 14px; }}
        .btn:hover {{ background-color: #2458db; }}
        .product-section {{ background: var(--card-bg); border-radius: 12px; padding: 25px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .product-title {{ margin-top: 0; color: var(--primary-color); border-bottom: 1px dashed var(--border-color); padding-bottom: 10px; }}
        .section-subtitle {{ font-size: 16px; color: #555; margin: 20px 0 10px 0; }}
        .global-inputs {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .global-img-box {{ width: 140px; text-align: center; font-size: 13px; color: var(--text-secondary); }}
        .global-img-box img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color); margin-bottom: 5px; }}
        .grid-container {{ display: flex; gap: 20px; overflow-x: auto; padding-bottom: 15px; }}
        .frame-column {{ flex: 0 0 280px; background: #fafafa; border-radius: 8px; border: 1px solid var(--border-color); padding: 15px; display: flex; flex-direction: column; }}
        .prompt-text {{ font-size: 14px; color: #d83931; line-height: 1.5; margin-bottom: 15px; min-height: 80px; white-space: pre-wrap; }}
        .original-prompt {{ font-size: 12px; color: #646a73; line-height: 1.4; margin-bottom: 10px; white-space: pre-wrap; }}
        .image-compare-box {{ margin-bottom: 15px; }}
        .image-compare-box h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--text-main); }}
        .image-compare-box img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border-color); cursor: pointer; transition: transform 0.2s; }}
        .image-compare-box img:hover {{ transform: scale(1.02); }}
        .my-model-title {{ color: var(--primary-color) !important; }}
        .my-rewrite-title {{ color: var(--rewrite-color) !important; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>🛍️ 电商图片编辑</h2>
        <button class="btn" onclick="togglePanel()">⚙️ 展开/隐藏配置面板</button>
    </div>

    <div class="control-panel" id="controlPanel">
        <h3>配置数据 (JSON Array)</h3>
        <p style="font-size: 12px; color: gray;">每个产品对应一个对象；合并多个产品时，把多个 config JSON 对象放进同一个数组即可。</p>
        <textarea id="mainConfig">{escaped_config_json}</textarea>
        <button class="btn" onclick="renderPage()">🚀 渲染网页</button>
    </div>

    <div id="mainContent"></div>
</div>

<script>
    function togglePanel() {{
        const panel = document.getElementById('controlPanel');
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }}

    function escapeHtml(text) {{
        if (!text) return '';
        return String(text).replace(/[&<>"]/g, function(ch) {{
            return {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}}[ch];
        }});
    }}

    function renderPage() {{
        try {{
            const data = JSON.parse(document.getElementById('mainConfig').value);
            const container = document.getElementById('mainContent');
            container.innerHTML = '';

            data.forEach(product => {{
                let html = `
                <div class="product-section">
                    <h2 class="product-title">${{escapeHtml(product.title)}}</h2>
                    <h3 class="section-subtitle">Global input:</h3>
                    <div class="global-inputs">
                `;

                product.globals.forEach(g => {{
                    html += `
                        <div class="global-img-box">
                            <img src="${{g.path}}" alt="${{escapeHtml(g.name)}}" onerror="this.src='https://dummyimage.com/150x150/ffcccc/f00&text=图片丢失'">
                            <div>${{escapeHtml(g.name)}}</div>
                        </div>
                    `;
                }});

                html += `
                    </div>
                    <h3 class="section-subtitle">Performance comparison:</h3>
                    <div class="grid-container">
                `;

                product.frames.forEach(frame => {{
                    let originalPromptHtml = '';
                    if (frame.original_prompt) {{
                        originalPromptHtml = `<div class="original-prompt"><b>Original:</b> ${{escapeHtml(frame.original_prompt)}}</div>`;
                    }}

                    let rewriteHtml = '';
                    if (frame.my_rewrite_path) {{
                        rewriteHtml = `
                            <div class="image-compare-box">
                                <h4 class="my-rewrite-title">My Model (Rewrite):</h4>
                                <img src="${{frame.my_rewrite_path}}" alt="My Rewrite Model" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失'">
                            </div>
                        `;
                    }}

                    html += `
                        <div class="frame-column">
                            <div class="prompt-text"><b>Rewrite Prompt:</b> ${{escapeHtml(frame.prompt)}}</div>
                            ${{originalPromptHtml}}

                            <div class="image-compare-box">
                                <h4>nano banana:</h4>
                                <img src="${{frame.nano_path}}" alt="Nano" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失'">
                            </div>

                            <div class="image-compare-box">
                                <h4 class="my-model-title">My Model:</h4>
                                <img src="${{frame.my_path}}" alt="My Model" onerror="this.src='https://dummyimage.com/280x350/ffcccc/f00&text=图片丢失'">
                            </div>

                            ${{rewriteHtml}}
                        </div>
                    `;
                }});

                html += `
                    </div>
                </div>
                `;
                container.innerHTML += html;
            }});
            document.getElementById('controlPanel').style.display = 'none';
        }} catch (error) {{
            alert("JSON 格式有误，请检查语法！\\n\\n错误信息: " + error.message);
        }}
    }}

    renderPage();
</script>
</body>
</html>
"""


def write_viewer_outputs(products_config):
    """写出可单独打开的 HTML 和可用于多产品合并的 config JSON。"""
    with open(VIEWER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(products_config[0], f, ensure_ascii=False, indent=4)

    html_text = build_viewer_html(products_config)
    with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"✅ 展示配置已保存: {VIEWER_CONFIG_PATH}")
    print(f"✅ 展示 HTML 已保存: {HTML_REPORT_PATH}")


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
    vram_limit=75
)

# GRPO round2 best checkpoint (eager-cloud-11, epoch 28, reward peak 0.491)
lora_path = "/data/phd/kousiqi/zhitao/full_all_products_rewrite_expanded_ultimate_resume_132000/step-40000.safetensors"
pipe.load_lora(pipe.dit, lora_path)

print("模型加载完毕！")

# ==========================================
# 3. 定位核心文件夹及准备参考图
# ==========================================
print(f"\n开始处理产品: {product_name}")

found_character_dir = None
found_reference_dir = None
found_target_dir = None

for root, dirs, files in os.walk(TARGET_WORKSPACE_DIR):
    if "character_refs" in dirs:
        found_character_dir = os.path.join(root, "character_refs")
    if "reference_imgs" in dirs:
        found_reference_dir = os.path.join(root, "reference_imgs")
    if "augmented_generated_images_v8_v6_context_pro" in dirs:
        found_target_dir = os.path.join(root, "augmented_generated_images_v8_v6_context_pro")

if not found_character_dir or not found_reference_dir or not found_target_dir:
    raise ValueError(f"❌ 未能在 {TARGET_WORKSPACE_DIR} 中找齐所有必须的子文件夹！")

# 获取并加载【人物图】
char_imgs = sorted([img for img in os.listdir(found_character_dir) if img.lower().endswith(IMAGE_EXTS)])
if not char_imgs:
    raise FileNotFoundError("找不到人物图！")
char_img_path = os.path.join(found_character_dir, char_imgs[0])

# 获取并加载所有的【产品图】
ref_imgs = sorted([img for img in os.listdir(found_reference_dir) if img.lower().endswith(IMAGE_EXTS)])
if not ref_imgs:
    raise FileNotFoundError("找不到产品图！")

# 准备用于推理的列表：动态加入所有产品图 + 1张人物图（这里固定不变，不加生成帧）
edit_image_list = []
viewer_globals = []
folder_name = os.path.basename(REPORT_DIR)

for idx, ref_img in enumerate(ref_imgs):
    ref_path = os.path.join(found_reference_dir, ref_img)
    edit_image_list.append(Image.open(ref_path).convert("RGB"))

    report_ref_name = f"p1_ref_{idx}.png"
    shutil.copy(ref_path, os.path.join(REPORT_DIR, report_ref_name))
    viewer_globals.append({"name": f"产品参考{idx + 1}", "path": f"{folder_name}/{report_ref_name}"})

edit_image_list.append(Image.open(char_img_path).convert("RGB"))
shutil.copy(char_img_path, os.path.join(REPORT_DIR, "p1_char.png"))
viewer_globals.append({"name": "人物参考", "path": f"{folder_name}/p1_char.png"})

print(f"✅ 参考图片已整理至 Report 文件夹。用于推理的图片共 {len(edit_image_list)} 张。")

# QwenVL 改写时使用的商品参考图列表（仅商品图，不含人物图；推理时不依赖目标图）

# ==========================================
# 4. 获取帧列表并排序
# ==========================================
target_files = [f for f in os.listdir(found_target_dir) if f.lower().endswith(".png")]
target_files.sort(key=get_frame_num)
print(f"共发现 {len(target_files)} 帧需要生成。")

# HTML 配置数据：单产品对象。多产品合并时，把多个这样的对象放入同一个数组。
product_viewer_config = {
    "title": f"{product_name} [{RUN_TAG}]",
    "globals": viewer_globals,
    "frames": []
}
prompt_rewrite_logs = []

# ==========================================
# 4.5 初始化本地路径收集（用于批量上传）
# ==========================================
frame_local_paths = []  # list of (ref_paths, char_path, nano_path, my_path)

# ==========================================
# 5. 开始独立逐帧推理与重命名整理
# ==========================================
for i, target_file in enumerate(target_files):
    json_file = target_file.replace(".png", ".json")
    json_path = os.path.join(found_target_dir, json_file)

    if not os.path.exists(json_path):
        continue

    # 读取 Prompt
    try:
        original_prompt_text = safe_load_prompt(json_path)
    except Exception as e:
        print(f"  [读取JSON失败] 文件: {json_file}, 报错: {e}")
        continue

    if not original_prompt_text:
        continue

    print(f"\n---> 正在生成第 {i} 帧...")

    # 先用 QwenVL 基于商品参考图 + 原始 prompt 改写出更详尽的 edit prompt，再进入 edit 模型推理
    nano_img_path = os.path.join(found_target_dir, target_file)
    
    # 保留原逻辑：在约束模板 + 改写prompt末尾追加”不要文字”的指令
    # prompt_text = original_prompt_text + NO_TEXT_SUFFIX
    prompt_text = original_prompt_text + NO_TEXT_SUFFIX
    print(f"Original Prompt: {original_prompt_text[:80]}...")
    print(f"Using original prompt directly...")

    # 保留原逻辑：反向提示词黑名单
    anti_text_negative_prompt = ANTI_TEXT_NEGATIVE_PROMPT

    # 推理生成
    try:
        generated_image = pipe(
            prompt=prompt_text,
            negative_prompt=anti_text_negative_prompt,
            edit_image=edit_image_list,
            seed=i,
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
        report_nano_name = f"p1_nano_{i + 1}.png"
        if os.path.exists(nano_img_path):
            shutil.copy(nano_img_path, os.path.join(REPORT_DIR, report_nano_name))

        # 3. 拷贝你刚生成的图，重命名为 p1_my_1.png ... (注意序号 i+1)
        report_my_name = f"p1_my_{i + 1}.png"
        if os.path.exists(my_img_path):
            shutil.copy(my_img_path, os.path.join(REPORT_DIR, report_my_name))

        # 4. 记录本地路径供后续批量上传
        frame_local_paths.append((nano_img_path, my_img_path))
        prompt_rewrite_logs.append({
            "frame_index": i,
            "target_file": target_file,
            "json_file": json_file,
            "original_prompt": original_prompt_text,
            "rewritten_prompt": original_prompt_text,
            "final_pipe_prompt": prompt_text,
        })

    except Exception as e:
        print(f"  [推理失败] 帧: {target_file}, 报错: {e}")

# ==========================================
# 6. 批量上传 → 替换 HTML 路径 → 生成展示 HTML
# ==========================================
print("\n开始上传文件并生成展示 HTML...")

# 收集需要上传的所有文件（参考图 + 每帧 nano + my）
ref_file_paths = [os.path.join(found_reference_dir, r) for r in ref_imgs] + [char_img_path]
frame_file_paths = [p for pair in frame_local_paths for p in pair]
all_upload_paths = ref_file_paths + frame_file_paths

urls = upload_files(all_upload_paths, BLOB_PREFIX)
url_map = dict(zip(all_upload_paths, urls))

# 用公网 URL 替换 globals 中的本地相对路径
for idx, ref_img in enumerate(ref_imgs):
    local = os.path.join(found_reference_dir, ref_img)
    viewer_globals[idx]["path"] = url_map.get(local, viewer_globals[idx]["path"])
viewer_globals[-1]["path"] = url_map.get(char_img_path, viewer_globals[-1]["path"])

# 用公网 URL 重建 frames（frame_local_paths 与推理循环中记录的 prompt_rewrite_logs 等长且顺序一致）
log_idx = 0
for i, target_file in enumerate(target_files):
    json_file = target_file.replace(".png", ".json")
    json_path = os.path.join(found_target_dir, json_file)
    if not os.path.exists(json_path):
        continue
    if log_idx >= len(frame_local_paths):
        break
    nano_local, my_local = frame_local_paths[log_idx]
    log = prompt_rewrite_logs[log_idx]
    product_viewer_config["frames"].append({
        "prompt": log["original_prompt"],
        "original_prompt": log["original_prompt"],
        "nano_path": url_map.get(nano_local, ""),
        "my_path": url_map.get(my_local, ""),
    })
    log_idx += 1

# 写 prompt 改写日志
with open(PROMPT_LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(prompt_rewrite_logs, f, ensure_ascii=False, indent=4)
print(f"✅ Prompt 改写日志已保存: {PROMPT_LOG_PATH}")

write_viewer_outputs([product_viewer_config])

folder_name = os.path.basename(REPORT_DIR)
zip_file_path = f"{REPORT_DIR}.zip"
try:
    print(f"正在把 {folder_name} 文件夹压缩成 ZIP 包...")
    os.system(f"cd {OUTPUT_BASE_DIR} && zip -r {folder_name}.zip {folder_name}/")
    # 上传 HTML 到 BlobStore，获取公网可访问 URL
    blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
    html_bs_key = f"{BLOB_PREFIX}/viewer/{os.path.basename(HTML_REPORT_PATH)}"
    blobstore.upload_binary_to_s3(HTML_REPORT_PATH, html_bs_key)
    html_url = f"{BLOBSTORE_CDN}/{html_bs_key}"
    print("==================================================")
    print(f"🎉 任务完美结束！")
    print(f"📦 压缩包下载路径: {zip_file_path}")
    print(f"🌐 HTML 公网访问地址: {html_url}")
    print(f"🧩 单产品可合并JSON: {VIEWER_CONFIG_PATH}")
    print("==================================================")
except Exception as e:
    print(f"压缩或上传失败，请手动检查。报错：{e}")

