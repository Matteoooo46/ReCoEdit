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
from apg_guidance import patch_pipeline_for_apg
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit RL inference 单产品推理脚本 (APG版)")
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
TARGET_WORKSPACE_DIR = cmd_args.workspace_dir or "/data/phd/lijiahui/data/0304_gen_meizhuang_kling_shu3/workspace_25833632310597_1772644206"

# 结果保存的父目录
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"

product_name = os.path.basename(TARGET_WORKSPACE_DIR)

# ⚠️ 每次换 ckpt 或换 prompt 策略时修改这一个，其余路径自动隔离
RUN_TAG = "rewrite_optimized_apg"

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

# ==========================================
# 1.1 QwenVL Prompt 改写模型配置（本地 Qwen3-VL-8B + LoRA）
# ==========================================
REWRITER_BASE_MODEL = "/data/phd/kousiqi/zhitao/models/Qwen3-VL-8B-Instruct"
REWRITER_LORA_PATH = "/data/phd/kousiqi/zhitao/qwen3-vl-8b-sft-v2/checkpoint-10000"

# 全局变量，在 main 中初始化
rewriter_model = None
rewriter_processor = None

PROMPT_REWRITE_SYSTEM = """
你是一个电商商品图像编辑 prompt 改写助手。你的任务是根据商品参考图和 original prompt，生成一段更清晰、更具体、更适合图像编辑模型执行的中文 prompt。

请遵守以下原则：

1. 以 original prompt 的编辑意图为主。
保留并扩写 original prompt 中的场景、背景、光线、氛围、构图、人物、动作、道具和风格要求。不要改变原始编辑目标，不要把不同 prompt 都改写成相似的商品展示图。

2. 适度补充商品细节。
根据参考图描述商品本体的关键视觉特征，包括商品类别、整体形状、主色/辅色、主要材质、明显 logo/文字/图案、最重要的结构细节。商品细节要足够帮助模型保持商品一致性，但不要写成参考图长 caption。

3. 不要过度描述参考图。
参考图只用于识别商品本体。不要描述参考图里的原始背景、桌面、墙面、光照、阴影、拍摄角度、摆放方式、手、模特或装饰物，除非 original prompt 明确要求保留。

4. 控制内容比例和长度。
最终 prompt 中，编辑目标约占 50%，商品细节约占 40%，质量约束约占 10%。总长度控制在 120～150 个中文字符。不要列小标题，不要分点输出。

5. 质量和约束。
保持商品颜色、形状、logo、文字、图案和材质尽量与参考图一致；不要新增无关文字、水印、乱码；不要让商品变形、镜像、错色或丢失关键标识。只描述单帧静态画面。

只输出一条最终 prompt，格式如下：
i2v描述：<优化后的 prompt>
"""

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
# CONSTRAINT_PREFIX = """【硬性约束 - 商品本体】
# 严格参照输入参考图保留商品主体：外观/形状/结构/比例/姿态、材质纹理与表面工艺、所有颜色（含色彩配比与明暗）、所有 logo 与商标、所有可见文字（逐字保留，不重排、不翻译、不更换字体）、所有图案/印花/装饰元素，以及任何用于识别该商品的关键细节。商品本体不得变形、不得缩放比例失衡、不得缺失部件、不得增加部件。
#
# 【冲突解决】
# 如下文描述与参考图商品本体存在出入，一律以参考图为准；下文描述仅用于补充画面其余部分（背景、场景、光照、构图、氛围、其他物体）。未提及的区域保持与参考图一致。
#
# 【画面描述】
# """
ANTI_TEXT_NEGATIVE_PROMPT = "text, watermark, signature, letters, writing, words, typography, logo, brand name, garbled characters, subtitles"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def load_rewriter_model():
    """加载 Qwen3-VL-8B + LoRA checkpoint，返回 (model, processor)。
    与 QwenImagePipeline 共用同一 GPU，顺序执行不会冲突。"""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        REWRITER_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, REWRITER_LORA_PATH)
    processor = AutoProcessor.from_pretrained(REWRITER_BASE_MODEL, trust_remote_code=True)
    return model, processor


def extract_rewritten_prompt(answer):
    """从 QwenVL 返回中提取 i2v描述 字段；解析失败时返回清洗后的原文。"""
    if not answer:
        return ""
    match = re.search(r"i2v描述\s*[:：]\s*(.+)", answer, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        text = answer.strip()

    # 去掉可能的 Markdown 包裹和多余换行
    text = text.strip().strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def rewrite_prompt_with_qwenvl(original_prompt, ref_image_paths):
    """
    根据商品参考图 + 原始 prompt 改写得到更适合 Qwen-Image-Edit 的 prompt。
    使用本地 Qwen3-VL-8B + LoRA checkpoint。
    ref_image_paths: 单个路径字符串或路径列表，作为商品参考送入模型。
    失败时不影响主流程，直接返回 original_prompt。
    """
    global rewriter_model, rewriter_processor

    if rewriter_model is None:
        print("  [警告] Rewriter 模型未加载，回退原 prompt。")
        return original_prompt

    try:
        if isinstance(ref_image_paths, str):
            ref_image_paths = [ref_image_paths]

        # 加载并 resize 参考图
        images = []
        valid_paths = []
        for p in ref_image_paths:
            if not os.path.exists(p):
                print(f"  [警告] 参考图不存在，跳过: {p}")
                continue
            img = Image.open(p).convert("RGB")
            img.thumbnail((1024, 1024))
            images.append(img)
            valid_paths.append(p)

        if not valid_paths:
            print("  [警告] 没有可用的商品参考图，回退原 prompt。")
            return original_prompt

        # 构造 messages，匹配训练格式：<image> + PROMPT_REWRITE_SYSTEM + original_prompt
        content = [{"type": "image", "image": img} for img in images]
        content.append({
            "type": "text",
            "text": PROMPT_REWRITE_SYSTEM + "\n\n" + original_prompt,
        })
        messages = [{"role": "user", "content": content}]

        start_time = time.time()

        text = rewriter_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = rewriter_processor(
            text=[text], images=images, return_tensors="pt",
        ).to(rewriter_model.device)

        with torch.no_grad():
            generated_ids = rewriter_model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        answer = rewriter_processor.tokenizer.decode(
            output_ids, skip_special_tokens=True,
        ).strip()

        rewritten_prompt = extract_rewritten_prompt(answer)
        print("[QwenVL 原始返回]", answer)
        print(f"[QwenVL 改写耗时] {time.time() - start_time:.2f} 秒")

        if rewritten_prompt:
            return rewritten_prompt
        return original_prompt
    except Exception as e:
        print(f"  [Prompt改写失败，回退原prompt] 参考图: {ref_image_paths}, 报错: {e}")
        return original_prompt


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
    vram_limit=135
)

# GRPO round2 best checkpoint (eager-cloud-11, epoch 28, reward peak 0.491)
lora_path = "/data/phd/kousiqi/zhitao/flow_grpo/best_checkpoints/grpo_round2_epoch28_for_inference.safetensors"
pipe.load_lora(pipe.dit, lora_path)

# ==========================================
# 2.5 启用 APG (Adaptive Projected Guidance) — 替代标准 CFG
# ==========================================
# APG 论文: "Eliminating Oversaturation and Artifacts of High Guidance Scales
#           in Diffusion Models" (https://huggingface.co/papers/2410.02416)
#
# 核心公式 (x0 空间):
#   x0 = latents - sigma * v                    (flow matching: velocity → x0)
#   diff = x0_cond - x0_uncond
#   parallel = proj(diff, x0_cond)              (投影到条件预测方向)
#   orthogonal = diff - parallel
#   apg_update = orthogonal + eta * parallel
#   x0_guided = x0_cond + (cfg_scale - 1) * apg_update
#   v_guided = (latents - x0_guided) / sigma    (x0 → velocity)
#
# 与标准 CFG 的区别:
#   CFG:  v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)   (velocity 空间)
#   APG:  在 x0 空间投影，分离 parallel/orthogonal，eta 抑制 parallel
#
# 推荐测试参数: cfg_scale=4.0, eta=0.0, beta=-0.5
patch_pipeline_for_apg(
    pipe,
    eta=0.0,            # 0.0 = 完全抑制 parallel 分量 (最激进，防过饱和最强)
                        # 0.25 = 轻度抑制, 1.0 = 退化为标准 CFG
    beta=-0.5,          # 反向动量系数 (paper: running = diff + beta * running)
                        # -0.5 = 中等, -0.75 = 强力, None = 禁用
    norm_threshold=None,  # Norm clipping, None = 禁用 (可选值如 15.0)
)

print("模型加载完毕！(APG 已启用)")

# 加载本地 Rewriter 模型 (Qwen3-VL-8B + LoRA)，与图像模型共用 GPU
print("加载 Rewriter 模型...")
rewriter_model, rewriter_processor = load_rewriter_model()
print(f"Rewriter 模型加载完毕！(LoRA: {REWRITER_LORA_PATH})")

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
ref_image_paths_for_rewrite = [os.path.join(found_reference_dir, r) for r in ref_imgs]

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
    rewritten_prompt_text = rewrite_prompt_with_qwenvl(original_prompt_text, ref_image_paths_for_rewrite)

    # prompt_text = CONSTRAINT_PREFIX + rewritten_prompt_text + NO_TEXT_SUFFIX
    prompt_text = rewritten_prompt_text + NO_TEXT_SUFFIX
    print(f"Original Prompt: {original_prompt_text[:80]}...")
    print(f"Rewrite Prompt: {rewritten_prompt_text[:120]}...")

    # 保留原逻辑：反向提示词黑名单
    anti_text_negative_prompt = ANTI_TEXT_NEGATIVE_PROMPT

    # 推理生成 (APG: x0空间投影 + reverse momentum)
    # cfg_scale 建议测试序列: 4.0, 5.0, 6.0, 7.5
    try:
        generated_image = pipe(
            prompt=prompt_text,
            negative_prompt=anti_text_negative_prompt,
            edit_image=edit_image_list,
            seed=1,
            num_inference_steps=40,
            height=1152,
            width=896,
            edit_image_auto_resize=True,
            zero_cond_t=True,
            cfg_scale=4.0,  # APG: 从与CFG相同的scale开始对比
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
            "rewritten_prompt": rewritten_prompt_text,
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
        "prompt": log["rewritten_prompt"],
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
