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
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit 35产品推理（APG+Rewriter）")
    parser.add_argument("--product_dir", type=str, default=None, help="产品目录路径")
    parser.add_argument("--gpu", type=int, default=None, help="GPU编号")
    return parser.parse_args()


cmd_args = parse_args()
if cmd_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd_args.gpu)

TARGET_PRODUCT_DIR = cmd_args.product_dir or "/data/phd/kousiqi/zhitao/batch_product_images_all_refs/item_25925859476711_203f18e6"
OUTPUT_BASE_DIR = "/data/phd/kousiqi/zhitao/qwen_inference_results_single"

product_name = os.path.basename(TARGET_PRODUCT_DIR)
RUN_TAG = "random35_apg_rewrite"

RAW_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_raw_{RUN_TAG}")
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_{RUN_TAG}")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

HTML_REPORT_PATH = os.path.join(OUTPUT_BASE_DIR, f"{product_name}_{RUN_TAG}_viewer.html")
VIEWER_CONFIG_PATH = os.path.join(REPORT_DIR, f"{product_name}_viewer_config.json")
PROMPT_LOG_PATH = os.path.join(REPORT_DIR, f"{product_name}_prompt_log.json")

# Rewriter model config (local Qwen3-VL-8B + LoRA)
REWRITER_BASE_MODEL = "/data/phd/kousiqi/zhitao/models/Qwen3-VL-8B-Instruct"
REWRITER_LORA_PATH = "/data/phd/kousiqi/zhitao/qwen3-vl-8b-sft-v2/checkpoint-10000"

rewriter_model = None
rewriter_processor = None

PROMPT_REWRITE_SYSTEM = """
你是一个电商商品图像编辑 prompt 改写助手。

你的任务是根据商品参考图和 original prompt，生成一段更清晰、更具体、更适合图像编辑模型执行的中文 prompt。

硬性字数要求（必须严格遵守）：
改写后的 prompt 总长度必须控制在 120～150 个中文字符以内，绝对不能超过 150 字。如果你写的超过了 150 字，请删减到 150 字以内。

改写原则：

1. 以 original prompt 的编辑意图为主。
保留并扩写 original prompt 中的场景、背景、光线、氛围、构图、人物、动作、道具和风格要求。不要改变原始编辑目标。

2. 适度补充商品细节。
根据参考图描述商品本体的关键视觉特征，包括商品类别、整体形状、主色/辅色、主要材质、明显 logo/文字/图案。只需 1-2 句核心特征，不要写长描述。

3. 不要过度描述参考图。
参考图只用于识别商品本体。不要描述参考图里的原始背景、桌面、墙面、光照、阴影、拍摄角度、摆放方式、手、模特或装饰物。

4. 保持商品颜色、形状、logo、文字、图案和材质尽量与参考图一致；不要新增无关文字、水印、乱码；不要让商品变形、镜像、错色或丢失关键标识。只描述单帧静态画面。

再次强调：最终输出必须严格控制在 120～150 个中文字符以内，不能超过 150 字。

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
    print(f"Uploading {len(valid_paths)} files...")
    for file_path in valid_paths:
        try:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            bs_key = f"{blob_prefix}/{parent_dir}/{file_name}"
            blobstore.upload_binary_to_s3(file_path, bs_key)
            path_to_url[file_path] = f"{BLOBSTORE_CDN}/{bs_key}"
        except Exception as e:
            print(f"  [Upload failed] {file_path}: {e}")
            path_to_url[file_path] = ""
    return [path_to_url.get(p, "") if p else "" for p in file_paths]


NO_TEXT_SUFFIX = "，画面中不要出现任何多余的文字、字幕、水印、签名、乱码或无关logo，仅保留商品原本的品牌文字且保持清晰一致。"
ANTI_TEXT_NEGATIVE_PROMPT = "text, watermark, signature, letters, writing, words, typography, logo, brand name, garbled characters, subtitles"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def load_rewriter_model():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        REWRITER_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, REWRITER_LORA_PATH)
    processor = AutoProcessor.from_pretrained(REWRITER_BASE_MODEL, trust_remote_code=True)
    return model, processor


def extract_rewritten_prompt(answer):
    if not answer:
        return ""
    match = re.search(r"i2v描述\s*[:：]\s*(.+)", answer, flags=re.DOTALL)
    text = match.group(1).strip() if match else answer.strip()
    text = text.strip().strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    # Safety truncation: if > 150 Chinese chars, cut to ~150
    if len(text) > 150:
        # Try to cut at a sentence boundary (。！？)
        truncated = text[:150]
        for sep in "。！？":
            idx = truncated.rfind(sep)
            if idx > 100:
                text = truncated[:idx+1]
                break
    return text


def rewrite_prompt_with_model(original_prompt, ref_image_paths):
    global rewriter_model, rewriter_processor
    if rewriter_model is None:
        print("  [Warn] Rewriter not loaded, using original prompt")
        return original_prompt
    try:
        if isinstance(ref_image_paths, str):
            ref_image_paths = [ref_image_paths]
        images = []
        for p in ref_image_paths:
            if not os.path.exists(p): continue
            img = Image.open(p).convert("RGB")
            img.thumbnail((1024, 1024))
            images.append(img)
        if not images:
            return original_prompt

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": PROMPT_REWRITE_SYSTEM + "\n\n" + original_prompt})
        messages = [{"role": "user", "content": content}]

        start_time = time.time()
        text = rewriter_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = rewriter_processor(text=[text], images=images, return_tensors="pt").to(rewriter_model.device)
        with torch.no_grad():
            generated_ids = rewriter_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        answer = rewriter_processor.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()
        rewritten = extract_rewritten_prompt(answer)
        print(f"[Rewriter] {time.time() - start_time:.1f}s  {rewritten[:80]}..." if rewritten else f"[Rewriter] empty output")
        return rewritten if rewritten else original_prompt
    except Exception as e:
        print(f"  [Rewrite failed] {e}")
        return original_prompt


# ==========================================
# 1. 初始化图像模型 + APG
# ==========================================
print("Loading image model and LoRA...")
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
    "offload_dtype": "disk", "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn, "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=MODEL_CONFIG_PATHS["dit"], **vram_config),
        ModelConfig(path=MODEL_CONFIG_PATHS["text_encoder"], **vram_config),
        ModelConfig(path=MODEL_CONFIG_PATHS["vae"], **vram_config)
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(path=MODEL_CONFIG_PATHS["processor"]),
    vram_limit=135
)

lora_path = "/data/phd/kousiqi/zhitao/flow_grpo/best_checkpoints/grpo_round2_epoch28_for_inference.safetensors"
pipe.load_lora(pipe.dit, lora_path)

# APG
patch_pipeline_for_apg(pipe, eta=0.0, beta=-0.5, norm_threshold=None)
print("Image model loaded! (APG enabled)")

# Rewriter
print("Loading Rewriter model...")
rewriter_model, rewriter_processor = load_rewriter_model()
print(f"Rewriter loaded! (LoRA: {REWRITER_LORA_PATH})")

# ==========================================
# 2. 读取 final_storyboard_script.json
# ==========================================
script_path = os.path.join(TARGET_PRODUCT_DIR, "final_storyboard_script.json")
if not os.path.exists(script_path):
    raise FileNotFoundError(f"final_storyboard_script.json not found in {TARGET_PRODUCT_DIR}")

with open(script_path, "r", encoding="utf-8") as f:
    script_data = json.load(f)
scripts = script_data.get("script", [])
prompts = [s.get("description", "") for s in scripts if s.get("description", "").strip()]
print(f"Product: {product_name}, {len(prompts)} prompts")

# ==========================================
# 3. 参考图
# ==========================================
ref_dir = os.path.join(TARGET_PRODUCT_DIR, "reference_imgs")
ref_imgs = sorted([img for img in os.listdir(ref_dir) if img.lower().endswith(IMAGE_EXTS)])
if not ref_imgs:
    raise FileNotFoundError("No reference images!")

edit_image_list = []
viewer_globals = []
folder_name = os.path.basename(REPORT_DIR)
ref_image_paths_for_rewrite = []

for idx, ref_img in enumerate(ref_imgs):
    ref_path = os.path.join(ref_dir, ref_img)
    edit_image_list.append(Image.open(ref_path).convert("RGB"))
    ref_image_paths_for_rewrite.append(ref_path)
    report_ref_name = f"p1_ref_{idx}.png"
    shutil.copy(ref_path, os.path.join(REPORT_DIR, report_ref_name))
    viewer_globals.append({"name": f"Ref {idx+1}", "path": f"{folder_name}/{report_ref_name}"})

# 人物图 final_lifestyle_image.png
char_path = os.path.join(TARGET_PRODUCT_DIR, "final_lifestyle_image.png")
if os.path.exists(char_path):
    edit_image_list.append(Image.open(char_path).convert("RGB"))
    shutil.copy(char_path, os.path.join(REPORT_DIR, "p1_char.png"))
    viewer_globals.append({"name": "Character", "path": f"{folder_name}/p1_char.png"})
    print(f"Character image added: final_lifestyle_image.png")

print(f"Total edit images: {len(edit_image_list)}")

# ==========================================
# 4. 推理
# ==========================================
product_viewer_config = {"title": f"{product_name} [{RUN_TAG}]", "globals": viewer_globals, "frames": []}
prompt_logs = []
frame_local_paths = []

for i, original_prompt in enumerate(prompts):
    print(f"\n---> Frame {i}: {original_prompt[:80]}...")

    rewritten = rewrite_prompt_with_model(original_prompt, ref_image_paths_for_rewrite)
    final_prompt = rewritten + NO_TEXT_SUFFIX

    try:
        generated_image = pipe(
            prompt=final_prompt, negative_prompt=ANTI_TEXT_NEGATIVE_PROMPT,
            edit_image=edit_image_list, seed=i, num_inference_steps=40,
            height=1152, width=896, edit_image_auto_resize=True,
            zero_cond_t=True, cfg_scale=4.0,
        )
        my_img_path = os.path.join(RAW_OUTPUT_DIR, f"frame_{i}.jpg")
        generated_image.save(my_img_path)
        print(f"  Saved: {my_img_path}")

        report_my_name = f"p1_my_{i + 1}.png"
        if os.path.exists(my_img_path):
            shutil.copy(my_img_path, os.path.join(REPORT_DIR, report_my_name))

        frame_local_paths.append(my_img_path)
        prompt_logs.append({"frame_index": i, "original_prompt": original_prompt, "rewritten_prompt": rewritten})
    except Exception as e:
        print(f"  [FAILED] Frame {i}: {e}")

# ==========================================
# 5. 上传 & HTML
# ==========================================
print("\nUploading...")
ref_file_paths = [os.path.join(ref_dir, r) for r in ref_imgs]
if os.path.exists(char_path):
    ref_file_paths.append(char_path)
all_upload_paths = ref_file_paths + frame_local_paths
urls = upload_files(all_upload_paths, BLOB_PREFIX)
url_map = dict(zip(all_upload_paths, urls))

for idx, ref_img in enumerate(ref_imgs):
    local = os.path.join(ref_dir, ref_img)
    viewer_globals[idx]["path"] = url_map.get(local, viewer_globals[idx]["path"])

for log_idx, log in enumerate(prompt_logs):
    my_local = frame_local_paths[log_idx]
    product_viewer_config["frames"].append({
        "prompt": log["rewritten_prompt"],
        "original_prompt": log["original_prompt"],
        "my_path": url_map.get(my_local, ""),
    })

with open(PROMPT_LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(prompt_logs, f, ensure_ascii=False, indent=4)
print(f"Prompt log saved: {PROMPT_LOG_PATH}")

# Simple viewer HTML
cfg_json = json.dumps([product_viewer_config], ensure_ascii=False)
escaped = html_lib.escape(cfg_json)
viewer_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{product_name} [{RUN_TAG}]</title>
<style>
:root{{--bg:#f5f6f7;--card:#fff;--text:#1f2329;--sub:#8f959e;--border:#dee0e3;--pri:#3370ff}}
body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}}
.container{{max-width:1400px;margin:0 auto}}h2{{color:var(--pri)}}
.product-section{{background:var(--card);border-radius:12px;padding:25px;margin-bottom:40px;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
.product-title{{margin-top:0;color:var(--pri);border-bottom:1px dashed var(--border);padding-bottom:10px}}
.global-inputs{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
.global-img-box img{{width:140px;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--border);margin-bottom:5px}}
.grid-container{{display:flex;gap:20px;overflow-x:auto;padding-bottom:15px}}
.frame-column{{flex:0 0 280px;background:#fafafa;border-radius:8px;border:1px solid var(--border);padding:15px}}
.prompt-text{{font-size:14px;color:#d83931;line-height:1.5;margin-bottom:15px;white-space:pre-wrap}}
.original-prompt{{font-size:12px;color:#646a73;margin-bottom:10px;white-space:pre-wrap}}
.image-compare-box img{{width:100%;border-radius:6px;border:1px solid var(--border)}}
.btn{{background:var(--pri);color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;margin-top:10px;font-size:14px}}
</style></head>
<body><div class="container">
<h2>{product_name} [{RUN_TAG}]</h2>
<div id="content"><p style="color:#999">Loading...</p></div>
</div>
<script>
var DATA={cfg_json};
(function(){{
    function e(t){{return t?String(t).replace(/[&<>"]/g,function(c){{return{{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}}):'';}}
    DATA.forEach(function(p){{
        var h='<div class="product-section"><h2 class="product-title">'+e(p.title)+'</h2><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Global input:</h3><div class="global-inputs">';
        p.globals.forEach(function(g){{h+='<div class="global-img-box"><img src="'+g.path+'" alt="'+e(g.name)+'" onerror="this.src=\\'https://dummyimage.com/150x150/ffcccc/f00\\'"><div>'+e(g.name)+'</div></div>';}});
        h+='</div><h3 style="font-size:16px;color:#555;margin:20px 0 10px;">Generated Images:</h3><div class="grid-container">';
        p.frames.forEach(function(f){{
            var o=f.original_prompt?'<div class="original-prompt"><b>Original:</b> '+e(f.original_prompt)+'</div>':'';
            h+='<div class="frame-column"><div class="prompt-text"><b>Rewrite:</b> '+e(f.prompt)+'</div>'+o+'<div class="image-compare-box"><h4>My Model</h4><img src="'+f.my_path+'" onerror="this.src=\\'https://dummyimage.com/280x350/ffcccc/f00\\'"></div></div>';
        }});
        h+='</div></div>';document.getElementById('content').innerHTML=h;
    }});
}})();
</script></body></html>"""

with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(viewer_html)

with open(VIEWER_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(product_viewer_config, f, ensure_ascii=False, indent=4)

try:
    print(f"Zipping {folder_name}...")
    os.system(f"cd {OUTPUT_BASE_DIR} && zip -r {folder_name}.zip {folder_name}/")
    blobstore = BlobStoreClient(BLOBSTORE_BUCKET)
    html_bs_key = f"{BLOB_PREFIX}/viewer/{os.path.basename(HTML_REPORT_PATH)}"
    blobstore.upload_binary_to_s3(HTML_REPORT_PATH, html_bs_key)
    html_url = f"{BLOBSTORE_CDN}/{html_bs_key}"
    print("=" * 50)
    print(f"Done! {len(prompt_logs)}/{len(prompts)} frames generated")
    print(f"HTML: {html_url}")
    print("=" * 50)
except Exception as e:
    print(f"Upload failed: {e}")
