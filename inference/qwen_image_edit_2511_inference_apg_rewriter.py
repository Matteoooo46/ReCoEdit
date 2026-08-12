"""
ReCoEdit inference — APG guidance + prompt rewriter.

The rewriter (Qwen3-VL-8B + LoRA) takes your product reference image(s) and a
raw editing prompt, then generates a richer, more structured prompt before
passing it to the image editing model. This typically improves product consistency.

Usage:
    python inference/qwen_image_edit_2511_inference_apg_rewriter.py \
        --input_image path/to/product.jpg \
        --prompt "把产品放到户外花园场景中" \
        --output result.png

Multiple reference images:
    python inference/qwen_image_edit_2511_inference_apg_rewriter.py \
        --input_image ref1.jpg ref2.jpg \
        --prompt "..." \
        --output result.png
"""

import argparse
import os
import re
import time
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from apg_guidance import patch_pipeline_for_apg
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


REWRITER_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REWRITER_LORA_REPO  = "Matteoooo46/ReCoEdit-rewriter"

REWRITE_SYSTEM_PROMPT = """
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", nargs="+", required=True,
                        help="Input image path(s). All images are used as product references.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Raw editing instruction (Chinese or English).")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output image path.")
    parser.add_argument("--negative_prompt", type=str,
                        default="text, watermark, signature, letters, writing, words, typography, logo, brand name, garbled characters, subtitles")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Defaults to CUDA_VISIBLE_DEVICES or GPU 0.")
    return parser.parse_args()


def load_rewriter():
    print("Loading rewriter model (Qwen3-VL-8B + LoRA)...")
    lora_dir = snapshot_download(REWRITER_LORA_REPO)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        REWRITER_BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_dir)
    processor = AutoProcessor.from_pretrained(REWRITER_BASE_MODEL, trust_remote_code=True)
    print("Rewriter loaded.")
    return model, processor


def rewrite_prompt(model, processor, original_prompt, images):
    """Rewrite a raw prompt using product reference images."""
    thumbs = []
    for img in images:
        t = img.copy()
        t.thumbnail((1024, 1024))
        thumbs.append(t)

    content = [{"type": "image", "image": t} for t in thumbs]
    content.append({"type": "text", "text": REWRITE_SYSTEM_PROMPT + "\n\n" + original_prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=thumbs, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    answer = processor.tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    print(f"Rewriter output ({time.time()-t0:.1f}s): {answer}")

    match = re.search(r"i2v描述\s*[:：]\s*(.+)", answer, re.DOTALL)
    rewritten = match.group(1).strip() if match else answer
    rewritten = re.sub(r"\s+", " ", rewritten).strip("`").strip()
    return rewritten or original_prompt


def load_pipeline(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print("Downloading base model from HuggingFace (cached after first run)...")
    base_dir = snapshot_download("Qwen/Qwen-Image-Edit-2511")
    lora_dir = snapshot_download("Matteoooo46/ReCoEdit-RL")

    vram_cfg = dict(
        offload_dtype="disk", offload_device="disk",
        onload_dtype=torch.float8_e4m3fn, onload_device="cpu",
        preparing_dtype=torch.float8_e4m3fn, preparing_device="cuda",
        computation_dtype=torch.bfloat16, computation_device="cuda",
    )

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[
                f"{base_dir}/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
                f"{base_dir}/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
                f"{base_dir}/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
                f"{base_dir}/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
                f"{base_dir}/transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
            ], **vram_cfg),
            ModelConfig(path=[
                f"{base_dir}/text_encoder/model-00001-of-00004.safetensors",
                f"{base_dir}/text_encoder/model-00002-of-00004.safetensors",
                f"{base_dir}/text_encoder/model-00003-of-00004.safetensors",
                f"{base_dir}/text_encoder/model-00004-of-00004.safetensors",
            ], **vram_cfg),
            ModelConfig(path=f"{base_dir}/vae/diffusion_pytorch_model.safetensors", **vram_cfg),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(path=f"{base_dir}/processor"),
        vram_limit=135,
    )

    pipe.load_lora(pipe.dit, f"{lora_dir}/adapter_model.safetensors")
    patch_pipeline_for_apg(pipe, eta=0.0, beta=-0.5, norm_threshold=None)
    print("Edit model loaded with APG enabled.")
    return pipe


def run(args):
    images = [Image.open(p).convert("RGB") for p in args.input_image]
    print(f"Loaded {len(images)} input image(s).")

    # Step 1: rewrite the prompt
    rewriter, processor = load_rewriter()
    rewritten_prompt = rewrite_prompt(rewriter, processor, args.prompt, images)
    print(f"Original prompt : {args.prompt}")
    print(f"Rewritten prompt: {rewritten_prompt}")

    # Free rewriter VRAM before loading the larger edit model
    del rewriter, processor
    torch.cuda.empty_cache()

    # Step 2: run the edit model
    pipe = load_pipeline(args.gpu)
    result = pipe(
        prompt=rewritten_prompt,
        negative_prompt=args.negative_prompt,
        edit_image=images,
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        edit_image_auto_resize=True,
        zero_cond_t=True,
        cfg_scale=args.cfg_scale,
    )

    result.save(args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    run(parse_args())
