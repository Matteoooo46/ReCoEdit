"""
ReCoEdit inference — APG guidance, no rewriter.

Usage:
    python inference/qwen_image_edit_2511_inference_apg.py \
        --input_image path/to/image.jpg \
        --prompt "把产品放到户外花园场景中" \
        --output result.png

Multiple reference images (the model attends to all of them):
    python inference/qwen_image_edit_2511_inference_apg.py \
        --input_image ref1.jpg ref2.jpg \
        --prompt "..." \
        --output result.png
"""

import argparse
import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from apg_guidance import patch_pipeline_for_apg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", nargs="+", required=True,
                        help="Input image path(s). Multiple paths = multiple reference images.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Editing instruction (Chinese or English).")
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


def load_pipeline(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print("Downloading models from HuggingFace (cached after first run)...")
    base_dir  = snapshot_download("Qwen/Qwen-Image-Edit-2511")
    lora_dir  = snapshot_download("Matteoooo46/ReCoEdit-RL")

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
    print("Model loaded with APG enabled.")
    return pipe


def run(args):
    pipe = load_pipeline(args.gpu)

    images = [Image.open(p).convert("RGB") for p in args.input_image]
    print(f"Loaded {len(images)} input image(s).")
    print(f"Prompt: {args.prompt}")

    result = pipe(
        prompt=args.prompt,
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
