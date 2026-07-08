from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

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



prompt = "一双干净的手温柔地捧着一堆形态优美的白茶茶叶，同时有更多茶叶从上方缓慢落下。，Cinematic, Photorealistic, 4K Commercial, High Quality, bright and vibrant lighting"
edit_image = [
Image.open("/data/phd/kousiqi/zhitao/shipin_data/shipin_23397479040558_consistency/reference_tea.jpg").convert("RGB"),

Image.open("/data/phd/lijiahui/data/batch_no_voice_gen_shipin/workspace_item_23397479040558/proj_item_23397479040558_1770000330_seg_0/character_refs/品茶女子.png").convert("RGB"),
    Image.open("/data/phd/lijiahui/data/batch_no_voice_gen_shipin/workspace_item_23397479040558/proj_item_23397479040558_1770000330_seg_0/reference_imgs/reference_img_0.png").convert("RGB"),
              Image.open("/data/phd/lijiahui/data/batch_no_voice_gen_shipin/workspace_item_23397479040558/proj_item_23397479040558_1770000330_seg_0/reference_imgs/reference_img_1.png").convert("RGB"),        
]

image = pipe(
    prompt,
    edit_image=edit_image,
    seed=42,
    num_inference_steps=40,
    height=1024,
    width=1024,
    # edit_image_auto_resize=True,
    zero_cond_t=True, # This is a special parameter introduced by Qwen-Image-Edit-2511
)
image.save("/data/phd/kousiqi/zhitao/shipin_data/shipin_23397479040558_consistency/test02.jpg")

# Qwen-Image-Edit-2511 is a multi-image editing model.
# Please use a list to input `edit_image`, even if the input contains only one image.
# edit_image = [Image.open("image.jpg")]
# Please do not input the image directly.
# edit_image = Image.open("image.jpg")
