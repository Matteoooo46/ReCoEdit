from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

MODEL_CONFIG_PATHS = {
    "dit": [
        "/path/to/data",
        "/path/to/data",
        "/path/to/data",
        "/path/to/data",
        "/path/to/data"
    ],
    "text_encoder": [
        "/path/to/data",
        "/path/to/data",
        "/path/to/data",
        "/path/to/data"
    ],
    "vae": "/path/to/data",

    "processor": "/path/to/data",
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
Image.open("/path/to/data").convert("RGB"),

Image.open("/path/to/data品茶女子.png").convert("RGB"),
    Image.open("/path/to/data").convert("RGB"),
              Image.open("/path/to/data").convert("RGB"),        
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
image.save("/path/to/data")

# Qwen-Image-Edit-2511 is a multi-image editing model.
# Please use a list to input `edit_image`, even if the input contains only one image.
# edit_image = [Image.open("image.jpg")]
# Please do not input the image directly.
# edit_image = Image.open("image.jpg")
