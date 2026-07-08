"""Quick CoT scoring test — single product, 2 images."""
import os, base64, re, sys
from io import BytesIO
for v in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY'):
    os.environ.pop(v, None)
from openai import OpenAI
from PIL import Image

client = OpenAI(base_url='http://10.15.2.90:8080/v1', api_key='flowgrpo', timeout=120)

results_dir = '/data/phd/kousiqi/zhitao/qwen_inference_results_single'
dirs = [d for d in sorted(os.listdir(results_dir))
        if d.endswith('_data_enhanced_prompt_enhanced') and not d.startswith('raw')]
if not dirs:
    print("No product directories found!")
    sys.exit(1)

prod_dir = os.path.join(results_dir, dirs[0])
files = os.listdir(prod_dir)
ref = [f for f in files if f.startswith('p1_ref_')][0]
my_imgs = [f for f in files if f.startswith('p1_my_')][:2]

ref_img = Image.open(os.path.join(prod_dir, ref)).convert('RGB')
print(f'Testing product: {dirs[0][:60]}')
print(f'Ref: {ref}  |  Images: {my_imgs}\n')

def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return f'data:image;base64,{base64.b64encode(buf.getvalue()).decode()}'

SCORING_PROMPT = """你是一位专业的商品图像质量审核专家。请逐步分析两张图片中的产品外观一致性。

产品名称：{product_name}
图片1：产品参考图
图片2：生成的图片

按以下步骤简要分析（每步1-2句话，总共控制在200字以内）：
Step 1 — 描述图片1中产品的关键特征（颜色/款式/图案/logo）。
Step 2 — 图片2中是否有该产品？
Step 3 — 逐项对比关键特征是否匹配。
Step 4 — 结论。

规则：有明显颜色/款式/图案差异即为 No。若人物穿戴产品，只对比产品本身。
最后一行仅输出：Yes 或 No"""

for fname in my_imgs:
    gen_img = Image.open(os.path.join(prod_dir, fname)).convert('RGB')
    print(f'--- Scoring {fname} ---')
    try:
        resp = client.chat.completions.create(
            model='Qwen3-VL-30B-A3B-Instruct',
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': SCORING_PROMPT.format(product_name=dirs[0])},
                {'type': 'image_url', 'image_url': {'url': pil_to_base64(ref_img)}},
                {'type': 'image_url', 'image_url': {'url': pil_to_base64(gen_img)}},
            ]}],
            temperature=0.2,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        print(raw)
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        verdict = lines[-1] if lines else '(empty)'
        print(f'\n>>> 最终判决: {verdict}\n')
    except Exception as e:
        print(f'ERROR: {e}\n')

print('Done.')
