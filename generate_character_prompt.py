import os, traceback
import uuid
import json
import shutil
import logging
import sys
import re
from urllib.parse import urljoin
import glob
import requests
from PIL import Image
import base64
import time
from loguru import logger 
from openai import OpenAI
import httpx

from google.protobuf.json_format import MessageToDict
from video_graph.common.client.kwaishop_product_item_center_client import KwaishopProductItemCenterClient

# =====================================================================
# 1. 大模型配置 (完全保留你原来的所有配置)
# =====================================================================
openai_api_key_qw25 = "EMPTY"
openai_api_base_qw25 = "http://10.15.2.232:6628/v1"
model_name_qw25 = "/data/liumingda/qwen2.5vl/weight-32b/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/6bcf1c9155874e6961bcf82792681b4f4421d2f7"
client_qw25 = OpenAI(api_key=openai_api_key_qw25, base_url=openai_api_base_qw25)
client_qw25._client.timeout = httpx.Timeout(180.0)

client_qw3 = OpenAI(base_url=f"http://10.82.237.219:8080/v1", api_key="None")
client_qw3._client.timeout = httpx.Timeout(180.0)
model_name_qw3 = "Qwen3-VL-30B-A3B-Instruct"

qwen_vl_client, qwen_vl_model_name = client_qw3, model_name_qw3

# 配置 deepseekr1 文本模型客户端
base_url = "https://qianfan.baidubce.com/v2"
api_key = "REDACTED"
appid = "app-H8t9051I"

client_deepseekr1 = OpenAI(
    base_url=base_url,
    api_key=api_key,
    default_headers={"appid": appid},
)

# =====================================================================
# 2. Gemini 人物 Prompt 生成核心依赖 (新增)
# =====================================================================
def get_base64_encoded_image_gemini(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"图片读取失败 {image_path}: {e}")
        return None

def parse_image_by_gemini(image_paths, prompt_gemini):
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-pro:generateContent"
    # ⚠️ 这里填入你的 Gemini API KEY
    API_KEY = "REDACTED" 
    
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    parts = []
    
    if isinstance(prompt_gemini, str):
        parts.append({"text": prompt_gemini})
    elif isinstance(prompt_gemini, list):
        parts.append({"text": " ".join(prompt_gemini)})
        
    for img_path in image_paths:
        ext = img_path.split(".")[-1].lower()
        MIME_TYPE = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        img_base64 = get_base64_encoded_image_gemini(img_path)
        if img_base64:
            parts.append({"inline_data": {"mime_type": MIME_TYPE, "data": img_base64}})
            
    payload = {"contents": [{"parts": parts}]}
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()
        raw_answer = None
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            for part in response_data["candidates"][0].get("content", {}).get("parts", []):
                if "text" in part: return part["text"]
        return None
    except Exception as e:
        logger.error(f"API请求发生错误: {e}")
        return None

def get_manju_character_by_gemini(reference_img_paths, product_info, existing_characters=None):
    context_str = ""
    if existing_characters:
        context_str = "\n\n# Existing Characters from previous segments:\n"
        for name, desc in existing_characters.items():
            context_str += f"- {name}: {desc}\n"
        context_str += "\nIMPORTANT: If the character in the current segment matches any of the above, MUST use the EXACT same name."

    prompt = f"""You are a professional character designer. Based on the product info and reference images, design ONE main character for a video.
    
    # Product Info:
    {product_info}
    {context_str}
    
    # Task:
    1. Identify the most suitable main character.
    2. Provide a name and a highly detailed visual description (age, clothing, hairstyle, personality). The description MUST be in English and suitable for AI image generation.
    3. If this character seems to be the same as one of the 'Existing Characters', use that name.
    
    # Output Format (Strict JSON):
    {{
        "name": "Character Name",
        "description": "Detailed visual description in English for AI image generation (e.g., A 25-year-old Chinese female, long wavy hair, wearing a white casual t-shirt, elegant and energetic...)"
    }}
    """
    
    raw_answer = parse_image_by_gemini(reference_img_paths, prompt)
    if not raw_answer: return None
        
    try:
        match = re.search(r'```json(.*?)```', raw_answer, re.S)
        json_text = match.group(1).strip() if match else raw_answer.strip()
        return json.loads(json_text)
    except Exception as e:
        logger.error(f"JSON 解析失败: {e}\n原文: {raw_answer}")
        return None


# =====================================================================
# 3. 原封不动保留的所有提取工具函数与 Qwen 选图逻辑
# =====================================================================
System_Prompt_Reference_Image = '''
你是一位专业的广告图像内容分析专家。请根据输入的商品广告图像序列，选择2张用于宣传的商品图像，图片不要有宣传文字。

**选择主图（商品整体图）**
- 主图应包含完整商品，背景简单。
- 优先级1：尽量选择无广告文字的图片作为主图。 
- 优先级2：尽量选择有模特穿戴商品的图作为主图。
- 优先级3：尽量选择商品的正面图作为主图。
- 补充：如果所有图片都有文字（包括商品表面的 logo 或花纹），请选出文字最少且清晰完整的图片作为主图。
- 主图必须选且只能选一张。

**选择副图（商品细节图）**
- 副图应包含商品的局部细节信息。
- 请尽量选择无文字的细节图。
- 如果所有细节图都有文字，选出文字最少且细节清晰的图片。
- 如果找不到满足要求的副图，副图序号可为 -1。

**严格要求**
- 禁止选取无商品的图！
- 选取的图必须包含目标商品！
- 输入图像的编号从 **1** 开始，按输入顺序依次编号：1、2、3、4、……

**输出格式要求（必须严格遵守）**
请仅输出一个合法的 JSON 对象，不要包含多余文字、注释或说明。

JSON 结构如下：
```json
{
    "主图": <int>,        // 主图的图片编号（从1开始）
    "副图": <int>,        // 副图的图片编号（从1开始，若无合适副图则为 -1）
    "理由": {
        "主图": "<string>",   // 说明为什么选择该图作为主图
        "副图": "<string>"    // 说明为什么选择该图作为副图
    }
}
'''

def convert_rgba_to_rgb(image, background_color=(255, 255, 255)):
    if image.mode != 'RGBA':
        return image.convert('RGB')
    rgb_image = Image.new('RGB', image.size, background_color)
    rgb_image.paste(image, mask=image.split()[3])
    return rgb_image

def resize_short_edge(input_path, output_path, short_edge=512):
    img = Image.open(input_path)
    w, h = img.size
    if w < h:
        new_w = short_edge
        new_h = int(h * short_edge / w)
    else:
        new_h = short_edge
        new_w = int(w * short_edge / h)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if img_resized.mode == 'RGBA':
        img_resized = convert_rgba_to_rgb(img_resized)
    elif img_resized.mode not in ('RGB', 'L'):
        img_resized = img_resized.convert('RGB')
    img_resized.save(output_path)
    print(f"✅ 图片已保存到: {output_path}")

def download_images(image_urls, save_dir, host_prefix="[https://s1-11661.kwimgs.com](https://s1-11661.kwimgs.com)"):
    os.makedirs(save_dir, exist_ok=True)
    local_paths = []
    for idx, url in enumerate(image_urls):
        full_url = urljoin(host_prefix, url) if not url.startswith("http") else url
        ext = os.path.splitext(full_url)[1]
        save_path = os.path.join(save_dir, f"image_{idx+1}{ext}")
        try:
            response = requests.get(full_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                local_paths.append(save_path)
            else:
                print(f"[Warning] Failed to download {full_url}, status: {response.status_code}")
        except Exception as e:
            print(f"[Error] Exception when downloading {full_url}: {e}")
    return local_paths

def extract_selling_points(json_data):
    if isinstance(json_data, str): data = json.loads(json_data)
    else: data = json_data
    selling_points = []
    item_category_props = data.get('itemCategoryProp', [])
    for prop in item_category_props:
        prop_name = prop.get('propName', '')
        propAlias = prop.get('propAlias', '')
        pvs = prop.get('pvs', [])
        if prop_name in ['品牌', '风格', '适用场景', '适用季节', '服饰功能', '流行元素'] or propAlias in ['功能']:
            for pv in pvs:
                prop_value_text = pv.get('propValueText', '')
                if prop_value_text:
                    if prop_name == '服饰功能': selling_points.append(f"{prop_value_text}")
                    elif prop_name == '品牌': selling_points.append(f"{prop_value_text}品牌")
                    else: selling_points.append(prop_value_text)
    unique_selling_points = []
    seen = set()
    for point in selling_points:
        if point not in seen:
            unique_selling_points.append(point)
            seen.add(point)
    return ','.join(unique_selling_points)

def extract_script_and_image_urls(json_data ):
    script = None
    script2 = None
    script_add1 = None
    script_add2 = None
    script_add3 = None
    image_urls = set()

    def extract_recursive(obj):
        nonlocal script, script2, script_add1, script_add2, script_add3
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "itemTitle" and script is None and isinstance(value, str): script = value
                if key == "itemId" and script2 is None and isinstance(value, str): script2 = value
                if key == "first_industry_name" and script_add3 is None and isinstance(value, str): script_add3 = value
                if key == "second_industry_name" and script_add1 is None and isinstance(value, str): script_add1 = value
                if key == "sellingPoint" and script_add2 is None and isinstance(value, str): script_add2 = value
                if key.lower() in {"url", "image", "src", "image_url"} and isinstance(value, str):
                    if re.search(r'\.(jpg|jpeg|png|gif|webp|svg|bmp|tiff)$', value, re.IGNORECASE):
                        image_urls.add(value)
                extract_recursive(value)
        elif isinstance(obj, list):
            for item in obj: extract_recursive(item)

    extract_recursive(json_data)
    class_name = ""
    if not script_add1 and not script_add3: class_name = ""
    elif not script_add1: class_name = "first industry name is " + script_add3
    else: class_name = "first industry name is " + script_add3 + ", second industry name is " + script_add1 + "."
    return script, script2, list(image_urls), class_name, script_add2

def set_proxy():
    os.environ["http_proxy"] = "[http://oversea-squid2.ko.txyun:11080](http://oversea-squid2.ko.txyun:11080)"
    os.environ["https_proxy"] = "[http://oversea-squid2.ko.txyun:11080](http://oversea-squid2.ko.txyun:11080)"
    os.environ["no_proxy"] = (
        "localhost,127.0.0.1,localaddress,localdomain.com,internal,"
        "corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
    )
    print("代理已设置 ✅")

def unset_proxy():
    for key in ["http_proxy", "https_proxy", "no_proxy"]:
        os.environ.pop(key, None)
    print("代理已取消 ❎")

def get_image_info_from_jsonl(json_data, save_root="./image_info" ):
    script_text, item_id, image_urls, script_add1, script_add2  = extract_script_and_image_urls(json_data)
    if not script_add2:
        script_add2 = extract_selling_points(json_data)
    else:
        script_add2 = ",".join([script_add2, extract_selling_points(json_data)])
    
    images_path = os.path.join(save_root, "images")
    if not os.path.exists(images_path) or len(os.listdir(images_path)) < len(image_urls) * 0.8:
        image_files = download_images(image_urls, images_path)
    
    return images_path, script_text or "[No script found]", item_id, script_add1, script_add2

def get_reference_image(images_dir: str , product_info: str):
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    image_contents = []
    for img_path in image_files:
        temp_dir = os.path.join(os.path.dirname(img_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        tmp_img_path = os.path.join(temp_dir, os.path.basename(img_path))
        resize_short_edge(img_path, tmp_img_path)
        img_path = tmp_img_path

        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        base64_url = f"data:{mime};base64,{encoded}"
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": base64_url}
        })

    batch_size = 6
    selected_image_paths = []
    selected_image_contents = []
    prompt = System_Prompt_Reference_Image
    
    for i in range(0, len(image_files), batch_size):
        batch_images = image_files[i:i + batch_size]
        batch_contents = image_contents[i:i + batch_size]
        
        if len(batch_images) < 2:  
            selected_image_paths.extend(batch_images)
            continue
            
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": batch_contents + [{"type": "text", "text": f"请从以上{len(batch_images)}张图片中选择符合要求的主图和细节图。商品的标题为：'{product_info}'。"}],
            },
        ]

        _retry_num = 0
        answer = None
        
        while _retry_num < 3:
            try:
                response = qwen_vl_client.chat.completions.create(model=qwen_vl_model_name, messages=messages)  
                answer = response.choices[0].message.content if hasattr(response, "choices") else ""
                answer = answer.strip("```json").strip("```").strip()
                logger.info(f'[+] 第{i//batch_size + 1}批选图回答内容：{answer}')
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing images: {e}")

            if not answer:
                _retry_num += 1
                time.sleep(5)
                continue
            break
        
        if not answer:
            logger.error(f'[!] 调用模型挑选第{i//batch_size + 1}批主图失败，选择第一张图片')
            selected_image_paths.append(batch_images[0])
        else:
            main_idx = sub_idx = None
            answer = json.loads(answer)
            main_idx = answer["主图"]
            logger.info(f'[+] 通过"主图:X"格式匹配到序号: {main_idx}')
            
            if main_idx and 1 <= main_idx <= len(batch_images):
                selected_image = batch_images[main_idx - 1]
                selected_content = batch_contents[main_idx - 1]
                selected_image_paths.append(selected_image)
                selected_image_contents.append(selected_content)
                logger.info(f'[+] 第{i//batch_size + 1}批选中图片：{main_idx}')
            else:
                logger.error(f"[!] 第{i//batch_size + 1}批无法解析有效序号，选择第一张图片")
                main_idx = 1
                selected_image_paths.append(batch_images[0])
                selected_image_contents.append(batch_contents[0])

            sub_idx = answer["副图"]
            logger.info(f'[+] 通过"副图:X"格式匹配到序号: {sub_idx}')
            
            if sub_idx and 1 <= sub_idx <= len(batch_images) and main_idx != sub_idx:
                selected_image = batch_images[sub_idx - 1]
                selected_content = batch_contents[sub_idx - 1]
                selected_image_paths.append(selected_image)
                selected_image_contents.append(selected_content)
                logger.info(f'[+] 第{i//batch_size + 1}批选中图片：{sub_idx}')
            else:
                if main_idx != 1:
                    logger.error(f"[!] 第{i//batch_size + 1}批无法解析有效序号，选择第一张图片")
                    selected_image_paths.append(batch_images[0])
                    selected_image_contents.append(batch_contents[0])

    ref_images_path = []
    if len(selected_image_paths) == 1:
        ref_images_path.append(selected_image_paths[0])
        logger.info('[+] 只选出一张图片，直接作参考图')
    elif len(selected_image_paths) > 1:        
        final_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": selected_image_contents + [{"type": "text", "text": f"请从以上{len(selected_image_paths)}张候选图片中选择符合要求的主图和细节图。商品的标题为：'{product_info}'。"}],
            },
        ]
        _retry_num = 0
        final_answer = None
        while _retry_num < 3:
            response = qwen_vl_client.chat.completions.create(model=qwen_vl_model_name, messages=final_messages) 
            final_answer = response.choices[0].message.content if hasattr(response, "choices") else ""
            final_answer = final_answer.strip("```json").strip("```").strip()
            if not final_answer:
                _retry_num += 1
                time.sleep(5)
                continue
            break
        
        if not final_answer:
            logger.error('[!] 调用模型进行最终选择失败，选择第一张候选图片作为主图')
            ref_images_path.append(selected_image_paths[0])
        else:
            main_idx = sub_idx = None
            answer = json.loads(final_answer)

            main_idx = answer["主图"]
            logger.info(f'[+] 通过"主图:X"格式匹配到序号: {main_idx}')
            
            if main_idx and 1 <= main_idx <= len(batch_images):
                selected_image = selected_image_paths[main_idx - 1]
                ref_images_path.append(selected_image)
                logger.info(f'[+] 第{i//batch_size + 1}批选中图片：{main_idx}')
            else:
                logger.error(f"[!] 第{i//batch_size + 1}批无法解析有效序号，选择第一张图片")
                main_idx = 1
                selected_image = selected_image_paths[0]
                ref_images_path.append(selected_image)

            sub_idx = answer["副图"]
            logger.info(f'[+] 通过"副图:X"格式匹配到序号: {sub_idx}')
            
            if sub_idx and 1 <= sub_idx <= len(batch_images) and main_idx != sub_idx:
                selected_image = selected_image_paths[sub_idx - 1]
                ref_images_path.append(selected_image)
                logger.info(f'[+] 第{i//batch_size + 1}批选中图片：{sub_idx}')
            else:
                if main_idx != 1:
                    logger.error(f"[!] 第{i//batch_size + 1}批无法解析有效序号，选择第一张图片")
                    selected_image = selected_image_paths[0]
                    ref_images_path.append(selected_image)
    else:
        logger.error('[!] 没有选出任何候选图片，选择第一张原始图片作为主图')
        ref_images_path.append(image_files[0])

    return ref_images_path

def setup_logger(log_file='output.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

class ImageExtractor():
    def __init__(self, root_dir="./workspace_extracted_images"):
        self.root_dir = root_dir

    def extract_item_image(self, item_id: int):
        root_dir = self.root_dir
        os.makedirs(root_dir, exist_ok=True)
        
        project_id = workspace = None
        for project_name in os.listdir(root_dir):
            workspace = f"{root_dir}/{project_name}"
            json_path = f"{workspace}/item_info.json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if int(data["itemId"]) == item_id:
                            project_id = project_name
                            break
                except Exception:
                    pass
                    
        if project_id is None:
            project_id = f"item_{item_id}_{str(uuid.uuid4())[:8]}" 
            workspace = f"{root_dir}/{project_id}"
            os.makedirs(workspace, exist_ok=True)

        logger = setup_logger(os.path.join(workspace, 'process.log'))
        logger.info(f"========== 开始处理商品: {item_id} ==========")
        logger.info(f"工作目录: {workspace}")
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        
        logger.info("[0] 解析获取商品的信息...")
        item_info_json_path = os.path.join(workspace, "item_info.json")
        json_data = None
        if not os.path.exists(item_info_json_path):
            client = KwaishopProductItemCenterClient()
            resp = client._sync_run(item_id=int(item_id))
            resp = MessageToDict(resp)
            item_id_str = str(item_id)
            json_data = resp["itemInfo"][item_id_str]

            with open(item_info_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        else:
            with open(item_info_json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        logger.info("商品信息已准备好。")

        logger.info("[2] 下载素材并提取商品信息...")
        extracted_item_info_json_path = f"{workspace}/extracted_item_info.json"
        
        images_path, product_info, item_id_ret, industry_name, selling_point = get_image_info_from_jsonl(json_data, workspace)
        
        if not os.path.exists(extracted_item_info_json_path):
            data = {
                "product_info": product_info,
                "item_id": item_id_ret,
                "selling_point": selling_point,
                "industry_name": industry_name
            }
            with open(extracted_item_info_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info("[3] 智能提取参考图片 (reference image)...")
        reference_img_dir = os.path.join(workspace, "reference_imgs")
        reference_image_path_list = []
        
        if os.path.exists(reference_img_dir) and len(os.listdir(reference_img_dir)) > 0:
            reference_image_path_list = [
                os.path.join(reference_img_dir, p) for p in sorted(os.listdir(reference_img_dir))
                if p.split(".")[-1] in ("png", "jpg", "jpeg")
            ]
        else:
            os.makedirs(reference_img_dir, exist_ok=True)
            reference_img_paths = get_reference_image(images_path, product_info)
            
            for idx, reference_img_path in enumerate(reference_img_paths):
                reference_new = os.path.join(reference_img_dir, f"reference_img_{idx}.png")
                shutil.copy(reference_img_path, reference_new)
                reference_image_path_list.append(reference_new)

        logger.info(f"成功提取 {len(reference_image_path_list)} 张参考图！")
        
        # 将 workspace 也返回出来，供后面大模型提取文案用
        return reference_image_path_list, workspace


# ======================================================================
# 4. 全自动主循环：打通 提取 + 生成 Prompt
# ======================================================================
if __name__ == '__main__':
    extractor = ImageExtractor(root_dir="/data/phd/kousiqi/zhitao/batch_product_images")
    
    # 填入你要处理的商品列表
    my_data_list = [
        [26062938906514, '食品饮料'],
        [26073006797009, '电商平台'],
    ]
    
    print(f"📦 收到 {len(my_data_list)} 个商品的任务，即将执行端到端自动化...\n")

    set_proxy()
    successful_count = 0
    
    for i, data in enumerate(my_data_list):
        current_item_id = data[0]
        print(f"[{i+1}/{len(my_data_list)}] 正在处理商品 ID: {current_item_id}")
        
        try:
            # ================= 步骤 A：提取数据与图片 =================
            result_img_paths, workspace = extractor.extract_item_image(item_id=current_item_id)
            
            if result_img_paths and workspace:
                print(f"✅ 图片提取成功！保存在: {result_img_paths[0]}")
                
                # ================= 步骤 B：自动无缝衔接生成 Prompt =================
                print(f"🤖 正在调用 Gemini 结合提取的商品信息自动生成定妆照 Prompt...")
                
                # 1. 自动读取刚提取的 JSON 组合文字
                info_json_path = os.path.join(workspace, "extracted_item_info.json")
                if os.path.exists(info_json_path):
                    with open(info_json_path, 'r', encoding='utf-8') as f:
                        item_info = json.load(f)
                        
                    title = item_info.get('product_info', '')
                    sp = item_info.get('selling_point', '')
                    PRODUCT_TEXT = f"商品标题：{title}\n商品卖点：{sp}"

                    
                    # 2. 发给大模型生成
                    character_data = get_manju_character_by_gemini(
                        reference_img_paths=result_img_paths,
                        product_info=PRODUCT_TEXT
                    )

                    
                    # 3. 解析与保存
                    if character_data and "description" in character_data:
                        char_name = character_data.get("name", "Unknown")
                        image_gen_prompt = character_data["description"]
                        
                        print(f"🎉 成功！为该商品设计的专属角色为: {char_name}")
                        print(f"🎯 Nano Banana 生图 Prompt: {image_gen_prompt[:80]}...")
                        
                        prompt_save_path = os.path.join(workspace, "character_gen_prompt.json")
                        with open(prompt_save_path, 'w', encoding='utf-8') as pf:
                            json.dump(character_data, pf, ensure_ascii=False, indent=4)
                        print(f"💾 专属 Prompt 已妥善保管在: {prompt_save_path}")
                        
                        successful_count += 1
                    else:
                        print("❌ Gemini 响应失败，未能生成人物特征。")
                else:
                    print(f"⚠️ 找不到提取的商品信息文件，跳过 Prompt 生成。")

            else:
                print(f"⚠️ 警告: 商品 {current_item_id} 未能成功提取数据。")
                
        except Exception as e:
            print(f"❌ 处理商品 {current_item_id} 时发生错误: {e}")
            traceback.print_exc()
            
        print("-" * 50)

    unset_proxy()
        
    print(f"\n🚀 全部流水线结束！共完成 {successful_count}/{len(my_data_list)} 个商品的“图片提取+文案提取+定妆照Prompt生成”！")