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
import pandas as pd

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
# 2. Gemini 人物 Prompt 生成核心依赖
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
    # ⚠️ 这里填入你的 Gemini API KEY (内网写字模型的Key)
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

    prompt = f"""You are a professional character designer. Based on the product info and reference images, design ONE main character for a highly realistic commercial video.
    
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


def generate_transfer_script_by_gemini(reference_json_data, product_desc, product_img_path, character_img_path):
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-pro:generateContent"
    API_KEY = "REDACTED"

    # 替换 Prompt 占位符
    prompt_text = System_Prompt_scene_plot_transfer_v2.replace(
        "{{reference_video_track}}", 
        json.dumps(reference_json_data, ensure_ascii=False, indent=2)
    ).replace(
        "{{product_name}}", 
        product_desc
    )
    prompt_text += "\n\n【附加指令】：本次请求附带了[新商品图]和[人物角色图]，请结合图片中的视觉特征辅助设计 first_frame 和 caption，严格遵守原有结构 1:1 复刻。"

    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    parts = [{"text": prompt_text}]
    
    # 依次加载商品图和人物图作为多模态输入
    image_paths = [img for img in [product_img_path, character_img_path] if img and os.path.exists(img)]
    for img_path in image_paths:
        ext = img_path.split(".")[-1].lower()
        mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        img_base64 = get_base64_encoded_image_gemini(img_path)
        if img_base64:
            parts.append({"inline_data": {"mime_type": mime_type, "data": img_base64}})

    payload = {"contents": [{"parts": parts}]}

    try:
        logger.info("🚀 正在调用 Gemini 结合图片进行分镜剧情迁移...")
        # ⚠️ 强制直连内网
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), proxies={"http": None, "https": None})
        response.raise_for_status()
        
        response_data = response.json()
        raw_answer = None
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            for part in response_data["candidates"][0].get("content", {}).get("parts", []):
                if "text" in part:
                    raw_answer = part["text"]
                    break
                    
        if not raw_answer: return None
            
        match = re.search(r'```json(.*?)```', raw_answer, re.S)
        json_text = match.group(1).strip() if match else raw_answer.strip()
        return json.loads(json_text)
        
    except Exception as e:
        logger.error(f"❌ 分镜剧情迁移失败: {e}")
        return None


# =====================================================================
# ★★★ 新增：基于 Novita API 的自动生图函数 ★★★
# =====================================================================
def generate_novita_image(prompt, api_key, save_path, reference_image_paths=None): # 1. 参数改为复数，预期接收列表
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 动态判断：如果传了参考图列表，且列表不为空，就走 Edit 垫图接口
    if reference_image_paths and isinstance(reference_image_paths, list) and len(reference_image_paths) > 0:
        url = "https://api.novita.ai/v3/gemini-2.5-flash-image-edit" 
        
        import base64
        img_base64_list = []
        # 2. 遍历你传进来的图片列表
        for img_path in reference_image_paths:
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    img_base64_list.append(f"data:image/png;base64,{img_b64}")
                    
        if img_base64_list:
            payload = {
                "prompt": prompt,
                "image_base64s": img_base64_list # 3. 把多张图的 Base64 数组全塞进去！
            }
        else:
            # 如果列表里的路径全失效了，防重保底：退回纯文生图
            url = "https://api.novita.ai/v3/gemini-2.5-flash-image-text-to-image"
            payload = {"prompt": prompt, "aspect_ratio": "9:16", "size": "1K"}
    else:
        url = "https://api.novita.ai/v3/gemini-2.5-flash-image-text-to-image" # 纯文生图接口
        payload = {
            "prompt": prompt,
            "aspect_ratio": "9:16", 
            "size": "1K"
        }
    
    try:
        start_time = time.time()
        # 注意：请求外网 API 时，一定要暂时关闭代理，或者确保代理畅通
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if "image_urls" in result and len(result["image_urls"]) > 0:
                image_url = result["image_urls"][0]
                img_response = requests.get(image_url, timeout=30)
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(img_response.content)
                logger.info(f"✅ Novita 生图成功！耗时 {time.time()-start_time:.1f}s")
                return True
            else:
                logger.error("Novita 响应中未找到 image_urls")
                return False
        else:
            logger.error(f"❌ Novita 生图接口报错: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ 请求 Novita 发生异常: {e}")
        return False


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

def download_images(image_urls, save_dir, host_prefix="https://s1-11661.kwimgs.com"):
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
    os.environ["http_proxy"] = "http://oversea-squid2.ko.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid2.ko.txyun:11080"
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


# 人物、场景参考图生成的格式
System_Prompt_scene_plot_v2 = """
你是一位专业的商品短视频导演和商品广告剧本作家。你的任务是根据下面提供的所有材料，创作一个专业的商品广告短视频剧本。
## 你将获得的输入材料
1. 商品图片: 一系列图片，包含模特展示图和纯商品图。
2. 商品广告创意标题: 一句简短的标题。
3. 优质素材结构化信息 (JSON): 一份详细的、结构化的营销卖点和创意参考信息。 

## 你的核心任务
你的首要任务是深度分析上述所有材料，并以此为核心依据，构思一个包含 剧情大纲、人物造型设定 和 6个分镜 的短视频剧本。剧本需要巧妙地将素材中提到的 用户痛点、产品功效、适用场景 等关键信息，通过视觉化的故事呈现出来。

第一步：构思剧情大纲 (outline)
1. 人物设定 (characters):
- 根据素材信息，设定核心人物。人物的设计应能反映出正在经历**“用户痛点”**的目标用户形象。  
- 为每个人物创建唯一的 人物ID (例如: "主角A")，并描述其基本特征。
- 如果故事完全没有人物，此项可以为空数组 []。
2. 场景设定 (scenes):
- 场景的设定必须主要参考素材信息中的**“拍摄场景”和“适用场景”**。场景要具体。
- 为每个场景创建唯一的 场景ID (例如: "场景1", "场景2")。
- 关键要求: 如果场景中会以陈设的方式出现输入图片中的商品（非人物穿戴），必须在描述中明确指出 具体是哪张图片中的商品 以及它在场景中的 空间位置关系。

第二步：人物造型设定 (characters_dressing)
此部分用于定义人物在不同分镜中的具体着装。
- 为每个造型创建一个唯一的 造型ID (例如: "造型A1")。
- 指明该造型属于哪个 人物ID。
- 详细描述这套造型的具体着装。
- 如果着装是输入图片中的商品，必须明确注明“参考输入图片[编号]” (编号从1开始)。
- 也可以是与商品无关的日常服装。

第三步：创作分镜剧本 (script)
核心创意要求: 6个分镜必须形成一个连贯的叙事，以视觉方式转化优质素材结构化信息中的核心逻辑。强烈建议遵循以下故事结构：
    - 呈现“用户痛点”: 开头展现用户在某个场景下的困扰。
    - 展示“产品功效”: 引入产品，并通过特写或对比演示其核心功能点。
    - 体现“适用场景”中的美好结果: 展示用户在使用产品后，在相应场景中获得了满足和愉悦的体验。
    - 强化信任与价值: 通过结尾镜头（例如产品特写、人物自信表情）来呼应**“促单成交”**中的信任感和价值感。
分镜格式: 每个分镜都是一个JSON对象，包含 shot_id, character_id, scene_id, product_image_number, description 字段。
    - character_id: 引用 'characters_dressing'中的造型ID，无人物则为 null。
    - product_image_number: 仅在无人物的空镜中用于指明场景中陈设的商品图片编号，否则为 null。
    - 视觉纯粹性: 严禁 在 description 中出现任何形式的文字信息。只关注视觉元素：人物、场景、动作、光影、色彩和氛围。

## 输出格式要求
你的最终输出必须是一个完整的 JSON 对象，包含三个顶级键："outline", "characters_dressing", 和 "script"。
## 输出格式示例 (基于拖鞋素材)

```json
{
  "outline": {
    "characters": [
      {
        "id": "主角A",
        "gender": "男性",
        "age": "28-35岁",
        "style": "注重生活品质的居家男士，对物品有洁癖",
        "figure": "标准身材"
      }
    ],
    "scenes": [
      {
        "id": "场景1",
        "description": "一间略显潮湿、光线昏暗的旧式卫生间，地砖接缝处有些许水渍。"
      },
      {
        "id": "场景2",
        "description": "一间现代、明亮、干爽的卫生间，有良好的通风和温暖的灯光。参考输入图片5中的EVA拖鞋被整齐地放置在淋浴区外。"
      }
    ]
  },
  "characters_dressing": [
    {
      "id": "造型A1",
      "character_id": "主角A",
      "description": "身穿一套深色的棉质居家服。"
    },
    {
      "id": "造型A2",
      "character_id": "主角A",
      "description": "穿着一件干净的白色浴袍，脚上穿着参考输入图片5的EVA拖鞋。"
    }
  ],
  "script": [
    {
      "shot_id": 1,
      "character_id": "造型A1",
      "scene_id": "场景1",
      "product_image_number": null,
      "description": "镜头从下往上摇，主角A皱着眉头，小心翼翼地抬起脚，他的旧拖鞋底部湿漉漉的，甚至有些发黑。他脸上露出嫌弃和困扰的表情，这是对传统拖鞋吸水发臭的痛点呈现。"
    },
    {
      "shot_id": 2,
      "character_id": null,
      "scene_id": "场景2",
      "product_image_number": [5],
      "description": "场景切换。一个干净的特写镜头，水流从花洒中喷涌而出，直接冲刷在EVA拖鞋上。水珠如同落在荷叶上一般迅速滑落，完全没有被吸收的迹象，直观展示其不吸水的物理特性。"
    },
    {
      "shot_id": 3,
      "character_id": null,
      "scene_id": "场景2",
      "product_image_number": [5],
      "description": "慢镜头特写。一把美工刀划过一块EVA材质样品，切口平滑且质地紧密，没有任何气孔或气泡。这个实验性画面，有力地展示了其高密度实心材质结构。"
    },
    {
      "shot_id": 4,
      "character_id": "造型A2",
      "scene_id": "场景2",
      "product_image_number": null,
      "description": "主角A刚洗完澡，穿着浴袍和干爽的EVA拖鞋，舒适地走出淋浴区。他踩在干燥的地面上，脸上是放松和满足的微笑，展现了产品在浴室这个适用场景中的完美体验。"
    },
    {
      "shot_id": 5,
      "character_id": "造型A2",
      "scene_id": "场景2",
      "product_image_number": null,
      "description": "主角A的脚部特写。他活动了一下脚趾，拖鞋柔软地贴合着他的脚型。镜头缓缓上移，最终定格在他充满信任和认可的眼神上，暗示了对比后建立的信任感。"
    },
    {
      "shot_id": 6,
      "character_id": null,
      "scene_id": "场景2",
      "product_image_number": [5],
      "description": "最终产品空镜。EVA拖鞋被放置在纯净、简约的背景前，柔和的灯光勾勒出其一体成型的设计和高品质的材质感。整个画面干净、专业，给人强烈的品质保障感。"
    }
  ]
}
```
"""

def generate_original_script_by_gemini(product_desc, product_img_path, character_img_path):
    """
    不依赖模板，直接根据商品信息、商品图和人物图生成分镜剧本
    """
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-pro:generateContent"
    API_KEY = "REDACTED"

    # 1. 组装 Prompt：将商品文案直接拼接到系统 Prompt 后面
    prompt_text = System_Prompt_scene_plot_v2 + f"\n\n【新商品文案信息】:\n{product_desc}"
    
    # 补充要求：明确告知大模型我们传了图片
    prompt_text += """
\n\n【附加视觉指令与一致性强化】：
1. 本次请求附带了[新商品首图]和[生成的模特定妆照]，请仔细观察图片中的人物外貌特征和商品细节。
2. 严禁生成任何卡通元素的描述。
3. 【核心指令：商品外观特征强注入】：在生成包含该商品的每一个分镜的 `description` 时，绝不能仅用简单的代词（如“商品”、“产品”、“鞋子”、“衣服”）带过。你**必须**从上述文案和图片中提取商品最核心的外观特征（如特定的颜色、标志性的材质、独特的形状设计、印花等），并强制写入该镜头的画面描述中。
   - 错误示范："特写镜头，主角A拿起了商品。"
   - 正确示范："特写镜头，主角A拿起了那件【雾霾蓝色的、表面带有细腻磨砂质感的、拉链处有银色金属环设计的】防晒衣。"
"""

    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    parts = [{"text": prompt_text}]
    
    # 2. 依次加载商品图和人物定妆照
    image_paths = [img for img in [product_img_path, character_img_path] if img and os.path.exists(img)]
    for img_path in image_paths:
        ext = img_path.split(".")[-1].lower()
        mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        img_base64 = get_base64_encoded_image_gemini(img_path)
        if img_base64:
            parts.append({"inlineData": {"mime_type": mime_type, "data": img_base64}})

    payload = {"contents": [{"parts": parts}]}

    # 3. 带有重试机制的请求调用
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🚀 正在调用 Gemini 直接创作分镜剧本 (尝试 {attempt + 1}/{max_retries})...")
            # ⚠️ 强制直连内网，绕过外网代理
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), proxies={"http": None, "https": None})
            response.raise_for_status()
            
            response_data = response.json()
            raw_answer = None
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                for part in response_data["candidates"][0].get("content", {}).get("parts", []):
                    if "text" in part:
                        raw_answer = part["text"]
                        break
                        
            if not raw_answer: 
                logger.error("❌ 模型未返回文本内容。")
                return None
                
            # 提取 JSON
            match = re.search(r'```json(.*?)```', raw_answer, re.S)
            json_text = match.group(1).strip() if match else raw_answer.strip()
            return json.loads(json_text)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [500, 502, 503, 504]:
                logger.warning(f"⚠️ 服务器繁忙 ({response.status_code})，等待 5 秒后重试...")
                time.sleep(5)
            else:
                logger.error(f"❌ 发生 HTTP 错误: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ 分镜剧本直接生成失败: {e}")
            return None
            
    logger.error("❌ 已达到最大重试次数，分镜生成失败。")
    return None

# ======================================================================
# 4. 全自动主循环：打通 提取 + 生成 Prompt + 自动生图
# ======================================================================
if __name__ == '__main__':
    # # 1. 使用 pandas 读取你的 CSV 文件
    # csv_file_path = "/data/phd/kousiqi/zhitao/可用的item.csv"
    # df = pd.read_csv(csv_file_path)
    
    # # 2. 过滤掉可能存在空值的无效行，并将 item_id 确保转换为整数类型
    # df = df.dropna(subset=['item_id', 'first_industry_name'])
    # df['item_id'] = df['item_id'].astype(int)
    
    # # 3. 自动将 DataFrame 转换为你需要的列表格式
    # my_data_list = df[['item_id', 'first_industry_name']].values.tolist()
    
    # # 打印前 5 个商品，检查格式是否正确
    # print(f"📦 成功读取 {len(my_data_list)} 个商品的任务！")
    # print(f"数据预览: {my_data_list[:5]}")
    
    extractor = ImageExtractor(root_dir="/data/phd/kousiqi/zhitao/batch_product_images")
    
    # 填入你要处理的商品列表
    my_data_list = [
        [26062938906514, '食品饮料'],
        [26073006797009, '电商平台'],
        [25688412798544, '日化'],
        [26099871476587, '食品饮料'],
        [26073148826708, '食品饮料'],
        [22161831549814, '日化'],
        [25917374915282, '美妆'],
        [26080096022033, '食品饮料'],
        [22715913571309, '母婴'],
        [26114033208077, '日化'],
        [23977521626106, '服装配饰'],
        [26066597478196, '服装配饰'],
        [24746892337721, '食品饮料'],
        [21829532608663, '日化'],
        [25570814654303, '美妆'],
        [24684391688695, '日化']
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
                        magic_suffix = ", Cinematic, Photorealistic, 4K Commercial, High Quality, bright and vibrant lighting, Portrait Mode (竖屏 9:16), Aspect Ratio: 9:16"
                        image_gen_prompt = f"{character_data['description']}{magic_suffix}"
                        
                        print(f"🎉 成功！为该商品设计的专属角色为: {char_name}")
                        print(f"🎯 Nano Banana 生图 Prompt: {image_gen_prompt[:80]}...")
                        
                        prompt_save_path = os.path.join(workspace, "character_gen_prompt.json")
                        with open(prompt_save_path, 'w', encoding='utf-8') as pf:
                            json.dump(character_data, pf, ensure_ascii=False, indent=4)
                        print(f"💾 专属 Prompt 已妥善保管在: {prompt_save_path}")
                        
                        # ================= 步骤 C：★★★ 无缝衔接自动生图 ★★★ =================
                        print(f"🎨 正在调用外网 Novita API 生成最终定妆照...")
                        
                        # ⚠️ 请确保这里替换成你真实的 Novita API KEY
                        NOVITA_API_KEY = "REDACTED"
                        
                        # 图片和 Prompt 存放在同级目录 (workspace 下)
                        final_image_save_path = os.path.join(workspace, "final_lifestyle_image.png")
                    
                        
                        # 开始生图
                        gen_success = generate_novita_image(
                            prompt=image_gen_prompt,
                            api_key=NOVITA_API_KEY,
                            save_path=final_image_save_path
                        )

                        
                        if gen_success:
                            print(f"🎉 终极成功！模特定妆照已生成并保存在同级目录: {final_image_save_path}")
                            
                            # =====================================================================
                            # 步骤 D：★★★ 自动化 T2I2V 流水线：从剧本到分镜图 ★★★
                            # =====================================================================
                            
                            # 1. 调用 Gemini 生成自由创新的分镜剧本 JSON
                            print(f"🎬 🎬 正在根据商品信息与图片，从零开始创作自由分镜剧本...")
                            new_script_json = generate_original_script_by_gemini(
                                product_desc=PRODUCT_TEXT,              # 提取的商品卖点
                                product_img_path=result_img_paths[0],   # 商品首图
                                character_img_path=final_image_save_path# 刚生成的模特图
                            )
                            
                            if new_script_json:
                                magic_suffix = ", Cinematic, Photorealistic, 4K Commercial, High Quality, bright and vibrant lighting, Portrait Mode (竖屏 9:16), Aspect Ratio: 9:16"
                                # 找到 JSON 里的 script 列表，遍历它
                                if isinstance(new_script_json, dict) and "script" in new_script_json:
                                    for scene in new_script_json["script"]:
                                        if isinstance(scene, dict) and "description" in scene:
                                            # 把旧的 description 拿出来，拼上后缀，再塞回去！
                                            original_desc = scene["description"]
                                            scene["description"] = f"{original_desc}{magic_suffix}"
                                            
                                # 如果模型很乖，直接给的列表，那就直接遍历列表
                                elif isinstance(new_script_json, list):
                                    for scene in new_script_json:
                                        if isinstance(scene, dict) and "description" in scene:
                                            original_desc = scene["description"]
                                            scene["description"] = f"{original_desc}{magic_suffix}"

                                # =========================================================
                                # 带着魔法后缀的全新 JSON，现在可以放心保存了！
                                # =========================================================
                                script_save_path = os.path.join(workspace, "final_storyboard_script.json")
                                with open(script_save_path, "w", encoding="utf-8") as f:
                                    json.dump(new_script_json, f, ensure_ascii=False, indent=2)
                                print(f"🎉 剧本生成完毕！成品（已包含画风约束）已保存至: {script_save_path}")
                                
                                # ===============================================================
                                # 💥 核心步骤：★★★ 开始循环生成 6 张分镜图 ★★★
                                # ===============================================================
                                scenes_list = []
                                if isinstance(new_script_json, dict):
                                    # 【核心修复 1】：精准提取 "script" 字段！绝不瞎抓！
                                    if "script" in new_script_json and isinstance(new_script_json["script"], list):
                                        scenes_list = new_script_json["script"]
                                    else:
                                        # 万一模型没按规矩叫 script，我们找最长的那个列表（肯定是分镜）
                                        for key, value in new_script_json.items():
                                            if isinstance(value, list) and len(value) > len(scenes_list):
                                                scenes_list = value
                                elif isinstance(new_script_json, list):
                                    scenes_list = new_script_json
                                    
                                if not scenes_list:
                                    scenes_list = [new_script_json]
                                    
                                # 【核心修复 2】：打印真正的分镜数量，而不是字典的 key 数量
                                print(f"🖼️  🖼️  正在根据剧本，自动循环生成分镜图片 (共 {len(scenes_list)} 个镜头)...")
                                
                                storyboard_images_save_dir = os.path.join(workspace, "storyboard_images")
                                os.makedirs(storyboard_images_save_dir, exist_ok=True)
                                
                                # 🌟🌟🌟 核心改造：使用列表维护历史帧，滑动窗口大小为 2 🌟🌟🌟
                                history_frames = [] 
                                
                                for idx, scene in enumerate(scenes_list):
                                    if not isinstance(scene, dict):
                                        continue
                                        
                                    scene_num = idx + 1
                                    print(f"   👉 正在生成镜头 [{scene_num}/{len(scenes_list)}] 的图片...")
                                    
                                    scene_prompt = scene.get('description', '')
                                    if not scene_prompt:
                                        print(f"   ⚠️ 镜头 {scene_num} 缺少 'description'，跳过。")
                                        continue
                                        
                                    # =========================================================
                                    # ★★★ 核心中枢：多图融合垫图策略 (2张历史帧 + 定妆/原图) ★★★
                                    # =========================================================
                                    char_id = scene.get('character_id')
                                    prod_nums = scene.get('product_image_number')
                                    
                                    ref_imgs_for_novita = []
                                    
                                    # 策略 1：先将缓存池中的历史帧（最多2张）加入垫图列表
                                    for hist_img in history_frames:
                                        if os.path.exists(hist_img):
                                            ref_imgs_for_novita.append(hist_img)
                                    
                                    if history_frames:
                                        print(f"      🔍 剧本指令：已垫入【{len(history_frames)} 张历史帧】保持环境与动作连贯")
                                        
                                    # 策略 2：再根据当前帧的剧本需求，加入绝对锚点（定妆照或商品原图）
                                    if char_id:
                                        ref_imgs_for_novita.append(final_image_save_path)
                                        print(f"      🔍 剧本指令：垫入【人物定妆照】死锁人脸和服装")
                                    elif prod_nums:
                                        ref_imgs_for_novita.append(result_img_paths[0])
                                        print(f"      🔍 剧本指令：垫入【商品原图】死锁商品细节")
                                        
                                    if not ref_imgs_for_novita:
                                        print("      🔍 剧本指令：无任何参考图，走纯文本生图")

                                    scene_image_save_path = os.path.join(storyboard_images_save_dir, f"scene_{scene_num}.png")
                                    
                                    # 调用 Novita 多图合并生图接口
                                    scene_gen_success = generate_novita_image(
                                        prompt=f"画面描述: {scene_prompt}", 
                                        api_key=NOVITA_API_KEY,
                                        save_path=scene_image_save_path,
                                        reference_image_paths=ref_imgs_for_novita # 这里需要是你上一步改好的支持列表的函数
                                    )
                                    
                                    if scene_gen_success:
                                        print(f"      ✅ 镜头 {scene_num} 生图成功！")
                                        
                                        # 🌟🌟🌟 动态更新历史帧队列 🌟🌟🌟
                                        # 将当前成功生成的帧加入队尾
                                        history_frames.append(scene_image_save_path)
                                        
                                        # 如果队列长度超过 2，就把最老的一张图（索引为0）踢出去
                                        if len(history_frames) > 2:
                                            history_frames.pop(0)
                                    else:
                                        print(f"      ❌ 镜头 {scene_num} 生图失败。")
                                        
                                print(f"🎉 💥 💥 此商品的端到端流水线彻底完成！成品已汇总至文件夹: {workspace}")
                                successful_count += 1
                                
                            else:
                                print("❌ 分镜剧本生成失败，后续生图流程已跳过。")
                        else:
                            print(f"❌ 定妆照生图失败，请检查报错日志。")
                            
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
        
    print(f"\n🚀 全部流水线结束！共完成 {successful_count}/{len(my_data_list)} 个商品的“图片提取 + 文案提取 + 定妆照一键生成”！")