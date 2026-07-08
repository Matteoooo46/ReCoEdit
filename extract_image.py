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

openai_api_key_qw25 = "EMPTY"
openai_api_base_qw25 = "http://10.15.2.232:6628/v1"
# model_name_qw25 = "Qwen2.5-VL-32B-Instruct"
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
    # http_client=http_client
)

# sys.path.append("/data/zgq/nieuwland/src/t2i2v/t2i2v_project/t2i2v_pipeline")

# 引入你项目内部的函数
# from text_to_video_function import (
#     get_image_info_from_jsonl,
#     get_reference_image,
#     setup_logger,
#     set_proxy,
#     unset_proxy
# )

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
    """
    将 RGBA 图片转换为 RGB

    Args:
        image: PIL Image 对象
        background_color: 背景颜色 (R, G, B)

    Returns:
        RGB 模式的图片
    """
    if image.mode != 'RGBA':
        return image.convert('RGB')

    # 创建背景
    rgb_image = Image.new('RGB', image.size, background_color)
    # 使用 alpha 通道合成
    rgb_image.paste(image, mask=image.split()[3])
    return rgb_image

def resize_short_edge(input_path, output_path, short_edge=512):
    # 打开图片
    img = Image.open(input_path)
    w, h = img.size

    # 计算缩放比例
    if w < h:
        new_w = short_edge
        new_h = int(h * short_edge / w)
    else:
        new_h = short_edge
        new_w = int(w * short_edge / h)

    # resize 图像
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 把 rgba 模式转化为 rgb 模式
    if img_resized.mode == 'RGBA':
        img_resized = convert_rgba_to_rgb(img_resized)
    elif img_resized.mode not in ('RGB', 'L'):
        img_resized = img_resized.convert('RGB')

    # 保存图片
    img_resized.save(output_path)
    print(f"✅ 图片已保存到: {output_path}")

def download_images(image_urls, save_dir, host_prefix="https://s1-11661.kwimgs.com"):
    os.makedirs(save_dir, exist_ok=True)
    local_paths = []

    for idx, url in enumerate(image_urls):
        # if idx > 10:
        #     break
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
    """
    从商品JSON中提取卖点
    
    Args:
        json_data: 商品信息的JSON字符串或字典
    
    Returns:
        str: 用逗号拼接的卖点字符串
    """
    # 如果输入是字符串，先转换为字典
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    selling_points = []
    
    # 从 itemCategoryProp 中提取卖点
    item_category_props = data.get('itemCategoryProp', [])
    
    for prop in item_category_props:
        prop_name = prop.get('propName', '')
        propAlias = prop.get('propAlias', '')
        pvs = prop.get('pvs', [])
        
        # 重点关注的卖点属性
        if prop_name in ['品牌', '风格', '适用场景', '适用季节', '服饰功能', '流行元素'] or propAlias in ['功能']:
            for pv in pvs:
                prop_value_text = pv.get('propValueText', '')
                if prop_value_text:
                    # 对于服饰功能，添加属性名前缀
                    if prop_name == '服饰功能':
                        selling_points.append(f"{prop_value_text}")
                    elif prop_name == '品牌':
                        selling_points.append(f"{prop_value_text}品牌")
                    else:
                        selling_points.append(prop_value_text)
    
    # 去重并保持顺序
    unique_selling_points = []
    seen = set()
    for point in selling_points:
        if point not in seen:
            unique_selling_points.append(point)
            seen.add(point)
    
    # 用逗号拼接
    return ','.join(unique_selling_points)

def extract_script_and_image_urls(json_data ):
    script = None
    script2 = None
    script_add1 = None
    script_add2 = None
    script_add3 = None
    image_urls = set()

    def extract_recursive(obj):
        nonlocal script
        nonlocal script2
        nonlocal script_add1
        nonlocal script_add2
        nonlocal script_add3
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "itemTitle" and script is None and isinstance(value, str):
                    script = value
                if key == "itemId" and script2 is None and isinstance(value, str):
                    script2 = value
                if key == "first_industry_name" and script_add3 is None and isinstance(value, str):
                    script_add3 = value
                if key == "second_industry_name" and script_add1 is None and isinstance(value, str):
                    script_add1 = value
                if key == "sellingPoint" and script_add2 is None and isinstance(value, str):
                    script_add2 = value
                if key.lower() in {"url", "image", "src", "image_url"} and isinstance(value, str):
                    if re.search(r'\.(jpg|jpeg|png|gif|webp|svg|bmp|tiff)$', value, re.IGNORECASE):
                        image_urls.add(value)
                extract_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_recursive(item)

    extract_recursive(json_data)
    class_name = ""
    if not script_add1 and not script_add3:
        class_name = ""
    elif not script_add1:
        class_name = "first industry name is " + script_add3
    else:
        class_name = "first industry name is " + script_add3 + ", second industry name is " + script_add1 + "."
    return script, script2, list(image_urls), class_name, script_add2

def set_proxy():
    """
    设置 HTTP/HTTPS/NO_PROXY 环境变量。
    """
    os.environ["http_proxy"] = "http://oversea-squid2.ko.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid2.ko.txyun:11080"
    os.environ["no_proxy"] = (
        "localhost,127.0.0.1,localaddress,localdomain.com,internal,"
        "corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
    )
    
    print("代理已设置 ✅")

def unset_proxy():
    """
    移除 HTTP/HTTPS/NO_PROXY 环境变量。
    """
    for key in ["http_proxy", "https_proxy", "no_proxy"]:
        os.environ.pop(key, None)
    print("代理已取消 ❎")

def get_image_info_from_jsonl(json_data, save_root="./image_info" ):
    """
    处理 jsonl 文件，提取 script 和下载图片到本地
    参数:
        jsonl_path: str - jsonl 文件路径
        save_root: str - 存储根目录
    返回:
        images_path: str - 图片保存目录路径
        info: str - 提取的 script 文本
    """
    
    script_text, item_id, image_urls, script_add1, script_add2  = extract_script_and_image_urls(json_data)
    if not script_add2:
        script_add2 = extract_selling_points(json_data)
    else:
        script_add2 = ",".join([script_add2, extract_selling_points(json_data)])

    # os.environ['HTTP_PROXY'] = "http://oversea-squid2.ko.txyun:11080"
    # os.environ['HTTPS_PROXY'] = "http://oversea-squid2.ko.txyun:11080"

    
    images_path = os.path.join(save_root, "images")
    if not os.path.exists(images_path) or len(os.listdir(images_path)) < len(image_urls) * 0.8:
        image_files = download_images(image_urls, images_path)
    
    # os.environ.pop('HTTP_PROXY', None)
    # os.environ.pop('HTTPS_PROXY', None)

    return images_path, script_text or "[No script found]", item_id, script_add1, script_add2

def get_reference_image(images_dir: str , product_info: str):
    """
    从一个图片目录中筛选主图和细节图，返回主图路径。

    参数：images_dir: 包含图片的目录
    返回：reference_image_path: 主图路径
    """
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # 编码图片内容
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

    # 第一阶段：每6张图片选1张, 图片太多，所以分次数
    batch_size = 6
    selected_image_paths = []
    selected_image_contents = []
    prompt = System_Prompt_Reference_Image
    
    for i in range(0, len(image_files), batch_size):
        batch_images = image_files[i:i + batch_size]
        batch_contents = image_contents[i:i + batch_size]
        
        if len(batch_images) < 2:  # 如果最后一批少于2张，直接加入候选
            selected_image_paths.extend(batch_images)
            continue
            
            
        # 生成传给模型的信息
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": batch_contents + [{"type": "text", "text": f"请从以上{len(batch_images)}张图片中选择符合要求的主图和细节图。商品的标题为：'{product_info}'。"}],
            },
        ]

        # 调用模型
        _retry_num = 0
        answer = None
        
        while _retry_num < 3:
            try:
                response = qwen_vl_client.chat.completions.create(model=qwen_vl_model_name, messages=messages)  # temperature=0.5 
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

            # 匹配主图 "主图: X"
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

            # 匹配副图 "副图: X"
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

    # 第二阶段：从选出的图片中再选出最终的一张
    ref_images_path = []
    if len(selected_image_paths) == 1:
        # 如果只选出了一张，直接使用
        ref_images_path.append(selected_image_paths[0])
        logger.info('[+] 只选出一张图片，直接作参考图')
    elif len(selected_image_paths) > 1:        
        # 生成最终选择的信息
        final_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": selected_image_contents + [{"type": "text", "text": f"请从以上{len(selected_image_paths)}张候选图片中选择符合要求的主图和细节图。商品的标题为：'{product_info}'。"}],
            },
        ]

        # 调用模型进行最终选择
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

            # 匹配主图 "主图: X"
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

            # 匹配副图 "副图: X"
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

    # 清除旧的 handler，防止多次添加重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 文件输出
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 输出格式
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
    



class ImageExtractor():
    def __init__(self, root_dir="./workspace_extracted_images"):
        # 简化初始化，去掉了不需要的 Redis 和 Reviewer
        self.root_dir = root_dir

    def extract_item_image(self, item_id: int):
        """
        专门用来提取商品主图的函数
        返回: 提取到的参考图路径列表
        """
        # 开始处理
        # set_proxy()
        root_dir = self.root_dir
        os.makedirs(root_dir, exist_ok=True)
        
        # 找找是不是已经建过这个商品的文件夹了
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
            project_id = f"item_{item_id}_{str(uuid.uuid4())[:8]}" # 让文件夹名字带上 item_id，方便你找
            workspace = f"{root_dir}/{project_id}"
            os.makedirs(workspace, exist_ok=True)

        # 日志配置
        logger = setup_logger(os.path.join(workspace, 'process.log'))
        logger.info(f"========== 开始处理商品: {item_id} ==========")
        logger.info(f"工作目录: {workspace}")
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        
        # [0] 解析获取商品的信息
        logger.info("[0] 解析获取商品的信息...")
        item_info_json_path = os.path.join(workspace, "item_info.json")
        json_data = None
        if not os.path.exists(item_info_json_path):
            client = KwaishopProductItemCenterClient()
            resp = client._sync_run(item_id=int(item_id))
            resp = MessageToDict(resp)
            item_id_str = str(item_id)
            json_data = resp["itemInfo"][item_id_str]

            # 【修复点 1】：修正了原来代码中 as f:a 的语法错误
            with open(item_info_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        else:
            with open(item_info_json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        logger.info("商品信息已准备好。")

        # 【修复点 2】：跳过原代码的 [1] 获取客户私域素材 (不需要且缺少函数)

        # [2] 下载 item 商品信息并提取标题和卖点
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

        # [3] 获取参考图片 reference image
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
            # 调用底层算法筛选最好的图
            reference_img_paths = get_reference_image(images_path, product_info)
            
            for idx, reference_img_path in enumerate(reference_img_paths):
                reference_new = os.path.join(reference_img_dir, f"reference_img_{idx}.png")
                shutil.copy(reference_img_path, reference_new)
                reference_image_path_list.append(reference_new)

        logger.info(f"成功提取 {len(reference_image_path_list)} 张参考图！")
        # unset_proxy()
        
        # 【修复点 3】：必须把路径 return 出来，否则外部拿不到
        return reference_image_path_list


# ======================================================================
# 开始执行批量任务
# ======================================================================
if __name__ == '__main__':
    # 实例化我们精简好的类 (指定所有文件夹存放在 ./batch_product_images 下)
    extractor = ImageExtractor(root_dir="/data/phd/kousiqi/zhitao/batch_product_images")
    
    # 【修复点 4】：在这里粘贴你要处理的数据列表
    # 格式：[item_id, 一级行业名] (这里行业名用不上，但为了兼容你的数据格式保留着)
    my_data_list = [
        [26062938906514, '食品饮料'],
        [26073006797009, '电商平台'],
        # ... 在这里粘贴你几十上百个数据 ...
    ]
    
    print(f"📦 收到 {len(my_data_list)} 个商品的提取任务，准备起飞...\n")

    set_proxy()
    
    successful_count = 0
    
    for i, data in enumerate(my_data_list):
        current_item_id = data[0]
        print(f"[{i+1}/{len(my_data_list)}] 正在处理商品 ID: {current_item_id}")
        
        try:
            # 核心调用
            result_img_paths = extractor.extract_item_image(item_id=current_item_id)
            
            if result_img_paths:
                print(f"✅ 成功！图片保存在: {result_img_paths[0]}")
                successful_count += 1
            else:
                print(f"⚠️ 警告: 商品 {current_item_id} 未找到合适的参考图。")
                
        except Exception as e:
            print(f"❌ 处理商品 {current_item_id} 时发生致命错误: {e}")
            traceback.print_exc() # 打印详细报错方便排查
            
        print("-" * 50)

    unset_proxy()
        
    print(f"\n🎉 全部任务执行完毕！共成功提取 {successful_count}/{len(my_data_list)} 个商品。")
    print(f"所有的图片和数据都保存在: {extractor.root_dir} 目录下。")