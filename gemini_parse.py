#from google import genai
import google.generativeai as genai
# from google.genai import types
# from genai import types
import requests
from loguru import logger
import requests
import json, os
import base64
import io, pdb
from PIL import Image
import re
import logging
import requests
import glob

# --- 配置API ---
API_KEY = "REDACTED"  # 你的内部网关密钥


System_Prompt_scene_plot_motes_with_scene_identical = """
# 土味儿电商服饰短视频剧本创作指令

## 角色定位
你是一位深谙当代中国县城市场审美的短视频导演，擅长运用生动、接地气的镜头语言，营造充满当代县城生活气息的动态场景。

## 核心任务
基于提供的**单张模特商品图**，构思一个由6个分镜组成的短视频剧本。**所有分镜必须使用参考图中的同一场景、同一模特和同一套服装**，仅通过模特的不同动作来展示服装，确保视频画面中的服装与模特商品图片中的服装保持完全一致。

## 核心创作原则

### 场景与人物一致性
1. **场景锁定**：所有分镜必须使用**参考图片中的完全相同的背景场景**，不得更换或修改场景环境
2. **模特与服装一致性**：所有分镜必须保持**参考图中完全相同的模特面部特征和服装**

### 动作设计原则
3. **动作多样化**：6个分镜必须包含**6种完全不同类型**的动作，避免重复或相似
4. **动态感强化**：每个动作都应体现**动作的连贯性和自然流动感**
5. **幅度差异化**：动作设计要包含不同幅度和方向的变化
6. **生活化表达**：动作要体现县城年轻人的真实日常行为

### 服装展示要求
7. **多角度展示**：通过动作自然展示服装的前后左右各个角度
8. **细节凸显**：包含一个自然的材质细节展示动作
9. **实用场景**：动作要体现服装在日常生活中的穿着状态

### 镜头语言
10. 以中景（全身）和近景（上半身）为主
11. 材质展示分镜需近景拍摄，同时模特面部或上半身仍在画面中

## 动态动作设计体系
动作分类要求（6个分镜必须涵盖以下所有类型）
1. 行进动态类
    - 具体动作：自然行走、轻快踱步、侧身移动
    - 展示重点：服装在运动中的垂坠感和整体轮廓
    - 动态特征：体现重心的自然转移和肢体协调
2. 生活互动类
    - 具体动作：整理衣领、调整背包、查看手机、挥手打招呼
    - 展示重点：服装在日常互动中的自然状态
    - 动态特征：手部与服装的自然接触和互动
3. 姿态转换类
    - 具体动作：从站立到半蹲、身体轻微晃动、重心转移
    - 展示重点：服装在不同姿态下的版型变化
    - 动态特征：体现姿态转换的流畅过程
4. 情绪表达类
    - 具体动作：开心时的自然摆动、自信的肢体语言
    - 展示重点：服装在情绪表达时的动态美感
    - 动态特征：配合面部表情的协调肢体动作
5. 展示性动作类
    - 具体动作：自然地展示袖口、衣摆、领口等细节
    - 展示重点：服装的特定设计细节
    - 动态特征：动作自然不做作，符合生活场景
6. 材质展示类
    - 具体动作：通过肢体位置变化让材质在光线下自然呈现
    - 展示重点：面料纹理、光泽度和质感
    - 动态特征：利用自然光线和角度展示材质特性


## 禁止项
- **禁止人物背对镜头或转身动作**
- **禁止静态、呆板的站立姿势**
- **禁止材质展示分镜中模特完全淡出画幅**
- **禁止材质展示分镜中出现拉扯、揉捏、抓握等模特主动展示衣物的动作**
- **禁止动作重复或相似度过高** 
- **禁止所有分镜使用相同的基本姿态**


## 输出格式
严格输出JSON格式：

```json
[
  ["分镜1首帧参考图片编号","分镜1画面描述：模特动作1","分镜1首帧图片编辑prompt：保留参考图场景和服装 + 修改为动作1"],
  ["分镜2首帧参考图片编号","分镜2画面描述：模特动作2","分镜2首帧图片编辑prompt：保留参考图场景和服装 + 修改为动作2"],
  ["分镜3首帧参考图片编号","分镜3画面描述：模特动作3","分镜3首帧图片编辑prompt：保留参考图场景和服装 + 修改为动作3"],
  ["分镜4首帧参考图片编号","分镜4画面描述：模特动作4","分镜4首帧图片编辑prompt：保留参考图场景和服装 + 修改为动作4"],
  ["分镜5首帧参考图片编号","分镜5画面描述：模特动作5","分镜5首帧图片编辑prompt：保留参考图场景和服装 + 修改为动作5"],
  ["分镜6首帧参考图片编号","分镜6画面描述：材质展示动作","分镜6首帧图片编辑prompt：保留参考图场景和服装 + 近景展示材质细节 + 模特上半身可见"]
]
"""



System_Prompt_scene_plot_parse = """
# Role
你是一名资深的视频逆向工程与视觉叙事分析专家。请仔细观看视频，你的任务是深入理解输入的视频内容，识别剪辑逻辑，将其拆解为精确的视频轨道数据。

# Goal
请严格按照以下步骤对视频进行解析，最终输出一个符合指定 Schema 的 JSON 对象。

# Workflow (必须严格按序执行)

## 第一步：视频分镜与轨道生成 (Video Track)
**核心原则：精准捕捉剪辑节奏，既要保持叙事连贯，又要敏锐识别画面焦点的变化。**

1.  **切镜标准 (Cut Detection)**：
    *   **硬切分 (Hard Cut)**：场景完全切换、视角发生物理位移。
    *   **景别突变 (Scale Shift)**：这是关键点。即使主体（如人物/商品）不变，如果画面从**全景/中景**瞬间切换到**局部特写**（或反之），必须切分为两个镜头。
        *   *Example*：男人穿大衣（全身） -> 大衣扣子特写（局部） -> 男人整理领口（全身）。这应被识别为 3 个独立镜头。
    *   **快节奏剪辑 (Fast-paced Editing)**：对于电商、卡点或混剪类视频，忽略时长限制，只要画面发生上述变化，即使单个镜头短于 1 秒，也必须记录。

2.  **合并逻辑 (Merge Logic)**：
    *   仅在以下情况合并：背景不变、主体不变、**且景别（拍摄距离）未发生显著改变**的连续片段。
    *   **禁止合并**：不要将展示整体效果的镜头与展示材质/细节的镜头合并，因为它们的语义重点不同。

3.  **时间轴连续性**：
    *   当前镜头的 `end_time` 必须严格等于下一个镜头的 `start_time`，单位为秒（Float类型），确保时间轴无缝衔接。

4.  **字段生成**：
    *   `caption`: 详细描述该镜头的视觉内容。如果是特写，请明确描述“特写展示了...细节”；如果是全景，请描述“全身展示了...效果”。
    *   `start_time`/`end_time`: 精确到秒（建议保留两位小数）。

# Output Schema (严格遵守)
请直接输出 JSON 代码，不要包含 markdown 代码块标记（```json ... ```）以外的任何分析过程或闲聊文字。

```json
{
  "video_track": [
    {
      "caption": "xxxx", 
      "start_time": 0.0,
      "end_time": 5.5
    }
  ]
}
"""


def encode_img_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def get_base64_encoded_image_gemini(image_path):
    """读取图片文件并返回Base64编码的字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        logger.info(f"错误: 图片文件未找到于 {image_path}")
        return None
    except Exception as e:
        logger.info(f"编码图片时发生错误: {e}")
        return None


def parse_by_gemini(prompt_gemini=""):
    # API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-flash-image-preview:generateContent"
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-pro:generateContent"
    
    
    # 构建请求头
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # 构建请求体
    parts = None
    if isinstance(prompt_gemini, str):
        parts = [{"text": prompt_gemini}]
    elif isinstance(prompt_gemini, list):
        parts = [{"text": " ".join(prompt_gemini)}]
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ]
    }
    raw_answer = None
    try:
        logger.info("正在发送API请求...")
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 如果响应状态码不是200，则抛出HTTPError异常

        response_data = response.json()
        logger.info("API响应成功:")
        # logger.info(json.dumps(response_data, indent=2, ensure_ascii=False)) # 打印完整的响应

        # 解析响应，仿照Gemini SDK的结构
        response_data["candidates"][0]["content"]["parts"]
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            # pdb.set_trace()
            for part in response_data["candidates"][0].get("content", {}).get("parts", []):
                if "text" in part:
                    raw_answer = part["text"]
                    logger.info(raw_answer)
                else:
                    logger.info(f"未知响应部分: {part}")
        else:
            logger.info("响应中未找到 'candidates' 或其为空。")

    except requests.exceptions.HTTPError as http_err:
        logger.info(f"HTTP错误: {http_err}")
        logger.info(f"响应内容: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.info(f"连接错误: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.info(f"请求超时: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.info(f"请求发生错误: {req_err}")
    except json.JSONDecodeError as json_err:
        logger.info(f"JSON解码错误: {json_err}")
        logger.info(f"原始响应文本: {response.text}")
    except Exception as e:
        logger.info(f"发生未知错误: {e}")

    return raw_answer
        
AD_CINEMATIC_PROMPT = """
You are an expert AI Video Director and Cinematographer specializing in high-end commercial advertisements. 
Analyze the provided image and generate a text-to-video prompt to animate this scene into a cinematic commercial shot.

Your output must focus on the following dynamic elements:
1. **Camera Movement**: Describe sophisticated camera moves (e.g., "Slow dolly in," "Truck left," "Orbital shot," "Rack focus," "Low angle tracking").
2. **Subject Motion**: Describe realistic and appealing movement within the scene (e.g., "Water droplets condensation sliding down," "Steam rising elegantly," "Hair blowing in slow motion," "Fabric flowing," "Light reflections shifting").
3. **Atmosphere & Lighting**: Enhance the commercial look (e.g., "Cinematic lighting," "4k," "High resolution," "Slow motion," "Volumetric fog," "Golden hour").

**Constraints:**
- The motion must be physically plausible and high-quality.
- Keep the description concise but descriptive (under 75 words).
- 输出的时候 用中文，ready to be pasted into a video generation tool (like Runway, Pika, or Sora).
"""

def parse_image_by_gemini(image_path, prompt_gemini):
    # API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-flash-image-preview:generateContent"
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-3.0-pro:generateContent"
    
    # 获取Base64编码
    # pdb.set_trace()
    if isinstance(image_path, list):
        image_path = image_path[0]

    ext = image_path.split(".")[1]
    MIME_TYPE = "image/png"
    if ext in ["jpg", "jpeg"]:
        MIME_TYPE = "image/jpg"
    img_base64 = get_base64_encoded_image_gemini(image_path)
    
    if img_base64:
        logger.info(f"Base64编码长度: {len(img_base64)}")

        # 构建请求头
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }

        # 构建请求体
        parts = None
        if isinstance(prompt_gemini, str):
            parts = [{"text": prompt_gemini}]
        elif isinstance(prompt_gemini, list):
            parts = [{"text": " ".join(prompt_gemini)}]
        parts.extend(
                        [
                            {
                                "inline_data": {
                                    "mime_type": MIME_TYPE,
                                    "data": img_base64
                                }
                            }
                    ])  
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ]
        }
        raw_answer = None
        try:
            logger.info("正在发送API请求...")
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # 如果响应状态码不是200，则抛出HTTPError异常

            response_data = response.json()
            logger.info("API响应成功:")
            # logger.info(json.dumps(response_data, indent=2, ensure_ascii=False)) # 打印完整的响应

            # 解析响应，仿照Gemini SDK的结构
            response_data["candidates"][0]["content"]["parts"]
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                for part in response_data["candidates"][0].get("content", {}).get("parts", []):
                    if "text" in part:
                        raw_answer = part["text"]
                        logger.info(raw_answer)
                    else:
                        logger.info(f"未知响应部分: {part}")
            else:
                logger.info("响应中未找到 'candidates' 或其为空。")

        except requests.exceptions.HTTPError as http_err:
            logger.info(f"HTTP错误: {http_err}")
            logger.info(f"响应内容: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.info(f"连接错误: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.info(f"请求超时: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.info(f"请求发生错误: {req_err}")
        except json.JSONDecodeError as json_err:
            logger.info(f"JSON解码错误: {json_err}")
            logger.info(f"原始响应文本: {response.text}")
        except Exception as e:
            logger.info(f"发生未知错误: {e}")
    else:
        logger.info("无法进行API调用，因为图片编码失败。")

    return raw_answer


def parse_video_by_gemini(video_path, prompt_gemini=System_Prompt_scene_plot_parse):
    API_URL = "http://llm-gateway-sgp.internal/ai-serve/v1/gemini-2.5-pro:generateContent"

    if isinstance(video_path, list):
        video_path = video_path[0]

    # 读取视频为 bytes -> base64
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    # MIME type 目前只处理 mp4
    MIME_TYPE = "video/mp4"

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # prompt 兼容 str / list
    if isinstance(prompt_gemini, str):
        parts = [{"text": prompt_gemini}]
    elif isinstance(prompt_gemini, list):
        parts = [{"text": " ".join(prompt_gemini)}]

    # 添加视频（inline_data + video_metadata）
    parts.extend([
        {
            "inline_data": {
                "mime_type": MIME_TYPE,
                "data": video_base64
            },
            "video_metadata": {
                "fps": 5   # ⭐ 与官方示例保持一致
            }
        }
    ])

    payload = {
        "contents": [
            {
                "parts": parts
            }
        ]
    }

    raw_answer = None

    try:
        logger.info("正在发送视频解析 API 请求...")
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        data = response.json()
        logger.info("视频解析成功")

        # 兼容内部网关的 candidates → content → parts → text
        if "candidates" in data and data["candidates"]:
            for part in data["candidates"][0]["content"]["parts"]:
                if "text" in part:
                    raw_answer = part["text"]
                    logger.info(raw_answer)
        else:
            logger.info("响应中未找到 candidates 字段")

    except Exception as e:
        logger.error(f"视频解析错误: {e}")
        logger.error(f"原始响应: {response.text if 'response' in locals() else '无'}")

    return raw_answer



# if __name__ == "__main__":
#     # image_path = "/data/zgq/nieuwland/src/t2i2v/t2i2v_project/t2i2v_pipeline/ref_img_720p.png"
#     # parse_image_by_gemini(image_path)

#     video_path = "/data/zgq/nieuwland/src/t2i2v/t2i2v_project/t2i2v_pipeline/videoplayback.mp4"
#     parse_video_by_gemini(video_path)
    

# ================= 主函数：批量处理 =================

def process_folder_for_ad_prompts(folder_path, prompt_text):
    """
    遍历文件夹下所有图片，生成动态效果Prompt，并保存为同名txt文件
    """
    # 1. 检查文件夹是否存在
    if not os.path.exists(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")
        return

    # 2. 获取所有图片文件 (支持 jpg, jpeg, png)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_files.sort() # 排序，方便查看进度
    total_files = len(image_files)
    logger.info(f"=== 开始处理，共发现 {total_files} 张图片 ===")

    # 3. 循环处理
    for index, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        logger.info(f"[{index+1}/{total_files}] 正在分析: {filename}")
        
        # 调用 Gemini 解析
        result_prompt = parse_image_by_gemini(img_path, prompt_text)

        if result_prompt:
            # 4. 保存结果到同名的 .txt 文件
            txt_path = os.path.splitext(img_path)[0] + "_prompt.txt"
            
            # 可选：也可以追加保存到一个汇总文件 summary.txt
            # summary_path = os.path.join(folder_path, "all_prompts_summary.txt")

            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(result_prompt)
                logger.info(f"--> 保存成功: {os.path.basename(txt_path)}")
                logger.info(f"--> 生成内容: {result_prompt}\n")
            except Exception as e:
                logger.error(f"保存文件失败: {e}")
        else:
            logger.warning(f"--> {filename} 生成失败或返回为空")

    logger.info("=== 全部处理完成 ===")

# ================= 执行入口 =================

if __name__ == "__main__":
    # 设置目标图片文件夹路径
    TARGET_FOLDER = r"./test_image2"  # 修改为你的实际路径
    
    # 使用之前定义的广告大片 Prompt
    # 如果你想让Gemini输出中文指导，可以在Prompt里加一句 "Please output in Chinese."
    process_folder_for_ad_prompts(TARGET_FOLDER, AD_CINEMATIC_PROMPT)
