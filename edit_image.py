import os
import requests
import base64
import time

def edit_image_with_novita(prompt, api_key, input_image_path, save_path):
    """
    专门用于根据输入图片和提示词进行图像编辑的函数
    """
    url = "https://api.novita.ai/v3/gemini-2.5-flash-image-edit"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. 读取原图并转换为 Base64 编码
    if not os.path.exists(input_image_path):
        print(f"❌ 找不到输入图片: {input_image_path}")
        return False

    try:
        with open(input_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
            # 统一封装为 data URI 格式
            img_data = f"data:image/png;base64,{img_b64}"
    except Exception as e:
        print(f"❌ 读取图片失败: {e}")
        return False

    # 2. 构建请求 Payload
    payload = {
        "prompt": prompt,
        "image_base64s": [img_data] # 传入要编辑的原图
    }

    # 3. 发送 API 请求
    try:
        print(f"⏳ 正在调用 Novita API 编辑图片...\n   👉 原图: {input_image_path}\n   👉 提示词: {prompt}")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if "image_urls" in result and len(result["image_urls"]) > 0:
                image_url = result["image_urls"][0]
                
                # 4. 下载生成的图片并保存
                print("⏳ 正在下载编辑后的图片...")
                img_response = requests.get(image_url, timeout=30)
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(img_response.content)
                    
                print(f"✅ 图片编辑成功！总耗时 {time.time()-start_time:.1f}s")
                print(f"💾 最终成品已保存至: {save_path}")
                return True
            else:
                print("❌ Novita 响应中未找到 image_urls，请检查账户余额或提示词合规性。")
                return False
        else:
            print(f"❌ Novita 生图接口报错 (状态码 {response.status_code}): {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 网络请求发生异常: {e}")
        return False

if __name__ == '__main__':
    # =====================================================================
    # 🌟 在这里输入你的配置参数
    # =====================================================================
    
    # 1. 你的 Novita API Key
    NOVITA_API_KEY = "REDACTED" 
    
    # 2. 你要编辑的【原图】绝对路径
    INPUT_IMAGE_PATH = "/data/phd/lijiahui/data/batch_no_voice_gen_fuzhuang/workspace_item_25917400924058/proj_item_25917400924058_1769968905_seg_0/character_refs/活力少女.png"  
    
    # 3. 你的【编辑要求/Prompt】(例如: "把人物的外套换成红色的羽绒服，保持其他不变")
    EDIT_PROMPT = "删去背景图中的箭头与云朵图案"
    
    # 4. 编辑完成后【新图】的保存路径
    OUTPUT_IMAGE_PATH = "/data/phd/lijiahui/data/batch_no_voice_gen_fuzhuang/workspace_item_25917400924058/proj_item_25917400924058_1769968905_seg_0/character_refs/活力少女1.png" 
    
    # =====================================================================
    
    # 执行编辑任务
    edit_image_with_novita(
        prompt=EDIT_PROMPT,
        api_key=NOVITA_API_KEY,
        input_image_path=INPUT_IMAGE_PATH,
        save_path=OUTPUT_IMAGE_PATH
    )