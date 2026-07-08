import requests
import json
import base64
from pathlib import Path
import time
import logging
from typing import Optional, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NanoBananaAIGenerator:
    def __init__(self, api_key: str = None):
        """
        初始化Nano Banana AI图像生成器
        
        Args:
            api_key: Novita AI API密钥，如果不传入会提示输入
        """
        if api_key is None:
            api_key = input("请输入您的Novita AI API密钥: ").strip()
        
        self.api_url = "https://api.novita.ai/v3/gemini-2.5-flash-image-text-to-image"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.api_key = api_key
        
        logger.info(f"API密钥已设置（前5位: {api_key[:5]}...）")
    
    def generate_image(self, prompt: str, output_path: str = None) -> dict:
        """
        根据文本提示生成图像，并自动下载保存
        """
        logger.info(f"开始生成图像，提示: {prompt[:50]}...")
        
        # 1. 严格遵守文档，只发这三个参数
        payload = {
            "prompt": prompt,
            "aspect_ratio": "1:1",  # 默认正方形，可改 16:9
            "size": "1K"
        }
        
        logger.info(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
        
        try:
            logger.info("发送API请求给 Novita...")
            start_time = time.time()
            
            # 发送请求 (设置60秒超时，给大模型慢慢画)
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
            
            result = response.json()
            
            # 2. 从返回值里提取图片 URL
            if "image_urls" in result and len(result["image_urls"]) > 0:
                image_url = result["image_urls"][0]
                logger.info(f"🎉 成功获取图片链接: {image_url}")
                logger.info("正在下载图片...")
                
                # 3. 下载图片内容
                img_response = requests.get(image_url, timeout=30)
                image_bytes = img_response.content
                
                # 4. 保存到本地
                if output_path:
                    self._save_image(image_bytes, output_path)
                    logger.info(f"✅ 图像已成功保存到: {output_path}")
                
                return {
                    "success": True, 
                    "image_size": len(image_bytes), 
                    "elapsed_time": elapsed_time
                }
            else:
                logger.error("响应中未找到 image_urls")
                return {"success": False, "error": "未找到图片链接"}
                
        except Exception as e:
            logger.error(f"发生异常: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _save_image(self, image_bytes: bytes, output_path: str) -> str:
        """
        保存图像到文件
        
        Args:
            image_bytes: 图像字节数据
            output_path: 保存路径
            
        Returns:
            实际保存的路径
        """
        # 确保目录存在
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(path, 'wb') as f:
            f.write(image_bytes)
        
        return str(path.absolute())
    
    def test_api_connection(self):
        """
        测试API连接
        """
        logger.info("测试API连接...")
        
        # 简单的测试请求
        test_payload = {
            # "model": "gemini-2.5-flash-image", # 这个接口URL已经指定了模型，不需要在这传
            "prompt": "test",
            "aspect_ratio": "1:1",
            "size": "1K"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=test_payload,
                timeout=60
            )
            
            logger.info(f"测试响应状态码: {response.status_code}")
            logger.info(f"测试响应内容: {response.text[:200]}")
            
            return response.status_code, response.text
        except Exception as e:
            logger.error(f"测试失败: {str(e)}")
            return None, str(e)

# ==================== 使用示例 ====================
def main():
    """
    主函数 - 使用示例
    """
    
    # 在这里直接填入你的API密钥
    API_KEY = "REDACTED"  # ← 在这里填入你的API密钥
    
    print("=" * 50)
    print("Nano Banana AI 图像生成器 - 调试版本")
    print("=" * 50)
    
    # 初始化生成器
    if API_KEY == "YOUR_API_KEY_HERE":
        # 如果用户忘记修改API_KEY，则提示输入
        generator = NanoBananaAIGenerator()
    else:
        generator = NanoBananaAIGenerator(api_key=API_KEY)
    
    # 测试API连接
    print("\n测试API连接...")
    status, response = generator.test_api_connection()
    
    if status == 200:
        print("✓ API连接成功")
    else:
        print(f"✗ API连接失败: {status}")
        print(f"响应: {response}")
        return
    
    # 使用更简单的参数生成图像
    print("\n生成一张简单的图像...")
    prompt = "A simple red apple on a white background"
    
    result = generator.generate_image(
        prompt=prompt,
        output_path="/data/phd/kousiqi/zhitao/gemini_test/test_apple.png",
    )
    
    if result["success"]:
        print(f"✓ 图像生成成功！")
        print(f"  图像大小: {result['image_size']:,} 字节")
        print(f"  耗时: {result['elapsed_time']:.2f} 秒")
        print(f"  保存到: output/test_apple.png")
    else:
        print(f"✗ 图像生成失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    
    # 运行主函数
    main()