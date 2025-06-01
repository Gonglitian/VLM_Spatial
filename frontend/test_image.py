#!/usr/bin/env python3

import requests
import base64
import io
from PIL import Image

def test_image_description():
    # 测试图像处理
    try:
        # 打开图像
        with open("image.png", "rb") as f:
            image_data = f.read()
            
        image = Image.open(io.BytesIO(image_data))
        print(f"原始图像模式: {image.mode}")
        print(f"原始图像尺寸: {image.size}")
        
        # 确保图像完全加载
        image.load()
        
        # 缩小图像以减少内存使用
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"调整后图像尺寸: {image.size}")
        
        # 转换为RGB
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        print(f"转换后图像模式: {image.mode}")
        
        # 转换为base64，使用较低质量以减少大小
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        print(f"Base64长度: {len(img_base64)}")
        
        # 构造更简单的请求
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    }
                ]
            }
        ]
        
        print("发送请求到VLLM API...")
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 100
            },
            timeout=90
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"完整响应: {result}")
            description = result["choices"][0]["message"]["content"]
            print(f"成功！描述: {description}")
        else:
            print(f"错误响应: {response.text}")
            # 尝试获取更多错误信息
            try:
                error_detail = response.json()
                print(f"错误详情: {error_detail}")
            except:
                pass
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_description() 