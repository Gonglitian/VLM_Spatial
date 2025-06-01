from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import requests
import base64
import io
from PIL import Image
import uvicorn

app = FastAPI(title="VLM图像描述服务")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# VLLM API配置
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页HTML"""
    return FileResponse("index.html")

@app.post("/describe-image/")
async def describe_image(file: UploadFile = File(...)):
    """上传图像并获取描述"""
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图像类型")
        
        # 读取并处理图像
        contents = await file.read()
        
        # 使用更安全的方式处理图像
        try:
            image = Image.open(io.BytesIO(contents))
            # 确保图像完全加载
            image.load()
            
            # 如果是RGBA模式，转换为RGB
            if image.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"图像处理失败: {str(img_error)}")
        
        # 将图像转换为base64，使用JPEG格式以获得更好的兼容性
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 构造消息
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
                        "text": "请用一句话描述这张图片。"
                    }
                ]
            }
        ]
        
        # 调用VLLM API
        try:
            response = requests.post(
                f"{VLLM_API_BASE}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 150
                },
                timeout=30
            )
        except requests.exceptions.RequestException as req_error:
            raise HTTPException(status_code=500, detail=f"API请求失败: {str(req_error)}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"API调用失败: {response.text}")
        
        try:
            result = response.json()
            description = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as parse_error:
            raise HTTPException(status_code=500, detail=f"API响应解析失败: {str(parse_error)}")
        
        return {
            "success": True,
            "description": description,
            "filename": file.filename
        }
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 捕获所有其他异常
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查VLLM API是否可用
        response = requests.get(f"{VLLM_API_BASE}/models", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "vllm_api": "available"}
        else:
            return {"status": "unhealthy", "vllm_api": "unavailable"}
    except:
        return {"status": "unhealthy", "vllm_api": "unavailable"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 