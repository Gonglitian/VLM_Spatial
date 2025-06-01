#!/bin/bash

# VLM-Spatial 服务启动脚本

echo "启动VLM图像描述服务..."

# 项目目录
PROJECT_DIR="/home/vla-reasoning/proj/VLM-Spatial"

# 检查是否在正确的环境中
if [[ "$CONDA_DEFAULT_ENV" != "vlmspatial" ]]; then
    echo "请先激活vlmspatial环境: conda activate vlmspatial"
    exit 1
fi

# 停止可能存在的旧进程
echo "停止可能存在的旧服务..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 2

# 启动VLLM API服务器
echo "启动VLLM API服务器 (端口 8000)..."
cd "$PROJECT_DIR"
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 &

VLLM_PID=$!
echo "VLLM API服务器 PID: $VLLM_PID"

# 等待VLLM API服务器启动
echo "等待VLLM API服务器启动..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "VLLM API服务器启动成功！"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "VLLM API服务器启动超时"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

# 启动FastAPI前端服务
echo "启动FastAPI前端服务 (端口 8080)..."
cd "$PROJECT_DIR/frontend"

# 确保static目录存在
mkdir -p static

uvicorn app:app --host 0.0.0.0 --port 8080 &
FASTAPI_PID=$!
echo "FastAPI前端服务 PID: $FASTAPI_PID"

# 等待FastAPI服务启动
echo "等待FastAPI服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "FastAPI服务启动成功！"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FastAPI服务启动超时"
        kill $VLLM_PID $FASTAPI_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

echo ""
echo "🎉 所有服务启动成功！"
echo "📊 VLLM API服务器: http://localhost:8000"
echo "🌐 Web界面: http://localhost:8080"
echo ""
echo "进程信息:"
echo "VLLM API (PID: $VLLM_PID)"
echo "FastAPI (PID: $FASTAPI_PID)"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo "停止服务..."; kill $VLLM_PID $FASTAPI_PID 2>/dev/null || true; exit 0' INT
wait 