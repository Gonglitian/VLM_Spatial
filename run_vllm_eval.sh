#!/bin/bash

# vLLM批量推理评估脚本
echo "🚀 启动vLLM批量推理评估..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:."

# 快速测试 (2个样本)
echo "📝 快速测试模式: 2个样本"
python scripts/model_eval.py --config-name=eval_config_vllm data.num_samples=2

echo "✅ 评估完成!" 