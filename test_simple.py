#!/usr/bin/env python3
print("🚀 开始测试...")

# 测试基本导入
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except:
    print("❌ PyTorch导入失败")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except:
    print("❌ Transformers导入失败")

try:
    from unsloth import FastVisionModel
    print("✅ Unsloth导入成功")
except:
    print("❌ Unsloth导入失败")

try:
    from peft import PeftModel
    print("✅ PEFT导入成功")
except:
    print("❌ PEFT导入失败")

print("🎉 测试完成!") 