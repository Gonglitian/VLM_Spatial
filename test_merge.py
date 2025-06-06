#!/usr/bin/env python3
"""
简单的模型加载测试脚本
"""
import torch
print(f"✅ PyTorch版本: {torch.__version__}")

try:
    from transformers import AutoModel, AutoTokenizer
    print("✅ Transformers导入成功")
    
    # 测试dtype映射
    DTYPE_MAP = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    
    dtype = "fp16"
    selected_torch_dtype = DTYPE_MAP.get(dtype.lower())
    print(f"✅ dtype映射测试通过: {dtype} -> {selected_torch_dtype}")
    
    # 测试模型加载
    base_model = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    print(f"🔗 尝试加载模型: {base_model}")
    
    model = AutoModel.from_pretrained(
        base_model,
        torch_dtype=selected_torch_dtype,
        trust_remote_code=True,
    )
    print("✅ 模型加载成功!")
    
    # 测试tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print("✅ Tokenizer加载成功!")
    
    print("🎉 所有测试通过!")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc() 