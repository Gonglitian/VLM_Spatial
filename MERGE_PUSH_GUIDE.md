# 🔧 Merge and Push 脚本修复指南

## 🚨 问题诊断

之前的错误是由于两个问题：

1. **AttributeError: module 'torch' has no attribute 'fp16'**
   - 原因：`torch.fp16` 不存在，应该是 `torch.float16`

2. **ValueError: Unrecognized configuration class Qwen2_5_VLConfig for AutoModelForCausalLM**
   - 原因：Qwen2.5-VL 是视觉语言模型，不能用 `AutoModelForCausalLM` 加载

## ✅ 修复方案

我已经重写了 `scripts/merge_and_push.py`，采用以下修复策略：

### 1. **修复 dtype 映射**
```python
DTYPE_MAP = {
    "fp16": torch.float16,      # ✅ 正确映射
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
```

### 2. **优先使用 unsloth**
对于 Qwen2.5-VL 模型，unsloth 提供了更可靠的支持：
```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(args.base)
```

### 3. **回退策略**
如果 unsloth 不可用，回退到 transformers：
```python
from transformers import AutoModel  # 而不是 AutoModelForCausalLM
model = AutoModel.from_pretrained(args.base, trust_remote_code=True)
```

## 🚀 使用方法

### 测试基本环境
```bash
python test_simple.py
```

### 执行模型合并
```bash
python scripts/merge_and_push.py \
    --base unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit \
    --adapter ./final_model \
    --repo your-username/merged-model-name \
    --private  # 可选，创建私有仓库
```

### 仅本地合并（不上传）
如果不想上传到 HuggingFace，脚本会在 `./merged_vlm/` 目录保存合并后的模型。

## 🔍 故障排除

### 如果 unsloth 不可用
```bash
pip install unsloth
# 或者
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 如果 transformers 版本过低
```bash
pip install --upgrade transformers
pip install --upgrade accelerate
```

### 如果 PEFT 不可用
```bash
pip install peft
```

### 如果 huggingface_hub 不可用
```bash
pip install huggingface_hub
```

## ✨ 改进特性

1. **更好的错误处理**：详细的错误信息和回退策略
2. **进度提示**：清晰的执行步骤显示
3. **灵活的 dtype 支持**：支持多种数据类型格式
4. **专门的 Qwen2.5-VL 优化**：针对视觉语言模型的特殊处理
5. **可选上传**：即使上传失败，本地合并仍然有效

## 🎯 预期输出

成功运行时你应该看到：
```
🚀 开始LoRA模型合并流程...
📁 基座模型: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit
🔗 LoRA适配器: ./final_model
📤 目标仓库: your-repo
🔗 Loading base model with unsloth...
✅ 基座模型加载成功
🔗 Loading LoRA adapter...
✅ LoRA适配器加载成功
📝 Merging weights...
✅ 权重合并完成
💾 Saving merged model...
✅ 模型保存完成
📝 生成模型卡片...
✅ 模型卡片生成完成
🔐 准备上传到 Hugging Face Hub...
🚀 Uploading files...
✅ All done! View: https://huggingface.co/your-repo
```

## 🗂️ 输出结构

```
./merged_vlm/
├── config.json              # 模型配置
├── model.safetensors.*       # 合并后的模型权重
├── tokenizer.json           # 分词器
├── tokenizer_config.json    # 分词器配置
├── README.md                # 自动生成的模型卡片
└── ...                      # 其他相关文件
```

现在你可以尝试运行修复后的脚本了！🎉 