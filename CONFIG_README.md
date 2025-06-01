# Hydra配置管理系统

本项目使用Hydra/OmegaConf进行配置管理，提供灵活且可重现的实验设置。

## 📁 目录结构

```
VLM-Spatial/
├── conf/                          # 配置文件目录
│   ├── train_qwen.yaml           # 默认训练配置（完整版）
│   └── train_qwen_small.yaml     # 小规模训练配置
├── scripts/
│   └── train.py                  # 主训练脚本
├── run_examples.sh              # 使用示例脚本
└── requirements_hydra.txt       # Hydra相关依赖
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install hydra-core omegaconf tqdm numpy
# 或者
pip install -r requirements_hydra.txt
```

### 2. 基本使用

```bash
# 使用默认配置
python scripts/train.py

# 使用指定配置文件
python scripts/train.py --config-name=train_qwen_small
```

### 3. 运行时参数覆盖 (⭐ 推荐)

```bash
# 调整LoRA参数
python scripts/train.py lora.r=96 lora.alpha=96

# 调整训练参数
python scripts/train.py train.batch_size=4 train.learning_rate=3e-4

# 组合多个参数
python scripts/train.py lora.r=48 train.batch_size=6 train.num_epochs=3
```

## 📋 配置文件结构

### 🏃 运行环境配置
```yaml
run:
  cuda_visible_devices: "1"          # GPU设备ID
  seed: 3407                         # 全局随机种子
  resume_from_checkpoint: null       # 从检查点恢复训练
```

### 🤖 模型配置
```yaml
model:
  name: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit
  load_in_4bit: true
  torch_dtype: "auto"               # 数据类型
  trust_remote_code: true           # 允许自定义代码
  max_seq_length: 4096              # 序列长度
  device_map: "auto"                # 设备映射
  rope_scaling: null                # RoPE缩放配置
```

### 🎯 LoRA配置
```yaml
lora:
  r: 64                           # LoRA矩阵的秩
  alpha: 64                       # 缩放因子
  dropout: 0.05                   # Dropout率
  bias: "none"                    # 偏置设置
  use_rslora: false               # 排序稳定LoRA
  target_modules: "all-linear"    # 目标模块
  # ... 微调选项
```

### 📊 数据配置
```yaml
data:
  dataset_name: "unsloth/LaTeX_OCR"
  split: "train"
  max_samples: null               # 样本数限制
  num_proc: 4                     # 处理进程数
  instruction: "Write the LaTeX representation for this image."
  shuffle: true                   # 数据打乱
  streaming: false                # 流式模式
```

### 🏋️ 训练配置
```yaml
train:
  # 基础参数
  batch_size: 8
  grad_accum: 2
  num_epochs: 2
  learning_rate: 2.0e-4
  
  # 优化器设置
  optim: "paged_adamw_8bit"
  lr_scheduler_type: "linear"
  warmup_steps: 50
  weight_decay: 0.01
  
  # 精度和内存
  fp16: false
  bf16: true
  gradient_checkpointing: false
  
  # 日志和保存
  logging_steps: 10
  save_strategy: "steps"
  save_steps: 100
  output_dir: "outputs"
```

### 📈 W&B配置
```yaml
wandb:
  project: "latex_ocr"
  entity: null                    # 团队名称
  run_name: "qwen_lora64"
  tags: ["qwen", "vision", "lora64"]
  notes: null
  log_model: true                 # 上传模型
```

## 🔧 主要特性和改进

### 1. **完整的参数分组**
- 🏃 运行环境：GPU、种子、检查点恢复
- 🤖 模型配置：完整的模型加载参数
- 🎯 LoRA配置：高级LoRA选项
- 📊 数据配置：数据处理和加载
- 🏋️ 训练配置：完整的训练参数
- 📈 W&B配置：实验跟踪设置

### 2. **增强的功能**
- ✅ 随机种子设置，确保可重现性
- ✅ 检查点恢复支持
- ✅ 可自定义的指令提示
- ✅ 数据打乱和流式加载
- ✅ 完整的W&B集成
- ✅ 进度条和详细日志

### 3. **灵活的配置管理**
- ✅ 运行时参数覆盖
- ✅ 多配置文件支持
- ✅ 完整的参数验证
- ✅ 清晰的配置分组

## 💡 高级用法示例

### 快速实验
```bash
# 快速测试（小数据量）
python scripts/train.py data.max_samples=100 train.num_epochs=1

# 调试模式（极小配置）
python scripts/train.py --config-name=train_qwen_small data.max_samples=10
```

### 性能优化
```bash
# 小GPU配置
python scripts/train.py lora.r=16 train.batch_size=2 train.grad_accum=8

# 大GPU配置  
python scripts/train.py lora.r=128 train.batch_size=16 train.grad_accum=1

# 启用DeepSpeed
python scripts/train.py train.deepspeed_config=./ds_config.json
```

### 实验管理
```bash
# 不同学习率实验
python scripts/train.py wandb.run_name=exp_lr_high train.learning_rate=5e-4
python scripts/train.py wandb.run_name=exp_lr_low train.learning_rate=1e-4

# 不同LoRA配置实验
python scripts/train.py wandb.run_name=lora32 lora.r=32 lora.alpha=32
python scripts/train.py wandb.run_name=lora64 lora.r=64 lora.alpha=64
```

### 从检查点恢复
```bash
# 从最新检查点恢复
python scripts/train.py run.resume_from_checkpoint=./outputs/checkpoint-100

# 更改配置继续训练
python scripts/train.py run.resume_from_checkpoint=./outputs/checkpoint-100 train.learning_rate=1e-4
```

## 🎯 配置验证

脚本会自动验证配置参数，并在启动时显示：
- 🔧 运行环境设置
- 🎲 随机种子
- 📈 W&B初始化状态
- 🚀 模型加载参数
- 🔧 LoRA配置
- 📊 数据集信息
- 🔄 数据转换进度

## 📚 更多信息

- 查看 `run_examples.sh` 了解使用示例
- 参考具体配置文件中的注释
- 所有配置都会自动记录到W&B中 