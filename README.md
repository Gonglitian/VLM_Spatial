# VLM-Spatial

A complete solution for training and deploying Vision Language Models (VLMs) with advanced configuration management.

## ✨ Features

- 🤖 **Qwen2.5-VL Model Training** - Fine-tune vision-language models with unsloth LoRA
- ⚡ **Single-GPU Optimized** - Efficient training with unsloth framework
- ⚙️ **Hydra Configuration** - Advanced config management with YAML files
- 🚀 **Production Ready** - Inference server with web interface
- 📊 **W&B Integration** - Experiment tracking and model logging
- 🔧 **Memory Efficient** - 4-bit quantization and gradient checkpointing

## 🚀 Quick Start

### 1. Setup Environment

```bash
conda create -n vlm-spatial python=3.10
conda activate vlm-spatial
pip install -r requirements.txt
```

### 2. Training

```bash
# Basic training with default config
python scripts/train.py

# Small-scale training
python scripts/train.py --config-name=train_qwen_small

# Custom parameters
python scripts/train.py model.name=Qwen/Qwen2.5-VL-7B-Instruct train.num_epochs=3
```
See `CONFIG_README.md` for detailed configuration options.
### 3. Inference

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/final_model \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

# Start web interface
cd frontend && uvicorn app:app --host 0.0.0.0 --port 8080
```

## 📁 Project Structure

```
VLM-Spatial/
├── conf/                    # Configuration files
│   ├── train_qwen.yaml     # Main training config
│   └── train_qwen_small.yaml
├── scripts/
│   └── train.py           # Training script with Hydra
├── frontend/              # Web inference interface
├── outputs/               # Training outputs
└── CONFIG_README.md       # Detailed config guide
```

## ⚙️ Configuration

The training script uses Hydra for flexible configuration management:

- **Model**: Choose model size, quantization, dtype
- **LoRA**: Configure rank, alpha, target modules
- **Training**: Set batch size, learning rate, optimizer
- **Data**: Specify dataset, processing options
- **W&B**: Enable experiment tracking

See `CONFIG_README.md` for detailed configuration options.

## 🔧 Key Components

### Training (`scripts/train.py`)
- Complete Hydra integration
- Automatic model loading and LoRA configuration
- Data processing with validation splits
- W&B logging and model upload

### Configuration (`conf/`)
- Structured YAML configs with Unicode organization
- Support for different training scales
- Easy parameter overrides via command line

### Web Interface (`frontend/`)
- Modern UI for image analysis
- API endpoints for programmatic access
- Real-time inference with uploaded models

## 📊 Training Features

- **Unsloth Optimization**: Ultra-fast single-GPU training
- **4-bit Quantization**: Reduce memory usage with QLoRA
- **Parallel Processing**: Multi-core data conversion
- **Validation Splits**: Automatic train/val separation
- **Mixed Precision**: FP16/BF16 support
- **Gradient Checkpointing**: Memory optimization
- **Resume Training**: Checkpoint support
- **Experiment Tracking**: W&B integration

## 🛠️ Advanced Usage

### Custom Dataset Training
```bash
python scripts/train.py \
  data.dataset_name=your/dataset \
  data.max_samples=1000 \
  data.val_split_ratio=0.1
```

### GPU Memory Optimization
```bash
python scripts/train.py \
  model.load_in_4bit=true \
  train.gradient_checkpointing=true \
  train.batch_size=2
```

### Hyperparameter Tuning
```bash
python scripts/train.py \
  lora.r=32 \
  lora.alpha=64 \
  train.learning_rate=2e-4 \
  train.batch_size=4
```

## 📈 Monitoring

- **W&B Dashboard**: Real-time training metrics
- **Local Logging**: Detailed console output
- **Model Checkpoints**: Automatic saving
- **Health Checks**: API status monitoring

## 🔍 Troubleshooting

**Memory Issues**: Reduce batch size, enable 4-bit quantization, or use gradient checkpointing
**Config Errors**: Check YAML syntax and parameter types
**Model Loading**: Verify model name and Hugging Face access
**W&B Issues**: Ensure API key is set and project exists
**Single GPU**: This project is optimized for single-GPU training with unsloth

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests. 

## Blender
lib install 

```python
~/apps/blender-4.4.3-linux-x64/4.4/python/bin/python3.11 -m pip install tqdm
```