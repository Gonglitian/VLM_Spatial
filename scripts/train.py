# train.py
import hydra
import torch
import wandb
import os
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from unsloth import FastVisionModel, is_bf16_supported
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_none_values(config_dict):
    """过滤掉None值的配置项"""
    return {k: v for k, v in config_dict.items() if v is not None}


def convert_to_conversation(sample):
    """将数据集样本转换为对话格式（适配question+answer格式）"""
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : sample["question"]},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["answer"]} ]
        },
    ]
    return { "messages" : conversation }


@hydra.main(version_base="1.1", config_path="../conf", config_name="train_qwen")
def main(cfg: DictConfig):
    print("=== 🔧 配置信息 ===")
    print(OmegaConf.to_yaml(cfg))
    
    # ======================================================================
    # 🏃 运行环境设置
    # ======================================================================
    if hasattr(cfg.run, 'cuda_visible_devices'):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.run.cuda_visible_devices)
        print(f"🔧 设置CUDA设备: {cfg.run.cuda_visible_devices}")
    
    if hasattr(cfg.run, 'seed') and cfg.run.seed is not None:
        set_seed(cfg.run.seed)
        print(f"🎲 设置随机种子: {cfg.run.seed}")
    
    # ======================================================================
    # 📈 W&B初始化
    # ======================================================================
    if cfg.wandb and cfg.train.report_to and "wandb" in cfg.train.report_to:
        wandb_config = filter_none_values({
            "project": cfg.wandb.project,
            "name": cfg.wandb.run_name,
            "entity": getattr(cfg.wandb, 'entity', None),
            "tags": getattr(cfg.wandb, 'tags', None),
            "notes": getattr(cfg.wandb, 'notes', None),
            "config": OmegaConf.to_container(cfg, resolve=True)
        })
        wandb.init(**wandb_config)
        print(f"📈 W&B初始化完成: {cfg.wandb.project}/{cfg.wandb.run_name}")
        
        # 记录关键配置参数，使其在W&B中更突出显示
        key_config = {
            "model_name": cfg.model.name,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.alpha,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.learning_rate,
            "num_epochs": cfg.train.num_epochs,
            "max_samples": cfg.data.max_samples,
            "dataset": cfg.data.dataset_name,
        }
        wandb.config.update(key_config)
        print("📋 关键配置参数已记录到W&B")
    
    # ======================================================================
    # 🤖 模型加载
    # ======================================================================
    print(f"🚀 加载模型: {cfg.model.name}")
    
    # 极简模型加载 - 只用必需参数
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.model.name,
        load_in_4bit=cfg.model.load_in_4bit
    )
    print("✅ 模型加载成功")
    
    # ======================================================================
    # 🎯 LoRA配置 - 极简化
    # ======================================================================
    print("🔧 配置LoRA参数")
    
    # 基础LoRA参数
    lora_config = {
        "model": model,
        "r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "finetune_vision_layers": cfg.lora.finetune_vision_layers,
        "finetune_language_layers": cfg.lora.finetune_language_layers,
        "finetune_attention_modules": cfg.lora.finetune_attention_modules,
        "finetune_mlp_modules": cfg.lora.finetune_mlp_modules,
    }
    
    # 自动添加可选参数（如果存在且不为None）
    optional_lora_params = ['bias', 'use_rslora', 'loftq_config']
    for param in optional_lora_params:
        if hasattr(cfg.lora, param):
            value = getattr(cfg.lora, param)
            if value is not None:
                lora_config[param] = value
    
    # 特殊处理target_modules
    if hasattr(cfg.lora, 'target_modules') and cfg.lora.target_modules:
        target_modules = cfg.lora.target_modules
        if isinstance(target_modules, str) and target_modules != "all-linear":
            lora_config["target_modules"] = [target_modules]
        elif isinstance(target_modules, (list, tuple)):
            lora_config["target_modules"] = target_modules
    
    model = FastVisionModel.get_peft_model(**lora_config)
    
    # ======================================================================
    # 📊 数据加载和处理 - 极简化
    # ======================================================================
    print(f"📊 加载数据集: {cfg.data.dataset_name}")
    
    # 加载数据集
    ds = load_dataset(
        path=cfg.data.dataset_name, 
        split=cfg.data.split,
        streaming=getattr(cfg.data, 'streaming', False)
    )
    
    # 验证集分割
    eval_dataset = None
    if hasattr(cfg.data, 'val_split_ratio') and cfg.data.val_split_ratio > 0:
        split_dataset = ds.train_test_split(
            test_size=cfg.data.val_split_ratio, 
            seed=getattr(cfg.run, 'seed', None)
        )
        ds, eval_dataset = split_dataset['train'], split_dataset['test']
        print(f"📊 训练集: {len(ds)} 样本, 验证集: {len(eval_dataset)} 样本")
    
    # 打乱和截取数据
    if getattr(cfg.data, 'shuffle', False):
        ds = ds.shuffle(seed=getattr(cfg.run, 'seed', None))
    
    if cfg.data.max_samples and cfg.data.max_samples > 0:
        ds = ds.select(range(min(cfg.data.max_samples, len(ds))))
        if eval_dataset:
            eval_samples = min(int(cfg.data.max_samples * cfg.data.val_split_ratio), len(eval_dataset))
            eval_dataset = eval_dataset.select(range(eval_samples))
    
    print(f"📊 最终训练样本数: {len(ds)}")
    if eval_dataset:
        print(f"📊 最终验证样本数: {len(eval_dataset)}")
    
    # 数据格式转换 - 使用每个样本的question作为指令
    converted_dataset = [convert_to_conversation(sample) for sample in tqdm(ds, desc="🔄 转换训练数据")]
    eval_converted_dataset = None
    if eval_dataset:
        eval_converted_dataset = [convert_to_conversation(sample) for sample in tqdm(eval_dataset, desc="🔄 转换验证数据")]
    
    print("✅ 数据转换完成!")
    
    # ======================================================================
    # 🏋️ 训练配置 - 极简化
    # ======================================================================
    print("🏋️ 开始训练")
    
    # 基础训练参数
    training_config = {
        "per_device_train_batch_size": cfg.train.batch_size,
        "gradient_accumulation_steps": cfg.train.grad_accum,
        "learning_rate": cfg.train.learning_rate,
        "bf16": cfg.train.bf16 and is_bf16_supported(),
        "fp16": cfg.train.fp16,
        "optim": cfg.train.optim,
        "output_dir": cfg.train.output_dir,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_seq_length": cfg.train.max_seq_length,
        "logging_steps": cfg.train.logging_steps,
        "save_strategy": cfg.train.save_strategy,
    }
    
    # 训练轮数或步数
    if hasattr(cfg.train, 'max_steps') and cfg.train.max_steps:
        training_config["max_steps"] = cfg.train.max_steps
    else:
        training_config["num_train_epochs"] = cfg.train.num_epochs
    
    # 自动添加所有可选训练参数
    optional_train_params = [
        'lr_scheduler_type', 'warmup_steps', 'weight_decay', 'gradient_checkpointing',
        'save_steps', 'dataloader_num_workers', 'disable_tqdm'
    ]
    for param in optional_train_params:
        if hasattr(cfg.train, param):
            value = getattr(cfg.train, param)
            if value is not None:
                training_config[param] = value
    
    # 特殊参数处理
    if hasattr(cfg.train, 'deepspeed_config') and cfg.train.deepspeed_config:
        training_config["deepspeed"] = cfg.train.deepspeed_config
    
    if hasattr(cfg.train, 'gradient_clipping'):
        training_config["max_grad_norm"] = cfg.train.gradient_clipping
    
    # W&B设置
    if cfg.train.report_to and "wandb" in cfg.train.report_to:
        training_config.update({
            "report_to": ["wandb"],
            "run_name": cfg.wandb.run_name
        })
    else:
        training_config["report_to"] = []
    
    if eval_converted_dataset:
        training_config["per_device_eval_batch_size"] = cfg.train.batch_size
    
    # 创建训练器
    trainer_config = {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
        "train_dataset": converted_dataset,
        "args": SFTConfig(**training_config),
    }
    
    if eval_converted_dataset:
        trainer_config["eval_dataset"] = eval_converted_dataset
    
    trainer = SFTTrainer(**trainer_config)
    
    # 开始训练
    resume_checkpoint = getattr(cfg.run, 'resume_from_checkpoint', None)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    print("✅ 训练完成")
    
    # ======================================================================
    # 💾 模型保存
    # ======================================================================
    save_path = os.path.join(cfg.train.output_dir, "final_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"💾 模型已保存到: {save_path}")
    
    # W&B模型上传
    if cfg.wandb and getattr(cfg.wandb, 'log_model', False):
        try:
            wandb.save(os.path.join(save_path, "*"))
            print("📤 模型已上传到W&B")
        except Exception as e:
            print(f"⚠️ W&B模型上传失败: {e}")
    
    if cfg.wandb and cfg.train.report_to and "wandb" in cfg.train.report_to:
        wandb.finish()


if __name__ == "__main__":
    main() 