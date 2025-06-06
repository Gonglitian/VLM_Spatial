"""
merge_and_push.py
-----------------
合并 LoRA 权重到基座模型，并推送到 Hugging Face Hub.
专门针对 Qwen2.5-VL 模型优化

用法: python scripts/merge_and_push.py --base unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit \
                               --adapter ./final_model \
                               --repo Litian2002/Qwen2.5-VL-3B-Spatial-bnb-4bit
"""
import argparse, os, json, datetime, textwrap, subprocess
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True, help="基座模型（HF hub id 或本地路径）")
parser.add_argument("--adapter", required=True, help="LoRA 权重目录")
parser.add_argument("--repo", required=True, help="想要上传到 HF 的 repo_id")
parser.add_argument("--private", action="store_true", help="是否创建私有仓库")
parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32", "auto"])
args = parser.parse_args()

out_dir = "./merged_vlm"
os.makedirs(out_dir, exist_ok=True)

print("🚀 开始LoRA模型合并流程...")
print(f"📁 基座模型: {args.base}")
print(f"🔗 LoRA适配器: {args.adapter}")
print(f"📤 目标仓库: {args.repo}")

# 1️⃣ 使用 unsloth 进行模型合并（对Qwen2.5-VL最可靠）
try:
    print("🔗 Loading base model with unsloth...")
    from unsloth import FastVisionModel
    
    # 加载基座模型
    model, tokenizer = FastVisionModel.from_pretrained(args.base)
    print("✅ 基座模型加载成功")
    
    # 加载LoRA适配器
    print("🔗 Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    print("✅ LoRA适配器加载成功")
    
    # 合并权重
    print("📝 Merging weights...")
    model = model.merge_and_unload()
    print("✅ 权重合并完成")
    
    # 设置为推理模式
    FastVisionModel.for_inference(model)
    
    # 保存合并后的模型
    print("💾 Saving merged model...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print("✅ 模型保存完成")
    
except ImportError:
    print("❌ unsloth 不可用，尝试使用 transformers...")
    try:
        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel
        
        # dtype字符串到torch类型的映射
        DTYPE_MAP = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        
        if args.dtype == "auto":
            selected_torch_dtype = "auto"
        else:
            selected_torch_dtype = DTYPE_MAP.get(args.dtype.lower())
            if selected_torch_dtype is None:
                raise ValueError(f"不支持的dtype: {args.dtype}")
        
        print("🔗 Loading base model with transformers...")
        model = AutoModel.from_pretrained(
            args.base,
            torch_dtype=selected_torch_dtype,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        print("✅ 基座模型加载成功")
        
        print("🔗 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, args.adapter)
        print("✅ LoRA适配器加载成功")
        
        print("📝 Merging weights...")
        model = model.merge_and_unload()
        print("✅ 权重合并完成")
        
        print("💾 Saving merged model...")
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        print("✅ 模型保存完成")
        
    except Exception as e:
        print(f"❌ transformers 加载失败: {e}")
        raise

except Exception as e:
    print(f"❌ unsloth 加载失败: {e}")
    print("请确保已安装 unsloth 或更新 transformers 版本")
    raise

# 2️⃣ 生成 README / model card
print("📝 生成模型卡片...")
readme_path = os.path.join(out_dir, "README.md")
card = textwrap.dedent(f"""
    ---
    license: apache-2.0
    base_model: {args.base}
    merges:
      - adapter: {os.path.basename(args.adapter)}
        method: merge_and_unload
        date: {datetime.date.today()}
    tags:
      - vision-language
      - lora-merged
      - qwen2.5-vl
    ---

    # 🐍 Merged Qwen2.5-VL Model (LoRA + Base)

    This repository contains the **merged** weights of **LoRA adapter** located at `{args.adapter}` and the base
    model **{args.base}**.  
    
    The merge was performed with `peft.merge_and_unload()` on {datetime.date.today()}.

    ## Usage

    ```python
    from unsloth import FastVisionModel
    
    model, tokenizer = FastVisionModel.from_pretrained("{args.repo}")
    model = FastVisionModel.for_inference(model) # Enable native 2x faster inference
    
    # Your inference code here
    ```

    Or with transformers:

    ```python
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained("{args.repo}", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("{args.repo}", trust_remote_code=True)
    ```
    """).strip()

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(card + "\n")
print("✅ 模型卡片生成完成")

# 3️⃣ Hugging Face Hub 上传
print("🔐 准备上传到 Hugging Face Hub...")
try:
    from huggingface_hub import HfApi, HfFolder, upload_folder, create_repo
    
    api = HfApi()
    
    # 若未登录会提示输入 token
    if HfFolder.get_token() is None:
        print("🔐 需要登录 Hugging Face Hub，一次即可：")
        subprocess.run(["huggingface-cli", "login"])
    
    # 创建仓库 (存在则 skip)
    try:
        create_repo(repo_id=args.repo, private=args.private, exist_ok=True)
        print(f"📁 Repository `{args.repo}` ready.")
    except Exception as e:
        print("⚠️ Repo create error(可能已存在)：", e)
    
    # 上传目录
    print("🚀 Uploading files (this may take a while for >1 GB weights)...")
    upload_folder(
        repo_id=args.repo,
        folder_path=out_dir,
        path_in_repo=".",           # 根目录
        commit_message="Upload merged Qwen2.5-VL model",
        ignore_patterns=["*.pt"],   # 如果你只保留 .safetensors，可略过 .pt
    )
    
    print(f"✅ All done! View: https://huggingface.co/{args.repo}")
    
except ImportError:
    print("❌ huggingface_hub 不可用，跳过上传")
    print(f"💾 合并的模型已保存到: {out_dir}")
    print("📝 请手动上传或安装 huggingface_hub")
    
except Exception as e:
    print(f"❌ 上传失败: {e}")
    print(f"💾 合并的模型已保存到: {out_dir}")
    print("📝 你可以手动上传这些文件")
