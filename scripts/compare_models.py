#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比脚本 - 比较微调前后的Qwen2.5-VL模型输出
对比原始模型和final_model的推理结果，智能评估答案质量
"""

import os
import torch
import random
import argparse
import re
from datasets import load_dataset
from unsloth import FastVisionModel
from PIL import Image
import pandas as pd
from tqdm import tqdm


def extract_numbers(text):
    """从文本中提取数字，包括小数"""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return [float(n) for n in numbers if n and n != '.']


def is_numerical_question(ground_truth):
    """判断是否为数值型问题"""
    numbers = extract_numbers(ground_truth)
    return len(numbers) > 0


def calculate_numerical_similarity(ground_truth, response):
    """专门计算数值型答案的相似度"""
    gt_numbers = extract_numbers(ground_truth)
    resp_numbers = extract_numbers(response)
    
    if not gt_numbers:
        return 0.0, "非数值型答案"
    
    if not resp_numbers:
        return 0.0, "回答中无数值"
    
    # 找到最接近的数值对
    min_diff = float('inf')
    best_gt = None
    best_resp = None
    
    for gt_num in gt_numbers:
        for resp_num in resp_numbers:
            diff = abs(gt_num - resp_num)
            if diff < min_diff:
                min_diff = diff
                best_gt = gt_num
                best_resp = resp_num
    
    # 计算相对误差和绝对误差
    if best_gt != 0:
        relative_error = min_diff / abs(best_gt)
    else:
        relative_error = min_diff
    
    # 精细化的数值相似度评分
    if min_diff == 0:
        return 1.0, f"数值完全匹配 ({best_resp})"
    elif min_diff <= 0.01:
        return 0.95, f"数值极度接近 (差值: {min_diff:.3f})"
    elif min_diff <= 0.05:
        return 0.90, f"数值非常接近 (差值: {min_diff:.3f})"
    elif min_diff <= 0.1:
        return 0.85, f"数值很接近 (差值: {min_diff:.3f})"
    elif min_diff <= 0.2:
        return 0.75, f"数值接近 (差值: {min_diff:.3f})"
    elif min_diff <= 0.5:
        return 0.60, f"数值较接近 (差值: {min_diff:.3f})"
    elif min_diff <= 1.0:
        return 0.40, f"数值有差距 (差值: {min_diff:.3f})"
    elif min_diff <= 2.0:
        return 0.25, f"数值差距较大 (差值: {min_diff:.3f})"
    elif relative_error <= 0.5:  # 相对误差50%以内
        return 0.15, f"数值差距大 (差值: {min_diff:.3f}, 相对误差: {relative_error:.2%})"
    else:
        return 0.05, f"数值差距很大 (差值: {min_diff:.3f}, 相对误差: {relative_error:.2%})"


def calculate_answer_similarity(ground_truth, response):
    """计算答案与真实值的相似度"""
    gt_lower = ground_truth.lower().strip()
    resp_lower = response.lower().strip()
    
    # 1. 完全匹配
    if gt_lower == resp_lower:
        return 1.0, "完全匹配"
    
    # 2. 包含匹配
    if gt_lower in resp_lower or resp_lower in gt_lower:
        return 0.8, "包含匹配"
    
    # 3. 优先处理数值型问题
    if is_numerical_question(ground_truth):
        numerical_sim, numerical_reason = calculate_numerical_similarity(ground_truth, response)
        # 对于数值型问题，如果找到了数值，优先使用数值相似度
        if "无数值" not in numerical_reason and "非数值" not in numerical_reason:
            return numerical_sim, f"[数值型] {numerical_reason}"
        # 如果没找到数值，降级处理但保持标识
        else:
            # 继续下面的文本匹配，但标记为数值型问题
            pass
    
    # 4. 传统数值匹配（作为备选）
    gt_numbers = extract_numbers(ground_truth)
    resp_numbers = extract_numbers(response)
    
    if gt_numbers and resp_numbers:
        # 找最接近的数值
        min_diff = float('inf')
        for gt_num in gt_numbers:
            for resp_num in resp_numbers:
                diff = abs(gt_num - resp_num)
                min_diff = min(min_diff, diff)
        
        # 基础数值相似度（保持原有逻辑作为兼容）
        if min_diff == 0:
            return 0.9, f"数值完全匹配"
        elif min_diff <= 0.5:
            return 0.7, f"数值接近 (差值: {min_diff:.2f})"
        elif min_diff <= 1.0:
            return 0.5, f"数值较接近 (差值: {min_diff:.2f})"
        else:
            return 0.2, f"数值相差较大 (差值: {min_diff:.2f})"
    
    # 5. 是否型答案匹配
    yes_words = ['yes', 'true', '是', '对', '正确']
    no_words = ['no', 'false', '否', '不', '错误']
    
    gt_is_yes = any(word in gt_lower for word in yes_words)
    gt_is_no = any(word in gt_lower for word in no_words)
    resp_is_yes = any(word in resp_lower for word in yes_words)
    resp_is_no = any(word in resp_lower for word in no_words)
    
    if (gt_is_yes and resp_is_yes) or (gt_is_no and resp_is_no):
        return 0.6, "是否型匹配"
    elif (gt_is_yes and resp_is_no) or (gt_is_no and resp_is_yes):
        return 0.1, "是否型相反"
    
    # 6. 词汇重叠度
    gt_words = set(gt_lower.split())
    resp_words = set(resp_lower.split())
    
    if gt_words and resp_words:
        overlap = len(gt_words.intersection(resp_words))
        union = len(gt_words.union(resp_words))
        jaccard = overlap / union if union > 0 else 0
        
        # 如果是数值型问题但没匹配到数值，降低词汇重叠度的权重
        if is_numerical_question(ground_truth):
            jaccard_penalty = 0.5  # 数值型问题的词汇匹配权重减半
            if jaccard > 0.5:
                return 0.2, f"[数值型-词汇] 词汇重叠度高但缺乏数值 ({jaccard:.2f})"
            elif jaccard > 0.2:
                return 0.15, f"[数值型-词汇] 词汇重叠度中等但缺乏数值 ({jaccard:.2f})"
            else:
                return 0.05, f"[数值型-词汇] 词汇重叠度低且缺乏数值 ({jaccard:.2f})"
        else:
            # 非数值型问题的正常词汇匹配
            if jaccard > 0.5:
                return 0.4, f"词汇重叠度高 ({jaccard:.2f})"
            elif jaccard > 0.2:
                return 0.3, f"词汇重叠度中等 ({jaccard:.2f})"
            else:
                return 0.1, f"词汇重叠度低 ({jaccard:.2f})"
    
    return 0.0, "无匹配"


def save_sample_image(image, sample_index, question, ground_truth, original_response, finetuned_response, better_model, output_dir="comparison_samples"):
    """保存样本图片和详细信息"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    image_path = os.path.join(output_dir, f"sample_{sample_index}.png")
    image.save(image_path)
    
    # 保存详细信息
    info_path = os.path.join(output_dir, f"sample_{sample_index}_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"样本索引: {sample_index}\n")
        f.write(f"问题: {question}\n")
        f.write(f"标准答案: {ground_truth}\n")
        f.write(f"原始模型回答: {original_response}\n")
        f.write(f"微调模型回答: {finetuned_response}\n")
        f.write(f"更好的模型: {better_model}\n")
    
    return image_path, info_path


def load_models():
    """加载原始模型和微调后的模型"""
    print("🚀 加载模型...")
    
    # 加载原始模型
    print("  📥 加载原始模型: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    original_model, original_tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        load_in_4bit=True
    )
    print("  ✅ 原始模型加载完成")
    
    # 加载微调后的模型
    finetuned_model_path = "outputs/final_model"
    if not os.path.exists(finetuned_model_path):
        # 尝试其他可能的路径
        possible_paths = [
            "final_model",
            "outputs/checkpoint-20/final_model", 
            "outputs/checkpoint-32/final_model",
            "outputs/2025-06-03/14-58-59/outputs/final_model"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                finetuned_model_path = path
                break
        else:
            raise FileNotFoundError("❌ 找不到微调后的模型！请检查路径")
    
    print(f"  📥 加载微调模型: {finetuned_model_path}")
    finetuned_model, finetuned_tokenizer = FastVisionModel.from_pretrained(
        model_name=finetuned_model_path,
        load_in_4bit=True
    )
    print("  ✅ 微调模型加载完成")
    
    # 设置为推理模式
    FastVisionModel.for_inference(original_model)
    FastVisionModel.for_inference(finetuned_model)
    
    return (original_model, original_tokenizer), (finetuned_model, finetuned_tokenizer)


def load_test_samples(num_samples=10):
    """从数据集中随机加载测试样本"""
    print(f"📊 加载测试数据集: Litian2002/spatialvlm_qa")
    
    try:
        dataset = load_dataset("Litian2002/spatialvlm_qa", split="train")
        print(f"  📈 数据集大小: {len(dataset)}")
        
        # 随机选择样本
        total_samples = len(dataset)
        random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        test_samples = []
        for idx in random_indices:
            sample = dataset[idx]
            test_samples.append({
                "index": idx,
                "image": sample["image"],
                "question": sample["question"],
                "ground_truth": sample["answer"]
            })
        
        print(f"  ✅ 随机选择了 {len(test_samples)} 个测试样本")
        return test_samples
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return []


def generate_response(model, tokenizer, image, question):
    """使用模型生成回答"""
    try:
        # 参考unsloth官方示例的正确格式
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},  # 注意：这里不需要传image参数
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # 应用聊天模板
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 正确的调用方式：image作为第一个参数
        inputs = tokenizer(
            image,           # 图像作为第一个参数
            input_text,      # 文本作为第二个参数
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 提取回答部分
        input_token_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_token_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        return response if response else "🤖 模型未产生有效回答"
        
    except Exception as e:
        print(f"⚠️ 推理错误详情: {e}")
        # 备用方案：仅文本推理
        try:
            simple_input = tokenizer(
                f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **simple_input,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            input_length = simple_input['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            return f"📝 [仅文本模式] {response}" if response else "❌ 备用推理也失败"
            
        except Exception as backup_error:
            return f"❌ 完全失败: {str(e)} | 备用: {str(backup_error)}"


def compare_models_on_samples(original_model_info, finetuned_model_info, test_samples):
    """对比两个模型在测试样本上的表现"""
    original_model, original_tokenizer = original_model_info
    finetuned_model, finetuned_tokenizer = finetuned_model_info
    
    print("\n🔍 开始模型对比测试...")
    
    results = []
    
    for i, sample in enumerate(tqdm(test_samples, desc="🧪 测试样本")):
        print(f"\n{'='*80}")
        print(f"📝 测试样本 {i+1}/{len(test_samples)} (数据集索引: {sample['index']})")
        print(f"{'='*80}")
        
        # 显示问题和图像信息
        print(f"❓ 问题: {sample['question']}")
        print(f"🖼️  图像尺寸: {sample['image'].size}")
        print(f"✅ 标准答案: {sample['ground_truth']}")
        
        # 原始模型推理
        print("\n🤖 原始模型推理中...")
        original_response = generate_response(
            original_model, original_tokenizer, 
            sample['image'], sample['question']
        )
        print(f"📤 原始模型回答: {original_response}")
        
        # 微调模型推理  
        print("\n🎯 微调模型推理中...")
        finetuned_response = generate_response(
            finetuned_model, finetuned_tokenizer,
            sample['image'], sample['question']
        )
        print(f"📤 微调模型回答: {finetuned_response}")
        
        # 智能评估答案质量
        original_similarity, original_reason = calculate_answer_similarity(
            sample['ground_truth'], original_response
        )
        finetuned_similarity, finetuned_reason = calculate_answer_similarity(
            sample['ground_truth'], finetuned_response
        )
        
        # 判断哪个模型更好
        if finetuned_similarity > original_similarity:
            better_model = "微调模型"
            winner_emoji = "🎯"
        elif original_similarity > finetuned_similarity:
            better_model = "原始模型"
            winner_emoji = "🤖"
        else:
            better_model = "平局"
            winner_emoji = "🤝"
        
        print(f"\n📊 智能评估:")
        print(f"  🤖 原始模型相似度: {original_similarity:.2f} ({original_reason})")
        print(f"  🎯 微调模型相似度: {finetuned_similarity:.2f} ({finetuned_reason})")
        print(f"  {winner_emoji} 更好的模型: {better_model}")
        
        # 保存样本图片和信息
        image_path, info_path = save_sample_image(
            sample['image'], sample['index'], sample['question'],
            sample['ground_truth'], original_response, finetuned_response, better_model
        )
        print(f"  💾 样本已保存: {image_path}")
        
        # 保存结果
        results.append({
            "sample_index": sample['index'],
            "question": sample['question'],
            "ground_truth": sample['ground_truth'],
            "original_response": original_response,
            "finetuned_response": finetuned_response,
            "original_similarity": original_similarity,
            "finetuned_similarity": finetuned_similarity,
            "original_reason": original_reason,
            "finetuned_reason": finetuned_reason,
            "better_model": better_model,
            "image_size": str(sample['image'].size),
            "image_path": image_path,
            "info_path": info_path
        })
    
    return results


def save_results(results, output_file="model_comparison_results.csv"):
    """保存对比结果到CSV文件"""
    print(f"\n💾 保存结果到: {output_file}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("✅ 结果保存完成!")
    
    # 显示详细统计信息
    print(f"\n📈 详细对比统计:")
    print(f"  总测试样本: {len(results)}")
    
    # 分类统计：数值型 vs 非数值型问题
    numerical_results = []
    non_numerical_results = []
    
    for result in results:
        if is_numerical_question(result['ground_truth']):
            numerical_results.append(result)
        else:
            non_numerical_results.append(result)
    
    print(f"  📊 数值型问题: {len(numerical_results)} ({len(numerical_results)/len(results)*100:.1f}%)")
    print(f"  📝 非数值型问题: {len(non_numerical_results)} ({len(non_numerical_results)/len(results)*100:.1f}%)")
    
    # 统计获胜情况
    original_wins = sum(1 for r in results if r['better_model'] == '原始模型')
    finetuned_wins = sum(1 for r in results if r['better_model'] == '微调模型')
    ties = sum(1 for r in results if r['better_model'] == '平局')
    
    print(f"\n🏆 总体模型对比结果:")
    print(f"  🤖 原始模型获胜: {original_wins}/{len(results)} ({original_wins/len(results)*100:.1f}%)")
    print(f"  🎯 微调模型获胜: {finetuned_wins}/{len(results)} ({finetuned_wins/len(results)*100:.1f}%)")
    print(f"  🤝 平局: {ties}/{len(results)} ({ties/len(results)*100:.1f}%)")
    
    # 数值型问题的专门统计
    if numerical_results:
        num_original_wins = sum(1 for r in numerical_results if r['better_model'] == '原始模型')
        num_finetuned_wins = sum(1 for r in numerical_results if r['better_model'] == '微调模型')
        num_ties = sum(1 for r in numerical_results if r['better_model'] == '平局')
        
        print(f"\n🔢 数值型问题对比结果:")
        print(f"  🤖 原始模型获胜: {num_original_wins}/{len(numerical_results)} ({num_original_wins/len(numerical_results)*100:.1f}%)")
        print(f"  🎯 微调模型获胜: {num_finetuned_wins}/{len(numerical_results)} ({num_finetuned_wins/len(numerical_results)*100:.1f}%)")
        print(f"  🤝 平局: {num_ties}/{len(numerical_results)} ({num_ties/len(numerical_results)*100:.1f}%)")
        
        # 数值型问题的平均相似度
        num_avg_original = sum(r['original_similarity'] for r in numerical_results) / len(numerical_results)
        num_avg_finetuned = sum(r['finetuned_similarity'] for r in numerical_results) / len(numerical_results)
        
        print(f"\n📊 数值型问题平均相似度:")
        print(f"  🤖 原始模型: {num_avg_original:.3f}")
        print(f"  🎯 微调模型: {num_avg_finetuned:.3f}")
        print(f"  📈 改进幅度: {(num_avg_finetuned - num_avg_original):.3f}")
        
        # 数值精度统计
        high_precision_finetuned = sum(1 for r in numerical_results if r['finetuned_similarity'] >= 0.8)
        high_precision_original = sum(1 for r in numerical_results if r['original_similarity'] >= 0.8)
        
        print(f"\n🎯 数值高精度匹配 (相似度≥0.8):")
        print(f"  🤖 原始模型: {high_precision_original}/{len(numerical_results)} ({high_precision_original/len(numerical_results)*100:.1f}%)")
        print(f"  🎯 微调模型: {high_precision_finetuned}/{len(numerical_results)} ({high_precision_finetuned/len(numerical_results)*100:.1f}%)")
    
    # 非数值型问题的统计
    if non_numerical_results:
        nnum_original_wins = sum(1 for r in non_numerical_results if r['better_model'] == '原始模型')
        nnum_finetuned_wins = sum(1 for r in non_numerical_results if r['better_model'] == '微调模型')
        nnum_ties = sum(1 for r in non_numerical_results if r['better_model'] == '平局')
        
        print(f"\n📝 非数值型问题对比结果:")
        print(f"  🤖 原始模型获胜: {nnum_original_wins}/{len(non_numerical_results)} ({nnum_original_wins/len(non_numerical_results)*100:.1f}%)")
        print(f"  🎯 微调模型获胜: {nnum_finetuned_wins}/{len(non_numerical_results)} ({nnum_finetuned_wins/len(non_numerical_results)*100:.1f}%)")
        print(f"  🤝 平局: {nnum_ties}/{len(non_numerical_results)} ({nnum_ties/len(non_numerical_results)*100:.1f}%)")
    
    # 总体平均相似度
    avg_original_similarity = sum(r['original_similarity'] for r in results) / len(results)
    avg_finetuned_similarity = sum(r['finetuned_similarity'] for r in results) / len(results)
    
    print(f"\n📊 总体平均相似度:")
    print(f"  🤖 原始模型: {avg_original_similarity:.3f}")
    print(f"  🎯 微调模型: {avg_finetuned_similarity:.3f}")
    print(f"  📈 改进幅度: {(avg_finetuned_similarity - avg_original_similarity):.3f}")
    
    # 最佳和最差样本
    best_improvement = max(results, key=lambda x: x['finetuned_similarity'] - x['original_similarity'])
    worst_regression = min(results, key=lambda x: x['finetuned_similarity'] - x['original_similarity'])
    
    print(f"\n🎯 关键样本:")
    print(f"  🚀 最大改进样本: #{best_improvement['sample_index']} (改进: {best_improvement['finetuned_similarity'] - best_improvement['original_similarity']:.3f})")
    print(f"  ⚠️  最大退步样本: #{worst_regression['sample_index']} (退步: {worst_regression['finetuned_similarity'] - worst_regression['original_similarity']:.3f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对比微调前后的Qwen2.5-VL模型")
    parser.add_argument("--num_samples", type=int, default=100, help="测试样本数量")
    parser.add_argument("--output", type=str, default="model_comparison_results.csv", help="输出文件名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("🔍 Qwen2.5-VL 模型对比工具 (增强数值型问题评估)")
    print("="*80)
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    try:
        # 1. 加载模型
        original_model_info, finetuned_model_info = load_models()
        
        # 2. 加载测试样本
        test_samples = load_test_samples(args.num_samples)
        if not test_samples:
            print("❌ 没有可用的测试样本")
            return
        
        # 3. 执行对比测试
        results = compare_models_on_samples(
            original_model_info, finetuned_model_info, test_samples
        )
        
        # 4. 保存结果
        save_results(results, args.output)
        
        print("\n🎉 模型对比完成!")
        print(f"📄 详细结果请查看: {args.output}")
        print(f"📁 测试样本保存在: comparison_samples/")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 