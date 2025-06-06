#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型评估脚本 - vLLM批量推理版本
使用Hydra配置管理参数，支持文本一致性和数值误差百分比双重评估指标
专注于vLLM批量推理，大幅提升评估速度
图表内容使用英文显示
"""

import os
import torch
import random
import hydra
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# vLLM相关导入
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("✅ vLLM库可用，启用批量推理加速")
except ImportError:
    VLLM_AVAILABLE = False
    print("❌ vLLM库不可用，请安装vLLM: pip install vllm")
    exit(1)

# 设置英文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_numbers(text):
    """从文本中提取数字，包括小数和负数"""
    if pd.isna(text) or text is None:
        return []
    numbers = re.findall(r'-?\d+\.?\d*', str(text))
    return [float(n) for n in numbers if n and n != '.']


def is_numerical_question(ground_truth):
    """判断是否为数值型问题"""
    numbers = extract_numbers(ground_truth)
    return len(numbers) > 0


def calculate_text_consistency(ground_truth, response, config):
    """
    指标1: 文本一致性计算
    使用多种方法计算文本相似度
    """
    if not ground_truth or not response:
        return 0.0, "空文本"
    
    gt_lower = str(ground_truth).lower().strip()
    resp_lower = str(response).lower().strip()
    
    # 1. 完全匹配
    if gt_lower == resp_lower:
        return 1.0, "完全匹配"
    
    # 2. 包含匹配
    if gt_lower in resp_lower or resp_lower in gt_lower:
        return 0.8, "包含匹配"
    
    # 3. 序列匹配度 (基于difflib)
    sequence_similarity = SequenceMatcher(None, gt_lower, resp_lower).ratio()
    
    # 4. 词汇重叠度 (Jaccard相似度)
    gt_words = set(gt_lower.split())
    resp_words = set(resp_lower.split())
    
    if gt_words and resp_words:
        intersection = len(gt_words.intersection(resp_words))
        union = len(gt_words.union(resp_words))
        jaccard_similarity = intersection / union if union > 0 else 0
    else:
        jaccard_similarity = 0
    
    # 5. 是否型答案匹配
    yes_words = ['yes', 'true', '是', '对', '正确', 'correct']
    no_words = ['no', 'false', '否', '不', '错误', 'incorrect', 'wrong']
    
    gt_is_yes = any(word in gt_lower for word in yes_words)
    gt_is_no = any(word in gt_lower for word in no_words)
    resp_is_yes = any(word in resp_lower for word in yes_words)
    resp_is_no = any(word in resp_lower for word in no_words)
    
    boolean_match = 0
    boolean_reason = ""
    if (gt_is_yes and resp_is_yes) or (gt_is_no and resp_is_no):
        boolean_match = 0.6
        boolean_reason = "是否型匹配"
    elif (gt_is_yes and resp_is_no) or (gt_is_no and resp_is_yes):
        boolean_match = 0.1
        boolean_reason = "是否型相反"
    
    # 使用配置中的权重进行综合评分
    weights = config.metrics.text_consistency
    weighted_score = (sequence_similarity * weights.sequence_weight + 
                     jaccard_similarity * weights.jaccard_weight + 
                     boolean_match * weights.boolean_weight)
    
    max_similarity = max(sequence_similarity, jaccard_similarity, boolean_match, weighted_score)
    
    if max_similarity >= 0.8:
        reason = f"高文本一致性 (序列:{sequence_similarity:.2f}, 词汇:{jaccard_similarity:.2f})"
    elif max_similarity >= 0.5:
        reason = f"中等文本一致性 (序列:{sequence_similarity:.2f}, 词汇:{jaccard_similarity:.2f})"
    elif boolean_reason:
        reason = boolean_reason
    else:
        reason = f"低文本一致性 (序列:{sequence_similarity:.2f}, 词汇:{jaccard_similarity:.2f})"
    
    return max_similarity, reason


def calculate_numerical_error_percentage(ground_truth, response):
    """
    指标2: 数值误差百分比计算
    专门针对数值型问题计算误差百分比
    """
    gt_numbers = extract_numbers(ground_truth)
    resp_numbers = extract_numbers(response)
    
    if not gt_numbers:
        return None, "非数值型问题"
    
    if not resp_numbers:
        return None, "回答中无数值"
    
    # 找到最接近的数值对
    min_error_percentage = float('inf')
    best_gt = None
    best_resp = None
    
    for gt_num in gt_numbers:
        for resp_num in resp_numbers:
            if gt_num != 0:
                error_percentage = abs(gt_num - resp_num) / abs(gt_num) * 100
            else:
                error_percentage = abs(resp_num) * 100  # 当真实值为0时
            
            if error_percentage < min_error_percentage:
                min_error_percentage = error_percentage
                best_gt = gt_num
                best_resp = resp_num
    
    return min_error_percentage, f"数值对比: {best_resp} vs {best_gt}"


def calculate_comprehensive_similarity(ground_truth, response, config):
    """
    综合相似度计算 - 结合两个指标
    """
    # 指标1: 文本一致性
    text_consistency, text_reason = calculate_text_consistency(ground_truth, response, config)
    
    # 指标2: 数值误差百分比（仅对数值型问题）
    if is_numerical_question(ground_truth):
        error_percentage, numerical_reason = calculate_numerical_error_percentage(ground_truth, response)
        
        if error_percentage is not None:
            # 使用配置中的评分策略将误差百分比转换为相似度分数
            scoring = config.metrics.numerical_scoring
            
            if error_percentage == 0:
                numerical_score = scoring.perfect_match
            elif error_percentage <= 1:
                numerical_score = scoring.excellent
            elif error_percentage <= 5:
                numerical_score = scoring.very_good
            elif error_percentage <= 10:
                numerical_score = scoring.good
            elif error_percentage <= 20:
                numerical_score = scoring.acceptable
            elif error_percentage <= 50:
                numerical_score = scoring.poor
            else:
                numerical_score = scoring.very_poor
            
            # 使用配置中的权重进行综合评分
            weights = config.metrics.score_weights
            final_score = (weights.numerical_precision * numerical_score + 
                          weights.text_consistency * text_consistency)
            reason = f"[数值型] 误差:{error_percentage:.1f}%, 文本一致性:{text_consistency:.2f}"
        else:
            # 数值型问题但未找到数值，使用降权系数
            penalty = config.metrics.score_weights.no_numerical_penalty
            final_score = text_consistency * penalty
            reason = f"[数值型-无数值] {text_reason}"
    else:
        # 非数值型问题，主要使用文本一致性
        final_score = text_consistency
        reason = f"[文本型] {text_reason}"
    
    return final_score, reason


def load_model(config):
    """加载模型 - 支持vLLM批量推理"""
    global VLLM_AVAILABLE  # 声明为全局变量
    print("🚀 加载模型...")
    
    model_id = config.model.model_id
    
    if VLLM_AVAILABLE:
        # 使用vLLM加载HuggingFace模型（不支持本地LoRA）
        print(f"  🚀 使用vLLM批量推理引擎加载: {model_id}")
        try:
            llm = LLM(
                model=model_id,
                trust_remote_code=True,
                max_model_len=4096,  # 根据需要调整
            )
            
            print("  ✅ vLLM模型加载完成")
            return llm, None, f"vLLM({model_id})"
            
        except Exception as e:
            print(f"❌ vLLM模型加载失败: {e}")
    else:
        # error
        raise ValueError(f"模型 {model_id} 不存在")
    
def load_test_samples(config):
    """直接加载数据集并随机采样测试样本"""
    num_samples = config.data.num_samples
    seed = config.run.seed
    dataset_name = config.data.dataset_name
    dataset_split = config.data.dataset_split
    
    print(f"📊 加载测试数据集: {dataset_name}")
    print(f"  🎯 目标样本数量: {num_samples}")
    
    try:
        # 设置随机种子确保可重现性
        random.seed(seed)
        
        # 直接加载整个数据集
        print("  📦 加载完整数据集...")
        dataset = load_dataset(dataset_name, split=dataset_split)
        print(f"  📈 数据集大小: {len(dataset)}")
        
        # 随机选择样本
        total_samples = len(dataset)
        if num_samples > total_samples:
            print(f"  ⚠️ 请求样本数({num_samples})大于数据集大小({total_samples})，使用全部样本")
            num_samples = total_samples
        
        random_indices = random.sample(range(total_samples), num_samples)
        random_indices.sort()  # 排序以便查看
        
        print(f"  🎲 随机选中索引: {random_indices}")
        
        test_samples = []
        for idx in random_indices:
            sample = dataset[idx]
            test_samples.append({
                "index": idx,
                "image": sample["image"],
                "question": sample["question"],
                "ground_truth": sample["answer"]
            })
        
        print(f"  ✅ 成功加载 {len(test_samples)} 个测试样本")
        return test_samples
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return []


def batch_generate_responses_vllm(llm, test_samples, config):
    """使用vLLM批量生成回答"""
    print("🚀 使用vLLM批量推理...")
    
    # 准备批量输入
    batch_inputs = []
    for sample in test_samples:
        # 参考train.py的数据格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["question"]},
                    {"type": "image", "image": sample["image"]}
                ]
            }
        ]
        
        # 应用聊天模板
        prompt = llm.get_tokenizer().apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": sample["image"]}
        })
    
    # 设置采样参数
    gen_config = config.generation
    sampling_params = SamplingParams(
        temperature=gen_config.temperature,
        max_tokens=gen_config.max_new_tokens,
        top_p=0.9,  # 添加top_p参数
        stop=None,
    )
    
    print(f"  📝 批量推理 {len(batch_inputs)} 个样本...")
    
    try:
        # 执行批量推理
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        
        # 提取回答
        responses = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                response = output.outputs[0].text.strip()
                responses.append(response if response else "🤖 模型未产生有效回答")
            else:
                responses.append("❌ 推理失败: 无输出")
        
        print(f"  ✅ 批量推理完成，共生成 {len(responses)} 个回答")
        return responses
        
    except Exception as e:
        print(f"❌ vLLM批量推理失败: {e}")
        raise

def save_sample_image(image, sample_index, question, ground_truth, response, score, config):
    """保存样本图片和详细信息"""
    base_dir = config.output.base_output_dir
    samples_dir = os.path.join(base_dir, config.output.samples_dir)
    os.makedirs(samples_dir, exist_ok=True)
    
    # 保存图片
    image_path = os.path.join(samples_dir, f"sample_{sample_index}.png")
    image.save(image_path)
    
    # 保存详细信息
    info_path = os.path.join(samples_dir, f"sample_{sample_index}_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"样本索引: {sample_index}\n")
        f.write(f"问题: {question}\n")
        f.write(f"标准答案: {ground_truth}\n")
        f.write(f"模型回答: {response}\n")
        f.write(f"评估分数: {score:.3f}\n")
    
    return image_path, info_path


def evaluate_model_on_samples(llm, model_path, test_samples, config):
    """评估模型在测试样本上的表现 - vLLM批量推理"""
    print(f"\n🔍 开始模型评估测试...")
    print(f"📋 评估模型: {model_path}")
    
    # 使用vLLM批量推理
    print("🚀 使用vLLM批量推理模式")
    responses = batch_generate_responses_vllm(llm, test_samples, config)
    
    # 处理结果
    results = []
    print(f"\n📊 处理评估结果...")
    
    for i, (sample, response) in enumerate(zip(test_samples, responses)):
        print(f"\n{'='*80}")
        print(f"📝 处理样本 {i+1}/{len(test_samples)} (数据集索引: {sample['index']})")
        print(f"{'='*80}")
        
        # 显示问题和图像信息
        print(f"❓ 问题: {sample['question']}")
        print(f"🖼️  图像尺寸: {sample['image'].size}")
        print(f"✅ 标准答案: {sample['ground_truth']}")
        print(f"📤 模型回答: {response}")
        
        # 计算综合相似度
        similarity, reason = calculate_comprehensive_similarity(
            sample['ground_truth'], response, config
        )
        
        # 计算数值误差百分比（如果适用）
        error_pct = None
        if is_numerical_question(sample['ground_truth']):
            error, _ = calculate_numerical_error_percentage(sample['ground_truth'], response)
            error_pct = error
        
        print(f"\n📊 评估结果:")
        print(f"  📈 相似度分数: {similarity:.3f} ({reason})")
        if error_pct is not None:
            print(f"  📊 数值误差: {error_pct:.1f}%")
        
        # 保存样本图片和信息
        image_path, info_path = save_sample_image(
            sample['image'], sample['index'], sample['question'],
            sample['ground_truth'], response, similarity, config
        )
        print(f"  💾 样本已保存: {image_path}")
        
        # 保存结果
        results.append({
            "sample_index": sample['index'],
            "question": sample['question'],
            "ground_truth": sample['ground_truth'],
            "response": response,
            "similarity": similarity,
            "reason": reason,
            "error_percentage": error_pct,
            "is_numerical": is_numerical_question(sample['ground_truth']),
            "image_size": str(sample['image'].size),
            "image_path": image_path,
            "info_path": info_path
        })
    
    return results


def save_results_to_csv(results, config):
    """保存评估结果到CSV文件"""
    base_dir = config.output.base_output_dir
    os.makedirs(base_dir, exist_ok=True)
    
    output_file = os.path.join(base_dir, config.output.csv_filename)
    print(f"\n💾 保存结果到: {output_file}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("✅ 结果保存完成!")
    return df


def create_visualization(df, config):
    """创建英文可视化图表"""
    print("\n🎨 生成可视化图表...")
    
    base_dir = config.output.base_output_dir
    output_dir = os.path.join(base_dir, config.output.visualization_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 应用配置中的可视化设置
    viz_config = config.visualization
    plt.style.use(viz_config.style)
    plt.rcParams['figure.figsize'] = viz_config.figure_size
    plt.rcParams['font.size'] = viz_config.font_size
    
    colors = viz_config.colors
    
    # 1. 创建主要评估图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 相似度分数分布直方图
    ax1.hist(df['similarity'], bins=15, color=colors.primary, alpha=0.7, edgecolor='black')
    ax1.axvline(df['similarity'].mean(), color=colors.error, linestyle='--', linewidth=2, 
                label=f'Mean: {df["similarity"].mean():.3f}')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Similarity Score Distribution', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2 问题类型分布饼图
    type_counts = df['is_numerical'].value_counts()
    
    # 根据实际数据动态生成标签
    type_labels = []
    for is_num in type_counts.index:
        if is_num:
            type_labels.append('Numerical Questions')
        else:
            type_labels.append('Non-numerical Questions')
    
    # 确保颜色数量足够
    available_colors = [colors.secondary, colors.accent][:len(type_counts)]
    
    wedges, texts, autotexts = ax2.pie(type_counts.values, labels=type_labels, 
                                      autopct='%1.1f%%', colors=available_colors, startangle=90)
    ax2.set_title('Question Type Distribution', fontweight='bold', fontsize=14)
    
    # 1.3 分类型相似度对比
    numerical_df = df[df['is_numerical']]
    non_numerical_df = df[~df['is_numerical']]
    
    categories = ['Numerical Questions', 'Non-numerical Questions']
    avg_scores = [
        numerical_df['similarity'].mean() if len(numerical_df) > 0 else 0,
        non_numerical_df['similarity'].mean() if len(non_numerical_df) > 0 else 0
    ]
    
    bars = ax3.bar(categories, avg_scores, color=[colors.accent, colors.secondary], alpha=0.8)
    ax3.set_ylabel('Average Similarity Score')
    ax3.set_title('Average Similarity by Question Type', fontweight='bold', fontsize=14)
    ax3.set_ylim(0, 1)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 1.4 相似度等级分布
    def get_similarity_level(score):
        if score >= 0.9:
            return 'Excellent (≥0.9)'
        elif score >= 0.7:
            return 'Good (0.7-0.9)'
        elif score >= 0.5:
            return 'Fair (0.5-0.7)'
        else:
            return 'Poor (<0.5)'
    
    df['similarity_level'] = df['similarity'].apply(get_similarity_level)
    level_counts = df['similarity_level'].value_counts()
    
    # 确保顺序正确
    level_order = ['Excellent (≥0.9)', 'Good (0.7-0.9)', 'Fair (0.5-0.7)', 'Poor (<0.5)']
    level_counts = level_counts.reindex(level_order, fill_value=0)
    
    bars = ax4.bar(range(len(level_counts)), level_counts.values, 
                   color=colors.primary, alpha=0.8)
    ax4.set_ylabel('Number of Samples')
    ax4.set_xlabel('Similarity Level')
    ax4.set_title('Similarity Level Distribution', fontweight='bold', fontsize=14)
    ax4.set_xticks(range(len(level_counts)))
    ax4.set_xticklabels(level_counts.index, rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path1 = os.path.join(output_dir, 'model_evaluation_overview.png')
    plt.savefig(output_path1, dpi=viz_config.dpi, bbox_inches='tight', 
                facecolor=colors.background)
    print(f"  💾 保存: {output_path1}")
    plt.close()
    
    # 2. 数值型问题专门分析
    if len(numerical_df) > 0:
        valid_numerical = numerical_df.dropna(subset=['error_percentage'])
        
        if len(valid_numerical) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 2.1 数值误差分布
            ax1.hist(valid_numerical['error_percentage'], bins=10, color=colors.accent, 
                    alpha=0.7, edgecolor='black')
            ax1.axvline(valid_numerical['error_percentage'].mean(), color=colors.error, 
                       linestyle='--', linewidth=2, 
                       label=f'Mean Error: {valid_numerical["error_percentage"].mean():.1f}%')
            ax1.set_xlabel('Error Percentage (%)')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Numerical Error Distribution', fontweight='bold', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2.2 数值精度等级分布
            def get_error_level(error):
                if pd.isna(error):
                    return 'Cannot Calculate'
                elif error == 0:
                    return 'Perfect (0%)'
                elif error <= 5:
                    return 'Excellent (≤5%)'
                elif error <= 20:
                    return 'Good (≤20%)'
                else:
                    return 'Fair (>20%)'
            
            valid_numerical['error_level'] = valid_numerical['error_percentage'].apply(get_error_level)
            error_level_counts = valid_numerical['error_level'].value_counts()
            
            # 确保顺序正确
            error_level_order = ['Perfect (0%)', 'Excellent (≤5%)', 'Good (≤20%)', 'Fair (>20%)']
            error_level_counts = error_level_counts.reindex(error_level_order, fill_value=0)
            
            # 为饼图准备颜色
            pie_colors = [colors.secondary, colors.accent, colors.primary, colors.error]
            available_pie_colors = pie_colors[:len(error_level_counts[error_level_counts > 0])]
            
            # 只显示有数据的等级
            non_zero_counts = error_level_counts[error_level_counts > 0]
            
            wedges, texts, autotexts = ax2.pie(non_zero_counts.values, labels=non_zero_counts.index,
                                              autopct='%1.1f%%', colors=available_pie_colors, startangle=90)
            ax2.set_title(f'Numerical Precision Level Distribution\n({len(valid_numerical)} Numerical Questions)', 
                         fontweight='bold', fontsize=14)
            
            # 2.3 数值型问题相似度 vs 误差散点图
            ax3.scatter(valid_numerical['error_percentage'], valid_numerical['similarity'], 
                       color=colors.primary, alpha=0.6, s=50)
            ax3.set_xlabel('Error Percentage (%)')
            ax3.set_ylabel('Similarity Score')
            ax3.set_title('Similarity vs Numerical Error Relationship', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(valid_numerical) > 1:
                z = np.polyfit(valid_numerical['error_percentage'], valid_numerical['similarity'], 1)
                p = np.poly1d(z)
                ax3.plot(valid_numerical['error_percentage'], p(valid_numerical['error_percentage']), 
                        color=colors.error, linestyle='--', alpha=0.8, label='Trend Line')
                ax3.legend()
            
            # 2.4 低分样本分析（相似度<0.5的样本）
            low_score_samples = df[df['similarity'] < 0.5]
            if len(low_score_samples) > 0:
                type_counts = low_score_samples['is_numerical'].value_counts()
                
                # 根据实际数据动态生成标签
                type_labels = []
                for is_num in type_counts.index:
                    if is_num:
                        type_labels.append('Numerical')
                    else:
                        type_labels.append('Non-numerical')
                
                # 确保颜色数量足够
                low_score_colors = [colors.secondary, colors.accent][:len(type_counts)]
                
                wedges, texts, autotexts = ax4.pie(type_counts.values, labels=type_labels,
                                                  autopct='%1.1f%%', colors=low_score_colors, startangle=90)
                ax4.set_title(f'Low Score Sample Type Distribution\n(Similarity<0.5, {len(low_score_samples)} samples)', 
                             fontweight='bold', fontsize=14)
            else:
                ax4.text(0.5, 0.5, 'No Low Score Samples\n(All samples ≥0.5)', 
                        ha='center', va='center', transform=ax4.transAxes, 
                        fontsize=16, fontweight='bold', color=colors.primary)
                ax4.set_title('Low Score Sample Analysis', fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            output_path2 = os.path.join(output_dir, 'numerical_questions_detailed_analysis.png')
            plt.savefig(output_path2, dpi=viz_config.dpi, bbox_inches='tight', 
                       facecolor=colors.background)
            print(f"  💾 保存: {output_path2}")
            plt.close()
    
    # 3. 生成评估报告
    if config.output.create_detailed_report:
        create_evaluation_report(df, config, output_dir)
    
    print(f"\n🎉 可视化完成！图表保存在: {output_dir}/")


def create_evaluation_report(df, config, output_dir):
    """创建详细评估报告"""
    print("📋 生成评估报告...")
    
    # 基本统计
    total_samples = len(df)
    numerical_count = len(df[df['is_numerical']])
    non_numerical_count = total_samples - numerical_count
    
    # 平均分数
    avg_similarity = df['similarity'].mean()
    
    # 数值型问题统计
    numerical_df = df[df['is_numerical']]
    numerical_stats = ""
    if len(numerical_df) > 0:
        valid_num_df = numerical_df.dropna(subset=['error_percentage'])
        if len(valid_num_df) > 0:
            avg_error = valid_num_df['error_percentage'].mean()
            min_error = valid_num_df['error_percentage'].min()
            max_error = valid_num_df['error_percentage'].max()
            
            # 精度等级统计
            perfect_count = len(valid_num_df[valid_num_df['error_percentage'] == 0])
            excellent_count = len(valid_num_df[valid_num_df['error_percentage'] <= 5])
            good_count = len(valid_num_df[valid_num_df['error_percentage'] <= 20])
            
            numerical_stats = f"""
📊 数值型问题详细分析:
  • 数值型问题数量: {len(numerical_df)} ({len(numerical_df)/total_samples*100:.1f}%)
  • 有效数值对比: {len(valid_num_df)} 个样本
  • 平均数值误差: {avg_error:.2f}%
  • 最小误差: {min_error:.2f}%
  • 最大误差: {max_error:.2f}%
  
  精度等级分布:
  • 完美匹配 (0%误差): {perfect_count} 个 ({perfect_count/len(valid_num_df)*100:.1f}%)
  • 优秀精度 (≤5%误差): {excellent_count} 个 ({excellent_count/len(valid_num_df)*100:.1f}%)
  • 良好精度 (≤20%误差): {good_count} 个 ({good_count/len(valid_num_df)*100:.1f}%)
"""
    
    # 相似度等级统计
    excellent_sim = len(df[df['similarity'] >= 0.9])
    good_sim = len(df[df['similarity'] >= 0.7])
    medium_sim = len(df[df['similarity'] >= 0.5])
    poor_sim = len(df[df['similarity'] < 0.5])
    
    # 模型信息
    model_info = f"模型ID: {config.model.model_id}"
    
    # 生成完整报告
    report = f"""
🔍 单模型评估结果报告
==================================================

📋 评估信息:
  • {model_info}
  • 评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
  • 总测试样本: {total_samples}
  • 随机种子: {config.run.seed}

📊 基本统计:
  • 数值型问题: {numerical_count} ({numerical_count/total_samples*100:.1f}%)
  • 非数值型问题: {non_numerical_count} ({non_numerical_count/total_samples*100:.1f}%)
  • 平均相似度: {avg_similarity:.3f}

📈 相似度等级分布:
  • 优秀 (≥0.9): {excellent_sim} 个 ({excellent_sim/total_samples*100:.1f}%)
  • 良好 (≥0.7): {good_sim} 个 ({good_sim/total_samples*100:.1f}%)
  • 中等 (≥0.5): {medium_sim} 个 ({medium_sim/total_samples*100:.1f}%)
  • 较差 (<0.5): {poor_sim} 个 ({poor_sim/total_samples*100:.1f}%)

📋 评估指标说明:
  1. 文本一致性计算: 基于序列匹配、词汇重叠度等多种方法
  2. 数值误差百分比: 专门针对数值型问题的精确度评估
  3. 综合评分: 根据问题类型动态调整权重

{numerical_stats}

🎯 总结:
  • 模型平均表现: {"优秀" if avg_similarity >= 0.8 else "良好" if avg_similarity >= 0.6 else "中等" if avg_similarity >= 0.4 else "较差"}
  • 整体评估分数: {avg_similarity:.3f}/1.000
  • 高质量回答比例: {good_sim/total_samples*100:.1f}% (相似度≥0.7)

==================================================
配置文件: eval_config.yaml
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 保存报告到reports子目录
    base_dir = config.output.base_output_dir
    reports_dir = os.path.join(base_dir, config.output.reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, '评估报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  💾 评估报告保存至: {report_path}")
    
    # 在控制台显示简要报告
    print("\n" + "="*60)
    print("📊 模型评估结果摘要:")
    print(f"  模型: {config.model.model_id}")
    print(f"  总样本: {total_samples}")
    print(f"  平均相似度: {avg_similarity:.3f}")
    print(f"  平均匹配程度: {avg_similarity*100:.1f}%")
    if numerical_stats:
        valid_num_df = df[df['is_numerical']].dropna(subset=['error_percentage'])
        if len(valid_num_df) > 0:
            avg_error = valid_num_df['error_percentage'].mean()
            print(f"  平均数值误差: {avg_error:.2f}%")
    print("="*60)


@hydra.main(version_base="1.1", config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig):
    """主函数"""
    print("=== 🔧 配置信息 ===")
    print(OmegaConf.to_yaml(cfg))
    
    print("🔍 单模型评估工具")
    print("使用双重评估指标: 1.文本一致性计算 2.数值误差百分比")
    print("="*80)
    
    # ======================================================================
    # 🏃 运行环境设置
    # ======================================================================
    if hasattr(cfg.run, 'cuda_visible_devices') and cfg.run.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.run.cuda_visible_devices)
        print(f"🔧 设置CUDA设备: {cfg.run.cuda_visible_devices}")
    
    # 设置随机种子
    if hasattr(cfg.run, 'seed') and cfg.run.seed is not None:
        set_seed(cfg.run.seed)
        print(f"🎲 设置随机种子: {cfg.run.seed}")
    
    try:
        # 1. 加载模型
        llm, _, model_path = load_model(cfg)
        
        # 2. 加载测试样本
        test_samples = load_test_samples(cfg)
        if not test_samples:
            print("❌ 没有可用的测试样本")
            return
        
        # 3. 执行评估测试
        results = evaluate_model_on_samples(llm, model_path, test_samples, cfg)
        
        # 4. 保存结果到CSV
        df = save_results_to_csv(results, cfg)
        
        # 5. 生成可视化
        create_visualization(df, cfg)
        
        print("\n🎉 模型评估完成!")
        print(f"📄 详细结果: {os.path.join(cfg.output.base_output_dir, cfg.output.csv_filename)}")
        print(f"📊 可视化图表: {os.path.join(cfg.output.base_output_dir, cfg.output.visualization_dir)}/")
        print(f"📁 测试样本: {os.path.join(cfg.output.base_output_dir, cfg.output.samples_dir)}/")
        print(f"📋 评估报告: {os.path.join(cfg.output.base_output_dir, cfg.output.reports_dir)}/")
        print(f"🗂️  总输出目录: {cfg.output.base_output_dir}/")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()