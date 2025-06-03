#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集检查器 - Spatial VLM QA Dataset
下载并分析 https://huggingface.co/datasets/Litian2002/spatialvlm_qa
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
import pandas as pd

def download_and_inspect_dataset():
    """下载并检查 spatialvlm_qa 数据集"""
    
    print("🔄 正在下载 Spatial VLM QA 数据集...")
    print("📍 数据集地址: https://huggingface.co/datasets/Litian2002/spatialvlm_qa")
    
    try:
        # 下载数据集
        dataset = load_dataset("Litian2002/spatialvlm_qa")
        print("✅ 数据集下载成功!")
        
        # 基本信息
        print("\n" + "="*60)
        print("📊 数据集基本信息")
        print("="*60)
        print(f"数据集名称: Spatial VLM QA")
        print(f"总样本数: {len(dataset['train'])}")
        print(f"数据分割: {list(dataset.keys())}")
        
        # 查看字段结构
        train_data = dataset['train']
        print(f"\n📋 数据字段:")
        for key, feature in train_data.features.items():
            print(f"  • {key}: {feature}")
        
        # 随机采样几个例子
        print("\n" + "="*60)
        print("🔍 随机样本检查 (前5个)")
        print("="*60)
        
        for i in range(min(5, len(train_data))):
            sample = train_data[i]
            print(f"\n样本 {i+1}:")
            print(f"  问题: {sample['question']}")
            print(f"  答案: {sample['answer']}")
            print(f"  图像尺寸: {sample['image'].size}")
        
        # 统计分析
        print("\n" + "="*60)
        print("📈 数据集统计分析")
        print("="*60)
        
        # 问题长度分析
        questions = [sample['question'] for sample in train_data.select(range(1000))]  # 取前1000个样本分析
        answers = [sample['answer'] for sample in train_data.select(range(1000))]
        
        q_lengths = [len(q.split()) for q in questions]
        a_lengths = [len(a.split()) for a in answers]
        
        print(f"问题长度统计:")
        print(f"  平均长度: {np.mean(q_lengths):.1f} 词")
        print(f"  最短: {min(q_lengths)} 词")
        print(f"  最长: {max(q_lengths)} 词")
        
        print(f"\n答案长度统计:")
        print(f"  平均长度: {np.mean(a_lengths):.1f} 词")
        print(f"  最短: {min(a_lengths)} 词")
        print(f"  最长: {max(a_lengths)} 词")
        
        # 问题类型分析
        print(f"\n问题类型分析 (前1000个样本):")
        question_types = {}
        for q in questions:
            q_lower = q.lower()
            if any(word in q_lower for word in ['distance', 'how far', 'how distant']):
                question_types['距离测量'] = question_types.get('距离测量', 0) + 1
            elif any(word in q_lower for word in ['front', 'back', 'behind']):
                question_types['前后关系'] = question_types.get('前后关系', 0) + 1
            elif any(word in q_lower for word in ['width', 'height', 'measure']):
                question_types['尺寸测量'] = question_types.get('尺寸测量', 0) + 1
            elif any(word in q_lower for word in ['positioned', 'located']):
                question_types['位置关系'] = question_types.get('位置关系', 0) + 1
            else:
                question_types['其他'] = question_types.get('其他', 0) + 1
        
        for qtype, count in question_types.items():
            print(f"  • {qtype}: {count} ({count/len(questions)*100:.1f}%)")
        
        # 答案类型分析  
        print(f"\n答案类型分析:")
        numeric_answers = 0
        yes_no_answers = 0
        other_answers = 0
        
        for a in answers:
            a_lower = a.lower().strip()
            if a_lower in ['yes', 'no', 'yes.', 'no.']:
                yes_no_answers += 1
            elif any(char.isdigit() for char in a_lower):
                numeric_answers += 1
            else:
                other_answers += 1
        
        print(f"  • 数值型答案: {numeric_answers} ({numeric_answers/len(answers)*100:.1f}%)")
        print(f"  • 是否型答案: {yes_no_answers} ({yes_no_answers/len(answers)*100:.1f}%)")
        print(f"  • 其他答案: {other_answers} ({other_answers/len(answers)*100:.1f}%)")
        
        # 保存几个示例图像
        print("\n" + "="*60)
        print("💾 保存示例图像")
        print("="*60)
        
        os.makedirs("dataset_samples", exist_ok=True)
        
        for i in range(min(3, len(train_data))):
            sample = train_data[i]
            img = sample['image']
            img.save(f"dataset_samples/sample_{i+1}.png")
            
            # 保存问答对
            with open(f"dataset_samples/sample_{i+1}_qa.txt", "w", encoding="utf-8") as f:
                f.write(f"问题: {sample['question']}\n")
                f.write(f"答案: {sample['answer']}\n")
            
            print(f"  保存样本 {i+1}: sample_{i+1}.png + sample_{i+1}_qa.txt")
        
        print(f"\n✅ 示例文件保存到: ./dataset_samples/")
        
        # 创建简单的可视化
        create_visualization(questions, answers, q_lengths, a_lengths)
        
        return dataset
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def create_visualization(questions, answers, q_lengths, a_lengths):
    """创建数据集统计可视化"""
    
    print("\n📊 创建统计图表...")
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 问题长度分布
    ax1.hist(q_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Question Length Distribution')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 答案长度分布
    ax2.hist(a_lengths, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Answer Length Distribution')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 问题关键词统计
    keywords = ['distance', 'front', 'back', 'width', 'height', 'measure', 'far']
    keyword_counts = []
    for keyword in keywords:
        count = sum(1 for q in questions if keyword in q.lower())
        keyword_counts.append(count)
    
    ax3.bar(keywords, keyword_counts, color='lightgreen', alpha=0.7)
    ax3.set_title('Question Keywords Frequency')
    ax3.set_xlabel('Keywords')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # 答案类型饼图
    yes_count = sum(1 for a in answers if a.lower().strip() in ['yes', 'yes.'])
    no_count = sum(1 for a in answers if a.lower().strip() in ['no', 'no.'])
    numeric_count = sum(1 for a in answers if any(c.isdigit() for c in a) and a.lower().strip() not in ['yes', 'no', 'yes.', 'no.'])
    other_count = len(answers) - yes_count - no_count - numeric_count
    
    labels = ['Yes', 'No', 'Numeric', 'Other']
    sizes = [yes_count, no_count, numeric_count, other_count]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Answer Type Distribution')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print("  📊 统计图表保存为: dataset_analysis.png")

def main():
    """主函数"""
    print("🚀 Spatial VLM QA 数据集检查器")
    print("=" * 60)
    
    dataset = download_and_inspect_dataset()
    
    if dataset:
        print("\n🎉 数据集检查完成!")
        print("\n📁 生成的文件:")
        print("  • dataset_samples/ - 示例图像和问答对")
        print("  • dataset_analysis.png - 统计分析图表")
        
        print("\n💡 使用建议:")
        print("  1. 这是一个合成的3D空间推理数据集")
        print("  2. 包含40k个图像-问答对")
        print("  3. 主要用于训练视觉语言模型的空间推理能力")
        print("  4. 问题类型涵盖距离测量、位置关系、尺寸等")
        
    else:
        print("\n❌ 数据集检查失败，请检查网络连接和依赖库")

if __name__ == "__main__":
    main()
