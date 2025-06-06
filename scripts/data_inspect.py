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
        
        print("⏳ 正在分析全部样本，请稍候...")
        
        # 分析全部样本
        questions = [sample['question'] for sample in train_data]
        answers = [sample['answer'] for sample in train_data]
        
        print(f"✅ 成功加载 {len(questions)} 个问答对")
        
        q_lengths = [len(q.split()) for q in questions]
        a_lengths = [len(a.split()) for a in answers]
        
        print(f"\n问题长度统计:")
        print(f"  平均长度: {np.mean(q_lengths):.1f} 词")
        print(f"  最短: {min(q_lengths)} 词")
        print(f"  最长: {max(q_lengths)} 词")
        print(f"  中位数: {np.median(q_lengths):.1f} 词")
        
        print(f"\n答案长度统计:")
        print(f"  平均长度: {np.mean(a_lengths):.1f} 词")
        print(f"  最短: {min(a_lengths)} 词")
        print(f"  最长: {max(a_lengths)} 词")
        print(f"  中位数: {np.median(a_lengths):.1f} 词")
        
        # 问题类型分析
        print(f"\n问题类型分析 (全部 {len(questions)} 个样本):")
        print("⏳ 正在分析问题类型...")
        
        question_types = {}
        for i, q in enumerate(questions):
            if i % 5000 == 0:
                print(f"  处理进度: {i}/{len(questions)} ({i/len(questions)*100:.1f}%)")
            
            q_lower = q.lower()
            if any(word in q_lower for word in ['distance', 'how far', 'how distant']):
                question_types['距离测量'] = question_types.get('距离测量', 0) + 1
            elif any(word in q_lower for word in ['front', 'back', 'behind']):
                question_types['前后关系'] = question_types.get('前后关系', 0) + 1
            elif any(word in q_lower for word in ['width', 'height', 'measure', 'size', 'big', 'large', 'small']):
                question_types['尺寸测量'] = question_types.get('尺寸测量', 0) + 1
            elif any(word in q_lower for word in ['positioned', 'located', 'where']):
                question_types['位置关系'] = question_types.get('位置关系', 0) + 1
            elif any(word in q_lower for word in ['color', 'colour']):
                question_types['颜色识别'] = question_types.get('颜色识别', 0) + 1
            elif any(word in q_lower for word in ['count', 'how many', 'number of']):
                question_types['数量统计'] = question_types.get('数量统计', 0) + 1
            else:
                question_types['其他'] = question_types.get('其他', 0) + 1
        
        print(f"\n✅ 问题类型分析完成:")
        for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {qtype}: {count:,} ({count/len(questions)*100:.1f}%)")
        
        # 答案类型分析  
        print(f"\n答案类型分析:")
        print("⏳ 正在分析答案类型...")
        
        numeric_answers = 0
        yes_no_answers = 0
        color_answers = 0
        other_answers = 0
        
        for i, a in enumerate(answers):
            if i % 5000 == 0:
                print(f"  处理进度: {i}/{len(answers)} ({i/len(answers)*100:.1f}%)")
            
            a_lower = a.lower().strip()
            if a_lower in ['yes', 'no', 'yes.', 'no.']:
                yes_no_answers += 1
            elif any(char.isdigit() for char in a_lower):
                numeric_answers += 1
            elif any(color in a_lower for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple']):
                color_answers += 1
            else:
                other_answers += 1
        
        print(f"\n✅ 答案类型分析完成:")
        print(f"  • 数值型答案: {numeric_answers:,} ({numeric_answers/len(answers)*100:.1f}%)")
        print(f"  • 是否型答案: {yes_no_answers:,} ({yes_no_answers/len(answers)*100:.1f}%)")
        print(f"  • 颜色型答案: {color_answers:,} ({color_answers/len(answers)*100:.1f}%)")
        print(f"  • 其他答案: {other_answers:,} ({other_answers/len(answers)*100:.1f}%)")
        
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
        create_visualization(questions, answers, q_lengths, a_lengths, question_types)
        
        return dataset
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def create_visualization(questions, answers, q_lengths, a_lengths, question_types):
    """创建数据集统计可视化"""
    
    print("\n📊 创建统计图表...")
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(16, 12))
    
    # 问题长度分布
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(q_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'Question Length Distribution\n(Total: {len(questions):,} samples)')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 答案长度分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(a_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title(f'Answer Length Distribution\n(Total: {len(answers):,} samples)')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 问题关键词统计
    ax3 = plt.subplot(2, 3, 3)
    keywords = ['distance', 'front', 'back', 'width', 'height', 'measure', 'far', 'color', 'count']
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
    ax4 = plt.subplot(2, 3, 4)
    yes_count = sum(1 for a in answers if a.lower().strip() in ['yes', 'yes.'])
    no_count = sum(1 for a in answers if a.lower().strip() in ['no', 'no.'])
    numeric_count = sum(1 for a in answers if any(c.isdigit() for c in a) and a.lower().strip() not in ['yes', 'no', 'yes.', 'no.'])
    color_count = sum(1 for a in answers if any(color in a.lower() for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple']) and not any(c.isdigit() for c in a))
    other_count = len(answers) - yes_count - no_count - numeric_count - color_count
    
    labels = ['Yes', 'No', 'Numeric', 'Color', 'Other']
    sizes = [yes_count, no_count, numeric_count, color_count, other_count]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Answer Type Distribution')
    
    # 问题类型分布
    ax5 = plt.subplot(2, 3, 5)
    qtypes = list(question_types.keys())
    qcounts = list(question_types.values())
    
    ax5.bar(qtypes, qcounts, color='lightsteelblue', alpha=0.7)
    ax5.set_title('Question Type Distribution')
    ax5.set_xlabel('Question Types')
    ax5.set_ylabel('Count')
    ax5.tick_params(axis='x', rotation=45)
    
    # 长度分布对比
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(q_lengths, a_lengths, alpha=0.3, s=1)
    ax6.set_xlabel('Question Length (words)')
    ax6.set_ylabel('Answer Length (words)')
    ax6.set_title('Question vs Answer Length')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("  📊 统计图表保存为: dataset_analysis.png")
    
    # 保存详细统计到文件
    with open('dataset_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("Spatial VLM QA Dataset - 完整统计分析\n")
        f.write("="*60 + "\n\n")
        f.write(f"总样本数: {len(questions):,}\n\n")
        
        f.write("问题长度统计:\n")
        f.write(f"  平均长度: {np.mean(q_lengths):.2f} 词\n")
        f.write(f"  中位数: {np.median(q_lengths):.2f} 词\n")
        f.write(f"  标准差: {np.std(q_lengths):.2f}\n")
        f.write(f"  最短: {min(q_lengths)} 词\n")
        f.write(f"  最长: {max(q_lengths)} 词\n\n")
        
        f.write("答案长度统计:\n")
        f.write(f"  平均长度: {np.mean(a_lengths):.2f} 词\n")
        f.write(f"  中位数: {np.median(a_lengths):.2f} 词\n")
        f.write(f"  标准差: {np.std(a_lengths):.2f}\n")
        f.write(f"  最短: {min(a_lengths)} 词\n")
        f.write(f"  最长: {max(a_lengths)} 词\n\n")
        
        f.write("问题类型分布:\n")
        for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {qtype}: {count:,} ({count/len(questions)*100:.2f}%)\n")
        
        f.write(f"\n答案类型分布:\n")
        f.write(f"  数值型: {numeric_count:,} ({numeric_count/len(answers)*100:.2f}%)\n")
        f.write(f"  是否型: {yes_count + no_count:,} ({(yes_count + no_count)/len(answers)*100:.2f}%)\n")
        f.write(f"  颜色型: {color_count:,} ({color_count/len(answers)*100:.2f}%)\n")
        f.write(f"  其他: {other_count:,} ({other_count/len(answers)*100:.2f}%)\n")
    
    print("  📄 详细统计保存为: dataset_statistics.txt")

def main():
    """主函数"""
    print("🚀 Spatial VLM QA 数据集检查器 (全量分析)")
    print("=" * 60)
    
    dataset = download_and_inspect_dataset()
    
    if dataset:
        print("\n🎉 数据集检查完成!")
        print("\n📁 生成的文件:")
        print("  • dataset_samples/ - 示例图像和问答对")
        print("  • dataset_analysis.png - 统计分析图表")
        print("  • dataset_statistics.txt - 详细统计数据")
        
        print("\n💡 使用建议:")
        print("  1. 这是一个合成的3D空间推理数据集")
        print("  2. 包含约40k个图像-问答对")
        print("  3. 主要用于训练视觉语言模型的空间推理能力")
        print("  4. 问题类型涵盖距离测量、位置关系、尺寸等")
        print("  5. 已完成全量样本分析，结果更准确全面")
        
    else:
        print("\n❌ 数据集检查失败，请检查网络连接和依赖库")

if __name__ == "__main__":
    main()
