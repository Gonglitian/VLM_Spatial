from datasets import Dataset, Features, Value, Image
from PIL import Image as PILImage
import datasets, os
from huggingface_hub import HfApi

# 1. 构造单条样本
img = PILImage.open("image.png")
data = {
    "image": [img],                # 列名须与代码保持一致
    "text" : ["x^2 + y^2 = z^2"]
}

# 2. 指定 feature 类型，确保 image 列被识别
features = Features({"image": Image(), "text": Value("string")})
ds = Dataset.from_dict(data, features=features)

# 3. 创建 Dataset Card 内容
dataset_card = """---
license: mit
task_categories:
- image-to-text
- visual-question-answering
language:
- en
tags:
- latex
- ocr
- mathematics
- vision
pretty_name: LaTeX OCR Tiny Dataset
size_categories:
- n<1K
---

# LaTeX OCR Tiny Dataset

## 数据集描述

这是一个用于LaTeX公式光学字符识别(OCR)的小型测试数据集。数据集包含图像和对应的LaTeX公式文本。

## 数据集结构

### 数据字段

- `image`: 包含数学公式的图像
- `text`: 对应的LaTeX格式文本

### 数据实例

```json
{
  "image": <PIL.Image>,
  "text": "x^2 + y^2 = z^2"
}
```

## 用途

这个数据集主要用于：
- 测试和验证LaTeX OCR模型
- 数学公式识别研究
- 视觉语言模型的微调

## 数据来源

手动创建的测试数据集。

## 许可证

MIT License

## 引用

如果您使用此数据集，请引用：

```bibtex
@dataset{latex_ocr_tiny_2024,
  title={LaTeX OCR Tiny Dataset},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/Litian2002/latex_ocr_tiny}
}
```
"""

# 4. 推送数据集到 Hub
print("正在上传数据集...")
ds.push_to_hub("Litian2002/latex_ocr_tiny", private=True,
               commit_message="init one-sample test set")

# 5. 上传 Dataset Card
print("正在上传 Dataset Card...")
api = HfApi()
api.upload_file(
    path_or_fileobj=dataset_card.encode('utf-8'),
    path_in_repo="README.md",
    repo_id="Litian2002/latex_ocr_tiny",
    repo_type="dataset",
    commit_message="Add dataset card"
)

print("✅ 数据集和Dataset Card上传完成！")
print("🔗 访问链接: https://huggingface.co/datasets/Litian2002/latex_ocr_tiny") 