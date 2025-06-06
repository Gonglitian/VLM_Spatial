# generate_spatial_qa_dataset.py
"""
将Blender渲染的图像数据集 + meta.json转换为视觉-语言问答数据集
并上传到 Hugging Face Hub。

现在使用Hydra配置管理系统来管理所有参数！

数据结构假设
-----------
* 文件夹结构
    dataset_root/
        images/          # 1000个PNG文件，命名如 00000.png …
        meta.json        # list[dict] – 每个图像一个字典（示例如下）

```
{
  "image": "00000.png",
  "camera": {"location": [...], "rotation_euler": [...]},
  "light": {...},
  "objects": [
      {
        "name": "cube_0",
        "shape": "cube",
        "location": [...],   # [x, y, z] Blender单位（米）
        "scale": [...],      # 每轴统一缩放（球体半径，立方体半边长等）
        "color": "blue",
        "metallic": true
      }, ...
  ]
}
```

* 每个对象使用**统一**缩放（三个元素相等）。宽度解释为该标量的两倍（XY平面直径）。
* 距离以位置相同的单位返回（默认：米）。
* 答案使用毫米精度四舍五入到**两位小数**（可通过配置文件修改）。
* 脚本随机生成**QA对**每张图片（可通过配置文件配置）。
* 最终HuggingFace数据集有三列：
    - `image`   : PIL.Image
    - `question`: str
    - `answer`  : str

使用方法
-----
```bash
pip install datasets pillow huggingface_hub tqdm hydra-core omegaconf

export HF_TOKEN="<your‑access‑token>"   # 或运行 `huggingface‑cli login`

python scripts/data_process.py
```
"""

import json
import math
import os
import random
import logging
from itertools import combinations
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi
from PIL import Image as PILImage
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# 常量和模板池
# ---------------------------

OBJ_A = "[A]"
OBJ_B = "[B]"
DIST = "[X]"

# 模板字典：{"类别": {"q": [列表], "a": [列表]}}
TEMPLATES = {
    "distance": {
        "q": [
            "What is the distance between [A] and [B]?",
            "How far apart are [A] and [B]?",
            "How distant is [A] from [B]?",
            "How far is [A] from [B]?",
            "How close is [A] from [B]?",
            "Could you measure the distance between [A] and [B]?",
            "Can you tell me the distance of [A] from [B]?",
            "How far away is [A] from [B]?",
            "Can you provide the distance measurement between [A] and [B]?",
            "Can you give me an estimation of the distance between [A] and [B]?",
            "Could you provide the distance between [A] and [B]?",
            "How much distance is there between [A] and [B]?",
            "Tell me the distance between [A] and [B].",
            "Give me the distance from [A] to [B].",
            "Measure the distance from [A] to [B].",
            "Measure the distance between [A] and [B].",
        ],
        "a": [
            "[X]",
            "[A] and [B] are [X] apart.",
            "[A] is [X] away from [B].",
            "A distance of [X] exists between [A] and [B].",
            "[A] is [X] from [B].",
            "[A] and [B] are [X] apart from each other.",
            "They are [X] apart.",
            "The distance of [A] from [B] is [X].",
        ],
    },
    "vertical_distance": {
        "q": [
            "What is the vertical distance between [A] and [B]?",
            "How far apart are [A] and [B] vertically?",
            "How distant is [A] from [B] vertically?",
            "How far is [A] from [B] vertically?",
            "Could you measure the vertical distance between [A] and [B]?",
            "Can you tell me the vertical distance between [A] and [B]?",
            "How far away is [A] from [B] vertically?",
            "Can you provide the measurement of the vertical distance between [A] and [B]?",
            "Estimate the vertical distance between [A] and [B].",
            "Could you provide the vertical distance between [A] and [B]?",
            "How much distance is there between [A] and [B] vertically?",
            "Tell me the distance between [A] and [B] vertically.",
            "Give me the vertical distance from [A] to [B].",
            "Measure the vertical distance from [A] to [B].",
            "Measure the distance between [A] and [B] vertically.",
        ],
        "a": [
            "[X]",
            "[A] and [B] are [X] apart vertically.",
            "[A] is [X] away from [B] vertically.",
            "A vertical distance of [X] exists between [A] and [B].",
            "[A] is [X] from [B] vertically.",
            "[A] and [B] are [X] apart vertically from each other.",
            "Vertically, They are [X] apart.",
            "The vertical distance of [A] from [B] is [X].",
            "They are [X] apart.",
            "It's approximately [X].",
        ],
    },
    "horizontal_distance": {
        "q": [
            "What is the horizontal distance between [A] and [B]?",
            "How far apart are [A] and [B] horizontally?",
            "How distant is [A] from [B] horizontally?",
            "How far is [A] from [B] horizontally?",
            "Could you measure the horizontal distance between [A] and [B]?",
            "Can you tell me the horizontal distance of [A] from [B]?",
            "How far away is [A] from [B] horizontally?",
            "Can you provide the measurement of the horizontal distance between [A] and [B]?",
            "Can you give me an estimation of the horizontal distance between [A] and [B]?",
            "Could you provide the horizontal distance between [A] and [B]?",
            "How much distance is there between [A] and [B] horizontally?",
            "Tell me the distance between [A] and [B] horizontally.",
            "Give me the horizontal distance from [A] to [B].",
            "Measure the horizontal distance from [A] to [B].",
            "Measure the distance between [A] and [B] horizontally.",
        ],
        "a": [
            "[X]",
            "[A] and [B] are [X] apart horizontally.",
            "[A] is [X] away from [B] horizontally.",
            "A horizontal distance of [X] exists between [A] and [B].",
            "[A] is [X] from [B] horizontally.",
            "[A] and [B] are [X] apart horizontally from each other.",
            "Horizontally, They are [X] apart.",
            "The horizontal distance of [A] from [B] is [X].",
            "They are [X] apart.",
            "It's approximately [X].",
        ],
    },
    "width": {
        "q": [
            "Measure the width of [A].",
            "Determine the horizontal dimensions of [A].",
            "Find out how wide [A] is.",
            "What is the width of [A]?",
            "How wide is [A]?",
            "What are the dimensions of [A] in terms of width?",
            "Could you tell me the horizontal size of [A]?",
            "What is the approximate width of [A]?",
            "How much space does [A] occupy horizontally?",
            "How big is [A]?",
            "How big is [A] in terms of width?",
            "What's the radius of [A]?",
        ],
        "a": [
            "[X]",
            "The width of [A] is [X].",
            "[A] is [X] wide.",
            "[A] is [X] in width.",
            "It's [X].",
        ],
    },
    "behind": {
        "q": [
            "Is [A] behind [B]?",
            "Is the position of [A] more distant than that of [B]?",
            "Does [A] lie behind [B]?",
            "Is [A] positioned behind [B]?",
            "Is [A] further to camera compared to [B]?",
            "Does [A] come behind [B]?",
            "Is [A] positioned at the back of [B]?",
            "Is [A] further to the viewer compared to [B]?",
        ],
        "true": [
            "Yes.",
            "Yes, it is.",
            "Yes, it's behind [B].",
            "That's True.",
            "Yes, [A] is further from the viewer.",
            "Yes, [A] is behind [B].",
        ],
        "false": [
            "No.",
            "No, it is not.",
            "No, it's in front of [B].",
            "That's False.",
            "No, [A] is closer to the viewer.",
            "No, [B] is in front of [A].",
        ],
    },
    "front": {
        "q": [
            "Is [A] in front of [B]?",
            "Is the position of [A] less distant than that of [B]?",
            "Does [A] lie in front of [B]?",
            "Is [A] positioned in front of [B]?",
            "Is [A] closer to camera compared to [B]?",
            "Does [A] come in front of [B]?",
            "Is [A] positioned before [B]?",
            "Is [A] closer to the viewer compared to [B]?",
        ],
        "true": [
            "Yes.",
            "Yes, it is.",
            "Yes, it's in front of [B].",
            "That's True.",
            "Yes, [A] is closer to the viewer.",
            "Yes, [A] is in front of [B].",
        ],
        "false": [
            "No.",
            "No, it is not.",
            "No, it's behind [B].",
            "That's False.",
            "No, [A] is further to the viewer.",
            "No, [B] is behind [A].",
        ],
    },
}

CATEGORIES = list(TEMPLATES.keys())  # 辅助变量

# ---------------------------
# 辅助函数
# ---------------------------

def euc_dist(a, b):
    """3D欧几里得距离"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def horiz_dist(a, b):
    """水平距离（XY平面）"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def vert_dist(a, b):
    """垂直距离（Z轴）"""
    return abs(a[2] - b[2])

def width_from_scale(obj):
    """从缩放计算宽度：假设统一缩放且直径 = 2 * scale[0]"""
    return 2 * obj["scale"][0]

def cam_distance(cam_loc, obj_loc):
    """相机到对象的距离"""
    return euc_dist(cam_loc, obj_loc)

def get_object_description(obj, all_objects, config):
    """
    生成对象的描述性名称，格式为 'the [color] [shape]'
    如果有重复的颜色+形状组合，会添加编号（根据配置决定）
    """
    color = obj["color"]
    shape = obj["shape"]
    base_desc = f"the {color} {shape}"
    
    # 根据配置决定是否添加材质信息
    if config.generation.object_description.include_metallic and obj.get("metallic", False):
        base_desc = f"the metallic {color} {shape}"
    
    # 根据配置决定是否在重复时添加编号
    if not config.generation.object_description.use_numbering:
        return base_desc
    
    # 检查是否有重复的颜色+形状组合
    same_color_shape = [o for o in all_objects 
                       if o["color"] == color and o["shape"] == shape]
    
    if len(same_color_shape) == 1:
        return base_desc
    else:
        # 如果有多个相同颜色和形状的对象，添加编号
        index = same_color_shape.index(obj) + 1
        return f"{base_desc} ({index})"

# ---------------------------
# 每张图片的QA生成
# ---------------------------

def make_qa_for_image(record, config, rng=None):
    """为单张图片生成QA对"""
    rng = rng or random
    objs = record["objects"]
    cam_loc = record["camera"]["location"]
    
    num_qas = config.data.num_qas
    dist_precision = config.data.dist_precision

    # 检查场景质量
    if config.quality.min_objects_per_scene and len(objs) < config.quality.min_objects_per_scene:
        logger.warning(f"场景对象数量 ({len(objs)}) 少于最小要求 ({config.quality.min_objects_per_scene})")
        return []
    
    # 检查QA数量限制
    if config.quality.max_qa_per_image and num_qas > config.quality.max_qa_per_image:
        logger.warning(f"QA数量 ({num_qas}) 超过最大限制 ({config.quality.max_qa_per_image})")
        num_qas = config.quality.max_qa_per_image

    # 预计算成对指标
    pairs = list(combinations(range(len(objs)), 2))
    rng.shuffle(pairs)
    qa_items = []

    # 根据权重创建类别选择池
    category_weights = config.generation.category_weights
    weighted_categories = []
    for cat, weight in category_weights.items():
        weighted_categories.extend([cat] * int(weight * 10))  # 权重转换为选择概率
    
    if not weighted_categories:
        weighted_categories = list(CATEGORIES)  # 回退到均匀分布

    # 混合类别直到我们有num_qas个示例
    while len(qa_items) < num_qas:
        cat = rng.choice(weighted_categories)
        if cat == "width":
            idx = rng.randrange(len(objs))
            obj = objs[idx]
            obj_desc = get_object_description(obj, objs, config)
            x = round(width_from_scale(obj), dist_precision)
            q = rng.choice(TEMPLATES[cat]["q"]).replace(OBJ_A, obj_desc)
            a = rng.choice(TEMPLATES[cat]["a"]).replace(OBJ_A, obj_desc).replace(DIST, f"{x}")
            qa_items.append({"question": q, "answer": a})
        elif cat in ("distance", "horizontal_distance", "vertical_distance"):
            if not pairs:
                pairs = list(combinations(range(len(objs)), 2))
                rng.shuffle(pairs)
            i, j = pairs.pop()
            obj_a, obj_b = objs[i], objs[j]
            obj_a_desc = get_object_description(obj_a, objs, config)
            obj_b_desc = get_object_description(obj_b, objs, config)
            dist_func = {
                "distance": euc_dist,
                "horizontal_distance": horiz_dist,
                "vertical_distance": vert_dist,
            }[cat]
            x = round(dist_func(obj_a["location"], obj_b["location"]), dist_precision)
            q_tmpl = rng.choice(TEMPLATES[cat]["q"])
            a_tmpl = rng.choice(TEMPLATES[cat]["a"])
            q = q_tmpl.replace(OBJ_A, obj_a_desc).replace(OBJ_B, obj_b_desc)
            a = (
                a_tmpl.replace(OBJ_A, obj_a_desc).replace(OBJ_B, obj_b_desc).replace(DIST, f"{x}")
            )
            qa_items.append({"question": q, "answer": a})
        else:  # front / behind 谓词
            if not pairs:
                pairs = list(combinations(range(len(objs)), 2))
                rng.shuffle(pairs)
            i, j = pairs.pop()
            obj_a, obj_b = objs[i], objs[j]
            obj_a_desc = get_object_description(obj_a, objs, config)
            obj_b_desc = get_object_description(obj_b, objs, config)
            # 计算真值：front = A比B更近，behind = A比B更远
            dist_a = cam_distance(cam_loc, obj_a["location"])
            dist_b = cam_distance(cam_loc, obj_b["location"])
            if cat == "front":
                truth = dist_a < dist_b
            else:  # behind
                truth = dist_a > dist_b
            answer_pool = TEMPLATES[cat]["true"] if truth else TEMPLATES[cat]["false"]
            q = rng.choice(TEMPLATES[cat]["q"]).replace(OBJ_A, obj_a_desc).replace(OBJ_B, obj_b_desc)
            a = rng.choice(answer_pool).replace(OBJ_A, obj_a_desc).replace(OBJ_B, obj_b_desc)
            qa_items.append({"question": q, "answer": a})

    return qa_items

# ---------------------------
# 数据集构建函数
# ---------------------------

def build_dataset(config):
    """构建QA数据集"""
    logger.info("🚀 开始构建数据集...")
    
    # 获取项目根目录路径，确保数据路径正确
    project_root = Path(__file__).parent.parent  # scripts 的上一级目录
    data_dir = project_root / config.data.data_dir
    
    meta_path = data_dir / "metadata.json"
    images_dir = data_dir / "images"
    
    logger.info(f"📂 检查数据目录: {data_dir}")
    assert meta_path.exists(), f"❌ 未找到元数据文件: {meta_path}"
    assert images_dir.exists(), f"❌ 未找到图像目录: {images_dir}"
    logger.info("✅ 数据目录检查完成")

    # 加载元数据
    logger.info("📋 加载元数据文件...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_records = json.load(f)
    logger.info(f"✅ 加载了 {len(meta_records)} 条元数据记录")

    # 可选的元数据验证
    if config.quality.validate_metadata:
        logger.info("🔍 验证元数据完整性...")
        valid_records = []
        for i, rec in enumerate(meta_records):
            if "objects" not in rec or len(rec["objects"]) < config.quality.min_objects_per_scene:
                logger.warning(f"跳过记录 {i}: 对象数量不足")
                continue
            valid_records.append(rec)
        meta_records = valid_records
        logger.info(f"✅ 元数据验证完成，保留 {len(meta_records)} 条有效记录")

    # 初始化随机数生成器
    rng = random.Random(config.run.random_seed)
    logger.info(f"🎲 初始化随机数生成器 (seed={config.run.random_seed})")

    # 构建行数据
    logger.info("🔄 开始生成QA对...")
    rows = []
    
    # 根据配置选择是否显示进度条
    iterator = tqdm(meta_records, desc="生成QA对", unit="图片", disable=not config.output.enable_progress_bar)
    
    for rec in iterator:
        qa_pairs = make_qa_for_image(rec, config, rng=rng)
        
        if not qa_pairs:  # 如果质量检查失败，跳过这个记录
            continue
            
        img_path = images_dir / rec["image"]
        
        if config.quality.check_image_exists and not img_path.exists():
            logger.warning(f"⚠️  警告: 图片文件不存在: {img_path}")
            continue
            
        img = PILImage.open(img_path)
        for qa in qa_pairs:
            rows.append({"image": img, **qa})

    logger.info(f"✅ 从 {len(meta_records)} 张图片生成了 {len(rows)} 个QA对")

    # 创建HuggingFace数据集
    logger.info("🏗️  构建HuggingFace数据集格式...")
    features = Features({"image": Image(), "question": Value("string"), "answer": Value("string")})
    ds = Dataset.from_dict({"image": [r["image"] for r in rows], 
                           "question": [r["question"] for r in rows],
                           "answer": [r["answer"] for r in rows]}, 
                          features=features)
    
    logger.info("✅ 数据集构建完成!")
    logger.info(f"📊 数据集统计:")
    logger.info(f"   - 总样本数: {len(ds)}")
    logger.info(f"   - 列: {list(ds.features.keys())}")
    
    # 可选的本地保存
    if config.output.save_local_copy:
        local_path = project_root / config.output.local_save_path
        logger.info(f"💾 保存本地副本到: {local_path}")
        ds.save_to_disk(str(local_path))
    
    return ds

# ---------------------------
# 数据集上传函数
# ---------------------------

def push_dataset(dataset, config):
    """上传数据集到HuggingFace Hub"""
    logger.info("🚀 开始上传数据集到HuggingFace Hub...")
    
    repo_id = config.hub.repo_id
    private = config.hub.private_repo
    dataset_info = config.hub.dataset_info
    
    # 数据集说明卡片
    dataset_card = f"""---
license: {dataset_info.license}
task_categories:
{chr(10).join(f'- {cat}' for cat in dataset_info.task_categories)}
language:
{chr(10).join(f'- {lang}' for lang in dataset_info.language)}
tags:
{chr(10).join(f'- {tag}' for tag in dataset_info.tags)}
pretty_name: {dataset_info.pretty_name}
size_categories: {dataset_info.size_categories}
---

# {dataset_info.pretty_name}

自动从Blender场景数据集生成。每个示例包含一张图片和一个问答对，用于探测**度量**（数值）和**关系**（真/假）空间推理技能。

* **图像**: 使用Blender渲染（场景，每个{config.quality.min_objects_per_scene}+个随机基元，随机相机和光照）。
* **元数据**: 对象名称、位置、缩放、颜色、材质标志。
* **问题**: 每张图片{config.data.num_qas}个，从手工制作的模板中抽取，涵盖：
    * 欧几里得/水平/垂直距离查询
    * 对象宽度查询
    * 相对于相机的前/后谓词

距离以Blender场景单位表示，四舍五入到{config.data.dist_precision}位小数精度。

## 字段
| 字段    | 类型    | 描述                                 |
|---------|---------|-------------------------------------|
| image   | image   | 渲染的场景                           |
| question| string  | 自然语言查询                         |
| answer  | string  | 真实答案                             |

## 配置参数
本数据集使用以下配置生成：
- 每张图片QA对数: {config.data.num_qas}
- 距离精度: {config.data.dist_precision}位小数
- 最少对象数: {config.quality.min_objects_per_scene}
- 随机种子: {config.run.random_seed}

## 引用
```
@misc{{synthetic_spatialvlm_qa,
  title  = {{{dataset_info.pretty_name}}},
  author = {{Generated with Hydra configuration}},
  year   = 2025,
  url    = {{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""

    # 推送数据集
    logger.info("📤 推送数据集...")
    dataset.push_to_hub(repo_id, private=private, commit_message="使用Hydra配置系统生成的合成空间问答数据集")
    logger.info("✅ 数据集推送完成")

    logger.info("📝 上传README文件...")
    HfApi().upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="README.md",
        path_or_fileobj=dataset_card.encode(),
        commit_message="添加数据集说明卡片",
    )
    logger.info("✅ README文件上传完成")
    logger.info(f"🎉 全部完成! 数据集地址: https://huggingface.co/datasets/{repo_id}")

# ---------------------------
# 主执行程序
# ---------------------------

@hydra.main(version_base="1.1", config_path="../configs", config_name="data_process")
def main(cfg: DictConfig):
    """主函数 - 使用Hydra配置管理"""
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, cfg.output.log_level))
    
    logger.info("=" * 60)
    logger.info("🎯 空间视觉语言问答数据集生成器 (Hydra配置版)")
    logger.info("=" * 60)
    
    # 显示配置信息
    logger.info("=== 🔧 配置信息 ===")
    logger.info(OmegaConf.to_yaml(cfg))
    
    logger.info(f"📁 数据目录: {cfg.data.data_dir}")
    logger.info(f"🏷️  仓库ID: {cfg.hub.repo_id}")
    logger.info(f"📊 每图QA数: {cfg.data.num_qas}")
    logger.info(f"🔒 私有仓库: {cfg.hub.private_repo}")
    logger.info(f"🔧 仅构建模式: {cfg.hub.build_only}")
    logger.info(f"🎲 随机种子: {cfg.run.random_seed}")
    logger.info("=" * 60)

    try:
        # 构建数据集
        dataset = build_dataset(cfg)

        if not cfg.hub.build_only:
            # 上传数据集
            push_dataset(dataset, cfg)
        else:
            logger.info("⚠️  仅构建模式已启用，跳过上传步骤")
            logger.info("✅ 数据集构建完成，可以通过设置hub.build_only=false来启用上传")
            
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
