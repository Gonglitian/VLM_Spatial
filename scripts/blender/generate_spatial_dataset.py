#!/usr/bin/env python3
"""
Generate a synthetic 3‑D dataset of simple geometric primitives (cube, sphere, cylinder, cone)
for training vision‑language models to reason about spatial relations.

### Usage (inside Blender ≥ 3.0)
Run this script directly in Blender's scripting interface or via:
```
blender --background --python generate_spatial_dataset.py
```
The script will render RGB images and save rich JSON metadata
for every object, camera, and light.

Now using Hydra configuration management for flexible parameter control.
"""

# 处理blender与hydra的冲突
import argparse, sys
parser = argparse.ArgumentParser(add_help=False)
# --background and -b   
parser.add_argument("--background", "-b", action="store_true")
# --python and -p
parser.add_argument("--python")          # 接脚本路径
# 解析并忽略未知
_, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers     # 剩下的交给 Hydra

import bpy
import random
import math
import os
import json
import sys
from mathutils import Vector

# Add the parent directory to sys.path to import hydra
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 获取项目根目录路径（脚本所在目录的上两级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    print("⚠️ Hydra not available, falling back to default parameters")
    HYDRA_AVAILABLE = False

# 尝试导入 tqdm 用于进度条显示
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️ tqdm not available, progress will be shown as simple text")
    TQDM_AVAILABLE = False

# --------------------------------------------------
# Helper utilities
# --------------------------------------------------

def clear_scene():
    """Delete every object and purge orphaned data blocks (keeps memory low)."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Purge orphans (repeat because Blender removes them in waves)
    for _ in range(4):
        bpy.ops.outliner.orphans_purge(do_recursive=True)


def create_material(name, color, metallic=False, config=None):
    """Build a simple physically‑based material with optional metallic look."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = 0.9 if metallic else 0.0
    
    if config and hasattr(config, 'materials'):
        roughness = config.materials.metallic_roughness if metallic else config.materials.non_metallic_roughness
    else:
        roughness = 0.25 if metallic else 0.55
    
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def random_camera_position(config):
    """Sample a camera position on a horizontal circle around the scene."""
    radius = config.camera.radius if config else 8.0
    height_range = config.camera.height_range if config else [3.0, 6.0]
    
    theta = random.uniform(0, 2 * math.pi)
    h = random.uniform(*height_range)
    return Vector((radius * math.cos(theta), radius * math.sin(theta), h))


def look_at(obj, target):
    """Rotate *obj* so its local ‑Z axis points at *target* (Blender camera forward)."""
    direction = Vector(target) - obj.location
    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()


def add_light(target, config):
    """Add a SUN light whose direction points toward *target*."""
    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    
    if config and hasattr(config, 'lighting'):
        energy_range = config.lighting.energy_range
        radius = config.lighting.radius
        height_range = config.lighting.height_range
    else:
        energy_range = [2.0, 6.0]
        radius = 10.0
        height_range = [5.0, 9.0]
    
    light_data.energy = random.uniform(*energy_range)
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector((
        radius * math.cos(random.uniform(0, 2 * math.pi)),
        radius * math.sin(random.uniform(0, 2 * math.pi)),
        random.uniform(*height_range)
    ))
    look_at(light_obj, target)
    return light_obj


def add_ground(config):
    """Add ground plane with configurable size and color."""
    if config and hasattr(config, 'materials'):
        size = config.materials.ground_size
        color = config.materials.ground_color
    else:
        size = 20
        color = [0.8, 0.8, 0.8]
    
    bpy.ops.mesh.primitive_plane_add(size=size)
    ground = bpy.context.object
    ground_mat = create_material("Ground", color, False, config)
    ground.data.materials.append(ground_mat)
    return ground

# --------------------------------------------------
# Random object factory
# --------------------------------------------------

def get_palette(config):
    """Get color palette from config or use defaults."""
    if config and hasattr(config, 'objects') and hasattr(config.objects, 'colors'):
        return [(color.name, color.rgb) for color in config.objects.colors]
    else:
        return [
            ("red",     [1.0, 0.0, 0.0]),
            ("green",   [0.0, 0.8, 0.0]),
            ("blue",    [0.0, 0.2, 1.0]),
            ("yellow",  [1.0, 1.0, 0.0]),
            ("purple",  [0.6, 0.2, 0.8]),
            ("orange",  [1.0, 0.55, 0.0]),
        ]


def get_shapes(config):
    """Get shapes list from config or use defaults."""
    if config and hasattr(config, 'objects') and hasattr(config.objects, 'shapes'):
        return config.objects.shapes
    else:
        return ["cube", "sphere", "cylinder", "cone"]


class ObjectPlacer:
    """Manages object placement to avoid overlaps."""
    
    def __init__(self, config=None):
        self.placed_xy = []
        self.config = config
    
    def clear(self):
        """Clear placed positions for new scene."""
        self.placed_xy.clear()
    
    def add_random_object(self, idx):
        """Add a random object with collision avoidance."""
        shapes = get_shapes(self.config)
        palette = get_palette(self.config)
        
        shape = random.choice(shapes)
        color_name, color_rgb = random.choice(palette)
        
        # Material properties
        if self.config and hasattr(self.config, 'materials'):
            metallic = random.random() < self.config.materials.metallic_probability
            size_range = self.config.objects.size_range
            placement_area = self.config.objects.placement_area
            min_distance = self.config.objects.min_distance
            max_attempts = self.config.objects.max_placement_attempts
        else:
            metallic = random.choice([True, False])
            size_range = [0.4, 1.2]
            placement_area = [-3.0, 3.0]
            min_distance = 1.2
            max_attempts = 100
        
        size = random.uniform(*size_range)

        # Sample a non‑overlapping 2‑D position
        for _ in range(max_attempts):
            x = random.uniform(*placement_area)
            y = random.uniform(*placement_area)
            if all((x - ox) ** 2 + (y - oy) ** 2 > min_distance ** 2 for ox, oy in self.placed_xy):
                self.placed_xy.append((x, y))
                break
        else:
            x, y = 0.0, 0.0  # Fallback if things get crowded

        # Create the primitive at the origin, then scale & lift
        if shape == "cube":
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0))
        elif shape == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(x, y, 0))
        elif shape == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=1, location=(x, y, 0))
        elif shape == "cone":
            bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1, location=(x, y, 0))

        obj = bpy.context.object
        obj.name = f"{shape}_{idx}"
        obj.scale = (size, size, size)

        # Let Blender update dimensions before we lift the object
        bpy.context.view_layer.update()
        obj.location.z = obj.dimensions.z / 2  # Sit perfectly on the ground

        # Assign PBR material
        mat = create_material(f"{color_name}_{idx}", color_rgb, metallic, self.config)
        obj.data.materials.append(mat)

        return {
            "name": obj.name,
            "shape": shape,
            "location": list(obj.location),
            "scale": list(obj.scale),
            "color": color_name,
            "metallic": metallic,
        }

# --------------------------------------------------
# Scene generator
# --------------------------------------------------

def generate_scene(scene_idx, config):
    """Generate a single scene with all objects, camera, and lighting."""
    clear_scene()
    
    # Set random seed for this scene
    seed = config.run.random_seed + scene_idx if config else scene_idx
    random.seed(seed)

    scene = bpy.context.scene
    
    # Configure output settings
    if config and hasattr(config, 'output'):
        scene.render.image_settings.file_format = config.output.image_format
    else:
        scene.render.image_settings.file_format = "PNG"
    
    if config and hasattr(config, 'dataset'):
        resolution = config.dataset.resolution
    else:
        resolution = [640, 480]
    
    scene.render.resolution_x, scene.render.resolution_y = resolution

    # Add ground
    add_ground(config)

    # Objects
    placer = ObjectPlacer(config)
    objects_per_scene = config.dataset.objects_per_scene if config else 5
    objects_meta = [placer.add_random_object(i) for i in range(objects_per_scene)]

    # Camera
    cam_loc = random_camera_position(config)
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    look_at(cam, Vector((0, 0, 0.8)))
    scene.camera = cam

    # Light
    sun = add_light(Vector((0, 0, 0)), config)

    # Environment brightness randomisation
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    
    if config and hasattr(config, 'lighting'):
        brightness_range = config.lighting.world_brightness_range
    else:
        brightness_range = [0.8, 1.6]
    
    bg.inputs[1].default_value = random.uniform(*brightness_range)

    # Filenames & render
    output_dir = config.dataset.output_dir if config else "blender_data/data"
    # 确保输出路径相对于项目根目录
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    
    padding = config.output.filename_padding if config else 5
    images_subdir = config.output.images_subdir if config else "images"
    
    img_name = f"{scene_idx:0{padding}d}.png"
    img_path = os.path.join(output_dir, images_subdir, img_name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    scene.render.filepath = img_path

    bpy.ops.render.render(write_still=True)

    return {
        "image": img_name,
        "camera": {
            "location": list(cam.location),
            "rotation_euler": list(cam.rotation_euler),
        },
        "light": {
            "location": list(sun.location),
            "energy": sun.data.energy,
        },
        "objects": objects_meta,
    }

# --------------------------------------------------
# Main execution functions
# --------------------------------------------------

def generate_dataset(config):
    """Generate the complete dataset."""
    print("=== 🔧 配置信息 ===")
    if HYDRA_AVAILABLE and config:
        print(OmegaConf.to_yaml(config))
    else:
        print("使用默认配置参数")
    
    print("🎬 开始生成Blender数据集...")
    
    # Get parameters from config or use defaults
    num_scenes = config.dataset.num_scenes if config else 100
    output_dir = config.dataset.output_dir if config else "blender_data/data"
    # 确保输出路径相对于项目根目录
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    
    metadata_filename = config.output.metadata_filename if config else "metadata.json"
    images_subdir = config.output.images_subdir if config else "images"

    # Ensure folder structure exists
    os.makedirs(os.path.join(output_dir, images_subdir), exist_ok=True)

    all_meta = []
    
    # 创建进度条或简单文本显示
    if TQDM_AVAILABLE:
        scene_iterator = tqdm(
            range(num_scenes),
            desc="🎬 生成场景",
            unit="scene",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        scene_iterator = range(num_scenes)
        print(f"🎬 开始生成 {num_scenes} 个场景...")
    
    for idx in scene_iterator:
        meta = generate_scene(idx, config)
        all_meta.append(meta)
        
        if TQDM_AVAILABLE:
            # 更新 tqdm 描述以显示当前图像信息
            scene_iterator.set_postfix_str(f"→ {meta['image']}")
        else:
            # 简单文本进度显示
            print(f"[Scene {idx + 1}/{num_scenes}] → {meta['image']}")

    # Write single JSON with metadata for the whole dataset
    meta_path = os.path.join(output_dir, metadata_filename)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(all_meta, fp, indent=2)
    print(f"✅ 数据集生成完成 → {meta_path}")


def main_with_hydra(cfg: DictConfig):
    """Main function when using Hydra."""
    generate_dataset(cfg)


def main_fallback():
    """Fallback main function when Hydra is not available."""
    print("⚠️ 在Blender环境中运行，使用默认参数")
    generate_dataset(None)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if HYDRA_AVAILABLE:
    # Set up Hydra decorator
    @hydra.main(version_base="1.1", config_path="../../configs", config_name="blender_dataset")
    def main(cfg: DictConfig):
        main_with_hydra(cfg)
    
    if __name__ == "__main__":
        main()
else:
    # Direct execution in Blender
    if __name__ == "__main__":
        main_fallback()
