#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_geo_dataset.py – 基础几何体合成数据生成
随机化: 形状·尺度·颜色·相机·光照
"""

import argparse
from isaacsim import SimulationApp

# ---------- 1. 启动无头 Isaac Sim ----------
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=500, help="总帧数")
parser.add_argument("--out", type=str, default="out", help="图片输出目录")
parser.add_argument("--res", nargs=2, type=int, default=[640, 480], help="分辨率")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
from omni.isaac.lab.sim import SimulationContext

sim = SimulationContext()            # 便于统一 dt & Physics
sim.set_simulation_dt(1 / 60.0)

# ---------- 2. 定义随机化函数 ----------
SHAPES = ["Cube", "Sphere", "Cylinder", "Cone"]

def random_shape():
    """在世界原点上方随机实例化一个几何体并随机属性"""
    prim = rep.randomizer.choice(
        rep.create.primitive,
        [dict(shape=s) for s in SHAPES]
    ).node
    with prim:
        # 尺寸 0.05–0.3 m，均匀缩放
        rep.modify.scale(rep.distribution.uniform(0.05, 0.3))
        # 纯 RGB 颜色随机
        rep.modify.color(rep.distribution.color(color_space="rgb"))
        # 随机平移到 xy ∈ [-0.3,0.3], z=shape_size/2
        rep.modify.pose(
            position=rep.distribution.uniform(
                (-0.3, -0.3, 0.15), (0.3, 0.3, 0.6)
            )
        )
    return prim

# ---------- 3. 相机与光照 ----------
camera = rep.create.camera(focal_length=12.0)
rep.create.render_product(camera, resolution=tuple(args.res))

with camera:
    # 相机在半径 1 m 的球面上均匀采样，始终朝向世界原点
    rep.modify.pose(
        position=rep.distribution.uniform((-1, -1, 0.5), (1, 1, 1.5)),
        look_at=(0, 0, 0)
    )

key_light = rep.create.light(light_type="sphere", temperature=5500, intensity=700)
with key_light:
    rep.modify.pose(
        position=rep.distribution.uniform((-2, -2, 2), (2, 2, 4))
    )

# ---------- 4. 输出设置 ----------
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=args.out,
    rgb=True,
    render_products=rep.get.render_products()
)

# ---------- 5. 每帧随机化并抓取 ----------
@rep.trigger.on_frame(num_frames=args.frames)
def on_frame():
    # 清空旧物体
    rep.modify.delete("/World/geo*")
    # 新建 4–8 个随机几何体
    for _ in range(rep.distribution.randint(4, 8).get_value()):
        random_shape()

# ---------- 6. 运行 ----------
rep.orchestrator.run()
app.close()
