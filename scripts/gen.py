import os
# cuda device os set to 0 and 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from isaacsim import SimulationApp


sim = SimulationApp({'headless': True})


from omni.replicator.isaac import Randomizer, Writer

# ----- 创建无界面仿真 -----

# ----- 随机器 -----
rand = Randomizer()
rand.add_shape_randomizer(           # Sphere / Cube / Cone / Cylinder
    shapes=['Sphere', 'Cube', 'Cone', 'Cylinder'],
    scale=(0.05, 0.3),              # 米
    color='random_hsv',             # 任意 RGB
)
rand.add_pose_randomizer(x=(-1, 1), y=(-1, 1), z=(0.1, 1.0),
                         yaw=(0, 360))
rand.add_light_randomizer(intensity=(200, 1200), color='white')

# 相机
cam = rand.create_random_camera(
    look_at='origin',
    azimuth=(-60, 60),
    elevation=(10, 70),
    fov=(35, 90),
    resolution=(512, 512)
)

# ----- 写入器 -----
writer = Writer(output_dir='out/', funcs=['rgb'])
writer.attach([cam])

# ----- 主循环 -----
for step in range(10_000):
    if step % 30 == 0:       # 每隔 30 帧全部重新随机
        rand.sample_all()
    sim.step()
    if step % 30 == 0:
        writer.write()       # 保存 PNG 至 out/
sim.close()
