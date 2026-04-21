import sapien.core as sapien
from sapien.utils.viewer import Viewer

def main():
    # 1. 场景创建 (SAPIEN 3.0 极简 API：不需要手动创建 Engine 和 Renderer 了)
    # 它会自动在后台调用你的 RTX 5070 Ti 和 Vulkan
    scene = sapien.Scene()
    scene.set_timestep(1 / 240.0) 
    
    # 2. 添加环境光和网格地面
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    scene.add_ground(altitude=0)
    
    # 3. 创建一个动态方块 Actor
    builder = scene.create_actor_builder()
    
    # 添加碰撞体积
    builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    
    # SAPIEN 3.0 添加视觉模型（移除旧版的 color 参数，使用默认材质）
    builder.add_box_visual(half_size=[0.5, 0.5, 0.5]) 
    
    box = builder.build(name="falling_box")
    # 设置初始位置：位于半空 z=2 的位置
    box.set_pose(sapien.Pose([0, 0, 2]))
    
    # 4. 可视化界面 (SAPIEN 3.0 的 Viewer 也不需要传 renderer 了)
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-3, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.1, y=0)
    
    # 5. 仿真主循环
    print("SAPIEN 3.0 环境启动成功！按 Esc 退出可视化窗口...")
    while not viewer.closed:
        scene.step()      # 推进物理世界
        viewer.render()   # 渲染当前帧

if __name__ == '__main__':
    main()