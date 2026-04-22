"""
Keyframe capture helper environment for manual joint posing.

Purpose:
1) Start the same excavator + particle-pool scene.
2) Default paused.
3) Disable excavator gravity and heavily suppress inertia so manual joint edits are stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sapien
from sapien.utils import Viewer

import excavator_pool_env as env

# ===== 可调参数（keyframe 录制专用） =====
# 关节驱动参数：越大越“跟手”，但也更容易数值僵硬，建议逐步调。
CAPTURE_DRIVE_STIFFNESS = 3000.0
CAPTURE_DRIVE_DAMPING = 600.0
CAPTURE_DRIVE_FORCE_LIMIT = 20000.0

# 连杆阻尼：用于快速吸收残余速度，减少惯性拖尾。
CAPTURE_LINK_LINEAR_DAMPING = 8.0
CAPTURE_LINK_ANGULAR_DAMPING = 8.0

# 去穿透速度上限：防止接触时出现过强回弹。
CAPTURE_MAX_DEPENETRATION_VEL = 0.20

# 键盘单次增量（弧度），按住会连续变化。
KEYBOARD_DELTA_RAD = 0.01


def _apply_capture_dynamics(robot: sapien.physx.PhysxArticulation) -> None:
    """Apply low-inertia settings for manual keyframe capture."""
    # 1) 关闭挖掘机重力
    for link in robot.links:
        link.set_disable_gravity(True)
        link.set_linear_damping(CAPTURE_LINK_LINEAR_DAMPING)
        link.set_angular_damping(CAPTURE_LINK_ANGULAR_DAMPING)
        link.set_max_depenetration_velocity(CAPTURE_MAX_DEPENETRATION_VEL)
        link.wake_up()

    # 2) 配置高刚度高阻尼驱动，保证手动调节后快速贴近目标
    env.configure_joint_drives(
        robot,
        stiffness=CAPTURE_DRIVE_STIFFNESS,
        damping=CAPTURE_DRIVE_DAMPING,
        force_limit=CAPTURE_DRIVE_FORCE_LIMIT,
    )

    # 3) 初始时把当前角度作为目标，避免启动瞬间抖动
    curr_q = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
    for joint, q in zip(robot.active_joints, curr_q, strict=False):
        joint.set_drive_target(float(q))


def _run_capture_viewer(scene: sapien.Scene, robot: sapien.physx.PhysxArticulation) -> None:
    """Run viewer with paused-start and no-physics manual joint editing."""
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=env.CAMERA_XYZ[0], y=env.CAMERA_XYZ[1], z=env.CAMERA_XYZ[2])
    viewer.set_camera_rpy(r=env.CAMERA_RPY[0], p=env.CAMERA_RPY[1], y=env.CAMERA_RPY[2])
    viewer.window.set_camera_parameters(near=0.01, far=30.0, fovy=np.deg2rad(58.0))

    # 需求：无论何种模式都默认暂停
    viewer.paused = True

    physx_system = scene.physx_system
    is_gpu_backend = isinstance(physx_system, sapien.physx.PhysxGpuSystem)

    if is_gpu_backend:
        try:
            physx_system.sync_poses_gpu_to_cpu()
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] Initial GPU pose sync failed: {exc}")

    print("[Info] Keyframe capture env started. paused=True")
    print("[Info] 本脚本不执行 scene.step()，挖掘机只会因手动调节/键盘输入而改变姿态。")
    print("[Info] 键盘: 1/2->J1-/+, 3/4->J2-/+, 5/6->J3-/+, 7/8->J4-/+")

    # 保持零速度/零力，避免任何残余动力学量影响显示状态。
    zero_q = np.zeros(robot.dof, dtype=np.float32)
    qlimits = np.asarray(robot.get_qlimits(), dtype=np.float32).reshape(-1, 2)

    def apply_joint_delta_by_keyboard() -> None:
        if robot.dof <= 0:
            return
        q = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
        changed = False
        key_pairs = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]
        max_dof = min(robot.dof, len(key_pairs))
        for j in range(max_dof):
            neg_key, pos_key = key_pairs[j]
            if viewer.window.key_down(neg_key):
                q[j] -= KEYBOARD_DELTA_RAD
                changed = True
            if viewer.window.key_down(pos_key):
                q[j] += KEYBOARD_DELTA_RAD
                changed = True

        if not changed:
            return

        q = np.clip(q, qlimits[:, 0], qlimits[:, 1]).astype(np.float32)
        robot.set_qpos(q)
        robot.set_qvel(zero_q)
        robot.set_qf(zero_q)
        for link in robot.links:
            link.wake_up()

    while not viewer.closed:
        apply_joint_delta_by_keyboard()
        robot.set_qvel(zero_q)
        robot.set_qf(zero_q)
        if is_gpu_backend:
            physx_system.sync_poses_gpu_to_cpu()

        scene.update_render()
        viewer.render()


def main() -> None:
    bootstrap_config_path = env.extract_config_path_from_argv(sys.argv[1:], env.DEFAULT_CONFIG_PATH)
    bootstrap_config = env.load_pose_config(bootstrap_config_path)
    model_choices = env.get_available_excavator_models_from_config(bootstrap_config)

    if "excavator_s010" in model_choices:
        default_model = "excavator_s010"
    elif "excavator_simple" in model_choices:
        default_model = "excavator_simple"
    else:
        default_model = model_choices[0]

    parser = argparse.ArgumentParser(description="Keyframe capture env: gravity-off + low-inertia manual joint posing.")
    parser.add_argument(
        "equipment_model",
        nargs="?",
        default=default_model,
        choices=model_choices,
        help="Excavator model preset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(bootstrap_config_path),
        help="Path to config.json that stores urdf candidates and per-model, per-scene initial poses.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU PhysX (default is CPU for stable manual keyframe capture).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    pose_config = env.load_pose_config(config_path)

    scene = env.create_scene(timestep=env.SIM_TIMESTEP, prefer_gpu=args.gpu)
    env.configure_lighting(scene, ground_half_size=env.GROUND_HALF_SIZE)

    env.build_platform(scene, center=env.PLATFORM_CENTER, half_size=env.PLATFORM_HALF_SIZE)
    env.build_particle_pool(
        scene,
        center=env.POOL_CENTER,
        inner_half_size=env.POOL_INNER_HALF_SIZE,
        wall_height=env.POOL_WALL_HEIGHT,
        wall_thickness=env.POOL_WALL_THICKNESS,
        bottom_thickness=env.POOL_BOTTOM_THICKNESS,
    )
    env.spawn_particles(
        scene,
        center=env.POOL_CENTER,
        inner_half_size=env.POOL_INNER_HALF_SIZE,
        particle_count=env.PARTICLE_COUNT,
        radius=env.PARTICLE_RADIUS,
        render_particles=env.PARTICLE_RENDER_ENABLED,
    )

    urdf_path = env.resolve_excavator_urdf_path(pose_config, args.equipment_model, user_path=None)
    init_qpos = env.get_initial_qpos_from_config(pose_config, args.equipment_model, env.SCENE_NAME)

    robot = env.load_excavator(
        scene=scene,
        equipment_model=args.equipment_model,
        urdf_path=urdf_path,
        platform_center=env.PLATFORM_CENTER,
        platform_half_height=env.PLATFORM_HALF_SIZE[2],
        init_qpos=init_qpos,
    )

    _apply_capture_dynamics(robot)
    env.maybe_init_gpu_physx(scene)
    _run_capture_viewer(scene, robot)


if __name__ == "__main__":
    main()
