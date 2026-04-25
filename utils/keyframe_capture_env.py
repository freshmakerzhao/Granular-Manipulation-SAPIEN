"""
Keyframe capture helper environment for manual joint posing.

Purpose:
1) Start the same excavator + particle-pool scene.
2) Default paused.
3) Disable excavator gravity and heavily suppress inertia so manual joint edits are stable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sapien
from sapien.utils import Viewer
try:
    from transforms3d.euler import quat2euler
except Exception:  # noqa: BLE001
    quat2euler = None

from envs import excavator_pool as env

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

# 每次按 9 记录关键帧时，t 的固定步长（用于后续 scripted 回放）
KEYFRAME_T_INTERVAL = 40

# 开启碰撞录制时，每次循环推进的物理子步数（提高接触稳定性）
CAPTURE_COLLISION_SUBSTEPS = 2
# 开启碰撞录制且 viewer 暂停时，检测到按键后额外推进子步数（确保肉眼可见位移）
CAPTURE_COLLISION_KEYPRESS_SUBSTEPS = 12


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


def _run_capture_viewer(
    scene: sapien.Scene,
    robot: sapien.physx.PhysxArticulation,
    record_file: Path,
    keyframe_t_interval: int,
    equipment_model: str,
    collision_mode: str,
    bucket_collision_debug_visuals: list[tuple[sapien.Entity, object, sapien.Pose]] | None = None,
) -> None:
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
    print("[Info] 键盘控制:")
    print("       F/H -> Swing 旋转 (J1-/+) | T/G -> Stick 伸缩 (J3-/+)")
    print("       I/K -> Boom 升降 (J2-/+) | J/L -> Bucket 开合 (J4-/+)")
    print("       9   -> 记录当前末端位姿为一个 keyframe")
    print("       0   -> 打印当前相机 CAMERA_XYZ / CAMERA_RPY（便于拷贝到代码）")
    print("[Info] 若按键无响应，请先鼠标左键点击一次 3D 视窗以获取键盘焦点。")
    print(f"[Info] 录制文件: {record_file}")
    collision_enabled = collision_mode == "on"
    if collision_enabled:
        print("[Info] 碰撞录制模式: ON（使用驱动目标 + 物理解算，按键移动不会直接穿模）。")
        print("[Info] 默认 paused=True。可按空格切换暂停；暂停时按键会触发多步解算。")
    else:
        print("[Info] 碰撞录制模式: OFF（直接写 qpos，不做物理解算，适合快速摆姿态）。")

    # 保持零速度/零力，避免任何残余动力学量影响显示状态。
    zero_q = np.zeros(robot.dof, dtype=np.float32)
    qlimits = np.asarray(robot.get_qlimits(), dtype=np.float32).reshape(-1, 2)
    unsupported_key_names: set[str] = set()
    prev_record_key_down = False
    prev_dump_camera_key_down = False
    drive_target_q = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1).copy()
    if collision_enabled:
        for joint, tgt in zip(robot.active_joints, drive_target_q, strict=False):
            joint.set_drive_target(float(tgt))

    ee_link_index, ee_link_name = env.resolve_ee_link(robot, preferred_name=None)
    active_joint_names = [joint.name for joint in robot.active_joints]

    keyframe_payload: dict = {
        "meta": {
            "equipment_model": equipment_model,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "ee_link_name": ee_link_name,
            "joint_names": active_joint_names,
            "keyboard_delta_rad": float(KEYBOARD_DELTA_RAD),
            "keyframe_t_interval": int(keyframe_t_interval),
            "collision_mode": collision_mode,
            "notes": "Generated by utils/keyframe_capture_env.py",
        },
        "keyframes": [],
    }

    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.write_text(json.dumps(keyframe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Info] 已清空并初始化 keyframe 文件。")

    def key_down_safe(key_name: str) -> bool:
        try:
            return bool(viewer.window.key_down(key_name))
        except RuntimeError:
            if key_name not in unsupported_key_names:
                unsupported_key_names.add(key_name)
                print(f"[Warn] Viewer 不支持键名 '{key_name}'，已自动忽略该按键。")
            return False

    def get_current_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        ee_pose = robot.links[ee_link_index].entity_pose
        xyz = np.asarray(ee_pose.p, dtype=np.float32).reshape(3)
        rpy = np.asarray(ee_pose.rpy, dtype=np.float32).reshape(3)
        return xyz, rpy

    def record_current_keyframe() -> None:
        q = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
        xyz, rpy = get_current_ee_pose()
        keyframes = keyframe_payload["keyframes"]
        kf_id = len(keyframes)
        t = int(kf_id * max(1, int(keyframe_t_interval)))
        keyframe_item = {
            "id": int(kf_id),
            "t": int(t),
            "xyz": xyz.tolist(),
            "rpy": rpy.tolist(),
            "qpos": q.tolist(),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        keyframes.append(keyframe_item)
        record_file.write_text(json.dumps(keyframe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[Info] 记录 keyframe #{kf_id}: t={t}, xyz={np.round(xyz, 4).tolist()}, "
            f"rpy={np.round(rpy, 4).tolist()}"
        )

    def print_current_camera_params() -> None:
        try:
            cam_p = np.asarray(viewer.window.get_camera_position(), dtype=np.float32).reshape(3)
            cam_q = np.asarray(viewer.window.get_camera_rotation(), dtype=np.float32).reshape(4)
            cam_pose = sapien.Pose(p=cam_p.tolist(), q=cam_q.tolist())
        except Exception:  # noqa: BLE001
            try:
                cam_pose = viewer.window.get_camera_pose()
                cam_p = np.asarray(cam_pose.p, dtype=np.float32).reshape(3)
            except Exception as exc:  # noqa: BLE001
                print(f"[Warn] 无法读取当前相机参数: {exc}")
                return

        def _wrap_to_pi(v: float) -> float:
            return float((v + np.pi) % (2.0 * np.pi) - np.pi)

        if quat2euler is not None:
            # Match SAPIEN viewer control convention exactly:
            # control_window.py uses quat2euler(q) then setRPY(r, -p, -y).
            r_raw, p_raw, y_raw = quat2euler(np.asarray(cam_pose.q, dtype=np.float64).reshape(4))
            cam_rpy = np.asarray([r_raw, -p_raw, -y_raw], dtype=np.float32).reshape(3)
        else:
            # Fallback when transforms3d is unavailable.
            pose_rpy = np.asarray(cam_pose.rpy, dtype=np.float32).reshape(3)
            cam_rpy = np.asarray([pose_rpy[0], -pose_rpy[1], -pose_rpy[2]], dtype=np.float32).reshape(3)
            print("[Warn] transforms3d 不可用，RPY 由 Pose.rpy 反推，可能存在欧拉角多解。")

        # Canonicalize for direct backfill to CAMERA_RPY.
        cam_rpy[0] = _wrap_to_pi(float(cam_rpy[0]))
        cam_rpy[1] = float(np.clip(cam_rpy[1], -1.57, 1.57))  # viewer controller pitch clamp
        cam_rpy[2] = _wrap_to_pi(float(cam_rpy[2]))
        p = np.round(cam_p, 5).tolist()
        r = np.round(cam_rpy, 5).tolist()
        q = np.round(np.asarray(cam_pose.q, dtype=np.float32).reshape(4), 6).tolist()
        print(
            "[Info] 当前相机参数（可直接回填）:\n"
            f"CAMERA_XYZ = ({p[0]}, {p[1]}, {p[2]})\n"
            f"CAMERA_RPY = ({r[0]}, {r[1]}, {r[2]})\n"
            f"# Raw camera quaternion (for debugging): CAMERA_Q = ({q[0]}, {q[1]}, {q[2]}, {q[3]})"
        )

    def apply_joint_delta_by_keyboard() -> bool:
        if robot.dof <= 0:
            return False
        if collision_enabled:
            q = drive_target_q.copy()
        else:
            q = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
        changed = False

        # 指定键位映射到 excavator 关节顺序 [j1_swing, j2_boom, j3_stick, j4_bucket]
        if robot.dof >= 1:
            # F/H: Swing (J1)
            if key_down_safe("f"):
                q[0] -= KEYBOARD_DELTA_RAD
                changed = True
            if key_down_safe("h"):
                q[0] += KEYBOARD_DELTA_RAD
                changed = True
        if robot.dof >= 3:
            # T/G: Stick (J3) [方向已反转]
            if key_down_safe("t"):
                q[2] -= KEYBOARD_DELTA_RAD
                changed = True
            if key_down_safe("g"):
                q[2] += KEYBOARD_DELTA_RAD
                changed = True
        if robot.dof >= 2:
            # I/K: Boom (J2) [方向已反转]
            if key_down_safe("i"):
                q[1] -= KEYBOARD_DELTA_RAD
                changed = True
            if key_down_safe("k"):
                q[1] += KEYBOARD_DELTA_RAD
                changed = True
        if robot.dof >= 4:
            # J/L: Bucket (J4)
            if key_down_safe("j"):
                q[3] -= KEYBOARD_DELTA_RAD
                changed = True
            if key_down_safe("l"):
                q[3] += KEYBOARD_DELTA_RAD
                changed = True

        if not changed:
            return False

        q = np.clip(q, qlimits[:, 0], qlimits[:, 1]).astype(np.float32)
        if collision_enabled:
            drive_target_q[:] = q
            for joint, tgt in zip(robot.active_joints, drive_target_q, strict=False):
                joint.set_drive_target(float(tgt))
        else:
            robot.set_qpos(q)
            robot.set_qvel(zero_q)
            robot.set_qf(zero_q)
            for link in robot.links:
                link.wake_up()
        return True

    while not viewer.closed:
        if bucket_collision_debug_visuals:
            env.update_bucket_collision_debug_visuals(bucket_collision_debug_visuals)
        key_changed = apply_joint_delta_by_keyboard()
        record_key_down = key_down_safe("9")
        if record_key_down and (not prev_record_key_down):
            record_current_keyframe()
        prev_record_key_down = record_key_down
        dump_camera_key_down = key_down_safe("0")
        if dump_camera_key_down and (not prev_dump_camera_key_down):
            print_current_camera_params()
        prev_dump_camera_key_down = dump_camera_key_down

        if collision_enabled:
            # 未暂停时持续解算；暂停时若检测到按键，执行更多子步让位移可见。
            if viewer.paused:
                step_count = max(1, int(CAPTURE_COLLISION_KEYPRESS_SUBSTEPS)) if key_changed else 0
            else:
                step_count = max(1, int(CAPTURE_COLLISION_SUBSTEPS))
            if step_count > 0:
                for _ in range(step_count):
                    scene.step()
                    if is_gpu_backend:
                        physx_system.sync_poses_gpu_to_cpu()
        else:
            robot.set_qvel(zero_q)
            robot.set_qf(zero_q)
            if is_gpu_backend:
                physx_system.sync_poses_gpu_to_cpu()

        scene.update_render()
        viewer.render()


def _set_entity_collision_enabled(entity: sapien.Entity, enabled: bool) -> int:
    """Enable/disable collisions for one entity by setting all shape groups."""
    if enabled:
        return 0

    component: sapien.physx.PhysxRigidBaseComponent | None = entity.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if component is None:
        component = entity.find_component_by_type(sapien.physx.PhysxArticulationLinkComponent)
    if component is None:
        return 0

    changed = 0
    for shape in component.collision_shapes:
        shape.set_collision_groups([0, 0, 0, 0])
        changed += 1
    return changed


def _set_robot_and_particles_collision(
    robot: sapien.physx.PhysxArticulation,
    particles: list[sapien.Entity],
    enabled: bool,
) -> None:
    """Unified collision switch for robot + particles."""
    if enabled:
        print("[Info] Collision mode: ON (robot + particles).")
        return

    changed_shapes = 0
    for link in robot.links:
        changed_shapes += _set_entity_collision_enabled(link.entity, enabled=False)
    for particle in particles:
        changed_shapes += _set_entity_collision_enabled(particle, enabled=False)
    print(f"[Info] Collision mode: OFF (robot + particles), disabled shapes={changed_shapes}.")


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
    parser.add_argument(
        "--record-file",
        type=str,
        required=True,
        help="输出 keyframe 文件路径（每次启动会清空重写）。",
    )
    parser.add_argument(
        "--keyframe-t-interval",
        type=int,
        default=KEYFRAME_T_INTERVAL,
        help="每次按 9 记录关键帧时，t 的固定步长。",
    )
    parser.add_argument(
        "--collision",
        type=str,
        choices=("on", "off"),
        default="on",
        help="统一设置粒子与挖掘机碰撞：on=开启，off=关闭。",
    )
    parser.add_argument(
        "--show-bucket-collision-boxes",
        action="store_true",
        help="显示铲斗 box 碰撞箱（半透明），用于手动调试碰撞箱大小和位置。",
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
        base_color=env.SOURCE_POOL_BASE_COLOR,
        name="source_particle_pool",
    )
    env.build_particle_pool(
        scene,
        center=env.RECEIVER_POOL_CENTER,
        inner_half_size=env.RECEIVER_POOL_INNER_HALF_SIZE,
        wall_height=env.RECEIVER_POOL_WALL_HEIGHT,
        wall_thickness=env.POOL_WALL_THICKNESS,
        bottom_thickness=env.POOL_BOTTOM_THICKNESS,
        base_color=env.RECEIVER_POOL_BASE_COLOR,
        name="receiver_particle_pool",
    )
    particles = env.spawn_particles(
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
    env.configure_bucket_particle_only_collision(robot, particles)
    bucket_collision_debug_visuals = (
        env.create_bucket_collision_debug_visuals(scene, robot)
        if args.show_bucket_collision_boxes
        else None
    )

    _set_robot_and_particles_collision(
        robot=robot,
        particles=particles,
        enabled=(args.collision == "on"),
    )
    _apply_capture_dynamics(robot)
    env.maybe_init_gpu_physx(scene)
    _run_capture_viewer(
        scene=scene,
        robot=robot,
        record_file=Path(args.record_file).expanduser().resolve(),
        keyframe_t_interval=max(1, int(args.keyframe_t_interval)),
        equipment_model=args.equipment_model,
        collision_mode=args.collision,
        bucket_collision_debug_visuals=bucket_collision_debug_visuals,
    )


if __name__ == "__main__":
    main()
