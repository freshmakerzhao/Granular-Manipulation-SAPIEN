"""
Excavator + granular pool environment in SAPIEN 3.0.

Layout:
1) Right side: an elevated platform for a fixed excavator.
2) Left side: a square pool (bottom + 4 walls) filled with small dynamic particles.
3) Ground plane + lighting + interactive viewer.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sapien
from sapien.utils import Viewer
from scripted_policy import (
    LinearEEKeyframePolicy,
    build_default_excavator_ee_keyframes,
    load_ee_keyframes_json,
)


SCENE_NAME = "excavator_pool_env"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.json"

# ============================================= 可调参数 ============================================= 

# ============================================= 地面 ============================================= 
GROUND_HALF_SIZE = (3.2, 2.4)  # 地面半尺寸 (x, y)
# ============================================= 地面 ============================================= 

# ============================================= 机器人平台 ============================================= 
PLATFORM_CENTER = (0.35, 0.0)  # 机器人平台中心 (x, y)
PLATFORM_HALF_SIZE = (0.29, 0.23, 0.06)  # 机器人平台半尺寸 (x, y, z)
# ============================================= 机器人平台 ============================================= 

# ============================================= 料池 ============================================= 
POOL_CENTER = (-0.35, 0.0)  # 颗粒池中心 (x, y)
POOL_INNER_HALF_SIZE = (0.28, 0.28)  # 颗粒池内部半尺寸 (x, y)
POOL_WALL_HEIGHT = 0.14  # 颗粒池墙高 (m)
POOL_WALL_THICKNESS = 0.017  # 颗粒池墙厚 (m)
POOL_BOTTOM_THICKNESS = 0.020  # 颗粒池底板半厚度 (m)
# ============================================= 料池 ============================================= 

# ============================================= 接料池 ============================================= 
RECEIVER_POOL_CENTER = (0.4, 0.60)  # 接料池中心
RECEIVER_POOL_INNER_HALF_SIZE = (0.16, 0.14)  # 接料池内部半尺寸 (x, y)
RECEIVER_POOL_WALL_HEIGHT = 0.14  # 接料池墙高 (m)
# ============================================= 接料池 ============================================= 

# ============================================= 相机 ============================================= 
CAMERA_XYZ = (0.0, -2.2, 1.08)  # Viewer 相机位置 (x, y, z)
CAMERA_RPY = (0.0, -0.24, 0.0)  # Viewer 相机姿态 (roll, pitch, yaw)
# ============================================= 相机 ============================================= 

SIM_TIMESTEP = 1 / 240.0  # 物理仿真步长 (s)


# ============================================= 颗粒 ============================================= 
PARTICLE_COUNT = 4000  # 颗粒总数
PARTICLE_RADIUS = 0.008  # 颗粒半径 (m)
PARTICLE_RENDER_ENABLED = True  # 是否渲染颗粒（仅影响显示，不影响物理）

SAND_STATIC_FRICTION = 1.75  # 颗粒静摩擦系数（越大越不易滑动）
SAND_DYNAMIC_FRICTION = 1.45  # 颗粒动摩擦系数
SAND_RESTITUTION = 0.0  # 颗粒弹性系数（0=不回弹）
SAND_DENSITY = 1700.0  # 颗粒密度 (kg/m^3)
SAND_LINEAR_DAMPING = 0.24  # 颗粒线速度阻尼（抑制平动抖动）
SAND_ANGULAR_DAMPING = 0.30  # 颗粒角速度阻尼（抑制旋转抖动）
SAND_SOLVER_POS_ITERS = 12  # 颗粒接触求解位置迭代次数
SAND_SOLVER_VEL_ITERS = 4  # 颗粒接触求解速度迭代次数
SAND_MAX_DEPENETRATION_VEL = 0.70  # 最大去穿透速度限制
SAND_MAX_LINEAR_VEL = 1.20  # 最大线速度限制
SAND_MAX_ANGULAR_VEL = 20.0  # 最大角速度限制
SAND_MAX_CONTACT_IMPULSE = 0.015  # 单次接触冲量上限（降低“弹飞”）
# ============================================= 颗粒 ============================================= 


# ============================================= 机械臂 ============================================= 
TOOL_STATIC_FRICTION = 1.60  # 机械臂/铲斗与环境静摩擦系数
TOOL_DYNAMIC_FRICTION = 1.30  # 机械臂/铲斗与环境动摩擦系数
TOOL_RESTITUTION = 0.0  # 机械臂/铲斗回弹系数
TOOL_SOLVER_POS_ITERS = 24  # 机械臂接触求解位置迭代次数
TOOL_SOLVER_VEL_ITERS = 8  # 机械臂接触求解速度迭代次数
TOOL_MAX_DEPENETRATION_VEL = 0.80  # 机械臂去穿透速度限制

JOINT_DRIVE_STIFFNESS = 900.0  # 关节驱动刚度（CPU 驱动模式）
JOINT_DRIVE_DAMPING = 120.0  # 关节驱动阻尼（CPU 驱动模式）
JOINT_DRIVE_FORCE_LIMIT = 4_000.0  # 关节驱动力上限（CPU 驱动模式）
# ============================================= 机械臂 ============================================= 


# ============================================= 计算 ============================================= 
GPU_MAX_RIGID_CONTACT_COUNT = 1_200_000  # GPU 接触点缓冲上限（高密颗粒场景）
GPU_MAX_RIGID_PATCH_COUNT = 240_000  # GPU 接触 patch 缓冲上限
# ============================================= 计算 ============================================= 


EE_IK_EPS = 1e-4  # IK 收敛阈值
EE_IK_MAX_ITERS = 120  # IK 最大迭代次数
EE_IK_DT = 0.08  # IK 内部步长
EE_IK_DAMP = 1e-4  # IK 阻尼项
EE_APPLY_MODE = "direct"  # EE-IK 结果应用方式（当前建议保持 direct）

SCRIPTED_TIME_SCALE = 1.0  # 关键帧时间缩放（>1 慢放，<1 快放）
SCRIPTED_LOOP = False  # 是否循环播放关键帧
POOL_STATS_TOP_MARGIN = 0.30  # 池内统计时，墙顶向上额外容忍高度（用于容纳堆积）
POOL_STATS_PRINT_KEY = "p"  # 运行时按该键打印池内质量/体积与一致性对比


def create_scene(timestep: float = 1 / 240.0, prefer_gpu: bool = True) -> sapien.Scene:
    """Create a SAPIEN scene and prefer GPU PhysX when available."""
    if prefer_gpu:
        try:
            # SAPIEN 3.x requires explicit GPU enable before creating PhysxGpuSystem.
            if not sapien.physx.is_gpu_enabled():
                sapien.physx.enable_gpu()
            sapien.physx.set_gpu_memory_config(
                max_rigid_contact_count=GPU_MAX_RIGID_CONTACT_COUNT,
                max_rigid_patch_count=GPU_MAX_RIGID_PATCH_COUNT,
            )
            scene = sapien.Scene([sapien.physx.PhysxGpuSystem("cuda"), sapien.render.RenderSystem()])
            print("[Info] Using PhysX GPU system (cuda).")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] GPU PhysX unavailable, fallback to CPU PhysX: {exc}")
            scene = sapien.Scene()
    else:
        scene = sapien.Scene()

    scene.set_timestep(timestep)
    return scene


def maybe_init_gpu_physx(scene: sapien.Scene) -> None:
    """Initialize GPU PhysX system when running on GPU backend."""
    physx_system = scene.physx_system
    if isinstance(physx_system, sapien.physx.PhysxGpuSystem):
        try:
            physx_system.gpu_init()
            print("[Info] GPU PhysX initialized.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] GPU PhysX init failed, simulation may fallback internally: {exc}")


def configure_lighting(scene: sapien.Scene, ground_half_size: tuple[float, float] = GROUND_HALF_SIZE) -> None:
    """Set practical lighting for full-scene inspection."""
    scene.set_ambient_light([0.38, 0.38, 0.38])
    scene.add_directional_light(direction=[0.25, -0.35, -1.0], color=[0.95, 0.92, 0.90], shadow=True)
    scene.add_point_light(position=[0.0, 0.0, 2.8], color=[0.45, 0.45, 0.45], shadow=False)
    scene.add_ground(altitude=0.0, render_half_size=list(ground_half_size))


def get_urdf_candidates_from_config(config: dict[str, Any], equipment_model: str) -> list[str]:
    """Read URDF candidate list for a model from config."""
    model_map = config.get("urdf_candidates", {})
    if not isinstance(model_map, dict):
        raise ValueError("Invalid config: 'urdf_candidates' must be an object.")
    candidates = model_map.get(equipment_model)
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise ValueError(f"No urdf_candidates found for equipment_model={equipment_model} in config.")
    for item in candidates:
        if not isinstance(item, str):
            raise ValueError(f"Invalid urdf candidate for {equipment_model}: expected string, got {type(item)}")
    return candidates


def resolve_excavator_urdf_path(
    config: dict[str, Any], equipment_model: str, user_path: str | None
) -> Path:
    """Resolve URDF path for excavator model."""
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF file not found: {path}")
        return path

    repo_root = Path(__file__).resolve().parent
    rel_candidates = get_urdf_candidates_from_config(config, equipment_model)
    for rel in rel_candidates:
        path = (repo_root / rel).resolve()
        if path.is_file():
            return path

    candidates_text = "\n".join(f"  - {repo_root / rel}" for rel in rel_candidates)
    raise FileNotFoundError(f"No excavator URDF found for {equipment_model}. Tried:\n{candidates_text}")


def build_platform(
    scene: sapien.Scene,
    center: tuple[float, float] = PLATFORM_CENTER,
    half_size: tuple[float, float, float] = PLATFORM_HALF_SIZE,
) -> sapien.Entity:
    """Build a kinematic platform on the right side."""
    platform_material = scene.create_physical_material(static_friction=1.25, dynamic_friction=1.05, restitution=0.02)
    platform_visual = sapien.render.RenderMaterial(
        base_color=[0.50, 0.52, 0.56, 1.0],
        roughness=0.88,
        specular=0.12,
        metallic=0.0,
    )

    builder = scene.create_actor_builder()
    pose = sapien.Pose(p=[center[0], center[1], half_size[2]])
    builder.add_box_collision(pose=pose, half_size=list(half_size), material=platform_material)
    builder.add_box_visual(pose=pose, half_size=list(half_size), material=platform_visual, name="platform_top")
    return builder.build_kinematic(name="equipment_platform")


def build_particle_pool(
    scene: sapien.Scene,
    center: tuple[float, float] = POOL_CENTER,
    inner_half_size: tuple[float, float] = POOL_INNER_HALF_SIZE,
    wall_height: float = POOL_WALL_HEIGHT,
    wall_thickness: float = POOL_WALL_THICKNESS,
    bottom_thickness: float = POOL_BOTTOM_THICKNESS,
    base_color: tuple[float, float, float, float] = (0.46, 0.56, 0.64, 1.0),
    name: str = "particle_pool",
) -> sapien.Entity:
    """Build a square pool (bottom + 4 walls)."""
    pool_material = scene.create_physical_material(
        static_friction=1.90,
        dynamic_friction=1.60,
        restitution=0.0,
    )
    pool_visual = sapien.render.RenderMaterial(
        base_color=list(base_color),
        roughness=0.82,
        specular=0.16,
        metallic=0.0,
    )

    cx, cy = center
    ix, iy = inner_half_size
    wall_center_z = wall_height / 2.0

    builder = scene.create_actor_builder()

    bottom_half = [ix + wall_thickness, iy + wall_thickness, bottom_thickness]
    bottom_pose = sapien.Pose(p=[cx, cy, -bottom_thickness])
    builder.add_box_collision(pose=bottom_pose, half_size=bottom_half, material=pool_material)
    builder.add_box_visual(pose=bottom_pose, half_size=bottom_half, material=pool_visual, name="pool_bottom")

    x_wall_half = [wall_thickness, iy + wall_thickness, wall_height / 2.0]
    x_wall_offset = ix + wall_thickness
    for sign in (+1.0, -1.0):
        pose = sapien.Pose(p=[cx + sign * x_wall_offset, cy, wall_center_z])
        builder.add_box_collision(pose=pose, half_size=x_wall_half, material=pool_material)
        builder.add_box_visual(pose=pose, half_size=x_wall_half, material=pool_visual)

    y_wall_half = [ix, wall_thickness, wall_height / 2.0]
    y_wall_offset = iy + wall_thickness
    for sign in (+1.0, -1.0):
        pose = sapien.Pose(p=[cx, cy + sign * y_wall_offset, wall_center_z])
        builder.add_box_collision(pose=pose, half_size=y_wall_half, material=pool_material)
        builder.add_box_visual(pose=pose, half_size=y_wall_half, material=pool_visual)

    return builder.build_kinematic(name=name)


def spawn_particles(
    scene: sapien.Scene,
    center: tuple[float, float] = POOL_CENTER,
    inner_half_size: tuple[float, float] = POOL_INNER_HALF_SIZE,
    particle_count: int = PARTICLE_COUNT,
    radius: float = PARTICLE_RADIUS,
    render_particles: bool = PARTICLE_RENDER_ENABLED,
) -> list[sapien.Entity]:
    """Spawn dynamic particles from near the pool bottom (not from high altitude)."""
    if particle_count <= 0:
        return []

    px, py = center
    ix, iy = inner_half_size
    margin = max(0.02, 3.0 * radius)
    spacing = radius * 2.0

    nx = max(2, int((2 * (ix - margin)) / spacing))
    ny = max(2, int((2 * (iy - margin)) / spacing))
    nz = int(np.ceil(particle_count / float(nx * ny)))

    x_coords = px + (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * spacing
    y_coords = py + (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * spacing
    # Bottom plate top surface is z=0. Spawn first layer just above bottom.
    z_start = radius + 0.0015
    z_coords = z_start + np.arange(nz, dtype=np.float32) * spacing

    rng = np.random.default_rng(seed=7)
    gx, gy, gz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    all_positions = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    total_available = all_positions.shape[0]

    # Uniformly sample from the full volume so low-count settings do not only fill one corner.
    if total_available > particle_count:
        indices = rng.choice(total_available, size=particle_count, replace=False)
        positions = all_positions[indices]
    else:
        positions = all_positions
    # XY jitter breaks perfect lattice; Z jitter stays positive to avoid initial floor penetration.
    xy_jitter = rng.uniform(
        low=-0.06 * radius,
        high=0.06 * radius,
        size=(positions.shape[0], 2),
    ).astype(np.float32)
    z_jitter = rng.uniform(
        low=0.0,
        high=0.06 * radius,
        size=(positions.shape[0], 1),
    ).astype(np.float32)
    positions += np.concatenate([xy_jitter, z_jitter], axis=1)

    particle_material = scene.create_physical_material(
        static_friction=SAND_STATIC_FRICTION,
        dynamic_friction=SAND_DYNAMIC_FRICTION,
        restitution=SAND_RESTITUTION,
    )
    particle_visual = sapien.render.RenderMaterial(
        base_color=[0.76, 0.66, 0.42, 1.0],
        roughness=0.95,
        specular=0.06,
        metallic=0.0,
    )

    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius, material=particle_material, density=SAND_DENSITY)
    if render_particles:
        builder.add_sphere_visual(radius=radius, material=particle_visual)

    particles: list[sapien.Entity] = []
    for pos in positions:
        builder.set_initial_pose(sapien.Pose(p=pos.tolist()))
        particle = builder.build()
        rigid = particle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        rigid.set_linear_damping(SAND_LINEAR_DAMPING)
        rigid.set_angular_damping(SAND_ANGULAR_DAMPING)
        rigid.set_solver_position_iterations(SAND_SOLVER_POS_ITERS)
        rigid.set_solver_velocity_iterations(SAND_SOLVER_VEL_ITERS)
        rigid.set_max_depenetration_velocity(SAND_MAX_DEPENETRATION_VEL)
        rigid.set_max_linear_velocity(SAND_MAX_LINEAR_VEL)
        rigid.set_max_angular_velocity(SAND_MAX_ANGULAR_VEL)
        rigid.set_max_contact_impulse(SAND_MAX_CONTACT_IMPULSE)
        particles.append(particle)

    print(f"[Info] Spawned {len(particles)} particles.")
    return particles


def get_particle_unit_volume(radius: float = PARTICLE_RADIUS) -> float:
    """Return per-particle sphere volume in m^3."""
    r = float(radius)
    return (4.0 / 3.0) * float(np.pi) * r * r * r


def get_particle_unit_mass(radius: float = PARTICLE_RADIUS, density: float = SAND_DENSITY) -> float:
    """Return per-particle mass in kg based on radius and density."""
    return get_particle_unit_volume(radius=radius) * float(density)


def get_particle_positions(particles: list[sapien.Entity]) -> np.ndarray:
    """Return particle world positions as (N, 3) array."""
    if len(particles) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pos = np.zeros((len(particles), 3), dtype=np.float32)
    for i, particle in enumerate(particles):
        pos[i] = np.asarray(particle.pose.p, dtype=np.float32).reshape(3)
    return pos


def compute_pool_particle_stats(
    particles: list[sapien.Entity],
    center: tuple[float, float],
    inner_half_size: tuple[float, float],
    wall_height: float,
    particle_radius: float = PARTICLE_RADIUS,
    particle_density: float = SAND_DENSITY,
    top_margin: float = POOL_STATS_TOP_MARGIN,
) -> dict[str, float]:
    """Compute count/mass/volume for particles inside a pool's XY footprint and Z window."""
    positions = get_particle_positions(particles)
    if positions.shape[0] == 0:
        return {
            "count": 0.0,
            "mass_kg": 0.0,
            "volume_m3": 0.0,
        }

    cx, cy = center
    hx, hy = inner_half_size
    z_min = 0.0
    z_max = float(wall_height + top_margin)

    in_x = np.abs(positions[:, 0] - float(cx)) <= float(hx)
    in_y = np.abs(positions[:, 1] - float(cy)) <= float(hy)
    in_z = (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
    in_pool = in_x & in_y & in_z

    count = int(np.sum(in_pool))
    unit_mass = get_particle_unit_mass(radius=particle_radius, density=particle_density)
    unit_volume = get_particle_unit_volume(radius=particle_radius)
    return {
        "count": float(count),
        "mass_kg": float(count * unit_mass),
        "volume_m3": float(count * unit_volume),
    }


def compute_transfer_consistency(
    source_initial_count: int,
    source_current_count: int,
    receiver_current_count: int,
    particle_radius: float = PARTICLE_RADIUS,
    particle_density: float = SAND_DENSITY,
) -> dict[str, float]:
    """Compare source decrease and receiver increase, return discrepancy metrics."""
    removed_from_source = max(0, int(source_initial_count) - int(source_current_count))
    received_in_target = max(0, int(receiver_current_count))
    discrepancy_count = removed_from_source - received_in_target
    unit_mass = get_particle_unit_mass(radius=particle_radius, density=particle_density)
    unit_volume = get_particle_unit_volume(radius=particle_radius)
    return {
        "source_removed_count": float(removed_from_source),
        "receiver_count": float(received_in_target),
        "discrepancy_count": float(discrepancy_count),
        "source_removed_mass_kg": float(removed_from_source * unit_mass),
        "receiver_mass_kg": float(received_in_target * unit_mass),
        "discrepancy_mass_kg": float(discrepancy_count * unit_mass),
        "source_removed_volume_m3": float(removed_from_source * unit_volume),
        "receiver_volume_m3": float(received_in_target * unit_volume),
        "discrepancy_volume_m3": float(discrepancy_count * unit_volume),
    }


def set_initial_joint_pose(robot: sapien.physx.PhysxArticulation, qpos: np.ndarray) -> None:
    """Apply initial articulation qpos if provided by config."""
    if robot.dof <= 0:
        return
    if robot.dof != len(qpos):
        raise ValueError(f"Initial qpos length mismatch: robot DOF={robot.dof}, provided={len(qpos)}")
    robot.set_qpos(qpos.astype(np.float32))


def load_pose_config(config_path: Path) -> dict[str, Any]:
    """Load config.json that stores per-model, per-scene initial poses."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}: root must be an object")
    return data


def extract_config_path_from_argv(argv: list[str], default_path: Path) -> Path:
    """Extract config path from raw argv supporting '--config X' and '--config=X'."""
    for i, token in enumerate(argv):
        if token == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    return default_path.expanduser().resolve()


def get_available_excavator_models_from_config(config: dict[str, Any]) -> list[str]:
    """Return sorted excavator model names from config URDF candidates."""
    model_map = config.get("urdf_candidates", {})
    if not isinstance(model_map, dict):
        raise ValueError("Invalid config: 'urdf_candidates' must be an object.")
    models = sorted(
        k
        for k, v in model_map.items()
        if isinstance(k, str) and k.startswith("excavator") and isinstance(v, list) and len(v) > 0
    )
    if not models:
        raise ValueError("No excavator models found in config['urdf_candidates'].")
    return models


def get_initial_qpos_from_config(config: dict[str, Any], equipment_model: str, scene_name: str) -> np.ndarray | None:
    """Get initial qpos with fallback order: model.scene -> model.default -> None."""
    model_map = config.get("initial_pose", {}).get(equipment_model)
    if not isinstance(model_map, dict):
        return None

    raw_pose = model_map.get(scene_name, model_map.get("default"))
    if raw_pose is None:
        return None
    if not isinstance(raw_pose, list):
        raise ValueError(
            f"Invalid pose for model={equipment_model}, scene={scene_name}: expected list, got {type(raw_pose)}"
        )
    return np.asarray(raw_pose, dtype=np.float32).reshape(-1)


def load_excavator(
    scene: sapien.Scene,
    equipment_model: str,
    urdf_path: Path,
    platform_center: tuple[float, float],
    platform_half_height: float,
    init_qpos: np.ndarray | None,
) -> sapien.physx.PhysxArticulation:
    """Load articulation and place it on the right platform."""
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.set_material(TOOL_STATIC_FRICTION, TOOL_DYNAMIC_FRICTION, TOOL_RESTITUTION)

    robot = loader.load(str(urdf_path), package_dir=str(urdf_path.parent))
    if robot is None:
        raise RuntimeError(f"Failed to load excavator: {urdf_path}")

    root_pose = sapien.Pose(p=[platform_center[0], platform_center[1], 2.0 * platform_half_height + 0.015])
    root_pose.set_rpy([0.0, 0.0, np.pi])
    robot.set_root_pose(root_pose)
    robot.set_solver_position_iterations(TOOL_SOLVER_POS_ITERS)
    robot.set_solver_velocity_iterations(TOOL_SOLVER_VEL_ITERS)

    for link in robot.links:
        link.set_max_depenetration_velocity(TOOL_MAX_DEPENETRATION_VEL)

    is_gpu_backend = isinstance(scene.physx_system, sapien.physx.PhysxGpuSystem)
    if init_qpos is not None and not is_gpu_backend:
        try:
            set_initial_joint_pose(robot, init_qpos)
        except (RuntimeError, ValueError) as exc:
            print(f"[Warn] Failed to set startup qpos ({exc}).")
    elif init_qpos is not None and is_gpu_backend:
        print("[Info] GPU backend detected: skip set_qpos to avoid slow/invalid host-side articulation write.")
    else:
        print(f"[Info] No initial pose configured for model={equipment_model}, skip set_qpos.")

    print(f"[Info] Loaded excavator model={equipment_model} (dof={robot.dof})")
    return robot


def configure_joint_drives(
    robot: sapien.physx.PhysxArticulation,
    stiffness: float = JOINT_DRIVE_STIFFNESS,
    damping: float = JOINT_DRIVE_DAMPING,
    force_limit: float = JOINT_DRIVE_FORCE_LIMIT,
) -> None:
    """Configure articulation active joint drives for target-position tracking."""
    for joint in robot.active_joints:
        joint.set_drive_properties(
            stiffness=float(stiffness),
            damping=float(damping),
            force_limit=float(force_limit),
            mode="force",
        )


def apply_joint_target(
    robot: sapien.physx.PhysxArticulation,
    target_qpos: np.ndarray,
    physx_system: sapien.physx.PhysxSystem,
) -> bool:
    """Apply a joint-space target in CPU/GPU backends.

    CPU: use per-joint drive targets.
    GPU: set articulation qpos + push to GPU via gpu_apply_articulation_qpos.
    """
    target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
    if target_qpos.shape[0] != robot.dof:
        raise ValueError(f"Target qpos length mismatch: got {target_qpos.shape[0]}, expected {robot.dof}")

    is_gpu_backend = isinstance(physx_system, sapien.physx.PhysxGpuSystem)
    if is_gpu_backend:
        gpu_system = physx_system
        try:
            robot.set_qpos(target_qpos)
            gpu_system.gpu_apply_articulation_qpos()
            return True
        except Exception:  # noqa: BLE001
            return False

    for joint, q in zip(robot.active_joints, target_qpos, strict=False):
        joint.set_drive_target(float(q))
    return True


def resolve_ee_link(
    robot: sapien.physx.PhysxArticulation,
    preferred_name: str | None = None,
) -> tuple[int, str]:
    """Resolve end-effector link index in articulation link order."""
    links = robot.links
    if preferred_name:
        for i, link in enumerate(links):
            if link.name == preferred_name:
                return i, link.name
        available = ", ".join(link.name for link in links)
        raise ValueError(f"EE link '{preferred_name}' not found. Available links: {available}")

    # Fallback: choose the deepest childless link if possible, else use last link.
    childless_indices = [i for i, link in enumerate(links) if len(link.children) == 0]
    if childless_indices:
        idx = childless_indices[-1]
    else:
        idx = len(links) - 1
    return idx, links[idx].name


def solve_ee_ik_target_qpos(
    robot: sapien.physx.PhysxArticulation,
    pin_model: sapien.PinocchioModel,
    ee_link_index: int,
    target_world_pose: sapien.Pose,
    qpos_seed: np.ndarray,
    active_qmask: np.ndarray,
    eps: float = EE_IK_EPS,
    max_iterations: int = EE_IK_MAX_ITERS,
    dt: float = EE_IK_DT,
    damp: float = EE_IK_DAMP,
) -> tuple[np.ndarray, bool, float]:
    """Solve IK for target EE pose in world frame, return qpos in articulation order."""
    target_local_pose = robot.pose.inv() * target_world_pose
    q, success, error = pin_model.compute_inverse_kinematics(
        link_index=ee_link_index,
        pose=target_local_pose,
        initial_qpos=np.asarray(qpos_seed, dtype=np.float32).reshape(-1),
        active_qmask=np.asarray(active_qmask, dtype=np.int32).reshape(-1),
        eps=float(eps),
        max_iterations=int(max_iterations),
        dt=float(dt),
        damp=float(damp),
    )
    q_out = np.asarray(q, dtype=np.float32).reshape(-1)

    # Different pinocchio backends may return scalar/buffer values.
    success_arr = np.asarray(success).reshape(-1)
    success_out = bool(success_arr[0]) if success_arr.size > 0 else bool(success)

    error_arr = np.asarray(error, dtype=np.float64).reshape(-1)
    if error_arr.size == 0:
        error_out = 0.0
    elif error_arr.size == 1:
        error_out = float(error_arr[0])
    else:
        error_out = float(np.linalg.norm(error_arr))

    return q_out, success_out, error_out


def run_viewer(
    scene: sapien.Scene,
    camera_xyz: tuple[float, float, float] = CAMERA_XYZ,
    camera_rpy: tuple[float, float, float] = CAMERA_RPY,
    start_paused: bool = True,
    robot: sapien.physx.PhysxArticulation | None = None,
    ee_scripted_policy: LinearEEKeyframePolicy | None = None,
    pin_model: sapien.PinocchioModel | None = None,
    ee_link_index: int | None = None,
    ee_ik_active_qmask: np.ndarray | None = None,
    ee_apply_mode: str = "direct",
    particles: list[sapien.Entity] | None = None,
    source_pool_center: tuple[float, float] = POOL_CENTER,
    source_pool_inner_half_size: tuple[float, float] = POOL_INNER_HALF_SIZE,
    source_pool_wall_height: float = POOL_WALL_HEIGHT,
    receiver_pool_center: tuple[float, float] = RECEIVER_POOL_CENTER,
    receiver_pool_inner_half_size: tuple[float, float] = RECEIVER_POOL_INNER_HALF_SIZE,
    receiver_pool_wall_height: float = RECEIVER_POOL_WALL_HEIGHT,
    source_initial_count: int | None = None,
) -> None:
    """Run the real-time viewer loop."""
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2])
    viewer.set_camera_rpy(r=camera_rpy[0], p=camera_rpy[1], y=camera_rpy[2])
    viewer.window.set_camera_parameters(near=0.01, far=30.0, fovy=np.deg2rad(58.0))
    viewer.paused = start_paused
    physx_system = scene.physx_system
    is_gpu_backend = isinstance(physx_system, sapien.physx.PhysxGpuSystem)

    # Ensure initial GPU state is visible to the renderer.
    if is_gpu_backend:
        try:
            physx_system.sync_poses_gpu_to_cpu()
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] Initial GPU pose sync failed: {exc}")

    sim_step_index = 0
    warned_control_failure = False
    warned_ik_failure = False
    qpos_seed: np.ndarray | None = None
    prev_stats_key_down = False
    tracked_particles = particles if particles is not None else []
    init_source_count = (
        int(source_initial_count)
        if source_initial_count is not None
        else len(tracked_particles)
    )

    print(f"[Info] Viewer started. paused={viewer.paused}. Close window to exit.")
    if len(tracked_particles) > 0:
        print(f"[Info] Press '{POOL_STATS_PRINT_KEY}' to print source/receiver pool mass-volume statistics.")
    while not viewer.closed:
        stats_key_down = bool(viewer.window.key_down(POOL_STATS_PRINT_KEY))
        if stats_key_down and (not prev_stats_key_down) and len(tracked_particles) > 0:
            source_stats = compute_pool_particle_stats(
                particles=tracked_particles,
                center=source_pool_center,
                inner_half_size=source_pool_inner_half_size,
                wall_height=source_pool_wall_height,
            )
            receiver_stats = compute_pool_particle_stats(
                particles=tracked_particles,
                center=receiver_pool_center,
                inner_half_size=receiver_pool_inner_half_size,
                wall_height=receiver_pool_wall_height,
            )
            consistency = compute_transfer_consistency(
                source_initial_count=init_source_count,
                source_current_count=int(source_stats["count"]),
                receiver_current_count=int(receiver_stats["count"]),
            )
            print(
                "[Stats] Source(count/mass/vol)="
                f"{int(source_stats['count'])}/{source_stats['mass_kg']:.6f}kg/{source_stats['volume_m3']:.6f}m3 | "
                "Receiver(count/mass/vol)="
                f"{int(receiver_stats['count'])}/{receiver_stats['mass_kg']:.6f}kg/{receiver_stats['volume_m3']:.6f}m3"
            )
            print(
                "[Stats] Transfer consistency: "
                f"source_removed={int(consistency['source_removed_count'])}, "
                f"receiver={int(consistency['receiver_count'])}, "
                f"delta={int(consistency['discrepancy_count'])} "
                f"(mass_delta={consistency['discrepancy_mass_kg']:.6f}kg, "
                f"vol_delta={consistency['discrepancy_volume_m3']:.6f}m3)"
            )
        prev_stats_key_down = stats_key_down

        if not viewer.paused:
            if (
                robot is not None
                and ee_scripted_policy is not None
                and pin_model is not None
                and ee_link_index is not None
                and ee_ik_active_qmask is not None
            ):
                xyz, rpy = ee_scripted_policy.query(sim_step_index)
                target_world_pose = sapien.Pose(p=xyz.tolist())
                target_world_pose.set_rpy(rpy.tolist())
                if qpos_seed is None:
                    qpos_seed = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
                q_ik, ik_success, ik_error = solve_ee_ik_target_qpos(
                    robot=robot,
                    pin_model=pin_model,
                    ee_link_index=ee_link_index,
                    target_world_pose=target_world_pose,
                    qpos_seed=qpos_seed,
                    active_qmask=ee_ik_active_qmask,
                    eps=EE_IK_EPS,
                    max_iterations=EE_IK_MAX_ITERS,
                    dt=EE_IK_DT,
                    damp=EE_IK_DAMP,
                )
                if (not is_gpu_backend) and ee_apply_mode == "direct":
                    # Deterministic replay path: directly write qpos so motion is always visible.
                    # This is useful for data collection and validating EE->IK trajectories.
                    robot.set_qpos(q_ik)
                    try:
                        robot.set_qvel(np.zeros_like(q_ik))
                    except Exception:  # noqa: BLE001
                        pass
                    for link in robot.links:
                        link.wake_up()
                    ok = True
                else:
                    ok = apply_joint_target(robot=robot, target_qpos=q_ik, physx_system=physx_system)
                if ok:
                    qpos_seed = q_ik
                if (not ik_success) and (not warned_ik_failure):
                    print(
                        f"[Warn] IK not fully converged (error={ik_error:.6f}). "
                        "Trajectory still uses best-effort qpos."
                    )
                    warned_ik_failure = True
                if (not ok) and (not warned_control_failure):
                    print(
                        "[Warn] Keyframe control failed on current backend. "
                        "Try `--cpu` for stable IK replay."
                    )
                    warned_control_failure = True
            scene.step()
            if is_gpu_backend:
                # GPU PhysX needs explicit pose sync for viewport updates.
                physx_system.sync_poses_gpu_to_cpu()
            sim_step_index += 1
        scene.update_render()
        viewer.render()


class GranularExcavatorEnv:
    """ACT-style environment wrapper: action comes from outside, env only executes it."""

    def __init__(
        self,
        equipment_model: str | None = None,
        config_path: str | Path | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else DEFAULT_CONFIG_PATH.expanduser().resolve()
        )
        self.pose_config = load_pose_config(self.config_path)
        self.model_choices = get_available_excavator_models_from_config(self.pose_config)
        if equipment_model is None:
            if "excavator_s010" in self.model_choices:
                self.equipment_model = "excavator_s010"
            elif "excavator_simple" in self.model_choices:
                self.equipment_model = "excavator_simple"
            else:
                self.equipment_model = self.model_choices[0]
        else:
            if equipment_model not in self.model_choices:
                raise ValueError(f"Unknown equipment_model={equipment_model}, choices={self.model_choices}")
            self.equipment_model = equipment_model

        self.prefer_gpu = bool(prefer_gpu)
        self.scene: sapien.Scene | None = None
        self.robot: sapien.physx.PhysxArticulation | None = None
        self.particles: list[sapien.Entity] = []
        self.source_initial_count: int = 0
        self.pin_model: sapien.PinocchioModel | None = None
        self.ee_link_index: int | None = None
        self._ee_qmask: np.ndarray | None = None
        self._ee_qseed: np.ndarray | None = None

    def reset(self) -> collections.OrderedDict[str, Any]:
        """Reset world and return first observation."""
        self.scene = create_scene(timestep=SIM_TIMESTEP, prefer_gpu=self.prefer_gpu)
        configure_lighting(self.scene, ground_half_size=GROUND_HALF_SIZE)

        build_platform(self.scene, center=PLATFORM_CENTER, half_size=PLATFORM_HALF_SIZE)
        build_particle_pool(
            self.scene,
            center=POOL_CENTER,
            inner_half_size=POOL_INNER_HALF_SIZE,
            wall_height=POOL_WALL_HEIGHT,
            wall_thickness=POOL_WALL_THICKNESS,
            bottom_thickness=POOL_BOTTOM_THICKNESS,
            base_color=(0.46, 0.56, 0.64, 1.0),
            name="source_particle_pool",
        )
        build_particle_pool(
            self.scene,
            center=RECEIVER_POOL_CENTER,
            inner_half_size=RECEIVER_POOL_INNER_HALF_SIZE,
            wall_height=RECEIVER_POOL_WALL_HEIGHT,
            wall_thickness=POOL_WALL_THICKNESS,
            bottom_thickness=POOL_BOTTOM_THICKNESS,
            base_color=(0.34, 0.62, 0.42, 1.0),
            name="receiver_particle_pool",
        )
        self.particles = spawn_particles(
            self.scene,
            center=POOL_CENTER,
            inner_half_size=POOL_INNER_HALF_SIZE,
            particle_count=PARTICLE_COUNT,
            radius=PARTICLE_RADIUS,
            render_particles=PARTICLE_RENDER_ENABLED,
        )
        self.source_initial_count = len(self.particles)

        urdf_path = resolve_excavator_urdf_path(self.pose_config, self.equipment_model, user_path=None)
        init_qpos = get_initial_qpos_from_config(self.pose_config, self.equipment_model, SCENE_NAME)
        self.robot = load_excavator(
            scene=self.scene,
            equipment_model=self.equipment_model,
            urdf_path=urdf_path,
            platform_center=PLATFORM_CENTER,
            platform_half_height=PLATFORM_HALF_SIZE[2],
            init_qpos=init_qpos,
        )
        # Always configure joint drives so external joint_pos action can be directly applied.
        configure_joint_drives(self.robot)
        maybe_init_gpu_physx(self.scene)

        self.pin_model = None
        self.ee_link_index = None
        self._ee_qmask = None
        self._ee_qseed = None
        self.scene.update_render()
        return self.get_observation()

    def _ensure_ready(self) -> None:
        if self.scene is None or self.robot is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

    def _ensure_ee_ik(self) -> None:
        self._ensure_ready()
        if self.pin_model is None:
            self.pin_model = self.robot.create_pinocchio_model()
            self.ee_link_index, _ = resolve_ee_link(self.robot, preferred_name=None)
            self._ee_qmask = np.zeros(self.robot.dof, dtype=np.int32)
            self._ee_qmask[: min(4, self.robot.dof)] = 1
            self._ee_qseed = np.asarray(self.robot.get_qpos(), dtype=np.float32).reshape(-1)

    def apply_action(self, action: np.ndarray | list[float] | dict[str, Any]) -> None:
        """Apply external action without stepping.

        Supported:
        - joint action: np.ndarray/list with shape (dof,) or {'joint_pos': ...}
        - ee action: {'ee_pose': {'xyz': [x,y,z], 'rpy': [r,p,y]}}
        """
        self._ensure_ready()
        assert self.scene is not None and self.robot is not None

        if isinstance(action, dict):
            if "joint_pos" in action:
                joint_target = np.asarray(action["joint_pos"], dtype=np.float32).reshape(-1)
                ok = apply_joint_target(self.robot, joint_target, self.scene.physx_system)
                if not ok:
                    raise RuntimeError("Failed to apply joint_pos action on current backend.")
                return
            if "ee_pose" in action:
                ee_pose = action["ee_pose"]
                if not isinstance(ee_pose, dict):
                    raise ValueError("action['ee_pose'] must be dict with keys xyz and rpy.")
                xyz = np.asarray(ee_pose["xyz"], dtype=np.float32).reshape(3)
                rpy = np.asarray(ee_pose["rpy"], dtype=np.float32).reshape(3)
                self._ensure_ee_ik()
                assert self.pin_model is not None and self.ee_link_index is not None and self._ee_qmask is not None
                if self._ee_qseed is None:
                    self._ee_qseed = np.asarray(self.robot.get_qpos(), dtype=np.float32).reshape(-1)
                target_world_pose = sapien.Pose(p=xyz.tolist())
                target_world_pose.set_rpy(rpy.tolist())
                q_ik, _, _ = solve_ee_ik_target_qpos(
                    robot=self.robot,
                    pin_model=self.pin_model,
                    ee_link_index=self.ee_link_index,
                    target_world_pose=target_world_pose,
                    qpos_seed=self._ee_qseed,
                    active_qmask=self._ee_qmask,
                    eps=EE_IK_EPS,
                    max_iterations=EE_IK_MAX_ITERS,
                    dt=EE_IK_DT,
                    damp=EE_IK_DAMP,
                )
                ok = apply_joint_target(self.robot, q_ik, self.scene.physx_system)
                if not ok:
                    raise RuntimeError("Failed to apply IK joint target on current backend.")
                self._ee_qseed = q_ik
                return
            raise ValueError("Unsupported action dict. Use {'joint_pos': ...} or {'ee_pose': {'xyz','rpy'}}")

        joint_target = np.asarray(action, dtype=np.float32).reshape(-1)
        ok = apply_joint_target(self.robot, joint_target, self.scene.physx_system)
        if not ok:
            raise RuntimeError("Failed to apply joint_pos action on current backend.")

    def step(
        self,
        action: np.ndarray | list[float] | dict[str, Any] | None = None,
        n_substeps: int = 1,
    ) -> tuple[collections.OrderedDict[str, Any], dict[str, Any]]:
        """Apply action then advance simulation."""
        self._ensure_ready()
        assert self.scene is not None
        if action is not None:
            self.apply_action(action)

        n_steps = max(1, int(n_substeps))
        for _ in range(n_steps):
            self.scene.step()
            if isinstance(self.scene.physx_system, sapien.physx.PhysxGpuSystem):
                self.scene.physx_system.sync_poses_gpu_to_cpu()
        self.scene.update_render()
        obs = self.get_observation()
        metrics = self.get_transfer_metrics()
        return obs, metrics

    def get_transfer_metrics(self) -> dict[str, Any]:
        """Return source/receiver stats and transfer consistency."""
        source_stats = compute_pool_particle_stats(
            particles=self.particles,
            center=POOL_CENTER,
            inner_half_size=POOL_INNER_HALF_SIZE,
            wall_height=POOL_WALL_HEIGHT,
        )
        receiver_stats = compute_pool_particle_stats(
            particles=self.particles,
            center=RECEIVER_POOL_CENTER,
            inner_half_size=RECEIVER_POOL_INNER_HALF_SIZE,
            wall_height=RECEIVER_POOL_WALL_HEIGHT,
        )
        consistency = compute_transfer_consistency(
            source_initial_count=self.source_initial_count,
            source_current_count=int(source_stats["count"]),
            receiver_current_count=int(receiver_stats["count"]),
        )
        return {
            "source_pool": source_stats,
            "receiver_pool": receiver_stats,
            "consistency": consistency,
        }

    def get_ee_pose_world(self) -> dict[str, np.ndarray]:
        """Return current end-effector world pose as {'xyz': (3,), 'rpy': (3,)}."""
        self._ensure_ready()
        assert self.robot is not None
        if self.ee_link_index is None:
            self.ee_link_index, _ = resolve_ee_link(self.robot, preferred_name=None)
        ee_pose = self.robot.links[self.ee_link_index].entity_pose
        return {
            "xyz": np.asarray(ee_pose.p, dtype=np.float32).reshape(3).copy(),
            "rpy": np.asarray(ee_pose.rpy, dtype=np.float32).reshape(3).copy(),
        }

    def get_observation(self) -> collections.OrderedDict[str, Any]:
        """Return ACT-style observation dict."""
        self._ensure_ready()
        assert self.robot is not None
        obs: collections.OrderedDict[str, Any] = collections.OrderedDict()
        obs["qpos"] = np.asarray(self.robot.get_qpos(), dtype=np.float32).copy()
        obs["qvel"] = np.asarray(self.robot.get_qvel(), dtype=np.float32).copy()
        obs["env_state"] = self.get_transfer_metrics()
        obs["images"] = {}
        return obs


def make_excavator_env(
    equipment_model: str | None = None,
    config_path: str | Path | None = None,
    prefer_gpu: bool = True,
) -> GranularExcavatorEnv:
    """Factory function aligned with ViPACT make_*_env style."""
    env = GranularExcavatorEnv(
        equipment_model=equipment_model,
        config_path=config_path,
        prefer_gpu=prefer_gpu,
    )
    env.reset()
    return env


def main() -> None:
    bootstrap_config_path = extract_config_path_from_argv(sys.argv[1:], DEFAULT_CONFIG_PATH)
    bootstrap_config = load_pose_config(bootstrap_config_path)
    model_choices = get_available_excavator_models_from_config(bootstrap_config)
    if "excavator_s010" in model_choices:
        default_model = "excavator_s010"
    elif "excavator_simple" in model_choices:
        default_model = "excavator_simple"
    else:
        default_model = model_choices[0]

    parser = argparse.ArgumentParser(description="Excavator + granular pool environment in SAPIEN 3.0.")
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
    parser.add_argument("--cpu", action="store_true", help="Force CPU PhysX.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("manual", "keyframe"),
        default="manual",
        help="运行模式：manual=手动调关节，keyframe=按末端关键帧自动运动。",
    )
    parser.add_argument(
        "--keyframes-json",
        type=str,
        default=None,
        help="末端关键帧 JSON 路径，仅在 mode=keyframe 时使用。",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    pose_config = load_pose_config(config_path)

    scene = create_scene(timestep=SIM_TIMESTEP, prefer_gpu=not args.cpu)
    configure_lighting(scene, ground_half_size=GROUND_HALF_SIZE)

    platform_center = PLATFORM_CENTER
    platform_half_size = PLATFORM_HALF_SIZE
    build_platform(scene, center=platform_center, half_size=platform_half_size)

    pool_center = POOL_CENTER
    pool_inner_half_size = POOL_INNER_HALF_SIZE
    build_particle_pool(
        scene,
        center=pool_center,
        inner_half_size=pool_inner_half_size,
        wall_height=POOL_WALL_HEIGHT,
        wall_thickness=POOL_WALL_THICKNESS,
        bottom_thickness=POOL_BOTTOM_THICKNESS,
        base_color=(0.46, 0.56, 0.64, 1.0),
        name="source_particle_pool",
    )
    build_particle_pool(
        scene,
        center=RECEIVER_POOL_CENTER,
        inner_half_size=RECEIVER_POOL_INNER_HALF_SIZE,
        wall_height=RECEIVER_POOL_WALL_HEIGHT,
        wall_thickness=POOL_WALL_THICKNESS,
        bottom_thickness=POOL_BOTTOM_THICKNESS,
        base_color=(0.34, 0.62, 0.42, 1.0),
        name="receiver_particle_pool",
    )
    particles = spawn_particles(
        scene,
        center=pool_center,
        inner_half_size=pool_inner_half_size,
        particle_count=PARTICLE_COUNT,
        radius=PARTICLE_RADIUS,
        render_particles=PARTICLE_RENDER_ENABLED,
    )

    urdf_path = resolve_excavator_urdf_path(pose_config, args.equipment_model, user_path=None)
    init_qpos = get_initial_qpos_from_config(pose_config, args.equipment_model, SCENE_NAME)
    print(f"[Info] Loading excavator URDF: {urdf_path}")
    print(f"[Info] Loading pose config: {config_path}")
    robot = load_excavator(
        scene=scene,
        equipment_model=args.equipment_model,
        urdf_path=urdf_path,
        platform_center=platform_center,
        platform_half_height=platform_half_size[2],
        init_qpos=init_qpos,
    )

    ee_scripted_policy: LinearEEKeyframePolicy | None = None
    pin_model: sapien.PinocchioModel | None = None
    ee_link_index: int | None = None
    ee_ik_active_qmask: np.ndarray | None = None

    if args.mode == "manual":
        print("[Info] Manual mode: 启动后默认暂停，可在 Viewer 中手动调节关节并控制开始/暂停。")
    elif args.mode == "keyframe":
        if isinstance(scene.physx_system, sapien.physx.PhysxGpuSystem):
            print("[Warn] Keyframe IK replay is recommended on CPU for better stability/debuggability.")
        else:
            configure_joint_drives(robot)
            print(
                "[Info] Keyframe control enabled (CPU drive target mode): "
                f"stiffness={JOINT_DRIVE_STIFFNESS}, damping={JOINT_DRIVE_DAMPING}, "
                f"force_limit={JOINT_DRIVE_FORCE_LIMIT}"
            )
            print(f"[Info] EE apply mode (CPU): {EE_APPLY_MODE}")

        ee_link_index, ee_link_name = resolve_ee_link(robot, preferred_name=None)
        print(f"[Info] EE link for IK: name={ee_link_name}, index={ee_link_index}")

        try:
            pin_model = robot.create_pinocchio_model()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to create pinocchio model for IK. "
                "Please ensure SAPIEN pinocchio backend is available."
            ) from exc

        if args.keyframes_json:
            ee_keyframes = load_ee_keyframes_json(args.keyframes_json)
            print(f"[Info] Loaded EE keyframes: {Path(args.keyframes_json).expanduser().resolve()}")
        else:
            ee_pose = robot.links[ee_link_index].entity_pose
            ee_keyframes = build_default_excavator_ee_keyframes(
                init_xyz=np.asarray(ee_pose.p, dtype=np.float32).reshape(3),
                init_rpy=np.asarray(ee_pose.rpy, dtype=np.float32).reshape(3),
            )
            print("[Info] Using built-in default EE keyframes.")

        ee_scripted_policy = LinearEEKeyframePolicy(
            keyframes=ee_keyframes,
            time_scale=SCRIPTED_TIME_SCALE,
            loop=SCRIPTED_LOOP,
        )

        # Keep only first 4 DOFs active for the excavator chain as requested.
        ee_ik_active_qmask = np.zeros(robot.dof, dtype=np.int32)
        ee_ik_active_qmask[: min(4, robot.dof)] = 1
        print(f"[Info] EE IK active DOFs: {int(np.sum(ee_ik_active_qmask))}/{robot.dof}")

    maybe_init_gpu_physx(scene)
    run_viewer(
        scene,
        camera_xyz=CAMERA_XYZ,
        camera_rpy=CAMERA_RPY,
        start_paused=True,
        robot=robot,
        ee_scripted_policy=ee_scripted_policy,
        pin_model=pin_model,
        ee_link_index=ee_link_index,
        ee_ik_active_qmask=ee_ik_active_qmask,
        ee_apply_mode=EE_APPLY_MODE,
        particles=particles,
        source_pool_center=POOL_CENTER,
        source_pool_inner_half_size=POOL_INNER_HALF_SIZE,
        source_pool_wall_height=POOL_WALL_HEIGHT,
        receiver_pool_center=RECEIVER_POOL_CENTER,
        receiver_pool_inner_half_size=RECEIVER_POOL_INNER_HALF_SIZE,
        receiver_pool_wall_height=RECEIVER_POOL_WALL_HEIGHT,
        source_initial_count=len(particles),
    )


if __name__ == "__main__":
    main()
