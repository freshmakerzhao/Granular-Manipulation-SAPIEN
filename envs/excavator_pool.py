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
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import sapien
try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None
try:
    from transforms3d.quaternions import axangle2quat as _axangle2quat
    from transforms3d.quaternions import qmult as _qmult
except Exception:  # noqa: BLE001
    _axangle2quat = None
    _qmult = None
from sapien.utils import Viewer
from scripted_policy import (
    LinearJointKeyframePolicy,
    build_default_excavator_keyframes,
    load_joint_keyframes_json,
)


SCENE_NAME = "excavator_pool_env"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.json"

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
SOURCE_POOL_BASE_COLOR = (0.46, 0.56, 0.64, 0.7)  # 料池颜色+透明度 RGBA（a 越小越透明）
# ============================================= 料池 ============================================= 

# ============================================= 接料池 ============================================= 
RECEIVER_POOL_CENTER = (0.4, 0.60)  # 接料池中心
RECEIVER_POOL_INNER_HALF_SIZE = (0.16, 0.14)  # 接料池内部半尺寸 (x, y)
RECEIVER_POOL_WALL_HEIGHT = 0.14  # 接料池墙高 (m)
RECEIVER_POOL_BASE_COLOR = (0.34, 0.62, 0.42, 0.7)   # 接料池颜色+透明度 RGBA（a 越小越透明）
# ============================================= 接料池 ============================================= 

# ============================================= 相机 ============================================= 
CAMERA_XYZ = (-0.41492998600006104, 1.4828300476074219, 1.3395099639892578)  # Viewer 相机位置 (x, y, z)
CAMERA_RPY = (-0.0, -0.6399999856948853, 1.2949999570846558)  # Viewer 相机姿态 (roll, pitch, yaw)
HEADLESS_CAMERA_RPY = (-0.0, -0.6399999856948853, 1.2949999570846558) # 后台回放离屏相机姿态（确保镜头朝向场景）

HEADLESS_CAMERA_WIDTH = 1280  # 后台回放导出分辨率宽
HEADLESS_CAMERA_HEIGHT = 720  # 后台回放导出分辨率高
# ============================================= 相机 ============================================= 

SIM_TIMESTEP = 1 / 240.0  # 物理仿真步长 (s)


# ============================================= 颗粒 ============================================= 
PARTICLE_COUNT = 5000  # 颗粒总数
PARTICLE_RADIUS = 0.007  # 颗粒半径 (m)
PARTICLE_RENDER_ENABLED = True  # 是否渲染颗粒（仅影响显示，不影响物理）

SAND_STATIC_FRICTION = 1.15   # 颗粒静摩擦系数（降低卡死与瞬时冲击）
SAND_DYNAMIC_FRICTION = 0.95  # 颗粒动摩擦系数（降低铲入时“顶飞”）
SAND_RESTITUTION = 0.0  # 颗粒弹性系数（0=不回弹）
SAND_DENSITY = 1700.0  # 颗粒密度 (kg/m^3)
SAND_LINEAR_DAMPING = 2.50   # 颗粒线速度阻尼
SAND_ANGULAR_DAMPING = 5.0  # 颗粒角速度阻尼（减少翻滚抛射）
SAND_SOLVER_POS_ITERS = 25  # 颗粒接触求解位置迭代次数（提高稳定性）
SAND_SOLVER_VEL_ITERS = 10  # 颗粒接触求解速度迭代次数
SAND_MAX_DEPENETRATION_VEL = 0.10  # 最大去穿透速度限制（防止接触瞬间弹飞）
SAND_MAX_LINEAR_VEL = 1.50  # 最大线速度限制（硬限速）
SAND_MAX_ANGULAR_VEL = 2.0  # 最大角速度限制（硬限速）
SAND_MAX_CONTACT_IMPULSE = 0.05  # 单次接触冲量上限（进一步抑制“弹飞”）

# ============================================= 颗粒 ============================================= 


# ============================================= 机械臂 ============================================= 
TOOL_STATIC_FRICTION = 1.10  # 机械臂/铲斗与环境静摩擦系数（降低刃口推挤冲击）
TOOL_DYNAMIC_FRICTION = 0.90  # 机械臂/铲斗与环境动摩擦系数
TOOL_RESTITUTION = 0.0  # 机械臂/铲斗回弹系数
TOOL_SOLVER_POS_ITERS = 24  # 机械臂接触求解位置迭代次数
TOOL_SOLVER_VEL_ITERS = 8  # 机械臂接触求解速度迭代次数
TOOL_MAX_DEPENETRATION_VEL = 0.25  # 机械臂去穿透速度限制（降低反冲）

JOINT_DRIVE_STIFFNESS = 900.0  # 关节驱动刚度（CPU 驱动模式）
JOINT_DRIVE_DAMPING = 120.0  # 关节驱动阻尼（CPU 驱动模式）
JOINT_DRIVE_FORCE_LIMIT = 4_000.0  # 关节驱动力上限（CPU 驱动模式）
# ============================================= 机械臂 ============================================= 


# ============================================= 计算 ============================================= 
GPU_MAX_RIGID_CONTACT_COUNT = 1_200_000  # GPU 接触点缓冲上限（高密颗粒场景）
GPU_MAX_RIGID_PATCH_COUNT = 240_000  # GPU 接触 patch 缓冲上限
# ============================================= 计算 ============================================= 


SCRIPTED_TIME_SCALE = 1.0  # 关键帧时间缩放（>1 慢放，<1 快放）
SCRIPTED_LOOP = False  # 是否循环播放关键帧
KEYFRAME_REPLAY_APPLY_MODE = "direct"  # 关键帧回放应用方式：direct=直接写 qpos，保证角度可达
KEYFRAME_DIRECT_MAX_DELTA_RAD = 0.020  # direct 回放单步最大关节变化（抑制“瞬移铲斗”导致飞粒）
BUCKET_ONLY_PARTICLE_COLLISION = True  # 铲斗仅与粒子碰撞（不与地面/池壁/车体碰撞）
BUCKET_DISABLE_MESH_COLLISION = True  # 关闭铲斗原始 mesh 碰撞（避免凸包封口，保留 box 内腔碰撞）
POOL_STATS_TOP_MARGIN = 0.30  # 池内统计时，墙顶向上额外容忍高度（用于容纳堆积）
POOL_STATS_PRINT_KEY = "p"  # 运行时按该键打印池内质量/体积与一致性对比

# ============================================= 回放前颗粒稳定等待 =============================================
SETTLE_PARTICLES_BEFORE_REPLAY = True  # 回放前先等待颗粒稳定，避免“落砂过程”干扰回放
SETTLE_MIN_STEPS = 120  # 最少先走这么多步再开始判稳（约 0.5 秒）
SETTLE_MAX_STEPS = 2400  # 最多等待步数（避免极端情况下卡住）
SETTLE_STABLE_WINDOW_STEPS = 60  # 需要连续满足阈值的步数
SETTLE_MEAN_SPEED_THRESHOLD = 0.030  # 判稳阈值：颗粒平均线速度上限 (m/s)
SETTLE_MAX_SPEED_THRESHOLD = 0.120  # 判稳阈值：颗粒最大线速度上限 (m/s)
# ============================================= 回放前颗粒稳定等待 =============================================


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

    repo_root = REPO_ROOT
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
    base_color: tuple[float, float, float, float] = SOURCE_POOL_BASE_COLOR,
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


def get_particle_linear_velocities(particles: list[sapien.Entity]) -> np.ndarray:
    """Return particle linear velocities as (N, 3) array."""
    if len(particles) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    vel = np.zeros((len(particles), 3), dtype=np.float32)
    for i, particle in enumerate(particles):
        rigid = particle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rigid is None:
            continue
        try:
            v = rigid.linear_velocity
        except Exception:  # noqa: BLE001
            v = rigid.get_linear_velocity()
        vel[i] = np.asarray(v, dtype=np.float32).reshape(3)
    return vel


def compute_particle_speed_stats(particles: list[sapien.Entity]) -> dict[str, float]:
    """Compute particle speed summary for settling checks."""
    vel = get_particle_linear_velocities(particles)
    if vel.shape[0] == 0:
        return {"mean_speed": 0.0, "max_speed": 0.0, "p95_speed": 0.0}

    speed = np.linalg.norm(vel, axis=1)
    return {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "p95_speed": float(np.percentile(speed, 95.0)),
    }


def settle_particles_before_replay(
    scene: sapien.Scene,
    particles: list[sapien.Entity],
    min_steps: int = SETTLE_MIN_STEPS,
    max_steps: int = SETTLE_MAX_STEPS,
    stable_window_steps: int = SETTLE_STABLE_WINDOW_STEPS,
    mean_speed_threshold: float = SETTLE_MEAN_SPEED_THRESHOLD,
    max_speed_threshold: float = SETTLE_MAX_SPEED_THRESHOLD,
) -> None:
    """Step physics until particle speeds become stable (or timeout)."""
    if len(particles) == 0:
        return

    min_steps = max(0, int(min_steps))
    max_steps = max(min_steps, int(max_steps))
    stable_window_steps = max(1, int(stable_window_steps))
    mean_speed_threshold = float(mean_speed_threshold)
    max_speed_threshold = float(max_speed_threshold)

    physx_system = scene.physx_system
    is_gpu_backend = isinstance(physx_system, sapien.physx.PhysxGpuSystem)
    start_ts = time.perf_counter()
    progress_interval = max(1, int(np.ceil(max_steps / 20.0)))

    stable_count = 0
    final_stats = compute_particle_speed_stats(particles)
    for step_idx in range(1, max_steps + 1):
        scene.step()
        if is_gpu_backend:
            physx_system.sync_poses_gpu_to_cpu()

        final_stats = compute_particle_speed_stats(particles)
        if step_idx < min_steps:
            continue

        is_stable_now = (
            final_stats["mean_speed"] <= mean_speed_threshold
            and final_stats["max_speed"] <= max_speed_threshold
        )
        stable_count = (stable_count + 1) if is_stable_now else 0
        if (step_idx % progress_interval) == 0 or step_idx == min_steps:
            elapsed = max(1e-6, time.perf_counter() - start_ts)
            pct = 100.0 * float(step_idx) / float(max_steps)
            print(
                "[Info] Settling particles: "
                f"step={step_idx}/{max_steps} ({pct:.1f}%), "
                f"mean={final_stats['mean_speed']:.5f}, max={final_stats['max_speed']:.5f}, "
                f"stable={stable_count}/{stable_window_steps}, elapsed={elapsed:.1f}s"
            )
        if stable_count >= stable_window_steps:
            elapsed = max(1e-6, time.perf_counter() - start_ts)
            print(
                "[Info] Particle settle done: "
                f"steps={step_idx}, mean_speed={final_stats['mean_speed']:.5f}, "
                f"max_speed={final_stats['max_speed']:.5f}, p95={final_stats['p95_speed']:.5f}, "
                f"elapsed={elapsed:.1f}s."
            )
            return

    elapsed = max(1e-6, time.perf_counter() - start_ts)
    print(
        "[Warn] Particle settle reached max steps without full convergence: "
        f"max_steps={max_steps}, mean_speed={final_stats['mean_speed']:.5f}, "
        f"max_speed={final_stats['max_speed']:.5f}, p95={final_stats['p95_speed']:.5f}, "
        f"elapsed={elapsed:.1f}s. "
        "Replay will still start."
    )


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


def apply_joint_target_direct(
    robot: sapien.physx.PhysxArticulation,
    target_qpos: np.ndarray,
    physx_system: sapien.physx.PhysxSystem,
) -> bool:
    """Directly set articulation qpos to guarantee replay reachability."""
    target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
    if target_qpos.shape[0] != robot.dof:
        raise ValueError(f"Target qpos length mismatch: got {target_qpos.shape[0]}, expected {robot.dof}")
    curr_q = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
    max_delta = float(max(1e-6, KEYFRAME_DIRECT_MAX_DELTA_RAD))
    target_qpos = curr_q + np.clip(target_qpos - curr_q, -max_delta, max_delta)
    try:
        robot.set_qpos(target_qpos)
        try:
            zero_q = np.zeros_like(target_qpos)
            robot.set_qvel(zero_q)
            robot.set_qf(zero_q)
        except Exception:  # noqa: BLE001
            pass
        if isinstance(physx_system, sapien.physx.PhysxGpuSystem):
            physx_system.gpu_apply_articulation_qpos()
        for link in robot.links:
            link.wake_up()
        return True
    except Exception:  # noqa: BLE001
        return False


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


def set_robot_and_particles_collision_enabled(
    robot: sapien.physx.PhysxArticulation,
    particles: list[sapien.Entity],
    enabled: bool,
) -> None:
    """Unified collision switch for robot + particles in scene mode."""
    if enabled:
        print("[Info] Collision: ON (robot + particles).")
        return

    changed_shapes = 0
    for link in robot.links:
        for shape in link.collision_shapes:
            shape.set_collision_groups([0, 0, 0, 0])
            changed_shapes += 1
    for particle in particles:
        rigid = particle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rigid is None:
            continue
        for shape in rigid.collision_shapes:
            shape.set_collision_groups([0, 0, 0, 0])
            changed_shapes += 1
    print(f"[Info] Collision: OFF (robot + particles), disabled_shapes={changed_shapes}.")


def configure_bucket_particle_only_collision(
    robot: sapien.physx.PhysxArticulation,
    particles: list[sapien.Entity],
    enabled: bool = BUCKET_ONLY_PARTICLE_COLLISION,
) -> None:
    """Configure collision filtering so bucket only collides with particles."""
    if not enabled:
        return

    default_layer = 1
    bucket_layer = 1 << 1
    particle_layer = 1 << 2
    bucket_groups = [bucket_layer, particle_layer, 0, 0]
    particle_groups = [default_layer | particle_layer, default_layer | bucket_layer, 0, 0]

    bucket_links = [link for link in robot.links if "bucket" in link.name.lower()]
    if not bucket_links:
        ee_idx, _ = resolve_ee_link(robot, preferred_name=None)
        bucket_links = [robot.links[ee_idx]]

    bucket_shape_count = 0
    bucket_mesh_disabled = 0
    for link in bucket_links:
        for shape in link.collision_shapes:
            shape_name = type(shape).__name__.lower()
            is_mesh_shape = "mesh" in shape_name
            if BUCKET_DISABLE_MESH_COLLISION and is_mesh_shape:
                shape.set_collision_groups([0, 0, 0, 0])
                bucket_mesh_disabled += 1
                continue
            shape.set_collision_groups(bucket_groups)
            bucket_shape_count += 1

    particle_shape_count = 0
    for particle in particles:
        rigid = particle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rigid is None:
            continue
        for shape in rigid.collision_shapes:
            shape.set_collision_groups(particle_groups)
            particle_shape_count += 1

    print(
        "[Info] Collision filter: bucket only interacts with particles "
        f"(bucket_shapes={bucket_shape_count}, bucket_mesh_disabled={bucket_mesh_disabled}, "
        f"particle_shapes={particle_shape_count})."
    )


def create_bucket_collision_debug_visuals(
    scene: sapien.Scene,
    robot: sapien.physx.PhysxArticulation,
) -> list[tuple[sapien.Entity, Any, sapien.Pose]]:
    """Create translucent visuals for bucket box-collision shapes."""
    bucket_links = [link for link in robot.links if "bucket" in link.name.lower()]
    if not bucket_links:
        ee_idx, _ = resolve_ee_link(robot, preferred_name=None)
        bucket_links = [robot.links[ee_idx]]

    colors = (
        [0.95, 0.20, 0.20, 0.35],  # 红
        [0.95, 0.85, 0.20, 0.35],  # 黄
        [0.20, 0.45, 0.95, 0.35],  # 蓝
        [0.20, 0.85, 0.30, 0.35],  # 绿
        [0.95, 0.95, 0.95, 0.35],  # 白
    )
    visuals: list[tuple[sapien.Entity, Any, sapien.Pose]] = []
    color_idx = 0

    for link in bucket_links:
        for shape in link.collision_shapes:
            if not isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
                continue
            half_size = np.asarray(shape.half_size, dtype=np.float32).reshape(3).tolist()
            local_pose = shape.local_pose
            material = sapien.render.RenderMaterial(
                base_color=colors[color_idx % len(colors)],
                roughness=0.2,
                specular=0.0,
                metallic=0.0,
            )
            color_idx += 1

            builder = scene.create_actor_builder()
            builder.add_box_visual(
                pose=sapien.Pose(),
                half_size=half_size,
                material=material,
                name=f"{link.name}_collision_debug_box",
            )
            actor = builder.build_kinematic(name=f"{link.name}_collision_debug_box_actor")
            actor.set_pose(link.entity_pose * local_pose)
            visuals.append((actor, link, local_pose))

    print(f"[Info] Bucket collision debug visuals enabled: count={len(visuals)}")
    return visuals


def update_bucket_collision_debug_visuals(
    visuals: list[tuple[sapien.Entity, Any, sapien.Pose]],
) -> None:
    """Update debug box actor poses to follow moving bucket link."""
    for actor, link, local_pose in visuals:
        actor.set_pose(link.entity_pose * local_pose)


def build_joint_policy_from_json(
    json_path: str | Path,
    dof: int,
    qlimits: np.ndarray | None = None,
    time_scale: float = SCRIPTED_TIME_SCALE,
    loop: bool = SCRIPTED_LOOP,
) -> LinearJointKeyframePolicy:
    """Load q/qpos keyframe JSON and build a joint replay policy."""
    joint_frames = load_joint_keyframes_json(
        json_path=json_path,
        dof=dof,
        qlimits=qlimits,
    )
    return LinearJointKeyframePolicy(
        keyframes=joint_frames,
        dof=dof,
        qlimits=qlimits,
        time_scale=float(time_scale),
        loop=bool(loop),
    )


def run_viewer(
    scene: sapien.Scene,
    camera_xyz: tuple[float, float, float] = CAMERA_XYZ,
    camera_rpy: tuple[float, float, float] = CAMERA_RPY,
    start_paused: bool = True,
    robot: sapien.physx.PhysxArticulation | None = None,
    joint_scripted_policy: LinearJointKeyframePolicy | None = None,
    keyframe_replay_apply_mode: str = KEYFRAME_REPLAY_APPLY_MODE,
    bucket_collision_debug_visuals: list[tuple[sapien.Entity, Any, sapien.Pose]] | None = None,
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
        if bucket_collision_debug_visuals:
            update_bucket_collision_debug_visuals(bucket_collision_debug_visuals)
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
            if robot is not None and joint_scripted_policy is not None:
                q_target = joint_scripted_policy.query(sim_step_index)
                if keyframe_replay_apply_mode == "direct":
                    ok = apply_joint_target_direct(robot=robot, target_qpos=q_target, physx_system=physx_system)
                else:
                    ok = apply_joint_target(robot=robot, target_qpos=q_target, physx_system=physx_system)
                if (not ok) and (not warned_control_failure):
                    print(
                        "[Warn] Joint keyframe control failed on current backend. "
                        "Try `--cpu` for deterministic replay."
                    )
                    warned_control_failure = True
            scene.step()
            if is_gpu_backend:
                # GPU PhysX needs explicit pose sync for viewport updates.
                physx_system.sync_poses_gpu_to_cpu()
            sim_step_index += 1
        scene.update_render()
        viewer.render()


def _capture_camera_rgb_uint8(camera: Any) -> np.ndarray:
    """Capture one RGB frame from a render camera and convert to uint8."""
    camera.take_picture()
    color = camera.get_picture("Color")
    rgb = np.clip(np.asarray(color[..., :3], dtype=np.float32), 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def _viewer_rpy_to_quaternion(rpy: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Convert Viewer-style (roll, pitch, yaw) to camera quaternion.

    This matches SAPIEN FPSCameraController update() convention:
    q = qmult(qmult(aa(up, -yaw), aa(left, -pitch)), aa(forward, roll))
    """
    r, p, y = [float(x) for x in np.asarray(rpy, dtype=np.float64).reshape(3)]
    if _axangle2quat is None or _qmult is None:
        # Fallback: keep previous behavior when transforms3d is unavailable.
        pose = sapien.Pose()
        pose.set_rpy([r, p, y])
        return np.asarray(pose.q, dtype=np.float32).reshape(4)

    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    left = np.cross(up, forward)
    q = _qmult(_qmult(_axangle2quat(up, -y), _axangle2quat(left, -p)), _axangle2quat(forward, r))
    return np.asarray(q, dtype=np.float32).reshape(4)


def run_headless_keyframe_replay(
    scene: sapien.Scene,
    robot: sapien.physx.PhysxArticulation,
    joint_scripted_policy: LinearJointKeyframePolicy,
    keyframe_replay_apply_mode: str,
    output_dir: Path,
    frame_interval_steps: int = 1,
    video_fps: float = 30.0,
    camera_xyz: tuple[float, float, float] = CAMERA_XYZ,
    camera_rpy: tuple[float, float, float] = HEADLESS_CAMERA_RPY,
    camera_width: int = HEADLESS_CAMERA_WIDTH,
    camera_height: int = HEADLESS_CAMERA_HEIGHT,
    total_steps: int | None = None,
) -> None:
    """Replay keyframes without UI and export frames + video."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for headless replay export.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_interval_steps = max(1, int(frame_interval_steps))
    video_fps = max(1e-6, float(video_fps))
    camera_width = max(64, int(camera_width))
    camera_height = max(64, int(camera_height))

    camera = scene.add_camera(
        name="headless_replay_camera",
        width=camera_width,
        height=camera_height,
        fovy=np.deg2rad(58.0),
        near=0.01,
        far=30.0,
    )
    cam_q = _viewer_rpy_to_quaternion(camera_rpy)
    cam_pose = sapien.Pose(p=list(camera_xyz), q=cam_q.tolist())
    camera.set_entity_pose(cam_pose)

    physx_system = scene.physx_system
    is_gpu_backend = isinstance(physx_system, sapien.physx.PhysxGpuSystem)

    if total_steps is None:
        total_steps = max(1, int(np.ceil(joint_scripted_policy.period)) + 1)
    else:
        total_steps = max(1, int(total_steps))
    print(
        "[Info] Headless replay started: "
        f"steps={total_steps}, frame_interval={frame_interval_steps}, "
        f"size={camera_width}x{camera_height}, fps={video_fps:.3f}"
    )
    start_ts = time.perf_counter()
    progress_interval = max(1, int(np.ceil(total_steps / 20.0)))

    video_path = output_dir / "replay.mp4"
    frame_count = 0
    sim_step_index = 0
    warned_control_failure = False
    writer: cv2.VideoWriter | None = None

    def export_frame(step_idx: int) -> None:
        nonlocal frame_count, writer
        scene.update_render()
        rgb = _capture_camera_rgb_uint8(camera)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_path = output_dir / f"step_{step_idx:06d}.png"
        ok = cv2.imwrite(str(frame_path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to save frame: {frame_path}")
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (bgr.shape[1], bgr.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
        writer.write(bgr)
        frame_count += 1

    try:
        export_frame(step_idx=0)
        for sim_step_index in range(1, total_steps + 1):
            q_target = joint_scripted_policy.query(sim_step_index)
            if keyframe_replay_apply_mode == "direct":
                ok = apply_joint_target_direct(robot=robot, target_qpos=q_target, physx_system=physx_system)
            else:
                ok = apply_joint_target(robot=robot, target_qpos=q_target, physx_system=physx_system)
            if (not ok) and (not warned_control_failure):
                print(
                    "[Warn] Joint keyframe control failed on current backend during headless replay. "
                    "Try `--cpu` for deterministic replay."
                )
                warned_control_failure = True

            scene.step()
            if is_gpu_backend:
                physx_system.sync_poses_gpu_to_cpu()

            if (sim_step_index % frame_interval_steps) == 0 or sim_step_index == total_steps:
                export_frame(step_idx=sim_step_index)
            if (sim_step_index % progress_interval) == 0 or sim_step_index == total_steps:
                elapsed = max(1e-6, time.perf_counter() - start_ts)
                pct = 100.0 * float(sim_step_index) / float(total_steps)
                step_rate = float(sim_step_index) / elapsed
                eta = max(0.0, float(total_steps - sim_step_index) / max(1e-6, step_rate))
                print(
                    "[Info] Headless replay progress: "
                    f"step={sim_step_index}/{total_steps} ({pct:.1f}%), "
                    f"frames={frame_count}, elapsed={elapsed:.1f}s, eta={eta:.1f}s"
                )
    finally:
        if writer is not None:
            writer.release()

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_steps": int(total_steps),
        "exported_frames": int(frame_count),
        "frame_interval_steps": int(frame_interval_steps),
        "video_fps": float(video_fps),
        "camera_xyz": [float(x) for x in camera_xyz],
        "camera_rpy": [float(x) for x in camera_rpy],
        "camera_size": [int(camera_width), int(camera_height)],
        "replay_apply_mode": str(keyframe_replay_apply_mode),
        "video_path": str(video_path),
    }
    (output_dir / "replay_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Info] Headless replay finished: frames={frame_count}, video={video_path}")


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
        self.ee_link_index: int | None = None

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
            base_color=SOURCE_POOL_BASE_COLOR,
            name="source_particle_pool",
        )
        build_particle_pool(
            self.scene,
            center=RECEIVER_POOL_CENTER,
            inner_half_size=RECEIVER_POOL_INNER_HALF_SIZE,
            wall_height=RECEIVER_POOL_WALL_HEIGHT,
            wall_thickness=POOL_WALL_THICKNESS,
            bottom_thickness=POOL_BOTTOM_THICKNESS,
            base_color=RECEIVER_POOL_BASE_COLOR,
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
        configure_bucket_particle_only_collision(self.robot, self.particles)
        # Always configure joint drives so external joint_pos action can be directly applied.
        configure_joint_drives(self.robot)
        maybe_init_gpu_physx(self.scene)

        self.ee_link_index = None
        self.scene.update_render()
        return self.get_observation()

    def _ensure_ready(self) -> None:
        if self.scene is None or self.robot is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

    def apply_action(self, action: np.ndarray | list[float] | dict[str, Any]) -> None:
        """Apply external action without stepping.

        Supported:
        - joint action: np.ndarray/list with shape (dof,) or {'joint_pos': ...}
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
            raise ValueError("Unsupported action dict. Use {'joint_pos': ...}.")

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
        help="运行模式：manual=手动调关节，keyframe=按关节关键帧自动运动。",
    )
    parser.add_argument(
        "--keyframes-json",
        type=str,
        default=None,
        help="关节关键帧 JSON 路径（支持 q 或 qpos 字段），仅在 mode=keyframe 时使用。",
    )

    # ==================== debug ==========================
    parser.add_argument(
        "--show-bucket-collision-boxes",
        action="store_true",
        help="显示铲斗 box 碰撞箱（半透明），用于手动调试碰撞箱大小和位置。",
    )
    parser.add_argument(
        "--collision",
        type=str,
        choices=("on", "off"),
        default="on",
        help="统一设置回放时粒子与挖掘机碰撞：on=开启，off=关闭。",
    )
    parser.add_argument(
        "--bucket-collision-mode",
        type=str,
        choices=("particle-only", "all"),
        default="particle-only",
        help="铲斗碰撞模式：particle-only=仅与粒子碰撞，all=与所有物体碰撞。",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=SCRIPTED_TIME_SCALE,
        help="关键帧时间缩放：>1 慢放，<1 快放。",
    )
    parser.add_argument(
        "--replay-apply-mode",
        type=str,
        choices=("direct", "drive"),
        default=KEYFRAME_REPLAY_APPLY_MODE,
        help="回放关节应用方式：direct=每步直接写qpos，drive=关节驱动跟踪（更物理）。",
    )
    parser.add_argument(
        "--headless-replay",
        action="store_true",
        help="后台回放（不打开UI），导出图片序列并自动生成视频（仅支持 mode=keyframe）。",
    )
    parser.add_argument(
        "--frame-interval-steps",
        type=int,
        default=8,
        help="后台回放导出图片间隔（单位：仿真步）。例如 10 表示每 10 step 导出 1 帧。",
    )
    parser.add_argument(
        "--headless-output-dir",
        type=str,
        default=None,
        help="后台回放输出目录。默认 outputs/headless_replay/<时间戳>/",
    )
    parser.add_argument(
        "--headless-video-fps",
        type=float,
        default=30.0,
        help="后台回放输出视频帧率。",
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
        base_color=SOURCE_POOL_BASE_COLOR,
        name="source_particle_pool",
    )
    build_particle_pool(
        scene,
        center=RECEIVER_POOL_CENTER,
        inner_half_size=RECEIVER_POOL_INNER_HALF_SIZE,
        wall_height=RECEIVER_POOL_WALL_HEIGHT,
        wall_thickness=POOL_WALL_THICKNESS,
        bottom_thickness=POOL_BOTTOM_THICKNESS,
        base_color=RECEIVER_POOL_BASE_COLOR,
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
    set_robot_and_particles_collision_enabled(robot, particles, enabled=(args.collision == "on"))
    if args.collision == "on":
        configure_bucket_particle_only_collision(
            robot,
            particles,
            enabled=(args.bucket_collision_mode == "particle-only"),
        )
    bucket_collision_debug_visuals = (
        create_bucket_collision_debug_visuals(scene, robot)
        if args.show_bucket_collision_boxes
        else None
    )

    joint_scripted_policy: LinearJointKeyframePolicy | None = None

    if args.mode == "manual":
        print("[Info] Manual mode: 启动后默认暂停，可在 Viewer 中手动调节关节并控制开始/暂停。")
    elif args.mode == "keyframe":
        if args.replay_apply_mode != "direct" and not isinstance(scene.physx_system, sapien.physx.PhysxGpuSystem):
            configure_joint_drives(robot)
            print(
                "[Info] Keyframe control enabled (CPU drive target mode): "
                f"stiffness={JOINT_DRIVE_STIFFNESS}, damping={JOINT_DRIVE_DAMPING}, "
                f"force_limit={JOINT_DRIVE_FORCE_LIMIT}"
            )
        elif args.replay_apply_mode == "direct":
            print("[Info] Keyframe replay mode: direct qpos (always reach target angles).")
        else:
            print("[Info] GPU keyframe replay: using direct set_qpos + gpu_apply_articulation_qpos.")
        print(f"[Info] Keyframe time scale: {float(args.time_scale):.3f}")

        qlimits = np.asarray(robot.get_qlimits(), dtype=np.float32).reshape(-1, 2)
        if args.keyframes_json:
            keyframes_path = Path(args.keyframes_json).expanduser().resolve()
            joint_scripted_policy = build_joint_policy_from_json(
                json_path=args.keyframes_json,
                dof=robot.dof,
                qlimits=qlimits,
                time_scale=float(args.time_scale),
                loop=SCRIPTED_LOOP,
            )
            print(f"[Info] Loaded joint keyframes: {keyframes_path}")
        else:
            default_keyframes = build_default_excavator_keyframes(
                init_qpos=np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1),
                qlimits=qlimits,
            )
            joint_scripted_policy = LinearJointKeyframePolicy(
                keyframes=default_keyframes,
                dof=robot.dof,
                qlimits=qlimits,
                time_scale=float(args.time_scale),
                loop=SCRIPTED_LOOP,
            )
            print("[Info] Using built-in default joint keyframes.")

    maybe_init_gpu_physx(scene)
    if args.mode == "keyframe" and SETTLE_PARTICLES_BEFORE_REPLAY:
        print(
            "[Info] Waiting particles to settle before keyframe replay... "
            f"(min_steps={SETTLE_MIN_STEPS}, max_steps={SETTLE_MAX_STEPS})"
        )
        settle_particles_before_replay(
            scene=scene,
            particles=particles,
            min_steps=SETTLE_MIN_STEPS,
            max_steps=SETTLE_MAX_STEPS,
            stable_window_steps=SETTLE_STABLE_WINDOW_STEPS,
            mean_speed_threshold=SETTLE_MEAN_SPEED_THRESHOLD,
            max_speed_threshold=SETTLE_MAX_SPEED_THRESHOLD,
        )

    if args.headless_replay:
        if args.mode != "keyframe" or joint_scripted_policy is None:
            raise ValueError("--headless-replay 仅支持 mode=keyframe 且需要可用 keyframe policy。")
        if args.headless_output_dir:
            output_dir = Path(args.headless_output_dir).expanduser().resolve()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (REPO_ROOT / "outputs" / "headless_replay" / ts).resolve()
        total_steps = int(np.ceil(float(joint_scripted_policy.period) * float(args.time_scale))) + 1
        run_headless_keyframe_replay(
            scene=scene,
            robot=robot,
            joint_scripted_policy=joint_scripted_policy,
            keyframe_replay_apply_mode=args.replay_apply_mode,
            output_dir=output_dir,
            frame_interval_steps=max(1, int(args.frame_interval_steps)),
            video_fps=float(args.headless_video_fps),
            camera_xyz=CAMERA_XYZ,
            camera_rpy=HEADLESS_CAMERA_RPY,
            total_steps=total_steps,
        )
        return

    run_viewer(
        scene,
        camera_xyz=CAMERA_XYZ,
        camera_rpy=CAMERA_RPY,
        start_paused=True,
        robot=robot,
        joint_scripted_policy=joint_scripted_policy,
        keyframe_replay_apply_mode=args.replay_apply_mode,
        bucket_collision_debug_visuals=bucket_collision_debug_visuals,
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
