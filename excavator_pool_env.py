"""
Excavator + granular pool environment in SAPIEN 3.0.

Layout:
1) Right side: an elevated platform for a fixed excavator.
2) Left side: a square pool (bottom + 4 walls) filled with small dynamic particles.
3) Ground plane + lighting + interactive viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sapien
from sapien.utils import Viewer


SCENE_NAME = "excavator_pool_env"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.json"


def create_scene(timestep: float = 1 / 240.0, prefer_gpu: bool = True) -> sapien.Scene:
    """Create a SAPIEN scene and prefer GPU PhysX when available."""
    if prefer_gpu:
        try:
            scene = sapien.Scene([sapien.physx.PhysxGpuSystem("cuda"), sapien.render.RenderSystem()])
            print("[Info] Using PhysX GPU system (cuda).")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] GPU PhysX unavailable, fallback to CPU PhysX: {exc}")
            scene = sapien.Scene()
    else:
        scene = sapien.Scene()

    scene.set_timestep(timestep)
    return scene


def configure_lighting(scene: sapien.Scene) -> None:
    """Set practical lighting for full-scene inspection."""
    scene.set_ambient_light([0.38, 0.38, 0.38])
    scene.add_directional_light(direction=[0.25, -0.35, -1.0], color=[0.95, 0.92, 0.90], shadow=True)
    scene.add_point_light(position=[0.0, 0.0, 2.8], color=[0.45, 0.45, 0.45], shadow=False)
    scene.add_ground(altitude=0.0, render_half_size=[7.0, 4.0])


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
    center: tuple[float, float] = (2.40, 0.0),
    half_size: tuple[float, float, float] = (0.90, 0.75, 0.20),
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
    center: tuple[float, float] = (-2.00, 0.0),
    inner_half_size: tuple[float, float] = (0.72, 0.72),
    wall_height: float = 0.36,
    wall_thickness: float = 0.04,
    bottom_thickness: float = 0.05,
) -> sapien.Entity:
    """Build a square pool (bottom + 4 walls) on the left side."""
    pool_material = scene.create_physical_material(static_friction=1.35, dynamic_friction=1.20, restitution=0.01)
    pool_visual = sapien.render.RenderMaterial(
        base_color=[0.46, 0.56, 0.64, 1.0],
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

    return builder.build_kinematic(name="particle_pool")


def spawn_particles(
    scene: sapien.Scene,
    center: tuple[float, float] = (-2.00, 0.0),
    inner_half_size: tuple[float, float] = (0.72, 0.72),
    particle_count: int = 2500,
    radius: float = 0.015,
) -> list[sapien.Entity]:
    """Spawn dynamic particles above the left pool and let them settle."""
    if particle_count <= 0:
        return []

    px, py = center
    ix, iy = inner_half_size
    margin = 0.07
    spacing = radius * 2.05

    nx = max(2, int((2 * (ix - margin)) / spacing))
    ny = max(2, int((2 * (iy - margin)) / spacing))
    nz = int(np.ceil(particle_count / float(nx * ny)))

    x_coords = px + (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * spacing
    y_coords = py + (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * spacing
    z_start = 0.65
    z_coords = z_start + np.arange(nz, dtype=np.float32) * spacing

    gx, gy, gz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    positions = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)[:particle_count]

    rng = np.random.default_rng(seed=7)
    positions += rng.uniform(
        low=-0.08 * radius,
        high=0.08 * radius,
        size=positions.shape,
    ).astype(np.float32)

    particle_material = scene.create_physical_material(static_friction=1.25, dynamic_friction=1.08, restitution=0.003)
    particle_visual = sapien.render.RenderMaterial(
        base_color=[0.76, 0.66, 0.42, 1.0],
        roughness=0.95,
        specular=0.06,
        metallic=0.0,
    )

    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius, material=particle_material, density=1550.0)
    builder.add_sphere_visual(radius=radius, material=particle_visual)

    particles: list[sapien.Entity] = []
    for pos in positions:
        builder.set_initial_pose(sapien.Pose(p=pos.tolist()))
        particle = builder.build()
        rigid = particle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        rigid.set_linear_damping(0.07)
        rigid.set_angular_damping(0.07)
        rigid.set_solver_position_iterations(8)
        rigid.set_solver_velocity_iterations(2)
        particles.append(particle)

    print(f"[Info] Spawned {len(particles)} particles.")
    return particles


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
    scale: float,
    init_qpos: np.ndarray | None,
) -> sapien.physx.PhysxArticulation:
    """Load articulation and place it on the right platform."""
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = float(scale)

    robot = loader.load(str(urdf_path), package_dir=str(urdf_path.parent))
    if robot is None:
        raise RuntimeError(f"Failed to load excavator: {urdf_path}")

    root_pose = sapien.Pose(p=[platform_center[0], platform_center[1], 2.0 * platform_half_height + 0.015])
    root_pose.set_rpy([0.0, 0.0, np.pi])
    robot.set_root_pose(root_pose)

    if init_qpos is not None:
        try:
            set_initial_joint_pose(robot, init_qpos)
        except (RuntimeError, ValueError) as exc:
            print(
                f"[Warn] Failed to set startup qpos ({exc}). "
                "Try running with --cpu for deterministic initial pose setup."
            )
    else:
        print(f"[Info] No initial pose configured for model={equipment_model}, skip set_qpos.")

    print(f"[Info] Loaded excavator model={equipment_model} (dof={robot.dof}, scale={scale:.3f})")
    return robot


def run_viewer(scene: sapien.Scene) -> None:
    """Run the real-time viewer loop."""
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=0.20, y=-6.00, z=2.25)
    viewer.set_camera_rpy(r=0.0, p=-0.24, y=0.0)
    viewer.window.set_camera_parameters(near=0.01, far=30.0, fovy=np.deg2rad(58.0))

    print("[Info] Viewer started. Close window to exit.")
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


def main() -> None:
    bootstrap_config_path = extract_config_path_from_argv(sys.argv[1:], DEFAULT_CONFIG_PATH)
    bootstrap_config = load_pose_config(bootstrap_config_path)
    model_choices = get_available_excavator_models_from_config(bootstrap_config)
    default_model = "excavator_simple" if "excavator_simple" in model_choices else model_choices[0]

    parser = argparse.ArgumentParser(description="Excavator + left particle pool environment in SAPIEN 3.0.")
    parser.add_argument(
        "equipment_model",
        nargs="?",
        default=default_model,
        choices=model_choices,
        help="Excavator model preset.",
    )
    parser.add_argument("--urdf", type=str, default=None, help="Optional custom excavator URDF path.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(bootstrap_config_path),
        help="Path to config.json that stores urdf candidates and per-model, per-scene initial poses.",
    )
    parser.add_argument(
        "--excavator-scale",
        type=float,
        default=0.32,
        help="Global URDF scale for excavator. Reduce if excavator is much larger than pool.",
    )
    parser.add_argument("--particle-count", type=int, default=2500, help="Number of dynamic particles in the pool.")
    parser.add_argument("--particle-radius", type=float, default=0.015, help="Particle radius in meters.")
    parser.add_argument("--timestep", type=float, default=1 / 240.0, help="Physics timestep.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU PhysX.")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    pose_config = load_pose_config(config_path)

    scene = create_scene(timestep=args.timestep, prefer_gpu=not args.cpu)
    configure_lighting(scene)

    platform_center = (1.35, 0.0)
    platform_half_size = (0.75, 0.65, 0.16)
    build_platform(scene, center=platform_center, half_size=platform_half_size)

    pool_center = (-0.40, 0.0)
    pool_inner_half_size = (0.85, 0.85)
    build_particle_pool(scene, center=pool_center, inner_half_size=pool_inner_half_size)
    spawn_particles(
        scene,
        center=pool_center,
        inner_half_size=pool_inner_half_size,
        particle_count=args.particle_count,
        radius=args.particle_radius,
    )

    urdf_path = resolve_excavator_urdf_path(pose_config, args.equipment_model, args.urdf)
    init_qpos = get_initial_qpos_from_config(pose_config, args.equipment_model, SCENE_NAME)
    print(f"[Info] Loading excavator URDF: {urdf_path}")
    print(f"[Info] Loading pose config: {config_path}")
    load_excavator(
        scene=scene,
        equipment_model=args.equipment_model,
        urdf_path=urdf_path,
        platform_center=platform_center,
        platform_half_height=platform_half_size[2],
        scale=args.excavator_scale,
        init_qpos=init_qpos,
    )

    run_viewer(scene)


if __name__ == "__main__":
    main()
