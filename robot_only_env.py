"""
Minimal SAPIEN 3.0 robot-only environment.

Goals:
1) Load the robot URDF into a clean scene
2) Print joint names and limits for ACT alignment
3) Launch viewer for quick inspection
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

SCENE_NAME = "robot_only_env"
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
    """Set simple lighting for robot inspection."""
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light(direction=[0.2, 0.5, -1.0], color=[0.8, 0.8, 0.8], shadow=True)
    scene.add_point_light(position=[1.0, -1.0, 1.5], color=[0.35, 0.35, 0.35], shadow=False)
    scene.add_ground(altitude=0.0, render_half_size=[1.2, 1.2])


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


def resolve_urdf_path(config: dict[str, Any], equipment_model: str, user_path: str | None) -> Path:
    """Resolve URDF path from equipment_model, unless explicitly overridden."""
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"URDF file not found: {p}")
        return p

    repo_root = Path(__file__).resolve().parent
    candidates = [(repo_root / rel).resolve() for rel in get_urdf_candidates_from_config(config, equipment_model)]

    for p in candidates:
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"No default URDF found for equipment_model={equipment_model}. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
    )


def load_robot(scene: sapien.Scene, urdf_path: Path) -> sapien.physx.PhysxArticulation:
    """Load robot from URDF with fixed base."""
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # package_dir helps resolve package:// style URDF resources.
    robot = loader.load(str(urdf_path), package_dir=str(urdf_path.parent))
    return robot


def print_robot_diagnostics(robot: sapien.physx.PhysxArticulation) -> None:
    """Print articulation details for ACT/SAPIEN joint mapping checks."""
    print("\n========== Robot Diagnostics ==========")
    print(f"Articulation name: {robot.name}")
    print(f"DOF: {robot.dof}")
    print(f"Link count: {len(robot.links)}")
    print(f"Joint count (all): {len(robot.joints)}")
    print(f"Joint count (active): {len(robot.active_joints)}")

    if robot.dof > 0:
        qpos = np.asarray(robot.get_qpos(), dtype=np.float64).reshape(-1)
        qlimits = np.asarray(robot.get_qlimits(), dtype=np.float64).reshape(-1, 2)
        print(f"Initial qpos: {np.array2string(qpos, precision=5, suppress_small=True)}")
        print("qlimits from articulation (active joints order):")
        for i, (low, high) in enumerate(qlimits):
            print(f"  dof[{i:02d}]: [{low:+.6f}, {high:+.6f}]")

    print("\nActive joints:")
    for i, joint in enumerate(robot.active_joints):
        limits = np.asarray(joint.get_limits(), dtype=np.float64).reshape(-1, 2)
        if len(limits) > 0:
            low, high = limits[0]
            limit_str = f"[{low:+.6f}, {high:+.6f}]"
        else:
            limit_str = "[]"
        print(f"  [{i:02d}] name={joint.name:20s} type={joint.type:18s} limit={limit_str}")
    print("=======================================\n")


def set_initial_joint_pose(robot: sapien.physx.PhysxArticulation, qpos: np.ndarray) -> None:
    """Apply initial articulation qpos before opening the viewer."""
    if robot.dof <= 0:
        return
    if robot.dof != len(qpos):
        raise ValueError(
            f"Initial qpos length mismatch: robot DOF={robot.dof}, provided={len(qpos)}"
        )
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


def get_available_models_from_config(config: dict[str, Any]) -> list[str]:
    """Return sorted model names that have URDF candidates in config."""
    model_map = config.get("urdf_candidates", {})
    if not isinstance(model_map, dict):
        raise ValueError("Invalid config: 'urdf_candidates' must be an object.")
    models = sorted(k for k, v in model_map.items() if isinstance(k, str) and isinstance(v, list) and len(v) > 0)
    if not models:
        raise ValueError("No available models found in config['urdf_candidates'].")
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


def run_viewer(scene: sapien.Scene, start_paused: bool = True) -> None:
    """Open viewer and keep simulation running."""
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.8, y=0.0, z=1.1)
    viewer.set_camera_rpy(r=0.0, p=-0.45, y=3.14)
    viewer.window.set_camera_parameters(near=0.01, far=20.0, fovy=np.deg2rad(60.0))
    viewer.paused = start_paused

    print(f"[Info] Viewer started. paused={viewer.paused}. Close the window to exit.")
    while not viewer.closed:
        if not viewer.paused:
            scene.step()
        scene.update_render()
        viewer.render()


def main() -> None:
    bootstrap_config_path = extract_config_path_from_argv(sys.argv[1:], DEFAULT_CONFIG_PATH)
    bootstrap_config = load_pose_config(bootstrap_config_path)
    model_choices = get_available_models_from_config(bootstrap_config)
    default_model = "fairino5_single" if "fairino5_single" in model_choices else model_choices[0]

    parser = argparse.ArgumentParser(description="Load selected equipment model into a minimal SAPIEN 3.0 scene.")
    parser.add_argument(
        "equipment_model",
        nargs="?",
        default=default_model,
        choices=model_choices,
        help="Equipment model preset. You can pass it directly as the first argument.",
    )
    parser.add_argument(
        "--equipment-model",
        type=str,
        dest="equipment_model_override",
        default=None,
        choices=model_choices,
        help="Optional override for equipment model preset.",
    )
    parser.add_argument("--urdf", type=str, default=None, help="Path to URDF. Uses local default if omitted.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(bootstrap_config_path),
        help="Path to config.json that stores urdf candidates and per-model, per-scene initial poses.",
    )
    parser.add_argument("--timestep", type=float, default=1 / 240.0, help="Simulation timestep.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU PhysX.")
    args = parser.parse_args()

    equipment_model = args.equipment_model_override or args.equipment_model
    config_path = Path(args.config).expanduser().resolve()
    pose_config = load_pose_config(config_path)
    urdf_path = resolve_urdf_path(pose_config, equipment_model, args.urdf)
    print(f"[Info] equipment_model={equipment_model}")
    print(f"[Info] Loading URDF: {urdf_path}")
    print(f"[Info] Loading pose config: {config_path}")

    scene = create_scene(timestep=args.timestep, prefer_gpu=not args.cpu)
    configure_lighting(scene)
    robot = load_robot(scene, urdf_path)

    init_qpos = get_initial_qpos_from_config(pose_config, equipment_model, SCENE_NAME)
    if init_qpos is not None:
        try:
            set_initial_joint_pose(robot, init_qpos)
        except (RuntimeError, ValueError) as exc:
            print(
                f"[Warn] Failed to set startup qpos ({exc}). "
                "Try running with --cpu for deterministic initial pose setup."
            )
    else:
        print(f"[Info] No initial pose configured for equipment_model={equipment_model}, skip set_qpos.")

    print_robot_diagnostics(robot)
    run_viewer(scene, start_paused=True)


if __name__ == "__main__":
    main()
