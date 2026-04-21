"""
Minimal SAPIEN 3.0 robot-only environment for Fairino FR5.

Goals:
1) Load the robot URDF into a clean scene
2) Print joint names and limits for ACT alignment
3) Launch viewer for quick inspection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sapien
from sapien.utils import Viewer

# Initial joint angles from your screenshot (radians): j1..j6
DEFAULT_INIT_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)

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


def resolve_urdf_path(user_path: str | None) -> Path:
    """Resolve URDF path from CLI arg or common local defaults."""
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"URDF file not found: {p}")
        return p

    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "assets" / "fairino5" / "fairino5_v6.urdf",
        repo_root / "assets" / "fairino5_v6.urdf",
    ]
    for p in candidates:
        if p.is_file():
            return p

    raise FileNotFoundError(
        "No default Fairino URDF found. Tried:\n  - " + "\n  - ".join(str(p) for p in candidates)
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
    parser = argparse.ArgumentParser(description="Load Fairino robot into a minimal SAPIEN 3.0 scene.")
    parser.add_argument("--urdf", type=str, default=None, help="Path to URDF. Uses local default if omitted.")
    parser.add_argument("--timestep", type=float, default=1 / 240.0, help="Simulation timestep.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU PhysX.")
    args = parser.parse_args()

    urdf_path = resolve_urdf_path(args.urdf)
    print(f"[Info] Loading URDF: {urdf_path}")

    scene = create_scene(timestep=args.timestep, prefer_gpu=not args.cpu)
    configure_lighting(scene)
    robot = load_robot(scene, urdf_path)

    # Load your desired startup pose (j1..j6) and open viewer in paused mode.
    try:
        # 设置初始位姿
        set_initial_joint_pose(robot, DEFAULT_INIT_QPOS)
    except RuntimeError as exc:
        print(
            f"[Warn] Failed to set startup qpos ({exc}). "
            "Try running with --cpu for deterministic initial pose setup."
        )

    print_robot_diagnostics(robot)
    run_viewer(scene, start_paused=True)


if __name__ == "__main__":
    main()
