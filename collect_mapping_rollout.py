"""Replay keyframes and export mapping-ready rollout data (robot + particles)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sapien

from envs import excavator_pool as ep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "回放关节 keyframe，并按固定步长导出建图数据："
            "粒子位置/速度 + 挖掘机关节 + 末端位姿 + 池子统计。"
        )
    )
    parser.add_argument(
        "equipment_model",
        nargs="?",
        default="excavator_s010",
        help="挖掘机模型名（需在 config.json 的 urdf_candidates 中存在）。",
    )
    parser.add_argument("--config", type=str, default=str(ep.DEFAULT_CONFIG_PATH), help="配置文件路径（config.json）。")
    parser.add_argument("--keyframes-json", type=str, required=True, help="关节 keyframe JSON（q 或 qpos）。")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU PhysX。")
    parser.add_argument(
        "--collision",
        type=str,
        choices=("on", "off"),
        default="on",
        help="碰撞开关：on=按环境规则启用，off=关闭机器人+粒子碰撞。",
    )
    parser.add_argument(
        "--bucket-collision-mode",
        type=str,
        choices=("particle-only", "all"),
        default="particle-only",
        help="当 collision=on 时，铲斗碰撞模式。",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=ep.SCRIPTED_TIME_SCALE,
        help="关键帧时间缩放：>1 慢放，<1 快放。",
    )
    parser.add_argument(
        "--replay-apply-mode",
        type=str,
        choices=("direct", "drive"),
        default=ep.KEYFRAME_REPLAY_APPLY_MODE,
        help="回放应用方式：direct=直接写 qpos，drive=关节驱动跟踪。",
    )
    parser.add_argument(
        "--sample-interval-steps",
        type=int,
        default=10,
        help="每隔多少仿真步采样一次状态（最后一步会强制采样）。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="最大回放步数。默认按 keyframe period 和 time-scale 自动计算。",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=0,
        help="最多导出多少颗粒子（0 表示全部）。",
    )
    parser.add_argument(
        "--particle-sample-seed",
        type=int,
        default=7,
        help="当 max-particles>0 时用于随机抽样粒子的随机种子。",
    )
    parser.add_argument(
        "--settle-before-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回放前是否等待颗粒速度稳定。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 hdf5 路径（.hdf5）；若不传则写到 outputs/mapping_rollouts/<timestamp>/rollout_data.hdf5",
    )
    return parser.parse_args()


def _resolve_output_path(output_arg: str | None) -> Path:
    if output_arg is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (ep.REPO_ROOT / "outputs" / "mapping_rollouts" / ts).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "rollout_data.hdf5"

    out_path = Path(output_arg).expanduser().resolve()
    if out_path.suffix.lower() == ".hdf5":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    if out_path.suffix != "":
        raise ValueError(f"--output only supports .hdf5 file path or directory, got: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / "rollout_data.hdf5"


def _choose_particle_indices(total_count: int, max_particles: int, seed: int) -> np.ndarray:
    max_particles = int(max_particles)
    if max_particles <= 0 or max_particles >= total_count:
        return np.arange(total_count, dtype=np.int32)
    rng = np.random.default_rng(seed=int(seed))
    return np.sort(rng.choice(total_count, size=max_particles, replace=False).astype(np.int32))


def _collect_sample(
    robot: sapien.physx.PhysxArticulation,
    particles: list[sapien.Entity],
    tracked_particle_indices: np.ndarray,
    ee_link_index: int,
) -> dict[str, Any]:
    qpos = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
    qvel = np.asarray(robot.get_qvel(), dtype=np.float32).reshape(-1)
    ee_pose = robot.links[ee_link_index].entity_pose
    ee_xyz = np.asarray(ee_pose.p, dtype=np.float32).reshape(3)
    ee_rpy = np.asarray(ee_pose.rpy, dtype=np.float32).reshape(3)

    particle_positions_all = ep.get_particle_positions(particles)
    particle_velocities_all = ep.get_particle_linear_velocities(particles)
    particle_positions = particle_positions_all[tracked_particle_indices]
    particle_velocities = particle_velocities_all[tracked_particle_indices]

    source_stats = ep.compute_pool_particle_stats(
        particles=particles,
        center=ep.POOL_CENTER,
        inner_half_size=ep.POOL_INNER_HALF_SIZE,
        wall_height=ep.POOL_WALL_HEIGHT,
    )
    receiver_stats = ep.compute_pool_particle_stats(
        particles=particles,
        center=ep.RECEIVER_POOL_CENTER,
        inner_half_size=ep.RECEIVER_POOL_INNER_HALF_SIZE,
        wall_height=ep.RECEIVER_POOL_WALL_HEIGHT,
    )

    return {
        "qpos": qpos,
        "qvel": qvel,
        "ee_xyz": ee_xyz,
        "ee_rpy": ee_rpy,
        "particle_positions": particle_positions,
        "particle_velocities": particle_velocities,
        "source_count": int(source_stats["count"]),
        "source_mass_kg": float(source_stats["mass_kg"]),
        "source_volume_m3": float(source_stats["volume_m3"]),
        "receiver_count": int(receiver_stats["count"]),
        "receiver_mass_kg": float(receiver_stats["mass_kg"]),
        "receiver_volume_m3": float(receiver_stats["volume_m3"]),
    }


def main() -> None:
    args = _parse_args()
    output_hdf5 = _resolve_output_path(args.output)
    output_meta = output_hdf5.with_name(f"{output_hdf5.stem}_meta.json")

    world = ep.create_excavator_pool_world(
        equipment_model=args.equipment_model,
        config_path=args.config,
        prefer_gpu=not args.cpu,
        collision=args.collision,
        bucket_collision_mode=args.bucket_collision_mode,
        show_bucket_collision_boxes=False,
    )
    scene = world.scene
    robot = world.robot
    particles = world.particles
    config_path = world.config_path
    urdf_path = world.urdf_path

    if args.replay_apply_mode == "drive" and not isinstance(scene.physx_system, sapien.physx.PhysxGpuSystem):
        ep.configure_joint_drives(robot)
        print(
            "[Info] Drive replay mode (CPU): "
            f"stiffness={ep.JOINT_DRIVE_STIFFNESS}, damping={ep.JOINT_DRIVE_DAMPING}, "
            f"force_limit={ep.JOINT_DRIVE_FORCE_LIMIT}"
        )
    else:
        print(f"[Info] Replay apply mode: {args.replay_apply_mode}")

    ep.maybe_init_gpu_physx(scene)

    if args.settle_before_replay:
        print(
            "[Info] Settling particles before replay... "
            f"(min_steps={ep.SETTLE_MIN_STEPS}, max_steps={ep.SETTLE_MAX_STEPS})"
        )
        ep.settle_particles_before_replay(
            scene=scene,
            particles=particles,
            min_steps=ep.SETTLE_MIN_STEPS,
            max_steps=ep.SETTLE_MAX_STEPS,
            stable_window_steps=ep.SETTLE_STABLE_WINDOW_STEPS,
            mean_speed_threshold=ep.SETTLE_MEAN_SPEED_THRESHOLD,
            max_speed_threshold=ep.SETTLE_MAX_SPEED_THRESHOLD,
        )

    qlimits = np.asarray(robot.get_qlimits(), dtype=np.float32).reshape(-1, 2)
    policy = ep.build_joint_policy_from_json(
        json_path=args.keyframes_json,
        dof=robot.dof,
        qlimits=qlimits,
        time_scale=float(args.time_scale),
        loop=False,
    )
    total_steps = (
        max(1, int(args.max_steps))
        if args.max_steps is not None
        else int(np.ceil(float(policy.period) * float(args.time_scale))) + 1
    )
    sample_interval_steps = max(1, int(args.sample_interval_steps))

    ee_link_index, ee_link_name = ep.resolve_ee_link(robot, preferred_name=None)
    tracked_particle_indices = _choose_particle_indices(
        total_count=len(particles),
        max_particles=int(args.max_particles),
        seed=int(args.particle_sample_seed),
    )
    print(
        "[Info] Rollout export started: "
        f"steps={total_steps}, sample_interval={sample_interval_steps}, "
        f"tracked_particles={tracked_particle_indices.shape[0]}/{len(particles)}, "
        f"ee_link={ee_link_name}"
    )

    is_gpu_backend = isinstance(scene.physx_system, sapien.physx.PhysxGpuSystem)
    start_ts = time.perf_counter()
    progress_interval = max(1, int(np.ceil(total_steps / 20.0)))

    sample_steps: list[int] = []
    sample_time_s: list[float] = []
    qpos_samples: list[np.ndarray] = []
    qvel_samples: list[np.ndarray] = []
    ee_xyz_samples: list[np.ndarray] = []
    ee_rpy_samples: list[np.ndarray] = []
    pos_samples: list[np.ndarray] = []
    vel_samples: list[np.ndarray] = []
    source_count_samples: list[int] = []
    source_mass_samples: list[float] = []
    source_vol_samples: list[float] = []
    receiver_count_samples: list[int] = []
    receiver_mass_samples: list[float] = []
    receiver_vol_samples: list[float] = []

    def add_sample(step_idx: int) -> None:
        sample = _collect_sample(
            robot=robot,
            particles=particles,
            tracked_particle_indices=tracked_particle_indices,
            ee_link_index=ee_link_index,
        )
        sample_steps.append(int(step_idx))
        sample_time_s.append(float(step_idx) * float(ep.SIM_TIMESTEP))
        qpos_samples.append(sample["qpos"])
        qvel_samples.append(sample["qvel"])
        ee_xyz_samples.append(sample["ee_xyz"])
        ee_rpy_samples.append(sample["ee_rpy"])
        pos_samples.append(sample["particle_positions"])
        vel_samples.append(sample["particle_velocities"])
        source_count_samples.append(sample["source_count"])
        source_mass_samples.append(sample["source_mass_kg"])
        source_vol_samples.append(sample["source_volume_m3"])
        receiver_count_samples.append(sample["receiver_count"])
        receiver_mass_samples.append(sample["receiver_mass_kg"])
        receiver_vol_samples.append(sample["receiver_volume_m3"])

    add_sample(step_idx=0)
    warned_control_failure = False
    for sim_step_index in range(1, total_steps + 1):
        q_target = policy.query(sim_step_index)
        if args.replay_apply_mode == "direct":
            ok = ep.apply_joint_target_direct(robot=robot, target_qpos=q_target, physx_system=scene.physx_system)
        else:
            ok = ep.apply_joint_target(robot=robot, target_qpos=q_target, physx_system=scene.physx_system)
        if (not ok) and (not warned_control_failure):
            print("[Warn] Failed to apply target on current backend. Remaining steps continue best-effort.")
            warned_control_failure = True

        scene.step()
        if is_gpu_backend:
            scene.physx_system.sync_poses_gpu_to_cpu()

        if (sim_step_index % sample_interval_steps) == 0 or sim_step_index == total_steps:
            add_sample(step_idx=sim_step_index)

        if (sim_step_index % progress_interval) == 0 or sim_step_index == total_steps:
            elapsed = max(1e-6, time.perf_counter() - start_ts)
            pct = 100.0 * float(sim_step_index) / float(total_steps)
            step_rate = float(sim_step_index) / elapsed
            eta = max(0.0, float(total_steps - sim_step_index) / max(1e-6, step_rate))
            print(
                "[Info] Replay progress: "
                f"{sim_step_index}/{total_steps} ({pct:.1f}%), "
                f"samples={len(sample_steps)}, elapsed={elapsed:.1f}s, eta={eta:.1f}s"
            )

    with h5py.File(output_hdf5, "w") as h5f:
        h5f.attrs["format"] = "granular_mapping_rollout_hdf5_v1"
        h5f.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        h5f.create_dataset("sample_steps", data=np.asarray(sample_steps, dtype=np.int32), compression="gzip")
        h5f.create_dataset("sample_time_s", data=np.asarray(sample_time_s, dtype=np.float32), compression="gzip")
        h5f.create_dataset("qpos", data=np.asarray(qpos_samples, dtype=np.float32), compression="gzip")
        h5f.create_dataset("qvel", data=np.asarray(qvel_samples, dtype=np.float32), compression="gzip")
        h5f.create_dataset("ee_xyz", data=np.asarray(ee_xyz_samples, dtype=np.float32), compression="gzip")
        h5f.create_dataset("ee_rpy", data=np.asarray(ee_rpy_samples, dtype=np.float32), compression="gzip")
        h5f.create_dataset("particle_indices", data=tracked_particle_indices.astype(np.int32), compression="gzip")
        h5f.create_dataset("particle_positions", data=np.asarray(pos_samples, dtype=np.float32), compression="gzip")
        h5f.create_dataset("particle_velocities", data=np.asarray(vel_samples, dtype=np.float32), compression="gzip")
        h5f.create_dataset("source_pool_count", data=np.asarray(source_count_samples, dtype=np.int32), compression="gzip")
        h5f.create_dataset(
            "source_pool_mass_kg",
            data=np.asarray(source_mass_samples, dtype=np.float32),
            compression="gzip",
        )
        h5f.create_dataset(
            "source_pool_volume_m3",
            data=np.asarray(source_vol_samples, dtype=np.float32),
            compression="gzip",
        )
        h5f.create_dataset(
            "receiver_pool_count",
            data=np.asarray(receiver_count_samples, dtype=np.int32),
            compression="gzip",
        )
        h5f.create_dataset(
            "receiver_pool_mass_kg",
            data=np.asarray(receiver_mass_samples, dtype=np.float32),
            compression="gzip",
        )
        h5f.create_dataset(
            "receiver_pool_volume_m3",
            data=np.asarray(receiver_vol_samples, dtype=np.float32),
            compression="gzip",
        )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "equipment_model": args.equipment_model,
        "config_path": str(config_path),
        "keyframes_json": str(Path(args.keyframes_json).expanduser().resolve()),
        "urdf_path": str(urdf_path),
        "backend": "cpu" if args.cpu else "gpu_or_auto",
        "collision": args.collision,
        "bucket_collision_mode": args.bucket_collision_mode,
        "replay_apply_mode": args.replay_apply_mode,
        "time_scale": float(args.time_scale),
        "sim_timestep": float(ep.SIM_TIMESTEP),
        "total_steps": int(total_steps),
        "sample_interval_steps": int(sample_interval_steps),
        "num_samples": int(len(sample_steps)),
        "num_particles_total": int(len(particles)),
        "num_particles_tracked": int(tracked_particle_indices.shape[0]),
        "tracked_particle_indices_path": "embedded:particle_indices",
        "ee_link_index": int(ee_link_index),
        "ee_link_name": ee_link_name,
        "settle_before_replay": bool(args.settle_before_replay),
        "output_hdf5": str(output_hdf5),
    }
    output_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Info] Export finished: {output_hdf5}")
    print(f"[Info] Metadata: {output_meta}")
    print(
        "[Info] Saved datasets (HDF5): "
        "sample_steps, sample_time_s, qpos, qvel, ee_xyz, ee_rpy, "
        "particle_indices, particle_positions, particle_velocities, "
        "source_pool_*, receiver_pool_*"
    )


if __name__ == "__main__":
    main()
