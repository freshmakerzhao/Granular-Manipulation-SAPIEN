"""Collect rollout data by replaying joint keyframes in GranularExcavatorEnv."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from envs.excavator_pool import make_excavator_env
from scripted_policy import (
    LinearJointKeyframePolicy,
    build_default_excavator_keyframes,
    load_joint_keyframes_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute joint keyframes and collect commanded/actual trajectory.")
    parser.add_argument("--equipment-model", type=str, default="excavator_s010", help="Excavator model preset.")
    parser.add_argument("--config", type=str, default=None, help="Optional config.json path.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend.")
    parser.add_argument(
        "--keyframes-json",
        type=str,
        default=None,
        help="Optional joint keyframes JSON path (supports q or qpos).",
    )
    parser.add_argument("--n-substeps", type=int, default=1, help="Physics substeps per policy step.")
    parser.add_argument("--out-dir", type=str, default="outputs/ee_rollouts", help="Output directory.")
    parser.add_argument("--name", type=str, default=None, help="Optional output run name.")
    args = parser.parse_args()

    env = make_excavator_env(
        equipment_model=args.equipment_model,
        config_path=args.config,
        prefer_gpu=not args.cpu,
    )
    obs0 = env.reset()
    assert env.robot is not None
    qlimits = np.asarray(env.robot.get_qlimits(), dtype=np.float32).reshape(-1, 2)

    if args.keyframes_json:
        joint_keyframes = load_joint_keyframes_json(
            json_path=args.keyframes_json,
            dof=env.robot.dof,
            qlimits=qlimits,
        )
        keyframe_source = str(Path(args.keyframes_json).expanduser().resolve())
    else:
        joint_keyframes = build_default_excavator_keyframes(
            init_qpos=np.asarray(obs0["qpos"], dtype=np.float32).reshape(-1),
            qlimits=qlimits,
        )
        keyframe_source = "scripted_policy.build_default_excavator_keyframes"

    policy = LinearJointKeyframePolicy(
        keyframes=joint_keyframes,
        dof=env.robot.dof,
        qlimits=qlimits,
        time_scale=1.0,
        loop=False,
    )
    total_steps = int(policy.period) + 1

    cmd_qpos = np.zeros((total_steps, env.robot.dof), dtype=np.float32)
    act_qpos = np.zeros((total_steps, env.robot.dof), dtype=np.float32)
    act_xyz = np.zeros((total_steps, 3), dtype=np.float32)
    act_rpy = np.zeros((total_steps, 3), dtype=np.float32)

    metrics_log: list[dict] = []

    for t in range(total_steps):
        q_target = policy.query(t)
        obs, metrics = env.step(
            action={"joint_pos": q_target.tolist()},
            n_substeps=max(1, int(args.n_substeps)),
        )
        ee_actual = env.get_ee_pose_world()

        cmd_qpos[t] = np.asarray(q_target, dtype=np.float32).reshape(-1)
        act_qpos[t] = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
        act_xyz[t] = ee_actual["xyz"]
        act_rpy[t] = ee_actual["rpy"]
        metrics_log.append(metrics)

    run_name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "trajectory.npz",
        cmd_qpos=cmd_qpos,
        act_qpos=act_qpos,
        act_xyz=act_xyz,
        act_rpy=act_rpy,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics_log, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "equipment_model": args.equipment_model,
        "backend": "cpu" if args.cpu else "gpu",
        "n_substeps": int(args.n_substeps),
        "total_steps": total_steps,
        "keyframe_source": keyframe_source,
        "output_npz": str((out_dir / "trajectory.npz").resolve()),
        "output_metrics_json": str((out_dir / "metrics.json").resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Info] Joint rollout collection done: {out_dir}")


if __name__ == "__main__":
    main()
