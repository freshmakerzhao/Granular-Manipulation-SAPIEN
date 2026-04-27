"""Inspect exported rollout_data.hdf5 for quick sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_KEYS = (
    "sample_steps",
    "sample_time_s",
    "qpos",
    "qvel",
    "ee_xyz",
    "ee_rpy",
    "particle_indices",
    "particle_positions",
    "particle_velocities",
    "source_pool_count",
    "source_pool_mass_kg",
    "source_pool_volume_m3",
    "receiver_pool_count",
    "receiver_pool_mass_kg",
    "receiver_pool_volume_m3",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 collect_mapping_rollout.py 导出的 hdf5 数据结构与基础统计。"
    )
    parser.add_argument("data_file", type=str, help="rollout_data.hdf5 路径")
    parser.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="可选：rollout_data_meta.json 路径；默认尝试同目录同名前缀。",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="打印前多少个采样点的 step/source/receiver 计数。",
    )
    return parser.parse_args()


def _fmt_shape(arr: np.ndarray) -> str:
    return "x".join(str(int(v)) for v in arr.shape)


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _load_data_map(data_path: Path) -> dict[str, np.ndarray]:
    suffix = data_path.suffix.lower()
    if suffix != ".hdf5":
        raise ValueError(f"Unsupported file type: {data_path}. Only .hdf5 is supported.")
    with h5py.File(data_path, "r") as h5f:
        data = {k: np.asarray(h5f[k]) for k in h5f.keys()}
    return data


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_file).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if args.meta_json is None:
        meta_path = data_path.with_name(f"{data_path.stem}_meta.json")
    else:
        meta_path = Path(args.meta_json).expanduser().resolve()
    meta = _load_meta(meta_path)

    data = _load_data_map(data_path)
    keys = sorted(data.keys())
    print(f"[Info] Data file: {data_path}")
    print("[Info] Format: hdf5")
    print(f"[Info] Keys ({len(keys)}): {keys}")

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"[Warn] Missing required keys: {missing}")
        return
    print("[Info] Required keys: OK")

    print("\n[Info] Tensor shapes")
    for k in REQUIRED_KEYS:
        if k in data:
            arr = data[k]
            print(f"  - {k:24s} shape={_fmt_shape(arr):>12s} dtype={arr.dtype}")

    sample_steps = np.asarray(data["sample_steps"], dtype=np.int64).reshape(-1)
    sample_time = np.asarray(data["sample_time_s"], dtype=np.float64).reshape(-1)
    source_count = np.asarray(data["source_pool_count"], dtype=np.int64).reshape(-1)
    receiver_count = np.asarray(data["receiver_pool_count"], dtype=np.int64).reshape(-1)
    particle_vel = np.asarray(data["particle_velocities"], dtype=np.float64)

    n_samples = int(sample_steps.shape[0])
    print("\n[Info] Basic stats")
    print(f"  - num_samples: {n_samples}")
    if n_samples > 0:
        print(f"  - step range: [{int(sample_steps.min())}, {int(sample_steps.max())}]")
        print(f"  - time range: [{float(sample_time.min()):.4f}, {float(sample_time.max()):.4f}] s")

    if n_samples >= 2:
        d_step = np.diff(sample_steps)
        d_time = np.diff(sample_time)
        print(
            "  - sample_step_interval: "
            f"min={int(d_step.min())}, max={int(d_step.max())}, median={int(np.median(d_step))}"
        )
        print(
            "  - sample_time_interval: "
            f"min={float(d_time.min()):.6f}, max={float(d_time.max()):.6f}, median={float(np.median(d_time)):.6f}"
        )
        strictly_inc = bool(np.all(d_step > 0))
        print(f"  - steps strictly increasing: {strictly_inc}")
    else:
        print("  - sample intervals: N/A (need >=2 samples)")

    speed = np.linalg.norm(particle_vel, axis=-1)
    print(
        "  - particle speed (m/s): "
        f"mean={float(np.mean(speed)):.6f}, p95={float(np.percentile(speed, 95.0)):.6f}, max={float(np.max(speed)):.6f}"
    )

    if n_samples > 0:
        source_init = int(source_count[0])
        source_end = int(source_count[-1])
        receiver_end = int(receiver_count[-1])
        removed = source_init - source_end
        delta = receiver_end - removed
        print(
            "  - transfer rough check (count): "
            f"source_init={source_init}, source_end={source_end}, removed={removed}, "
            f"receiver_end={receiver_end}, delta={delta}"
        )

    head_n = max(0, int(args.head))
    if head_n > 0 and n_samples > 0:
        print(f"\n[Info] First {min(head_n, n_samples)} samples")
        limit = min(head_n, n_samples)
        for i in range(limit):
            print(
                f"  - i={i:03d} step={int(sample_steps[i]):6d} "
                f"source={int(source_count[i]):6d} receiver={int(receiver_count[i]):6d}"
            )

    if meta is None:
        print(f"\n[Info] Meta JSON not found or unreadable: {meta_path}")
    else:
        print(f"\n[Info] Meta: {meta_path}")
        for k in (
            "equipment_model",
            "keyframes_json",
            "replay_apply_mode",
            "collision",
            "bucket_collision_mode",
            "time_scale",
            "sim_timestep",
            "total_steps",
            "sample_interval_steps",
            "num_particles_total",
            "num_particles_tracked",
        ):
            if k in meta:
                print(f"  - {k}: {meta[k]}")


if __name__ == "__main__":
    main()
