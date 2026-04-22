"""
Joint-space scripted policy utilities for excavator control.

This module provides:
1) A linear keyframe interpolator in joint space.
2) JSON keyframe loading.
3) A default excavator digging trajectory template.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class JointKeyframe:
    """A single joint-space keyframe at simulation step `t`."""

    t: int
    q: np.ndarray


def _clip_to_limits(q: np.ndarray, qlimits: np.ndarray | None) -> np.ndarray:
    """Clip a qpos vector to articulation limits when available."""
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if qlimits is None:
        return q
    qlimits = np.asarray(qlimits, dtype=np.float32).reshape(-1, 2)
    if qlimits.shape[0] != q.shape[0]:
        raise ValueError(f"qlimits shape mismatch: q={q.shape[0]} dof, qlimits={qlimits.shape[0]} dof")
    low = qlimits[:, 0]
    high = qlimits[:, 1]
    return np.clip(q, low, high)


def _normalize_keyframes(
    keyframes: Iterable[JointKeyframe],
    dof: int,
    qlimits: np.ndarray | None = None,
) -> list[JointKeyframe]:
    """Sort and validate keyframes, returning a canonical list."""
    frames = sorted(keyframes, key=lambda x: x.t)
    if not frames:
        raise ValueError("At least one keyframe is required.")
    if frames[0].t != 0:
        first = JointKeyframe(t=0, q=np.asarray(frames[0].q, dtype=np.float32).reshape(-1))
        frames = [first, *frames]

    normalized: list[JointKeyframe] = []
    prev_t = -1
    for frame in frames:
        q = np.asarray(frame.q, dtype=np.float32).reshape(-1)
        if q.shape[0] != dof:
            raise ValueError(f"Keyframe dof mismatch at t={frame.t}: expected {dof}, got {q.shape[0]}")
        if frame.t <= prev_t:
            raise ValueError("Keyframe times must be strictly increasing.")
        q = _clip_to_limits(q, qlimits)
        normalized.append(JointKeyframe(t=int(frame.t), q=q))
        prev_t = frame.t
    return normalized


class LinearJointKeyframePolicy:
    """Linear interpolation policy over joint-space keyframes."""

    def __init__(
        self,
        keyframes: Iterable[JointKeyframe],
        dof: int,
        qlimits: np.ndarray | None = None,
        time_scale: float = 1.0,
        loop: bool = False,
    ) -> None:
        if time_scale <= 0:
            raise ValueError(f"time_scale must be > 0, got {time_scale}")
        self._frames = _normalize_keyframes(keyframes, dof=dof, qlimits=qlimits)
        self._dof = int(dof)
        self._time_scale = float(time_scale)
        self._loop = bool(loop)
        self._period = int(self._frames[-1].t)

    @property
    def period(self) -> int:
        return self._period

    @property
    def dof(self) -> int:
        return self._dof

    def query(self, sim_step: int) -> np.ndarray:
        """Return interpolated qpos target for the given simulation step."""
        t_scaled = float(sim_step) / self._time_scale
        if self._loop and self._period > 0:
            t = t_scaled % float(self._period)
        else:
            t = min(t_scaled, float(self._period))

        times = [f.t for f in self._frames]
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = max(0, min(idx, len(self._frames) - 1))
        if idx >= len(self._frames) - 1:
            return self._frames[-1].q.copy()

        curr = self._frames[idx]
        nxt = self._frames[idx + 1]
        duration = float(nxt.t - curr.t)
        if duration <= 1e-8:
            return nxt.q.copy()
        frac = float((t - curr.t) / duration)
        return (curr.q + (nxt.q - curr.q) * frac).astype(np.float32)


@dataclass(frozen=True)
class EEKeyframe:
    """A single end-effector keyframe at simulation step `t`."""

    t: int
    xyz: np.ndarray
    rpy: np.ndarray


def _normalize_ee_keyframes(keyframes: Iterable[EEKeyframe]) -> list[EEKeyframe]:
    """Sort and validate end-effector keyframes."""
    frames = sorted(keyframes, key=lambda x: x.t)
    if not frames:
        raise ValueError("At least one EE keyframe is required.")
    if frames[0].t != 0:
        first = EEKeyframe(
            t=0,
            xyz=np.asarray(frames[0].xyz, dtype=np.float32).reshape(3),
            rpy=np.asarray(frames[0].rpy, dtype=np.float32).reshape(3),
        )
        frames = [first, *frames]

    normalized: list[EEKeyframe] = []
    prev_t = -1
    for i, frame in enumerate(frames):
        xyz = np.asarray(frame.xyz, dtype=np.float32).reshape(-1)
        rpy = np.asarray(frame.rpy, dtype=np.float32).reshape(-1)
        if xyz.shape[0] != 3:
            raise ValueError(f"EE keyframe #{i} xyz must be 3D, got shape={xyz.shape}")
        if rpy.shape[0] != 3:
            raise ValueError(f"EE keyframe #{i} rpy must be 3D, got shape={rpy.shape}")
        if frame.t <= prev_t:
            raise ValueError("EE keyframe times must be strictly increasing.")
        normalized.append(EEKeyframe(t=int(frame.t), xyz=xyz, rpy=rpy))
        prev_t = frame.t
    return normalized


class LinearEEKeyframePolicy:
    """Linear interpolation policy over end-effector xyz+rpy keyframes."""

    def __init__(
        self,
        keyframes: Iterable[EEKeyframe],
        time_scale: float = 1.0,
        loop: bool = False,
    ) -> None:
        if time_scale <= 0:
            raise ValueError(f"time_scale must be > 0, got {time_scale}")
        self._frames = _normalize_ee_keyframes(keyframes)
        self._time_scale = float(time_scale)
        self._loop = bool(loop)
        self._period = int(self._frames[-1].t)

    @property
    def period(self) -> int:
        return self._period

    def query(self, sim_step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return interpolated (xyz, rpy) for the given simulation step."""
        t_scaled = float(sim_step) / self._time_scale
        if self._loop and self._period > 0:
            t = t_scaled % float(self._period)
        else:
            t = min(t_scaled, float(self._period))

        times = [f.t for f in self._frames]
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = max(0, min(idx, len(self._frames) - 1))
        if idx >= len(self._frames) - 1:
            last = self._frames[-1]
            return last.xyz.copy(), last.rpy.copy()

        curr = self._frames[idx]
        nxt = self._frames[idx + 1]
        duration = float(nxt.t - curr.t)
        if duration <= 1e-8:
            return nxt.xyz.copy(), nxt.rpy.copy()

        frac = float((t - curr.t) / duration)
        xyz = curr.xyz + (nxt.xyz - curr.xyz) * frac
        rpy = curr.rpy + (nxt.rpy - curr.rpy) * frac
        return xyz.astype(np.float32), rpy.astype(np.float32)


def load_joint_keyframes_json(
    json_path: str | Path,
    dof: int,
    qlimits: np.ndarray | None = None,
) -> list[JointKeyframe]:
    """Load joint keyframes from JSON.

    Supported JSON formats:
    1) [{"t": 0, "q": [..]}, ...]
    2) {"keyframes": [{"t": 0, "q": [..]}, ...]}
    """
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Keyframe JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_frames = data.get("keyframes")
    else:
        raw_frames = data

    if not isinstance(raw_frames, list) or len(raw_frames) == 0:
        raise ValueError(f"Invalid keyframe JSON format in {path}.")

    keyframes: list[JointKeyframe] = []
    for i, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            raise ValueError(f"Keyframe #{i} must be an object.")
        if "t" not in item or "q" not in item:
            raise ValueError(f"Keyframe #{i} must contain 't' and 'q'.")
        t = int(item["t"])
        q = np.asarray(item["q"], dtype=np.float32).reshape(-1)
        if q.shape[0] != dof:
            raise ValueError(f"Keyframe #{i} dof mismatch: expected {dof}, got {q.shape[0]}")
        q = _clip_to_limits(q, qlimits)
        keyframes.append(JointKeyframe(t=t, q=q))
    return _normalize_keyframes(keyframes, dof=dof, qlimits=qlimits)


def load_ee_keyframes_json(json_path: str | Path) -> list[EEKeyframe]:
    """Load end-effector keyframes from JSON.

    Supported JSON formats:
    1) [{"t": 0, "xyz": [x,y,z], "rpy": [r,p,y]}, ...]
    2) {"keyframes": [{"t": 0, "xyz": [...], "rpy": [...]}, ...]}
    """
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EE keyframe JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_frames = data.get("keyframes")
    else:
        raw_frames = data

    if not isinstance(raw_frames, list) or len(raw_frames) == 0:
        raise ValueError(f"Invalid EE keyframe JSON format in {path}.")

    frames: list[EEKeyframe] = []
    for i, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            raise ValueError(f"EE keyframe #{i} must be an object.")
        if "t" not in item or "xyz" not in item or "rpy" not in item:
            raise ValueError(f"EE keyframe #{i} must contain 't', 'xyz', and 'rpy'.")
        t = int(item["t"])
        xyz = np.asarray(item["xyz"], dtype=np.float32).reshape(-1)
        rpy = np.asarray(item["rpy"], dtype=np.float32).reshape(-1)
        if xyz.shape[0] != 3:
            raise ValueError(f"EE keyframe #{i} xyz must have length 3, got {xyz.shape[0]}")
        if rpy.shape[0] != 3:
            raise ValueError(f"EE keyframe #{i} rpy must have length 3, got {rpy.shape[0]}")
        frames.append(EEKeyframe(t=t, xyz=xyz, rpy=rpy))
    return _normalize_ee_keyframes(frames)


def build_default_excavator_ee_keyframes(init_xyz: np.ndarray, init_rpy: np.ndarray) -> list[EEKeyframe]:
    """Build a conservative default EE trajectory around an initial pose."""
    init_xyz = np.asarray(init_xyz, dtype=np.float32).reshape(3)
    init_rpy = np.asarray(init_rpy, dtype=np.float32).reshape(3)

    def f(t: int, dxyz: list[float], drpy: list[float]) -> EEKeyframe:
        xyz = init_xyz + np.asarray(dxyz, dtype=np.float32)
        rpy = init_rpy + np.asarray(drpy, dtype=np.float32)
        return EEKeyframe(t=t, xyz=xyz, rpy=rpy)

    return [
        f(0, [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]),
        f(90, [-0.02, 0.02, -0.03], [0.00, 0.25, 0.00]),
        f(180, [-0.05, 0.04, -0.07], [0.00, 0.45, 0.05]),
        f(280, [0.03, -0.02, 0.05], [0.00, -0.15, -0.05]),
        f(360, [0.10, -0.06, 0.08], [0.00, -0.35, -0.10]),
        f(460, [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]),
    ]


# 采集用末端关键帧（世界坐标）。直接修改这里即可驱动环境执行并采集数据。
# 格式: (t, [x, y, z], [roll, pitch, yaw])，t 为仿真 step。
EXCAVATOR_EE_COLLECTION_KEYFRAMES: list[tuple[int, list[float], list[float]]] = [
    (0, [0.42, 0.02, 0.42], [0.0, -0.20, 0.00]),
    (80, [0.32, 0.05, 0.32], [0.0, 0.15, 0.00]),
    (160, [0.24, 0.10, 0.24], [0.0, 0.45, 0.05]),
    (260, [0.42, 0.24, 0.38], [0.0, -0.10, -0.10]),
    (340, [0.52, 0.35, 0.40], [0.0, -0.35, -0.20]),
    (420, [0.42, 0.02, 0.42], [0.0, -0.20, 0.00]),
]


def build_collection_excavator_ee_keyframes() -> list[EEKeyframe]:
    """Build EE keyframes from `EXCAVATOR_EE_COLLECTION_KEYFRAMES` for rollout/recording."""
    frames: list[EEKeyframe] = []
    for i, item in enumerate(EXCAVATOR_EE_COLLECTION_KEYFRAMES):
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"Invalid collection EE keyframe #{i}: expected (t, xyz, rpy).")
        t, xyz, rpy = item
        frames.append(
            EEKeyframe(
                t=int(t),
                xyz=np.asarray(xyz, dtype=np.float32).reshape(3),
                rpy=np.asarray(rpy, dtype=np.float32).reshape(3),
            )
        )
    return _normalize_ee_keyframes(frames)


def build_default_excavator_keyframes(
    init_qpos: np.ndarray,
    qlimits: np.ndarray | None = None,
) -> list[JointKeyframe]:
    """Build a conservative digging-like scripted trajectory template.

    Joint order follows articulation active joints:
    typically [j1_swing, j2_boom, j3_stick, j4_bucket, ...].
    """
    init_qpos = np.asarray(init_qpos, dtype=np.float32).reshape(-1)
    dof = int(init_qpos.shape[0])
    if dof <= 0:
        raise ValueError("init_qpos must have positive DOF.")

    def with_delta(delta4: list[float]) -> np.ndarray:
        q = init_qpos.copy()
        n = min(4, dof)
        q[:n] += np.asarray(delta4[:n], dtype=np.float32)
        return _clip_to_limits(q, qlimits)

    # A simple scoop cycle: approach -> press -> curl/lift -> swing -> dump -> return.
    return [
        JointKeyframe(t=0, q=with_delta([0.00, 0.00, 0.00, 0.00])),
        JointKeyframe(t=90, q=with_delta([0.05, -0.25, 0.35, -0.25])),
        JointKeyframe(t=180, q=with_delta([0.10, -0.55, 0.80, -0.85])),
        JointKeyframe(t=260, q=with_delta([0.15, -0.40, 0.62, -0.10])),
        JointKeyframe(t=360, q=with_delta([-0.35, -0.35, 0.50, -0.05])),
        JointKeyframe(t=430, q=with_delta([-0.45, -0.28, 0.42, 0.45])),
        JointKeyframe(t=520, q=with_delta([-0.10, -0.20, 0.25, 0.10])),
        JointKeyframe(t=620, q=with_delta([0.00, 0.00, 0.00, 0.00])),
    ]
