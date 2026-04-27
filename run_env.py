"""Unified root entry for scene viewer and keyframe capture."""

from __future__ import annotations

import argparse
import sys


def _build_root_parser(*, add_help: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Root entry: scene / capture / collect / inspect.",
        add_help=add_help,
    )
    parser.add_argument(
        "--entry",
        choices=("scene", "capture", "collect", "inspect"),
        default="scene",
        help=(
            "scene: 打开环境场景；"
            "capture: 进入 keyframe 录制模式；"
            "collect: 回放并导出建图数据；"
            "inspect: 检查导出的 rollout 数据（hdf5）。"
        ),
    )
    return parser


def main() -> None:
    argv = sys.argv[1:]
    has_explicit_entry = any(arg == "--entry" or arg.startswith("--entry=") for arg in argv)
    if (not has_explicit_entry) and any(arg in {"-h", "--help"} for arg in argv):
        _build_root_parser(add_help=True).parse_args(argv)
        return

    # Keep all remaining args and forward them to the selected sub-entry.
    args, remaining = _build_root_parser(add_help=False).parse_known_args(argv)

    if args.entry == "scene":
        from envs.excavator_pool import main as scene_main

        sys.argv = ["envs/excavator_pool.py", *remaining]
        scene_main()
        return

    if args.entry == "capture":
        from utils.keyframe_capture_env import main as capture_main

        sys.argv = ["utils/keyframe_capture_env.py", *remaining]
        capture_main()
        return

    if args.entry == "collect":
        from collect_mapping_rollout import main as collect_main

        sys.argv = ["collect_mapping_rollout.py", *remaining]
        collect_main()
        return

    from inspect_mapping_rollout import main as inspect_main

    sys.argv = ["inspect_mapping_rollout.py", *remaining]
    inspect_main()


if __name__ == "__main__":
    main()
