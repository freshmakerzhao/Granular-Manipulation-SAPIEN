"""Unified root entry for scene viewer and keyframe capture."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Root entry: choose scene viewer or keyframe capture mode."
    )
    parser.add_argument(
        "--entry",
        choices=("scene", "capture"),
        default="scene",
        help="scene: 打开环境场景；capture: 进入 keyframe 录制模式。",
    )

    # Keep all remaining args and forward them to the selected sub-entry.
    args, remaining = parser.parse_known_args()

    if args.entry == "scene":
        from envs.excavator_pool import main as scene_main

        sys.argv = ["envs/excavator_pool.py", *remaining]
        scene_main()
        return

    from utils.keyframe_capture_env import main as capture_main

    sys.argv = ["utils/keyframe_capture_env.py", *remaining]
    capture_main()


if __name__ == "__main__":
    main()
