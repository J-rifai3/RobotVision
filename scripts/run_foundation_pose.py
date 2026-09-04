#!/usr/bin/env python3
"""Run FoundationPose on RGB-D + mask.

Usage:
  python scripts/run_foundation_pose.py --frame data/rgbd/demo/frame_000000 --mask data/masks/mask.png
"""

from robotvision.cli import pose_main

if __name__ == "__main__":
    import tyro

    tyro.cli(pose_main)
