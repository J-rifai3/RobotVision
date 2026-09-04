#!/usr/bin/env python3
"""Run Grounded SAM 2 segmentation.

Usage:
  python scripts/run_grounded_sam2.py --prompt "red mug"
  python scripts/run_grounded_sam2.py --config configs/grounded_sam2.yaml --image path/to/color.png
"""

from robotvision.cli import segment_main

if __name__ == "__main__":
    import tyro

    tyro.cli(segment_main)
