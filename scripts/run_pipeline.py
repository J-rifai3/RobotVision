#!/usr/bin/env python3
"""Run full pipeline: prompt → mask → 6D pose.

Usage:
  python scripts/run_pipeline.py --config configs/pipeline.yaml
"""

from robotvision.cli import pipeline_main

if __name__ == "__main__":
    import tyro

    tyro.cli(pipeline_main)
