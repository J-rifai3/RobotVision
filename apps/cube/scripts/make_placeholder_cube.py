#!/usr/bin/env python3
"""Create a placeholder cube mesh when you don't have the SolidWorks export yet.

Usage:
  python scripts/make_placeholder_cube.py --edge-m 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--edge-m", type=float, default=0.05, help="Cube edge length in meters")
    p.add_argument("--output", type=Path, default=ROOT / "mesh" / "cube.obj")
    args = p.parse_args()

    mesh = trimesh.creation.box(extents=[args.edge_m, args.edge_m, args.edge_m])
    mesh.apply_translation(-mesh.bounding_box.centroid)
    rgba = np.tile(np.array([180, 140, 90, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = rgba
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(f"Wrote placeholder cube ({args.edge_m} m) → {args.output}")
    print("Replace this with prepare_mesh.py output from your SolidWorks export for a real test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
