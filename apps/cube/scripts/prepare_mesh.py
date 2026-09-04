#!/usr/bin/env python3
"""Convert a SolidWorks STL/OBJ export into a FoundationPose-ready cube mesh.

FoundationPose expects:
  - meters
  - a triangle mesh (.obj recommended)
  - optional texture (simple solid color is fine)

Usage:
  # After exporting cube_raw.stl from SolidWorks (usually in mm):
  python scripts/prepare_mesh.py

  # Or pass paths / scale explicitly:
  python scripts/prepare_mesh.py --input mesh/cube_raw.stl --scale 0.001
  python scripts/prepare_mesh.py --input mesh/cube_raw.stl --force-edge-m 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_solid_texture(mesh: trimesh.Trimesh, color=(180, 140, 90, 255)) -> trimesh.Trimesh:
    """Assign a flat vertex color so FoundationPose has appearance cues."""
    mesh = mesh.copy()
    rgba = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = rgba
    return mesh


def prepare_mesh(
    input_path: Path,
    output_path: Path,
    scale: float = 0.001,
    force_edge_m: float | None = None,
) -> trimesh.Trimesh:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing CAD export: {input_path}\n"
            "Export your SolidWorks part as STL or OBJ into that path first.\n"
            "File → Save As → STL (Binary) or OBJ."
        )

    loaded = trimesh.load(input_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError(f"Could not load a triangle mesh from {input_path}")

    mesh = mesh.copy()
    mesh.apply_scale(float(scale))

    # Center at origin (FoundationPose works in object frame centered on the mesh).
    mesh.apply_translation(-mesh.bounding_box.centroid)

    extents = mesh.bounding_box.extents
    edge = float(np.max(extents))
    print(f"After scale={scale}: extents = {extents} m  (max edge ≈ {edge:.6f} m)")

    if force_edge_m is not None and edge > 1e-9:
        extra = float(force_edge_m) / edge
        mesh.apply_scale(extra)
        extents = mesh.bounding_box.extents
        print(f"Forced max edge to {force_edge_m} m → extents = {extents} m")

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight. Pose can still work, but check the export.")

    mesh = make_solid_texture(mesh)
    mesh.fix_normals()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    print(f"Wrote FoundationPose mesh → {output_path}")
    print(f"  vertices={len(mesh.vertices)}  faces={len(mesh.faces)}")
    print(f"  diameter≈{float(mesh.bounding_sphere.primitive.radius) * 2:.4f} m")
    return mesh


def main() -> int:
    cfg = load_config()
    mesh_cfg = cfg.get("mesh", {})
    default_in = ROOT / mesh_cfg.get("source_file", "mesh/cube_raw.stl")
    default_out = ROOT / cfg["paths"].get("mesh_file", "mesh/cube.obj")
    default_scale = float(mesh_cfg.get("scale", 0.001))
    force = mesh_cfg.get("force_edge_m", None)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument("--output", type=Path, default=default_out)
    p.add_argument("--scale", type=float, default=default_scale, help="Unit scale (0.001 = mm→m)")
    p.add_argument(
        "--force-edge-m",
        type=float,
        default=None if force is None else float(force),
        help="Optional: rescale so the longest edge equals this many meters",
    )
    args = p.parse_args()

    prepare_mesh(args.input, args.output, scale=args.scale, force_edge_m=args.force_edge_m)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
