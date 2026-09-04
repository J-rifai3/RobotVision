#!/usr/bin/env python3
"""Evaluate FoundationPose accuracy against a ground-truth 4x4 pose.

Handles cube rotational symmetry (24 orientations) when enabled in config.

Usage:
  python scripts/evaluate_accuracy.py --est debug/ob_in_cam/0000.txt
  python scripts/evaluate_accuracy.py --est debug/ob_in_cam/0000.txt --gt gt/ob_in_cam.txt
"""

from __future__ import annotations

import argparse
from itertools import permutations, product
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rotation_geodesic_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_est @ R_gt.T
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def cube_rotation_group() -> list[np.ndarray]:
    """All 24 proper rotations that map a cube onto itself."""
    rots: list[np.ndarray] = []
    for perm in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if abs(float(np.linalg.det(R)) - 1.0) < 1e-8:
                rots.append(R)
    assert len(rots) == 24, f"expected 24 cube rotations, got {len(rots)}"
    return rots


def pose_errors(
    T_est: np.ndarray,
    T_gt: np.ndarray,
    handle_cube_symmetry: bool = True,
) -> dict[str, float]:
    t_err = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    R_est, R_gt = T_est[:3, :3], T_gt[:3, :3]

    if not handle_cube_symmetry:
        r_err = rotation_geodesic_deg(R_est, R_gt)
        return {"translation_m": t_err, "rotation_deg": r_err, "symmetry_applied": 0.0}

    best = None
    for S in cube_rotation_group():
        # Estimated pose may match GT up to a cube symmetry S in the object frame:
        # T_est ≈ T_gt @ S
        r = rotation_geodesic_deg(R_est, R_gt @ S)
        if best is None or r < best:
            best = r
    return {
        "translation_m": t_err,
        "rotation_deg": float(best),
        "symmetry_applied": 1.0,
    }


def main() -> int:
    cfg = load_config()
    default_gt = cfg.get("accuracy", {}).get("gt_pose_file", "gt/ob_in_cam.txt")
    handle_sym = bool(cfg.get("accuracy", {}).get("handle_cube_symmetry", True))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--est", type=Path, required=True, help="Estimated 4x4 pose .txt")
    p.add_argument("--gt", type=Path, default=ROOT / default_gt)
    p.add_argument("--no-symmetry", action="store_true")
    args = p.parse_args()

    T_est = np.loadtxt(args.est).reshape(4, 4)
    if not args.gt.exists():
        print(f"No ground-truth pose at {args.gt}")
        print("Estimated pose (camera ← object):")
        np.set_printoptions(precision=6, suppress=True)
        print(T_est)
        print("\nTo measure accuracy, place the cube at a known pose and save it as:")
        print(f"  {args.gt}")
        print("Then re-run this script.")
        return 0

    T_gt = np.loadtxt(args.gt).reshape(4, 4)
    err = pose_errors(T_est, T_gt, handle_cube_symmetry=handle_sym and not args.no_symmetry)

    print("=== FoundationPose cube accuracy ===")
    print(f"translation error : {err['translation_m'] * 1000:.2f} mm")
    print(f"rotation error    : {err['rotation_deg']:.2f} deg")
    if err["symmetry_applied"]:
        print("(cube 24-fold symmetry accounted for)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
