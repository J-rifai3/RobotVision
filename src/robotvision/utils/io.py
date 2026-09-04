"""Config loading and RGB-D I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config, resolving simple `extends:` chains."""
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.pop("extends", None)
    if extends:
        base_path = (path.parent / extends).resolve()
        base = load_config(base_path)
        base.update(cfg)
        return base
    return cfg


def load_rgbd_frame(frame_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load color + depth from a frame directory.

    Expects color.png/.jpg and depth.png or depth.npy.
    PASTE: adapt if your capture layout differs.
    """
    frame_dir = Path(frame_dir)
    color_path = _find_first(frame_dir, ["color.png", "color.jpg", "rgb.png"])
    depth_path = _find_first(frame_dir, ["depth.npy", "depth.png"])

    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Could not read color image: {color_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    if depth_path.suffix == ".npy":
        depth = np.load(depth_path)
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    return color, depth


def save_pose(pose: np.ndarray, path: Path) -> None:
    """Save 4×4 pose as .npy and human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), pose)
    with path.with_suffix(".json").open("w") as f:
        json.dump({"object_in_camera": pose.tolist()}, f, indent=2)


def _find_first(directory: Path, names: list[str]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of {names} found in {directory}")
