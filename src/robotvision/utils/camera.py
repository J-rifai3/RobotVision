"""Camera intrinsics and depth conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1000.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CameraIntrinsics:
        intr = cfg["intrinsics"]
        return cls(
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            depth_scale=float(cfg.get("depth_scale", 1000.0)),
        )

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def depth_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert raw depth to meters (float32).

    PASTE: override if your depth format differs (e.g. already float meters).
    """
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        return depth.astype(np.float32)
    return depth.astype(np.float32) / depth_scale
