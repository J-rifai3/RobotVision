"""FoundationPose wrapper.

PASTE TARGET
------------
Copy your existing FoundationPose code here:
  - mesh loading & scaling
  - register() on first frame with mask + RGB-D
  - track() on subsequent frames
  - pose refinement loop

Keep the public API stable so scripts/ and pipeline/combo.py stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PoseResult:
    """6D pose of object in camera frame."""

    object_in_camera: np.ndarray  # 4×4 SE(3)
    score: float | None = None


class FoundationPoseEstimator:
    """Model-based 6D pose estimation via FoundationPose."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Load mesh and FoundationPose models from config.

        PASTE: your FoundationPose bootstrap (estimator, scorer, refiner) here.
        """
        self.config = config
        raise NotImplementedError("Paste your FoundationPose setup code in __init__")

    def register(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        mesh_path: Path | None = None,
    ) -> PoseResult:
        """Estimate initial pose from RGB-D + mask.

        PASTE: your register / first-frame pose code here.
        """
        raise NotImplementedError("Paste your register implementation")

    def track(
        self,
        color: np.ndarray,
        depth: np.ndarray,
    ) -> PoseResult:
        """Track pose on subsequent frames (after register).

        PASTE: your track loop here.
        """
        raise NotImplementedError("Paste your track implementation")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> FoundationPoseEstimator:
        from robotvision.utils.io import load_config

        return cls(load_config(config_path))
