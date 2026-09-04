"""End-to-end pipeline: text prompt → mask → 6D pose.

PASTE TARGET
------------
Copy your existing combo-model / full-stack integration here.
This is the glue that wires GroundedSAM2 → FoundationPose (and any
temporal filtering, multi-object selection, etc. you already built).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotvision.pose.foundation_pose import FoundationPoseEstimator, PoseResult
from robotvision.segmentation.grounded_sam2 import GroundedSAM2, SegmentationResult


@dataclass
class PipelineResult:
    """Output of one pipeline step (single frame or sequence)."""

    pose: PoseResult
    segmentation: SegmentationResult
    debug_image: np.ndarray | None = None


class ComboPipeline:
    """Grounded SAM 2 + FoundationPose combined pipeline."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Build segmenter + pose estimator from pipeline config.

        PASTE: your combo init (shared camera config, path setup, filters).
        """
        self.config = config
        raise NotImplementedError("Paste your ComboPipeline __init__ here")

    def run_frame(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        prompt: str,
        *,
        mesh_path: Path | None = None,
        register: bool = True,
    ) -> PipelineResult:
        """Single frame: segment by prompt, then estimate / track pose.

        PASTE: your main per-frame loop here.
        """
        raise NotImplementedError("Paste your run_frame implementation")

    def run_sequence(
        self,
        rgbd_dir: Path,
        prompt: str,
        mesh_path: Path,
        output_dir: Path,
    ) -> list[PipelineResult]:
        """Process an RGB-D sequence: register on frame 0, track rest.

        PASTE: your sequence / video pipeline here.
        """
        raise NotImplementedError("Paste your run_sequence implementation")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ComboPipeline:
        from robotvision.utils.io import load_config

        return cls(load_config(config_path))
