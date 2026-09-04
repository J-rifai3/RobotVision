"""Grounded SAM 2 wrapper.

PASTE TARGET
------------
Copy your existing Grounded SAM 2 inference code here:
  - model loading (Grounding DINO / Florence-2 + SAM 2)
  - text-prompted detection → box → mask
  - optional video mask propagation (SAM 2 predictor)

Keep the public API stable so scripts/ and pipeline/combo.py stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SegmentationResult:
    """Output of a single segmentation pass."""

    mask: np.ndarray  # H×W bool or uint8 {0, 255}
    boxes: np.ndarray | None = None  # N×4 xyxy
    scores: np.ndarray | None = None  # N
    labels: list[str] = field(default_factory=list)


class GroundedSAM2:
    """Open-vocabulary segmentation via Grounded SAM 2."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Load models from config (see configs/grounded_sam2.yaml).

        PASTE: your __init__ / model-loading logic here.
        """
        self.config = config
        raise NotImplementedError("Paste your Grounded SAM 2 setup code in __init__")

    def segment_image(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> SegmentationResult:
        """Run detection + SAM2 on a single RGB frame.

        PASTE: your single-image inference loop here.
        """
        raise NotImplementedError("Paste your segment_image implementation")

    def segment_video(
        self,
        frames_dir: Path,
        prompt: str,
        output_dir: Path,
    ) -> list[SegmentationResult]:
        """Optional: propagate masks across frames with SAM 2 video mode.

        PASTE: your video / mask-tracking code here.
        """
        raise NotImplementedError("Paste your segment_video implementation")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> GroundedSAM2:
        from robotvision.utils.io import load_config

        return cls(load_config(config_path))
