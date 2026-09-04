"""Debug visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(color: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask onto an RGB image."""
    out = color.copy()
    if mask.dtype != bool:
        mask = mask > 0
    color_mask = np.zeros_like(out)
    color_mask[mask] = (0, 255, 128)
    out = cv2.addWeighted(out, 1.0, color_mask, alpha, 0)
    return out


def draw_pose_axes(
    color: np.ndarray,
    object_in_camera: np.ndarray,
    intrinsics: np.ndarray,
    axis_length: float = 0.05,
) -> np.ndarray:
    """Draw XYZ axes for a 4×4 pose on an RGB image.

    PASTE: replace with your existing pose overlay if you use Open3D / trimesh.
    """
    _ = (object_in_camera, intrinsics, axis_length)
    return color.copy()
