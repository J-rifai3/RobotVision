#!/usr/bin/env python3
"""FoundationPose-only ZED cube tracker with a manually supplied first-frame mask.

This follows the official FoundationPose model-based workflow:

  pose = estimator.register(K, rgb, depth, ob_mask, ...)
  pose = estimator.track_one(rgb, depth, K, ...)

No Grounding DINO, Grounded-SAM, SAM2, or other segmentation model is imported.
The user draws the required binary mask on the first frame. FoundationPose needs
the mask only for registration; subsequent frames use ``track_one``.

Controls
--------
Mask editor:
  Left click       add polygon point
  Right click      undo last point
  c                clear polygon
  Enter / Space    accept polygon and register
  q                quit

Tracking:
  r                freeze a new frame and draw a new mask
  q                quit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# These knobs are supported by the locally patched FoundationPose installation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ZedCamera:
    """Minimal ZED RGB-D source; independent of scan_candle.py."""

    def __init__(self, cfg: dict[str, Any]):
        import pyzed.sl as sl

        zed_cfg = cfg["zed"]
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, zed_cfg["resolution"])
        init.depth_mode = getattr(sl.DEPTH_MODE, zed_cfg["depth_mode"])
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init.depth_minimum_distance = float(zed_cfg["depth_minimum_distance"])
        init.depth_maximum_distance = float(zed_cfg["depth_maximum_distance"])

        self.sl = sl
        self.camera = sl.Camera()
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        calibration = (
            self.camera.get_camera_information()
            .camera_configuration.calibration_parameters.left_cam
        )
        self.K = np.array(
            [
                [calibration.fx, 0.0, calibration.cx],
                [0.0, calibration.fy, calibration.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()

    def capture(self, warmup: int = 1) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(warmup):
            if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("ZED grab failed")

        self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT)
        self.camera.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)
        rgba = self.image.get_data()
        rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGB)
        depth = self.depth.get_data().copy()
        depth = np.nan_to_num(
            depth, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        return rgb, depth

    def close(self) -> None:
        self.camera.close()


class PolygonMaskEditor:
    """Small OpenCV polygon editor that returns a binary object mask."""

    WINDOW = "FoundationPose manual mask"

    def __init__(self):
        self.points: list[tuple[int, int]] = []

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def edit(self, rgb: np.ndarray) -> np.ndarray | None:
        self.points = []
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        h, w = rgb.shape[:2]

        while True:
            vis = rgb.copy()
            if self.points:
                pts = np.asarray(self.points, dtype=np.int32)
                if len(pts) >= 3:
                    tint = vis.copy()
                    cv2.fillPoly(tint, [pts], (0, 255, 80))
                    vis = cv2.addWeighted(vis, 0.60, tint, 0.40, 0)
                    cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
                else:
                    cv2.polylines(vis, [pts], False, (255, 255, 0), 2)
                for point in self.points:
                    cv2.circle(vis, point, 4, (255, 80, 80), -1)

            instructions = [
                "Click around cube silhouette (at least 3 points)",
                "LEFT add | RIGHT undo | c clear",
                "ENTER/SPACE register | q quit",
            ]
            for i, text in enumerate(instructions):
                y = 28 + 27 * i
                cv2.putText(
                    vis,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(self.WINDOW, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(self.WINDOW)
                return None
            if key == ord("c"):
                self.points.clear()
            if key in (13, ord(" ")) and len(self.points) >= 3:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(
                    mask,
                    [np.asarray(self.points, dtype=np.int32)],
                    1,
                )
                cv2.destroyWindow(self.WINDOW)
                return mask.astype(bool)


class FoundationPoseOnlyTracker:
    """Direct wrapper around the official FoundationPose register/track API."""

    def __init__(self, cfg: dict[str, Any], mesh_path: Path):
        fp_dir = Path(cfg["paths"]["foundationpose_dir"])
        if not fp_dir.is_absolute():
            fp_dir = (ROOT / fp_dir).resolve()
        else:
            fp_dir = fp_dir.resolve()
        if str(fp_dir) not in sys.path:
            sys.path.insert(0, str(fp_dir))

        # Import only FoundationPose and its own dependencies.
        import nvdiffrast.torch as dr
        from estimater import (
            FoundationPose,
            PoseRefinePredictor,
            ScorePredictor,
            set_logging_format,
            set_seed,
        )

        set_logging_format()
        set_seed(0)
        # FoundationPose prints a lot of INFO per frame; keep the UI responsive.
        logging.getLogger().setLevel(logging.WARNING)
        for noisy in ("estimater", "predict_pose_refine", "predict_score", "Utils"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        loaded = trimesh.load(mesh_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
        if not isinstance(loaded, trimesh.Trimesh):
            raise RuntimeError(f"Could not load triangle mesh: {mesh_path}")
        self.mesh = loaded
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.axis_scale = float(np.max(extents)) * 0.8

        fp_cfg = cfg["foundationpose"]
        os.environ["FOUNDATIONPOSE_RENDER_BS"] = str(
            fp_cfg.get("render_batch_size", 32)
        )
        os.environ["FOUNDATIONPOSE_WARP_CHUNK"] = str(
            fp_cfg.get("warp_chunk", 16)
        )
        debug_dir = ROOT / "debug" / "foundationpose_only"
        (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)
        (debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)
        self.debug_dir = debug_dir
        self.log_every = int(fp_cfg.get("log_every", 15))
        self.save_every = int(fp_cfg.get("save_every", 30))

        self.est_refine_iter = int(fp_cfg.get("est_refine_iter", 2))
        self.track_refine_iter = int(fp_cfg.get("track_refine_iter", 1))
        rotation_views = int(fp_cfg.get("rotation_views", 8))
        inplane_step = int(fp_cfg.get("inplane_step", 90))

        class FastFoundationPose(FoundationPose):
            """Fewer register hypotheses = much faster first-frame pose."""

            def __init__(self, *args, rotation_views: int = 8, inplane_step: int = 90, **kwargs):
                super().__init__(*args, **kwargs)
                self.make_rotation_grid(min_n_views=rotation_views, inplane_step=inplane_step)

        self.estimator = FastFoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=str(debug_dir),
            debug=int(fp_cfg.get("debug", 0)),
            glctx=dr.RasterizeCudaContext(),
            rotation_views=rotation_views,
            inplane_step=inplane_step,
        )
        logging.getLogger().setLevel(logging.INFO)
        logging.info(
            "FoundationPose ready | register_iter=%d track_iter=%d views=%d",
            self.est_refine_iter,
            self.track_refine_iter,
            rotation_views,
        )

    def register(
        self,
        K: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        return self.estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(bool),
            iteration=self.est_refine_iter,
        )

    def track(
        self, K: np.ndarray, rgb: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        return self.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=K,
            iteration=self.track_refine_iter,
        )

    @staticmethod
    def pose_xyz_euler(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return translation (m) and XYZ Euler angles (deg) from a 4x4 pose."""
        position = pose[:3, 3].copy()
        R = pose[:3, :3]
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0
        euler_deg = np.degrees(np.array([x, y, z], dtype=np.float64))
        return position, euler_deg

    @staticmethod
    def _project(pt: np.ndarray, K: np.ndarray, pose: np.ndarray) -> tuple[int, int] | None:
        cam = pose @ np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
        if cam[2] <= 1e-6:
            return None
        uv = K @ cam[:3]
        return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))

    def overlay(
        self, K: np.ndarray, rgb: np.ndarray, pose: np.ndarray
    ) -> np.ndarray:
        from Utils import draw_posed_3d_box, draw_xyz_axis

        center_pose = pose @ np.linalg.inv(self.to_origin)
        position, euler = self.pose_xyz_euler(pose)

        vis = draw_posed_3d_box(
            K,
            img=rgb.copy(),
            ob_in_cam=center_pose,
            bbox=self.bbox,
            line_color=(0, 255, 0),
            linewidth=2,
        )
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=self.axis_scale,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        # Axis end labels (RGB image: R=X, G=Y, B=Z)
        for name, tip, color in (
            ("X", np.array([self.axis_scale, 0.0, 0.0]), (255, 40, 40)),
            ("Y", np.array([0.0, self.axis_scale, 0.0]), (40, 220, 40)),
            ("Z", np.array([0.0, 0.0, self.axis_scale]), (40, 120, 255)),
        ):
            uv = self._project(tip, K, center_pose)
            if uv is None:
                continue
            cv2.putText(
                vis,
                name,
                (uv[0] + 4, uv[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        hud = [
            f"xyz m:  [{position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}]",
            f"rpy deg:[{euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}]",
            "axes: R=X  G=Y  B=Z   |  q quit  r redraw mask",
        ]
        for i, line in enumerate(hud):
            y = 28 + i * 26
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return vis


def resolve_mesh(cfg: dict[str, Any], override: Path | None) -> Path:
    path = override or Path(cfg["paths"]["mesh_file"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mesh", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    mesh_path = resolve_mesh(cfg, args.mesh)

    camera = ZedCamera(cfg)
    try:
        tracker = FoundationPoseOnlyTracker(cfg, mesh_path)
        editor = PolygonMaskEditor()
        tracking = False
        frame_number = 0

        while True:
            rgb, depth = camera.capture(warmup=2 if not tracking else 1)

            if not tracking:
                mask = editor.edit(rgb)
                if mask is None:
                    break

                valid_depth = depth[mask]
                valid_depth = valid_depth[valid_depth > 0]
                if len(valid_depth) < 20:
                    logging.error(
                        "Mask has too few valid depth pixels. Redraw it or move "
                        "the cube into the ZED depth range."
                    )
                    continue

                cv2.imwrite(
                    str(tracker.debug_dir / "initial_mask.png"),
                    mask.astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    str(tracker.debug_dir / "initial_rgb.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
                pose = tracker.register(camera.K, rgb, depth, mask)
                tracking = True
                logging.info(
                    "Registered. Tracking no longer uses a mask. "
                    "Press r to draw a new one."
                )
            else:
                pose = tracker.track(camera.K, rgb, depth)

            xyz, rpy = tracker.pose_xyz_euler(pose)
            if frame_number % tracker.log_every == 0:
                logging.info(
                    "xyz=[%.4f, %.4f, %.4f] m  rpy=[%.1f, %.1f, %.1f] deg",
                    *xyz,
                    *rpy,
                )
                np.savetxt(
                    tracker.debug_dir / "ob_in_cam" / f"{frame_number:06d}.txt",
                    pose.reshape(4, 4),
                )
            vis = tracker.overlay(camera.K, rgb, pose)
            # Show FPS-ish timing every log_every frames via HUD already in overlay
            cv2.imshow("FoundationPose tracking", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if frame_number % tracker.save_every == 0:
                cv2.imwrite(
                    str(
                        tracker.debug_dir
                        / "track_vis"
                        / f"{frame_number:06d}.jpg"
                    ),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                )
            frame_number += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.estimator.pose_last = None
                tracking = False

        cv2.destroyAllWindows()
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
