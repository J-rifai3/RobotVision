#!/usr/bin/env python3
"""Model-based FoundationPose accuracy test for a physical cube + SolidWorks CAD.

Uses the candle-scan conda env (ZED + Grounded-SAM-2 + FoundationPose) but loads
your CAD mesh instead of reconstructing from depth.

Workflow:
  1. Export SolidWorks part → mesh/cube_raw.stl
  2. python scripts/prepare_mesh.py
  3. conda activate candle-scan
  4. python scan_cube.py scan          # one-shot pose + overlay image
  5. python scan_cube.py live          # continuous tracking with on-screen markers
  6. Optionally save gt/ob_in_cam.txt and run scripts/evaluate_accuracy.py

Controls (live / scan register):
  Put the cube inside the yellow center box, then:
    SPACE / ENTER  capture mask & lock pose
    [ / ]          shrink / grow the box
    r              re-register (back to box preview; works during tracking too)
    q              quit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FACE_LABELS = (
    ("+X", np.array([1.0, 0.0, 0.0]), (0, 0, 255)),
    ("-X", np.array([-1.0, 0.0, 0.0]), (0, 80, 255)),
    ("+Y", np.array([0.0, 1.0, 0.0]), (0, 255, 0)),
    ("-Y", np.array([0.0, -1.0, 0.0]), (0, 180, 0)),
    ("+Z", np.array([0.0, 0.0, 1.0]), (255, 80, 0)),
    ("-Z", np.array([0.0, 0.0, -1.0]), (255, 160, 0)),
)


def _load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (ROOT / "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(value: str | Path) -> Path:
    """Resolve config paths relative to apps/cube/ when not absolute."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _bootstrap_candle_scan(cfg: dict[str, Any]):
    sam_root = _resolve_path(cfg["paths"]["sam_foundation_dir"])
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))

    import scan_candle as sc

    def _ensure_paths(_cfg: dict[str, Any] | None = None) -> dict[str, Path]:
        c = _cfg or cfg
        p = c["paths"]
        return {
            "grounded_sam2": _resolve_path(p["grounded_sam2_dir"]),
            "foundationpose": _resolve_path(p["foundationpose_dir"]),
            "sam2_checkpoint": _resolve_path(p["sam2_checkpoint"]),
            "sam2_config": p["sam2_config"],
            "fp_weights": _resolve_path(p["fp_weights_dir"]),
            "work_dir": _resolve_path(p.get("work_dir", ".")),
        }

    sc._ensure_paths = _ensure_paths  # type: ignore[attr-defined]
    return sc


def _project_point(pt_obj: np.ndarray, K: np.ndarray, ob_in_cam: np.ndarray) -> tuple[int, int] | None:
    homo = np.array([pt_obj[0], pt_obj[1], pt_obj[2], 1.0], dtype=np.float64)
    cam = ob_in_cam @ homo
    if cam[2] <= 1e-6:
        return None
    uv = K @ cam[:3]
    return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))


def _face_visible(direction: np.ndarray, pt: np.ndarray, center_pose: np.ndarray) -> bool:
    face_normal_cam = center_pose[:3, :3] @ direction
    face_center_cam = (center_pose @ np.array([pt[0], pt[1], pt[2], 1.0]))[:3]
    return float(np.dot(face_normal_cam, -face_center_cam)) > 0


def draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray | None = None,
    label: str | None = None,
    alpha: float = 0.45,
    hud: bool = True,
) -> np.ndarray:
    vis = rgb.copy()
    color = np.array([0, 255, 120], dtype=np.float32)
    m = mask.astype(bool)
    if np.any(m):
        blend = vis[m].astype(np.float32) * (1.0 - alpha) + color * alpha
        vis[m] = np.clip(blend, 0, 255).astype(np.uint8)
        mask_u8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    if box is not None and len(box) >= 4:
        x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)

    if hud:
        lines = ["MASK CHECK: green = selected object"]
        if label:
            lines.append(f"label: {label}")
        lines.append("SPACE accept   r back to box   q quit")
        for i, line in enumerate(lines):
            y = 28 + i * 26
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_cube_overlay(
    rgb: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    to_origin: np.ndarray,
    bbox: np.ndarray,
    position: np.ndarray,
    euler_deg: np.ndarray,
    axis_scale: float | None = None,
) -> np.ndarray:
    from Utils import draw_posed_3d_box, draw_xyz_axis

    center_pose = pose @ np.linalg.inv(to_origin)
    extents = bbox[1] - bbox[0]
    half = extents * 0.5
    if axis_scale is None:
        axis_scale = float(np.max(extents)) * 0.75

    vis = rgb.copy()
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 0), linewidth=2)
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=axis_scale,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )

    h, w = vis.shape[:2]
    for name, direction, color_bgr in FACE_LABELS:
        pt = direction * half
        if not _face_visible(direction, pt, center_pose):
            continue
        uv = _project_point(pt, K, center_pose)
        if uv is None:
            continue
        u, v = uv
        if not (0 <= u < w and 0 <= v < h):
            continue
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        cv2.circle(vis, (u, v), 6, color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, name, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb, 2, cv2.LINE_AA)

    for name, tip, color_bgr in (
        ("X", np.array([axis_scale, 0.0, 0.0]), (0, 0, 255)),
        ("Y", np.array([0.0, axis_scale, 0.0]), (0, 255, 0)),
        ("Z", np.array([0.0, 0.0, axis_scale]), (255, 0, 0)),
    ):
        uv = _project_point(tip, K, center_pose)
        if uv is None:
            continue
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        cv2.putText(vis, name, (uv[0] + 4, uv[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_rgb, 2, cv2.LINE_AA)

    hud = [
        f"pos m: [{position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}]",
        f"euler deg: [{euler_deg[0]:+.1f}, {euler_deg[1]:+.1f}, {euler_deg[2]:+.1f}]",
        "R=X  G=Y  B=Z   |  q quit  r re-register",
    ]
    for i, line in enumerate(hud):
        y = 28 + i * 26
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


class CubeScanner:
    """CAD mesh + ZED + center-box SAM register + FoundationPose track."""

    def __init__(self, config_path: Path | None = None, text_prompt: str | None = None):
        self.cfg = _load_config(config_path)
        if text_prompt is not None:
            self.cfg["segmentation"]["text_prompt"] = text_prompt

        self.sc = _bootstrap_candle_scan(self.cfg)
        self.sc._load_config = lambda _p=None: self.cfg  # type: ignore[attr-defined]

        self.scanner = self.sc.CandleScanner(
            config_path=ROOT / "config.yaml",
            text_prompt=self.cfg["segmentation"]["text_prompt"],
        )
        self.scanner.cfg = self.cfg
        self.scanner.paths = self.sc._ensure_paths(self.cfg)
        self.sc._setup_import_paths(self.scanner.paths)

        self._box_frac = float(self.cfg["segmentation"].get("center_box_frac", 0.35))

    def load_cad_mesh(self, mesh_path: Path | None = None) -> trimesh.Trimesh:
        path = Path(mesh_path) if mesh_path else ROOT / self.cfg["paths"]["mesh_file"]
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"CAD mesh not found: {path}\n"
                "1) Export SolidWorks → mesh/cube_raw.stl\n"
                "2) python scripts/prepare_mesh.py"
            )
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"Failed to load mesh from {path}")

        extents = mesh.bounding_box.extents
        logging.info("Loaded CAD mesh %s extents=%s m", path, extents)
        self.scanner._fp_mesh = mesh
        # Defer FoundationPose init until after first mask (saves VRAM for SAM).
        # Still compute bbox/to_origin for overlays after FP init.
        return mesh

    def close(self) -> None:
        self.scanner.close()

    @staticmethod
    def _flush_keys(settle_ms: int = 120) -> None:
        end = time.monotonic() + settle_ms * 0.001
        while time.monotonic() < end:
            cv2.waitKey(1)

    def _center_box(self, h: int, w: int) -> np.ndarray:
        side = int(min(h, w) * self._box_frac)
        side = max(20, side)
        x0 = int((w - side) * 0.5)
        y0 = int((h - side) * 0.5)
        return np.array([x0, y0, x0 + side, y0 + side], dtype=np.float32)

    def _ensure_sam2_only(self) -> None:
        """Load SAM2 only (no Grounding DINO). Unloads FoundationPose first for VRAM."""
        if self.scanner._exclusive_gpu_models():
            self.scanner._unload_foundationpose()

        if self.scanner._sam2_predictor is not None:
            return

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.scanner._free_cuda_memory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scanner._device = device
        self.scanner._use_bf16 = device == "cuda" and torch.cuda.get_device_properties(0).major >= 8

        logging.info("Loading SAM2 only (no Grounding DINO)...")
        sam2_model = build_sam2(
            self.scanner.paths["sam2_config"],
            str(self.scanner.paths["sam2_checkpoint"]),
            device=device,
        )
        self.scanner._sam2_predictor = SAM2ImagePredictor(sam2_model)
        # Mark loaded so later unload clears SAM2; grounding stays None.
        self.scanner._segmentation_loaded = True
        self.scanner._active_gpu_stack = "segmentation"
        self.scanner._log_gpu_memory("after SAM2-only load")

    def _sam_center_mask(self, frame) -> tuple[np.ndarray, dict[str, Any]]:
        self._ensure_sam2_only()
        rgb = frame.rgb
        h, w = rgb.shape[:2]
        box = self._center_box(h, w)

        self.scanner._sam2_predictor.set_image(rgb)
        with self.scanner._segmentation_autocast():
            masks, mask_scores, _ = self.scanner._sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask = masks[0].astype(bool)
        if not np.any(mask):
            raise RuntimeError("SAM returned an empty mask. Center the cube in the yellow box.")

        meta = {
            "boxes": box.reshape(1, 4),
            "scores": np.array([1.0], dtype=np.float32),
            "labels": ["center-box"],
            "mask_score": float(mask_scores[0]),
            "prompt": "center-box",
            "pick_mode": "center_box",
            "dist_norm": 0.0,
            "area_frac": float(mask.mean()),
        }
        return mask, meta

    def _overlay(self, frame, result, mask: np.ndarray | None = None) -> np.ndarray:
        vis = draw_cube_overlay(
            rgb=frame.rgb,
            K=frame.K,
            pose=result.pose,
            to_origin=self.scanner._fp_to_origin,
            bbox=self.scanner._fp_bbox,
            position=result.position,
            euler_deg=self.scanner.rotation_matrix_to_euler_xyz(result.rotation_matrix),
        )
        if mask is not None and self.cfg["segmentation"].get("show_mask", True):
            faint = draw_mask_overlay(vis, mask, alpha=0.22, hud=False)
            m = mask.astype(bool)
            vis[m] = faint[m]
        return vis

    def _draw_box_preview(self, rgb: np.ndarray) -> np.ndarray:
        vis = rgb.copy()
        h, w = vis.shape[:2]
        box = self._center_box(h, w)
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
        lines = [
            "Put CUBE inside yellow box",
            f"box size={self._box_frac:.0%}   [ shrink   ] grow",
            "SPACE capture mask   r reset   q quit",
        ]
        for i, line in enumerate(lines):
            y = 28 + i * 26
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    def _segment_until_accepted(self):
        """Live center-box registration. No Grounding DINO. r always returns to this UI."""
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        confirm = bool(self.cfg["segmentation"].get("confirm_mask", True))

        logging.info(
            "Center-box register: put cube in yellow box, SPACE to mask. "
            "[ / ] resize box. r resets. q quits."
        )
        self._flush_keys(200)

        while True:
            frame = self.scanner.capture_frame()
            preview = self._draw_box_preview(frame.rgb)
            cv2.imshow("cube scan", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return None, None, None
            if key == ord("["):
                self._box_frac = max(0.12, self._box_frac - 0.05)
                continue
            if key == ord("]"):
                self._box_frac = min(0.80, self._box_frac + 0.05)
                continue
            if key == ord("r"):
                # Already in register UI — ignore (user wants a clean reset of preview).
                continue
            if key not in (ord(" "), 13):
                continue

            # SPACE: run SAM on current frame's center box
            try:
                mask, meta = self._sam_center_mask(frame)
            except Exception as exc:  # noqa: BLE001
                logging.error("SAM mask failed: %s", exc)
                err = self._draw_box_preview(frame.rgb)
                cv2.putText(
                    err,
                    f"SAM failed: {str(exc)[:70]}",
                    (12, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 80, 80),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("cube scan", cv2.cvtColor(err, cv2.COLOR_RGB2BGR))
                self._flush_keys(300)
                # Wait for any key then return to live box preview (do NOT require r)
                cv2.waitKey(800)
                continue

            box = meta["boxes"][0]
            mask_vis = draw_mask_overlay(frame.rgb, mask, box=box, label=str(meta["labels"][0]))
            mask_path = debug_dir / "last_mask.jpg"
            cv2.imwrite(str(mask_path), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
            logging.info("Mask saved → %s  area_frac=%.3f", mask_path, meta["area_frac"])

            if not confirm:
                return frame, mask, meta

            cv2.imshow("cube scan", cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
            self._flush_keys(250)
            print("Mask check: SPACE accept, r back to box, q quit")

            while True:
                key2 = cv2.waitKey(30) & 0xFF
                if key2 in (ord(" "), 13):
                    self._flush_keys(150)
                    return frame, mask, meta
                if key2 == ord("r"):
                    logging.info("Back to center-box preview")
                    self._flush_keys(200)
                    break  # outer loop → live preview again
                if key2 == ord("q"):
                    return None, None, None

    def scan_once(self, save_pose: bool = True) -> Any:
        frame, mask, meta = self._segment_until_accepted()
        if frame is None:
            return None
        result = self.scanner.estimate_pose(frame, mask=mask)

        vis = self._overlay(frame, result, mask=mask)
        out_dir = ROOT / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)
        vis_path = out_dir / "overlay_0000.jpg"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        if save_pose:
            out = out_dir / "ob_in_cam" / "0000.txt"
            np.savetxt(out, result.pose.reshape(4, 4))
            pos = result.position
            print(f"detected: {meta['labels'][0]}")
            print(f"position (m): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"overlay image: {vis_path}")

        cv2.imshow("cube scan", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("Press any key in the window to close...")
        self._flush_keys(100)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result

    def live(self) -> None:
        logging.info("Live cube tracking. r = re-register (center box). q = quit.")
        # Quiet FoundationPose's per-frame INFO spam during tracking.
        for noisy in ("estimater", "predict_pose_refine", "predict_score", "Utils"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        tracking = False
        last_mask: np.ndarray | None = None
        frame_i = 0
        ignore_r_until = 0.0
        fp_cfg = self.cfg.get("foundationpose", {})
        log_every = int(fp_cfg.get("log_every", 15))
        save_every = int(fp_cfg.get("save_every", 30))
        pose_dir = ROOT / "debug" / "ob_in_cam"
        vis_dir = ROOT / "debug" / "track_vis"
        pose_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        while True:
            if not tracking:
                frame, mask, meta = self._segment_until_accepted()
                if frame is None:
                    break
                last_mask = mask
                result = self.scanner.estimate_pose(frame, mask=mask)
                tracking = True
                self._flush_keys(300)
                ignore_r_until = time.monotonic() + 1.0
                logging.info("Locked — tracking. Press r to re-register.")
            else:
                frame = self.scanner.capture_frame()
                result = self.scanner.track_pose(frame)

            if frame_i % log_every == 0:
                euler = self.scanner.rotation_matrix_to_euler_xyz(result.rotation_matrix)
                logging.info(
                    "pos=[%.3f, %.3f, %.3f] m  euler_xyz=[%.1f, %.1f, %.1f] deg",
                    *result.position,
                    *euler,
                )
                np.savetxt(pose_dir / f"{frame_i:04d}.txt", result.pose.reshape(4, 4))

            vis = self._overlay(frame, result, mask=last_mask)
            cv2.imshow("cube scan", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            if frame_i % save_every == 0:
                cv2.imwrite(str(vis_dir / f"{frame_i:04d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and time.monotonic() >= ignore_r_until:
                logging.info("Re-register: returning to center-box UI")
                tracking = False
                last_mask = None
                if self.scanner._fp_estimator is not None:
                    self.scanner._fp_estimator.pose_last = None
                self._flush_keys(250)

        cv2.destroyAllWindows()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--prompt", type=str, default=None, help="Unused in center-box mode")
    p.add_argument("--mesh", type=Path, default=None, help="Override CAD mesh path")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="One-shot pose estimate with overlay")
    sub.add_parser("live", help="Live tracking with box / axes / face labels")
    sub.add_parser("save-gt", help="One estimate copied to gt/ob_in_cam.txt")
    args = p.parse_args()

    cube = CubeScanner(config_path=args.config, text_prompt=args.prompt)
    try:
        cube.load_cad_mesh(args.mesh)
        if args.command == "scan":
            cube.scan_once()
        elif args.command == "live":
            cube.live()
        elif args.command == "save-gt":
            result = cube.scan_once(save_pose=True)
            if result is None:
                return 1
            gt_path = ROOT / cube.cfg["accuracy"]["gt_pose_file"]
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(gt_path, result.pose.reshape(4, 4))
            print(f"\nWrote gt/ pose → {gt_path}")
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    finally:
        cube.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
