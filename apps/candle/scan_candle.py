"""
Modeless candle 6D pose estimation with a ZED camera.

Pipeline:
  1. Grounded-SAM-2 segments the candle from a text prompt (no trained detector).
  2. FoundationPose estimates 6D pose using a mesh built on-the-fly from depth
     (model-free: no CAD model required).

Usage:
  conda activate candle-scan
  ./setup.sh   # first time only
  python scan_candle.py capture-refs --num-views 8
  python scan_candle.py scan
  python scan_candle.py scan --live
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent

# Reduce CUDA fragmentation on 8 GB GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = cfg["paths"]
    return {
        "grounded_sam2": (ROOT / paths["grounded_sam2_dir"]).resolve(),
        "foundationpose": (ROOT / paths["foundationpose_dir"]).resolve(),
        "sam2_checkpoint": (ROOT / paths["sam2_checkpoint"]).resolve(),
        "sam2_config": paths["sam2_config"],
        "fp_weights": (ROOT / paths["fp_weights_dir"]).resolve(),
        "work_dir": (ROOT / paths["work_dir"]).resolve(),
    }


def _setup_import_paths(paths: dict[str, Path]) -> None:
    gsam = str(paths["grounded_sam2"])
    fp = str(paths["foundationpose"])
    for entry in (gsam, fp):
        if entry not in sys.path:
            sys.path.insert(0, entry)


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    mask: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PoseResult:
    pose: np.ndarray
    position: np.ndarray
    rotation_matrix: np.ndarray
    score: float | None = None


class CandleScanner:
  """ZED + Grounded-SAM-2 + model-free FoundationPose in one object."""

  def __init__(
      self,
      config_path: str | Path | None = None,
      text_prompt: str | None = None,
      mesh_mode: str | None = None,
      debug: int | None = None,
  ):
    self.cfg = _load_config(Path(config_path) if config_path else None)
    self.paths = _ensure_paths(self.cfg)
    self.paths["work_dir"].mkdir(parents=True, exist_ok=True)
    _setup_import_paths(self.paths)

    if text_prompt is not None:
      self.cfg["segmentation"]["text_prompt"] = text_prompt
    if mesh_mode is not None:
      self.cfg["reconstruction"]["mode"] = mesh_mode
    if debug is not None:
      self.cfg["foundationpose"]["debug"] = debug

    self._zed: Any = None
    self._runtime: Any = None
    self._image_mat: Any = None
    self._depth_mat: Any = None
    self._K: np.ndarray | None = None

    self._sam2_predictor = None
    self._grounding_processor = None
    self._grounding_model = None
    self._device: str = "cpu"

    self._fp_estimator = None
    self._fp_mesh: trimesh.Trimesh | None = None
    self._fp_to_origin: np.ndarray | None = None
    self._fp_bbox: np.ndarray | None = None

    self._reference_views: list[Frame] = []
    self._reference_poses: list[np.ndarray] = []
    self._segmentation_loaded = False
    self._grounding_device = "cpu"
    self._active_gpu_stack: str | None = None

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    self._apply_vram_defaults()
    self._init_zed()
    if not self.cfg.get("memory", {}).get("lazy_load_segmentation", True):
      self._load_segmentation_models()

  def _apply_vram_defaults(self) -> None:
    import torch

    if not torch.cuda.is_available():
      return

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    fp = self.cfg.setdefault("foundationpose", {})
    mem = self.cfg.setdefault("memory", {})

    if vram_gb < 10:
      fp.setdefault("render_batch_size", 16)
      fp.setdefault("warp_chunk", 8)
      fp.setdefault("rotation_views", 10)
      fp.setdefault("inplane_step", 90)
      fp.setdefault("est_refine_iter", 2)
      fp.setdefault("track_refine_iter", 1)
      mem.setdefault("lazy_load_segmentation", True)
      mem.setdefault("unload_segmentation_during_pose", True)
      mem.setdefault("exclusive_gpu_models", True)
      mem.setdefault("grounding_on_cpu", True)
      seg = self.cfg.setdefault("segmentation", {})
      seg.setdefault("grounding_device", "cpu")
      logging.info("Low VRAM mode (%.1f GB): exclusive model loading enabled", vram_gb)

  def _log_gpu_memory(self, label: str) -> None:
    import torch

    if not torch.cuda.is_available():
      return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / (1024**3)
    logging.info(
        "%s | GPU active=%s | allocated=%.2f GiB | free=%.2f/%.2f GiB",
        label,
        self._active_gpu_stack,
        alloc,
        free / (1024**3),
        total / (1024**3),
    )

  def _exclusive_gpu_models(self) -> bool:
    return bool(self.cfg.get("memory", {}).get("exclusive_gpu_models", True))

  def _unload_foundationpose(self) -> None:
    if self._fp_estimator is None:
      return
    del self._fp_estimator
    self._fp_estimator = None
    if self._active_gpu_stack == "foundationpose":
      self._active_gpu_stack = None
    self._free_cuda_memory()
    logging.info("Unloaded FoundationPose to free GPU memory")
    self._log_gpu_memory("after FoundationPose unload")

  def _ensure_foundationpose_ready(self) -> None:
    if self._fp_estimator is not None:
      return
    if self._fp_mesh is None:
      raise RuntimeError("No object mesh loaded. Run build-mesh or calibrate_object first.")
    if self._exclusive_gpu_models():
      self._unload_segmentation_models()
    self._init_foundationpose(self._fp_mesh)

  def _load_segmentation_models(self) -> None:
    if self._segmentation_loaded:
      return
    if self._exclusive_gpu_models():
      self._unload_foundationpose()
    self._init_grounded_sam2()
    self._segmentation_loaded = True
    self._active_gpu_stack = "segmentation"
    self._log_gpu_memory("after segmentation load")

  def _free_cuda_memory(self) -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  def _unload_segmentation_models(self) -> None:
    if not self._segmentation_loaded:
      return
    if self._sam2_predictor is not None:
      del self._sam2_predictor
      self._sam2_predictor = None
    if self._grounding_model is not None:
      del self._grounding_model
      self._grounding_model = None
    self._segmentation_loaded = False
    if self._active_gpu_stack == "segmentation":
      self._active_gpu_stack = None
    self._free_cuda_memory()
    logging.info("Unloaded segmentation models to free GPU memory")
    self._log_gpu_memory("after segmentation unload")

  def _configure_foundationpose_vram(self) -> None:
    fp = self.cfg.get("foundationpose", {})
    os.environ["FOUNDATIONPOSE_RENDER_BS"] = str(fp.get("render_batch_size", 16))
    os.environ["FOUNDATIONPOSE_WARP_CHUNK"] = str(fp.get("warp_chunk", 8))
    logging.info(
        "FoundationPose VRAM settings: render_bs=%s warp_chunk=%s rotation_views=%s",
        os.environ["FOUNDATIONPOSE_RENDER_BS"],
        os.environ["FOUNDATIONPOSE_WARP_CHUNK"],
        fp.get("rotation_views", 10),
    )

  def _init_zed(self) -> None:
    try:
      import pyzed.sl as sl
    except ImportError as exc:
      raise RuntimeError(
          "Stereolabs ZED Python API is not installed. "
          "Do NOT use `pip install pyzed` (wrong package). "
          "Install the ZED SDK, then run: python /usr/local/zed/get_python_api.py"
      ) from exc

    zed_cfg = self.cfg["zed"]
    resolution = getattr(sl.RESOLUTION, zed_cfg["resolution"])
    depth_mode = getattr(sl.DEPTH_MODE, zed_cfg["depth_mode"])

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.depth_mode = depth_mode
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_minimum_distance = zed_cfg["depth_minimum_distance"]
    init_params.depth_maximum_distance = zed_cfg["depth_maximum_distance"]

    camera = sl.Camera()
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
      raise RuntimeError(f"Failed to open ZED camera: {status}")

    calib = camera.get_camera_information().camera_configuration.calibration_parameters
    left = calib.left_cam
    self._K = np.array(
        [[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    self._zed = camera
    self._runtime = sl.RuntimeParameters()
    self._image_mat = sl.Mat()
    self._depth_mat = sl.Mat()
    logging.info("ZED camera ready (%s, %s)", zed_cfg["resolution"], zed_cfg["depth_mode"])

  def _init_grounded_sam2(self) -> None:
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    self._use_bf16 = False
    seg = self.cfg["segmentation"]
    mem = self.cfg.get("memory", {})
    if mem.get("grounding_on_cpu", False):
      self._grounding_device = "cpu"
    else:
      self._grounding_device = seg.get("grounding_device", self._device)

    if self._device == "cuda":
      self._use_bf16 = torch.cuda.get_device_properties(0).major >= 8
      if self._use_bf16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(
        self.paths["sam2_config"],
        str(self.paths["sam2_checkpoint"]),
        device=self._device,
    )
    self._sam2_predictor = SAM2ImagePredictor(sam2_model)

    model_id = seg["grounding_model"]
    self._grounding_processor = AutoProcessor.from_pretrained(model_id)
    self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        self._grounding_device
    )
    logging.info(
        "Grounded-SAM-2 ready | sam2=%s grounding=%s (prompt: %r)",
        self._device,
        self._grounding_device,
        seg["text_prompt"],
    )

  def _init_foundationpose(self, mesh: trimesh.Trimesh) -> None:
    if self._fp_estimator is not None:
      return

    if self.cfg.get("memory", {}).get("unload_segmentation_during_pose", True):
      self._unload_segmentation_models()

    self._configure_foundationpose_vram()
    self._free_cuda_memory()

    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_logging_format, set_seed

    class CandleFoundationPose(FoundationPose):
      """Fewer rotation hypotheses for 8 GB GPUs."""

      def __init__(self, *args, rotation_views: int = 10, inplane_step: int = 90, **kwargs):
        super().__init__(*args, **kwargs)
        self.make_rotation_grid(min_n_views=rotation_views, inplane_step=inplane_step)

    set_logging_format()
    set_seed(0)

    fp_cfg = self.cfg["foundationpose"]
    debug_dir = str((self.paths["work_dir"] / "debug").resolve())
    os.makedirs(debug_dir, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    self._fp_mesh = mesh
    self._fp_to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    self._fp_bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    self._fp_estimator = CandleFoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=fp_cfg["debug"],
        glctx=glctx,
        rotation_views=int(fp_cfg.get("rotation_views", 10)),
        inplane_step=int(fp_cfg.get("inplane_step", 90)),
    )
    self._active_gpu_stack = "foundationpose"
    logging.info("FoundationPose ready (diameter %.3f m)", self._fp_estimator.diameter)
    self._log_gpu_memory("after FoundationPose load")

  # ------------------------------------------------------------------
  # ZED capture
  # ------------------------------------------------------------------

  def capture_frame(self, warmup: int = 5) -> Frame:
    import pyzed.sl as sl

    for _ in range(warmup):
      if self._zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED grab failed during warmup")

    if self._zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
      raise RuntimeError("ZED grab failed")

    self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
    self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)

    rgba = self._image_mat.get_data()
    rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGB)
    depth = self._depth_mat.get_data().copy()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return Frame(rgb=rgb, depth=depth, K=self._K.copy())

  def close(self) -> None:
    if self._zed is not None:
      self._zed.close()
      self._zed = None

  # ------------------------------------------------------------------
  # Grounded-SAM-2 segmentation
  # ------------------------------------------------------------------

  def _segmentation_autocast(self):
    import torch

    if self._device == "cuda" and self._use_bf16:
      return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)

  def _foundationpose_autocast(self):
    import torch

    # FoundationPose uses matrix inverses that break under bfloat16 autocast.
    return torch.autocast(
        device_type="cuda" if self._device == "cuda" else "cpu",
        enabled=False,
    )

  def _grounding_detect(
      self,
      rgb: np.ndarray,
      prompt: str,
      box_threshold: float,
      text_threshold: float,
  ) -> dict[str, Any]:
    import torch
    from PIL import Image

    image = Image.fromarray(rgb)
    inputs = self._grounding_processor(images=image, text=prompt, return_tensors="pt").to(
        self._grounding_device
    )
    with self._segmentation_autocast():
      with torch.no_grad():
        outputs = self._grounding_model(**inputs)

    return self._grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

  def _normalize_prompt(self, prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
      prompt += "."
    return prompt

  def _prompt_candidates(self, text_prompt: str | None = None) -> list[str]:
    seg = self.cfg["segmentation"]
    primary = self._normalize_prompt(text_prompt or seg["text_prompt"])
    prompts = [primary]
    for item in seg.get("fallback_prompts", []):
      normalized = self._normalize_prompt(item)
      if normalized not in prompts:
        prompts.append(normalized)
    return prompts

  def _save_detection_debug(self, rgb: np.ndarray, tag: str) -> Path:
    debug_dir = self.paths["work_dir"] / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{tag}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path

  def segment_candle(
      self,
      frame: Frame | np.ndarray,
      text_prompt: str | None = None,
      pick_largest: bool | None = None,
  ) -> tuple[np.ndarray, dict[str, Any]]:
    self._load_segmentation_models()
    rgb = frame.rgb if isinstance(frame, Frame) else frame
    seg = self.cfg["segmentation"]
    box_threshold = float(seg["box_threshold"])
    text_threshold = float(seg["text_threshold"])
    min_box = float(seg.get("min_box_threshold", max(0.1, box_threshold - 0.15)))
    min_text = float(seg.get("min_text_threshold", max(0.1, text_threshold - 0.1)))

    # pick_mode: largest | center | center_score
    if pick_largest is True:
      pick_mode = "largest"
    elif pick_largest is False:
      pick_mode = str(seg.get("pick_mode", "center"))
    else:
      pick_mode = str(seg.get("pick_mode", "largest"))

    self._sam2_predictor.set_image(rgb)

    results = None
    used_prompt = None
    used_box_threshold = box_threshold
    used_text_threshold = text_threshold

    threshold_steps = (
        (box_threshold, text_threshold),
        (max(min_box, box_threshold * 0.75), max(min_text, text_threshold * 0.75)),
        (min_box, min_text),
    )

    for prompt in self._prompt_candidates(text_prompt):
      for box_t, text_t in threshold_steps:
        candidate = self._grounding_detect(rgb, prompt, box_t, text_t)
        if len(candidate["boxes"]) > 0:
          results = candidate
          used_prompt = prompt
          used_box_threshold = box_t
          used_text_threshold = text_t
          break
      if results is not None:
        break

    if results is None:
      debug_path = self._save_detection_debug(rgb, "no_detection")
      tried = ", ".join(repr(p) for p in self._prompt_candidates(text_prompt))
      raise RuntimeError(
          f"No detections for prompts [{tried}] "
          f"(box>={min_box}, text>={min_text}). "
          f"Saved camera frame to {debug_path}. "
          "Try better lighting, move the object closer, or pass --prompt."
      )

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    h, w = rgb.shape[:2]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_frac = areas / float(h * w)
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    img_cx, img_cy = 0.5 * w, 0.5 * h
    dist_norm = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2) / (
        np.sqrt(img_cx**2 + img_cy**2) + 1e-6
    )

    min_area = float(seg.get("min_area_frac", 0.0005))
    max_area = float(seg.get("max_area_frac", 0.35))
    valid = (area_frac >= min_area) & (area_frac <= max_area)
    if not np.any(valid):
      logging.warning(
          "No detections in area_frac [%.4f, %.2f]; using all %d boxes",
          min_area,
          max_area,
          len(boxes),
      )
      valid = np.ones(len(boxes), dtype=bool)

    if pick_mode == "center":
      # Nearest to image center among size-filtered boxes
      ranking = np.where(valid, dist_norm, np.inf)
      idx = int(np.argmin(ranking))
    elif pick_mode == "center_score":
      # High detector score near center; lightly prefer compact objects
      center_weight = float(seg.get("center_weight", 3.0))
      ranking = np.where(
          valid,
          scores * ((1.0 - dist_norm) ** 2) / (1.0 + area_frac),
          -np.inf,
      )
      # Optional extra center emphasis
      ranking = np.where(valid, ranking * (1.0 + center_weight * (1.0 - dist_norm)), -np.inf)
      idx = int(np.argmax(ranking))
    else:
      # largest (legacy): area * score
      ranking = np.where(valid, areas * scores, -np.inf)
      idx = int(np.argmax(ranking))

    boxes = boxes[idx : idx + 1]
    scores = scores[idx : idx + 1]
    labels = [labels[idx]]
    chosen_dist = float(dist_norm[idx])
    chosen_area = float(area_frac[idx])

    with self._segmentation_autocast():
      masks, mask_scores, _ = self._sam2_predictor.predict(
          point_coords=None,
          point_labels=None,
          box=boxes,
          multimask_output=False,
      )
    if masks.ndim == 4:
      masks = masks.squeeze(1)
    mask = masks[0].astype(bool)

    logging.info(
        "Detected %r score=%.2f pick=%s dist_norm=%.2f area_frac=%.3f "
        "(box_thresh=%.2f, text_thresh=%.2f)",
        labels[0],
        float(scores[0]),
        pick_mode,
        chosen_dist,
        chosen_area,
        used_box_threshold,
        used_text_threshold,
    )

    meta = {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "mask_score": float(mask_scores[0]),
        "prompt": used_prompt,
        "box_threshold": used_box_threshold,
        "text_threshold": used_text_threshold,
        "pick_mode": pick_mode,
        "dist_norm": chosen_dist,
        "area_frac": chosen_area,
    }
    return mask, meta

  # ------------------------------------------------------------------
  # Model-free mesh reconstruction
  # ------------------------------------------------------------------

  def capture_reference_views(
      self,
      num_views: int | None = None,
      interval_s: float | None = None,
      text_prompt: str | None = None,
  ) -> list[Frame]:
    if self._exclusive_gpu_models():
      self._unload_foundationpose()

    rec = self.cfg["reconstruction"]
    if num_views is None:
      if rec["mode"] == "fast":
        n = int(rec.get("fast_num_reference_views", 1))
      else:
        n = int(rec["num_reference_views"])
    else:
      n = num_views
    wait_s = interval_s or rec["reference_capture_interval_s"]

    if rec["mode"] == "fast" and n == 1:
      logging.info(
          "Fast mode: capture 1 view. Keep the candle still in front of the camera."
      )
    else:
      logging.info(
          "Capture %d views. Leave the candle in the SAME place on the table and "
          "only ROTATE it a little between shots. Do not slide it to new locations.",
          n,
      )
    views: list[Frame] = []
    poses: list[np.ndarray] = []

    for i in range(n):
      while True:
        prompt = (
            f"[{i + 1}/{n}] Rotate the candle slightly in place, then press Enter..."
            if n > 1
            else f"[{i + 1}/{n}] Center the candle in view, then press Enter..."
        )
        input(prompt)
        frame = self.capture_frame()
        try:
          mask, meta = self.segment_candle(frame, text_prompt=text_prompt)
          break
        except RuntimeError as exc:
          logging.error("%s", exc)
          choice = input("Retry? [Enter]=recapture, [p]=new prompt, [q]=quit: ").strip().lower()
          if choice == "q":
            raise
          if choice == "p":
            text_prompt = input("New prompt (e.g. wax candle.): ").strip() or text_prompt

      frame.mask = mask
      views.append(frame)

      if i == 0:
        poses.append(np.eye(4, dtype=np.float64))
      else:
        rel = self._estimate_relative_pose_icp(views[0], frame)
        poses.append(rel)

      if i < n - 1:
        time.sleep(wait_s)

    self._reference_views = views
    self._reference_poses = poses
    self._save_reference_views(views, poses)
    return views

  def build_object_mesh(
      self,
      views: list[Frame] | None = None,
      init_pose_model: bool = True,
  ) -> trimesh.Trimesh:
    views = views or self._reference_views
    if not views:
      raise RuntimeError("No reference views. Run capture_reference_views() first.")

    mode = self.cfg["reconstruction"]["mode"]
    if mode == "neural":
      mesh = self._build_mesh_neural(views)
    else:
      mesh = self._build_mesh_fast(views[0])

    mesh_path = self.paths["work_dir"] / "candle_mesh.obj"
    mesh.export(mesh_path)
    logging.info("Saved reconstructed mesh to %s", mesh_path)
    self._fp_mesh = mesh
    if init_pose_model:
      self._init_foundationpose(mesh)
    elif self._exclusive_gpu_models():
      self._unload_foundationpose()
    return mesh

  def _masked_point_cloud(self, frame: Frame) -> o3d.geometry.PointCloud:
    if frame.mask is None:
      raise ValueError("Frame mask is required")

    ys, xs = np.where(frame.mask)
    if len(xs) == 0:
      raise RuntimeError("Empty mask")

    z = frame.depth[ys, xs]
    valid = z > 0.01
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(z) < 50:
      raise RuntimeError("Too few valid depth points in mask")

    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    colors = frame.rgb[ys, xs].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    return pcd

  def _build_mesh_fast(self, frame: Frame) -> trimesh.Trimesh:
    pcd = self._masked_point_cloud(frame)
    if len(pcd.points) < 100:
      raise RuntimeError("Not enough points for mesh reconstruction")

    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )
    mesh_o3d = mesh_o3d.remove_degenerate_triangles()
    mesh_o3d = mesh_o3d.remove_duplicated_triangles()
    mesh_o3d = mesh_o3d.remove_non_manifold_edges()

    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_o3d = mesh_o3d.crop(bbox)

    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    if len(faces) == 0:
      raise RuntimeError("Poisson reconstruction produced an empty mesh")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    return mesh

  def _estimate_relative_pose_icp(self, ref: Frame, cur: Frame) -> np.ndarray:
    source = self._masked_point_cloud(ref)
    target = self._masked_point_cloud(cur)

    init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=0.02,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    ob_in_cam = reg.transformation
    return ob_in_cam

  def _build_mesh_neural(self, views: list[Frame]) -> trimesh.Trimesh:
    from bundlesdf.run_nerf import run_neural_object_field

    if self._reference_poses and len(self._reference_poses) == len(views):
      cam_in_obs = [np.linalg.inv(p) for p in self._reference_poses]
    else:
      cam_in_obs = [np.eye(4) for _ in views]

    cfg_path = self.paths["foundationpose"] / "bundlesdf" / "config_ycbv.yml"
    with open(cfg_path, "r", encoding="utf-8") as f:
      nerf_cfg = yaml.safe_load(f)
    nerf_cfg["n_step"] = 1000
    nerf_cfg["mesh_resolution"] = self.cfg["reconstruction"]["mesh_resolution"]

    K = views[0].K
    rgbs = [v.rgb for v in views]
    depths = [v.depth for v in views]
    masks = [v.mask.astype(np.uint8) * 255 for v in views]

    save_dir = str(self.paths["work_dir"] / "nerf")
    mesh = run_neural_object_field(
        nerf_cfg,
        K,
        rgbs,
        depths,
        masks,
        cam_in_obs,
        debug=0,
        save_dir=save_dir,
    )
    return mesh

  def _save_reference_views(self, views: list[Frame], poses: list[np.ndarray]) -> None:
    ref_dir = self.paths["work_dir"] / "reference_views"
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(ref_dir / "K.txt", views[0].K)
    for i, (frame, pose) in enumerate(zip(views, poses)):
      cv2.imwrite(str(ref_dir / f"rgb_{i:04d}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
      cv2.imwrite(str(ref_dir / f"mask_{i:04d}.png"), (frame.mask.astype(np.uint8) * 255))
      depth_mm = (frame.depth * 1000).astype(np.uint16)
      cv2.imwrite(str(ref_dir / f"depth_{i:04d}.png"), depth_mm)
      np.savetxt(ref_dir / f"ob_in_cam_{i:04d}.txt", pose)

  def load_reference_views(self, ref_dir: str | Path | None = None) -> list[Frame]:
    ref_dir = Path(ref_dir or self.paths["work_dir"] / "reference_views")
    K = np.loadtxt(ref_dir / "K.txt")
    views: list[Frame] = []
    poses: list[np.ndarray] = []
    rgb_files = sorted(ref_dir.glob("rgb_*.png"))
    for rgb_file in rgb_files:
      idx = rgb_file.stem.split("_")[1]
      rgb = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)
      depth = cv2.imread(str(ref_dir / f"depth_{idx}.png"), cv2.IMREAD_UNCHANGED) / 1000.0
      mask = cv2.imread(str(ref_dir / f"mask_{idx}.png"), cv2.IMREAD_GRAYSCALE) > 0
      pose_file = ref_dir / f"ob_in_cam_{idx}.txt"
      views.append(Frame(rgb=rgb, depth=depth.astype(np.float32), K=K.copy(), mask=mask))
      if pose_file.exists():
        poses.append(np.loadtxt(pose_file))
    self._reference_views = views
    self._reference_poses = poses
    logging.info("Loaded %d reference views from %s", len(views), ref_dir)
    return views

  # ------------------------------------------------------------------
  # Pose estimation
  # ------------------------------------------------------------------

  def estimate_pose(self, frame: Frame | None = None, mask: np.ndarray | None = None) -> PoseResult:
    frame = frame or self.capture_frame()
    if mask is None:
      mask, _ = self.segment_candle(frame)
    else:
      self._unload_segmentation_models()

    self._ensure_foundationpose_ready()

    fp_cfg = self.cfg["foundationpose"]
    with self._foundationpose_autocast():
      pose = self._fp_estimator.register(
          K=frame.K,
          rgb=frame.rgb,
          depth=frame.depth,
          ob_mask=mask.astype(bool),
          iteration=fp_cfg["est_refine_iter"],
      )
    return self._pose_result(pose)

  def track_pose(self, frame: Frame | None = None) -> PoseResult:
    self._unload_segmentation_models()
    self._ensure_foundationpose_ready()

    frame = frame or self.capture_frame()
    fp_cfg = self.cfg["foundationpose"]
    with self._foundationpose_autocast():
      pose = self._fp_estimator.track_one(
          rgb=frame.rgb,
          depth=frame.depth,
          K=frame.K,
          iteration=fp_cfg["track_refine_iter"],
      )
    return self._pose_result(pose)

  def _pose_result(self, pose: np.ndarray) -> PoseResult:
    position = pose[:3, 3].copy()
    rotation = pose[:3, :3].copy()
    return PoseResult(pose=pose, position=position, rotation_matrix=rotation)

  @staticmethod
  def rotation_matrix_to_euler_xyz(rotation: np.ndarray, degrees: bool = True) -> np.ndarray:
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
      x = np.arctan2(rotation[2, 1], rotation[2, 2])
      y = np.arctan2(-rotation[2, 0], sy)
      z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
      x = np.arctan2(-rotation[1, 2], rotation[1, 1])
      y = np.arctan2(-rotation[2, 0], sy)
      z = 0.0
    euler = np.array([x, y, z])
    return np.degrees(euler) if degrees else euler

  # ------------------------------------------------------------------
  # High-level workflows
  # ------------------------------------------------------------------

  def calibrate_object(self, num_views: int | None = None) -> trimesh.Trimesh:
    rec = self.cfg["reconstruction"]
    if num_views is None and rec["mode"] == "fast":
      num_views = int(rec.get("fast_num_reference_views", 1))
    self.capture_reference_views(num_views=num_views)
    return self.build_object_mesh()

  def scan_once(self) -> PoseResult:
    frame = self.capture_frame()
    mask, meta = self.segment_candle(frame)
    logging.info("Detection: %s (score %.2f)", meta["labels"], float(meta["scores"][0]))
    return self.estimate_pose(frame, mask=mask)

  def scan_loop(self, display: bool = True) -> None:
    logging.info("Live scan loop. Press q to quit.")
    logging.info(
        "VRAM mode: only one model stack on GPU at a time (%s).",
        "exclusive" if self._exclusive_gpu_models() else "shared",
    )
    tracking = self._fp_estimator is not None and self._fp_estimator.pose_last is not None

    while True:
      frame = self.capture_frame()
      if not tracking:
        mask, _ = self.segment_candle(frame)
        self._ensure_foundationpose_ready()
        result = self.estimate_pose(frame, mask=mask)
        tracking = True
      else:
        result = self.track_pose(frame)

      euler = self.rotation_matrix_to_euler_xyz(result.rotation_matrix)
      logging.info(
          "pos=[%.3f, %.3f, %.3f] m  euler_xyz=[%.1f, %.1f, %.1f] deg",
          *result.position,
          *euler,
      )

      if display:
        from Utils import draw_posed_3d_box, draw_xyz_axis

        vis = frame.rgb.copy()
        if self._fp_to_origin is not None and self._fp_bbox is not None:
          center_pose = result.pose @ np.linalg.inv(self._fp_to_origin)
          axis_scale = float(np.max(self._fp_bbox[1] - self._fp_bbox[0])) * 0.75
          vis = draw_posed_3d_box(
              frame.K, img=vis, ob_in_cam=center_pose, bbox=self._fp_bbox, line_color=(0, 255, 0), linewidth=2
          )
          vis = draw_xyz_axis(
              vis,
              ob_in_cam=center_pose,
              scale=max(axis_scale, 0.05),
              K=frame.K,
              thickness=3,
              transparency=0,
              is_input_rgb=True,
          )
          hud = (
              f"pos=[{result.position[0]:+.3f},{result.position[1]:+.3f},{result.position[2]:+.3f}] m  "
              f"xyz=[{euler[0]:+.1f},{euler[1]:+.1f},{euler[2]:+.1f}] deg"
          )
          cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
          cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("candle scan", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break

    if display:
      cv2.destroyAllWindows()


def main() -> None:
  parser = argparse.ArgumentParser(description="Modeless candle pose estimation with ZED")
  parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
  parser.add_argument("--prompt", type=str, default=None, help='Text prompt, e.g. "candle."')
  parser.add_argument(
      "--mesh-mode",
      choices=("fast", "neural"),
      default=None,
      help="fast = single-view depth mesh, neural = BundleSDF NeRF",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p_cap = sub.add_parser("capture-refs", help="Capture multi-view reference images")
  p_cap.add_argument("--num-views", type=int, default=None)

  sub.add_parser("build-mesh", help="Build mesh from saved reference views")
  sub.add_parser("scan", help="Estimate pose for one frame")
  p_live = sub.add_parser("live", help="Continuous pose tracking")
  p_live.add_argument("--no-display", action="store_true")

  p_all = sub.add_parser("run", help="Capture refs, build mesh, then live track")
  p_all.add_argument("--num-views", type=int, default=None)
  p_all.add_argument("--skip-capture", action="store_true")

  args = parser.parse_args()
  scanner = CandleScanner(
      config_path=args.config,
      text_prompt=args.prompt,
      mesh_mode=args.mesh_mode,
  )

  try:
    if args.command == "capture-refs":
      scanner.capture_reference_views(num_views=args.num_views)
    elif args.command == "build-mesh":
      scanner.load_reference_views()
      scanner.build_object_mesh()
    elif args.command == "scan":
      scanner.load_reference_views()
      scanner.build_object_mesh(init_pose_model=False)
      result = scanner.scan_once()
      euler = scanner.rotation_matrix_to_euler_xyz(result.rotation_matrix)
      print("Pose (4x4):\n", result.pose)
      print(f"Position (m): {result.position}")
      print(f"Euler XYZ (deg): {euler}")
    elif args.command == "live":
      scanner.load_reference_views()
      scanner.build_object_mesh(init_pose_model=False)
      scanner.scan_loop(display=not args.no_display)
    elif args.command == "run":
      if not args.skip_capture:
        scanner.calibrate_object(num_views=args.num_views)
      else:
        scanner.load_reference_views()
        scanner.build_object_mesh()
      scanner.scan_loop()
  finally:
    scanner.close()


if __name__ == "__main__":
  main()
