# RobotVision

Perception stack for robot manipulation: **Grounded SAM 2** (open-vocabulary segmentation) → **FoundationPose** (6D pose estimation), with a combined pipeline on top.

This repo is an outline. Paste your existing FoundationPose, GroundedSAM2, and combo-model code into the slots below.

## Pipeline

```
Text prompt ──► Grounded SAM 2 ──► mask(s)
                                      │
RGB-D + mesh ─────────────────────────┴──► FoundationPose ──► 6D pose
```

## Project layout

```
RobotVision/
├── external/                  # Upstream repos (clone or paste whole trees here)
│   ├── FoundationPose/
│   └── GroundedSAM2/
├── src/robotvision/           # Your wrappers & glue code
│   ├── segmentation/          # ← paste Grounded SAM 2 integration
│   ├── pose/                  # ← paste FoundationPose integration
│   ├── pipeline/              # ← paste combo model
│   └── utils/
├── configs/                   # Model paths, camera intrinsics, prompts
├── data/
│   ├── meshes/                # CAD / reconstructed meshes for FoundationPose
│   ├── rgbd/                  # RGB + depth captures
│   ├── masks/                 # Segmentation outputs
│   └── outputs/               # Poses, debug overlays, videos
├── scripts/                   # CLI entry points
└── notebooks/                 # Exploratory / demo notebooks
```

## Quick start

```bash
cd RobotVision
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. Drop upstream repos into external/ (or git submodule add)
# 2. Paste your code into src/robotvision/{segmentation,pose,pipeline}/
# 3. Fill in configs/*.yaml with checkpoint paths and camera params

python scripts/run_grounded_sam2.py --config configs/grounded_sam2.yaml --prompt "red mug"
python scripts/run_foundation_pose.py --config configs/foundation_pose.yaml --mesh data/meshes/object.obj
python scripts/run_pipeline.py --config configs/pipeline.yaml --prompt "red mug"
```

## Where to paste what

| You have… | Paste into… |
|-----------|-------------|
| FoundationPose repo / checkpoints | `external/FoundationPose/` |
| Grounded SAM 2 repo / weights | `external/GroundedSAM2/` |
| SAM2 inference / tracking code | `src/robotvision/segmentation/grounded_sam2.py` |
| FoundationPose register / track code | `src/robotvision/pose/foundation_pose.py` |
| End-to-end detect → segment → pose | `src/robotvision/pipeline/combo.py` |
| Camera intrinsics / depth scale helpers | `src/robotvision/utils/camera.py` |
| I/O for RGB-D, masks, meshes | `src/robotvision/utils/io.py` |
| Pose / mask overlays | `src/robotvision/utils/viz.py` |
| One-off demos | `notebooks/` or `scripts/` |

See `external/README.md` for clone commands and env notes per upstream repo.

## Configs

| File | Purpose |
|------|---------|
| `configs/camera.yaml` | Intrinsics, depth scale, image size |
| `configs/grounded_sam2.yaml` | Detector + SAM2 checkpoint paths, thresholds |
| `configs/foundation_pose.yaml` | Mesh path, FoundationPose weights, refine iters |
| `configs/pipeline.yaml` | Full stack: prompt, paths, temporal filter settings |

## Data conventions

**RGB-D frame** (under `data/rgbd/<session>/`):

```
frame_000000/
  color.png       # or .jpg
  depth.png       # uint16 mm, or depth.npy float32 meters
  meta.json       # optional: timestamp, camera frame id
```

**Mesh** (under `data/meshes/`): `.obj` or `.ply` with metric scale aligned to depth.

**Mask** (under `data/masks/`): single-channel PNG, same resolution as color.

**Output pose**: 4×4 `object_in_camera` matrix saved as `.npy` or `pose.json` under `data/outputs/`.

## References

- [FoundationPose](https://github.com/NVlabs/FoundationPose)
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)
