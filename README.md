# RobotVision

Unified workspace for robot perception experiments: **Grounded SAM 2** (open-vocabulary segmentation) + **FoundationPose** (6D pose), plus the Boxer/BEV/SLAM stack.

## Layout

```
RobotVision/
├── apps/
│   ├── candle/          # Model-free candle pose (text prompt → depth mesh → FP)
│   └── cube/            # CAD cube pose (SolidWorks mesh + ZED + FP)
├── external/
│   ├── FoundationPose/  # NVLabs FoundationPose (+ weights)
│   ├── GroundedSAM2/    # IDEA-Research Grounded-SAM-2 (+ checkpoints)
│   └── zed_object_pipeline/  # Boxer / BEV / ORB-SLAM3 desktop stack
├── data/
│   ├── meshes/          # Shared object meshes (candle, cube, …)
│   ├── workspaces/candle/   # Reference views, reconstructed mesh, debug
│   ├── svo/             # ZED .svo2 recordings
│   ├── demos/           # Upstream FoundationPose demo sequences
│   ├── rgbd/ masks/ outputs/
├── configs/             # Shared YAML for the src/robotvision package
├── scripts/             # Thin CLI entry points
├── src/robotvision/     # Package wrappers (segment / pose / combo pipeline)
└── vendor/              # Optional local wheels (e.g. pyzed)
```

## Quick start (candle / cube)

Uses the existing `candle-scan` conda env.

```bash
cd apps/candle
chmod +x setup.sh
./setup.sh                  # first time: env, deps, extensions
conda activate candle-scan

# Model-free candle
python scan_candle.py run --num-views 8

# CAD cube (from apps/cube)
cd ../cube
python scan_cube.py live
# or pure FoundationPose with a hand-drawn mask:
python foundationpose_original_mask.py
```

See `apps/candle/` and `apps/cube/` configs for prompts, ZED settings, and GPU knobs (tuned for 8 GB VRAM).

## Package CLI (scaffold)

```bash
pip install -e .
rv-segment --help
rv-pose --help
rv-pipeline --help
```

`src/robotvision/` is the cleaner library API; the battle-tested live demos are still under `apps/`.

## External deps

| Path | Source |
|------|--------|
| `external/FoundationPose` | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) |
| `external/GroundedSAM2` | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| `external/zed_object_pipeline` | BoxerNet OBB + BEV + SLAM desktop pipeline |

Details and install notes: [`external/README.md`](external/README.md).
