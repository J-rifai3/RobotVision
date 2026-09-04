# External dependencies

Upstream checkouts live here (not meant to be edited casually). App configs point at these paths.

## FoundationPose

```
external/FoundationPose/
```

Cloned from [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose). Weights expected under `weights/`:

- `2023-10-28-18-33-37` (refiner)
- `2024-01-11-20-02-45` (scorer)

Download: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i

Demo sequences: `data/demos/foundationpose/` (and zip archives under `data/demos/foundationpose_archives/`).

## Grounded SAM 2

```
external/GroundedSAM2/
```

Cloned from [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2).

- SAM2 checkpoints: `checkpoints/sam2.1_hiera_*.pt`
- Optional local Grounding DINO weights: `gdino_checkpoints/`
- Apps default to HuggingFace `IDEA-Research/grounding-dino-tiny`

## zed_object_pipeline

```
external/zed_object_pipeline/
```

Desktop stack: Zenoh/ZED bus → ORB-SLAM3 → BEV → BoxerNet 7-DoF OBB. See that folder's own README for setup (`scripts/setup.sh`, SAM3, SLAM build).

## PYTHONPATH

If you run upstream scripts directly (outside `apps/`):

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/FoundationPose:$(pwd)/external/GroundedSAM2"
```

Prefer `apps/candle/setup.sh` + the app entrypoints; they wire paths via `config.yaml`.
