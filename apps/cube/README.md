# Cube pose (CAD / model-based)

Live 6D pose of a physical cube with a SolidWorks mesh, ZED, and FoundationPose.

```bash
conda activate candle-scan
python scripts/prepare_mesh.py          # STL → meters .obj
python scan_cube.py live                # center-box SAM2 mask + track
python foundationpose_original_mask.py  # hand-drawn mask only
```

Config: `config.yaml`. Imports shared helpers from `apps/candle` (`scan_candle`).

Quality presets: `python scripts/set_quality_preset.py NEURAL|QUALITY|PERFORMANCE`
