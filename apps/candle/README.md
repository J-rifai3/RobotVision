# Candle pose (model-free)

ZED + Grounded-SAM-2 + FoundationPose without a CAD model. Text prompt → mask → depth mesh → 6D pose.

```bash
conda activate candle-scan
./setup.sh    # once
python scan_candle.py run --num-views 8
```

Config: `config.yaml` (paths point at `../../external/` and `../../data/workspaces/candle`).

Shared env with `apps/cube`. Full notes on VRAM / CUDA matching are in the repo root README and the troubleshooting comments in `setup.sh`.
