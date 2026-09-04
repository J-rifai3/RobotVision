# External dependencies

Drop full upstream checkouts here, or add as git submodules.

## FoundationPose

```bash
git clone https://github.com/NVlabs/FoundationPose.git external/FoundationPose
# Follow upstream install (CUDA, nvdiffrast, etc.)
```

Paste checkpoints under `external/FoundationPose/weights/` (or path your config uses).

## Grounded SAM 2

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git external/GroundedSAM2
# Follow upstream install (Grounding DINO + SAM 2)
```

Paste detector / SAM2 weights where your existing setup expects them, then point `configs/grounded_sam2.yaml` at those paths.

## PYTHONPATH

If upstream code is not pip-installed, extend the path in your scripts or shell:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/FoundationPose:$(pwd)/external/GroundedSAM2"
```

Or wire paths in `src/robotvision/utils/paths.py` after you paste your bootstrap code.
