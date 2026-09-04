#!/usr/bin/env python3
"""Print the torch+cuda tag used by prebuilt GPU extension wheels (e.g. pt2.6.0cu124)."""

from __future__ import annotations

import sys

try:
    import torch
except ImportError:
    print("torch is not installed", file=sys.stderr)
    raise SystemExit(1)

version = torch.__version__.split("+")[0]
cuda = torch.version.cuda
if not cuda:
    print("cpu", end="")
    raise SystemExit(0)

cuda_tag = "cu" + cuda.replace(".", "")
print(f"pt{version}{cuda_tag}", end="")
