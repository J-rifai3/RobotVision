#!/usr/bin/env python3
"""Apply speed/quality presets to FoundationPoseTest/config.yaml.

Presets
-------
  NEURAL      Pre-optimization settings (best depth, slowest)
  QUALITY     Middle ground
  PERFORMANCE Current fast settings (fastest)

Usage
-----
  python scripts/set_quality_preset.py NEURAL
  python scripts/set_quality_preset.py QUALITY
  python scripts/set_quality_preset.py PERFORMANCE
  python scripts/set_quality_preset.py          # interactive prompt
  python scripts/set_quality_preset.py --show   # print current + presets
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

# Values that change with each preset.
PRESETS: dict[str, dict[str, object]] = {
    "NEURAL": {
        # Exact pre-optimization profile
        "depth_mode": "NEURAL",
        "est_refine_iter": 5,
        "track_refine_iter": 2,
        "render_batch_size": 16,
        "warp_chunk": 8,
        "rotation_views": 10,
        "inplane_step": 90,
        "debug": 1,
        "log_every": 1,
        "save_every": 30,
    },
    "QUALITY": {
        # Between accuracy and speed
        "depth_mode": "QUALITY",
        "est_refine_iter": 3,
        "track_refine_iter": 2,
        "render_batch_size": 24,
        "warp_chunk": 12,
        "rotation_views": 9,
        "inplane_step": 90,
        "debug": 0,
        "log_every": 5,
        "save_every": 30,
    },
    "PERFORMANCE": {
        # Current optimized profile
        "depth_mode": "PERFORMANCE",
        "est_refine_iter": 2,
        "track_refine_iter": 1,
        "render_batch_size": 32,
        "warp_chunk": 16,
        "rotation_views": 8,
        "inplane_step": 90,
        "debug": 0,
        "log_every": 15,
        "save_every": 30,
    },
}

# Map config key -> which YAML section it lives under (for inserts if missing).
KEY_SECTION = {
    "depth_mode": "zed",
    "est_refine_iter": "foundationpose",
    "track_refine_iter": "foundationpose",
    "render_batch_size": "foundationpose",
    "warp_chunk": "foundationpose",
    "rotation_views": "foundationpose",
    "inplane_step": "foundationpose",
    "debug": "foundationpose",
    "log_every": "foundationpose",
    "save_every": "foundationpose",
}


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _set_key(text: str, key: str, value: object) -> str:
    """Replace `key: ...` if present; otherwise insert under its section."""
    formatted = _format_value(value)
    pattern = re.compile(rf"^([ \t]*{re.escape(key)}:[ \t]*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{formatted}", text, count=1)

    section = KEY_SECTION[key]
    section_pat = re.compile(rf"^({re.escape(section)}:[ \t]*\n)", re.MULTILINE)
    match = section_pat.search(text)
    if not match:
        raise RuntimeError(f"Could not find section '{section}:' to insert {key}")
    insert_at = match.end()
    return text[:insert_at] + f"  {key}: {formatted}\n" + text[insert_at:]


def apply_preset(preset_name: str, config_path: Path = CONFIG_PATH) -> None:
    name = preset_name.strip().upper()
    if name not in PRESETS:
        raise SystemExit(
            f"Unknown preset {preset_name!r}. Choose: {', '.join(PRESETS)}"
        )
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    for key, value in PRESETS[name].items():
        text = _set_key(text, key, value)
    config_path.write_text(text, encoding="utf-8")

    print(f"Applied preset: {name}")
    print(f"Updated: {config_path}")
    print()
    for key, value in PRESETS[name].items():
        print(f"  {key}: {value}")


def show_presets(config_path: Path = CONFIG_PATH) -> None:
    print(f"Config: {config_path}")
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        print("\nCurrent values:")
        for key in PRESETS["PERFORMANCE"]:
            m = re.search(rf"^[ \t]*{re.escape(key)}:[ \t]*(.*)$", text, re.MULTILINE)
            print(f"  {key}: {m.group(1).strip() if m else '(missing)'}")
    print("\nAvailable presets:")
    for name, values in PRESETS.items():
        print(f"\n{name}:")
        for key, value in values.items():
            print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "preset",
        nargs="?",
        choices=sorted(PRESETS.keys()),
        help="NEURAL | QUALITY | PERFORMANCE",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current values and preset table, then exit",
    )
    args = parser.parse_args()

    if args.show:
        show_presets(args.config)
        return 0

    preset = args.preset
    if preset is None:
        print("Choose a preset:")
        print("  1) NEURAL      — best depth / slowest (pre-optimization)")
        print("  2) QUALITY     — middle ground")
        print("  3) PERFORMANCE — fastest (current optimizations)")
        choice = input("Enter name or number: ").strip()
        mapping = {"1": "NEURAL", "2": "QUALITY", "3": "PERFORMANCE"}
        preset = mapping.get(choice, choice)

    apply_preset(preset, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
