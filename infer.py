#!/usr/bin/env python
"""Run Hyper-DM inference (M×N sampling) from a YAML config + checkpoint.

Usage
-----
    python infer.py --config configs/ct_luna16.yaml --ckpt path/to/hnet.ckpt
    python infer.py --config configs/knee_singlecoil.yaml --ckpt path/to/hnet.ckpt --split val

Override config fields via CLI:
    python infer.py --config configs/ct_luna16.yaml --ckpt hnet.ckpt inference.M=20 inference.N=20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch


def _deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_dotted_overrides(args: list[str]) -> dict:
    result: dict = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        for caster in (int, float):
            try:
                val = caster(val)  # type: ignore[assignment]
                break
            except (ValueError, TypeError):
                pass
        if val == "true":
            val = True  # type: ignore[assignment]
        elif val == "false":
            val = False  # type: ignore[assignment]
        d[parts[-1]] = val
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyper-DM Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to HyperNet checkpoint.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test).")
    parser.add_argument("--no-save", action="store_true", help="Skip saving .npy outputs.")
    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    overrides = _parse_dotted_overrides(unknown)
    if overrides:
        _deep_update(cfg, overrides)

    torch.backends.cudnn.benchmark = True

    from hyper_dm.inference import load_hyperdm, infer_hyperdm

    unet, hnet = load_hyperdm(cfg, args.ckpt)
    results = infer_hyperdm(cfg, unet, hnet, split=args.split, save=not args.no_save)

    if results:
        print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
