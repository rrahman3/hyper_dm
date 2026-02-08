#!/usr/bin/env python
"""Train a Hyper-DM (or downstream) model from a YAML config file.

Usage
-----
    python train.py --config configs/ct_luna16.yaml
    python train.py --config configs/knee_singlecoil.yaml
    python train.py --config configs/brain_multicoil.yaml
    python train.py --config configs/segmentation.yaml
    python train.py --config configs/downstream.yaml

Override individual fields via CLI:
    python train.py --config configs/ct_luna16.yaml training.epochs=100 training.batch_size=16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_dotted_overrides(args: list[str]) -> dict:
    """Parse ``key.subkey=value`` overrides into a nested dict."""
    result: dict = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Try to parse value as int/float/bool
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
    parser = argparse.ArgumentParser(description="Hyper-DM Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    overrides = _parse_dotted_overrides(unknown)
    if overrides:
        _deep_update(cfg, overrides)

    torch.backends.cudnn.benchmark = True

    task = cfg.get("task", "ct_reconstruction")

    if task in ("ct_reconstruction", "mri_reconstruction"):
        from hyper_dm.trainers import train_hyperdm

        unet, hnet = train_hyperdm(cfg)
        print("Training complete.")

    elif task == "segmentation":
        from hyper_dm.trainers import train_segmentation

        model = train_segmentation(cfg)
        print("Segmentation training complete.")

    elif task == "downstream_classification":
        from hyper_dm.trainers import train_downstream_classifier

        model = train_downstream_classifier(cfg)
        print("Downstream training complete.")

    else:
        print(f"Unknown task: '{task}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
