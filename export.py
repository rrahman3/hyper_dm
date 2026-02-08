#!/usr/bin/env python
"""Export Hyper-DM reconstructions to disk for downstream tasks.

Usage
-----
    python export.py --config configs/downstream.yaml

This will:
  1. Load the HyperNet checkpoint specified in the config.
  2. Iterate over all sinograms in the data root.
  3. Save mean / AU / EU reconstruction maps as .npy files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Hyper-DM Reconstructions")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    torch.backends.cudnn.benchmark = True

    rcfg = cfg["reconstruction"]

    from hyper_dm.models.unet import build_backbone
    from hyper_dm.models.hypernet import HyperNet
    from hyper_dm.inference.export import export_reconstructions

    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = build_backbone(rcfg["backbone"], in_ch=3, base=32).to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    hnet = HyperNet(unet, z_dim=rcfg["z_dim"], hidden=rcfg["hidden"]).to(device)
    hnet.load_state_dict(torch.load(rcfg["hnet_ckpt"], map_location=device))
    hnet.eval()

    export_reconstructions(
        unet=unet,
        hnet=hnet,
        data_root=cfg["data"]["data_root"],
        out_dir=cfg["data"]["recon_dir"],
        z_dim=rcfg["z_dim"],
        steps=rcfg.get("ddim_steps", 20),
        batch_size=cfg["training"]["batch_size"],
    )
    print("Export complete.")


if __name__ == "__main__":
    main()
