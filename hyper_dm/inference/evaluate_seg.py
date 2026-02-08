"""Multi-source segmentation evaluation.

Runs the same segmentation model on reconstructions from different sources
(GT CT, normal recons, EU recons, AU recons) and compares Dice scores.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.segmentation import NoduleSegDataset
from ..utils.metrics import dice_score, dice_loss


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> float:
    """Compute average Dice score on a segmentation DataLoader."""
    model.eval()
    total_dice = 0.0
    n = 0
    for x, y in tqdm(loader, desc="seg eval", leave=False):
        x, y = x.to(device), y.to(device)
        if x.ndim > 4:
            x = x.squeeze(1)
        logits = model(x)
        total_dice += dice_score(logits, y).item() * x.size(0)
        n += x.size(0)
    return total_dice / max(n, 1)


def compare_reconstruction_sources(
    model: nn.Module,
    mask_dir: str | pathlib.Path,
    sources: dict[str, str | pathlib.Path],
    *,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate segmentation across multiple reconstruction source directories.

    Parameters
    ----------
    model : trained segmentation model.
    mask_dir : directory with binary mask ``.npy`` files.
    sources : mapping ``{source_name: ct_dir_path}``, e.g.
              ``{"GT CT": ".../ct", "EU recon": ".../recons_EU/mean", ...}``.
    batch_size : evaluation batch size.
    device : torch device.

    Returns
    -------
    Dict ``{source_name: avg_dice_score}``.
    """
    results: dict[str, float] = {}
    model = model.to(device)

    for name, ct_dir in sources.items():
        ds = NoduleSegDataset(str(ct_dir), str(mask_dir), split="test", train_frac=0.0)
        dl = DataLoader(ds, batch_size=batch_size)
        avg_dice = evaluate_segmentation(model, dl, device)
        results[name] = avg_dice
        print(f"  {name:20s}  Dice = {avg_dice:.4f}")

    return results
