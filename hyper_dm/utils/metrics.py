"""Evaluation metrics for image reconstruction and segmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    return float(peak_signal_noise_ratio(gt, pred, data_range=data_range))


def ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Structural Similarity Index (higher is better)."""
    return float(structural_similarity(gt, pred, data_range=data_range))


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice score (not loss — higher is better).

    Parameters
    ----------
    logits : (B, 1, H, W) raw logits from the segmentation model.
    targets : (B, 1, H, W) binary ground truth.
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=[2, 3])
    den = probs.sum(dim=[2, 3]) + targets.sum(dim=[2, 3]) + eps
    return (num / den).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss: ``1 − Dice``."""
    return 1.0 - dice_score(logits, targets, eps)


class SegmentationLoss(torch.nn.Module):
    """Combined segmentation loss: ``0.5 × BCE + 0.5 × Dice``."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(logits, targets) + 0.5 * dice_loss(logits, targets)
