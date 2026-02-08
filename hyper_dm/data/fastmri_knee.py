"""fastMRI knee single-coil dataset — .h5 k-space → zero-filled / GT pairs."""

from __future__ import annotations

import pathlib
import random
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc


class KneeSingleCoilDataset(Dataset):
    """fastMRI knee single-coil dataset.

    Each sample is a single MRI slice yielding:
    * ``cond`` — zero-filled magnitude (under-sampled), shape ``(1, img_size, img_size)``.
    * ``gt``   — ground-truth RSS reconstruction,       shape ``(1, img_size, img_size)``.
    * ``sid``  — slice identifier string.

    Parameters
    ----------
    root : path to the directory containing ``.h5`` volumes.
    split : ``"train"`` / ``"val"`` / ``"test"`` — file-level split.
    img_size : spatial resolution after bicubic resize.
    center_fractions, accelerations : under-sampling mask parameters.
    train_frac, val_frac : proportions.
    seed : deterministic shuffle seed.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        img_size: int = 128,
        center_fractions: list[float] | None = None,
        accelerations: list[int] | None = None,
        train_frac: float = 0.1,
        val_frac: float = 0.1,
        seed: int = 0,
    ) -> None:
        root = pathlib.Path(root)
        files = sorted(root.rglob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No .h5 files under {root}")

        rng = random.Random(seed)
        rng.shuffle(files)
        n = len(files)
        train_end = int(train_frac * n)
        val_end = train_end + int(val_frac * n)

        if split == "train":
            files = files[:train_end]
        elif split == "val":
            files = files[train_end:val_end]
        else:
            files = files[val_end:]

        # Build a flat index of (file_path, slice_idx)
        self.index: list[tuple[pathlib.Path, int]] = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                n_slices = f["kspace"].shape[0]
            self.index.extend((fp, s) for s in range(n_slices))

        self.img_size = img_size
        self.mask_func = RandomMaskFunc(
            center_fractions=center_fractions or [0.08],
            accelerations=accelerations or [4],
        )

    def _zero_filled(self, kslice: np.ndarray) -> np.ndarray:
        """Apply under-sampling mask → IFFT → magnitude → resize → normalise."""
        k = T.to_tensor(kslice)
        k_masked, *_ = T.apply_mask(k, self.mask_func)
        img = fastmri.ifft2c(k_masked)
        mag = fastmri.complex_abs(img)
        if mag.dim() == 3:
            mag = fastmri.rss(mag, dim=0)

        mag = mag.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        mag = F.interpolate(mag, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        mag = mag.squeeze()  # (H, W)
        vmax = mag.max()
        if vmax > 0:
            mag = mag / vmax
        return mag.numpy().astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        fpath, sidx = self.index[idx]
        with h5py.File(fpath, "r") as f:
            kspace = f["kspace"][sidx]
            gt_raw = f["reconstruction_rss"][sidx]

        cond = torch.from_numpy(self._zero_filled(kspace)).unsqueeze(0)  # (1, H, W)

        # Resize GT to match
        gt_max = np.max(gt_raw)
        gt_norm = (gt_raw / gt_max).astype(np.float32) if gt_max > 0 else gt_raw.astype(np.float32)
        gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gt_resized = F.interpolate(gt_tensor, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        gt = gt_resized.squeeze(0)  # (1, H, W)

        sid = f"{fpath.stem}_{sidx:03d}"
        return cond, gt, sid
