"""LUNA16 sparse-view CT dataset — paired sinogram / CT slices (.npy)."""

from __future__ import annotations

import pathlib
import random
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class CTSliceDataset(Dataset):
    """Load paired ``(sinogram, ct_slice)`` from ``<root>/sino/`` and ``<root>/ct/``.

    Files are ``.npy`` arrays of shape ``(1, H, W)`` with values in ``[0, 1]``.
    A deterministic random split is applied at the *file* level.

    Parameters
    ----------
    root : directory containing ``ct/`` and ``sino/`` sub-folders.
    split : one of ``"train"``, ``"val"``, ``"test"``.
    train_frac, val_frac : proportions for train / val (test = remainder).
    seed : random seed for reproducible splitting.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 0,
    ) -> None:
        root = pathlib.Path(root)
        ct_files = sorted((root / "ct").glob("*.npy"))
        if not ct_files:
            raise FileNotFoundError(f"No .npy files found in {root / 'ct'}")

        rng = random.Random(seed)
        rng.shuffle(ct_files)
        n = len(ct_files)

        train_end = int(train_frac * n)
        val_end = train_end + int(val_frac * n)

        if split == "train":
            self.ct_files = ct_files[:train_end]
        elif split == "val":
            self.ct_files = ct_files[train_end:val_end]
        else:
            self.ct_files = ct_files[val_end:]

        self.sino_dir = root / "sino"

    def __len__(self) -> int:
        return len(self.ct_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        ct_path = self.ct_files[idx]
        sino_path = self.sino_dir / ct_path.name

        ct = torch.from_numpy(np.load(ct_path).astype(np.float32))
        sino = torch.from_numpy(np.load(sino_path).astype(np.float32))

        # Ensure shape (1, H, W)
        if ct.ndim == 2:
            ct = ct.unsqueeze(0)
        if sino.ndim == 2:
            sino = sino.unsqueeze(0)

        sid = ct_path.stem
        return sino, ct, sid
