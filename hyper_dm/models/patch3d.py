"""3-D patch-based classifier for nodule candidate detection (LUNA16 downstream)."""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Tiny3DCNN(nn.Module):
    """Small 3-D CNN for binary nodule-candidate classification."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CandidatePatchDataset(Dataset):
    """Extracts 3-D patches around candidate coordinates for classification.

    Parameters
    ----------
    recon_dir : directory with per-UID slice ``.npy`` files
                (naming: ``{uid}_{sliceidx:04d}.npy``).
    candidates_csv : CSV with columns ``seriesuid, coordX, coordY, coordZ, class``.
    raw_ct_dir : directory with ``.mhd`` headers (for world-to-voxel mapping).
    split : patient-level split.
    patch_size : side length of the cubic patch.
    train_frac, val_frac : patient-level proportions.
    """

    def __init__(
        self,
        recon_dir: str | pathlib.Path,
        candidates_csv: str | pathlib.Path,
        raw_ct_dir: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        patch_size: int = 32,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        import pandas as pd

        self.recon_dir = pathlib.Path(recon_dir)
        self.raw_ct_dir = pathlib.Path(raw_ct_dir)
        self.patch_size = patch_size

        df = pd.read_csv(candidates_csv)
        uids = sorted(df.seriesuid.unique())

        n = len(uids)
        train_end = int(train_frac * n)
        val_end = train_end + int(val_frac * n)
        if split == "train":
            sel = set(uids[:train_end])
        elif split == "val":
            sel = set(uids[train_end:val_end])
        else:
            sel = set(uids[val_end:])

        self.df = df[df.seriesuid.isin(sel)].reset_index(drop=True)
        self._vol_cache: dict[str, np.ndarray] = {}

    def _load_volume(self, uid: str) -> np.ndarray:
        if uid not in self._vol_cache:
            slices = sorted(self.recon_dir.glob(f"{uid}_*.npy"))
            self._vol_cache[uid] = np.stack(
                [np.load(f).squeeze() for f in slices], axis=0
            )
        return self._vol_cache[uid]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import SimpleITK as sitk

        row = self.df.iloc[idx]
        uid = row.seriesuid
        vol = self._load_volume(uid)

        # World → voxel via .mhd header
        mhd_path = self.raw_ct_dir / f"{uid}.mhd"
        itk = sitk.ReadImage(str(mhd_path))
        org = np.array(itk.GetOrigin())
        spc = np.array(itk.GetSpacing())
        vx, vy, vz = ((np.array([row.coordX, row.coordY, row.coordZ]) - org) / spc).round().astype(int)

        ps = self.patch_size // 2
        zmin, zmax = max(vz - ps, 0), min(vz + ps, vol.shape[0] - 1)
        ymin, ymax = max(vy - ps, 0), min(vy + ps, vol.shape[1] - 1)
        xmin, xmax = max(vx - ps, 0), min(vx + ps, vol.shape[2] - 1)

        patch = np.zeros((self.patch_size, self.patch_size, self.patch_size), np.float32)
        patch[: (zmax - zmin + 1), : (ymax - ymin + 1), : (xmax - xmin + 1)] = vol[
            zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1
        ]

        return (
            torch.from_numpy(patch).unsqueeze(0),  # (1, Z, Y, X)
            torch.tensor(int(row["class"]), dtype=torch.long),
        )
