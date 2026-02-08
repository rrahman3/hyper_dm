"""Classification datasets for fastMRI downstream tasks (knee & brain).

Loads `.npy` reconstruction files produced by Hyper-DM inference and pairs
them with integer classification labels read from a CSV file.

CSV format (columns): ``file``, ``slice``, ``label_idx``

Supports both 1-channel (mean reconstruction only) and 3-channel
(mean + AU + EU uncertainty maps) inputs.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def _resolve_npy_path(
    recon_dir: pathlib.Path,
    file_id: str,
    slice_idx: int,
) -> pathlib.Path:
    """Build ``<recon_dir>/<file_id>_<slice>.npy`` and return the path.

    Falls back to ``<recon_dir>/<file_id>/<slice>.npy`` if the flat layout
    does not exist.
    """
    flat = recon_dir / f"{file_id}_{slice_idx}.npy"
    if flat.exists():
        return flat
    nested = recon_dir / file_id / f"{slice_idx}.npy"
    if nested.exists():
        return nested
    # Return flat as canonical (caller will get a descriptive FileNotFoundError)
    return flat


def _split_df(
    df: pd.DataFrame,
    split: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Deterministic train/val/test split on rows."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    if split == "train":
        return df.iloc[:train_end]
    if split == "val":
        return df.iloc[train_end:val_end]
    return df.iloc[val_end:]


# ──────────────────────────────────────────────────────────────────
#  1-channel dataset (standard or EU-mean recons)
# ──────────────────────────────────────────────────────────────────

class MRIClsDataset(Dataset):
    """Single-channel MRI classification: ``(image, label)`` pairs.

    Parameters
    ----------
    recon_dir : directory with reconstruction ``.npy`` files.
    csv_path : path to a CSV with columns ``file``, ``slice``, ``label_idx``.
    split : ``"train"`` / ``"val"`` / ``"test"``.
    train_frac, val_frac : proportions.
    img_size : optional resize target (square).
    seed : random seed for deterministic splits.
    """

    def __init__(
        self,
        recon_dir: str | pathlib.Path,
        csv_path: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        img_size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.recon_dir = pathlib.Path(recon_dir)
        self.img_size = img_size

        df = pd.read_csv(csv_path)
        required = {"file", "slice", "label_idx"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must have columns {required}, got {set(df.columns)}"
            )

        self.df = _split_df(df, split, train_frac, val_frac, seed)
        self.num_classes: int = int(df["label_idx"].max()) + 1

    # ── helpers ────────────────────────────────────────────

    def _resize(self, t: torch.Tensor) -> torch.Tensor:
        if self.img_size is None:
            return t
        import torch.nn.functional as F

        if t.ndim == 2:
            t = t.unsqueeze(0)
        return F.interpolate(
            t.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # ── Dataset interface ──────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        file_id = str(row["file"])
        slice_idx = int(row["slice"])
        label = int(row["label_idx"])

        npy_path = _resolve_npy_path(self.recon_dir, file_id, slice_idx)
        img = np.load(npy_path).astype(np.float32)
        img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)  # (1, H, W)

        img = self._resize(img)
        return img, label


# ──────────────────────────────────────────────────────────────────
#  3-channel dataset (mean + AU + EU uncertainty maps)
# ──────────────────────────────────────────────────────────────────

class MRIClsUncertaintyDataset(Dataset):
    """Three-channel MRI classification: ``(mean+AU+EU, label)`` pairs.

    Parameters
    ----------
    mean_dir, au_dir, eu_dir : directories for the three channels.
    csv_path : path to a CSV with columns ``file``, ``slice``, ``label_idx``.
    split : ``"train"`` / ``"val"`` / ``"test"``.
    train_frac, val_frac : proportions.
    img_size : optional resize target (square).
    seed : random seed for deterministic splits.
    """

    def __init__(
        self,
        mean_dir: str | pathlib.Path,
        au_dir: str | pathlib.Path,
        eu_dir: str | pathlib.Path,
        csv_path: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        img_size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.mean_dir = pathlib.Path(mean_dir)
        self.au_dir = pathlib.Path(au_dir)
        self.eu_dir = pathlib.Path(eu_dir)
        self.img_size = img_size

        df = pd.read_csv(csv_path)
        required = {"file", "slice", "label_idx"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must have columns {required}, got {set(df.columns)}"
            )

        self.df = _split_df(df, split, train_frac, val_frac, seed)
        self.num_classes: int = int(df["label_idx"].max()) + 1

    # ── helpers ────────────────────────────────────────────

    def _resize_3ch(self, t: torch.Tensor) -> torch.Tensor:
        """Resize a (3, H, W) tensor."""
        if self.img_size is None:
            return t
        import torch.nn.functional as F

        return F.interpolate(
            t.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # ── Dataset interface ──────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        file_id = str(row["file"])
        slice_idx = int(row["slice"])
        label = int(row["label_idx"])

        mean = np.load(
            _resolve_npy_path(self.mean_dir, file_id, slice_idx)
        ).astype(np.float32).squeeze()
        au = np.load(
            _resolve_npy_path(self.au_dir, file_id, slice_idx)
        ).astype(np.float32).squeeze()
        eu = np.load(
            _resolve_npy_path(self.eu_dir, file_id, slice_idx)
        ).astype(np.float32).squeeze()

        img = torch.from_numpy(np.stack([mean, au, eu], axis=0))  # (3, H, W)
        img = self._resize_3ch(img)
        return img, label
