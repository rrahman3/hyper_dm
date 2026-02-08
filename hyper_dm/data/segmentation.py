"""Segmentation datasets for LUNA16 nodule segmentation (1-ch and 3-ch uncertainty)."""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class NoduleSegDataset(Dataset):
    """Single-channel nodule segmentation: ``(image, mask)`` pairs.

    Parameters
    ----------
    ct_dir : directory with reconstruction ``.npy`` files (1×128×128).
    mask_dir : directory with binary segmentation masks.
    split : ``"train"`` / ``"val"`` / ``"test"``.
    train_frac : proportion of files used for training.
    """

    def __init__(
        self,
        ct_dir: str | pathlib.Path,
        mask_dir: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.5,
    ) -> None:
        ct_dir = pathlib.Path(ct_dir)
        mask_dir = pathlib.Path(mask_dir)

        common = sorted(
            p for p in ct_dir.glob("*.npy") if (mask_dir / p.name).exists()
        )
        if not common:
            raise RuntimeError(f"No matching CT/mask pairs in {ct_dir} & {mask_dir}")

        n = len(common)
        train_end = int(train_frac * n)
        val_end = train_end + int(0.3 * n)  # val = 30% of remaining

        if split == "train":
            self.slices = common[:train_end]
        elif split == "val":
            self.slices = common[train_end:val_end]
        else:
            self.slices = common[val_end:]

        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.slices[idx]
        img = np.load(p).astype(np.float32)
        msk = np.load(self.mask_dir / p.name).astype(np.float32)

        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)

        # Ensure (1, H, W) shape
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if msk.ndim == 2:
            msk = msk.unsqueeze(0)
        if img.ndim == 3 and img.shape[0] != 1:
            img = img.unsqueeze(0)

        return img, msk


class NoduleSegUncertaintyDataset(Dataset):
    """Three-channel nodule segmentation: ``(mean+AU+EU, mask)`` pairs.

    Parameters
    ----------
    mean_dir, au_dir, eu_dir : directories for the three channels.
    mask_dir : directory with binary segmentation masks.
    split : ``"train"`` / ``"val"`` / ``"test"``.
    train_frac : proportion of files used for training.
    """

    def __init__(
        self,
        mean_dir: str | pathlib.Path,
        au_dir: str | pathlib.Path,
        eu_dir: str | pathlib.Path,
        mask_dir: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.5,
    ) -> None:
        mean_dir = pathlib.Path(mean_dir)
        au_dir = pathlib.Path(au_dir)
        eu_dir = pathlib.Path(eu_dir)
        mask_dir = pathlib.Path(mask_dir)

        common = sorted(
            p
            for p in mean_dir.glob("*.npy")
            if (au_dir / p.name).exists()
            and (eu_dir / p.name).exists()
            and (mask_dir / p.name).exists()
        )
        if not common:
            raise RuntimeError("No matching mean/AU/EU/mask quadruplets found.")

        n = len(common)
        train_end = int(train_frac * n)
        val_end = train_end + int(0.3 * n)

        if split == "train":
            self.slices = common[:train_end]
        elif split == "val":
            self.slices = common[train_end:val_end]
        else:
            self.slices = common[val_end:]

        self.mean_dir = mean_dir
        self.au_dir = au_dir
        self.eu_dir = eu_dir
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.slices[idx].name

        mean = np.load(self.mean_dir / stem).astype(np.float32).squeeze()
        au = np.load(self.au_dir / stem).astype(np.float32).squeeze()
        eu = np.load(self.eu_dir / stem).astype(np.float32).squeeze()

        img = np.stack([mean, au, eu], axis=0)  # (3, H, W)

        msk = np.load(self.mask_dir / stem).astype(np.float32)
        msk = (msk > 0).astype(np.float32).squeeze()
        if msk.ndim == 2:
            msk = msk[np.newaxis]  # (1, H, W)

        return torch.from_numpy(img), torch.from_numpy(msk)


class ReconDownstreamDataset(Dataset):
    """Binary classification on reconstructed CT slices.

    Parameters
    ----------
    recon_dir : directory with reconstructed ``.npy`` slices.
    label_dir : directory with per-slice label ``.npy`` files.
    split : ``"train"`` / ``"val"`` / ``"test"``.
    train_frac, val_frac : proportions.
    """

    def __init__(
        self,
        recon_dir: str | pathlib.Path,
        label_dir: str | pathlib.Path,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        recon_dir = pathlib.Path(recon_dir)
        label_dir = pathlib.Path(label_dir)

        self.recons = sorted(recon_dir.glob("*.npy"))
        n = len(self.recons)
        train_end = int(train_frac * n)
        val_end = train_end + int(val_frac * n)

        if split == "train":
            self.recons = self.recons[:train_end]
        elif split == "val":
            self.recons = self.recons[train_end:val_end]
        else:
            self.recons = self.recons[val_end:]

        self.label_dir = label_dir

    def __len__(self) -> int:
        return len(self.recons)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rpath = self.recons[idx]
        img = torch.from_numpy(np.load(rpath).astype(np.float32))
        if img.ndim == 2:
            img = img.unsqueeze(0)

        label = torch.from_numpy(np.load(self.label_dir / rpath.name)).long()
        return img, label


class SinogramDataset(Dataset):
    """Stream sinogram .npy files for reconstruction export."""

    def __init__(self, root: str | pathlib.Path) -> None:
        self.sinos = sorted(pathlib.Path(root, "sino").glob("*.npy"))

    def __len__(self) -> int:
        return len(self.sinos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.sinos[idx]
        sino = torch.from_numpy(np.load(path).astype(np.float32))
        if sino.ndim == 2:
            sino = sino.unsqueeze(0)
        return sino, path.stem
