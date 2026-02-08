"""fastMRI brain multi-coil dataset — k-space → RSS image-domain pairs."""

from __future__ import annotations

import pathlib
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from fastmri.data.transforms import to_tensor, apply_mask
from fastmri.data.subsample import create_mask_for_mask_type

from ..utils.mri_transforms import kspace_to_nchw, nchw_to_rss_image, center_crop


class BrainMultiCoilDataset(Dataset):
    """fastMRI brain multi-coil dataset.

    Returns ``dict`` with keys:
    * ``"full_kspace"``   — centre-cropped full k-space, shape ``(2·coils, img_size, img_size)``.
    * ``"masked_kspace"`` — under-sampled k-space (same shape).
    * ``"mask"``          — sampling mask.
    * ``"sid"``           — slice identifier string.

    Parameters
    ----------
    root : parent directory (e.g. ``D:/data``).
    split, modality : sub-path components (e.g. ``"train"`` / ``"brain/"``).
    mask_type : ``"equispaced"`` or ``"random"``.
    acc_factor : acceleration factor for the under-sampling mask.
    center_fraction : fraction of k-space centre to keep.
    img_size : spatial resolution of the centre-cropped k-space.
    filter_split : ``"train"`` / ``"val"`` / ``"test"`` — slice-level split.
    train_frac, val_frac : proportions for splitting.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        split: str = "train",
        modality: str = "brain/",
        mask_type: str = "equispaced",
        acc_factor: int = 4,
        center_fraction: float = 0.08,
        img_size: int = 320,
        filter_split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        pattern = pathlib.Path(root) / modality / split
        files = sorted(pattern.rglob("*.h5"))
        if not files:
            raise RuntimeError(f"No .h5 files found in {pattern}")

        # Build flat slice index
        slice_tuples: list[tuple[int, int]] = []
        self.files = files
        for file_idx, fp in enumerate(files):
            with h5py.File(fp, "r") as hf:
                n_slices = hf["kspace"].shape[0]
            slice_tuples.extend((file_idx, s) for s in range(n_slices))

        # Slice-level split
        n = len(slice_tuples)
        train_end = int(train_frac * n)
        val_end = train_end + int(val_frac * n)

        if filter_split == "train":
            self.slice_tuples = slice_tuples[:train_end]
        elif filter_split == "val":
            self.slice_tuples = slice_tuples[train_end:val_end]
        else:
            self.slice_tuples = slice_tuples[val_end:]

        self.mask_func = create_mask_for_mask_type(
            mask_type, [center_fraction], [acc_factor]
        )
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.slice_tuples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        file_idx, slice_idx = self.slice_tuples[idx]
        fname = self.files[file_idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][slice_idx]

        # Ensure shape (coils, H, W, 2)  — complex pairs
        if kspace.ndim == 3 and kspace.shape[-1] == 2:
            kspace = kspace[np.newaxis, ...]
        elif kspace.ndim == 3 and kspace.shape[0] < 32:
            if np.iscomplexobj(kspace):
                kspace = np.stack([np.real(kspace), np.imag(kspace)], axis=-1)
            else:
                kspace = np.stack([kspace, np.zeros_like(kspace)], axis=-1)
        elif kspace.ndim == 2:
            if np.iscomplexobj(kspace):
                kspace = np.stack([np.real(kspace), np.imag(kspace)], axis=-1)
            else:
                kspace = np.stack([kspace, np.zeros_like(kspace)], axis=-1)
            kspace = kspace[np.newaxis, ...]

        # Full k-space in channel-stacked format
        full_kspace = kspace_to_nchw(kspace)

        # Apply mask
        kspace_torch = to_tensor(kspace)
        masked_kspace_torch, mask, _ = apply_mask(kspace_torch, self.mask_func)
        masked_kspace = kspace_to_nchw(masked_kspace_torch.numpy())

        # Centre-crop to img_size
        full_kspace = center_crop(full_kspace, (self.img_size, self.img_size))
        masked_kspace = center_crop(masked_kspace, (self.img_size, self.img_size))

        sid = f"{fname.stem}_{slice_idx:03d}"
        return {
            "full_kspace": full_kspace,
            "masked_kspace": masked_kspace,
            "mask": mask.float(),
            "sid": sid,
        }
