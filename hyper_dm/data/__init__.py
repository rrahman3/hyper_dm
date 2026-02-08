"""Data sub-package — dataset registry and factory."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from .ct_dataset import CTSliceDataset
from .fastmri_knee import KneeSingleCoilDataset
from .segmentation import (
    NoduleSegDataset,
    NoduleSegUncertaintyDataset,
    ReconDownstreamDataset,
    SinogramDataset,
)
from .mri_classification import MRIClsDataset, MRIClsUncertaintyDataset

# Brain import is conditional (requires fastmri extras)
try:
    from .fastmri_brain import BrainMultiCoilDataset
except ImportError:
    BrainMultiCoilDataset = None  # type: ignore[assignment, misc]


def build_dataset(cfg: dict[str, Any], split: str = "train") -> Dataset:
    """Instantiate a dataset from a parsed YAML config ``data:`` section.

    Parameters
    ----------
    cfg : the full parsed YAML config dict.
    split : ``"train"`` / ``"val"`` / ``"test"``.
    """
    data = cfg["data"]
    name = data["dataset"]

    if name == "ct_luna16":
        return CTSliceDataset(
            root=data["root"],
            split=split,
            train_frac=data.get("train_frac", 0.8),
            val_frac=data.get("val_frac", 0.1),
            seed=data.get("split_seed", 0),
        )

    if name == "fastmri_knee":
        mask_cfg = data.get("mask", {})
        return KneeSingleCoilDataset(
            root=data["root"],
            split=split,
            img_size=data.get("img_size", 128),
            center_fractions=mask_cfg.get("center_fractions", [0.08]),
            accelerations=mask_cfg.get("accelerations", [4]),
            train_frac=data.get("train_frac", 0.1),
            val_frac=data.get("val_frac", 0.1),
            seed=data.get("split_seed", 0),
        )

    if name == "fastmri_brain":
        if BrainMultiCoilDataset is None:
            raise ImportError("fastmri package required for brain dataset")
        mask_cfg = data.get("mask", {})
        return BrainMultiCoilDataset(
            root=data["root"],
            split=data.get("split", "train"),
            modality=data.get("modality", "brain/"),
            mask_type=mask_cfg.get("type", "equispaced"),
            acc_factor=mask_cfg.get("accelerations", [4])[0],
            center_fraction=mask_cfg.get("center_fractions", [0.08])[0],
            img_size=data.get("img_size", 320),
            filter_split=split,
            train_frac=data.get("train_frac", 0.8),
            val_frac=data.get("val_frac", 0.1),
        )

    if name == "nodule_seg":
        if data.get("use_uncertainty_channels", False):
            return NoduleSegUncertaintyDataset(
                mean_dir=data["ct_dir"],
                au_dir=data["au_dir"],
                eu_dir=data["eu_dir"],
                mask_dir=data["mask_dir"],
                split=split,
                train_frac=data.get("train_frac", 0.5),
            )
        return NoduleSegDataset(
            ct_dir=data["ct_dir"],
            mask_dir=data["mask_dir"],
            split=split,
            train_frac=data.get("train_frac", 0.5),
        )

    if name == "recon_downstream":
        return ReconDownstreamDataset(
            recon_dir=data["recon_dir"],
            label_dir=data["label_dir"],
            split=split,
            train_frac=data.get("train_frac", 0.8),
            val_frac=data.get("val_frac", 0.1),
        )

    if name == "mri_cls":
        if data.get("use_uncertainty_channels", False):
            return MRIClsUncertaintyDataset(
                mean_dir=data["recon_dir"],
                au_dir=data["au_dir"],
                eu_dir=data["eu_dir"],
                csv_path=data["csv_path"],
                split=split,
                train_frac=data.get("train_frac", 0.8),
                val_frac=data.get("val_frac", 0.1),
                img_size=data.get("img_size"),
                seed=data.get("split_seed", 0),
            )
        return MRIClsDataset(
            recon_dir=data["recon_dir"],
            csv_path=data["csv_path"],
            split=split,
            train_frac=data.get("train_frac", 0.8),
            val_frac=data.get("val_frac", 0.1),
            img_size=data.get("img_size"),
            seed=data.get("split_seed", 0),
        )

    raise ValueError(f"Unknown dataset '{name}'")


__all__ = [
    "CTSliceDataset",
    "KneeSingleCoilDataset",
    "BrainMultiCoilDataset",
    "NoduleSegDataset",
    "NoduleSegUncertaintyDataset",
    "MRIClsDataset",
    "MRIClsUncertaintyDataset",
    "ReconDownstreamDataset",
    "SinogramDataset",
    "build_dataset",
]
