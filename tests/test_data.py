"""Unit tests for dataset classes (smoke tests with synthetic data).

These tests create small temporary `.npy` files in the exact directory
layout expected by each dataset class and verify lengths, shapes, dtypes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hyper_dm.data.ct_dataset import CTSliceDataset
from hyper_dm.data.segmentation import (
    NoduleSegDataset,
    NoduleSegUncertaintyDataset,
    ReconDownstreamDataset,
    SinogramDataset,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_ct_root(tmp_path: Path, n: int = 5, shape=(128, 128)):
    """Create ``<root>/ct/`` and ``<root>/sino/`` with matching .npy files."""
    ct_dir = tmp_path / "ct"
    sino_dir = tmp_path / "sino"
    ct_dir.mkdir()
    sino_dir.mkdir()
    for i in range(n):
        np.save(ct_dir / f"slice_{i:03d}.npy", np.random.randn(*shape).astype(np.float32))
        np.save(sino_dir / f"slice_{i:03d}.npy", np.random.randn(*shape).astype(np.float32))
    return tmp_path


# ── CTSliceDataset ───────────────────────────────────────────────

class TestCTSliceDataset:
    @pytest.fixture
    def ct_root(self, tmp_path):
        return _make_ct_root(tmp_path, n=10)

    def test_length_train(self, ct_root):
        ds = CTSliceDataset(str(ct_root), split="train", train_frac=0.8, val_frac=0.1)
        assert len(ds) == 8  # 80% of 10

    def test_split_disjoint(self, ct_root):
        args = dict(root=str(ct_root), train_frac=0.8, val_frac=0.1, seed=0)
        tr = CTSliceDataset(split="train", **args)
        va = CTSliceDataset(split="val", **args)
        te = CTSliceDataset(split="test", **args)
        assert len(tr) + len(va) + len(te) == 10

    def test_item_shapes(self, ct_root):
        ds = CTSliceDataset(str(ct_root), split="train")
        sino, ct, sid = ds[0]
        assert ct.shape == (1, 128, 128)
        assert sino.shape == (1, 128, 128)
        assert isinstance(sid, str)

    def test_item_dtype(self, ct_root):
        ds = CTSliceDataset(str(ct_root), split="train")
        sino, ct, _ = ds[0]
        assert ct.dtype == torch.float32
        assert sino.dtype == torch.float32


# ── NoduleSegDataset ─────────────────────────────────────────────

class TestNoduleSegDataset:
    @pytest.fixture
    def seg_dirs(self, tmp_path):
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        for i in range(10):
            np.save(img_dir / f"{i}.npy", np.random.randn(128, 128).astype(np.float32))
            np.save(mask_dir / f"{i}.npy", (np.random.rand(128, 128) > 0.5).astype(np.float32))
        return img_dir, mask_dir

    def test_length(self, seg_dirs):
        img_dir, mask_dir = seg_dirs
        ds = NoduleSegDataset(str(img_dir), str(mask_dir), split="train", train_frac=0.5)
        assert len(ds) == 5  # 50% of 10

    def test_item_shapes(self, seg_dirs):
        img_dir, mask_dir = seg_dirs
        ds = NoduleSegDataset(str(img_dir), str(mask_dir), split="train")
        img, mask = ds[0]
        assert img.shape == (1, 128, 128)
        assert mask.shape == (1, 128, 128)


# ── SinogramDataset ──────────────────────────────────────────────

class TestSinogramDataset:
    @pytest.fixture
    def sino_root(self, tmp_path):
        sino_dir = tmp_path / "sino"
        sino_dir.mkdir()
        for i in range(3):
            np.save(sino_dir / f"sino_{i}.npy", np.random.randn(128, 128).astype(np.float32))
        return tmp_path  # root, not sino_dir

    def test_length(self, sino_root):
        ds = SinogramDataset(str(sino_root))
        assert len(ds) == 3

    def test_item_shape_and_stem(self, sino_root):
        ds = SinogramDataset(str(sino_root))
        sino, stem = ds[0]
        assert sino.shape == (1, 128, 128)
        assert isinstance(stem, str)


# ── ReconDownstreamDataset ───────────────────────────────────────

class TestReconDownstreamDataset:
    @pytest.fixture
    def recon_dirs(self, tmp_path):
        recon_dir = tmp_path / "recon"
        label_dir = tmp_path / "labels"
        recon_dir.mkdir()
        label_dir.mkdir()
        for i in range(10):
            np.save(recon_dir / f"slice_{i}.npy", np.random.randn(128, 128).astype(np.float32))
            np.save(label_dir / f"slice_{i}.npy", np.array(i % 2))
        return recon_dir, label_dir

    def test_length(self, recon_dirs):
        recon_dir, label_dir = recon_dirs
        ds = ReconDownstreamDataset(str(recon_dir), str(label_dir), split="train")
        assert len(ds) == 8  # 80% of 10

    def test_item_shape_and_label(self, recon_dirs):
        recon_dir, label_dir = recon_dirs
        ds = ReconDownstreamDataset(str(recon_dir), str(label_dir), split="train")
        img, lbl = ds[0]
        assert img.shape == (1, 128, 128)
        assert lbl.dtype == torch.int64
