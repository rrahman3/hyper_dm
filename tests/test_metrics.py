"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hyper_dm.utils.metrics import psnr, ssim, dice_score, dice_loss, SegmentationLoss


class TestPSNR:
    def test_identical_images(self):
        img = np.random.rand(64, 64).astype(np.float32)
        assert psnr(img, img) == float("inf") or psnr(img, img) > 60

    def test_different_images(self):
        gt = np.zeros((64, 64), dtype=np.float32)
        pred = np.ones((64, 64), dtype=np.float32)
        val = psnr(gt, pred, data_range=1.0)
        assert val == pytest.approx(0.0, abs=0.5)


class TestSSIM:
    def test_identical_images(self):
        img = np.random.rand(64, 64).astype(np.float32)
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-4)

    def test_range(self):
        a = np.random.rand(64, 64).astype(np.float32)
        b = np.random.rand(64, 64).astype(np.float32)
        val = ssim(a, b)
        assert -1.0 <= val <= 1.0


class TestDiceScore:
    def test_perfect_match(self):
        logits = torch.full((2, 1, 16, 16), 10.0)  # strong positive
        targets = torch.ones(2, 1, 16, 16)
        score = dice_score(logits, targets)
        assert score.item() == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        logits = torch.full((2, 1, 16, 16), -10.0)  # strong negative
        targets = torch.ones(2, 1, 16, 16)
        score = dice_score(logits, targets)
        assert score.item() < 0.01


class TestDiceLoss:
    def test_inverse_of_score(self):
        logits = torch.full((2, 1, 16, 16), 10.0)
        targets = torch.ones(2, 1, 16, 16)
        loss = dice_loss(logits, targets)
        assert loss.item() == pytest.approx(0.0, abs=0.01)


class TestSegmentationLoss:
    def test_is_scalar(self):
        seg_loss = SegmentationLoss()
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = seg_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_backward(self):
        seg_loss = SegmentationLoss()
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = seg_loss(logits, targets)
        loss.backward()
        assert logits.grad is not None
