"""Tests for Score-MRI baseline, SALM2D, Tiny3DCNN, eval_epoch, and visualization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hyper_dm.models.score_mri import (
    ScoreMRINet,
    ScoreMRIScheduler,
    score_matching_loss,
    pc_sampler,
)
from hyper_dm.models.patch3d import Tiny3DCNN
from hyper_dm.models.diffusion import ddim_sample_steps
from hyper_dm.models.unet import UNetTiny
from hyper_dm.models.hypernet import HyperNet, flat_to_state


# ── Score-MRI ────────────────────────────────────────────────────

class TestScoreMRINet:
    def test_forward_shape(self):
        model = ScoreMRINet(base=16, cond=True)
        x = torch.randn(2, 1, 32, 32)
        sigma = torch.rand(2)
        sino = torch.randn(2, 1, 32, 32)
        out = model(x, sigma, sino)
        assert out.shape == (2, 1, 32, 32)

    def test_forward_no_cond(self):
        model = ScoreMRINet(base=16, cond=False)
        x = torch.randn(2, 1, 32, 32)
        sigma = torch.rand(2)
        out = model(x, sigma)
        assert out.shape == (2, 1, 32, 32)

    def test_cond_required(self):
        model = ScoreMRINet(base=16, cond=True)
        with pytest.raises(ValueError, match="cond=True"):
            model(torch.randn(1, 1, 16, 16), torch.rand(1))


class TestScoreMRIScheduler:
    def test_length(self):
        sched = ScoreMRIScheduler(K=20)
        assert len(sched) == 20

    def test_descending(self):
        sched = ScoreMRIScheduler(sigma_min=0.01, sigma_max=1.0, K=10)
        sigmas = sched.get()
        deltas = sigmas[1:] - sigmas[:-1]
        assert (deltas < 0).all()


class TestScoreMatchingLoss:
    def test_scalar_output(self):
        model = ScoreMRINet(base=8, cond=True)
        sched = ScoreMRIScheduler(K=5)
        gt = torch.randn(2, 1, 16, 16)
        sino = torch.randn(2, 1, 16, 16)
        loss = score_matching_loss(model, gt, sino, sched)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_backward(self):
        model = ScoreMRINet(base=8, cond=True)
        sched = ScoreMRIScheduler(K=5)
        gt = torch.randn(2, 1, 16, 16)
        sino = torch.randn(2, 1, 16, 16)
        loss = score_matching_loss(model, gt, sino, sched)
        loss.backward()


class TestPCSampler:
    def test_output_shape(self):
        model = ScoreMRINet(base=8, cond=True)
        sched = ScoreMRIScheduler(K=3)
        sino = torch.randn(2, 1, 16, 16)
        out = pc_sampler(model, sino, scheduler=sched)
        assert out.shape == (2, 1, 16, 16)

    def test_output_clamped(self):
        model = ScoreMRINet(base=8, cond=True)
        sched = ScoreMRIScheduler(K=3)
        sino = torch.randn(1, 1, 16, 16)
        out = pc_sampler(model, sino, scheduler=sched)
        assert out.min() >= 0
        assert out.max() <= 1


# ── Tiny3DCNN ───────────────────────────────────────────────────

class TestTiny3DCNN:
    def test_output_shape(self):
        model = Tiny3DCNN(num_classes=2)
        x = torch.randn(4, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (4, 2)


# ── ddim_sample_steps ───────────────────────────────────────────

class TestDDIMSampleSteps:
    def test_returns_list_of_correct_length(self):
        backbone = UNetTiny(in_ch=3, base=8)
        hnet = HyperNet(backbone, z_dim=4, hidden=16)
        z = torch.randn(1, 4)
        weights = flat_to_state(hnet(z)[0], backbone)
        cond = torch.randn(1, 1, 32, 32)
        steps = 5
        result = ddim_sample_steps(backbone, cond, weights, steps=steps, T_train=50)
        assert isinstance(result, list)
        assert len(result) == steps

    def test_each_step_correct_shape(self):
        backbone = UNetTiny(in_ch=3, base=8)
        hnet = HyperNet(backbone, z_dim=4, hidden=16)
        z = torch.randn(1, 4)
        weights = flat_to_state(hnet(z)[0], backbone)
        cond = torch.randn(2, 1, 32, 32)
        result = ddim_sample_steps(backbone, cond, weights, steps=3, T_train=50)
        for step_tensor in result:
            assert step_tensor.shape == (2, 1, 32, 32)
