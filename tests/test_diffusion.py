"""Unit tests for diffusion schedule, noise loss, DDIM sampler, and EU spread."""

from __future__ import annotations

import pytest
import torch

from hyper_dm.models.diffusion import (
    GaussianDiffusion,
    NoisePredLoss,
    ddim_sample,
    pairwise_l1,
)
from hyper_dm.models.unet import UNetTiny
from hyper_dm.models.hypernet import HyperNet, flat_to_state, n_params


# ── GaussianDiffusion ───────────────────────────────────────────

class TestGaussianDiffusion:
    def test_alphabar_monotonically_decreasing(self):
        diff = GaussianDiffusion(T=100)
        deltas = diff.alphabar[1:] - diff.alphabar[:-1]
        assert (deltas < 0).all(), "alphabar should be strictly decreasing"

    def test_alphabar_range(self):
        diff = GaussianDiffusion(T=100)
        assert diff.alphabar[0] > 0.99, "First alphabar should be close to 1"
        assert diff.alphabar[-1] > 0, "Last alphabar should be > 0"

    def test_q_sample_shape(self):
        diff = GaussianDiffusion(T=100)
        B, C, H, W = 4, 1, 32, 32
        x0 = torch.randn(B, C, H, W)
        t = torch.randint(0, 100, (B,))
        noise = torch.randn_like(x0)
        xt = diff.q_sample(x0, t, noise)
        assert xt.shape == (B, C, H, W)

    def test_q_sample_t0_close_to_x0(self):
        """At t=0 (very low noise), x_t should be close to x_0."""
        diff = GaussianDiffusion(T=100)
        x0 = torch.randn(2, 1, 16, 16)
        t = torch.zeros(2, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = diff.q_sample(x0, t, noise)
        # alphabar[0] ≈ 1 so sqrt(alphabar)*x0 dominates
        assert torch.allclose(xt, x0, atol=0.05)

    def test_num_timesteps(self):
        diff = GaussianDiffusion(T=200)
        assert diff.num_timesteps == 200


# ── NoisePredLoss ────────────────────────────────────────────────

class TestNoisePredLoss:
    def test_loss_is_scalar(self):
        diff = GaussianDiffusion(T=50)
        loss_fn = NoisePredLoss(diff)

        model = UNetTiny(in_ch=3, base=8)

        cond = torch.randn(2, 1, 32, 32)
        x0 = torch.randn(2, 1, 32, 32)
        loss = loss_fn(model, cond, x0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_loss_backward(self):
        """Loss should be differentiable w.r.t. model params."""
        diff = GaussianDiffusion(T=50)
        loss_fn = NoisePredLoss(diff)
        model = UNetTiny(in_ch=3, base=8)

        cond = torch.randn(2, 1, 32, 32)
        x0 = torch.randn(2, 1, 32, 32)
        loss = loss_fn(model, cond, x0)
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Expected at least some gradients"


# ── DDIM Sampler ─────────────────────────────────────────────────

class TestDDIMSampler:
    def test_output_shape(self):
        backbone = UNetTiny(in_ch=3, base=8)
        hnet = HyperNet(backbone, z_dim=4, hidden=16)

        z = torch.randn(1, 4)
        flat = hnet(z)[0]
        weights = flat_to_state(flat, backbone)

        cond = torch.randn(2, 1, 32, 32)
        out = ddim_sample(backbone, cond, weights, steps=5)
        assert out.shape == (2, 1, 32, 32)

    def test_custom_img_shape(self):
        backbone = UNetTiny(in_ch=3, base=8)
        hnet = HyperNet(backbone, z_dim=4, hidden=16)

        z = torch.randn(1, 4)
        flat = hnet(z)[0]
        weights = flat_to_state(flat, backbone)

        cond = torch.randn(1, 1, 32, 32)
        out = ddim_sample(backbone, cond, weights, steps=3, img_shape=(1, 64, 64))
        assert out.shape == (1, 1, 64, 64)


# ── pairwise_l1 ─────────────────────────────────────────────────

class TestPairwiseL1:
    def test_identical_preds_zero(self):
        preds = torch.ones(3, 2, 1, 8, 8)
        assert pairwise_l1(preds).item() == pytest.approx(0.0)

    def test_positive_for_different_preds(self):
        preds = torch.randn(4, 2, 1, 8, 8)
        assert pairwise_l1(preds).item() > 0

    def test_two_samples(self):
        a = torch.zeros(2, 1, 4, 4)
        b = torch.ones(2, 1, 4, 4)
        preds = torch.stack([a, b])  # (2, 2, 1, 4, 4)
        # only 1 pair: mean |a - b| = 1.0
        assert pairwise_l1(preds).item() == pytest.approx(1.0)

    def test_shape_is_scalar(self):
        preds = torch.randn(5, 2, 1, 16, 16)
        out = pairwise_l1(preds)
        assert out.shape == ()
