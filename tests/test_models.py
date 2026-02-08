"""Unit tests for model forward passes and shape correctness."""

from __future__ import annotations

import pytest
import torch

from hyper_dm.models import (
    UNetTiny,
    UNetTiny3Level,
    HyperUNet,
    SegmentationUNet,
    TinyCNN,
    build_backbone,
)
from hyper_dm.models.hypernet import HyperNet, n_params, flat_to_state
from hyper_dm.models.blocks import ConvGNAct, SEBlock, DownBlock, UpBlock, ConvBlock


# ── Helpers ──────────────────────────────────────────────────────

B, H, W = 2, 64, 64


def _noise_unet_inputs(batch: int = B, h: int = H, w: int = W):
    """Return (xt, cond, t_norm) tensors for noise-prediction U-Nets."""
    xt = torch.randn(batch, 1, h, w)
    cond = torch.randn(batch, 1, h, w)
    t_norm = torch.rand(batch)
    return xt, cond, t_norm


# ── Blocks ───────────────────────────────────────────────────────

class TestBlocks:
    def test_conv_gn_act_shape(self):
        block = ConvGNAct(16, 32)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 32, 32, 32)

    def test_se_block_shape(self):
        se = SEBlock(16, reduction=4)
        x = torch.randn(2, 16, 32, 32)
        out = se(x)
        assert out.shape == x.shape

    def test_down_block_halves_spatial(self):
        down = DownBlock(16, 32)
        x = torch.randn(2, 16, 64, 64)
        out = down(x)
        assert out.shape == (2, 32, 32, 32)

    def test_down_block_with_se(self):
        down = DownBlock(16, 32, use_se=True)
        x = torch.randn(2, 16, 64, 64)
        out = down(x)
        assert out.shape == (2, 32, 32, 32)

    def test_up_block_doubles_spatial(self):
        up = UpBlock(48, 16)  # c_in = c_low + c_skip = 32 + 16
        x = torch.randn(2, 32, 16, 16)
        skip = torch.randn(2, 16, 32, 32)
        out = up(x, skip)
        assert out.shape == (2, 16, 32, 32)

    def test_conv_block_shape(self):
        block = ConvBlock(1, 32)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 64, 64)


# ── Noise-prediction U-Nets ─────────────────────────────────────

class TestUNetTiny:
    def test_output_shape(self):
        model = UNetTiny(in_ch=3, base=16)
        xt, cond, t_norm = _noise_unet_inputs()
        out = model(xt, cond, t_norm)
        assert out.shape == (B, 1, H, W)

    def test_different_cond_size(self):
        """Conditioning can be a different spatial size → bilinear interpolation."""
        model = UNetTiny(in_ch=3, base=16)
        xt = torch.randn(B, 1, H, W)
        cond = torch.randn(B, 1, 32, 32)  # smaller
        t_norm = torch.rand(B)
        out = model(xt, cond, t_norm)
        assert out.shape == (B, 1, H, W)


class TestUNetTiny3Level:
    def test_output_shape(self):
        model = UNetTiny3Level(in_ch=3, base=16)
        xt, cond, t_norm = _noise_unet_inputs()
        out = model(xt, cond, t_norm)
        assert out.shape == (B, 1, H, W)


class TestHyperUNet:
    def test_output_shape(self):
        model = HyperUNet(in_ch=3, base=16)
        xt, cond, t_norm = _noise_unet_inputs()
        out = model(xt, cond, t_norm)
        assert out.shape == (B, 1, H, W)


class TestSegmentationUNet:
    def test_output_shape(self):
        model = SegmentationUNet(in_ch=1, n_classes=1)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)

    def test_multichannel_input(self):
        model = SegmentationUNet(in_ch=3, n_classes=2)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 2, 64, 64)


class TestTinyCNN:
    def test_output_shape(self):
        model = TinyCNN(in_ch=1, num_classes=2)
        x = torch.randn(4, 1, 128, 128)
        out = model(x)
        assert out.shape == (4, 2)


# ── HyperNet ────────────────────────────────────────────────────

class TestHyperNet:
    @pytest.fixture
    def backbone(self):
        return UNetTiny(in_ch=3, base=16)

    def test_output_dim_matches_backbone(self, backbone):
        hnet = HyperNet(backbone, z_dim=8, hidden=32)
        z = torch.randn(B, 8)
        out = hnet(z)
        assert out.shape == (B, n_params(backbone))

    def test_flat_to_state_keys(self, backbone):
        hnet = HyperNet(backbone, z_dim=8, hidden=32)
        z = torch.randn(1, 8)
        flat = hnet(z)[0]
        state = flat_to_state(flat, backbone)
        expected_keys = {n for n, _ in backbone.named_parameters()}
        assert set(state.keys()) == expected_keys

    def test_flat_to_state_shapes(self, backbone):
        hnet = HyperNet(backbone, z_dim=8, hidden=32)
        z = torch.randn(1, 8)
        flat = hnet(z)[0]
        state = flat_to_state(flat, backbone)
        for name, param in backbone.named_parameters():
            assert state[name].shape == param.shape

    def test_functional_call_forward(self, backbone):
        """Generated weights should be usable via functional_call."""
        hnet = HyperNet(backbone, z_dim=8, hidden=32)
        z = torch.randn(1, 8)
        flat = hnet(z)[0]
        state = flat_to_state(flat, backbone)

        xt, cond, t_norm = _noise_unet_inputs(batch=1)
        out = backbone(xt, cond, t_norm, weights=state)
        assert out.shape == (1, 1, H, W)


# ── build_backbone factory ──────────────────────────────────────

class TestBuildBackbone:
    @pytest.mark.parametrize(
        "name, kwargs",
        [
            ("unet_tiny", {"base": 16}),
            ("unet_tiny_3level", {"base": 16}),
            ("hyper_unet", {"base": 16}),
            ("seg_unet", {}),
            ("tiny_cnn", {}),
        ],
    )
    def test_registry_returns_module(self, name, kwargs):
        model = build_backbone(name, **kwargs)
        assert isinstance(model, torch.nn.Module)

    def test_unknown_name_raises(self):
        with pytest.raises((KeyError, ValueError)):
            build_backbone("non_existent_model")
