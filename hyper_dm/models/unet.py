"""U-Net backbones for Hyper-DM noise prediction and segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DownBlock, UpBlock, ConvBlock


# ──────────────────────────────────────────────────────────────────
#  Noise-prediction U-Nets  (used inside the diffusion framework)
# ──────────────────────────────────────────────────────────────────

class UNetTiny(nn.Module):
    """2-level U-Net (base → 2×base).  3-channel input ``[x_t, cond, t_map]``."""

    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.inc = nn.Sequential(ConvGNAct(in_ch, base), ConvGNAct(base, base))
        self.down1 = DownBlock(base, base * 2)
        self.up1 = UpBlock(base * 2 + base, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        t_norm: torch.Tensor,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xt : (B, 1, H, W) noisy image at timestep *t*.
        cond : (B, 1, H', W') conditioning signal (sinogram / zero-filled MRI).
        t_norm : (B,) normalised timestep in [0, 1].
        weights : optional external weight dict (for HyperNet functional_call).
        """
        if weights is not None:
            return torch.func.functional_call(self, weights, (xt, cond, t_norm))

        B, _, H, W = xt.shape
        cond = F.interpolate(cond, (H, W), mode="bilinear", align_corners=False)
        t_map = t_norm.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([xt, cond, t_map], dim=1)  # (B, 3, H, W)

        s0 = self.inc(x)
        bottleneck = self.down1(s0)
        x = self.up1(bottleneck, s0)
        return self.outc(x)


class UNetTiny3Level(nn.Module):
    """3-level U-Net (base → 2×base → 4×base).  Used for brain data at 320px."""

    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.inc = nn.Sequential(ConvGNAct(in_ch, base), ConvGNAct(base, base))
        self.down1 = DownBlock(base, base * 2)
        self.down2 = DownBlock(base * 2, base * 4)
        self.up2 = UpBlock(base * 4 + base * 2, base * 2)
        self.up1 = UpBlock(base * 2 + base, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        t_norm: torch.Tensor,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if weights is not None:
            return torch.func.functional_call(self, weights, (xt, cond, t_norm))

        B, _, H, W = xt.shape
        cond = F.interpolate(cond, (H, W), mode="bilinear", align_corners=False)
        t_map = t_norm.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([xt, cond, t_map], dim=1)

        s0 = self.inc(x)
        d1 = self.down1(s0)
        d2 = self.down2(d1)
        u2 = self.up2(d2, d1)
        u1 = self.up1(u2, s0)
        return self.outc(u1)


class HyperUNet(nn.Module):
    """3-level U-Net with SE attention (base=64 → 128 → 256)."""

    def __init__(self, in_ch: int = 3, base: int = 64) -> None:
        super().__init__()
        self.inc = nn.Sequential(ConvGNAct(in_ch, base), ConvGNAct(base, base))
        self.down1 = DownBlock(base, base * 2, use_se=True)
        self.down2 = DownBlock(base * 2, base * 4, use_se=True)
        self.up1 = UpBlock(base * 4 + base * 2, base * 2, use_se=True)
        self.up2 = UpBlock(base * 2 + base, base, use_se=True)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        t_norm: torch.Tensor,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if weights is not None:
            return torch.func.functional_call(self, weights, (xt, cond, t_norm))

        B, _, H, W = xt.shape
        cond = F.interpolate(cond, (H, W), mode="bilinear", align_corners=False)
        t_map = t_norm.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([xt, cond, t_map], dim=1)

        s0 = self.inc(x)
        s1 = self.down1(s0)
        bottleneck = self.down2(s1)
        x = self.up1(bottleneck, s1)
        x = self.up2(x, s0)
        return self.outc(x)


# ──────────────────────────────────────────────────────────────────
#  Segmentation U-Net
# ──────────────────────────────────────────────────────────────────

class SegmentationUNet(nn.Module):
    """4-level U-Net for binary segmentation (e.g. lung nodules)."""

    def __init__(self, in_ch: int = 1, n_classes: int = 1) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_ch, 32)
        self.down2 = ConvBlock(32, 64)
        self.down3 = ConvBlock(64, 128)
        self.down4 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)

        self.up4 = ConvBlock(512 + 256, 256)
        self.up3 = ConvBlock(256 + 128, 128)
        self.up2 = ConvBlock(128 + 64, 64)
        self.up1 = ConvBlock(64 + 32, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))

        x = self.up4(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), d4], 1))
        x = self.up3(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), d3], 1))
        x = self.up2(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), d2], 1))
        x = self.up1(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), d1], 1))
        return self.outc(x)


# ──────────────────────────────────────────────────────────────────
#  Downstream classifier
# ──────────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """Simple classifier on reconstructed slices (any spatial size)."""

    def __init__(self, in_ch: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),           # always outputs (16, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────

def build_backbone(name: str, **kwargs) -> nn.Module:
    """Instantiate a backbone by config name."""
    registry = {
        "unet_tiny": UNetTiny,
        "unet_tiny_3level": UNetTiny3Level,
        "hyper_unet": HyperUNet,
        "seg_unet": SegmentationUNet,
        "tiny_cnn": TinyCNN,
    }
    if name not in registry:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(registry.keys())}")
    return registry[name](**kwargs)
