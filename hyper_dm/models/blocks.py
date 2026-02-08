"""Reusable convolutional building blocks for Hyper-DM backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Sequential):
    """Conv2d → GroupNorm(8) → SiLU activation block."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
        )


class SEBlock(nn.Module):
    """Squeeze-and-Excite channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class DownBlock(nn.Module):
    """Double conv (+ optional SE) followed by 2× max-pool."""

    def __init__(self, c_in: int, c_out: int, use_se: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvGNAct(c_in, c_out), ConvGNAct(c_out, c_out)]
        if use_se:
            layers.append(SEBlock(c_out))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(self.conv(x), 2)


class UpBlock(nn.Module):
    """2× nearest upsample → concat skip → double conv (+ optional SE)."""

    def __init__(self, c_in: int, c_out: int, use_se: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvGNAct(c_in, c_out), ConvGNAct(c_out, c_out)]
        if use_se:
            layers.append(SEBlock(c_out))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


class ConvBlock(nn.Module):
    """Simple double Conv2d + ReLU block used in segmentation U-Nets."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
