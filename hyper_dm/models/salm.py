"""SALM2D — SAM/ViT-based 2-D segmentation model (comparison baseline).

Uses a pre-trained ViT-B/16 backbone (via ``timm``) with a learnable
convolutional decoder for binary segmentation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None  # type: ignore[assignment]


class Decoder2D(nn.Module):
    """Three-stage up-sampler from ViT patch tokens to spatial mask.

    Input: ``(B, N, C)`` patch tokens where ``N = H' × W'``.
    """

    def __init__(self, embed_dim: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2)
        self.up2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.final = nn.Conv2d(embed_dim // 8, out_channels, 1)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        B, N, C = tokens.shape
        h, w = grid
        x = tokens.permute(0, 2, 1).reshape(B, C, h, w)
        x = F.relu(self.conv0(x))
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        return self.final(x)


class SALM2D(nn.Module):
    """ViT-B/16 encoder + CNN decoder for 2-D segmentation.

    Parameters
    ----------
    in_chans : number of input channels (1 for single-channel CT/MRI).
    embed_dim : ViT embedding dimension (768 for ViT-B).
    out_channels : number of output classes (1 for binary segmentation).
    """

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 768,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("Install timm: pip install timm")

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            in_chans=in_chans,
            num_classes=0,  # no classification head
        )
        if self.vit.embed_dim != embed_dim:
            raise RuntimeError(
                f"embed_dim mismatch: decoder expects {embed_dim}, "
                f"ViT has {self.vit.embed_dim}"
            )
        self.decoder = Decoder2D(embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W) → logits (B, out_channels, H, W)``."""
        B, _, H, W = x.shape
        ps = self.vit.patch_embed.patch_size[0]  # 16
        if (H % ps) or (W % ps):
            raise ValueError(f"(H, W)=({H},{W}) must be divisible by patch_size={ps}")

        tokens = self.vit.forward_features(x)  # (B, N+1, C) with CLS
        if tokens.shape[1] == (H // ps) * (W // ps) + 1:
            tokens = tokens[:, 1:, :]  # drop CLS token

        gh, gw = H // ps, W // ps
        logits = self.decoder(tokens, grid=(gh, gw))
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
