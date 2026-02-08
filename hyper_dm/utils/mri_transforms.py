"""MRI-specific transforms: k-space ↔ image domain conversions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def kspace_to_nchw(kspace: np.ndarray) -> torch.Tensor:
    """Convert complex k-space ``(coils, H, W, 2)`` to real channel-stacked ``(2·coils, H, W)``.

    Parameters
    ----------
    kspace : numpy array of shape ``(coils, H, W, 2)`` (real/imag last dim).

    Returns
    -------
    Torch tensor of shape ``(2·coils, H, W)``.
    """
    assert kspace.ndim == 4 and kspace.shape[-1] == 2, (
        f"Expected (coils, H, W, 2), got {kspace.shape}"
    )
    t = torch.from_numpy(kspace).float()
    t = t.permute(0, 3, 1, 2)  # (coils, 2, H, W)
    return t.reshape(-1, t.shape[2], t.shape[3])  # (2·coils, H, W)


def nchw_to_rss_image(kspace_ch: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Channel-stacked k-space ``(2·coils, H, W)`` → RSS magnitude image ``(1, H, W)``.

    Applies FFT-shift → IFFT → magnitude → root-sum-of-squares.
    """
    if isinstance(kspace_ch, np.ndarray):
        kspace_ch = torch.from_numpy(kspace_ch)
    if kspace_ch.ndim == 4 and kspace_ch.shape[0] == 1:
        kspace_ch = kspace_ch[0]

    n_coils = kspace_ch.shape[0] // 2
    kspace_r = kspace_ch[:n_coils]
    kspace_i = kspace_ch[n_coils:]
    kspace_c = torch.complex(kspace_r, kspace_i)  # (coils, H, W)

    kspace_c = torch.fft.fftshift(kspace_c, dim=(-2, -1))
    img = torch.fft.ifft2(kspace_c, norm="ortho")
    img = torch.fft.ifftshift(img, dim=(-2, -1))
    rss = torch.sqrt((img.abs() ** 2).sum(dim=0, keepdim=True))  # (1, H, W)
    return rss


def kspace_to_image(kspace_ch: torch.Tensor | np.ndarray) -> np.ndarray:
    """Channel-stacked k-space → normalised RSS magnitude numpy array ``(H, W)``."""
    if isinstance(kspace_ch, np.ndarray):
        kspace_ch = torch.from_numpy(kspace_ch)
    if kspace_ch.ndim == 4 and kspace_ch.shape[0] == 1:
        kspace_ch = kspace_ch[0]

    n_ch = kspace_ch.shape[0] // 2
    kspace_r = kspace_ch[:n_ch]
    kspace_i = kspace_ch[n_ch:]
    kspace_c = torch.complex(kspace_r, kspace_i)

    kspace_c = torch.fft.fftshift(kspace_c, dim=(-2, -1))
    img = torch.fft.ifft2(kspace_c, norm="ortho")
    img = torch.fft.ifftshift(img, dim=(-2, -1))
    rss = torch.sqrt((img.abs() ** 2).sum(dim=0))
    rss = rss / rss.max().clamp(min=1e-8)
    return rss.cpu().numpy()


def center_crop(
    tensor: torch.Tensor | np.ndarray,
    crop_size: tuple[int, int],
) -> torch.Tensor | np.ndarray:
    """Centre-crop the last two spatial dimensions to ``(ch, cw)``."""
    h, w = tensor.shape[-2], tensor.shape[-1]
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return tensor[..., top : top + ch, left : left + cw]
