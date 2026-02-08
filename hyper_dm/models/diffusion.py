"""Gaussian diffusion schedulers, noise-prediction loss, and DDIM sampler."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
#  Schedulers
# ──────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """Linear-β DDPM schedule.

    Parameters
    ----------
    T : number of diffusion timesteps.
    beta_min, beta_max : endpoints of the linear β schedule.
    device : torch device for pre-computed tensors.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        device: str = "cpu",
    ) -> None:
        betas = torch.linspace(beta_min, beta_max, T, device=device)
        alphas = 1.0 - betas
        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alphabar = torch.cumprod(alphas, dim=0)

    @property
    def num_timesteps(self) -> int:
        return self.T

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: sample ``x_t`` given ``x_0`` and noise."""
        a = self.alphabar[t].view(-1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


# ──────────────────────────────────────────────────────────────────
#  Noise-prediction loss
# ──────────────────────────────────────────────────────────────────

class NoisePredLoss(nn.Module):
    """MSE loss between predicted and true Gaussian noise.

    Randomly samples a timestep *t* and noise *ε*, applies forward diffusion,
    then evaluates the model's noise prediction.
    """

    def __init__(self, diffusion: GaussianDiffusion) -> None:
        super().__init__()
        self.diff = diffusion

    def forward(
        self,
        model_fn: Callable[..., torch.Tensor],
        cond: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        model_fn : ``(x_t, cond, t_norm) → ε_pred``.
        cond : conditioning tensor (sinogram / zero-filled image).
        x0 : clean target image.
        """
        B = x0.size(0)
        t = torch.randint(0, self.diff.num_timesteps, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.diff.q_sample(x0, t, eps)
        t_norm = t.float() / (self.diff.num_timesteps - 1)
        eps_pred = model_fn(xt, cond, t_norm)
        return F.mse_loss(eps_pred, eps)


# ──────────────────────────────────────────────────────────────────
#  DDIM-style deterministic sampler
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    unet: nn.Module,
    cond: torch.Tensor,
    weights: dict[str, torch.Tensor],
    steps: int = 20,
    img_shape: tuple[int, ...] | None = None,
    alpha: float = 0.9,
) -> torch.Tensor:
    """Simple deterministic reverse-process sampler.

    Parameters
    ----------
    unet : noise-prediction backbone.
    cond : (B, 1, H, W) conditioning signal.
    weights : parameter dict for ``functional_call``.
    steps : number of denoising steps.
    img_shape : spatial shape of the generated image (C, H, W).
                Defaults to ``(1, cond.H, cond.W)`` if *None*.
    alpha : step-size parameter (analogous to αₜ in DDIM).

    Returns
    -------
    (B, 1, H, W) denoised reconstruction.
    """
    B = cond.size(0)
    device = cond.device

    if img_shape is None:
        img_shape = (1, cond.shape[2], cond.shape[3])

    x = torch.randn(B, *img_shape, device=device)

    for k in reversed(range(steps)):
        t = torch.full((B,), k / max(steps - 1, 1), device=device)
        eps = unet(x, cond, t, weights)
        x = (x - (1 - alpha) * eps) / math.sqrt(alpha)

    return x


@torch.no_grad()
def ddim_sample_steps(
    unet: nn.Module,
    cond: torch.Tensor,
    weights: dict[str, torch.Tensor],
    steps: int = 20,
    img_shape: tuple[int, ...] | None = None,
    T_train: int = 100,
    alpha: float = 0.9,
) -> list[torch.Tensor]:
    """DDIM sampler that returns intermediate denoising steps.

    Useful for visualising the diffusion trajectory with
    :func:`~hyper_dm.utils.visualization.plot_diffusion_steps`.

    Parameters
    ----------
    unet : noise-prediction backbone.
    cond : (B, 1, H, W) conditioning signal.
    weights : parameter dict for ``functional_call``.
    steps : number of denoising steps.
    img_shape : spatial shape ``(C, H, W)``.
    T_train : number of timesteps used during training (for schedule alignment).
    alpha : step-size parameter.

    Returns
    -------
    List of ``(B, 1, H, W)`` tensors from step 0 (noisiest) to step T (cleanest).
    """
    import numpy as np

    B = cond.size(0)
    device = cond.device
    if img_shape is None:
        img_shape = (1, cond.shape[2], cond.shape[3])

    x = torch.randn(B, *img_shape, device=device)
    all_steps: list[torch.Tensor] = []

    t_schedule = np.linspace(T_train - 1, 0, steps).astype(np.int64)
    for k in range(steps):
        t_idx = t_schedule[k]
        t_norm = torch.full((B,), t_idx / (T_train - 1), device=device)
        eps = unet(x, cond, t_norm, weights)
        x = (x - (1 - alpha) * eps) / math.sqrt(alpha)
        all_steps.append(x.clone().cpu())

    all_steps.reverse()  # step 0 = noisiest, step T = cleanest
    return all_steps


# ──────────────────────────────────────────────────────────────────
#  EU spread loss
# ──────────────────────────────────────────────────────────────────

def pairwise_l1(preds: torch.Tensor) -> torch.Tensor:
    """Mean pair-wise L1 distance between *M* predictions.

    Parameters
    ----------
    preds : (M, B, 1, H, W) – reconstructions from M different weight samples.

    Returns
    -------
    Scalar spread loss.
    """
    M = preds.size(0)
    total = 0.0
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            total = total + (preds[i] - preds[j]).abs().mean()
            count += 1
    return total / max(count, 1)
