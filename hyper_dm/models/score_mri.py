"""Score-MRI baseline: score-based model with predictor-corrector sampling.

This is a comparison baseline for the Hyper-DM paper. It implements a
VE-style score-matching model (Song & Ermon) conditioned on under-sampled
MRI / sinograms, with a simple predictor-corrector (PC) reverse sampler.
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .blocks import ConvGNAct


# ── Internal UNet blocks (light) ────────────────────────────────

class _Down(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(ConvGNAct(cin, cout), ConvGNAct(cout, cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(self.conv(x), 2)


class _Up(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(ConvGNAct(cin, cout), ConvGNAct(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


# ── ScoreMRINet ──────────────────────────────────────────────────

class ScoreMRINet(nn.Module):
    """2-level conditional UNet for score prediction.

    Input channels: ``x_noisy (1) + sigma_map (1) [+ conditioning (1)]``.
    """

    def __init__(self, base: int = 32, cond: bool = True) -> None:
        super().__init__()
        self.cond = cond
        in_ch = 1 + (1 if cond else 0) + 1  # noisy + (cond?) + sigma-map
        self.inc = nn.Sequential(ConvGNAct(in_ch, base), ConvGNAct(base, base))
        self.down1 = _Down(base, base * 2)
        self.up1 = _Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        sino: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_noisy : (B, 1, H, W) corrupted image.
        sigma : (B,) noise level per sample.
        sino : (B, 1, H, W) conditioning signal (zero-filled / sinogram).
        """
        B, _, H, W = x_noisy.shape
        feats = [x_noisy]

        if self.cond:
            if sino is None:
                raise ValueError("cond=True but sino is None")
            if sino.shape[-2:] != (H, W):
                sino = F.interpolate(sino, (H, W), mode="bilinear", align_corners=False)
            feats.append(sino)

        smap = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)
        feats.append(smap)

        x = torch.cat(feats, dim=1)
        s0 = self.inc(x)
        bott = self.down1(s0)
        x = self.up1(bott, s0)
        return self.outc(x)


# ── VE noise schedule ───────────────────────────────────────────

class ScoreMRIScheduler:
    """Log-spaced noise levels σ₁ > … > σ_K (VE style)."""

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        K: int = 10,
        device: str = "cpu",
    ) -> None:
        self.sigmas = torch.logspace(
            math.log10(sigma_max), math.log10(sigma_min), steps=K, device=device
        )

    def __len__(self) -> int:
        return self.sigmas.numel()

    def to(self, device: str) -> "ScoreMRIScheduler":
        self.sigmas = self.sigmas.to(device)
        return self

    def get(self) -> torch.Tensor:
        return self.sigmas


# ── Score-matching loss ──────────────────────────────────────────

def score_matching_loss(
    model: ScoreMRINet,
    gt: torch.Tensor,
    sino: torch.Tensor | None,
    scheduler: ScoreMRIScheduler,
    *,
    anneal_power: float = 2.0,
) -> torch.Tensor:
    """Denoising score matching with annealed weighting.

    Parameters
    ----------
    model : score network.
    gt : (B, 1, H, W) clean target.
    sino : (B, 1, H, W) conditioning or *None*.
    scheduler : noise-level schedule.
    anneal_power : exponent for the per-sigma weighting.
    """
    B = gt.size(0)
    sigmas = scheduler.get()
    idx = torch.randint(0, len(sigmas), (B,), device=gt.device)
    sigma = sigmas[idx]

    noise = torch.randn_like(gt) * sigma.view(B, 1, 1, 1)
    x_noisy = gt + noise

    score_pred = model(x_noisy, sigma, sino)
    target = -noise / (sigma.view(B, 1, 1, 1) ** 2)

    loss = (score_pred - target) ** 2
    w = (sigma ** anneal_power).view(B, 1, 1, 1)
    return (loss * w).mean()


# ── Predictor-corrector sampler ──────────────────────────────────

@torch.no_grad()
def pc_sampler(
    model: ScoreMRINet,
    sino: torch.Tensor,
    *,
    scheduler: ScoreMRIScheduler | None = None,
    sigmas: Sequence[float] | None = None,
    steps_each: int = 1,
    step_size: float = 0.1,
    correct_noise: float = 0.0,
    dc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predictor-corrector reconstruction (Score-MRI style).

    Parameters
    ----------
    model : trained score network.
    sino : (B, 1, H, W) undersampled conditioning.
    scheduler : provides noise levels; alternatively pass ``sigmas`` directly.
    dc_fn : optional data-consistency operator ``(x, sino) → x_dc``.
    """
    device = sino.device
    if scheduler is not None:
        sigmas_t = scheduler.get()
    elif sigmas is not None:
        sigmas_t = torch.as_tensor(sigmas, device=device)
    else:
        raise ValueError("Provide scheduler or sigmas.")

    B, _, H, W = sino.shape
    x = torch.randn(B, 1, H, W, device=device)

    for sigma in reversed(sigmas_t):
        sigma_b = torch.full((B,), float(sigma), device=device)
        for _ in range(steps_each):
            score = model(x, sigma_b, sino)
            x = x + step_size * (sigma ** 2) * score
            if correct_noise > 0:
                x = x + correct_noise * torch.randn_like(x)
        if dc_fn is not None:
            x = dc_fn(x, sino)

    return x.clamp(0, 1)


# ── Training loop ────────────────────────────────────────────────

def train_score_mri(
    cfg: dict[str, Any],
) -> tuple[ScoreMRINet, ScoreMRIScheduler]:
    """Train the Score-MRI baseline from a YAML config.

    Config sections used: ``data``, ``score_mri``, ``training``, ``checkpoint``.

    Returns
    -------
    ``(model, scheduler)``
    """
    from ..data import build_dataset  # late import avoids circular

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    scfg = cfg.get("score_mri", {})

    # Data
    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0)

    # Model
    base = scfg.get("base", 32)
    cond = scfg.get("cond", True)
    model = ScoreMRINet(base=base, cond=cond).to(device)

    sched = ScoreMRIScheduler(
        sigma_min=scfg.get("sigma_min", 0.01),
        sigma_max=scfg.get("sigma_max", 1.0),
        K=scfg.get("K", 10),
        device=str(device),
    )

    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt_dir = pathlib.Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, tcfg["epochs"] + 1):
        model.train()
        run_loss = 0.0
        for cond_batch, gt_batch, *_ in tqdm(train_dl, desc=f"score-mri {ep:02d}", leave=False):
            cond_batch, gt_batch = cond_batch.to(device), gt_batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss = score_matching_loss(model, gt_batch, cond_batch if cond else None, sched)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * gt_batch.size(0)

        train_loss = run_loss / max(len(train_ds), 1)
        val_psnr, val_ssim = validate_score_mri(model, val_dl, sched, cond=cond, device=device)
        print(f"ep {ep:02d} | loss {train_loss:.4f} | val PSNR {val_psnr:.2f}  SSIM {val_ssim:.4f}")

        if ep % cfg["checkpoint"].get("save_every", 5) == 0:
            torch.save(model.state_dict(), ckpt_dir / f"scoremri_ep{ep:02d}.pt")

    torch.save(model.state_dict(), ckpt_dir / "scoremri_final.pt")
    return model, sched


@torch.no_grad()
def validate_score_mri(
    model: ScoreMRINet,
    val_dl: DataLoader,
    sched: ScoreMRIScheduler,
    *,
    cond: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Quick validation: PC-sample each slice and compute PSNR/SSIM."""
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    model.eval()
    ps, ss = [], []
    for cond_batch, gt_batch, *_ in tqdm(val_dl, desc="val score-mri", leave=False):
        cond_batch, gt_batch = cond_batch.to(device), gt_batch.to(device)
        recon = pc_sampler(model, cond_batch, scheduler=sched, dc_fn=lambda x, y: 0.5 * x + 0.5 * y)
        g = gt_batch.cpu().numpy().squeeze()
        r = recon.cpu().numpy().squeeze()
        ps.append(_psnr(g, r, data_range=1.0))
        ss.append(_ssim(g, r, data_range=1.0))
    return float(np.mean(ps)), float(np.mean(ss))
