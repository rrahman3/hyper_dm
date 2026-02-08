"""Hyper-DM training loops: standard, epistemic-uncertainty (EU), and aleatoric-uncertainty (AU)."""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.unet import build_backbone
from ..models.hypernet import HyperNet, flat_to_state
from ..models.diffusion import GaussianDiffusion, NoisePredLoss, ddim_sample, pairwise_l1
from ..data import build_dataset
from ..utils.metrics import psnr, ssim
from ..utils.mri_transforms import nchw_to_rss_image
from ..utils.tracking import RunTracker
from ..utils.profiling import Profiler


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def _prepare_batch_ct(batch: tuple, device: torch.device):
    """Unpack a CT dataset batch → (cond, gt)."""
    sino, ct, *_ = batch
    return sino.to(device), ct.to(device)


def _prepare_batch_knee(batch: tuple, device: torch.device):
    """Unpack a knee single-coil batch → (cond, gt)."""
    cond, gt, *_ = batch
    return cond.to(device), gt.to(device)


def _prepare_batch_brain(batch: dict, device: torch.device):
    """Unpack a brain multi-coil batch → (cond, gt) in image domain."""
    masked = batch["masked_kspace"].to(device)
    full = batch["full_kspace"].to(device)

    # Convert k-space to RSS image domain (per sample)
    conds, gts = [], []
    for i in range(masked.shape[0]):
        c = nchw_to_rss_image(masked[i]).to(device)
        g = nchw_to_rss_image(full[i]).to(device)
        # z-score normalise
        c = (c - c.mean()) / (c.std() + 1e-8)
        g = (g - g.mean()) / (g.std() + 1e-8)
        conds.append(c.unsqueeze(0))
        gts.append(g.unsqueeze(0))
    return torch.cat(conds, dim=0), torch.cat(gts, dim=0)


def _get_batch_fn(dataset_name: str):
    """Return the appropriate batch-unpacking function."""
    if dataset_name == "ct_luna16":
        return _prepare_batch_ct
    if dataset_name == "fastmri_knee":
        return _prepare_batch_knee
    if dataset_name == "fastmri_brain":
        return _prepare_batch_brain
    raise ValueError(f"No batch function for dataset '{dataset_name}'")


# ──────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    unet: nn.Module,
    hnet: nn.Module,
    loader: DataLoader,
    batch_fn,
    z_dim: int,
    steps: int,
    img_shape: tuple[int, ...] | None = None,
) -> tuple[float, float]:
    """Compute mean PSNR & SSIM over a validation set (single weight sample)."""
    device = next(hnet.parameters()).device
    unet.eval()
    hnet.eval()
    psnr_list, ssim_list = [], []

    for batch in loader:
        cond, gt = batch_fn(batch, device)
        w = flat_to_state(hnet(torch.randn(1, z_dim, device=device))[0], unet)
        pred = ddim_sample(unet, cond, w, steps=steps, img_shape=img_shape).clamp(0, 1)
        gt_np = gt.cpu().numpy().squeeze()
        pred_np = pred.cpu().numpy().squeeze()
        psnr_list.append(psnr(gt_np, pred_np))
        ssim_list.append(ssim(gt_np, pred_np))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# ──────────────────────────────────────────────────────────────────
#  Standard training
# ──────────────────────────────────────────────────────────────────

def _train_standard(
    unet: nn.Module,
    hnet: nn.Module,
    diff: GaussianDiffusion,
    train_loader: DataLoader,
    batch_fn,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_dir: pathlib.Path,
    tracker: RunTracker | None = None,
) -> None:
    lossfn = NoisePredLoss(diff)
    tcfg = cfg["training"]
    z_dim = cfg["hypernet"]["z_dim"]
    opt = torch.optim.Adam(hnet.parameters(), lr=tcfg["lr"])
    scaler = torch.amp.GradScaler(enabled=tcfg.get("amp", True))
    prof = Profiler()

    for ep in range(1, tcfg["epochs"] + 1):
        prof.epoch_start()
        unet.eval()
        hnet.train()
        running = 0.0

        for batch in tqdm(train_loader, desc=f"train {ep:03d}", leave=False):
            prof.batch_start()
            cond, gt = batch_fn(batch, device)
            z = torch.randn(1, z_dim, device=device)
            weights = flat_to_state(hnet(z)[0], unet)

            with torch.amp.autocast(device_type=str(device), enabled=tcfg.get("amp", True)):
                loss = lossfn(lambda xt, s, t: unet(xt, s, t, weights), cond, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running += loss.item() * gt.size(0)
            prof.batch_end()

        epoch_loss = running / max(len(train_loader.dataset), 1)
        perf = prof.epoch_end()
        print(f"ep {ep:03d} | loss {epoch_loss:.4f} | {perf['time/epoch_s']:.1f}s"
              f" | GPU peak {perf.get('gpu/mem_peak_GB', 0):.2f} GB")
        if tracker:
            tracker.log_metrics({"train/loss": epoch_loss, **perf}, step=ep)

        if ep % cfg["checkpoint"].get("save_every", 5) == 0:
            tracker.save_checkpoint(
                hnet.state_dict(), f"hnet_ep{ep:03d}.ckpt", ckpt_dir,
            )

    if tracker:
        tracker.log_metrics(prof.run_summary())


# ──────────────────────────────────────────────────────────────────
#  EU (epistemic uncertainty) training
# ──────────────────────────────────────────────────────────────────

def _train_eu(
    unet: nn.Module,
    hnet: nn.Module,
    diff: GaussianDiffusion,
    train_loader: DataLoader,
    batch_fn,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_dir: pathlib.Path,
    tracker: RunTracker | None = None,
) -> None:
    lossfn = NoisePredLoss(diff)
    tcfg = cfg["training"]
    z_dim = cfg["hypernet"]["z_dim"]
    m_weights = tcfg.get("m_weights", 2)
    lambda_eu = tcfg.get("lambda_eu", 0.25)
    steps = tcfg.get("ddim_steps", 20)

    opt = torch.optim.Adam(hnet.parameters(), lr=tcfg["lr"])
    scaler = torch.amp.GradScaler(enabled=tcfg.get("amp", True))
    prof = Profiler()

    for ep in range(1, tcfg["epochs"] + 1):
        prof.epoch_start()
        unet.eval()
        hnet.train()
        running = 0.0

        for batch in tqdm(train_loader, desc=f"EU {ep:03d}", leave=False):
            prof.batch_start()
            cond, gt = batch_fn(batch, device)

            # Sample M weight sets and generate predictions
            preds = []
            last_weights = None
            for _ in range(m_weights):
                z = torch.randn(1, z_dim, device=device)
                weights = flat_to_state(hnet(z)[0], unet)
                last_weights = weights
                with torch.amp.autocast(device_type=str(device), enabled=tcfg.get("amp", True)):
                    x_hat = ddim_sample(unet, cond, weights, steps=steps)
                preds.append(x_hat)
            preds_stack = torch.stack(preds, dim=0)  # (M, B, 1, H, W)

            # Reconstruction loss (last weight set)
            with torch.amp.autocast(device_type=str(device), enabled=tcfg.get("amp", True)):
                rec_loss = lossfn(
                    lambda xt, s, t: unet(xt, s, t, last_weights), cond, gt
                )

            # EU spread loss
            spread = pairwise_l1(preds_stack)
            loss = rec_loss + lambda_eu * spread

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running += loss.item() * gt.size(0)
            prof.batch_end()
            torch.cuda.empty_cache()

        epoch_loss = running / max(len(train_loader.dataset), 1)
        perf = prof.epoch_end()
        print(f"ep {ep:03d} | EU loss {epoch_loss:.4f} | {perf['time/epoch_s']:.1f}s"
              f" | GPU peak {perf.get('gpu/mem_peak_GB', 0):.2f} GB")
        if tracker:
            tracker.log_metrics({"train/eu_loss": epoch_loss, **perf}, step=ep)

        if ep % cfg["checkpoint"].get("save_every", 5) == 0:
            tracker.save_checkpoint(
                hnet.state_dict(), f"hnet_eu_ep{ep:03d}.ckpt", ckpt_dir,
            )

    if tracker:
        tracker.log_metrics(prof.run_summary())


# ──────────────────────────────────────────────────────────────────
#  AU (aleatoric uncertainty) training
# ──────────────────────────────────────────────────────────────────

def _train_au(
    unet: nn.Module,
    hnet: nn.Module,
    diff: GaussianDiffusion,
    train_loader: DataLoader,
    batch_fn,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_dir: pathlib.Path,
    tracker: RunTracker | None = None,
) -> None:
    lossfn = NoisePredLoss(diff)
    tcfg = cfg["training"]
    z_dim = cfg["hypernet"]["z_dim"]
    n_noisy = tcfg.get("n_noisy", 2)
    sigma_noise = tcfg.get("sigma_noise", 0.04)
    lambda_au = tcfg.get("lambda_au", 1.0)

    opt = torch.optim.Adam(hnet.parameters(), lr=tcfg["lr"])
    scaler = torch.amp.GradScaler(enabled=tcfg.get("amp", True))
    prof = Profiler()

    for ep in range(1, tcfg["epochs"] + 1):
        prof.epoch_start()
        unet.eval()
        hnet.train()
        running = 0.0

        for batch in tqdm(train_loader, desc=f"AU {ep:03d}", leave=False):
            prof.batch_start()
            cond, gt = batch_fn(batch, device)

            z = torch.randn(1, z_dim, device=device)
            weights = flat_to_state(hnet(z)[0], unet)

            with torch.amp.autocast(device_type=str(device), enabled=tcfg.get("amp", True)):
                # Standard noise-prediction loss
                loss_rec = lossfn(
                    lambda xt, s, t: unet(xt, s, t, weights), cond, gt
                )

                # AU: build N noisy copies of the conditioning
                noisy_conds = torch.cat(
                    [cond + sigma_noise * torch.randn_like(cond) for _ in range(n_noisy)],
                    dim=0,
                )
                t = torch.randint(
                    1, diff.num_timesteps, (1,), device=device
                ).long().expand(noisy_conds.size(0))

                gt_rep = gt.repeat(n_noisy, 1, 1, 1)
                noise_rep = torch.randn_like(gt_rep)
                xt_noisy = diff.q_sample(gt_rep, t, noise_rep)

                eps_pred = unet(
                    xt_noisy,
                    noisy_conds,
                    t.float() / (diff.num_timesteps - 1),
                    weights,
                )
                eps_pred = eps_pred.view(n_noisy, *gt.shape)
                var_term = eps_pred.var(dim=0, unbiased=False).mean()

                loss = loss_rec + lambda_au * var_term

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running += loss.item() * gt.size(0)
            prof.batch_end()

        epoch_loss = running / max(len(train_loader.dataset), 1)
        perf = prof.epoch_end()
        print(f"ep {ep:03d} | AU loss {epoch_loss:.4f} | {perf['time/epoch_s']:.1f}s"
              f" | GPU peak {perf.get('gpu/mem_peak_GB', 0):.2f} GB")
        if tracker:
            tracker.log_metrics({"train/au_loss": epoch_loss, **perf}, step=ep)

        if ep % cfg["checkpoint"].get("save_every", 5) == 0:
            tracker.save_checkpoint(
                hnet.state_dict(), f"hnet_au_ep{ep:03d}.ckpt", ckpt_dir,
            )

    if tracker:
        tracker.log_metrics(prof.run_summary())


# ──────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────

def train_hyperdm(cfg: dict[str, Any]) -> tuple[nn.Module, nn.Module]:
    """Train a Hyper-DM model from a parsed YAML config.

    Returns ``(unet, hnet)`` after training completes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["data"]["dataset"]

    # Build backbone & freeze
    mcfg = cfg["model"]
    unet = build_backbone(
        mcfg["backbone"],
        in_ch=mcfg.get("in_ch", 3),
        base=mcfg.get("base_ch", 32),
    ).to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    # Build HyperNet
    hcfg = cfg["hypernet"]
    hnet = HyperNet(unet, z_dim=hcfg["z_dim"], hidden=hcfg["hidden"]).to(device)

    # Diffusion schedule
    dcfg = cfg["diffusion"]
    diff = GaussianDiffusion(
        T=dcfg["T"],
        beta_min=dcfg["beta_min"],
        beta_max=dcfg["beta_max"],
        device=str(device),
    )

    # Data
    tcfg = cfg["training"]
    train_ds = build_dataset(cfg, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    # Checkpoint fallback directory (only used when MLflow is disabled)
    ckpt_dir = pathlib.Path(cfg["checkpoint"]["dir"])

    batch_fn = _get_batch_fn(dataset_name)
    mode = tcfg.get("mode", "standard")

    # ── Experiment tracking ──────────────────────────────────────
    tracker = RunTracker(cfg)
    if not tracker.enabled:
        # No MLflow → save to disk in per-run subdirectory
        ckpt_dir = tracker.make_run_dir(ckpt_dir)
        print(f"[run] checkpoints → {ckpt_dir}")
    else:
        print(f"[run] checkpoints → MLflow artifacts (run {tracker.run_id})")

    tracker.log_params(cfg)
    tracker.log_config_yaml(cfg)
    tracker.set_tag("mode", mode)
    tracker.set_tag("dataset", dataset_name)

    if mode == "standard":
        _train_standard(unet, hnet, diff, train_loader, batch_fn, cfg, device, ckpt_dir, tracker)
    elif mode == "eu":
        _train_eu(unet, hnet, diff, train_loader, batch_fn, cfg, device, ckpt_dir, tracker)
    elif mode == "au":
        _train_au(unet, hnet, diff, train_loader, batch_fn, cfg, device, ckpt_dir, tracker)
    else:
        tracker.end()
        raise ValueError(f"Unknown training mode '{mode}'. Use 'standard', 'eu', or 'au'.")

    tracker.end()
    return unet, hnet
