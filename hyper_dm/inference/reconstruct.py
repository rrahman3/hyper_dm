"""Hyper-DM inference: M×N sampling → mean / AU / EU maps + PSNR / SSIM."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.unet import build_backbone
from ..models.hypernet import HyperNet, flat_to_state
from ..models.diffusion import GaussianDiffusion, ddim_sample
from ..data import build_dataset
from ..utils.metrics import psnr, ssim
from ..utils.mri_transforms import nchw_to_rss_image
from ..utils.profiling import InferenceProfiler
from ..utils.tracking import RunTracker


def load_hyperdm(
    cfg: dict[str, Any],
    hnet_ckpt: str,
    device: str | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Recreate model skeletons and load a trained HyperNet checkpoint.

    Parameters
    ----------
    cfg : parsed YAML config.
    hnet_ckpt : path to the saved ``hnet.state_dict()``.
    device : torch device (auto-detected if *None*).

    Returns
    -------
    ``(unet, hnet)`` in eval mode with gradients disabled.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    hcfg = cfg["hypernet"]

    unet = build_backbone(
        mcfg["backbone"],
        in_ch=mcfg.get("in_ch", 3),
        base=mcfg.get("base_ch", 32),
    ).to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    hnet = HyperNet(unet, z_dim=hcfg["z_dim"], hidden=hcfg["hidden"]).to(device)
    hnet.load_state_dict(torch.load(hnet_ckpt, map_location=device))
    hnet.eval()
    for p in hnet.parameters():
        p.requires_grad_(False)

    return unet, hnet


@torch.no_grad()
def infer_hyperdm(
    cfg: dict[str, Any],
    unet: nn.Module,
    hnet: nn.Module,
    split: str = "test",
    save: bool = True,
    tracker: RunTracker | None = None,
) -> dict[str, float]:
    """Run M×N inference and optionally save mean / AU / EU maps.

    Parameters
    ----------
    cfg : parsed YAML config (needs ``inference`` and ``data`` sections).
    unet, hnet : loaded models from :func:`load_hyperdm`.
    split : dataset split to evaluate on.
    save : whether to save ``.npy`` output maps.
    tracker : optional :class:`RunTracker` – if provided, metrics and output
              images are logged to MLflow and saved in a per-run directory.

    Returns
    -------
    Dict with ``"psnr"`` and ``"ssim"`` averages (if ground truth is available),
    plus profiling metrics (``infer/*``, ``gpu/*``).
    """
    device = next(hnet.parameters()).device
    icfg = cfg["inference"]
    M = icfg["M"]
    N = icfg["N"]
    z_dim = cfg["hypernet"]["z_dim"]
    steps = icfg["ddim_steps"]
    dataset_name = cfg["data"]["dataset"]

    # Per-run output directory (isolated per run_id)
    out_base = pathlib.Path(icfg["output_dir"])
    if tracker:
        out_dir = tracker.make_run_dir(out_base)
    else:
        out_dir = out_base / "default"
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] outputs → {out_dir}")

    ds = build_dataset(cfg, split=split)
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    n_sample_images = icfg.get("n_sample_images", 5)  # how many PNGs to log to MLflow
    print(f"Inference on {len(ds)} samples (M={M}, N={N}, steps={steps})")

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    psnr_list, ssim_list = [], []
    iprof = InferenceProfiler()
    sample_count = 0

    for batch in tqdm(dl, desc="infer"):
        iprof.sample_start()
        # Unpack batch depending on dataset type
        if dataset_name in ("ct_luna16", "fastmri_knee"):
            cond, gt, sids = batch
            cond, gt = cond.to(device), gt.to(device)
            sid = sids[0] if isinstance(sids, (list, tuple)) else sids
        elif dataset_name == "fastmri_brain":
            cond_k = batch["masked_kspace"].to(device)[0]
            gt_k = batch["full_kspace"].to(device)[0]
            cond = nchw_to_rss_image(cond_k).to(device).unsqueeze(0)
            gt = nchw_to_rss_image(gt_k).to(device).unsqueeze(0)
            cond = (cond - cond.mean()) / (cond.std() + 1e-8)
            gt = (gt - gt.mean()) / (gt.std() + 1e-8)
            sid = batch["sid"][0]
        else:
            raise ValueError(f"Inference not implemented for dataset '{dataset_name}'")

        img_size = cond.shape[-1]
        preds = torch.zeros(M, N, 1, 1, img_size, img_size, device=device)

        for i in range(M):
            w_vec = hnet(torch.randn(1, z_dim, device=device))[0]
            weights = flat_to_state(w_vec, unet)
            for j in range(N):
                preds[i, j] = ddim_sample(
                    unet, cond, weights, steps=steps,
                    img_shape=(1, img_size, img_size),
                )

        mean_recon = preds.mean((0, 1)).cpu().numpy().squeeze()
        au_map = preds.var(dim=1).mean(0).cpu().numpy().squeeze()
        eu_map = preds.mean(dim=1).var(0).cpu().numpy().squeeze()

        # Bulk .npy outputs → disk only (never MLflow)
        if save:
            np.save(out_dir / f"{sid}_mean.npy", mean_recon)
            np.save(out_dir / f"{sid}_au.npy", au_map)
            np.save(out_dir / f"{sid}_eu.npy", eu_map)

        # First N samples → PNG to MLflow for visual QA
        if tracker and sample_count < n_sample_images:
            tracker.log_image(mean_recon, f"{sid}_mean.png", artifact_path="samples")
            tracker.log_image(au_map, f"{sid}_au.png", artifact_path="samples", cmap="hot")
            tracker.log_image(eu_map, f"{sid}_eu.png", artifact_path="samples", cmap="hot")
            sample_count += 1

        iprof.sample_end()

        gt_np = gt.cpu().numpy().squeeze()
        try:
            psnr_list.append(psnr(gt_np, mean_recon))
            ssim_list.append(ssim(gt_np, mean_recon))
        except ValueError:
            pass  # shape mismatch — skip metrics

    results = {}
    if psnr_list:
        results["psnr"] = float(np.mean(psnr_list))
        results["ssim"] = float(np.mean(ssim_list))
        print(f"Average PSNR: {results['psnr']:.2f}  SSIM: {results['ssim']:.4f}")

    perf = iprof.summary()
    results.update(perf)
    print(f"Inference: {perf.get('infer/latency_mean_ms', 0):.1f} ms/sample"
          f" | {perf.get('infer/throughput_samples_per_s', 0):.2f} samples/s"
          f" | GPU peak {perf.get('gpu/mem_peak_GB', 0):.2f} GB")

    # ── Log metrics to MLflow (outputs stay on disk) ──────────────
    if tracker:
        tracker.log_metrics(results)

    return results
