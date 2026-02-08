"""Export Hyper-DM reconstructions to disk as .npy files (for downstream tasks)."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.hypernet import flat_to_state
from ..models.diffusion import ddim_sample
from ..data.segmentation import SinogramDataset


@torch.no_grad()
def export_reconstructions(
    unet: nn.Module,
    hnet: nn.Module,
    data_root: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    z_dim: int = 8,
    steps: int = 20,
    batch_size: int = 4,
    M: int = 1,
    N: int = 1,
) -> None:
    """Generate reconstructions from sinograms and save to ``out_dir``.

    When ``M > 1`` or ``N > 1``, saves ``*_mean.npy``, ``*_au.npy``, ``*_eu.npy``.
    Otherwise saves a single ``*.npy`` per sample.

    Parameters
    ----------
    unet, hnet : loaded Hyper-DM models.
    data_root : directory containing ``sino/`` sub-folder.
    out_dir : output directory for ``.npy`` files.
    z_dim : latent dimension of the HyperNet.
    steps : DDIM denoising steps.
    batch_size : loader batch size.
    M : number of weight samples.
    N : number of DDIM draws per weight sample.
    """
    device = next(hnet.parameters()).device
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SinogramDataset(data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    multi = M > 1 or N > 1

    if multi:
        (out_dir / "mean").mkdir(exist_ok=True)
        (out_dir / "au").mkdir(exist_ok=True)
        (out_dir / "eu").mkdir(exist_ok=True)

    for sino_batch, stems in tqdm(dl, desc="export recons"):
        sino_batch = sino_batch.to(device)
        B = sino_batch.size(0)
        img_size = sino_batch.shape[-1]

        if multi:
            preds = torch.zeros(M, N, B, 1, img_size, img_size, device=device)
            for i in range(M):
                w = flat_to_state(hnet(torch.randn(1, z_dim, device=device))[0], unet)
                for j in range(N):
                    preds[i, j] = ddim_sample(unet, sino_batch, w, steps=steps)

            mean = preds.mean((0, 1)).cpu().numpy()
            au = preds.var(dim=1).mean(0).cpu().numpy()
            eu = preds.mean(dim=1).var(0).cpu().numpy()

            for k, name in enumerate(stems):
                np.save(out_dir / "mean" / f"{name}.npy", mean[k])
                np.save(out_dir / "au" / f"{name}.npy", au[k])
                np.save(out_dir / "eu" / f"{name}.npy", eu[k])
        else:
            w = flat_to_state(hnet(torch.randn(1, z_dim, device=device))[0], unet)
            recon = ddim_sample(unet, sino_batch, w, steps=steps)
            for k, name in enumerate(stems):
                np.save(out_dir / f"{name}.npy", recon[k].cpu().numpy())
