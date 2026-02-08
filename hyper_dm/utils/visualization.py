"""Visualization utilities for Hyper-DM reconstruction and uncertainty maps."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Helpers ──────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    """Convert tensor or array to squeezed numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().squeeze()
    return np.asarray(x).squeeze()


def _robust_minmax(
    arr: np.ndarray,
    clip: tuple[float, float] = (0.02, 0.995),
) -> tuple[float, float]:
    """Quantile-based robust min/max (avoids outlier stretching)."""
    lo, hi = np.quantile(arr, clip)
    return float(lo), float(hi)


def _resize_to(img: np.ndarray, size: int, mode: str = "bilinear") -> np.ndarray:
    """Resize a 2-D image to ``(size, size)`` via torch interpolation."""
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    kw = {"align_corners": False} if mode != "nearest" else {}
    t_r = F.interpolate(t, size=(size, size), mode=mode, **kw)
    return t_r.squeeze().numpy()


# ── 5-panel grid: Cond | GT | Recon | AU | EU ────────────────────

def show_results(
    img_cond,
    img_gt,
    recon,
    au,
    eu,
    sid: str = "",
    *,
    cmap_img: str = "gray",
    cmap_unc: str = "inferno",
    clip: tuple[float, float] = (0.02, 0.995),
    share_scale: bool = False,
    au_vmin: float | None = None,
    au_vmax: float | None = None,
    eu_vmin: float | None = None,
    eu_vmax: float | None = None,
    cbar_width_in: float = 0.16,
    cbar_pad_in: float = 0.06,
    cbar_ticksize: int = 20,
    title_size: int = 12,
    figsize: tuple[int, int] = (18, 4),
    single_figsize: tuple[float, float] = (4.0, 4.2),
    layout: str = "grid",
    save_path: str | None = None,
) -> None:
    """Plot the standard 5-panel Hyper-DM visualisation.

    Parameters
    ----------
    img_cond : conditioning image (zero-filled / sinogram).
    img_gt : ground-truth image.
    recon : mean reconstruction.
    au, eu : aleatoric / epistemic uncertainty maps.
    layout : ``"grid"`` for a single 1×5 figure, ``"separate"`` for 5 figures.
    save_path : if given, save the figure(s) instead of calling ``plt.show()``.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualisation — pip install matplotlib")

    titles = ["Condition", "Ground Truth", "Reconstruction", "Aleatoric Unc.", "Epistemic Unc."]
    imgs = [_to_np(img_cond), _to_np(img_gt), _to_np(recon), _to_np(au), _to_np(eu)]
    cmaps = [cmap_img, cmap_img, cmap_img, cmap_unc, cmap_unc]

    # ---------- Compute AU/EU colour scales ----------
    if share_scale and any(v is None for v in (au_vmin, au_vmax, eu_vmin, eu_vmax)):
        lo, hi = _robust_minmax(np.concatenate([imgs[3].ravel(), imgs[4].ravel()]), clip)
        au_vmin = au_vmin if au_vmin is not None else lo
        au_vmax = au_vmax if au_vmax is not None else hi
        eu_vmin = eu_vmin if eu_vmin is not None else lo
        eu_vmax = eu_vmax if eu_vmax is not None else hi
    else:
        if au_vmin is None or au_vmax is None:
            lo, hi = _robust_minmax(imgs[3], clip)
            au_vmin = au_vmin if au_vmin is not None else lo
            au_vmax = au_vmax if au_vmax is not None else hi
        if eu_vmin is None or eu_vmax is None:
            lo, hi = _robust_minmax(imgs[4], clip)
            eu_vmin = eu_vmin if eu_vmin is not None else lo
            eu_vmax = eu_vmax if eu_vmax is not None else hi

    def _draw(ax, idx, im_kwargs):
        im = ax.imshow(imgs[idx], cmap=cmaps[idx], **im_kwargs)
        ax.set_title(titles[idx], fontsize=title_size)
        ax.axis("off")
        return im

    def _add_cbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size=axes_size.Fixed(cbar_width_in),
            pad=axes_size.Fixed(cbar_pad_in),
        )
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=cbar_ticksize)

    if layout == "grid":
        fig, axes = plt.subplots(1, 5, figsize=figsize)
        for i, ax in enumerate(axes):
            kw = {}
            if i == 3:
                kw = dict(vmin=au_vmin, vmax=au_vmax)
            elif i == 4:
                kw = dict(vmin=eu_vmin, vmax=eu_vmax)
            im = _draw(ax, i, kw)
            if i in (3, 4):
                _add_cbar(fig, ax, im)
        if sid:
            fig.suptitle(sid, fontsize=title_size + 2)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.02)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
    else:
        for i in range(5):
            fig, ax = plt.subplots(1, 1, figsize=single_figsize)
            kw = {}
            if i == 3:
                kw = dict(vmin=au_vmin, vmax=au_vmax)
            elif i == 4:
                kw = dict(vmin=eu_vmin, vmax=eu_vmax)
            im = _draw(ax, i, kw)
            if i in (3, 4):
                _add_cbar(fig, ax, im)
            fig.subplots_adjust(left=0.02, right=0.98)
            if save_path:
                base, ext = save_path.rsplit(".", 1) if "." in save_path else (save_path, "png")
                fig.savefig(f"{base}_{titles[i].replace(' ', '_').lower()}.{ext}",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()


# ── Diffusion trajectory ──────────────────────────────────────────

def plot_diffusion_steps(
    all_steps: list[torch.Tensor],
    img_cond: torch.Tensor,
    img_gt: torch.Tensor,
    sid: str = "sample",
    *,
    max_panels: int = 7,
    figsize_per_col: float = 3.0,
    save_path: str | None = None,
) -> None:
    """Visualise selected DDIM denoising steps with per-step PSNR/SSIM.

    Parameters
    ----------
    all_steps : list of (B, 1, H, W) tensors, ordered from step 0 (noisiest)
                to step T (cleanest).
    img_cond : conditioning image.
    img_gt : ground-truth image.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required")
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    num_steps = len(all_steps)
    indices = np.linspace(0, num_steps - 1, min(max_panels, num_steps), dtype=int)
    ncols = len(indices) + 2  # +2 for cond & gt

    fig, axes = plt.subplots(1, ncols, figsize=(figsize_per_col * ncols, 4))

    axes[0].imshow(_to_np(img_cond), cmap="gray")
    axes[0].set_title("Condition")
    axes[0].axis("off")

    axes[1].imshow(_to_np(img_gt), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    gt_np = _to_np(img_gt)
    for i, k in enumerate(indices):
        pred_np = _to_np(all_steps[k])
        p = _psnr(gt_np, pred_np, data_range=1.0)
        s = _ssim(gt_np, pred_np, data_range=1.0)
        ax = axes[i + 2]
        ax.imshow(pred_np, cmap="gray")
        ax.set_title(f"Step {k + 1}", fontsize=12)
        ax.set_xlabel(f"PSNR={p:.2f}\nSSIM={s:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Diffusion Trajectory: {sid}", fontsize=18)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_diffusion_steps_grid(
    all_steps: list[torch.Tensor],
    img_cond: torch.Tensor,
    img_gt: torch.Tensor,
    sid: str = "sample",
    *,
    ncols: int = 4,
    max_steps: int = 25,
    save_path: str | None = None,
) -> dict[str, float]:
    """Grid layout for diffusion trajectory with best-step metrics.

    Returns dict with ``max_psnr``, ``max_ssim`` and their step indices.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required")
    import math as _math
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    num_steps = len(all_steps)
    step_imgs = min(num_steps, max_steps)
    indices = np.linspace(0, num_steps - 1, step_imgs, dtype=int)

    total_imgs = step_imgs + 2
    nrows = _math.ceil(total_imgs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    axes[0].imshow(_to_np(img_cond), cmap="gray")
    axes[0].set_title("Condition")
    axes[0].axis("off")
    axes[1].imshow(_to_np(img_gt), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    gt_np = _to_np(img_gt)
    psnrs: dict[int, float] = {}
    ssims: dict[int, float] = {}

    for i, k in enumerate(indices):
        pred_np = _to_np(all_steps[k])
        p = _psnr(gt_np, pred_np, data_range=1.0)
        s = _ssim(gt_np, pred_np, data_range=1.0)
        psnrs[k] = p
        ssims[k] = s
        ax = axes[i + 2]
        ax.imshow(pred_np, cmap="gray")
        ax.set_title(f"Step {num_steps - k + 1}", fontsize=12)
        ax.set_xlabel(f"PSNR={p:.2f}\nSSIM={s:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(total_imgs, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Diffusion Trajectory: {sid}", fontsize=18)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    best_p_k = max(psnrs, key=psnrs.get)
    best_s_k = max(ssims, key=ssims.get)
    return {
        "max_psnr": psnrs[best_p_k],
        "max_psnr_step": int(num_steps - best_p_k),
        "max_ssim": ssims[best_s_k],
        "max_ssim_step": int(num_steps - best_s_k),
    }
