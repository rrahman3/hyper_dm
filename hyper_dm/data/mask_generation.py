"""LUNA16 mask generation from annotations.csv using SimpleITK.

Creates per-slice binary nodule masks from world-coordinate annotations.
"""

from __future__ import annotations

import pathlib

import numpy as np
from tqdm.auto import tqdm


def find_mhd(uid: str, raw_dirs: list[pathlib.Path]) -> pathlib.Path:
    """Locate the ``.mhd`` file for a given UID across LUNA16 subsets."""
    for d in raw_dirs:
        p = d / f"{uid}.mhd"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {uid}.mhd in any of {raw_dirs}")


def generate_masks(
    ct_dir: str | pathlib.Path,
    mask_dir: str | pathlib.Path,
    annotations_csv: str | pathlib.Path,
    raw_dirs: list[str | pathlib.Path],
    *,
    skip_existing: bool = True,
) -> int:
    """Generate per-slice binary nodule masks from LUNA16 annotations.

    Parameters
    ----------
    ct_dir : directory with preprocessed CT slice ``.npy`` files
             (naming: ``{uid}_{sliceidx:04d}.npy``).
    mask_dir : output directory for mask ``.npy`` files (same naming).
    annotations_csv : path to ``annotations.csv`` with columns
                      ``seriesuid, coordX, coordY, coordZ, diameter_mm``.
    raw_dirs : list of LUNA16 subset directories containing ``.mhd`` files.
    skip_existing : skip generation if ``mask_dir`` already has files.

    Returns
    -------
    Number of mask files written.
    """
    import pandas as pd
    import SimpleITK as sitk

    ct_dir = pathlib.Path(ct_dir)
    mask_dir = pathlib.Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    raw_dirs_p = [pathlib.Path(d) for d in raw_dirs]

    if skip_existing and any(mask_dir.glob("*.npy")):
        existing = len(list(mask_dir.glob("*.npy")))
        print(f"[mask] {existing} files exist — skipping generation")
        return existing

    ann = pd.read_csv(annotations_csv)
    count = 0

    for uid, grp in tqdm(ann.groupby("seriesuid"), desc="generate masks"):
        ct_files = sorted(ct_dir.glob(f"{uid}_*.npy"))
        if not ct_files:
            continue

        try:
            itk = sitk.ReadImage(str(find_mhd(uid, raw_dirs_p)))
        except FileNotFoundError:
            continue

        org = np.array(itk.GetOrigin())
        sp = np.array(itk.GetSpacing())
        size = np.array(itk.GetSize())

        H, W = np.load(ct_files[0]).shape[-2:]
        scale_x, scale_y = W / size[0], H / size[1]

        masks: dict[int, np.ndarray] = {}
        for _, r in grp.iterrows():
            vx, vy, vz = ((np.array([r.coordX, r.coordY, r.coordZ]) - org) / sp).round().astype(int)
            if 0 <= vz < size[2]:
                rx, ry = int(vx * scale_x), int(vy * scale_y)
                rad = int((r.diameter_mm / 2) / sp[0] * scale_x)
                canvas = masks.setdefault(vz, np.zeros((H, W), np.uint8))
                yy, xx = np.ogrid[:H, :W]
                canvas[(yy - ry) ** 2 + (xx - rx) ** 2 <= rad ** 2] = 1

        for p in ct_files:
            idx = int(p.stem.split("_")[-1])
            m = masks.get(idx, np.zeros((H, W), np.uint8))
            np.save(mask_dir / p.name, m[None])
            count += 1

    print(f"[mask] generated {count} files → {mask_dir}")
    return count
