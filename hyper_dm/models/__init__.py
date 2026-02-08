"""Model registry — public API for the models sub-package."""

from .blocks import ConvGNAct, SEBlock, DownBlock, UpBlock, ConvBlock
from .unet import (
    UNetTiny,
    UNetTiny3Level,
    HyperUNet,
    SegmentationUNet,
    TinyCNN,
    build_backbone,
)
from .hypernet import HyperNet, n_params, flat_to_state
from .diffusion import (
    GaussianDiffusion,
    NoisePredLoss,
    ddim_sample,
    ddim_sample_steps,
    pairwise_l1,
)
from .score_mri import (
    ScoreMRINet,
    ScoreMRIScheduler,
    score_matching_loss,
    pc_sampler,
    train_score_mri,
    validate_score_mri,
)
from .salm import SALM2D, Decoder2D
from .patch3d import Tiny3DCNN, CandidatePatchDataset

__all__ = [
    # blocks
    "ConvGNAct",
    "SEBlock",
    "DownBlock",
    "UpBlock",
    "ConvBlock",
    # backbones
    "UNetTiny",
    "UNetTiny3Level",
    "HyperUNet",
    "SegmentationUNet",
    "TinyCNN",
    "build_backbone",
    # hypernet
    "HyperNet",
    "n_params",
    "flat_to_state",
    # diffusion
    "GaussianDiffusion",
    "NoisePredLoss",
    "ddim_sample",
    "ddim_sample_steps",
    "pairwise_l1",
    # score-mri baseline
    "ScoreMRINet",
    "ScoreMRIScheduler",
    "score_matching_loss",
    "pc_sampler",
    "train_score_mri",
    "validate_score_mri",
    # salm
    "SALM2D",
    "Decoder2D",
    # 3d patch
    "Tiny3DCNN",
    "CandidatePatchDataset",
]
