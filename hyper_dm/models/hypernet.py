"""HyperNetwork that maps a latent code *z* to the full parameter vector of a target model."""

from __future__ import annotations

import torch
import torch.nn as nn


def n_params(model: nn.Module) -> int:
    """Total number of learnable scalar parameters."""
    return sum(p.numel() for p in model.parameters())


def flat_to_state(
    vec: torch.Tensor,
    template: nn.Module,
) -> dict[str, torch.Tensor]:
    """Reshape a flat parameter vector into a ``{name: tensor}`` state dict.

    Parameters
    ----------
    vec : (D,) flat vector of length ``n_params(template)``.
    template : the target network whose ``.named_parameters()`` defines the layout.
    """
    out: dict[str, torch.Tensor] = {}
    idx = 0
    for name, param in template.named_parameters():
        n = param.numel()
        out[name] = vec[idx : idx + n].view_as(param)
        idx += n
    return out


class HyperNet(nn.Module):
    """Two-hidden-layer MLP:  ``z → hidden → hidden → target_dim``.

    Parameters
    ----------
    target_model : the backbone whose parameters will be generated.
    z_dim : dimensionality of the latent code *z*.
    hidden : width of the two hidden layers.
    """

    def __init__(
        self,
        target_model: nn.Module,
        z_dim: int = 8,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        target_dim = n_params(target_model)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent codes to flat parameter vectors.

        Parameters
        ----------
        z : (B, z_dim)

        Returns
        -------
        (B, target_dim)
        """
        return self.net(z)
