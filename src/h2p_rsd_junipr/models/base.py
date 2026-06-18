"""Model abstraction & registry (§3): one contract, many posterior families.

The trainer, validation suite, and serving layer only ever touch `log_prob`,
`sample`, `map_estimate` — they never know which family they hold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..geometry import Geometry
from ..inference.point_estimate import LundPointEstimate

_REGISTRY: dict[str, type[PosteriorModel]] = {}


def register_model(*names: str):
    def deco(cls):
        for n in names:
            _REGISTRY[n] = cls
        return cls

    return deco


class PosteriorModel(nn.Module, ABC):
    @abstractmethod
    def log_prob(self, batch: dict) -> torch.Tensor:
        """(B,) log q_phi(y | x)."""
        ...

    @abstractmethod
    def sample(self, xf: torch.Tensor, nx: torch.Tensor, n: int) -> list:
        """`n` posterior draws (cell chains) for one jet."""
        ...

    @abstractmethod
    def map_estimate(self, xf: torch.Tensor, nx: torch.Tensor) -> LundPointEstimate:
        ...


def build_model(cfg, geometry: Geometry) -> PosteriorModel:
    # import for side-effect registration
    from . import ar_junipr, cinn, diffusion  # noqa: F401

    name = cfg.model.name
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, geometry)


def registered_models() -> list[str]:
    from . import ar_junipr, cinn, diffusion  # noqa: F401

    return sorted(_REGISTRY)
