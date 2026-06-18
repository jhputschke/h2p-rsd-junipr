"""Model abstraction & registry (§3): one contract, many posterior families.

The trainer, validation suite, and serving layer only ever touch `log_prob`,
`sample`, `map_estimate` — they never know which family they hold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
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

    def length_pmf(self, xf, nx, mults=None, n_samples: int = 500) -> np.ndarray:
        """The model's per-jet length belief P(n|x) as a normalized pmf over n=0,1,...

        Default (sampler-based: AR and any family without an explicit length head):
        the empirical multiplicity histogram of posterior draws. If `mults` (the
        per-draw multiplicities the caller already computed) is given it is reused —
        no second sample. cINN/diffusion override this with their exact softmax head.
        """
        if mults is None:
            mults = [len(d) for d in self.sample(xf, nx, n_samples)]
        counts = np.bincount(np.asarray(mults, dtype=int))
        total = counts.sum()
        if total == 0:
            return np.array([1.0])  # degenerate (no draws): all mass at n=0
        return counts / total


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
