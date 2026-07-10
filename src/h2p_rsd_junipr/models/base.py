"""Model abstraction & registry (§3): one contract, many posterior families.

The trainer, validation suite, and serving layer only ever touch `log_prob`,
`sample`, `map_estimate` — they never know which family they hold.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate

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

    def describe_cells(self, xf, nx, cells) -> LundPointEstimate:
        """One posterior draw (a cell chain) -> LundPointEstimate: nodes placed at
        their Lund-cell centres plus the model's joint log-density of that chain.

        The MBR winner (`inference.mbr.mbr_select`) is a genuine drawn tree, so its
        coordinates are exactly its cell centres. This family-agnostic fallback
        scores it via the trained density (`log_prob`); AR overrides it with its
        staged decode (continuous within-cell coordinate modes). No new abstract
        method — every family inherits MBR through `sample`/`log_prob` alone."""
        geom = self.geometry
        cells = [int(c) for c in cells]
        dev = xf.device
        nodes, rows = [], []
        for t, c in enumerate(cells):
            u, v = geom.cell_center(c)
            rows.append([u, v, 0.0, 0.0])
            nodes.append(
                LundNode(
                    depth=t, parent=t - 1, cell=c,
                    ln_invDelta=u, ln_kt=v, ln_z=0.0, psi=0.0,
                    kt=math.exp(v), delta_R=math.exp(-u), z=1.0,
                    logp_split=0.0, logp_coord=0.0, logp_cont=0.0,
                )
            )
        L = len(cells)
        if L > 0:
            yc = torch.tensor([cells], dtype=torch.long, device=dev)
            yraw = torch.tensor([rows], dtype=torch.float32, device=dev)
        else:
            yc = torch.zeros(1, 0, dtype=torch.long, device=dev)
            yraw = torch.zeros(1, 0, 4, dtype=torch.float32, device=dev)
        batch = {"xf": xf, "nx": nx, "yc": yc,
                 "ny": torch.tensor([L], device=dev), "yraw": yraw}
        with torch.inference_mode():
            logprob = float(self.log_prob(batch)[0])
        return LundPointEstimate(nodes=nodes, logprob=logprob, multiplicity=L)

    def map_or_mbr(self, xf, nx, *, draws=None, **decode) -> LundPointEstimate:
        """Point estimate dispatched by ``decode['point_estimator']``: ``"map"``
        (default) -> ``map_estimate``; ``"mbr"`` -> minimum-Bayes-risk selection over
        posterior draws (`inference.mbr.mbr_select`, reusing ``draws`` when given).
        A thin convenience so all three families gain MBR with no per-family code;
        the ``"map"`` branch imports no OT backend, preserving parity."""
        if str(decode.get("point_estimator", "map")) == "mbr":
            from ..inference.mbr import mbr_kwargs_from_decode, mbr_select

            return mbr_select(self, xf, nx, draws=draws, geom=self.geometry,
                              **mbr_kwargs_from_decode(decode))
        return self.map_estimate(xf, nx, **decode)

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
