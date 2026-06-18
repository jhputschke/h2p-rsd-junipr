"""A learned per-jet lower bound on MAP multiplicity (quantile floor of P(n|x)).

The joint-argmax MAP is length-biased *low* even after the hard `min_emissions`
floor. The model's own length distribution P(n|x) is its unbiased length belief, so
flooring the MAP length at a low quantile of P(n|x) transfers that belief into the
point estimate and cuts the residual under-count. One knob `alpha` spans it:
`alpha->0` reproduces today's hard floor; `alpha->median` ~ a length-conditioned MAP
at that quantile.

These helpers are deliberately kept out of the parity-critical `point_estimate.py`:
the effective floor `max(min_emissions, quantile_floor(P(n|x)))` is passed straight
into the unchanged `map_estimate` as `min_emissions=`, so `alpha=0.0` short-circuits
before any new code path runs (structural parity preserved).
"""

from __future__ import annotations

import numpy as np


def quantile_floor(pmf: np.ndarray, alpha: float) -> int:
    """Smallest n with cdf(n) >= alpha, clamped to [0, len(pmf)-1].

    `pmf` is a length distribution over n=0,1,...; `np.searchsorted` on the cdf gives
    the quantile index directly. alpha<=0 -> 0 (the empty-tree bound)."""
    pmf = np.asarray(pmf, dtype=float)
    if pmf.size == 0:
        return 0
    cdf = np.cumsum(pmf)
    n = int(np.searchsorted(cdf, alpha))
    return int(np.clip(n, 0, pmf.size - 1))


def learned_min_emissions(
    model, xf, nx, *, quantile: float, base_floor: int, mults=None, n_samples: int = 500
) -> int:
    """Effective per-jet MAP floor = max(base_floor, quantile_floor(P(n|x), quantile)).

    `quantile<=0` short-circuits to `base_floor` with no pmf computed (so the learned
    floor is strictly opt-in and only ever *raises* the bound, preserving the n>=1
    guarantee). Otherwise it reads the model's P(n|x) via `model.length_pmf` (reusing
    the caller's `mults` for sampler-based families, exact for cINN/diffusion)."""
    if quantile <= 0.0:
        return int(base_floor)
    pmf = model.length_pmf(xf, nx, mults=mults, n_samples=n_samples)
    return max(int(base_floor), quantile_floor(pmf, quantile))
