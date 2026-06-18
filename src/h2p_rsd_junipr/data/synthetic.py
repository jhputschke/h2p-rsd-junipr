"""Matched-pair synthetic hadronization simulator (was `_sample_parton_sequence`,
`_hadronize`, `synthetic_matched_dataset`).

This is a qualitative physics stand-in, NOT the Lund discretisation, so its
sampling bounds are the fixed generation ranges (0, 6) of the v2 script — kept
constant here so `synthetic_matched_dataset(n, seed)` is bit-identical to the
script's, which the verification relies on. The real data path is `rntuple.py`.
"""

from __future__ import annotations

import math

import numpy as np

# Generation ranges of the synthetic physics model (match the v2 script globals).
LN_INVDELTA_RANGE = (0.0, 6.0)
LN_KT_RANGE = (0.0, 6.0)


def _sample_parton_sequence(rng: np.random.Generator, max_emissions: int = 20):
    """A parton-level primary Lund sequence: angular ordered, kt drifting down.
    (Qualitative stand-in only; not a physical kinematic configuration.)"""
    n = min(max(1, int(rng.poisson(6))), max_emissions)
    ln_invd = 0.0
    li, lk, lz, ps = [], [], [], []
    for t in range(n):
        ln_invd += rng.exponential(0.8)  # angular ordering: 1/DeltaR grows
        if ln_invd > LN_INVDELTA_RANGE[1]:
            break
        ln_kt = rng.normal(4.0 - 0.3 * t, 1.0)  # kt drifts down deeper in the shower
        if ln_kt < LN_KT_RANGE[0]:
            continue  # below floor -> removed by RSD
        li.append(ln_invd)
        lk.append(ln_kt)
        lz.append(-float(rng.exponential(0.7)) - 0.05)  # ln z < 0 (z < 1)
        ps.append(float(rng.uniform(-math.pi, math.pi)))
    if not li:
        li, lk, lz, ps = [0.5], [3.0], [-0.3], [0.0]
    return (
        np.array(li, np.float32),
        np.array(lk, np.float32),
        np.array(lz, np.float32),
        np.array(ps, np.float32),
    )


def _hadronize(parton, rng: np.random.Generator):
    """Forward model y -> x: a kt-dependent smearing + soft migration. The
    smearing WIDTH grows as ln kt decreases (tight at high kt, loose near the
    floor) -- the property grooming exploits."""
    li, lk, lz, ps = parton
    xi, xk, xz, xp = [], [], [], []
    for a, b, c, d in zip(li, lk, lz, ps):
        sigma = 0.25 + 0.35 * max(0.0, 3.0 - b)  # wider smear at low kt
        p_drop = 0.05 + 0.25 * max(0.0, 2.0 - b)  # soft nodes migrate out of the groomed tree
        if rng.random() < p_drop:
            continue
        xi.append(a + rng.normal(0.0, sigma))
        xk.append(b + rng.normal(0.0, sigma))
        xz.append(min(-1e-3, c + rng.normal(0.0, sigma)))
        xp.append(((d + rng.normal(0.0, 0.3)) + math.pi) % (2 * math.pi) - math.pi)
    if rng.random() < 0.20:  # spurious soft hadron-level declustering
        xi.append(float(rng.uniform(*LN_INVDELTA_RANGE)))
        xk.append(float(rng.uniform(LN_KT_RANGE[0], 2.0)))
        xz.append(-float(rng.exponential(0.7)) - 0.05)
        xp.append(float(rng.uniform(-math.pi, math.pi)))
    if not xi:  # never return empty
        xi, xk, xz, xp = [li[0]], [lk[0]], [lz[0]], [ps[0]]
    order = np.argsort(xi)  # keep angular ordering
    xi = np.clip(np.array(xi, np.float32)[order], *LN_INVDELTA_RANGE)
    xk = np.clip(np.array(xk, np.float32)[order], *LN_KT_RANGE)
    xz = np.array(xz, np.float32)[order]
    xp = np.array(xp, np.float32)[order]
    return xi, xk, xz, xp


def synthetic_matched_dataset(n_jets: int, seed: int = 0, max_emissions: int = 20):
    """List of per-jet dicts {weight, x, y}, bit-identical to the v2 script for a
    given (n_jets, seed)."""
    rng = np.random.default_rng(seed)
    jets = []
    for _ in range(n_jets):
        y = _sample_parton_sequence(rng, max_emissions=max_emissions)
        x = _hadronize(y, rng)
        jets.append(dict(weight=1.0, x=x, y=y, event=None, generator="synthetic"))
    return jets
