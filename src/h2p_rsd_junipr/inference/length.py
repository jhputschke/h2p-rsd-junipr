"""Per-jet length decisions read off the model's own P(n|x): a lower bound (the
quantile floor) and an emptiness ceiling (the empty gate).

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

`empty_gate` / `empty_threshold_for_rate` are the mirror image — a *ceiling* rather
than a floor. The parton target really is the empty tree for ~17% of jets, and no
point estimator under the default decode can say so: the MAP is `argmax_n q(n|x)`,
whose peak lands at 0 essentially never, and MBR's imbalance penalty makes an empty
cloud near-maximal risk. That is a property of the decision rule, not of the fit —
the model separates the two classes at AUC 0.77. `models.base.map_or_mbr` consumes
them, before any shape decode (docs/PLAN_empty_parton_tree.md).
"""

from __future__ import annotations

import math

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


def recalibrate_pmf(pmf, temperature: float = 1.0, tilt: float = 0.0) -> np.ndarray:
    """`softmax(log p / T + tilt * n)` — post-hoc affine recalibration of a length belief.

    Two scalars, fitted on held-out jets, no retraining:

    * `temperature` rescales about the mode — `T > 1` flattens toward uniform, `T < 1`
      sharpens. It is the plan's original proposal and is the `tilt = 0` special case.
    * `tilt` adds a term LINEAR IN n, which is what moves mass between short and long
      trees. A negative tilt shifts toward the empty end.

    The tilt is not decoration: the measured miscalibration of `ar_junipr_v3`'s head is a
    *monotone ramp* across n (empirical/predicted = 1.90, 0.96, 0.93, 0.80, 0.68, 0.50 at
    n = 0..5), and a temperature cannot produce a ramp — it is symmetric about the mode.
    See `fit_length_recalibration`.

    Both defaults are the identity and return the input untouched."""
    p = np.asarray(pmf, dtype=float)
    if (temperature == 1.0 and tilt == 0.0) or p.size == 0:
        return p
    if temperature <= 0.0:
        raise ValueError(f"length temperature must be > 0, got {temperature!r}")
    z = np.log(np.clip(p, 1e-300, None)) / float(temperature)
    if tilt:
        z = z + float(tilt) * np.arange(p.size, dtype=float)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def _length_nll(logp, obs, temperature: float, tilt: float) -> float:
    total = 0.0
    for z, n in zip(logp, obs):
        if n >= z.size:          # past the categorical support: skip, never clamp —
            continue             # clamping is the bias the WP4 support guard exists for
        s = z / temperature + (tilt * np.arange(z.size) if tilt else 0.0)
        total += float(np.log(np.exp(s - s.max()).sum()) + s.max() - s[n])
    return total / max(len(logp), 1)


def _golden(f, lo: float, hi: float, tol: float) -> float:
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def fit_length_recalibration(pmfs, n_true, *, with_tilt: bool = True,
                             tol: float = 1e-4) -> tuple[float, float]:
    """`(temperature, tilt)` minimising the NLL of the observed multiplicities.

    Takes the UNCALIBRATED pmfs (one per held-out jet) rather than logits, so it is
    family-agnostic: softmax is shift-invariant, so `log p` serves as the logits.
    Coordinate descent with golden-section on each axis — the objective is smooth and
    2-D, and this keeps the package free of a scipy dependency.

    `with_tilt=False` restricts it to the plan's original scalar temperature, which is
    kept because it is the honest baseline: on `ar_junipr_v3` it fits `T = 1.10` and buys
    0.0008 nat, leaving `mean q(0|x)` at 0.091 against a truth of 0.161. **No scalar
    temperature can close that gap** — sweeping `T` over `[0.1, 20]` tops out at 0.125,
    because `q(0|x)` sits above uniform and below the mode, so flattening pulls it *down*
    toward `1/(max_emissions+1)` and sharpening pulls it to zero.

    Fit on held-out jets and FREEZE, exactly like `empty_threshold_for_rate`."""
    obs = np.asarray(n_true, dtype=int)
    logp = [np.log(np.clip(np.asarray(p, dtype=float), 1e-300, None)) for p in pmfs]
    if not logp or obs.size != len(logp):
        raise ValueError("fit_length_recalibration needs one pmf per observed multiplicity")

    t, b = 1.0, 0.0
    for _ in range(6 if with_tilt else 1):
        # `_b=b` / `_t=t` bind the current iterate into the closure rather than leaving it
        # to late-binding: `_golden` runs immediately so it would be correct either way,
        # but the default-arg form is what makes that independent of the caller.
        t = math.exp(_golden(lambda lt, _b=b: _length_nll(logp, obs, math.exp(lt), _b),
                             math.log(0.05), math.log(20.0), tol))
        if not with_tilt:
            break
        nb = _golden(lambda x, _t=t: _length_nll(logp, obs, _t, x), -3.0, 3.0, tol)
        if abs(nb - b) < tol:
            b = nb
            break
        b = nb
    return float(t), float(b)


def empty_gate(pmf, tau: float) -> bool:
    """True when the model's own `P(N=0|x)` clears `tau` — decide the EMPTY tree.

    `tau <= 0` is always False, so the default decode never enters this path. The
    comparison is `>=` so a `tau` returned by `empty_threshold_for_rate` fires on the
    jet that defined it."""
    if tau <= 0.0:
        return False
    pmf = np.asarray(pmf, dtype=float)
    return bool(pmf.size) and bool(pmf[0] >= tau)


def empty_threshold_for_rate(pmfs, rate: float) -> float:
    """The `tau` reproducing a target empty `rate` over `pmfs` (one per held-out jet).

    Thresholds the RANKING of `q(0|x)`, so the head's miscalibrated *scale* (the ~2x
    under-confidence in docs/PLAN_empty_parton_tree.md F5) does not move it — only the
    ordering matters. Fit on held-out jets and FREEZE: this is a quantile, hence
    sample-dependent, and a tau carried across a selection change silently mis-sets the
    rate. Re-fit per pT window.

    Ties are resolved in favour of firing, so the achieved rate is `rate` or slightly
    above it — except for jets holding exactly zero mass at n=0, which can never be
    called empty and so cap the achievable rate from below."""
    q0 = np.array([float(p[0]) if len(p) else 0.0 for p in pmfs], dtype=float)
    if q0.size == 0:
        return float("inf")                       # nothing to fit on -> never fire
    if rate <= 0.0:                               # above every jet's q0 -> never fire
        return float(np.nextafter(float(q0.max()), np.inf))
    k = min(max(int(round(float(rate) * q0.size)), 1), q0.size)
    tau = float(np.sort(q0)[::-1][k - 1])
    # tau <= 0 is `empty_gate`'s "off" sentinel, and a jet with q(0|x) == 0 must never be
    # called empty anyway, so lift it just above zero rather than silently disabling.
    return tau if tau > 0.0 else float(np.nextafter(0.0, 1.0))


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
