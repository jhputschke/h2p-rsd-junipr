"""Continuous density helpers (was the `_gauss_logpdf` etc. block).

Device-safe: only elementwise ops, no `torch.special`, so every term runs and
differentiates on MPS. Each `*_logpdf` returns a proper log-density; the unit
tests check they integrate to 1.

Each density is paired with its **CDF** (`std_normal_cdf`, `trunc_normal_cdf`,
`vonmises_cdf`) — the probability-integral transforms the per-coordinate PIT
calibration diagnostic evaluates at the truth (`eval/calibration.py`, WP2 of
docs/PLAN_UPDATES.md). The CDFs share the logpdfs' device-safety constraint.

The two bounded densities are also paired with a **sampler**
(`trunc_normal_sample`, `vonmises_sample`), which is what turns a coordinate head
into a posterior draw (`PosteriorModel.sample_coordinates`). `torch.distributions`
is deliberately not used: `VonMises.sample` casts to float64 and therefore raises
on MPS, and neither sampler there accepts the per-element bounds this head needs.
"""

from __future__ import annotations

import math

import torch

_LOG_2PI = math.log(2.0 * math.pi)


def gauss_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * z * z - torch.log(sigma) - 0.5 * _LOG_2PI


def std_normal_cdf(t):
    return 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))


def gauss_cdf(x, mu, sigma):
    """CDF of N(mu, sigma^2) — the PIT of a Gaussian coordinate head."""
    return std_normal_cdf((x - mu) / sigma)


def trunc_normal_logpdf(x, mu, sigma, lo, hi):
    """log density of N(mu, sigma^2) TRUNCATED to [lo, hi] (scalars), at x in
    [lo, hi]. Subtracting the in-interval mass makes the within-cell offset a
    proper density, so cell-prob x offset-density integrates to a proper density
    over (ln 1/DeltaR, ln kt)."""
    base = gauss_logpdf(x, mu, sigma)
    Z = (std_normal_cdf((hi - mu) / sigma) - std_normal_cdf((lo - mu) / sigma)).clamp(min=1e-6)
    return base - torch.log(Z)


def trunc_normal_cdf(x, mu, sigma, lo, hi):
    """CDF of the same truncated normal `trunc_normal_logpdf` describes:

        F(x) = (Phi((x-mu)/s) - Phi((lo-mu)/s)) / (Phi((hi-mu)/s) - Phi((lo-mu)/s)).

    Built from the *same* `std_normal_cdf` the normalizer already uses, so the PIT
    and the likelihood cannot drift apart. Values are clamped to [0, 1] for x
    outside [lo, hi] (the caller clamps the within-cell offset the same way)."""
    a = std_normal_cdf((lo - mu) / sigma)
    b = std_normal_cdf((hi - mu) / sigma)
    return ((std_normal_cdf((x - mu) / sigma) - a) / (b - a).clamp(min=1e-6)).clamp(0.0, 1.0)


def trunc_normal_icdf(q, mu, sigma, lo, hi):
    """Inverse CDF of the truncated normal at `q` in [0, 1] — the quantile function.

        x = mu + sigma * Phi^{-1}(a + q (b - a)),  a = Phi((lo-mu)/s), b = Phi((hi-mu)/s).

    `trunc_normal_sample` IS this evaluated at a uniform draw, and saying so in code
    rather than in a comment is what keeps the sampler and the CDF from drifting — they
    are now the same expression read in two directions. `torch.erfinv` is elementwise and
    runs on MPS."""
    # clamp keeps erfinv off its +-inf endpoints when the interval sits far in a tail
    a = std_normal_cdf((lo - mu) / sigma)
    b = std_normal_cdf((hi - mu) / sigma)
    p = (a + (b - a) * q).clamp(1e-6, 1.0 - 1e-6)
    return (mu + sigma * math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)).clamp(lo, hi)


def trunc_normal_sample(mu, sigma, lo, hi, *, generator=None):
    """One draw from the truncated normal `trunc_normal_logpdf` describes, per element
    of the broadcast `(mu, sigma)`, by inverting its CDF:

        x = mu + sigma * Phi^{-1}(a + U (b - a)),  a = Phi((lo-mu)/s), b = Phi((hi-mu)/s).

    Exact (no rejection loop, so no data-dependent runtime) and built from the same
    `std_normal_cdf` the density's normalizer uses, so sampler and likelihood cannot
    drift apart."""
    mu, sigma = torch.broadcast_tensors(torch.as_tensor(mu), torch.as_tensor(sigma))
    u = torch.rand(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return trunc_normal_icdf(u, mu, sigma, lo, hi)


# ---------------------------------------------------------------------------
# Monotone rational-quadratic spline (docs/PLAN_lnz_spline_head.md)
# ---------------------------------------------------------------------------
# Durkan, Bekasov, Murray & Papamakarios, *Neural Spline Flows*, arXiv:1906.04032,
# eqs. (4)-(8). A monotone map S: [0, 1] -> [0, 1] built from K rational-quadratic
# pieces, parameterised by K widths, K heights and the K-1 INTERNAL knot derivatives
# (the two boundary derivatives are pinned to 1, which is what makes the identity
# reachable and keeps S onto [0, 1] exactly).
#
# The spline is placed on the soft-drop interval through the AFFINE map
#
#     t = (x - lo) / (hi - lo),   F(x) = S(t),   p(x) = S'(t) / (hi - lo),
#
# so the density on (lo, hi] is the spline's own derivative and nothing else. The support
# closure v1's WP-A bought (0.83% below-soft-drop and 3.94% above z = 1/2 -> 0.0000%) is
# kept by construction: t is an affine bijection of the interval onto [0, 1] and S maps
# [0, 1] onto itself, so no draw can leave.
#
# WHY THE BASE HAS NO PARAMETERS, which is a measured decision and not a simplification.
# The first implementation warped the TRUNCATED NORMAL's CDF instead, `F(x) = S(F_tn(x))`,
# to keep today's head as the identity special case. That parameterization is
# NON-IDENTIFIABLE: once S carries the shape, any (mu, sigma) leaving F_tn roughly linear
# on the interval gives the same composed density, so the pair drifts along a flat
# direction. Measured on seed 2 of the first 3-seed run, it drifted until it broke —
# `lnz_mean` reached -533 against an interval of [-2.303, -0.693], `lnz_sig` reached 85,
# F_tn saturated to 0 or 1 on 100% of emissions, the gradient through S died and val NLL
# went 4.19 -> 19.2 at epoch 4 and never recovered. Seeds 0 and 1 were on the same flat
# direction and had merely not walked as far. A fixed base removes the redundancy at the
# root rather than bounding its symptom; `lnz_head="truncnorm"` remains available as its
# own path, which is what the parity flag is for.
_MIN_BIN = 1e-3      # floor on a bin's width/height: keeps s = h/w finite
_MIN_DERIV = 1e-3    # floor on a knot derivative: keeps the map strictly increasing
# Shift that makes raw == 0 the IDENTITY spline: with uniform widths/heights every
# s_k = 1, so the piece is linear exactly when its knot derivatives are 1, and
# `softplus(_IDENTITY_SHIFT) + _MIN_DERIV == 1`. A head whose last layer starts near
# zero therefore starts near the truncated normal it generalises.
_IDENTITY_SHIFT = math.log(math.exp(1.0 - _MIN_DERIV) - 1.0)


def rq_spline_n_params(n_bins: int) -> int:
    """How many numbers per node the head must emit for a `n_bins`-piece spline."""
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError(f"an RQ spline needs at least 2 bins, got {n_bins}")
    return 3 * n_bins - 1


def _rq_knots(raw, n_bins: int):
    """Raw `(..., 3K-1)` head output -> `(cum_x, cum_y, deriv)` knots.

    `cum_x`/`cum_y` are `(..., K+1)` and start at 0 and end at 1 by construction (the
    softmax sums to one and the floors are subtracted back out), so no renormalisation
    is needed downstream and the map is onto [0, 1] whatever the head emits."""
    K = int(n_bins)
    span = 1.0 - K * _MIN_BIN
    w = torch.softmax(raw[..., :K], dim=-1) * span + _MIN_BIN
    h = torch.softmax(raw[..., K:2 * K], dim=-1) * span + _MIN_BIN
    inner = _MIN_DERIV + torch.nn.functional.softplus(raw[..., 2 * K:] + _IDENTITY_SHIFT)
    ones = torch.ones_like(inner[..., :1])
    d = torch.cat([ones, inner, ones], dim=-1)                     # (..., K+1)
    zero = torch.zeros_like(w[..., :1])
    cum_x = torch.cat([zero, torch.cumsum(w, dim=-1)], dim=-1)     # (..., K+1)
    cum_y = torch.cat([zero, torch.cumsum(h, dim=-1)], dim=-1)
    return cum_x, cum_y, d


def _rq_bin(t, cum, n_bins: int):
    """Index of the piece containing `t`, as `(t >= interior knots).sum()`.

    Deliberately NOT `torch.searchsorted`: this runs inside the coordinate likelihood on
    every node of every batch, and the comparison-sum is elementwise + a reduction, which
    is the device-safety constraint the rest of this module is written to (see the module
    docstring). With K ~ 8 the two cost the same."""
    idx = (t.unsqueeze(-1) >= cum[..., 1:-1]).sum(dim=-1)
    return idx.clamp(0, int(n_bins) - 1).unsqueeze(-1)


def _rq_gather(cum_x, cum_y, d, k):
    """The four knot quantities of the selected piece, plus its width/height/slope."""
    x_k = cum_x.gather(-1, k).squeeze(-1)
    x_k1 = cum_x.gather(-1, k + 1).squeeze(-1)
    y_k = cum_y.gather(-1, k).squeeze(-1)
    y_k1 = cum_y.gather(-1, k + 1).squeeze(-1)
    d_k = d.gather(-1, k).squeeze(-1)
    d_k1 = d.gather(-1, k + 1).squeeze(-1)
    w = (x_k1 - x_k).clamp(min=1e-9)
    h = y_k1 - y_k
    return x_k, y_k, d_k, d_k1, w, h, h / w


def rq_spline_forward(t, raw, n_bins: int):
    """`(S(t), log S'(t))` for `t` in [0, 1] — eqs. (4) and (5) of arXiv:1906.04032.

    Returns the log-derivative alongside the value because every caller needs both: the
    density is `p_base * S'` and the PIT is `S` itself, and computing them in one place
    is what stops the two from drifting apart (the same reason `trunc_normal_cdf` is
    built from the normalizer's `std_normal_cdf`)."""
    cum_x, cum_y, d = _rq_knots(raw, n_bins)
    t = t.clamp(0.0, 1.0)
    k = _rq_bin(t, cum_x, n_bins)
    x_k, y_k, d_k, d_k1, w, h, s = _rq_gather(cum_x, cum_y, d, k)
    xi = ((t - x_k) / w).clamp(0.0, 1.0)
    xi1 = 1.0 - xi
    curv = d_k1 + d_k - 2.0 * s
    denom = (s + curv * xi * xi1).clamp(min=1e-9)
    y = y_k + h * (s * xi * xi + d_k * xi * xi1) / denom
    deriv = s * s * (d_k1 * xi * xi + 2.0 * s * xi * xi1 + d_k * xi1 * xi1) / (denom * denom)
    return y.clamp(0.0, 1.0), torch.log(deriv.clamp(min=1e-12))


def rq_spline_inverse(y, raw, n_bins: int):
    """`S^{-1}(y)` for `y` in [0, 1] — eq. (6)-(8) of arXiv:1906.04032.

    A rational quadratic inverts in closed form (one quadratic per piece), which is what
    makes the sampler exact and its runtime data-independent — the same property
    `trunc_normal_sample` has and for the same reason."""
    cum_x, cum_y, d = _rq_knots(raw, n_bins)
    y = y.clamp(0.0, 1.0)
    k = _rq_bin(y, cum_y, n_bins)
    x_k, y_k, d_k, d_k1, w, h, s = _rq_gather(cum_x, cum_y, d, k)
    dy = (y - y_k).clamp(min=0.0)
    curv = d_k1 + d_k - 2.0 * s
    a = h * (s - d_k) + dy * curv
    b = h * d_k - dy * curv
    c = -s * dy
    disc = (b * b - 4.0 * a * c).clamp(min=0.0)
    # The `2c / (-b - sqrt(disc))` root is the numerically stable one and degrades
    # gracefully to the linear solution when a -> 0 (a linear piece, which is what the
    # identity spline is made of). The guard covers the one case that form cannot handle,
    # `-b - sqrt(disc) == 0`; xi is clamped to the piece regardless.
    root_den = -b - torch.sqrt(disc)
    root_den = torch.where(root_den.abs() < 1e-9, torch.full_like(root_den, -1e-9), root_den)
    xi = (2.0 * c / root_den).clamp(0.0, 1.0)
    return (x_k + xi * w).clamp(0.0, 1.0)


def _unit(x, lo, hi):
    """The interval mapped onto [0, 1], and its width. One place, so the density, the
    CDF and the sampler cannot disagree about where the interval is.

    `lo`/`hi` may be tensors (the cell-conditional `ln z` bounds) or plain floats (the
    constant `+-half_v` of the within-cell offsets), so the width is materialised as a
    tensor rather than assumed to be one."""
    width = torch.as_tensor(hi - lo, dtype=x.dtype, device=x.device).clamp(min=1e-9)
    return ((x - lo) / width).clamp(0.0, 1.0), width


def rq_interval_logpdf(x, lo, hi, raw, n_bins: int):
    """log density of the RQ spline on `(lo, hi]`, evaluated at `x`.

    `log S'(t) - log(hi - lo)`: the spline's derivative is the density on the unit
    interval and the affine map contributes its constant Jacobian. At raw = 0 the spline
    is the identity, so this is the UNIFORM density on the interval — the maximum-entropy
    starting point, and a stable one (no parameter can saturate a fixed affine map)."""
    t, width = _unit(x, lo, hi)
    _y, log_deriv = rq_spline_forward(t, raw, n_bins)
    return log_deriv - torch.log(width)


def rq_interval_cdf(x, lo, hi, raw, n_bins: int):
    """CDF of the same density: `S(t)`. This is the PIT the G3 gate reads."""
    t, _ = _unit(x, lo, hi)
    y, _ = rq_spline_forward(t, raw, n_bins)
    return y


def rq_interval_icdf(q, lo, hi, raw, n_bins: int):
    """Quantile function `lo + (hi - lo) S^{-1}(q)` — the sampler, and the decode-time
    median (a spline density has no closed-form mode)."""
    return lo + (hi - lo) * rq_spline_inverse(q, raw, n_bins)


def rq_interval_sample(lo, hi, raw, n_bins: int, *, generator=None):
    """One exact draw per element, by inverting the CDF at a uniform. Cannot leave
    `[lo, hi]`: `S^{-1}` lands in [0, 1] and the affine map carries that to the
    interval."""
    shape = torch.broadcast_shapes(torch.as_tensor(lo).shape, raw.shape[:-1])
    u = torch.rand(shape, device=raw.device, dtype=raw.dtype, generator=generator)
    return rq_interval_icdf(u, lo, hi, raw, n_bins)


def log_bessel_i0(x):
    """log I0(x) for x >= 0 via the Abramowitz & Stegun 9.8.1/9.8.2 polynomial
    approximations -- elementwise only, so it runs (and differentiates) on MPS,
    where torch.special.i0e support is not guaranteed. Both branches are kept
    finite everywhere so autograd through torch.where is clean."""
    t = x / 3.75
    t2 = t * t
    small = 1.0 + t2 * (
        3.5156229
        + t2 * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))))
    )
    log_small = torch.log(small)
    xl = x.clamp(min=3.75)  # keep large branch finite for x < 3.75
    u = 3.75 / xl
    large = 0.39894228 + u * (
        0.01328592
        + u
        * (
            0.00225319
            + u
            * (
                -0.00157565
                + u
                * (0.00916281 + u * (-0.02057706 + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377))))
            )
        )
    )
    log_large = xl - 0.5 * torch.log(xl) + torch.log(large)
    return torch.where(x <= 3.75, log_small, log_large)


def vonmises_logpdf(psi, mu, kappa):
    """log density of a von Mises (circular Gaussian) on psi in (-pi, pi]."""
    return kappa * torch.cos(psi - mu) - _LOG_2PI - log_bessel_i0(kappa)


def wrap_to_pi(theta):
    """Wrap an angle into the principal branch (-pi, pi] — the branch every psi
    density and every psi CDF here is defined on."""
    return theta - 2.0 * math.pi * torch.floor((theta + math.pi) / (2.0 * math.pi))


def bessel_i_ratios(kappa, n_terms: int, n_extra: int = 40):
    """`I_j(kappa)/I_0(kappa)` for j = 1..n_terms, stacked on a new leading axis.

    Uses the backward continued-fraction recurrence for r_j = I_j/I_{j-1},

        r_j = 1 / (2j/kappa + r_{j+1}),   r_{J+1} := 0,

    which follows from I_{j-1} = I_{j+1} + (2j/kappa) I_j. Every r_j lies in (0, 1),
    so — unlike a direct series or a forward recurrence — this cannot overflow at any
    kappa, and it is elementwise only (runs on MPS, like `log_bessel_i0`). The ratios
    are the cumulative products of the r_j."""
    k = torch.as_tensor(kappa).clamp(min=1e-6)
    J = int(n_terms + n_extra + math.ceil(float(k.detach().max())))
    r = torch.zeros_like(k)
    kept: list = [None] * (n_terms + 1)
    for j in range(J, 0, -1):
        r = 1.0 / (2.0 * j / k + r)
        if j <= n_terms:
            kept[j] = r
    out, acc = [], torch.ones_like(k)
    for j in range(1, n_terms + 1):
        acc = acc * kept[j]
        out.append(acc)
    return torch.stack(out, dim=0)  # (n_terms, *kappa.shape)


def vonmises_cdf(psi, mu, kappa, n_terms: int | None = None):
    """CDF of the von Mises `vonmises_logpdf` describes, measured from the branch cut
    at mu - pi (i.e. of the *wrapped deviation* t = wrap(psi - mu)):

        F(t) = 1/2 + t/(2 pi) + (1/pi) sum_{j>=1} [I_j(kappa)/(j I_0(kappa))] sin(j t).

    This is the standard Fourier series of the von Mises distribution function
    (Mardia & Jupp, *Directional Statistics*, §3.5.4). It is the natural PIT for a
    circular coordinate: if psi ~ vM(mu, kappa) then F is Uniform(0, 1) — the branch
    cut is placed opposite the mode, so it carries (almost) no mass and the transform
    is smooth where the density is.

    `n_terms` defaults to a truncation that is exact to ~1e-12 for the kappa given
    (the terms decay like exp(-j^2 / 2 kappa) for j << kappa)."""
    k = torch.as_tensor(kappa).clamp(min=1e-6)
    if n_terms is None:
        n_terms = max(24, int(4.0 * math.sqrt(float(k.detach().max())) + 16))
    t = wrap_to_pi(psi - mu)
    # kappa may be a scalar against a batched psi: broadcast before the recurrence so
    # the (n_terms, *shape) ratio stack lines up with sin(j*t).
    k, t = torch.broadcast_tensors(k, t)
    ratios = bessel_i_ratios(k, n_terms)                       # (n_terms, *t.shape)
    j = torch.arange(1, n_terms + 1, device=t.device, dtype=t.dtype)
    j = j.view(-1, *([1] * t.dim()))
    series = (ratios * torch.sin(j * t) / j).sum(0)
    return (0.5 + t / (2.0 * math.pi) + series / math.pi).clamp(0.0, 1.0)


def vonmises_sample(mu, kappa, *, generator=None, max_iter: int = 64):
    """One draw from the von Mises `vonmises_logpdf` describes, per element of the
    broadcast `(mu, kappa)`, by Best & Fisher's (1979) wrapped-Cauchy rejection scheme
    — the same algorithm numpy uses, vectorised and elementwise so it runs on MPS.

    Inverting `vonmises_cdf` would share more code, but that CDF costs a Bessel
    recurrence per evaluation and a bisection needs dozens of them; rejection is a
    handful of elementwise ops per attempt.

    The setup constant is rearranged to be cancellation-free. The textbook form
    `rho = (tau - sqrt(2 tau)) / (2 kappa)` subtracts two nearly equal numbers as
    kappa -> 0 (tau -> 2), which in float32 collapses to rho = 0 and then r = inf.
    Rationalising twice gives the algebraically identical

        rho = 2 kappa tau / ((g + 1) (tau + sqrt(2 tau))),   g = sqrt(1 + 4 kappa^2),

    in which every term is positive and well conditioned; rho -> kappa/2 smoothly.

    Acceptance is >= ~0.65 for every kappa, so `max_iter=64` leaves a straggler
    probability below 1e-29 per element; any straggler keeps its initialised value,
    the mode `mu`, which is the harmless direction to be wrong in."""
    mu, k = torch.broadcast_tensors(torch.as_tensor(mu), torch.as_tensor(kappa))
    k = k.clamp(min=1e-6)  # kappa -> 0 is the uniform limit; keep r finite
    shape, dev, dt = mu.shape, mu.device, mu.dtype

    def _u():
        return torch.rand(shape, device=dev, dtype=dt, generator=generator)

    g = torch.sqrt(1.0 + 4.0 * k * k)
    tau = 1.0 + g
    rho = 2.0 * k * tau / ((g + 1.0) * (tau + torch.sqrt(2.0 * tau)))
    r = (1.0 + rho * rho) / (2.0 * rho)

    f = torch.ones_like(k)  # acos(1) = 0 -> psi = mu, the straggler fallback
    done = torch.zeros_like(k, dtype=torch.bool)
    for _ in range(max_iter):
        z = torch.cos(math.pi * _u())
        f_try = (1.0 + r * z) / (r + z)
        c = k * (r - f_try)
        u2 = _u().clamp(min=1e-12)  # keeps log(c/u2) off the 0/0 branch
        accept = (c * (2.0 - c) - u2 > 0) | (torch.log(c / u2) + 1.0 - c >= 0)
        f = torch.where(accept & ~done, f_try, f)
        done = done | accept
        if bool(done.all()):
            break
    # the sign of the deviation is a fair coin -- the wrapped Cauchy is symmetric
    theta = torch.sign(_u() - 0.5) * torch.acos(f.clamp(-1.0, 1.0))
    return wrap_to_pi(mu + theta)
