"""Continuous density helpers (was the `_gauss_logpdf` etc. block).

Device-safe: only elementwise ops, no `torch.special`, so every term runs and
differentiates on MPS. Each `*_logpdf` returns a proper log-density; the unit
tests check they integrate to 1.

Each density is paired with its **CDF** (`std_normal_cdf`, `trunc_normal_cdf`,
`vonmises_cdf`) — the probability-integral transforms the per-coordinate PIT
calibration diagnostic evaluates at the truth (`eval/calibration.py`, WP2 of
docs/PLAN_UPDATES.md). The CDFs share the logpdfs' device-safety constraint.
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
