"""Continuous density helpers (was the `_gauss_logpdf` etc. block).

Device-safe: only elementwise ops, no `torch.special`, so every term runs and
differentiates on MPS. Each `*_logpdf` returns a proper log-density; the unit
tests check they integrate to 1.
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


def trunc_normal_logpdf(x, mu, sigma, lo, hi):
    """log density of N(mu, sigma^2) TRUNCATED to [lo, hi] (scalars), at x in
    [lo, hi]. Subtracting the in-interval mass makes the within-cell offset a
    proper density, so cell-prob x offset-density integrates to a proper density
    over (ln 1/DeltaR, ln kt)."""
    base = gauss_logpdf(x, mu, sigma)
    Z = (std_normal_cdf((hi - mu) / sigma) - std_normal_cdf((lo - mu) / sigma)).clamp(min=1e-6)
    return base - torch.log(Z)


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
