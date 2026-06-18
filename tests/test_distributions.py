import math

import torch

from h2p_rsd_junipr.distributions import (
    gauss_logpdf,
    log_bessel_i0,
    trunc_normal_logpdf,
    vonmises_logpdf,
)


def _integrate(logpdf_vals, x):
    return torch.trapz(logpdf_vals.exp(), x).item()


def test_gauss_integrates_to_one():
    x = torch.linspace(-12, 12, 20001)
    mu = torch.tensor(0.7)
    sig = torch.tensor(1.3)
    assert abs(_integrate(gauss_logpdf(x, mu, sig), x) - 1.0) < 1e-3


def test_trunc_normal_integrates_to_one_over_interval():
    lo, hi = -0.3, 0.3
    x = torch.linspace(lo, hi, 20001)
    val = trunc_normal_logpdf(x, torch.tensor(0.05), torch.tensor(0.2), lo, hi)
    assert abs(_integrate(val, x) - 1.0) < 1e-3


def test_vonmises_integrates_to_one():
    psi = torch.linspace(-math.pi, math.pi, 20001)
    for kappa in (0.5, 2.0, 10.0):
        val = vonmises_logpdf(psi, torch.tensor(0.4), torch.tensor(kappa))
        assert abs(_integrate(val, psi) - 1.0) < 1e-3


def test_log_bessel_i0_matches_reference():
    x = torch.tensor([0.0, 1.0, 3.7, 3.8, 10.0, 40.0])
    ref = torch.special.i0(x).log()
    got = log_bessel_i0(x)
    assert torch.allclose(got, ref, atol=2e-3)
