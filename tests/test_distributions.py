import math

import torch

from h2p_rsd_junipr.distributions import (
    bessel_i_ratios,
    gauss_logpdf,
    log_bessel_i0,
    trunc_normal_cdf,
    trunc_normal_logpdf,
    trunc_normal_sample,
    vonmises_cdf,
    vonmises_logpdf,
    vonmises_sample,
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


# --- samplers ---------------------------------------------------------------
# Each sampler is checked against its OWN CDF: if X ~ p then F(X) ~ Uniform(0,1), so
# the KS distance of the transformed draws to the uniform is the whole correctness
# statement. This is stronger than comparing moments (it sees the shape) and it pins
# sampler and density together, which is the property that matters — a sampler that
# drifted from the likelihood would silently bias every posterior-predictive curve.

N_DRAW = 20_000
KS_TOL = 4.0 / math.sqrt(N_DRAW)   # ~2.8 sigma of the KS null at this sample size


def _ks_uniform(u):
    """sup|F_n(u) - u| for u on [0, 1]."""
    us = torch.sort(u.flatten()).values
    n = us.numel()
    i = torch.arange(1, n + 1, dtype=us.dtype) / n
    return float(torch.maximum((i - us).max(), (us - (i - 1.0 / n)).max()))


def test_trunc_normal_sample_pit_is_uniform():
    torch.manual_seed(0)
    lo, hi = -0.3, 0.3
    for mu, sig in ((0.0, 0.3), (0.25, 0.1), (-0.9, 0.2)):   # centred, off-centre, far tail
        m = torch.full((N_DRAW,), mu)
        s = torch.full((N_DRAW,), sig)
        x = trunc_normal_sample(m, s, lo, hi)
        assert (x >= lo).all() and (x <= hi).all(), "draw left the head's support"
        assert _ks_uniform(trunc_normal_cdf(x, m, s, lo, hi)) < KS_TOL


def test_vonmises_sample_pit_is_uniform():
    torch.manual_seed(0)
    for kappa in (0.01, 0.3, 1.0, 4.0, 20.0):
        mu = torch.full((N_DRAW,), 0.7)
        k = torch.full((N_DRAW,), kappa)
        psi = vonmises_sample(mu, k)
        assert (psi > -math.pi - 1e-5).all() and (psi <= math.pi + 1e-5).all()
        assert _ks_uniform(vonmises_cdf(psi, mu, k)) < KS_TOL, f"kappa={kappa}"


def test_vonmises_sample_resultant_length():
    """E[cos(psi - mu)] = I1(kappa)/I0(kappa) — the concentration actually lands."""
    torch.manual_seed(0)
    for kappa in (0.3, 2.0, 8.0):
        mu = torch.full((N_DRAW,), -1.1)
        k = torch.full((N_DRAW,), kappa)
        got = float(torch.cos(vonmises_sample(mu, k) - mu).mean())
        assert abs(got - float(bessel_i_ratios(torch.tensor(kappa), 1)[0])) < 0.02


def test_vonmises_sample_survives_tiny_kappa():
    """The textbook setup constant cancels catastrophically as kappa -> 0 and collapses
    to rho = 0, r = inf, all-NaN draws in float32. The rationalised form must not."""
    torch.manual_seed(0)
    psi = vonmises_sample(torch.zeros(512), torch.full((512,), 1e-9))
    assert torch.isfinite(psi).all()
    assert float(psi.std()) > 1.0    # kappa -> 0 is the uniform limit, not a spike
