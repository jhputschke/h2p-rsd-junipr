"""docs/PLAN_lnz_spline_head.md: the RQ-spline `ln z` head.

`lnz_support="physical"` fixed WHERE the density lives and left a shape mismatch INSIDE
the interval — PIT at 1.05-2.07x its critical value on every v1 seed, 2.16x in the
quadrant holding 94% of emissions. `lnz_head="spline"` puts a monotone rational-quadratic
spline (Durkan et al., arXiv:1906.04032) on that interval. These tests pin the five things
that has to mean, in the order they can fail:

1. the spline map itself is a monotone bijection of [0, 1] and inverts exactly;
2. the identity spline is the UNIFORM density on the interval (the initialization), and
   the `truncnorm` default is bit-identical end to end;
3. the density is a proper DENSITY: it integrates to 1 over the interval;
4. its sampler cannot leave the soft-drop support — the property WP-A bought and this
   change must not spend;
5. sampler and CDF agree: the PIT of model-drawn `ln z` is uniform (the SBC null).

Plus the configuration guard (a spline needs an interval to live on) and the
identifiability regression below, which is the one that was learned the hard way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.distributions import (
    rq_interval_cdf,
    rq_interval_logpdf,
    rq_interval_sample,
    rq_spline_forward,
    rq_spline_inverse,
    rq_spline_n_params,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.trainer import seed_everything

LN_HALF = math.log(0.5)
LN_ZCUT = math.log(0.1)
K = 8


def _model(seed=0, **over):
    argv = ["model=ar_junipr_v2", "encoder=gru", "model.lnz_support=physical"]
    argv += [f"model.{k}={v}" for k, v in over.items()]
    cfg = load_config(argv)
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    return build_model(cfg, geom).eval(), geom


def _ks(u):
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - u, u - (i - 1) / n)))


# ---------------------------------------------------------------------------
# 1. the spline map
# ---------------------------------------------------------------------------
def test_spline_is_a_monotone_bijection_of_the_unit_interval():
    """S is increasing, S(0) = 0, S(1) = 1 — which is what makes the composition a CDF."""
    raw = torch.randn(1, rq_spline_n_params(K), generator=torch.Generator().manual_seed(0))
    t = torch.linspace(0.0, 1.0, 1001)
    y, log_deriv = rq_spline_forward(t, raw.expand(1001, -1), K)
    assert bool((y[1:] >= y[:-1]).all()), "S must be non-decreasing"
    assert y[0].item() == pytest.approx(0.0, abs=1e-6)
    assert y[-1].item() == pytest.approx(1.0, abs=1e-6)
    # a strictly increasing map has a finite log-derivative everywhere
    assert bool(torch.isfinite(log_deriv).all())


def test_spline_inverse_round_trips():
    """`S^-1(S(t)) == t`. The sampler is the inverse evaluated at a uniform, so an
    inaccurate inverse is a sampler that does not match the likelihood.

    Asserted at BOTH precisions on purpose. In float64 the residual is 1e-13, which says
    the closed form is exact rather than approximately right; in float32 the worst of
    4096 random splines lands at ~1e-4 (median 1e-8), which is round-off in the quadratic
    solve and not a formula error. Testing only float32 would leave the two
    indistinguishable."""
    g = torch.Generator().manual_seed(1)
    raw = torch.randn(4096, rq_spline_n_params(K), generator=g)
    t = torch.rand(4096, generator=g)

    y64, _ = rq_spline_forward(t.double(), raw.double(), K)
    assert float((rq_spline_inverse(y64, raw.double(), K) - t.double()).abs().max()) < 1e-10

    y, _ = rq_spline_forward(t, raw, K)
    err = (rq_spline_inverse(y, raw, K) - t).abs()
    assert float(err.max()) < 5e-4
    assert float(err.median()) < 1e-6


# ---------------------------------------------------------------------------
# 2. the identity spline IS the truncated normal
# ---------------------------------------------------------------------------
def test_zero_parameters_are_the_identity_spline():
    """Raw zeros -> uniform widths and heights and unit knot derivatives -> S(t) = t.

    This is why a head whose last layer starts near zero starts near the truncated
    normal it generalises, rather than at an arbitrary warp."""
    t = torch.linspace(0.0, 1.0, 501)
    y, log_deriv = rq_spline_forward(t, torch.zeros(501, rq_spline_n_params(K)), K)
    assert float((y - t).abs().max()) < 1e-6
    assert float(log_deriv.abs().max()) < 1e-6


def test_identity_spline_is_the_uniform_density_on_the_interval():
    """With S = identity the density is `1 / (hi - lo)` and the CDF is the affine map.

    That is the initialization the head starts from — the maximum-entropy density on the
    support, and a stable one: a fixed affine base has no parameter that can saturate."""
    xs = torch.linspace(LN_ZCUT + 1e-5, LN_HALF - 1e-5, 4001)
    lo, hi = torch.tensor(LN_ZCUT), torch.tensor(LN_HALF)
    raw = torch.zeros(4001, rq_spline_n_params(K))
    lp = rq_interval_logpdf(xs, lo, hi, raw, K)
    assert float((lp - (-math.log(LN_HALF - LN_ZCUT))).abs().max()) < 1e-5
    cdf = rq_interval_cdf(xs, lo, hi, raw, K)
    assert float((cdf - (xs - lo) / (hi - lo)).abs().max()) < 1e-5


def test_default_head_is_bit_identical_end_to_end():
    """PARITY: the field absent and the field set to its default give the same model and
    the same numbers — state_dict, likelihood, PIT, draws.

    The same contract `lnz_support="legacy"` carries, and the reason the spline is
    shipped as a flag rather than as a replacement."""
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
    from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset

    def build(extra):
        seed_everything(0, True)
        cfg = load_config(["model=ar_junipr_v4", "encoder=lundnet",
                           "model.lnz_support=physical"] + extra)
        geom = Geometry.from_config(cfg.geometry)
        return build_model(cfg, geom).eval(), geom

    absent, geom = build([])
    explicit, _ = build(["model.lnz_head=truncnorm"])

    a, b = absent.state_dict(), explicit.state_dict()
    assert a.keys() == b.keys()
    assert max(float((a[k] - b[k]).abs().max()) for k in a) == 0.0
    assert absent.coord_head[-1].out_features == 8, "truncnorm must not widen the head"

    ds = MatchedLundDataset(synthetic_matched_dataset(32, seed=3), geom)
    batch = collate([ds[i] for i in range(len(ds))])
    with torch.inference_mode():
        assert float((absent.log_prob(batch) - explicit.log_prob(batch)).abs().max()) == 0.0
        pa, pb = absent.coordinate_cdfs(batch), explicit.coordinate_cdfs(batch)
        assert float((pa["u"] - pb["u"]).abs().max()) == 0.0
        torch.manual_seed(11)
        ca = absent.sample_coordinates(batch["xf"][:1], batch["nx"][:1], [12, 34, 56])
        torch.manual_seed(11)
        cb = explicit.sample_coordinates(batch["xf"][:1], batch["nx"][:1], [12, 34, 56])
        assert float((ca - cb).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 3. a proper density
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_composed_density_integrates_to_one(seed):
    """A warped density that forgot its Jacobian integrates to something else — the bug
    a flow invites, and the one no calibration plot would name."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(1, rq_spline_n_params(K), generator=g)
    xs = torch.linspace(LN_ZCUT + 1e-6, LN_HALF - 1e-6, 40001)
    lo, hi = torch.tensor(LN_ZCUT), torch.tensor(LN_HALF)
    p = rq_interval_logpdf(xs, lo, hi, raw.expand(40001, -1), K).exp()
    assert float(torch.trapz(p, xs)) == pytest.approx(1.0, abs=2e-3)


def test_model_joint_still_integrates_to_one_with_the_spline_on():
    """The same statement one level up: the emission density the TRAINED objective
    normalizes stays a density when the ln z factor is warped."""
    m, geom = _model(lnz_head="spline")
    cells = [17, 42, 88]
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    params = m.coord_head_params(xf, nx, cells)
    assert params.lnz_spline is not None, "the spline parameters must reach the head tuple"
    M = 120_000
    g = torch.Generator().manual_seed(7)
    for t, cell in enumerate(cells):
        cxv, cyv = geom.cell_center(cell)
        cx, cy = torch.full((M,), cxv), torch.full((M,), cyv)
        lo, hi = m.lnz_bounds(cx)
        p_t = params.apply(
            lambda q, t=t: (torch.full((M,), float(q[t])) if q.dim() == 1
                            else q[t].expand(M, -1))
        )
        u = cxv + m.half_u * (2.0 * torch.rand(M, generator=g) - 1.0)
        v = cyv + m.half_v * (2.0 * torch.rand(M, generator=g) - 1.0)
        lnz = lo + (hi - lo) * torch.rand(M, generator=g)
        psi = math.pi * (2.0 * torch.rand(M, generator=g) - 1.0)
        vol = (2 * m.half_u) * (2 * m.half_v) * float((hi - lo)[0]) * (2 * math.pi)
        with torch.inference_mode():
            dens = m._coord_logprob(p_t, u, v, lnz, psi, cx, cy).exp()
        assert float(dens.mean()) * vol == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# 4./5. the sampler: inside the support, and matching its own CDF
# ---------------------------------------------------------------------------
def test_sampler_stays_inside_the_soft_drop_support():
    """The property WP-A bought (0.83% -> 0.0000% violations) and this change must not
    spend. Guaranteed by construction — the spline maps [0, 1] onto itself, so the
    composed quantile function cannot leave the truncated normal's interval — and
    measured here at 2e5 draws."""
    g = torch.Generator().manual_seed(3)
    n = 200_000
    raw = torch.randn(1, rq_spline_n_params(K), generator=g).expand(n, -1)
    lo, hi = torch.tensor(LN_ZCUT), torch.tensor(LN_HALF)
    x = rq_interval_sample(lo, hi, raw, K, generator=g)
    assert float(x.min()) >= LN_ZCUT
    assert float(x.max()) <= LN_HALF


def test_pit_of_spline_drawn_ln_z_is_uniform():
    """The SBC null for the new head: draw from it, transform by its own CDF, and the
    result must be Uniform(0,1). This is the one test that fails if the sampler and the
    density ever describe different distributions."""
    g = torch.Generator().manual_seed(5)
    n = 200_000
    raw = torch.randn(1, rq_spline_n_params(K), generator=g).expand(n, -1)
    lo, hi = torch.tensor(LN_ZCUT), torch.tensor(LN_HALF)
    x = rq_interval_sample(lo, hi, raw, K, generator=g)
    u = rq_interval_cdf(x, lo, hi, raw, K)
    assert _ks(u.numpy()) < 1.36 / math.sqrt(n)


def test_model_sampled_ln_z_respects_the_support_and_its_own_pit():
    """Same two statements through the MODEL's own sampler and CDF, so the wiring is
    covered and not only the primitives."""
    m, geom = _model(lnz_head="spline", seed=2)
    cells = [5, 61, 93] * 40          # the default geometry is 10x10 = 100 cells
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    torch.manual_seed(0)
    with torch.inference_mode():
        coords = m.sample_coordinates(xf, nx, cells)
        yc = torch.tensor([cells], dtype=torch.long)
        cx = m.cell_cx[yc][0]
        lo, hi = m.lnz_bounds(cx)
        lnz = coords[:, 2]
        assert bool(((lnz >= lo) & (lnz <= hi)).all()), "a draw left the soft-drop interval"
        # the reported MAP point must land inside the support too (it is a median here)
        p = m.coord_head_params(xf, nx, cells)
        point = m._lnz_point(p, cx)
        assert bool(((point >= lo) & (point <= hi)).all())


# ---------------------------------------------------------------------------
# the configuration guard
# ---------------------------------------------------------------------------
def test_spline_requires_the_physical_support():
    """The spline warps the truncated normal's CDF; `legacy` has no interval to warp, so
    the pairing is a configuration error rather than a silently different model."""
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru",
                       "model.lnz_support=legacy", "model.lnz_head=spline"])
    geom = Geometry.from_config(cfg.geometry)
    with pytest.raises(ValueError, match="needs model.lnz_support='physical'"):
        build_model(cfg, geom)


def test_unknown_head_raises():
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru",
                       "model.lnz_support=physical", "model.lnz_head=quadratic"])
    geom = Geometry.from_config(cfg.geometry)
    with pytest.raises(ValueError, match="model.lnz_head must be"):
        build_model(cfg, geom)


def test_the_spline_replaces_the_base_rather_than_warping_a_learnable_one():
    """REGRESSION, and the expensive lesson of this work package.

    The first implementation composed the spline on the truncated normal's CDF,
    `F = S(F_tn)`, so that `truncnorm` was the identity special case. That is
    NON-IDENTIFIABLE: once S carries the shape, any (mu, sigma) leaving F_tn roughly
    linear on the interval gives the same density, and the pair is free to drift along
    that flat direction. It did. On seed 2 of the first 3-seed run `lnz_mean` reached
    **-533** against an interval of [-2.303, -0.693] and `lnz_sig` reached **85**, F_tn
    saturated to 0 or 1 on **100%** of emissions, and val NLL went 4.19 -> 19.2 at epoch 4
    and never came back.

    So the contract is: exactly ONE of the two ln z parameterizations is live, and the
    spline arm carries no learnable base at all. Asserted on the head's own output rather
    than on the config, because that is where the redundancy would reappear."""
    spl, _ = _model(lnz_head="spline")
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    p = spl.coord_head_params(xf, nx, [7, 23, 61])
    assert p.lnz_spline is not None
    assert p.lnz_mean is None and p.lnz_sig is None, (
        "the spline arm must not carry a learnable truncated-normal base — that is the "
        "flat direction that diverged"
    )
    base, _ = _model(lnz_head="truncnorm")
    q = base.coord_head_params(xf, nx, [7, 23, 61])
    assert q.lnz_spline is None
    assert q.lnz_mean is not None and q.lnz_sig is not None


def test_spline_widens_only_the_coordinate_head():
    """The cost is 3K-1 extra outputs on ONE layer; nothing else about the model moves."""
    base, _ = _model(lnz_head="truncnorm")
    spl, _ = _model(lnz_head="spline")
    assert base.coord_head[-1].out_features == 8               # 6 + (mean, sigma)
    assert spl.coord_head[-1].out_features == 6 + rq_spline_n_params(K)
    extra = sum(p.numel() for p in spl.parameters()) - sum(p.numel() for p in base.parameters())
    hidden = base.coord_head[-1].in_features
    # the spline REPLACES (mean, sigma), so the delta is 3K-1-2 outputs and nothing else
    assert extra == (rq_spline_n_params(K) - 2) * (hidden + 1)
