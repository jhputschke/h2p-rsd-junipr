"""Conditional flow matching with an exact ODE likelihood (docs/PLAN_UPDATES.md WP1).

The claim this family exists to make is that `log_prob` is a NORMALIZED density on
the physical support — so the tests are the ones that can falsify that claim:

  1. `test_divergence_is_exact` — the 4-VJP trace equals the full autograd Jacobian
     trace (the divergence is the whole likelihood; a wrong trace is a silent bias).
  2. `test_density_integrates_to_one` — Monte-Carlo integrate exp(log p) over the
     physical box. This is the test the discretized grid head could never pass, and
     it is sharp: flipping the divergence sign reads ~0.86 instead of ~1.0 (it caught
     exactly that error during development).
  3. `test_forward_reverse_roundtrip` — the density of a forward-ODE sample matches
     the change-of-variables accumulated on the way out.
  4. contract tests — `exact_likelihood` is True, `training_objective` is the
     flow-matching regression (NOT -log_prob), and the family is a registry drop-in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate
from h2p_rsd_junipr.models.base import build_model

CELL = 42


def _model(extra=None, geom=None, scale=1.0, double=False):
    """A CFM whose vector field is scaled up by `scale`: at random init the field is
    nearly zero, i.e. nearly the identity flow, which would let a broken divergence
    term pass unnoticed. Scaling makes `div v` genuinely non-zero."""
    cfg = load_config(["model=cfm", "encoder=gru"] + (extra or []))
    geom = geom or Geometry.from_config(cfg.geometry)
    torch.manual_seed(0)
    m = build_model(cfg, geom)
    if double:
        m = m.double()
    for p in m.field.net.parameters():
        p.data.mul_(scale)
    return m.eval(), geom


def _ctx(m, geom, n, dtype=torch.float32):
    torch.manual_seed(1)
    cell = torch.tensor([CELL])
    e = torch.randn(1, m.ctx_dim, dtype=dtype)
    return torch.cat([e, m.cell_emb(cell)], -1).expand(n, -1), cell.expand(n)


# ---------------------------------------------------------------------------
# 1. the exact divergence
# ---------------------------------------------------------------------------
def test_divergence_is_exact():
    """4 vector-Jacobian products == trace of the full autograd Jacobian."""
    m, geom = _model(scale=2.0)
    ctx, _ = _ctx(m, geom, 6)
    x = torch.randn(6, 4, requires_grad=True)
    t = torch.rand(6, 1)
    v = m.field(x, t, ctx)
    got = m._divergence(v, x, create_graph=False)
    for i in range(6):
        J = torch.autograd.functional.jacobian(
            lambda z, i=i: m.field(z, t[i : i + 1], ctx[i : i + 1]).squeeze(0),
            x[i : i + 1].detach(),
        ).reshape(4, 4)
        assert float(got[i]) == pytest.approx(float(torch.diagonal(J).sum()), abs=1e-5)


# ---------------------------------------------------------------------------
# 2. normalization on the PHYSICAL support
# ---------------------------------------------------------------------------
def _mc_integral(m, geom, n_mc=40000, sp=1.8, sz=2.5, seed=1, sign=+1.0):
    """Importance-sampling estimate of the 4-d coordinate integral over the physical
    support, with a proposal defined in PHYSICAL coordinates (a tanh-squashed normal
    on the bounded dims, a wide normal on ln z). Building the proposal independently
    of the model's own bijection is what makes the Jacobian bookkeeping testable.

    `sign=-1` flips the divergence contribution — the control that shows the test
    actually discriminates. Runs in float64: atanh saturates in float32 out in the
    tails where the proposal puts mass."""
    hu, hv = geom.half_u, geom.half_v
    cx, cy = geom.cell_center(CELL)
    ctx, yc = _ctx(m, geom, n_mc, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n_mc, 4, generator=g, dtype=torch.float64)
    du, dv = hu * torch.tanh(sp * s[:, 0]), hv * torch.tanh(sp * s[:, 1])
    lnz, ps = sz * s[:, 2], math.pi * torch.tanh(sp * s[:, 3])

    def log_q_tanh(x, h):
        t = (x / h).clamp(-1 + 1e-14, 1 - 1e-14)
        z = torch.atanh(t) / sp
        return (-0.5 * z**2 - math.log(sp) - 0.5 * math.log(2 * math.pi)
                - math.log(h) - torch.log1p(-t * t))

    log_q = (log_q_tanh(du, hu) + log_q_tanh(dv, hv) + log_q_tanh(ps, math.pi)
             + (-0.5 * (lnz / sz) ** 2 - math.log(sz) - 0.5 * math.log(2 * math.pi)))
    yraw = torch.stack([cx + du, cy + dv, lnz, ps], -1)
    s1, ldj = m._to_std(yraw, yc)
    s0, acc = m._ode(s1, ctx, reverse=True, with_divergence=True)
    log_p = (-0.5 * s0**2 - 0.5 * math.log(2 * math.pi)).sum(-1) - sign * acc + ldj
    w = (log_p - log_q).exp()
    return float(w.mean()), float(w.std() / math.sqrt(n_mc))


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_density_integrates_to_one(scale):
    """exp(log p_cfm) integrates to 1 over the physical (box x R x circle) support."""
    m, geom = _model(scale=scale, double=True, extra=["model.n_ode_steps=48"])
    est, se = _mc_integral(m, geom)
    assert abs(est - 1.0) < max(4.0 * se, 0.05), f"integral = {est:.4f} +- {se:.4f}"


def test_normalization_test_would_catch_a_sign_error():
    """The control: with the divergence contribution negated the integral misses 1 by
    ~16 standard errors, so the test above is a real constraint and not a tautology.

    Measured at scale=2 (n=40k): correct 1.004 +- 0.009, flipped 0.856 +- 0.008. The
    scale matters — at scale<=1 the field is nearly the identity flow and BOTH signs
    read 1.0; well above it the importance-sampling proposal stops covering the
    density and neither is meaningful."""
    m, geom = _model(scale=2.0, double=True, extra=["model.n_ode_steps=48"])
    good, good_se = _mc_integral(m, geom, sign=+1.0)
    bad, _ = _mc_integral(m, geom, sign=-1.0)
    assert abs(good - 1.0) < 0.05
    assert abs(bad - 1.0) > 0.10 and abs(bad - 1.0) > 8.0 * good_se


# ---------------------------------------------------------------------------
# 3. forward/reverse round trip
# ---------------------------------------------------------------------------
def test_forward_reverse_roundtrip():
    """Push base points OUT through the ODE accumulating the change of variables, then
    ask `coord_log_prob` for the density of the result: the two must agree, and the
    trajectory must come back to where it started."""
    m, geom = _model(scale=2.0, double=True, extra=["model.n_ode_steps=128"])
    ctx, yc = _ctx(m, geom, 32, dtype=torch.float64)
    torch.manual_seed(4)
    s0 = torch.randn(32, 4, dtype=torch.float64)
    s1, acc = m._ode(s0, ctx, reverse=False, with_divergence=True)
    base = (-0.5 * s0**2 - 0.5 * math.log(2 * math.pi)).sum(-1)
    coords = m._to_phys(s1, yc)
    _, ldj = m._to_std(coords, yc)
    expected = base + acc + ldj                      # accumulated on the way OUT
    got = m.coord_log_prob(coords, yc, ctx)          # recomputed on the way BACK
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()
    s0_back, _ = m._ode(s1, ctx, reverse=True, with_divergence=False)
    assert torch.allclose(s0_back, s0, atol=1e-5)


def test_heun_and_rk4_agree_at_fine_steps():
    """The two solvers integrate the same ODE: at 256 steps they must agree, so
    `ode_solver` is a cost knob and not a semantics knob."""
    ctx = None
    outs = []
    for solver in ("rk4", "heun"):
        m, geom = _model(scale=2.0, double=True,
                         extra=["model.n_ode_steps=256", f"model.ode_solver={solver}"])
        ctx, yc = _ctx(m, geom, 16, dtype=torch.float64)
        torch.manual_seed(5)
        yraw = torch.stack([
            torch.full((16,), geom.cell_center(CELL)[0], dtype=torch.float64) + 0.1 * torch.randn(16, dtype=torch.float64),
            torch.full((16,), geom.cell_center(CELL)[1], dtype=torch.float64) + 0.1 * torch.randn(16, dtype=torch.float64),
            torch.randn(16, dtype=torch.float64),
            torch.rand(16, dtype=torch.float64) * 2 - 1,
        ], dim=-1)
        outs.append(m.coord_log_prob(yraw, yc, ctx))
    assert torch.allclose(outs[0], outs[1], atol=1e-4), (outs[0] - outs[1]).abs().max()


# ---------------------------------------------------------------------------
# 4. contract
# ---------------------------------------------------------------------------
def test_registry_and_contract_flags(batch):
    b, geom = batch
    m = build_model(load_config(["model=cfm", "encoder=gru"]), geom).eval()
    assert m.exact_likelihood is True          # the whole point vs `diffusion`
    assert m.supports_coordinate_pit is True
    lp = m.log_prob(b)
    assert lp.shape == (b["xf"].shape[0],) and torch.isfinite(lp).all()
    assert torch.allclose(m.per_jet_nll(b), -lp)


def test_training_objective_is_the_regression_not_the_nll(batch):
    """`log_prob` stays exact BECAUSE training runs a different objective — if these
    two ever coincide, someone has quietly re-pointed one at the other."""
    b, geom = batch
    m = build_model(load_config(["model=cfm", "encoder=gru"]), geom).eval()
    torch.manual_seed(0)
    obj = m.training_objective(b)
    assert obj.shape == (b["xf"].shape[0],) and torch.isfinite(obj).all()
    assert not torch.allclose(obj, -m.log_prob(b), atol=1e-3)
    # ...and it is differentiable, unlike the (detached) ODE likelihood
    m.zero_grad()
    m.training_objective(b).sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.field.parameters())


def test_other_families_keep_maximum_likelihood_training(batch):
    """The `training_objective` hook must not change any existing family."""
    b, geom = batch
    for sel in (["model=ar_junipr_v2"], ["model=ar_junipr_v3"], ["model=cinn"],
                ["model=diffusion"]):
        m = build_model(load_config(sel), geom).eval()
        torch.manual_seed(0)
        a = m.training_objective(b)
        torch.manual_seed(0)
        assert torch.allclose(a, -m.log_prob(b), atol=1e-6), sel


def test_sample_and_map_are_valid(batch):
    b, geom = batch
    m = build_model(load_config(["model=cfm", "encoder=gru"]), geom).eval()
    xf, nx = b["xf"][:1], b["nx"][:1]
    draws = m.sample(xf, nx, 5)
    assert len(draws) == 5 and all(0 <= c < geom.n_cells for d in draws for c in d)
    mp = m.map_estimate(xf, nx)
    assert isinstance(mp, LundPointEstimate)
    assert mp.multiplicity == len(mp.nodes) >= 1        # default min_emissions=1
    assert np.isfinite(mp.logprob)
    assert m.map_estimate(xf, nx, min_emissions=3).multiplicity >= 3
    for n in mp.nodes:                                  # coordinates on the support
        u, v = geom.cell_center(n.cell)
        assert abs(n.ln_invDelta - u) <= geom.half_u + 1e-6
        assert abs(n.ln_kt - v) <= geom.half_v + 1e-6
        assert -math.pi <= n.psi <= math.pi


def test_map_ascent_mode_reaches_a_higher_density(batch):
    """`cfm_map="ascent"` must actually climb the exact density — it is the mode that
    needs gradients through the ODE, so this also pins `differentiable=True`."""
    b, geom = batch
    xf, nx = b["xf"][:1], b["nx"][:1]
    base = build_model(load_config(["model=cfm", "model.n_ode_steps=8"]), geom).eval()
    asc = build_model(
        load_config(["model=cfm", "model.n_ode_steps=8", "model.cfm_map=ascent"]), geom
    ).eval()
    asc.load_state_dict(base.state_dict())
    a = base.map_estimate(xf, nx, min_emissions=3, ascent_steps=15)
    c = asc.map_estimate(xf, nx, min_emissions=3, ascent_steps=15)
    assert a.multiplicity == c.multiplicity            # same cells, only coords move
    assert sum(n.logp_coord for n in c.nodes) > sum(n.logp_coord for n in a.nodes)


def test_length_pmf_is_exact_and_normalized(batch):
    b, geom = batch
    m = build_model(load_config(["model=cfm"]), geom).eval()
    pmf = m.length_pmf(b["xf"][:1], b["nx"][:1])
    assert pmf.shape == (m.max_emissions + 1,)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)


def test_coordinate_pit_is_reported_in_latent_space(batch):
    b, geom = batch
    m = build_model(load_config(["model=cfm", "model.n_ode_steps=8"]), geom).eval()
    out = m.coordinate_cdfs(b)
    assert out["space"] == "latent" and len(out["names"]) == 4
    u = out["u"][out["mask"]]
    assert bool(((u >= 0.0) & (u <= 1.0)).all()) and torch.isfinite(u).all()


def test_checkpoint_roundtrip(tmp_path, batch):
    from h2p_rsd_junipr.train.checkpoint import load_for_inference, save_checkpoint
    from h2p_rsd_junipr.train.trainer import build_components

    b, geom = batch
    cfg = load_config(["model=cfm", "encoder=gru", "model.n_ode_steps=8"])
    model, opt, sched = build_components(cfg, geom, torch.device("cpu"))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "cfm.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=1, step=1, best_val=0.0, cfg=cfg)
    info = load_for_inference(path)
    assert info["model_name"] == "cfm"
    m2 = build_model(cfg, geom)
    m2.load_state_dict(info["model_state"])            # strict
    model.eval(), m2.eval()
    assert torch.allclose(model.log_prob(b), m2.log_prob(b), atol=1e-5)


def test_invalid_solver_and_map_mode_are_rejected(batch):
    _, geom = batch
    for bad in (["model.ode_solver=euler"], ["model.cfm_map=argmax"]):
        with pytest.raises(ValueError):
            build_model(load_config(["model=cfm"] + bad), geom)
