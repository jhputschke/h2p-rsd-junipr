"""WP-E of docs/PLAN_prod_test_edit.md: the bounded-support `ln z` head on the EDIT family.

`tests/test_lnz_support.py` pins the same intervention on the AR families. This file pins
it here, where the density is a two-component mixture and the coordinates come out of a
constrained lattice rather than teacher forcing — so what can break is not the same:

1. `legacy` is bit-identical to the unbounded Normal (the switch is a no-op off), and it
   is also the `e_v1_legacy_lnz` attribution arm, so it must still leak;
2. `physical` is a proper density **in each mixture component separately** — a truncated
   component that forgot its normalizer integrates to `1/Z`, and mixing it with a proper
   one produces something finite, smooth and wrong;
3. neither draw path can leave the support at 1e5 draws, on BOTH walls (gate E2's target):
   the constrained `sample_coordinates` *and* the ancestral `_draw_emission`;
4. the general-`beta` bound, which the fielded `beta = 0` files cannot exercise;
5. **`_log_cell_mass` is provably unchanged by the port** — the plan §2 fact that let the
   constrained forward-backward go untouched. Asserted here, not hoped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.distributions import (
    gauss_logpdf,
    trunc_normal_logpdf,
    vonmises_logpdf,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models import edit_dp
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.models.edit import _EmitParams

LN_HALF = math.log(0.5)
LN_ZCUT = math.log(0.1)

FAMILIES = [["model=edit_v1", "encoder=gru"],
            ["model=edit_v2", "encoder=gru"],
            ["model=edit_v1", "encoder=gru", "model.physics_width=false"]]
IDS = ["v1", "v2", "v1-free-width"]


def _model(sel, support="physical", *, seed=0, **over):
    argv = list(sel) + [f"model.lnz_support={support}"]
    argv += [f"model.{k}={v}" for k, v in over.items()]
    cfg = load_config(argv)
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    return build_model(cfg, geom).eval(), geom


def _batch(model, geom, B=6, L=3, seed=1):
    """A frozen batch whose truth `ln z` lies inside the physical support."""
    g = torch.Generator().manual_seed(seed)
    xf = torch.randn(B, 5, 5, generator=g)
    nx = torch.full((B,), 5, dtype=torch.long)
    yc = torch.randint(0, geom.n_cells, (B, L), generator=g)
    ny = torch.full((B,), L, dtype=torch.long)
    cx, cy = model.cell_cx[yc], model.cell_cy[yc]
    lnz = LN_ZCUT + (LN_HALF - LN_ZCUT) * torch.rand(B, L, generator=g)
    psi = math.pi * (2.0 * torch.rand(B, L, generator=g) - 1.0)
    yraw = torch.stack([cx, cy, lnz, psi], dim=-1)
    return {"xf": xf, "nx": nx, "yc": yc, "ny": ny, "yraw": yraw}


def _emit_params(model, b, Ny):
    """The emission-head block at every lattice state of `b`, as `_lattice` builds it."""
    S, e, anchor, ok = model._encode(b["xf"], b["nx"])
    C = model._prefix_states(b["yc"], e)[:, :Ny] if model.prefix_conditioning else None
    return model._emit_params(model._emit_input(S, e, C),
                              anchor[:, :, None, :], ok[:, :, None])


def _chains(n_chains, length, n_cells, seed=13):
    """`n_chains` cell chains of `length` cells each, spread over the whole grid.

    Long chains rather than many short ones on purpose: `sample_coordinates_many`
    defaults to a python loop over draws for this family, so 1e5 emissions as 20 000
    chains of 5 is 20 000 lattice builds while 200 chains of 500 is 200. The DP is
    `O(n_x . n_y)` either way and the coordinate draws are what is being counted."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, n_cells, (n_chains, length), generator=g).tolist()


def _one_state(p: _EmitParams, n: int) -> _EmitParams:
    """Lattice state `(i, t) = (0, 0)` of a parameter block, broadcast to `n` draws.

    Every field is a scalar there except `cell_lp`, whose trailing axis is the cell
    categorical and is left alone."""
    return _EmitParams(*[f[0, 0, 0].expand(n) if f.ndim == 3 else f[0, 0, 0] for f in p])


# ---------------------------------------------------------------------------
# 1. parity — the switch is a no-op when off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_legacy_is_bit_identical_to_the_unbounded_normal(sel):
    """`legacy` must reproduce the pre-WP-E arithmetic EXACTLY, not merely closely: the
    point of the default is that the shipped edit checkpoints keep their likelihood.

    Checked where it can actually drift — the `ln z` factor of each component, and then
    the whole component rebuilt by hand in the SAME summation order the model uses.
    `torch.equal`, not `allclose`: a truncation normalizer left behind is a small
    constant, and small constants are exactly what a tolerance hides. (Rebuilding by
    subtracting and re-adding a term would not be bit-exact in float32 for reasons that
    have nothing to do with the port, so the sum is written out instead.)"""
    m, geom = _model(sel, "legacy")
    b = _batch(m, geom)
    assert m.lnz_bounds(b["yraw"][..., 0]) is None
    with torch.inference_mode():
        p = _emit_params(m, b, b["yc"].shape[1])
        u, v, lz, psi = (t[:, None, :] for t in m._targets(b["yraw"]))
        cell = b["yc"].clamp(min=0)[:, None, :]

        # the ln z factor itself: in `legacy` it IS `gauss_logpdf`, bit for bit
        assert torch.equal(m._log_lnz(lz, p.mu_z, p.sig_z, u),
                           gauss_logpdf(lz, p.mu_z, p.sig_z))
        assert torch.equal(m._log_lnz(lz, p.f_lz_m, p.f_lz_s, u),
                           gauss_logpdf(lz, p.f_lz_m, p.f_lz_s))

        anch_hand = trunc_normal_logpdf(u, p.mu_u, p.sig_u, m.lo_u, m.hi_u)
        anch_hand = anch_hand + trunc_normal_logpdf(v, p.mu_v, p.sig_v, m.lo_v, m.hi_v)
        anch_hand = anch_hand + gauss_logpdf(lz, p.mu_z, p.sig_z)
        anch_hand = anch_hand + vonmises_logpdf(psi, p.mu_psi, p.kappa)

        du = (u - m.cell_cx[cell]).clamp(-m.half_u, m.half_u)
        dv = (v - m.cell_cy[cell]).clamp(-m.half_v, m.half_v)
        free_hand = m._gather_cell_lp(p.cell_lp, cell)
        free_hand = free_hand + trunc_normal_logpdf(du, p.f_du_m, p.f_du_s,
                                                    -m.half_u, m.half_u)
        free_hand = free_hand + trunc_normal_logpdf(dv, p.f_dv_m, p.f_dv_s,
                                                    -m.half_v, m.half_v)
        free_hand = free_hand + gauss_logpdf(lz, p.f_lz_m, p.f_lz_s)
        free_hand = free_hand + vonmises_logpdf(psi, p.f_psi_m, p.f_kappa)

        anch = m._log_f_anch(p, u, v, lz, psi)
        free = m._log_f_free(p, cell, u, v, lz, psi)
    assert torch.equal(anch, anch_hand), "legacy anchored ln z is not the plain Normal"
    assert torch.equal(free, free_hand), "legacy free ln z is not the plain Normal"


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_switch_adds_no_parameter_and_no_buffer(sel):
    """No state may appear with the switch on: the `legacy` state_dict has to stay
    byte-identical, so a `physical` checkpoint remains loadable by a `legacy` build and
    `scripts/verify_parity.py` keeps its meaning."""
    leg, _ = _model(sel, "legacy")
    phy, _ = _model(sel, "physical")
    assert list(leg.state_dict().keys()) == list(phy.state_dict().keys())
    for k, v in leg.state_dict().items():
        assert torch.equal(v, phy.state_dict()[k])


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_physical_moves_the_likelihood_and_nothing_else(sel):
    """Guard against a silent no-op, and against an over-broad one.

    `ln z` sits inside the emission density, which sits inside the lattice, so there is no
    separable `coord_ll` term to isolate the way `nll_terms` does for the AR family — the
    number that must move is `log_prob` itself. What must NOT move is the op half of the
    lattice, i.e. the exact `q(N|x)`, which is the family's whole claim against v1."""
    leg, geom = _model(sel, "legacy")
    phy, _ = _model(sel, "physical")
    b = _batch(leg, geom)
    with torch.inference_mode():
        assert not torch.allclose(leg.log_prob(b), phy.log_prob(b))
    assert np.array_equal(leg.length_pmf(b["xf"][:1], b["nx"][:1]),
                          phy.length_pmf(b["xf"][:1], b["nx"][:1]))


# ---------------------------------------------------------------------------
# 2. each mixture component is a proper density
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_anchored_component_normalizes_over_the_four_coordinate_box(sel):
    """MC integral of `TN(u) TN(v) TN(ln z | u) vM(psi)` over
    `[lo_u, hi_u] x [lo_v, hi_v] x (lo_z(u), ln 1/2] x (-pi, pi]` is 1.

    Each component is integrated separately, because a mixture of a proper density and an
    improper one still looks plausible: `p*1 + (1-p)/Z` is finite and smooth and wrong.
    The `ln z` leg is `u`-dependent, so its width rides inside the integrand rather than
    factoring out of it."""
    m, geom = _model(sel, "physical")
    b = _batch(m, geom, B=1, L=1)
    M = 400_000
    g = torch.Generator().manual_seed(7)
    with torch.inference_mode():
        q = _one_state(_emit_params(m, b, 1), M)
        u = m.lo_u + (m.hi_u - m.lo_u) * torch.rand(M, generator=g)
        v = m.lo_v + (m.hi_v - m.lo_v) * torch.rand(M, generator=g)
        lo, hi = m.lnz_bounds(u)
        lz = lo + (hi - lo) * torch.rand(M, generator=g)
        psi = math.pi * (2.0 * torch.rand(M, generator=g) - 1.0)
        w = (m.hi_u - m.lo_u) * (m.hi_v - m.lo_v) * (2 * math.pi) * (hi - lo)
        got = float((w * torch.exp(m._log_f_anch(q, u, v, lz, psi))).mean())
    assert got == pytest.approx(1.0, abs=0.02), f"anchored component integrates to {got}"


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_free_component_normalizes_cell_by_cell(sel):
    """The insertion component is a cell categorical times within-cell offsets, so its
    integral over one cell must be that cell's probability — and the categorical sums to
    1 by construction. Integrating a few cells and checking the categorical separately is
    the same statement as one global integral, at a fraction of the variance."""
    m, geom = _model(sel, "physical")
    b = _batch(m, geom, B=1, L=1)
    M = 200_000
    g = torch.Generator().manual_seed(9)
    with torch.inference_mode():
        q = _one_state(_emit_params(m, b, 1), M)
        assert float(q.cell_lp.exp().sum()) == pytest.approx(1.0, abs=1e-5)
        for cell in (0, 17, geom.n_cells // 2, geom.n_cells - 1):
            cx, cy = geom.cell_center(cell)
            u = cx + m.half_u * (2.0 * torch.rand(M, generator=g) - 1.0)
            v = cy + m.half_v * (2.0 * torch.rand(M, generator=g) - 1.0)
            lo, hi = m.lnz_bounds(u)
            lz = lo + (hi - lo) * torch.rand(M, generator=g)
            psi = math.pi * (2.0 * torch.rand(M, generator=g) - 1.0)
            c = torch.full((M,), cell, dtype=torch.long)
            w = (2 * m.half_u) * (2 * m.half_v) * (2 * math.pi) * (hi - lo)
            got = float((w * torch.exp(m._log_f_free(q, c, u, v, lz, psi))).mean())
            want = float(q.cell_lp[cell].exp())
            assert got == pytest.approx(want, rel=0.03), f"cell {cell}: {got} vs {want}"


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_log_cell_mass_and_the_constrained_alignment_are_untouched(sel):
    """The plan's §2 claim, asserted rather than trusted.

    A cell is a box in `(u, v)` only. `_log_cell_mass` integrates those two and
    marginalises `ln z` and `psi`, whose factors integrate to 1 — and `int f(ln z | u)
    d ln z = 1` for EVERY `u`, bound or no bound. So the emission weights of the
    constrained lattice, and therefore every alignment `sample_coordinates` draws from it,
    are bit-identical across the switch; only the `ln z` drawn inside the chosen component
    differs. That is what makes WP-E a `ln z`-only change rather than a new family."""
    leg, geom = _model(sel, "legacy")
    phy, _ = _model(sel, "physical")
    b = _batch(leg, geom)
    L = b["yc"].shape[1]
    out = []
    with torch.inference_mode():
        for m in (leg, phy):
            S, e, _a, _o = m._encode(b["xf"], b["nx"])
            log_stay, log_emit = m._op_logprobs(S, e)
            p = _emit_params(m, b, L)
            mass, anch = m._log_cell_mass(p, b["yc"][:, None, :])
            edge = log_emit[:, :, None] + mass
            stay = log_stay[:, :, None].expand(*edge.shape[:2], L + 1)
            cols = edit_dp.sample_alignment(stay[0], edge[0], int(b["nx"][0]), L,
                                            generator=torch.Generator().manual_seed(5))
            out.append((mass, anch, list(cols)))
    assert torch.equal(out[0][0], out[1][0]), "the cell mass moved — the constrained DP " \
                                              "is not safe under the port"
    assert torch.equal(out[0][1], out[1][1]), "the anchored share moved"
    assert out[0][2] == out[1][2], "the sampled alignment moved"


# ---------------------------------------------------------------------------
# 3. neither draw path can leave the support (gate E2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_sample_coordinates_never_violates_either_wall(sel):
    """1e5 drawn emissions, hard zero on BOTH walls. Gate E2's target as a unit test:
    a nonzero count here is a bug, not a finding."""
    m, geom = _model(sel, "physical")
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    g = torch.Generator().manual_seed(11)
    coords = m.sample_coordinates_many(
        xf, nx, _chains(200, 500, geom.n_cells), generator=g)   # 200 x 500 = 1e5
    lnz = torch.cat([c[:, 2] for c in coords])
    assert lnz.numel() == 100_000 and torch.isfinite(lnz).all()
    assert int((lnz > LN_HALF).sum()) == 0, "draw above the kinematic z <= 1/2 bound"
    assert int((lnz < LN_ZCUT).sum()) == 0, "draw below the soft-drop bound"


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_ancestral_draw_never_violates_either_wall(sel):
    """The OTHER draw path, and the one the audit does not read.

    `sample` walks the lattice and calls `_draw_emission`, which has its own `ln z` draw
    per mixture component. A port that fixed only `sample_coordinates` would pass the
    support audit and still leak everywhere the ancestral walk is used — MBR candidates,
    `length_pmf` cross-checks, the closure draws."""
    m, geom = _model(sel, "physical")
    xf, nx = torch.randn(1, 6, 5), torch.tensor([6])
    K, n_rounds = 20_000, 5                       # 5 x 20k = 1e5, chunked to bound memory
    S, e, anchor, ok = m._encode(xf, nx)
    idx = torch.zeros(K, dtype=torch.long)
    parts = [S[0, idx]]
    if m.prefix_conditioning:
        parts.append(torch.tanh(m.pred_h0(e)).expand(K, -1))
    parts.append(e.expand(K, -1))
    torch.manual_seed(0)
    n_lo = n_hi = n = 0
    with torch.inference_mode():
        p = m._emit_params(torch.cat(parts, dim=-1), anchor[0, idx], ok[0, idx])
        # both components must actually be exercised, or the zeros below mean nothing
        assert 0.01 < float(p.log_p_anch.exp().mean()) < 0.99
        for _ in range(n_rounds):
            lnz = m._draw_emission(p)[0][:, 2]
            assert torch.isfinite(lnz).all()
            n += lnz.numel()
            n_lo += int((lnz < LN_ZCUT).sum())
            n_hi += int((lnz > LN_HALF).sum())
    assert n == 100_000
    assert (n_lo, n_hi) == (0, 0)


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_map_point_estimate_stays_inside_the_support(sel):
    """`_emission_mode` clamps the modal `ln z` into the support. Without it the decode
    could return a point estimate that the very support audit scoring it would flag —
    and the unclamped modes do sit outside: `mu_z` is an anchor plus a free shift."""
    m, geom = _model(sel, "physical")
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 6, 5), torch.tensor([6])
    pe = m.map_estimate(xf, nx, min_emissions=3)
    assert pe.multiplicity >= 3
    for n in pe.nodes:
        assert LN_ZCUT <= n.ln_z <= LN_HALF, f"MAP node outside the support: {n.ln_z}"


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_legacy_sampler_does_leak(sel):
    """The attribution arm. `e_v1_legacy_lnz` exists to reproduce the v0 / `v1_legacy_lnz`
    support failure under identical data; if the legacy path stopped leaking, that arm
    would attribute nothing and E2's second clause would be vacuous."""
    m, geom = _model(sel, "legacy")
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    g = torch.Generator().manual_seed(11)
    coords = m.sample_coordinates_many(xf, nx, _chains(20, 500, geom.n_cells), generator=g)
    lnz = torch.cat([c[:, 2] for c in coords])
    assert int((lnz > LN_HALF).sum()) + int((lnz < LN_ZCUT).sum()) > 0


# ---------------------------------------------------------------------------
# 4. the bound formula, at the betas the fielded files never exercise
# ---------------------------------------------------------------------------
def test_bounds_match_soft_drop_at_beta_zero():
    m, _ = _model(FAMILIES[0], "physical", lnz_zcut=0.1, lnz_beta=0.0)
    u = torch.linspace(0.0, 6.0, 13)
    lo, hi = m.lnz_bounds(u)
    assert torch.allclose(lo, torch.full_like(lo, LN_ZCUT))
    assert torch.allclose(hi, torch.full_like(hi, LN_HALF))
    assert float(hi[0] - lo[0]) == pytest.approx(math.log(5.0), abs=1e-6)


def test_general_beta_bound_is_the_exact_per_node_boundary():
    """`lo(u) = ln z_cut - beta*u`, hand-computed at both signs.

    This family reads the bound at the node's OWN `u` — which its factorization supports
    and the AR families' does not — so it is the exact Soft Drop boundary, and the SAME
    expression `data.stats.check_lnz_support` verifies the truth against. There is no
    `|beta|*half_u` slack to audit here because there is no cell-conditional loosening."""
    for beta in (0.5, -0.5, 2.0):
        m, _ = _model(FAMILIES[0], "physical", lnz_zcut=0.2, lnz_beta=beta)
        u = torch.linspace(-1.0, 6.0, 29)
        lo, hi = m.lnz_bounds(u)
        assert torch.allclose(lo, math.log(0.2) - beta * u, atol=1e-6)
        assert torch.allclose(hi, torch.full_like(hi, LN_HALF))


def test_general_beta_draws_respect_the_tilted_boundary():
    """A tilted bound is where a node-conditional implementation and a cell-conditional
    one visibly differ, so the draws are checked against the exact per-node line rather
    than against a constant."""
    m, geom = _model(FAMILIES[0], "physical", lnz_zcut=0.2, lnz_beta=0.5)
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    g = torch.Generator().manual_seed(3)
    coords = torch.cat(m.sample_coordinates_many(
        xf, nx, _chains(20, 500, geom.n_cells), generator=g))
    lo = math.log(0.2) - 0.5 * coords[:, 0]
    assert int((coords[:, 2] < lo - 1e-5).sum()) == 0
    assert int((coords[:, 2] > LN_HALF + 1e-5).sum()) == 0


def test_zcut_out_of_range_is_rejected():
    for bad in (0.6, 0.0):
        with pytest.raises(ValueError, match="lnz_zcut"):
            _model(FAMILIES[0], "physical", lnz_zcut=bad)


def test_unknown_support_is_rejected():
    with pytest.raises(ValueError, match="lnz_support"):
        _model(FAMILIES[0], "truncated")


# ---------------------------------------------------------------------------
# 5. the grooming-record guard is family-agnostic, and now covers edit
# ---------------------------------------------------------------------------
def test_check_lnz_support_reads_the_edit_config():
    """`data.stats.check_lnz_support` selects `model.lnz_support` through OmegaConf, so
    declaring the three fields on `EditTransducerConfig` buys the WP-A guard unchanged —
    which is why WP-E chose the same names rather than family-specific ones."""
    from h2p_rsd_junipr.data.stats import check_lnz_support

    def jets(lnz=(-1.5, -1.0), u=(2.0, 3.0), z_cut=0.1):
        x = (np.asarray(u, np.float32), np.asarray([2.0] * len(u), np.float32),
             np.asarray(lnz, np.float32), np.asarray([0.0] * len(u), np.float32))
        return [{"weight": 1.0, "z_cut": z_cut, "beta": 0.0, "kt_floor": 1.0, "x": x, "y": x}]

    cfg = load_config(["model=edit_v1", "encoder=gru", "model.lnz_support=physical",
                       "model.lnz_zcut=0.1", "model.lnz_beta=0.0"])
    out = check_lnz_support(jets(), cfg)
    assert out["ok"] and out["checked"] and out["frac_outside"] == 0.0
    with pytest.raises(ValueError, match="OUTSIDE"):
        check_lnz_support(jets(lnz=(-0.2,), u=(2.0,)), cfg)   # z = 0.82 > 1/2
    with pytest.raises(ValueError, match="z_cut"):
        check_lnz_support(jets(z_cut=0.05, lnz=(-2.9,), u=(2.0,)), cfg)

    leg = load_config(["model=edit_v1", "encoder=gru", "model.lnz_support=legacy"])
    assert check_lnz_support(jets(z_cut=0.3), leg) == {"support": "legacy"}
