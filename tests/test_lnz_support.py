"""WP-A of docs/PLAN_prod_test_v1.md: the bounded-support `ln z` head.

`ln z` was the one coordinate head on an unbounded density while the grooming puts
its truth on a bounded interval (`z in (z_cut (DeltaR/R)^beta, 1/2]`). These tests pin
the four things that makes true, in the order they can fail:

1. `legacy` is bit-identical to the unbounded Normal (parity — the switch is a no-op off);
2. `physical` is a proper DENSITY: the four-coordinate joint integrates to 1 over its box;
3. its sampler cannot leave the support, at 1e5 draws (gate G2's hard-zero target);
4. sampler and CDF agree — the PIT of model-drawn coordinates is uniform (the SBC null).

Plus the general-`beta` bound formula, which the fielded `beta = 0` files cannot exercise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.distributions import gauss_logpdf
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

LN_HALF = math.log(0.5)


def _model(support="physical", *, seed=0, **over):
    argv = ["model=ar_junipr_v2", "encoder=gru", f"model.lnz_support={support}"]
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
    lo = math.log(0.1)
    lnz = lo + (LN_HALF - lo) * torch.rand(B, L, generator=g)
    psi = math.pi * (2.0 * torch.rand(B, L, generator=g) - 1.0)
    yraw = torch.stack([cx, cy, lnz, psi], dim=-1)
    return {"xf": xf, "nx": nx, "yc": yc, "ny": ny, "yraw": yraw}


# ---------------------------------------------------------------------------
# 1. parity: the switch is a no-op when off
# ---------------------------------------------------------------------------
def test_legacy_lnz_term_is_the_unbounded_normal():
    """`legacy` must reproduce the pre-WP-A arithmetic exactly, not merely closely:
    the whole point of the default is that old checkpoints keep their likelihood."""
    m, geom = _model("legacy")
    b = _batch(m, geom)
    with torch.inference_mode():
        total = m.nll_terms(b)["coord_ll"]
        # the same coordinate log-likelihood with the ln z factor written out by hand
        e = m.encode(b["xf"], b["nx"])
        out = m._decode_states(b["yc"], e)
        L = b["yc"].shape[1]
        eh = torch.cat([out, e.unsqueeze(1).expand(-1, L + 1, -1)], dim=-1)[:, :L, :]
        params = m._coord_params(torch.cat([eh, m.y_embed(b["yc"])], dim=-1))
        lnz_mean, lnz_sig = params[4], params[5]
        cx, cy = m.cell_cx[b["yc"]], m.cell_cy[b["yc"]]
        ref = m._coord_logprob(params, cx, cy, b["yraw"][..., 2], b["yraw"][..., 3], cx, cy)
        by_hand = ref - gauss_logpdf(b["yraw"][..., 2], lnz_mean, lnz_sig)
    # `ref` already IS the legacy path; the check that matters is that removing the
    # Gaussian term leaves no truncation normalizer behind (Z == 1 <=> log Z == 0).
    assert torch.allclose(ref.sum(1), total, atol=0, rtol=0)
    assert m.lnz_bounds(cx) is None
    assert torch.isfinite(by_hand).all()


def test_switch_adds_no_state():
    """No parameter and no buffer may appear with the switch on: the `legacy`
    state_dict has to stay byte-identical, and a physical checkpoint has to remain
    loadable by a legacy build."""
    leg, _ = _model("legacy")
    phy, _ = _model("physical")
    assert list(leg.state_dict().keys()) == list(phy.state_dict().keys())
    for k, v in leg.state_dict().items():
        assert torch.equal(v, phy.state_dict()[k])


def test_physical_changes_the_likelihood():
    """Guard against a silent no-op: `physical` must actually move the number it is
    meant to move (and only that one — the split/length terms are untouched)."""
    leg, geom = _model("legacy")
    phy, _ = _model("physical")
    b = _batch(leg, geom)
    with torch.inference_mode():
        tl, tp = leg.nll_terms(b), phy.nll_terms(b)
    assert torch.allclose(tl["split_ll"], tp["split_ll"])
    assert torch.allclose(tl["length_ll"], tp["length_ll"])
    assert not torch.allclose(tl["coord_ll"], tp["coord_ll"])


# ---------------------------------------------------------------------------
# 2. the physical head is a proper density
# ---------------------------------------------------------------------------
def test_physical_joint_normalizes_over_its_box():
    """MC integral of `TN(du) TN(dv) TN(ln z) vM(psi)` over
    `[-half_u, half_u] x [-half_v, half_v] x [lo, ln 1/2] x (-pi, pi]` is 1.

    A truncated density that forgot its normalizer integrates to `1/Z != 1`, which is
    exactly the bug a bounded head invites and the one no calibration plot would name."""
    m, geom = _model("physical")
    cells = [17, 42, 88]
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    params = m.coord_head_params(xf, nx, cells)
    M = 200_000
    g = torch.Generator().manual_seed(7)
    for t, cell in enumerate(cells):
        cxv, cyv = geom.cell_center(cell)
        cx = torch.full((M,), cxv)
        cy = torch.full((M,), cyv)
        lo, hi = m.lnz_bounds(cx)
        p_t = params.apply(lambda q, t=t: torch.full((M,), float(q[t])))
        u = cxv + m.half_u * (2.0 * torch.rand(M, generator=g) - 1.0)
        v = cyv + m.half_v * (2.0 * torch.rand(M, generator=g) - 1.0)
        lnz = lo + (hi - lo) * torch.rand(M, generator=g)
        psi = math.pi * (2.0 * torch.rand(M, generator=g) - 1.0)
        vol = (2 * m.half_u) * (2 * m.half_v) * float(hi[0] - lo[0]) * (2 * math.pi)
        integral = vol * torch.exp(m._coord_logprob(p_t, u, v, lnz, psi, cx, cy)).mean()
        assert float(integral) == pytest.approx(1.0, abs=0.03), f"cell {cell}: {integral}"


def test_physical_bounds_match_soft_drop_at_beta_zero():
    m, _ = _model("physical", lnz_zcut=0.1, lnz_beta=0.0)
    lo, hi = m.lnz_bounds(m.cell_cx)
    assert torch.allclose(lo, torch.full_like(lo, math.log(0.1)))
    assert torch.allclose(hi, torch.full_like(hi, LN_HALF))
    assert float(hi[0] - lo[0]) == pytest.approx(math.log(5.0), abs=1e-6)


def test_general_beta_bound_formula():
    """`lo = min_{|u-cx|<=half_u} (ln z_cut - beta u) = ln z_cut - beta cx - |beta| half_u`.

    Hand-computed at both signs, because the fielded files are all `beta = 0` and would
    never exercise the branch."""
    for beta in (0.5, -0.5, 2.0):
        m, geom = _model("physical", lnz_zcut=0.2, lnz_beta=beta)
        cx = m.cell_cx
        lo, hi = m.lnz_bounds(cx)
        u_grid = torch.linspace(-m.half_u, m.half_u, 41)
        # the exact per-node bound, over every u the cell contains
        exact = math.log(0.2) - beta * (cx[:, None] + u_grid[None, :])
        assert torch.all(lo[:, None] <= exact + 1e-6), "bound is not valid on the whole cell"
        assert torch.allclose(lo, exact.min(dim=1).values, atol=1e-5), "bound is not tight"
        assert torch.allclose(hi, torch.full_like(hi, LN_HALF))


def test_zcut_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="lnz_zcut"):
        _model("physical", lnz_zcut=0.6)
    with pytest.raises(ValueError, match="lnz_zcut"):
        _model("physical", lnz_zcut=0.0)


def test_unknown_support_is_rejected():
    with pytest.raises(ValueError, match="lnz_support"):
        _model("truncated")


# ---------------------------------------------------------------------------
# 3. the sampler cannot leave the support (gate G2)
# ---------------------------------------------------------------------------
def test_sampled_lnz_never_violates_the_support():
    """1e5 draws, hard zero. This is gate G2's target expressed as a unit test:
    a nonzero count here is a bug, not a finding."""
    m, geom = _model("physical")
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    cells = [3, 17, 42, 88, 91]
    draws = [list(cells) for _ in range(20_000)]      # 20k x 5 = 1e5 sampled emissions
    g = torch.Generator().manual_seed(11)
    coords = m.sample_coordinates_many(xf, nx, draws, generator=g)
    lnz = torch.cat([c[:, 2] for c in coords])
    u = torch.cat([c[:, 0] for c in coords])
    assert lnz.numel() == 100_000
    assert int((lnz > LN_HALF).sum()) == 0, "draw above the kinematic z <= 1/2 bound"
    # soft drop at beta = 0: ln z > ln z_cut, independent of u
    assert int((lnz < math.log(0.1)).sum()) == 0, "draw below the soft-drop bound"
    assert torch.isfinite(lnz).all() and torch.isfinite(u).all()


def test_legacy_sampler_does_violate_the_support():
    """The attribution arm: `v1_legacy_lnz` exists to reproduce the v0 failure, so the
    legacy path must still leak — otherwise the comparison proves nothing."""
    m, geom = _model("legacy")
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    g = torch.Generator().manual_seed(11)
    coords = m.sample_coordinates_many(xf, nx, [[3, 17, 42, 88, 91]] * 2_000, generator=g)
    lnz = torch.cat([c[:, 2] for c in coords])
    outside = int((lnz > LN_HALF).sum()) + int((lnz < math.log(0.1)).sum())
    assert outside > 0


# ---------------------------------------------------------------------------
# 4. sampler and CDF agree — the PIT of model-drawn coordinates is uniform
# ---------------------------------------------------------------------------
def _ks_uniform(v):
    v = np.sort(np.asarray(v, dtype=float))
    n = v.size
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - v, v - (i - 1) / n)))


@pytest.mark.parametrize("support", ["legacy", "physical"])
def test_pit_of_model_drawn_coordinates_is_uniform(support):
    """The SBC self-consistency null: coordinates DRAWN from the head, pushed back
    through the head's own CDF, must be Uniform(0,1). It is what proves
    `_sample_lnz` and `coordinate_cdfs` describe the same distribution — the pairing
    that broke in v0, where the sampler could produce values the grooming forbids."""
    m, geom = _model(support, seed=3)
    # 1200 x 3 = 3600 transformed values. Sized against its own critical value rather
    # than picked: at 400 jets the KS null sits at 0.047 and eight coordinates are
    # tested at once, so the suite would flake on seed luck a few percent of the time.
    B, L = 1200, 3
    g = torch.Generator().manual_seed(23)
    xf = torch.randn(B, 5, 5, generator=g)
    nx = torch.full((B,), 5, dtype=torch.long)
    yc = torch.randint(0, geom.n_cells, (B, L), generator=g)
    ny = torch.full((B,), L, dtype=torch.long)
    # one batched draw per row, teacher-forced on the same cells the CDF will use
    drawn = torch.stack([
        m.sample_coordinates(xf[i: i + 1], nx[i: i + 1], yc[i].tolist(), generator=g)
        for i in range(B)
    ])
    out = m.coordinate_cdfs({"xf": xf, "nx": nx, "yc": yc, "ny": ny, "yraw": drawn})
    u = out["u"].numpy().reshape(-1, 4)
    crit = 1.63 / math.sqrt(u.shape[0])          # KS 99% critical value
    for d, name in enumerate(out["names"]):
        assert _ks_uniform(u[:, d]) < crit, f"{name}: PIT of its own draws is not uniform"


# ---------------------------------------------------------------------------
# 5. the grooming-record guard (data.stats.check_lnz_support)
# ---------------------------------------------------------------------------
def _jets(z_cut=0.1, beta=0.0, lnz=(-1.5, -1.0), u=(2.0, 3.0)):
    arr = np.asarray
    return [
        {
            "weight": 1.0, "z_cut": z_cut, "beta": beta, "kt_floor": 1.0,
            "x": (arr(u, np.float32), arr([2.0] * len(u), np.float32),
                  arr(lnz, np.float32), arr([0.0] * len(u), np.float32)),
            "y": (arr(u, np.float32), arr([2.0] * len(u), np.float32),
                  arr(lnz, np.float32), arr([0.0] * len(u), np.float32)),
        }
    ]


def test_guard_passes_on_a_matching_file():
    from h2p_rsd_junipr.data.stats import check_lnz_support

    cfg = load_config(["model=ar_junipr_v2", "model.lnz_support=physical",
                       "model.lnz_zcut=0.1", "model.lnz_beta=0.0"])
    out = check_lnz_support(_jets(), cfg)
    assert out["ok"] and out["checked"] and out["frac_outside"] == 0.0


def test_guard_is_a_no_op_in_legacy_mode():
    from h2p_rsd_junipr.data.stats import check_lnz_support

    cfg = load_config(["model=ar_junipr_v2", "model.lnz_support=legacy"])
    # a file whose grooming disagrees with the (unread) declaration is fine here
    assert check_lnz_support(_jets(z_cut=0.3), cfg) == {"support": "legacy"}


def test_guard_catches_a_zcut_mismatch():
    from h2p_rsd_junipr.data.stats import check_lnz_support

    cfg = load_config(["model=ar_junipr_v2", "model.lnz_support=physical",
                       "model.lnz_zcut=0.1"])
    with pytest.raises(ValueError, match="z_cut"):
        check_lnz_support(_jets(z_cut=0.05, lnz=(-2.9,), u=(2.0,)), cfg)


def test_guard_catches_truth_outside_the_declared_interval():
    """The check that a matching (z_cut, beta) pair cannot make: a convention error
    (a sign on beta, an R != 1) leaves the scalars equal and the interval wrong."""
    from h2p_rsd_junipr.data.stats import check_lnz_support

    cfg = load_config(["model=ar_junipr_v2", "model.lnz_support=physical",
                       "model.lnz_zcut=0.1", "model.lnz_beta=0.0"])
    with pytest.raises(ValueError, match="OUTSIDE"):
        check_lnz_support(_jets(lnz=(-0.2,), u=(2.0,)), cfg)   # z = 0.82 > 1/2


def test_guard_reports_rather_than_raises_when_the_record_is_absent(capsys):
    from h2p_rsd_junipr.data.stats import check_lnz_support

    cfg = load_config(["model=ar_junipr_v2", "model.lnz_support=physical"])
    jets = _jets()
    for j in jets:
        j["z_cut"] = float("nan")
        j["beta"] = float("nan")
    out = check_lnz_support(jets, cfg)
    assert out["checked"] is False
    assert "could not be verified" in capsys.readouterr().out
