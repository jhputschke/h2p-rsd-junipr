"""What licenses gate E6 of docs/PLAN_prod_test_edit.md: the cross-FAMILY NLL A/B.

E6 puts `e_v1`'s held-out NLL/jet beside `v1_contstop`'s and reads the difference as a
statement about the two factorizations. That is only a measurement if both numbers are
densities on the same space, normalized the same way, and reported under the same
per-jet convention — three claims that are easy to assert and easy to get wrong, and
whose failure looks exactly like a win.

* **Same space, both normalized.** The AR family factorizes an emission as
  `q(cell) . f(du) f(dv) f(ln z) f(psi)`; the edit family as a two-component mixture over
  `(u, v, ln z, psi)` inside a lattice. Written down they share nothing. Integrated over
  the same box they must both give 1 — that is the whole content of "comparable".
* **Same `ln z` normalization.** A truncated `ln z` concentrates its mass on a 1.61-wide
  interval and *gains* NLL against a Normal on `R` for reasons that have nothing to do
  with fit quality. So the box test is run at `lnz_support=physical` on both sides, and
  the mismatch is shown to move the number — which is why the plan makes E6 conditional
  on E2 and why `scripts/prod_test_edit_gates.py` prints a `!` on any mixed row.
* **Same per-jet convention.** Both report `-log_prob` per jet, weighted-mean over the
  validation set (`train.trainer.Trainer._validate`), with no per-emission division on
  either side.

The box is the geometry window x the grooming interval x the psi circle, at `beta = 0`
so the `ln z` bound is a constant and the two families' conventions coincide exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

LN_HALF = math.log(0.5)
LN_ZCUT = math.log(0.1)

# The two arms E6 compares: `v1_contstop` is ar_junipr_v4 with the implicit continue/stop
# length model, and `e_v1` is the stage-1 edit transducer. Both `physical`, per E6.
AR = ["model=ar_junipr_v4", "encoder=gru", "model.use_multiplicity_head=false",
      "model.lnz_support=physical"]
EDIT = ["model=edit_v1", "encoder=gru", "model.lnz_support=physical"]


def _model(sel, seed=0):
    cfg = load_config(list(sel))
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    return build_model(cfg, geom).eval(), geom


def _context(geom, seed=1):
    """ONE frozen context — the same `x` on both sides, so the only thing that differs
    between the two integrals is the factorization being integrated."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 5, 5, generator=g), torch.tensor([5])


def _box_points(m, geom, M, seed):
    """`M` uniform points in `[lo_u, hi_u] x [lo_v, hi_v] x (ln z_cut, ln 1/2] x (-pi, pi]`,
    with the cell each one falls in — the common integration domain."""
    g = torch.Generator().manual_seed(seed)
    lo_u, hi_u = geom.ln_invdelta_range
    lo_v, hi_v = geom.ln_kt_range
    u = lo_u + (hi_u - lo_u) * torch.rand(M, generator=g)
    v = lo_v + (hi_v - lo_v) * torch.rand(M, generator=g)
    lz = LN_ZCUT + (LN_HALF - LN_ZCUT) * torch.rand(M, generator=g)
    psi = math.pi * (2.0 * torch.rand(M, generator=g) - 1.0)
    cell = torch.tensor([geom.to_cell(float(a), float(b)) for a, b in zip(u, v)],
                        dtype=torch.long)
    vol = (hi_u - lo_u) * (hi_v - lo_v) * (LN_HALF - LN_ZCUT) * (2 * math.pi)
    return u, v, lz, psi, cell, float(vol)


def _ar_emission_density(m, geom, xf, nx, u, v, lz, psi, cell):
    """`q(cell) . f(du, dv, ln z, psi | cell)` at the FIRST emission of the AR decoder.

    The coordinate head is conditioned on the cell, so the parameters are computed for
    every cell once and gathered per point — which is what makes a single global box
    integral possible rather than 100 per-cell ones."""
    all_cells = list(range(geom.n_cells))
    params = m.coord_head_params(xf, nx, all_cells)         # each (n_cells,)
    e = m.encode(xf, nx)
    out = m._decode_states(torch.zeros(1, 0, dtype=torch.long), e, m.xattn_kv(xf, nx))
    eh = torch.cat([out, e.unsqueeze(1).expand(-1, out.shape[1], -1)], dim=-1)
    cell_lp = torch.log_softmax(m.split_head(eh[:, 0, :]), dim=-1)[0]   # (n_cells,)
    p_t = params.apply(lambda q: q[cell])
    cx, cy = m.cell_cx[cell], m.cell_cy[cell]
    return torch.exp(cell_lp[cell] + m._coord_logprob(p_t, u, v, lz, psi, cx, cy))


def _edit_emission_density(m, xf, nx, u, v, lz, psi, cell):
    """The edit family's emission density at lattice state `(i, t) = (0, 0)`: the same
    four coordinates, as a two-component mixture rather than a cell factorization."""
    from h2p_rsd_junipr.models.edit import _EmitParams

    S, e, anchor, ok = m._encode(xf, nx)
    p = m._emit_params(m._emit_input(S, e, None), anchor[:, :, None, :], ok[:, :, None])
    M = u.shape[0]
    q = _EmitParams(*[f[0, 0, 0].expand(M) if f.ndim == 3 else f[0, 0, 0] for f in p])
    return torch.exp(m._log_f_emit(q, cell, u, v, lz, psi))


# ---------------------------------------------------------------------------
# the box integral — both factorizations are densities on the same space
# ---------------------------------------------------------------------------
def test_both_families_integrate_to_one_over_the_same_box():
    """The claim E6 rests on, stated as arithmetic.

    `coord_head_params` is a teacher-forced replay, so its cell-conditioning is exact
    here; the edit block is read at one lattice state. Both are the per-emission
    coordinate density the respective NLL sums over."""
    ar, geom = _model(AR)
    ed, _ = _model(EDIT)
    xf, nx = _context(geom)
    M = 400_000
    u, v, lz, psi, cell, vol = _box_points(ar, geom, M, seed=5)
    with torch.inference_mode():
        got_ar = float(vol * _ar_emission_density(ar, geom, xf, nx, u, v, lz, psi, cell).mean())
        got_ed = float(vol * _edit_emission_density(ed, xf, nx, u, v, lz, psi, cell).mean())
    assert got_ar == pytest.approx(1.0, abs=0.03), f"ar_junipr_v4/contstop: {got_ar}"
    assert got_ed == pytest.approx(1.0, abs=0.03), f"edit_v1: {got_ed}"


def test_a_legacy_ln_z_head_does_not_integrate_to_one_over_that_box():
    """Why E6 is conditional on E2 rather than merely accompanied by it.

    `legacy` is a Normal on all of `R`, so over the grooming interval it integrates to
    LESS than 1 — it spends probability mass outside the box the other arm normalizes
    over, which is precisely the constant offset that would masquerade as a fit
    difference. Checked on BOTH families, since either side could be the mismatched one."""
    M = 200_000
    for sel, label in ((AR, "ar"), (EDIT, "edit")):
        legacy = [s.replace("physical", "legacy") for s in sel]
        m, geom = _model(legacy)
        xf, nx = _context(geom)
        u, v, lz, psi, cell, vol = _box_points(m, geom, M, seed=5)
        with torch.inference_mode():
            f = (_ar_emission_density(m, geom, xf, nx, u, v, lz, psi, cell) if label == "ar"
                 else _edit_emission_density(m, xf, nx, u, v, lz, psi, cell))
            got = float(vol * f.mean())
        assert got < 0.97, (f"{label}: the legacy head integrates to {got} over the "
                            f"grooming box — if that were ~1 the two supports would be "
                            f"interchangeable and E6 would need no E2 precondition")


# ---------------------------------------------------------------------------
# the per-jet convention — both report the same quantity
# ---------------------------------------------------------------------------
def test_both_families_report_nll_under_the_same_per_jet_convention(batch):
    """`Trainer._validate` computes `-log_prob(batch)` and takes the weight-weighted mean
    over jets. So "NLL/jet" means the same thing on both sides only if `log_prob` is
    per-jet, padding-invariant, and equal to `-training_objective` — no per-emission
    division anywhere. All three, both families."""
    b, geom = batch
    B = b["xf"].shape[0]
    for sel in (AR, EDIT):
        cfg = load_config(list(sel))
        m = build_model(cfg, geom).eval()
        assert m.exact_likelihood is True, f"{sel[0]}: NLL is not a normalized density"
        with torch.inference_mode():
            lp = m.log_prob(b)
            assert lp.shape == (B,) and torch.isfinite(lp).all()
            assert torch.allclose(m.training_objective(b), -lp, atol=1e-5)
            # padding-invariance: the batched number and the per-jet number are one model
            for i in range(3):
                one = {k: v[i: i + 1] for k, v in b.items() if k != "w"}
                assert float(m.log_prob(one)[0]) == pytest.approx(
                    float(lp[i]), rel=1e-5, abs=1e-4)


def test_the_two_families_are_not_trivially_the_same_number(batch):
    """A sanity floor on the tests above: if the two `log_prob`s happened to coincide,
    the box test would be passing for the wrong reason."""
    b, geom = batch
    ar = build_model(load_config(AR), geom).eval()
    ed = build_model(load_config(EDIT), geom).eval()
    with torch.inference_mode():
        assert not torch.allclose(ar.log_prob(b), ed.log_prob(b))
