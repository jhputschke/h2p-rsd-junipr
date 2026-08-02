"""`sample_coordinates_many` — one jet's K draws in one forward pass.

The per-draw hook re-runs `encode()` and `xattn_kv()` for every draw of the same jet,
which was ~67 of the 109 minutes of `notebooks/prod_test_v0.ipynb`
(docs/PLAN_prod_test_speedup.md §2). The batched sibling computes them once and decodes
the whole `(K, L_max)` block.

**Exactness is not claimed and must not be tested for**: padding a block to `L_max` and
drawing the whole thing at once consumes the RNG in a different order, by design. What
IS claimed, and is what these tests pin:

1. the shapes are per-row, exactly the shapes the loop returns;
2. the draws come from the SAME conditional — marginal means and sigmas agree with the
   loop to within Monte-Carlo error over ~2000 draws;
3. a family with no coordinate density returns a list of `None`, so every caller's
   `c is None -> cont_ok = False` degradation path still fires;
4. the default implementation on the contract loops the per-draw hook, so `cfm`,
   `cinn` and `diffusion` inherit today's behaviour untouched.
"""

from __future__ import annotations

import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.models.base import build_model

# mixed lengths, a repeat, and the empty draw — the padding cases in one list
DRAWS = [[0, 5, 12, 5], [7], [], [3, 3], [12, 0, 9]]
N_MC = 2000


def _model(sel, geom):
    return build_model(load_config(sel), geom).eval()


@pytest.mark.parametrize("sel", [["model=ar_junipr_v4", "encoder=gru"],
                                 ["model=ar_junipr_v2", "encoder=gru"],
                                 ["model=cinn", "encoder=deepsets"],
                                 ["model=cfm", "encoder=gru", "model.n_ode_steps=4"]],
                         ids=lambda s: s[0].split("=")[1])
def test_shapes_match_the_per_draw_loop(sel, batch):
    """Row `k` must be `(len(draws[k]), 4)` — including the empty draw, whose `(0, 4)`
    is what keeps `_leading_coords` returning None rather than raising."""
    b, geom = batch
    model = _model(sel, geom)
    xf, nx = b["xf"][:1], b["nx"][:1]
    torch.manual_seed(0)
    loop = [model.sample_coordinates(xf, nx, d) for d in DRAWS]
    torch.manual_seed(0)
    many = model.sample_coordinates_many(xf, nx, DRAWS)
    assert len(many) == len(DRAWS)
    for k, (a, c) in enumerate(zip(loop, many)):
        assert a.shape == c.shape == (len(DRAWS[k]), 4), f"row {k}"
        assert torch.isfinite(c).all()


def test_cells_land_in_their_own_cell(batch):
    """The padding must not leak: row `k`'s coordinates must sit within `(half_u,
    half_v)` of the centres of row `k`'s OWN cells, not of the padded zeros."""
    b, geom = batch
    model = _model(["model=ar_junipr_v4", "encoder=gru"], geom)
    xf, nx = b["xf"][:1], b["nx"][:1]
    torch.manual_seed(0)
    many = model.sample_coordinates_many(xf, nx, DRAWS)
    for cells, got in zip(DRAWS, many):
        for t, c in enumerate(cells):
            cx, cy = geom.cell_center(int(c))
            assert abs(float(got[t, 0]) - cx) <= model.half_u + 1e-5
            assert abs(float(got[t, 1]) - cy) <= model.half_v + 1e-5


def test_marginals_agree_with_the_loop_within_mc_error(batch):
    """The distributional claim. Same conditional, different RNG stream: means and
    sigmas of each coordinate must agree to within Monte-Carlo error over 2000 draws.

    5 standard errors, not 2: this is 16 simultaneous comparisons in a unit test that
    must not flake, and the effect it is guarding against (a batched path reading the
    wrong row's head parameters) is orders of magnitude larger than the band."""
    b, geom = batch
    model = _model(["model=ar_junipr_v4", "encoder=gru"], geom)
    xf, nx = b["xf"][:1], b["nx"][:1]
    chain = DRAWS[0]

    torch.manual_seed(0)
    loop = torch.stack([model.sample_coordinates(xf, nx, chain) for _ in range(N_MC)]).double()
    torch.manual_seed(1)
    many = torch.stack(model.sample_coordinates_many(xf, nx, [chain] * N_MC)).double()
    assert loop.shape == many.shape == (N_MC, len(chain), 4)

    m1, m2 = loop.mean(0), many.mean(0)
    s1, s2 = loop.std(0), many.std(0)
    se_mean = torch.sqrt((s1 ** 2 + s2 ** 2) / N_MC)
    se_sig = torch.sqrt((s1 ** 2 + s2 ** 2) / (2 * N_MC))   # sigma's own standard error
    assert torch.all((m1 - m2).abs() <= 5 * se_mean + 1e-9), (
        f"means differ by more than MC error:\n{(m1 - m2) / se_mean}"
    )
    assert torch.all((s1 - s2).abs() <= 5 * se_sig + 1e-9), (
        f"sigmas differ by more than MC error:\n{(s1 - s2) / se_sig}"
    )


def test_no_coordinate_density_returns_a_list_of_none(batch):
    """`ar_junipr_v1` has no coordinate head. Callers key on `c is None` to turn the
    continuous branch off (`eval/closure.py`, `scripts/leading_estimators.py`), so the
    batched hook must return that per row rather than an empty list or a tensor."""
    b, geom = batch
    model = _model(["model=ar_junipr_v1", "encoder=gru"], geom)
    assert model.has_continuous_coords is False
    got = model.sample_coordinates_many(b["xf"][:1], b["nx"][:1], DRAWS)
    assert got == [None] * len(DRAWS)


def test_empty_draw_list_and_all_empty_draws(batch):
    """Two degenerate shapes the closure loop can hand it: no draws at all, and a jet
    all of whose draws are the empty tree (`L_max == 0`)."""
    b, geom = batch
    model = _model(["model=ar_junipr_v4", "encoder=gru"], geom)
    xf, nx = b["xf"][:1], b["nx"][:1]
    assert model.sample_coordinates_many(xf, nx, []) == []
    allempty = model.sample_coordinates_many(xf, nx, [[], [], []])
    assert len(allempty) == 3 and all(t.shape == (0, 4) for t in allempty)


def test_contract_default_loops_the_per_draw_hook(batch):
    """Families that do not override it must be bit-identical to the loop — that is
    what makes the new hook a safe addition to the contract rather than a change."""
    b, geom = batch
    model = _model(["model=cinn", "encoder=deepsets"], geom)
    assert "sample_coordinates_many" not in vars(type(model)), (
        "cinn now overrides the batched hook; this test asserts the CONTRACT default and "
        "must be pointed at a family that still inherits it"
    )
    xf, nx = b["xf"][:1], b["nx"][:1]
    torch.manual_seed(3)
    loop = [model.sample_coordinates(xf, nx, d) for d in DRAWS]
    torch.manual_seed(3)
    many = model.sample_coordinates_many(xf, nx, DRAWS)
    for a, c in zip(loop, many):
        assert torch.equal(a, c)
