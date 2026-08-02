"""`EditTransducer.sample_coordinates_many` — the batched coordinate sampler.

The per-draw `sample_coordinates` is ~6 ms whose cost is FLAT in `L` and `n_x`: it is
~100 tiny op launches, not arithmetic. `eval.support.run_support_audit` calls it 2 000
jets x 200 draws = 400 000 times per arm (~1 h 50 m), which is what made
`scripts/refresh_support_audit.py` unaffordable on this family. The override collapses
that to one pass per jet — the same fix `PLAN_prod_test_speedup.md` §2 made for the AR
family, plus the piece AR never needed: its coordinates are conditionally independent
given the cell chain, so its batched path is a padded teacher-forced replay, while here
the alignment is LATENT and must be sampled per draw before any coordinate can be.

**Bit-identity is not the bar and asserting it would be wrong.** The batched walk consumes
one `rand(B)` per lattice step where the loop consumes one `rand(())` per step per draw,
so the draws are from the same conditional and are NOT the same numbers — exactly what
`ARJunipr.sample_coordinates_many` documents for itself. So the equivalence tested here is
DISTRIBUTIONAL, and the invariants that must hold exactly are tested exactly:

* every drawn coordinate discretises back to the cell it was conditioned on;
* zero support violations under `physical`, on both walls (gate E2's target);
* the sampled alignment stays monotone (0 crossing pairs) — the RNN-T lattice cannot
  produce one, so a crossing is a bug in the batched walk, not a finding;
* the single-draw path is UNCHANGED by the shared factoring (`tests/test_edit_model.py`
  covers its behaviour; the extraction was verified bit-identical when it landed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models import edit_dp
from h2p_rsd_junipr.models.base import build_model

LN_HALF, LN_ZCUT = math.log(0.5), math.log(0.1)
FAMILIES = [["model=edit_v1", "encoder=gru"],
            ["model=edit_v2", "encoder=gru"],
            ["model=edit_v1", "encoder=gru", "model.physics_width=false"]]
IDS = ["v1", "v2", "v1-free-width"]


def _model(sel, support="physical", seed=0):
    cfg = load_config(list(sel) + [f"model.lnz_support={support}"])
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    return build_model(cfg, geom).eval(), geom


def _chains(n, n_cells, seed=13, lo=1, hi=6):
    g = torch.Generator().manual_seed(seed)
    return [[int(c) for c in torch.randint(0, n_cells, (int(torch.randint(lo, hi, (1,),
                                                                          generator=g)),),
                                           generator=g)] for _ in range(n)]


def _ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic, without scipy."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / a.size
    cb = np.searchsorted(b, allv, side="right") / b.size
    return float(np.abs(ca - cb).max())


# ---------------------------------------------------------------------------
# 1. the batched draws are draws from the same conditional
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_batched_and_looped_draws_agree_in_distribution(sel):
    """The claim the override rests on. One fixed jet and one fixed cell chain, so both
    paths sample the SAME conditional `q(coords | cells, x)`; 4 000 draws each; a
    two-sample KS per coordinate against its own 99.9% critical value.

    The alignment is latent and marginalised here, so this also tests the batched walk:
    a walk that visited the wrong columns would put the anchored component on the wrong
    hadron node and move `ln 1/DeltaR` and `ln k_t` first."""
    m, geom = _model(sel)
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 4, 5), torch.tensor([4])
    chain = [17, 42, 88]
    N = 4000
    with torch.inference_mode():
        loop = torch.stack([m.sample_coordinates(
            xf, nx, chain, generator=torch.Generator().manual_seed(1000 + i))
            for i in range(N)])
        bat = torch.cat(m.sample_coordinates_many(
            xf, nx, [chain] * N, generator=torch.Generator().manual_seed(7))).reshape(
                N, len(chain), 4)
    # 1.95/sqrt(n_eff) is the 99.9% two-sample KS point; the suite runs 12 of these
    crit = 1.95 * math.sqrt(2.0 / N)
    for t in range(len(chain)):
        for d, name in enumerate(("ln_invDelta", "ln_kt", "ln_z", "psi")):
            ks = _ks(loop[:, t, d].numpy(), bat[:, t, d].numpy())
            assert ks < crit, (f"node {t} {name}: batched and looped draws differ, "
                               f"KS {ks:.4f} vs crit {crit:.4f} — the batched path is "
                               f"not sampling the same conditional")


def test_the_batched_alignment_walk_reproduces_the_scalar_one():
    """`sample_alignment_batch` against `sample_alignment`, on a synthetic lattice, as a
    per-node histogram over which column emitted each node. Tested here rather than only
    through the model because it is the one piece with no AR counterpart."""
    torch.manual_seed(0)
    n_col, L, nx, ny, N = 5, 4, 4, 3, 20_000
    stay = torch.randn(1, n_col, L + 1).log_softmax(-1) * 0.5
    edge = torch.randn(1, n_col, L) * 0.5
    g = torch.Generator().manual_seed(1)
    scal = np.array([edit_dp.sample_alignment(stay[0], edge[0], nx, ny, generator=g)
                     for _ in range(N)])
    bat = edit_dp.sample_alignment_batch(
        stay.expand(N, -1, -1), edge.expand(N, -1, -1), torch.tensor([nx]),
        torch.tensor([ny]), generator=torch.Generator().manual_seed(2)).numpy()[:, :ny]
    for t in range(ny):
        a = np.bincount(scal[:, t], minlength=n_col) / N
        b = np.bincount(bat[:, t], minlength=n_col) / N
        assert np.abs(a - b).max() < 0.02, f"node {t}: column histograms differ"
    # the RNN-T lattice cannot produce a crossing; a nonzero count is a bug in the walk
    assert bool((np.diff(bat, axis=1) >= 0).all()), "the batched walk is not monotone"


def test_the_walk_respects_per_row_lengths():
    """Rows are padded to `L_max`, so a short row must stop at its own `ny` and leave the
    padded tail alone — otherwise a K-draw block would silently lengthen its short draws."""
    torch.manual_seed(0)
    n_col, L, B = 6, 5, 64
    stay = torch.randn(B, n_col, L + 1).log_softmax(-1)
    edge = torch.randn(B, n_col, L)
    ny = torch.arange(B) % (L + 1)                    # 0..L, every length represented
    cols = edit_dp.sample_alignment_batch(stay, edge, torch.tensor([n_col - 1]), ny,
                                          generator=torch.Generator().manual_seed(3))
    assert cols.shape == (B, L)
    for b in range(B):
        n = int(ny[b])
        assert bool((cols[b, :n] <= n_col - 1).all()) and bool((cols[b, :n] >= 0).all())
        assert bool((torch.diff(cols[b, :n]) >= 0).all()), f"row {b} is not monotone"
        assert bool((cols[b, n:] == 0).all()), f"row {b} wrote past its own length"


# ---------------------------------------------------------------------------
# 2. the exact invariants, which batching must not weaken
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_every_batched_coordinate_lands_in_its_own_cell(sel):
    """The per-draw contract, now over a block of ragged chains: each coordinate must
    discretise back to the cell it was conditioned on, and each row must come back at its
    own length. Padding to `L_max` is where this would break."""
    m, geom = _model(sel)
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    chains = _chains(64, geom.n_cells)
    got = m.sample_coordinates_many(xf, nx, chains,
                                    generator=torch.Generator().manual_seed(5))
    assert len(got) == len(chains)
    for c, g in zip(chains, got):
        assert g.shape == (len(c), 4) and torch.isfinite(g).all()
        assert [geom.to_cell(float(r[0]), float(r[1])) for r in g] == c


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_batched_path_cannot_leave_the_ln_z_support(sel):
    """Gate E2's hard zero, on the path the support audit actually calls. 1e5 emissions."""
    m, geom = _model(sel, "physical")
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 5, 5), torch.tensor([5])
    lnz = torch.cat([c[:, 2] for c in m.sample_coordinates_many(
        xf, nx, [[3, 17, 42, 88, 91]] * 20_000,
        generator=torch.Generator().manual_seed(11))])
    assert lnz.numel() == 100_000 and torch.isfinite(lnz).all()
    assert int((lnz > LN_HALF).sum()) == 0 and int((lnz < LN_ZCUT).sum()) == 0


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_degenerate_blocks(sel):
    """Empty draw list, all-empty chains, and a mix of empty and non-empty rows — the
    shapes `run_support_audit` and `run_closure` actually hand it."""
    m, geom = _model(sel)
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 3, 5), torch.tensor([3])
    assert m.sample_coordinates_many(xf, nx, []) == []
    allempty = m.sample_coordinates_many(xf, nx, [[], [], []])
    assert len(allempty) == 3 and all(c.shape == (0, 4) for c in allempty)
    mixed = m.sample_coordinates_many(xf, nx, [[], [7], [], [3, 9]])
    assert [tuple(c.shape) for c in mixed] == [(0, 4), (1, 4), (0, 4), (2, 4)]


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_chunking_does_not_change_the_result_shape(sel):
    """`edit_v2` chunks the block for memory (`_block_rows`); the split must be invisible
    to the caller. Forced to 1 row per chunk, which is the loop's own shape."""
    m, geom = _model(sel)
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 4, 5), torch.tensor([4])
    chains = _chains(24, geom.n_cells, seed=21)
    big = m.sample_coordinates_many(xf, nx, chains,
                                    generator=torch.Generator().manual_seed(2))
    m._block_rows = lambda n_col, L: 1                       # one row per chunk
    small = m.sample_coordinates_many(xf, nx, chains,
                                      generator=torch.Generator().manual_seed(2))
    assert [tuple(c.shape) for c in big] == [tuple(c.shape) for c in small]
    for g in small:
        assert torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# 3. it is actually faster — the whole point
# ---------------------------------------------------------------------------
def test_the_batched_path_is_much_faster_than_the_loop():
    """A regression guard on the reason this code exists. The measured factor is ~220x
    (`edit_v1`) and ~40x (`edit_v2`) at K = 200; 5x is a floor that only a genuine
    regression — a per-draw call sneaking back into the block — would breach."""
    import time

    m, geom = _model(FAMILIES[0])
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 4, 5), torch.tensor([4])
    chains = _chains(200, geom.n_cells, seed=31)
    with torch.inference_mode():
        t0 = time.perf_counter()
        [m.sample_coordinates(xf, nx, c) for c in chains]
        loop = time.perf_counter() - t0
        t0 = time.perf_counter()
        m.sample_coordinates_many(xf, nx, chains)
        batched = time.perf_counter() - t0
    assert loop / max(batched, 1e-9) > 5.0, (
        f"batched path is only {loop / batched:.1f}x the loop — the per-jet work is being "
        f"repeated per draw again")
