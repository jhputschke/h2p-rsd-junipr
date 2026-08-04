"""`eval/stability.py` — the WP4a loss-stability columns, and the boundary that keeps them
out of the uncertainty budget (docs/PLAN_PosteriorClusters.md §8.5, §8.6).

The executable half of §8.6. The tempting misuse is specific — folding
`d(y_linear, y_bounded)` into the systematics beside `generator_spread` — and a docstring
saying "don't" is not a guard. `test_loss_spread_not_in_systematics` is.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from h2p_rsd_junipr.eval import stability as st

SRC = Path(__file__).resolve().parents[1] / "src" / "h2p_rsd_junipr"


def _clique_matrix(n_empty=6, n_dense=5, n_far=9, empty_gap=40.0, dense_scale=0.5, seed=0):
    """A `D` shaped like the §8.4 hazard: an EXACT zero-diameter empty clique, a tight
    non-empty cluster, and a diffuse remainder. Multiplicities come back beside it so the
    N = 0 stratum is identifiable the way `loss_stability_row` expects."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        np.zeros(n_empty),                                   # the empty clique, all at 0
        empty_gap + rng.normal(0.0, dense_scale, n_dense),   # a tight non-empty cluster
        empty_gap + rng.normal(0.0, 12.0, n_far),            # the diffuse remainder
    ])
    D = np.abs(X[:, None] - X[None, :])
    D[:n_empty, :n_empty] = 0.0     # `mbr._empty_value` returns EXACTLY 0 for two empties
    mults = np.array([0] * n_empty + [3] * (n_dense + n_far))
    return D, mults


# ---------------------------------------------------------------------------
# §8.6 — the module boundary, enforced rather than conventioned
# ---------------------------------------------------------------------------
def test_loss_spread_not_in_systematics():
    """`eval/systematics.py` must neither import from `stability` nor emit any of its keys.

    `generator_spread` varies something unknown about NATURE, so its spread is a real
    uncertainty on a fixed target. Loss choice varies something the ANALYST decides, and
    `linear` and `bounded` are not two approximations to one quantity — they are the
    Frechet median and a density mode, two different functionals of one posterior. Quoting
    their spread as a systematic is quoting the mean-minus-median difference as a
    systematic on the mean, and it double-counts a width the cluster radii already report.
    """
    src = (SRC / "eval" / "systematics.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "stability" not in (node.module or ""), (
                "eval/systematics.py imports eval/stability.py — the module boundary IS "
                "the guard against a later quadrature sum (plan §8.6)"
            )
        if isinstance(node, ast.Import):
            assert not any("stability" in a.name for a in node.names)
    for key in st.STABILITY_COLUMNS:
        assert key not in src, (
            f"eval/systematics.py mentions {key!r}; the loss spread is a STABILITY check "
            f"reported beside the answer, never folded into the uncertainty budget"
        )
    # ...and the same in the other direction: the emitted dict says so about itself, so a
    # reader of the artifact alone cannot mistake it for an error bar.
    D, mults = _clique_matrix()
    out = st.summarise_stability([st.loss_stability_row(D, mults=mults)])
    assert out["is_a_systematic"] is False and "not a systematic" in out["note"]


def test_generator_spread_signature_carries_no_loss_knob():
    """A defence against the other route in: adding an `mbr_loss=` parameter to the
    systematic would let a caller produce the forbidden number without importing anything."""
    from h2p_rsd_junipr.eval.systematics import generator_spread

    params = set(inspect.signature(generator_spread).parameters)
    assert not (params & {"mbr_loss", "loss", "losses", "diagnostic_losses"})


# ---------------------------------------------------------------------------
# The columns themselves
# ---------------------------------------------------------------------------
def test_columns_are_all_present_even_when_undefined():
    """Every column exists on every row, NaN where it could not be computed. A missing key
    reads downstream as "never asked"; NaN reads as "asked, unavailable"."""
    D = np.zeros((6, 6))                       # no positive distance -> no bandwidth
    row = st.loss_stability_row(D, mults=np.zeros(6, dtype=int))
    assert set(row) == set(st.STABILITY_COLUMNS)
    assert row["eps_per_jet"] == 0.0
    assert np.isnan(row["argmin_moved"]), "a jet with no bandwidth is not a 'did not move'"
    summ = st.summarise_stability([row])
    assert summ["n_scored"] == 0 and summ["n_no_bandwidth"] == 1


def _from_positions(pos):
    p = np.asarray(pos, dtype=float)
    return np.abs(p[:, None] - p[None, :]), np.full(p.size, 3, dtype=int)


def test_argmin_moved_fires_exactly_on_the_medoid_in_the_valley():
    """The one column that survives WP4b's closure: no scikit-learn, valid at small K, and
    available on REAL data where the truth-based gate G2' is not.

    Both directions are pinned on hand-computable fixtures, because a flag that is always
    True (or always False) is indistinguishable from a broken one.

    **Fires** on §8.1's exact condition — clusters of unequal RADIUS. Ten coincident draws
    (a zero-radius minority) against twenty-six spread over [20, 40] (a broad majority).
    Mean distance from the tight point is 26*30/36 = 21.7 and from the broad centre is
    (10*30 + ~26*5)/36 = 11.9, so the LINEAR medoid takes the broad lobe; at eps = 2.4 the
    tight point has 10 neighbours and a broad point ~7, so the bounded argmin takes the
    tight one. Integrated density and peak density disagree, which is the whole of §8.1.

    **Does not fire** when the two agree: thirty coincident draws with an ISOLATED tail at
    10..24. The blob is both the most central (mean 5.8 vs 9.2) and the densest (30 vs 6 at
    eps = 5), so the flag is correctly silent."""
    D, mults = _clique_matrix()
    row = st.loss_stability_row(D, mults=mults, gamma=0.10)
    assert isinstance(row["argmin_moved"], bool)
    assert row["eps_per_jet"] > 0.0

    unequal_radii, m = _from_positions([0.0] * 10 + list(np.linspace(20.0, 40.0, 26)))
    assert st.loss_stability_row(unequal_radii, mults=m, gamma=0.10)["argmin_moved"] is True

    agree, m2 = _from_positions([0.0] * 30 + list(range(10, 25)))
    assert st.loss_stability_row(agree, mults=m2, gamma=0.10)["argmin_moved"] is False


def test_empty_clique_dominance_is_measured_at_the_same_eps(monkeypatch):
    """Gate G8'. `mbr._empty_value` puts all empty draws at mutual distance exactly 0, so
    an empty candidate's neighbour count is the clique size FOR ANY eps, while the clique
    is invisible to the bandwidth rule (which takes only positive distances). Both halves
    have to be measured or a near-miss looks like a pass."""
    D, mults = _clique_matrix(n_empty=9, n_dense=4, n_far=7)
    row = st.loss_stability_row(D, mults=mults, gamma=0.10)
    assert row["empty_clique_size"] == 9, "the clique's neighbour count IS its size"
    assert row["best_nonempty_count"] < 9
    assert row["empty_clique_wins"] is True and row["bounded_is_empty"] is True
    summ = st.summarise_stability([row])
    assert summ["empty_clique_wins"] == 1.0 and summ["G8prime_pass"] is False
    # ...and with the clique small enough, the non-empty cluster wins and the gate passes
    D2, m2 = _clique_matrix(n_empty=3, n_dense=9, n_far=6, dense_scale=0.2)
    row2 = st.loss_stability_row(D2, mults=m2, gamma=0.10)
    assert row2["empty_clique_wins"] is False and row2["bounded_is_empty"] is False
    assert st.summarise_stability([row2])["G8prime_pass"] is True


def test_bounded_is_members0_needs_an_exemplar_to_compare_against():
    D, mults = _clique_matrix()
    assert np.isnan(st.loss_stability_row(D, mults=mults)["bounded_is_members0"])
    row = st.loss_stability_row(D, mults=mults, top_exemplar=0)
    assert isinstance(row["bounded_is_members0"], bool)


def test_truth_distances_only_appear_when_a_truth_is_supplied():
    D, mults = _clique_matrix()
    rng = np.random.default_rng(0)
    dt = rng.uniform(1.0, 20.0, D.shape[0])
    bare = st.loss_stability_row(D, mults=mults)
    assert np.isnan(bare["d_mbr"]) and np.isnan(bare["d_bounded"])
    row = st.loss_stability_row(D, mults=mults, top_exemplar=3, d_to_truth=dt)
    for k in ("d_mbr", "d_bounded", "d_top"):
        assert np.isfinite(row[k])
    assert row["d_top"] == pytest.approx(dt[3])


def test_summary_reports_what_it_dropped():
    """"No silent caps": a jet that could not be scored has to show up in the denominator,
    or a smaller sample reads as "covered everything"."""
    D, mults = _clique_matrix()
    rows = [st.loss_stability_row(D, mults=mults) for _ in range(3)]
    rows.append(st.loss_stability_row(np.zeros((5, 5)), mults=np.zeros(5, dtype=int)))
    summ = st.summarise_stability(rows)
    assert summ["n_jets"] == 4 and summ["n_scored"] == 3 and summ["n_no_bandwidth"] == 1
