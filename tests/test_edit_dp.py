"""The edit transducer's lattice numerics (`models/edit_dp.py`).

The entire novel numerical risk of the family lives in one recursion, so it is pinned
against the one thing that cannot be wrong: for small `(n_x, n_y)`, **every monotone path
enumerated explicitly**. Everything else here (the exact length marginal, the
forward-backward responsibilities, Viterbi, batching) is checked against that same
recursion or against a closed-form invariant.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from h2p_rsd_junipr.models import edit_dp

SMALL = [(0, 0), (0, 3), (3, 0), (1, 1), (2, 2), (4, 4), (1, 4), (4, 1), (3, 2)]


def _lattice(nx, ny, seed=0, J=None):
    """A random single-jet lattice: `(log_stay (1,n_col,J), log_emit_edge (1,n_col,J-1))`.

    STAY and EMIT share one categorical (as they do in the model, which is what closes the
    normalization) and the emit edge carries an extra log-density factor on top."""
    torch.manual_seed(seed)
    n_col, J = nx + 1, (ny + 1 if J is None else J)
    lp = torch.log_softmax(torch.randn(1, n_col, J, 2, dtype=torch.float64), dim=-1)
    dens = torch.randn(1, n_col, J - 1, dtype=torch.float64)
    return lp[..., 0], lp[..., 1][:, :, : J - 1] + dens


def _enumerate(log_stay, log_emit_edge, nx, ny) -> float:
    """log q(y|x) by brute force: every interleaving of `nx` STAYs and `ny` EMITs."""
    stay = log_stay[0].tolist()
    edge = log_emit_edge[0].tolist()
    terms = []
    for emits in itertools.combinations(range(nx + ny), ny):
        chosen, i, j, s = set(emits), 0, 0, 0.0
        for k in range(nx + ny):
            if k in chosen:
                s += edge[i][j]
                j += 1
            else:
                s += stay[i][j]
                i += 1
        terms.append(s + stay[nx][ny])
    return float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), dim=0))


# --- the headline test ------------------------------------------------------
@pytest.mark.parametrize("nx,ny", SMALL)
def test_forward_recursion_reproduces_every_enumerated_path(nx, ny):
    stay, edge = _lattice(nx, ny)
    got = float(
        edit_dp.forward_logsumexp(stay, edge, torch.tensor([nx]), torch.tensor([ny]))[0]
    )
    assert got == pytest.approx(_enumerate(stay, edge, nx, ny), abs=1e-6)


def test_path_count_is_the_binomial_it_should_be():
    """A uniform lattice makes the recursion count paths: `C(nx+ny, ny)` of them. If the
    lattice ever gained or lost an edge this is what would notice."""
    nx, ny = 5, 4
    n_col, J = nx + 1, ny + 1
    zero = torch.zeros(1, n_col, J, dtype=torch.float64)
    got = edit_dp.forward_logsumexp(
        zero, zero[:, :, : J - 1], torch.tensor([nx]), torch.tensor([ny])
    )
    assert float(got.exp()) == pytest.approx(math.comb(nx + ny, ny), rel=1e-9)


# --- normalization ----------------------------------------------------------
@pytest.mark.parametrize("nx", [0, 1, 4])
def test_the_structural_process_is_normalized(nx):
    """`sum_n q(N=n|x) == 1` BEFORE any renormalization: the lattice is a proper
    generative process, which is what earns `exact_likelihood = True`.

    Checked on the raw terminal values from `forward_alpha`, not on the renormalized pmf
    (which would sum to 1 by construction and test nothing)."""
    torch.manual_seed(3)
    n_col, max_n = nx + 1, 400
    lp = torch.log_softmax(torch.randn(1, n_col, 2, dtype=torch.float64), dim=-1)
    stay = lp[:, :, 0, None].expand(1, n_col, max_n + 1)
    edge = lp[:, :, 1, None].expand(1, n_col, max_n)
    alpha = edit_dp.forward_alpha(stay, edge)
    mass = torch.exp(alpha[0, nx, :] + lp[0, nx, 0]).sum()
    assert float(mass) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("nx", [0, 2, 5])
def test_structural_length_pmf_sums_to_one_and_matches_the_forward(nx):
    torch.manual_seed(4)
    n_col, max_n = nx + 1, 12
    lp = torch.log_softmax(torch.randn(1, n_col, 2, dtype=torch.float64), dim=-1)
    pmf = edit_dp.structural_length_pmf(lp[..., 0], lp[..., 1], torch.tensor([nx]), max_n)
    assert pmf.shape == (1, max_n + 1)
    assert float(pmf.sum()) == pytest.approx(1.0, abs=1e-9)
    # ...and each entry is the SAME quantity the coordinate-carrying forward gives when
    # every emission density is 1 (log 0), i.e. the coordinates really do marginalise out.
    # Compared as RATIOS, because the pmf is renormalized over n <= max_n and the mass past
    # the cap genuinely exists -- that truncation is documented, not an error.
    ref = None
    for n in (0, 1, 4, max_n):
        stay = lp[:, :, 0, None].expand(1, n_col, n + 1)
        edge = lp[:, :, 1, None].expand(1, n_col, n)
        lq = float(
            edit_dp.forward_logsumexp(stay, edge, torch.tensor([nx]), torch.tensor([n]))[0]
        )
        if ref is None:
            ref = (lq, float(pmf[0, n]))
            continue
        assert math.exp(lq - ref[0]) == pytest.approx(float(pmf[0, n]) / ref[1], rel=1e-6)


# --- forward-backward -------------------------------------------------------
@pytest.mark.parametrize("nx,ny", [(3, 2), (4, 4), (0, 3), (3, 0)])
def test_backward_meets_the_forward_and_responsibilities_sum_to_one(nx, ny):
    stay, edge = _lattice(nx, ny, seed=7)
    nxt, nyt = torch.tensor([nx]), torch.tensor([ny])
    log_z = float(edit_dp.forward_logsumexp(stay, edge, nxt, nyt)[0])
    beta = edit_dp.backward_beta(stay, edge, nxt, nyt)
    assert float(beta[0, 0, 0]) == pytest.approx(log_z, abs=1e-6)

    resp = edit_dp.forward_backward_responsibilities(stay, edge, nxt, nyt)
    assert float(resp["log_z"][0]) == pytest.approx(log_z, abs=1e-6)
    # every parton node is emitted exactly once, from somewhere
    for t in range(ny):
        assert float(resp["gamma_emit"][0, :, t].sum()) == pytest.approx(1.0, abs=1e-6)
    # ...and every hadron node is advanced past exactly once
    for i in range(nx):
        assert float(resp["gamma_stay"][0, i, :].sum()) == pytest.approx(1.0, abs=1e-6)


def test_responsibilities_vanish_outside_a_jets_own_lattice():
    """A jet padded to the batch maximum must not pick up posterior mass on columns it
    does not have -- that would silently feed the closure diagnostics padding."""
    stay, edge = _lattice(6, 6, seed=11)
    resp = edit_dp.forward_backward_responsibilities(
        stay, edge, torch.tensor([2]), torch.tensor([3])
    )
    g = resp["gamma_emit"][0]
    assert float(g[3:, :].abs().max()) == pytest.approx(0.0, abs=1e-9)   # i > nx
    assert float(g[:, 3:].abs().max()) == pytest.approx(0.0, abs=1e-9)   # t >= ny


# --- Viterbi ----------------------------------------------------------------
@pytest.mark.parametrize("nx,ny", [(3, 2), (4, 4), (1, 4), (3, 0)])
def test_viterbi_is_a_single_path_and_never_beats_the_sum(nx, ny):
    stay, edge = _lattice(nx, ny, seed=13)
    score, cols = edit_dp.viterbi_path(stay[0], edge[0], nx, min_n=ny)
    total = float(edit_dp.forward_logsumexp(stay, edge, torch.tensor([nx]), torch.tensor([ny]))[0])
    assert len(cols) == ny
    assert cols == sorted(cols) and all(0 <= c <= nx for c in cols)  # monotone
    assert score <= total + 1e-9
    # the best of the enumerated paths IS the Viterbi score
    best = max(
        _path_score(stay[0].tolist(), edge[0].tolist(), emits, nx, ny)
        for emits in itertools.combinations(range(nx + ny), ny)
    )
    assert score == pytest.approx(best, abs=1e-9)


def _path_score(stay, edge, emits, nx, ny):
    chosen, i, j, s = set(emits), 0, 0, 0.0
    for k in range(nx + ny):
        if k in chosen:
            s += edge[i][j]
            j += 1
        else:
            s += stay[i][j]
            i += 1
    return s + stay[nx][ny]


@pytest.mark.parametrize("nx,ny", [(4, 4), (5, 3), (2, 6)])
def test_one_per_column_restricts_to_the_edit_semantics(nx, ny):
    """`one_per_column=True` is the decode that says a hadron node is kept-and-smeared or
    deleted, never duplicated. It searches a strict SUBSET of the paths, so it can only
    score lower — and every hadron column it uses appears at most once."""
    stay, edge = _lattice(nx, ny, seed=19)
    free_score, _free_cols = edit_dp.viterbi_path(stay[0], edge[0], nx, min_n=ny)
    score, cols = edit_dp.viterbi_path(stay[0], edge[0], nx, min_n=ny, one_per_column=True)
    total = float(
        edit_dp.forward_logsumexp(stay, edge, torch.tensor([nx]), torch.tensor([ny]))[0]
    )
    assert len(cols) == ny and cols == sorted(cols)
    assert score <= free_score + 1e-9 <= total + 1e-9
    hadron = [c for c in cols if c < nx]          # the terminal column is insertions
    assert len(set(hadron)) == len(hadron)
    # it IS the best such path: check against enumeration over the restricted class
    best = max(
        (_path_score(stay[0].tolist(), edge[0].tolist(), emits, nx, ny)
         for emits in itertools.combinations(range(nx + ny), ny)
         if _one_per_column(emits, nx, ny)),
        default=None,
    )
    assert best is not None and score == pytest.approx(best, abs=1e-9)


def _one_per_column(emits, nx, ny):
    """Does this interleaving emit at most once from each non-terminal column?"""
    chosen, i, seen = set(emits), 0, []
    for k in range(nx + ny):
        if k in chosen:
            if i < nx:
                seen.append(i)
        else:
            i += 1
    return len(set(seen)) == len(seen)


def test_viterbi_honours_the_minimum_length():
    stay, edge = _lattice(4, 0, seed=17, J=6)
    for floor in (0, 1, 3):
        _score, cols = edit_dp.viterbi_path(stay[0], edge[0], 4, min_n=floor)
        assert len(cols) >= floor


# --- batching ---------------------------------------------------------------
def test_batched_masked_agrees_with_a_per_jet_loop():
    """The property that makes training and evaluation the same model: a jet padded into
    a batch must score exactly as it does alone."""
    torch.manual_seed(23)
    nxs, nys = [0, 2, 5, 3], [3, 0, 2, 4]
    n_col, J = max(nxs) + 1, max(nys) + 1
    lp = torch.log_softmax(torch.randn(4, n_col, J, 2, dtype=torch.float64), dim=-1)
    stay, edge = lp[..., 0], lp[..., 1][:, :, : J - 1] + torch.randn(4, n_col, J - 1, dtype=torch.float64)
    batched = edit_dp.forward_logsumexp(stay, edge, torch.tensor(nxs), torch.tensor(nys))
    for b, (nx, ny) in enumerate(zip(nxs, nys)):
        alone = edit_dp.forward_logsumexp(
            stay[b : b + 1, : nx + 1, : ny + 1],
            edge[b : b + 1, : nx + 1, :ny],
            torch.tensor([nx]), torch.tensor([ny]),
        )
        assert float(batched[b]) == pytest.approx(float(alone[0]), abs=1e-9)
        assert float(alone[0]) == pytest.approx(
            _enumerate(stay[b : b + 1, : nx + 1, : ny + 1],
                       edge[b : b + 1, : nx + 1, :ny], nx, ny),
            abs=1e-6,
        )


def test_sample_alignment_returns_monotone_paths_of_the_right_length():
    stay, edge = _lattice(5, 4, seed=29)
    torch.manual_seed(0)
    for _ in range(20):
        cols = edit_dp.sample_alignment(stay[0], edge[0], 5, 4)
        assert len(cols) == 4
        assert cols == sorted(cols) and all(0 <= c <= 5 for c in cols)


def test_sample_alignment_recovers_the_posterior_it_was_built_from():
    """A statistical check on the walk: the empirical frequency of the column that emitted
    `y_0` must match the forward-backward responsibility for it."""
    nx, ny = 3, 2
    stay, edge = _lattice(nx, ny, seed=31)
    resp = edit_dp.forward_backward_responsibilities(
        stay, edge, torch.tensor([nx]), torch.tensor([ny])
    )
    want = resp["gamma_emit"][0, :, 0].tolist()
    torch.manual_seed(5)
    counts = [0] * (nx + 1)
    K = 4000
    for _ in range(K):
        counts[edit_dp.sample_alignment(stay[0], edge[0], nx, ny)[0]] += 1
    got = [c / K for c in counts]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=0.03)
