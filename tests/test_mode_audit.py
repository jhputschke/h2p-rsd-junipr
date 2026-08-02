"""The mode-mass audit (docs/PLAN_ModeMassAudit.md §8): exactness, certificates and
the two consistency identities.

The audit's whole claim is that its numbers are CERTIFICATES rather than estimates —
`M_1 > 1/2` is a proof of dominance, not evidence for it. That claim rests on two
things, and both are pinned here rather than argued: the best-first search returns the
same masses *and the same order* as an exhaustive product (T1/T6), and the bookkeeping
identity `sum_i M_i + frontier + pruned = 1` holds at every termination point, however
aggressively the search prunes (T2). If either fails, every fraction the audit reports
is wrong in a way no downstream plot would reveal.

T4 is the one statistical test in the file: the enumerated `M_1` must agree with the
frequency the model's own sampler produces. It is the check that the search and the
sampler describe the same distribution — a normalization slip in either shows up here
and nowhere else.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import audit_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.eval.mode_audit import jet_strata, run_mode_audit, spearman
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.mode_audit import (
    SkeletonEnumeration,
    entropy_from_draws,
    enumerate_skeletons,
    skeleton_log_prob,
)
from h2p_rsd_junipr.models.base import build_model

# A model small enough to enumerate EXHAUSTIVELY: 2x2 = 4 cells, 4 emissions ->
# sum_n 4^n = 341 skeletons. Everything below compares against that full product.
TINY = ["geometry.n_bins=2", "encoder.emb_dim=8", "encoder.hidden_dim=8"]
# Width knobs are per-FAMILY (the schemas are polymorphic and reject unknown keys), so
# they are named at the call site rather than shared — `model.dec_dim` on an edit
# transducer is a config error, not a wider net.
DIMS_AR = ["model.ctx_dim=8", "model.dec_dim=8"]
DIMS_FACTORIZED = ["model.ctx_dim=8", "model.hidden_dim=8"]
MAX_EM = 4
# The skeleton marginal is a product of float32 softmaxes, so the accounting identity
# closes to float32 precision (~1e-7 per head), not float64. A defect above this is a
# bug in the search; below it is the head's own arithmetic.
MASS_TOL = 1e-6


def _tiny(extra=None, seed=0, dims=None):
    torch.manual_seed(seed)
    cfg = load_config(TINY + (DIMS_AR if dims is None else list(dims))
                      + [f"model.max_emissions={MAX_EM}"] + (extra or []))
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).eval()
    jets = synthetic_matched_dataset(8, seed=seed)
    ds = MatchedLundDataset(jets, geom)
    return cfg, geom, model, ds, jets


def _jet(ds, i=0):
    item = ds[i]
    return item["xf"].unsqueeze(0), torch.tensor([item["nx"]]), item


def _brute_force(model, xf, nx, n_cells, max_em=MAX_EM):
    """Every skeleton up to `max_em`, scored by the teacher-forced scorer, descending."""
    out = []
    for n in range(max_em + 1):
        for cells in itertools.product(range(n_cells), repeat=n):
            out.append((list(cells), skeleton_log_prob(model, list(cells), xf, nx)))
    out.sort(key=lambda kv: -kv[1])
    return out


# ---------------------------------------------------------------------------
# T1 — brute-force exactness (continue/stop family)
# ---------------------------------------------------------------------------
def test_t1_best_first_matches_brute_force_masses_and_order():
    _cfg, geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    ref = _brute_force(model, xf, nx, geom.n_cells)

    enum = enumerate_skeletons(model, xf, nx, k=16, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM)
    assert len(enum.skeletons) == 16
    for (cells, lm), (rc, rlm) in zip(enum.skeletons, ref[:16]):
        assert cells == rc, "best-first popped a different skeleton than the exact order"
        assert lm == pytest.approx(rlm, abs=1e-6)
    # descending by construction — the property the dominance statements rest on
    lms = [lm for _c, lm in enum.skeletons]
    assert all(a >= b - 1e-12 for a, b in zip(lms, lms[1:]))
    assert enum.certified and enum.certified_top1


def test_t1_total_mass_is_one():
    """With no depth cap in play the enumerated + frontier + pruned mass is exactly 1:
    the coordinate factors integrate out, so the skeleton marginal is a proper pmf."""
    _cfg, _geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    enum = enumerate_skeletons(model, xf, nx, k=8, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM)
    assert math.exp(enum.total_log_mass) == pytest.approx(1.0, abs=MASS_TOL)


# ---------------------------------------------------------------------------
# T2 — the certificate property, under every pruning setting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kw",
    [
        {"k": 4, "budget": 3},                                   # budget-limited
        {"k": 32, "budget": 100_000, "prune_rel": 1e-2},         # relative pruning
        {"k": 32, "budget": 100_000, "topk_children": 1},        # child cap
        {"k": 32, "budget": 100_000, "max_frontier": 3},         # frontier eviction
        {"k": 341, "budget": 100_000, "prune_rel": 0.0, "topk_children": 0},  # exhaustive
    ],
)
def test_t2_mass_accounting_and_monotonicity(kw):
    _cfg, _geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    enum = enumerate_skeletons(model, xf, nx, max_emissions=MAX_EM,
                               **{"prune_rel": 0.0, "topk_children": 0, **kw})
    assert math.exp(enum.total_log_mass) == pytest.approx(1.0, abs=MASS_TOL), (
        "completed + frontier + pruned must be 1 at EVERY termination point — that "
        "identity is what makes the coverage remainder a bound rather than a guess"
    )
    lms = [lm for _c, lm in enum.skeletons]
    assert all(a >= b - 1e-12 for a, b in zip(lms, lms[1:]))
    assert enum.coverage <= 1.0 + MASS_TOL
    assert 1.0 - enum.coverage <= enum.remainder_bound + MASS_TOL


def test_t2_certified_flag_is_conservative():
    """A budget too small to reach k completions is never certified, and a search that
    IS certified agrees with brute force — the flag has to mean something."""
    _cfg, geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    ref = _brute_force(model, xf, nx, geom.n_cells)

    starved = enumerate_skeletons(model, xf, nx, k=32, budget=2, prune_rel=0.0,
                                  topk_children=0, max_emissions=MAX_EM)
    assert not starved.certified

    ok = enumerate_skeletons(model, xf, nx, k=8, budget=100_000, prune_rel=1e-3,
                             topk_children=0, max_emissions=MAX_EM)
    if ok.certified:
        for (cells, _lm), (rc, _rlm) in zip(ok.skeletons, ref[:8]):
            assert cells == rc


def test_t2_dominant_top1_is_self_certifying():
    """`M_1 > 1/2` needs no certificate: the total mass is 1, so nothing else can be
    bigger. The audit quotes its dominance thresholds at 0.5 for exactly this reason."""
    _cfg, _geom, model, ds, _jets = _tiny()
    for i in range(len(ds)):
        xf, nx, _ = _jet(ds, i)
        enum = enumerate_skeletons(model, xf, nx, k=1, budget=100_000, prune_rel=1e-2,
                                   topk_children=1, max_emissions=MAX_EM)
        if enum.m1 > 0.5:
            assert enum.certified_top1


# ---------------------------------------------------------------------------
# T3 — scorer consistency against the trained likelihood
# ---------------------------------------------------------------------------
def test_t3_skeleton_log_prob_is_the_discrete_part_of_the_likelihood():
    """`skeleton_log_prob` walks `_step`; `nll_terms` teacher-forces `_decode_states`.
    Two code paths, one number: `length_ll + split_ll` IS the skeleton marginal, and
    the coordinate term is exactly what marginalises away."""
    _cfg, geom, model, ds, _jets = _tiny()
    batch = collate([ds[i] for i in range(6)])
    with torch.inference_mode():
        terms = model.nll_terms(batch)
    for i in range(6):
        xf = batch["xf"][i : i + 1]
        nx = batch["nx"][i : i + 1]
        ny = int(batch["ny"][i])
        cells = [int(c) for c in batch["yc"][i, :ny].tolist()]
        got = skeleton_log_prob(model, cells, xf, nx)
        want = float(terms["length_ll"][i] + terms["split_ll"][i])
        assert got == pytest.approx(want, abs=1e-5)


def test_t3_scorer_agrees_with_the_search():
    """Every enumerated mass must be reproducible by the scorer on its own cells —
    otherwise the top-k list and the truth's mass live on different scales and their
    comparison (question ii) is meaningless."""
    _cfg, _geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    enum = enumerate_skeletons(model, xf, nx, k=12, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM)
    for cells, lm in enum.skeletons:
        assert skeleton_log_prob(model, cells, xf, nx) == pytest.approx(lm, abs=1e-6)


# ---------------------------------------------------------------------------
# T4 — the sampled-frequency consistency check
# ---------------------------------------------------------------------------
def test_t4_top1_mass_matches_the_samplers_frequency():
    """The search and the sampler must describe the same distribution.

    The band is the normal binomial interval at `n = 4000` rather than an exact
    Clopper-Pearson one: scipy is not a runtime dependency here, and at these `n` and
    `p` (`np > 5` comfortably) the two agree to well inside the 4-sigma width used, which
    is chosen so a correct implementation does not fail once a year on the seed."""
    _cfg, _geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    enum = enumerate_skeletons(model, xf, nx, k=4, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM)
    top1 = enum.top1_cells
    torch.manual_seed(1234)
    n_draws = 4000
    draws = model.sample(xf, nx, n_draws, max_emissions=MAX_EM)
    freq = sum(1 for d in draws if [int(c) for c in d] == top1) / n_draws
    sigma = math.sqrt(max(enum.m1 * (1 - enum.m1), 1e-12) / n_draws)
    assert abs(freq - enum.m1) <= 4.0 * sigma + 1.0 / n_draws, (
        f"sampler frequency {freq:.4f} vs enumerated M_1 {enum.m1:.4f}"
    )


# ---------------------------------------------------------------------------
# T5 — the empty-skeleton identity
# ---------------------------------------------------------------------------
def test_t5_empty_skeleton_mass_is_the_models_q0():
    """The N=0 row is a first-class skeleton, and its mass must be the same `q(0|x)`
    the empty-tree analysis thresholds — reached here through `describe_cells`, which
    never touches the incremental step the search uses."""
    _cfg, _geom, model, ds, _jets = _tiny()
    for i in range(4):
        xf, nx, _ = _jet(ds, i)
        enum = enumerate_skeletons(model, xf, nx, k=64, budget=100_000, prune_rel=0.0,
                                   topk_children=0, max_emissions=MAX_EM)
        empty = [lm for cells, lm in enum.skeletons if not cells]
        assert empty, "the empty skeleton must appear in a k=64 enumeration of 341"
        with torch.inference_mode():
            q0 = float(model.describe_cells(xf, nx, []).logprob)
        assert math.exp(empty[0]) == pytest.approx(math.exp(q0), abs=1e-6)


# ---------------------------------------------------------------------------
# T6 — the n_head adapter, and the factorized one beside it
# ---------------------------------------------------------------------------
def test_t6_nhead_adapter_matches_brute_force():
    """The G8 winner lives in this family, so its adapter is not optional. The per-N
    fixed-length searches merge on one heap; the merge is exact for the same reason the
    variable-length search is."""
    _cfg, geom, model, ds, _jets = _tiny(["model=ar_junipr_v3"])
    assert model.use_multiplicity_head
    xf, nx, _ = _jet(ds)
    ref = _brute_force(model, xf, nx, geom.n_cells)

    enum = enumerate_skeletons(model, xf, nx, k=16, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM, eps_n=0.0)
    assert enum.kind == "nhead"
    for (cells, lm), (rc, rlm) in zip(enum.skeletons, ref[:16]):
        assert cells == rc
        assert lm == pytest.approx(rlm, abs=1e-6)
    # q(N|x) has support to model.max_emissions, so the mass beyond MAX_EM is pruned,
    # accounted, and the identity still closes.
    assert math.exp(enum.total_log_mass) == pytest.approx(1.0, abs=MASS_TOL)


def test_t6_factorized_adapter_matches_brute_force():
    _cfg, geom, model, ds, _jets = _tiny(["model=cinn"], dims=DIMS_FACTORIZED)
    xf, nx, _ = _jet(ds)
    ref = _brute_force(model, xf, nx, geom.n_cells)

    enum = enumerate_skeletons(model, xf, nx, k=16, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM, eps_n=0.0)
    assert enum.kind == "factorized"
    # This family's cells are conditionally independent given the jet, so a permutation
    # of the same cells is a DIFFERENT skeleton (order is part of S) with an IDENTICAL
    # mass. The mass sequence is therefore the exact claim; which member of a tied group
    # is popped first is not, and pinning it would pin an implementation detail of the
    # heap rather than the posterior.
    for (cells, lm), (_rc, rlm) in zip(enum.skeletons, ref[:16]):
        assert lm == pytest.approx(rlm, abs=1e-6)
        assert skeleton_log_prob(model, cells, xf, nx) == pytest.approx(lm, abs=1e-6)
    assert len({tuple(c) for c, _lm in enum.skeletons}) == len(enum.skeletons)


def test_family_without_an_adapter_raises_by_name():
    """Loud, not silent: a None here would let the audit publish an empty table for a
    family it never searched."""
    _cfg, _geom, model, ds, _jets = _tiny(["model=edit_v1"], dims=["model.ctx_dim=8"])
    xf, nx, _ = _jet(ds)
    with pytest.raises(NotImplementedError, match="skeleton"):
        model.skeleton_search_spec(xf, nx)


# ---------------------------------------------------------------------------
# The runner: the artifact's shape, and the two identities it reports
# ---------------------------------------------------------------------------
def test_runner_produces_the_preregistered_quantities():
    _cfg, geom, model, ds, jets = _tiny()
    aud = {**audit_params(), "k": 8, "budget": 2000, "thresholds": [0.3, 0.5, 0.7]}
    out = run_mode_audit(model, ds, jets, geom, torch.device("cpu"),
                         n_jets=6, K=16, audit=aud, verbose=False)

    assert out["n_jets"] == 6
    o = out["overall"]
    for key in ("M1_p50", "M1_p90", "certified", "F", "truth_is_top1", "truth_in_top3",
                "truth_in_top10", "truth_median_mass"):
        assert key in o
    assert set(o["F"]) == {"0.3", "0.5", "0.7"}
    for f in o["F"].values():
        lo, hi = f["wilson95"]
        # a proper interval, and it brackets the point estimate except at k = 0 or
        # k = n, where the Wilson score interval is deliberately shrunk away from the
        # boundary (that asymmetry is the reason it is used instead of the normal one)
        assert 0.0 <= lo <= hi <= 1.0
        if 0 < f["k"] < f["n"]:
            assert lo <= f["frac"] <= hi
    assert set(out["spearman_M1_vs"]["predicted_sign"]) == {
        "lnkt_lead", "d_boundary", "d_floor", "n_x"}
    # the validity checks are arithmetic and must be tight
    assert out["validity"]["max_abs_mass_defect"] < 1e-6
    assert out["validity"]["max_abs_q0_deviation"] < 1e-6
    r = out["records"][0]
    for key in ("M", "cells_top1", "M1", "M2", "C_k", "frontier", "pruned", "certified",
                "H_hat", "n_x", "lnkt_lead", "d_boundary", "d_floor", "M_truth",
                "rank_truth", "top1_is_truth"):
        assert key in r
    assert r["M_truth"] >= 0.0
    # a rank that IS reported must reproduce the truth's own mass
    for rec in out["records"]:
        if rec["rank_truth"] == 1:
            assert rec["M_truth"] == pytest.approx(rec["M1"], rel=1e-9)


def test_runner_dominance_and_correctness_stay_separate():
    """The two questions must be answerable independently: a jet can have a dominant
    skeleton that is not the truth, and the summary must not merge them."""
    _cfg, geom, model, ds, jets = _tiny()
    out = run_mode_audit(model, ds, jets, geom, torch.device("cpu"), n_jets=6, K=0,
                         audit={**audit_params(), "k": 8, "budget": 2000}, verbose=False)
    o = out["overall"]
    assert o["F"]["0.5"]["n"] == o["truth_is_top1"]["n"] == out["n_jets"]
    # ...and they are not the same number by construction
    assert o["F"]["0.5"] is not o["truth_is_top1"]


def test_entropy_estimate_and_its_lower_bound():
    """`H_hat` over draws is the estimator; `-sum_i M_i log M_i` over the enumerated set
    is its certified lower bound. The bound has to actually bound."""
    _cfg, _geom, model, ds, _jets = _tiny()
    xf, nx, _ = _jet(ds)
    enum = enumerate_skeletons(model, xf, nx, k=64, budget=100_000, prune_rel=0.0,
                               topk_children=0, max_emissions=MAX_EM)
    torch.manual_seed(7)
    draws = model.sample(xf, nx, 2000, max_emissions=MAX_EM)
    from h2p_rsd_junipr.inference.mode_audit import skeleton_log_probs

    est = entropy_from_draws(skeleton_log_probs(model, [list(d) for d in draws], xf, nx))
    assert est["n_draws"] == 2000
    assert est["eff_skeletons"] == pytest.approx(math.exp(est["H_hat"]), rel=1e-9)
    assert enum.entropy_lower_bound() <= est["H_hat"] + 0.15


def test_strata_are_read_off_the_hadron_side():
    """Every stratification axis has to be computable on data — i.e. from `x` alone."""
    _cfg, geom, _model, _ds, jets = _tiny()
    s = jet_strata(jets[0], geom, z_cut=0.1, beta=0.0, kt_floor=0.5)
    assert s["n_x"] == len(jets[0]["x"][0])
    assert np.isfinite(s["lnkt_lead"]) and np.isfinite(s["d_boundary"])
    assert s["region"] in ("wide_soft", "wide_hard", "narrow_soft", "narrow_hard")
    # an unknown grooming leaves the distance UNSET rather than inventing a boundary
    s2 = jet_strata(jets[0], geom, z_cut=float("nan"), beta=float("nan"),
                    kt_floor=float("nan"))
    assert math.isnan(s2["d_boundary"]) and math.isnan(s2["d_floor"])


def test_spearman_matches_a_known_case():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_empty_enumeration_has_defined_summaries():
    """A degenerate enumeration must not raise on the properties every summary reads."""
    e = SkeletonEnumeration()
    assert e.m1 == 0.0 and e.m2 == 0.0 and e.top1_cells == []
    assert e.rank_of([1, 2]) == 0
    assert e.coverage == 0.0
