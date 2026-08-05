"""Loss-choice **stability**, reported beside the answer and never folded into it
(docs/PLAN_PosteriorClusters.md WP4a, §8.5, §8.6).

`inference.mbr._reduce_risk` supports three reductions over one distance matrix. This
module measures what changes when the reduction changes, at zero additional EMD cost: the
matrix is already built, and every column below is one array comparison on it.

**This is not `eval/systematics.py`, and the module boundary is the guard.**
`generator_spread` and the `frag_weights` variations vary something *unknown about
nature* — which fragmentation model is right — so their spread is a real uncertainty on a
fixed target. Loss choice varies something *the analyst decides*, and `linear` and
`bounded` are not two approximations to one quantity: they are **different functionals of
the same posterior**, the Frechet median and a density mode. Quoting their spread as an
uncertainty is quoting the mean-minus-median difference as a systematic on the mean. Both
are exactly right; they answer different questions.

It would also **double-count**: the uncertainty on the parton-level tree is the posterior
width, which `PosteriorSetEstimate.radii[0]` and the cluster masses already report.

The near-precedent that does not carry: unfolding analyses do quote a
regularization-strength systematic, varying the IBU iteration count (D'Agostini, *NIM A*
**362** (1995) 487). That varies a regularization *strength* on one estimand. Here the
estimand itself changes. The analogy fails at the point that matters.

What these columns are for instead:

  - **`argmin_moved` is a 1-bit multimodality flag that needs no clustering.** No
    scikit-learn, valid at small K, and — the reason it earns its place — **it works on
    real data**, where there is no truth and so the truth-based gate G2' is unavailable. It
    is a coarse proxy for `entropy`, degenerate with it wherever clusters exist and
    available where they do not.
  - **`empty_clique_size` vs `best_nonempty_count`** is gate G8', measured rather than
    predicted: `inference.mbr._empty_value` puts all empty draws at mutual distance exactly
    0, so at the measured ~17% empty rate and K = 500 that is ~85 draws forming a
    zero-diameter clique whose neighbour count is ~85 *for any* epsilon. A bounded loss can
    therefore collapse to the empty tree — reproducing exactly the MAP degeneracy the README
    credits MBR with removing structurally.
  - **`d_bounded` / `d_mbr` / `d_top`** answer gate G8 without shipping anything.

The configuration to avoid, stated explicitly: `bounded` as the *reported* point estimate
with no cluster layer. That is where both hazards bite and neither safeguard is present.
For a single tree, `linear` is the safe default and `predict_set().members[0]` the
principled one.
"""

from __future__ import annotations

import numpy as np

from ..inference.mbr import _reduce_risk, bandwidth_quantile

# The reductions measured beside the linear one. `kernel` rides along because it is the
# same array comparison and it separates "the top hat's edge moved it" from "the density
# really is elsewhere".
DIAGNOSTIC_LOSSES = ("bounded", "kernel")

# Every per-jet column this module emits. Named once, so `test_loss_spread_not_in_
# systematics` can assert none of them reaches the uncertainty budget by string rather
# than by a convention nobody re-reads.
STABILITY_COLUMNS = (
    "argmin_moved", "kernel_moved", "bounded_is_members0", "bounded_is_empty",
    "empty_clique_size", "best_nonempty_count", "empty_clique_wins", "eps_per_jet",
    "d_bounded", "d_mbr", "d_top", "n_draws",
)


def loss_stability_row(D, *, mults, w=None, gamma: float = 0.10, top_exemplar=None,
                       d_to_truth=None) -> dict:
    """The §8.5 columns for ONE jet, from that jet's already-built `D`.

    `mults` is the per-draw multiplicity (so the N = 0 stratum is identifiable without
    re-deriving it from the clouds); `top_exemplar` is `PosteriorSetEstimate`'s highest-mass
    exemplar index, or None when no cluster layer ran; `d_to_truth` is the per-draw distance
    to the true tree — `K` EMD solves against the `K^2` already spent, and the only input
    here that consults the truth.

    Epsilon is **per-jet**: `Q_gamma` of that jet's own positive off-diagonal distances.
    Within a jet the neighbour counts are compared at a common epsilon, which is all the
    argmin needs — this is a variable-bandwidth (nearest-neighbour) KDE in the sense of
    Loftsgaarden & Quesenberry, *Ann. Math. Statist.* **36** (1965) 1049, not an ad hoc
    choice, and it removes the `fitted_under` freeze machinery entirely since a per-jet
    statistic is not a fitted scalar. It is also exactly what a *productionised* bounded
    loss could not inherit: a per-jet bandwidth makes `.risk` comparable within a jet and
    not across jets, and the closure scripts aggregate across jets (§8.3)."""
    D = np.asarray(D, dtype=float)
    K = int(D.shape[1])
    m = np.asarray(mults, dtype=int)
    eps = bandwidth_quantile(D, gamma)
    row = {c: float("nan") for c in STABILITY_COLUMNS}
    row["n_draws"] = int(K)
    row["eps_per_jet"] = float(eps)
    if K == 0 or D.shape[0] == 0:
        return row

    risk_lin = _reduce_risk(D, w, loss="linear")
    win_lin = int(np.argmin(risk_lin))
    row["d_mbr"] = float(d_to_truth[win_lin]) if d_to_truth is not None else float("nan")

    if eps > 0:
        win_b = int(np.argmin(_reduce_risk(D, w, loss="bounded", eps=eps)))
        win_k = int(np.argmin(_reduce_risk(D, w, loss="kernel", eps=eps)))
        row["argmin_moved"] = bool(win_b != win_lin)
        row["kernel_moved"] = bool(win_k != win_lin)
        row["bounded_is_empty"] = bool(win_b < m.size and m[win_b] == 0)
        if top_exemplar is not None:
            row["bounded_is_members0"] = bool(win_b == int(top_exemplar))
        if d_to_truth is not None:
            row["d_bounded"] = float(d_to_truth[win_b])
        # --- gate G8': the empty clique against the best non-empty candidate ----------
        # Counted at the SAME epsilon the reduction uses. The clique is invisible to the
        # bandwidth rule (empty-empty pairs are exactly 0 and `bandwidth_quantile` takes
        # only positive entries) and decisive in the tally, which is the whole hazard.
        nb = (D <= eps).sum(axis=1)
        empty_rows = np.flatnonzero(m[: D.shape[0]] == 0)
        nonempty_rows = np.flatnonzero(m[: D.shape[0]] != 0)
        row["empty_clique_size"] = int(nb[empty_rows].max()) if empty_rows.size else 0
        row["best_nonempty_count"] = int(nb[nonempty_rows].max()) if nonempty_rows.size else 0
        row["empty_clique_wins"] = bool(row["empty_clique_size"] > row["best_nonempty_count"])
    if top_exemplar is not None and d_to_truth is not None:
        row["d_top"] = float(d_to_truth[int(top_exemplar)])
    return row


def _frac(rows, key) -> float:
    """Mean of a boolean column over the jets where it is defined (NaN if none are)."""
    vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean(rows, key) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")


def summarise_stability(rows, verbose: bool = False) -> dict:
    """Aggregate the per-jet rows into the table gates G8 and G8' are read off.

    Reports what was DROPPED as well as what was kept: a jet with no positive distance
    (every draw identical, or every draw empty) has no bandwidth and contributes to no
    column, and a silently smaller denominator reads as "covered everything"."""
    rows = [r for r in rows if r]
    n = len(rows)
    scored = [r for r in rows if isinstance(r.get("argmin_moved"), bool)]
    out = {
        "n_jets": int(n),
        "n_scored": int(len(scored)),
        "n_no_bandwidth": int(n - len(scored)),
        "argmin_moved": _frac(scored, "argmin_moved"),
        "kernel_moved": _frac(scored, "kernel_moved"),
        "bounded_is_members0": _frac(scored, "bounded_is_members0"),
        "bounded_is_empty": _frac(scored, "bounded_is_empty"),
        "empty_clique_wins": _frac(scored, "empty_clique_wins"),
        "empty_clique_size_mean": _mean(scored, "empty_clique_size"),
        "best_nonempty_count_mean": _mean(scored, "best_nonempty_count"),
        "eps_per_jet_mean": _mean(rows, "eps_per_jet"),
        "eps_per_jet_p16": float("nan"),
        "eps_per_jet_p84": float("nan"),
        "d_mbr": _mean(rows, "d_mbr"),
        "d_bounded": _mean(rows, "d_bounded"),
        "d_top": _mean(rows, "d_top"),
        # §8.6, recorded in the artifact itself so a later reader cannot mistake these for
        # an uncertainty. The key is verbose on purpose.
        "is_a_systematic": False,
        "note": ("linear vs bounded is a STABILITY check, not a systematic: they are "
                 "different functionals of one posterior (Frechet median vs density mode), "
                 "not two approximations to one quantity. Never add this spread in "
                 "quadrature — the posterior width is already reported by the cluster "
                 "radii. See docs/PLAN_PosteriorClusters.md §8.6."),
    }
    eps = np.asarray([r["eps_per_jet"] for r in rows
                      if np.isfinite(r.get("eps_per_jet", np.nan))], dtype=float)
    if eps.size:
        out["eps_per_jet_p16"] = float(np.percentile(eps, 16))
        out["eps_per_jet_p84"] = float(np.percentile(eps, 84))
    # The gate verdicts, spelled out rather than left to a reader with the plan open.
    out["G8prime_pass"] = (bool(out["empty_clique_wins"] <= 0.01)
                           if np.isfinite(out["empty_clique_wins"]) else None)
    if verbose:
        print("\nloss stability (docs/PLAN_PosteriorClusters.md §8.5) — a DIAGNOSTIC, "
              "never an uncertainty:")
        print(f"  jets scored = {out['n_scored']} of {out['n_jets']}"
              + (f"   ({out['n_no_bandwidth']} had no positive distance, so no bandwidth)"
                 if out["n_no_bandwidth"] else ""))
        print(f"  bounded argmin differs from linear : {out['argmin_moved']:.3f}"
              f"   (kernel {out['kernel_moved']:.3f})"
              f"   — near zero closes WP4b outright")
        print(f"  ...and when it moves, it lands on the top-mass exemplar : "
              f"{out['bounded_is_members0']:.3f}")
        print(f"  gate G8' — the N=0 clique wins the bounded argmin : "
              f"{out['empty_clique_wins']:.4f}"
              f"   (clique {out['empty_clique_size_mean']:.1f} vs best non-empty "
              f"{out['best_nonempty_count_mean']:.1f}; > 0.01 blocks a bounded ship)")
        print(f"  per-jet eps = {out['eps_per_jet_mean']:.3f} "
              f"[{out['eps_per_jet_p16']:.3f}, {out['eps_per_jet_p84']:.3f}]"
              f"   — a single FROZEN eps is viable only if this is narrow")
        if np.isfinite(out["d_mbr"]):
            print(f"  gate G8 — distance to truth: linear medoid = {out['d_mbr']:.3f}"
                  f"   bounded = {out['d_bounded']:.3f}   top-mass exemplar = "
                  f"{out['d_top']:.3f}   (the second comparison is the meaningful one)")
    return out
