"""Posterior-cluster diagnostics on held-out jets (docs/PLAN_PosteriorClusters.md WP5/WP6).

One measurement pass per jet, from **one** `K x K` distance matrix, producing everything
the pre-registered gates read:

  - **G2 (necessity, truth-free)** — the fraction of jets whose linear medoid lies in the
    dominant cluster. `>= 0.90` closes WP4 as unnecessary. Truth-free, so it transfers to
    real data.
  - **G2' (set value, truth-based)** — whether the *set* recovers what the point estimate
    misses, against a **mass-matched random-partition null**. G2 alone asks only whether
    the medoid is centrally placed: it can pass while the set is worthless, or fail while
    it is valuable, so the two are joint gates and may legitimately disagree.
  - **G3 (empty stratum)** — the N = 0 draws must form exactly one zero-radius cluster
    whose mass agrees with `length_pmf`'s q(0|x). Disagreement is a metric-convention bug,
    not a finding.
  - **G5 (budget stability)** — `top_mass` / `entropy` with their binomial errors, so a
    K = 250 run and a K = 1000 run can be compared without re-deriving the error.
  - **G6 (reliability)** — the reliability diagram of `top_mass` against the realized
    "truth in top cluster" frequency, with ECE and the Brier decomposition into
    reliability / resolution / uncertainty (Murphy, *J. Appl. Meteor.* **12** (1973) 595;
    Gneiting & Raftery, *JASA* **102** (2007) 359). The decomposition matters because
    "the numbers are miscalibrated" and "the numbers carry no information" are different
    failures with different fixes.
  - **G7 (conformal coverage)** — the split-conformal set threshold and its realized
    coverage. Marginal over jets, **not** conditional on x.
  - **G8 / G8'** — the WP4a loss-stability columns, from `eval/stability.py`, on the same
    `D`.

**G2' and G6 are one pass, deliberately.** Binning jets by `m_top` and asking how often
the truth landed in the top cluster *is* the reliability diagram; G2' asks whether the
truth is much closer to one exemplar than the others and G6 asks how often that one is the
highest-mass exemplar. Same nearest-exemplar assignment, same loop, implemented once.

**Scope discipline.** "The jet population is bimodal" and "p(y|x) is bimodal for this jet"
are different claims and only the second is in scope. A bimodal population marginal (quark-
vs gluon-initiated) yields unimodal conditionals wherever x separates them; a unimodal
marginal can have bimodal conditionals wherever the forward map folds. G2' is therefore
reported stratified by whether the top two clusters differ in N, so a split *between*
strata is distinguishable from a split *within* one.

**Why this is its own module rather than more of `run_closure`.** Every number here needs
the `K x K` matrix, and `run_closure`'s `map_or_mbr` builds and discards one per jet. A
second pass inside it would double the dominant cost of the whole suite (§14's budget
note); one pass that keeps `D` and reads the point estimate, the clusters and the
stability columns off it costs what the closure MBR already costs. It is gated by
`experiment.cluster_diagnostics` and lands at `metrics["clusters"]`, the same shape as
`support_audit` / `exposure` / `mode_audit`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..inference.clusters import (
    assign_truth,
    cluster_posterior,
    fit_set_threshold,
    pool_coverage_bound,
    pool_covered,
    random_partition_null,
    set_size_for,
    support_radii,
)
from ..inference.mbr import (
    _qn_importance_weights,
    _reduce_risk,
    lund_cloud,
    lund_emd_matrix,
    mbr_kwargs_from_decode,
    posterior_distances,
)
from .calibration import REGION_LABELS, cell_region, wilson_interval
from .closure import leading_emission_cell
from .stability import loss_stability_row, summarise_stability

# Below this many jets in a stratum the numbers are reported but NOT scored, per the
# standing v0/v1 convention. A coverage on 12 jets is a number, not a measurement.
MIN_STRATUM_N = 30
# Reliability bins for `top_mass`. Equal-width in the probability itself, so the diagram
# is read the way every reliability diagram is read.
N_RELIABILITY_BINS = 10


# ---------------------------------------------------------------------------
# Calibration statistics for a probability that is claimed, not fitted
# ---------------------------------------------------------------------------
def reliability(probs, hits, n_bins: int = N_RELIABILITY_BINS) -> dict:
    """Reliability diagram + ECE + the Brier decomposition for a claimed probability.

    `probs[i]` is jet `i`'s claimed `top_mass`; `hits[i]` is whether the truth actually
    landed in that cluster. Bins are equal-width in `p`; empty bins are dropped rather
    than imputed.

    The Brier score decomposes as `BS = REL - RES + UNC` (Murphy 1973), and the three
    terms answer different questions: `REL -> 0` says the claimed numbers match the
    realized frequencies, `RES` says they *vary* with the jet (a constant forecaster has
    RES = 0 and can still have perfect reliability), and `UNC` is the base rate's own
    variance, which no forecaster controls. Reporting only ECE cannot tell a calibrated
    but uninformative predictor from a useful one."""
    p = np.asarray(probs, dtype=float)
    o = np.asarray(hits, dtype=float)
    keep = np.isfinite(p) & np.isfinite(o)
    p, o = p[keep], o[keep]
    n = int(p.size)
    if n == 0:
        return {"n": 0, "ece": float("nan"), "brier": float("nan"), "bins": []}
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    obar = float(o.mean())
    bins, ece, rel, res = [], 0.0, 0.0, 0.0
    for b in range(int(n_bins)):
        sel = idx == b
        nb = int(sel.sum())
        if nb == 0:
            continue
        f, ob = float(p[sel].mean()), float(o[sel].mean())
        lo, hi = wilson_interval(int(o[sel].sum()), nb)
        bins.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": nb,
            "claimed": f, "observed": ob, "wilson95": [lo, hi],
            "scored": bool(nb >= MIN_STRATUM_N),
        })
        ece += nb / n * abs(f - ob)
        rel += nb / n * (f - ob) ** 2
        res += nb / n * (ob - obar) ** 2
    unc = obar * (1.0 - obar)
    # Weighted least squares of observed on claimed, weights = bin counts. A slope of 1 is
    # calibrated; < 1 is over-confident (the classic sign for a tree posterior), > 1
    # under-confident. Reported with its own standard error, because a slope estimated
    # from eight bins of a few dozen jets each is not a number to read to two decimals.
    slope = intercept = slope_se = float("nan")
    if len(bins) >= 2:
        f = np.array([b["claimed"] for b in bins])
        ob = np.array([b["observed"] for b in bins])
        wt = np.array([b["n"] for b in bins], dtype=float)
        fm, om = np.average(f, weights=wt), np.average(ob, weights=wt)
        sxx = float((wt * (f - fm) ** 2).sum())
        if sxx > 0:
            slope = float((wt * (f - fm) * (ob - om)).sum() / sxx)
            intercept = float(om - slope * fm)
            resid = ob - (intercept + slope * f)
            dof = max(len(bins) - 2, 1)
            slope_se = float(math.sqrt(max((wt * resid ** 2).sum() / dof / sxx, 0.0)))
    return {
        "n": n,
        "ece": float(ece),
        "brier": float(np.mean((p - o) ** 2)),
        "brier_reliability": float(rel),
        "brier_resolution": float(res),
        "brier_uncertainty": float(unc),
        "base_rate": obar,
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "slope_ci95": ([slope - 1.96 * slope_se, slope + 1.96 * slope_se]
                       if np.isfinite(slope_se) else [float("nan")] * 2),
        "bins": bins,
        "scored": bool(n >= MIN_STRATUM_N),
    }


def fit_mass_temperature(mass_vectors, hits, *, grid=None) -> dict:
    """One temperature on the cluster mass vector, fitted on validation and FROZEN.

    `m_j(T) = m_j^{1/T} / sum_k m_k^{1/T}` — the vector analogue of Guo, Pleiss, Sun &
    Weinberger (ICML 2017, arXiv:1706.04599), chosen by minimising the log loss of the
    binary "truth in top cluster" outcome against the tempered `top_mass`. `T = 1` is off
    and exactly reproduces the raw masses.

    Same discipline as `fit_length_recalibration`'s `(T, tilt)`: the value comes back with
    its provenance, and the caller freezes it before test. A temperature refitted on the
    set it is then scored on measures nothing."""
    grid = np.geomspace(0.25, 4.0, 61) if grid is None else np.asarray(grid, dtype=float)
    h = np.asarray(hits, dtype=float)
    vecs = [np.asarray(m, dtype=float) for m in mass_vectors]
    keep = [i for i, m in enumerate(vecs) if m.size and np.isfinite(h[i])]
    if not keep:
        return {"value": 1.0, "fitted_under": {"n": 0, "reason": "no usable jets"}}
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        tops = np.array([temper_top_mass(vecs[i], float(T)) for i in keep])
        q = np.clip(tops, 1e-9, 1 - 1e-9)
        nll = float(-(h[keep] * np.log(q) + (1 - h[keep]) * np.log(1 - q)).mean())
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return {
        "value": best_T,
        "fitted_under": {
            "n": len(keep),
            "grid": [float(grid[0]), float(grid[-1]), int(grid.size)],
            "objective": "log loss of truth-in-top-cluster vs tempered top_mass",
            "nll": best_nll,
            "note": "FREEZE before test; a temperature refitted on its own score set "
                    "measures nothing",
        },
    }


def temper_top_mass(masses, T: float) -> float:
    """`max_j m_j^{1/T} / sum_k m_k^{1/T}`; `T = 1` returns `masses.max()` unchanged."""
    m = np.asarray(masses, dtype=float)
    m = m[m > 0]
    if m.size == 0:
        return float("nan")
    if T == 1.0:
        return float(m.max() / m.sum()) if m.sum() > 0 else float("nan")
    p = m ** (1.0 / float(T))
    return float(p.max() / p.sum())


# ---------------------------------------------------------------------------
# The measurement pass
# ---------------------------------------------------------------------------
def _truth_cloud(item, geom, **cloud_kw):
    """The true parton tree as a Lund cloud, in the SAME coordinates as the draws.

    Built from `yraw` (the continuous truth), not from `yc`: placing the truth at cell
    centres would compare a quantised truth against unquantised draws, and the whole G2'
    effect is of order the inter-cluster separation, which is a few cells."""
    y = np.asarray(item["yraw"].numpy(), dtype=float)
    return lund_cloud([row for row in y], geom, **cloud_kw)


def run_cluster_diagnostics(model, val_ds, val_jets, geometry, device, *, K=200, n_jets=300,
                            decode=None, verbose=True, draws_by_jet=None, null_reps=20,
                            alpha=0.32, cluster_kwargs=None):
    """Gates G2, G2', G3, G5, G6, G7, G8 and G8' in one pass over held-out jets.

    `decode` is a `decode_params(cfg)` dict; the metric settings come from it and are
    checked for admissibility (gate G4) before the first jet, so a non-metric `D` raises
    once rather than producing a table of numbers nobody may use.

    `draws_by_jet` reuses posterior draws the caller already has — the same pattern as
    `run_closure(draws_by_jet=)` and `mbr_select(draws=)` — which makes this pass exactly
    PAIRED with the closure table rather than merely comparable to it."""
    from ..inference.clusters import assert_cluster_metric_ok

    dec = dict(decode or {})
    mk = mbr_kwargs_from_decode(dec)
    mk.pop("n_samples", None)
    mk.pop("resample_to_qn", None)
    n_cand = mk.pop("n_candidates", 0)
    assert_cluster_metric_ok({**dec, "mbr_n_candidates": n_cand}, geometry)
    resample = bool(dec.get("mbr_resample_to_qn", False))
    ck = dict(
        method=str(dec.get("cluster_method", "hdbscan")),
        min_mass=float(dec.get("cluster_min_mass", 0.05)),
        min_cluster_size=int(dec.get("cluster_min_cluster_size", 0)),
        eps_quantile=float(dec.get("cluster_eps_quantile", 0.10)),
        backend=str(dec.get("mbr_backend", "pot")),
    )
    ck.update(cluster_kwargs or {})
    split = bool(dec.get("cluster_split", False))
    cloud_kw = dict(lnkt_cut=mk["lnkt_cut"], weight=mk["weight"], coords=mk["coords"])

    n_jets = min(int(n_jets), len(val_ds))
    rows: list[dict] = []
    stab: list[dict] = []
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        draws = (model.sample_batch(xf, nx, K) if draws_by_jet is None else draws_by_jet[i])
        if not draws:
            continue
        _d, clouds, cand_idx, D = posterior_distances(
            model, xf, nx, draws=draws, geom=geometry, n_candidates=0, **mk)
        if not cand_idx:
            continue
        mults = np.array([len(d) for d in draws], dtype=int)
        w = _qn_importance_weights(model, xf, nx, draws) if resample else None
        # The point estimate, from the same D the clusters come from — so "the medoid is
        # in the dominant cluster" is a statement about ONE object, not two.
        win_lin = int(np.argmin(_reduce_risk(D, w, loss="linear")))

        split_index = None
        if split:
            split_index = np.zeros(len(draws), dtype=bool)
            split_index[::2] = True
        cs = cluster_posterior(D, weights=w, split_index=split_index, **ck)

        # --- the truth, and every draw's distance to it ------------------------------
        ny = int(item["ny"])
        tc = _truth_cloud(item, geometry, **cloud_kw)
        dt = lund_emd_matrix([tc], clouds, R=mk["R"], beta=mk["beta"], norm=mk["norm"],
                             periodic_phi=mk["periodic_phi"], phi_col=mk["phi_col"],
                             backend=ck["backend"], geom=geometry)[0]
        d_ex = np.array([dt[e] for e in cs.exemplars], dtype=float)
        j_truth = assign_truth(d_ex, support_radii(D, cs.labels, cs.exemplars))
        # WP3: the same question judged at the POOL's resolution rather than at each
        # cluster's internal tightness, so the answer does not move with the method.
        pool_bound = pool_coverage_bound(D)

        # --- G2, G2' and their controls ---------------------------------------------
        lab_med = int(cs.labels[win_lin]) if cs.labels.size else -1
        d_best = float(d_ex.min()) if d_ex.size else float("nan")
        null = random_partition_null(D, cs.masses, dt, n_reps=null_reps, seed=i, weights=w)
        n_top = len(draws[cs.exemplars[0]]) if cs.exemplars else -1
        n_second = len(draws[cs.exemplars[1]]) if len(cs.exemplars) > 1 else -1
        q0 = float(np.asarray(model.length_pmf(xf, nx, mults=mults.tolist()), dtype=float)[0])
        empty_mass = float(w[mults == 0].sum() / w.sum()) if w is not None else float(
            np.mean(mults == 0))
        lead = leading_emission_cell(item["yc"].tolist(), geometry)

        row = {
            "jet": int(i),
            "n_clusters": int(cs.n_clusters),
            "top_mass": float(cs.top_mass),
            "entropy": float(cs.entropy),
            "radius_top": float(cs.radii[0]) if cs.radii.size else float("nan"),
            "residual_mass": float(cs.residual_mass),
            "noise_mass": float(cs.noise_mass),
            "silhouette": float(cs.silhouette),
            "separation": float(cs.separation),
            "masses": cs.masses.tolist(),
            # G2 — truth-free, so this one transfers to real data
            "medoid_in_top": bool(lab_med == 0),
            "medoid_unassigned": bool(lab_med < 0),
            # G2' — oracle; diagnostic only, never a headline
            "d_top": float(d_ex[0]) if d_ex.size else float("nan"),
            "d_best": d_best,
            "d_mbr": float(dt[win_lin]),
            "d_best_rand": float(null["d_best_rand"]),
            "d_best_rand_sd": float(null["sd"]),
            # the separation-over-width precondition, computed BEFORE any truth is used
            "precondition": bool(np.isfinite(cs.separation) and cs.radii.size
                                 and cs.separation > float(np.max(cs.radii))),
            "n_top": int(n_top), "n_second": int(n_second),
            "strata_differ": bool(n_second >= 0 and n_top != n_second),
            # G6 — the reliability pair, from the same nearest-exemplar assignment
            "truth_in_top": bool(j_truth == 0),
            "truth_unassigned": bool(j_truth < 0),
            "truth_cluster": int(j_truth),
            "cum_mass_to_truth": (float(np.cumsum(cs.masses)[j_truth]) if j_truth >= 0
                                  else float("nan")),
            "pool_bound": float(pool_bound),
            "pool_covered_all": bool(pool_covered(dt, cs.labels,
                                                  range(cs.n_clusters), pool_bound)),
            "d_nearest_draw": float(dt.min()),
            # G3 — the empty stratum against the model's own q(0|x)
            "q0": q0,
            "empty_draw_mass": empty_mass,
            "ny_true": ny,
            "region": cell_region(lead, geometry),
            "ln_pt": float("nan"),
            "K": int(len(draws)),
        }
        try:  # ln_pt is a registered aux feature, but the jet file may predate the columns
            from ..features import AUX_FEATURES

            row["ln_pt"] = float(AUX_FEATURES["ln_pt"](val_jets[i]))
        except Exception:
            pass
        rows.append(row)
        stab.append(loss_stability_row(
            D, mults=mults, w=w, gamma=float(dec.get("cluster_eps_quantile", 0.10)),
            top_exemplar=(cs.exemplars[0] if cs.exemplars else None), d_to_truth=dt))

    metrics = summarise_clusters(rows, stab, alpha=alpha, verbose=verbose)
    metrics["config"] = {
        "K": int(K), "n_jets": int(n_jets), "method": ck["method"],
        "min_mass": ck["min_mass"], "min_cluster_size": ck["min_cluster_size"],
        "eps_quantile": ck["eps_quantile"], "backend": ck["backend"],
        "cluster_split": bool(split), "resample_to_qn": resample,
        "mbr_R": mk["R"], "mbr_beta": mk["beta"], "mbr_coords": mk["coords"],
    }
    metrics["per_jet"] = rows
    return metrics


# ---------------------------------------------------------------------------
# Aggregation — where the gates get their verdicts
# ---------------------------------------------------------------------------
def _mean(rows, key):
    v = [float(r[key]) for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(v)) if v else float("nan")


def _frac(rows, key):
    v = [bool(r[key]) for r in rows if isinstance(r.get(key), bool)]
    return float(np.mean(v)) if v else float("nan")


def _g2prime(rows) -> dict:
    """`<d_best_real>` against the mass-matched random-partition null, on the scored subset.

    **The signal is `d_best_real` vs `d_best_rand`, not `d_best` vs `d_mbr`.** Taking a
    minimum over n exemplars improves the distance even for a random partition, purely as
    an order statistic, so `d_best < d_mbr` is not evidence of anything. Restricted to jets
    with at least two clusters (there is no min-over-exemplars otherwise) and reported
    beside the silhouette precondition: where the precondition fails, the bimodality is
    unresolvable at this metric and budget whether or not it is real."""
    sel = [r for r in rows if r["n_clusters"] >= 2 and np.isfinite(r["d_best_rand"])]
    pre = [r for r in sel if r["precondition"]]
    def block(rs):
        if not rs:
            return {"n": 0, "d_best": float("nan"), "d_best_rand": float("nan"),
                    "gain": float("nan"), "gain_sem": float("nan"), "scored": False}
        db = np.array([r["d_best"] for r in rs], dtype=float)
        dr = np.array([r["d_best_rand"] for r in rs], dtype=float)
        gain = dr - db  # > 0 => the real partition beats the mass-matched null
        return {
            "n": len(rs),
            "d_best": float(np.nanmean(db)),
            "d_best_rand": float(np.nanmean(dr)),
            "d_top": float(np.nanmean([r["d_top"] for r in rs])),
            "d_mbr": float(np.nanmean([r["d_mbr"] for r in rs])),
            "gain": float(np.nanmean(gain)),
            "gain_sem": (float(np.nanstd(gain, ddof=1) / math.sqrt(len(rs)))
                         if len(rs) > 1 else float("nan")),
            "scored": bool(len(rs) >= MIN_STRATUM_N),
        }
    out = {"all": block(sel), "precondition_holds": block(pre),
           "precondition_rate": _frac(rows, "precondition")}
    # Stratified by whether the top two clusters differ in N — a split BETWEEN strata is a
    # different physical claim from a split WITHIN one (§10.1d).
    out["strata_differ"] = block([r for r in pre if r["strata_differ"]])
    out["strata_same"] = block([r for r in pre if not r["strata_differ"]])
    return out


def summarise_clusters(rows, stability_rows=None, *, alpha=0.32, verbose=True) -> dict:
    """Per-jet rows -> the gate table. Pure aggregation; no model, no EMD, no truth access
    beyond what the rows already carry."""
    n = len(rows)
    if n == 0:
        return {"n_jets": 0, "scored": False}
    tops = np.array([r["top_mass"] for r in rows], dtype=float)
    hits = np.array([r["truth_in_top"] for r in rows], dtype=float)
    assigned = [r for r in rows if not r["truth_unassigned"]]
    rel = reliability([r["top_mass"] for r in assigned],
                      [r["truth_in_top"] for r in assigned])
    temp = fit_mass_temperature([r["masses"] for r in assigned],
                                [r["truth_in_top"] for r in assigned])
    rel_T = reliability([temper_top_mass(r["masses"], temp["value"]) for r in assigned],
                        [r["truth_in_top"] for r in assigned])

    # --- G7: the conformal threshold and the coverage it buys ---------------------
    # `nan` for a jet whose truth no prefix covers -- kept, not dropped: `fit_set_threshold`
    # reads it as "never covered", which is its true rank. Dropping them would condition the
    # guarantee on assignment and report a coverage that cannot fail for the reason it most
    # needs to (see that function).
    scores = [r["cum_mass_to_truth"] for r in rows]
    conf = None
    if scores:
        conf = fit_set_threshold(scores, alpha=alpha)
        covered, sizes = [], []
        for r in rows:
            k = set_size_for(r["masses"], conf["value"])
            sizes.append(k)
            covered.append(bool(0 <= r["truth_cluster"] < k))
        lo, hi = wilson_interval(int(np.sum(covered)), len(covered))
        conf["coverage"] = float(np.mean(covered))
        conf["coverage_wilson95"] = [lo, hi]
        conf["nominal"] = float(1.0 - alpha)
        conf["mean_set_size"] = float(np.mean(sizes))
        conf["pass"] = bool(hi >= 1.0 - alpha)

    # --- G5: the Monte-Carlo error on a mass, so two K tiers are comparable -------
    Ks = np.array([r["K"] for r in rows], dtype=float)
    mc_err = float(np.mean(np.sqrt(np.clip(tops * (1 - tops), 0, None) / np.maximum(Ks, 1))))

    out = {
        "n_jets": int(n),
        "n_truth_assigned": int(len(assigned)),
        "unassigned_rate": _frac(rows, "truth_unassigned"),
        # WP3, reported BESIDE the exemplar rule and never instead of it. The exemplar
        # rule asks "is the truth inside the region this exemplar represents"; this asks
        # "did the pool put a draw near the truth at all", judged at the pool's own
        # resolution so it cannot move with the clustering method (35.7% under hdbscan vs
        # 8.2% under pam was the same jets, the same draws, a different partition).
        "pool_covered_rate": _frac(rows, "pool_covered_all"),
        "pool_bound_mean": _mean(rows, "pool_bound"),
        "d_nearest_draw_mean": _mean(rows, "d_nearest_draw"),
        "n_clusters_mean": _mean(rows, "n_clusters"),
        "frac_multimodal": float(np.mean([r["n_clusters"] >= 2 for r in rows])),
        "top_mass_mean": float(np.nanmean(tops)),
        "entropy_mean": _mean(rows, "entropy"),
        "radius_top_mean": _mean(rows, "radius_top"),
        "residual_mass_mean": _mean(rows, "residual_mass"),
        "silhouette_mean": _mean(rows, "silhouette"),
        "truth_in_top_rate": float(np.nanmean(hits)),
        # --- G2 ------------------------------------------------------------------
        "G2_medoid_in_top": _frac(rows, "medoid_in_top"),
        "G2_medoid_unassigned": _frac(rows, "medoid_unassigned"),
        "G2_pass_wp4_closed": None,
        # --- G2' -----------------------------------------------------------------
        "G2prime": _g2prime(rows),
        # --- G3: the N=0 stratum's mass against the model's own q(0|x). Any gap is a
        #     metric-convention bug (the empty-empty distance is 0 by construction, so the
        #     stratum IS a cluster), not a finding about the posterior.
        "G3_empty_mass_vs_q0": float(np.mean(
            [abs(r["empty_draw_mass"] - r["q0"]) for r in rows])),
        "G3_within_mc_error": float(np.mean(
            [abs(r["empty_draw_mass"] - r["q0"]) <= 3.0 * math.sqrt(
                max(r["q0"] * (1 - r["q0"]), 0.0) / max(r["K"], 1)) + 1e-12
             for r in rows])),
        # --- G5 ------------------------------------------------------------------
        "top_mass_mc_error": mc_err,
        # --- G6 ------------------------------------------------------------------
        "G6_reliability": rel,
        "G6_temperature": temp,
        "G6_reliability_recalibrated": rel_T,
        "G6_pass": None,
        # --- G7 ------------------------------------------------------------------
        "G7_conformal": conf,
    }
    out["G2_pass_wp4_closed"] = (bool(out["G2_medoid_in_top"] >= 0.90)
                                 if np.isfinite(out["G2_medoid_in_top"]) else None)
    out["G6_pass"] = (bool(rel_T["ece"] <= 0.05) if np.isfinite(rel_T.get("ece", np.nan))
                      else None)

    # --- stratification: ln p_T (must be FLAT) and the Lund quadrants -------------
    out["by_ln_pt"] = _stratify_entropy(rows)
    out["by_region"] = {
        r: _stratum(list(filter(lambda x: x["region"] == r, rows))) for r in REGION_LABELS
    }
    out["region_min_n"] = MIN_STRATUM_N

    if stability_rows:
        out["stability"] = summarise_stability(stability_rows, verbose=verbose)

    if verbose:
        _print_clusters(out)
    return out


def _stratum(rows) -> dict:
    if not rows:
        return {"n_jets": 0, "scored": False}
    lo, hi = wilson_interval(int(np.sum([r["truth_in_top"] for r in rows])), len(rows))
    return {
        "n_jets": len(rows),
        "top_mass": _mean(rows, "top_mass"),
        "entropy": _mean(rows, "entropy"),
        "n_clusters": _mean(rows, "n_clusters"),
        "truth_in_top": _frac(rows, "truth_in_top"),
        "truth_in_top_wilson95": [lo, hi],
        "medoid_in_top": _frac(rows, "medoid_in_top"),
        "scored": bool(len(rows) >= MIN_STRATUM_N),
    }


def _stratify_entropy(rows, n_bins: int = 4) -> dict:
    """Entropy binned in `ln p_T`. It must be FLAT.

    A cluster split that tracks jet scale indicates **incomplete conditioning** rather than
    physical ambiguity — the model is spreading over explanations it could have ruled out
    from x. `ln_pt` is already a registered aux feature, so this costs nothing beyond the
    binning. Flat => the splits are the discrete emission-count explanations the plan is
    after."""
    vals = np.array([r["ln_pt"] for r in rows], dtype=float)
    ok = np.isfinite(vals)
    if ok.sum() < n_bins:
        return {"available": False,
                "reason": "ln_pt is absent from this file (pre-aux-column jets.root)"}
    q = np.quantile(vals[ok], np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9
    out: dict = {"available": True, "edges": [float(x) for x in q], "bins": []}
    for b in range(n_bins):
        sel = [r for r, v, k in zip(rows, vals, ok)
               if k and q[b] < v <= q[b + 1]]
        e = _stratum(sel)
        e["ln_pt_lo"], e["ln_pt_hi"] = float(q[b]), float(q[b + 1])
        out["bins"].append(e)
    ent = [b["entropy"] for b in out["bins"] if b["n_jets"] >= MIN_STRATUM_N]
    out["entropy_spread"] = float(np.nanmax(ent) - np.nanmin(ent)) if len(ent) >= 2 else float("nan")
    out["flat"] = bool(out["entropy_spread"] <= 0.1) if np.isfinite(
        out["entropy_spread"]) else None
    return out


def _print_clusters(m: dict) -> None:
    print("\nposterior clusters (docs/PLAN_PosteriorClusters.md WP5/WP6):")
    print(f"  {m['n_jets']} jets   <n_clusters> = {m['n_clusters_mean']:.2f}"
          f"   multimodal on {m['frac_multimodal']:.1%} of jets"
          f"   <top_mass> = {m['top_mass_mean']:.3f}"
          f"   <H> = {m['entropy_mean']:.3f} nats"
          f"   <radius_top> = {m['radius_top_mean']:.3f}")
    print(f"  gate G2 (truth-free): the linear medoid lies in the dominant cluster on "
          f"{m['G2_medoid_in_top']:.3f} of jets"
          + ("   -> >= 0.90, so WP4 closes as unnecessary"
             if m["G2_pass_wp4_closed"] else "   -> < 0.90"))
    g = m["G2prime"]
    for label in ("all", "precondition_holds"):
        e = g[label]
        if not e["n"]:
            continue
        print(f"  gate G2' [{label}, n={e['n']}]: d_best = {e['d_best']:.3f} vs the "
              f"mass-matched random-partition null {e['d_best_rand']:.3f}"
              f"   gain = {e['gain']:+.3f} +/- {e['gain_sem']:.3f}"
              f"   (d_top = {e['d_top']:.3f}, d_mbr = {e['d_mbr']:.3f}; ORACLE — never a "
              f"headline)")
    print(f"      silhouette precondition holds on {g['precondition_rate']:.1%} of jets; "
          f"truth unassigned on {m['unassigned_rate']:.1%} (out of the pool's support)")
    print(f"  gate G3: |mass(N=0 draws) - q(0|x)| = {m['G3_empty_mass_vs_q0']:.4f} "
          f"(a metric-convention bug if this is not ~0)")
    r, rT = m["G6_reliability"], m["G6_reliability_recalibrated"]
    print(f"  gate G6: ECE = {r['ece']:.4f} raw -> {rT['ece']:.4f} at T = "
          f"{m['G6_temperature']['value']:.3f}   slope = {r['slope']:.2f} +/- "
          f"{r['slope_se']:.2f} (1.0 is calibrated; < 1 over-confident)")
    print(f"      Brier {r['brier']:.4f} = reliability {r['brier_reliability']:.4f} "
          f"- resolution {r['brier_resolution']:.4f} + uncertainty "
          f"{r['brier_uncertainty']:.4f}"
          f"   (resolution ~ 0 means the numbers carry no information, which is a "
          f"different failure from miscalibration)")
    c = m.get("G7_conformal")
    if c:
        print(f"  gate G7: conformal threshold {c['value']:.3f} at alpha = {c['alpha']:.2f}"
              f" -> coverage {c['coverage']:.3f} "
              f"[{c['coverage_wilson95'][0]:.3f}, {c['coverage_wilson95'][1]:.3f}]"
              f" vs nominal {c['nominal']:.3f}   mean set size {c['mean_set_size']:.2f}")
        print("      the guarantee is MARGINAL over jets, not conditional on x")
    p = m.get("by_ln_pt", {})
    if p.get("available"):
        # `flat is None` means NOT SCORED (no bin cleared MIN_STRATUM_N), which is a
        # different statement from "not flat" and must not print as a failure.
        if p.get("flat") is None:
            print(f"  entropy vs ln p_T: not scored — no quantile bin reached "
                  f"{MIN_STRATUM_N} jets, so the spread is undefined rather than large")
        else:
            print(f"  entropy vs ln p_T: spread {p['entropy_spread']:.3f} nats across "
                  f"{len(p['bins'])} quantile bins"
                  + ("   (flat — the splits are emission-count explanations, not scale)"
                     if p["flat"] else
                     "   (NOT flat — a split that tracks jet scale is incomplete "
                     "conditioning, not physical ambiguity)"))
