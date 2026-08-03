"""The mode-mass audit (docs/PLAN_ModeMassAudit.md WP-3): per-jet, does the posterior
concentrate on a single parton SKELETON, and when it does, is that skeleton the true one?

Two questions, kept apart in every table here and in the write-up, because they are
logically independent:

1. **Does a dominant skeleton exist** — the `M_1` distribution and
   `F(m) = frac(M_1 >= m)` at the pre-registered thresholds.
2. **Is it correct** — the truth skeleton's rank and mass on the same jets.

A model can be sharply dominant and wrong, or diffuse and centred on the truth; nothing
below ever mixes the two into one number.

**Question 1 is resolution-relative and question 2 is not.** A cell's probability is
`~ density x area`, so `M_1` — and with it `F(m)` — scales with `n_bins`, which is a
discretization choice and not a physics one (`inference/mode_audit.py`'s module docstring
derives this). Every `F(m)` here is therefore a SAME-GEOMETRY, same-checkpoint comparison
and never a grid-free claim that a dominant parton skeleton exists. The `resolution`
block is the companion that IS grid-free: the Lund-plane AREA the first splitting's
posterior occupies (`node_hpd_area`), in physical units, beside the width the coordinate
head claims for itself. Read that block before quoting a mode mass — where the head is
truncation-saturated, a small `M_1` says the grid is finer than the model's own
resolution, not that the posterior is fragmented.

This is a DESCRIPTIVE audit. Nothing is adopted or rejected on these numbers, so there
are no pass/fail gates — but the quoted quantities and strata are pre-registered in the
plan's §7 before the run, and the two VALIDITY checks (mass accounting, and the
empty-skeleton identity against the model's own `q(0|x)`) are reported beside every
table. Those are arithmetic checks, not physics gates: a nonzero defect means the search
is wrong, not that the model is.

The stratification axes are the ones the non-perturbative smearing
`sigma = sigma_0 + Lambda_eff/k_t` predicts dominance to separate on (Dasgupta, Magnea &
Salam, arXiv:0712.3014), and all three are computed from `x` alone — an analysis can cut
on them on data:

* `lnkt_lead` — the hardest primary emission of the hadron-level jet.
* `d_boundary` — how far the closest emission sits above the soft-drop boundary,
  `min_t (ln z_t - ln z_cut + beta * ln(1/DeltaR_t))` (a straight line at `beta = 0`;
  Larkoski, Marzani, Soyez & Thaler, arXiv:1402.2657).
* `d_floor` — the same for the generator's `k_t` floor.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..features import node_raw
from ..inference.mode_audit import (
    entropy_from_draws,
    enumerate_skeletons,
    node_hpd_area,
    skeleton_log_prob,
    skeleton_log_probs,
)
from .calibration import REGION_LABELS, cell_region, wilson_interval
from .closure import leading_emission_cell
from .support import grooming_from_jets

# Per-jet masses kept in the artifact. The full list stays in the returned records for a
# notebook to use; a 2 000-jet run at k = 64 would otherwise write 128 000 floats whose
# tail nobody reads, and C_k already carries what the tail contributes.
STORE_TOP = 10


def spearman(a, b) -> float:
    """Spearman rank correlation, ties averaged — Pearson on the ranks.

    Hand-rolled rather than `scipy.stats.spearmanr` because scipy is not a runtime
    dependency of this package (the same reason `chi2_crit95` is hand-rolled in
    `eval/calibration.py`), and the rank transform is four lines."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")

    def _rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        # average the ranks inside each tie group, so a constant column gives a defined
        # (zero-variance -> nan) answer rather than an ordering artefact
        u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(u.size)
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    rx, ry = _rank(x), _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def jet_strata(jet, geometry, *, z_cut, beta, kt_floor) -> dict:
    """The three per-jet scalars the audit stratifies on, plus `n_x` and the Lund
    quadrant — all read off the HADRON side, so every one of them is available on data.

    An empty coordinate is NaN rather than a sentinel: a jet with no emission above the
    floor has no distance to it, and a filler zero would land in the middle of the
    distribution and be silently stratified as if it were measured."""
    a = np.asarray(node_raw(*jet["x"]), dtype=float).reshape(-1, 4)
    n_x = int(a.shape[0])
    out = {"n_x": n_x, "lnkt_lead": float("nan"), "d_boundary": float("nan"),
           "d_floor": float("nan"), "region": None}
    if n_x == 0:
        return out
    u, v, lnz = a[:, 0], a[:, 1], a[:, 2]
    out["lnkt_lead"] = float(v.max())
    if z_cut == z_cut and z_cut > 0.0:
        b = 0.0 if beta != beta else float(beta)
        out["d_boundary"] = float(np.min(lnz - (math.log(z_cut) - b * u)))
    if kt_floor == kt_floor and kt_floor > 0.0:
        out["d_floor"] = float(np.min(v - math.log(kt_floor)))
    cells = [int(c) for c in geometry.seq_cells(u, v)]
    out["region"] = cell_region(leading_emission_cell(cells, geometry), geometry)
    return out


def _pct(values, q) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return float(np.percentile(v, q)) if v.size else float("nan")


def _mean_of(records, key) -> float:
    v = np.array([r.get(key, float("nan")) for r in records], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _fraction(mask, label="") -> dict:
    """A proportion with its 95% Wilson interval (Brown, Cai & DasGupta, *Statist.
    Sci.* **16** (2001) 101) — the honest error bar on a few hundred jets, where the
    normal approximation is both too wide in the middle and nonsensical at the edges."""
    m = np.asarray(mask, dtype=bool)
    n, k = int(m.size), int(m.sum())
    lo, hi = wilson_interval(k, n)
    out = {"n": n, "k": k, "frac": (float(k) / n if n else float("nan")),
           "wilson95": [lo, hi]}
    if label:
        out["label"] = label
    return out


def _stratum_summary(rec_idx, records, thresholds) -> dict:
    """Everything §7.1–§7.2 asks for, on one subset of jets. Certified fractions travel
    WITH the numbers rather than in a separate table: an `F(0.5)` quoted without the
    certified rate beside it is not a certificate."""
    if not rec_idx:
        return {"n": 0}
    m1 = np.array([records[i]["M1"] for i in rec_idx], dtype=float)
    cert = np.array([records[i]["certified"] for i in rec_idx], dtype=bool)
    cert1 = np.array([records[i]["certified_top1"] for i in rec_idx], dtype=bool)
    top1_truth = np.array([records[i]["top1_is_truth"] for i in rec_idx], dtype=bool)
    rank = np.array([records[i]["rank_truth"] for i in rec_idx], dtype=int)
    m_truth = np.array([records[i]["M_truth"] for i in rec_idx], dtype=float)
    out = {
        "n": len(rec_idx),
        "M1_p50": float(np.median(m1)), "M1_p90": _pct(m1, 90.0),
        "M1_mean": float(m1.mean()),
        "certified": _fraction(cert), "certified_top1": _fraction(cert1),
        "F": {f"{t:g}": _fraction(m1 >= t) for t in thresholds},
        # question (ii), never merged into question (i) above
        "truth_is_top1": _fraction(top1_truth),
        "truth_in_top3": _fraction((rank >= 1) & (rank <= 3)),
        "truth_in_top10": _fraction((rank >= 1) & (rank <= 10)),
        "truth_median_mass": float(np.median(m_truth)) if m_truth.size else float("nan"),
        "truth_found_in_topk": _fraction(rank >= 1),
    }
    return out


@torch.inference_mode()
def audit_jet(model, item, jet, geometry, device, *, audit, draws=None, groom=None,
              index=0, store_top=STORE_TOP, hpd=True):
    """ONE jet's audit: `(record, SkeletonEnumeration)`.

    Split out of the runner so a notebook that already holds the jet's posterior draws
    (and wants the enumeration object itself, to plot the mass spectrum) produces the
    SAME record the CLI writes, rather than a second definition of the same numbers —
    the failure mode `docs/PLAN_prod_test_v0.md` §7 records for two closure populations.

    `draws` given -> the entropy estimate AND the length marginal `q(N|x)` reuse them at
    zero extra sampling cost; None leaves both unset rather than quietly starting a
    sampling pass the caller did not budget for.

    `hpd=True` adds the grid-free positional region (`node_hpd_area`) — the quantity that
    survives a change of `n_bins`, and the one to read before quoting `M_1`.
    """
    groom = grooming_from_jets([jet]) if groom is None else groom
    search_kw = {k: audit[k] for k in
                 ("k", "budget", "prune_rel", "topk_children", "max_frontier", "eps_n")
                 if k in audit}
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)
    spec = model.skeleton_search_spec(xf, nx)
    enum = enumerate_skeletons(model, xf, nx, spec=spec, **search_kw)

    ny = int(item["ny"])
    truth_cells = [int(c) for c in item["yc"].tolist()[:ny]]
    log_truth = skeleton_log_prob(model, truth_cells, xf, nx, spec=spec)
    rank = enum.rank_of(truth_cells)

    # VALIDITY, not physics: the empty skeleton's enumerated mass against the model's
    # own log q(N=0|x) through a different code path (the teacher-forced
    # `describe_cells`, which never touches the incremental step the search uses). A
    # mismatch is a bug in the search, and it would silently bias every fraction above.
    q0_enum = math.exp(skeleton_log_prob(model, [], xf, nx, spec=spec))
    q0_model = math.exp(float(model.describe_cells(xf, nx, []).logprob))

    ent = {"H_hat": float("nan"), "eff_skeletons": float("nan"), "n_draws": 0}
    q_n = [float("nan")] * 3
    if draws:
        ent = entropy_from_draws(
            skeleton_log_probs(model, [list(d) for d in draws], xf, nx, spec=spec)
        )
        # The LENGTH belief, kept beside the skeleton masses because it is the grid-free
        # half of the same factorization: q(S|x) = q(N|x) q(cells|N,x), and only the
        # second factor carries the cell-area scaling. "Is there a splitting at all" is
        # answerable without reference to any binning; "which cell" is not.
        pmf = np.asarray(model.length_pmf(xf, nx, mults=[len(d) for d in draws]),
                         dtype=float)
        q_n = [float(pmf[0]) if pmf.size else float("nan"),
               float(pmf[1]) if pmf.size > 1 else 0.0,
               float(pmf[2:].sum()) if pmf.size > 2 else 0.0]

    region = node_hpd_area(model, xf, nx, spec=spec) if hpd else None
    strata = jet_strata(jet, geometry, z_cut=groom["z_cut"], beta=groom["beta"],
                        kt_floor=groom["kt_floor"])
    masses = enum.masses
    record = {
        "i": int(index),
        "M": [float(m) for m in masses[:store_top]],
        "n_enumerated": len(masses),
        "cells_top1": enum.top1_cells,
        "M1": float(enum.m1), "M2": float(enum.m2),
        # the log mass RATIO, the scale-free "how far ahead is the mode" number. inf
        # when there is no runner-up inside the enumeration, which is a statement about
        # k, not about the posterior.
        "log_M1_M2": (float(enum.skeletons[0][1] - enum.skeletons[1][1])
                      if len(enum.skeletons) > 1 else float("inf")),
        "C_k": float(enum.coverage),
        "frontier": float(enum.frontier), "pruned": float(enum.pruned),
        "remainder_bound": float(enum.remainder_bound),
        "certified": bool(enum.certified), "certified_top1": bool(enum.certified_top1),
        "n_expansions": int(enum.n_expansions),
        "mass_defect": float(math.exp(enum.total_log_mass) - 1.0),
        "H_hat": ent["H_hat"], "eff_skeletons": ent["eff_skeletons"],
        "n_entropy_draws": int(ent["n_draws"]),
        "H_enumerated_lower": float(enum.entropy_lower_bound()),
        "M_truth": float(math.exp(log_truth)) if math.isfinite(log_truth) else 0.0,
        "rank_truth": int(rank),
        "top1_is_truth": bool(rank == 1),
        "n_truth": ny,
        "q0_enum": float(q0_enum), "q0_model": float(q0_model),
        # the grid-FREE half of q(S|x) = q(N|x) q(cells|N,x)
        "qN_0": q_n[0], "qN_1": q_n[1], "qN_ge2": q_n[2],
        "weight": float(jet.get("weight", 1.0)),
        "kind": spec.kind,
        **strata,
    }
    if region is not None:
        # The grid-free companion to M_1, flattened into the record so a stratified table
        # can read it the same way it reads everything else.
        record.update({
            "hpd_area_50": region["area"]["0.5"], "hpd_area_90": region["area"]["0.9"],
            "hpd_sqrt_50": region["sqrt_area"]["0.5"],
            "hpd_cells_50": region["n_cells_equivalent"]["0.5"],
            "hpd_over_sigma_box_50": region["area_over_sigma_box"]["0.5"],
            "sigma_u": region["sigma_u"], "sigma_v": region["sigma_v"],
            "truncation_saturated": region["truncation_saturated"],
            "modal_cell_mass": region["modal_cell_mass"],
            "hpd_quadrature": region["mass_quadrature"],
        })
    return record, enum


@torch.inference_mode()
def run_mode_audit(model, val_ds, val_jets, geometry, device, *, n_jets=300, K=0,
                   audit=None, draws_by_jet=None, store_top=STORE_TOP,
                   verbose=True) -> dict:
    """Enumerate each jet's top-k skeletons exactly and summarise §7's quantities.

    `K > 0` additionally scores posterior draws for the entropy estimate
    `H_hat = -mean_k log q(S^(k)|x)` (unbiased for `H(S|x)`; `e^H_hat` is the effective
    skeleton count). Pass `draws_by_jet` to reuse draws the caller already has — the
    audit never needs its own sampling pass.

    The search is sequential per jet by construction, so this runs on whatever device it
    is handed; CPU is the right one at `dec_dim` 64 (per-step MPS sync dominates), and
    the batched multi-jet variant is deferred with the sampling-speed work.
    """
    from ..config import audit_params

    audit = audit_params(None) if audit is None else dict(audit)
    thresholds = [float(t) for t in audit.get("thresholds", [0.3, 0.5, 0.7])]
    n_jets = min(int(n_jets), len(val_ds))
    groom = grooming_from_jets(val_jets)

    records: list[dict] = []
    kind = ""
    for i in range(n_jets):
        item = val_ds[i]
        draws = None
        if K:
            draws = (draws_by_jet[i] if draws_by_jet is not None else
                     model.sample(item["xf"].unsqueeze(0).to(device),
                                  torch.tensor([item["nx"]], device=device), int(K)))
        rec, _enum = audit_jet(model, item, val_jets[i], geometry, device, audit=audit,
                               draws=draws, groom=groom, index=i, store_top=store_top)
        kind = rec["kind"]
        records.append(rec)

    return summarise_mode_audit(records, thresholds=thresholds, audit=audit,
                                kind=kind, groom=groom, K=int(K), verbose=verbose)


def summarise_mode_audit(records, *, thresholds=(0.3, 0.5, 0.7), audit=None, kind="",
                         groom=None, K=0, verbose=True) -> dict:
    """The run-level block, split out so a notebook can re-summarise its own per-jet
    records (a different jet selection, a different threshold grid) without re-running
    the search — and get numbers computed by the same code that wrote the artifact."""
    thresholds = [float(t) for t in thresholds]
    n = len(records)
    if n == 0:
        return {"n_jets": 0, "records": []}
    idx_all = list(range(n))
    m1 = np.array([r["M1"] for r in records], dtype=float)
    lnkt = np.array([r["lnkt_lead"] for r in records], dtype=float)
    d_b = np.array([r["d_boundary"] for r in records], dtype=float)
    d_f = np.array([r["d_floor"] for r in records], dtype=float)
    n_x = np.array([r["n_x"] for r in records], dtype=float)

    # §7.4 — the mixture-population statement. The split is the SAMPLE MEDIAN of each
    # axis, so it is defined without a tuned constant; a column that is entirely NaN
    # (no z_cut / kt_floor in the file) drops out of the conjunction instead of
    # silently emptying the stratum.
    def _hi(v):
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return np.ones(v.size, dtype=bool), float("nan")
        med = float(np.median(finite))
        return (v >= med) | ~np.isfinite(v), med

    hi_kt, med_kt = _hi(lnkt)
    hi_b, med_b = _hi(d_b)
    hi_f, med_f = _hi(d_f)
    mix = hi_kt & hi_b & hi_f

    by_region: dict = {}
    for label in REGION_LABELS:
        sel = [i for i in idx_all if records[i].get("region") == label]
        if sel:
            by_region[label] = _stratum_summary(sel, records, thresholds)

    # The empty skeleton is a first-class row of the enumeration, and "the posterior is
    # sure there is NOTHING there" is a different physical claim from "the posterior is
    # sure WHICH splitting is there". Pooling the two into one F(m) credits a model for
    # the easy class; they are split here so a reader cannot avoid seeing both.
    # `top1_*` is an inference-time cut (an analysis could make it on data); `truth_*` is
    # a decomposition that uses the parton answer and is labelled as such.
    empty_top1 = [i for i in idx_all if not records[i]["cells_top1"]]
    empty_truth = [i for i in idx_all if not records[i]["n_truth"]]
    by_class = {
        "top1_is_empty": _stratum_summary(empty_top1, records, thresholds),
        "top1_is_a_splitting": _stratum_summary(
            [i for i in idx_all if i not in set(empty_top1)], records, thresholds),
        "truth_is_empty": _stratum_summary(empty_truth, records, thresholds),
        "truth_is_a_splitting": _stratum_summary(
            [i for i in idx_all if i not in set(empty_truth)], records, thresholds),
    }

    # The resolution block: what survives a change of n_bins. `M_1` does not, so nothing
    # in `overall` / `by_region` / `by_class` above is comparable across geometries and
    # these are the numbers that are.
    def _med(key):
        v = np.array([r.get(key, float("nan")) for r in records], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    sat = [bool(r.get("truncation_saturated", False)) for r in records
           if "truncation_saturated" in r]
    resolution = {
        "note": "the smallest Lund-plane region holding a fraction of the FIRST "
                "splitting's positional posterior, in ln(1/DeltaR) x ln(kt) units. "
                "Unlike M_1 this has a limit as n_bins refines, so it is comparable "
                "across geometries and families.",
        "hpd_area_50": _med("hpd_area_50"), "hpd_area_90": _med("hpd_area_90"),
        "hpd_sqrt_50": _med("hpd_sqrt_50"),
        "hpd_cells_50": _med("hpd_cells_50"),
        "hpd_over_sigma_box_50": _med("hpd_over_sigma_box_50"),
        "sigma_u": _med("sigma_u"), "sigma_v": _med("sigma_v"),
        "modal_cell_mass": _med("modal_cell_mass"),
        # A head whose sigma exceeds the half-cell on both axes cannot express its own
        # width within a cell, so it carries that width in the CELL distribution instead.
        # Where this is high, a small M_1 says the grid is finer than the model's
        # resolution -- it is not evidence that the posterior is fragmented.
        "frac_truncation_saturated": (float(np.mean(sat)) if sat else float("nan")),
        "q_N_mean": {"0": _mean_of(records, "qN_0"), "1": _mean_of(records, "qN_1"),
                     ">=2": _mean_of(records, "qN_ge2")},
    }

    ent = np.array([r["H_hat"] for r in records], dtype=float)
    eff = np.array([r["eff_skeletons"] for r in records], dtype=float)
    ent_lo = np.array([r["H_enumerated_lower"] for r in records], dtype=float)
    defect = np.array([abs(r["mass_defect"]) for r in records], dtype=float)
    q0_dev = np.array([abs(r["q0_enum"] - r["q0_model"]) for r in records], dtype=float)

    out = {
        "n_jets": n,
        "family_kind": str(kind),
        "search": dict(audit or {}),
        "grooming": dict(groom or {}),
        "K_entropy_draws": int(K),
        "overall": _stratum_summary(idx_all, records, thresholds),
        "geometry_dependence": "M_1 and every F(m) below scale with the cell area, so "
                               "they are same-geometry, same-checkpoint comparisons and "
                               "not grid-free claims. See `resolution`.",
        "resolution": resolution,
        "by_region": by_region,
        "by_class": by_class,
        # §7.3 — predicted signs +, +, +, -. Quoted as correlations, not as a fit: the
        # prediction is about ORDERING (more perturbative -> more dominant), and a
        # Spearman is the statement that survives the monotone reparametrisations
        # (ln kt vs kt) the physics does not fix.
        "spearman_M1_vs": {
            "lnkt_lead": spearman(m1, lnkt), "d_boundary": spearman(m1, d_b),
            "d_floor": spearman(m1, d_f), "n_x": spearman(m1, n_x),
            "predicted_sign": {"lnkt_lead": "+", "d_boundary": "+", "d_floor": "+",
                               "n_x": "-"},
        },
        "mixture": {
            "definition": "lnkt_lead, d_boundary and d_floor all at or above the "
                          "sample median (NaN columns do not constrain)",
            "medians": {"lnkt_lead": med_kt, "d_boundary": med_b, "d_floor": med_f},
            "stratum": _stratum_summary([i for i in idx_all if mix[i]], records, thresholds),
            "complement": _stratum_summary([i for i in idx_all if not mix[i]], records,
                                           thresholds),
        },
        "entropy": {
            "H_hat_p50": _pct(ent, 50.0), "H_hat_p90": _pct(ent, 90.0),
            "eff_skeletons_p50": _pct(eff, 50.0), "eff_skeletons_p90": _pct(eff, 90.0),
            "H_enumerated_lower_p50": _pct(ent_lo, 50.0),
            "note": "H_hat = -mean_k log q(S_k|x) over posterior draws (unbiased for "
                    "H(S|x)); H_enumerated_lower = -sum_i M_i log M_i over the "
                    "enumerated set, its certified lower bound",
        },
        # Arithmetic, not physics. Both must be ~0 for anything above to mean anything.
        "validity": {
            "max_abs_mass_defect": float(defect.max()) if defect.size else 0.0,
            "max_abs_q0_deviation": float(q0_dev.max()) if q0_dev.size else 0.0,
            "note": "mass_defect = (sum_i M_i + frontier + pruned) - 1 per jet; "
                    "q0 deviation = enumerated M(N=0) vs the model's own q(N=0|x) "
                    "through describe_cells",
        },
        "records": records,
    }
    if verbose:
        _print_summary(out)
    return out


def _print_resolution(out) -> None:
    """The grid-free block. Printed between the two questions rather than at the end,
    because it is what tells the reader how to read the F(m) line above it."""
    r = out.get("resolution") or {}
    if not np.isfinite(r.get("hpd_area_50", float("nan"))):
        return
    q = r.get("q_N_mean", {})
    print("    grid-FREE companion (first splitting's position, medians):")
    print(f"      50% region {r['hpd_area_50']:.3f} ln^2  = {r['hpd_sqrt_50']:.2f} ln "
          f"across  = {r['hpd_cells_50']:.0f} cells   90% region "
          f"{r['hpd_area_90']:.3f} ln^2")
    print(f"      head's own width sigma_u {r['sigma_u']:.3f} sigma_v {r['sigma_v']:.3f}"
          f"  ->  the region is {r['hpd_over_sigma_box_50']:.1f}x its +-1sigma box")
    if np.isfinite(r.get("frac_truncation_saturated", float("nan"))):
        sat = r["frac_truncation_saturated"]
        print(f"      truncation-saturated (sigma > half-cell on both axes) for "
              f"{sat:.0%} of jets" + ("   <- the grid is FINER than the model's own "
                                      "resolution, so a small M_1 above is a statement "
                                      "about n_bins" if sat > 0.5 else ""))
    if np.isfinite(q.get("0", float("nan"))):
        print(f"      length belief (grid-free): q(N=0) {q['0']:.3f}   q(N=1) "
              f"{q['1']:.3f}   q(N>=2) {q['>=2']:.3f}")


def _print_summary(out) -> None:
    o = out["overall"]
    print(f"\nmode-mass audit ({out['n_jets']} jets, family kind {out['family_kind']!r}, "
          f"k = {out['search'].get('k')}, budget = {out['search'].get('budget')}):")
    print(f"    M_1: median {o['M1_p50']:.3f}   p90 {o['M1_p90']:.3f}   "
          f"mean {o['M1_mean']:.3f}")
    print(f"    certified top-k {o['certified']['frac']:.1%}   "
          f"certified top-1 {o['certified_top1']['frac']:.1%}")
    for t, f in o["F"].items():
        lo, hi = f["wilson95"]
        print(f"    F({t}) = frac(M_1 >= {t}) = {f['frac']:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]   ({f['k']}/{f['n']} jets)")
    print("    ^ SAME-GEOMETRY numbers: a cell's mass is ~ density x area, so these "
          "scale with n_bins.")
    _print_resolution(out)
    print("    dominance and correctness are INDEPENDENT — the next block is the "
          "second question:")
    print(f"    truth = top-1 {o['truth_is_top1']['frac']:.3f}   in top-3 "
          f"{o['truth_in_top3']['frac']:.3f}   in top-10 {o['truth_in_top10']['frac']:.3f}"
          f"   median M_truth {o['truth_median_mass']:.4f}")
    # ...and the same two numbers split by the empty class, which the pooled ones mix.
    c = out["by_class"]
    for name, label in (("top1_is_empty", "mode is the EMPTY skeleton"),
                        ("top1_is_a_splitting", "mode is a splitting")):
        b = c[name]
        if b.get("n"):
            print(f"      {label:<28} {b['n']:>6} jets   median M_1 {b['M1_p50']:.3f}   "
                  f"F(0.5) {b['F'].get('0.5', {}).get('frac', float('nan')):.3f}   "
                  f"truth = top-1 {b['truth_is_top1']['frac']:.3f}")
    e = out["entropy"]
    print(f"    H_hat median {e['H_hat_p50']:.3f} nat  ->  "
          f"{e['eff_skeletons_p50']:.2f} effective skeletons")
    sp = out["spearman_M1_vs"]
    print("    Spearman(M_1, .): " + "   ".join(
        f"{k} {sp[k]:+.3f} (pred {sp['predicted_sign'][k]})"
        for k in ("lnkt_lead", "d_boundary", "d_floor", "n_x")))
    mx = out["mixture"]
    if mx["stratum"].get("n") and mx["complement"].get("n"):
        f_s = mx["stratum"]["F"].get("0.5", {}).get("frac", float("nan"))
        f_c = mx["complement"]["F"].get("0.5", {}).get("frac", float("nan"))
        print(f"    mixture: F(0.5) = {f_s:.3f} on the perturbative stratum "
              f"({mx['stratum']['n']} jets) vs {f_c:.3f} on its complement "
              f"({mx['complement']['n']} jets)")
    v = out["validity"]
    print(f"    validity (arithmetic, not a gate): max |mass defect| "
          f"{v['max_abs_mass_defect']:.2e}   max |q0 deviation| "
          f"{v['max_abs_q0_deviation']:.2e}")
