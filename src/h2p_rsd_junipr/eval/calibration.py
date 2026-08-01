"""Posterior calibration (§8): coverage, PIT, simulation-based calibration (SBC;
Talts et al., arXiv:1804.06788), and — WP2 of docs/PLAN_UPDATES.md — per-coordinate
PITs, region-stratified metrics, and TARP expected coverage (Lemos et al.,
arXiv:2302.03026).

Conditional-generator posteriors are not automatically calibrated (the original
cINN unfolding came out too narrow, arXiv:2006.06685), so this gates
"trustworthy".

**Why the suite grew.** The v1 statistic is the SBC rank of the *multiplicity* n:
for a calibrated posterior the rank of the true n among posterior draws is uniform
on {0..K}. That is a real test for the implicit continue/stop length model
(`ar_junipr_v2`) — but `ar_junipr_v3` trains `q(N|x)` by direct NLL on N, so
SBC-on-N certifies *the very marginal the model optimizes*, near-tautologically. A
v2-vs-v3 A/B judged on SBC-N is therefore biased toward v3 by construction. The
three additions below test what SBC-N cannot:

1. `coordinate_pits` — the *kinematics*, coordinate by coordinate, via each family's
   exact conditional CDFs (`PosteriorModel.coordinate_cdfs`). Miscalibrated widths
   show up as U-shaped (over-confident) or dome-shaped (over-dispersed) histograms.
2. Region stratification — every metric additionally binned by the Lund-plane
   quadrant of the leading emission, so a calibration that only holds *on average*
   over the plane cannot pass. This is the precondition for any localized
   (e.g. heavy-ion) claim.
3. `run_tarp` — expected coverage of the *whole tree* under the perturbative-Lund
   EMD (`inference.mbr`), i.e. a joint test in the metric the physics cares about,
   not a marginal one.

Everything new is **off by default**: with the switches off `run_calibration`
returns exactly the v1 metric dict, so CI numbers and published tables are stable.
"""

from __future__ import annotations

import numpy as np
import torch

from .closure import leading_emission_cell

# Lund-plane quadrants of the LEADING emission, used to stratify every metric.
# u = ln 1/DeltaR (low => wide angle, high => narrow/collinear),
# v = ln kt       (low => soft,       high => hard).
REGION_LABELS = ("wide_soft", "wide_hard", "narrow_soft", "narrow_hard")


def wilson_interval(k, n, z=1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — the honest error bar on
    `coverage_68` and every per-region coverage beside it.

    These are proportions on a few hundred jets (a few dozen per region), where the
    normal approximation `p +- z*sqrt(p(1-p)/n)` is both too wide in the middle and
    nonsensical at the edges (it can leave [0, 1], and it collapses to zero width at
    p = 0 or 1 — exactly where a near-empty Lund quadrant lands). Wilson does neither.
    Returns `(nan, nan)` for n = 0."""
    n = int(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(k) / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def chi2_crit95(dof: int) -> float:
    """95% point of chi^2(dof), via Wilson-Hilferty — no scipy dependency.

    `sbc_chi2_uniform` is quoted with no scale, which makes "2.4" and "24" read the
    same. The reference is chi^2(n_bins - 1): at the default 10 bins that is 16.92,
    and the approximation below returns 16.90 (0.1% low, and it improves with dof).
    Below 5 dof it degrades, so it is documented as indicative there, not exact."""
    if dof <= 0:
        return float("nan")
    z95 = 1.6448536269514722
    return float(dof * (1.0 - 2.0 / (9.0 * dof) + z95 * np.sqrt(2.0 / (9.0 * dof))) ** 3)


def cell_region(cell, geometry) -> str | None:
    """Lund-plane quadrant of a cell id, as a stable string label (or None)."""
    if cell is None:
        return None
    u, v = geometry.cell_center(int(cell))
    lo_u, hi_u = geometry.ln_invdelta_range
    lo_v, hi_v = geometry.ln_kt_range
    narrow = u >= 0.5 * (lo_u + hi_u)
    hard = v >= 0.5 * (lo_v + hi_v)
    return f"{'narrow' if narrow else 'wide'}_{'hard' if hard else 'soft'}"


# ---------------------------------------------------------------------------
# Uniformity statistics
# ---------------------------------------------------------------------------
def _ks_uniform(values) -> float:
    """One-sample Kolmogorov-Smirnov distance of `values` to Uniform(0,1).

    The headline per-coordinate number: 0 is perfect, and the 95% critical value is
    ~1.36/sqrt(n), so it is directly readable as "how far from calibrated"."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - v, v - (i - 1) / n)))


def _chi2_uniform(values, n_bins: int) -> float:
    hist, _ = np.histogram(np.asarray(values, dtype=float), bins=n_bins, range=(0.0, 1.0))
    expected = len(values) / n_bins if len(values) else 1.0
    return float(np.sum((hist - expected) ** 2 / max(expected, 1e-8)))


def _uniformity_report(values, n_bins: int) -> dict:
    v = np.asarray(values, dtype=float)
    hist, edges = np.histogram(v, bins=n_bins, range=(0.0, 1.0))
    return {
        "n": int(v.size),
        "ks": _ks_uniform(v),
        "chi2": _chi2_uniform(v, n_bins),
        "mean": float(v.mean()) if v.size else float("nan"),
        "hist": [int(c) for c in hist],
        "edges": [float(x) for x in edges],
    }


# ---------------------------------------------------------------------------
# WP2.1 — per-coordinate PITs
# ---------------------------------------------------------------------------
def coordinate_pits(model, val_ds, geometry, device, n_jets=300, n_bins=10,
                    stratify_regions=False, max_index=6, verbose=True) -> dict | None:
    """Per-coordinate probability-integral transforms, teacher-forced on the truth.

    Asks the model for `coordinate_cdfs(batch)` (exact closed-form CDFs for the AR
    coordinate head; base-space CDFs for the flow families; None for `diffusion`,
    which has no density) and turns each coordinate's transformed values into a rank
    histogram plus its KS distance to Uniform(0,1).

    Two breakdowns come for free and are the point of the test:
    `by_emission_index` (is the *late* emission calibrated, or only the first? — the
    exposure-bias signature) and, with `stratify_regions=True`, `by_region`.

    Returns None when the family cannot provide a transform, so callers degrade
    gracefully rather than branching on the family."""
    n_jets = min(int(n_jets), len(val_ds))
    if n_jets == 0:
        return None
    from ..data.dataset import collate

    batch = collate([val_ds[i] for i in range(n_jets)])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model.coordinate_cdfs(batch)
    if out is None:
        return None

    u = out["u"].detach().cpu().numpy()          # (B, L, D)
    mask = out["mask"].detach().cpu().numpy()    # (B, L)
    names = list(out["names"])
    B, L = mask.shape
    idx = np.broadcast_to(np.arange(L)[None, :], (B, L))

    regions = None
    if stratify_regions:
        regions = np.array(
            [str(cell_region(leading_emission_cell(val_ds[i]["yc"].tolist(), geometry), geometry))
             for i in range(n_jets)]
        )
        regions = np.broadcast_to(regions[:, None], (B, L))

    report: dict = {"space": out.get("space", "physical"), "names": names,
                    "n_jets": int(n_jets), "coords": {}}
    for d, name in enumerate(names):
        vals = u[..., d][mask]
        entry = _uniformity_report(vals, n_bins)
        entry["by_emission_index"] = {
            str(t): _uniformity_report(u[..., d][mask & (idx == t)], n_bins)
            for t in range(min(L, int(max_index)))
            if bool((mask & (idx == t)).any())
        }
        if regions is not None:
            entry["by_region"] = {
                r: _uniformity_report(u[..., d][mask & (regions == r)], n_bins)
                for r in REGION_LABELS
                if bool((mask & (regions == r)).any())
            }
        report["coords"][name] = entry
    report["ks_max"] = float(
        np.nanmax([report["coords"][n]["ks"] for n in names]) if names else float("nan")
    )

    if verbose:
        space = report["space"]
        print(f"\nper-coordinate PIT ({space} space, {n_jets} jets, "
              f"{int(mask.sum())} emissions):")
        print(f"    {'coord':>6} {'n':>7} {'KS':>8} {'mean':>7}   (KS -> 0, mean -> 0.5)")
        for name in names:
            e = report["coords"][name]
            print(f"    {name:>6} {e['n']:>7} {e['ks']:>8.4f} {e['mean']:>7.3f}")
        crit = 1.36 / max(np.sqrt(max(int(mask.sum()), 1)), 1e-9)
        print(f"    KS 95% critical value at this sample size = {crit:.4f}"
              f"   (KS above it => significant miscalibration)")
    return report


# ---------------------------------------------------------------------------
# WP2.3 — TARP expected coverage on tree-valued posteriors
# ---------------------------------------------------------------------------
def _tarp_reference_pool(model, val_ds, geometry, device, n_jets, kind, n_refs, rng):
    """The TARP reference distribution, as a list of cell chains.

    `"pooled"` draws references from the pooled *posterior* draws of the evaluation
    jets (support-covering by construction: wherever the posterior puts mass, a
    reference can appear). `"prior"` uses the held-out *truth* trees instead — the
    empirical prior over y. TARP's guarantee holds for any reference distribution
    with support covering the posterior's; the two choices differ in variance and in
    which failure they are most sensitive to, so the choice is reported."""
    if kind == "prior":
        pool = [val_ds[i]["yc"].tolist() for i in range(len(val_ds))]
    else:
        pool = []
        per_jet = max(1, int(np.ceil(n_refs / max(n_jets, 1))))
        for i in range(n_jets):
            item = val_ds[i]
            xf = item["xf"].unsqueeze(0).to(device)
            nx = torch.tensor([item["nx"]], device=device)
            pool.extend(model.sample_batch(xf, nx, per_jet))
    if len(pool) > n_refs:
        sel = rng.choice(len(pool), size=n_refs, replace=False)
        pool = [pool[int(s)] for s in sel]
    return pool


def run_tarp(model, val_ds, geometry, device, K=200, n_jets=300, n_refs=100,
             reference="pooled", n_alpha=21, mbr_kwargs=None, seed=0,
             verbose=True) -> dict:
    """TARP expected coverage (Lemos et al., arXiv:2302.03026) on TREE-valued posteriors.

    For each jet, draw a reference tree `r` from the reference distribution and
    compute the credibility level

        f = (1/K) * [ #{k : d(r, y_k) < d(r, y_true)}
                      + 0.5 * #{k : d(r, y_k) == d(r, y_true)} ],

    with `d` the perturbative-Lund EMD between the trees' weighted Lund clouds
    (`inference.mbr.lund_emd`, the *same* distance the MBR point estimator minimizes,
    with the decode-configured `mbr_*` metric). The half-weight on exact ties is the
    mid-rank convention the SBC statistic above already uses — without it the discrete
    cell chains tie often enough to push every `f` down and fake over-dispersion.

    Under a calibrated posterior each `f` is Uniform(0,1), so the expected-coverage
    probability `ECP(alpha) = P(f < alpha)` equals `alpha`.
    `tarp_max_dev = max_alpha |ECP(alpha) - alpha|` is the headline
    number; the *sign* diagnoses the failure — ECP below the diagonal means the
    posterior is **over-confident** (too narrow), above means over-dispersed.

    This is a JOINT test over the whole tree in a physics metric — the thing neither
    SBC-on-N nor the per-coordinate marginals can see.

    Cost is one distance matrix row per jet: `n_jets * (K + 1)` EMD solves."""
    from ..inference.mbr import lund_cloud, lund_emd_matrix

    mk = dict(mbr_kwargs or {})
    cloud_kw = {k: mk[k] for k in ("lnkt_cut", "weight", "coords") if k in mk}
    emd_kw = {k: mk[k] for k in ("R", "beta", "norm", "periodic_phi", "phi_col", "backend")
              if k in mk}
    rng = np.random.default_rng(seed)
    n_jets = min(int(n_jets), len(val_ds))
    pool = _tarp_reference_pool(model, val_ds, geometry, device, n_jets,
                                str(reference), int(n_refs), rng)
    if not pool:
        return {"tarp_max_dev": float("nan"), "n_jets": 0, "reference": str(reference)}

    f_levels = []
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        draws = model.sample_batch(xf, nx, K)
        ref = pool[int(rng.integers(len(pool)))]
        truth = item["yc"].tolist()
        clouds = [lund_cloud(d, geometry, **cloud_kw) for d in [*draws, truth]]
        D = lund_emd_matrix([lund_cloud(ref, geometry, **cloud_kw)], clouds,
                            geom=geometry, **emd_kw)[0]
        d_true, d_draws = D[-1], D[:-1]
        f_levels.append(
            float(np.mean(d_draws < d_true) + 0.5 * np.mean(d_draws == d_true))
        )

    f = np.asarray(f_levels, dtype=float)
    alpha = np.linspace(0.0, 1.0, int(n_alpha))
    ecp = np.array([float(np.mean(f < a)) for a in alpha])
    max_dev = float(np.max(np.abs(ecp - alpha)))
    signed = float(np.mean(ecp - alpha))
    # ECP read off at the standard credibility levels — the physically quotable form
    # ("at 90% credibility the posterior actually covered X%"). The endpoints alpha=0/1
    # are pinned by construction, so the *interior* levels carry the diagnosis.
    ecp_at = {f"{a:.2f}": float(np.interp(a, alpha, ecp)) for a in (0.5, 0.68, 0.9, 0.95)}
    # `tarp_max_dev` is a sup-norm deviation of an empirical CDF from the diagonal —
    # i.e. a KS statistic, whose 95% null value is ~1.36/sqrt(n). At 300 jets that is
    # 0.078, so a "max dev = 0.06" is a PASS, not a small failure. Quoted, because the
    # bare number invites reading any nonzero deviation as a defect.
    null_floor = 1.36 / np.sqrt(max(n_jets, 1))
    metrics = {
        "alpha": [float(a) for a in alpha],
        "ecp": [float(e) for e in ecp],
        "ecp_at": ecp_at,
        "tarp_max_dev": max_dev,
        "tarp_null_floor95": float(null_floor),
        "tarp_exceeds_null": bool(max_dev > null_floor),
        "tarp_signed_bias": signed,
        "n_jets": int(n_jets),
        "n_refs": int(len(pool)),
        "reference": str(reference),
        "backend": str(emd_kw.get("backend", "pot")),
    }
    if verbose:
        dev68 = ecp_at["0.68"] - 0.68
        verdict = ("over-confident (too narrow)" if dev68 < -0.03
                   else "over-dispersed (too broad)" if dev68 > 0.03
                   else "consistent with calibrated")
        print(f"\nTARP expected coverage ({metrics['reference']} references, "
              f"{metrics['n_refs']} refs, {n_jets} jets, EMD backend "
              f"{metrics['backend']}):")
        print(f"  max |ECP(alpha) - alpha| = {max_dev:.3f}   "
              f"(95% null floor 1.36/sqrt({n_jets}) = {null_floor:.3f};"
              f" {'ABOVE it => real deviation' if metrics['tarp_exceeds_null'] else 'below it => consistent with calibrated'})"
              f"   mean signed deviation = {signed:+.3f}")
        print("    " + "   ".join(f"ECP({a}) = {v:.3f}" for a, v in ecp_at.items())
              + f"   => {verdict}")
    return metrics


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------
def run_calibration(model, val_ds, geometry, device, K=200, n_jets=300, n_rank_bins=10,
                    verbose=True, pit_coords=False, stratify_regions=False, tarp=False,
                    tarp_refs=100, tarp_reference="pooled", mbr_kwargs=None, seed=0,
                    min_region_n=30, draws_by_jet=None):
    """SBC / PIT / coverage on held-out jets, plus the opt-in WP2 additions.

    With `pit_coords=stratify_regions=tarp=False` (the defaults) the returned dict is
    exactly the v1 dict plus the *uncertainty* keys — same RNG consumption, same
    values for every key that already existed. Each switch only ADDS keys.

    `min_region_n` is the per-region jet count below which a Lund quadrant is reported
    but marked `scored: false`. The quadrants are not equally populated (and the low-u
    strip is kinematically unreachable), so without a stated floor a 40-jet region's
    coverage gets quoted with the same confidence as a 200-jet one.

    `draws_by_jet` reuses posterior draws the caller already has (`draws_by_jet[i]` for
    jet `i`) instead of sampling K per jet here — see `eval.closure.run_closure` for
    the same argument and what sharing one pass costs. The SBC/PIT/coverage block is
    the only consumer; `tarp=True` still draws its own, since it needs the reference
    pool as well. Default None is today's behaviour exactly."""
    ranks = []
    coverage_hits = []
    pit_values = []
    regions: list[str | None] = []
    covered_regions: list[str | None] = []
    n_jets = min(n_jets, len(val_ds))
    if draws_by_jet is not None and len(draws_by_jet) < n_jets:
        raise ValueError(
            f"draws_by_jet has {len(draws_by_jet)} entries but n_jets={n_jets} are scored "
            f"— the shared draws must be aligned with val_ds[0..n_jets)"
        )
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        ny_true = int(item["ny"])
        draws = model.sample_batch(xf, nx, K) if draws_by_jet is None else draws_by_jet[i]
        mults = np.array([len(d) for d in draws])

        # SBC rank of the true multiplicity among posterior draws (Talts et al.).
        rank = int(np.sum(mults < ny_true) + 0.5 * np.sum(mults == ny_true))
        ranks.append(rank / max(len(mults), 1))

        # PIT: fraction of draws with multiplicity <= true (should be ~Uniform).
        pit_values.append(float(np.mean(mults <= ny_true)))

        # leading-cell central 68% coverage
        lead = [c for c in (leading_emission_cell(d, geometry) for d in draws) if c is not None]
        ly = leading_emission_cell(item["yc"].tolist(), geometry)
        regions.append(cell_region(ly, geometry))
        if ly is not None and lead:
            vals, counts = np.unique(np.array(lead), return_counts=True)
            order = np.argsort(-counts)
            cum = np.cumsum(counts[order]) / counts.sum()
            k68 = int(np.searchsorted(cum, 0.68)) + 1
            hpd = set(int(c) for c in vals[order][:k68])
            coverage_hits.append(1.0 if ly in hpd else 0.0)
            covered_regions.append(cell_region(ly, geometry))

    ranks = np.array(ranks)
    hist, _ = np.histogram(ranks, bins=n_rank_bins, range=(0.0, 1.0))
    expected = len(ranks) / n_rank_bins if len(ranks) else 1.0
    chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1e-8)))
    cov_hits = np.array(coverage_hits, dtype=float)
    cov_lo, cov_hi = wilson_interval(cov_hits.sum(), cov_hits.size)
    crit = chi2_crit95(n_rank_bins - 1)
    metrics = {
        "sbc_chi2_uniform": chi2,
        # Without its reference the chi^2 is unreadable: the same number is a pass at
        # 10 bins and a failure at 3. Quote the 95% point of chi^2(n_bins - 1) beside it.
        "sbc_chi2_dof": int(n_rank_bins - 1),
        "sbc_chi2_crit95": crit,
        "sbc_chi2_exceeds_crit95": bool(chi2 > crit),
        "sbc_rank_mean": float(np.mean(ranks)) if len(ranks) else float("nan"),
        "pit_mean": float(np.mean(pit_values)) if pit_values else float("nan"),
        "coverage_68": float(np.mean(coverage_hits)) if coverage_hits else float("nan"),
        # `coverage_68` is a binomial proportion on a few hundred jets; without its
        # interval "0.66 vs 0.68" reads as a finding when it is a coin flip.
        "n_coverage": int(cov_hits.size),
        "coverage_68_ci": [cov_lo, cov_hi],
        "coverage_68_consistent": bool(cov_lo <= 0.68 <= cov_hi) if cov_hits.size else False,
        "n_jets": int(n_jets),
    }
    if verbose:
        print("\nposterior calibration (SBC / PIT / coverage):")
        print(f"  SBC rank-uniformity chi^2 ({n_rank_bins} bins) = {metrics['sbc_chi2_uniform']:.2f}"
              f"   (95% point of chi^2({metrics['sbc_chi2_dof']}) = {crit:.2f};"
              f" {'ABOVE it => non-uniform' if metrics['sbc_chi2_exceeds_crit95'] else 'below it => consistent with uniform'})")
        print(f"  SBC mean rank = {metrics['sbc_rank_mean']:.3f}   PIT mean = {metrics['pit_mean']:.3f}"
              f"   (target ~0.5)")
        print(f"  leading-cell 68% coverage = {metrics['coverage_68']:.2f}"
              f"  95% Wilson [{cov_lo:.2f}, {cov_hi:.2f}] on {cov_hits.size:d} jets"
              f"   (target 0.68 —"
              f" {'inside' if metrics['coverage_68_consistent'] else 'OUTSIDE'} the interval)")
        if not getattr(model, "exact_likelihood", True):
            print("  NOTE: this family reports a SURROGATE log_prob (exact_likelihood=False);"
                  " the SBC/PIT/coverage above are sampling-based and still valid, its NLL is not.")

    # --- WP2.2: the same metrics, binned by leading-emission Lund quadrant ------
    if stratify_regions:
        rank_reg = np.array([r if r is not None else "none" for r in regions])
        cov_reg = np.array([r if r is not None else "none" for r in covered_regions])
        cov = np.array(coverage_hits, dtype=float)
        pit = np.array(pit_values, dtype=float)
        by_region = {}
        for label in REGION_LABELS:
            sel_r = rank_reg == label
            sel_c = cov_reg == label
            if not sel_r.any() and not sel_c.any():
                continue
            n_cov = int(sel_c.sum())
            lo, hi = wilson_interval(cov[sel_c].sum() if sel_c.any() else 0, n_cov)
            by_region[label] = {
                "n_jets": int(sel_r.sum()),
                "sbc_chi2_uniform": _chi2_uniform(ranks[sel_r], n_rank_bins) if sel_r.any() else float("nan"),
                "sbc_rank_mean": float(ranks[sel_r].mean()) if sel_r.any() else float("nan"),
                "sbc_rank_ks": _ks_uniform(ranks[sel_r]) if sel_r.any() else float("nan"),
                "pit_mean": float(pit[sel_r].mean()) if sel_r.any() else float("nan"),
                "coverage_68": float(cov[sel_c].mean()) if sel_c.any() else float("nan"),
                # A quadrant with 40 jets carries a +-0.15 interval: quoting it beside a
                # quadrant with 200 as though both were measurements is how "no quadrant
                # fails the band" gets asserted about a coin flip.
                "n_coverage": n_cov,
                "coverage_68_ci": [lo, hi],
                "coverage_68_consistent": bool(lo <= 0.68 <= hi) if n_cov else None,
                # Below `min_region_n` the region is REPORTED but not SCORED — the
                # verdict column is None rather than a number a reader would act on.
                "scored": bool(n_cov >= int(min_region_n)),
            }
        metrics["by_region"] = by_region
        metrics["region_min_n"] = int(min_region_n)
        # The split points, so a reader can check them against the kinematics rather
        # than assume all four quadrants are reachable. u = ln(1/DeltaR) is bounded
        # below by ln(1/R) (0.92 for R = 0.4), so the low-u strip of the `wide_*`
        # quadrants is empty by construction, not by the model's choice — and whichever
        # quadrant that leaves near-empty is a geometry fact, not a calibration failure.
        lo_u, hi_u = geometry.ln_invdelta_range
        lo_v, hi_v = geometry.ln_kt_range
        metrics["region_split"] = {"u": 0.5 * (lo_u + hi_u), "v": 0.5 * (lo_v + hi_v)}
        if verbose:
            print("  region-stratified (leading-emission Lund quadrant):")
            print(f"    {'region':>12} {'jets':>6} {'SBC chi2':>9} {'rank mean':>10}"
                  f" {'cov68':>7} {'95% Wilson':>16} {'n':>5}   (targets: low, 0.5, 0.68)")
            for label, e in by_region.items():
                ci = e["coverage_68_ci"]
                flag = "" if e["scored"] else f"  < n={min_region_n}, NOT SCORED"
                print(f"    {label:>12} {e['n_jets']:>6} {e['sbc_chi2_uniform']:>9.2f}"
                      f" {e['sbc_rank_mean']:>10.3f} {e['coverage_68']:>7.2f}"
                      f"   [{ci[0]:.2f}, {ci[1]:.2f}] {e['n_coverage']:>5}{flag}")
            print(f"    quadrant split at u = {metrics['region_split']['u']:.2f},"
                  f" v = {metrics['region_split']['v']:.2f};  u = ln(1/DeltaR) >= ln(1/R)"
                  f" = 0.92 at R = 0.4, so the low-u strip is kinematically unreachable")

    # --- WP2.1: per-coordinate PITs -------------------------------------------
    if pit_coords:
        pits = coordinate_pits(model, val_ds, geometry, device, n_jets=n_jets,
                               n_bins=n_rank_bins, stratify_regions=stratify_regions,
                               verbose=verbose)
        if pits is None:
            if verbose:
                print("\nper-coordinate PIT: unavailable for this family "
                      "(no exact coordinate density) — skipped.")
        else:
            metrics["pit_coords"] = pits
            metrics["pit_coords_ks_max"] = pits["ks_max"]

    # --- WP2.3: TARP -----------------------------------------------------------
    if tarp:
        t = run_tarp(model, val_ds, geometry, device, K=K, n_jets=n_jets,
                     n_refs=tarp_refs, reference=tarp_reference,
                     mbr_kwargs=mbr_kwargs, seed=seed, verbose=verbose)
        metrics["tarp"] = t
        metrics["tarp_max_dev"] = t["tarp_max_dev"]
    return metrics
