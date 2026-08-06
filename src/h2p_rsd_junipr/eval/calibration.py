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

import math

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


def wilson_diff_interval(k1, n1, k2, n2, z=1.96) -> tuple[float, float]:
    """95% interval for the DIFFERENCE of two independent proportions, `p1 - p2`
    (Newcombe, *Statist. Med.* **17** (1998) 873, his method 10 — the hybrid score).

    Built from the two Wilson intervals, so it inherits their behaviour at the edges and
    needs no scipy:

        lower = (p1 - p2) - sqrt((p1 - l1)^2 + (u2 - p2)^2)
        upper = (p1 - p2) + sqrt((u1 - p1)^2 + (p2 - l2)^2)

    **Why this exists rather than "is p1 inside p2's interval".** That test throws away
    p1's own error, so it is anti-conservative exactly when p1 is the *noisier* of the two
    — which is the case it is usually reached for. `coverage_68` is measured on ~500 jets
    (Wilson half-width ≈ 0.044) and its null on ~8 800 pseudo-truths (≈ 0.010), so asking
    whether the observation lands inside the null's interval rejects a **perfectly
    calibrated** arm most of the time. Returns `(nan, nan)` if either `n` is 0."""
    if int(n1) <= 0 or int(n2) <= 0:
        return (float("nan"), float("nan"))
    p1, p2 = float(k1) / int(n1), float(k2) / int(n2)
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (float(lo), float(hi))


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


def tarp_null_band(n_jets: int, n_alpha: int = 21, n_reps: int = 2000, seed: int = 0
                   ) -> dict:
    """The null distribution of `max_alpha |ECP(alpha) - alpha|`, by Monte Carlo at the
    run's OWN `(n_jets, alpha grid)` (docs/PLAN_prod_test_v1.md WP-D.2).

    Under a calibrated posterior each credibility level `f_i` is Uniform(0,1), so the
    null needs no model: draw `n_jets` uniforms, build the same ECP on the same alpha
    grid, take the sup deviation, repeat. This differs from the analytic `1.36/sqrt(n)`
    in two ways that matter at the sizes actually used — the deviation is evaluated on a
    FINITE alpha grid (which lowers it) and `1.36/sqrt(n)` is asymptotic (which is loose
    at a few hundred) — and, unlike a formula, it is quoted at the size of the run that
    produced the number.

    The gate reads `p95`; `floor_ok` is the plan's precondition that the band be tight
    enough for the statistic to mean anything at all (v0's 0.079 floor at n = 300 could
    not have detected a 0.05 deviation, so quoting "max dev 0.037, passes" was quoting
    the sample size)."""
    rng = np.random.default_rng(seed)
    alpha = np.linspace(0.0, 1.0, int(n_alpha))
    f = rng.random((int(n_reps), int(n_jets)))
    # ECP(alpha) = mean_i [f_i < alpha], vectorised over reps and the alpha grid
    ecp = (f[:, None, :] < alpha[None, :, None]).mean(axis=2)
    dev = np.abs(ecp - alpha[None, :]).max(axis=1)
    p95 = float(np.percentile(dev, 95))
    return {
        "n_jets": int(n_jets), "n_alpha": int(n_alpha), "n_reps": int(n_reps),
        "p95": p95, "p99": float(np.percentile(dev, 99)), "mean": float(dev.mean()),
        "analytic_floor95": float(1.36 / np.sqrt(max(n_jets, 1))),
        # A band whose own 95% point is above 0.05 cannot resolve a 5% miscalibration,
        # so a "pass" against it is a statement about n, not about the posterior.
        "floor_ok": bool(p95 < 0.05),
        "floor_target": 0.05,
    }


def run_tarp(model, val_ds, geometry, device, K=200, n_jets=300, n_refs=100,
             reference="pooled", n_alpha=21, mbr_kwargs=None, seed=0,
             null_reps=0, stratify=False, min_region_n=30,
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
    regions: list[str | None] = []
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
        regions.append(cell_region(leading_emission_cell(truth, geometry), geometry))

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
    # --- WP-D.2: the null band recomputed at THIS run's (n, alpha grid) -------------
    if int(null_reps) > 0:
        band = tarp_null_band(n_jets, n_alpha=int(n_alpha), n_reps=int(null_reps),
                              seed=int(seed))
        metrics["null_band"] = band
        metrics["tarp_exceeds_null_mc"] = bool(max_dev > band["p95"])
        # Gate G7 has two clauses, and this is the second: the band's own floor must be
        # below 0.05 before "inside the band" is a statement about the posterior.
        metrics["tarp_quotable"] = bool(band["floor_ok"])
        metrics["tarp_passes_g7"] = bool(band["floor_ok"] and max_dev <= band["p95"])
    # --- WP-D.2: the same statistic per Lund quadrant -------------------------------
    if stratify:
        reg = np.array([r if r is not None else "none" for r in regions])
        by_region = {}
        for label in REGION_LABELS:
            sel = reg == label
            n_r = int(sel.sum())
            if not n_r:
                continue
            e_r = np.array([float(np.mean(f[sel] < a)) for a in alpha])
            dev_r = float(np.max(np.abs(e_r - alpha)))
            entry = {"n_jets": n_r, "tarp_max_dev": dev_r,
                     "tarp_signed_bias": float(np.mean(e_r - alpha)),
                     "ecp_at": {f"{a:.2f}": float(np.interp(a, alpha, e_r))
                                for a in (0.5, 0.68, 0.9, 0.95)},
                     # A quadrant's null is its OWN n, which is a fraction of the total —
                     # so a per-region band is always looser, and saying which regions are
                     # scoreable at all is the point of reporting it.
                     "scored": bool(n_r >= int(min_region_n))}
            if int(null_reps) > 0:
                b_r = tarp_null_band(n_r, n_alpha=int(n_alpha), n_reps=int(null_reps),
                                     seed=int(seed) + 1)
                entry["null_p95"] = b_r["p95"]
                entry["floor_ok"] = b_r["floor_ok"]
                entry["exceeds_null"] = bool(dev_r > b_r["p95"])
            by_region[label] = entry
        metrics["by_region"] = by_region
        metrics["region_min_n"] = int(min_region_n)
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
        band = metrics.get("null_band")
        if band:
            print(f"  null band recomputed at THIS run's (n = {band['n_jets']},"
                  f" {band['n_alpha']} alpha, {band['n_reps']} reps):"
                  f" 95% = {band['p95']:.3f}, 99% = {band['p99']:.3f}"
                  f"   (analytic 1.36/sqrt(n) = {band['analytic_floor95']:.3f})")
            print(f"    max dev {max_dev:.3f} is"
                  f" {'ABOVE' if metrics['tarp_exceeds_null_mc'] else 'inside'} it;"
                  f" band floor {'<' if band['floor_ok'] else '>='} {band['floor_target']}"
                  f" => the statistic is"
                  f" {'quotable' if band['floor_ok'] else 'NOT quotable at this n'}"
                  f"   => gate G7 {'PASS' if metrics['tarp_passes_g7'] else 'FAIL'}")
        reg = metrics.get("by_region")
        if reg:
            print("  TARP by Lund quadrant:")
            print(f"    {'region':>12} {'jets':>6} {'max dev':>9} {'null 95%':>9}"
                  f" {'signed':>8}")
            for label, e in reg.items():
                flag = "" if e["scored"] else f"   < n={metrics['region_min_n']}, NOT SCORED"
                p95 = e.get("null_p95")
                print(f"    {label:>12} {e['n_jets']:>6} {e['tarp_max_dev']:>9.3f}"
                      f" {(f'{p95:.3f}' if p95 is not None else 'n/a'):>9}"
                      f" {e['tarp_signed_bias']:>+8.3f}{flag}")
    return metrics


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------
def run_calibration(model, val_ds, geometry, device, K=200, n_jets=300, n_rank_bins=10,
                    verbose=True, pit_coords=False, stratify_regions=False, tarp=False,
                    tarp_refs=100, tarp_reference="pooled", mbr_kwargs=None, seed=0,
                    min_region_n=30, draws_by_jet=None, tarp_null_reps=0,
                    tarp_stratify=False, coverage_null_reps=0):
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
    coverage_null_hits = []      # WP4: the same statistic with the MODEL as the truth
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
            if coverage_null_reps:
                # The statistic's OWN null: score held-out draws from the SAME posterior
                # as pseudo-truths, through the identical HPD construction above. Under
                # model == truth a calibrated statistic returns 0.68; anything lower is
                # the estimator, not the model. The HPD is built from K draws, so it
                # cannot contain a cell of probability < 1/K that a genuine draw still
                # visits — that loss is exactly what this measures, and it is why
                # "0.53 vs 0.68" cannot be read as over-confidence on its own.
                #
                # Drawn inside `fork_rng`, so this DIAGNOSTIC cannot change which draws
                # the next jet gets. Without it the switch is not additive: every key
                # that already existed moves, and a switch that perturbs the statistic it
                # exists to explain is worse than no switch. Same discipline as
                # `PosteriorModel.decode_generator` (tests/test_shared_draws.py), reached
                # here through the RNG state because `sample_batch` takes no generator.
                with torch.random.fork_rng(devices=([xf.device] if xf.device.type == "cuda"
                                                    else []), enabled=True):
                    extra = model.sample_batch(xf, nx, int(coverage_null_reps))
                for d in extra:
                    lp = leading_emission_cell(d, geometry)
                    if lp is not None:
                        coverage_null_hits.append(1.0 if lp in hpd else 0.0)

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
    if coverage_null_reps:
        nul = np.array(coverage_null_hits, dtype=float)
        n_lo, n_hi = wilson_interval(nul.sum(), nul.size)
        metrics["coverage_68_null"] = float(nul.mean()) if nul.size else float("nan")
        metrics["coverage_68_null_ci"] = [n_lo, n_hi]
        metrics["n_coverage_null"] = int(nul.size)
        # The comparison that matters is coverage vs its OWN null, not vs the nominal 0.68.
        metrics["coverage_68_vs_null"] = (
            float(metrics["coverage_68"] - metrics["coverage_68_null"])
            if nul.size and cov_hits.size else float("nan"))
        metrics["coverage_68_null_explains_deficit"] = (
            bool(n_lo <= metrics["coverage_68"] <= n_hi) if nul.size and cov_hits.size
            else None)
        # ...and the SAME question asked properly. The key above compares a point estimate
        # against the other estimate's interval, which discards `coverage_68`'s own error —
        # and that is the larger of the two by ~4x (a few hundred jets against a few
        # thousand pseudo-truths), so it rejects a perfectly calibrated arm most of the
        # time. `wilson_diff_interval` prices both (docs/PLAN_next_steps.md B3). The old
        # key is kept, unchanged in value, because it is in committed artifacts.
        d_lo, d_hi = wilson_diff_interval(cov_hits.sum(), cov_hits.size,
                                          nul.sum(), nul.size)
        metrics["coverage_68_vs_null_ci"] = [d_lo, d_hi]
        metrics["coverage_68_null_explains_deficit_paired"] = (
            bool(d_lo <= 0.0 <= d_hi) if np.isfinite(d_lo) else None)
        metrics["coverage_68_null_note"] = (
            "the empirical HPD-68 is built from K draws and cannot contain a cell of "
            "probability < 1/K, so it under-covers even a PERFECT model. Read "
            "coverage_68 against this null, not against 0.68. Score it with "
            "`coverage_68_null_explains_deficit_paired`, which is the Newcombe interval "
            "on the DIFFERENCE and prices both errors; "
            "`coverage_68_null_explains_deficit` asks the narrower question 'is the "
            "observation inside the null's own interval', which ignores the observation's "
            "error and is therefore too strict."
        )
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
        if coverage_null_reps:
            ex = metrics["coverage_68_null_explains_deficit"]
            print(f"      ...against its OWN null (the model as truth, same K-draw HPD): "
                  f"{metrics['coverage_68_null']:.3f} "
                  f"[{metrics['coverage_68_null_ci'][0]:.2f}, "
                  f"{metrics['coverage_68_null_ci'][1]:.2f}] on "
                  f"{metrics['n_coverage_null']} pseudo-truths")
            print("      -> the deficit is "
                  + ("THE STATISTIC (an HPD from K draws misses cells with p < 1/K); "
                     "compare to the null, not to 0.68" if ex else
                     "REAL: below even the null, so the posterior is genuinely too narrow"))
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
            # --- WP-D.3: the region x coordinate cross, as a scannable summary ------
            # The cross already exists inside `pit_coords` when both switches are on, one
            # nested dict per coordinate. Flattened here because that is the instrument
            # gate G5 reads — "which coordinate, in which quadrant" — and a reader should
            # not have to walk four nested dicts to find the worst cell of the table.
            # Ranked by KS / its own 95% critical value, NOT by raw KS. The regions have
            # very different counts, and the critical value is 1.36/sqrt(n): a KS of 0.15
            # on 52 emissions is INSIDE its null while 0.077 on 2671 exceeds it threefold.
            # Ranking on the raw number would name the small region every time.
            worst = {"coord": None, "region": None, "ks": float("nan"), "n": 0,
                     "crit95": float("nan"), "ks_over_crit": float("nan")}
            cross = {}
            for name in pits["names"]:
                by_r = pits["coords"][name].get("by_region")
                if not by_r:
                    continue
                cross[name] = {}
                for r, e in by_r.items():
                    crit = 1.36 / max(np.sqrt(max(int(e["n"]), 1)), 1e-9)
                    ratio = e["ks"] / crit if crit else float("nan")
                    cross[name][r] = {"ks": e["ks"], "n": e["n"], "mean": e["mean"],
                                      "crit95": float(crit), "ks_over_crit": float(ratio),
                                      "scored": bool(e["n"] >= int(min_region_n))}
                    if e["n"] >= int(min_region_n) and not (ratio <= worst["ks_over_crit"]):
                        worst = {"coord": name, "region": r, "ks": e["ks"], "n": e["n"],
                                 "crit95": float(crit), "ks_over_crit": float(ratio)}
            if cross:
                metrics["pit_coords_by_region"] = cross
                metrics["pit_coords_by_region_worst"] = worst
                if verbose:
                    print("\n  region x coordinate PIT (the WP-D.3 attribution "
                          f"instrument; each cell is KS / its own 95% critical value "
                          f"1.36/sqrt(n), so >1 fails. regions with n < {min_region_n} "
                          f"not scored):")
                    regs = sorted({r for v in cross.values() for r in v})
                    print(f"    {'coord':>6} " + " ".join(f"{r:>18}" for r in regs))
                    for name, v in cross.items():
                        cells = []
                        for r in regs:
                            e = v.get(r)
                            cells.append("n/a".rjust(18) if e is None else
                                         (f"{e['ks_over_crit']:.2f}x "
                                          f"({e['ks']:.3f}, n={e['n']})"
                                          + ("" if e["scored"] else "*")).rjust(18))
                        print(f"    {name:>6} " + " ".join(cells))
                    if worst["coord"]:
                        print(f"    worst scored cell: {worst['coord']} x {worst['region']}"
                              f"  KS = {worst['ks']:.3f} on n = {worst['n']}"
                              f"  = {worst['ks_over_crit']:.2f}x its critical value "
                              f"{worst['crit95']:.3f}")

    # --- WP2.3 / WP-D.2: TARP, with its null band recomputed at this run's size ----
    if tarp:
        t = run_tarp(model, val_ds, geometry, device, K=K, n_jets=n_jets,
                     n_refs=tarp_refs, reference=tarp_reference,
                     mbr_kwargs=mbr_kwargs, seed=seed, verbose=verbose,
                     null_reps=int(tarp_null_reps), stratify=bool(tarp_stratify),
                     min_region_n=int(min_region_n))
        metrics["tarp"] = t
        metrics["tarp_max_dev"] = t["tarp_max_dev"]
    return metrics
