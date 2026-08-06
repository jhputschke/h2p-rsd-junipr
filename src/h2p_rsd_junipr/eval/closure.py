"""Closure diagnostics (§8): on held-out generator data, the MAP recovers the
true y and posterior draws bracket it. Node-alignment-free observables only
(leading-emission Lund distance, multiplicity bias), since there is no per-node
x<->y correspondence (§4).

Was `leading_emission_cell`, `lund_distance`, `_tree_coords`, `lund_tree_str`, and
the closure block of the v2 script's `main()`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..features import node_raw
from ..geometry import Geometry
from ..inference.length import learned_min_emissions
from ..inference.point_estimate import LundPointEstimate

# True-multiplicity strata for the signed-bias breakdown (label, lo, hi inclusive).
_N_BINS = [("1-3", 1, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11+", 11, 10**9)]


def leading_emission_cell(cells, geometry: Geometry):
    """Hardest (largest ln kt) primary emission cell — the most perturbative,
    node-alignment-free observable. Returns its cell id or None."""
    if not cells:
        return None
    best, best_kt = cells[0], geometry.cell_center(cells[0])[1]
    for c in cells[1:]:
        kt = geometry.cell_center(c)[1]
        if kt > best_kt:
            best, best_kt = c, kt
    return best


def lund_distance(cell_a, cell_b, geometry: Geometry):
    """Euclidean distance in (ln 1/DeltaR, ln kt) between two cell centres."""
    if cell_a is None or cell_b is None:
        return float("nan")
    ax, ay = geometry.cell_center(cell_a)
    bx, by = geometry.cell_center(cell_b)
    return math.hypot(ax - bx, ay - by)


def medoid_cell(cells, geometry: Geometry):
    """The drawn leading cell of least MEAN `lund_distance` to all the draws.

    The loss-matched counterpart of the modal cell: `mode` is the argmin of expected
    0-1 loss, but this observable is scored by `lund_distance`, so the mode is the
    estimator for a loss nobody is measuring. The medoid is MBR (`inference.mbr`)
    applied to a one-node cloud — argmin over the same support of the quantity
    actually reported — so under the model's own posterior it cannot do worse.

    Empirically (2000 val jets, ar_junipr_v3): mode 1.030x identity(x), medoid
    0.944x. The whole "plain RSD wins the leading emission" result was the mode."""
    if not cells:
        return None
    vals, counts = np.unique(np.asarray(cells), return_counts=True)
    ctr = np.array([geometry.cell_center(int(c)) for c in vals])
    risk = (np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1) * counts[None, :]).sum(1)
    return int(vals[int(risk.argmin())])


def geometric_median(points, iters: int = 64, eps: float = 1e-9) -> np.ndarray:
    """`argmin_a sum_i ||a - p_i||` by Weiszfeld — the L1 Bayes point in the plane.

    Unlike `medoid_cell` this is not restricted to the drawn support, so it is the
    exact minimiser rather than the best available draw (the restriction exists for
    trees, which cannot be averaged; a point in the Lund plane can)."""
    P = np.asarray(points, dtype=float)
    a = P.mean(0)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(P - a, axis=1), eps)
        a_new = (P / d[:, None]).sum(0) / (1.0 / d).sum()
        if np.linalg.norm(a_new - a) < eps:
            return a_new
        a = a_new
    return a


def _rate(num, cond):
    """Mean of `num` over the entries where `cond` is true — recall with
    `(pred, true)`, precision with `(true, pred)`. NaN on an empty denominator, which
    is the honest answer: a sample with no truth-empty jet has no recall."""
    n, c = np.asarray(num, dtype=bool), np.asarray(cond, dtype=bool)
    return float(n[c].mean()) if c.any() else float("nan")


def _leading_row(arr):
    """Hardest-kt row of an `(n, 4)` node_raw table, WHOLE — `(u, v, ln z, psi)`.

    One `argmax(a[:, 1])` for every coordinate a caller wants, rather than one selection
    per slice: the 2-D and 3-D rulers below must be reading the *same emission* or the
    difference between them is not a difference of rulers (docs/PLAN_z_aware.md WP-1)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        return None
    return a[int(np.argmax(a[:, 1]))]


def _leading_coords(arr, ncols: int = 2):
    """The first `ncols` coordinates of `_leading_row` — `(ln 1/DeltaR, ln kt)` by default.

    `ncols` defaults to 2 so every pre-existing caller is unchanged *by construction*;
    `ncols=3` adds `ln z`, which is what the `dlund3_*` / `dlnz_*` series are built from."""
    row = _leading_row(arr)
    return None if row is None else row[:ncols]


def _mbr_leading_row(hat):
    """`(u, v, ln z)` of the point estimate's hardest-kt node — or None.

    None (⇒ NaN downstream) unless the estimate carries genuinely DRAWN coordinates.
    `map_or_mbr(point_estimator="mbr")` reaches `describe_cells`, which draws from
    `q(coords | cells, x)` and stamps `coords_source="sample"`, so the MBR row IS a
    posterior sample of `ln z` today with no plumbing at all (docs/PLAN_z_aware.md WP-1).
    Under `ar_junipr_v1` — or any `cell_center` fallback — `ln z` is the placeholder 0,
    i.e. `z = 1`, and scoring it would manufacture a number out of a filler constant."""
    if hat is None or not getattr(hat, "nodes", None):
        return None
    if str(getattr(hat, "coords_source", "")) != "sample":
        return None
    rows = np.array([[n.ln_invDelta, n.ln_kt, n.ln_z] for n in hat.nodes], dtype=float)
    return rows[int(np.argmax(rows[:, 1]))]


def _dist3(a, b):
    """`(||Δ(u,v,ln z)||, ||Δ(u,v)||, |Δ ln z|)` between two 3-vectors.

    A NaN anywhere in either row propagates to the components that read it, which is the
    "asked, unavailable" convention (`ln z = 0` means `z = 1` and must never be scored —
    see `models.base.describe_cells`). The 2-D component is deliberately still finite when only
    `ln z` is missing: a cell-centre estimator has a real `(u, v)` and no `ln z` at all."""
    if a is None or b is None:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    d = a[:3] - b[:3]
    return (float(np.linalg.norm(d)), float(np.linalg.norm(d[:2])), float(abs(d[2])))


def _nanmean(vals):
    """`nanmean` of a list, NaN (not a warning, not 0) when it is empty or all-NaN."""
    a = np.asarray(vals, dtype=float)
    finite = a[np.isfinite(a)]
    return float(finite.mean()) if finite.size else float("nan")


def _tree_coords(obj):
    if isinstance(obj, LundPointEstimate):
        return [(n.ln_invDelta, n.ln_kt, n.ln_z, n.psi) for n in obj.nodes]
    arr = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    return [(float(u), float(v), float(lz), float(ps)) for u, v, lz, ps in arr]


def lund_tree_str(obj, title: str, geometry: Geometry, ref=None) -> str:
    is_pe = isinstance(obj, LundPointEstimate)
    coords = _tree_coords(obj)
    rcoords = _tree_coords(ref) if ref is not None else None
    head = (
        f"{title}: {len(coords)} primary splittings"
        + (f", log q(y_hat|x) = {obj.logprob:.3f}" if is_pe else "")
    )
    rows = []
    for t, (u, v, lz, ps) in enumerate(coords):
        if is_pe:
            n = obj.nodes[t]
            tail = f"  logP={n.logp_split + n.logp_coord:+.2f}"
        else:
            tail = f"  cell={geometry.to_cell(u, v):3d}"
        if rcoords is not None and t < len(rcoords):
            tail += f"  dLund={math.hypot(u - rcoords[t][0], v - rcoords[t][1]):.3f}"
        rows.append(
            f"  [{t}] kt={math.exp(v):6.2f} GeV  DeltaR={math.exp(-u):5.3f}  "
            f"z={math.exp(lz):5.3f}  psi={ps:+5.2f}  "
            f"(ln1/DR={u:4.2f}, lnkt={v:4.2f}, lnz={lz:5.2f}){tail}"
        )
    return "\n".join([head, *rows]) if rows else head + "\n  (empty)"


def run_closure(model, val_ds, val_jets, geometry, device, K=200, n_closure=300,
                verbose=True, decode=None, continuous=False, draws_by_jet=None,
                per_jet=False, coords_by_jet=None):
    """Closure + calibration on held-out jets (cell-level, as the v2 script). Returns
    a metrics dict and (optionally) prints the same summary lines. `decode` is a
    decode_params(cfg) dict threaded into sampling (n_posterior_samples ignored here;
    K wins) — kept for a uniform call signature with print_point_estimate.

    When `decode.point_estimator == "mbr"` an **MBR** series (minimum expected
    perturbative-Lund EMD over the K draws, `inference.mbr`) is added beside the
    posterior-mode estimator, so the leading-emission-distance and multiplicity-bias
    panels can be compared MBR-vs-mode. This is O(K^2) EMD solves per jet — shrink it
    with `decode.mbr_n_candidates` / a smaller `experiment.closure_jets`. The default
    (`point_estimator="map"`) skips it entirely (no cost, no OT-backend import).

    `dlund_posterior_medoid` is reported beside `dlund_posterior_mode` unconditionally
    (it is pure numpy over cells already drawn, so it costs nothing) — see
    `medoid_cell` for why the mode is the wrong estimator for a distance-valued score.

    `continuous=True` adds the same comparison OFF the cell grid, via
    `sample_coordinates_many`. At this geometry cells are ~0.6 wide and the cell-level
    distances are ~0.6, so the cell metric is quantisation-limited and cannot resolve
    what the model is doing; `*_cont` can. Cost is ONE batched coordinate call per jet
    (it used to be one per draw, i.e. `n_closure * K` forward passes — the bulk of
    docs/PLAN_prod_test_speedup.md's 109 min). Families with no coordinate density
    (`ar_junipr_v1`) return None from the hook and the `*_cont` keys come back NaN.

    The continuous block also carries a **`ln z`-aware** ruler and, for the first time, an
    **MBR row** (docs/PLAN_z_aware.md WP-1). Three series come off the same leading
    emission, so they differ only in what the ruler looks at:

      * `dlund_*_cont`  — `||d(u, v)||`, the plane distance (pre-existing, unchanged);
      * `dlund3_*_cont` — `||d(u, v, ln z)||`, the same emission with `ln z` restored;
      * `dlnz_*`        — `|d ln z|` alone.

    This exists because `dlund_mbr` — the decode headline — compares leading-emission
    *cell centres*, so it is blind to `ln z` AND to the within-cell offsets, while the
    continuous block carried no MBR row at all. The ruler that scored the RQ-spline
    `ln z` head could therefore not register what the head improved, on either axis
    (docs/SUMMARY_Model_Status.md §2.5). **Additive only**: `decode_headline` stays
    `dlund_mbr` and every pre-existing key keeps its exact value — the same discipline
    `coverage_null_reps`' `fork_rng` follows (`eval/calibration.py`), because a switch
    that perturbs the statistic it exists to explain is worse than no switch.

    Unavailable is **NaN, never 0**. `dlund3_posterior_mode_cont` and
    `dlnz_posterior_mode` are NaN by construction: the modal *cell* has a real `(u, v)`
    centre and no `ln z` at all, and a placeholder `ln z = 0` means `z = 1`, the softer
    prong taking the whole jet (`models/base.py`). The MBR row is scored only where the
    point estimate carries genuinely DRAWN coordinates (`coords_source == "sample"`);
    under `ar_junipr_v1`, or any `cell_center` fallback, its keys are NaN too.

    `per_jet=True` adds `metrics["per_jet"]` — one row per scored jet, in jet order, with
    every distance above and NaN where a series is undefined for that jet (shape
    precedent: `eval/clusters.py`'s `metrics["per_jet"]`). Off by default, so
    `eval_metrics.json` is byte-identical without it. The rows are what makes a PAIRED
    analysis possible at all: every published `dlund_*` comparison in this repo WAS a
    difference of unpaired means until one of them was finally paired, and the conclusion
    read off it did not survive (docs/PLAN_z_aware.md §11).

    `draws_by_jet` reuses posterior draws the caller already has — `draws_by_jet[i]`
    for jet `i`, in place of the internal `sample_batch` — the same pattern as
    `mbr_select(draws=)` and `learned_min_emissions(mults=)`. Default None keeps
    today's behaviour, so `h2p-rsd-junipr eval` is untouched. Sharing one sampling pass
    across sections makes their comparisons exactly PAIRED, and makes the run not
    bit-comparable to one that re-sampled per section.

    `coords_by_jet` is its coordinate half — `coords_by_jet[i]` an UNFILTERED,
    index-aligned list of `(m_k, 4)` tables from `inference.mbr.coords_for_draws`. Taken
    once per jet and reused for BOTH the point estimate (`map_or_mbr`'s
    `coords_by_draw`, which is what `decode.mbr_cloud_source="coords"` selects on) and
    the `cont_ok` block below, so the two cannot disagree about where a draw sits.

    **The bit-identity hazard, and where it is fenced off** (docs/PLAN_z_aware.md §7.1).
    This function's own continuous call is *filtered* — `[list(d) for d in draws if
    len(d)]` — and `sample_coordinates_many` pads to `L_max` over the list it is handed,
    so an unfiltered call changes the block shape, reorders RNG consumption, and moves
    `dlund_*_cont` and the psi block. An unfiltered table therefore enters here **only**
    when the caller supplied one or `decode.mbr_cloud_source="coords"` asked for it; with
    the switch off the filtered call is still the one taken, byte for byte."""
    dec = dict(decode or {})
    # Either MBR variant produces the same series: a drawn tree selected off the same D.
    want_mbr = str(dec.get("point_estimator", "map")) in ("mbr", "mbr_n")
    # `decode.mbr_cloud_source="coords"` makes the SELECTION read continuous coordinates
    # (docs/PLAN_z_aware.md §4/WP-3). It only reaches an estimator that builds clouds, so
    # it is a no-op under `point_estimator="map"` — asserted here rather than left implicit,
    # because a MAP decode that silently drew K coordinate tables per jet would pay the
    # whole cost of the switch for none of its effect.
    cloud_source = str(dec.get("mbr_cloud_source", "cells"))
    needs_coords = bool(want_mbr and cloud_source == "coords")
    # The edit family's emergent-alignment readout (docs/PLAN_EditTransducer.md). Gated on
    # the model exposing it, so every other family's metric dict is untouched.
    edit_summary = getattr(model, "edit_summary", None)
    edit_acc: dict[str, list[float]] = {"frac_anchored": [], "delete_rate": [], "insert_rate": []}
    d_id, d_mode, d_medoid = [], [], []
    n_id_bias, n_mean_bias, n_median_bias = [], [], []
    d_mbr, n_mbr_bias = [], []
    dc_id, dc_mode, dc_geomed = [], [], []   # continuous (no grid), opt-in
    dc_mbr = []                              # ...and the MBR row the block never had
    # The same four estimators under a `ln z`-aware ruler: 3-D over (u, v, ln z), and
    # |d ln z| alone (docs/PLAN_z_aware.md WP-1). Keyed by estimator so the four series
    # cannot drift apart in the accumulation.
    _EST = ("identity", "posterior_mode", "posterior_geomedian", "mbr")
    d3 = {k: [] for k in _EST}
    dlnz = {k: [] for k in _EST}
    per_jet_rows: list[dict] = []
    empty_true, empty_pred = [], []          # the FULL population, incl. truth-empty jets
    cont_ok = bool(continuous)
    true_ns = []  # true N per kept jet, aligned with the bias lists (for the per-N table)
    # ...and the same two over the FULL population, including the truth-empty jets the
    # leading-emission selection drops. Gate G4's <N> clause is about this pair: selecting
    # jets by `N_true >= 1` and comparing them to the posterior mean is regression to the
    # mean, so its deficit is negative by construction (docs/PLAN_prod_test_v1.md WP-B.1).
    post_mean_all, true_n_all = [], []
    # --- psi resultant (gate G6). |R| = |<e^{i psi}>| pooled over nodes: 0 for a uniform
    #     azimuth, 1 for a pinned one. v0's decode reported 0.69 against a truth of 0.045,
    #     because it attached the MODE of a von Mises whose median kappa is 0.022 — a
    #     direction that is not identified. Truth, point estimate and posterior draws are
    #     accumulated as complex sums so the pooled resultant is exact, not a mean of means.
    psi_sum = {"truth": 0j, "point": 0j, "posterior": 0j}
    psi_n = {"truth": 0, "point": 0, "posterior": 0}
    psi_unident, psi_nodes_scored = 0, 0
    covered = []
    n_closure = min(n_closure, len(val_ds))
    if draws_by_jet is not None and len(draws_by_jet) < n_closure:
        raise ValueError(
            f"draws_by_jet has {len(draws_by_jet)} entries but n_closure={n_closure} jets "
            f"are scored — the shared draws must be aligned with val_ds[0..n_closure)"
        )
    if coords_by_jet is not None and len(coords_by_jet) < n_closure:
        raise ValueError(
            f"coords_by_jet has {len(coords_by_jet)} entries but n_closure={n_closure} "
            f"jets are scored — the shared coordinates must be aligned with "
            f"val_ds[0..n_closure), one unfiltered table per draw"
        )
    for i in range(n_closure):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        y_true = item["yc"].tolist()
        x_cells = geometry.seq_cells(val_jets[i]["x"][0], val_jets[i]["x"][1]).tolist()
        ny_true = len(y_true)

        # Appended BY REFERENCE and filled in below, so the `continue` that drops the
        # truth-empty jets still leaves a row for them — a paired analysis pairs on the
        # jet index, and a row that is silently absent is indistinguishable from a jet
        # the other arm also dropped.
        row = {"jet": int(i), "ny_true": int(ny_true), "kept": False}
        if per_jet:
            per_jet_rows.append(row)

        draws = model.sample_batch(xf, nx, K) if draws_by_jet is None else draws_by_jet[i]
        mults = np.array([len(d) for d in draws])
        lead = [c for c in (leading_emission_cell(d, geometry) for d in draws) if c is not None]
        ly = leading_emission_cell(y_true, geometry)

        # --- empty-tree accounting: BEFORE the `continue` below, deliberately -------
        # `ly is None` IS the truth-empty case, so the leading-emission selection that
        # follows drops exactly the jets this measures. Anything computed after it is
        # blind to them by construction and `p_empty_true` would read a flat 0.0.
        # The point estimate is taken here (not after) for the same reason, and reused
        # by the MBR block below so the `point_estimator="mbr"` path pays for it once.
        # The coordinate half of this jet's pool, taken ONCE and reused by both the point
        # estimate and the `cont_ok` block. None on the default path, where nothing has
        # asked for it and the filtered call below is still what runs.
        jet_coords = None
        if coords_by_jet is not None:
            jet_coords = coords_by_jet[i]
        elif needs_coords:
            from ..inference.mbr import coords_for_draws

            jet_coords = coords_for_draws(model, xf, nx, draws)
        hat = model.map_or_mbr(xf, nx, draws=draws, coords_by_draw=jet_coords, **dec)
        empty_true.append(ny_true == 0)
        empty_pred.append(hat.multiplicity == 0)
        post_mean_all.append(float(mults.mean()) if mults.size else 0.0)
        true_n_all.append(ny_true)
        row["n_hat"] = int(hat.multiplicity)
        row["post_mean"] = post_mean_all[-1]

        if getattr(model, "has_continuous_coords", False):
            tpsi = item["yraw"][:, 3].numpy() if ny_true else np.zeros(0)
            psi_sum["truth"] += complex(np.exp(1j * tpsi).sum())
            psi_n["truth"] += int(tpsi.size)
            hpsi = np.array([n.psi for n in hat.nodes], dtype=float)
            psi_sum["point"] += complex(np.exp(1j * hpsi).sum())
            psi_n["point"] += int(hpsi.size)
            psi_unident += hat.n_psi_unidentified
            psi_nodes_scored += hat.multiplicity

        if edit_summary is not None:
            # over EVERY jet, like the empty-tree block above: a truth-empty jet is the
            # delete-all path, which is precisely a deletion-rate measurement.
            s = edit_summary({
                "xf": xf, "nx": nx,
                "yc": item["yc"].unsqueeze(0).to(device),
                "ny": torch.tensor([ny_true], device=device),
                "yraw": item["yraw"].unsqueeze(0).to(device),
            })
            for key, arr in s.items():
                edit_acc[key].append(float(arr[0]))

        if ly is None or not lead:
            continue

        cells_arr = np.array(lead)
        vals, counts = np.unique(cells_arr, return_counts=True)
        mode_cell = int(vals[counts.argmax()])
        d_mode.append(lund_distance(mode_cell, ly, geometry))
        d_medoid.append(lund_distance(medoid_cell(lead, geometry), ly, geometry))
        d_id.append(lund_distance(leading_emission_cell(x_cells, geometry), ly, geometry))
        row.update(kept=True, dlund_identity=d_id[-1], dlund_posterior_mode=d_mode[-1],
                   dlund_posterior_medoid=d_medoid[-1])

        if cont_ok:  # the same estimators, off the grid — and now under three rulers
            rows3 = []
            # ONE batched call for the jet's K draws, not K calls: the per-draw hook
            # re-runs encode()/xattn_kv() every time on identical conditioning
            # (docs/PLAN_prod_test_speedup.md §2). When `jet_coords` already exists it is
            # reused rather than re-drawn — one table per jet, feeding the selection and
            # the ruler alike. That table is UNFILTERED, so it carries the empty draws too;
            # they contribute no psi and no leading row, exactly as the filtered call's
            # missing entries did, but the padded block shape (hence the RNG stream)
            # differs — which is why this substitution stays behind the switch.
            tables = (jet_coords if jet_coords is not None else
                      model.sample_coordinates_many(
                          xf, nx, [list(d) for d in draws if len(d)]))
            for c in tables:
                if c is None:      # family has no coordinate density -> stop asking
                    cont_ok = False
                    break
                arr = (c.detach().cpu().double().numpy().reshape(-1, 4)
                       if hasattr(c, "detach") else
                       np.asarray(c, dtype=float).reshape(-1, 4))
                # the posterior's own psi resultant, from the same draws (G6's reference)
                psi_sum["posterior"] += complex(np.exp(1j * arr[:, 3]).sum())
                psi_n["posterior"] += int(arr.shape[0])
                p = _leading_coords(arr, 3)
                if p is not None:
                    rows3.append(p)
            y_lead = _leading_coords(item["yraw"].numpy(), 3)
            x_lead = _leading_coords(node_raw(*val_jets[i]["x"]), 3)
            if cont_ok and len(rows3) >= 2 and y_lead is not None:
                rows3 = np.asarray(rows3)
                pts = rows3[:, :2]   # the pre-existing 2-D block, off the same argmax
                dc_id.append(float(np.linalg.norm(x_lead[:2] - y_lead[:2]))
                             if x_lead is not None else float("nan"))
                dc_mode.append(float(np.linalg.norm(
                    np.asarray(geometry.cell_center(mode_cell)) - y_lead[:2])))
                dc_geomed.append(float(np.linalg.norm(geometric_median(pts) - y_lead[:2])))
                # --- the `ln z`-aware ruler (docs/PLAN_z_aware.md WP-1) ---------------
                # The modal CELL has a centre and no `ln z`, so its third component is
                # NaN and propagates: "asked, unavailable", never a scored placeholder.
                # `geometric_median` is dimension-agnostic, so the 3-D row is the 3-D
                # Bayes point rather than the 2-D one with a coordinate stapled on.
                est_rows = {
                    "identity": x_lead,
                    "posterior_mode": np.array(
                        [*geometry.cell_center(mode_cell), float("nan")]),
                    "posterior_geomedian": geometric_median(rows3),
                    "mbr": _mbr_leading_row(hat) if want_mbr else None,
                }
                for name, r in est_rows.items():
                    d3v, d2v, dzv = _dist3(r, y_lead)
                    d3[name].append(d3v)
                    dlnz[name].append(dzv)
                    if name == "mbr":
                        dc_mbr.append(d2v)
                    row[f"dlund3_{name}_cont"] = d3v
                    row[f"dlnz_{name}"] = dzv
                row["dlund_identity_cont"] = dc_id[-1]
                row["dlund_posterior_mode_cont"] = dc_mode[-1]
                row["dlund_posterior_geomedian_cont"] = dc_geomed[-1]
                row["dlund_mbr_cont"] = dc_mbr[-1]
                row["point_coords_source"] = str(getattr(hat, "coords_source", "unknown"))

        n_id_bias.append(len(x_cells) - ny_true)
        n_mean_bias.append(mults.mean() - ny_true)
        n_median_bias.append(np.median(mults) - ny_true)
        true_ns.append(ny_true)

        if want_mbr:  # `hat` above already IS the MBR estimate under this `dec`
            lead_mbr = leading_emission_cell([n.cell for n in hat.nodes], geometry)
            if lead_mbr is not None:
                d_mbr.append(lund_distance(lead_mbr, ly, geometry))
                row["dlund_mbr"] = d_mbr[-1]
            n_mbr_bias.append(hat.multiplicity - ny_true)

        order = np.argsort(-counts)
        cum = np.cumsum(counts[order]) / counts.sum()
        k68 = int(np.searchsorted(cum, 0.68)) + 1
        hpd_set = set(int(c) for c in vals[order][:k68])
        covered.append(1.0 if ly in hpd_set else 0.0)

    d_id, d_mode = np.array(d_id), np.array(d_mode)
    tn_all = np.array(true_n_all, dtype=float)
    pm_all = np.array(post_mean_all, dtype=float)
    tn_kept = np.array(true_ns, dtype=float)
    metrics = {
        "mean_mult_true": float(np.mean([len(val_ds[i]["yc"]) for i in range(n_closure)])),
        "mean_mult_hadron": float(np.mean([len(val_jets[i]["x"][0]) for i in range(n_closure)])),
        # Over the FULL population and correctly paired. It used to be
        # `mean(b + len(val_ds[i]["yc"]) for i, b in enumerate(n_mean_bias))`, where `i`
        # indexes the KEPT jets and `val_ds[i]` the unfiltered dataset — so each kept jet's
        # bias was added to a different jet's truth. That mispairing, plus the truth-nonempty
        # selection, is where v0's "posterior 1.15 vs truth 1.40" came from.
        "mean_mult_posterior": float(pm_all.mean()) if pm_all.size else float("nan"),
        "mean_mult_ratio": (float(pm_all.mean() / tn_all.mean())
                            if pm_all.size and tn_all.mean() else float("nan")),
        # The same pair restricted to jets with a truth leading emission — the population
        # every `dlund_*` row below lives on. Reported so the two are never confused again,
        # and flagged: conditioning on the truth makes this comparison biased low.
        "mean_mult_true_kept": float(tn_kept.mean()) if tn_kept.size else float("nan"),
        "mean_mult_posterior_kept": (
            float((np.array(n_mean_bias) + tn_kept).mean()) if tn_kept.size else float("nan")
        ),
        "mean_mult_kept_is_truth_selected": True,
        # The empty tree. `mult_bias_*` is provably blind to this failure — a MAP that
        # answers 1 wherever the truth is 0 lands at mean multiplicity 1.41 against a
        # true 1.42 while recovering 0% of them — so it is reported explicitly.
        # Computed over EVERY jet, unlike the dlund_* keys below (see `n_kept`).
        "p_empty_true": float(np.mean(empty_true)) if empty_true else float("nan"),
        "p_empty_pred": float(np.mean(empty_pred)) if empty_pred else float("nan"),
        "recall_empty": _rate(empty_pred, empty_true),
        "precision_empty": _rate(empty_true, empty_pred),
        "n_empty_true": int(np.sum(empty_true)),
        "n_jets_scored": len(empty_true),
        # ...whereas every dlund_* number below is conditioned on the TRUTH having a
        # leading emission (the `ly is None` continue), i.e. it is p(leading | n_y > 0).
        "n_kept_leading": len(d_id),
        "dlund_identity": float(np.nanmean(d_id)),
        "dlund_posterior_mode": float(np.nanmean(d_mode)),
        "dlund_posterior_medoid": float(np.nanmean(d_medoid)) if d_medoid else float("nan"),
        "mult_bias_identity": float(np.mean(n_id_bias)),
        "mult_bias_posterior": float(np.mean(n_mean_bias)),
        "mult_bias_posterior_median": float(np.mean(n_median_bias)),
        "coverage_68": float(np.mean(covered)),
    }
    if edit_summary is not None:
        # NaN where the rate is undefined for a jet (no parton nodes to anchor, or no
        # hadron nodes to delete), so these are nanmeans over the jets that have one.
        for key, vals in edit_acc.items():
            finite = [v for v in vals if v == v]
            metrics[key] = float(np.mean(finite)) if finite else float("nan")
    # Which row is the DECODE headline (docs/PLAN_prod_test_v1.md WP-C.3). The MAP is a
    # diagnostic: it is the argmax of a high-entropy sequence posterior, an estimator for
    # a loss nobody is measuring here (Stahlberg & Byrne, arXiv:1908.10090; Eikema & Aziz,
    # arXiv:2005.10283). The population headline is the decode-free posterior series.
    metrics["decode_headline"] = "dlund_mbr" if want_mbr else "dlund_posterior_medoid"
    metrics["map_is_diagnostic"] = True
    if want_mbr:
        metrics["dlund_mbr"] = float(np.nanmean(d_mbr)) if d_mbr else float("nan")
        metrics["mult_bias_mbr"] = float(np.mean(n_mbr_bias)) if n_mbr_bias else float("nan")
        metrics["mbr_backend"] = str(dec.get("mbr_backend", "pot"))
    if continuous:
        # NaN (not absent) when the family has no coordinate density, so a consumer
        # reading these keys sees "asked, unavailable" rather than "never asked".
        nan = float("nan")
        metrics["dlund_identity_cont"] = float(np.nanmean(dc_id)) if dc_id else nan
        metrics["dlund_posterior_mode_cont"] = float(np.nanmean(dc_mode)) if dc_mode else nan
        metrics["dlund_posterior_geomedian_cont"] = (
            float(np.nanmean(dc_geomed)) if dc_geomed else nan
        )
        metrics["n_continuous_jets"] = int(len(dc_geomed))
        # --- the `ln z`-aware ruler, and the MBR row this block never had --------------
        # Same jets, same leading emission, three rulers (docs/PLAN_z_aware.md WP-1):
        # `dlund_*_cont` is the plane distance, `dlund3_*_cont` restores `ln z`, `dlnz_*`
        # is that coordinate alone. Every one of these is NEW — nothing above moved.
        metrics["dlund_mbr_cont"] = _nanmean(dc_mbr)
        for name in _EST:
            metrics[f"dlund3_{name}_cont"] = _nanmean(d3[name])
            metrics[f"dlnz_{name}"] = _nanmean(dlnz[name])
        # How many jets each ruler actually scored. `dlund3_posterior_mode_cont` is NaN on
        # all of them by construction (a cell centre has no `ln z`), and the MBR row is NaN
        # wherever the estimate was empty or its coordinates were not drawn — so a reader
        # can tell an unscorable series from an unremarkable one.
        metrics["n_continuous_scored"] = {
            "mbr_cont": int(np.isfinite(dc_mbr).sum()),
            **{f"dlund3_{k}": int(np.isfinite(d3[k]).sum()) for k in _EST},
            **{f"dlnz_{k}": int(np.isfinite(dlnz[k]).sum()) for k in _EST},
        }
    if per_jet:
        metrics["per_jet"] = per_jet_rows

    # Signed multiplicity bias stratified by true N: does the marginal-multiplicity bias
    # (posterior-mean/median) survive in MBR, and does it vary with the true length?
    tn = np.array(true_ns, dtype=int)
    mean_arr, median_arr = np.array(n_mean_bias), np.array(n_median_bias)
    mbr_arr = np.array(n_mbr_bias) if (want_mbr and n_mbr_bias) else None
    mult_bias_by_N = {}
    for label, lo, hi in _N_BINS:
        sel = (tn >= lo) & (tn <= hi)
        entry = {
            "n_jets": int(sel.sum()),
            "posterior_mean": float(mean_arr[sel].mean()) if sel.any() else float("nan"),
            "posterior_median": float(median_arr[sel].mean()) if sel.any() else float("nan"),
        }
        if mbr_arr is not None:
            entry["mbr"] = float(mbr_arr[sel].mean()) if sel.any() else float("nan")
        mult_bias_by_N[label] = entry
    metrics["mult_bias_by_N"] = mult_bias_by_N

    # --- psi identifiability + resultant (gate G6) -----------------------------------
    def _R(key):
        return abs(psi_sum[key]) / psi_n[key] if psi_n[key] else float("nan")

    def _R_null(key):
        """`E[|R|]` for `n` i.i.d. UNIFORM angles = sqrt(pi)/(2 sqrt(n)).

        A resultant without this is unreadable. `|R|` is a norm, so it is positive
        under uniformity too, and its floor moves with `n` — the truth series and the
        posterior series are pooled over wildly different node counts here, so the same
        `|R|` means different things in each row. Reporting the ratio of two numbers
        that are both at their own noise floors would be reporting noise."""
        n = psi_n[key]
        return math.sqrt(math.pi) / (2.0 * math.sqrt(n)) if n else float("nan")

    def _rayleigh_p(key):
        """Rayleigh test of uniformity: `p ~ exp(-n |R|^2)` (Mardia & Jupp §6.3.1).
        Small p => a genuinely preferred azimuth; large p => consistent with isotropic."""
        n = psi_n[key]
        return math.exp(-n * _R(key) ** 2) if n else float("nan")

    metrics["psi"] = {
        "resultant_truth": _R("truth"),
        "resultant_point_estimate": _R("point"),
        # the uniform floor for each row's own node count, and the Rayleigh p beside it
        "resultant_null_truth": _R_null("truth"),
        "resultant_null_point_estimate": _R_null("point"),
        "resultant_null_posterior": _R_null("posterior"),
        "rayleigh_p_truth": _rayleigh_p("truth"),
        "rayleigh_p_point_estimate": _rayleigh_p("point"),
        "rayleigh_p_posterior": _rayleigh_p("posterior"),
        # NaN unless `experiment.closure_continuous=true`: the posterior's psi only
        # exists once coordinates are drawn, and drawing them for this alone would be a
        # second sampling pass. Asked-and-unavailable, not never-asked.
        "resultant_posterior": _R("posterior"),
        "n_nodes_truth": int(psi_n["truth"]),
        "n_nodes_point_estimate": int(psi_n["point"]),
        "n_nodes_posterior": int(psi_n["posterior"]),
        "ratio_point_over_truth": (_R("point") / _R("truth")) if _R("truth") else float("nan"),
        "frac_psi_unidentified": (psi_unident / psi_nodes_scored
                                  if psi_nodes_scored else float("nan")),
        "kappa_min_mode": float(getattr(model, "kappa_min_mode", 0.0)),
        "point_coords_source": str(getattr(hat, "coords_source", "unknown")) if n_closure else "",
    }
    if verbose and psi_n["truth"]:
        p = metrics["psi"]
        print("  psi resultant |R| = |<e^(i psi)>|, each row against the UNIFORM floor"
              " for its own node count:")
        print(f"      {'series':>16} {'nodes':>7} {'|R|':>8} {'uniform E|R|':>13}"
              f" {'Rayleigh p':>11}")
        for label, key in (("truth", "truth"), ("point estimate", "point_estimate"),
                           ("posterior", "posterior")):
            n = p[f"n_nodes_{key}"]
            if not n:
                print(f"      {label:>16} {'n/a':>7}"
                      "   (needs experiment.closure_continuous=true)")
                continue
            print(f"      {label:>16} {n:>7} {p['resultant_' + key]:>8.4f}"
                  f" {p['resultant_null_' + key]:>13.4f} {p['rayleigh_p_' + key]:>11.3f}")
        print(f"      point/truth = {p['ratio_point_over_truth']:.2f}x"
              f"   (gate G6 wants within 2x — but read it beside the floors above:"
              f" a ratio of two numbers at their own noise floors is noise)")
        print(f"      psi mode not identified (kappa < {p['kappa_min_mode']:g}) for"
              f" {p['frac_psi_unidentified']:.1%} of point-estimate nodes;"
              f" coordinates carried as {p['point_coords_source']!r}")

    if verbose:
        print("\nclosure + calibration on held-out jets:")
        print(
            f"  mean multiplicity, ALL {metrics['n_jets_scored']} jets"
            f"   :  true y = {metrics['mean_mult_true']:.3f}"
            f"   hadron x = {metrics['mean_mult_hadron']:.3f}"
            f"   posterior = {metrics['mean_mult_posterior']:.3f}"
            f"   ratio = {metrics['mean_mult_ratio']:.3f}   (gate G4 reads this row)"
        )
        print(
            f"  the same on the {metrics['n_kept_leading']} truth-NONEMPTY jets"
            f" :  true y = {metrics['mean_mult_true_kept']:.3f}"
            f"   posterior = {metrics['mean_mult_posterior_kept']:.3f}"
            f"   (SELECTED ON TRUTH — biased low by construction, not a second measurement)"
        )
        print(
            f"  leading-emission Lund distance to true y :  identity(x) = {metrics['dlund_identity']:.3f}"
            f"   posterior-mode = {metrics['dlund_posterior_mode']:.3f}"
            f"   posterior-medoid = {metrics['dlund_posterior_medoid']:.3f}   (lower is better;"
            f" medoid is the loss-matched estimator, mode is kept for continuity)"
        )
        if continuous:
            print(
                f"  the same, OFF the cell grid ({metrics['n_continuous_jets']} jets) :"
                f"  identity(x) = {metrics['dlund_identity_cont']:.3f}"
                f"   posterior-mode = {metrics['dlund_posterior_mode_cont']:.3f}"
                f"   posterior-geo-median = {metrics['dlund_posterior_geomedian_cont']:.3f}"
            )
            print(
                f"      (cells are ~{(geometry.ln_kt_range[1] - geometry.ln_kt_range[0]) / geometry.n_bins:.2f}"
                " wide, so the cell-level row above is quantisation-limited)"
            )
            # The `ln z`-aware ruler beside the plane one, on the same emission. The MBR
            # column is the point of the table: the fielded headline `dlund_mbr` compares
            # CELL CENTRES, so it sees neither `ln z` nor the within-cell offset.
            print("      the same leading emission under three rulers"
                  "  (2-D = ||d(u,v)||, 3-D adds ln z, dlnz is ln z alone):")
            print(f"      {'estimator':>22} {'2-D cont':>10} {'3-D cont':>10} {'|dlnz|':>10}")
            for name in _EST:
                two = {"identity": metrics["dlund_identity_cont"],
                       "posterior_mode": metrics["dlund_posterior_mode_cont"],
                       "posterior_geomedian": metrics["dlund_posterior_geomedian_cont"],
                       "mbr": metrics["dlund_mbr_cont"]}[name]
                print(f"      {name:>22} {two:>10.4f}"
                      f" {metrics['dlund3_' + name + '_cont']:>10.4f}"
                      f" {metrics['dlnz_' + name]:>10.4f}")
            print("        (posterior_mode is NaN under the ln z rulers by construction:"
                  " a modal CELL has no ln z)")
        print(
            f"  multiplicity signed bias  <n - n_true>   :  identity(x) = {metrics['mult_bias_identity']:+.3f}"
            f"   posterior-mean = {metrics['mult_bias_posterior']:+.3f}"
            f"   posterior-median = {metrics['mult_bias_posterior_median']:+.3f}   (closer to 0 is better)"
        )
        print(
            f"  posterior 68% coverage of true leading cell = {metrics['coverage_68']:.2f}"
            f"   (target ~0.68; <0.68 => over-confident)"
        )
        _tau = float(dec.get("empty_threshold", 0.0))
        print(
            f"  empty parton tree (ALL {metrics['n_jets_scored']} jets; the dlund_* rows"
            f" above keep only the {metrics['n_kept_leading']} with a truth leading"
            f" emission):\n"
            f"      truth = {metrics['p_empty_true']:.3f}"
            f"   predicted = {metrics['p_empty_pred']:.3f}"
            f"   recall = {metrics['recall_empty']:.3f}"
            f"   precision = {metrics['precision_empty']:.3f}"
            f"   (decode.empty_threshold = {_tau:g}"
            + ("; 0 == off, so predicted ~0 is the DECODE, not the fit)" if _tau <= 0
               else ")")
        )
        if edit_summary is not None:
            print(
                f"  latent alignment (edit transducer; posterior over alignments, never"
                f" supervised):\n"
                f"      kept & smeared = {metrics['frac_anchored']:.3f}"
                f"   inserted = {metrics['insert_rate']:.3f}"
                f"   deleted = {metrics['delete_rate']:.3f}"
                f"   (of n_y, n_y, n_x respectively; n_y = n_x - #del + #ins)"
            )
        if want_mbr:
            print(
                f"  MBR ({metrics['mbr_backend']}) vs true y :"
                f"  leading-emission Lund distance = {metrics['dlund_mbr']:.3f}"
                f"   multiplicity bias <n - n_true> = {metrics['mult_bias_mbr']:+.3f}"
                f"   (floor-free; compare to posterior-mode / mean above)"
            )
        print("  multiplicity signed bias stratified by true N (mean over jets in bin):")
        head = f"    {'true N':>7} {'jets':>6} {'post-mean':>11} {'post-median':>12}"
        if mbr_arr is not None:
            head += f" {'MBR':>9}"
        print(head)
        for label, _lo, _hi in _N_BINS:
            e = mult_bias_by_N[label]
            row = f"    {label:>7} {e['n_jets']:>6} {e['posterior_mean']:>+11.3f} {e['posterior_median']:>+12.3f}"
            if mbr_arr is not None:
                row += f" {e['mbr']:>+9.3f}"
            print(row)
    return metrics


def print_point_estimate(model, val_ds, val_jets, geometry, device, n_samples=500, decode=None):
    """Per-jet point estimate: plain RSD (hadron x) vs model MAP vs truth (jet 0).
    `decode` is a decode_params(cfg) dict (beam keys steer the MAP, e.g. min_emissions).
    When `decode.length_floor_quantile > 0` the MAP multiplicity is floored per jet at
    the learned quantile of P(n|x), reusing the posterior draws below.

    The MAP row goes through `map_or_mbr`, not `map_estimate`, so `decode.empty_threshold`
    reaches it. It used to call `map_estimate` directly, which cannot return the empty
    tree — so with the gate on, this block printed a non-empty MAP for the very jets
    `run_closure`'s `p_empty_pred` had just counted as empty, and the two halves of one
    `eval` disagreed (docs/PLAN_prod_test_v0.md check 7)."""
    dec = dict(decode or {})
    want_mbr = str(dec.get("point_estimator", "map")) in ("mbr", "mbr_n")
    item = val_ds[0]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)
    draws = model.sample_batch(xf, nx, n_samples)
    mults = np.array([len(d) for d in draws])
    alpha = float(dec.get("length_floor_quantile", 0.0))
    if alpha > 0.0:  # learned per-jet floor reuses the draws above (no double-sample)
        dec["min_emissions"] = learned_min_emissions(
            model, xf, nx, quantile=alpha,
            base_floor=int(dec.get("min_emissions", 1)), mults=mults,
        )
    # the MAP is always shown; MBR (floor-free) is shown beside it when opted in.
    # `point_estimator="map"` pins the branch while still passing through the empty gate.
    y_hat = model.map_or_mbr(xf, nx, draws=draws, **{**dec, "point_estimator": "map"})
    mbr_hat = model.map_or_mbr(xf, nx, draws=draws, **dec) if want_mbr else None

    x_raw = node_raw(*val_jets[0]["x"])
    y_truth = item["yraw"]
    lead_truth = leading_emission_cell(item["yc"].tolist(), geometry)
    d_rsd = lund_distance(
        leading_emission_cell(geometry.seq_cells(*val_jets[0]["x"][:2]).tolist(), geometry),
        lead_truth, geometry,
    )
    d_map = lund_distance(
        leading_emission_cell([n.cell for n in y_hat.nodes], geometry), lead_truth, geometry
    )

    print("\nper-jet point estimate q_phi(y | x) for one validation jet"
          + ("  (MAP shown as a DIAGNOSTIC; MBR is the decode headline)" if want_mbr
             else "  (MAP is a DIAGNOSTIC: the argmax of a high-entropy sequence"
                  " posterior — set decode.point_estimator=mbr for the headline)")
          + ":")
    print(
        f"  multiplicity:  truth y = {item['ny']}   model MAP = {y_hat.multiplicity}   "
        f"plain RSD (hadron x) = {len(x_raw)}   "
        f"posterior = {mults.mean():.2f} +/- {mults.std():.2f} "
        f"(median {np.median(mults):.0f}, 68% CR [{np.percentile(mults, 16):.0f}, {np.percentile(mults, 84):.0f}])"
    )
    print(
        f"  leading-emission Lund distance to truth:  plain RSD = {d_rsd:.3f}   "
        f"model MAP = {d_map:.3f}   (lower is better)"
    )
    if mbr_hat is not None:
        d_mbr = lund_distance(
            leading_emission_cell([n.cell for n in mbr_hat.nodes], geometry), lead_truth, geometry
        )
        print(
            f"  MBR ({dec.get('mbr_backend', 'pot')}, floor-free):  multiplicity = {mbr_hat.multiplicity}"
            f"   leading-emission Lund distance = {d_mbr:.3f}"
            f"   risk = {mbr_hat.risk:.3f}   (mean expected Lund-EMD to the posterior; not an NLL)"
        )
    print("\n" + lund_tree_str(y_hat, "model MAP groomed shower", geometry, ref=y_truth))
    if mbr_hat is not None:
        print("\n" + lund_tree_str(mbr_hat, "model MBR groomed shower (perturbative Lund)", geometry, ref=y_truth))
    print("\n" + lund_tree_str(x_raw, "plain RSD groomed shower (hadron-level x)", geometry, ref=y_truth))
    print("\n" + lund_tree_str(y_truth, "true groomed shower", geometry))
