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


def _leading_coords(arr):
    """Hardest-kt row of an `(n, 4)` node_raw table, as `(ln 1/DeltaR, ln kt)`."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        return None
    return a[int(np.argmax(a[:, 1])), :2]


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
                verbose=True, decode=None, continuous=False):
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
    `sample_coordinates`. At this geometry cells are ~0.6 wide and the cell-level
    distances are ~0.6, so the cell metric is quantisation-limited and cannot resolve
    what the model is doing; `*_cont` can. Cost is one `sample_coordinates` call per
    draw per jet (`n_closure * K` forward passes), which is why it is opt-in
    (`experiment.closure_continuous`). Families with no coordinate density
    (`ar_junipr_v1`) return None from the hook and the `*_cont` keys come back NaN."""
    dec = dict(decode or {})
    want_mbr = str(dec.get("point_estimator", "map")) == "mbr"
    d_id, d_mode, d_medoid = [], [], []
    n_id_bias, n_mean_bias, n_median_bias = [], [], []
    d_mbr, n_mbr_bias = [], []
    dc_id, dc_mode, dc_geomed = [], [], []   # continuous (no grid), opt-in
    empty_true, empty_pred = [], []          # the FULL population, incl. truth-empty jets
    cont_ok = bool(continuous)
    true_ns = []  # true N per kept jet, aligned with the bias lists (for the per-N table)
    covered = []
    n_closure = min(n_closure, len(val_ds))
    for i in range(n_closure):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        y_true = item["yc"].tolist()
        x_cells = geometry.seq_cells(val_jets[i]["x"][0], val_jets[i]["x"][1]).tolist()
        ny_true = len(y_true)

        draws = model.sample_batch(xf, nx, K)
        mults = np.array([len(d) for d in draws])
        lead = [c for c in (leading_emission_cell(d, geometry) for d in draws) if c is not None]
        ly = leading_emission_cell(y_true, geometry)

        # --- empty-tree accounting: BEFORE the `continue` below, deliberately -------
        # `ly is None` IS the truth-empty case, so the leading-emission selection that
        # follows drops exactly the jets this measures. Anything computed after it is
        # blind to them by construction and `p_empty_true` would read a flat 0.0.
        # The point estimate is taken here (not after) for the same reason, and reused
        # by the MBR block below so the `point_estimator="mbr"` path pays for it once.
        hat = model.map_or_mbr(xf, nx, draws=draws, **dec)
        empty_true.append(ny_true == 0)
        empty_pred.append(hat.multiplicity == 0)

        if ly is None or not lead:
            continue

        cells_arr = np.array(lead)
        vals, counts = np.unique(cells_arr, return_counts=True)
        mode_cell = int(vals[counts.argmax()])
        d_mode.append(lund_distance(mode_cell, ly, geometry))
        d_medoid.append(lund_distance(medoid_cell(lead, geometry), ly, geometry))
        d_id.append(lund_distance(leading_emission_cell(x_cells, geometry), ly, geometry))

        if cont_ok:  # the same three estimators, off the grid
            pts = []
            for d in draws:
                if not len(d):
                    continue
                c = model.sample_coordinates(xf, nx, list(d))
                if c is None:      # family has no coordinate density -> stop asking
                    cont_ok = False
                    break
                p = _leading_coords(c.detach().cpu().double().numpy().reshape(-1, 4))
                if p is not None:
                    pts.append(p)
            y_lead = _leading_coords(item["yraw"].numpy())
            x_lead = _leading_coords(node_raw(*val_jets[i]["x"]))
            if cont_ok and len(pts) >= 2 and y_lead is not None:
                pts = np.asarray(pts)
                dc_id.append(float(np.linalg.norm(x_lead - y_lead))
                             if x_lead is not None else float("nan"))
                dc_mode.append(float(np.linalg.norm(
                    np.asarray(geometry.cell_center(mode_cell)) - y_lead)))
                dc_geomed.append(float(np.linalg.norm(geometric_median(pts) - y_lead)))

        n_id_bias.append(len(x_cells) - ny_true)
        n_mean_bias.append(mults.mean() - ny_true)
        n_median_bias.append(np.median(mults) - ny_true)
        true_ns.append(ny_true)

        if want_mbr:  # `hat` above already IS the MBR estimate under this `dec`
            lead_mbr = leading_emission_cell([n.cell for n in hat.nodes], geometry)
            if lead_mbr is not None:
                d_mbr.append(lund_distance(lead_mbr, ly, geometry))
            n_mbr_bias.append(hat.multiplicity - ny_true)

        order = np.argsort(-counts)
        cum = np.cumsum(counts[order]) / counts.sum()
        k68 = int(np.searchsorted(cum, 0.68)) + 1
        hpd_set = set(int(c) for c in vals[order][:k68])
        covered.append(1.0 if ly in hpd_set else 0.0)

    d_id, d_mode = np.array(d_id), np.array(d_mode)
    metrics = {
        "mean_mult_true": float(np.mean([len(val_ds[i]["yc"]) for i in range(n_closure)])),
        "mean_mult_hadron": float(np.mean([len(val_jets[i]["x"][0]) for i in range(n_closure)])),
        "mean_mult_posterior": float(
            np.mean([b + len(val_ds[i]["yc"]) for i, b in enumerate(n_mean_bias)])
        ),
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

    if verbose:
        print("\nclosure + calibration on held-out jets:")
        print(
            f"  mean multiplicity            :  true y = {metrics['mean_mult_true']:.2f}"
            f"   hadron x = {metrics['mean_mult_hadron']:.2f}"
            f"   posterior = {metrics['mean_mult_posterior']:.2f}"
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
    want_mbr = str(dec.get("point_estimator", "map")) == "mbr"
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

    print("\nper-jet point estimate q_phi(y | x) for one validation jet:")
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
