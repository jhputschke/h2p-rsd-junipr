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
                verbose=True, decode=None):
    """Closure + calibration on held-out jets (cell-level, as the v2 script). Returns
    a metrics dict and (optionally) prints the same summary lines. `decode` is a
    decode_params(cfg) dict threaded into sampling (n_posterior_samples ignored here;
    K wins) — kept for a uniform call signature with print_point_estimate.

    When `decode.point_estimator == "mbr"` an **MBR** series (minimum expected
    perturbative-Lund EMD over the K draws, `inference.mbr`) is added beside the
    posterior-mode estimator, so the leading-emission-distance and multiplicity-bias
    panels can be compared MBR-vs-mode. This is O(K^2) EMD solves per jet — shrink it
    with `decode.mbr_n_candidates` / a smaller `experiment.closure_jets`. The default
    (`point_estimator="map"`) skips it entirely (no cost, no OT-backend import)."""
    dec = dict(decode or {})
    want_mbr = str(dec.get("point_estimator", "map")) == "mbr"
    d_id, d_mode = [], []
    n_id_bias, n_mean_bias, n_median_bias = [], [], []
    d_mbr, n_mbr_bias = [], []
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
        if ly is None or not lead:
            continue

        cells_arr = np.array(lead)
        vals, counts = np.unique(cells_arr, return_counts=True)
        mode_cell = int(vals[counts.argmax()])
        d_mode.append(lund_distance(mode_cell, ly, geometry))
        d_id.append(lund_distance(leading_emission_cell(x_cells, geometry), ly, geometry))

        n_id_bias.append(len(x_cells) - ny_true)
        n_mean_bias.append(mults.mean() - ny_true)
        n_median_bias.append(np.median(mults) - ny_true)

        if want_mbr:  # MBR reuses the same draws (no resample); O(K^2) EMD per jet
            mbr_hat = model.map_or_mbr(xf, nx, draws=draws, **dec)
            lead_mbr = leading_emission_cell([n.cell for n in mbr_hat.nodes], geometry)
            if lead_mbr is not None:
                d_mbr.append(lund_distance(lead_mbr, ly, geometry))
            n_mbr_bias.append(mbr_hat.multiplicity - ny_true)

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
        "dlund_identity": float(np.nanmean(d_id)),
        "dlund_posterior_mode": float(np.nanmean(d_mode)),
        "mult_bias_identity": float(np.mean(n_id_bias)),
        "mult_bias_posterior": float(np.mean(n_mean_bias)),
        "mult_bias_posterior_median": float(np.mean(n_median_bias)),
        "coverage_68": float(np.mean(covered)),
    }
    if want_mbr:
        metrics["dlund_mbr"] = float(np.nanmean(d_mbr)) if d_mbr else float("nan")
        metrics["mult_bias_mbr"] = float(np.mean(n_mbr_bias)) if n_mbr_bias else float("nan")
        metrics["mbr_backend"] = str(dec.get("mbr_backend", "pot"))
    if verbose:
        print("\nclosure + calibration on held-out jets:")
        print(
            f"  mean multiplicity            :  true y = {metrics['mean_mult_true']:.2f}"
            f"   hadron x = {metrics['mean_mult_hadron']:.2f}"
            f"   posterior = {metrics['mean_mult_posterior']:.2f}"
        )
        print(
            f"  leading-emission Lund distance to true y :  identity(x) = {metrics['dlund_identity']:.3f}"
            f"   posterior-mode = {metrics['dlund_posterior_mode']:.3f}   (lower is better)"
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
        if want_mbr:
            print(
                f"  MBR ({metrics['mbr_backend']}) vs true y :"
                f"  leading-emission Lund distance = {metrics['dlund_mbr']:.3f}"
                f"   multiplicity bias <n - n_true> = {metrics['mult_bias_mbr']:+.3f}"
                f"   (floor-free; compare to posterior-mode / mean above)"
            )
    return metrics


def print_point_estimate(model, val_ds, val_jets, geometry, device, n_samples=500, decode=None):
    """Per-jet point estimate: plain RSD (hadron x) vs model MAP vs truth (jet 0).
    `decode` is a decode_params(cfg) dict (beam keys steer the MAP, e.g. min_emissions).
    When `decode.length_floor_quantile > 0` the MAP multiplicity is floored per jet at
    the learned quantile of P(n|x), reusing the posterior draws below."""
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
    # the MAP is always shown; MBR (floor-free) is shown beside it when opted in
    y_hat = model.map_estimate(xf, nx, **{k: v for k, v in dec.items() if k != "point_estimator"})
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
