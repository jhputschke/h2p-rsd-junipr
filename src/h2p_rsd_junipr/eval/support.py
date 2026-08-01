"""WP-D.1 of docs/PLAN_prod_test_v1.md: the support audit, as a SCORED metric.

v0 found 0.88% of sampled emissions below the soft-drop boundary — a *support* error,
not a calibration one — and found it ad hoc, in a notebook section, after the fact.
Nothing in the scored suite would have caught it, and nothing would catch its
recurrence. This module makes the three violation rates first-class metrics with a hard
zero target (gate G2): a nonzero value is a bug, not a finding.

The four boundaries, all of them properties of the GENERATOR the training data came
from, not opinions about the model:

* **window** — `(ln 1/DeltaR, ln kt)` outside the geometry's own ranges. Outside it a
  draw is not merely improbable, it is outside the space the cell grid tiles.
* **soft drop** — `ln z <= ln z_cut - beta * ln(1/DeltaR)`; Soft Drop discards those
  splittings, so the training data crosses that line exactly zero times
  (Larkoski et al., arXiv:1402.2657; RSD: Dreyer et al., arXiv:1804.03657).
* **`z <= 1/2`** — `z = min(pT1,pT2)/(pT1+pT2)` by construction, so `ln z > ln(1/2)` is
  not a soft prong at all. v0 never measured this half of the `ln z` leak.
* **`k_t` floor** — `ln kt` below the traversal's floor, likewise a hard generator cut.

The truth series is audited beside the posterior in the same pass. It is the control:
if truth shows a nonzero rate, the audit's own boundaries are wrong and no statement
about the model follows.
"""

from __future__ import annotations

import math

import numpy as np
import torch

_KEYS = ("out_of_window", "soft_drop", "z_above_half", "kt_floor")


def grooming_from_jets(jets) -> dict:
    """`(z_cut, beta, kt_floor)` from the loaded jets' own record, or NaNs.

    Read from the DATA rather than the config: the audit's whole value is that it tests
    the model against the boundaries its training file actually enforced."""
    out = {"z_cut": float("nan"), "beta": float("nan"), "kt_floor": float("nan")}
    for key in out:
        vals = [float(j[key]) for j in (jets or []) if key in j and j[key] == j[key]]
        if vals:
            out[key] = float(np.median(vals))
    return out


def violations(coords, geometry, *, z_cut, beta) -> dict:
    """Violation COUNTS of an `(m, 4)` coordinate table, plus `n` — see the module
    docstring for what each boundary is. `z_cut` NaN leaves the two `ln z` rows NaN
    rather than inventing a boundary."""
    a = np.asarray(coords, dtype=float).reshape(-1, 4)
    n = int(a.shape[0])
    lo_u, hi_u = geometry.ln_invdelta_range
    lo_v, hi_v = geometry.ln_kt_range
    if n == 0:
        return {"n": 0, **{k: 0 for k in _KEYS}}
    u, v, lnz = a[:, 0], a[:, 1], a[:, 2]
    out = {
        "n": n,
        "out_of_window": int(((u < lo_u) | (u > hi_u) | (v < lo_v) | (v > hi_v)).sum()),
        "kt_floor": int((v < lo_v).sum()),
    }
    if z_cut == z_cut:  # not NaN
        b = 0.0 if beta != beta else float(beta)
        out["soft_drop"] = int((lnz <= math.log(z_cut) - b * u).sum())
        out["z_above_half"] = int((lnz > math.log(0.5)).sum())
    else:
        out["soft_drop"] = out["z_above_half"] = -1   # unknown, not zero
    return out


def _rates(counts) -> dict:
    n = counts["n"]
    out = {"n_emissions": int(n)}
    for k in _KEYS:
        c = counts[k]
        out[k] = float("nan") if c < 0 else (float(c) / n if n else 0.0)
        out[f"n_{k}"] = int(c)
    finite = [out[k] for k in _KEYS if out[k] == out[k]]
    out["max_rate"] = float(max(finite)) if finite else float("nan")
    # Gate G2: the target is a hard zero, so `passes` is an equality, not a tolerance.
    out["passes"] = bool(finite and max(finite) == 0.0)
    return out


def run_support_audit(model, val_ds, val_jets, geometry, device, n_jets=300, K=200,
                      draws_by_jet=None, z_cut=None, beta=None, verbose=True) -> dict:
    """Violation rates of the SAMPLED posterior and of the truth, on the same jets.

    Costs one batched `sample_coordinates_many` per jet on top of draws the caller
    already has (pass `draws_by_jet`). A family with no coordinate density
    (`ar_junipr_v1`) can only be audited on the two cell-grid boundaries, and says so
    by returning `posterior: null` — the cell centres it would otherwise report are
    inside the window by construction, so auditing them would be a tautology."""
    n_jets = min(int(n_jets), len(val_ds))
    groom = grooming_from_jets(val_jets)
    z_cut = groom["z_cut"] if z_cut is None else float(z_cut)
    beta = groom["beta"] if beta is None else float(beta)

    truth = {"n": 0, **{k: 0 for k in _KEYS}}
    post = {"n": 0, **{k: 0 for k in _KEYS}}
    can_sample = bool(getattr(model, "has_continuous_coords", False))

    for i in range(n_jets):
        item = val_ds[i]
        t = violations(item["yraw"].numpy(), geometry, z_cut=z_cut, beta=beta)
        for k in ("n", *_KEYS):
            truth[k] = truth[k] + t[k] if not (truth[k] < 0 or t[k] < 0) else -1
        if not can_sample:
            continue
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        draws = (draws_by_jet[i] if draws_by_jet is not None
                 else model.sample_batch(xf, nx, K))
        chains = [list(d) for d in draws if len(d)]
        if not chains:
            continue
        for c in model.sample_coordinates_many(xf, nx, chains):
            if c is None:
                can_sample = False
                break
            p = violations(c.detach().cpu().numpy(), geometry, z_cut=z_cut, beta=beta)
            for k in ("n", *_KEYS):
                post[k] = post[k] + p[k] if not (post[k] < 0 or p[k] < 0) else -1

    out = {
        "z_cut": z_cut, "beta": beta, "kt_floor": groom["kt_floor"],
        "window": {"ln_invdelta": list(geometry.ln_invdelta_range),
                   "ln_kt": list(geometry.ln_kt_range)},
        "n_jets": int(n_jets),
        "truth": _rates(truth),
        "posterior": _rates(post) if can_sample and post["n"] else None,
        # `lnz_support` is what the model was TRAINED under; the audit is what the draws
        # actually do. Recorded together so the two can never be reported apart.
        "lnz_support": str(getattr(model, "lnz_support", "n/a")),
    }
    out["passes"] = bool(out["posterior"] is not None and out["posterior"]["passes"])
    if verbose:
        print(f"\nsupport audit ({n_jets} jets, z_cut = {z_cut:g}, beta = {beta:g}, "
              f"model.lnz_support = {out['lnz_support']!r}) — target is a hard ZERO:")
        print(f"    {'series':>10} {'emissions':>10} " + " ".join(f"{k:>16}" for k in _KEYS))
        for name in ("truth", "posterior"):
            e = out[name]
            if e is None:
                print(f"    {name:>10}   (no coordinate density — not auditable)")
                continue
            print(f"    {name:>10} {e['n_emissions']:>10} "
                  + " ".join(f"{e[k]:>16.5%}" if e[k] == e[k] else f"{'n/a':>16}"
                             for k in _KEYS))
        if out["truth"]["n_emissions"] and not out["truth"]["passes"]:
            print("    WARNING: the TRUTH violates a boundary — the audit's own bounds are "
                  "wrong, and nothing it says about the model follows.")
        elif out["posterior"] is not None:
            print(f"    verdict: {'PASS' if out['passes'] else 'FAIL'}"
                  f"  (gate G2; a nonzero rate is a bug, not a finding)")
    return out
