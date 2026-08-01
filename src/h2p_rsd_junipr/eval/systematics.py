"""Generator systematic (§8): train on generator A (e.g. PYTHIA), evaluate the
MAP/posterior spread against a generator-B-trained copy (e.g. HERWIG) and
alternative tunes (arXiv:2203.11601; arXiv:1512.01178). The inter-model spread
*is* the dominant systematic and must be quoted.

This module compares two trained models on the same evaluation jets and reports
the spread of per-jet point estimates and multiplicities between them.
"""

from __future__ import annotations

import numpy as np
import torch

from .closure import leading_emission_cell, lund_distance


def generator_spread(model_a, model_b, val_ds, geometry, device, n_jets=300, verbose=True,
                     decode=None):
    """Per-jet spread between two generator-trained posteriors on shared jets.

    `decode` is a `decode_params(cfg)` dict, applied identically to both models — the
    spread is only meaningful when the two are decoded the same way. It used to call
    `map_estimate` with no arguments at all, so the quantity billed as "the dominant
    systematic" was measured under the signature defaults rather than under the decode
    the run was configured with, and could never see the empty tree whatever
    `decode.empty_threshold` said (docs/PLAN_prod_test_v0.md check 7). `decode=None`
    reproduces that old behaviour exactly."""
    dec = dict(decode or {})
    d_lead = []
    d_mult = []
    n_jets = min(n_jets, len(val_ds))
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        ya = model_a.map_or_mbr(xf, nx, **dec)
        yb = model_b.map_or_mbr(xf, nx, **dec)
        la = leading_emission_cell([n.cell for n in ya.nodes], geometry)
        lb = leading_emission_cell([n.cell for n in yb.nodes], geometry)
        d_lead.append(lund_distance(la, lb, geometry))
        d_mult.append(abs(ya.multiplicity - yb.multiplicity))
    # All-NaN is a real outcome, not an error: with the empty gate on for both models
    # neither has a leading emission to compare. Report NaN rather than letting
    # `np.nanmean` warn — and never a 0, which would read as perfect agreement.
    lead = np.asarray(d_lead, dtype=float)
    metrics = {
        "lead_lund_spread_mean": (float(np.nanmean(lead)) if np.isfinite(lead).any()
                                  else float("nan")),
        "n_lead_compared": int(np.isfinite(lead).sum()),
        "mult_spread_mean": float(np.mean(d_mult)),
        "n_jets": int(n_jets),
        # The spread depends on HOW both models were decoded, so the number is not
        # interpretable without it — a systematic quoted from a different decode is a
        # different systematic.
        "point_estimator": str(dec.get("point_estimator", "map")),
        "empty_threshold": float(dec.get("empty_threshold", 0.0)),
        "min_emissions": int(dec.get("min_emissions", 1)),
    }
    if verbose:
        print("\ngenerator systematic (model A vs model B MAP spread):")
        print(f"  leading-emission Lund spread = {metrics['lead_lund_spread_mean']:.3f}")
        print(f"  multiplicity spread = {metrics['mult_spread_mean']:.3f}")
        print("  (this inter-model spread is the dominant systematic and must be quoted)")
    return metrics
