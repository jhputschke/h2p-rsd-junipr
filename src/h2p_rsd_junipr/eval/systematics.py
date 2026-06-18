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


def generator_spread(model_a, model_b, val_ds, geometry, device, n_jets=300, verbose=True):
    """Per-jet spread between two generator-trained posteriors on shared jets."""
    d_lead = []
    d_mult = []
    n_jets = min(n_jets, len(val_ds))
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        ya = model_a.map_estimate(xf, nx)
        yb = model_b.map_estimate(xf, nx)
        la = leading_emission_cell([n.cell for n in ya.nodes], geometry)
        lb = leading_emission_cell([n.cell for n in yb.nodes], geometry)
        d_lead.append(lund_distance(la, lb, geometry))
        d_mult.append(abs(ya.multiplicity - yb.multiplicity))
    metrics = {
        "lead_lund_spread_mean": float(np.nanmean(d_lead)),
        "mult_spread_mean": float(np.mean(d_mult)),
        "n_jets": int(n_jets),
    }
    if verbose:
        print("\ngenerator systematic (model A vs model B MAP spread):")
        print(f"  leading-emission Lund spread = {metrics['lead_lund_spread_mean']:.3f}")
        print(f"  multiplicity spread = {metrics['mult_spread_mean']:.3f}")
        print("  (this inter-model spread is the dominant systematic and must be quoted)")
    return metrics
