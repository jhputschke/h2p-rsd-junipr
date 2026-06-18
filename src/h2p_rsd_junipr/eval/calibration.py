"""Posterior calibration (§8): coverage, PIT, and simulation-based calibration
(SBC; Talts et al., arXiv:1804.06788).

Conditional-generator posteriors are not automatically calibrated (the original
cINN unfolding came out too narrow, arXiv:2006.06685), so this gates
"trustworthy". The SBC rank statistic here uses the multiplicity n as the test
quantity: for a calibrated posterior, the rank of the true n among posterior
draws is uniform on {0..K}.
"""

from __future__ import annotations

import numpy as np
import torch

from .closure import leading_emission_cell


def run_calibration(model, val_ds, geometry, device, K=200, n_jets=300, n_rank_bins=10, verbose=True):
    ranks = []
    coverage_hits = []
    pit_values = []
    n_jets = min(n_jets, len(val_ds))
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        ny_true = int(item["ny"])
        draws = model.sample_batch(xf, nx, K)
        mults = np.array([len(d) for d in draws])

        # SBC rank of the true multiplicity among posterior draws (Talts et al.).
        rank = int(np.sum(mults < ny_true) + 0.5 * np.sum(mults == ny_true))
        ranks.append(rank / max(len(mults), 1))

        # PIT: fraction of draws with multiplicity <= true (should be ~Uniform).
        pit_values.append(float(np.mean(mults <= ny_true)))

        # leading-cell central 68% coverage
        lead = [c for c in (leading_emission_cell(d, geometry) for d in draws) if c is not None]
        ly = leading_emission_cell(item["yc"].tolist(), geometry)
        if ly is not None and lead:
            vals, counts = np.unique(np.array(lead), return_counts=True)
            order = np.argsort(-counts)
            cum = np.cumsum(counts[order]) / counts.sum()
            k68 = int(np.searchsorted(cum, 0.68)) + 1
            hpd = set(int(c) for c in vals[order][:k68])
            coverage_hits.append(1.0 if ly in hpd else 0.0)

    ranks = np.array(ranks)
    hist, _ = np.histogram(ranks, bins=n_rank_bins, range=(0.0, 1.0))
    expected = len(ranks) / n_rank_bins if len(ranks) else 1.0
    chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1e-8)))
    metrics = {
        "sbc_chi2_uniform": chi2,
        "sbc_rank_mean": float(np.mean(ranks)) if len(ranks) else float("nan"),
        "pit_mean": float(np.mean(pit_values)) if pit_values else float("nan"),
        "coverage_68": float(np.mean(coverage_hits)) if coverage_hits else float("nan"),
        "n_jets": int(n_jets),
    }
    if verbose:
        print("\nposterior calibration (SBC / PIT / coverage):")
        print(f"  SBC rank-uniformity chi^2 ({n_rank_bins} bins) = {metrics['sbc_chi2_uniform']:.2f}"
              f"   (lower => more uniform => better calibrated)")
        print(f"  SBC mean rank = {metrics['sbc_rank_mean']:.3f}   PIT mean = {metrics['pit_mean']:.3f}"
              f"   (target ~0.5)")
        print(f"  leading-cell 68% coverage = {metrics['coverage_68']:.2f}   (target ~0.68)")
    return metrics
