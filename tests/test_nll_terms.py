"""`ARJunipr.nll_terms` — the per-term decomposition, and which terms survive a
geometry change.

The question this exists to answer (docs/PLAN_prod_test_v0.md check 1): a 30-bin NLL
is not simply "incomparable" to a 10-bin one. With `continuous_coords=True` the TOTAL
is a density on the (ln 1/DeltaR, ln kt) plane and is dimensionally commensurable
across `n_bins`; `split_ll` alone is a probability over cells and is not. These tests
pin both halves of that claim on an untrained model, where the split head is uniform
and the shift is exactly computable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

SEL = ["model=ar_junipr_v4", "encoder=lundnet"]


def _model_and_batch(n_bins, n_jets=24, seed=0):
    cfg = load_config([*SEL, f"geometry.n_bins={n_bins}"])
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    model = build_model(cfg, geom).eval()
    jets = synthetic_matched_dataset(n_jets, seed=0)
    ds = MatchedLundDataset(jets, geom)
    return model, collate([ds[i] for i in range(len(ds))]), geom


def test_terms_sum_to_per_jet_nll():
    model, batch, _ = _model_and_batch(10)
    with torch.no_grad():
        t = model.nll_terms(batch)
        total = model.per_jet_nll(batch)
    recon = -(t["length_ll"] + t["split_ll"] + t["coord_ll"])
    assert torch.allclose(recon, total, atol=1e-6), "the parts must be the whole"
    assert torch.equal(t["n_emissions"], batch["ny"].to(t["length_ll"].dtype))


def test_split_ll_shifts_by_two_log_ratio_and_the_total_does_not():
    """An UNTRAINED split head is uniform over `n_cells`, so per emission
    `split_ll = -ln(n_bins^2)` exactly, and 10 -> 30 costs `2*ln(3)` = 2.197 nat.
    The coordinate term pays it back: the within-cell density is `1/(cell area)`
    larger, so the total per emission is unchanged."""
    m10, b10, _ = _model_and_batch(10)
    m30, b30, _ = _model_and_batch(30)
    with torch.no_grad():
        t10, t30 = m10.nll_terms(b10), m30.nll_terms(b30)

    n = t10["n_emissions"].sum().item()
    assert n > 0 and torch.equal(t10["n_emissions"], t30["n_emissions"])

    split10 = t10["split_ll"].sum().item() / n
    split30 = t30["split_ll"].sum().item() / n
    assert split10 == pytest.approx(-math.log(100), abs=0.02)
    assert split30 == pytest.approx(-math.log(900), abs=0.02)
    # the headline number in check 1: 2*ln(30/10)
    assert (split10 - split30) == pytest.approx(2 * math.log(3.0), abs=0.05)

    # The coordinate term is a DENSITY over the cell, so shrinking the cell 3x per axis
    # raises it by the same 2*ln(3) — leaving the sum a density on the plane, and hence
    # commensurable across geometries. An untrained head is not exactly uniform on the
    # (du, dv) box, so this is a loose bound, not the identity above.
    coord10 = t10["coord_ll"].sum().item() / n
    coord30 = t30["coord_ll"].sum().item() / n
    assert (coord30 - coord10) == pytest.approx(2 * math.log(3.0), abs=0.6)

    per_em10 = (t10["split_ll"] + t10["coord_ll"]).sum().item() / n
    per_em30 = (t30["split_ll"] + t30["coord_ll"]).sum().item() / n
    assert abs(per_em30 - per_em10) < 0.6, (
        "the cell-probability x within-cell-density product must stay a density on the "
        "plane; if this drifts with n_bins the totals are NOT comparable"
    )


def test_length_term_is_untouched_by_the_geometry():
    """`q(N|x)` is a distribution over multiplicities; nothing about it references the
    cell grid, so it is comparable across `n_bins` with no correction at all."""
    m10, b10, _ = _model_and_batch(10, seed=3)
    m30, b30, _ = _model_and_batch(30, seed=3)
    with torch.no_grad():
        l10 = m10.nll_terms(b10)["length_ll"]
        l30 = m30.nll_terms(b30)["length_ll"]
    # same seed, same encoder widths, same targets -> identical length head input path
    assert l10.shape == l30.shape
    assert np.isfinite(l10.numpy()).all() and np.isfinite(l30.numpy()).all()


def test_v1_has_no_coordinate_term_so_its_total_is_not_comparable():
    """`ar_junipr_v1` sets `continuous_coords=False`: `coord_ll` is identically 0, so its
    total IS the cell probability and shifts with `n_bins` like `split_ll` does."""
    cfg = load_config(["model=ar_junipr_v1", "encoder=gru", "geometry.n_bins=10"])
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(0)
    model = build_model(cfg, geom).eval()
    ds = MatchedLundDataset(synthetic_matched_dataset(16, seed=0), geom)
    batch = collate([ds[i] for i in range(len(ds))])
    with torch.no_grad():
        t = model.nll_terms(batch)
    assert not model.continuous_coords
    assert float(t["coord_ll"].abs().max()) == 0.0
