"""The leading-emission point estimators (`eval/closure.py`).

`run_closure` used to report only `dlund_posterior_mode` — the MODAL leading cell,
which minimises expected 0-1 loss while the score is `lund_distance`. That mismatch,
not the model, is what made plain RSD look unbeatable on the most perturbative
observable in the jet. `medoid_cell` is the loss-matched estimator over the same
support and `geometric_median` the unrestricted one; both are pinned here, along with
the additive-keys contract that keeps old eval numbers readable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import experiment_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.eval.closure import (
    geometric_median,
    leading_emission_cell,
    lund_distance,
    medoid_cell,
    run_closure,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model


def _risk(cell, draws, geom):
    """Mean Lund distance from `cell` to the drawn leading cells — what the medoid
    minimises and what the reported score measures."""
    return float(np.mean([lund_distance(cell, c, geom) for c in draws]))


# --- the estimators -------------------------------------------------------------
def test_medoid_minimises_the_reported_risk_over_the_drawn_support():
    """The defining property: no drawn cell has lower mean distance to the draws.
    This is what makes the medoid never worse than the mode under the model's own
    posterior — the mode has no such guarantee for a distance-valued score."""
    geom = Geometry()
    rng = np.random.default_rng(0)
    for _ in range(20):
        draws = [int(c) for c in rng.integers(0, geom.n_cells, size=40)]
        med = medoid_cell(draws, geom)
        best = _risk(med, draws, geom)
        assert all(_risk(int(c), draws, geom) >= best - 1e-9 for c in set(draws))


def test_medoid_beats_the_mode_when_the_mode_is_an_outlier():
    """A concrete failure of the mode: a plurality lump far from a broad bulk that
    carries most of the mass. The mode follows the lump; the medoid follows the mass."""
    geom = Geometry()
    lump = geom.to_cell(0.3, 0.3)                       # tight plurality, cornered
    bulk = [geom.to_cell(u, v) for u in (2.8, 3.0, 3.2) for v in (2.8, 3.0, 3.2)]
    draws = [lump] * 12 + [c for c in bulk for _ in range(3)]   # 12 vs 27, spread out
    vals, counts = np.unique(np.asarray(draws), return_counts=True)
    assert int(vals[counts.argmax()]) == lump           # the mode really is the lump
    assert _risk(medoid_cell(draws, geom), draws, geom) < _risk(lump, draws, geom)


def test_medoid_and_geometric_median_handle_degenerate_input():
    geom = Geometry()
    assert medoid_cell([], geom) is None
    assert medoid_cell([7], geom) == 7                  # single draw is its own medoid
    p = np.array([[1.0, 2.0]])
    assert np.allclose(geometric_median(p), [1.0, 2.0])


def test_geometric_median_is_the_l1_optimum_not_the_mean():
    """Weiszfeld beats the mean on total L1 distance, and resists a single outlier
    that drags the mean — the reason it, not the mean, is the reported estimator."""
    P = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [9.0, 9.0]])
    gm, mu = geometric_median(P), P.mean(0)
    l1 = lambda a: float(np.linalg.norm(P - a, axis=1).sum())  # noqa: E731
    assert l1(gm) <= l1(mu) + 1e-9
    assert np.linalg.norm(gm) < np.linalg.norm(mu)      # not dragged to the outlier


# --- the run_closure contract ---------------------------------------------------
@pytest.fixture
def tiny(small_jets):
    cfg = load_config(["model=ar_junipr_v3", "data.n_jets=64"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(torch.device("cpu"))
    model.eval()
    jets = small_jets[:24]
    return model, MatchedLundDataset(jets, geom), jets, geom


def test_medoid_is_reported_unconditionally_and_old_keys_survive(tiny):
    """It is pure numpy over cells already drawn, so it costs nothing and is always
    on — but `dlund_posterior_mode` must remain, since scripts/ab_v2_v3.py and
    docs/PLAN_ProductionAssessment.md read it."""
    model, ds, jets, geom = tiny
    m = run_closure(model, ds, jets, geom, torch.device("cpu"), K=16, n_closure=12,
                    verbose=False)
    assert "dlund_identity" in m and "dlund_posterior_mode" in m
    assert np.isfinite(m["dlund_posterior_medoid"])
    # continuous is opt-in: asking for nothing must add nothing
    assert not any(k.endswith("_cont") for k in m)


def test_continuous_keys_are_additive_and_off_the_grid(tiny):
    """`continuous=True` only ADDS keys (the WP2 contract), and the continuous
    distances are genuinely un-quantised — a cell-centre distance is a multiple of
    the cell pitch, these are not."""
    model, ds, jets, geom = tiny
    dev = torch.device("cpu")
    base = run_closure(model, ds, jets, geom, dev, K=16, n_closure=12, verbose=False)
    got = run_closure(model, ds, jets, geom, dev, K=16, n_closure=12, verbose=False,
                      continuous=True)
    assert set(base) <= set(got)
    for k in ("dlund_identity_cont", "dlund_posterior_mode_cont",
              "dlund_posterior_geomedian_cont"):
        assert np.isfinite(got[k])
    assert got["n_continuous_jets"] > 0


def test_continuous_degrades_to_nan_without_a_coordinate_density(tiny):
    """ar_junipr_v1 returns None from `sample_coordinates`. The keys must still be
    present and NaN — "asked, unavailable" is a different fact from "never asked"."""
    _, _, jets, _ = tiny
    cfg = load_config(["model=ar_junipr_v1", "data.n_jets=64"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(torch.device("cpu"))
    model.eval()
    ds = MatchedLundDataset(jets, geom)
    assert model.sample_coordinates(ds[0]["xf"].unsqueeze(0),
                                    torch.tensor([ds[0]["nx"]]), [0]) is None
    m = run_closure(model, ds, jets, geom, torch.device("cpu"), K=16, n_closure=12,
                    verbose=False, continuous=True)
    assert np.isnan(m["dlund_posterior_geomedian_cont"])
    assert m["n_continuous_jets"] == 0


def test_closure_continuous_switch_defaults_off_and_backfills():
    """Same tolerance contract as the WP2 switches: a snapshot written before this
    field evaluates with it off rather than crashing."""
    from omegaconf import OmegaConf

    assert experiment_params(load_config([]))["closure_continuous"] is False
    assert load_config(["experiment.closure_continuous=true"]).experiment.closure_continuous
    old = OmegaConf.create({"experiment": {"name": "closure", "closure_jets": 7}})
    got = experiment_params(old)
    assert got["closure_jets"] == 7 and got["closure_continuous"] is False


def test_leading_emission_cell_agrees_with_the_medoid_helpers_on_types(tiny):
    """`medoid_cell` must accept exactly what `leading_emission_cell` produces over
    a batch of draws (python ints), not just numpy arrays."""
    model, ds, _, geom = tiny
    item = ds[0]
    draws = model.sample_batch(item["xf"].unsqueeze(0), torch.tensor([item["nx"]]), 16)
    lead = [c for c in (leading_emission_cell(d, geom) for d in draws) if c is not None]
    if not lead:
        pytest.skip("no non-empty draw in this sample")
    assert isinstance(medoid_cell(lead, geom), int)
