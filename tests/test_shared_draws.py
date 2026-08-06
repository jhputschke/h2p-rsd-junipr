"""`draws_by_jet=` — one sampling pass shared across sections.

`notebooks/prod_test_v0.ipynb` drew K=200 posterior samples for the same 2000-jet tier
FOUR times (occupancy, `run_calibration`, `run_closure`, `collect`), ~24 of its 109
minutes, and no section shared with another (docs/PLAN_prod_test_speedup.md §4). The
optional argument added to `run_closure`, `run_calibration` and `collect` lets a caller
draw once and hand the same draws to every consumer.

Two properties matter and are pinned here:

* **it really does not re-sample** — asserted by making `sample_batch` raise, which is
  the only way to catch a helper that accepts the draws and then quietly draws its own;
* **the default is untouched** — every new argument is optional, so
  `h2p-rsd-junipr eval` and every existing caller keep today's behaviour.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.eval.calibration import run_calibration
from h2p_rsd_junipr.eval.closure import run_closure
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

DEV = torch.device("cpu")
K, N = 12, 10


@pytest.fixture
def tiny(small_jets):
    cfg = load_config(["model=ar_junipr_v3", "data.n_jets=64"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(DEV).eval()
    jets = small_jets[:N]
    return model, MatchedLundDataset(jets, geom), jets, geom


def _draws(model, ds, n=N, k=K):
    torch.manual_seed(0)
    out = []
    for i in range(n):
        item = ds[i]
        out.append(model.sample_batch(item["xf"].unsqueeze(0).to(DEV),
                                      torch.tensor([item["nx"]], device=DEV), k))
    return out


def _forbid_sampling(model, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("re-sampled despite being handed draws_by_jet")

    monkeypatch.setattr(model, "sample_batch", boom)
    monkeypatch.setattr(model, "sample", boom)


def test_run_closure_reuses_the_draws_it_is_given(tiny, monkeypatch):
    model, ds, jets, geom = tiny
    draws = _draws(model, ds)
    _forbid_sampling(model, monkeypatch)
    m = run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False,
                    continuous=True, draws_by_jet=draws)
    assert m["n_jets_scored"] == N
    assert np.isfinite(m["dlund_posterior_medoid"])


def test_run_closure_shared_draws_reproduce_the_cell_level_numbers(tiny):
    """Handing over the very draws the helper would have drawn itself must give the
    same cell-level answer — the argument is a plumbing change, not a metric change.
    (The `*_cont` keys are excluded: those draw coordinates, and the batched
    coordinate hook consumes the RNG in a different order by design.)"""
    model, ds, jets, geom = tiny
    torch.manual_seed(0)
    ref = run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False)
    got = run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False,
                      draws_by_jet=_draws(model, ds))
    for key, v in ref.items():
        if isinstance(v, float) and np.isfinite(v):
            assert got[key] == pytest.approx(v), key


def test_run_calibration_reuses_the_draws_it_is_given(tiny, monkeypatch):
    model, ds, _, geom = tiny
    draws = _draws(model, ds)
    _forbid_sampling(model, monkeypatch)
    m = run_calibration(model, ds, geom, DEV, K=K, n_jets=N, verbose=False,
                        draws_by_jet=draws)
    assert m["n_jets"] == N and np.isfinite(m["sbc_chi2_uniform"])


def test_run_calibration_shared_draws_reproduce_the_metrics(tiny):
    model, ds, _, geom = tiny
    torch.manual_seed(0)
    ref = run_calibration(model, ds, geom, DEV, K=K, n_jets=N, verbose=False)
    got = run_calibration(model, ds, geom, DEV, K=K, n_jets=N, verbose=False,
                          draws_by_jet=_draws(model, ds))
    for key, v in ref.items():
        if isinstance(v, float) and np.isfinite(v):
            assert got[key] == pytest.approx(v), key


def test_collect_reuses_the_draws_it_is_given(tiny, monkeypatch):
    """`scripts/leading_estimators.collect` is imported by path from the notebook, so
    it is exercised the same way here."""
    spec = importlib.util.spec_from_file_location(
        "leading_estimators", Path(__file__).resolve().parents[1] / "scripts" / "leading_estimators.py")
    le = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(le)

    model, ds, jets, geom = tiny
    draws = _draws(model, ds)
    _forbid_sampling(model, monkeypatch)
    R, C = le.collect(model, ds, jets, geom, DEV, n_jets=N, K=K, n_cont=N,
                      draws_by_jet=draws)
    assert len(R) > 0 and R.shape[1] == 7


def test_short_draws_fail_loudly(tiny):
    """A truncated list is a silent mis-pairing of jets to draws — every number would
    still come out, describing the wrong jets."""
    model, ds, jets, geom = tiny
    short = _draws(model, ds, n=N - 3)
    with pytest.raises(ValueError, match="draws_by_jet"):
        run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False,
                    draws_by_jet=short)
    with pytest.raises(ValueError, match="draws_by_jet"):
        run_calibration(model, ds, geom, DEV, K=K, n_jets=N, verbose=False,
                        draws_by_jet=short)


def test_default_is_none_everywhere(tiny):
    """The CLI must be untouched: every new argument is keyword-optional and defaults
    to today's behaviour."""
    import inspect

    for fn in (run_closure, run_calibration):
        assert inspect.signature(fn).parameters["draws_by_jet"].default is None


# ---------------------------------------------------------------------------
# WP-3 — `coords_by_jet=`, the coordinate half (docs/PLAN_z_aware.md §4/WP-3, §7.1)
#
# The largest risk in that work package: `run_closure`'s existing continuous call is
# FILTERED (`[list(d) for d in draws if len(d)]`) and `sample_coordinates_many` pads to
# `L_max` over the list it is handed, so an unfiltered call changes the block shape,
# reorders RNG consumption, and moves `dlund_*_cont` and the psi block. It has to live
# strictly behind the switch, and this is where that is pinned.
# ---------------------------------------------------------------------------
def _coords(model, ds, draws, n=N):
    from h2p_rsd_junipr.inference.mbr import coords_for_draws

    out = []
    for i in range(n):
        item = ds[i]
        out.append(coords_for_draws(model, item["xf"].unsqueeze(0).to(DEV),
                                    torch.tensor([item["nx"]], device=DEV), draws[i]))
    return out


def test_run_closure_reuses_the_coordinates_it_is_given(tiny, monkeypatch):
    """Handed `coords_by_jet`, it must not draw a second table — asserted by making the
    sampler raise, which is the only way to catch a helper that accepts them and then
    quietly draws its own."""
    model, ds, jets, geom = tiny
    draws = _draws(model, ds)
    coords = _coords(model, ds, draws)

    def boom(*a, **kw):
        raise AssertionError("re-drew coordinates despite being handed coords_by_jet")

    monkeypatch.setattr(model, "sample_coordinates_many", boom)
    m = run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False,
                    continuous=True, draws_by_jet=draws, coords_by_jet=coords)
    assert m["n_jets_scored"] == N
    assert np.isfinite(m["dlund_posterior_geomedian_cont"])


def test_with_the_switch_off_the_filtered_call_is_still_taken(tiny, monkeypatch):
    """The bit-identity hazard, pinned at the call site rather than at the numbers.

    With `mbr_cloud_source` at its default the continuous block must still hand
    `sample_coordinates_many` the FILTERED list — the empty draws dropped — because that
    is what every committed `dlund_*_cont` was produced under. Under `"coords"` the same
    block gets the unfiltered table, which is a different `L_max` and therefore a
    different RNG stream: correct, deliberate, and exactly why it is opt-in."""
    model, ds, jets, geom = tiny
    draws = _draws(model, ds)
    seen: list[list[int]] = []
    original = model.sample_coordinates_many

    def spy(xf, nx, d, **kw):
        seen.append([len(x) for x in d])
        return original(xf, nx, d, **kw)

    monkeypatch.setattr(model, "sample_coordinates_many", spy)
    run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False,
                continuous=True, draws_by_jet=draws)
    assert seen, "the continuous block did not call the coordinate sampler at all"
    for lens, d in zip(seen, draws):
        assert lens == [len(x) for x in d if len(x)], "the call was NOT filtered"


def test_short_coords_fail_loudly(tiny):
    """Same contract as `draws_by_jet`: a truncated list is a silent mis-pairing."""
    model, ds, jets, geom = tiny
    draws = _draws(model, ds)
    with pytest.raises(ValueError, match="coords_by_jet"):
        run_closure(model, ds, jets, geom, DEV, K=K, n_closure=N, verbose=False,
                    continuous=True, draws_by_jet=draws,
                    coords_by_jet=_coords(model, ds, draws, n=N - 3))


def test_coords_by_jet_defaults_to_none(tiny):
    import inspect

    assert inspect.signature(run_closure).parameters["coords_by_jet"].default is None
    from h2p_rsd_junipr.eval.clusters import run_cluster_diagnostics

    assert (inspect.signature(run_cluster_diagnostics)
            .parameters["coords_by_jet"].default is None)
