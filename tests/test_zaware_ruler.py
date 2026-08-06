"""The `ln z`-aware closure ruler (`eval/closure.py`, docs/PLAN_z_aware.md WP-1).

`dlund_mbr` — the decode headline, and the row that regressed on 4/4 arms when the
RQ-spline `ln z` head was fielded — is the Euclidean distance between leading-emission
**cell centres**. It is blind to `ln z` *and* to the within-cell `du`/`dv` offsets, and
the continuous block carried no MBR row at all. So the ruler that scored the spline could
not register what the spline improved, on either axis.

Three properties are pinned here, and the third is the one that makes the measurement
readable at all:

* the new series measure what they claim — a hand-computed 3-D value, the 2-D component
  and `|d ln z|` off the SAME emission;
* **unavailable is NaN, never 0** — a modal *cell* has no `ln z`, and `ln z = 0` means
  `z = 1`, the softer prong taking the whole jet;
* **additive only** — `decode_headline` and every pre-existing key keep their exact
  meaning and value, so the new ruler cannot perturb the statistic it exists to explain.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.eval.closure import (
    _dist3,
    _leading_coords,
    _leading_row,
    _mbr_leading_row,
    run_closure,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.point_estimate import LundNode, LundPointEstimate
from h2p_rsd_junipr.models.base import build_model

POT_OK = importlib.util.find_spec("ot") is not None
DEV = torch.device("cpu")

# (u, v, ln z, psi) rows; the hardest-kt row is the middle one.
TABLE = np.array([
    [1.0, 2.0, -1.0, 0.1],
    [2.0, 5.0, -2.0, 0.2],
    [3.0, 4.0, -1.5, 0.3],
])


# --- the selectors --------------------------------------------------------------
def test_leading_row_is_the_hardest_kt_row_whole():
    assert np.allclose(_leading_row(TABLE), [2.0, 5.0, -2.0, 0.2])
    assert _leading_row(np.zeros((0, 4))) is None
    assert _leading_row(np.zeros(4)) is None          # not a table


def test_leading_coords_defaults_to_two_columns():
    """The default is what keeps every pre-existing caller unchanged BY CONSTRUCTION
    rather than by review."""
    assert np.allclose(_leading_coords(TABLE), [2.0, 5.0])
    assert _leading_coords(TABLE).shape == (2,)


def test_leading_coords_three_columns_come_off_the_same_row():
    """`(u, v, ln z)` from ONE argmax, not three selections — otherwise the 2-D and 3-D
    rulers could be scoring different emissions and their difference would not be a
    difference of rulers."""
    three = _leading_coords(TABLE, 3)
    assert np.allclose(three, [2.0, 5.0, -2.0])
    assert np.allclose(three[:2], _leading_coords(TABLE))


# --- the distance triple --------------------------------------------------------
def test_dist3_is_hand_computable():
    a = np.array([1.0, 2.0, -1.0])
    b = np.array([4.0, 6.0, -3.0])
    d3, d2, dz = _dist3(a, b)
    assert d2 == pytest.approx(5.0)                       # 3-4-5
    assert dz == pytest.approx(2.0)
    assert d3 == pytest.approx(np.sqrt(25.0 + 4.0))       # the 2-D leg and `ln z` in quadrature
    assert d3 == pytest.approx(np.hypot(d2, dz))


def test_dist3_propagates_a_missing_ln_z_as_nan_and_keeps_the_plane_leg():
    """A cell centre has a real `(u, v)` and no `ln z` at all. The 2-D component must
    survive that — the mode's `dlund_*_cont` row is not in question — while the two
    `ln z` rulers go NaN rather than scoring a placeholder."""
    a = np.array([1.0, 2.0, np.nan])
    b = np.array([4.0, 6.0, -3.0])
    d3, d2, dz = _dist3(a, b)
    assert d2 == pytest.approx(5.0)
    assert np.isnan(d3) and np.isnan(dz)
    assert all(np.isnan(x) for x in _dist3(None, b))


# --- the MBR row ----------------------------------------------------------------
def _pe(coords_source, lnz=(-1.0, -2.0)):
    nodes = [
        LundNode(depth=t, parent=t - 1, cell=t, ln_invDelta=float(t), ln_kt=float(t),
                 ln_z=float(z), psi=0.0, kt=1.0, delta_R=1.0, z=1.0,
                 logp_split=0.0, logp_coord=0.0, logp_cont=0.0)
        for t, z in enumerate(lnz)
    ]
    return LundPointEstimate(nodes=nodes, logprob=0.0, multiplicity=len(nodes),
                             coords_source=coords_source)


def test_mbr_row_is_scored_only_when_the_coordinates_were_drawn():
    """`map_or_mbr(point_estimator="mbr")` reaches `describe_cells`, which DRAWS from
    `q(coords | cells, x)` and stamps `coords_source="sample"` — so the MBR row is a
    genuine posterior sample of `ln z` today, with no plumbing. Under `ar_junipr_v1`, or
    any `cell_center` fallback, `ln z` is the placeholder 0 and must not be scored."""
    assert np.allclose(_mbr_leading_row(_pe("sample")), [1.0, 1.0, -2.0])
    assert _mbr_leading_row(_pe("cell_center")) is None
    assert _mbr_leading_row(_pe("mode")) is None
    assert _mbr_leading_row(None) is None


# --- run_closure ----------------------------------------------------------------
@pytest.fixture
def tiny(small_jets):
    cfg = load_config(["model=ar_junipr_v4", "encoder=gru", "data.n_jets=64",
                       "decode.point_estimator=mbr", "decode.mbr_backend=pot",
                       "decode.mbr_n_candidates=6", "decode.min_emissions=0"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(DEV).eval()
    jets = small_jets[:24]
    return model, MatchedLundDataset(jets, geom), jets, geom, decode_params(cfg)


def _closure(model, ds, jets, geom, dec, **kw):
    """One `run_closure` from a reset RNG state — BOTH streams.

    `decode_generator` is private to the decode layer and advances per call, persisting on
    the model, so two runs in one process are not comparable on `torch.manual_seed` alone.
    That separation is deliberate (`models/base.py`: a point estimate must not change
    which posterior draws the NEXT jet gets), and it is exactly why an equality assertion
    across two runs has to reset it explicitly rather than assume it."""
    torch.manual_seed(0)
    model.__dict__.pop("_decode_generators", None)
    return run_closure(model, ds, jets, geom, DEV, K=12, n_closure=16, verbose=False,
                       decode=dec, continuous=True, **kw)


def _same(a, b) -> bool:
    """Structural equality with `NaN == NaN` — the metric dict is full of honest NaNs."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            _same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


NEW_KEYS = (
    "dlund_mbr_cont",
    "dlund3_identity_cont", "dlund3_posterior_mode_cont",
    "dlund3_posterior_geomedian_cont", "dlund3_mbr_cont",
    "dlnz_identity", "dlnz_posterior_mode",
    "dlnz_posterior_geomedian", "dlnz_mbr",
)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_the_new_ruler_is_additive_and_the_headline_does_not_move(tiny):
    """The WP-1 contract in one test. A switch that perturbs the statistic it exists to
    explain is worse than no switch (same discipline as `coverage_null_reps`' fork_rng)."""
    model, ds, jets, geom, dec = tiny
    base = _closure(model, ds, jets, geom, dec)
    assert set(NEW_KEYS) <= set(base)
    assert base["decode_headline"] == "dlund_mbr"
    # the pre-existing continuous rows are still there and still 2-D
    for k in ("dlund_identity_cont", "dlund_posterior_mode_cont",
              "dlund_posterior_geomedian_cont", "n_continuous_jets"):
        assert k in base
    # ...and no `dlund3_*` / `dlnz_*` key appears without `continuous=True`
    torch.manual_seed(0)
    off = run_closure(model, ds, jets, geom, DEV, K=12, n_closure=16, verbose=False,
                      decode=dec)
    assert not any(k.startswith("dlund3_") or k.startswith("dlnz_") for k in off)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_the_modal_cell_has_no_ln_z_and_says_so(tiny):
    """NaN, not 0, and not silently absent: the modal CELL is a legitimate 2-D estimator
    with no third coordinate, so its two `ln z` rows are unscorable by construction and
    `n_continuous_scored` says how many jets each ruler actually scored."""
    model, ds, jets, geom, dec = tiny
    m = _closure(model, ds, jets, geom, dec)
    assert np.isnan(m["dlund3_posterior_mode_cont"])
    assert np.isnan(m["dlnz_posterior_mode"])
    assert np.isfinite(m["dlund_posterior_mode_cont"])     # the plane row is unaffected
    scored = m["n_continuous_scored"]
    assert scored["dlund3_posterior_mode"] == 0 and scored["dlnz_posterior_mode"] == 0
    assert scored["dlund3_posterior_geomedian"] == m["n_continuous_jets"]


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_a_fixed_point_estimate_puts_its_three_rulers_in_a_right_triangle(tiny):
    """For identity and MBR the estimator is ONE point, so the three rulers are the
    hypotenuse and its two legs on the same emission: `dlund3 = hypot(dlund_cont, dlnz)`
    exactly. That is what makes them readable together."""
    model, ds, jets, geom, dec = tiny
    m = _closure(model, ds, jets, geom, dec, per_jet=True)
    seen = 0
    for row in m["per_jet"]:
        for name, two in (("identity", "dlund_identity_cont"),
                          ("mbr", "dlund_mbr_cont")):
            d3, d2, dz = (row.get(f"dlund3_{name}_cont"), row.get(two),
                          row.get(f"dlnz_{name}"))
            if not all(v is not None and np.isfinite(v) for v in (d3, d2, dz)):
                continue
            assert d3 == pytest.approx(float(np.hypot(d2, dz)), rel=1e-9, abs=1e-9)
            seen += 1
    assert seen > 0


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_the_3d_geometric_median_is_a_3d_estimator_not_a_2d_one_relabelled(tiny):
    """The geometric median is the one estimator that MOVES when the ruler changes:
    `geometric_median` is dimension-agnostic, so the 3-D row is the argmin over
    `(u, v, ln z)` and its plane components are NOT the 2-D median's. The right-triangle
    identity above therefore fails for it — deliberately, and `dlund3` can even sit
    *below* `dlund_cont`, because the 3-D Bayes point trades plane accuracy for `ln z`.

    Pinned because the cheap alternative (score the 2-D median's `ln z`) would have
    produced a `dlund3` that is a relabelling rather than a measurement."""
    model, ds, jets, geom, dec = tiny
    m = _closure(model, ds, jets, geom, dec, per_jet=True)
    differ = 0
    for row in m["per_jet"]:
        d3 = row.get("dlund3_posterior_geomedian_cont")
        d2 = row.get("dlund_posterior_geomedian_cont")
        dz = row.get("dlnz_posterior_geomedian")
        if not all(v is not None and np.isfinite(v) for v in (d3, d2, dz)):
            continue
        differ += int(abs(d3 - float(np.hypot(d2, dz))) > 1e-6)
    assert differ > 0


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_per_jet_rows_are_off_by_default_and_cover_every_scored_jet(tiny):
    """WP-0 consumes these rows. One per scored jet INCLUDING the truth-empty ones the
    leading-emission selection drops — a paired analysis pairs on the jet index, and a
    row that is silently absent is indistinguishable from one the other arm also
    dropped."""
    model, ds, jets, geom, dec = tiny
    off = _closure(model, ds, jets, geom, dec)
    assert "per_jet" not in off
    on = _closure(model, ds, jets, geom, dec, per_jet=True)
    rows = on["per_jet"]
    assert [r["jet"] for r in rows] == list(range(on["n_jets_scored"]))
    assert sum(r["kept"] for r in rows) == on["n_kept_leading"]
    # the aggregates ARE the mean of the rows — the rows are not a second measurement
    for key in ("dlund_identity", "dlund_mbr", "dlund3_mbr_cont", "dlnz_mbr"):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        if vals and np.isfinite(on[key]):
            assert float(np.mean(vals)) == pytest.approx(on[key], rel=1e-9, abs=1e-12)
    # ...and collecting them must not move a single aggregate
    assert _same({k: v for k, v in on.items() if k != "per_jet"}, off)


def test_new_keys_are_nan_not_zero_without_a_coordinate_density(small_jets):
    """`ar_junipr_v1` returns None from `sample_coordinates`. Every new key must be
    present and NaN — "asked, unavailable" is a different fact from "never asked", and a
    0 here would read as a perfect `ln z` match."""
    cfg = load_config(["model=ar_junipr_v1", "encoder=gru", "data.n_jets=64"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(DEV).eval()
    jets = small_jets[:16]
    ds = MatchedLundDataset(jets, geom)
    m = run_closure(model, ds, jets, geom, DEV, K=8, n_closure=12, verbose=False,
                    continuous=True)
    for k in NEW_KEYS:
        assert k in m, k
        assert np.isnan(m[k]), (k, m[k])
