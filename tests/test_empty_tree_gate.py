"""The empty-tree decision (docs/PLAN_empty_parton_tree.md).

The parton target really is the empty tree for ~17% of jets, and no point estimator
under the default decode can say so: the MAP is `argmax_n q(n|x)`, whose peak lands at
0 essentially never however much mass sits there, and MBR's perturbative-Lund EMD
charges an imbalance penalty that makes an empty cloud near-maximal risk. Both are
properties of the DECODE, not of the fit. `decode.empty_threshold` adds the decision
before either shape decode.

The test that matters most is parity: `empty_threshold = 0.0` must reproduce today's
point estimate exactly, for every family.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import empty_gate, empty_threshold_for_rate
from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate
from h2p_rsd_junipr.models.base import build_model

FAMILIES = ["ar_junipr_v1", "ar_junipr_v2", "ar_junipr_v3", "ar_junipr_v4", "cinn"]


def _model(name, small_jets):
    cfg = load_config([f"model={name}", "data.n_jets=64"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(torch.device("cpu"))
    model.eval()
    ds = MatchedLundDataset(small_jets[:16], geom)
    return model, ds, decode_params(cfg)


def _jet(ds, i=0):
    item = ds[i]
    return item["xf"].unsqueeze(0), torch.tensor([item["nx"]])


# --- empty_gate ------------------------------------------------------------------
def test_empty_gate_off_by_default_whatever_the_pmf():
    """`tau <= 0` is the off sentinel — it must not fire even on a pmf that is all
    mass at n=0, or the default decode would silently change."""
    certain = np.array([1.0, 0.0, 0.0])
    assert empty_gate(certain, 0.0) is False
    assert empty_gate(certain, -1.0) is False
    assert empty_gate(certain, 0.5) is True


def test_empty_gate_boundary_is_inclusive():
    """`>=`, so the tau returned by `empty_threshold_for_rate` fires on the very jet
    that defined it — otherwise the fitted rate would be short by one."""
    pmf = np.array([0.25, 0.75])
    assert empty_gate(pmf, 0.25) is True
    assert empty_gate(pmf, np.nextafter(0.25, 1.0)) is False


def test_empty_gate_handles_a_degenerate_pmf():
    assert empty_gate(np.zeros(0), 0.5) is False
    assert empty_gate([0.9], 0.5) is True


# --- empty_threshold_for_rate ----------------------------------------------------
@pytest.mark.parametrize("rate", [0.05, 0.17, 0.4, 0.9])
def test_threshold_reproduces_its_fitted_rate(rate):
    """The headline property: fit tau for a target rate, and gating at it on the same
    jets reproduces that rate to within one jet."""
    rng = np.random.default_rng(0)
    pmfs = [np.array([q, 1.0 - q]) for q in rng.uniform(0.01, 0.99, size=500)]
    tau = empty_threshold_for_rate(pmfs, rate)
    got = np.mean([empty_gate(p, tau) for p in pmfs])
    assert abs(got - rate) <= 1.0 / len(pmfs) + 1e-12


def test_threshold_thresholds_the_ranking_not_the_scale():
    """A monotone rescaling of q(0|x) — exactly the ~2x under-confidence the plan
    measures (F5) — must select the SAME jets, since only the ordering matters."""
    rng = np.random.default_rng(1)
    q = rng.uniform(0.01, 0.45, size=300)
    pmfs = [np.array([v, 1 - v]) for v in q]
    squashed = [np.array([v / 2.0, 1 - v / 2.0]) for v in q]     # under-confident by 2x
    picked = [empty_gate(p, empty_threshold_for_rate(pmfs, 0.2)) for p in pmfs]
    picked_sq = [empty_gate(p, empty_threshold_for_rate(squashed, 0.2)) for p in squashed]
    assert picked == picked_sq


def test_threshold_rate_zero_never_fires_and_empty_input_is_safe():
    pmfs = [np.array([0.9, 0.1]), np.array([0.2, 0.8])]
    assert not any(empty_gate(p, empty_threshold_for_rate(pmfs, 0.0)) for p in pmfs)
    assert empty_threshold_for_rate([], 0.5) == float("inf")


def test_threshold_never_returns_the_off_sentinel():
    """A tau of exactly 0.0 would read as "gate off" and silently disable the stage.
    All-zero q(0|x) is the case that would produce it."""
    pmfs = [np.array([0.0, 1.0])] * 10
    tau = empty_threshold_for_rate(pmfs, 0.5)
    assert tau > 0.0
    assert not any(empty_gate(p, tau) for p in pmfs)   # q0 == 0 is never called empty


# --- the decode stage ------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_threshold_zero_is_exact_parity(family, small_jets):
    """`empty_threshold = 0.0` reproduces today's point estimate EXACTLY — the whole
    opt-in contract. Checked for both estimators on every family."""
    model, ds, dec = _model(family, small_jets)
    xf, nx = _jet(ds)
    torch.manual_seed(0)
    draws = model.sample_batch(xf, nx, 8)
    for est in ("map", "mbr"):
        if est == "mbr":
            pytest.importorskip("ot")
        d = {**dec, "point_estimator": est}
        a = model.map_or_mbr(xf, nx, draws=draws, **d)
        b = model.map_or_mbr(xf, nx, draws=draws, **{**d, "empty_threshold": 0.0})
        assert a.multiplicity == b.multiplicity
        assert [n.cell for n in a.nodes] == [n.cell for n in b.nodes]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_fired_gate_returns_a_well_formed_empty_estimate(family, small_jets):
    """tau just above 0 fires on any jet with non-zero q(0|x). The result must be a
    real LundPointEstimate — consumers assuming `multiplicity >= 1` are what the plan
    warns about, so the type contract has to hold."""
    model, ds, dec = _model(family, small_jets)
    xf, nx = _jet(ds)
    torch.manual_seed(0)
    draws = model.sample_batch(xf, nx, 8)
    pmf = model.length_pmf(xf, nx, mults=[len(d) for d in draws])
    if not (pmf.size and pmf[0] > 0.0):
        pytest.skip(f"{family}: q(0|x) == 0 on this jet, nothing to gate")
    hat = model.map_or_mbr(xf, nx, draws=draws,
                           **{**dec, "empty_threshold": float(pmf[0])})
    assert isinstance(hat, LundPointEstimate)
    assert hat.multiplicity == 0 and hat.nodes == []
    assert np.isfinite(hat.logprob)


def test_the_gate_beats_min_emissions_zero(small_jets):
    """The distinction the plan is built on: lifting `min_emissions` cannot produce the
    empty tree, because with a multiplicity head the MAP is argmax_n q(n|x) and the
    clamp never binds. Only the gate can."""
    model, ds, dec = _model("ar_junipr_v3", small_jets)
    xf, nx = _jet(ds)
    torch.manual_seed(0)
    pmf = model.length_pmf(xf, nx)
    if int(np.argmax(pmf)) == 0:
        pytest.skip("untrained head happens to peak at n=0; nothing to distinguish")
    unfloored = model.map_or_mbr(xf, nx, **{**dec, "min_emissions": 0})
    assert unfloored.multiplicity > 0                      # the floor was not the issue
    gated = model.map_or_mbr(xf, nx, **{**dec, "empty_threshold": float(pmf[0])})
    assert gated.multiplicity == 0


# --- check 7: every consumer must tolerate the empty tree once the gate can fire ----
def _always_empty(dec):
    """A tau small enough to fire on every jet — the audit condition."""
    return {**dec, "empty_threshold": 1e-12}


def test_print_point_estimate_honours_the_gate(small_jets, capsys):
    """It used to call `map_estimate` directly, which structurally cannot return the
    empty tree. With the gate on, this block printed a non-empty MAP for the very jets
    `run_closure`'s `p_empty_pred` had just counted as empty — one `eval`, two answers."""
    from h2p_rsd_junipr.eval.closure import print_point_estimate

    model, ds, dec = _model("ar_junipr_v4", small_jets)
    geom = Geometry.from_config(load_config(["model=ar_junipr_v4"]).geometry)
    print_point_estimate(model, ds, small_jets[:16], geom, torch.device("cpu"),
                         n_samples=8, decode=_always_empty(dec))
    out = capsys.readouterr().out
    assert "model MAP = 0" in out, out
    assert "(empty)" in out, "the MAP tree block must render the empty tree, not nothing"


def test_generator_spread_takes_a_decode_and_defaults_to_the_old_behaviour(small_jets):
    """The "dominant systematic" was measured under `map_estimate()`'s signature
    defaults — no `min_emissions`, no gate, whatever the run was configured with."""
    from h2p_rsd_junipr.eval.systematics import generator_spread

    model_a, ds, dec = _model("ar_junipr_v4", small_jets)
    torch.manual_seed(11)
    model_b, _, _ = _model("ar_junipr_v4", small_jets)
    geom = Geometry.from_config(load_config(["model=ar_junipr_v4"]).geometry)
    dev = torch.device("cpu")

    base = generator_spread(model_a, model_b, ds, geom, dev, n_jets=6, verbose=False)
    assert base["empty_threshold"] == 0.0            # decode=None == the old defaults
    assert base["point_estimator"] == "map"

    # both models gated empty => they agree exactly, so the spread collapses to 0
    gated = generator_spread(model_a, model_b, ds, geom, dev, n_jets=6, verbose=False,
                             decode=_always_empty(dec))
    assert gated["empty_threshold"] == 1e-12
    assert gated["mult_spread_mean"] == 0.0
    assert np.isnan(gated["lead_lund_spread_mean"]), (
        "two empty trees have no leading emission to compare; the spread must be NaN, "
        "not a silent 0 that reads as perfect agreement"
    )


def test_serving_predict_and_run_closure_agree_under_the_gate(small_jets):
    """The two consumers that already honoured the gate must keep doing so, and must
    report the SAME emptiness for the same jet."""
    from h2p_rsd_junipr.eval.closure import run_closure
    from h2p_rsd_junipr.serving.api import predict

    model, ds, dec = _model("ar_junipr_v4", small_jets)
    geom = Geometry.from_config(load_config(["model=ar_junipr_v4"]).geometry)
    dev = torch.device("cpu")
    gate = _always_empty(dec)

    r = run_closure(model, ds, small_jets[:16], geom, dev, K=8, n_closure=8,
                    verbose=False, decode=gate)
    assert r["p_empty_pred"] == 1.0

    x = {"lnInvDelta": [0.3, 1.3], "lnkt": [4.7, 4.4], "lnz": [-1.1, -0.2],
         "psi": [-3.0, -2.8]}
    out = predict(model, geom, dev, x, decode=gate)
    assert out["map_multiplicity"] == 0 and out["map_nodes"] == []


def test_decode_params_backfills_on_a_pre_field_snapshot():
    """A checkpoint written before this field must evaluate with the gate off rather
    than crash — the same tolerance contract as every other decode knob."""
    from omegaconf import OmegaConf

    assert decode_params(load_config([]))["empty_threshold"] == 0.0
    old = OmegaConf.create({"decode": {"beam_width": 4, "min_emissions": 1}})
    got = decode_params(old)
    assert got["beam_width"] == 4 and got["empty_threshold"] == 0.0
    assert load_config(["decode.empty_threshold=0.17"]).decode.empty_threshold == 0.17
