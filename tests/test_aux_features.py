"""Aux conditioning: registry guards, dataset/collate widths, per-family build +
train step, the synthetic refusal, and the serving round trip (docs/PLAN_Input.md).

The physics A/B lives on the PYTHIA RNTuple path (see
`notebooks/aux_input_ab.ipynb`); everything here is plumbing, so the aux source
columns are INJECTED onto synthetic fixture dicts rather than proxied — a proxy would
be a function of x and would fake the very information gain aux exists to measure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.features import (
    MG_EPS,
    N_NODE_FEAT,
    PT_REF,
    aux_source_fields,
    aux_vector,
    configured_aux_names,
    node_features,
    with_aux,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

AUX = ["ln_mg_pt", "nsec", "ln_pt"]


def _with_aux_columns(jets, *, mg=5.0, nsec=2, pt=120.0):
    """Inject the C++-written aux source columns onto fixture jets."""
    out = []
    for j in jets:
        k = dict(j)
        k.update(jet_pt=pt, x_mg=mg, x_nsec=nsec, generator="pythia-fixture")
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_aux_vector_values_and_order():
    jet = {"jet_pt": 200.0, "x_mg": 20.0, "x_nsec": 3}
    v = aux_vector(jet, AUX)
    assert v.shape == (3,) and v.dtype == np.float32
    assert v[0] == pytest.approx(math.log(20.0 / 200.0), rel=1e-6)
    assert v[1] == pytest.approx(math.log1p(3), rel=1e-6)
    assert v[2] == pytest.approx(math.log(200.0 / PT_REF), rel=1e-6)
    # order follows the configured list, not the registry
    assert aux_vector(jet, ["ln_pt", "ln_mg_pt"]) == pytest.approx(v[[2, 0]])


def test_mg_eps_floors_the_single_prong_zero_mass():
    """A groomed jet with nothing resolved has m_g == 0 exactly: log must not blow up."""
    v = aux_vector({"jet_pt": 100.0, "x_mg": 0.0, "x_nsec": 0}, ["ln_mg_pt"])
    assert v[0] == pytest.approx(math.log(MG_EPS / 100.0), rel=1e-6)
    assert math.isfinite(float(v[0]))


@pytest.mark.parametrize(
    "jet",
    [
        {"jet_pt": 100.0, "x_nsec": 1},                          # x_mg missing
        {"x_mg": 5.0, "x_nsec": 1},                              # jet_pt missing
        {"jet_pt": float("nan"), "x_mg": 5.0, "x_nsec": 1},      # NaN sentinel
        {"jet_pt": 100.0, "x_mg": float("nan"), "x_nsec": 1},    # NaN sentinel
        {"jet_pt": 100.0, "x_mg": 5.0, "x_nsec": -1},            # absent-column sentinel
        {"jet_pt": 0.0, "x_mg": 5.0, "x_nsec": 1},               # unphysical scale
        {"jet_pt": 100.0, "x_mg": -1.0, "x_nsec": 1},            # unphysical mass
    ],
)
def test_aux_vector_rejects_missing_and_sentinel_sources(jet):
    with pytest.raises(ValueError):
        aux_vector(jet, AUX)


def test_unknown_aux_name_raises():
    with pytest.raises(KeyError):
        aux_vector({"jet_pt": 100.0}, ["girth"])
    with pytest.raises(KeyError):
        configured_aux_names(load_config(["encoder.aux_features=[girth]"]).encoder)


def test_configured_aux_names_defaults_off_and_is_getattr_tolerant():
    assert configured_aux_names(load_config([]).encoder) == ()
    assert configured_aux_names(object()) == ()  # pre-aux checkpoint config snapshot
    cfg = load_config([f"encoder.aux_features=[{','.join(AUX)}]"])
    assert configured_aux_names(cfg.encoder) == tuple(AUX)


def test_aux_source_fields_dedupes():
    assert aux_source_fields(AUX) == ("x_mg", "jet_pt", "x_nsec")


# ---------------------------------------------------------------------------
# dataset / collate widths
# ---------------------------------------------------------------------------
def test_dataset_broadcasts_aux_onto_every_node(small_jets):
    jets = _with_aux_columns(small_jets[:16])
    geom = Geometry()
    plain = MatchedLundDataset(jets, geom)
    wide = MatchedLundDataset(jets, geom, AUX)
    for i in range(len(jets)):
        a, b = plain[i]["xf"], wide[i]["xf"]
        assert b.shape == (a.shape[0], N_NODE_FEAT + len(AUX))
        assert torch.equal(b[:, :N_NODE_FEAT], a)          # node features untouched
        if b.shape[0] > 1:                                  # aux is CONSTANT per node
            assert torch.equal(b[0, N_NODE_FEAT:], b[-1, N_NODE_FEAT:])
        assert b.shape[0] == plain[i]["nx"]                 # nx unchanged


def test_empty_x_keeps_zero_rows_at_the_widened_width():
    """The nx == 0 case: shape must widen even with no rows to broadcast onto."""
    xf = node_features([], [], [], [])
    wide = with_aux(xf, aux_vector({"jet_pt": 100.0, "x_mg": 1.0, "x_nsec": 0}, AUX))
    assert wide.shape == (0, N_NODE_FEAT + len(AUX))


def test_collate_infers_width_on_mixed_lengths(small_jets):
    jets = _with_aux_columns(small_jets[:16])
    ds = MatchedLundDataset(jets, Geometry(), AUX)
    b = collate([ds[i] for i in range(8)])
    assert b["xf"].shape[2] == N_NODE_FEAT + len(AUX)
    assert b["yf"].shape[2] == N_NODE_FEAT and b["yraw"].shape[2] == 4  # target side untouched
    # padding rows stay zero, so masked consumers are unaffected by the wider x
    for i, n in enumerate(b["nx"].tolist()):
        assert torch.count_nonzero(b["xf"][i, n:]) == 0


def test_collate_rejects_mixed_widths(small_jets):
    jets = _with_aux_columns(small_jets[:8])
    geom = Geometry()
    plain, wide = MatchedLundDataset(jets, geom), MatchedLundDataset(jets, geom, AUX)
    with pytest.raises(ValueError, match="mixed x feature widths"):
        collate([plain[0], wide[1]])


def test_synthetic_source_refuses_aux(small_jets):
    """No proxies: the synthetic generator has no secondary planes."""
    with pytest.raises(ValueError, match="synthetic"):
        MatchedLundDataset(small_jets[:8], Geometry(), AUX)


def test_missing_columns_on_a_real_file_name_the_writer(small_jets):
    jets = [dict(j, generator="PYTHIA-8.3:tune-Monash") for j in small_jets[:8]]
    with pytest.raises(ValueError, match="cpp/ writer"):
        MatchedLundDataset(jets, Geometry(), AUX)


# ---------------------------------------------------------------------------
# models: widened first layer, parity when off, one training step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ["ar_junipr_v2", "ar_junipr_v3", "cinn", "diffusion", "cfm"])
@pytest.mark.parametrize("encoder", ["gru", "lundnet", "deepsets"])
def test_family_builds_with_widened_input_and_trains_one_step(model_name, encoder, small_jets):
    jets = _with_aux_columns(small_jets[:16])
    cfg = load_config([f"model={model_name}", f"encoder={encoder}",
                       f"encoder.aux_features=[{','.join(AUX)}]"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom)
    assert model.aux_feature_names == tuple(AUX)

    b = collate([MatchedLundDataset(jets, geom, AUX)[i] for i in range(8)])
    loss = model.training_objective(b).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.encoder_net.parameters() if p.grad is not None]
    assert grads and any(torch.any(g != 0) for g in grads)


@pytest.mark.parametrize("model_name", ["ar_junipr_v2", "cinn", "cfm"])
def test_off_path_state_dict_is_byte_identical(model_name, small_jets):
    """`aux_features=[]` must leave the module list and every shape untouched."""
    geom = Geometry.from_config(load_config([]).geometry)
    torch.manual_seed(0)
    base = build_model(load_config([f"model={model_name}"]), geom)
    torch.manual_seed(0)
    explicit = build_model(load_config([f"model={model_name}", "encoder.aux_features=[]"]), geom)
    assert base.aux_feature_names == ()
    a, b = base.state_dict(), explicit.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)

    plain = MatchedLundDataset(small_jets[:16], geom)
    batch = collate([plain[i] for i in range(8)])
    with torch.inference_mode():  # eval(): the encoder dropout is stochastic in train mode
        assert torch.allclose(base.eval().log_prob(batch), explicit.eval().log_prob(batch))


def test_aux_changes_the_encoder_input_width_only(small_jets):
    geom = Geometry.from_config(load_config([]).geometry)
    off = build_model(load_config(["model=ar_junipr_v2"]), geom)
    on = build_model(load_config(["model=ar_junipr_v2", f"encoder.aux_features=[{AUX[0]}]"]), geom)
    assert off.encoder_net.x_feat[0].in_features == N_NODE_FEAT
    assert on.encoder_net.x_feat[0].in_features == N_NODE_FEAT + 1
    # every other parameter shape is unchanged
    off_shapes = {k: v.shape for k, v in off.state_dict().items() if "x_feat.0" not in k}
    on_shapes = {k: v.shape for k, v in on.state_dict().items() if "x_feat.0" not in k}
    assert off_shapes == on_shapes


def test_aux_actually_reaches_the_likelihood(small_jets):
    """Changing an aux VALUE must change log_prob — otherwise the column is inert."""
    geom = Geometry.from_config(load_config([]).geometry)
    torch.manual_seed(0)
    model = build_model(
        load_config(["model=ar_junipr_v2", f"encoder.aux_features=[{','.join(AUX)}]"]), geom
    ).eval()
    lo = MatchedLundDataset(_with_aux_columns(small_jets[:8], mg=1.0, nsec=0), geom, AUX)
    hi = MatchedLundDataset(_with_aux_columns(small_jets[:8], mg=30.0, nsec=9), geom, AUX)
    with torch.inference_mode():
        a = model.log_prob(collate([lo[i] for i in range(8)]))
        b = model.log_prob(collate([hi[i] for i in range(8)]))
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# datamodule fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_separates_aux_widths(small_jets):
    from h2p_rsd_junipr.data.datamodule import _fingerprint

    d = load_config([]).data
    assert _fingerprint(small_jets, d, ()) != _fingerprint(small_jets, d, tuple(AUX))


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
def _request(jet):
    li, lk, lz, ps = jet["x"]
    return {"lnInvDelta": li.tolist(), "lnkt": lk.tolist(),
            "lnz": lz.tolist(), "psi": ps.tolist()}


def test_serving_round_trip_without_aux(small_jets):
    from h2p_rsd_junipr.serving.api import predict

    geom = Geometry.from_config(load_config([]).geometry)
    model = build_model(load_config(["model=ar_junipr_v2"]), geom).eval()
    out = predict(model, geom, torch.device("cpu"), _request(small_jets[0]),
                  decode={"n_posterior_samples": 8})
    assert out["aux_features"] == []
    assert out["map_multiplicity"] >= 1


def test_serving_round_trip_with_aux(small_jets):
    from h2p_rsd_junipr.serving.api import predict

    geom = Geometry.from_config(load_config([]).geometry)
    model = build_model(
        load_config(["model=ar_junipr_v2", f"encoder.aux_features=[{','.join(AUX)}]"]), geom
    ).eval()
    req = _request(small_jets[0])
    req["aux"] = {"jet_pt": 150.0, "x_mg": 8.0, "x_nsec": 2}
    out = predict(model, geom, torch.device("cpu"), req, decode={"n_posterior_samples": 8})
    assert out["aux_features"] == AUX


def test_serving_requires_aux_when_the_model_was_trained_with_it(small_jets):
    from h2p_rsd_junipr.serving.api import predict

    geom = Geometry.from_config(load_config([]).geometry)
    model = build_model(
        load_config(["model=ar_junipr_v2", f"encoder.aux_features=[{','.join(AUX)}]"]), geom
    ).eval()
    with pytest.raises(ValueError, match="aux"):  # no aux at all
        predict(model, geom, torch.device("cpu"), _request(small_jets[0]),
                decode={"n_posterior_samples": 4})
    req = _request(small_jets[0])
    req["aux"] = {"jet_pt": 150.0}  # incomplete
    with pytest.raises(ValueError):
        predict(model, geom, torch.device("cpu"), req, decode={"n_posterior_samples": 4})


# ---------------------------------------------------------------------------
# end-to-end on the real RNTuple, when it is present
# ---------------------------------------------------------------------------
def test_rntuple_path_carries_the_aux_columns():
    from pathlib import Path

    from h2p_rsd_junipr.data.rntuple import load_rntuple

    path = Path(__file__).resolve().parents[1] / "cpp" / "test_data" / "jets_aux.root"
    if not path.exists():
        pytest.skip("cpp/test_data/jets_aux.root not present")
    jets = load_rntuple(str(path))
    if jets is None:
        pytest.skip("uproot could not read the RNTuple")
    assert all(j["x_nsec"] >= 0 and j["jet_pt"] > 0 and j["x_mg"] >= 0 for j in jets[:200])
    ds = MatchedLundDataset(jets[:200], Geometry(), AUX)
    assert ds[0]["xf"].shape[1] == N_NODE_FEAT + len(AUX)


def test_old_rntuple_without_aux_fails_loud():
    from pathlib import Path

    from h2p_rsd_junipr.data.rntuple import load_rntuple

    path = Path(__file__).resolve().parents[1] / "cpp" / "test_data" / "jets.root"
    if not path.exists():
        pytest.skip("cpp/test_data/jets.root not present")
    jets = load_rntuple(str(path))
    if jets is None:
        pytest.skip("uproot could not read the RNTuple")
    assert jets[0]["x_nsec"] == -1 and math.isnan(jets[0]["x_mg"])
    with pytest.raises(ValueError):
        MatchedLundDataset(jets[:32], Geometry(), AUX)
