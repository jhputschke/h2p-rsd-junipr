"""Decode-config plumbing + serving output: the MAP floor reaches the serving layer
and the posterior summary now carries a median."""

import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.serving.api import predict


def _model(sel):
    cfg = load_config(sel)
    geom = Geometry.from_config(cfg.geometry)
    return build_model(cfg, geom).eval(), geom, cfg


def test_predict_reports_median_and_nonempty_map():
    model, geom, cfg = _model(["model=ar_junipr_v2", "encoder=gru"])
    x = {"lnInvDelta": [0.3, 1.3, 4.3], "lnkt": [4.7, 4.4, 3.6],
         "lnz": [-1.1, -0.2, -0.9], "psi": [-3.0, -2.8, -0.3]}
    out = predict(model, geom, torch.device("cpu"), x, decode=decode_params(cfg))
    assert out["map_multiplicity"] >= 1                     # floor honored end-to-end
    assert out["map_multiplicity"] == len(out["map_nodes"])
    assert "posterior_mult_median" in out
    assert "posterior_mult_mean" in out and "posterior_mult_68CR" in out


def test_predict_without_decode_still_floors():
    """No decode dict passed -> model signature default (min_emissions=1) still holds."""
    model, geom, _ = _model(["model=ar_junipr_v2", "encoder=gru"])
    x = {"lnInvDelta": [0.3, 4.3], "lnkt": [4.7, 3.6], "lnz": [-1.1, -0.9], "psi": [-3.0, -0.3]}
    out = predict(model, geom, torch.device("cpu"), x)
    assert out["map_multiplicity"] >= 1
    assert "posterior_mult_median" in out
