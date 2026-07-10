"""Decode-config plumbing + serving output: the MAP floor reaches the serving layer,
the posterior summary carries a median, and the MBR estimator is selectable end-to-end."""

import importlib.util

import pytest
import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.serving.api import predict


class _FixedHead(torch.nn.Module):
    """A multiplicity head returning fixed logits, regardless of the encoded jet —
    lets a test pin P(n|x) to a known, right-skewed shape (low mode, high quantile)."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, e):
        return self.logits.expand(e.shape[0], -1)


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


@pytest.mark.skipif(importlib.util.find_spec("ot") is None, reason="POT not installed")
def test_predict_mbr_selectable_and_map_unchanged():
    """point_estimator='mbr' returns a tree plus additive mbr_risk/mbr_backend fields
    (reusing the same draws); the default 'map' output carries no MBR fields."""
    model, geom, cfg = _model(["model=ar_junipr_v2", "encoder=gru"])
    x = {"lnInvDelta": [0.3, 1.3, 4.3], "lnkt": [4.7, 4.4, 3.6],
         "lnz": [-1.1, -0.2, -0.9], "psi": [-3.0, -2.8, -0.3]}
    dec = decode_params(cfg)
    out_map = predict(model, geom, torch.device("cpu"), x, decode=dec)
    assert "mbr_risk" not in out_map and "mbr_backend" not in out_map  # map path: no MBR fields
    out_mbr = predict(model, geom, torch.device("cpu"), x,
                      decode={**dec, "point_estimator": "mbr", "n_posterior_samples": 64})
    assert "mbr_risk" in out_mbr and out_mbr["mbr_backend"] == "pot"    # additive, non-breaking
    assert out_mbr["map_multiplicity"] == len(out_mbr["map_nodes"])     # a valid tree
    assert "posterior_mult_median" in out_mbr


def test_predict_learned_floor_only_raises_map():
    """length_floor_quantile>0 floors the MAP multiplicity up per jet, never below the
    alpha=0 baseline. P(n|x) is pinned right-skewed (mode 1, 0.9-quantile 6) — exactly
    the length-bias case the learned floor is meant to correct."""
    torch.manual_seed(0)
    model, geom, cfg = _model(["model=cinn", "encoder=deepsets"])
    # p = [0, .4, .1, .1, .1, .1, .1, .1, 0...]: argmax (mode) = 1, cdf hits 0.9 at n=6
    width = model.max_emissions + 1
    p = torch.zeros(width)
    p[1] = 0.4
    p[2:8] = 0.1
    model.n_head = _FixedHead(torch.log(p.clamp(min=1e-12)))

    x = {"lnInvDelta": [0.3, 1.3, 4.3], "lnkt": [4.7, 4.4, 3.6],
         "lnz": [-1.1, -0.2, -0.9], "psi": [-3.0, -2.8, -0.3]}
    dec = decode_params(cfg)
    base = predict(model, geom, torch.device("cpu"), x, decode=dec)
    hi = predict(model, geom, torch.device("cpu"), x, decode={**dec, "length_floor_quantile": 0.9})

    assert base["map_multiplicity"] == 1                  # mode-driven MAP (floored at 1)
    assert hi["map_multiplicity"] == 6                    # raised to the learned 0.9 quantile
    assert hi["map_multiplicity"] >= base["map_multiplicity"]
    assert hi["map_multiplicity"] == len(hi["map_nodes"])
    # alpha back at 0 is a no-op even with the skewed head
    same = predict(model, geom, torch.device("cpu"), x, decode={**dec, "length_floor_quantile": 0.0})
    assert same["map_multiplicity"] == base["map_multiplicity"]
