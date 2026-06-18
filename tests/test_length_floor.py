"""Learned per-jet MAP floor (quantile of P(n|x)): the helpers in
`inference/length.py`, the `length_pmf` accessor on every family, and the
opt-in flooring behavior (alpha=0 no-op, alpha>0 raises the MAP length)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.inference.length import learned_min_emissions, quantile_floor
from h2p_rsd_junipr.models.base import build_model

MODELS = [
    ["model=ar_junipr_v2", "encoder=gru"],
    ["model=ar_junipr_v1", "encoder=gru"],
    ["model=cinn", "encoder=deepsets"],
    ["model=diffusion", "encoder=lundnet"],
]
HEAD_MODELS = [["model=cinn", "encoder=deepsets"], ["model=diffusion", "encoder=lundnet"]]


def _jet(batch):
    b, geom = batch
    return b["xf"][:1], b["nx"][:1], geom


# --- quantile_floor unit cases --------------------------------------------------
def test_quantile_floor_units():
    # all mass at n=2: any alpha>0 floors to 2
    assert quantile_floor([0.0, 0.0, 1.0], 0.5) == 2
    # cdf = [0.2, 0.5, 1.0]: smallest n with cdf(n) >= alpha
    pmf = [0.2, 0.3, 0.5]
    assert quantile_floor(pmf, 0.6) == 2     # 0.6 lands above cdf[1]=0.5
    assert quantile_floor(pmf, 0.5) == 1     # cdf[1] == 0.5
    assert quantile_floor(pmf, 0.2) == 0     # cdf[0] == 0.2
    # alpha <= 0 -> the empty-tree bound; alpha >= 1 -> last index, clamped
    assert quantile_floor(pmf, 0.0) == 0
    assert quantile_floor(pmf, 1.0) == 2
    assert quantile_floor(pmf, 1.5) == 2     # over-unity clamps to len-1
    # degenerate empty pmf
    assert quantile_floor([], 0.5) == 0


# --- length_pmf is a valid distribution for every family ------------------------
@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_length_pmf_sums_to_one(sel, batch):
    torch.manual_seed(0)
    xf, nx, _ = _jet(batch)
    model = build_model(load_config(sel), _jet(batch)[2]).eval()
    pmf = model.length_pmf(xf, nx, n_samples=64)
    assert pmf.ndim == 1 and pmf.size >= 1
    assert (pmf >= 0).all()
    assert pmf.sum() == pytest.approx(1.0, abs=1e-5)


# --- cINN/diffusion length_pmf is exactly the softmax of the multiplicity head ---
@pytest.mark.parametrize("sel", HEAD_MODELS, ids=lambda s: s[0].split("=")[1])
def test_length_pmf_matches_softmax_head(sel, batch):
    xf, nx, _ = _jet(batch)
    model = build_model(load_config(sel), _jet(batch)[2]).eval()
    pmf = model.length_pmf(xf, nx)
    with torch.inference_mode():
        ref = F.softmax(model.n_head(model.encode(xf, nx)), dim=-1).squeeze(0).cpu().numpy()
    assert np.allclose(pmf, ref, atol=1e-6)
    # mults are ignored by the exact head: passing them changes nothing
    assert np.allclose(model.length_pmf(xf, nx, mults=[0, 1, 2]), ref, atol=1e-6)


# --- learned_min_emissions short-circuits and only ever raises the floor ---------
@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_learned_floor_alpha_zero_is_base_floor(sel, batch):
    xf, nx, _ = _jet(batch)
    model = build_model(load_config(sel), _jet(batch)[2]).eval()
    # quantile<=0 short-circuits to base_floor with no pmf evaluated
    assert learned_min_emissions(model, xf, nx, quantile=0.0, base_floor=1) == 1
    assert learned_min_emissions(model, xf, nx, quantile=-1.0, base_floor=3) == 3
    # a positive quantile never drops below base_floor
    eff = learned_min_emissions(model, xf, nx, quantile=0.01, base_floor=5, n_samples=64)
    assert eff >= 5


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_alpha_zero_map_is_unchanged(sel, batch):
    """alpha=0 is a structural no-op: the floored MAP equals the default MAP."""
    torch.manual_seed(0)
    xf, nx, _ = _jet(batch)
    model = build_model(load_config(sel), _jet(batch)[2]).eval()
    base = model.map_estimate(xf, nx)  # default min_emissions=1
    eff = learned_min_emissions(model, xf, nx, quantile=0.0, base_floor=1)
    floored = model.map_estimate(xf, nx, min_emissions=eff)
    assert floored.multiplicity == base.multiplicity


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_high_alpha_floors_map_up(sel, batch):
    """A high quantile floors the MAP multiplicity to at least the learned bound."""
    torch.manual_seed(0)
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(sel), geom).eval()
    draws = model.sample(xf, nx, 128)
    mults = [len(d) for d in draws]
    eff = learned_min_emissions(model, xf, nx, quantile=0.9, base_floor=1, mults=mults)
    assert eff >= 1
    mp = model.map_estimate(xf, nx, min_emissions=eff)
    assert mp.multiplicity >= eff
    assert mp.multiplicity == len(mp.nodes)
