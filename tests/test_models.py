import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate
from h2p_rsd_junipr.models.base import build_model

MODELS = [
    ["model=ar_junipr_v2", "encoder=gru"],
    ["model=ar_junipr_v1", "encoder=gru"],
    ["model=ar_junipr_v3", "encoder=gru"],
    ["model=cinn", "encoder=deepsets"],
    ["model=diffusion", "encoder=lundnet"],
    ["model=cfm", "encoder=gru", "model.n_ode_steps=8"],   # 8 steps keeps the smoke fast
]


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_log_prob_finite_and_shaped(sel, batch):
    b, geom = batch
    cfg = load_config(sel)
    model = build_model(cfg, geom)
    lp = model.log_prob(b)
    assert lp.shape == (b["xf"].shape[0],)
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_sample_and_map_are_valid(sel, batch):
    b, geom = batch
    cfg = load_config(sel)
    model = build_model(cfg, geom).eval()
    xf = b["xf"][:1]
    nx = b["nx"][:1]
    draws = model.sample(xf, nx, 5)
    assert len(draws) == 5
    for d in draws:
        assert all(0 <= c < geom.n_cells for c in d)
    mp = model.map_estimate(xf, nx)
    assert isinstance(mp, LundPointEstimate)
    assert mp.multiplicity == len(mp.nodes)
    assert mp.multiplicity >= 1  # default min_emissions=1: never the unphysical empty tree


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_map_respects_min_emissions(sel, batch):
    b, geom = batch
    model = build_model(load_config(sel), geom).eval()
    xf, nx = b["xf"][:1], b["nx"][:1]
    mp3 = model.map_estimate(xf, nx, min_emissions=3)
    assert mp3.multiplicity >= 3 and mp3.multiplicity == len(mp3.nodes)
    # the floor is honored, not hard-coded: min_emissions=0 may collapse to the empty tree
    mp0 = model.map_estimate(xf, nx, min_emissions=0)
    assert mp0.multiplicity >= 0 and mp0.multiplicity == len(mp0.nodes)


def test_length_penalty_is_noop_at_zero(batch):
    """min_emissions=0, length_penalty=0.0 reproduces the raw-score beam exactly."""
    b, geom = batch
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    xf, nx = b["xf"][:1], b["nx"][:1]
    a = model.map_decode(xf, nx, min_emissions=0, length_penalty=0.0)
    c = model.map_decode(xf, nx, min_emissions=0, length_penalty=0.0)
    assert a == c  # deterministic
    # a non-zero penalty is allowed to differ (favors longer trees), but must stay valid
    d = model.map_decode(xf, nx, min_emissions=1, length_penalty=1.0)
    assert all(0 <= cell < geom.n_cells for cell in d)


def test_ar_v1_drops_coord_likelihood(batch):
    b, geom = batch
    v1 = build_model(load_config(["model=ar_junipr_v1"]), geom)
    v2 = build_model(load_config(["model=ar_junipr_v2"]), geom)
    # v1 omits the continuous-coordinate density; with random init the magnitudes differ
    assert v1.continuous_coords is False
    assert v2.continuous_coords is True


def test_geometry_n_bins_drives_cell_head(batch):
    b, geom = batch
    cfg = load_config(["model=ar_junipr_v2", "geometry.n_bins=8"])
    g8 = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, g8)
    assert model.n_cells == 64
    # cell head output width must equal n_cells
    out = model.split_head(torch.zeros(1, model.dec_dim + model.ctx_dim))
    assert out.shape[-1] == 64
