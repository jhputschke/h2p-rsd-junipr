import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate
from h2p_rsd_junipr.models.base import build_model

MODELS = [
    ["model=ar_junipr_v2", "encoder=gru"],
    ["model=ar_junipr_v1", "encoder=gru"],
    ["model=cinn", "encoder=deepsets"],
    ["model=diffusion", "encoder=lundnet"],
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
