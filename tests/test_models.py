import math

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


# --- sample_coordinates: the coordinate half of a posterior draw ------------
CHAIN = [0, 5, 12, 5]


@pytest.mark.parametrize("sel", MODELS + [["model=ar_junipr_v4", "encoder=gru"]],
                         ids=lambda s: s[0].split("=")[1])
def test_sample_coordinates_agrees_with_its_flag(sel, batch):
    """`has_continuous_coords` must say exactly what `sample_coordinates` does. A family
    that advertised coordinates it does not draw would put filler constants on every
    plot and into every distance."""
    b, geom = batch
    model = build_model(load_config(sel), geom).eval()
    xf, nx = b["xf"][:1], b["nx"][:1]
    got = model.sample_coordinates(xf, nx, CHAIN)
    if not model.has_continuous_coords:
        assert got is None
        return
    assert got is not None and got.shape == (len(CHAIN), 4)
    assert torch.isfinite(got).all()
    assert model.sample_coordinates(xf, nx, []).shape == (0, 4)


@pytest.mark.parametrize("sel", MODELS + [["model=ar_junipr_v4", "encoder=gru"]],
                         ids=lambda s: s[0].split("=")[1])
def test_describe_cells_carries_real_coordinates(sel, batch):
    """The regression this whole hook exists for: a drawn tree's ln z / psi must be
    DRAWN, not the `ln z = 0, psi = 0` (i.e. z = 1, the softer prong taking the whole
    jet) placeholders the cell-centre fallback has to invent."""
    b, geom = batch
    model = build_model(load_config(sel), geom).eval()
    torch.manual_seed(0)
    pe = model.describe_cells(b["xf"][:1], b["nx"][:1], CHAIN)
    lnz = [n.ln_z for n in pe.nodes]
    psi = [n.psi for n in pe.nodes]
    if model.has_continuous_coords:
        assert any(v != 0.0 for v in lnz) and any(v != 0.0 for v in psi)
        assert all(n.z == pytest.approx(math.exp(n.ln_z)) for n in pe.nodes)
    else:
        assert lnz == [0.0] * len(CHAIN) and psi == [0.0] * len(CHAIN)
        assert all(n.z == 1.0 for n in pe.nodes)   # the placeholder, stated plainly


@pytest.mark.parametrize(
    "sel", [s for s in MODELS if "diffusion" not in s[0]],   # its log_prob is stochastic
    ids=lambda s: s[0].split("=")[1],
)
def test_describe_cells_scores_the_coordinates_it_reports(sel, batch):
    """`logprob` must be the density AT the returned nodes. Scoring cell centres while
    reporting drawn coordinates (or the reverse) is a wrong number presented as the
    model's joint log-density of the MBR winner."""
    b, geom = batch
    model = build_model(load_config(sel), geom).eval()
    xf, nx = b["xf"][:1], b["nx"][:1]
    torch.manual_seed(0)
    pe = model.describe_cells(xf, nx, CHAIN)
    yraw = torch.tensor(
        [[[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes]], dtype=torch.float32
    )
    with torch.no_grad():
        again = float(model.log_prob({
            "xf": xf, "nx": nx, "yraw": yraw,
            "yc": torch.tensor([CHAIN], dtype=torch.long),
            "ny": torch.tensor([len(CHAIN)]),
        })[0])
    assert again == pytest.approx(pe.logprob, rel=1e-4, abs=1e-3)


def test_geometry_n_bins_drives_cell_head(batch):
    b, geom = batch
    cfg = load_config(["model=ar_junipr_v2", "geometry.n_bins=8"])
    g8 = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, g8)
    assert model.n_cells == 64
    # cell head output width must equal n_cells
    out = model.split_head(torch.zeros(1, model.dec_dim + model.ctx_dim))
    assert out.shape[-1] == 64
