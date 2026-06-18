import torch

from h2p_rsd_junipr.geometry import Geometry


def test_n_cells_derived():
    g = Geometry(n_bins=10)
    assert g.n_cells == 100
    assert g.start_token == 100
    g16 = Geometry(n_bins=16)
    assert g16.n_cells == 256


def test_cell_roundtrip_centres_are_stable():
    g = Geometry()
    # the centre of a cell must map back to that same cell
    for cell in range(g.n_cells):
        u, v = g.cell_center(cell)
        assert g.to_cell(u, v) == cell


def test_to_cell_clips_to_range():
    g = Geometry()
    assert 0 <= g.to_cell(-10.0, -10.0) < g.n_cells
    assert 0 <= g.to_cell(100.0, 100.0) < g.n_cells
    # bottom-left corner -> cell 0, top-right -> last cell
    assert g.to_cell(0.0, 0.0) == 0
    assert g.to_cell(6.0, 6.0) == g.n_cells - 1


def test_cell_center_tensors_match_cell_center():
    g = Geometry()
    cx, cy = g.cell_center_tensors()
    for cell in (0, 1, 10, 55, 99):
        u, v = g.cell_center(cell)
        assert torch.allclose(cx[cell], torch.tensor(u))
        assert torch.allclose(cy[cell], torch.tensor(v))


def test_from_config_matches_default():
    from h2p_rsd_junipr.config import load_config

    cfg = load_config([])
    g = Geometry.from_config(cfg.geometry)
    assert g.n_bins == 10
    assert g.ln_invdelta_range == (0.0, 6.0)
