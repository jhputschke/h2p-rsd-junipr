import numpy as np
import torch

from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.features import N_NODE_FEAT
from h2p_rsd_junipr.geometry import Geometry


def test_synthetic_is_deterministic():
    a = synthetic_matched_dataset(50, seed=0)
    b = synthetic_matched_dataset(50, seed=0)
    assert len(a) == len(b) == 50
    for ja, jb in zip(a, b):
        assert np.array_equal(ja["x"][0], jb["x"][0])
        assert np.array_equal(ja["y"][1], jb["y"][1])


def test_collate_padding_and_masking():
    jets = synthetic_matched_dataset(16, seed=1)
    ds = MatchedLundDataset(jets, Geometry())
    items = [ds[i] for i in range(8)]
    b = collate(items)
    B = 8
    assert b["xf"].shape[0] == B
    assert b["xf"].shape[2] == N_NODE_FEAT
    Mx = int(b["nx"].max())
    assert b["xf"].shape[1] == Mx
    # padded rows beyond nx must be zero
    for i in range(B):
        n = int(b["nx"][i])
        if n < Mx:
            assert torch.count_nonzero(b["xf"][i, n:]) == 0
    # cell targets within valid range
    for i in range(B):
        ny = int(b["ny"][i])
        assert (b["yc"][i, :ny] >= 0).all()
        assert (b["yc"][i, :ny] < Geometry().n_cells).all()


def test_datamodule_trailing_split_matches_script():
    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule

    cfg = load_config(["data.n_jets=1000", "data.seed=0"])
    dm = LundDataModule(cfg, Geometry()).setup()
    # n_val = max(200, 1000//10) = 200, trailing split (no event ids in synthetic)
    assert len(dm.val_jets) == 200
    assert len(dm.train_jets) == 800
