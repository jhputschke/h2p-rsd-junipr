from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.features import N_NODE_FEAT
from h2p_rsd_junipr.geometry import Geometry

JETS_AUX = Path(__file__).resolve().parents[1] / "cpp" / "test_data" / "jets_aux.root"


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


# ---------------------------------------------------------------------------
# RNTuple loader: correctness of the column-hoisted read
# ---------------------------------------------------------------------------
def _jets_or_skip(path=JETS_AUX):
    from h2p_rsd_junipr.data.rntuple import load_rntuple

    if not path.exists():
        pytest.skip(f"{path.name} not present")
    jets = load_rntuple(str(path))
    if jets is None:
        pytest.skip("uproot could not read the RNTuple")
    return jets


def test_rntuple_sequences_are_per_jet_slices_not_shared():
    """The loader flattens each jagged column into ONE buffer and hands out slices of
    it. The bug that would introduce is jets aliasing each other's data, so check the
    offsets actually line up with the per-jet lengths and contents."""
    import awkward as ak
    import uproot

    jets = _jets_or_skip()
    with uproot.open(str(JETS_AUX)) as f:
        raw = f["Jets"].arrays(["x_lnkt", "y_lnkt"], library="ak")

    for key, col in (("x", "x_lnkt"), ("y", "y_lnkt")):
        # the offsets are right iff concatenating every per-jet view reproduces the
        # column's flat content exactly -- an off-by-one would survive a length check
        flat = np.asarray(ak.to_numpy(ak.flatten(raw[col])), dtype=np.float32)
        cat = np.concatenate([j[key][1] for j in jets] + [np.empty(0, np.float32)])
        assert np.array_equal(cat, flat)
        # per-jet lengths must match the source too
        assert [len(j[key][1]) for j in jets] == list(np.asarray(ak.num(raw[col])))
        # distinct jets must occupy DISJOINT regions of the shared buffer
        non_empty = [j[key][1] for j in jets if len(j[key][1]) > 0][:200]
        for a, b in zip(non_empty, non_empty[1:]):
            assert not np.shares_memory(a, b)

    # all four components of a jet's sequence have the same length
    for j in jets[:1000]:
        assert len({len(c) for c in j["x"]}) == 1
        assert len({len(c) for c in j["y"]}) == 1


def test_rntuple_sequence_buffers_are_read_only():
    """Slices are views into a shared buffer, so an accidental write would corrupt a
    neighbouring jet. It must raise instead."""
    jets = _jets_or_skip()
    seq = next(j["x"][0] for j in jets if len(j["x"][0]) > 0)
    with pytest.raises(ValueError):
        seq[0] = 0.0


def test_rntuple_carries_the_secondary_floor_provenance():
    """`kt_floor_sec` is the OFF-SPINE floor the aux traversal used. A file written
    before asymmetric floors existed has no such column, and there the honest default
    is `kt_floor` itself — mirroring is exactly what those files did — NOT NaN."""
    jets = _jets_or_skip()
    j = jets[0]
    assert "kt_floor_sec" in j
    if np.isfinite(j["kt_floor"]):
        # jets_aux.root predates the column -> must mirror, not report a sentinel
        assert j["kt_floor_sec"] == j["kt_floor"]


def test_rntuple_dtypes_and_sentinels():
    jets = _jets_or_skip()
    j = jets[0]
    for c in j["x"] + j["y"]:
        assert c.dtype == np.float32
    assert isinstance(j["weight"], float) and isinstance(j["x_nsec"], int)
    assert isinstance(j["generator"], str) and isinstance(j["event"], int)
    # aux columns present in this file -> real values, not sentinels
    assert j["jet_pt"] > 0 and j["x_nsec"] >= 0 and np.isfinite(j["x_mg"])


def test_rntuple_missing_optional_columns_become_sentinels():
    """An older file lacking the aux columns must read, with the sentinels that
    features.aux_vector rejects -- not silently-plausible zeros."""
    from h2p_rsd_junipr.data.rntuple import load_rntuple

    old = Path(__file__).resolve().parents[1] / "cpp" / "test_data" / "jets.root"
    if not old.exists():
        pytest.skip("jets.root not present")
    jets = load_rntuple(str(old))
    if jets is None:
        pytest.skip("uproot could not read the RNTuple")
    j = jets[0]
    # columns this file genuinely lacks -> sentinels
    assert np.isnan(j["x_mg"]) and np.isnan(j["x_ptg"])
    assert j["x_nsec"] == -1 and j["x_sec_attach"] == -1
    assert j["x_kt_sec_max"] == -1.0 and j["x_kt_sec_sum"] == -1.0
    # ...but jet_pt and jet_eta have been in the schema all along; they were merely
    # never READ into the jet dicts before the aux work, so they are real values here
    assert j["jet_pt"] > 0 and np.isfinite(j["jet_eta"])


# ---------------------------------------------------------------------------
# Jet-pT window (data.pt_min / data.pt_max / data.pt_var)
# ---------------------------------------------------------------------------
def _fake_jets(pts, col="jet_pt"):
    """Minimal jet dicts: the selector only ever reads the one pT column."""
    return [{col: p, "x": (np.zeros(1, np.float32),), "y": (np.zeros(1, np.float32),)}
            for p in pts]


def test_select_pt_range_is_half_open_and_off_by_default():
    from h2p_rsd_junipr.data.datamodule import select_pt_range

    jets = _fake_jets([20.0, 100.0, 149.9, 150.0, 400.0])
    # off path returns the SAME list object -- nothing downstream can change
    assert select_pt_range(jets) is jets
    kept = select_pt_range(jets, "jet_pt", 100.0, 150.0, verbose=False)
    assert [j["jet_pt"] for j in kept] == [100.0, 149.9]  # lower incl., upper excl.
    assert [j["jet_pt"] for j in
            select_pt_range(jets, "jet_pt", 150.0, None, verbose=False)] == [150.0, 400.0]
    assert [j["jet_pt"] for j in
            select_pt_range(jets, "jet_pt", None, 100.0, verbose=False)] == [20.0]


def test_select_pt_range_groomed_column_and_alias():
    from h2p_rsd_junipr.data.datamodule import select_pt_range

    jets = [{"jet_pt": 200.0, "x_ptg": 40.0}, {"jet_pt": 50.0, "x_ptg": 120.0}]
    for var in ("x_ptg", "ptg"):
        kept = select_pt_range(jets, var, 100.0, None, verbose=False)
        assert [j["x_ptg"] for j in kept] == [120.0]  # cuts on the GROOMED pT, not jet_pt


def test_select_pt_range_rejects_unusable_windows():
    from h2p_rsd_junipr.data.datamodule import select_pt_range

    jets = _fake_jets([20.0, 100.0, 400.0])
    with pytest.raises(ValueError, match="not a selectable"):
        select_pt_range(jets, "pt_groomed", 10.0, None)
    with pytest.raises(ValueError, match="empty jet-pT window"):
        select_pt_range(jets, "jet_pt", 200.0, 100.0)
    with pytest.raises(ValueError, match="negative"):
        select_pt_range(jets, "jet_pt", -1.0, None)
    with pytest.raises(ValueError, match="kept 0 of 3"):
        select_pt_range(jets, "jet_pt", 5000.0, None)
    # every jet unset (synthetic data / pre-aux file) must NAME the column, not
    # hand back an empty dataset
    with pytest.raises(ValueError, match="NO jet carries a finite jet_pt"):
        select_pt_range(_fake_jets([np.nan, np.nan]), "jet_pt", 100.0, None)


def test_select_pt_range_drops_unset_jets_but_keeps_the_rest():
    from h2p_rsd_junipr.data.datamodule import select_pt_range

    jets = _fake_jets([np.nan, 120.0, 130.0])
    assert len(select_pt_range(jets, "jet_pt", 100.0, 150.0, verbose=False)) == 2


def test_datamodule_pt_window_applies_before_split_and_fingerprint():
    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule

    base = ["data=rntuple", f"data.path={JETS_AUX}"]
    if not JETS_AUX.exists():
        pytest.skip("jets_aux.root not present")
    wide = LundDataModule(load_config(base), Geometry()).setup()
    if wide.jets is None or not np.isfinite(wide.jets[0]["jet_pt"]):
        pytest.skip("uproot could not read the RNTuple")
    cut = LundDataModule(
        load_config(base + ["data.pt_min=100", "data.pt_max=150"]), Geometry()
    ).setup()

    assert 0 < len(cut.jets) < len(wide.jets)
    # the window bounds BOTH splits, not just the loaded list
    for j in cut.train_jets + cut.val_jets:
        assert 100.0 <= j["jet_pt"] < 150.0
    # ...and ties the run to a different dataset than the unwindowed one
    assert cut.fingerprint != wide.fingerprint


def test_datamodule_pt_window_off_leaves_the_fingerprint_untouched():
    """The off path must hash exactly as it did before the knob existed, or every
    cached tensor file and recorded run<->data linkage is invalidated."""
    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule, _fingerprint

    cfg = load_config(["data.n_jets=300", "data.seed=0", "data.min_val=32"])
    dm = LundDataModule(cfg, Geometry()).setup()
    legacy = {k: v for k, v in dm.cfg.data.items() if not k.startswith("pt_")}
    assert dm.fingerprint == _fingerprint(dm.jets, OmegaConf.create(legacy))


def test_datamodule_pt_window_that_starves_the_train_split_raises():
    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule

    if not JETS_AUX.exists():
        pytest.skip("jets_aux.root not present")
    cfg = load_config(["data=rntuple", f"data.path={JETS_AUX}", "data.pt_min=400"])
    with pytest.raises(ValueError, match="train split would be empty"):
        LundDataModule(cfg, Geometry()).setup()
