"""Minimum-Bayes-risk (MBR) point estimate under the perturbative-Lund EMD metric
(`inference/mbr.py`): the cloud adapter, the OT backends (pot / energyflow /
surrogate), the batched distance matrix, and the estimator's headline property —
it never collapses to the empty tree when the posterior is non-empty-dominated,
*with no floor* (`min_emissions=0`).

The optional backends are guarded: `pot` needs POT, `energyflow` needs a working
`energyflow`/`wasserstein` build (some platforms — notably Apple-Silicon wheels of
`wasserstein` — ship a non-functional solver, so we probe it and skip if broken).
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference import mbr
from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate
from h2p_rsd_junipr.models.base import build_model


# --- backend availability -------------------------------------------------------
def _pot_ok() -> bool:
    return importlib.util.find_spec("ot") is not None


def _energyflow_ok() -> bool:
    """energyflow is importable AND its underlying EMD solver actually works here.

    Goes through `mbr._import_ef` rather than importing energyflow raw: that is the
    entry point production uses, and it is where the macOS duplicate-OpenMP guard
    runs. Probing raw would commit wasserstein's OpenMP build before the guard could
    opt out of it, which on a duplicate-libomp host segfaults the whole session."""
    try:
        ef = mbr._import_ef()
        ef.emd.emd(np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), R=1.0, gdim=2)
        return True
    except Exception:
        return False


POT_OK, EF_OK = _pot_ok(), _energyflow_ok()
BACKENDS = [
    pytest.param("pot", marks=[] if POT_OK else pytest.mark.skip(reason="POT not installed")),
    pytest.param("energyflow", marks=[] if EF_OK else pytest.mark.skip(reason="energyflow solver unavailable")),
    pytest.param("surrogate", marks=[]),  # pure-numpy, always available
]

MODELS = [
    ["model=ar_junipr_v2", "encoder=gru"],
    ["model=ar_junipr_v1", "encoder=gru"],
    ["model=ar_junipr_v3", "encoder=gru"],
    ["model=cinn", "encoder=deepsets"],
    ["model=diffusion", "encoder=lundnet"],
]

GEOM = Geometry()  # default (0,6)^2, n_bins=10


def _cloud(cells, weight="kt", coords="lnDR_lnkt", lnkt_cut=None):
    return mbr.lund_cloud(cells, GEOM, lnkt_cut=lnkt_cut, weight=weight, coords=coords)


def _kwargs(backend):
    kw = dict(R=8.485, beta=1.0, norm=False, backend=backend)
    if backend == "surrogate":
        kw["geom"] = GEOM
    return kw


def _jet(batch):
    b, geom = batch
    return b["xf"][:1], b["nx"][:1], geom


# --- cloud adapter --------------------------------------------------------------
def test_lund_cloud_drops_below_lnkt_cut():
    # cell 0 has ln kt centre 0.3; cell 9 has ln kt centre 5.7 (n_bins=10, range (0,6))
    assert GEOM.cell_center(0)[1] < 1.0 < GEOM.cell_center(9)[1]
    pts, w = _cloud([0, 9], lnkt_cut=3.0)
    assert pts.shape == (1, 2) and w.shape == (1,)         # only the hard cell 9 survives
    assert pts[0, 1] == pytest.approx(GEOM.cell_center(9)[1])


def test_lund_cloud_weights_are_raw_not_normalised():
    pts, w = _cloud([3, 9, 27], weight="kt", lnkt_cut=0.0)
    assert w.shape == (3,)
    # kt weight == exp(ln kt) of each cell centre; NOT pre-normalised to unit sum
    assert np.allclose(w, [np.exp(GEOM.cell_center(c)[1]) for c in (3, 9, 27)])
    assert not np.isclose(w.sum(), 1.0)
    _, wu = _cloud([3, 9, 27], weight="unit", lnkt_cut=0.0)
    assert np.allclose(wu, 1.0)


def test_lund_cloud_empty_and_coords_gdim():
    for coords, g in (("lnDR_lnkt", 2), ("+lnz", 3), ("+psi", 4)):
        pts, w = _cloud([5, 6], coords=coords, lnkt_cut=0.0)
        assert pts.shape == (2, g)
        e_pts, e_w = _cloud([], coords=coords)
        assert e_pts.shape == (0, g) and e_w.shape == (0,)


# --- single-pair distance -------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_lund_emd_metric_properties(backend):
    a = _cloud([12, 34, 56], lnkt_cut=0.0)
    b = _cloud([12, 30], lnkt_cut=0.0)
    empty = _cloud([], lnkt_cut=0.0)
    kw = _kwargs(backend)
    # identity, symmetry, non-negativity, finiteness on ragged cardinalities
    assert mbr.lund_emd(a, a, **kw) == pytest.approx(0.0, abs=1e-6)
    dab = mbr.lund_emd(a, b, **kw)
    dba = mbr.lund_emd(b, a, **kw)
    assert dab == pytest.approx(dba, rel=1e-6, abs=1e-6)
    assert np.isfinite(dab) and dab >= 0.0
    # the empty cloud is expensive against a non-empty one (empty-tree-never-wins)
    assert mbr.lund_emd(empty, a, **kw) > 0.0
    assert mbr.lund_emd(empty, empty, **kw) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_pot_imbalance_penalty_is_exactly_R_times_dW():
    # d(empty, cloud) == R * total weight (pure mass-imbalance term, by construction)
    a = _cloud([12, 34, 56], lnkt_cut=0.0)
    empty = _cloud([], lnkt_cut=0.0)
    R = 8.485
    assert mbr.lund_emd(empty, a, R=R, backend="pot") == pytest.approx(R * a[1].sum(), rel=1e-6)


# --- backend agreement: pot vs energyflow ---------------------------------------
@pytest.mark.skipif(not (POT_OK and EF_OK), reason="need both pot and a working energyflow")
def test_pot_energyflow_agree_on_argmin_and_ratio():
    cands = [_cloud(c, lnkt_cut=0.0) for c in ([12, 34], [5, 9, 27], [40], [12, 34, 56, 7])]
    supp = [_cloud(c, lnkt_cut=0.0) for c in ([12, 30], [5, 9], [40, 41], [7, 12, 34])]
    R, beta = 8.485, 1.0
    D_pot = mbr.lund_emd_matrix(cands, supp, R=R, beta=beta, backend="pot")
    D_ef = mbr.lund_emd_matrix(cands, supp, R=R, beta=beta, backend="energyflow")
    # same MBR winner (argmin of the mean-risk), the property that actually matters
    assert int(D_pot.mean(1).argmin()) == int(D_ef.mean(1).argmin())
    # and the documented 1/R scale mapping for beta=1 (pot == R * energyflow), elementwise
    ratio = D_pot[D_ef > 1e-9] / D_ef[D_ef > 1e-9]
    assert np.allclose(ratio, R, rtol=1e-3)


@pytest.mark.skipif(not EF_OK, reason="energyflow solver unavailable")
def test_energyflow_emds_matches_looped_emd():
    cands = [_cloud(c, lnkt_cut=0.0) for c in ([12, 34], [5, 9, 27])]
    supp = [_cloud(c, lnkt_cut=0.0) for c in ([12, 30], [5, 9], [40, 41])]
    D = mbr.lund_emd_matrix(cands, supp, backend="energyflow")
    for i, ca in enumerate(cands):
        for j, cb in enumerate(supp):
            assert D[i, j] == pytest.approx(mbr.lund_emd(ca, cb, backend="energyflow"), rel=1e-6)


@pytest.mark.skipif(not EF_OK, reason="energyflow solver unavailable")
def test_energyflow_matrix_uses_the_batched_path():
    """`emds` (OpenMP over all cores) is the only parallel path; the per-pair fallback
    is single-threaded and ~15x slower for identical numbers. Guard that the NumPy-2
    copy-semantics retry keeps us on it, so the cost cannot regress silently."""
    cands = [_cloud(c, lnkt_cut=0.0) for c in ([12, 34], [5, 9, 27])]
    supp = [_cloud(c, lnkt_cut=0.0) for c in ([12, 30], [5, 9])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mbr.lund_emd_matrix(cands, supp, backend="energyflow")
    fallback = [w for w in caught if issubclass(w.category, RuntimeWarning)
                and "per-pair" in str(w.message)]
    assert not fallback, f"fell back to the serial per-pair emd: {fallback[0].message}"


@pytest.mark.skipif(not EF_OK, reason="energyflow solver unavailable")
def test_batched_emds_never_runs_multithreaded_on_duplicate_openmp():
    """The invariant that keeps a macOS session alive.

    PyTorch's bundled libomp plus a `wasserstein` linked against a *different* libomp
    is two OpenMP runtimes in one process; creating the `emds` thread team then
    segfaults, killing the interpreter with no traceback. `mbr._import_ef` is supposed
    to notice and either select wasserstein's no-OpenMP build or pin `emds` to one
    thread. Assert the state, not the crash -- a segfault cannot be caught in-process,
    so a test that merely called `emds` would take the suite down with it."""
    mbr._import_ef()
    if len(mbr._loaded_omp_runtimes()) <= 1:
        pytest.skip("single OpenMP runtime: the parallel path is safe here")
    from wasserstein import _wasserstein as wext

    assert not wext.cvar.COMPILED_WITH_OPENMP or mbr._EMDS_N_JOBS == 1, (
        "duplicate OpenMP runtimes with wasserstein's OpenMP build live and `emds` "
        "unpinned -- the next batched call segfaults the interpreter"
    )


@pytest.mark.skipif(not EF_OK, reason="energyflow solver unavailable")
def test_wasserstein_numpy2_compat_restores_the_module_namespace():
    """The shim patches `wasserstein.wasserstein`'s `np` global, not the real numpy,
    and must put it back even when the wrapped call raises."""
    ww = pytest.importorskip("wasserstein.wasserstein")
    before = ww.np
    with pytest.raises(ZeroDivisionError):
        with mbr._wasserstein_numpy2_compat() as patched:
            assert patched and ww.np is not before
            assert ww.np.ndarray is np.ndarray          # forwards everything else
            1 / 0
    assert ww.np is before


# --- the estimator: headline property -------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_mbr_never_empty_when_nonempty_draws_dominate(backend, batch):
    """The property the MinCut floors had to *clamp*: with a non-empty-dominated
    posterior, MBR selects a non-empty tree with NO floor (`min_emissions=0`)."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    # a controlled posterior: 90% non-empty draws, 10% empty
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 3 + [[], []]
    pe = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend=backend,
                        **({} if backend != "surrogate" else {}))
    assert isinstance(pe, LundPointEstimate)
    assert pe.multiplicity >= 1                        # never the empty tree, no min_emissions
    assert pe.multiplicity == len(pe.nodes)
    assert pe.risk is not None and np.isfinite(pe.risk)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mbr_reflects_empty_dominated_posterior(backend, batch):
    """Conversely (documented, intended): a posterior that is mostly empty makes MBR
    pick a short/empty tree — it reflects the posterior, unlike a manufactured floor."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=cinn", "encoder=deepsets"]), geom).eval()
    draws = [[], [], [], [], [], [], [], [], [12, 34], [5]]  # 80% empty
    pe = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend=backend)
    assert pe.multiplicity == 0                        # correct: the posterior is empty-dominated


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_mbr_returns_valid_tree_all_families(sel, batch):
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(sel), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 3
    pe = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot")
    assert isinstance(pe, LundPointEstimate)
    assert pe.multiplicity == len(pe.nodes) >= 1
    assert np.isfinite(pe.logprob) and pe.risk is not None


def test_mbr_is_deterministic_given_draws(batch):
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 2
    a = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    b = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    assert a.multiplicity == b.multiplicity
    assert [n.cell for n in a.nodes] == [n.cell for n in b.nodes]
    assert a.risk == pytest.approx(b.risk)


@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_point_estimator_map_is_structural_noop(sel, batch):
    """point_estimator='map' short-circuits to map_estimate — identical to today,
    and imports no OT backend."""
    torch.manual_seed(0)
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(sel), geom).eval()
    base = model.map_estimate(xf, nx)
    viad = model.map_or_mbr(xf, nx, point_estimator="map")
    assert viad.multiplicity == base.multiplicity
    assert [n.cell for n in viad.nodes] == [n.cell for n in base.nodes]


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_n_candidates_shrinks_candidate_set(batch):
    """mbr_n_candidates restricts C but keeps the full S for the expectation; the
    winner must be drawn from the first n_candidates draws."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 9, 27], [40, 41], [7], [12, 30, 56], []]
    pe = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend="pot", n_candidates=3)
    assert [n.cell for n in pe.nodes] in [list(d) for d in draws[:3]]
