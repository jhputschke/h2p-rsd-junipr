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
import platform
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
    ["model=edit_v1", "encoder=gru"],
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


def _coord_rows(cells, lnz=(-1.0, -2.0), psi=(0.4, -0.7)):
    """A continuous-coordinate table over the same cells — the representation a cell
    chain cannot supply. `(u, v)` are the cell centres so the 2-D distances are directly
    comparable to the chain's; `ln z` / `psi` are what the chain does not have."""
    return [[*GEOM.cell_center(int(c)), float(lz), float(ps)]
            for c, lz, ps in zip(cells, lnz, psi)]


def test_lund_cloud_cell_chain_is_two_dimensional():
    """A cell chain under the 2-D ground metric: unchanged, and the ONLY thing a cell
    chain can honestly produce (the Lund grid discretizes `(u, v)` alone)."""
    pts, w = _cloud([5, 6], coords="lnDR_lnkt", lnkt_cut=0.0)
    assert pts.shape == (2, 2) and w.shape == (2,)
    e_pts, e_w = _cloud([], coords="lnDR_lnkt")
    assert e_pts.shape == (0, 2) and e_w.shape == (0,)


@pytest.mark.parametrize("coords,g", [("lnDR_lnkt", 2), ("+lnz", 3), ("+psi", 4)])
def test_lund_cloud_coordinate_table_carries_the_real_columns(coords, g):
    """A coordinate TABLE gives every `coords` mode its true columns — the third is the
    supplied `ln z`, not a constant."""
    rows = _coord_rows([5, 6])
    pts, w = _cloud(rows, coords=coords, lnkt_cut=0.0)
    assert pts.shape == (2, g) and w.shape == (2,)
    if g >= 3:
        assert np.allclose(pts[:, 2], [-1.0, -2.0])       # the ln z that was supplied
        assert not np.allclose(pts[:, 2], 0.0)            # ...and NOT the old filler
    if g >= 4:
        assert np.allclose(pts[:, 3], [0.4, -0.7])
    # an empty draw is untouched by the guard: honestly empty, nothing fabricated
    e_pts, e_w = _cloud([], coords=coords)
    assert e_pts.shape == (0, g) and e_w.shape == (0,)


@pytest.mark.parametrize("coords", ["+lnz", "+psi"])
def test_lund_cloud_cell_chain_raises_above_two_columns(coords):
    """**This test replaces one that asserted the opposite** (`(2, 3)` for a cell chain
    under `+lnz`), and the old form passed only because the knob was INERT: `lund_cloud`
    hard-coded `lz = ps = 0.0` for cell chains, so `mbr_coords="+lnz"` appended a
    constant-zero third column and changed no distance. `sample_batch` returns cell
    chains, so that was the path the whole pipeline took — `+lnz` was not merely off by
    default, it could not be switched on (docs/PLAN_z_aware.md §2a, WP-2).

    It is the test that would have caught this had it been written against a coordinate
    table rather than against the representation the code happened to accept."""
    with pytest.raises(ValueError, match="CELL CHAIN"):
        _cloud([5, 6], coords=coords, lnkt_cut=0.0)


def test_lund_cloud_weight_z_is_no_longer_silently_unit():
    """`mbr_weight="z"` read column 2, which a cell chain fills with 0 — so every weight
    was `exp(0) = 1` and `"z"` was bit-identical to `"unit"`. It now raises on a chain and
    means what it says on a table."""
    with pytest.raises(ValueError, match="CELL CHAIN"):
        _cloud([5, 6], weight="z", coords="lnDR_lnkt", lnkt_cut=0.0)
    rows = _coord_rows([5, 6])
    pts, w = _cloud(rows, weight="z", coords="lnDR_lnkt", lnkt_cut=0.0)
    assert pts.shape == (2, 2)                              # gdim still follows `coords`
    assert np.allclose(w, np.exp([-1.0, -2.0]))
    _, w_unit = _cloud(rows, weight="unit", coords="lnDR_lnkt", lnkt_cut=0.0)
    assert not np.allclose(w, w_unit)                       # the regression this pins


def test_lund_cloud_short_row_raises_rather_than_padding():
    """A row with too few columns is the same silent `ln z = 0` by another route."""
    with pytest.raises(ValueError, match="an emission row has 2"):
        _cloud([[1.0, 2.0], [1.5, 2.5]], coords="+lnz", lnkt_cut=0.0)
    with pytest.raises(ValueError, match="an emission row has 3"):
        _cloud([[1.0, 2.0, -1.0]], coords="+psi", lnkt_cut=0.0)


def test_needs_continuous_coords_is_the_one_place_the_rule_lives():
    """The guard's own predicate, exported so `report.py`, the tests and the scripts ask
    one function instead of re-deriving `gdim > 2 or weight == "z"` each."""
    assert not mbr.needs_continuous_coords("lnDR_lnkt", "kt")
    assert not mbr.needs_continuous_coords("lnDR_lnkt", "unit")
    assert mbr.needs_continuous_coords("lnDR_lnkt", "z")     # weight reads column 2
    assert mbr.needs_continuous_coords("+lnz", "kt")
    assert mbr.needs_continuous_coords("+psi", "kt")
    assert mbr.cloud_columns_needed("lnDR_lnkt", "z") == 3
    assert mbr.cloud_columns_needed("+psi", "kt") == 4


def test_lnz_is_no_longer_inert_in_the_distance():
    """Two draws with IDENTICAL cells and different `ln z` are at distance 0 under the
    2-D ground metric and at a real distance under `+lnz`.

    The whole content of §2a in one assertion: before WP-2 both were 0, because the third
    column was the constant the adapter invented."""
    a = _coord_rows([5, 6], lnz=(-1.0, -1.0))
    b = _coord_rows([5, 6], lnz=(-2.5, -2.5))
    kw = _kwargs("surrogate" if not POT_OK else "pot")
    d2 = mbr.lund_emd(_cloud(a, lnkt_cut=0.0), _cloud(b, lnkt_cut=0.0), **kw)
    assert d2 == pytest.approx(0.0, abs=1e-9)
    kw3 = dict(kw)
    d3 = mbr.lund_emd(_cloud(a, coords="+lnz", lnkt_cut=0.0),
                      _cloud(b, coords="+lnz", lnkt_cut=0.0), **kw3)
    assert d3 > 1e-6


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


# --- the surrogate follows the model's resolution --------------------------------
def test_surrogate_image_bins_at_geom_n_bins():
    """`_lund_image` used to hard-code a 10x10 grid, which silently made the surrogate
    risk COARSER than the model at any other geometry (docs/PLAN_prod_test_v0.md check
    6). It must follow `geom.n_bins` — and still reproduce the old numbers at 10."""
    from h2p_rsd_junipr.geometry import Geometry

    g30 = Geometry(n_bins=30)
    pts, w = mbr.lund_cloud([0, 5, 9], GEOM, lnkt_cut=0.0)
    assert mbr._lund_image(pts, w, GEOM).size == 10 * 10
    assert mbr._lund_image(pts, w, g30).size == 30 * 30
    # explicit nb still wins, and nb=10 on the old geometry is unchanged
    assert mbr._lund_image(pts, w, g30, nb=10).size == 10 * 10
    assert np.allclose(mbr._lund_image(pts, w, GEOM), mbr._lund_image(pts, w, GEOM, nb=10))


def test_surrogate_resolves_what_the_coarse_image_could_not():
    """Two clouds inside one 0.6-wide cell but in different 0.2-wide cells: chi2 is 0
    at nb=10 (indistinguishable) and > 0 at nb=30."""
    from h2p_rsd_junipr.geometry import Geometry

    g30 = Geometry(n_bins=30)
    a = (np.array([[0.05, 0.05]]), np.array([1.0]))
    b = (np.array([[0.55, 0.55]]), np.array([1.0]))   # same 0.6-cell, different 0.2-cell
    assert mbr.lund_emd(a, b, backend="surrogate", geom=GEOM) == pytest.approx(0.0)
    assert mbr.lund_emd(a, b, backend="surrogate", geom=g30) > 0.0


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
    if platform.system() != "Darwin":
        pytest.skip("duplicate-runtime abort is a Darwin phenomenon; libgomp coexists "
                    "with torch's runtime on Linux and the guard no-ops there")
    mbr._import_ef()
    if not mbr._OMP_CONFLICT:
        pytest.skip("wasserstein shares an already-mapped runtime (or brought the only "
                    "one): the parallel path is safe here")
    from wasserstein import _wasserstein as wext

    assert not wext.cvar.COMPILED_WITH_OPENMP or mbr._EMDS_N_JOBS == 1, (
        "duplicate OpenMP runtimes with wasserstein's OpenMP build live and `emds` "
        "unpinned -- the next batched call segfaults the interpreter"
    )


def test_loaded_omp_runtimes_is_total_across_platforms():
    """It probes dyld, whose `_dyld_image_count` does not exist off macOS — so the
    ctypes lookup raised `undefined symbol` and every caller died ON THE PROBE instead
    of learning there was nothing to probe. Needs no energyflow: the failure was in the
    platform check, not the solver."""
    got = mbr._loaded_omp_runtimes()
    assert isinstance(got, set)
    if platform.system() != "Darwin":
        assert got == set(), (
            "off Darwin this must be empty: what it counts is runtimes that make a "
            "thread team fatal, and that is a macOS phenomenon"
        )


def test_a_third_party_libomp_does_not_downgrade_a_shared_build():
    """The guard must identify *wasserstein's* runtime, not count every runtime loaded.

    scikit-learn's macOS wheel vendors `sklearn/.dylibs/libomp.dylib`, and POT imports
    sklearn whenever it is installed -- so every `import energyflow` maps a second runtime
    that wasserstein never touches. Counting took that from one to two and switched a build
    correctly sharing PyTorch's libomp to the single-threaded one, paying ~11x for a
    segfault that could not happen. Pure set logic, so it runs off Darwin too."""
    torch_omp, sk_omp, conda_omp = "/t/torch/libomp.dylib", "/t/sk/libomp.dylib", "/t/libomp.dylib"

    # The regression: a stranger's copy was already mapped, and wasserstein added nothing.
    assert not mbr._wasserstein_omp_conflict(
        {torch_omp, sk_omp}, {torch_omp, sk_omp}, precommitted=False
    )
    # The real hazard: the dlopen resolved to a libomp that was not already there.
    assert mbr._wasserstein_omp_conflict(
        {torch_omp}, {torch_omp, conda_omp}, precommitted=False
    )
    # Sharing torch's runtime, and being the only runtime, are both safe.
    assert not mbr._wasserstein_omp_conflict({torch_omp}, {torch_omp}, precommitted=False)
    assert not mbr._wasserstein_omp_conflict(set(), {conda_omp}, precommitted=False)
    # Pre-committed: the resolve already happened, so the diff is empty for the wrong
    # reason and only the conservative count is left.
    assert mbr._wasserstein_omp_conflict(
        {torch_omp, sk_omp}, {torch_omp, sk_omp}, precommitted=True
    )
    assert not mbr._wasserstein_omp_conflict({torch_omp}, {torch_omp}, precommitted=True)


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
            1 / 0  # noqa: B018 — deliberate raise; the point is the shim unwinds
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
def test_mbr_never_empty_for_the_edit_family_either(backend, batch):
    """MBR is the DEFAULT point estimator for the edit family (its MAP is a Viterbi
    surrogate), so the headline invariant has to hold through its `describe_cells` —
    which, unlike the other families', draws coordinates through a constrained
    forward-backward before it can score the winner at all."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=edit_v1", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 3 + [[], []]
    torch.manual_seed(0)
    pe = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend=backend)
    assert pe.multiplicity >= 1 and pe.multiplicity == len(pe.nodes)
    assert np.isfinite(pe.logprob) and pe.risk is not None and np.isfinite(pe.risk)
    # the winner is a genuine drawn tree, placed at genuine drawn coordinates
    assert [n.cell for n in pe.nodes] in [list(d) for d in draws]
    assert all(geom.to_cell(n.ln_invDelta, n.ln_kt) == n.cell for n in pe.nodes)


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


# ---------------------------------------------------------------------------
# WP-C of docs/PLAN_prod_test_v1.md: the medoid carries its own sample, and the psi
# mode is reported only where the head identifies one.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_medoid_carries_its_own_sampled_coordinates(batch):
    """The winner's coordinates must be a DRAW, not the head's modes re-attached.

    Pinned two ways, because "it is a sample" is not observable from one number:
    supplying the coordinates the caller already drew reproduces them EXACTLY in the
    returned nodes, and drawing them internally does not reproduce the mode."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 2
    coords = [model.sample_coordinates(xf, nx, d) for d in draws]

    pe = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot",
                        coords_by_draw=coords)
    assert pe.coords_source == "sample"
    win = [list(d) for d in draws].index([n.cell for n in pe.nodes])
    for t, n in enumerate(pe.nodes):
        for j, got in enumerate((n.ln_invDelta, n.ln_kt, n.ln_z, n.psi)):
            assert got == pytest.approx(float(coords[win][t, j]), abs=1e-5), (
                f"node {t} coordinate {j} is not the supplied draw"
            )
    # ...and the modes are a different thing entirely
    modes = model.describe_sequence(xf, nx, draws[win])
    assert modes.coords_source == "mode"
    assert any(abs(a.ln_z - b.ln_z) > 1e-6 for a, b in zip(pe.nodes, modes.nodes))


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_medoid_logprob_is_the_density_of_what_it_returns(batch):
    """The `logprob` contract: it is the joint log-density OF THE RETURNED
    CONFIGURATION. Carrying sampled coordinates without re-evaluating the density
    would leave a number describing a tree nobody was shown."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56]] * 2
    coords = [model.sample_coordinates(xf, nx, d) for d in draws]
    pe = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot",
                        coords_by_draw=coords)
    cells = [n.cell for n in pe.nodes]
    yraw = torch.tensor([[[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes]],
                        dtype=torch.float32)
    b = {"xf": xf, "nx": nx, "yc": torch.tensor([cells], dtype=torch.long),
         "ny": torch.tensor([len(cells)]), "yraw": yraw}
    with torch.inference_mode():
        assert pe.logprob == pytest.approx(float(model.log_prob(b)[0]), abs=2e-4)


def test_kappa_gate_substitutes_a_draw_and_flags_it(batch):
    """Below `kappa_min_mode` the reported psi is a draw and the node says so; above
    it the mode is reported unchanged. Driven by pinning kappa on both sides of the
    bound, so the test does not depend on what an untrained head happens to hold."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    cells = [12, 34, 56]
    real = model._coord_params

    def pinned(kappa_value):
        def _params(coord_in):
            p = real(coord_in)
            return p._replace(kappa=torch.full_like(p.kappa, kappa_value))
        return _params

    model.kappa_min_mode = 0.5
    model._coord_params = pinned(5.0)                      # well identified
    hi = model.describe_sequence(xf, nx, cells)
    assert [n.psi_identified for n in hi.nodes] == [True] * 3
    assert hi.n_psi_unidentified == 0
    assert all(n.kappa == pytest.approx(5.0) for n in hi.nodes)
    # the mode is deterministic, so two calls agree exactly
    assert [n.psi for n in hi.nodes] == [n.psi for n in
                                         model.describe_sequence(xf, nx, cells).nodes]

    model._coord_params = pinned(0.02)                     # v0's measured median kappa
    torch.manual_seed(1)
    lo = model.describe_sequence(xf, nx, cells)
    assert [n.psi_identified for n in lo.nodes] == [False] * 3
    assert lo.n_psi_unidentified == 3
    torch.manual_seed(2)
    again = model.describe_sequence(xf, nx, cells)
    assert [n.psi for n in lo.nodes] != [n.psi for n in again.nodes], (
        "a substituted psi must be a DRAW, not a second deterministic value"
    )
    model._coord_params = real


def test_kappa_gate_off_is_the_unconditional_mode(batch):
    """`kappa_min_mode = 0.0` is the pinned reference path: the ungated mode, exactly
    as before WP-C, and deterministic."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru",
                                     "decode.kappa_min_mode=0.0"]), geom).eval()
    assert model.kappa_min_mode == 0.0
    a = model.map_estimate(xf, nx)
    b = model.map_estimate(xf, nx)
    assert [n.psi for n in a.nodes] == [n.psi for n in b.nodes]
    assert all(n.psi_identified is True for n in a.nodes)
    assert a.logprob == pytest.approx(b.logprob)


def test_psi_flag_is_none_for_a_carried_sample(batch):
    """Mode identifiability is not a question about a draw, so the flag is None there
    rather than a True/False a reader would act on."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    cells = [12, 34]
    pe = model.describe_cells(xf, nx, cells)
    assert pe.coords_source == "sample"
    assert all(n.psi_identified is None for n in pe.nodes)
    assert pe.n_psi_unidentified == 0
    assert all(n.kappa is not None for n in pe.nodes)


def test_kappa_gate_default_is_on():
    from h2p_rsd_junipr.config import decode_params

    assert decode_params(load_config(["model=ar_junipr_v2"]))["kappa_min_mode"] == 0.5
    m = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), GEOM)
    assert m.kappa_min_mode == 0.5


def test_decode_draws_never_touch_the_sampling_stream(batch):
    """A point estimate must not change which posterior draws come next.

    The psi gate and the medoid's coordinates are both DRAWS, so routing them through
    the global RNG would make a run that computed a MAP disagree with one that did not
    on every sampled number downstream — silently, and only in the numbers, never in a
    shape. They go through `decode_generator` instead."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()

    torch.manual_seed(0)
    plain = model.sample(xf, nx, 32)

    torch.manual_seed(0)
    model.map_estimate(xf, nx)                    # psi gate fires here
    model.describe_cells(xf, nx, [12, 34, 56])    # and a coordinate draw here
    assert model.sample(xf, nx, 32) == plain


# ---------------------------------------------------------------------------
# docs/PLAN_PosteriorClusters.md §4 — the metric audit, as a regression test.
#
# Clustering imposes requirements on `D` that the point estimator does not: HDBSCAN's
# mutual-reachability construction assumes a METRIC, and at beta != 1 the perturbative-Lund
# EMD is not one. §4 measured this on 40 synthetic clouds / 64 000 triples; the fixture
# below is the same construction at a size the unit suite can afford, and the beta = 2
# NEGATIVE control is what proves the check can detect a violation at all.
# ---------------------------------------------------------------------------
def _audit_clouds(n=14, seed=0):
    """`n` synthetic Lund clouds of 0-5 points each — the §4 fixture, in miniature.

    Deliberately includes empty clouds: `_empty_value` is a separate code path and it is
    the one the empty-clique hazard runs through."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = int(rng.integers(0, 6))
        pts = rng.uniform([0.0, 0.0], [6.0, 6.0], size=(m, 2))
        w = np.exp(pts[:, 1]) if m else np.zeros(0)
        out.append((pts, w))
    return out


def _triangle_violations(D, tol=1e-9):
    """Count `d(i,k) > d(i,j) + d(j,k) + tol` over every ordered triple."""
    viol = D[:, :, None] - (D[:, None, :] + D[None, :, :]) > tol
    np.fill_diagonal(viol.any(-1), False)
    return int(viol.sum())


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
@pytest.mark.parametrize("beta,expect_metric", [(1.0, True), (2.0, False)])
def test_metric_audit_triangle_inequality(beta, expect_metric):
    clouds = _audit_clouds()
    D = mbr.lund_emd_matrix(clouds, clouds, R=8.485, beta=beta, backend="pot")
    assert np.abs(D - D.T).max() < 1e-10, "symmetry is asserted, not assumed"
    assert np.abs(np.diag(D)).max() < 1e-10
    n_viol = _triangle_violations(D)
    if expect_metric:
        assert n_viol == 0, (
            f"beta = 1 must satisfy the triangle inequality (KMT's condition for the EMD "
            f"to be a metric); got {n_viol} violations"
        )
    else:
        # The negative control. Without it a green triangle test proves only that the
        # checker never fires, and `assert_cluster_metric_ok`'s beta guard would be
        # defending against nothing.
        assert n_viol > 0, "beta = 2 is expected to VIOLATE the triangle inequality"


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_surrogate_is_blind_to_the_mass_the_strata_are_made_of():
    """§4's mass-sensitivity row, and the reason `mbr_backend='surrogate'` may not produce
    a quoted mass vector: `_lund_image` NORMALISES, so scaling every weight by 10 leaves
    the surrogate distance at exactly 0 while `pot` sees 100+."""
    a = _cloud([12, 34, 56], lnkt_cut=0.0)
    b = (a[0], a[1] * 10.0)
    assert mbr.lund_emd(a, b, backend="surrogate", geom=GEOM) == pytest.approx(0.0)
    assert mbr.lund_emd(a, b, backend="pot") > 1.0


# ---------------------------------------------------------------------------
# WP4a — `_reduce_risk`, and gate G1 (the linear path is bit-identical)
# ---------------------------------------------------------------------------
def test_reduce_risk_against_a_hand_computed_matrix():
    """A 4x4 by hand, all three losses, so the implementation is checked against
    arithmetic rather than against itself."""
    D = np.array([
        [0.0, 1.0, 5.0, 9.0],
        [1.0, 0.0, 4.0, 8.0],
        [5.0, 4.0, 0.0, 4.0],
        [9.0, 8.0, 4.0, 0.0],
    ])
    # row sums 15, 13, 13, 21 over K = 4
    assert np.allclose(mbr._reduce_risk(D, None, loss="linear"),
                       [15 / 4, 13 / 4, 13 / 4, 21 / 4])
    # at eps = 4 the rows have {0,1}, {1,0,4}, {4,0,4}, {4,0} within eps -> counts 2,3,3,2
    counts = np.array([2, 3, 3, 2])
    assert np.array_equal((D <= 4.0).sum(1), counts)
    got = mbr._reduce_risk(D, None, loss="bounded", eps=4.0)
    assert np.allclose(got, 1.0 - counts / 4.0), "risk == 1 - (neighbour fraction)"
    assert int(np.argmin(got)) == int(np.argmax(counts)), (
        "the bounded argmin MAXIMISES the number of neighbours within eps"
    )
    ker = mbr._reduce_risk(D, None, loss="kernel", eps=4.0)
    assert np.allclose(ker, -np.exp(-0.5 * (D / 4.0) ** 2).mean(1))
    assert ker.max() < 0.0, "the kernel risk is a NEGATED density, so its argmin is a mode"
    with pytest.raises(ValueError, match="unknown mbr_loss"):
        mbr._reduce_risk(D, None, loss="huber", eps=1.0)
    with pytest.raises(ValueError, match="bandwidth"):
        mbr._reduce_risk(D, None, loss="bounded", eps=None)


def test_reduce_risk_linear_is_bit_identical_to_the_merged_expression():
    """Gate G1: `max|delta risk| == 0.0`, elementwise, weighted and unweighted. Not
    `approx` — the merged behaviour is `D.mean(axis=1)` and the dispatch must BE it, not
    agree with it to within a tolerance."""
    rng = np.random.default_rng(0)
    D = rng.uniform(0.0, 40.0, (32, 32))
    assert np.abs(mbr._reduce_risk(D, None, loss="linear") - D.mean(axis=1)).max() == 0.0
    w = rng.uniform(0.2, 3.0, 32)
    merged = (D * w[None, :]).sum(axis=1) / w.sum()
    assert np.abs(mbr._reduce_risk(D, w, loss="linear") - merged).max() == 0.0


def test_bandwidth_quantile_excludes_the_zero_pairs():
    """eps is `Q_gamma` of the POSITIVE off-diagonal distances. The exclusion is not
    tidiness: `_empty_value` returns exactly 0 for two empty clouds, so the empty clique is
    invisible to the bandwidth rule while remaining decisive in the neighbour tally — the
    §8.4 hazard in one line."""
    D = np.zeros((10, 10))       # rows 0-4 an empty clique: mutual distance EXACTLY 0
    D[:5, 5:] = 20.0             # ...at a large constant distance from every non-empty draw
    D[5:, :5] = 20.0
    assert mbr.bandwidth_quantile(D, 0.10) == pytest.approx(20.0), (
        "the 50 zero pairs inside the clique are half the matrix and must not set eps"
    )
    assert mbr.bandwidth_quantile(np.zeros((4, 4)), 0.10) == 0.0


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_diagnostic_losses_side_channel(batch):
    """WP4a's containment guarantee: `diagnostic_losses=()` is bit-identical to merged, and
    a non-empty tuple returns the side channel WITHOUT mutating the returned estimate.

    `.risk` keeps meaning "the achieved mean distance" — it has fourteen consumers, five of
    which aggregate it across jets, and none of them break loudly if it silently becomes a
    neighbour deficit."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 3 + [[], []]
    base = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    assert isinstance(base, LundPointEstimate)

    out = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot",
                         diagnostic_losses=("bounded", "kernel"))
    assert isinstance(out, tuple) and len(out) == 2
    pe, side = out
    assert pe.risk == base.risk                      # bit-identical, not approx
    assert [n.cell for n in pe.nodes] == [n.cell for n in base.nodes]
    assert set(side) >= {"linear", "bounded", "kernel", "eps"}
    assert side["eps"] > 0.0
    for k in ("linear", "bounded", "kernel"):
        assert 0 <= side[k] < len(draws)
    # the side channel is a side channel: nothing about it reaches the estimate
    assert pe.cluster_mass is None and pe.cluster_entropy is None


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_empty_clique_dominance(batch):
    """Gate G8' as an executable regression: at a small eps the bounded loss selects the
    EMPTY tree, and at an eps above the clique scale it does not.

    This is the MAP degeneracy the README credits MBR with removing *structurally* — the
    linear loss is immune because an empty cloud pays the full imbalance penalty inside the
    mean, and the bounded loss reintroduces it because `_empty_value` puts every empty draw
    at mutual distance exactly 0."""
    xf, nx, geom = _jet(batch)
    build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    # ~17% empty, the measured rate; the non-empty draws are deliberately DIVERSE, so no
    # non-empty candidate has a large neighbourhood at small eps.
    non_empty = [[c] for c in range(0, 90, 3)][:29]
    draws = non_empty + [[]] * 6
    clouds = [mbr.lund_cloud(d, geom, lnkt_cut=0.0) for d in draws]
    D = mbr.lund_emd_matrix(clouds, clouds, backend="pot")
    mults = np.array([len(d) for d in draws])

    lin = int(np.argmin(mbr._reduce_risk(D, None, loss="linear")))
    assert mults[lin] > 0, "the LINEAR loss must never pick the empty tree here"

    small = 0.5 * float(D[D > 0].min())          # below every non-empty separation
    tiny = int(np.argmin(mbr._reduce_risk(D, None, loss="bounded", eps=small)))
    assert mults[tiny] == 0, "at a small eps the zero-diameter empty clique wins"

    floor = float(np.quantile(D[D > 0], 0.60))   # an eps floor above the clique scale
    big = int(np.argmin(mbr._reduce_risk(D, None, loss="bounded", eps=floor)))
    assert mults[big] > 0, "above the clique scale a non-empty candidate wins again"


# ---------------------------------------------------------------------------
# WP2 — the set-valued prediction, across every family
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not POT_OK, reason="POT not installed")
@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_predict_set_returns_genuine_draws_for_every_family(sel, batch):
    """The cluster layer touches only `D`, so family-agnosticism is a cheap invariant to
    assert — and every member must be a tree the model actually generated."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(sel), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 4 + [[]] * 8
    ps = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                           cluster_method="pam", cluster_min_mass=0.05)
    assert len(ps) == len(ps.masses) == len(ps.radii) >= 1
    assert ps.masses.sum() <= 1.0 + 1e-9
    assert list(ps.masses) == sorted(ps.masses, reverse=True)
    assert ps.top_mass == pytest.approx(ps.masses[0])
    assert ps.point is ps.members[0]
    for m, mass in zip(ps.members, ps.masses):
        assert [n.cell for n in m.nodes] in [list(d) for d in draws]
        assert m.cluster_mass == pytest.approx(mass)
        assert m.cluster_entropy == pytest.approx(ps.entropy)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_predict_set_leaves_the_point_estimate_bit_identical(batch):
    """The plan's headline claim, at the level a caller can observe: taking a SET changes
    nothing about the point estimate, because nothing in the cluster layer touches
    `risk = D.mean(axis=1)`."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 3
    before = model.map_or_mbr(xf, nx, draws=list(draws), point_estimator="mbr",
                              mbr_backend="pot")
    model.predict_set(xf, nx, draws=list(draws), point_estimator="mbr", mbr_backend="pot",
                      cluster_method="pam")
    after = model.map_or_mbr(xf, nx, draws=list(draws), point_estimator="mbr",
                             mbr_backend="pot")
    assert after.risk == before.risk
    assert [n.cell for n in after.nodes] == [n.cell for n in before.nodes]
    assert before.cluster_mass is None and after.cluster_mass is None


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_predict_set_refuses_a_rectangular_D(batch):
    """`notebooks/inference_demo.ipynb` sets MBR_N_CANDIDATES = 24, which is exactly the
    setting that leaves `D` rectangular. Raise rather than silently override: overriding
    would change the point estimate the caller asked for."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56], [12, 34], [5, 34, 56]] * 3
    with pytest.raises(ValueError, match="mbr_n_candidates"):
        model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                          mbr_n_candidates=3, cluster_method="pam")


def test_cluster_decode_defaults_are_all_off():
    """Parity rule: every new switch defaults off with a bit-identical OFF path, and reads
    through the tolerant `decode_params` backfill so a pre-change checkpoint still loads."""
    from h2p_rsd_junipr.config import decode_params

    dec = decode_params(load_config(["model=ar_junipr_v2"]))
    assert dec["cluster_posterior"] is False
    assert dec["cluster_method"] == "hdbscan"
    assert dec["cluster_split"] is False
    assert dec["cluster_min_mass"] == 0.05
    assert dec["cluster_min_cluster_size"] == 0
    assert dec["cluster_eps_quantile"] == 0.10
    assert dec["set_alpha"] == 0.32
    # ...and an OLD snapshot with no cluster block at all backfills rather than raising
    from omegaconf import OmegaConf

    old = OmegaConf.create({"decode": {"point_estimator": "mbr"}})
    assert decode_params(old)["cluster_posterior"] is False


# ---------------------------------------------------------------------------
# N-first (stratified) MBR — docs/PLAN_StratifiedMBR.md WP1
#
# `mbr_select` minimises a mean over EVERY multiplicity stratum at once, and the EMD's
# imbalance term charges ~R|W_a - W_b| across strata — so the medoid is pulled toward
# whatever N is most populous and can represent none of them. The stratified estimator
# decides N from the calibrated q(N|x) and takes the medoid WITHIN that stratum.
# ---------------------------------------------------------------------------
def _strata_fixture():
    """A `D` whose GLOBAL medoid is in a stratum the N decision will not choose.

    mults = [0, 0, 2, 2, 2, 3]. Row 5 (the lone N=3 draw) sits 3 from everything, so it
    has the smallest global mean (2.5) and `mbr_select` would take it. Within the N=2
    stratum the sub-block is [[0,2,4],[2,0,3],[4,3,0]] with row means [2, 5/3, 7/3], so
    the conditional medoid is draw 3. The two answers differ — which is the whole point."""
    D = np.array([
        [0.0, 0.0, 8.0, 8.0, 8.0, 3.0],
        [0.0, 0.0, 8.0, 8.0, 8.0, 3.0],
        [8.0, 8.0, 0.0, 2.0, 4.0, 3.0],
        [8.0, 8.0, 2.0, 0.0, 3.0, 3.0],
        [8.0, 8.0, 4.0, 3.0, 0.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 3.0, 0.0],
    ])
    return D, np.array([0, 0, 2, 2, 2, 3])


def test_stratified_medoid_against_a_hand_computed_matrix():
    D, mults = _strata_fixture()
    assert np.allclose(D, D.T), "the fixture must be a distance matrix"
    # the global reduction picks the N=3 draw...
    assert int(np.argmin(mbr._reduce_risk(D, None, loss="linear"))) == 5
    # ...and the stratified one never can, at n_hat = 2
    win, risk, n_used = mbr.stratified_medoid(D, mults, 2)
    assert (win, n_used) == (3, 2)
    assert risk == pytest.approx(5.0 / 3.0)
    # the reduction IS the sub-block row mean, to the exactness convention of gate G1
    sub = D[np.ix_([2, 3, 4], [2, 3, 4])]
    assert np.abs(mbr._reduce_risk(sub, None, loss="linear") - sub.mean(axis=1)).max() == 0.0
    # the empty stratum is a zero-diameter clique -> risk exactly 0
    assert mbr.stratified_medoid(D, mults, 0) == (0, 0.0, 0)


def test_stratified_medoid_weights_are_a_no_op_when_they_come_from_qn():
    """`_qn_importance_weights` assigns ONE weight per multiplicity, so within a stratum it
    is constant and cancels out of the weighted mean. This estimator is the exact form of
    the correction `mbr_resample_to_qn` approximates by reweighting — assert it rather than
    leave it as a docstring claim."""
    D, mults = _strata_fixture()
    base = mbr.stratified_medoid(D, mults, 2)
    w = np.array([0.3, 0.3, 2.7, 2.7, 2.7, 1.1])      # constant within each stratum
    got = mbr.stratified_medoid(D, mults, 2, w=w)
    # The SELECTION is what must not move; the risk agrees to float precision but not
    # bit-for-bit, because the weighted branch is `(D*w).sum(1)/w.sum()` rather than
    # `D.mean(1)` — a different summation order for the same quantity.
    assert (got[0], got[2]) == (base[0], base[2])
    assert got[1] == pytest.approx(base[1])
    # a genuinely per-draw weight IS applied, so the knob is not silently ignored
    w2 = np.array([1.0, 1.0, 1.0, 1e-6, 1.0, 1.0])    # starve the winner
    assert mbr.stratified_medoid(D, mults, 2, w=w2)[0] != base[0]


def test_stratified_medoid_falls_back_to_the_nearest_populated_stratum():
    """An unrealised median is a legitimate runtime state for an explicit-`q(N|x)` family
    (exact softmax vs a finite pool), so it degrades rather than raising — and it degrades
    INSIDE the realised support, never to the global medoid that would reintroduce the
    smearing on exactly the most N-ambiguous jets."""
    D, mults = _strata_fixture()                        # present {0: 2, 2: 3, 3: 1}
    assert mbr.stratified_medoid(D, mults, 1)[2] == 2, "tie |1-0|=|1-2| -> larger mass"
    assert mbr.stratified_medoid(D, mults, 5)[2] == 3, "nearest populated below"
    assert mbr.stratified_medoid(D, mults, 4)[2] == 3
    # equal mass at equal distance -> smaller n, for determinism
    even = np.array([0, 0, 2, 2])
    assert mbr._nearest_populated(even, 1) == 0


def test_stratified_medoid_guards():
    D, mults = _strata_fixture()
    with pytest.raises(ValueError, match="mbr_n_candidates"):
        mbr.stratified_medoid(D[:4], mults, 2)             # rectangular
    with pytest.raises(ValueError, match="mults has"):
        mbr.stratified_medoid(D, mults[:3], 2)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_stratified_is_a_no_op_when_there_is_nothing_to_stratify(batch):
    """The structural anchor: with every draw at one multiplicity the stratum IS the pool,
    so the estimator must return the same tree and a BIT-IDENTICAL risk as `mbr_select`."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34], [5, 9], [40, 41], [7, 12], [30, 56]]   # all N = 2
    a = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    b = mbr.mbr_select_stratified(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    assert [n.cell for n in a.nodes] == [n.cell for n in b.nodes]
    assert a.risk == b.risk                                   # not approx
    assert a.estimator == "mbr" and b.estimator == "mbr_n"


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_stratified_leaves_mbr_select_bit_identical(batch):
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56]] * 4 + [[5, 9]] * 5 + [[]] * 3
    before = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    mbr.mbr_select_stratified(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    after = mbr.mbr_select(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    assert after.risk == before.risk
    assert [n.cell for n in after.nodes] == [n.cell for n in before.nodes]


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_stratified_selects_the_median_stratum_and_labels_itself(batch):
    """The end-to-end claim: the returned tree's multiplicity IS the posterior median, not
    whatever the global mean distance preferred."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    # median N = 2 (5 of 12 draws at N=2 with 4 below), but the lone N=3 draw is central
    draws = [[]] * 4 + [[12, 34]] * 5 + [[12, 34, 56]] * 3
    pe = mbr.mbr_select_stratified(model, xf, nx, draws=list(draws), geom=geom, backend="pot")
    assert pe.estimator == "mbr_n"
    assert pe.multiplicity == 2, "the N decision is the calibrated median, not the medoid's"
    assert pe.risk is not None and np.isfinite(pe.risk)
    assert pe.pretty().startswith("MBR-N (stratified) groomed shower")
    assert [n.cell for n in pe.nodes] in [list(d) for d in draws]     # H = {pool}


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_stratified_carries_its_own_sampled_coordinates(batch):
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34], [5, 9], [40, 41]] * 2
    coords = [model.sample_coordinates(xf, nx, d) for d in draws]
    pe = mbr.mbr_select_stratified(model, xf, nx, draws=list(draws), geom=geom,
                                   backend="pot", coords_by_draw=coords)
    assert pe.coords_source == "sample"
    win = [list(d) for d in draws].index([n.cell for n in pe.nodes])
    for t, n in enumerate(pe.nodes):
        for j, got in enumerate((n.ln_invDelta, n.ln_kt, n.ln_z, n.psi)):
            assert got == pytest.approx(float(coords[win][t, j]), abs=1e-5)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_stratified_empty_median_answers_the_empty_tree(batch):
    """With the gate off and an empty-dominated posterior the median IS 0, and the honest
    answer is the empty tree at risk exactly 0 — the empty clique has zero diameter."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[]] * 8 + [[12, 34]] * 2
    pe = mbr.mbr_select_stratified(model, xf, nx, draws=draws, geom=geom, backend="pot")
    assert pe.multiplicity == 0 and pe.risk == pytest.approx(0.0)
    assert pe.estimator == "mbr_n"
    assert "the N decision itself" in pe.pretty()
    from h2p_rsd_junipr.inference.clusters import assert_ancestral_draws

    with pytest.raises(ValueError, match="ANCESTRAL"):
        assert_ancestral_draws([pe])


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_stratified_rejects_a_candidate_cap(batch):
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34], [5, 9], [40, 41]] * 2
    with pytest.raises(ValueError, match="mbr_n_candidates"):
        mbr.mbr_select_stratified(model, xf, nx, draws=draws, geom=geom, backend="pot",
                                  n_candidates=3)
    with pytest.raises(ValueError, match="draws"):
        mbr.mbr_select_stratified(model, xf, nx, geom=geom, backend="pot",
                                  D=np.zeros((6, 6)))


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
@pytest.mark.parametrize("sel", MODELS, ids=lambda s: s[0].split("=")[1])
def test_map_or_mbr_dispatches_mbr_n_for_every_family(sel, batch):
    """`point_estimator="mbr_n"` is a new VALUE on the existing knob, so it reaches every
    family through the same dispatch — and the empty gate still runs BEFORE it."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(sel), geom).eval()
    draws = [[12, 34, 56]] * 4 + [[5, 9]] * 5 + [[]] * 3
    pe = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr_n", mbr_backend="pot")
    assert isinstance(pe, LundPointEstimate) and pe.estimator == "mbr_n"
    assert pe.multiplicity == len(pe.nodes)
    assert np.isfinite(pe.logprob) and pe.risk is not None


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_the_empty_gate_is_stage_0_of_the_stratified_decode(batch):
    """The gate runs BEFORE dispatch, so it wins over the N decision when it fires.

    Pinned on a continue/stop family deliberately: there `length_pmf` IS the histogram of
    these draws, so `q(0|x)` is the empty fraction and the gate is exercised. A family with
    an explicit `q(N|x)` head reads its own softmax instead, which an untrained head puts
    almost nowhere near 0 — the gate not firing there is the head talking, not a bug."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[]] * 9 + [[12, 34]]
    assert not hasattr(model, "n_head"), "this test needs the sampler-histogram family"

    gated = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr_n",
                             mbr_backend="pot", empty_threshold=0.5)
    assert gated.estimator == "empty_gate" and gated.multiplicity == 0
    # ...and with the gate off the N decision reaches the same answer on its own, by the
    # median rather than by tau — the two stages agree here, which is why the composition
    # is safe: any sensible tau is below 0.5, so a gate that does NOT fire implies
    # q(0|x) < 0.5 and the median cannot be 0.
    ungated = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr_n",
                               mbr_backend="pot")
    assert ungated.estimator == "mbr_n" and ungated.multiplicity == 0


def test_mbr_n_is_recognised_by_every_point_estimator_consumer():
    """Six call sites tested `point_estimator == "mbr"` exactly; a new value that any of
    them missed would silently drop the MBR series, mark live knobs inert, or lose the
    serving draw-reuse."""
    from h2p_rsd_junipr.eval.report import inert_decode_keys

    class _Stub:
        cont_head = object()

    dec = {"point_estimator": "mbr_n", "mbr_backend": "pot", "mbr_R": 8.485,
           "cluster_posterior": True, "set_alpha": 0.32}
    inert = {e["key"] for e in inert_decode_keys(_Stub(), dec)}
    assert not (inert & {"mbr_backend", "mbr_R"}), "mbr_* knobs are LIVE under mbr_n"
