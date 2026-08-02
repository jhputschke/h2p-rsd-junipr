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
    if len(mbr._loaded_omp_runtimes()) <= 1:
        pytest.skip("single OpenMP runtime: the parallel path is safe here")
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
            p = list(real(coord_in))
            p[7] = torch.full_like(p[7], kappa_value)
            return tuple(p)
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
