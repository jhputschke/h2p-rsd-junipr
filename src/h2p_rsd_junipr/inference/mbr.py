"""Minimum-Bayes-risk (MBR) point estimate under a perturbative-Lund EMD metric.

Post-hoc inference, kept out of the parity-critical `point_estimate.py` (mirrors
`inference/length.py`). Given the posterior draws the caller already takes, MBR
selects the drawn tree of least *expected* Energy-Mover's-Distance to the posterior
(Kumar & Byrne, HLT-NAACL 2004; Eikema & Aziz, COLING 2020 / EMNLP 2022):

    y_hat = argmin_{h in C}  (1/K) sum_k  d(h, y^(k)),   C subset of the K draws.

The empty tree is never selected *without a floor*: an empty Lund cloud pays the
full mass-imbalance penalty against typical non-empty draws, so it can never
minimise the risk — the brevity bias the MinCut floors had to clamp is removed
structurally (`decode.min_emissions` is unnecessary for this estimator).

`d` is the perturbative-Lund EMD (Komiske, Metodiev & Thaler, *Phys. Rev. Lett.*
**123** (2019) 041801, arXiv:1902.02346): each draw becomes a weighted point cloud
in the Lund plane (emissions below `lnkt_cut` dropped) and clouds are compared by
optimal transport with a mass-imbalance penalty of radius `R`.

The OT solve is pluggable via `backend`:
  - ``"pot"`` (default, self-contained): explicit augmented cost + sink particle,
    solved with ``ot.emd2``. No physics package; the imbalance term ``R*|dW|`` is
    written out by hand, matching the EMD equation exactly.
  - ``"energyflow"`` (reference): ``energyflow.emd.emd`` / ``energyflow.emd.emds``.
    EnergyFlow normalises ground distances by ``R`` internally, so (for ``beta=1``)
    its value equals the ``pot`` value divided by ``R`` — the two backends agree on
    the *argmin* but not the numeric scale (pick one backend per analysis).
  - ``"surrogate"``: a fully vectorised binned Lund-image chi^2 (no OT) — a fast
    ranker/pre-filter.

Both ``ot`` and ``energyflow`` are lazy-imported *inside* their path and guarded
with an actionable ``ImportError``, so the default ``point_estimator="map"`` imports
neither and parity stays dependency-free.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import math
import os
import platform
import warnings

import numpy as np

# Which coordinate columns enter the ground metric -> ground dimension gdim.
_COORD_GDIM = {"lnDR_lnkt": 2, "+lnz": 3, "+psi": 4}


def cloud_columns_needed(coords: str = "lnDR_lnkt", weight: str = "kt") -> int:
    """How many coordinate columns per emission this `(coords, weight)` pair reads.

    `coords` fixes the ground dimension; `weight="z"` additionally reads column 2 even at
    `gdim = 2`, because the point weight is `exp(ln z)`. Both are counted here so no caller
    has to re-derive the rule (docs/PLAN_z_aware.md WP-2)."""
    return max(_COORD_GDIM[coords], 3 if weight == "z" else 2)


# What a posterior draw becomes before the ground metric sees it (`decode.mbr_cloud_source`).
CLOUD_SOURCES = ("cells", "coords")


def check_cloud_source(cloud_source: str) -> str:
    """Validate `decode.mbr_cloud_source` and return it — one place, so every entry point
    rejects a typo identically rather than silently falling back to `"cells"`."""
    if cloud_source not in CLOUD_SOURCES:
        raise ValueError(
            f"unknown mbr_cloud_source={cloud_source!r}; expected one of {list(CLOUD_SOURCES)}. "
            f"'cells' builds every cloud from the drawn cell CHAIN (the fielded path); "
            f"'coords' builds it from the continuous coordinate table drawn alongside "
            f"(`coords_for_draws`), which de-quantizes (u, v) and supplies ln z."
        )
    return cloud_source


def needs_continuous_coords(coords: str = "lnDR_lnkt", weight: str = "kt") -> bool:
    """Does this decode configuration need something a Lund CELL cannot supply?

    The grid discretizes `(ln 1/DeltaR, ln kt)` only (`geometry.py`), so a cell centre has
    two coordinates and no `ln z` or `psi` at all. Anything above two columns therefore
    requires a continuous-coordinate table — which `sample_batch` does not return, since
    every family samples cell chains.

    One function rather than the rule re-derived at each site — `lund_cloud`'s own guard,
    the tests, and any caller that wants to check *before* getting an exception. The
    version of this rule that was implicit is what let `mbr_coords="+lnz"` stay inert for
    the whole of the v1 campaign, measuring 2-D numbers under a 3-D label."""
    return cloud_columns_needed(coords, weight) > 2


# ---------------------------------------------------------------------------
# decode -> mbr_select kwargs
# ---------------------------------------------------------------------------
def mbr_kwargs_from_decode(decode: dict) -> dict:
    """Map a `decode_params(cfg)` dict onto `mbr_select`'s keyword arguments."""
    return dict(
        n_samples=int(decode.get("n_posterior_samples", 200)),
        n_candidates=int(decode.get("mbr_n_candidates", 0)),
        lnkt_cut=decode.get("mbr_lnkt_cut", None),
        weight=str(decode.get("mbr_weight", "kt")),
        coords=str(decode.get("mbr_coords", "lnDR_lnkt")),
        cloud_source=str(decode.get("mbr_cloud_source", "cells")),
        R=float(decode.get("mbr_R", 8.485)),
        beta=float(decode.get("mbr_beta", 1.0)),
        norm=bool(decode.get("mbr_norm", False)),
        periodic_phi=bool(decode.get("mbr_periodic_phi", False)),
        phi_col=int(decode.get("mbr_phi_col", -1)),
        backend=str(decode.get("mbr_backend", "pot")),
        resample_to_qn=bool(decode.get("mbr_resample_to_qn", False)),
    )


def cluster_kwargs_from_decode(decode: dict) -> dict:
    """Map a `decode_params(cfg)` dict onto `mbr_cluster_set`'s cluster-layer kwargs.

    Sibling of `mbr_kwargs_from_decode`, kept separate because the two are *orthogonal*:
    the cluster layer reads more off `D`, the risk reduction reduces over `D`, and neither
    sees the other's output (docs/PLAN_PosteriorClusters.md §8.1)."""
    return dict(
        method=str(decode.get("cluster_method", "hdbscan")),
        min_cluster_size=int(decode.get("cluster_min_cluster_size", 0)),
        min_mass=float(decode.get("cluster_min_mass", 0.05)),
        eps_quantile=float(decode.get("cluster_eps_quantile", 0.10)),
        split=bool(decode.get("cluster_split", False)),
        # The SAME knob `map_or_mbr` reads, meaning the same thing at the same stage: with
        # it set, the emptiness decision is the calibrated gate's rather than the cluster
        # mass argmax's. No new config field — see `mbr_cluster_set` for why the argmax is
        # the wrong rule for the N = 0 stratum.
        empty_threshold=float(decode.get("empty_threshold", 0.0)),
    )


# ---------------------------------------------------------------------------
# Cloud adapter (cells -> centres, or v2 continuous nodes -> coords)
# ---------------------------------------------------------------------------
def lund_cloud(draw, geom, *, lnkt_cut=None, weight="kt", coords="lnDR_lnkt"):
    """One posterior draw -> ``(pts (m,g), w (m,))`` weighted Lund cloud.

    ``draw`` is a cell chain (list of cell ids; every model family samples these)
    or an ``(m, >=2)`` continuous-coordinate array. Emissions with ``ln kt <
    lnkt_cut`` are dropped (perturbative support); weights are RAW — *not*
    pre-normalised (``mbr_norm`` decides). ``coords`` selects ``g in {2,3,4}``.
    ``lnkt_cut=None`` inherits the geometry's ``ln_kt`` floor (the region cut).

    **A cell chain under a configuration that reads more than two columns RAISES**
    (docs/PLAN_z_aware.md WP-2). It used to fill ``ln z = psi = 0`` from a cell centre,
    which made ``mbr_coords="+lnz"`` append a constant-zero third column — so it changed
    no distance, and ``mbr_weight="z"`` was silently identical to ``unit``. The knob was
    not merely off by default; it could not be switched on, and nothing said so. Since
    ``sample_batch`` returns cell chains, that is the path the whole pipeline took.
    ``ln z = 0`` also means ``z = 1`` — the softer prong taking the whole jet — so the
    filler is not a neutral default but an unphysical point in the metric.

    Raise rather than warn, matching ``assert_cluster_metric_ok`` and the
    ``lnz_head='spline'`` + ``lnz_support='legacy'`` guard: a decode that silently
    measures something other than what its config says is worse than one that stops. An
    EMPTY draw is untouched — it yields an honestly empty cloud and fabricates nothing."""
    g = _COORD_GDIM[coords]
    need = cloud_columns_needed(coords, weight)
    if lnkt_cut is None:
        lnkt_cut = float(geom.ln_kt_range[0])
    pts, ws = [], []
    for c in draw:
        if isinstance(c, (int, np.integer)):
            if need > 2:
                raise ValueError(
                    f"lund_cloud(coords={coords!r}, weight={weight!r}) reads {need} "
                    f"coordinate columns per emission, but this draw is a CELL CHAIN: a "
                    f"Lund cell centre supplies (ln 1/DeltaR, ln kt) and nothing else, so "
                    f"ln z / psi would be filled with 0 (ln z = 0 means z = 1). Supply an "
                    f"(m, >= {need}) continuous-coordinate array per draw, or set "
                    f"decode.mbr_coords='lnDR_lnkt' with decode.mbr_weight in "
                    f"{{'kt', 'unit'}}. `needs_continuous_coords(coords, weight)` is the "
                    f"same test for a caller that wants to check before getting here."
                )
            u, v = geom.cell_center(int(c))
            lz = ps = 0.0
        else:
            arr = np.asarray(c, dtype=float).ravel()
            if arr.size < need:
                raise ValueError(
                    f"lund_cloud(coords={coords!r}, weight={weight!r}) reads {need} "
                    f"coordinate columns per emission, but an emission row has "
                    f"{arr.size}. A short row was padded with zeros, which is the same "
                    f"silent ln z = 0 (i.e. z = 1) the cell-chain path used to take."
                )
            u, v = float(arr[0]), float(arr[1])
            lz = float(arr[2]) if arr.size > 2 else 0.0
            ps = float(arr[3]) if arr.size > 3 else 0.0
        if v < lnkt_cut:  # perturbative-region restriction = metric support
            continue
        pts.append([u, v, lz, ps][:g])
        if weight == "kt":
            ws.append(math.exp(v))       # IRC-safe momentum scale
        elif weight == "z":
            ws.append(math.exp(lz))
        else:                            # unit
            ws.append(1.0)
    if not pts:
        return np.zeros((0, g), dtype=float), np.zeros((0,), dtype=float)
    return np.asarray(pts, dtype=float), np.asarray(ws, dtype=float)


def _as_coord_table(t) -> np.ndarray:
    """A model's coordinate tensor (or an array) -> a float64 ``(m, 4)`` NumPy table.

    Duck-typed on ``detach`` rather than importing torch: this module is otherwise pure
    NumPy and the ``point_estimator="map"`` path must keep importing nothing new."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().double().numpy()
    return np.asarray(t, dtype=float).reshape(-1, 4)


def coords_for_draws(model, xf, nx, draws) -> list:
    """One jet's `K` draws -> `K` continuous coordinate tables, **index-aligned**.

    ONE batched `sample_coordinates_many` for the whole pool (the per-draw hook re-runs
    `encode()` / `xattn_kv()` on identical conditioning every time — 67 of the 109 min of
    `docs/PLAN_prod_test_speedup.md` §2), returned as float64 `(m_k, 4)` arrays in
    `features.node_raw` column order `(ln 1/DeltaR, ln kt, ln z, psi)`.

    **Unfiltered.** `draws[k]`'s table is `out[k]`, including the empty draws — an empty
    draw yields an honest `(0, 4)` and keeps its slot. Filtering is what `run_closure`'s
    pre-existing continuous block does (`[list(d) for d in draws if len(d)]`) and it is
    exactly what makes that call unusable here: a cloud list must line up with `draws`
    position by position or the winner index means nothing. It also changes `L_max`, hence
    the padded block shape, hence RNG consumption — which is why the unfiltered call lives
    strictly behind `decode.mbr_cloud_source="coords"` (docs/PLAN_z_aware.md §7.1).

    **Raises by family name** when the family has no coordinate density, rather than
    returning `None`s for a caller to trip over later: `ar_junipr_v1` samples cell chains
    and nothing else, so a `"coords"` decode of it is a misconfiguration, not a data state.
    Same convention as `skeleton_search_spec` (docs/PLAN_z_aware.md §4/WP-3)."""
    fam = getattr(model, "model_name", None) or type(model).__name__
    if not getattr(model, "has_continuous_coords", False):
        raise ValueError(
            f"decode.mbr_cloud_source='coords' needs continuous coordinates, but the "
            f"model family {fam!r} ({type(model).__name__}) has "
            f"has_continuous_coords=False — it samples cell chains and has no "
            f"q(coords | cells, x) to draw from, so every cloud would be built from "
            f"placeholders (ln z = 0 means z = 1). Set decode.mbr_cloud_source='cells' "
            f"for this family."
        )
    tables = model.sample_coordinates_many(xf, nx, [list(d) for d in draws])
    if len(tables) != len(draws):
        raise ValueError(
            f"sample_coordinates_many returned {len(tables)} tables for {len(draws)} "
            f"draws; `coords_for_draws` is index-aligned by contract."
        )
    out = []
    for k, (t, d) in enumerate(zip(tables, draws)):
        if t is None:
            raise ValueError(
                f"model family {fam!r} returned None coordinates for draw {k} while "
                f"has_continuous_coords is True — the flag and the hook must agree "
                f"(models/base.py sample_coordinates)."
            )
        a = _as_coord_table(t)
        if a.shape[0] != len(d):
            raise ValueError(
                f"coordinate table for draw {k} has {a.shape[0]} rows for a "
                f"{len(d)}-emission draw; the tables must be index-aligned AND "
                f"row-aligned with the cell chains they complete."
            )
        out.append(a)
    return out


def cloud_to_event(pts, w) -> np.ndarray:
    """EnergyFlow event layout: ``(m, 1+g)`` with the weight in column 0."""
    pts = np.asarray(pts, dtype=float)
    w = np.asarray(w, dtype=float)
    g = pts.shape[1] if pts.ndim == 2 else 0
    if pts.shape[0] == 0:
        return np.zeros((0, g + 1), dtype=float)
    return np.concatenate([w[:, None], pts], axis=1)


# ---------------------------------------------------------------------------
# Ground metric + backend primitives
# ---------------------------------------------------------------------------
def _ground(pa, pb, beta, periodic_phi, phi_col):
    """Pairwise ``||p_i - p'_j||^beta`` with an optional periodic (wrapped) column."""
    diff = pa[:, None, :] - pb[None, :, :]
    if periodic_phi and pa.shape[1] > 0:
        col = phi_col if phi_col >= 0 else pa.shape[1] - 1
        diff = diff.copy()
        d = np.abs(diff[..., col])
        diff[..., col] = np.minimum(d, 2.0 * np.pi - d)
    dist = np.sqrt((diff**2).sum(-1))
    return dist**beta


def _empty_value(wa, wb, R, norm, ground_scale):
    """Distance for a pair where at least one cloud is empty (pure imbalance term).

    ``ground_scale`` is ``R`` for the hand-rolled `pot` cost and ``1.0`` for
    EnergyFlow's R-normalised convention."""
    Wa, Wb = float(np.sum(wa)), float(np.sum(wb))
    if Wa == 0.0 and Wb == 0.0:
        return 0.0
    if norm:  # one empty, one unit-normalised cloud: a full move at radius R
        return float(ground_scale)
    return abs(Wa - Wb) * ground_scale


def _emd_pot(pa, wa, pb, wb, *, R, beta, norm, periodic_phi, phi_col) -> float:
    """Augmented-cost EMD via ``ot.emd2``: pad each cloud with a sink particle that
    absorbs the other's total weight at ground distance ``R``. The optimal transport
    then routes the excess mass ``|Wa-Wb|`` to a sink at cost ``R``, reproducing the
    ``sum f_ij ||p_i-p'_j||^beta + R|Wa-Wb|`` EMD exactly."""
    try:
        import ot  # lazy, per-backend
    except ImportError as e:  # pragma: no cover - exercised only without POT
        raise ImportError(
            "mbr_backend='pot' requires POT: pip install 'pot>=0.9' "
            "(or `pip install -e \".[mbr]\"`)."
        ) from e
    ma, mb = pa.shape[0], pb.shape[0]
    if ma == 0 and mb == 0:
        return 0.0
    Wa, Wb = float(wa.sum()), float(wb.sum())
    if norm and Wa > 0 and Wb > 0:  # removes the imbalance term (mbr_norm)
        wa, wb = wa / Wa, wb / Wb
        Wa = Wb = 1.0
    C = np.zeros((ma + 1, mb + 1), dtype=np.float64)
    if ma and mb:
        C[:ma, :mb] = _ground(pa, pb, beta, periodic_phi, phi_col)
    C[:ma, mb] = R  # real A -> B-sink (unmatched mass)
    C[ma, :mb] = R  # A-sink  -> real B
    C[ma, mb] = 0.0
    a = np.concatenate([wa, [Wb]]).astype(np.float64)
    b = np.concatenate([wb, [Wa]]).astype(np.float64)
    if a.sum() <= 0:
        return 0.0
    return float(ot.emd2(a, b, C))


_OMP_RUNTIME_NAMES = ("libomp.dylib", "libiomp5.dylib", "libgomp")


def _loaded_omp_runtimes() -> set:
    """Realpaths of every OpenMP runtime currently mapped into this process.

    Walks dyld's image list; `realpath` so that a symlinked duplicate counts once.

    An inventory, not a verdict: which of these can make a thread team *fatal* is
    `_wasserstein_omp_conflict`'s call, taken on two snapshots of this set rather than on
    its size, because a stranger's vendored copy is mapped here too and is harmless.

    **Darwin only, and empty everywhere else — deliberately, not as a stub.** The abort
    it feeds is a macOS phenomenon: two LLVM OpenMP runtimes abort, whereas on Linux GCC's
    libgomp coexists with the one PyTorch bundles, which is why `_guard_wasserstein_openmp`
    also returns early there. Enumerating `/proc/self/maps` on Linux would return paths
    that read like a hazard and are not one.

    It must not RAISE off Darwin either: `_dyld_image_count` is a dyld symbol, so the
    ctypes lookup fails with `undefined symbol` on Linux, and any caller probing the
    state — including the test that asserts the pinning invariant — died on the probe
    rather than skipping."""
    if platform.system() != "Darwin":
        return set()
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    found = set()
    for i in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(i)
        if not raw:
            continue
        name = raw.decode(errors="replace")
        if any(k in os.path.basename(name) for k in _OMP_RUNTIME_NAMES):
            found.add(os.path.realpath(name))
    return found


_REBUILD_HINT = (
    "Numbers are unaffected, but the batched solve loses ~11x. Fix: rebuild "
    "wasserstein against the same libomp PyTorch loads -- see the [energyflow] "
    "notes in README.md."
)

# Threads for the batched `emds`; None = energyflow's default (every core). Set to 1
# only by the guard below, when a duplicate OpenMP runtime makes a team fatal.
_EMDS_N_JOBS = None
_OPENMP_GUARDED = False
# Whether wasserstein's own runtime is the duplicate. None until the guard has run;
# it is the hazard the guard acts on, and what a test must skip on.
_OMP_CONFLICT = None


def _wasserstein_omp_conflict(before: set, after: set, *, precommitted: bool) -> bool:
    """Did *wasserstein* pull in a second OpenMP runtime -- as opposed to merely running
    in a process that happens to hold several?

    ``before``/``after`` bracket the dlopen of wasserstein's OpenMP extension, so
    ``after - before`` is whatever its ``@rpath/libomp.dylib`` resolved to, and is empty
    when that resolved to a runtime already mapped.

    Identifying wasserstein's runtime is the point; counting all of them is what this
    replaces. Third-party wheels vendor private copies -- ``sklearn/.dylibs/libomp.dylib``
    is one, and POT imports `sklearn` when it is installed, so *every* `import energyflow`
    maps it. Merely having scikit-learn in the environment therefore took the count from
    one to two and downgraded a build that was correctly sharing PyTorch's runtime,
    costing ~11x for a hazard that was not there. A stranger's copy is invisible to
    wasserstein, whose team is created inside the runtime *it* linked; only a different
    one under its own rpath can make that team fatal.

    ``precommitted`` covers something having imported the extension before the guard ran:
    the resolve already happened, so ``after - before`` is empty for the wrong reason and
    nothing is left but the conservative count."""
    if precommitted:
        return len(after) > 1
    return bool(after - before) and len(after) > 1


def _guard_wasserstein_openmp() -> None:
    """macOS: keep `wasserstein`'s OpenMP off when it links a runtime of its own.

    PyTorch bundles `torch/lib/libomp.dylib`, and `import energyflow` loads it before
    we get here (energyflow -> POT -> `ot.backend` imports torch). If `wasserstein`'s
    extension then resolves ``@rpath/libomp.dylib`` to a *different* file -- conda's
    ``$CONDA_PREFIX/lib/libomp.dylib``, which is what the README's macOS build command
    linked -- the process holds two LLVM OpenMP runtimes. Creating the batched `emds`
    thread team then **segfaults**: a Jupyter kernel simply dies, no traceback. The
    ``KMP_DUPLICATE_LIB_OK=TRUE`` this function used to set unconditionally is exactly
    what downgraded OpenMP's own self-explaining "Error #15" abort into that silent
    SIGSEGV, so it is no longer set unless it is the only thing left to try.

    Probe rather than guess: dlopen the OpenMP extension -- which resolves its rpath but
    starts no parallel region, so loading is safe even though `emds` is not -- bracketed
    by a snapshot of the mapped runtimes, so the diff names the one *it* linked rather
    than every one in the process (see `_wasserstein_omp_conflict`). Which repair is
    available depends on who got here first:

      * Nothing has touched the extension yet -> take wasserstein's own documented
        Darwin opt-out (`without_openmp()`), which selects its no-OpenMP build. Clean:
        no parallel region is ever entered, and no unsupported env var is involved.
      * Something already ran an EMD (a probe, an earlier cell) -> the OpenMP build is
        committed and cannot be swapped. Fall back to suppressing the duplicate-runtime
        abort and pinning `emds` to `n_jobs=1`: with a team of one no second team is
        spawned, which is what actually avoids the crash. Verified elementwise against
        the single-runtime result: max deviation 0.0.

    Cost, measured on an M-series 16-core (100x100 clouds of 3-20 points): 2.9 us/pair
    with one shared runtime, 32.7 single-threaded, 43.1 on the per-pair `emd`. The
    batched path is still worth taking, but a shared runtime is worth ~11x more --
    hence a warning that names the repair rather than a silent degradation."""
    global _EMDS_N_JOBS, _OPENMP_GUARDED, _OMP_CONFLICT
    if _OPENMP_GUARDED or platform.system() != "Darwin":
        return  # Linux: GCC's libgomp coexists with torch's runtime without aborting
    try:
        import wasserstein
        import wasserstein.config as wconfig
    except ImportError:  # pragma: no cover - energyflow imports wasserstein itself
        return
    sos = glob.glob(os.path.join(os.path.dirname(wasserstein.__file__),
                                 "_wasserstein_omp*.so"))
    if not sos:  # pragma: no cover - no OpenMP build exists, so nothing to guard
        return
    before = _loaded_omp_runtimes()  # bracket the dlopen: the diff is wasserstein's own
    try:
        ctypes.CDLL(sos[0], mode=ctypes.RTLD_LOCAL)  # resolve its libomp; no OMP init
    except OSError:  # pragma: no cover - an unloadable extension fails at first use
        return
    _OPENMP_GUARDED = True
    precommitted = not wconfig._CAN_SET_OPENMP  # extension already imported by someone
    _OMP_CONFLICT = _wasserstein_omp_conflict(
        before, _loaded_omp_runtimes(), precommitted=precommitted
    )
    if not _OMP_CONFLICT:
        return  # shares the mapped runtime -> the parallel `emds` is safe, and ~11x faster
    if not precommitted:
        wconfig.without_openmp()  # no extension loaded yet: pick the no-OpenMP build
        warnings.warn(
            "Two OpenMP runtimes are loaded (PyTorch's bundled libomp and the one "
            "`wasserstein` links); its parallel `emds` would segfault, so wasserstein "
            "was switched to its single-threaded build. " + _REBUILD_HINT,
            RuntimeWarning, stacklevel=3,
        )
        return
    from wasserstein import _wasserstein as _wext
    if not _wext.cvar.COMPILED_WITH_OPENMP:
        return  # already committed to the no-OpenMP build; nothing can go wrong
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # or the team creation aborts outright
    _EMDS_N_JOBS = 1
    warnings.warn(
        "Two OpenMP runtimes are loaded (PyTorch's bundled libomp and the one "
        "`wasserstein` links) and its OpenMP build is already committed, so `emds` "
        "is pinned to one thread to avoid a segfault. " + _REBUILD_HINT,
        RuntimeWarning, stacklevel=3,
    )


def _import_ef():
    try:
        import energyflow as ef  # lazy, per-backend
    except ImportError as e:  # pragma: no cover - exercised only without energyflow
        raise ImportError(
            "mbr_backend='energyflow' requires the optional [energyflow] extra: "
            "pip install 'energyflow>=1.3' (and a working `wasserstein` build)."
        ) from e
    _guard_wasserstein_openmp()  # after energyflow, which is what pulls in torch's libomp
    return ef


class _NumpyCopyShim:
    """Proxy for the ``numpy`` module that restores NumPy-1 ``copy=False`` semantics.

    NumPy 1 read ``copy=False`` as "copy only if you must"; NumPy 2 reads it as
    "never copy, raise if you must" and spells the old meaning ``copy=None``.
    Everything but ``array`` forwards untouched."""

    def __init__(self, np_mod):
        self._np = np_mod

    def __getattr__(self, name):
        return getattr(self._np, name)

    def array(self, obj, *args, **kwargs):
        if kwargs.get("copy", True) is False:
            kwargs["copy"] = None
        return self._np.array(obj, *args, **kwargs)


@contextlib.contextmanager
def _wasserstein_numpy2_compat():
    """Swap `wasserstein.wasserstein`'s module-global ``np`` for the shim above.

    `wasserstein` 1.1.0's `_store_events` does
    ``np.array(event[:, 0], order='C', copy=False)``; a column slice is strided, so
    ``order='C'`` requires a copy and NumPy >= 2 raises rather than copying. That kills
    the *only* multi-core path (`emds` -> `PairwiseEMD`, OpenMP over all pairs) and
    drops us to the per-pair `emd` on one core — measured ~16x on a 20-core host, for
    bit-identical numbers. It is a break in wasserstein's *Python* layer, so rebuilding
    the C++ extension does not fix it.

    Patching that one module's namespace (not the global `numpy`) keeps the blast
    radius to the batched call. Yields False when `wasserstein` is not importable."""
    try:
        import wasserstein.wasserstein as _ww
    except ImportError:  # pragma: no cover - energyflow imports wasserstein itself
        yield False
        return
    original = _ww.np
    _ww.np = _NumpyCopyShim(original)
    try:
        yield True
    finally:
        _ww.np = original


def _emds_block(ef, eventsC, eventsS, *, R, beta, norm, gdim, periodic_phi, shape):
    """One batched, OpenMP-parallel ``emds`` call over the non-empty sub-block.

    ``n_jobs`` is passed only when `_guard_wasserstein_openmp` pinned it, so the
    default stays energyflow's own (every core) rather than a number we chose."""
    extra = {} if _EMDS_N_JOBS is None else {"n_jobs": _EMDS_N_JOBS}
    return np.asarray(
        ef.emd.emds(eventsC, eventsS, R=R, beta=beta, norm=norm,
                    gdim=gdim, periodic_phi=periodic_phi, **extra),
        dtype=float,
    ).reshape(shape)


def _emd_ef(pa, wa, pb, wb, *, R, beta, norm, periodic_phi) -> float:
    ef = _import_ef()
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return _empty_value(wa, wb, R, norm, 1.0)  # EnergyFlow's 1/R ground scale
    g = pa.shape[1]
    return float(
        ef.emd.emd(cloud_to_event(pa, wa), cloud_to_event(pb, wb),
                   R=R, beta=beta, norm=norm, gdim=g, periodic_phi=periodic_phi)
    )


def _lund_image(pts, w, geom, nb=None) -> np.ndarray:
    """Normalised binned Lund image over ``(ln 1/DeltaR, ln kt)`` — the surrogate.

    ``nb`` defaults to ``geom.n_bins``, i.e. the surrogate bins at exactly the
    resolution the model decides at. It used to be hard-coded to 10, which happened to
    agree with the only geometry in use; at ``n_bins: 30`` that made the risk function
    bin at 0.2-wide truth in 0.6-wide cells, so the surrogate was *coarser than the
    model* and MBR could not tell apart candidates the model distinguishes. Pass ``nb``
    explicitly only to deliberately decouple the two."""
    nb = int(geom.n_bins if nb is None else nb)
    img = np.zeros((nb, nb), dtype=float)
    if pts.shape[0] == 0:
        return img.ravel()
    lo_u, hi_u = geom.ln_invdelta_range
    lo_v, hi_v = geom.ln_kt_range
    iu = np.clip(((pts[:, 0] - lo_u) / (hi_u - lo_u) * nb).astype(int), 0, nb - 1)
    iv = np.clip(((pts[:, 1] - lo_v) / (hi_v - lo_v) * nb).astype(int), 0, nb - 1)
    np.add.at(img, (iu, iv), w)
    s = img.sum()
    if s > 0:
        img /= s
    return img.ravel()


def _chi2(ha, hb) -> float:
    denom = ha + hb
    mask = denom > 0
    return float(0.5 * (((ha - hb) ** 2)[mask] / denom[mask]).sum())


def _infer_gdim(*cloud_lists) -> int:
    for clouds in cloud_lists:
        for pts, _ in clouds:
            if pts.shape[0] > 0:
                return int(pts.shape[1])
    return 2


# ---------------------------------------------------------------------------
# Public distance API
# ---------------------------------------------------------------------------
def lund_emd(cloud_a, cloud_b, *, R=8.485, beta=1.0, norm=False,
             periodic_phi=False, phi_col=-1, backend="pot", geom=None) -> float:
    """Single-pair perturbative-Lund EMD. Each cloud is a ``(pts, w)`` pair."""
    pa, wa = cloud_a
    pb, wb = cloud_b
    if backend == "energyflow":
        return _emd_ef(pa, wa, pb, wb, R=R, beta=beta, norm=norm, periodic_phi=periodic_phi)
    if backend == "surrogate":
        if geom is None:
            raise ValueError("mbr_backend='surrogate' needs `geom`")
        return _chi2(_lund_image(pa, wa, geom), _lund_image(pb, wb, geom))
    return _emd_pot(pa, wa, pb, wb, R=R, beta=beta, norm=norm,
                    periodic_phi=periodic_phi, phi_col=phi_col)


def _matrix_ef(clouds_C, clouds_S, *, R, beta, norm, periodic_phi) -> np.ndarray:
    """``|C| x |S|`` matrix via one batched ``energyflow.emd.emds`` call on the
    non-empty sub-block; empty rows/cols get the imbalance-only value."""
    ef = _import_ef()
    g = _infer_gdim(clouds_C, clouds_S)
    D = np.zeros((len(clouds_C), len(clouds_S)), dtype=float)
    nzC = [i for i, (p, _) in enumerate(clouds_C) if p.shape[0] > 0]
    nzS = [j for j, (p, _) in enumerate(clouds_S) if p.shape[0] > 0]
    if nzC and nzS:
        eventsC = [cloud_to_event(*clouds_C[i]) for i in nzC]
        eventsS = [cloud_to_event(*clouds_S[j]) for j in nzS]
        kw = dict(R=R, beta=beta, norm=norm, gdim=g, periodic_phi=periodic_phi,
                  shape=(len(nzC), len(nzS)))
        try:  # one batched, OpenMP-parallel call for the whole non-empty block
            sub = _emds_block(ef, eventsC, eventsS, **kw)
        except (ValueError, TypeError):
            try:  # same call, with wasserstein's NumPy-1 copy semantics restored
                with _wasserstein_numpy2_compat() as patched:
                    if not patched:
                        raise
                    sub = _emds_block(ef, eventsC, eventsS, **kw)
            except (ValueError, TypeError):
                # Batched path unavailable for some other reason. The per-pair `emd`
                # is unaffected and gives identical numbers, but runs on one core --
                # loud, because it is the difference between seconds and minutes.
                warnings.warn(
                    "energyflow's batched `emds` is unavailable; falling back to the "
                    "per-pair `emd`. Results are identical but single-threaded (~16x "
                    "slower on a 20-core host). See the [energyflow] notes in README.md.",
                    RuntimeWarning, stacklevel=2,
                )
                sub = np.array([[float(ef.emd.emd(ea, eb, R=R, beta=beta, norm=norm,
                                                  gdim=g, periodic_phi=periodic_phi))
                                 for eb in eventsS] for ea in eventsC])
        for a, i in enumerate(nzC):
            for b, j in enumerate(nzS):
                D[i, j] = sub[a, b]
    for i, (p, w) in enumerate(clouds_C):
        for j, (q, ww) in enumerate(clouds_S):
            if p.shape[0] == 0 or q.shape[0] == 0:
                D[i, j] = _empty_value(w, ww, R, norm, 1.0)
    return D


def lund_emd_matrix(clouds_C, clouds_S, *, R=8.485, beta=1.0, norm=False,
                    periodic_phi=False, phi_col=-1, backend="pot", geom=None) -> np.ndarray:
    """``|C| x |S|`` pairwise-distance matrix — the primitive MBR uses.

    ``energyflow`` is one batched ``emds`` call; ``pot`` loops ``ot.emd2``;
    ``surrogate`` is a vectorised chi^2 over binned Lund images."""
    nC, nS = len(clouds_C), len(clouds_S)
    if backend == "energyflow":
        return _matrix_ef(clouds_C, clouds_S, R=R, beta=beta, norm=norm, periodic_phi=periodic_phi)
    if backend == "surrogate":
        if geom is None:
            raise ValueError("mbr_backend='surrogate' needs `geom`")
        imgC = [_lund_image(p, w, geom) for p, w in clouds_C]
        imgS = [_lund_image(p, w, geom) for p, w in clouds_S]
        D = np.zeros((nC, nS), dtype=float)
        for i in range(nC):
            for j in range(nS):
                D[i, j] = _chi2(imgC[i], imgS[j])
        return D
    D = np.zeros((nC, nS), dtype=float)  # pot (default)
    for i, (pa, wa) in enumerate(clouds_C):
        for j, (pb, wb) in enumerate(clouds_S):
            D[i, j] = _emd_pot(pa, wa, pb, wb, R=R, beta=beta, norm=norm,
                               periodic_phi=periodic_phi, phi_col=phi_col)
    return D


# ---------------------------------------------------------------------------
# q(N|x) exposure-bias correction (decode layer only)
# ---------------------------------------------------------------------------
def _qn_importance_weights(model, xf, nx, draws) -> np.ndarray:
    """Importance weights over the support draws that reweight the empirical
    posterior multiplicity marginal to the model's calibrated ``q(N|x)``:

        w_k = q(N=|y^(k)| | x) / p_emp(N=|y^(k)|).

    Corrects the Monte-Carlo risk's multiplicity marginal at the decoding layer
    only — the trained likelihood is untouched (contrast: minimum-risk / sequence
    fine-tuning). For a family without an explicit head, ``length_pmf`` reuses these
    same draws, so ``p_emp == q(N|x)`` and the weights collapse to uniform (a no-op)."""
    mults = np.array([len(d) for d in draws], dtype=int)
    K = len(mults)
    if K == 0:
        return np.ones(0, dtype=float)
    pmf = np.asarray(model.length_pmf(xf, nx, mults=mults.tolist()), dtype=float)
    p_emp = np.bincount(mults) / K
    w = np.array(
        [(float(pmf[n]) if n < pmf.size else 0.0) / p_emp[n] if p_emp[n] > 0 else 0.0
         for n in mults],
        dtype=float,
    )
    if w.sum() <= 0:  # calibrated pmf disjoint from the draws -> fall back to uniform
        return np.ones(K, dtype=float)
    return w


# ---------------------------------------------------------------------------
# The risk reduction (docs/PLAN_PosteriorClusters.md WP4a)
# ---------------------------------------------------------------------------
def bandwidth_quantile(D, gamma: float = 0.10) -> float:
    """`Q_gamma` of the POSITIVE off-diagonal distances — the pre-registered epsilon.

    `gamma = 0.10` is fixed before any test run and recorded with `fitted_under`. Tuning
    epsilon against closure metrics is forbidden: it is the one free parameter the bounded
    construction turns on, and a closure-tuned bandwidth makes gate G7 circular. The
    quantile form also makes epsilon invariant to the `mbr_norm` / `energyflow` 1/R
    convention, which is *why* it is a quantile rather than an absolute.

    Only positive entries enter, which excludes both the zero diagonal and the
    empty-empty pairs. That exclusion is not a convenience — it is exactly the §8.4
    hazard: `_empty_value` puts every empty draw at mutual distance 0, so the empty clique
    is invisible to the bandwidth rule while remaining decisive in the neighbour tally."""
    d = np.asarray(D, dtype=float)
    pos = d[d > 0]
    if pos.size == 0:
        return 0.0
    return float(np.quantile(pos, float(gamma)))


def _reduce_risk(D, w=None, *, loss: str = "linear", eps=None) -> np.ndarray:
    """Row-wise Bayes risk of each candidate under the configured loss.

    The general Bayes estimator is `y_hat = argmin_{y' in H} E_{y ~ q}[Delta(y', y)]`, and
    the *character* of the answer is fixed by `Delta` (Goel & Byrne, *Computer Speech &
    Language* **14** (2000) 115; Berger, *Statistical Decision Theory and Bayesian
    Analysis*, Springer 1985, §2.4):

      - ``linear``  — `Delta = d`, so the argmin is the Frechet median restricted to the
        sample. **This is the merged behaviour and is bit-identical**: with `w=None` the
        expression below is literally `D.mean(axis=1)`, and with weights it is literally
        the `resample_to_qn` line it replaced.
      - ``bounded`` — `Delta = 1[d > eps]`, so the risk is `1 - (neighbour fraction)` and
        the argmin MAXIMISES the number of neighbours within `eps`: a Parzen window
        (Silverman, *Density Estimation*, Chapman & Hall 1986, §3) evaluated on the pool,
        i.e. a KDE mode restricted to valid draws.
      - ``kernel``  — the same idea with a Gaussian window instead of a top hat.

    `w` is uniform unless `resample_to_qn`, so the existing q(N|x) correction composes with
    all three losses unchanged. **Cost: zero additional EMD calls.**

    Two warnings that belong at the definition rather than in a plan (§8.3, §8.4):
    under `bounded`/`kernel` the returned number is NOT an EMD — it is dimensionless and
    in [0, 1] (bounded) or negative (kernel) — and the N = 0 stratum forms a zero-diameter
    clique whose neighbour count is its own size at any `eps`, so a small `eps` can collapse
    the estimate to the empty tree. That is why WP4a keeps this an eval-only side channel
    and `.risk` keeps the linear value."""
    D = np.asarray(D, dtype=float)
    if loss == "linear":
        return D.mean(axis=1) if w is None else (D * w[None, :]).sum(axis=1) / w.sum()
    if eps is None or not np.isfinite(eps) or eps <= 0:
        raise ValueError(
            f"mbr_loss={loss!r} needs a positive bandwidth eps; got {eps!r}. Use "
            f"`bandwidth_quantile(D, gamma)` — the pre-registered per-jet rule."
        )
    if loss == "bounded":
        hit = (D <= float(eps)).astype(float)
    elif loss == "kernel":
        hit = np.exp(-0.5 * (D / float(eps)) ** 2)
    else:
        raise ValueError(f"unknown mbr_loss={loss!r}; expected linear | bounded | kernel")
    num = hit.sum(axis=1) if w is None else (hit * w[None, :]).sum(axis=1)
    den = float(D.shape[1]) if w is None else float(w.sum())
    return -num / den if loss == "kernel" else 1.0 - num / den


def stratified_medoid(D, mults, n_hat, *, w=None) -> tuple[int, float, int]:
    """The Frechet medoid of ONE multiplicity stratum (docs/PLAN_StratifiedMBR.md WP1).

    `argmin` over the rows of ``D[stratum, stratum]`` with
    ``stratum = {k : mults[k] == n_used}`` — the candidate set *and* the expectation are
    both restricted, which is what distinguishes this from `_reduce_risk` (which varies the
    per-pair loss and always keeps every column). Returns
    ``(win_idx, risk, n_used)`` with `win_idx` a **global** draw index.

    **Why restrict the expectation.** The perturbative-Lund EMD carries a mass-imbalance
    term, so a draw of multiplicity `m` pays `~R|W_a - W_b|` against every draw of a
    different multiplicity. The global medoid therefore minimises a mean over *all* strata
    and can sit in the wrong one, or between two — measured on 600 held-out jets, that
    smearing is what leaves the linear medoid 2.349 from truth while the closest cluster
    exemplar is 1.476, with 83% of the resolvable posterior ambiguity being between-N.
    Conditioning on `N` removes the imbalance term from the reduction entirely: within a
    stratum every pair has equal total weight, so what is left is pure shape.

    **`w` is an exact no-op here when it comes from `_qn_importance_weights`,** which
    assigns one weight per multiplicity — constant within a stratum, so it cancels out of
    the weighted mean. That is not a coincidence: this estimator is the exact form of the
    correction `decode.mbr_resample_to_qn` approximates by reweighting. It is still
    accepted and applied, for a caller with genuinely per-draw weights.

    Zero EMD calls: `D` is already built. Pure NumPy — no model, no torch."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"stratified_medoid needs a square K x K distance matrix, got {D.shape}. With "
            f"mbr_n_candidates != 0 the row indices are not the column indices, so "
            f"restricting both to one stratum is undefined — reset the candidate cap."
        )
    m = np.asarray(mults, dtype=int).reshape(-1)
    if m.size != D.shape[0]:
        raise ValueError(f"mults has {m.size} entries for a {D.shape[0]}-draw matrix")
    if m.size == 0:
        raise ValueError("stratified_medoid needs at least one draw")

    n_used = _nearest_populated(m, int(n_hat))
    idx = np.flatnonzero(m == n_used)
    sub = D[np.ix_(idx, idx)]
    ws = None if w is None else np.asarray(w, dtype=float).reshape(-1)[idx]
    # The uniform branch is literally `sub.mean(axis=1)` — the same expression, and the
    # same exactness convention, as `_reduce_risk`'s linear path.
    risk = _reduce_risk(sub, ws, loss="linear")
    best = int(np.argmin(risk))
    return int(idx[best]), float(risk[best]), int(n_used)


def _nearest_populated(mults, n_hat: int) -> int:
    """`n_hat` if the pool realises it, else the nearest multiplicity that it does.

    The median of a *histogram* pmf is always realised — `quantile_floor` returns the
    smallest n with `cdf(n) >= alpha`, which forces `pmf[n] > 0` — so on a continue/stop
    family, whose `length_pmf` IS the draw histogram, this never fires. It exists for a
    family with an EXPLICIT `q(N|x)` head, where an exact softmax median can fall on a
    multiplicity the finite pool happens not to contain.

    Nearest by `|n - n_hat|`, ties to the larger pool mass (the posterior's own vote
    between two equidistant strata), then to the smaller `n` for determinism.

    **Not a raise:** an unrealised median is a legitimate runtime state, not a
    misconfiguration — the repo raises for non-metric or rectangular `D` and degrades with
    a note for data states (`_qn_importance_weights` falls back to uniform; a degenerate
    `D` becomes one zero-radius cluster). **And not the global medoid:** that would
    silently revert to the estimator this one exists to replace, on precisely the jets
    where the length belief and the sampler disagree — the most N-ambiguous ones, where
    the smearing it removes is worst. Staying inside the realised support is the same
    `H = {pool}` discipline `mbr_cluster_set` applies to the empty stratum."""
    m = np.asarray(mults, dtype=int).reshape(-1)
    present, counts = np.unique(m, return_counts=True)
    if np.any(present == int(n_hat)):
        return int(n_hat)
    # |n - n_hat| is the L1 loss whose Bayes estimator is the median, so "nearest
    # populated" is that same decision restricted to the answers the pool can give.
    order = np.lexsort((present, -counts, np.abs(present - int(n_hat))))
    return int(present[order[0]])


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
def posterior_distances(model, xf, nx, *, draws=None, geom, n_samples=200, n_candidates=0,
                        lnkt_cut=None, weight="kt", coords="lnDR_lnkt", R=8.485, beta=1.0,
                        norm=False, periodic_phi=False, phi_col=-1, backend="pot",
                        coords_by_draw=None, cloud_source="cells"):
    """Everything both the point estimate and the cluster layer need, computed once.

    Returns `(draws, clouds, cand_idx, D)`. Factored out of `mbr_select` so `predict_set`
    and the WP4a diagnostics read the SAME `D` the point estimate was selected from —
    recomputing it would be `K^2` EMD solves for a matrix that is already in hand, and
    (worse) it would let the two products drift apart under a config change.

    `cloud_source` (`decode.mbr_cloud_source`) picks what each draw becomes:

      * ``"cells"`` — the drawn cell chain, i.e. cell centres. The fielded path, and the
        one every committed artifact was produced under.
      * ``"coords"`` — the continuous coordinate table in `coords_by_draw[k]`, which
        de-quantizes `(u, v)` *and* supplies `ln z` / `psi`. This is what makes
        `mbr_coords="+lnz"` and `mbr_weight="z"` functional instead of fatal
        (docs/PLAN_z_aware.md §4/WP-3, measured in §13).

    **It never draws coordinates itself, deliberately.** Under `"coords"` it RAISES when
    `coords_by_draw` is missing, naming `coords_for_draws`. The draw has to happen once,
    in the caller, so the same array feeds both the clouds and the winner's
    `describe_cells(..., win_coords)` — otherwise the tree that gets shown sits at
    coordinates its own cloud never saw, and a second draw is a second RNG consumption
    nobody asked for.

    Under `"cells"`, `coords_by_draw` is **ignored** and the result is bit-identical to
    not passing it: two notebook generators already supply it for winner decoration."""
    check_cloud_source(cloud_source)
    if draws is None:
        draws = model.sample_batch(xf, nx, n_samples)
    K = len(draws)
    if cloud_source == "coords":
        if coords_by_draw is None:
            raise ValueError(
                "posterior_distances(cloud_source='coords') needs `coords_by_draw` and "
                "will not draw it: call `coords_for_draws(model, xf, nx, draws)` once in "
                "the caller and pass the SAME array here and to describe_cells, so the "
                "reported tree sits at the coordinates its cloud was built from."
            )
        if len(coords_by_draw) != K:
            raise ValueError(
                f"coords_by_draw has {len(coords_by_draw)} entries for {K} draws; "
                f"`coords_for_draws` is unfiltered and index-aligned by contract, and an "
                f"empty draw keeps its slot as a (0, 4) table."
            )
        sources = coords_by_draw
    else:
        sources = draws
    clouds_S = [
        lund_cloud(s, geom, lnkt_cut=lnkt_cut, weight=weight, coords=coords) for s in sources
    ]
    if n_candidates and 0 < n_candidates < K:
        cand_idx = list(range(n_candidates))
    else:
        cand_idx = list(range(K))
    if not cand_idx:
        return draws, clouds_S, cand_idx, np.zeros((0, 0), dtype=float)
    clouds_C = [clouds_S[i] for i in cand_idx]
    D = lund_emd_matrix(clouds_C, clouds_S, R=R, beta=beta, norm=norm,
                        periodic_phi=periodic_phi, phi_col=phi_col, backend=backend, geom=geom)
    return draws, clouds_S, cand_idx, D


def _draw_coords_once(model, xf, nx, draws, coords_by_draw, cloud_source):
    """The ONE place the coordinate table for a jet is drawn, shared by the three
    estimators. Returns `coords_by_draw` unchanged under `"cells"`, or when the caller
    already supplied one — so a double draw is structurally impossible rather than merely
    avoided by discipline (docs/PLAN_z_aware.md §4/WP-3)."""
    check_cloud_source(cloud_source)
    if cloud_source != "coords" or coords_by_draw is not None:
        return coords_by_draw
    return coords_for_draws(model, xf, nx, draws)


def mbr_select(model, xf, nx, *, draws=None, geom, n_samples=200, n_candidates=0,
               lnkt_cut=None, weight="kt", coords="lnDR_lnkt", R=8.485, beta=1.0,
               norm=False, periodic_phi=False, phi_col=-1, backend="pot",
               resample_to_qn=False, coords_by_draw=None, cloud_source="cells",
               diagnostic_losses=(), loss_quantile=0.10):
    """Sampling-based MBR (Eikema & Aziz, EMNLP 2022): pick the drawn tree of least
    mean perturbative-Lund EMD to the ``K`` draws.

    Returns a ``LundPointEstimate`` (the same type as ``map_estimate`` — a drop-in
    for every consumer) with ``.risk`` = the achieved mean distance and ``.logprob``
    = the model's joint log-density of the selected tree. ``draws`` are reused when
    given (no resample — same pattern as ``learned_min_emissions(..., mults=)``);
    ``n_candidates>0`` shrinks the candidate set ``C`` (asymmetric MBR) while the
    expectation still runs over all ``K`` draws. ``resample_to_qn=True`` reweights the
    support to the calibrated ``q(N|x)`` marginal (``_qn_importance_weights``), an
    opt-in decode-layer exposure-bias correction; off keeps the plain mean risk.

    ``cloud_source="coords"`` (``decode.mbr_cloud_source``) builds every cloud from the
    jet's continuous coordinate table instead of from cell centres, which is what makes
    ``coords="+lnz"`` and ``weight="z"`` functional rather than fatal. The table is drawn
    **once**, here, and the same array feeds both the clouds and the winner's
    ``describe_cells`` — so the tree that comes back sits at exactly the coordinates its
    cloud was built from. Off by default and bit-identical off (docs/PLAN_z_aware.md
    §4/WP-3; measured in §13, where it recovers 47-70% of the ``|Δ ln z|`` ceiling).

    The winner keeps **its own** continuous coordinates (docs/PLAN_prod_test_v1.md
    WP-C.1): ``describe_cells`` draws them, or ``coords_by_draw`` supplies the ones the
    caller already drew alongside the cells. It used to re-attach the head modes, which
    threw away the one property that makes a medoid worth reporting — that it is a
    genuine posterior sample. v0 measured the price: a psi resultant ``|R| = 0.69``
    against a truth of 0.045, out of a head whose median ``kappa`` is 0.022.

    ``diagnostic_losses`` (WP4a of docs/PLAN_PosteriorClusters.md) is an **eval-only side
    channel**: pass e.g. ``("bounded", "kernel")`` and the call returns
    ``(point_estimate, {"linear": win_idx, "bounded": win_idx, ..., "eps": eps})``
    instead of the bare estimate. The returned ``LundPointEstimate`` is untouched — same
    tree, same ``.risk``, still the linear medoid — so ``.risk`` keeps meaning "the
    achieved mean distance" for all fourteen of its consumers and no config field, serving
    surface or config-hash churn is involved. It costs zero additional EMD calls: every
    loss is another reduction over the `D` already built."""
    if check_cloud_source(cloud_source) == "coords" and draws is None:
        # Hoisted out of `posterior_distances` only on this branch, because the
        # coordinates have to be drawn from the same `draws` the clouds are built from.
        # The default path still samples inside `posterior_distances`, at the same point
        # in the RNG stream it always did.
        draws = model.sample_batch(xf, nx, n_samples)
    coords_by_draw = _draw_coords_once(model, xf, nx, draws, coords_by_draw, cloud_source)
    draws, clouds_S, cand_idx, D = posterior_distances(
        model, xf, nx, draws=draws, geom=geom, n_samples=n_samples, n_candidates=n_candidates,
        lnkt_cut=lnkt_cut, weight=weight, coords=coords, R=R, beta=beta, norm=norm,
        periodic_phi=periodic_phi, phi_col=phi_col, backend=backend,
        coords_by_draw=coords_by_draw, cloud_source=cloud_source,
    )
    if not cand_idx:  # no draws at all -> honest empty tree (reflects the posterior)
        pe = model.describe_cells(xf, nx, [])
        pe.risk = 0.0
        pe.estimator = "mbr"
        return (pe, {}) if diagnostic_losses else pe
    # match the support's multiplicity marginal to calibrated q(N|x); None == uniform, and
    # the uniform branch of `_reduce_risk` is literally `D.mean(axis=1)` (gate G1).
    w = _qn_importance_weights(model, xf, nx, draws) if resample_to_qn else None
    risk = _reduce_risk(D, w, loss="linear")
    best = int(np.argmin(risk))
    win_idx = cand_idx[best]
    winner = draws[win_idx]
    win_coords = None
    if coords_by_draw is not None and win_idx < len(coords_by_draw):
        win_coords = coords_by_draw[win_idx]
    # genuine drawn tree -> LundPointEstimate, carrying its own sampled coordinates
    pe = model.describe_cells(xf, nx, winner, win_coords)
    pe.risk = float(risk[best])
    pe.estimator = "mbr"
    if not diagnostic_losses:
        return pe
    eps = bandwidth_quantile(D, loss_quantile)
    side = {"linear": win_idx, "eps": eps, "loss_quantile": float(loss_quantile)}
    for loss in diagnostic_losses:
        if loss == "linear":
            continue
        r = _reduce_risk(D, w, loss=loss, eps=eps) if eps > 0 else np.full(len(cand_idx), np.nan)
        side[loss] = cand_idx[int(np.argmin(r))] if eps > 0 else win_idx
    return pe, side


# ---------------------------------------------------------------------------
# N-first (stratified) MBR — docs/PLAN_StratifiedMBR.md WP1
# ---------------------------------------------------------------------------
def mbr_select_stratified(model, xf, nx, *, draws=None, geom, n_samples=200, n_candidates=0,
                          lnkt_cut=None, weight="kt", coords="lnDR_lnkt", R=8.485, beta=1.0,
                          norm=False, periodic_phi=False, phi_col=-1, backend="pot",
                          resample_to_qn=False, coords_by_draw=None, cloud_source="cells",
                          n_quantile=0.5, D=None):
    """Two-stage point estimate: decide **N** from the calibrated marginal, then the
    **conditional medoid** within that stratum (`decode.point_estimator="mbr_n"`).

        n_hat = Q_0.5(q(N|x))        # the Bayes estimator under L(n,m) = |n - m|
        y_hat = argmin_{h : |h| = n_hat}  mean_{k : |y_k| = n_hat} d(h, y_k)

    A **sibling** of `mbr_select`, not a mode of it: `mbr_select` keeps its own contract
    (`.risk` = the achieved mean distance over all K draws) and its G1 bit-identity
    untouched, the same separation `mbr_cluster_set` has.

    **Why.** `mbr_select` minimises a mean over every stratum at once, and the EMD's
    mass-imbalance term charges `~R|W_a - W_b|` across strata — so the medoid is pulled
    toward whatever multiplicity is most populous and can land between strata,
    representing none. On the 600-jet K=200 arm that leaves it 2.349 from truth against a
    1.476 oracle over cluster exemplars, with **83% of the resolvable ambiguity between
    N strata** and `q(N|x)` itself calibrated (G4 ratio 0.977; SBC-on-N at the 88th
    percentile of its own null). Deciding N by the calibrated marginal and the shape by
    within-stratum centrality uses each channel where it is trustworthy.

    This is also `docs/PLAN_empty_parton_tree.md`'s deferred "general argmin over an
    explicit loss on n", concretely: `L = |n - m|` gives the median, and the empty gate is
    its `n = 0` special case.

    **Composition with the empty gate.** Stage 0 is `models.base.map_or_mbr`'s
    `decode.empty_threshold`, which runs *before* dispatch — it is deliberately not
    duplicated here, or the config path would gate twice. The interaction is benign: any
    sensible tau is below 0.5, so "the gate did not fire" implies `q(0|x) < 0.5` and the
    median cannot be 0 on the gated path. With the gate off a median of 0 honestly returns
    the empty medoid, at risk exactly 0.0 (the empty clique has zero diameter).

    `.risk` is the **within-stratum** mean — the achieved risk of the decision that
    produced this tree, which is the only meaning `.risk` has. It is a different number
    from `mbr_select`'s global mean, and `estimator="mbr_n"` is the provenance that keeps
    the two from being averaged together.

    Pass `D=` (with the `draws` that produced it) when the caller already built the matrix
    and no EMD is solved at all."""
    from .length import quantile_floor

    if D is None:
        if check_cloud_source(cloud_source) == "coords" and draws is None:
            draws = model.sample_batch(xf, nx, n_samples)
        coords_by_draw = _draw_coords_once(model, xf, nx, draws, coords_by_draw, cloud_source)
        draws, _clouds, cand_idx, D = posterior_distances(
            model, xf, nx, draws=draws, geom=geom, n_samples=n_samples,
            n_candidates=n_candidates, lnkt_cut=lnkt_cut, weight=weight, coords=coords,
            R=R, beta=beta, norm=norm, periodic_phi=periodic_phi, phi_col=phi_col,
            backend=backend, coords_by_draw=coords_by_draw, cloud_source=cloud_source,
        )
        if not cand_idx:  # no draws at all -> honest empty tree (reflects the posterior)
            pe = model.describe_cells(xf, nx, [])
            pe.risk, pe.estimator = 0.0, "mbr_n"
            return pe
    elif draws is None:
        raise ValueError("mbr_select_stratified(D=...) also needs the `draws` that produced it")
    if n_candidates:
        raise ValueError(
            f"mbr_select_stratified requires mbr_n_candidates == 0 (a square K x K "
            f"matrix), got {n_candidates}: restricting both the candidates and the "
            f"expectation to one stratum needs the row and column indices to agree."
        )

    mults = np.array([len(d) for d in draws], dtype=int)
    # Reuses the draws rather than sampling again — the `learned_min_emissions(mults=)`
    # pattern. For a continue/stop family this pmf IS the histogram of these draws, so the
    # median is realised by construction and `_nearest_populated` never fires.
    pmf = model.length_pmf(xf, nx, mults=mults.tolist())
    n_hat = int(quantile_floor(pmf, float(n_quantile)))
    w = _qn_importance_weights(model, xf, nx, draws) if resample_to_qn else None
    win_idx, risk, n_used = stratified_medoid(D, mults, n_hat, w=w)

    win_coords = None
    if coords_by_draw is not None and win_idx < len(coords_by_draw):
        win_coords = coords_by_draw[win_idx]
    pe = model.describe_cells(xf, nx, draws[win_idx], win_coords)
    pe.risk = float(risk)
    pe.estimator = "mbr_n"
    return pe


# ---------------------------------------------------------------------------
# WP2 — the set-valued prediction
# ---------------------------------------------------------------------------
def mbr_cluster_set(model, xf, nx, *, draws=None, geom, n_samples=200, n_candidates=0,
                    lnkt_cut=None, weight="kt", coords="lnDR_lnkt", R=8.485, beta=1.0,
                    norm=False, periodic_phi=False, phi_col=-1, backend="pot",
                    resample_to_qn=False, coords_by_draw=None, cloud_source="cells",
                    method="hdbscan", min_cluster_size=0, min_mass=0.05,
                    eps_quantile=0.10, split=False, screening_only=False,
                    set_threshold=None, fitted_under=None, D=None,
                    empty_threshold=0.0):
    """One `LundPointEstimate` per posterior cluster, each a genuine draw, with the
    cluster's posterior mass and radius (docs/PLAN_PosteriorClusters.md WP2).

    Runs at **stock MBR settings**: `mbr_select`'s point estimate is bit-identical whether
    or not this is called, because nothing here touches `risk = D.mean(axis=1)`. `D` is the
    same matrix — pass it in (`D=`) when the caller already built it, and no EMD is solved
    at all.

    Each member goes through `describe_cells(xf, nx, winner, win_coords)`, so every
    exemplar carries its own sampled coordinates and `coords_source="sample"` exactly as
    the WP-C.1 medoid does. The hypothesis space stays `H = {pool}`: nothing here
    constructs a tree the model did not generate.

    The two per-jet scalars (WP3) ride along on every member as `cluster_mass` /
    `cluster_entropy`, so existing single-estimate consumers carry them without a
    signature change. They are deliberately not folded into one +/-: `top_mass` is a
    probability, `entropy` is an ambiguity over discrete alternatives, and only `radii[0]`
    is a width.

    ``empty_threshold`` (default ``0.0`` == off, bit-identical) makes the **emptiness**
    decision the same one `map_or_mbr` already takes, instead of leaving it to the mass
    argmax. It reuses `decode.empty_threshold` and adds no config field, because it is the
    same knob meaning the same thing at the same stage.

    **Why the mass argmax is the wrong rule for the N = 0 stratum.** `_empty_value` returns
    exactly `0` for two empty clouds, so every empty draw collapses into ONE zero-radius
    cluster carrying the whole of `q(0|x)` — while the non-empty draws live on a continuum
    and get *fragmented* into several clusters by the density method. The argmax therefore
    compares one atomic lump against the largest of a fragmented competitor set, and the
    empty stratum wins on far more jets than its own mass warrants: measured 29.8% against a
    true rate of 16.7% on 600 held-out jets at K = 200 (~9 sigma). That is a partition-
    granularity artifact, not physics.

    Gate G3 says the empty cluster's mass and `length_pmf`'s `q(0|x)` are the SAME NUMBER
    (`|difference| ~ 0`). So the two rules differ only in what that number is compared
    against — a fragmented competitor set, or a threshold fitted by rate-matching and
    frozen (`inference.length.empty_threshold_for_rate`, docs/PLAN_empty_parton_tree.md).
    Same information, calibrated decision rule.

    What moves and what does not: `members`, `masses`, `radii` and the conformal prefix are
    **untouched** and stay mass-descending, so every existing consumer is unaffected. Only
    `.point` moves, via `point_index`. `members[0]` keeps meaning "the top-mass exemplar" so
    the two rules can be compared on the same object.

    Note the gate is rate-matched, not per-jet accurate — AUC ~0.76-0.82, recall ~0.36 on
    the measured arm. It fixes the empty RATE by construction; whether it fixes the right
    JETS is a separate measurement."""
    from .clusters import (
        PosteriorSetEstimate,
        assert_cluster_metric_ok,
        cluster_posterior,
        set_size_for,
    )

    assert_cluster_metric_ok(
        {"mbr_beta": beta, "mbr_R": R, "mbr_coords": coords, "mbr_n_candidates": n_candidates},
        geom,
    )
    if D is None:
        if check_cloud_source(cloud_source) == "coords" and draws is None:
            draws = model.sample_batch(xf, nx, n_samples)
        coords_by_draw = _draw_coords_once(model, xf, nx, draws, coords_by_draw, cloud_source)
        draws, _clouds, cand_idx, D = posterior_distances(
            model, xf, nx, draws=draws, geom=geom, n_samples=n_samples, n_candidates=0,
            lnkt_cut=lnkt_cut, weight=weight, coords=coords, R=R, beta=beta, norm=norm,
            periodic_phi=periodic_phi, phi_col=phi_col, backend=backend,
            coords_by_draw=coords_by_draw, cloud_source=cloud_source,
        )
        if not cand_idx:  # no draws at all -> an honestly empty set, not a fabricated one
            return PosteriorSetEstimate(
                members=[], masses=np.zeros(0), radii=np.zeros(0),
                top_mass=float("nan"), entropy=float("nan"),
                clusters=cluster_posterior(np.zeros((1, 1)), method="pam", backend=backend),
            )
    elif draws is None:
        raise ValueError("mbr_cluster_set(D=...) also needs the `draws` that produced it")
    w = _qn_importance_weights(model, xf, nx, draws) if resample_to_qn else None
    # A deterministic exchangeable split: the draws are i.i.d. from q(y|x), so even/odd is
    # as valid a split as any RNG draw and it is reproducible without carrying a seed.
    split_index = None
    if split:
        split_index = np.zeros(len(draws), dtype=bool)
        split_index[::2] = True
    cs = cluster_posterior(
        D, method=method, min_mass=min_mass, min_cluster_size=min_cluster_size,
        eps_quantile=eps_quantile, weights=w, backend=backend,
        screening_only=screening_only, split_index=split_index,
    )
    members = []
    for j, e in enumerate(cs.exemplars):
        ec = coords_by_draw[e] if (coords_by_draw is not None and e < len(coords_by_draw)) else None
        pe = model.describe_cells(xf, nx, draws[e], ec)
        pe.cluster_mass = float(cs.masses[j])
        pe.cluster_entropy = float(cs.entropy)
        pe.estimator = "cluster"
        members.append(pe)

    # --- which member is the RECOMMENDED tree: the emptiness decision -------------
    # The N = 0 stratum's cluster is identified by its EXEMPLAR being empty. That is exact
    # rather than heuristic: empty draws sit at mutual distance 0 and at a large constant
    # distance from every non-empty draw, so a cluster holding any of them holds only them
    # and its medoid is one of them.
    empty_cluster = next((j for j, m in enumerate(members) if m.multiplicity == 0), None)
    point_index, gate_fired, policy = 0, None, "include"
    tau = float(empty_threshold or 0.0)
    if tau > 0.0 and members:
        from .length import empty_gate

        policy = "gate"
        pmf = model.length_pmf(xf, nx, mults=[len(d) for d in draws])
        gate_fired = bool(empty_gate(pmf, tau))
        if gate_fired:
            # The gate says empty. Recommend the empty explanation IF the posterior has
            # one — never fabricate it: H = {pool}, and a q(0|x) above tau with no empty
            # draw is a disagreement between the length head and the sampler, not a tree.
            point_index = empty_cluster if empty_cluster is not None else 0
        elif empty_cluster == 0 and len(members) > 1:
            # The gate says NOT empty but the mass argmax landed on the empty stratum —
            # the granularity artifact. Recommend the top-mass NON-empty explanation.
            point_index = 1
        elif empty_cluster == 0:
            point_index = 0   # the empty cluster is the only one: nothing else to offer

    return PosteriorSetEstimate(
        members=members,
        masses=cs.masses,
        radii=cs.radii,
        top_mass=cs.top_mass,
        entropy=cs.entropy,
        clusters=cs,
        set_size=(set_size_for(cs.masses, set_threshold) if set_threshold is not None else None),
        set_threshold=(float(set_threshold) if set_threshold is not None else None),
        fitted_under=fitted_under,
        point_index=int(point_index),
        empty_policy=policy,
        empty_cluster=empty_cluster,
        empty_gate_fired=gate_fired,
    )
