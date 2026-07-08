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

import math
import os

import numpy as np

# Which coordinate columns enter the ground metric -> ground dimension gdim.
_COORD_GDIM = {"lnDR_lnkt": 2, "+lnz": 3, "+psi": 4}


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
        R=float(decode.get("mbr_R", 8.485)),
        beta=float(decode.get("mbr_beta", 1.0)),
        norm=bool(decode.get("mbr_norm", False)),
        periodic_phi=bool(decode.get("mbr_periodic_phi", False)),
        phi_col=int(decode.get("mbr_phi_col", -1)),
        backend=str(decode.get("mbr_backend", "pot")),
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
    ``lnkt_cut=None`` inherits the geometry's ``ln_kt`` floor (the region cut)."""
    g = _COORD_GDIM[coords]
    if lnkt_cut is None:
        lnkt_cut = float(geom.ln_kt_range[0])
    pts, ws = [], []
    for c in draw:
        if isinstance(c, (int, np.integer)):
            u, v = geom.cell_center(int(c))
            lz = ps = 0.0
        else:
            arr = np.asarray(c, dtype=float).ravel()
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


def _import_ef():
    # EnergyFlow's `wasserstein` OpenMP extension and PyTorch both link an OpenMP
    # runtime; loading both in one process aborts with "OMP: Error #15 ... libomp
    # already initialized" on macOS. Allow them to coexist (set before the first
    # wasserstein call, which is where its runtime initialises). The MBR solves are
    # independent LPs, so the duplicate-runtime caveat does not affect correctness.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import energyflow as ef  # lazy, per-backend
        return ef
    except ImportError as e:  # pragma: no cover - exercised only without energyflow
        raise ImportError(
            "mbr_backend='energyflow' requires the optional [energyflow] extra: "
            "pip install 'energyflow>=1.3' (and a working `wasserstein` build)."
        ) from e


def _emd_ef(pa, wa, pb, wb, *, R, beta, norm, periodic_phi) -> float:
    ef = _import_ef()
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return _empty_value(wa, wb, R, norm, 1.0)  # EnergyFlow's 1/R ground scale
    g = pa.shape[1]
    return float(
        ef.emd.emd(cloud_to_event(pa, wa), cloud_to_event(pb, wb),
                   R=R, beta=beta, norm=norm, gdim=g, periodic_phi=periodic_phi)
    )


def _lund_image(pts, w, geom, nb=10) -> np.ndarray:
    """Normalised binned Lund image over ``(ln 1/DeltaR, ln kt)`` — the surrogate."""
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
        try:  # one batched, multiprocessed call for the whole non-empty block
            sub = np.asarray(
                ef.emd.emds(eventsC, eventsS, R=R, beta=beta, norm=norm,
                            gdim=g, periodic_phi=periodic_phi),
                dtype=float,
            ).reshape(len(nzC), len(nzS))
        except (ValueError, TypeError):
            # wasserstein's batched `_store_events` uses `np.array(..., copy=False)`,
            # which raises under numpy>=2 ("Unable to avoid copy"). Fall back to the
            # per-pair `emd` (unaffected, identical result), so the backend still works.
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
# The estimator
# ---------------------------------------------------------------------------
def mbr_select(model, xf, nx, *, draws=None, geom, n_samples=200, n_candidates=0,
               lnkt_cut=None, weight="kt", coords="lnDR_lnkt", R=8.485, beta=1.0,
               norm=False, periodic_phi=False, phi_col=-1, backend="pot"):
    """Sampling-based MBR (Eikema & Aziz, EMNLP 2022): pick the drawn tree of least
    mean perturbative-Lund EMD to the ``K`` draws.

    Returns a ``LundPointEstimate`` (the same type as ``map_estimate`` — a drop-in
    for every consumer) with ``.risk`` = the achieved mean distance and ``.logprob``
    = the model's joint log-density of the selected tree. ``draws`` are reused when
    given (no resample — same pattern as ``learned_min_emissions(..., mults=)``);
    ``n_candidates>0`` shrinks the candidate set ``C`` (asymmetric MBR) while the
    expectation still runs over all ``K`` draws."""
    if draws is None:
        draws = model.sample_batch(xf, nx, n_samples)
    K = len(draws)
    clouds_S = [
        lund_cloud(d, geom, lnkt_cut=lnkt_cut, weight=weight, coords=coords) for d in draws
    ]
    if n_candidates and 0 < n_candidates < K:
        cand_idx = list(range(n_candidates))
    else:
        cand_idx = list(range(K))
    if not cand_idx:  # no draws at all -> honest empty tree (reflects the posterior)
        pe = model.describe_cells(xf, nx, [])
        pe.risk = 0.0
        return pe
    clouds_C = [clouds_S[i] for i in cand_idx]
    D = lund_emd_matrix(clouds_C, clouds_S, R=R, beta=beta, norm=norm,
                        periodic_phi=periodic_phi, phi_col=phi_col, backend=backend, geom=geom)
    risk = D.mean(axis=1)
    best = int(np.argmin(risk))
    winner = draws[cand_idx[best]]
    pe = model.describe_cells(xf, nx, winner)  # genuine drawn tree -> LundPointEstimate
    pe.risk = float(risk[best])
    return pe
