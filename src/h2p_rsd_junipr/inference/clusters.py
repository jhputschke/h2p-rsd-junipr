"""Posterior clusters over the MBR distance matrix (docs/PLAN_PosteriorClusters.md WP1).

`mbr_select` returns the Fréchet median restricted to the sample — the draw of least
*mean* perturbative-Lund EMD to the posterior. That is a **centrality** criterion, and
it is the right default; it is the wrong criterion when the posterior is **multimodal**,
because the medoid of a two-lobed posterior can land in the sparse valley between the
lobes, minimising mean distance while representing neither explanation.

The sample space here is transdimensional (`Y = disjoint union over N of C^N`) and the
strata are metrically separated by the EMD's imbalance term, so "one hard emission" and
"two softer emissions consistent with the same observed x" are *discrete alternative
explanations* rather than two ends of one continuum. This module reads those alternatives
off the distance matrix `mbr_select` already builds.

**The structural fact the whole module rests on:** the cluster layer consumes only `D`,
and `D` is identical under every loss (`cluster_posterior` and `inference.mbr._reduce_risk`
both consume `D` and never see each other's output). So the partition, masses, radii and
the two per-jet scalars are *bit-identical* across `mbr_loss in {linear, bounded, kernel}`
— changing the reduction over `D` and reading more off `D` are orthogonal
(`tests/test_clusters.py::test_losses_do_not_move_clusters` asserts it rather than
trusting this paragraph).

Distance-matrix methods only: nothing here embeds `Y` in a vector space, because trees
have no mean. Three methods, all `metric="precomputed"`:

  - ``hdbscan`` (default) — density-based, no fixed k, native noise label (Campello,
    Moulavi & Sander, PAKDD 2013). Takes ``min_cluster_size``, not a bandwidth, so the
    default path pre-registers no epsilon.
  - ``dbscan``  — the epsilon-explicit fallback (Ester, Kriegel, Sander & Xu, KDD 1996),
    sharing the bandwidth rule with `inference.mbr.bandwidth_quantile`.
  - ``pam``     — k-medoids with k by silhouette (Kaufman & Rousseeuw, *Finding Groups in
    Data*, Wiley 1990), pure NumPy and deterministic: the control arm that says whether
    gate G2's verdict is method-dependent.

`scikit-learn` is needed for the first two only, and is declared under the `[mbr]` extra
rather than as a core dependency — the `point_estimator="map"` path must import nothing
new. `pam` runs on NumPy alone, which is also why the CI fast tier can exercise the layer
on a host with no sklearn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # no runtime import: this module stays numpy-only and model-free
    from .point_estimate import LundPointEstimate

# Ground-metric diameter per `mbr_coords` mode, as a multiplier on the Lund-plane
# diagonal. `+lnz` and `+psi` add coordinate columns and so lengthen the diagonal; the
# admissibility check below needs the *active* diameter, not the 2-D one.
_COORD_COLS = {"lnDR_lnkt": ("u", "v"), "+lnz": ("u", "v", "lnz"), "+psi": ("u", "v", "lnz", "psi")}
# Spans of the two coordinates the geometry does not carry. ln z runs over the soft-drop
# interval and psi over the circle; both are model-level supports, not geometry fields,
# so they are stated here and used only to compute a ground DIAMETER (a bound, not a value).
_LNZ_SPAN = 5.0      # ln z in ~[-5, 0] on the physical support
_PSI_SPAN = 2.0 * math.pi


# ---------------------------------------------------------------------------
# §4 metric audit — the guards, as raises rather than doc notes
# ---------------------------------------------------------------------------
def ground_diameter(geom, coords: str = "lnDR_lnkt") -> float:
    """Maximum ground distance between two points of the Lund cloud, for `mbr_coords`.

    The EMD is a metric only when the imbalance radius `R` is at least half the maximum
    ground distance (Komiske, Metodiev & Thaler, *Phys. Rev. Lett.* **123** (2019) 041801,
    arXiv:1902.02346), and HDBSCAN's mutual-reachability construction assumes a metric —
    so this is what `assert_cluster_metric_ok` compares `mbr_R` against. Computed from the
    geometry's own ranges rather than hard-coded at 8.485, so a non-default `geometry`
    block cannot silently break the inequality."""
    if coords not in _COORD_COLS:
        raise ValueError(f"unknown mbr_coords={coords!r}; expected one of {sorted(_COORD_COLS)}")
    du = float(geom.ln_invdelta_range[1]) - float(geom.ln_invdelta_range[0])
    dv = float(geom.ln_kt_range[1]) - float(geom.ln_kt_range[0])
    spans = [du, dv]
    if coords in ("+lnz", "+psi"):
        spans.append(_LNZ_SPAN)
    if coords == "+psi":
        spans.append(_PSI_SPAN)
    return float(math.sqrt(sum(s * s for s in spans)))


def assert_cluster_metric_ok(decode: dict, geom) -> None:
    """Raise unless the decode's metric settings make `D` a metric (gate G4).

    Three conditions, all measured in docs/PLAN_PosteriorClusters.md §4 rather than
    assumed, and all raising rather than warning — a warning on a mass vector nobody can
    see is a number that gets quoted anyway:

    1. **`mbr_beta == 1.0`.** At beta = 2 the triangle inequality fails on 300 of 64 000
       measured triples; the clustering's mutual-reachability distance is then not a
       distance and its output is not a partition of anything.
    2. **`mbr_R >= ground_diameter/2`.** KMT's condition for the EMD to be a metric.
    3. **`mbr_n_candidates == 0`.** Otherwise `D` is `|C| x K` (asymmetric MBR) and there
       is no `K x K` matrix to cluster. Raised rather than silently overridden: overriding
       would change the point estimate the caller asked for.

    `mbr_backend` is checked in `cluster_posterior` instead, where the screening escape
    hatch lives."""
    beta = float(decode.get("mbr_beta", 1.0))
    if beta != 1.0:
        raise ValueError(
            f"cluster_posterior requires mbr_beta == 1.0, got {beta:g}. At beta != 1 the "
            f"perturbative-Lund EMD violates the triangle inequality (measured: 300 "
            f"violations / 64 000 triples at beta = 2), so the mutual-reachability "
            f"distance HDBSCAN builds is not a metric — see docs/PLAN_PosteriorClusters.md §4."
        )
    n_cand = int(decode.get("mbr_n_candidates", 0))
    if n_cand:
        raise ValueError(
            f"cluster_posterior requires mbr_n_candidates == 0 (a square K x K distance "
            f"matrix), got {n_cand}. With a candidate cap D is |C| x K and there is no "
            f"pairwise matrix over the posterior to cluster. Reset the knob rather than "
            f"letting this be overridden — the cap changes which point estimate you get."
        )
    R = float(decode.get("mbr_R", 8.485))
    coords = str(decode.get("mbr_coords", "lnDR_lnkt"))
    diam = ground_diameter(geom, coords)
    if R < 0.5 * diam - 1e-9:
        raise ValueError(
            f"cluster_posterior requires mbr_R >= half the maximum ground distance "
            f"(KMT's metric condition): mbr_R = {R:g} but the ground diameter for "
            f"mbr_coords={coords!r} at this geometry is {diam:.3f}, so R must be at least "
            f"{0.5 * diam:.3f}."
        )


def symmetrize(D) -> np.ndarray:
    """`0.5 * (D + D.T)` with an exactly zero diagonal — always, before any clustering.

    The measured asymmetry of the `pot` backend is at solver round-off (max |D - D.T| =
    1.4e-14 over 64 000 triples), so this changes no number that matters. It is here
    because `sklearn` raises on its own exact-symmetry check, and because the `energyflow`
    batched path (`inference.mbr._matrix_ef`) has not been audited to the same standard —
    a defensive symmetrisation is cheaper than finding out in a run."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"expected a square distance matrix, got shape {D.shape}")
    out = 0.5 * (D + D.T)
    np.fill_diagonal(out, 0.0)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# The product type
# ---------------------------------------------------------------------------
@dataclass
class PosteriorClusterSet:
    """A labelling of the K posterior draws, with the per-cluster posterior mass.

    `masses` and `radii` are mass-descending and aligned with `exemplars`; every reported
    cluster carries at least `min_mass`. Everything below that threshold, plus whatever
    the density method labelled noise, is pooled into `residual_mass` so the mass vector
    stays short and interpretable — `masses.sum() + residual_mass == 1` up to float error.

    The three numbers a consumer should take away are documented at their fields and are
    deliberately **not** foldable into one +/-: `top_mass` is a probability, `entropy` is
    an ambiguity over *discrete* alternatives, and only `radii[0]` is a width."""

    labels: np.ndarray                 # (K,) int; -1 = noise or sub-threshold residual
    exemplars: list[int]               # draw index per reported cluster (within-cluster medoid)
    masses: np.ndarray                 # (n_clusters,) posterior mass, descending, sums <= 1
    radii: np.ndarray                  # (n_clusters,) mean within-cluster EMD to the exemplar
    top_mass: float                    # masses[0] — the posterior mass of the selected explanation
    entropy: float                     # -sum m log m over [*masses, residual]; natural log
    method: str
    eps: float | None                  # bandwidth, when the method takes one (dbscan)
    min_cluster_size: int              # hdbscan's control instead
    min_mass: float
    n_draws: int
    backend: str                       # provenance: mass vectors from "surrogate" are invalid
    noise_mass: float = 0.0            # the density method's OWN noise label, a subset of...
    residual_mass: float = 0.0         # ...everything not in a reported cluster
    silhouette: float = float("nan")   # separation-over-width; the G2' precondition
    separation: float = float("nan")   # smallest exemplar-to-exemplar EMD; NaN below 2 clusters
    screening_only: bool = False       # True => this labelling may not be quoted (§4.3)
    weighted: bool = False             # were masses computed under q(N|x) importance weights
    split: bool = False                # were masses estimated on a held-out half (WP5.1)
    notes: list = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return int(len(self.exemplars))

    def summary(self) -> str:
        head = (f"{self.n_clusters} cluster(s) over {self.n_draws} draws "
                f"[{self.method}, {self.backend}]  top_mass={self.top_mass:.3f}  "
                f"H={self.entropy:.3f} nats")
        rows = [f"    #{j}  mass={m:.3f}  radius={r:.3f}  exemplar=draw[{e}]"
                for j, (m, r, e) in enumerate(zip(self.masses, self.radii, self.exemplars))]
        if self.residual_mass > 0:
            rows.append(f"    residual (noise + clusters below min_mass={self.min_mass:g}): "
                        f"{self.residual_mass:.3f}")
        if self.screening_only:
            rows.append("    SCREENING ONLY — the surrogate is mass-blind; do not quote "
                        "these masses (docs/PLAN_PosteriorClusters.md §4.3)")
        return "\n".join([head, *rows])


@dataclass
class PosteriorSetEstimate:
    """The set-valued prediction: one `LundPointEstimate` per posterior cluster.

    Each member is a **genuine posterior draw** carrying its own sampled coordinates
    (`coords_source="sample"`), exactly as the WP-C.1 medoid does — the hypothesis space
    stays `H = {pool}` and nothing here constructs a tree the model did not generate.

    `members[0]` is the highest-mass exemplar. It is the estimator to compare a bounded
    loss against (docs/PLAN_PosteriorClusters.md §8.1): `members[0]` maximises integrated
    density (mass), a bounded loss maximises peak density, and the two disagree exactly
    when the clusters have unequal `radii`."""

    members: list[LundPointEstimate]   # exemplars, mass-descending
    masses: np.ndarray
    radii: np.ndarray
    top_mass: float
    entropy: float
    clusters: PosteriorClusterSet      # the full labelling, for diagnostics
    set_size: int | None = None        # conformal set size at `set_threshold`, when fitted
    set_threshold: float | None = None
    fitted_under: dict | None = None   # provenance of a frozen threshold (v0 §7 pattern)

    def __len__(self) -> int:
        return len(self.members)

    @property
    def point(self):
        """The mass-maximising member — the set-valued answer's single-tree summary."""
        return self.members[0] if self.members else None

    def conformal_members(self) -> list:
        """The smallest mass-descending prefix whose accumulated mass reaches the frozen
        threshold. Falls back to every member when no threshold has been fitted."""
        if self.set_threshold is None:
            return list(self.members)
        acc, out = 0.0, []
        for m, mem in zip(self.masses, self.members):
            out.append(mem)
            acc += float(m)
            if acc >= self.set_threshold:
                break
        return out


# ---------------------------------------------------------------------------
# Cluster-quality statistics computed straight off D (no truth consulted)
# ---------------------------------------------------------------------------
def _mean_silhouette(D, labels) -> float:
    """Mean silhouette over the assigned draws (Rousseeuw, *J. Comput. Appl. Math.* **20**
    (1987) 53), from a precomputed `D`.

    This is the **separation-over-width precondition** of gate G2': `d(exemplar, truth)`
    carries the within-cluster scatter even in the correct lobe, so a set can only beat a
    point estimate when the inter-cluster distance exceeds the radius. It is computable
    before any truth is consulted, which is what makes it a precondition rather than a
    post-hoc excuse."""
    lab = np.asarray(labels)
    keep = lab >= 0
    uniq = np.unique(lab[keep])
    if uniq.size < 2 or keep.sum() < 2:
        return float("nan")
    idx = np.flatnonzero(keep)
    sub = D[np.ix_(idx, idx)]
    sl = lab[idx]
    sil = np.full(idx.size, np.nan)
    for c in uniq:
        inc = sl == c
        n_in = int(inc.sum())
        if n_in < 2:
            sil[inc] = 0.0  # a singleton has no within-cluster scatter to compare against
            continue
        a = sub[np.ix_(inc, inc)].sum(1) / (n_in - 1)
        b = np.full(n_in, np.inf)
        for other in uniq:
            if other == c:
                continue
            out = sl == other
            if not out.any():
                continue
            b = np.minimum(b, sub[np.ix_(inc, out)].mean(1))
        denom = np.maximum(a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(denom > 0, (b - a) / np.where(denom > 0, denom, 1.0), 0.0)
        sil[inc] = s
    return float(np.nanmean(sil))


def _medoid(D, members, w) -> int:
    """Within-cluster medoid: the member of least weighted mean distance to the cluster."""
    sub = D[np.ix_(members, members)]
    ww = w[members]
    risk = (sub * ww[None, :]).sum(1) / max(float(ww.sum()), 1e-300)
    return int(members[int(np.argmin(risk))])


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------
def _require_sklearn(method: str):
    try:
        import sklearn.cluster as skc
    except ImportError as e:  # pragma: no cover - exercised only without scikit-learn
        raise ImportError(
            f"cluster_method={method!r} needs scikit-learn >= 1.3: "
            f"pip install 'scikit-learn>=1.3' (or `pip install -e \".[mbr]\"`). "
            f"cluster_method='pam' is pure NumPy and needs nothing."
        ) from e
    return skc


def _labels_hdbscan(D, *, min_cluster_size: int, min_samples=None) -> np.ndarray:
    skc = _require_sklearn("hdbscan")
    est = skc.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(max(2, min_cluster_size)),
        min_samples=min_samples,
        # A unimodal posterior must come back as ONE cluster, not as all-noise. sklearn's
        # default refuses a single cluster (it is the root of the condensed tree), which
        # would report `n_clusters=0, top_mass=nan` for exactly the jets the plan calls
        # "effectively unimodal" — the modal outcome under the kill criterion.
        allow_single_cluster=True,
        copy=True,
    )
    return np.asarray(est.fit_predict(D), dtype=int)


def _labels_dbscan(D, *, eps: float, min_samples: int) -> np.ndarray:
    skc = _require_sklearn("dbscan")
    est = skc.DBSCAN(eps=float(eps), min_samples=int(max(2, min_samples)), metric="precomputed")
    return np.asarray(est.fit_predict(D), dtype=int)


def _pam_build(D, k: int) -> list:
    """PAM's deterministic BUILD phase (Kaufman & Rousseeuw 1990, §2.4).

    Greedy: the first medoid minimises total distance; each next one is the point whose
    addition most reduces the total distance to the nearest medoid. No RNG, so `pam` is
    reproducible run to run — which is the entire reason it is the control arm for G2's
    method dependence."""
    first = int(np.argmin(D.sum(1)))
    med = [first]
    nearest = D[first].copy()
    while len(med) < k:
        gain = np.maximum(nearest[None, :] - D, 0.0).sum(1)
        gain[med] = -np.inf
        nxt = int(np.argmax(gain))
        if not np.isfinite(gain[nxt]) or gain[nxt] <= 0:
            break  # no candidate reduces the cost: k is larger than the data supports
        med.append(nxt)
        nearest = np.minimum(nearest, D[nxt])
    return med


def _pam_swap(D, med: list, max_iter: int = 32) -> list:
    """PAM's SWAP phase: exchange a medoid for a non-medoid while the total cost falls."""
    med = list(med)
    for _ in range(max_iter):
        cost = D[med].min(0).sum()
        best, best_cost = None, cost
        for a in range(len(med)):
            trial = list(med)
            others = [m for j, m in enumerate(med) if j != a]
            base = D[others].min(0) if others else np.full(D.shape[0], np.inf)
            cand_cost = np.minimum(base[None, :], D).sum(1)
            cand_cost[med] = np.inf
            h = int(np.argmin(cand_cost))
            if cand_cost[h] < best_cost - 1e-12:
                trial[a] = h
                best, best_cost = trial, float(cand_cost[h])
        if best is None:
            break
        med = best
    return sorted(med)


def _labels_pam(D, *, k_max: int, min_silhouette: float) -> tuple[np.ndarray, int]:
    """k-medoids with k chosen by mean silhouette over k = 2..k_max.

    **k = 1 is the answer when the best silhouette does not clear `min_silhouette`.** The
    default 0.50 is Kaufman & Rousseeuw's own interpretation boundary (*Finding Groups in
    Data*, Wiley 1990, §2.2): above 0.5 "a reasonable structure has been found", 0.25-0.5
    is weak and "could be artificial", below 0.25 there is no substantial structure. A
    threshold of 0 is not a threshold — k-medoids will cheerfully cut an isotropic
    Gaussian blob into three pieces at silhouette 0.32, and reporting that as three
    posterior explanations is the failure this method exists to control for. The
    kill-criterion outcome ("the posterior is effectively unimodal in this metric and at
    this budget") has to be REACHABLE, and this is where PAM can reach it."""
    n = D.shape[0]
    k_max = int(min(max(2, k_max), max(2, n - 1)))
    best_lab, best_sil, best_k = np.zeros(n, dtype=int), -np.inf, 1
    for k in range(2, k_max + 1):
        med = _pam_swap(D, _pam_build(D, k))
        if len(med) < 2:
            continue
        lab = np.asarray(np.argmin(D[med], axis=0), dtype=int)
        sil = _mean_silhouette(D, lab)
        if np.isfinite(sil) and sil > best_sil:
            best_lab, best_sil, best_k = lab, float(sil), k
    if not np.isfinite(best_sil) or best_sil <= float(min_silhouette):
        return np.zeros(n, dtype=int), 1
    return best_lab, best_k


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------
def cluster_posterior(
    D,
    *,
    method: str = "hdbscan",
    min_mass: float = 0.05,
    min_cluster_size: int = 0,
    eps: float | None = None,
    eps_quantile: float = 0.10,
    weights=None,
    backend: str = "pot",
    screening_only: bool = False,
    pam_k_max: int = 8,
    pam_min_silhouette: float = 0.50,
    split_index=None,
) -> PosteriorClusterSet:
    """Cluster the K posterior draws from their pairwise distance matrix alone.

    `D` is the square `K x K` matrix `inference.mbr` already builds; nothing is recomputed
    and no EMD is solved here. `weights` accepts `inference.mbr._qn_importance_weights`, so
    `decode.mbr_resample_to_qn` composes with the masses exactly as it composes with the
    risk. `min_cluster_size = 0` means `max(5, ceil(min_mass * K))` — the HDBSCAN control
    that corresponds to "a cluster worth reporting".

    `split_index` (WP5.1) is a boolean mask over the draws: True entries form pool **A**,
    on which the partition and the exemplars are found, and the masses are then estimated
    from pool **B** by nearest-exemplar assignment. `R_j` is otherwise defined using the
    same draws whose membership is counted, so `top_mass` is biased **high**
    (post-selection inference: Berk, Brown, Buja, Zhang & Zhao, *Ann. Statist.* **41**
    (2013) 802). The split costs `|C| x K`, not `K^2`.

    `backend="surrogate"` raises unless `screening_only=True`: `_lund_image` normalises,
    so the surrogate is *exactly* blind to total kt and multiplicity and collapses the
    N-stratum separation that makes these clusters physical. It is admissible as a G2
    screening pass and never for a quoted mass vector.
    """
    if backend == "surrogate" and not screening_only:
        raise ValueError(
            "mbr_backend='surrogate' cannot produce a quotable mass vector: `_lund_image` "
            "normalises, so the surrogate is exactly blind to total kt and multiplicity "
            "and collapses the N-stratum separation the clusters are made of "
            "(docs/PLAN_PosteriorClusters.md §4.3). Pass screening_only=True to use it as "
            "the gate-G2 first pass, or switch to mbr_backend='pot'."
        )
    D = symmetrize(D)
    K = int(D.shape[0])
    if K == 0:
        raise ValueError("cluster_posterior needs at least one draw; D is 0 x 0")
    w = (np.ones(K, dtype=float) if weights is None
         else np.asarray(weights, dtype=float).reshape(-1).copy())
    if w.size != K:
        raise ValueError(f"weights has {w.size} entries for {K} draws")
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite and sum to a positive number")

    notes: list = []

    # --- pool A: the draws the partition is FOUND on ---------------------------
    if split_index is None:
        idx_A = np.arange(K)
    else:
        idx_A = np.flatnonzero(np.asarray(split_index, dtype=bool))
        if idx_A.size < 2:
            raise ValueError("split_index selects fewer than two draws for pool A")
    D_A = D[np.ix_(idx_A, idx_A)]
    # The auto size is a fraction of the pool ACTUALLY CLUSTERED, not of K. Under a sample
    # split pool A is half the draws, so deriving it from K would silently double the
    # effective threshold and change the partition's granularity — which would make gate
    # G9 (split vs no-split) a measurement of granularity plus selection bias rather than
    # of selection bias alone.
    mcs = int(min_cluster_size) or max(5, int(math.ceil(float(min_mass) * idx_A.size)))

    # --- the partition ---------------------------------------------------------
    off = D_A[~np.eye(D_A.shape[0], dtype=bool)]
    if D_A.shape[0] < 2 or not (off > 0).any():
        # Every draw at mutual distance zero: one cluster of radius zero. This is the
        # all-empty-draw case exactly (`mbr._empty_value` returns 0 for two empty clouds),
        # and it is gate G3 in miniature — no clustering routine is asked, because a
        # zero-diameter set has no density structure to find and sklearn's estimators
        # disagree about what to do with it.
        lab_A = np.zeros(D_A.shape[0], dtype=int)
        eps_used = 0.0 if method == "dbscan" else None
        notes.append("degenerate D (all pairwise distances zero) -> one zero-radius cluster")
    elif method == "hdbscan":
        lab_A, eps_used = _labels_hdbscan(D_A, min_cluster_size=mcs), None
    elif method == "dbscan":
        eps_used = float(eps) if eps is not None else float(
            np.quantile(off[off > 0], float(eps_quantile)))
        lab_A = _labels_dbscan(D_A, eps=eps_used, min_samples=mcs)
    elif method == "pam":
        lab_A, _k = _labels_pam(D_A, k_max=pam_k_max, min_silhouette=pam_min_silhouette)
        eps_used = None
    else:
        raise ValueError(f"unknown cluster_method={method!r}; expected hdbscan | dbscan | pam")

    # --- exemplars on pool A, masses on the assignment pool --------------------
    labels = np.full(K, -1, dtype=int)
    labels[idx_A] = lab_A
    found = [c for c in np.unique(lab_A) if c >= 0]
    exemplars_all = [_medoid(D, list(idx_A[lab_A == c]), w) for c in found]

    if split_index is None:
        assign = labels.copy()
    else:
        # Fresh pool B assigned to the A-exemplars by nearest EMD; pool A keeps its own
        # labels so the exemplars stay members of the clusters they name.
        assign = labels.copy()
        idx_B = np.setdiff1d(np.arange(K), idx_A, assume_unique=False)
        if exemplars_all and idx_B.size:
            near = np.argmin(D[np.ix_(idx_B, exemplars_all)], axis=1)
            assign[idx_B] = np.asarray(found, dtype=int)[near]
        notes.append(f"masses sample-split: |A| = {idx_A.size}, |B| = {int(idx_B.size)} (WP5.1)")

    wsum = float(w.sum())
    raw_mass = np.array([float(w[assign == c].sum()) / wsum for c in found], dtype=float)
    noise_mass = float(w[assign < 0].sum()) / wsum

    # --- the min_mass merge ----------------------------------------------------
    keep = [j for j, m in enumerate(raw_mass) if m >= float(min_mass)]
    order = sorted(keep, key=lambda j: -raw_mass[j])
    residual_mass = float(1.0 - sum(raw_mass[j] for j in order))
    dropped = [found[j] for j in range(len(found)) if j not in set(order)]
    for c in dropped:  # sub-threshold clusters join the residual bucket
        labels[labels == c] = -1
        assign[assign == c] = -1

    exemplars = [exemplars_all[j] for j in order]
    masses = np.array([raw_mass[j] for j in order], dtype=float)
    radii = np.array(
        [float((D[e, assign == found[j]] * w[assign == found[j]]).sum()
               / max(float(w[assign == found[j]].sum()), 1e-300))
         for j, e in zip(order, exemplars)],
        dtype=float,
    )
    # Relabel the survivors 0..n-1 in mass-descending order, so `labels` indexes `masses`.
    # The returned labelling is the ASSIGNMENT pool's: with `split_index` set, pool B's
    # draws carry their nearest-exemplar label, which is the labelling the masses were
    # actually counted on and the only one a consumer can align with `masses`.
    remap = {found[j]: new for new, j in enumerate(order)}
    labels = np.array([remap.get(int(c), -1) for c in assign], dtype=int)

    # Entropy over the reported vector WITH the residual as one entry: a mass vector that
    # does not sum to one has no entropy, and dropping the residual would make a jet whose
    # posterior is mostly unclustered look confident.
    m_full = np.array([*masses, residual_mass], dtype=float) if residual_mass > 1e-12 else masses
    m_full = m_full[m_full > 0]
    # max(0, .) only to turn the -0.0 a single unit mass produces into 0.0; every term
    # -m log m is non-negative for m in (0, 1], so this can never mask a real negative.
    entropy = (max(0.0, float(-(m_full * np.log(m_full)).sum()))
               if m_full.size else float("nan"))

    sep = float("nan")
    if len(exemplars) >= 2:
        sub = D[np.ix_(exemplars, exemplars)]
        sep = float(sub[~np.eye(len(exemplars), dtype=bool)].min())

    return PosteriorClusterSet(
        labels=labels,
        exemplars=exemplars,
        masses=masses,
        radii=radii,
        top_mass=float(masses[0]) if masses.size else float("nan"),
        entropy=entropy,
        method=str(method),
        eps=eps_used,
        min_cluster_size=mcs,
        min_mass=float(min_mass),
        n_draws=K,
        backend=str(backend),
        noise_mass=noise_mass,
        residual_mass=max(residual_mass, 0.0),
        silhouette=_mean_silhouette(D, labels),
        separation=sep,
        screening_only=bool(screening_only and backend == "surrogate"),
        weighted=weights is not None,
        split=split_index is not None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# WP5 — conformal set threshold
# ---------------------------------------------------------------------------
def fit_set_threshold(scores, alpha: float = 0.32) -> dict:
    """Split-conformal threshold on the ACCUMULATED cluster mass (WP5, conformal fallback).

    `scores[i]` is the mass-descending cumulative mass at which jet `i`'s truth was first
    covered — so emitting the smallest mass-descending prefix whose cumulative mass reaches
    the returned threshold gives, under exchangeability, finite-sample marginal coverage
    `>= 1 - alpha` **however wrong q_phi's calibration is** (Vovk, Gammerman & Shafer,
    *Algorithmic Learning in a Random World*, Springer 2005; Angelopoulos & Bates,
    arXiv:2107.07511).

    The `ceil((n+1)(1-alpha))/n` order statistic is the finite-sample correction, not a
    rounding convenience; without it the guarantee is asymptotic only.

    **The guarantee is marginal over jets, not conditional on x** — the same coverage
    notion TARP tests. It must be documented that way wherever it is quoted: a per-jet
    reading of a marginal guarantee is the standard misuse.

    Returns the `fitted_under` record, not a bare float, so a frozen threshold carries its
    own provenance (the `tau.fitted_under` pattern of docs/PLAN_prod_test_v1.md v0 §7)."""
    s = np.asarray([v for v in np.asarray(scores, dtype=float).ravel() if np.isfinite(v)])
    n = int(s.size)
    if n == 0:
        raise ValueError("fit_set_threshold needs at least one finite calibration score")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if rank > n:  # too few calibration jets for this alpha: the honest threshold is "all"
        thr, exact = 1.0, False
    else:
        thr, exact = float(np.sort(s)[rank - 1]), True
    return {
        "value": float(min(max(thr, 0.0), 1.0)),
        "alpha": float(alpha),
        "fitted_under": {
            "n_calibration": n,
            "rank": rank,
            "finite_sample_exact": bool(exact),
            "coverage": "marginal over jets, NOT conditional on x",
            "score": "cumulative cluster mass at which the truth is first covered",
        },
    }


def assert_ancestral_draws(estimates, *, where: str = "an aggregate pushforward") -> None:
    """Raise if any of `estimates` is a SELECTED tree rather than an ancestral draw.

    Pushforward hygiene, asserted rather than commented (docs/PLAN_PosteriorClusters.md
    §10.6). A selected exemplar is systematically closer to the bulk than a typical draw —
    that is what selecting it *means* — so an exemplar-built pushforward is
    under-dispersed exactly in the soft/wide-angle corner it most needs to get right. The
    same applies to the MBR medoid.

    The provenance is already on the object: `.risk` is set only by `mbr_select`, and
    `.cluster_mass` only by `mbr_cluster_set`. Any consumer building an aggregate series
    from per-jet objects should call this first; it is cheap and it fails loudly, which a
    silently narrow distribution does not."""
    bad = []
    for k, pe in enumerate(estimates or []):
        if getattr(pe, "cluster_mass", None) is not None:
            bad.append(f"[{k}] cluster exemplar (cluster_mass is set)")
        elif getattr(pe, "risk", None) is not None:
            bad.append(f"[{k}] MBR medoid (risk is set)")
    if bad:
        raise ValueError(
            f"{where} must consume ANCESTRAL draws, not selected trees — a selected tree "
            f"is systematically closer to the bulk than a typical draw, so the resulting "
            f"series is under-dispersed exactly in the soft/wide-angle corner it most "
            f"needs to get right. Offending entries: " + "; ".join(bad[:5])
            + ("" if len(bad) <= 5 else f" (+{len(bad) - 5} more)")
        )


def set_size_for(masses, threshold: float) -> int:
    """How many mass-descending clusters reach `threshold` — the conformal set size."""
    m = np.asarray(masses, dtype=float)
    if m.size == 0:
        return 0
    acc = np.cumsum(m)
    hit = np.flatnonzero(acc >= float(threshold))
    return int(hit[0] + 1) if hit.size else int(m.size)


# ---------------------------------------------------------------------------
# WP6 §10.1b — the truth-based set diagnostic (G2'), with its mandatory controls
# ---------------------------------------------------------------------------
def support_radii(D, labels, exemplars, q: float = 0.95) -> np.ndarray:
    """Per-cluster `q`-quantile of the member-to-exemplar distance — the support bound.

    The companion of `radii` (which is a MEAN): a mean radius says how tight the cluster
    is, this says how far its members actually reach, and "is the truth inside this
    cluster's support" is a question about the reach. Used by `assign_truth`."""
    D = np.asarray(D, dtype=float)
    lab = np.asarray(labels)
    out = []
    for j, e in enumerate(exemplars):
        mem = np.flatnonzero(lab == j)
        d = D[int(e), mem] if mem.size else np.zeros(0)
        out.append(float(np.quantile(d, float(q))) if d.size else float("nan"))
    return np.asarray(out, dtype=float)


def assign_truth(d_to_exemplars, bounds, *, slack: float = 1.0) -> int:
    """Nearest-exemplar assignment of the truth, with an explicit **unassigned** verdict.

    Returns the exemplar index, or `-1` when the truth is farther from the nearest exemplar
    than that cluster's own support bound (`support_radii`, the 95th percentile of its
    members' distance to the exemplar), scaled by `slack`.

    Nearest-exemplar assignment is itself a decision, and it is wrong when the truth sits
    between clusters or outside the pool's support. Force-assigning would hide that inside
    a calibration statistic, where it would read as a miscalibrated probability rather than
    as an out-of-support jet. The rate of `-1` is a per-jet out-of-support indicator no
    calibration statistic sees, and it belongs beside the existing `support_audit` numbers.

    `bounds` may be a scalar or per-cluster; NaN entries never reject."""
    d = np.asarray(d_to_exemplars, dtype=float)
    if d.size == 0:
        return -1
    j = int(np.argmin(d))
    b = np.broadcast_to(np.asarray(bounds, dtype=float).reshape(-1), d.shape)
    if np.isfinite(b[j]) and d[j] > float(slack) * b[j]:
        return -1
    return j


def random_partition_null(D, masses, d_to_truth, *, n_reps: int = 20, seed: int = 0,
                          weights=None) -> dict:
    """The **decisive** control for G2': a mass-matched random partition of the same pool.

    Taking a minimum over `n` exemplars improves the distance to truth even for a random
    partition, purely as an order statistic — so `d_best < d_mbr` is not evidence of
    anything. The signal is `d_best_real` vs `d_best_rand`, and this computes the latter:
    partition the draws at random into the same number of groups with the same masses,
    take within-group medoids as exemplars, and recompute the min-over-exemplars distance.

    Reuses `D` and the already-computed per-draw truth distances; no new EMD calls."""
    D = symmetrize(D)
    K = int(D.shape[0])
    m = np.asarray(masses, dtype=float)
    dt = np.asarray(d_to_truth, dtype=float)
    if m.size < 2 or dt.size != K:
        return {"d_best_rand": float("nan"), "n_reps": 0, "sd": float("nan")}
    w = np.ones(K) if weights is None else np.asarray(weights, dtype=float)
    sizes = np.maximum(1, np.round(m / m.sum() * K).astype(int))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_reps)):
        perm = rng.permutation(K)
        start, best = 0, np.inf
        for s in sizes:
            grp = perm[start:start + int(s)]
            start += int(s)
            if grp.size == 0:
                continue
            best = min(best, float(dt[_medoid(D, list(grp), w)]))
        if np.isfinite(best):
            vals.append(best)
    if not vals:
        return {"d_best_rand": float("nan"), "n_reps": 0, "sd": float("nan")}
    a = np.asarray(vals, dtype=float)
    return {
        "d_best_rand": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n_reps": int(a.size),
    }
