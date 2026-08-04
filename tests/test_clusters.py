"""The posterior cluster layer (`inference/clusters.py`, docs/PLAN_PosteriorClusters.md).

Three things are pinned here and nothing else would catch them:

1. **The partition is recovered, and so are the masses.** A two-lobe distance matrix must
   come back as two clusters at the right weights under every method, because the mass IS
   the deliverable — an exemplar with the wrong mass is worse than no set at all.
2. **The degenerate cases are answers, not crashes.** An all-zero matrix is one cluster;
   an all-empty-draw posterior is one *zero-radius* cluster, which is gate G3 in
   miniature (`mbr._empty_value` returns 0 for two empty clouds, so the N = 0 stratum is a
   zero-diameter clique any density method finds by construction).
3. **The §4 guards raise.** A non-metric `D` produces a partition of nothing, and a
   surrogate-backend mass vector is blind to exactly the quantity that separates the
   strata. Both are silent failures otherwise.

`scikit-learn` is optional (the `[mbr]` extra), so the `hdbscan` / `dbscan` methods skip
where it is absent; `pam` is pure NumPy and always runs, which is why the invariants that
must hold *everywhere* are asserted on it.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference import clusters as cl

SKLEARN = importlib.util.find_spec("sklearn") is not None
METHODS = [
    pytest.param("hdbscan", marks=[] if SKLEARN else pytest.mark.skip(reason="no scikit-learn")),
    pytest.param("dbscan", marks=[] if SKLEARN else pytest.mark.skip(reason="no scikit-learn")),
    pytest.param("pam", marks=[]),  # pure numpy, always available
]
GEOM = Geometry()


def _two_lobe(n_a=40, n_b=60, sep=6.0, scale=0.2, seed=0):
    """Two well-separated Gaussian blobs in the plane -> their distance matrix.

    Points are only ever used to BUILD `D`; the clustering never sees a coordinate, which
    is the property that lets the same code run on trees."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, scale, (n_a, 2)), rng.normal(sep, scale, (n_b, 2))])
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1), n_a, n_b


def _one_lobe(n=100, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, scale, (n, 2))
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


# ---------------------------------------------------------------------------
# 1. the partition and the masses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_two_lobes_recovered_with_the_right_masses(method):
    D, n_a, n_b = _two_lobe()
    cs = cl.cluster_posterior(D, method=method)
    assert cs.n_clusters == 2
    assert cs.masses.sum() + cs.residual_mass == pytest.approx(1.0, abs=1e-9)
    # mass-descending, and the bigger lobe is first
    assert cs.masses[0] >= cs.masses[1]
    assert cs.top_mass == pytest.approx(n_b / (n_a + n_b), abs=0.06)
    # the exemplars are genuine members, one per lobe
    assert len(set(cs.exemplars)) == 2
    assert (cs.exemplars[0] >= n_a) != (cs.exemplars[1] >= n_a)
    # radii are the WITHIN-lobe scatter, far below the separation
    assert float(cs.radii.max()) < 1.0 < cs.separation
    assert cs.silhouette > 0.5
    # Entropy is the ambiguity over the REPORTED vector, residual bucket included — a
    # density method that leaves a few draws as noise is genuinely less certain than one
    # that assigns them, and dropping the residual would hide exactly that.
    m = np.array([*cs.masses, cs.residual_mass])
    m = m[m > 0]
    assert cs.entropy == pytest.approx(float(-(m * np.log(m)).sum()), abs=1e-9)
    # ...and it is a real two-way ambiguity: well above the 0 of a unimodal answer, and
    # below the log 3 that three equal explanations would give.
    assert 0.5 < cs.entropy < np.log(3.0)


@pytest.mark.parametrize("method", METHODS)
def test_labels_index_the_mass_vector(method):
    """`labels[k] == j` must mean "draw k is in the cluster whose mass is `masses[j]`".
    Nothing downstream re-derives this, so a relabelling bug would silently swap masses."""
    D, _a, _b = _two_lobe()
    cs = cl.cluster_posterior(D, method=method)
    for j in range(cs.n_clusters):
        assert cs.labels[cs.exemplars[j]] == j
        assert (cs.labels == j).mean() == pytest.approx(cs.masses[j], abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_weights_move_the_masses_and_nothing_else(method):
    """`weights` is how `mbr_resample_to_qn` composes here. It reweights the MASSES; it
    must not silently reweight the partition, or the q(N|x) correction would change which
    explanations exist rather than how probable they are."""
    D, n_a, _n_b = _two_lobe()
    w = np.ones(D.shape[0])
    w[:n_a] = 4.0                                   # up-weight the smaller lobe 4x
    base = cl.cluster_posterior(D, method=method)
    up = cl.cluster_posterior(D, method=method, weights=w)
    assert up.n_clusters == base.n_clusters
    assert set(up.labels.tolist()) == set(base.labels.tolist())
    assert up.weighted and not base.weighted
    assert up.masses[0] != pytest.approx(base.masses[0], abs=1e-6)


# ---------------------------------------------------------------------------
# 2. the degenerate cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_all_zero_matrix_is_one_cluster(method):
    cs = cl.cluster_posterior(np.zeros((16, 16)), method=method)
    assert cs.n_clusters == 1
    assert cs.top_mass == pytest.approx(1.0)
    assert cs.radii[0] == pytest.approx(0.0)
    assert cs.entropy == pytest.approx(0.0)
    assert cs.residual_mass == pytest.approx(0.0)


@pytest.mark.parametrize("method", METHODS)
def test_all_empty_draws_are_one_zero_radius_cluster(method):
    """Gate G3 in miniature, on the real distance function rather than a stand-in.

    `mbr._empty_value` returns exactly 0 for two empty clouds, so a posterior of nothing
    but empty trees has an all-zero `D`. The N = 0 stratum must come back as ONE cluster of
    radius 0 carrying all the mass — that mass is q(0|x), the quantity v1 measured as
    well-calibrated (AUC ~ 0.820) while every point estimator mishandled it."""
    from h2p_rsd_junipr.inference import mbr

    clouds = [mbr.lund_cloud([], GEOM) for _ in range(12)]
    D = mbr.lund_emd_matrix(clouds, clouds, backend="surrogate", geom=GEOM)
    assert np.allclose(D, 0.0)
    cs = cl.cluster_posterior(D, method=method, backend="surrogate", screening_only=True)
    assert cs.n_clusters == 1 and cs.radii[0] == pytest.approx(0.0)
    assert cs.top_mass == pytest.approx(1.0)


@pytest.mark.parametrize("method", METHODS)
def test_the_empty_stratum_separates_from_the_non_empty_one(method):
    """The structural claim the plan exploits: empty draws sit at mutual distance 0 and at
    a large CONSTANT distance from every non-empty draw, so the N = 0 stratum is its own
    cluster by construction rather than by luck."""
    from h2p_rsd_junipr.inference import mbr

    # 16 non-empty draws and 8 empty ones: the empty clique has to clear the default
    # min_cluster_size = max(5, ceil(0.05 K)), or a density method is right to call it noise.
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 4 + [[]] * 8
    clouds = [mbr.lund_cloud(d, GEOM, lnkt_cut=0.0) for d in draws]
    D = mbr.lund_emd_matrix(clouds, clouds, backend="surrogate", geom=GEOM)
    cs = cl.cluster_posterior(D, method=method, backend="surrogate", screening_only=True,
                              min_mass=0.05)
    empty = np.array([len(d) == 0 for d in draws])
    lab_empty = {int(c) for c in cs.labels[empty]}
    lab_other = {int(c) for c in cs.labels[~empty]}
    assert len(lab_empty) == 1, "the empty draws must not be split across clusters"
    assert not (lab_empty & lab_other), "the N=0 stratum must not share a cluster"


def test_single_lobe_is_one_cluster_under_pam():
    """The kill-criterion outcome must be reachable: an effectively unimodal posterior
    comes back as ONE cluster, not as a partition of noise. `pam` decides this by
    silhouette, so it is the method where the decision is explicit."""
    cs = cl.cluster_posterior(_one_lobe(), method="pam")
    assert cs.n_clusters == 1
    assert cs.top_mass == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. the §4 guards
# ---------------------------------------------------------------------------
def test_rectangular_D_raises():
    with pytest.raises(ValueError, match="square"):
        cl.cluster_posterior(np.zeros((4, 9)), method="pam")


def test_n_candidates_guard_raises_and_names_the_knob():
    with pytest.raises(ValueError, match="mbr_n_candidates"):
        cl.assert_cluster_metric_ok({"mbr_n_candidates": 24}, GEOM)


def test_beta_guard_raises():
    """beta != 1 breaks the triangle inequality (measured: 300 / 64 000 triples at beta=2),
    so HDBSCAN's mutual-reachability distance is not a distance."""
    with pytest.raises(ValueError, match="mbr_beta"):
        cl.assert_cluster_metric_ok({"mbr_beta": 2.0}, GEOM)


def test_R_guard_uses_the_geometry_not_a_hard_coded_diameter():
    """KMT's condition is `R >= R_max/2` for the ACTIVE coordinates, so the bound has to
    move with `geometry` and with `mbr_coords` — hard-coding 8.485 would pass a geometry
    it should fail."""
    assert cl.ground_diameter(GEOM) == pytest.approx(6.0 * np.sqrt(2.0), rel=1e-9)
    assert cl.ground_diameter(GEOM, "+lnz") > cl.ground_diameter(GEOM)
    assert cl.ground_diameter(GEOM, "+psi") > cl.ground_diameter(GEOM, "+lnz")
    cl.assert_cluster_metric_ok({"mbr_R": 8.485}, GEOM)             # the default passes
    with pytest.raises(ValueError, match="ground diameter"):
        cl.assert_cluster_metric_ok({"mbr_R": 1.0}, GEOM)
    wide = Geometry(ln_invdelta_range=(0.0, 20.0), ln_kt_range=(0.0, 20.0))
    with pytest.raises(ValueError, match="ground diameter"):
        cl.assert_cluster_metric_ok({"mbr_R": 8.485}, wide)


def test_surrogate_backend_refused_outside_screening():
    """`_lund_image` normalises, so the surrogate is EXACTLY blind to total kt and
    multiplicity — the quantity that separates the N strata. A mass vector from it is not
    wrong by a little."""
    D, _a, _b = _two_lobe()
    with pytest.raises(ValueError, match="surrogate"):
        cl.cluster_posterior(D, method="pam", backend="surrogate")
    cs = cl.cluster_posterior(D, method="pam", backend="surrogate", screening_only=True)
    assert cs.screening_only is True and "SCREENING ONLY" in cs.summary()


def test_symmetrize_is_defensive_not_cosmetic():
    D, _a, _b = _two_lobe()
    D[0, 1] += 1e-12                       # solver round-off, as measured on `pot`
    S = cl.symmetrize(D)
    assert np.allclose(S, S.T, atol=0.0)
    assert np.all(np.diag(S) == 0.0)


# ---------------------------------------------------------------------------
# G2' — the random-partition null, with BOTH controls
# ---------------------------------------------------------------------------
def test_random_partition_null_on_a_single_lobe_is_the_negative_control():
    """On a single-lobe matrix the real partition must NOT beat the mass-matched null.

    Without this the whole of G2' is an untested order statistic: taking a minimum over n
    exemplars improves the distance even for a random partition, so `d_best < d_mbr` is
    evidence of nothing at all."""
    D = _one_lobe(n=120, seed=3)
    rng = np.random.default_rng(7)
    d_truth = rng.normal(4.0, 0.5, D.shape[0])       # truth equidistant-ish from everything
    cs = cl.cluster_posterior(D, method="pam", pam_min_silhouette=-1.0)  # force a split
    d_best_real = min(float(d_truth[e]) for e in cs.exemplars)
    null = cl.random_partition_null(D, cs.masses, d_truth, n_reps=40, seed=0)
    assert null["n_reps"] == 40
    # within ~2 sigma of the null: the "structure" carries no information about the truth
    assert abs(d_best_real - null["d_best_rand"]) < 3.0 * max(null["sd"], 1e-6)


def test_random_partition_null_positive_control_on_two_lobes():
    """...and on a genuinely two-lobed posterior whose truth sits inside the MINORITY
    lobe, the real partition must beat the null by a lot. Paired with the negative control
    above, this is what makes G2' a measurement."""
    D, n_a, n_b = _two_lobe(n_a=40, n_b=60, sep=8.0, seed=5)
    d_truth = np.concatenate([np.full(n_a, 0.5), np.full(n_b, 8.0)])   # truth in lobe A
    cs = cl.cluster_posterior(D, method="pam")
    d_best_real = min(float(d_truth[e]) for e in cs.exemplars)
    null = cl.random_partition_null(D, cs.masses, d_truth, n_reps=40, seed=0)
    assert d_best_real < null["d_best_rand"] - 1.0


def test_assign_truth_reports_unassigned_rather_than_forcing_it():
    d = np.array([10.0, 12.0])
    bounds = np.array([0.5, 0.5])
    assert cl.assign_truth(np.array([0.1, 12.0]), bounds) == 0
    assert cl.assign_truth(d, bounds) == -1, "a truth outside every cluster's support "
    assert cl.assign_truth(np.zeros(0), bounds) == -1


# ---------------------------------------------------------------------------
# §8.1 — the load-bearing orthogonality claim, as a test rather than a paragraph
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_losses_do_not_move_clusters(method):
    """`cluster_posterior` consumes `D`; `_reduce_risk` consumes `D` and emits a K-vector
    whose argmin is the point estimate. The partition never sees the risk vector, so
    `labels`, `exemplars`, `masses` and `radii` are BIT-IDENTICAL across all three losses.

    §8.1 is the plan's load-bearing orthogonality claim — everything from "the cluster
    products run at stock MBR settings" to "WP4b moves only the single point estimate"
    rests on it — so it is asserted, not assumed."""
    from h2p_rsd_junipr.inference.mbr import _reduce_risk, bandwidth_quantile

    D, _a, _b = _two_lobe()
    eps = bandwidth_quantile(D, 0.10)
    ref = cl.cluster_posterior(D, method=method)
    argmins = set()
    for loss in ("linear", "bounded", "kernel"):
        risk = _reduce_risk(D, None, loss=loss, eps=eps)
        argmins.add(int(np.argmin(risk)))
        got = cl.cluster_posterior(D, method=method)      # same D, whatever the loss was
        assert np.array_equal(got.labels, ref.labels)
        assert got.exemplars == ref.exemplars
        assert np.array_equal(got.masses, ref.masses)
        assert np.array_equal(got.radii, ref.radii)
        assert got.top_mass == ref.top_mass and got.entropy == ref.entropy
    assert argmins, "the losses were never actually evaluated"


# ---------------------------------------------------------------------------
# WP5 — the conformal threshold
# ---------------------------------------------------------------------------
def test_fit_set_threshold_covers_at_least_1_minus_alpha():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 500)
    fit = cl.fit_set_threshold(scores, alpha=0.2)
    assert fit["fitted_under"]["finite_sample_exact"]
    assert float(np.mean(scores <= fit["value"])) >= 0.8 - 0.02
    assert "marginal" in fit["fitted_under"]["coverage"]


def test_fit_set_threshold_degrades_honestly_below_the_sample_it_needs():
    """With too few calibration jets for the requested alpha the exact order statistic does
    not exist. The honest answer is "emit everything", not a threshold that silently
    under-covers."""
    fit = cl.fit_set_threshold([0.3, 0.4], alpha=0.05)
    assert fit["value"] == 1.0 and not fit["fitted_under"]["finite_sample_exact"]


def test_set_size_for_is_the_smallest_prefix():
    m = np.array([0.5, 0.3, 0.15, 0.05])
    assert cl.set_size_for(m, 0.4) == 1
    assert cl.set_size_for(m, 0.6) == 2
    assert cl.set_size_for(m, 0.99) == 4
    assert cl.set_size_for(m, 1.5) == 4      # unreachable -> everything, not an error


# ---------------------------------------------------------------------------
# WP5.1 — the sample split, and the direction of the bias it removes
# ---------------------------------------------------------------------------
def test_sample_split_masses_are_estimated_off_the_selection_pool():
    D, _a, _b = _two_lobe(n_a=60, n_b=60, seed=11)
    split = np.zeros(D.shape[0], dtype=bool)
    split[::2] = True
    cs = cl.cluster_posterior(D, method="pam", split_index=split)
    assert cs.split is True and any("sample-split" in n for n in cs.notes)
    assert cs.n_clusters == 2
    # every draw still carries a label: pool B is assigned to the A-exemplars
    assert (cs.labels >= 0).all()
    assert cs.masses.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# §10.6 — pushforward hygiene, asserted on the provenance flag
# ---------------------------------------------------------------------------
def test_assert_ancestral_draws_rejects_selected_trees():
    from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate

    draw = LundPointEstimate(nodes=[], logprob=0.0, multiplicity=0)
    medoid = LundPointEstimate(nodes=[], logprob=0.0, multiplicity=0, risk=1.0)
    exemplar = LundPointEstimate(nodes=[], logprob=0.0, multiplicity=0, cluster_mass=0.6)
    cl.assert_ancestral_draws([draw, draw])          # ancestral draws are fine
    with pytest.raises(ValueError, match="ANCESTRAL"):
        cl.assert_ancestral_draws([draw, medoid])
    with pytest.raises(ValueError, match="ANCESTRAL"):
        cl.assert_ancestral_draws([exemplar])
