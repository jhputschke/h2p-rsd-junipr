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

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference import clusters as cl
from h2p_rsd_junipr.models.base import build_model

SKLEARN = importlib.util.find_spec("sklearn") is not None
_POT_OK = importlib.util.find_spec("ot") is not None
METHODS = [
    pytest.param("hdbscan", marks=[] if SKLEARN else pytest.mark.skip(reason="no scikit-learn")),
    pytest.param("dbscan", marks=[] if SKLEARN else pytest.mark.skip(reason="no scikit-learn")),
    pytest.param("pam", marks=[]),  # pure numpy, always available
]
GEOM = Geometry()


def _jet(batch):
    b, geom = batch
    return b["xf"][:1], b["nx"][:1], geom


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


# ---------------------------------------------------------------------------
# The emptiness decision: the mass argmax is the wrong rule for the N = 0 stratum
# ---------------------------------------------------------------------------
def _empty_dominated_draws():
    """A posterior whose EMPTY stratum wins the mass argmax without being a majority.

    This is the granularity artifact in miniature: the empty draws sit at mutual distance
    exactly 0 and so form ONE atomic cluster, while the non-empty draws spread out and get
    fragmented into several. 8 empty draws (a third of the pool) beat three non-empty
    clusters of 6/5/5 — even though two jets in three are not empty."""
    return ([[]] * 8
            + [[12, 34, 56]] * 6
            + [[5, 9]] * 5
            + [[40, 41, 42, 43]] * 5)


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_mass_argmax_lets_the_empty_stratum_win_without_a_majority(batch):
    """The failure the gate exists to fix, pinned as a measurement rather than described.

    Measured on 600 held-out jets at K = 200: `members[0]` answers EMPTY on 29.8% against a
    true rate of 16.7%. The mechanism is here in miniature."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = _empty_dominated_draws()
    ps = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                           cluster_method="pam", cluster_min_mass=0.05)
    assert ps.empty_cluster == 0, "the empty stratum should have won the mass argmax here"
    assert ps.masses[0] < 0.5, (
        "...and it wins with a PLURALITY, not a majority — which is the whole artifact: "
        "one atomic lump against a fragmented competitor set"
    )
    assert ps.point.multiplicity == 0          # the default rule recommends the empty tree
    assert ps.empty_policy == "include" and ps.empty_gate_fired is None


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_empty_gate_moves_the_recommendation_but_not_the_set(batch):
    """`decode.empty_threshold` decides emptiness for the SET too, and only `.point` moves.

    `members`, `masses` and `radii` are untouched, so the conformal prefix and every
    existing consumer are unaffected — the set still carries the empty explanation, because
    a rejected alternative is still a reported alternative."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = _empty_dominated_draws()
    base = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                             cluster_method="pam")
    # tau above q(0|x) = 8/24 -> the gate does NOT fire -> recommend the top NON-empty
    off = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                            cluster_method="pam", empty_threshold=0.90)
    assert off.empty_gate_fired is False and off.empty_policy == "gate"
    assert off.point_index == 1 and off.point.multiplicity > 0
    # ...and tau below it -> the gate fires -> the empty explanation is the answer
    on = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                           cluster_method="pam", empty_threshold=0.10)
    assert on.empty_gate_fired is True and on.point.multiplicity == 0

    # the SET is identical in all three: only `.point` moved
    for other in (off, on):
        assert np.array_equal(other.masses, base.masses)
        assert np.array_equal(other.radii, base.radii)
        assert [m.multiplicity for m in other.members] == [m.multiplicity for m in base.members]
        assert other.members[0].multiplicity == 0, "members[0] keeps meaning top-mass"


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_empty_threshold_zero_is_bit_identical(batch):
    """The parity rule: the new knob's OFF path is the merged behaviour exactly."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = _empty_dominated_draws()
    kw = dict(point_estimator="mbr", mbr_backend="pot", cluster_method="pam")
    a = model.predict_set(xf, nx, draws=draws, **kw)
    b = model.predict_set(xf, nx, draws=draws, empty_threshold=0.0, **kw)
    assert a.point_index == b.point_index == 0
    assert a.empty_policy == b.empty_policy == "include"
    assert np.array_equal(a.masses, b.masses)


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_the_gate_never_fabricates_an_empty_tree(batch):
    """H = {pool}. A `q(0|x)` above tau with NO empty draw in the pool is a disagreement
    between the length head and the sampler, not an explanation — so the recommendation
    stays inside the posterior rather than inventing the tree the gate asked for.

    Needs a family with an EXPLICIT `q(N|x)` head. For the continue/stop families the length
    belief *is* the sampler histogram, so `q(0|x)` is identically 0 when no draw is empty
    and the gate cannot fire at all — a real property, but it makes them unable to exhibit
    the disagreement this test is about."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=cinn", "encoder=deepsets"]), geom).eval()
    draws = [[12, 34, 56]] * 8 + [[5, 9]] * 8      # nothing empty anywhere
    if float(np.asarray(model.length_pmf(xf, nx))[0]) <= 0.0:
        pytest.skip("this head puts no mass at N=0, so the gate cannot fire")
    ps = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                           cluster_method="pam", empty_threshold=1e-12)  # fires on any mass
    assert ps.empty_gate_fired is True and ps.empty_cluster is None
    assert ps.point.multiplicity > 0, "no empty draw exists, so none may be recommended"
    assert ps.point is ps.members[0]


# ---------------------------------------------------------------------------
# `estimator` — which DECISION produced this tree
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_a_gated_empty_answer_does_not_report_itself_as_a_MAP(batch):
    """`pretty()` used `risk is not None` as the MAP-vs-MBR discriminator, and
    `map_or_mbr`'s empty gate returns BEFORE `mbr_select` runs — so an MBR decode that
    answered the empty tree carried no risk and printed as a MAP. The tree was right and
    the label was wrong, which is the failure mode this repo tracks most carefully."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[]] * 7 + [[12, 34]] * 3          # q(0|x) = 0.7, so any sane tau fires
    pe = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                          empty_threshold=0.5)
    assert pe.multiplicity == 0 and pe.risk is None      # the gate short-circuited
    assert pe.estimator == "empty_gate"
    assert "MAP" not in pe.pretty(), "a gate-decided answer is not a MAP"
    assert "EMPTY-GATED" in pe.pretty()
    assert "before any shape decode" in pe.pretty().lower()


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_estimator_labels_every_decision_and_leaves_the_others_unchanged(batch):
    """The four labels, and the parity rule: nothing that used to print "MAP"/"MBR" moves."""
    xf, nx, geom = _jet(batch)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    draws = [[12, 34, 56]] * 6 + [[5, 9]] * 5 + [[]] * 5

    mp = model.map_or_mbr(xf, nx, draws=draws, point_estimator="map")
    assert mp.estimator == "map" and mp.pretty().startswith("MAP groomed shower")

    mbr = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot")
    assert mbr.estimator == "mbr" and mbr.pretty().startswith("MBR groomed shower")

    ps = model.predict_set(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot",
                           cluster_method="pam")
    assert all(m.estimator == "cluster" for m in ps.members)
    assert ps.members[0].pretty().startswith("cluster-exemplar groomed shower")

    # a bare `describe_cells` is a SCORING call, not an estimator: it keeps the default,
    # so its printed form is unchanged from before the field existed.
    assert model.describe_cells(xf, nx, [12, 34]).estimator == "map"


def test_assert_ancestral_draws_rejects_a_gated_empty_tree():
    """The one a `risk`/`cluster_mass` check alone would wave through: the gate returns
    before either estimator runs, so a gate-decided empty tree carries neither — while
    being every bit as much a decision, and pushing it forward would bias the multiplicity
    marginal toward zero on exactly the jets the gate fired on."""
    from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate

    gated = LundPointEstimate(nodes=[], logprob=0.0, multiplicity=0,
                              estimator="empty_gate")
    assert gated.risk is None and gated.cluster_mass is None    # invisible to the old check
    with pytest.raises(ValueError, match="ANCESTRAL"):
        cl.assert_ancestral_draws([gated])


def test_never_covered_jets_are_counted_not_dropped():
    """A truth no prefix covers has NO finite score, and dropping it is the tempting thing
    to do — it silently conditions the guarantee on assignment and reports a coverage that
    cannot fail for the one reason it most needs to.

    At an unassigned rate `u` no threshold reaches coverage above `1 - u`, so with
    `1 - alpha` above that ceiling the honest answer is "emit everything, and it still
    under-covers" rather than a threshold that looks like it worked."""
    rng = np.random.default_rng(0)
    covered = rng.uniform(0.0, 0.6, 60)
    scores = np.concatenate([covered, np.full(40, np.nan)])   # 40% never covered

    fit = cl.fit_set_threshold(scores, alpha=0.32)            # wants 0.68, ceiling is 0.60
    assert fit["max_achievable_coverage"] == pytest.approx(0.60)
    assert fit["reachable"] is False
    assert fit["value"] == 1.0, "the best available answer is the full set"
    assert fit["fitted_under"]["n_never_covered"] == 40
    assert fit["fitted_under"]["n_calibration"] == 100, "the dropped jets were the point"

    # ...and with the nominal INSIDE the ceiling it behaves normally again
    ok = cl.fit_set_threshold(scores, alpha=0.60)             # wants 0.40, ceiling 0.60
    assert ok["reachable"] is True and ok["value"] < 1.0
    assert float(np.mean(np.nan_to_num(scores, nan=np.inf) <= ok["value"])) >= 0.40 - 0.02


def test_all_finite_scores_are_unchanged_by_the_never_covered_path():
    """Parity: with every jet assigned, the threshold is the plain order statistic."""
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 400)
    fit = cl.fit_set_threshold(s, alpha=0.2)
    assert fit["reachable"] is True
    assert fit["max_achievable_coverage"] == 1.0
    assert fit["fitted_under"]["n_never_covered"] == 0
    assert fit["value"] == pytest.approx(float(np.sort(s)[int(np.ceil(401 * 0.8)) - 1]))


# ---------------------------------------------------------------------------
# WP3 — a coverage bound that does not move with the clustering method
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_pool_coverage_bound_is_method_stable(method):
    """The property the exemplar-support rule cannot have.

    `assign_truth` compares the truth to each CLUSTER's support radius, so a finer
    partition rejects more: measured 35.7% "unassigned" under hdbscan against 8.2% under
    pam, on the same jets and the same draws. A statement about whether the MODEL covers
    the truth must not swing 4x with how its output was later cut up. This bound
    references the pool's own nearest-neighbour scale and nothing else."""
    D, n_a, _n_b = _two_lobe()
    bound = cl.pool_coverage_bound(D)
    cs = cl.cluster_posterior(D, method=method)
    assert cs.n_clusters == 2
    # the bound is a property of D alone — the partition cannot move it
    assert cl.pool_coverage_bound(D) == bound
    assert 0.0 < bound < 1.0, "the within-lobe nearest-neighbour scale"

    # a truth sitting inside lobe A is covered by the full set, and NOT by lobe B alone
    d_truth = np.concatenate([np.full(n_a, 0.5 * bound), np.full(D.shape[0] - n_a, 6.0)])
    lab_a = int(cs.labels[0])
    assert cl.pool_covered(d_truth, cs.labels, range(cs.n_clusters), bound)
    assert not cl.pool_covered(d_truth, cs.labels,
                               [j for j in range(cs.n_clusters) if j != lab_a], bound)
    assert not cl.pool_covered(d_truth, cs.labels, [], bound), "an empty set covers nothing"


def test_pool_coverage_bound_handles_the_degenerate_pool():
    """All draws identical -> nearest-neighbour distance 0 -> a bound of 0, which is the
    honest answer (the pool resolves nothing) rather than a NaN."""
    assert cl.pool_coverage_bound(np.zeros((8, 8))) == 0.0
    assert cl.pool_coverage_bound(np.zeros((1, 1))) == 0.0
    # ...and the off-diagonal must not be poisoned by the self-distance mask
    D, _a, _b = _two_lobe()
    assert np.isfinite(cl.pool_coverage_bound(D))


# ---------------------------------------------------------------------------
# WP-3 / A3 — the truth and the draws in the SAME representation
# (docs/PLAN_z_aware.md §4/WP-3 inset, docs/PLAN_next_steps.md A2/A3)
#
# `_truth_cloud` has always built the truth from the continuous `yraw` rows while every
# DRAW cloud came from cell centres. Under the default `mbr_weight="kt"` that weights the
# truth by `exp(v_continuous)` and the draws by `exp(v_cell_centre)` — a per-point
# mismatch plus a Jensen inflation of the truth's total mass, which the EMD charges at
# `R*|dW|`. `d_top`, `d_best`, `d_mbr`, `d_nearest_draw` and gates G2'/G6/G7 all sit on it.
# `mbr_cloud_source="coords"` removes it as a side effect, and these tests pin BOTH the
# defect and its fix, so neither can regress unnoticed.
# ---------------------------------------------------------------------------
def test_R_guard_is_coords_dependent(monkeypatch):
    """The same `R` can be admissible at 2-D and inadmissible at `+lnz`.

    `assert_cluster_metric_ok` already reads `mbr_coords`; this pins that it does, because
    WP-3 is what finally makes a caller set that knob to something other than the default —
    and an `R` chosen for the 2-D diameter is the first thing that would silently break."""
    d2 = cl.ground_diameter(GEOM, "lnDR_lnkt")
    d3 = cl.ground_diameter(GEOM, "+lnz")
    assert d3 > d2
    R = 0.5 * (d2 / 2.0 + d3 / 2.0)              # clears the 2-D bound, misses the 3-D one
    cl.assert_cluster_metric_ok({"mbr_R": R, "mbr_coords": "lnDR_lnkt"}, GEOM)
    with pytest.raises(ValueError, match="ground diameter"):
        cl.assert_cluster_metric_ok({"mbr_R": R, "mbr_coords": "+lnz"}, GEOM)


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_truth_and_draws_share_a_representation_only_under_coords(small_jets):
    """`|W_truth - W_draw| / W` is ~0 under `"coords"` and demonstrably nonzero under
    `"cells"` — the defect and the fix in one assertion.

    Constructed so the comparison is exact: the "draws" ARE the truth's own cells, so any
    residual weight difference is the representation and nothing else."""
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.eval.clusters import _truth_cloud, _weight_audit
    from h2p_rsd_junipr.inference.mbr import lund_cloud

    geom = Geometry()
    ds = MatchedLundDataset(small_jets, geom)
    item = next(ds[i] for i in range(len(ds)) if int(ds[i]["ny"]) >= 2)
    yraw = np.asarray(item["yraw"].numpy(), dtype=float)
    cells = [int(c) for c in item["yc"].tolist()]
    kw = dict(lnkt_cut=-1e9, weight="kt", coords="lnDR_lnkt")

    tc = _truth_cloud(item, geom, **kw)
    cells_cloud = lund_cloud(cells, geom, **kw)          # the DRAWS' representation
    coords_cloud = lund_cloud([row for row in yraw], geom, **kw)

    mismatched = _weight_audit(tc, cells_cloud, [cells_cloud], R=8.485)
    matched = _weight_audit(tc, coords_cloud, [coords_cloud], R=8.485)
    assert matched["W_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert matched["R_dW"] == pytest.approx(0.0, abs=1e-9)
    # ...and the "cells" side is NOT ~0: the truth carries a different total mass from the
    # cell-centre representation of the very same emissions.
    assert abs(mismatched["W_ratio"] - 1.0) > 1e-6
    assert mismatched["R_dW"] > 1e-6


@pytest.mark.skipif(not _POT_OK, reason="POT not installed")
def test_truth_and_draws_are_both_three_dimensional_under_lnz(small_jets):
    """Under `+lnz` the truth's third column is its real `ln z` and so is every draw's —
    both non-constant. Before WP-3 the draw side was the constant 0 the adapter invented,
    which is what made `+lnz` measure 2-D numbers under a 3-D label."""
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.eval.clusters import _truth_cloud
    from h2p_rsd_junipr.inference.mbr import lund_cloud

    geom = Geometry()
    ds = MatchedLundDataset(small_jets, geom)
    item = next(ds[i] for i in range(len(ds)) if int(ds[i]["ny"]) >= 3)
    yraw = np.asarray(item["yraw"].numpy(), dtype=float)
    kw = dict(lnkt_cut=-1e9, weight="kt", coords="+lnz")

    tpts, _ = _truth_cloud(item, geom, **kw)
    dpts, _ = lund_cloud([row for row in yraw], geom, **kw)
    assert tpts.shape[1] == 3 and dpts.shape[1] == 3
    assert tpts[:, 2].std() > 0 and dpts[:, 2].std() > 0
    # the cell-chain representation cannot even be asked for the third column
    with pytest.raises(ValueError, match="CELL CHAIN"):
        lund_cloud([int(c) for c in item["yc"].tolist()], geom, **kw)


@pytest.mark.skipif(not (_POT_OK and SKLEARN), reason="needs POT and scikit-learn")
def test_run_cluster_diagnostics_reports_the_weight_audit(small_jets):
    """The audit is a REPORTED number on every run, not an assumption. §4/WP-3 asked for
    `W_truth/W_draw` and `R*|dW|` against the typical `d` — priced in the units the metric
    charges in, because a ratio near 1 is not evidence of a small effect until it is."""
    import torch

    from h2p_rsd_junipr.config import decode_params
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.eval.clusters import run_cluster_diagnostics

    cfg = load_config(["model=ar_junipr_v2", "encoder=gru",
                       "decode.point_estimator=mbr", "decode.mbr_backend=pot"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).eval()
    jets = small_jets[:16]
    ds = MatchedLundDataset(jets, geom)
    m = run_cluster_diagnostics(model, ds, jets, geom, torch.device("cpu"), K=10,
                                n_jets=8, decode=decode_params(cfg), verbose=False,
                                null_reps=2)
    wa = m["weight_audit"]
    for k in ("W_truth_over_W_truth_as_drawn", "R_dW_mean", "R_dW_physical_mean",
              "R_dW_over_d_nearest_draw", "R_dW_over_R_dW_physical", "matched"):
        assert k in wa
    assert wa["matched"] is False          # the fielded "cells" path IS mismatched
    assert all("W_ratio" in r and "R_dW" in r for r in m["per_jet"])
    assert m["config"]["mbr_cloud_source"] == "cells"

    # ...and under "coords" the same call reports it matched, exactly (the ratio is 1 by
    # construction there, not by measurement).
    mc = run_cluster_diagnostics(model, ds, jets, geom, torch.device("cpu"), K=10,
                                 n_jets=8, verbose=False, null_reps=2,
                                 decode={**decode_params(cfg),
                                         "mbr_cloud_source": "coords"})
    assert mc["weight_audit"]["matched"] is True
    assert mc["weight_audit"]["R_dW_mean"] == pytest.approx(0.0, abs=1e-9)
