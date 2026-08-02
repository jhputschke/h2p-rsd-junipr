"""WP-B of docs/PLAN_prod_test_v1.md: the sampling-time continue temperature and the
multiplicity diagnostics that decide whether it is needed.

`decode.continue_temperature` tempers the CONTINUE/STOP logit at sampling only. The
tests that matter are the ones that pin what it must NOT touch: the trained likelihood,
the beam-search MAP, and — at `T = 1.0` — the sampler's own numbers, bit for bit.

Beside it, the two diagnostics WP-B.1 puts before any remedy:

* `length_marginal` reports `<N>` on the full population *and* on the truth-nonempty
  one, because comparing `E_q[N|x]` to jets selected by `N_true >= 1` is regression to
  the mean and yields a deficit for a perfectly calibrated posterior;
* `sbc_n_selfconsistency_null` replaces chi^2(9) — the null for a *continuous* rank —
  with the statistic's own distribution at this discreteness.

Both are tested against models whose answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.eval.exposure import (
    _sbc_midranks,
    continue_prob_by_depth,
    length_marginal,
    sbc_n_selfconsistency_null,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import fit_continue_temperature
from h2p_rsd_junipr.models.base import build_model

DEV = torch.device("cpu")


def _built(model_name="ar_junipr_v2", *, seed=0, **decode):
    argv = [f"model={model_name}", "encoder=gru"]
    argv += [f"decode.{k}={v}" for k, v in decode.items()]
    cfg = load_config(argv)
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    return build_model(cfg, geom).eval(), geom, cfg


@pytest.fixture(scope="module")
def jets():
    return synthetic_matched_dataset(200, seed=0)


@pytest.fixture(scope="module")
def dataset(jets):
    return MatchedLundDataset(jets, Geometry())


# ---------------------------------------------------------------------------
# 1. T = 1.0 is off, bit for bit
# ---------------------------------------------------------------------------
def test_default_is_one_and_is_a_no_op(dataset):
    m1, _, _ = _built(continue_temperature=1.0)
    m0, _, _ = _built()
    assert m0.continue_temperature == 1.0 and m1.continue_temperature == 1.0
    item = dataset[0]
    xf, nx = item["xf"].unsqueeze(0), torch.tensor([item["nx"]])
    torch.manual_seed(5)
    a = m0.sample(xf, nx, 200)
    torch.manual_seed(5)
    b = m1.sample(xf, nx, 200)
    assert a == b, "T = 1.0 must not perturb the RNG stream or the draws"


def test_temperature_never_touches_the_likelihood(dataset):
    """The whole claim of an inference-layer object: `per_jet_nll` is the trained
    likelihood and a decode knob may not move it."""
    b = collate([dataset[i] for i in range(16)])
    hot, _, _ = _built(continue_temperature=2.5)
    cold, _, _ = _built(continue_temperature=1.0)
    with torch.inference_mode():
        assert torch.equal(hot.per_jet_nll(b), cold.per_jet_nll(b))
        assert torch.equal(hot.log_prob(b), cold.log_prob(b))


def test_temperature_never_touches_the_map(dataset):
    """Beam search reads `cont_head` through `_step`, not `_step_batched`, so the MAP is
    the untempered argmax by construction — asserted, because the two step functions are
    one edit apart."""
    item = dataset[0]
    xf, nx = item["xf"].unsqueeze(0), torch.tensor([item["nx"]])
    hot, _, _ = _built(continue_temperature=3.0)
    cold, _, _ = _built(continue_temperature=1.0)
    assert hot.map_decode(xf, nx) == cold.map_decode(xf, nx)
    assert hot.map_estimate(xf, nx).logprob == pytest.approx(cold.map_estimate(xf, nx).logprob)


def _pin_p_cont(model, p):
    """Force the continue head to a constant probability, so the temperature's effect is
    read against a known `p_cont` instead of whatever an untrained head happens to hold
    (which sits at ~1/2, where the knob has nothing to move)."""
    with torch.no_grad():
        model.cont_head.weight.zero_()
        model.cont_head.bias.fill_(float(np.log(p / (1.0 - p))))


@pytest.mark.parametrize("p_cont,direction", [(0.2, +1), (0.8, -1)])
def test_temperature_moves_the_sampled_lengths(dataset, p_cont, direction):
    """T > 1 pulls `p_cont` toward 1/2 — which LENGTHENS trees where the head is
    confident to stop and SHORTENS them where it is confident to continue. Both
    directions are pinned, because "raising T lengthens" is only half true and the half
    that is false is the one that would make a fit run the wrong way."""
    item = dataset[0]
    xf, nx = item["xf"].unsqueeze(0), torch.tensor([item["nx"]])
    means = []
    for T in (0.5, 1.0, 2.0, 4.0):
        m, _, _ = _built(continue_temperature=T)
        _pin_p_cont(m, p_cont)
        torch.manual_seed(7)
        means.append(float(np.mean([len(d) for d in m.sample(xf, nx, 400)])))
    ordered = means if direction > 0 else means[::-1]
    assert ordered == sorted(ordered), f"mean length not monotone in T: {means}"
    assert abs(means[-1] - means[0]) > 0.1


def test_config_round_trip_and_default():
    dec = decode_params(load_config(["model=ar_junipr_v2"]))
    assert dec["continue_temperature"] == 1.0
    dec = decode_params(load_config(["model=ar_junipr_v2", "decode.continue_temperature=1.7"]))
    assert dec["continue_temperature"] == 1.7


def test_no_op_for_the_multiplicity_head_family(dataset, capsys):
    """A family with an explicit q(N|x) head takes no per-step continue decision. The
    knob must be inert AND say so, rather than silently doing nothing."""
    from h2p_rsd_junipr.eval.report import inert_decode_keys

    m, _, cfg = _built("ar_junipr_v3", continue_temperature=2.0)
    assert "NO-OP" in capsys.readouterr().out
    item = dataset[0]
    xf, nx = item["xf"].unsqueeze(0), torch.tensor([item["nx"]])
    torch.manual_seed(3)
    hot = m.sample(xf, nx, 200)
    m.continue_temperature = 1.0
    torch.manual_seed(3)
    assert m.sample(xf, nx, 200) == hot
    keys = {e["key"] for e in inert_decode_keys(m, decode_params(cfg))}
    assert "continue_temperature" in keys


# ---------------------------------------------------------------------------
# 2. the fit
# ---------------------------------------------------------------------------
def test_fit_recovers_a_known_temperature():
    """Bisection on a monotone surrogate whose root is known exactly."""
    truth = 1.8

    def mean_n(T):
        return 1.0 + 0.5 * (T - truth)      # monotone, root at T = truth -> <N> = 1.0

    T, info = fit_continue_temperature(mean_n, 1.0)
    assert T == pytest.approx(truth, abs=2e-3)
    assert info["bracketed"] and info["achieved_mean_n"] == pytest.approx(1.0, abs=2e-3)


def test_fit_reports_an_unbracketed_target_instead_of_extrapolating():
    T, info = fit_continue_temperature(lambda T: 1.0, 5.0, lo=0.5, hi=2.0)
    assert info["bracketed"] is False
    assert T in (0.5, 2.0)


def test_fit_drives_the_sampler_to_a_target(dataset):
    """End to end on the real sampler: the fitted T reproduces the requested mean
    multiplicity to within Monte-Carlo noise."""
    m, _, _ = _built(seed=1)
    _pin_p_cont(m, 0.25)     # a head with something for the knob to move
    item = dataset[0]
    xf, nx = item["xf"].unsqueeze(0), torch.tensor([item["nx"]])

    def mean_n(T):
        m.continue_temperature = float(T)
        torch.manual_seed(19)                # common random numbers: a smooth objective
        return float(np.mean([len(d) for d in m.sample(xf, nx, 600)]))

    # +0.25 rather than a larger step: `p_cont -> 1/2` is the T -> inf limit, so the
    # reachable range is bounded and a target outside it is an unbracketed fit, which
    # `test_fit_reports_an_unbracketed_target_instead_of_extrapolating` covers instead.
    base = mean_n(1.0)
    target = base + 0.25
    T, info = fit_continue_temperature(mean_n, target)
    assert info["bracketed"]
    assert mean_n(T) == pytest.approx(target, abs=0.1)
    assert set(info) >= {"T", "target_mean_n", "achieved_mean_n", "lo", "hi", "bracketed"}


# ---------------------------------------------------------------------------
# 3. the N-marginal diagnostic
# ---------------------------------------------------------------------------
def test_length_marginal_separates_the_two_populations(dataset):
    m, _, _ = _built("ar_junipr_v3", seed=2)
    out = length_marginal(m, dataset, DEV, n_jets=64, K=32, verbose=False)
    assert out["posterior_source"].startswith("exact")
    assert out["gate_population"] == "full"
    assert out["truth_nonempty"]["selection_biased"] is True
    assert out["full"]["n_jets"] >= out["truth_nonempty"]["n_jets"]
    # the ratio is the mean pair, not a mean of ratios
    f = out["full"]
    assert f["ratio"] == pytest.approx(f["mean_n_posterior"] / f["mean_n_truth"])
    assert f["signed_bias"] == pytest.approx(f["mean_n_posterior"] - f["mean_n_truth"], abs=1e-6)


def test_truth_selection_biases_the_nonempty_row_low():
    """The artefact itself, on a posterior that is calibrated BY CONSTRUCTION: draw the
    truth from the model's own q(N|x), then select on it. The full-population ratio is 1
    and the truth-nonempty ratio is below it — which is why gate G4 reads the first."""
    rng = np.random.default_rng(0)
    n_jets, support = 20_000, 6
    P = rng.dirichlet(np.ones(support) * 0.7, size=n_jets)
    cdf = np.cumsum(P, axis=1)
    n_true = (cdf < rng.random((n_jets, 1))).sum(axis=1)      # truth ~ q(N|x): calibrated
    n_post = P @ np.arange(support)
    keep = n_true >= 1
    assert n_post.mean() / n_true.mean() == pytest.approx(1.0, abs=0.02)
    assert n_post[keep].mean() / n_true[keep].mean() < 0.95


# ---------------------------------------------------------------------------
# 4. SBC-on-N against its own null
# ---------------------------------------------------------------------------
def test_midranks_match_a_brute_force_draw():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(5), size=3)
    for i in range(3):
        for n in range(5):
            draws = rng.choice(5, size=200_000, p=P[i])
            brute = np.mean(draws < n) + 0.5 * np.mean(draws == n)
            assert _sbc_midranks(P[i: i + 1], np.array([n]))[0] == pytest.approx(brute, abs=0.005)


def test_selfconsistency_null_accepts_a_calibrated_discrete_posterior(dataset):
    """The point of the whole exercise: a model whose truth is drawn from its own
    q(N|x) must NOT be flagged — even though its chi^2 is far above chi^2(9), because
    a discrete mid-rank cannot be uniform on [0, 1]."""
    m, geom, _ = _built("ar_junipr_v3", seed=4)

    class _Calibrated:
        """`dataset`, with each jet's truth multiplicity redrawn from the model's own
        q(N|x) — calibrated by construction, at exactly this discreteness."""

        def __init__(self, base, model, seed=0):
            import torch.nn.functional as F

            self.items = []
            rng = np.random.default_rng(seed)
            with torch.inference_mode():
                for i in range(len(base)):
                    it = dict(base[i])
                    e = model.encode(it["xf"].unsqueeze(0), torch.tensor([it["nx"]]))
                    p = F.softmax(model.recalibrated_n_logits(model.n_head(e)),
                                  dim=-1).numpy()[0]
                    n = int(rng.choice(p.size, p=p))
                    it["ny"] = n
                    it["yc"] = torch.zeros(n, dtype=torch.long)
                    it["yraw"] = torch.zeros(n, 4)
                    it["yf"] = torch.zeros(n, 5)
                    self.items.append(it)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ds = _Calibrated(dataset, m, seed=3)
    out = sbc_n_selfconsistency_null(m, ds, DEV, n_jets=len(ds), n_reps=100, verbose=False)
    assert out is not None
    assert out["sbc_n_exceeds_null95"] is False, (
        f"a calibrated discrete posterior was flagged: chi2={out['sbc_n_chi2']:.1f} "
        f"vs null p95={out['sbc_n_null_p95']:.1f}"
    )
    # ...and the null is materially wider than the continuous-rank one it replaces
    # (chi^2(9) has mean 9), which is the whole reason it has to be recomputed.
    assert out["sbc_n_null_mean"] > 1.4 * (out["sbc_n_chi2_bins"] - 1)


def test_discreteness_alone_inflates_the_sbc_chi2():
    """The mechanism, in isolation and without a model: when `q(N|x)` concentrates on a
    handful of values — which is what the fielded multiplicity marginal does — the
    mid-rank statistic lands on a handful of atoms, and its chi^2 is an order of
    magnitude above chi^2(9) for a posterior that is calibrated BY CONSTRUCTION.

    This is why v0's "SBC-on-N chi^2 107 vs crit 16.90" measured the reference, not the
    model."""
    from h2p_rsd_junipr.eval.exposure import _chi2_uniform

    rng = np.random.default_rng(0)
    n_jets = 2000
    for support, expect_above in ((3, True), (40, False)):
        P = rng.dirichlet(np.ones(support), size=n_jets)
        cdf = np.cumsum(P, axis=1)
        n_true = (cdf < rng.random((n_jets, 1))).sum(axis=1)      # calibrated by construction
        chi2 = _chi2_uniform(_sbc_midranks(P, n_true), 10)
        assert (chi2 > 5 * 16.92) is expect_above, (
            f"support {support}: chi^2 {chi2:.1f} against the chi^2(9) 95% point 16.92"
        )


def test_selfconsistency_null_is_unavailable_without_an_exact_head(dataset):
    m, _, _ = _built("ar_junipr_v2")
    assert sbc_n_selfconsistency_null(m, dataset, DEV, n_jets=16, n_reps=5, verbose=False) is None


# ---------------------------------------------------------------------------
# 5. the exposure probe
# ---------------------------------------------------------------------------
def test_continue_prob_by_depth_shape_and_family_guard(dataset):
    m, _, _ = _built("ar_junipr_v2", seed=6)
    out = continue_prob_by_depth(m, dataset, DEV, n_jets=32, K=16, max_depth=4, verbose=False)
    assert out is not None and out["by_depth"]
    for r in out["by_depth"]:
        assert 0.0 <= r["p_cont_teacher_forced"] <= 1.0
        assert 0.0 <= r["p_cont_on_policy"] <= 1.0
        assert r["gap"] == pytest.approx(r["p_cont_on_policy"] - r["p_cont_teacher_forced"])
    # a family with an explicit head has no per-step continue decision to read
    nh, _, _ = _built("ar_junipr_v3", seed=6)
    assert continue_prob_by_depth(nh, dataset, DEV, n_jets=8, K=4, verbose=False) is None
