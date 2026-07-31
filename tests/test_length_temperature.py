"""Post-hoc recalibration of the multiplicity head (docs/PLAN_empty_parton_tree.md,
second work item).

`q(N|x)` is measurably under-confident about emptiness, and the existing suite does not
flag it: SBC ranks against the sampler's own draws, so a uniformly squashed length belief
still passes. A single scalar temperature on the `n_head` logits, fitted post-hoc on
held-out jets, is the cheapest correction — no retraining, no new weights.

The boundary that matters: it is a DECODE-layer transform. It moves `length_pmf` and the
`N` drawn by `sample`; it must never touch `log_prob` or the `logprob` a point estimate
reports, which are the trained likelihood.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import fit_length_recalibration, recalibrate_pmf
from h2p_rsd_junipr.models.base import build_model

HEAD_FAMILIES = ["ar_junipr_v3", "ar_junipr_v4", "cinn", "diffusion"]


def _model(name, small_jets, **over):
    cfg = load_config([f"model={name}", "data.n_jets=64", *[f"{k}={v}" for k, v in over.items()]])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(torch.device("cpu"))
    model.eval()
    return model, MatchedLundDataset(small_jets[:16], geom), decode_params(cfg)


def _jet(ds, i=0):
    it = ds[i]
    return it["xf"].unsqueeze(0), torch.tensor([it["nx"]])


# --- recalibrate_pmf ------------------------------------------------------------------
def test_recalibrate_pmf_identity_at_one():
    p = np.array([0.1, 0.6, 0.3])
    assert np.array_equal(recalibrate_pmf(p, 1.0), p)   # exact, not approx


def test_recalibrate_pmf_direction_and_normalisation():
    """T>1 flattens toward uniform, T<1 sharpens toward the mode; both stay pmfs."""
    p = np.array([0.1, 0.6, 0.3])
    hot, cold = recalibrate_pmf(p, 5.0), recalibrate_pmf(p, 0.2)
    for q in (hot, cold):
        assert q.sum() == pytest.approx(1.0)
        assert (q >= 0).all()
    assert hot.max() < p.max() and cold.max() > p.max()
    assert np.argmax(cold) == np.argmax(p)          # the mode never moves
    assert abs(hot - 1 / 3).max() < abs(p - 1 / 3).max()


def test_recalibrate_pmf_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="must be > 0"):
        recalibrate_pmf(np.array([0.5, 0.5]), 0.0)


# --- fit_length_recalibration ------------------------------------------------------
def test_fit_recovers_a_planted_temperature():
    """Take a true pmf, squash it by a known T, and check the fit inverts it: NLL of
    draws from the TRUE distribution is minimised at the T that undoes the squash."""
    rng = np.random.default_rng(0)
    true = np.array([0.17, 0.30, 0.25, 0.15, 0.08, 0.05])
    planted = 2.5
    squashed = recalibrate_pmf(true, planted)
    obs = rng.choice(len(true), size=6000, p=true)
    t, _ = fit_length_recalibration([squashed] * len(obs), obs, with_tilt=False)
    assert t == pytest.approx(1.0 / planted, rel=0.15)   # undoes it


def test_fit_returns_the_identity_when_already_calibrated():
    rng = np.random.default_rng(1)
    p = np.array([0.17, 0.30, 0.25, 0.15, 0.08, 0.05])
    obs = rng.choice(len(p), size=6000, p=p)
    t, b = fit_length_recalibration([p] * len(obs), obs)
    assert t == pytest.approx(1.0, rel=0.15) and b == pytest.approx(0.0, abs=0.06)


def test_fit_never_increases_the_nll():
    """Whatever it returns must be at least as good as leaving T=1 — the only property
    worth guaranteeing when the family is misspecified."""
    rng = np.random.default_rng(2)
    p = recalibrate_pmf(np.array([0.05, 0.5, 0.25, 0.12, 0.08]), 3.0)
    obs = rng.choice(5, size=2000, p=np.array([0.05, 0.5, 0.25, 0.12, 0.08]))
    pmfs = [p] * len(obs)

    def nll(t, b=0.0):
        q = recalibrate_pmf(p, t, b)
        return -np.mean([np.log(q[n]) for n in obs])

    assert nll(*fit_length_recalibration(pmfs, obs)) <= nll(1.0) + 1e-9


def test_fit_skips_multiplicities_outside_the_support():
    """A truth past the categorical support must be dropped, not clamped into the last
    bin — clamping is exactly the bias the WP4 support guard exists to catch."""
    p = np.array([0.2, 0.5, 0.3])
    t, b = fit_length_recalibration([p, p, p], np.array([1, 1, 99]))
    assert np.isfinite(t) and t > 0 and np.isfinite(b)
    with pytest.raises(ValueError, match="one pmf per"):
        fit_length_recalibration([p, p], np.array([0, 1, 2]))


# --- the tilt: what a temperature structurally cannot do -------------------------
def test_tilt_moves_mass_along_n_where_a_temperature_cannot():
    """The finding that motivated it. A temperature is symmetric about the mode, so it
    can only pull a non-modal class toward uniform or toward zero — for `ar_junipr_v3`
    the best mean q(0|x) reachable over ALL scalar T is 0.125 against a truth of 0.161.
    A tilt is monotone in n and has no such ceiling."""
    # ar_junipr_v3's measured shape, on its real 26-class support — the class count
    # matters: flattening pulls q(0) toward 1/26 = 0.038, and it is only *because* 0.085
    # sits above that and below the mode that the scalar family is boxed in.
    p = np.full(26, 1e-4)
    p[:6] = [0.085, 0.426, 0.318, 0.136, 0.031, 0.004]
    p /= p.sum()
    best_T = max(np.geomspace(0.1, 20, 200), key=lambda t: recalibrate_pmf(p, t)[0])
    assert recalibrate_pmf(p, best_T)[0] < 0.14          # the scalar family tops out...
    assert recalibrate_pmf(p, 1.0, -0.6)[0] > 0.16       # ...where the tilt clears 0.161
    assert recalibrate_pmf(p, 1.0, -0.6).sum() == pytest.approx(1.0)


def test_fit_recovers_a_planted_tilt():
    rng = np.random.default_rng(3)
    true = np.array([0.17, 0.41, 0.29, 0.10, 0.02, 0.01])
    skewed = recalibrate_pmf(true, 1.0, 0.6)             # push mass toward long trees
    obs = rng.choice(len(true), size=8000, p=true)
    t, b = fit_length_recalibration([skewed] * len(obs), obs)
    assert b == pytest.approx(-0.6, abs=0.12)            # and the fit pushes it back
    assert 0.6 < t < 1.7


def test_tilt_is_a_noop_at_zero_and_reaches_the_model(small_jets):
    model, ds, _ = _model("ar_junipr_v3", small_jets)
    xf, nx = _jet(ds)
    base = model.length_pmf(xf, nx).copy()
    model.length_tilt = 0.0
    assert np.array_equal(model.length_pmf(xf, nx), base)   # exact
    model.length_tilt = -0.4
    tilted = model.length_pmf(xf, nx)
    assert tilted[0] > base[0] and tilted[-1] < base[-1]    # mass moved toward n=0


# --- the model-side boundary -----------------------------------------------------
@pytest.mark.parametrize("family", HEAD_FAMILIES)
def test_temperature_one_is_bit_identical(family, small_jets):
    """Division by 1.0 is exact, so the off path must match to the last bit — not
    merely to a tolerance. This is what keeps every pre-existing number valid."""
    a, ds, _ = _model(family, small_jets)
    b, _, _ = _model(family, small_jets, **{"decode.length_temperature": 1.0})
    b.load_state_dict(a.state_dict())
    xf, nx = _jet(ds)
    assert np.array_equal(a.length_pmf(xf, nx), b.length_pmf(xf, nx))


@pytest.mark.parametrize("family", HEAD_FAMILIES)
def test_temperature_moves_length_pmf_but_not_log_prob(family, small_jets):
    """The whole design boundary in one assertion: recalibration is a decode-layer
    transform, so the trained likelihood must be untouched."""
    model, ds, _ = _model(family, small_jets)
    xf, nx = _jet(ds)
    batch = collate([ds[i] for i in range(4)])

    def logp():
        # `diffusion` sets exact_likelihood=False and its DSM surrogate draws noise, so
        # two identical calls differ by ~1 nat. Seed, or this compares the RNG.
        torch.manual_seed(0)
        with torch.inference_mode():
            return model.log_prob(batch).clone()

    base_pmf, base_lp = model.length_pmf(xf, nx).copy(), logp()
    model.length_temperature = 3.0
    assert not np.allclose(model.length_pmf(xf, nx), base_pmf)   # the belief moved
    assert torch.equal(logp(), base_lp)                          # the likelihood did not


@pytest.mark.parametrize("t", [0.5, 1.0, 3.0])
def test_sample_draws_from_the_tempered_belief(t, small_jets):
    """The coupling the work item needs: the temperature must reach `sample`, not only
    `length_pmf`, and the two must stay the same distribution. Fixing the posterior
    series (the closure notebooks draw 9.7% empty against a truth 17%) is half the point,
    and a pmf-only change would silently leave every draw stale."""
    model, ds, _ = _model("ar_junipr_v3", small_jets)
    xf, nx = _jet(ds)
    model.length_temperature = t
    pmf = model.length_pmf(xf, nx)
    torch.manual_seed(0)
    n = np.array([len(d) for d in model.sample(xf, nx, n=8000)])
    emp = np.bincount(n, minlength=pmf.size)[:pmf.size] / len(n)
    tv = 0.5 * float(np.abs(emp - pmf).sum())
    assert tv < 0.05, f"T={t}: drawn lengths disagree with length_pmf (TV={tv:.3f})"


def test_the_drawn_lengths_actually_move_with_it(small_jets):
    """An UNTRAINED n_head is ~uniform over its 26 classes, so tempering it barely moves
    anything and the test would be about the initialisation. Plant a peaked head — zero
    weights, a sloped bias — so the pmf is a fixed decaying shape and the mechanism is
    what is being measured."""
    model, ds, _ = _model("ar_junipr_v3", small_jets)
    xf, nx = _jet(ds)
    last = model.n_head[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.linspace(4.0, -4.0, last.out_features))

    def drawn(t):
        model.length_temperature = t
        torch.manual_seed(0)
        c = np.bincount([len(d) for d in model.sample(xf, nx, n=4000)], minlength=26)
        return c[:26] / c.sum()

    sharp, flat = drawn(0.4), drawn(4.0)
    assert 0.5 * np.abs(sharp - flat).sum() > 0.2
    assert sharp[0] > flat[0]          # sharpening concentrates on the planted mode


def test_build_model_reads_it_from_the_config_snapshot(small_jets):
    model, _, dec = _model("ar_junipr_v3", small_jets,
                           **{"decode.length_temperature": 2.5})
    assert dec["length_temperature"] == pytest.approx(2.5)
    assert model.length_temperature == pytest.approx(2.5)


def test_it_is_a_flagged_noop_without_a_multiplicity_head(small_jets, capsys):
    """ar_junipr_v2's length belief IS the sampler histogram; tempering one without the
    other would decouple them, so the knob does nothing and must say so."""
    model, ds, _ = _model("ar_junipr_v2", small_jets,
                          **{"decode.length_temperature": 4.0})
    assert "NO-OP" in capsys.readouterr().out
    assert not hasattr(model, "n_head")
    xf, nx = _jet(ds)
    torch.manual_seed(0)
    a = model.length_pmf(xf, nx, mults=[0, 1, 1, 2])
    model.length_temperature = 1.0
    torch.manual_seed(0)
    assert np.array_equal(a, model.length_pmf(xf, nx, mults=[0, 1, 1, 2]))


def test_it_does_not_enter_the_checkpoint(small_jets):
    """A plain float attribute, not a Parameter or Buffer — so a recalibrated model
    still loads strictly into a fresh one and old checkpoints stay valid."""
    model, _, _ = _model("ar_junipr_v3", small_jets)
    model.length_temperature = 7.0
    assert not any("length_temperature" in k for k in model.state_dict())
