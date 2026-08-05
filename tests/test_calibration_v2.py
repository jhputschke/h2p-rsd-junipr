"""Calibration suite v2 (docs/PLAN_UPDATES.md WP2): the von Mises CDF, per-coordinate
PITs, region stratification, and TARP expected coverage.

The four properties that make the suite trustworthy:
  1. `vonmises_cdf` IS the CDF of `vonmises_logpdf` (to 1e-6 against quadrature);
  2. the PIT of data drawn FROM the model is uniform (the SBC null) — and a
     deliberately over-confident head (sigma x 0.5) breaks it in the documented
     U-shaped direction;
  3. TARP reads Uniform on a self-consistent posterior and drops below the diagonal
     for an over-confident one;
  4. every switch off => the metric dict is bit-for-bit the pre-WP2 dict.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
import torch
from omegaconf.errors import ConfigKeyError

from h2p_rsd_junipr.config import experiment_params, load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.distributions import (
    trunc_normal_cdf,
    trunc_normal_logpdf,
    vonmises_cdf,
    vonmises_logpdf,
    wrap_to_pi,
)
from h2p_rsd_junipr.eval.calibration import (
    REGION_LABELS,
    cell_region,
    chi2_crit95,
    coordinate_pits,
    run_calibration,
    run_tarp,
    wilson_interval,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.ar_junipr import CoordParams
from h2p_rsd_junipr.models.base import build_model

POT_OK = importlib.util.find_spec("ot") is not None


def _ks(u):
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - u, u - (i - 1) / n)))


# ---------------------------------------------------------------------------
# 1. the CDFs are the CDFs of the densities the likelihood uses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.01, 0.5, 2.0, 10.0, 50.0])
def test_vonmises_cdf_matches_quadrature(kappa):
    """F(psi) == integral of the SAME logpdf from the branch cut mu-pi to psi."""
    mu = torch.tensor(0.4, dtype=torch.float64)
    k = torch.tensor(kappa, dtype=torch.float64)
    for psi in (-3.0, -1.0, 0.0, 0.4, 1.7, 3.14):
        p = torch.tensor(psi, dtype=torch.float64)
        t = float(wrap_to_pi(p - mu) + mu)             # the point on the principal branch
        grid = torch.linspace(float(mu) - math.pi, t, 200001, dtype=torch.float64)
        ref = float(torch.trapz(vonmises_logpdf(grid, mu, k).exp(), grid))
        assert float(vonmises_cdf(p, mu, k)) == pytest.approx(ref, abs=1e-6)


def test_vonmises_cdf_endpoints_and_monotone():
    mu, k = torch.tensor(0.0), torch.tensor(3.0)
    psi = torch.linspace(-math.pi + 1e-6, math.pi - 1e-6, 401)
    cdf = vonmises_cdf(psi, mu, k)
    assert float(cdf[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(cdf[-1]) == pytest.approx(1.0, abs=1e-6)
    assert bool((cdf[1:] >= cdf[:-1] - 1e-7).all())


def test_trunc_normal_cdf_matches_quadrature():
    lo, hi = -0.3, 0.3
    mu, sig = torch.tensor(0.05, dtype=torch.float64), torch.tensor(0.2, dtype=torch.float64)
    for x in (-0.29, -0.1, 0.0, 0.12, 0.29):
        grid = torch.linspace(lo, x, 100001, dtype=torch.float64)
        ref = float(torch.trapz(trunc_normal_logpdf(grid, mu, sig, lo, hi).exp(), grid))
        got = float(trunc_normal_cdf(torch.tensor(x, dtype=torch.float64), mu, sig, lo, hi))
        assert got == pytest.approx(ref, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. per-coordinate PIT: the SBC null and the over-confidence signature
# ---------------------------------------------------------------------------
def _self_consistent_batch(model, batch, sigma_scale=1.0, seed=0):
    """Replace the true coordinates by draws FROM the model's own coordinate head
    (optionally with the head's widths scaled), keeping cells/lengths fixed.

    Under `sigma_scale=1` the resulting data is *exactly* a sample from q(coords|...),
    so its PIT must be Uniform(0,1) — the SBC null for the coordinate heads. Scaling
    the widths down makes the model over-confident about data it did not generate."""
    g = torch.Generator().manual_seed(seed)
    yc, ny = batch["yc"], batch["ny"]
    B, L = yc.shape
    e = model.encode(batch["xf"], batch["nx"])
    out = model._decode_states(yc, e)
    eh_t = torch.cat([out[:, :L, :], e.unsqueeze(1).expand(-1, L, -1)], dim=-1)
    du_m, dv_m, du_s, dv_s, lnz_m, lnz_s, mu, kappa = model._coord_params(
        torch.cat([eh_t, model.y_embed(yc.clamp(min=0))], dim=-1)
    )[:8]
    du_s, dv_s, lnz_s = du_s * sigma_scale, dv_s * sigma_scale, lnz_s * sigma_scale

    def _trunc_normal(m, s, lo, hi):
        from h2p_rsd_junipr.distributions import std_normal_cdf

        a, b = std_normal_cdf((lo - m) / s), std_normal_cdf((hi - m) / s)
        p = a + torch.rand(m.shape, generator=g) * (b - a)
        return m + s * math.sqrt(2.0) * torch.erfinv((2.0 * p - 1.0).clamp(-1 + 1e-7, 1 - 1e-7))

    du = _trunc_normal(du_m, du_s, -model.half_u, model.half_u)
    dv = _trunc_normal(dv_m, dv_s, -model.half_v, model.half_v)
    lnz = lnz_m + lnz_s * torch.randn(lnz_m.shape, generator=g)
    psi = torch.distributions.VonMises(mu, kappa / (sigma_scale**2)).sample()
    cx, cy = model.cell_cx[yc], model.cell_cy[yc]
    yraw = torch.stack([cx + du, cy + dv, lnz, psi], dim=-1)
    return {**batch, "yraw": yraw.detach(), "yc": yc, "ny": ny}


class _OneBatchDS:
    """Minimal val_ds view over one collated batch (coordinate_pits re-collates)."""

    def __init__(self, batch):
        self.b = batch
        self.n = batch["yc"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ny = int(self.b["ny"][i])
        return {"xf": self.b["xf"][i, : int(self.b["nx"][i])], "nx": int(self.b["nx"][i]),
                "yf": self.b["yf"][i, :ny], "yc": self.b["yc"][i, :ny],
                "yraw": self.b["yraw"][i, :ny], "ny": ny, "w": self.b["w"][i]}


def _pit_values(model, batch):
    out = model.coordinate_cdfs(batch)
    u, mask = out["u"].numpy(), out["mask"].numpy()
    return {n: u[..., d][mask] for d, n in enumerate(out["names"])}


def test_coordinate_pit_of_self_generated_data_is_uniform(batch):
    """The SBC null: PIT of data drawn from the model itself is Uniform(0,1)."""
    b, geom = batch
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    big = collate([_OneBatchDS(b)[i] for i in range(b["yc"].shape[0])] * 24)  # ~1k emissions
    with torch.inference_mode():
        synth = _self_consistent_batch(model, big, sigma_scale=1.0, seed=3)
        pits = _pit_values(model, synth)
    for name, vals in pits.items():
        crit = 1.9 * 1.36 / math.sqrt(len(vals))   # ~1e-3 false-positive rate
        assert _ks(vals) < crit, f"{name}: KS={_ks(vals):.4f} >= {crit:.4f}"
        assert vals.mean() == pytest.approx(0.5, abs=0.06)


def _pin_coords(model, *, sig=0.05, kappa=20.0, scale=1.0):
    """Pin the coordinate head to fixed widths, scaled by `scale`.

    A random-init head predicts sigma ~ 0.7 against a box of half-width 0.3, so its
    truncated normal is already ~uniform on the cell and rescaling it changes almost
    nothing. Pinning a width well inside the box is what makes the over-confidence
    signature measurable at all — `scale=0.5` is then a genuinely too-narrow model."""
    def _params(coord_in):
        shape = coord_in.shape[:-1]
        z = torch.zeros(shape)
        s = torch.full(shape, float(sig) * float(scale))
        return CoordParams(z, z, s, s, z, s, z,
                           torch.full(shape, float(kappa) / float(scale) ** 2))

    model._coord_params = _params


def test_coordinate_pit_flags_an_overconfident_head(batch):
    """Documented signature: a head with HALF the generating width sees data outside
    its body, so its PIT piles up at 0 and 1 (U-shaped) and the KS distance explodes."""
    b, geom = batch
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    big = collate([_OneBatchDS(b)[i] for i in range(b["yc"].shape[0])] * 24)
    with torch.inference_mode():
        _pin_coords(model, scale=1.0)                 # generate at the true width
        data = _self_consistent_batch(model, big, seed=5)
        honest = _pit_values(model, data)             # judged by the generating widths
        _pin_coords(model, scale=0.5)                 # ...now judge with HALF of it
        narrow = _pit_values(model, data)
    for name in ("du", "dv", "ln_z", "psi"):
        crit = 1.36 / math.sqrt(len(honest[name]))
        assert _ks(honest[name]) < 2.5 * crit, name       # measured: <= 0.053 @ crit 0.040
        vals = narrow[name]
        assert _ks(vals) > 3.5 * crit, name               # measured: >= 0.174
        tails = float(np.mean((vals < 0.05) | (vals > 0.95)))
        assert tails > 0.25, f"{name}: U-shape not visible (tail mass {tails:.3f}, uniform 0.10)"


def test_coordinate_pit_report_shape_and_v1_optout(batch):
    b, geom = batch
    ds = MatchedLundDataset.__new__(MatchedLundDataset)
    ds.geometry, ds.items = geom, [_OneBatchDS(b)[i] for i in range(b["yc"].shape[0])]
    v2 = build_model(load_config(["model=ar_junipr_v2"]), geom).eval()
    rep = coordinate_pits(v2, ds, geom, torch.device("cpu"), n_jets=8,
                          stratify_regions=True, verbose=False)
    assert rep["space"] == "physical" and rep["names"] == ["du", "dv", "ln_z", "psi"]
    assert set(rep["coords"]["du"]) >= {"ks", "chi2", "mean", "hist", "by_emission_index"}
    assert 0.0 <= rep["ks_max"] <= 1.0
    # v1 has no coordinate density and must opt out rather than fake one
    v1 = build_model(load_config(["model=ar_junipr_v1"]), geom).eval()
    assert v1.coordinate_cdfs(b) is None
    assert coordinate_pits(v1, ds, geom, torch.device("cpu"), n_jets=8, verbose=False) is None


def test_cinn_reports_latent_space_pit(batch):
    b, geom = batch
    cinn = build_model(load_config(["model=cinn", "encoder=gru"]), geom).eval()
    out = cinn.coordinate_cdfs(b)
    assert out["space"] == "latent" and list(out["names"]) == ["z0", "z1", "z2", "z3"]
    u = out["u"][out["mask"]]
    assert bool(((u >= 0.0) & (u <= 1.0)).all())


def test_diffusion_opts_out_of_pit_and_flags_surrogate(batch):
    b, geom = batch
    dif = build_model(load_config(["model=diffusion", "encoder=gru"]), geom).eval()
    assert dif.exact_likelihood is False      # the WP1 honesty flag
    assert dif.coordinate_cdfs(b) is None     # no exact density => no PIT


# ---------------------------------------------------------------------------
# 3. region stratification
# ---------------------------------------------------------------------------
def test_cell_region_partitions_the_plane():
    geom = Geometry()
    labels = {cell_region(c, geom) for c in range(geom.n_cells)}
    assert labels == set(REGION_LABELS)
    assert cell_region(None, geom) is None
    assert cell_region(0, geom) == "wide_soft"                    # low u, low v
    assert cell_region(geom.n_cells - 1, geom) == "narrow_hard"   # high u, high v


# ---------------------------------------------------------------------------
# 4. TARP
# ---------------------------------------------------------------------------
class _StubDS:
    """val_ds of cell chains with a controllable truth distribution."""

    def __init__(self, truths):
        self.truths = truths

    def __len__(self):
        return len(self.truths)

    def __getitem__(self, i):
        return {"xf": torch.zeros(2, 5), "nx": 2,
                "yc": torch.tensor(self.truths[i], dtype=torch.long),
                "ny": len(self.truths[i]), "yraw": torch.zeros(len(self.truths[i]), 4)}


class _StubModel:
    """`sample_batch` is all `run_tarp` needs. `spread` controls the posterior width.

    Emissions are jittered along the ANGLE bin only, at a fixed `ln k_t` bin: the
    default Lund cloud weights points by `exp(ln k_t)`, so moving in k_t would make the
    EMD a weight-mismatch measure and blunt the geometric signal this test is about."""

    KT_BIN = 5

    def __init__(self, centers, spread, seed=0):
        self.centers, self.spread = centers, spread
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def _jitter(self, base, spread, rng):
        return [int(np.clip(ix + rng.integers(-spread, spread + 1), 0, 9)) * 10 + self.KT_BIN
                for ix in base]

    def sample_batch(self, xf, nx, K, **kw):
        base = self.centers[self.calls % len(self.centers)]
        self.calls += 1
        return [self._jitter(base, self.spread, self.rng) for _ in range(K)]


def _tarp_case(truth_spread, post_spread, n_jets=200, seed=0):
    """Centers uniform over the angle axis; truth and posterior jittered around them
    with independently controllable widths."""
    rng = np.random.default_rng(seed)
    centers = [[int(c) for c in rng.integers(0, 10, size=3)] for _ in range(n_jets)]
    model = _StubModel(centers, post_spread, seed=seed + 1)
    truths = [model._jitter(ctr, truth_spread, rng) for ctr in centers]
    return model, _StubDS(truths)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
@pytest.mark.parametrize("spread", [2, 4])
def test_tarp_tracks_the_diagonal_for_a_self_consistent_posterior(spread):
    """Truth and draws from the SAME conditional => ECP(alpha) = alpha (the null)."""
    model, ds = _tarp_case(spread, spread)
    t = run_tarp(model, ds, Geometry(), torch.device("cpu"), K=40, n_jets=200,
                 n_refs=100, seed=0, verbose=False)
    assert t["tarp_max_dev"] < 0.12
    assert abs(t["ecp_at"]["0.68"] - 0.68) < 0.10
    assert abs(t["ecp_at"]["0.90"] - 0.90) < 0.10
    assert len(t["alpha"]) == len(t["ecp"])


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_tarp_detects_an_overconfident_posterior():
    """A posterior far narrower than the truth spread leaves the truth outside its
    credible regions: ECP falls BELOW the diagonal (the documented under-coverage)."""
    model, ds = _tarp_case(truth_spread=4, post_spread=1)
    t = run_tarp(model, ds, Geometry(), torch.device("cpu"), K=40, n_jets=200,
                 n_refs=100, seed=0, verbose=False)
    assert t["tarp_max_dev"] > 0.15
    assert t["ecp_at"]["0.68"] < 0.68 - 0.08     # under-covers at 68% credibility
    assert t["ecp_at"]["0.90"] < 0.90 - 0.15     # ...and worse at 90%
    assert t["tarp_signed_bias"] < 0.0


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_tarp_detects_an_overdispersed_posterior():
    """The mirror failure: a posterior much broader than the truth spread over-covers,
    so ECP sits ABOVE the diagonal."""
    model, ds = _tarp_case(truth_spread=1, post_spread=4)
    t = run_tarp(model, ds, Geometry(), torch.device("cpu"), K=40, n_jets=200,
                 n_refs=100, seed=0, verbose=False)
    assert t["ecp_at"]["0.68"] > 0.68 + 0.05
    assert t["tarp_signed_bias"] > 0.0


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_tarp_prior_reference_pool_is_supported():
    model, ds = _tarp_case(2, 2, n_jets=80)
    t = run_tarp(model, ds, Geometry(), torch.device("cpu"), K=20, n_jets=80,
                 n_refs=40, reference="prior", seed=0, verbose=False)
    assert t["reference"] == "prior" and t["n_refs"] == 40
    assert np.isfinite(t["tarp_max_dev"])


# ---------------------------------------------------------------------------
# 4b. the uncertainty the coverage numbers are quoted with (check 12)
# ---------------------------------------------------------------------------
def test_wilson_interval_brackets_and_survives_the_edges():
    lo, hi = wilson_interval(68, 100)
    assert lo < 0.68 < hi
    assert 0.57 < lo < 0.60 and 0.76 < hi < 0.78          # textbook [0.582, 0.768]
    # The edges are where the normal approximation gives a zero-width interval and
    # the ends of a near-empty Lund quadrant land.
    for k, n in ((0, 40), (40, 40)):
        a, b = wilson_interval(k, n)
        assert 0.0 <= a < b <= 1.0, "a degenerate count must still get a real interval"
    assert all(np.isnan(x) for x in wilson_interval(0, 0))  # no jets, no claim
    # width shrinks like 1/sqrt(n): 40 jets cannot resolve what 400 can
    w40 = np.subtract(*reversed(wilson_interval(27, 40)))
    w400 = np.subtract(*reversed(wilson_interval(272, 400)))
    assert w40 > 2.5 * w400


def test_chi2_crit95_matches_the_table():
    # the reference `sbc_chi2_uniform` is quoted against, at the default 10 rank bins
    assert chi2_crit95(9) == pytest.approx(16.919, abs=0.05)
    assert chi2_crit95(1) == pytest.approx(3.841, abs=0.1)
    assert chi2_crit95(20) == pytest.approx(31.410, abs=0.05)
    assert math.isnan(chi2_crit95(0))
    assert chi2_crit95(30) > chi2_crit95(9)                # monotone in dof


def test_region_strata_carry_counts_and_a_scoring_floor(batch, small_jets):
    b, geom = batch
    ds = MatchedLundDataset(small_jets[:64], geom)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    torch.manual_seed(3)
    m = run_calibration(model, ds, geom, torch.device("cpu"), K=16, n_jets=40,
                        verbose=False, stratify_regions=True, min_region_n=1000)
    assert m["region_min_n"] == 1000
    assert set(m["region_split"]) == {"u", "v"}
    for label, e in m["by_region"].items():
        assert label in REGION_LABELS
        assert e["scored"] is False, "no region has 1000 jets here; none may be scored"
        lo, hi = e["coverage_68_ci"]
        if e["n_coverage"]:
            assert lo <= e["coverage_68"] <= hi
        else:
            assert math.isnan(lo) and math.isnan(hi)


def test_tarp_quotes_its_null_floor(small_jets):
    geom = Geometry()
    ds = MatchedLundDataset(small_jets[:32], geom)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    t = run_tarp(model, ds, geom, torch.device("cpu"), K=20, n_jets=25, n_refs=20,
                 seed=0, verbose=False)
    # a sup-norm CDF deviation is a KS statistic; its 95% null value is 1.36/sqrt(n)
    assert t["tarp_null_floor95"] == pytest.approx(1.36 / math.sqrt(25))
    assert t["tarp_exceeds_null"] == (t["tarp_max_dev"] > t["tarp_null_floor95"])


# ---------------------------------------------------------------------------
# 5. the all-off path is the pre-WP2 path
# ---------------------------------------------------------------------------
def test_all_switches_off_reproduces_the_v1_metric_dict(batch, small_jets):
    b, geom = batch
    ds = MatchedLundDataset(small_jets[:24], geom)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    dev = torch.device("cpu")
    torch.manual_seed(7)
    base = run_calibration(model, ds, geom, dev, K=16, n_jets=12, verbose=False)
    torch.manual_seed(7)
    again = run_calibration(model, ds, geom, dev, K=16, n_jets=12, verbose=False,
                            pit_coords=False, stratify_regions=False, tarp=False)
    assert base == again
    # The v1 point estimates, unchanged. The uncertainty keys beside them
    # (docs/PLAN_prod_test_v0.md check 12) are annotations on these same numbers —
    # a Wilson interval and the chi^2 reference point — computed from what was already
    # there, so they cost no extra RNG draws and move no published value.
    v1 = {"sbc_chi2_uniform", "sbc_rank_mean", "pit_mean", "coverage_68", "n_jets"}
    uncertainty = {"sbc_chi2_dof", "sbc_chi2_crit95", "sbc_chi2_exceeds_crit95",
                   "n_coverage", "coverage_68_ci", "coverage_68_consistent"}
    assert set(base) == v1 | uncertainty
    lo, hi = base["coverage_68_ci"]
    assert lo <= base["coverage_68"] <= hi
    assert base["sbc_chi2_crit95"] == pytest.approx(16.92, abs=0.05)   # chi^2(9), 95%
    assert base["sbc_chi2_exceeds_crit95"] == (
        base["sbc_chi2_uniform"] > base["sbc_chi2_crit95"]
    )
    # switching WP2 on only ADDS keys and leaves the shared ones untouched
    torch.manual_seed(7)
    grown = run_calibration(model, ds, geom, dev, K=16, n_jets=12, verbose=False,
                            pit_coords=True, stratify_regions=True)
    assert all(grown[k] == base[k] for k in base)
    assert "pit_coords" in grown and "by_region" in grown
    assert set(grown["by_region"]) <= set(REGION_LABELS)


def test_experiment_params_defaults_and_backfill():
    cfg = load_config(["model=ar_junipr_v3"])
    exp = experiment_params(cfg)
    assert exp["pit_coords"] is False and exp["stratify_regions"] is False
    assert exp["tarp"] is False and exp["tarp_refs"] == 100
    assert exp["tarp_reference"] == "pooled"
    # a pre-WP2 snapshot (no new keys at all) backfills instead of raising
    from omegaconf import OmegaConf

    old = OmegaConf.create({"experiment": {"name": "closure", "closure_jets": 7}})
    got = experiment_params(old)
    assert got["closure_jets"] == 7 and got["tarp"] is False


def test_experiment_switches_are_schema_checked():
    cfg = load_config(["experiment.tarp=true", "experiment.tarp_refs=32"])
    assert cfg.experiment.tarp is True and cfg.experiment.tarp_refs == 32
    with pytest.raises(ConfigKeyError):     # unknown key rejected at load, as designed
        load_config(["experiment.tarpp=true"])


# ---------------------------------------------------------------------------
# WP-D of docs/PLAN_prod_test_v1.md: support audit, TARP power, region x coordinate PITs
# ---------------------------------------------------------------------------
def _v1_model_ds(small_jets, n=40, extra=None):
    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model

    cfg = load_config(["model=ar_junipr_v2", "encoder=gru", *(extra or [])])
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(0)
    model = build_model(cfg, geom).eval()
    jets = small_jets[:n]
    return model, MatchedLundDataset(jets, geom), jets, geom


def test_support_audit_counts_every_boundary():
    """`violations` on hand-placed points, one per boundary, so a miscounted column
    cannot hide behind a rate that happens to look plausible."""
    from h2p_rsd_junipr.eval.support import violations
    from h2p_rsd_junipr.geometry import Geometry

    geom = Geometry()                       # (0, 6)^2
    lnz_ok, lnz_lo, lnz_hi = math.log(0.3), math.log(0.05), math.log(0.8)
    pts = [
        [3.0, 3.0, lnz_ok, 0.0],            # clean
        [7.0, 3.0, lnz_ok, 0.0],            # u above the window
        [3.0, -1.0, lnz_ok, 0.0],           # v below the window == below the kt floor
        [3.0, 3.0, lnz_lo, 0.0],            # below soft drop
        [3.0, 3.0, lnz_hi, 0.0],            # z > 1/2
    ]
    v = violations(pts, geom, z_cut=0.1, beta=0.0)
    assert v == {"n": 5, "out_of_window": 2, "kt_floor": 1,
                 "soft_drop": 1, "z_above_half": 1, "n_at_boundary": 0}
    # an unknown grooming record must read "unknown", never "zero"
    u = violations(pts, geom, z_cut=float("nan"), beta=float("nan"))
    assert u["soft_drop"] == -1 and u["z_above_half"] == -1


def test_support_audit_target_is_a_hard_zero(small_jets):
    from h2p_rsd_junipr.eval.support import run_support_audit

    model, ds, jets, geom = _v1_model_ds(small_jets, n=8)
    for j in jets:                            # synthetic jets carry no grooming record
        j["z_cut"], j["beta"], j["kt_floor"] = 0.1, 0.0, 1.0
    out = run_support_audit(model, ds, jets, geom, torch.device("cpu"),
                            n_jets=8, K=4, verbose=False)
    assert out["z_cut"] == 0.1 and out["beta"] == 0.0
    assert out["posterior"] is not None and out["truth"]["n_emissions"] > 0
    # `passes` is an equality against zero, not a tolerance
    assert out["posterior"]["passes"] == (out["posterior"]["max_rate"] == 0.0)
    assert set(out["posterior"]) >= {"out_of_window", "soft_drop", "z_above_half",
                                     "kt_floor", "max_rate", "passes"}


def test_physical_lnz_head_passes_the_lnz_columns(small_jets):
    """The WP-A/WP-D.1 join: a truncated `ln z` head cannot produce either violation,
    so the audit's two `ln z` columns are exactly zero by construction."""
    from h2p_rsd_junipr.eval.support import run_support_audit

    model, ds, jets, geom = _v1_model_ds(small_jets, n=8,
                                         extra=["model.lnz_support=physical"])
    for j in jets:
        j["z_cut"], j["beta"], j["kt_floor"] = 0.1, 0.0, 1.0
    out = run_support_audit(model, ds, jets, geom, torch.device("cpu"),
                            n_jets=8, K=8, verbose=False)
    p = out["posterior"]
    assert p["soft_drop"] == 0.0 and p["z_above_half"] == 0.0
    assert out["lnz_support"] == "physical"


def test_tarp_null_band_is_recomputed_at_the_runs_own_size():
    from h2p_rsd_junipr.eval.calibration import tarp_null_band

    small = tarp_null_band(300, n_reps=4000, seed=0)
    large = tarp_null_band(2000, n_reps=4000, seed=0)
    assert small["p95"] > large["p95"], "the null band must tighten with n"
    # the plan's precondition for quoting the statistic at all
    assert small["floor_ok"] is False and large["floor_ok"] is True
    # ...and the MC band is below the asymptotic 1.36/sqrt(n) at these sizes, because
    # the deviation is evaluated on a finite alpha grid
    for b in (small, large):
        assert b["p95"] < b["analytic_floor95"]
        assert b["mean"] < b["p95"] < b["p99"]


def test_tarp_null_band_calibrates_its_own_false_positive_rate():
    """A 95% point must reject 5% of calibrated runs — no more. Checked by simulation,
    because the whole purpose of replacing 1.36/sqrt(n) is that its rate was wrong."""
    import numpy as np

    from h2p_rsd_junipr.eval.calibration import tarp_null_band

    n_jets, n_alpha = 300, 21
    band = tarp_null_band(n_jets, n_alpha=n_alpha, n_reps=6000, seed=0)
    rng = np.random.default_rng(99)
    alpha = np.linspace(0.0, 1.0, n_alpha)
    f = rng.random((3000, n_jets))
    dev = np.abs((f[:, None, :] < alpha[None, :, None]).mean(axis=2) - alpha).max(axis=1)
    rate = float(np.mean(dev > band["p95"]))
    assert 0.03 < rate < 0.07, f"false-positive rate {rate:.3f}, expected ~0.05"


def test_run_tarp_reports_the_band_and_the_quotability_verdict(small_jets):
    from h2p_rsd_junipr.eval.calibration import run_tarp

    model, ds, _, geom = _v1_model_ds(small_jets, n=12)
    t = run_tarp(model, ds, geom, torch.device("cpu"), K=6, n_jets=12, n_refs=8,
                 null_reps=400, stratify=True, min_region_n=4, verbose=False)
    assert "null_band" in t and t["null_band"]["n_jets"] == 12
    assert t["tarp_passes_g7"] == (t["null_band"]["floor_ok"]
                                   and t["tarp_max_dev"] <= t["null_band"]["p95"])
    # 12 jets can never make the floor: the gate must SAY so rather than pass quietly
    assert t["tarp_quotable"] is False
    if "by_region" in t:
        for e in t["by_region"].values():
            assert {"tarp_max_dev", "n_jets", "scored"} <= set(e)


def test_region_by_coordinate_pit_cross_is_surfaced(small_jets):
    """The cross already existed nested inside `pit_coords`; gate G5 reads it, so it is
    flattened to a scannable table with the worst SCORED cell named."""
    from h2p_rsd_junipr.eval.calibration import run_calibration

    model, ds, _, geom = _v1_model_ds(small_jets, n=40)
    m = run_calibration(model, ds, geom, torch.device("cpu"), K=8, n_jets=40,
                        pit_coords=True, stratify_regions=True, min_region_n=5,
                        verbose=False)
    cross = m["pit_coords_by_region"]
    assert set(cross) <= {"du", "dv", "ln_z", "psi"} and cross
    for v in cross.values():
        for r, e in v.items():
            assert r in REGION_LABELS
            assert 0.0 <= e["ks"] <= 1.0 and e["scored"] == (e["n"] >= 5)
    worst = m["pit_coords_by_region_worst"]
    if worst["coord"] is not None:
        assert worst["ks"] == cross[worst["coord"]][worst["region"]]["ks"]
        assert cross[worst["coord"]][worst["region"]]["scored"]


def test_new_switches_are_all_off_by_default(small_jets):
    """Every WP-D addition is opt-in: with the switches at their defaults the metric
    dict gains no key, so published tables stay stable."""
    from h2p_rsd_junipr.config import experiment_params, load_config
    from h2p_rsd_junipr.eval.calibration import run_calibration

    exp = experiment_params(load_config([]))
    assert exp["support_audit"] is False and exp["tarp_stratify"] is False
    assert exp["tarp_null_reps"] == 0 and exp["exposure_diagnostic"] is False
    model, ds, _, geom = _v1_model_ds(small_jets, n=8)
    m = run_calibration(model, ds, geom, torch.device("cpu"), K=4, n_jets=8, verbose=False)
    for k in ("pit_coords_by_region", "pit_coords_by_region_worst", "tarp", "by_region"):
        assert k not in m


def test_psi_resultant_carries_its_uniform_floor(small_jets):
    """`|R|` is a norm: it is positive under uniformity too, and its floor moves as
    `1/sqrt(n)`. Gate G6 compares two rows pooled over very different node counts, so
    without each row's own floor the ratio is a ratio of noise."""
    from h2p_rsd_junipr.eval.closure import run_closure

    model, ds, jets, geom = _v1_model_ds(small_jets, n=24)
    m = run_closure(model, ds, jets, geom, torch.device("cpu"), K=16, n_closure=24,
                    verbose=False, continuous=True)
    p = m["psi"]
    for key in ("truth", "point_estimate", "posterior"):
        n = p[f"n_nodes_{key}"]
        if not n:
            continue
        assert p[f"resultant_null_{key}"] == pytest.approx(
            math.sqrt(math.pi) / (2 * math.sqrt(n)), rel=1e-9
        ), key
        assert 0.0 <= p[f"rayleigh_p_{key}"] <= 1.0
    # the null must fall as 1/sqrt(n): the posterior row pools far more nodes than truth
    if p["n_nodes_posterior"] > p["n_nodes_truth"] > 0:
        assert p["resultant_null_posterior"] < p["resultant_null_truth"]


def test_rayleigh_p_flags_a_genuinely_anisotropic_sample():
    """The statistic itself, on samples whose answer is known: uniform angles give a
    large p, concentrated ones a vanishing p."""
    import numpy as np

    rng = np.random.default_rng(0)
    for n in (200, 2000):
        uni = rng.uniform(-math.pi, math.pi, n)
        R = abs(np.exp(1j * uni).sum()) / n
        assert math.exp(-n * R**2) > 0.01, "a uniform sample was flagged as anisotropic"
        conc = rng.normal(0.4, 0.5, n)          # a real preferred direction
        Rc = abs(np.exp(1j * conc).sum()) / n
        assert math.exp(-n * Rc**2) < 1e-6
        assert Rc > math.sqrt(math.pi) / (2 * math.sqrt(n))


def test_boundary_draws_are_not_counted_as_violations():
    """A truncated sampler CLAMPS to its bound, and the bound is only representable to
    float32. A strict comparison then counts a draw sitting exactly ON the soft-drop cut
    as a crossing of it — 8 in 575 525 on the first trained physical arm, all of them
    arithmetic. They are reported as `n_at_boundary` instead, and the violation columns
    use the same `EDGE_TOL` the training-time guard already used.

    The tolerance masks nothing: an unbounded head misses the boundary by O(0.1)."""
    import numpy as np

    from h2p_rsd_junipr.eval.support import EDGE_TOL, violations
    from h2p_rsd_junipr.geometry import Geometry

    geom = Geometry()
    lo, hi = math.log(0.1), math.log(0.5)
    pts = [
        [3.0, 3.0, float(np.float32(lo)), 0.0],    # on the bound after a float32 round trip
        [3.0, 3.0, float(np.float32(hi)), 0.0],    # ditto, upper
        [3.0, 3.0, lo - 0.5, 0.0],                 # genuinely below
        [3.0, 3.0, -0.2, 0.0],                     # genuinely above (z = 0.82)
    ]
    v = violations(pts, geom, z_cut=0.1, beta=0.0)
    assert v["soft_drop"] == 1 and v["z_above_half"] == 1, "a real violation was masked"
    assert v["n_at_boundary"] == 2, "boundary draws must be counted, just not as violations"
    # a leak is orders of magnitude larger than the tolerance, so the tolerance is inert
    leak = violations([[3.0, 3.0, lo - 10 * EDGE_TOL, 0.0]], geom, z_cut=0.1, beta=0.0)
    assert leak["soft_drop"] == 1


def test_edge_tolerance_matches_the_training_time_guard():
    """Two audits of the same boundary must not disagree about which side of it a point
    lies on. `data.stats.check_lnz_support` uses 1e-6; so does the eval-time audit."""
    import inspect

    from h2p_rsd_junipr.data import stats
    from h2p_rsd_junipr.eval.support import EDGE_TOL

    src = inspect.getsource(stats.check_lnz_support)
    assert f"lo - {EDGE_TOL:g}" in src or "lo - 1e-6" in src
    assert EDGE_TOL == 1e-6


# ---------------------------------------------------------------------------
# `coverage_68`'s own null (docs/PLAN_StratifiedMBR.md WP4)
#
# The leading-cell HPD-68 is built from the K DRAWS, so it cannot contain a cell of
# probability < 1/K that a calibrated truth still visits. The statistic therefore reads
# BELOW 0.68 even for a perfect model, and "0.53 against 0.68" cannot be called
# over-confidence until that loss is measured. v1 already caught the same shape of trap in
# SBC-on-N (a mid-rank statistic on a 7-valued discrete N against a continuous chi^2(9)).
# ---------------------------------------------------------------------------
def _hpd68(counts_by_cell):
    """The production HPD-68 construction, in isolation, for a hand-checkable case."""
    vals, counts = np.unique(np.asarray(counts_by_cell), return_counts=True)
    order = np.argsort(-counts)
    cum = np.cumsum(counts[order]) / counts.sum()
    k68 = int(np.searchsorted(cum, 0.68)) + 1
    return set(int(c) for c in vals[order][:k68])


def test_the_hpd_from_k_draws_undercovers_a_perfect_model():
    """The mechanism, on a categorical where the model IS the truth.

    A long-tailed distribution has mass spread over cells no finite sample reaches, so the
    empirical HPD misses them. That is the deficit the null exists to quantify — it is a
    property of the estimator, and no model change removes it."""
    rng = np.random.default_rng(0)
    p = np.ones(200) / 200.0            # flat over 200 cells: the HPD needs 136 of them
    K = 40                              # ...and 40 draws cannot supply that many
    hits = []
    for _ in range(400):
        draws = rng.choice(len(p), size=K, p=p)
        hpd = _hpd68(draws)
        truth = int(rng.choice(len(p), p=p))       # the truth IS a draw from the model
        hits.append(truth in hpd)
    covered = float(np.mean(hits))
    assert covered < 0.5, (
        f"a perfect model should still under-cover at K={K} over 200 cells; got {covered:.3f}"
    )


def test_coverage_null_is_reported_and_default_off(batch):
    """Opt-in and additive: `coverage_null_reps=0` adds no keys (the `tarp` convention),
    and > 0 adds the null with its own Wilson interval and the verdict line."""
    from h2p_rsd_junipr.config import experiment_params, load_config
    from h2p_rsd_junipr.eval.calibration import run_calibration
    from h2p_rsd_junipr.models.base import build_model

    assert experiment_params(load_config([]))["coverage_null_reps"] == 0

    b, geom = batch
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset

    ds = MatchedLundDataset(synthetic_matched_dataset(24, seed=0), geom)
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    dev = torch.device("cpu")

    # Seeded identically before EACH call: two sequential calls otherwise start from
    # different RNG states and would differ for a reason that has nothing to do with the
    # switch. This is the actual additivity claim -- same seed in, same pre-existing keys
    # out -- and it holds only because the null's extra draws are taken inside
    # `fork_rng`, so they cannot shift the stream the NEXT jet samples from.
    torch.manual_seed(0)
    off = run_calibration(model, ds, geom, dev, K=16, n_jets=8, verbose=False)
    assert "coverage_68_null" not in off, "off must add no keys"

    torch.manual_seed(0)
    on = run_calibration(model, ds, geom, dev, K=16, n_jets=8, verbose=False,
                         coverage_null_reps=5)
    assert on["coverage_68"] == off["coverage_68"], "the real statistic must not move"
    assert on["sbc_chi2_uniform"] == off["sbc_chi2_uniform"]
    for k in ("coverage_68_null", "coverage_68_null_ci", "n_coverage_null",
              "coverage_68_vs_null", "coverage_68_null_explains_deficit"):
        assert k in on
    assert 0.0 <= on["coverage_68_null"] <= 1.0
    assert "cannot contain a cell of probability < 1/K" in on["coverage_68_null_note"]
