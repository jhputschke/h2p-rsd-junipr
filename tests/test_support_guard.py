"""The multiplicity-support guard and the A/B grid (docs/PLAN_UPDATES.md WP4).

A categorical `q(N|x)` head has finite support: a truth sequence with
`N > model.max_emissions` is clamped into the last bin, gets the wrong likelihood,
and biases the length marginal — silently, with no signature in the loss curve. The
guard is the cheap check that turns that into a startup error; these tests pin the
thresholds, the family gating (v1/v2 have unbounded support and must NOT fire), and
that the synthetic generator stays inside the shipped default.
"""

from __future__ import annotations

import numpy as np
import pytest

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.stats import (
    SUPPORT_TAIL_ERROR,
    SUPPORT_TAIL_WARN,
    check_multiplicity_support,
    model_support,
    multiplicity_stats,
)


def _jets(lengths, z_cut=0.1, beta=0.0, kt_floor=1.0):
    """Minimal jet dicts: only `y` lengths and the grooming context are read."""
    return [
        {
            "y": (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)),
            "x": (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)),
            "weight": 1.0, "z_cut": z_cut, "beta": beta, "kt_floor": kt_floor,
        }
        for n in lengths
    ]


# --- which families the guard applies to -----------------------------------
@pytest.mark.parametrize(
    "sel,expected",
    [
        (["model=ar_junipr_v1"], None),   # implicit continue/stop: unbounded support
        (["model=ar_junipr_v2"], None),
        (["model=ar_junipr_v3"], 25),     # categorical q(N|x): bounded
        (["model=cinn"], 25),
        (["model=diffusion"], 25),
        (["model=cfm"], 25),
        (["model=ar_junipr_v2", "model.use_multiplicity_head=true"], 25),
        # the edit transducer's length model is the open-ended STOP/EMIT lattice, so its
        # max_emissions is the exact-q(N|x) readout width, not a likelihood support
        (["model=edit_v1"], None),
        (["model=edit_v2"], None),
    ],
)
def test_model_support_is_gated_on_having_a_head(sel, expected):
    assert model_support(load_config(sel)) == expected


def test_the_edit_family_is_never_refused_for_a_long_truth():
    """`model.max_emissions` is inert in the edit family's likelihood — a 100-emission
    truth is merely improbable there, exactly as it is for `ar_junipr_v2`. Firing the guard
    would refuse to train on data the DP handles correctly."""
    cfg = load_config(["model=edit_v1"])
    stats = check_multiplicity_support(_jets([1, 2, 100] * 50), cfg)
    assert stats["support"] is None and stats["tail_fraction"] == 0.0


def test_v2_never_raises_however_long_the_truth():
    """`max_emissions` is inert for the continue/stop model, so a 100-emission truth
    is merely improbable — not mis-normalized. The guard must stay quiet."""
    cfg = load_config(["model=ar_junipr_v2"])
    stats = check_multiplicity_support(_jets([1, 2, 100] * 50), cfg)
    assert stats["support"] is None and stats["tail_fraction"] == 0.0


# --- the thresholds ---------------------------------------------------------
def test_silent_when_the_tail_is_empty():
    cfg = load_config(["model=ar_junipr_v3"])
    stats = check_multiplicity_support(_jets([0, 1, 2, 3, 25] * 100), cfg)
    assert stats["tail_fraction"] == 0.0 and stats["support"] == 25


def test_hard_error_above_1e_3(capsys):
    cfg = load_config(["model=ar_junipr_v3"])
    jets = _jets([2] * 1000 + [30] * 2)          # 2/1002 ~ 2e-3 > 1e-3
    with pytest.raises(ValueError, match="multiplicity support exceeded"):
        check_multiplicity_support(jets, cfg)


def test_error_message_names_the_fix_and_the_grooming():
    cfg = load_config(["model=ar_junipr_v3"])
    jets = _jets([2] * 500 + [40] * 3, z_cut=0.05, beta=1.5, kt_floor=0.5)
    with pytest.raises(ValueError) as exc:
        check_multiplicity_support(jets, cfg)
    msg = str(exc.value)
    assert "model.max_emissions" in msg and ">= 40" in msg      # the actionable bound
    assert "z_cut=0.05" in msg and "beta=1.5" in msg and "kt_floor=0.5" in msg
    assert "grooming" in msg


def test_warning_between_1e_4_and_1e_3(capsys):
    cfg = load_config(["model=ar_junipr_v3"])
    jets = _jets([2] * 5000 + [30])              # 1/5001 = 2e-4, between the thresholds
    stats = check_multiplicity_support(jets, cfg)
    assert SUPPORT_TAIL_WARN < stats["tail_fraction"] <= SUPPORT_TAIL_ERROR
    assert "WARNING" in capsys.readouterr().out


def test_strict_false_downgrades_to_a_warning(capsys):
    """At `eval` the model is already trained: report, do not refuse."""
    cfg = load_config(["model=ar_junipr_v3"])
    jets = _jets([2] * 100 + [30] * 5)
    stats = check_multiplicity_support(jets, cfg, strict=False)
    assert stats["tail_fraction"] > SUPPORT_TAIL_ERROR
    assert "WARNING" in capsys.readouterr().out


def test_raising_max_emissions_clears_the_error():
    jets = _jets([2] * 500 + [40] * 3)
    with pytest.raises(ValueError):
        check_multiplicity_support(jets, load_config(["model=ar_junipr_v3"]))
    ok = check_multiplicity_support(
        jets, load_config(["model=ar_junipr_v3", "model.max_emissions=45"])
    )
    assert ok["tail_fraction"] == 0.0


# --- the shipped defaults stay inside the support ---------------------------
def test_synthetic_generator_fits_the_default_support(small_jets):
    """The default `data.max_emissions=20` must stay under `model.max_emissions=25`,
    or every v3/cINN/CFM run on synthetic data would be quietly mis-normalized."""
    cfg = load_config(["model=ar_junipr_v3"])
    stats = check_multiplicity_support(small_jets, cfg)
    assert stats["tail_fraction"] == 0.0
    assert stats["max"] <= cfg.model.max_emissions


def test_multiplicity_stats_shape(small_jets):
    s = multiplicity_stats(small_jets)
    assert s["n_jets"] == len(small_jets)
    assert s["counts"].sum() == len(small_jets)
    assert 0.0 <= s["frac_empty"] <= 1.0
    assert multiplicity_stats([])["n_jets"] == 0


# --- the A/B decode grid ----------------------------------------------------
def test_ab_decode_grid_prunes_only_duplicates():
    """The pruned cells must be genuinely redundant: floors steer the MAP only, and
    the q(N|x) reweighting steers MBR only."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ab_v2_v3 import cell_label, decode_grid

    grid = decode_grid(with_mbr=True)
    assert all(not c["mbr_resample_to_qn"] for c in grid if c["point_estimator"] == "map")
    mbr = [c for c in grid if c["point_estimator"] == "mbr"]
    assert len(mbr) == 2 and {c["mbr_resample_to_qn"] for c in mbr} == {False, True}
    assert len([c for c in grid if c["point_estimator"] == "map"]) == 4   # 2 floors x 2 alphas
    assert len({cell_label(c) for c in grid}) == len(grid)                # labels unique
    assert all(c["point_estimator"] == "map" for c in decode_grid(with_mbr=False))
