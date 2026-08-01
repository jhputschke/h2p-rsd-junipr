"""What `eval` may lift over a checkpoint snapshot, and what it must not.

`eval` follows the checkpoint for everything by default — that is what makes a metrics
file reproducible. Two groups are inference-time choices and are liftable over it:

  * `data`   — WHICH jets to report on. Without this there is no held-out TEST set at
               all: the datamodule only ever produces train/val, so every eval reported
               on the same jets model selection used.
  * `decode` — HOW the posterior becomes a point estimate.

Both accept the full composition surface (`group=name`, a `base=` preset's `defaults:`
or inline block, dotted `group.field=value`), which is the part that used to be silently
inert: only dotted `decode.*` tokens were read, so a decode preset changed nothing and
said nothing.

`geometry` and `encoder` stay pinned to the checkpoint — they set tensor widths and the
model contract, so lifting them would describe a different model, not a re-run.
"""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from h2p_rsd_junipr.cli import _lift_onto_snapshot, cmd_eval
from h2p_rsd_junipr.config import explicit_group_keys, load_config
from h2p_rsd_junipr.data.datamodule import LundDataModule
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.train.logging import CSVJSONLLogger
from h2p_rsd_junipr.train.trainer import Trainer, build_components

# keep the suite fast: 3 jets x 8 draws is enough to exercise every code path
EVAL_BUDGET = ["experiment.closure_jets=3", "experiment.n_closure_samples=8"]


# ---------------------------------------------------------------------------
# explicit_group_keys — "did this invocation NAME the field?"
# ---------------------------------------------------------------------------
def test_unnamed_group_is_empty():
    """The distinction the whole mechanism rests on: a fully populated composed config
    is not evidence that anyone asked for those values."""
    assert explicit_group_keys(["model=cinn", "optim.lr=1e-3"], "decode") == set()
    assert explicit_group_keys([], "data") == set()


def test_dotted_tokens_are_explicit():
    keys = explicit_group_keys(["decode.beam_width=16", "decode.point_estimator=mbr"], "decode")
    assert keys == {"beam_width", "point_estimator"}
    assert explicit_group_keys(["data.path=x.root", "trainer.seed=3"], "data") == {"path"}


def test_group_selector_names_every_field_of_the_file():
    keys = explicit_group_keys(["decode=default"], "decode")
    assert {"beam_width", "min_emissions", "point_estimator", "mbr_backend"} <= keys
    assert "lr" not in keys  # a group file may only carry its own group's fields


def test_base_preset_defaults_and_inline_block(tmp_path):
    """A `base=` file contributes through both of its channels, and its own directory is
    searched for group files first — the shadowing rule load_config documents."""
    (tmp_path / "decode").mkdir()
    (tmp_path / "decode" / "study.yaml").write_text("point_estimator: mbr\nmbr_backend: pot\n")
    preset = tmp_path / "study.yaml"
    preset.write_text(
        "defaults:\n  decode: study\n"      # resolves from tmp_path/decode/, not configs/
        "data:\n  n_jets: 64\n"             # inline block -> an explicit `data` field
    )
    assert explicit_group_keys([f"base={preset}"], "decode") == {"point_estimator", "mbr_backend"}
    assert explicit_group_keys([f"base={preset}"], "data") == {"n_jets"}


def test_cli_selector_overrides_preset_selector(tmp_path):
    preset = tmp_path / "p.yaml"
    preset.write_text("defaults:\n  decode: default\n")
    keys = explicit_group_keys([f"base={preset}", "decode.beam_width=4"], "decode")
    assert "beam_width" in keys and "point_estimator" in keys


# ---------------------------------------------------------------------------
# _lift_onto_snapshot — only named fields move, and the move is reported
# ---------------------------------------------------------------------------
def test_lift_moves_only_named_fields_and_reports_them():
    snapshot = OmegaConf.create({"decode": {"beam_width": 8, "point_estimator": "map"}})
    cfg = load_config(["decode.beam_width=16"])
    applied = _lift_onto_snapshot(snapshot, cfg, ["decode.beam_width=16"], "decode")
    assert applied == {"beam_width": (8, 16)}
    assert snapshot.decode.beam_width == 16
    assert snapshot.decode.point_estimator == "map"   # unnamed -> untouched


def test_lift_backfills_a_field_the_snapshot_predates():
    """Old checkpoints have no `point_estimator` key at all; naming it must add it rather
    than raise, the same tolerance decode_params provides on the read side."""
    snapshot = OmegaConf.create({"decode": {"beam_width": 8}})
    argv = ["decode.point_estimator=mbr"]
    applied = _lift_onto_snapshot(snapshot, load_config(argv), argv, "decode")
    assert applied == {"point_estimator": (None, "mbr")}
    assert snapshot.decode.point_estimator == "mbr"


def test_lift_is_a_no_op_when_the_value_already_matches():
    snapshot = OmegaConf.create({"decode": {"beam_width": 8}})
    argv = ["decode.beam_width=8"]
    assert _lift_onto_snapshot(snapshot, load_config(argv), argv, "decode") == {}


# ---------------------------------------------------------------------------
# end to end through cmd_eval
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_ckpt(tmp_path_factory):
    """One fast_dev checkpoint on synthetic data, reused by every eval below."""
    run_dir = tmp_path_factory.mktemp("run")
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru", "trainer=fast_dev",
                       "data.n_jets=128", "data.min_val=16"])
    geom = Geometry.from_config(cfg.geometry)
    dm = LundDataModule(cfg, geom).setup()
    model, opt, sched = build_components(cfg, geom, torch.device("cpu"))
    logger = CSVJSONLLogger(run_dir)
    Trainer(model, opt, sched, dm.loaders(), cfg, logger, torch.device("cpu"), run_dir,
            dm.fingerprint).fit()
    logger.close()
    return run_dir / "best.ckpt"


def _metrics(ckpt):
    return json.loads((ckpt.parent / "eval_metrics.json").read_text())


def test_plain_eval_follows_the_checkpoint(trained_ckpt):
    assert cmd_eval([str(trained_ckpt), *EVAL_BUDGET]) == 0
    m = _metrics(trained_ckpt)
    assert m["data"]["scope"] == "val_split"        # unchanged behaviour: the training val split
    assert m["data"]["source"] == "synthetic"
    assert m["data"]["overrides"] == {} and m["decode_overrides"] == {}
    assert m["decode"]["point_estimator"] == "map"  # the checkpoint's decode, not a CLI default
    assert m["model"] == "ar_junipr_v2"


def test_data_override_swaps_the_sample_and_uses_every_jet(trained_ckpt):
    """G1. A synthetic-trained checkpoint reported on a different sample entirely — the
    mechanism a held-out test file needs. The named sample is a TEST set, so it is
    evaluated whole rather than re-split 90/10."""
    before = _metrics(trained_ckpt)["data"]["fingerprint"]
    argv = [str(trained_ckpt), "data.n_jets=64", "data.seed=7", *EVAL_BUDGET]
    assert cmd_eval(argv) == 0
    m = _metrics(trained_ckpt)
    assert m["data"]["scope"] == "all"
    assert m["data"]["n_eval_jets"] == 64           # every jet, not the 16-jet val split
    assert m["data"]["fingerprint"] != before       # and the metrics file says which jets
    assert m["data"]["overrides"] == {"n_jets": 64, "seed": 7}


def test_decode_preset_binds_and_the_cli_still_wins(trained_ckpt, tmp_path):
    """G2. A decode group file selected through a `base=` preset used to be silently
    inert at eval; now it binds, and a dotted CLI token still outranks it."""
    (tmp_path / "decode").mkdir()
    (tmp_path / "decode" / "study.yaml").write_text("beam_width: 16\nmin_emissions: 3\n")
    preset = tmp_path / "study.yaml"
    preset.write_text("defaults:\n  decode: study\n")

    assert cmd_eval([str(trained_ckpt), f"base={preset}", *EVAL_BUDGET]) == 0
    m = _metrics(trained_ckpt)
    assert m["decode"]["beam_width"] == 16 and m["decode"]["min_emissions"] == 3
    assert m["decode_overrides"] == {"beam_width": 16, "min_emissions": 3}

    assert cmd_eval([str(trained_ckpt), f"base={preset}", "decode.min_emissions=1",
                     *EVAL_BUDGET]) == 0
    m = _metrics(trained_ckpt)
    assert m["decode"]["min_emissions"] == 1        # CLI last, as in load_config
    assert m["decode"]["beam_width"] == 16          # the rest of the preset still applies


def test_geometry_is_not_liftable(trained_ckpt):
    """The negative case that keeps the mechanism honest: a CLI geometry must not reach
    the model. n_bins changes the cell count, so a lifted one would silently reinterpret
    every cell id the checkpoint was trained on."""
    assert cmd_eval([str(trained_ckpt), "geometry.n_bins=16", *EVAL_BUDGET]) == 0
    cfg_ckpt = OmegaConf.create(
        torch.load(trained_ckpt, map_location="cpu", weights_only=False)["config"]
    )
    assert cfg_ckpt.geometry.n_bins == 10           # checkpoint untouched
    m = _metrics(trained_ckpt)
    assert m["data"]["overrides"] == {}             # and nothing was lifted on its behalf
