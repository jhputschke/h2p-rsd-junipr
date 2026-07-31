import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from h2p_rsd_junipr.config import CONFIGS, config_hash, decode_params, load_config

PRESETS = CONFIGS.parent / "presets"


def test_defaults_load():
    cfg = load_config([])
    assert cfg.model.name == "ar_junipr_v2"
    assert cfg.encoder.name == "gru"
    assert cfg.geometry.n_bins == 10
    assert cfg.optim.lr == pytest.approx(2e-3)


def test_group_selection_binds_schema():
    cfg = load_config(["model=cinn", "encoder=lundnet"])
    assert cfg.model.name == "cinn"
    assert cfg.model.n_blocks == 6  # CINNConfig field present
    assert cfg.encoder.name == "lundnet"
    assert cfg.encoder.k == 4  # LundNetEncoderConfig field present


def test_dotted_override():
    cfg = load_config(["optim.lr=1e-3", "geometry.n_bins=16", "trainer.max_epochs=5"])
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.geometry.n_bins == 16
    assert cfg.trainer.max_epochs == 5


def test_unknown_key_rejected():
    with pytest.raises((ConfigKeyError, KeyError, Exception)):
        load_config(["optim.lrr=1e-3"])  # typo


def test_bad_type_rejected():
    with pytest.raises((ValidationError, ValueError, Exception)):
        load_config(["geometry.n_bins=ten"])  # not an int


def test_run_name_interpolation():
    cfg = load_config(["model=cinn", "encoder=deepsets"])
    assert cfg.run_name == "cinn_deepsets"


# --- custom top-level config file (`base=<path>`) ---------------------------------
def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_base_file_selects_groups_and_globals(tmp_path):
    """A custom base need only list what it changes; group files still come from configs/."""
    base = _write(tmp_path / "my_config.yaml", """
defaults:
  model: ar_junipr_v3
  trainer: fast_dev
run_root: runs/mbr_study
""")
    cfg = load_config([f"base={base}"])
    assert cfg.model.name == "ar_junipr_v3"
    assert cfg.model.use_multiplicity_head is True   # resolved from configs/model/ar_junipr_v3.yaml
    assert cfg.trainer.fast_dev_run is True
    assert cfg.run_root == "runs/mbr_study"
    assert cfg.encoder.name == "gru"                 # unlisted group inherits the repo default


def test_base_file_dir_shadows_repo_group_files(tmp_path):
    """`<base dir>/<group>/<name>.yaml` wins over configs/, and a new name resolves there."""
    _write(tmp_path / "decode" / "mbr_study.yaml", "point_estimator: mbr\nbeam_width: 16\n")
    _write(tmp_path / "optim" / "default.yaml", "lr: 5.0e-4\n")   # same name as the repo file
    base = _write(tmp_path / "my_config.yaml", "defaults:\n  decode: mbr_study\n")
    cfg = load_config([f"base={base}"])
    assert cfg.decode.point_estimator == "mbr" and cfg.decode.beam_width == 16
    assert cfg.decode.topk_cells == 6                # unspecified keys keep schema defaults
    assert cfg.optim.lr == pytest.approx(5e-4)       # shadowed configs/optim/default.yaml


def test_cli_still_beats_the_base_file(tmp_path):
    base = _write(tmp_path / "my_config.yaml",
                  "defaults:\n  model: ar_junipr_v3\n  trainer: fast_dev\n")
    cfg = load_config([f"base={base}", "model=cinn", "trainer.max_epochs=7"])
    assert cfg.model.name == "cinn"                  # CLI selector overrides the base defaults
    assert cfg.trainer.max_epochs == 7               # dotted override merged last


def test_base_file_inline_group_overrides(tmp_path):
    """Top-level blocks other than `defaults:` are value overrides: a patch on the group
    file, merged after it and before the CLI. Pins the behaviour of the globals merge."""
    base = _write(tmp_path / "my_config.yaml", """
defaults:
  model: ar_junipr_v3
  encoder: lundnet
model:
  dec_dim: 128
optim:
  lr: 1.0e-3
geometry:
  n_bins: 16
""")
    cfg = load_config([f"base={base}"])
    assert cfg.model.dec_dim == 128
    assert cfg.model.use_multiplicity_head is True     # group file still applied underneath
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.optim.weight_decay == pytest.approx(3e-4)  # untouched field keeps the group value
    assert cfg.geometry.n_bins == 16                   # a group not re-selected can be tuned
    assert cfg.encoder.k == 4                          # configs/encoder/lundnet.yaml untouched
    # the CLI is merged last
    assert load_config([f"base={base}", "optim.lr=5e-4"]).optim.lr == pytest.approx(5e-4)


def test_base_file_inline_overrides_are_schema_checked(tmp_path):
    bad_key = _write(tmp_path / "bad_key.yaml", "optim:\n  lrr: 1.0e-3\n")
    wrong_family = _write(tmp_path / "wrong_family.yaml",
                          "defaults:\n  model: cinn\nmodel:\n  dec_dim: 128\n")
    for path in (bad_key, wrong_family):
        with pytest.raises(ConfigKeyError):
            load_config([f"base={path}"])


def test_base_file_cannot_pick_the_model_family_inline(tmp_path):
    """`model.name` is re-set from the selector, so an inline name is a no-op (not a
    half-applied family) — documented in CONFIGURATION.md §0."""
    base = _write(tmp_path / "my_config.yaml", "model:\n  name: cinn\n")
    assert load_config([f"base={base}"]).model.name == "ar_junipr_v2"


def test_shipped_preset_loads():
    """presets/mbr_study.yaml is the template CONFIGURATION.md §0 points at — keep it valid
    (a renamed decode/model field must not leave the documented example broken)."""
    cfg = load_config([f"base={PRESETS / 'mbr_study.yaml'}"])
    assert cfg.model.name == "ar_junipr_v3" and cfg.model.use_multiplicity_head is True
    assert cfg.encoder.name == "lundnet"
    dec = decode_params(cfg)
    assert dec["point_estimator"] == "mbr" and dec["mbr_backend"] == "pot"
    assert dec["mbr_lnkt_cut"] == pytest.approx(cfg.geometry.ln_kt_range[0])  # interpolated
    assert cfg.model.dec_dim == 128            # inline override
    assert cfg.model.max_emissions == 25       # from configs/model/ar_junipr_v3.yaml


def test_missing_base_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config([f"base={tmp_path / 'nope.yaml'}"])


def test_missing_group_file_raises():
    """No silent fallback to the schema defaults on a typo'd selector."""
    with pytest.raises(FileNotFoundError) as e:
        load_config(["decode=mbr_studdy"])
    assert "default" in str(e.value)                 # the message lists what is available


def test_missing_group_file_from_base_raises(tmp_path):
    base = _write(tmp_path / "my_config.yaml", "defaults:\n  trainer: nonexistent\n")
    with pytest.raises(FileNotFoundError):
        load_config([f"base={base}"])


def test_config_hash_stable():
    # NOTE: the hash value changed when min_emissions/length_penalty/cell_label_smoothing
    # were added to the schema; this only asserts determinism + sensitivity, not a fixed value.
    assert config_hash(load_config([])) == config_hash(load_config([]))
    assert config_hash(load_config([])) != config_hash(load_config(["optim.lr=9e-3"]))


def test_decode_config_has_new_fields():
    cfg = load_config([])
    assert cfg.decode.min_emissions == 1
    assert cfg.decode.length_penalty == pytest.approx(0.0)
    assert cfg.decode.length_floor_quantile == pytest.approx(0.0)
    # MBR knobs: point_estimator=map reproduces today; pot is the default backend
    assert cfg.decode.point_estimator == "map"
    assert cfg.decode.mbr_backend == "pot"
    assert cfg.decode.mbr_lnkt_cut is None
    assert cfg.decode.mbr_norm is False


def test_decode_params_full():
    dec = decode_params(load_config([]))
    assert set(dec) == {"beam_width", "topk_cells", "max_emissions", "n_posterior_samples",
                        "cont_temperature", "min_emissions", "length_penalty",
                        "length_floor_quantile", "empty_threshold",
                        "point_estimator", "mbr_backend",
                        "mbr_n_candidates", "mbr_lnkt_cut", "mbr_weight", "mbr_coords",
                        "mbr_R", "mbr_beta", "mbr_norm", "mbr_periodic_phi", "mbr_phi_col",
                        "mbr_resample_to_qn"}
    assert dec["min_emissions"] == 1
    assert dec["length_floor_quantile"] == pytest.approx(0.0)
    assert dec["empty_threshold"] == pytest.approx(0.0)  # the gate is opt-in
    assert dec["point_estimator"] == "map" and dec["mbr_backend"] == "pot"
    assert dec["mbr_lnkt_cut"] is None  # None default preserved through the tolerant read


def test_decode_params_tolerates_old_snapshot():
    """An old checkpoint's decode block lacking the new keys must backfill, not raise."""
    old = OmegaConf.create({"decode": {"beam_width": 8, "topk_cells": 6, "max_emissions": 25,
                                       "n_posterior_samples": 500, "cont_temperature": 1.0}})
    dec = decode_params(old)
    assert dec["min_emissions"] == 1 and dec["length_penalty"] == pytest.approx(0.0)
    assert dec["length_floor_quantile"] == pytest.approx(0.0)
    # the MBR keys backfill too (a snapshot predating them must not raise)
    assert dec["point_estimator"] == "map" and dec["mbr_backend"] == "pot"
    assert dec["mbr_lnkt_cut"] is None and dec["mbr_R"] == pytest.approx(8.485)
    # and a config with no decode block at all
    assert decode_params(OmegaConf.create({}))["min_emissions"] == 1
    assert decode_params(OmegaConf.create({}))["length_floor_quantile"] == pytest.approx(0.0)
    assert decode_params(OmegaConf.create({}))["point_estimator"] == "map"
