import pytest
from omegaconf.errors import ConfigKeyError, ValidationError

from h2p_rsd_junipr.config import config_hash, load_config


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


def test_config_hash_stable():
    assert config_hash(load_config([])) == config_hash(load_config([]))
    assert config_hash(load_config([])) != config_hash(load_config(["optim.lr=9e-3"]))
