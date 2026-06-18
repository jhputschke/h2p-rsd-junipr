"""Integration smoke train (§11): fast_dev_run for each mandatory model on
synthetic data — loss finite and a posterior draw valid."""

import math

import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.datamodule import LundDataModule
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.train.logging import CSVJSONLLogger
from h2p_rsd_junipr.train.trainer import Trainer, build_components

MODELS = ["ar_junipr_v1", "ar_junipr_v2", "cinn"]


@pytest.mark.parametrize("model_name", MODELS)
def test_fast_dev_train(model_name, tmp_path):
    cfg = load_config(
        [f"model={model_name}", "trainer=fast_dev", "data.n_jets=128", "data.min_val=16"]
    )
    device = torch.device("cpu")
    geom = Geometry.from_config(cfg.geometry)
    dm = LundDataModule(cfg, geom).setup()
    model, opt, sched = build_components(cfg, geom, device)
    logger = CSVJSONLLogger(tmp_path)
    trainer = Trainer(model, opt, sched, dm.loaders(), cfg, logger, device, tmp_path)
    best = trainer.fit()
    logger.close()
    assert math.isfinite(best)

    # a posterior draw is valid
    _, val_ds = dm.datasets()
    xf = val_ds[0]["xf"].unsqueeze(0)
    nx = torch.tensor([val_ds[0]["nx"]])
    draws = model.sample(xf, nx, 4)
    assert len(draws) == 4
    assert all(all(0 <= c < geom.n_cells for c in d) for d in draws)
    assert (tmp_path / "last.ckpt").exists()


def test_loss_decreases_over_two_steps():
    cfg = load_config(["model=ar_junipr_v2", "data.n_jets=256", "data.min_val=16"])
    device = torch.device("cpu")
    geom = Geometry.from_config(cfg.geometry)
    dm = LundDataModule(cfg, geom).setup()
    model, opt, sched = build_components(cfg, geom, device)
    train_loader, _ = dm.loaders()
    losses = []
    model.train()
    for i, batch in enumerate(train_loader):
        opt.zero_grad()
        loss = (-model.log_prob(batch)).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i >= 20:
            break
    assert losses[-1] < losses[0]  # NLL goes down
