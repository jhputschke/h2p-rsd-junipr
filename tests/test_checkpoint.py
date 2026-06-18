import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.train.checkpoint import (
    load_checkpoint,
    load_for_inference,
    restore_into,
    save_checkpoint,
)
from h2p_rsd_junipr.train.trainer import build_components


def test_save_resume_roundtrip(tmp_path, batch):
    b, geom = batch
    cfg = load_config(["model=ar_junipr_v2"])
    device = torch.device("cpu")
    model, opt, sched = build_components(cfg, geom, device)

    # take a couple of optimiser steps so state is non-trivial
    for _ in range(2):
        opt.zero_grad()
        loss = (-model.log_prob(b)).mean()
        loss.backward()
        opt.step()
    sched.step()

    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=3, step=2, best_val=1.23, cfg=cfg)

    # fresh components, restore, and compare
    model2, opt2, sched2 = build_components(cfg, geom, device)
    state = load_checkpoint(path, map_location=device)
    epoch, step, best = restore_into(model2, opt2, sched2, scaler, state)
    assert (epoch, step, best) == (3, 2, 1.23)

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    # forward parity after restore (eval mode: dropout off, deterministic)
    model.eval()
    model2.eval()
    with torch.inference_mode():
        assert torch.allclose(model.log_prob(b), model2.log_prob(b), atol=1e-6)


def test_load_for_inference_ignores_optimizer(tmp_path, batch):
    b, geom = batch
    cfg = load_config(["model=ar_junipr_v2"])
    model, opt, sched = build_components(cfg, geom, torch.device("cpu"))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=1, step=1, best_val=0.0, cfg=cfg)
    info = load_for_inference(path)
    assert info["model_name"] == "ar_junipr_v2"
    assert "model_state" in info and "config" in info
