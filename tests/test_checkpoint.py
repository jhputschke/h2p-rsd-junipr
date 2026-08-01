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


def test_edit_transducer_round_trips(tmp_path, batch):
    """The fourth family through the same save/restore path. Worth its own case: its
    parameters include the physics-width scalars `(sigma_0, Lambda_eff)`, which are the
    fitted quantity the whole family is meant to report — a checkpoint that dropped them
    would still load and still train."""
    b, geom = batch
    cfg = load_config(["model=edit_v1", "encoder=gru"])
    device = torch.device("cpu")
    model, opt, sched = build_components(cfg, geom, device)
    for _ in range(2):
        opt.zero_grad()
        (-model.log_prob(b)).mean().backward()
        opt.step()

    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "edit.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=2, step=2, best_val=9.0, cfg=cfg)

    model2, opt2, sched2 = build_components(cfg, geom, device)
    restore_into(model2, opt2, sched2, scaler, load_checkpoint(path, map_location=device))
    model.eval()
    model2.eval()
    with torch.inference_mode():
        assert torch.allclose(model.log_prob(b), model2.log_prob(b), atol=1e-6)
    assert model2.physics_width_params() == model.physics_width_params()
    assert load_for_inference(path)["model_name"] == "edit_v1"


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


def test_load_for_inference_carries_the_training_summary(tmp_path, batch):
    """A consumer that RESTORES a run from cache instead of retraining still has to
    report which checkpoint it got. These keys were missing, so `info.get("best_val",
    nan)` at the call site reported `nan` rather than failing — a silent NaN in an A/B
    results table (notebooks/aux_input_ab.ipynb)."""
    b, geom = batch
    cfg = load_config(["model=ar_junipr_v2"])
    model, opt, sched = build_components(cfg, geom, torch.device("cpu"))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=7, step=42, best_val=4.5901, cfg=cfg)
    info = load_for_inference(path)
    assert abs(info["best_val_nll"] - 4.5901) < 1e-9
    assert info["epoch"] == 7
    # the key the checkpoint actually stores; reading "best_val" was the bug
    assert "best_val" not in info


def test_old_snapshot_backward_compat(tmp_path, batch):
    """A checkpoint whose config predates min_emissions/length_penalty/cell_label_smoothing
    must still rebuild, read decode params tolerantly, and round-trip its own config_hash."""
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import config_hash, decode_params
    from h2p_rsd_junipr.models.base import build_model

    b, geom = batch
    cfg = load_config(["model=ar_junipr_v2"])
    model, opt, sched = build_components(cfg, geom, torch.device("cpu"))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=1, step=1, best_val=0.0, cfg=cfg)

    # simulate an OLD snapshot by dropping the newly added schema fields (incl. the
    # whole MBR block, which a pre-MBR checkpoint would not have)
    snap = OmegaConf.create(load_for_inference(path)["config"])
    OmegaConf.set_struct(snap, False)
    del snap.decode["min_emissions"]
    del snap.decode["length_penalty"]
    del snap.model["cell_label_smoothing"]
    for k in ("point_estimator", "mbr_backend", "mbr_lnkt_cut", "mbr_R", "mbr_norm"):
        snap.decode.pop(k, None)

    # rebuild + tolerant decode read must not raise on the trimmed config
    m2 = build_model(snap, geom)
    m2.load_state_dict(load_for_inference(path)["model_state"])
    assert m2.cell_label_smoothing == 0.0           # getattr default kicked in
    dec = decode_params(snap)
    assert dec["min_emissions"] == 1                # backfilled, no struct-mode crash
    assert dec["point_estimator"] == "map"          # MBR block backfills to the map default
    assert dec["mbr_backend"] == "pot" and dec["mbr_lnkt_cut"] is None
    assert config_hash(snap) == config_hash(snap)     # deterministic round-trip
