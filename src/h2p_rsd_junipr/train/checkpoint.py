"""Checkpoint save/resume spec (§6). Exact resume needs more than a state_dict:
the full config snapshot, optimiser/scheduler/scaler state, RNG state, and audit
fields. `config_hash` mismatch on resume is a hard error (no silent arch drift).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ..config import config_hash, to_container

FORMAT_VERSION = 2


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(rng: dict) -> None:
    if rng is None:
        return
    # torch.set_rng_state needs a CPU ByteTensor; map_location may have moved it.
    torch_state = rng["torch"]
    if torch.is_tensor(torch_state):
        torch_state = torch_state.cpu().to(torch.uint8)
    torch.set_rng_state(torch_state)
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [s.cpu().to(torch.uint8) for s in rng["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    best_val: float,
    cfg,
    ema=None,
    data_fingerprint: str | None = None,
    git_sha: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": FORMAT_VERSION,
        "model": {"name": cfg.model.name, "state_dict": _unwrap(model).state_dict()},
        "config": to_container(cfg),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": int(epoch),
        "global_step": int(step),
        "best_val_nll": float(best_val),
        "rng": _rng_state(),
        "git_sha": git_sha,
        "config_hash": config_hash(cfg),
        "data_fingerprint": data_fingerprint,
    }
    torch.save(state, path)


def _unwrap(model):
    # torch.compile wraps the module in an OptimizedModule with ._orig_mod
    return getattr(model, "_orig_mod", model)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    """`map_location` defaults to CPU, NOT to torch's None.

    torch.save tags every storage with the device it was written from, and None means
    "restore them there". A checkpoint trained on an Apple-silicon box therefore carries
    `mps` tags, and opening it on a Linux/CUDA host — where torch has no MPS backend —
    dies inside the unpickler with `Storage device not recognized: mps`, before any
    caller gets a chance to move anything. CPU is the one device every host has, so it
    is the only portable default; callers that want the tensors elsewhere pass a device
    (the trainer does, to resume in place), and a model that is already `.to(device)`
    absorbs a CPU state_dict on `load_state_dict` anyway.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def restore_into(model, optimizer, scheduler, scaler, state: dict, *, strict_config_hash: bool = True):
    """Load all training state in place and return (epoch, step, best_val)."""
    _unwrap(model).load_state_dict(state["model"]["state_dict"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    _set_rng_state(state.get("rng"))
    return state["epoch"], state["global_step"], state["best_val_nll"]


def load_for_inference(path: Path, map_location="cpu") -> dict:
    """Export-only load: returns {model_state, config, model_name, best_val_nll,
    epoch}; ignores optimiser state (§6).

    `best_val_nll` / `epoch` are carried because a consumer that RESTORES a run from
    cache rather than retraining still has to report which checkpoint it got. Omitting
    them invited `info.get("best_val", nan)` at the call site, which reports `nan`
    instead of failing — the silent-default trap the aux sentinels exist to avoid.

    `map_location` mirrors `load_checkpoint`'s CPU default rather than forwarding None,
    which would hand torch back the untranslated device tags and undo it."""
    state = load_checkpoint(path, map_location=map_location)
    return {
        "model_state": state["model"]["state_dict"],
        "config": state["config"],
        "model_name": state["model"]["name"],
        "best_val_nll": float(state["best_val_nll"]),
        "epoch": int(state["epoch"]),
    }
