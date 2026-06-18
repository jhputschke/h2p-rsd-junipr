"""The training engine (§6): a lean, framework-free loop driving any
`PosteriorModel` through its `log_prob`. No Lightning — the §5.1 model is
overhead-bound, so a ~150-line loop owns masking, weighted NLL, scheduling,
checkpointing, and logging more transparently.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ..config import config_hash
from ..geometry import Geometry
from ..models.base import build_model
from .checkpoint import load_checkpoint, restore_into, save_checkpoint


# ---------------------------------------------------------------------------
# Device + determinism + builders
# ---------------------------------------------------------------------------
def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)


def build_scheduler(cfg, optimizer):
    if cfg.optim.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.trainer.max_epochs, eta_min=cfg.optim.eta_min
        )
    if cfg.optim.scheduler in (None, "none"):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    raise ValueError(f"unknown scheduler {cfg.optim.scheduler!r}")


def build_components(cfg, geometry: Geometry, device: torch.device):
    model = build_model(cfg, geometry).to(device)
    opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt)
    return model, opt, sched


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """~150-line training loop for any PosteriorModel. Owns the epoch loop,
    optional AMP/compile, grad clipping, scheduling, checkpointing, and
    lightweight logging."""

    def __init__(self, model, optimizer, scheduler, loaders, cfg, logger, device, run_dir,
                 data_fingerprint=None):
        self.model = model.to(device)
        self.opt, self.sched = optimizer, scheduler
        self.train_loader, self.val_loader = loaders
        self.cfg, self.log, self.device, self.run_dir = cfg, logger, device, Path(run_dir)
        self.data_fingerprint = data_fingerprint
        self.scaler = torch.amp.GradScaler(device.type, enabled=cfg.trainer.amp)
        self.epoch, self.step, self.best_val = 0, 0, float("inf")
        if cfg.trainer.compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")

    # -- main loop -----------------------------------------------------------
    def fit(self):
        # range() is evaluated once up front, so resuming from self.epoch is correct.
        for self.epoch in range(self.epoch, self.cfg.trainer.max_epochs):  # noqa: B020
            train_nll = self._train_epoch()
            val_nll = self._validate()
            self.sched.step()
            self.log.log(
                self.step,
                {"epoch": self.epoch + 1, "train_nll": train_nll, "val_nll": val_nll,
                 "lr": self.sched.get_last_lr()[0]},
            )
            print(
                f"epoch {self.epoch + 1:2d}   train NLL/jet = {train_nll:8.3f}   "
                f"val NLL/jet = {val_nll:8.3f}"
            )
            self.save("last.ckpt")
            if val_nll < self.best_val:
                self.best_val = val_nll
                self.save("best.ckpt")
            if self.cfg.trainer.fast_dev_run:
                break
        return self.best_val

    def _move(self, b):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def _train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            batch = self._move(batch)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.cfg.trainer.amp):
                nll = -self.model.log_prob(batch)  # (B,)
                loss = (batch["w"] * nll).sum() / batch["w"].sum().clamp(min=1e-8)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.step += 1
            total += loss.item()
            n += 1
            if self.cfg.trainer.fast_dev_run and self.step >= 2:
                break
        return total / max(n, 1)

    @torch.inference_mode()
    def _validate(self):
        self.model.eval()
        num = den = 0.0
        for batch in self.val_loader:
            batch = self._move(batch)
            nll = -self.model.log_prob(batch)
            num += (batch["w"] * nll).sum().item()
            den += batch["w"].sum().item()
            if self.cfg.trainer.fast_dev_run:
                break
        return num / max(den, 1e-8)

    # -- checkpoint ----------------------------------------------------------
    def save(self, name):
        save_checkpoint(
            self.run_dir / name,
            model=self.model, optimizer=self.opt, scheduler=self.sched, scaler=self.scaler,
            epoch=self.epoch + 1, step=self.step, best_val=self.best_val, cfg=self.cfg,
            data_fingerprint=self.data_fingerprint,
        )

    def _restore(self, state):
        self.epoch, self.step, self.best_val = restore_into(
            self.model, self.opt, self.sched, self.scaler, state
        )

    @classmethod
    def resume(cls, path, geometry, loaders, logger, device, run_dir, data_fingerprint=None):
        """Rebuild model/opt/sched from the snapshotted config, restore all state
        (incl. RNG, epoch, step, best_val), then continue. config_hash mismatch
        is a hard error."""
        from ..config import OmegaConf  # local import to avoid top-level dependency cycles
        state = load_checkpoint(path, map_location=device)
        cfg = OmegaConf.create(state["config"])
        if config_hash(cfg) != state.get("config_hash"):
            raise RuntimeError(
                "config_hash mismatch on resume: refusing silent architecture drift."
            )
        model, opt, sched = build_components(cfg, geometry, device)
        trainer = cls(model, opt, sched, loaders, cfg, logger, device, run_dir, data_fingerprint)
        trainer._restore(state)
        return trainer
