"""Training callbacks (§6): EMA, early stopping, LR monitor. The Trainer keeps
these optional and lightweight; none are required for the synthetic verification.
"""

from __future__ import annotations

import copy

import torch


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad = 0

    def step(self, val: float) -> bool:
        """Return True if training should stop."""
        if val < self.best - self.min_delta:
            self.best = val
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience
