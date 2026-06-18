"""Encoder ABC. Any encoder maps the hadron-level primary sequence to a context
vector e(x) of width `out_dim` (== ctx_dim); any encoder pairs with any decoder
family (§3)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

_ENCODER_REGISTRY: dict[str, type[Encoder]] = {}


def register_encoder(name: str):
    def deco(cls):
        _ENCODER_REGISTRY[name] = cls
        return cls

    return deco


class Encoder(nn.Module, ABC):
    out_dim: int

    @abstractmethod
    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """(B, Mx, n_node_feat), (B,) -> (B, out_dim) context e(x)."""
        ...


def build_encoder(enc_cfg, ctx_dim: int, n_node_feat: int) -> Encoder:
    # import for side-effect registration
    from . import deepsets, gru, lundnet  # noqa: F401

    name = enc_cfg.name
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; registered: {sorted(_ENCODER_REGISTRY)}")
    return _ENCODER_REGISTRY[name](enc_cfg, ctx_dim, n_node_feat)
