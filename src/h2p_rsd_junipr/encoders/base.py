"""Encoder ABC. Any encoder maps the hadron-level primary sequence to a context
vector e(x) of width `out_dim` (== ctx_dim); any encoder pairs with any decoder
family (§3).

Encoders may ADDITIONALLY expose their per-node states before pooling, via
`forward_seq` / `returns_sequence` (docs/PLAN_UPDATES.md WP3). Pooling every
hadron-level node into one `ctx_dim` vector is the classic fixed-length bottleneck:
the parton-level decoder sees the whole hadron sequence only through that vector,
and LundNet's graph structure is likewise flattened before the decoder ever looks
at it. A decoder that can cross-attend (`ARJuniprConfig.use_cross_attention`) reads
the states instead. `forward` is untouched by all of this, so the pooled path — and
therefore likelihood parity — is unaffected."""

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
    # Per-node states available through `forward_seq`? A capability flag, not a new
    # ABC: `build_model` checks it before wiring a cross-attending decoder.
    returns_sequence: bool = False
    # Width of those states (set by implementations; often != out_dim, since the
    # states are pre-projection). The decoder projects them to its own width.
    seq_dim: int = 0

    @abstractmethod
    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """(B, Mx, n_node_feat), (B,) -> (B, out_dim) context e(x)."""
        ...

    def forward_seq(self, xf: torch.Tensor, nx: torch.Tensor):
        """(B, Mx, n_node_feat), (B,) -> ((B, Mx, seq_dim), (B, Mx) bool mask).

        The per-node states BEFORE pooling, with `True` at valid positions. Only
        valid when `returns_sequence` is True."""
        raise NotImplementedError(
            f"{type(self).__name__} has no per-node states (returns_sequence=False); "
            "implement forward_seq to use it with decoder cross-attention."
        )


def build_encoder(enc_cfg, ctx_dim: int, n_node_feat: int) -> Encoder:
    # import for side-effect registration
    from . import deepsets, gru, lundnet  # noqa: F401

    name = enc_cfg.name
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; registered: {sorted(_ENCODER_REGISTRY)}")
    return _ENCODER_REGISTRY[name](enc_cfg, ctx_dim, n_node_feat)
