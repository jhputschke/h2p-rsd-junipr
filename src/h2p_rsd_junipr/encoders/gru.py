"""Bi-GRU encoder (the v2 script's `encode`): per-node features -> bi-GRU ->
mean-pool over valid positions -> append hadron multiplicity -> project to e(x).

With the default config (emb_dim=32, hidden_dim=64, num_layers=1, bidirectional)
this reproduces the script's encoder exactly. `dropout` is now wired in (applied
in train mode; identity in eval, so it does not affect the likelihood / parity).

`forward_seq` returns the pre-pool per-node GRU states — the tensor `forward`
already computes and then averages away — for a cross-attending decoder (WP3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder, register_encoder


@register_encoder("gru")
class GRUEncoder(Encoder):
    returns_sequence = True

    def __init__(self, cfg, ctx_dim: int, n_node_feat: int):
        super().__init__()
        emb = int(cfg.emb_dim)
        hid = int(cfg.hidden_dim)
        layers = int(cfg.num_layers)
        bidir = bool(cfg.bidirectional)
        self.out_dim = int(ctx_dim)

        self.x_feat = nn.Sequential(nn.Linear(n_node_feat, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.encoder = nn.GRU(
            emb, hid, num_layers=layers, batch_first=True, bidirectional=bidir
        )
        n_dir = 2 if bidir else 1
        self.seq_dim = n_dir * hid
        self.to_ctx = nn.Linear(n_dir * hid + 1, self.out_dim)  # +1: hadron multiplicity
        self.drop = nn.Dropout(float(cfg.dropout))

    def _states(self, xf: torch.Tensor, nx: torch.Tensor):
        """(states (B, Mx, n_dir*hid), mask (B, Mx) float) — shared by both paths, so
        the pooled and sequence views can never diverge.

        `Mx == 0` is short-circuited: `nn.GRU` raises on a zero-length sequence, but a
        jet whose groomed hadron-level tree is EMPTY is physical and common (~7% of the
        PYTHIA sample in `cpp/test_data/jets.root`). Batched training never hit it —
        `collate` pads to the batch maximum — but every per-jet inference path does,
        since there `Mx = nx`. The empty sequence pools to zeros, so `e(x)` is the
        projection's bias: the encoder's "I was told nothing" context."""
        Mx = xf.shape[1]
        mask = (torch.arange(Mx, device=xf.device)[None, :] < nx[:, None]).float()
        if Mx == 0:
            out = xf.new_zeros(xf.shape[0], 0, self.seq_dim)
        else:
            out, _ = self.encoder(self.x_feat(xf))  # (B, Mx, n_dir*hid)
        return out, mask

    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        out, mask = self._states(xf, nx)
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = self.drop(pooled)
        nx_feat = torch.log1p(nx.float()).unsqueeze(-1)
        return self.to_ctx(torch.cat([pooled, nx_feat], dim=-1))  # (B, ctx)

    def forward_seq(self, xf: torch.Tensor, nx: torch.Tensor):
        out, mask = self._states(xf, nx)
        return out, mask.bool()
