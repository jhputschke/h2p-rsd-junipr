"""Bi-GRU encoder (the v2 script's `encode`): per-node features -> bi-GRU ->
mean-pool over valid positions -> append hadron multiplicity -> project to e(x).

With the default config (emb_dim=32, hidden_dim=64, num_layers=1, bidirectional)
this reproduces the script's encoder exactly. `dropout` is now wired in (applied
in train mode; identity in eval, so it does not affect the likelihood / parity).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder, register_encoder


@register_encoder("gru")
class GRUEncoder(Encoder):
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
        self.to_ctx = nn.Linear(n_dir * hid + 1, self.out_dim)  # +1: hadron multiplicity
        self.drop = nn.Dropout(float(cfg.dropout))

    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        B, Mx, _ = xf.shape
        out, _ = self.encoder(self.x_feat(xf))  # (B, Mx, n_dir*hid)
        mask = (torch.arange(Mx, device=xf.device)[None, :] < nx[:, None]).float()
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = self.drop(pooled)
        nx_feat = torch.log1p(nx.float()).unsqueeze(-1)
        return self.to_ctx(torch.cat([pooled, nx_feat], dim=-1))  # (B, ctx)
