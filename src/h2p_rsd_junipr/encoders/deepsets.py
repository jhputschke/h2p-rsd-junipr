"""DeepSets baseline encoder (Zaheer et al., NeurIPS 2017): a per-node MLP phi,
masked mean-pool, then a projection rho -> e(x). Permutation-invariant; the
simplest drop-in alternative to the GRU."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder, register_encoder


@register_encoder("deepsets")
class DeepSetsEncoder(Encoder):
    returns_sequence = True

    def __init__(self, cfg, ctx_dim: int, n_node_feat: int):
        super().__init__()
        emb = int(cfg.emb_dim)
        hid = int(cfg.hidden_dim)
        layers = max(1, int(cfg.num_layers))
        self.out_dim = int(ctx_dim)
        self.seq_dim = hid

        phi = [nn.Linear(n_node_feat, hid), nn.ReLU()]
        for _ in range(layers - 1):
            phi += [nn.Linear(hid, hid), nn.ReLU()]
        self.phi = nn.Sequential(*phi)
        self.drop = nn.Dropout(float(cfg.dropout))
        self.rho = nn.Linear(hid + 1, self.out_dim)  # +1: hadron multiplicity
        self.emb = emb  # kept for symmetry with other encoders

    def _states(self, xf: torch.Tensor, nx: torch.Tensor):
        Mx = xf.shape[1]
        h = self.phi(xf)  # (B, Mx, hid)
        mask = (torch.arange(Mx, device=xf.device)[None, :] < nx[:, None]).float()
        return h, mask

    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        h, mask = self._states(xf, nx)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = self.drop(pooled)
        nx_feat = torch.log1p(nx.float()).unsqueeze(-1)
        return self.rho(torch.cat([pooled, nx_feat], dim=-1))

    def forward_seq(self, xf: torch.Tensor, nx: torch.Tensor):
        """Per-node phi embeddings before the sum. Attention over them is still
        permutation-equivariant in x, so the Deep Sets symmetry survives."""
        h, mask = self._states(xf, nx)
        return h, mask.bool()
