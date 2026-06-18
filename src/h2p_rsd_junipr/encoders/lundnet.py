"""LundNet-style encoder (Dreyer & Qu, JHEP 03 (2021) 052, arXiv:2012.08526).

The full LundNet uses EdgeConv over the Lund tree with torch-geometric (the
optional `[lundnet]` extra). To keep the package importable without that heavy
dependency, this provides a dependency-free EdgeConv over the *primary chain*
(the Lund caterpillar's consecutive-node edges): for each node, an edge MLP on
[h_i, h_{i+1}-h_i] is max/mean-aggregated, stacked `num_layers` times, then
masked-mean-pooled and projected to e(x). If torch-geometric is installed a
proper kNN EdgeConv could be swapped in behind the same `forward` signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder, register_encoder


class _ChainEdgeConv(nn.Module):
    """EdgeConv on the chain graph i -> i+1 (and self): h'_i = max_j MLP([h_i, h_j - h_i])."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim), nn.ReLU()
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # neighbours: self and the next node along the primary chain (shifted).
        h_next = torch.cat([h[:, 1:, :], h[:, -1:, :]], dim=1)  # last node self-loops
        e_self = self.mlp(torch.cat([h, torch.zeros_like(h)], dim=-1))
        e_next = self.mlp(torch.cat([h, h_next - h], dim=-1))
        out = torch.maximum(e_self, e_next)
        return out * mask.unsqueeze(-1)


@register_encoder("lundnet")
class LundNetEncoder(Encoder):
    def __init__(self, cfg, ctx_dim: int, n_node_feat: int):
        super().__init__()
        hid = int(cfg.hidden_dim)
        layers = max(1, int(cfg.num_layers))
        self.out_dim = int(ctx_dim)

        self.embed = nn.Linear(n_node_feat, hid)
        self.blocks = nn.ModuleList([_ChainEdgeConv(hid, hid) for _ in range(layers)])
        self.drop = nn.Dropout(float(cfg.dropout))
        self.to_ctx = nn.Linear(hid + 1, self.out_dim)

    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        B, Mx, _ = xf.shape
        mask = (torch.arange(Mx, device=xf.device)[None, :] < nx[:, None]).float()
        h = torch.relu(self.embed(xf)) * mask.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, mask)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = self.drop(pooled)
        nx_feat = torch.log1p(nx.float()).unsqueeze(-1)
        return self.to_ctx(torch.cat([pooled, nx_feat], dim=-1))
