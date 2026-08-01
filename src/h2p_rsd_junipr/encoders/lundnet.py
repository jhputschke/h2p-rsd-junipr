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

    def __init__(self, in_dim: int, out_dim: int, mask_padding: bool = True):
        super().__init__()
        self.mask_padding = bool(mask_padding)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim), nn.ReLU()
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Neighbours: self and the next node along the primary chain. The last node of
        # EACH JET must self-loop; `h[:, -1:]` is the last row of the PADDED tensor, so
        # under batching a jet's final real node read its neighbour out of the padding
        # and its edge feature became `0 - h = -h`. For the ~40% of jets with nx = 1 that
        # is the only node, so the whole context depended on what else was in the batch.
        # `mask_padding=False` reproduces that, and exists only for `verify_parity.py`:
        # the original v2 script has the same defect and parity is measured against it.
        if self.mask_padding:
            L = h.shape[1]
            idx = torch.arange(L, device=h.device)
            last = (mask.sum(1).long() - 1).clamp(min=0)             # last REAL node per jet
            nxt = torch.minimum(idx.unsqueeze(0) + 1, last.unsqueeze(1))
            h_next = torch.gather(h, 1, nxt.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        else:
            h_next = torch.cat([h[:, 1:, :], h[:, -1:, :]], dim=1)
        e_self = self.mlp(torch.cat([h, torch.zeros_like(h)], dim=-1))
        e_next = self.mlp(torch.cat([h, h_next - h], dim=-1))
        out = torch.maximum(e_self, e_next)
        return out * mask.unsqueeze(-1)


@register_encoder("lundnet")
class LundNetEncoder(Encoder):
    returns_sequence = True

    def __init__(self, cfg, ctx_dim: int, n_node_feat: int):
        super().__init__()
        hid = int(cfg.hidden_dim)
        layers = max(1, int(cfg.num_layers))
        self.out_dim = int(ctx_dim)
        self.seq_dim = hid

        # Default FALSE, unlike the schema's True: this fallback is only reached by a
        # checkpoint snapshot written before the field existed, and such a run was
        # trained with the defect. Evaluating it masked would put it in a regime it
        # never saw. New configs always carry the key, and it is True.
        mp = bool(getattr(cfg, "mask_padding", False))
        self.embed = nn.Linear(n_node_feat, hid)
        self.blocks = nn.ModuleList([_ChainEdgeConv(hid, hid, mp) for _ in range(layers)])
        self.drop = nn.Dropout(float(cfg.dropout))
        self.to_ctx = nn.Linear(hid + 1, self.out_dim)

    def _states(self, xf: torch.Tensor, nx: torch.Tensor):
        """Per-node EdgeConv states after the last block, before the readout — the
        graph structure that pooling would otherwise flatten away."""
        Mx = xf.shape[1]
        mask = (torch.arange(Mx, device=xf.device)[None, :] < nx[:, None]).float()
        h = torch.relu(self.embed(xf)) * mask.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, mask)
        return h, mask

    def forward(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        h, mask = self._states(xf, nx)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = self.drop(pooled)
        nx_feat = torch.log1p(nx.float()).unsqueeze(-1)
        return self.to_ctx(torch.cat([pooled, nx_feat], dim=-1))

    def forward_seq(self, xf: torch.Tensor, nx: torch.Tensor):
        h, mask = self._states(xf, nx)
        return h, mask.bool()
