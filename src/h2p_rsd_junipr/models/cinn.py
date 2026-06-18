"""§5.2 conditional normalizing-flow posterior (cINN; Bellagente et al., SciPost
Phys. 9 (2020) 074, arXiv:2006.06685).

This is a contract-complete *baseline* realization of the §5.2 family: a drop-in
with the same `log_prob`/`sample`/`map_estimate` interface, exact log-likelihood,
and a multiplicity head — not a full rewrite of the autoregressive model. It
factorises q_phi(y|x) into

    P(n | e) * prod_i P(cell_i | e) * prod_i p_flow(coords_i | e, cell_i)

with `P(n|e)` a categorical multiplicity head, `P(cell|e)` a categorical cell
head, and `p_flow` a conditional RealNVP over the 4 continuous coordinates (exact
change-of-variables). The cell/coord factors are order-independent (the cINN
contrast to the autoregressive backbone). Phase 5 of the roadmap replaces this
with the full structured-latent flow; the point here is the registry drop-in.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.base import build_encoder
from ..features import N_NODE_FEAT
from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate
from .base import PosteriorModel, register_model

_LOG_2PI = math.log(2.0 * math.pi)


class _CondRealNVP(nn.Module):
    """Conditional RealNVP over `dim` continuous coords, conditioned on a context
    vector. Alternating-mask affine coupling; exact log-density."""

    def __init__(self, dim: int, ctx_dim: int, hidden: int, n_blocks: int):
        super().__init__()
        self.dim = dim
        self.masks = nn.ParameterList(
            [nn.Parameter(self._mask(i, dim), requires_grad=False) for i in range(n_blocks)]
        )
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim + ctx_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, 2 * dim),
                )
                for _ in range(n_blocks)
            ]
        )

    @staticmethod
    def _mask(i: int, dim: int) -> torch.Tensor:
        m = torch.zeros(dim)
        m[(i % 2) :: 2] = 1.0  # alternate which coords are passed through
        return m

    def _st(self, net, x_masked, ctx):
        h = net(torch.cat([x_masked, ctx], dim=-1))
        s, t = h.chunk(2, dim=-1)
        s = torch.tanh(s)  # stabilise the scale
        return s, t

    def log_prob(self, x, ctx):
        z = x
        logdet = torch.zeros(x.shape[:-1], device=x.device)
        for mask, net in zip(self.masks, self.nets):
            xm = z * mask
            s, t = self._st(net, xm, ctx)
            s = s * (1 - mask)
            t = t * (1 - mask)
            z = xm + (1 - mask) * (z * torch.exp(s) + t)
            logdet = logdet + s.sum(-1)
        base = (-0.5 * (z**2) - 0.5 * _LOG_2PI).sum(-1)
        return base + logdet

    @torch.no_grad()
    def inverse(self, z, ctx):
        x = z
        for mask, net in zip(reversed(self.masks), reversed(self.nets)):
            xm = x * mask
            s, t = self._st(net, xm, ctx)
            s = s * (1 - mask)
            t = t * (1 - mask)
            x = xm + (1 - mask) * ((x - t) * torch.exp(-s))
        return x


@register_model("cinn")
class CINN(PosteriorModel):
    def __init__(self, cfg, geometry: Geometry):
        super().__init__()
        m = cfg.model
        self.geometry = geometry
        self.n_cells = geometry.n_cells
        self.ctx_dim = int(m.ctx_dim)
        self.max_emissions = int(m.max_emissions)

        self.encoder_net = build_encoder(cfg.encoder, self.ctx_dim, N_NODE_FEAT)
        self.cell_emb = nn.Embedding(self.n_cells, int(cfg.encoder.emb_dim))
        self.n_head = nn.Sequential(
            nn.Linear(self.ctx_dim, self.ctx_dim), nn.ReLU(),
            nn.Linear(self.ctx_dim, self.max_emissions + 1),
        )
        self.cell_head = nn.Sequential(
            nn.Linear(self.ctx_dim, self.ctx_dim), nn.ReLU(),
            nn.Linear(self.ctx_dim, self.n_cells),
        )
        self.flow = _CondRealNVP(
            dim=4, ctx_dim=self.ctx_dim + int(cfg.encoder.emb_dim),
            hidden=int(m.hidden_dim), n_blocks=int(m.n_blocks),
        )
        cx, cy = geometry.cell_center_tensors()
        self.register_buffer("cell_cx", cx)
        self.register_buffer("cell_cy", cy)

    def encode(self, xf, nx):
        return self.encoder_net(xf, nx)

    def per_jet_nll(self, batch) -> torch.Tensor:
        xf, nx = batch["xf"], batch["nx"]
        yc, ny = batch["yc"], batch["ny"]
        yraw = batch["yraw"]
        e = self.encode(xf, nx)
        B, L = yc.shape
        dev = yc.device

        n_lp = F.log_softmax(self.n_head(e), dim=-1)  # (B, max+1)
        n_clamped = ny.clamp(max=self.max_emissions)
        logp_n = n_lp.gather(-1, n_clamped.unsqueeze(-1)).squeeze(-1)  # (B,)

        cell_lp = F.log_softmax(self.cell_head(e), dim=-1)  # (B, n_cells)
        mask = (torch.arange(L, device=dev).unsqueeze(0) < ny.unsqueeze(1)).float()

        logp_cell = torch.zeros(B, device=dev)
        logp_coord = torch.zeros(B, device=dev)
        if L > 0:
            per_cell = cell_lp.gather(-1, yc.clamp(min=0))  # (B, L)
            logp_cell = (per_cell * mask).sum(1)

            ctx = torch.cat(
                [e.unsqueeze(1).expand(-1, L, -1), self.cell_emb(yc.clamp(min=0))], dim=-1
            )
            coord_lp = self.flow.log_prob(yraw, ctx)  # (B, L)
            logp_coord = (coord_lp * mask).sum(1)

        return -(logp_n + logp_cell + logp_coord)

    def log_prob(self, batch) -> torch.Tensor:
        return -self.per_jet_nll(batch)

    @torch.inference_mode()
    def sample(self, xf, nx, n, **kw):
        self.eval()
        e = self.encode(xf, nx)  # (1, ctx)
        n_probs = F.softmax(self.n_head(e), dim=-1).squeeze(0)
        cell_probs = F.softmax(self.cell_head(e), dim=-1).squeeze(0)
        ns = torch.multinomial(n_probs, n, replacement=True)  # (n,)
        out = []
        for k in range(n):
            m = int(ns[k].item())
            if m == 0:
                out.append([])
            else:
                cells = torch.multinomial(cell_probs, m, replacement=True).tolist()
                out.append([int(c) for c in cells])
        return out

    def sample_batch(self, xf, nx, n_samples, max_emissions: int = 25):
        return self.sample(xf, nx, n_samples)

    @torch.inference_mode()
    def map_estimate(self, xf, nx, **kw) -> LundPointEstimate:
        self.eval()
        dev = xf.device
        e = self.encode(xf, nx)
        n_star = int(F.log_softmax(self.n_head(e), dim=-1).argmax(-1).item())
        cell_lp = F.log_softmax(self.cell_head(e), dim=-1).squeeze(0)
        # floor the MAP multiplicity so it is never the unphysical empty tree (the
        # constrained MAP under a minimum-emission floor; default min_emissions=1)
        n_star = max(n_star, int(kw.get("min_emissions", 1)))
        top = torch.topk(cell_lp, k=min(n_star, self.n_cells))
        cells = top.indices.tolist()

        ctx = torch.cat(
            [e.expand(len(cells), -1), self.cell_emb(torch.tensor(cells, device=dev))], dim=-1
        )
        coords = self.flow.inverse(torch.zeros(len(cells), 4, device=dev), ctx)  # flow mode
        nodes, total = [], 0.0
        n_lp = F.log_softmax(self.n_head(e), dim=-1).squeeze(0)
        total += float(n_lp[min(n_star, self.max_emissions)])
        for t, c in enumerate(cells):
            c = int(c)
            u, v, lz, ps = (float(coords[t, j]) for j in range(4))
            ls = float(cell_lp[c])
            lk = float(self.flow.log_prob(coords[t : t + 1], ctx[t : t + 1])[0])
            total += ls + lk
            nodes.append(
                LundNode(
                    depth=t, parent=t - 1, cell=c,
                    ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                    kt=math.exp(v), delta_R=math.exp(-u), z=math.exp(lz),
                    logp_split=ls, logp_coord=lk, logp_cont=0.0,
                )
            )
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=len(cells))
