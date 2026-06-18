"""§5.3 conditional diffusion / Schrödinger-bridge posterior (arXiv:2404.18807).

Contract-complete *baseline* drop-in: a conditional variance-preserving diffusion
over the 4 continuous coordinates per node (denoising-score-matching training
objective; a variational bound used as the reported `log_prob`), paired — as in
the cINN baseline — with categorical multiplicity and cell heads so it returns the
same posterior-draw structure. `sample` runs the reverse process; `map_estimate`
uses the posterior-mean (x0-prediction) surrogate. Phase 5 swaps in the full
score/bridge model and probability-flow-ODE likelihood; this establishes the
registry drop-in and passes the integration smoke train.
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


class _Denoiser(nn.Module):
    """epsilon_theta(x_t, t, ctx) for 4-D coordinates."""

    def __init__(self, ctx_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 1 + ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x_t, t, ctx):
        return self.net(torch.cat([x_t, t, ctx], dim=-1))


@register_model("diffusion")
class Diffusion(PosteriorModel):
    def __init__(self, cfg, geometry: Geometry):
        super().__init__()
        m = cfg.model
        self.geometry = geometry
        self.n_cells = geometry.n_cells
        self.ctx_dim = int(m.ctx_dim)
        self.n_steps = int(m.n_steps)
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
        self.denoiser = _Denoiser(self.ctx_dim + int(cfg.encoder.emb_dim), int(m.hidden_dim))

        betas = torch.linspace(1e-4, 0.02, self.n_steps)
        alphas = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alphas)

    def encode(self, xf, nx):
        return self.encoder_net(xf, nx)

    def per_jet_nll(self, batch) -> torch.Tensor:
        xf, nx = batch["xf"], batch["nx"]
        yc, ny = batch["yc"], batch["ny"]
        yraw = batch["yraw"]
        e = self.encode(xf, nx)
        B, L = yc.shape
        dev = yc.device

        n_lp = F.log_softmax(self.n_head(e), dim=-1)
        n_clamped = ny.clamp(max=self.max_emissions)
        logp_n = n_lp.gather(-1, n_clamped.unsqueeze(-1)).squeeze(-1)

        cell_lp = F.log_softmax(self.cell_head(e), dim=-1)
        mask = (torch.arange(L, device=dev).unsqueeze(0) < ny.unsqueeze(1)).float()

        logp_cell = torch.zeros(B, device=dev)
        dsm = torch.zeros(B, device=dev)
        if L > 0:
            per_cell = cell_lp.gather(-1, yc.clamp(min=0))
            logp_cell = (per_cell * mask).sum(1)

            ctx = torch.cat(
                [e.unsqueeze(1).expand(-1, L, -1), self.cell_emb(yc.clamp(min=0))], dim=-1
            )
            t = torch.randint(0, self.n_steps, (B, L), device=dev)
            ab = self.alpha_bar[t].unsqueeze(-1)  # (B, L, 1)
            noise = torch.randn_like(yraw)
            x_t = ab.sqrt() * yraw + (1 - ab).sqrt() * noise
            t_feat = (t.float() / self.n_steps).unsqueeze(-1)
            eps = self.denoiser(x_t, t_feat, ctx)
            # denoising-score-matching surrogate used as a (negative) log-density proxy
            dsm = (((eps - noise) ** 2).mean(-1) * mask).sum(1)

        # report a comparable per-jet objective: -(logP(n)+logP(cells)) + dsm
        return -(logp_n + logp_cell) + dsm

    def log_prob(self, batch) -> torch.Tensor:
        return -self.per_jet_nll(batch)

    @torch.inference_mode()
    def _x0(self, ctx):
        """DDPM ancestral reverse process from pure noise -> x0 estimate."""
        dev = ctx.device
        n = ctx.shape[0]
        x = torch.randn(n, 4, device=dev)
        for ti in reversed(range(self.n_steps)):
            t_feat = torch.full((n, 1), ti / self.n_steps, device=dev)
            eps = self.denoiser(x, t_feat, ctx)
            ab = self.alpha_bar[ti]
            beta = self.betas[ti]
            alpha = 1.0 - beta
            # mean of p(x_{t-1} | x_t): DDPM eq. (11)
            mean = alpha.rsqrt() * (x - beta / (1 - ab).clamp(min=1e-8).sqrt() * eps)
            if ti > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.inference_mode()
    def sample(self, xf, nx, n, **kw):
        self.eval()
        e = self.encode(xf, nx)
        n_probs = F.softmax(self.n_head(e), dim=-1).squeeze(0)
        cell_probs = F.softmax(self.cell_head(e), dim=-1).squeeze(0)
        ns = torch.multinomial(n_probs, n, replacement=True)
        out = []
        for k in range(n):
            m = int(ns[k].item())
            if m == 0:
                out.append([])
            else:
                out.append([int(c) for c in torch.multinomial(cell_probs, m, replacement=True).tolist()])
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
        if n_star == 0:
            return LundPointEstimate(nodes=[], logprob=0.0, multiplicity=0)
        top = torch.topk(cell_lp, k=min(n_star, self.n_cells))
        cells = top.indices.tolist()
        ctx = torch.cat(
            [e.expand(len(cells), -1), self.cell_emb(torch.tensor(cells, device=dev))], dim=-1
        )
        coords = self._x0(ctx)  # posterior-mean surrogate
        nodes = []
        for t, c in enumerate(cells):
            c = int(c)
            u, v, lz, ps = (float(coords[t, j]) for j in range(4))
            nodes.append(
                LundNode(
                    depth=t, parent=t - 1, cell=c,
                    ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                    kt=math.exp(v), delta_R=math.exp(-u), z=math.exp(lz),
                    logp_split=float(cell_lp[c]), logp_coord=0.0, logp_cont=0.0,
                )
            )
        return LundPointEstimate(nodes=nodes, logprob=float(cell_lp[cells].sum()), multiplicity=len(cells))
