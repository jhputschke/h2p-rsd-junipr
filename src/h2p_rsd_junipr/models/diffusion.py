"""§5.3 conditional diffusion / Schrödinger-bridge posterior (arXiv:2404.18807).

Contract-complete *baseline* drop-in: a conditional variance-preserving diffusion
over the 4 continuous coordinates per node, paired — as in the cINN baseline —
with categorical multiplicity and cell heads so it returns the same posterior-draw
structure. `sample` runs the reverse process; `map_estimate` uses the
posterior-mean (x0-prediction) surrogate.

> **`log_prob` here is NOT a normalized density.** The coordinate term is the
> denoising-score-matching regression residual used as a (negative) log-density
> *proxy* — it is not the diffusion ELBO and it is not the probability-flow-ODE
> likelihood, so it carries an unknown, context-dependent offset. Consequently this
> family sets `exact_likelihood = False`: its NLL must not be compared against
> `cinn`/`cfm`/`ar_junipr_*`, and its log-ratios are not likelihood ratios. Use
> `model=cfm` ([`cfm.py`](cfm.py)) for the exact-likelihood member of the
> continuous-time family; `diffusion` is kept as the registry's cheap-sampler
> baseline (docs/PLAN_UPDATES.md WP1).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.base import build_encoder
from ..features import N_NODE_FEAT, configured_aux_names
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
    # THE one family whose `log_prob` is a training surrogate, not a density. Every
    # NLL/log-ratio consumer warns on this flag instead of naming the family.
    exact_likelihood = False
    # It does have real coordinates (the reverse process emits them) -- it just has no
    # closed-form CDF for them, hence True here and False for `supports_coordinate_pit`.
    has_continuous_coords = True

    def __init__(self, cfg, geometry: Geometry):
        super().__init__()
        m = cfg.model
        self.geometry = geometry
        self.n_cells = geometry.n_cells
        self.ctx_dim = int(m.ctx_dim)
        self.n_steps = int(m.n_steps)
        self.max_emissions = int(m.max_emissions)

        # Aux conditioning (docs/PLAN_Input.md): the groomed per-jet scalars ride as
        # constant extra COLUMNS of xf, so the only diff is the encoder's input width.
        # () is the default -> n_in == N_NODE_FEAT -> byte-identical state_dict.
        self.aux_feature_names = configured_aux_names(cfg.encoder)
        n_in = N_NODE_FEAT + len(self.aux_feature_names)
        self.encoder_net = build_encoder(cfg.encoder, self.ctx_dim, n_in)
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
        """NOT a normalized log-density — see the module docstring and
        `exact_likelihood = False`. Kept as the training objective and as a relative
        model-selection score *within* this family only."""
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
        n_probs = F.softmax(self.recalibrated_n_logits(self.n_head(e)),
                            dim=-1).squeeze(0)
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

    def _coord_ctx(self, e, cells):
        """`(L, ctx_dim + emb_dim)` conditioning for the denoiser: the jet embedding
        tiled against the per-node cell embedding."""
        return torch.cat([e.expand(len(cells), -1), self.cell_emb(cells)], dim=-1)

    @torch.inference_mode()
    def sample_coordinates(self, xf, nx, cells, *, generator=None):
        # `generator` is accepted for the contract and not threaded: the reverse process draws from the global stream.
        """A draw per cell from the reverse process — `_x0` verbatim, which already
        starts from pure noise and injects noise at every step but the last, so it is
        a genuine ancestral sample rather than a mode."""
        cells = [int(c) for c in cells]
        dev = xf.device
        if not cells:
            return torch.zeros(0, 4, device=dev)
        self.eval()
        return self._x0(self._coord_ctx(self.encode(xf, nx),
                                        torch.tensor(cells, dtype=torch.long, device=dev)))

    @torch.inference_mode()
    def length_pmf(self, xf, nx, mults=None, n_samples: int = 500) -> np.ndarray:
        """Exact P(n|x) from the categorical multiplicity head (no sampling)."""
        self.eval()
        e = self.encode(xf, nx)
        return F.softmax(self.recalibrated_n_logits(self.n_head(e)),
                         dim=-1).squeeze(0).cpu().numpy()

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
        ctx = self._coord_ctx(e, torch.tensor(cells, dtype=torch.long, device=dev))
        # NOT a mode: `_x0` is the ancestral reverse process, so these coordinates are a
        # DRAW and this "MAP" is stochastic in its continuous half. Only the cells are
        # argmaxed. Left as-is (changing it would move every diffusion MAP number); the
        # honest reading is that `map_estimate` and `sample_coordinates` agree here.
        coords = self._x0(ctx)
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
