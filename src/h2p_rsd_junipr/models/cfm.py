"""§5.4 conditional flow matching with an EXACT probability-flow-ODE likelihood.

The registry's third continuous-coordinate family, and the one that repairs the
contract `diffusion` breaks: `log_prob` here is a genuinely normalized log-density
on the physical support, not a training surrogate (`exact_likelihood = True`).

Method: a conditional vector field `v_theta(x, t, ctx)` regressed onto the
optimal-transport path target `u_t(x|x_1) = x_1 - x_0` (Lipman et al., ICLR 2023,
arXiv:2210.02747), which is the family simulation-based inference converged on
(FMPE; Wildberger, Dax et al., NeurIPS 2023, arXiv:2305.17161) and the one used for
substructure-scale generative unfolding (Huetsch et al., SciPost Phys. 18 (2025)
070, arXiv:2404.18807; Petitjean et al., arXiv:2510.19906). At evaluation the same
field defines the probability-flow ODE (Song et al., ICLR 2021, arXiv:2011.13456)
whose instantaneous change of variables

    d log p / dt = - div v_theta

is integrated to give the density. The coordinate dimension is **4**, so the
divergence is computed EXACTLY with 4 vector-Jacobian products per step — no
Hutchinson estimator, hence no stochastic likelihood.

The discrete structure mirrors the proven cINN factorization exactly:

    q(y|x) = q(N|x) . prod_t q(cell_t|x) . prod_t p_cfm(coords_t | x, cell_t)

so `n_head` / `cell_head` / `sample` / `length_pmf` are the cINN treatment
unchanged, and only the coordinate density is new.

**Support.** The field lives in an unbounded standardized space; the physical
coordinates are recovered by fixed bijections whose closed-form log-Jacobians are
added to the density: a tanh box map onto the within-cell offsets
`(+-half_u, +-half_v)`, an angle wrap plus tanh box onto psi in `(-pi, pi)`, and the
identity on `ln z`. The density is therefore normalized on exactly the physical
support the AR heads use — the property a discretized grid can never satisfy.
Unlike the AR von Mises head the psi map is not periodic across the branch cut; the
seam is placed at +-pi (see `_to_std`), and closing it structurally needs Riemannian
flow matching, which is deliberately out of scope here.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributions import std_normal_cdf, wrap_to_pi
from ..encoders.base import build_encoder
from ..features import N_NODE_FEAT
from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate
from .base import PosteriorModel, register_model

_LOG_2PI = math.log(2.0 * math.pi)
_ATANH_CLAMP = 1.0 - 1e-6  # keeps atanh finite at the box edges

# Base-space dimension names for the latent-space PIT report (WP2).
_LATENT_NAMES = ("s0", "s1", "s2", "s3")


class _VectorField(nn.Module):
    """v_theta(x, t, ctx): R^4 x [0,1] x R^ctx -> R^4, with Fourier time features.

    Sinusoidal time features are what let one MLP represent a field that changes
    shape along the path; a raw scalar t makes the early (noise-dominated) and late
    (data-dominated) regimes fight over the same weights."""

    def __init__(self, ctx_dim: int, hidden: int, time_features: int):
        super().__init__()
        self.time_features = int(time_features)
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(self.time_features // 2).float() * math.pi
        )
        self.net = nn.Sequential(
            nn.Linear(4 + self.time_features + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def _t_feat(self, t):
        a = t * self.freqs                      # (..., F/2)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def forward(self, x, t, ctx):
        """x: (..., 4), t: (..., 1), ctx: (..., ctx_dim) -> (..., 4)."""
        return self.net(torch.cat([x, self._t_feat(t), ctx], dim=-1))


@register_model("cfm")
class CFM(PosteriorModel):
    exact_likelihood = True          # the point of this family
    supports_coordinate_pit = True   # PIT via the reverse ODE into the base space

    def __init__(self, cfg, geometry: Geometry):
        super().__init__()
        m = cfg.model
        self.geometry = geometry
        self.n_cells = geometry.n_cells
        self.ctx_dim = int(m.ctx_dim)
        self.max_emissions = int(m.max_emissions)
        self.n_ode_steps = int(m.n_ode_steps)
        self.sigma_min = float(m.sigma_min)
        self.ode_solver = str(getattr(m, "ode_solver", "rk4"))
        if self.ode_solver not in ("rk4", "heun"):
            raise ValueError(f"model.ode_solver must be rk4|heun, got {self.ode_solver!r}")
        self.map_mode = str(getattr(m, "cfm_map", "ode_mode"))
        if self.map_mode not in ("ode_mode", "ascent"):
            raise ValueError(f"model.cfm_map must be ode_mode|ascent, got {self.map_mode!r}")
        self.half_u = geometry.half_u
        self.half_v = geometry.half_v

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
        self.field = _VectorField(
            self.ctx_dim + int(cfg.encoder.emb_dim), int(m.hidden_dim), int(m.time_features)
        )
        cx, cy = geometry.cell_center_tensors()
        self.register_buffer("cell_cx", cx)
        self.register_buffer("cell_cy", cy)

    def encode(self, xf, nx):
        return self.encoder_net(xf, nx)

    # ------------------------------------------------------------------
    # Physical <-> standardized coordinates (fixed bijections, exact log-Jacobians)
    # ------------------------------------------------------------------
    def _to_std(self, yraw, yc):
        """(ln 1/DeltaR, ln kt, ln z, psi) -> standardized s in R^4, with
        `log |ds/dx|` (the term ADDED to the standardized log-density to get the
        physical one).

        du, dv are the within-cell offsets on `(+-half_u, +-half_v)`; psi is wrapped
        into `(-pi, pi)` first. Each bounded coordinate goes through `atanh(x/h)`,
        whose Jacobian `1/(h (1 - (x/h)^2))` is closed-form. ln z passes through."""
        cx, cy = self.cell_cx[yc], self.cell_cy[yc]
        du = ((yraw[..., 0] - cx) / self.half_u).clamp(-_ATANH_CLAMP, _ATANH_CLAMP)
        dv = ((yraw[..., 1] - cy) / self.half_v).clamp(-_ATANH_CLAMP, _ATANH_CLAMP)
        ps = (wrap_to_pi(yraw[..., 3]) / math.pi).clamp(-_ATANH_CLAMP, _ATANH_CLAMP)
        s = torch.stack([torch.atanh(du), torch.atanh(dv), yraw[..., 2], torch.atanh(ps)], -1)
        log_djac = (
            -math.log(self.half_u) - torch.log1p(-du * du)
            - math.log(self.half_v) - torch.log1p(-dv * dv)
            - math.log(math.pi) - torch.log1p(-ps * ps)
        )
        return s, log_djac

    def _to_phys(self, s, yc):
        """The inverse bijection: standardized s -> physical coordinates."""
        cx, cy = self.cell_cx[yc], self.cell_cy[yc]
        return torch.stack(
            [
                cx + self.half_u * torch.tanh(s[..., 0]),
                cy + self.half_v * torch.tanh(s[..., 1]),
                s[..., 2],
                math.pi * torch.tanh(s[..., 3]),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Flow matching: the training regression and the ODE
    # ------------------------------------------------------------------
    def _fm_loss(self, s, ctx, mask):
        """Conditional flow-matching regression on the OT path (Lipman Eq. 22-23):
        with x_0 ~ N(0, I) and x_1 = s, the interpolant is
        `x_t = (1 - (1-sigma_min) t) x_0 + t x_1` and the target field is
        `u_t = x_1 - (1 - sigma_min) x_0`, constant along the path."""
        x1 = s
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[:-1] + (1,), device=x1.device, dtype=x1.dtype)
        a = 1.0 - (1.0 - self.sigma_min) * t
        x_t = a * x0 + t * x1
        target = x1 - (1.0 - self.sigma_min) * x0
        v = self.field(x_t, t, ctx)
        return (((v - target) ** 2).mean(-1) * mask).sum(1)

    def _divergence(self, v, xx, *, create_graph: bool):
        """EXACT `div v` by one vector-Jacobian product per coordinate.

        The state is 4-dimensional, so the full trace costs 4 VJPs — cheap enough that
        the Hutchinson estimator every high-dimensional CNF is forced into is
        unnecessary here, and the likelihood comes out deterministic rather than
        stochastic (the property that makes it usable for model selection)."""
        div = torch.zeros(xx.shape[:-1], device=xx.device, dtype=xx.dtype)
        for i in range(4):
            e = torch.zeros_like(v)
            e[..., i] = 1.0
            g = torch.autograd.grad(v, xx, grad_outputs=e, retain_graph=True,
                                    create_graph=create_graph)[0]
            div = div + g[..., i]
        return div

    def _ode(self, s, ctx, *, reverse: bool, with_divergence: bool, n_steps=None,
             differentiable: bool = False):
        """Integrate the probability-flow ODE `ds/dt = v_theta(s, t, ctx)` with a
        fixed-step solver, optionally accumulating the divergence along the way.

        Direction: `reverse=False` runs t: 0 -> 1 (base -> data, i.e. sampling);
        `reverse=True` runs t: 1 -> 0 (data -> base, i.e. likelihood).

        `differentiable=True` keeps the whole trajectory in the autograd graph so a
        caller can take gradients of the resulting density w.r.t. the coordinates
        (`cfm_map="ascent"`). The default detaches at every step — a large constant
        factor cheaper, and gradients through the likelihood are never wanted."""
        n_steps = int(n_steps or self.n_ode_steps)
        solver = self.ode_solver
        dt = (-1.0 if reverse else 1.0) / n_steps
        # dtype from `s`, not the default: the likelihood is evaluated in float64 by the
        # normalization test, and a float32 time would only survive by type promotion.
        t = (torch.ones if reverse else torch.zeros)(
            s.shape[:-1] + (1,), device=s.device, dtype=s.dtype
        )
        x = s
        acc = torch.zeros(s.shape[:-1], device=s.device, dtype=s.dtype) if with_divergence else None

        def f(xx, tt):
            if not with_divergence:
                return self.field(xx, tt, ctx), None
            if differentiable:
                v = self.field(xx, tt, ctx)
                return v, self._divergence(v, xx, create_graph=True)
            xx = xx.detach().requires_grad_(True)
            with torch.enable_grad():
                v = self.field(xx, tt, ctx)
                div = self._divergence(v, xx, create_graph=False)
            return v.detach(), div.detach()

        for _ in range(n_steps):
            if solver == "heun":                      # 2 field evals per step
                k1, d1 = f(x, t)
                k2, d2 = f(x + dt * k1, t + dt)
                dx, ddiv = 0.5 * (k1 + k2), 0.5 * (d1 + d2) if with_divergence else None
            else:                                     # rk4 (default), 4 field evals
                k1, d1 = f(x, t)
                k2, d2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3, d3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4, d4 = f(x + dt * k3, t + dt)
                dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                ddiv = (d1 + 2 * d2 + 2 * d3 + d4) / 6.0 if with_divergence else None
            x = x + dt * dx
            if with_divergence:
                # d log p / dt = -div v, integrated ALONG the direction of travel, so
                # `acc` always means log p(end) - log p(start) — whichever way we run.
                # Forward (base -> data): log p(x_1) = log N(x_0) + acc.
                # Reverse (data -> base): log p(x_1) = log N(x_0) - acc.
                acc = acc - dt * ddiv
            t = t + dt
        return x, acc

    def coord_log_prob(self, yraw, yc, ctx, n_steps=None, differentiable=False) -> torch.Tensor:
        """log p_cfm(coords | ctx) on the PHYSICAL support, exactly.

        Reverse-integrates to the base point while accumulating the divergence, then
        applies the instantaneous change of variables and the bijections' Jacobian:

            log p(s_1) = log N(s_0; 0, I) - integral_0^1 div v dt,
            log p_phys(x) = log p(s_1) + log |ds/dx|.

        `_ode` returns `acc = log p(end) - log p(start)` along its direction of travel;
        here the travel is data -> base, so `acc = log N(s_0) - log p(s_1)` and it
        enters with a MINUS sign. `test_cfm.py::test_density_integrates_to_one` is what
        pins this down — the wrong sign reads as an integral of ~0.86, not ~1.0 — and
        `test_forward_reverse_roundtrip` pins the other direction.

        The `inference_mode(False)` escape hatch is required: the exact divergence
        needs autograd, and inference tensors created by an enclosing `inference_mode`
        (the trainer's validation pass, every `@torch.inference_mode()` decode path)
        cannot take part in an autograd graph. Cloning inside a normal-mode block is
        the documented way back. Note the returned value is a NUMBER, not a training
        objective — `training_objective` is the differentiable one."""
        with torch.inference_mode(False):
            s1, log_djac = self._to_std(yraw.clone(), yc)
            s0, acc = self._ode(s1, ctx.clone(), reverse=True, with_divergence=True,
                                n_steps=n_steps, differentiable=differentiable)
            base = (-0.5 * (s0**2) - 0.5 * _LOG_2PI).sum(-1)
            return base - acc + log_djac

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------
    def _discrete_terms(self, batch, e):
        """log q(N|x) and sum_t log q(cell_t|x) — the cINN treatment, unchanged."""
        yc, ny = batch["yc"], batch["ny"]
        B, L = yc.shape
        n_lp = F.log_softmax(self.n_head(e), dim=-1)
        logp_n = n_lp.gather(-1, ny.clamp(max=self.max_emissions).unsqueeze(-1)).squeeze(-1)
        cell_lp = F.log_softmax(self.cell_head(e), dim=-1)
        mask = (torch.arange(L, device=yc.device).unsqueeze(0) < ny.unsqueeze(1)).float()
        logp_cell = torch.zeros(B, device=yc.device)
        if L > 0:
            logp_cell = (cell_lp.gather(-1, yc.clamp(min=0)) * mask).sum(1)
        return logp_n, logp_cell, mask

    def _coord_ctx(self, e, yc):
        """The coordinate field's conditioning vector: context + cell embedding —
        built exactly the way the AR `coord_head` input is, minus the AR state."""
        L = yc.shape[1]
        return torch.cat(
            [e.unsqueeze(1).expand(-1, L, -1), self.cell_emb(yc.clamp(min=0))], dim=-1
        )

    def training_objective(self, batch) -> torch.Tensor:
        """(B,) discrete NLL + the flow-matching regression on the coordinates.

        NOT an NLL — the coordinate term is a regression residual with its own scale.
        This is why `training_objective` exists as a hook: `log_prob` below stays the
        exact density, so validation NLL and every likelihood-ratio consumer keep
        their meaning while training stays cheap (no ODE in the loop)."""
        e = self.encode(batch["xf"], batch["nx"])
        logp_n, logp_cell, mask = self._discrete_terms(batch, e)
        fm = torch.zeros_like(logp_n)
        if batch["yc"].shape[1] > 0:
            s, _ = self._to_std(batch["yraw"], batch["yc"])
            fm = self._fm_loss(s, self._coord_ctx(e, batch["yc"]), mask)
        return -(logp_n + logp_cell) + fm

    def per_jet_nll(self, batch) -> torch.Tensor:
        return -self.log_prob(batch)

    def log_prob(self, batch) -> torch.Tensor:
        """(B,) exact log q(y|x) = log q(N|x) + sum log q(cell|x) + sum log p_cfm."""
        e = self.encode(batch["xf"], batch["nx"])
        logp_n, logp_cell, mask = self._discrete_terms(batch, e)
        logp_coord = torch.zeros_like(logp_n)
        if batch["yc"].shape[1] > 0:
            per = self.coord_log_prob(batch["yraw"], batch["yc"], self._coord_ctx(e, batch["yc"]))
            logp_coord = (per.to(mask.device) * mask).sum(1)
        return logp_n + logp_cell + logp_coord

    @torch.inference_mode()
    def coordinate_cdfs(self, batch) -> dict | None:
        """PIT in the flow's BASE space, reached by the reverse ODE (see the cINN
        docstring for why a base-space PIT is the right per-dimension test here)."""
        yc, ny = batch["yc"], batch["ny"]
        B, L = yc.shape
        if L == 0:
            empty = torch.zeros(B, 0, 4, device=yc.device)
            return {"names": _LATENT_NAMES, "u": empty, "mask": empty[..., 0].bool(),
                    "space": "latent"}
        e = self.encode(batch["xf"], batch["nx"])
        with torch.inference_mode(False):
            s1, _ = self._to_std(batch["yraw"].clone(), yc)
            s0, _ = self._ode(s1, self._coord_ctx(e, yc).clone(), reverse=True,
                              with_divergence=False)
        mask = torch.arange(L, device=yc.device).unsqueeze(0) < ny.unsqueeze(1)
        return {"names": _LATENT_NAMES, "u": std_normal_cdf(s0.detach()), "mask": mask,
                "space": "latent"}

    # -- sampling (cells only; coordinates are integrated on demand) ----------
    @torch.inference_mode()
    def sample(self, xf, nx, n, **kw):
        """N ~ q(N|x) then N i.i.d. cells ~ q(cell|x) — the cINN sampler. The ODE is
        NOT run here: the contract's draws are cell chains, and every consumer that
        needs coordinates (`describe_cells`, `map_estimate`) integrates them itself."""
        self.eval()
        e = self.encode(xf, nx)
        n_probs = F.softmax(self.n_head(e), dim=-1).squeeze(0)
        cell_probs = F.softmax(self.cell_head(e), dim=-1).squeeze(0)
        ns = torch.multinomial(n_probs, n, replacement=True)
        out = []
        for k in range(n):
            m = int(ns[k].item())
            out.append([] if m == 0
                       else [int(c) for c in torch.multinomial(cell_probs, m, replacement=True)])
        return out

    def sample_batch(self, xf, nx, n_samples, max_emissions: int = 25):
        return self.sample(xf, nx, n_samples)

    @torch.inference_mode()
    def sample_coords(self, cells, ctx, n_steps=None) -> torch.Tensor:
        """Coordinates for a given cell chain: push base draws through the FORWARD
        ODE (t: 0 -> 1) and back through the bijections."""
        with torch.inference_mode(False):
            s0 = torch.randn(ctx.shape[:-1] + (4,), device=ctx.device)
            s1, _ = self._ode(s0, ctx.clone(), reverse=False, with_divergence=False,
                              n_steps=n_steps)
        return self._to_phys(s1.detach(), cells)

    @torch.inference_mode()
    def length_pmf(self, xf, nx, mults=None, n_samples: int = 500) -> np.ndarray:
        """Exact P(n|x) from the categorical multiplicity head (no sampling)."""
        self.eval()
        e = self.encode(xf, nx)
        return F.softmax(self.n_head(e), dim=-1).squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def map_estimate(self, xf, nx, **kw) -> LundPointEstimate:
        """`N* = max(argmax q(N|x), min_emissions)`, the top-`N*` cells, and a
        coordinate point per cell.

        `cfm_map="ode_mode"` (default) pushes the base mode `s0 = 0` through the
        forward ODE — the cheap, deterministic analogue of the cINN's `flow.inverse(0)`.
        `cfm_map="ascent"` instead climbs the EXACT `log p_cfm` from that point, which
        is the true conditional mode but costs `ascent_steps` ODE likelihood evaluations."""
        self.eval()
        dev = xf.device
        e = self.encode(xf, nx)
        n_lp = F.log_softmax(self.n_head(e), dim=-1).squeeze(0)
        n_star = max(int(n_lp.argmax().item()), int(kw.get("min_emissions", 1)))
        n_star = min(n_star, int(kw.get("max_emissions", self.max_emissions)), self.n_cells)
        cell_lp = F.log_softmax(self.cell_head(e), dim=-1).squeeze(0)
        cells = torch.topk(cell_lp, k=max(n_star, 1)).indices if n_star > 0 else \
            torch.zeros(0, dtype=torch.long, device=dev)
        if n_star == 0:
            return LundPointEstimate(nodes=[], logprob=float(n_lp[0]), multiplicity=0)

        ctx = torch.cat([e.expand(len(cells), -1), self.cell_emb(cells)], dim=-1)
        with torch.inference_mode(False):
            s = self._ode(torch.zeros(len(cells), 4, device=dev), ctx.clone(),
                          reverse=False, with_divergence=False)[0].detach()
            if self.map_mode == "ascent":
                s = self._ascend(s, cells, ctx.clone(), int(kw.get("ascent_steps", 25)))
        coords = self._to_phys(s, cells)
        per = self.coord_log_prob(coords, cells, ctx)
        nodes, total = [], float(n_lp[min(n_star, self.max_emissions)])
        for t, c in enumerate(cells.tolist()):
            u, v, lz, ps = (float(coords[t, j]) for j in range(4))
            ls, lk = float(cell_lp[c]), float(per[t])
            total += ls + lk
            nodes.append(
                LundNode(
                    depth=t, parent=t - 1, cell=int(c),
                    ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                    kt=math.exp(v), delta_R=math.exp(-u), z=math.exp(lz),
                    logp_split=ls, logp_coord=lk, logp_cont=0.0,
                )
            )
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=len(nodes))

    def _ascend(self, s, cells, ctx, steps: int, lr: float = 0.05):
        """Gradient ascent on the EXACT physical log-density, parameterised in
        standardized space (so the iterate can never leave the physical box).

        `differentiable=True` keeps the ODE trajectory in the graph — without it the
        density is a detached number and the ascent silently does nothing."""
        s = s.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([s], lr=lr)
        for _ in range(max(int(steps), 0)):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = -self.coord_log_prob(
                    self._to_phys(s, cells), cells, ctx, differentiable=True
                ).sum()
                loss.backward()
            opt.step()
        return s.detach()
