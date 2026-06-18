"""§5.1 autoregressive JUNIPR posterior (was `ConditionalPrimaryLundJUNIPR`).

A pluggable encoder e(x) feeds the conditioned autoregressive decoder over the
groomed parton tree y, with a discrete cell head, a continue/stop head, and (v2)
a continuous coordinate head per node. v1 (`continuous_coords=False`) drops the
coordinate density and keeps only the categorical-cell autoregressive backbone.

With the default config this is numerically identical to the v2 script: the same
encoder (gru), the same three heads, the same staged MAP (beam search + conditional
modes). The verification copies the script's weights into this model and checks
`log_prob` matches bit-for-bit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributions import gauss_logpdf, trunc_normal_logpdf, vonmises_logpdf
from ..encoders.base import build_encoder
from ..features import N_NODE_FEAT
from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate, beam_search_cells
from ..inference.sampling import ancestral_sample_cells
from .base import PosteriorModel, register_model


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Module:
    """An (n_layers)-deep MLP: n_layers==1 -> single Linear; else Linear/ReLU
    stack ending in Linear(hidden, out_dim). n_layers==2 reproduces the script's
    split_head / coord_head."""
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


@register_model("ar_junipr_v1", "ar_junipr_v2", "ar_junipr")
class ARJunipr(PosteriorModel):
    def __init__(self, cfg, geometry: Geometry):
        super().__init__()
        m = cfg.model
        self.geometry = geometry
        self.n_cells = geometry.n_cells
        self.n_bins = geometry.n_bins
        self.ctx_dim = int(m.ctx_dim)
        self.dec_dim = int(m.dec_dim)
        self.dec_layers = int(m.dec_layers)
        self.continuous_coords = bool(m.continuous_coords)
        self.sigma_floor = float(m.sigma_floor)
        self.kappa_max = float(m.kappa_max)
        # default 0.0 == off; getattr tolerates old checkpoint configs lacking the field
        self.cell_label_smoothing = float(getattr(m, "cell_label_smoothing", 0.0))
        self.half_u = geometry.half_u
        self.half_v = geometry.half_v

        emb = int(cfg.encoder.emb_dim)  # shared emb dim (encoder x_feat <-> decoder y_embed)
        self.emb_dim = emb

        # ---- encoder e(x) (pluggable) --------------------------------------
        self.encoder_net = build_encoder(cfg.encoder, self.ctx_dim, N_NODE_FEAT)

        # ---- decoder over y: cell-token embedding + context -> GRU ----------
        self.y_embed = nn.Embedding(self.n_cells + 1, emb)  # +1 for START
        self.dec_in = nn.Linear(emb + self.ctx_dim, self.dec_dim)
        self.decoder = nn.GRU(self.dec_dim, self.dec_dim, num_layers=self.dec_layers, batch_first=True)
        self.h0_proj = nn.Linear(self.ctx_dim, self.dec_dim * self.dec_layers)

        # ---- heads ----------------------------------------------------------
        self.cont_head = nn.Linear(self.dec_dim + self.ctx_dim, 1)
        self.split_head = _mlp(
            self.dec_dim + self.ctx_dim, self.dec_dim, self.n_cells, int(m.split_head_layers)
        )
        self.coord_head = _mlp(
            self.dec_dim + self.ctx_dim + emb, self.dec_dim, 8, int(m.coord_head_layers)
        )

        cx, cy = geometry.cell_center_tensors()
        self.register_buffer("cell_cx", cx)
        self.register_buffer("cell_cy", cy)

    @property
    def start_token(self) -> int:
        return self.n_cells

    # -- encoder -------------------------------------------------------------
    def encode(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(xf, nx)

    def _init_hidden(self, e: torch.Tensor) -> torch.Tensor:
        B = e.shape[0]
        h0 = torch.tanh(self.h0_proj(e))  # (B, dec*layers)
        return h0.view(B, self.dec_layers, self.dec_dim).transpose(0, 1).contiguous()

    # -- teacher-forced decoder states --------------------------------------
    def _decode_states(self, yc: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        B, L = yc.shape
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=yc.device)
        tokens = torch.cat([start, yc], dim=1)  # (B, L+1)
        tok_emb = self.y_embed(tokens)
        e_seq = e.unsqueeze(1).expand(-1, L + 1, -1)
        inp = self.dec_in(torch.cat([tok_emb, e_seq], dim=-1))
        out, _ = self.decoder(inp, self._init_hidden(e))
        return out

    # -- continuous coordinate head -----------------------------------------
    def _coord_params(self, coord_in: torch.Tensor):
        p = self.coord_head(coord_in)
        du_mean = self.half_u * torch.tanh(p[..., 0])
        dv_mean = self.half_v * torch.tanh(p[..., 1])
        du_sig = F.softplus(p[..., 2]) + self.sigma_floor
        dv_sig = F.softplus(p[..., 3]) + self.sigma_floor
        lnz_mean = p[..., 4]
        lnz_sig = F.softplus(p[..., 5]) + self.sigma_floor
        a, b = p[..., 6], p[..., 7]
        kappa = torch.sqrt(a * a + b * b).clamp(1e-3, self.kappa_max)
        mu = torch.atan2(b, a)
        return du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, mu, kappa

    def _coord_logprob(self, params, u, v, lnz, psi, cx, cy):
        du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, mu, kappa = params
        du = (u - cx).clamp(-self.half_u, self.half_u)
        dv = (v - cy).clamp(-self.half_v, self.half_v)
        ll = trunc_normal_logpdf(du, du_mean, du_sig, -self.half_u, self.half_u)
        ll = ll + trunc_normal_logpdf(dv, dv_mean, dv_sig, -self.half_v, self.half_v)
        ll = ll + gauss_logpdf(lnz, lnz_mean, lnz_sig)
        ll = ll + vonmises_logpdf(psi, mu, kappa)
        return ll

    # -- likelihood ----------------------------------------------------------
    def per_jet_nll(self, batch) -> torch.Tensor:
        xf, nx = batch["xf"], batch["nx"]
        yc, ny = batch["yc"], batch["ny"]
        yraw = batch["yraw"]
        e = self.encode(xf, nx)
        out = self._decode_states(yc, e)
        B, Lp1, _ = out.shape
        L = Lp1 - 1
        dev = yc.device

        eh = torch.cat([out, e.unsqueeze(1).expand(-1, Lp1, -1)], dim=-1)
        cont_logit = self.cont_head(eh).squeeze(-1)  # (B, L+1)

        idx = torch.arange(Lp1, device=dev).unsqueeze(0)
        n = ny.unsqueeze(1)
        cont_mask = (idx <= n).float()
        cont_tgt = (idx < n).float()
        cont_ll = -F.binary_cross_entropy_with_logits(cont_logit, cont_tgt, reduction="none")
        cont_ll = (cont_ll * cont_mask).sum(1)

        split_ll = torch.zeros(B, device=dev)
        coord_ll = torch.zeros(B, device=dev)
        if L > 0:
            eh_t = eh[:, :L, :]
            split_lp = F.log_softmax(self.split_head(eh_t), dim=-1)
            split_per = split_lp.gather(-1, yc.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            if self.cell_label_smoothing > 0.0:  # off by default -> per_jet_nll parity
                eps = self.cell_label_smoothing
                split_per = (1.0 - eps) * split_per + eps * split_lp.mean(-1)

            split_mask = (torch.arange(L, device=dev).unsqueeze(0) < n).float()
            split_ll = (split_per * split_mask).sum(1)

            if self.continuous_coords:
                cell_emb = self.y_embed(yc.clamp(min=0))
                params = self._coord_params(torch.cat([eh_t, cell_emb], dim=-1))
                cx, cy = self.cell_cx[yc], self.cell_cy[yc]
                coord_per = self._coord_logprob(
                    params, yraw[..., 0], yraw[..., 1], yraw[..., 2], yraw[..., 3], cx, cy
                )
                coord_ll = (coord_per * split_mask).sum(1)

        return -(cont_ll + split_ll + coord_ll)

    def log_prob(self, batch) -> torch.Tensor:
        return -self.per_jet_nll(batch)

    # -- single-jet / batched decoder steps ---------------------------------
    def _step(self, tok: torch.Tensor, e: torch.Tensor, h):
        inp = self.dec_in(torch.cat([self.y_embed(tok), e.unsqueeze(1)], dim=-1))
        out, h = self.decoder(inp, h)
        hv = torch.cat([out[:, -1, :], e], dim=-1)
        p_cont = torch.sigmoid(self.cont_head(hv)).item()
        logp_split = F.log_softmax(self.split_head(hv), dim=-1).squeeze(0)
        return p_cont, logp_split, h

    def _step_batched(self, tok: torch.Tensor, e: torch.Tensor, h):
        inp = self.dec_in(torch.cat([self.y_embed(tok), e.unsqueeze(1)], dim=-1))
        out, h = self.decoder(inp, h)
        hv = torch.cat([out[:, -1, :], e], dim=-1)
        p_cont = torch.sigmoid(self.cont_head(hv)).squeeze(-1)
        split_logits = self.split_head(hv)
        return p_cont, split_logits, h

    # -- contract: sample ----------------------------------------------------
    @torch.inference_mode()
    def sample(self, xf, nx, n, max_emissions: int = 25, cont_temperature: float = 1.0):
        self.eval()
        dev = xf.device
        e = self.encode(xf, nx).expand(n, -1).contiguous()
        h0 = self._init_hidden(e)
        return ancestral_sample_cells(
            self._step_batched, e, h0, self.start_token, n, dev,
            max_emissions=max_emissions, cont_temperature=cont_temperature,
        )

    # back-compat alias used by closure diagnostics
    def sample_batch(self, xf, nx, n_samples, max_emissions: int = 25):
        return self.sample(xf, nx, n_samples, max_emissions=max_emissions)

    # -- contract: MAP -------------------------------------------------------
    @torch.inference_mode()
    def map_decode(self, xf, nx, beam_width: int = 8, topk_cells: int = 6,
                   max_emissions: int = 25, min_emissions: int = 1, length_penalty: float = 0.0):
        self.eval()
        e = self.encode(xf, nx)
        h0 = self._init_hidden(e)
        return beam_search_cells(
            self._step, e, h0, self.start_token, xf.device,
            beam_width=beam_width, topk_cells=topk_cells, max_emissions=max_emissions,
            min_emissions=min_emissions, length_penalty=length_penalty,
        )

    @torch.inference_mode()
    def describe_sequence(self, xf, nx, cells) -> LundPointEstimate:
        """Attach continuous coordinates (head modes for v2; cell centres for v1)
        and the per-node + total log-density to a primary cell sequence. The total
        equals -per_jet_nll for (x, y_hat) when y_hat's continuous targets are
        these modes — the full joint log-density of the returned config."""
        self.eval()
        dev = xf.device
        L = len(cells)
        e = self.encode(xf, nx)
        yc = torch.tensor([list(cells)], dtype=torch.long, device=dev)
        out = self._decode_states(yc, e)
        eh = torch.cat([out, e.unsqueeze(1).expand(-1, L + 1, -1)], dim=-1)
        cont_logit = self.cont_head(eh).squeeze(-1).squeeze(0)
        logp_cont = F.logsigmoid(cont_logit)
        logp_stop = F.logsigmoid(-cont_logit)

        nodes, total = [], 0.0
        if L > 0:
            eh_t = eh[:, :L, :]
            split_lp = F.log_softmax(self.split_head(eh_t), dim=-1).squeeze(0)
            chosen = split_lp.gather(-1, yc[0].unsqueeze(-1)).squeeze(-1)
            cx, cy = self.cell_cx[yc], self.cell_cy[yc]

            if self.continuous_coords:
                cell_emb = self.y_embed(yc)
                params = self._coord_params(torch.cat([eh_t, cell_emb], dim=-1))
                du_mean, dv_mean, _, _, lnz_mean, _, mu, _ = params
                u_mode, v_mode = cx + du_mean, cy + dv_mean
                coord_per = self._coord_logprob(params, u_mode, v_mode, lnz_mean, mu, cx, cy).squeeze(0)
            else:
                u_mode, v_mode = cx, cy
                lnz_mean = torch.zeros_like(cx)
                mu = torch.zeros_like(cx)
                coord_per = torch.zeros(L, device=dev)

            for t, c in enumerate(cells):
                c = int(c)
                lc, ls, lk = float(logp_cont[t]), float(chosen[t]), float(coord_per[t])
                total += lc + ls + lk
                u, v = float(u_mode[0, t]), float(v_mode[0, t])
                lz, ps = float(lnz_mean[0, t]), float(mu[0, t])
                nodes.append(
                    LundNode(
                        depth=t, parent=t - 1, cell=c,
                        ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                        kt=math.exp(v), delta_R=math.exp(-u), z=math.exp(lz),
                        logp_split=ls, logp_coord=lk, logp_cont=lc,
                    )
                )
        total += float(logp_stop[L])
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=L)

    # beam-search keys map_estimate forwards to map_decode (sampling keys like
    # n_posterior_samples / cont_temperature are silently ignored, so a single
    # decode_params() dict can be splatted into both map_estimate and sample).
    _BEAM_KEYS = ("beam_width", "topk_cells", "max_emissions", "min_emissions", "length_penalty")

    @torch.inference_mode()
    def map_estimate(self, xf, nx, **beam_kwargs) -> LundPointEstimate:
        beam = {k: beam_kwargs[k] for k in self._BEAM_KEYS if k in beam_kwargs}
        cells = self.map_decode(xf, nx, **beam)
        return self.describe_sequence(xf, nx, cells)

    # back-compat alias
    def map_tree(self, xf, nx, **beam_kwargs) -> LundPointEstimate:
        return self.map_estimate(xf, nx, **beam_kwargs)
