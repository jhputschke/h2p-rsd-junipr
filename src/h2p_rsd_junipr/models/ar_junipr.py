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
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributions import (
    gauss_cdf,
    gauss_logpdf,
    rq_interval_cdf,
    rq_interval_icdf,
    rq_interval_logpdf,
    rq_interval_sample,
    rq_spline_n_params,
    trunc_normal_cdf,
    trunc_normal_logpdf,
    trunc_normal_sample,
    vonmises_cdf,
    vonmises_logpdf,
    vonmises_sample,
)
from ..encoders.base import build_encoder
from ..features import N_NODE_FEAT, configured_aux_names
from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate, beam_search_cells
from ..inference.sampling import ancestral_sample_cells, ancestral_sample_cells_fixed_length
from .base import PosteriorModel, register_model

# Physical coordinate names, in the column order of `yraw` / the coordinate head.
# Shared with the WP2 per-coordinate PIT report.
_COORD_NAMES = ("du", "dv", "ln_z", "psi")


class CoordParams(NamedTuple):
    """One node's coordinate-head parameters.

    The first eight fields are the tuple this used to be, unchanged and in the same
    order — so `p[:4]` still gives the within-cell offset parameters and the numbers are
    identical under the default `lnz_head="truncnorm"`.

    The ln z head fills EITHER `(lnz_mean, lnz_sig)` — the truncated normal — OR
    `lnz_spline`, the `(..., 3K-1)` raw spline parameters; the other is `None`. They are
    alternatives rather than a base and a warp: a spline on top of a *learnable*
    truncated normal is non-identifiable and diverges (see `distributions.py`'s spline
    section for the measurement). Nothing outside `_lnz_*` should read either field."""

    du_mean: torch.Tensor
    dv_mean: torch.Tensor
    du_sig: torch.Tensor
    dv_sig: torch.Tensor
    lnz_mean: torch.Tensor | None
    lnz_sig: torch.Tensor | None
    mu: torch.Tensor
    kappa: torch.Tensor
    lnz_spline: torch.Tensor | None = None

    def apply(self, fn) -> CoordParams:
        """The same parameters with `fn` applied to each — indexing, broadcasting, a
        device move. A field the active head does not use stays `None` rather than
        becoming `fn(None)`.

        NOTE the spline tensor carries a TRAILING parameter axis the other fields do not,
        so an `fn` that indexes the last dimension must account for it."""
        return CoordParams(*(None if t is None else fn(t) for t in self))


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


@register_model("ar_junipr_v1", "ar_junipr_v2", "ar_junipr_v3", "ar_junipr_v4", "ar_junipr")
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
        # exact closed-form coordinate CDFs exist exactly when the coordinate head does
        self.supports_coordinate_pit = self.continuous_coords
        # v1 is THE family with no coordinates at all: its nodes carry cell centres and
        # ln z / psi placeholders, and `sample_coordinates` returns None to say so.
        self.has_continuous_coords = self.continuous_coords
        self.sigma_floor = float(m.sigma_floor)
        self.kappa_max = float(m.kappa_max)
        # default 0.0 == off; getattr tolerates old checkpoint configs lacking the field
        self.cell_label_smoothing = float(getattr(m, "cell_label_smoothing", 0.0))
        # first-class multiplicity head (q(y|x) = q(N|x) q(y|N,x)); getattr-tolerant so
        # old checkpoint configs (no field) load as the implicit continue/stop model.
        self.use_multiplicity_head = bool(getattr(m, "use_multiplicity_head", False))
        self.max_emissions = int(getattr(m, "max_emissions", 25))
        # decoder cross-attention over the hadron-node states (WP3); getattr-tolerant
        # so pre-WP3 checkpoint configs rebuild as the pooled-context model.
        self.use_cross_attention = bool(getattr(m, "use_cross_attention", False))
        self.half_u = geometry.half_u
        self.half_v = geometry.half_v
        # bounded-support ln z head (docs/PLAN_prod_test_v1.md WP-A); getattr-tolerant so
        # pre-WP-A checkpoint configs rebuild as the unbounded Normal they were trained as.
        self.lnz_support = str(getattr(m, "lnz_support", "legacy"))
        if self.lnz_support not in ("legacy", "physical"):
            raise ValueError(
                f"model.lnz_support must be 'legacy' or 'physical', got {self.lnz_support!r}"
            )
        self.lnz_physical = self.lnz_support == "physical"
        self.lnz_zcut = float(getattr(m, "lnz_zcut", 0.1))
        self.lnz_beta = float(getattr(m, "lnz_beta", 0.0))
        if self.lnz_physical and not (0.0 < self.lnz_zcut < 0.5):
            raise ValueError(
                f"model.lnz_support='physical' needs 0 < model.lnz_zcut < 0.5 (the soft-drop "
                f"lower bound must sit below the kinematic z <= 1/2), got {self.lnz_zcut!r}"
            )
        # ln z_cut and ln(1/2) as plain floats; the bounds themselves are cell-conditional
        # and built on the fly from `cell_cx` (see `lnz_bounds`), so NO buffer is added and
        # the `legacy` state_dict stays byte-identical.
        self._ln_zcut = math.log(self.lnz_zcut) if self.lnz_physical else float("-inf")
        self._ln_half = math.log(0.5)
        # ln z SHAPE (docs/PLAN_lnz_spline_head.md); getattr-tolerant so every checkpoint
        # config written before the field rebuilds as the truncated normal it was trained
        # as. `truncnorm` adds no head outputs, so its state_dict is byte-identical.
        self.lnz_head = str(getattr(m, "lnz_head", "truncnorm"))
        if self.lnz_head not in ("truncnorm", "spline"):
            raise ValueError(
                f"model.lnz_head must be 'truncnorm' or 'spline', got {self.lnz_head!r}"
            )
        self.lnz_spline_bins = int(getattr(m, "lnz_spline_bins", 8))
        self.lnz_spline = self.lnz_head == "spline"
        if self.lnz_spline and not self.lnz_physical:
            # The spline is composed on the TRUNCATED normal's CDF, which is what keeps
            # the soft-drop support exact. On `legacy` there is no interval to warp — the
            # base is an unbounded Normal — so the pairing is a configuration error rather
            # than a silently different model.
            raise ValueError(
                "model.lnz_head='spline' needs model.lnz_support='physical': the spline "
                "warps the truncated normal's CDF on the soft-drop interval, and 'legacy' "
                "has no such interval (docs/PLAN_lnz_spline_head.md §2)."
            )
        # Head outputs the ln z density needs: 2 for the truncated normal (mean, sigma),
        # 3K-1 for the spline, which REPLACES them rather than adding to them. So the
        # truncnorm width stays exactly today's 8 = 6 + 2 and no output is ever dead.
        self.lnz_n_params = (rq_spline_n_params(self.lnz_spline_bins)
                             if self.lnz_spline else 2)

        emb = int(cfg.encoder.emb_dim)  # shared emb dim (encoder x_feat <-> decoder y_embed)
        self.emb_dim = emb

        # ---- encoder e(x) (pluggable) --------------------------------------
        # Aux conditioning (docs/PLAN_Input.md): the groomed per-jet scalars ride as
        # constant extra COLUMNS of xf, so the only diff is the encoder's input width.
        # () is the default -> n_in == N_NODE_FEAT -> byte-identical state_dict.
        self.aux_feature_names = configured_aux_names(cfg.encoder)
        n_in = N_NODE_FEAT + len(self.aux_feature_names)
        self.encoder_net = build_encoder(cfg.encoder, self.ctx_dim, n_in)

        # ---- optional cross-attention onto the encoder's per-node states ----
        # Built ONLY when on, so the off-path module list / state_dict are unchanged.
        if self.use_cross_attention:
            if not getattr(self.encoder_net, "returns_sequence", False):
                raise ValueError(
                    f"model.use_cross_attention=true needs an encoder exposing per-node "
                    f"states, but encoder={cfg.encoder.name!r} has returns_sequence=False. "
                    f"Implement Encoder.forward_seq there, or use encoder=gru|lundnet|deepsets."
                )
            heads = int(getattr(m, "xattn_heads", 4))
            if self.dec_dim % heads:
                raise ValueError(
                    f"model.xattn_heads={heads} must divide model.dec_dim={self.dec_dim}"
                )
            self.kv_proj = nn.Linear(int(self.encoder_net.seq_dim), self.dec_dim)
            self.xattn = nn.MultiheadAttention(self.dec_dim, heads, batch_first=True)

        # ---- decoder over y: cell-token embedding + context -> GRU ----------
        self.y_embed = nn.Embedding(self.n_cells + 1, emb)  # +1 for START
        self.dec_in = nn.Linear(emb + self.ctx_dim, self.dec_dim)
        self.decoder = nn.GRU(self.dec_dim, self.dec_dim, num_layers=self.dec_layers, batch_first=True)
        self.h0_proj = nn.Linear(self.ctx_dim, self.dec_dim * self.dec_layers)

        # ---- heads ----------------------------------------------------------
        # Length model: an explicit categorical q(N|x) head (v3) OR the implicit
        # per-step continue/stop Bernoulli (default). Exactly one is built, so the
        # off-path state_dict is byte-identical to today and old checkpoints load.
        if self.use_multiplicity_head:
            self.n_head = nn.Sequential(
                nn.Linear(self.ctx_dim, self.ctx_dim), nn.ReLU(),
                nn.Linear(self.ctx_dim, self.max_emissions + 1),
            )
        else:
            self.cont_head = nn.Linear(self.dec_dim + self.ctx_dim, 1)
        self.split_head = _mlp(
            self.dec_dim + self.ctx_dim, self.dec_dim, self.n_cells, int(m.split_head_layers)
        )
        # Layout: [du_mean, dv_mean, du_sig, dv_sig] + <ln z block> + [psi a, psi b].
        # The ln z block is 2 wide for the truncated normal and 3K-1 for the spline, so
        # `truncnorm` is exactly today's 8 outputs in today's order (parity) and `spline`
        # spends its width on the spline instead of carrying two unused numbers.
        self.coord_head = _mlp(
            self.dec_dim + self.ctx_dim + emb, self.dec_dim, 6 + self.lnz_n_params,
            int(m.coord_head_layers),
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

    # -- cross-attention over the hadron-node states (WP3, opt-in) -----------
    def xattn_kv(self, xf: torch.Tensor, nx: torch.Tensor):
        """`(kv (B, Mx, dec_dim), key_padding_mask (B, Mx))` for the decoder's attention,
        or None when cross-attention is off.

        Computed ONCE per jet by each decode entry point and threaded into the
        incremental steps, so a K-draw sample does not re-encode the hadron sequence K
        times."""
        if not self.use_cross_attention:
            return None
        seq, mask = self.encoder_net.forward_seq(xf, nx)
        if seq.shape[1] == 0:  # every jet here has an EMPTY hadron tree: nothing to attend to
            return None
        return self.kv_proj(seq), ~mask.bool()  # nn.MultiheadAttention masks where True

    def _apply_xattn(self, out: torch.Tensor, kv) -> torch.Tensor:
        """Residual cross-attention from decoder states onto the hadron-node states.

        RESIDUAL on purpose: every head's input width (`dec_dim + ctx_dim [+ emb]`) is
        unchanged, so with the switch off the module list and `state_dict` are
        byte-identical to today and old checkpoints keep loading strictly. Attention is
        over `x` only, so the AR factorization over `y` stays causal and the sampling /
        beam paths inherit the change with no decode-logic edits."""
        if kv is None:
            return out
        k, key_padding_mask = kv
        if k.shape[0] != out.shape[0]:  # one jet, K draws: broadcast the same keys
            k = k.expand(out.shape[0], -1, -1)
            key_padding_mask = key_padding_mask.expand(out.shape[0], -1)
        attn, _ = self.xattn(out, k, k, key_padding_mask=key_padding_mask,
                             need_weights=False)
        # A jet with an EMPTY hadron tree has every key masked. Softmax over nothing is
        # undefined and some torch versions return NaN there, which would silently
        # poison a whole batch's gradients — so drop the residual for those rows, which
        # is also the right semantics: nothing to attend to contributes nothing.
        empty = key_padding_mask.all(dim=1)
        if bool(empty.any()):
            attn = torch.where(empty[:, None, None], torch.zeros_like(attn), attn)
        return out + attn

    # -- teacher-forced decoder states --------------------------------------
    def _decode_states(self, yc: torch.Tensor, e: torch.Tensor, kv=None) -> torch.Tensor:
        B, L = yc.shape
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=yc.device)
        tokens = torch.cat([start, yc], dim=1)  # (B, L+1)
        tok_emb = self.y_embed(tokens)
        e_seq = e.unsqueeze(1).expand(-1, L + 1, -1)
        inp = self.dec_in(torch.cat([tok_emb, e_seq], dim=-1))
        out, _ = self.decoder(inp, self._init_hidden(e))
        return self._apply_xattn(out, kv)

    # -- continuous coordinate head -----------------------------------------
    def _coord_params(self, coord_in: torch.Tensor) -> CoordParams:
        """The per-node coordinate-head parameters, as a NAMED tuple.

        The first eight fields are exactly the eight this returned before the spline head
        existed, in the same order and computed from the same output slots, so
        `lnz_head="truncnorm"` is numerically unchanged and `p[:4]` keeps working for the
        callers that want only the within-cell offsets. `lnz_spline` is the ninth and is
        `None` unless the spline is on — the flag is read once, here, and every ln z
        density call goes through `_lnz_*` below rather than re-testing it."""
        p = self.coord_head(coord_in)
        du_mean = self.half_u * torch.tanh(p[..., 0])
        dv_mean = self.half_v * torch.tanh(p[..., 1])
        du_sig = F.softplus(p[..., 2]) + self.sigma_floor
        dv_sig = F.softplus(p[..., 3]) + self.sigma_floor
        w = self.lnz_n_params
        if self.lnz_spline:
            lnz_mean = lnz_sig = None
            spline = p[..., 4:4 + w]
        else:  # today's slots 4 and 5, unchanged
            lnz_mean = p[..., 4]
            lnz_sig = F.softplus(p[..., 5]) + self.sigma_floor
            spline = None
        a, b = p[..., 4 + w], p[..., 5 + w]
        kappa = torch.sqrt(a * a + b * b).clamp(1e-3, self.kappa_max)
        mu = torch.atan2(b, a)
        return CoordParams(du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig,
                           mu, kappa, spline)

    # -- the ln z density, in ONE place --------------------------------------
    # Three call sites need it (likelihood, PIT, sampler) and a fourth needs a point
    # summary. Each dispatches here rather than testing `lnz_head` itself, which is what
    # keeps the sampler from ever drawing from a density the likelihood does not
    # normalize — the failure `lnz_support="physical"` was introduced to remove.
    def _lnz_logprob(self, p: CoordParams, lnz, cx):
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            return gauss_logpdf(lnz, p.lnz_mean, p.lnz_sig)
        lo, hi = bounds
        x = lnz.clamp(min=lo, max=hi)
        if p.lnz_spline is None:
            return trunc_normal_logpdf(x, p.lnz_mean, p.lnz_sig, lo, hi)
        return rq_interval_logpdf(x, lo, hi, p.lnz_spline, self.lnz_spline_bins)

    def _lnz_cdf(self, p: CoordParams, lnz, cx):
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            return gauss_cdf(lnz, p.lnz_mean, p.lnz_sig)
        lo, hi = bounds
        x = torch.clamp(lnz, min=lo, max=hi)
        if p.lnz_spline is None:
            return trunc_normal_cdf(x, p.lnz_mean, p.lnz_sig, lo, hi)
        return rq_interval_cdf(x, lo, hi, p.lnz_spline, self.lnz_spline_bins)

    def _lnz_point(self, p: CoordParams, cx):
        """The ln z a MODE-based decode reports.

        For the truncated normal that is the untruncated mean clamped into the support —
        its mode, and bit-identical to what this path did before. A spline density has no
        closed-form mode, so the **median** `F^{-1}(1/2)` is reported instead: it is
        exact, monotone-equivariant, and cheap. The distinction only reaches the MAP
        decode, which is not the fielded point estimate (the MBR medoid is, and it carries
        genuine sampled coordinates)."""
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            return p.lnz_mean
        lo, hi = bounds
        if p.lnz_spline is None:
            return torch.clamp(p.lnz_mean, min=lo, max=hi)
        half = torch.full(p.lnz_spline.shape[:-1], 0.5, device=p.lnz_spline.device,
                          dtype=p.lnz_spline.dtype)
        return rq_interval_icdf(half, lo, hi, p.lnz_spline, self.lnz_spline_bins)

    # -- WP-A: the physical ln z support -------------------------------------
    def lnz_bounds(self, cx):
        """`(lo, hi)` of the `ln z` support for nodes in the cells whose centres are
        `cx`, or `None` in `legacy` mode.

        Soft Drop keeps a splitting iff `z > z_cut (DeltaR/R)^beta` (Larkoski et al.,
        arXiv:1402.2657; RSD: Dreyer et al., arXiv:1804.03657), i.e. — with this repo's
        `u = ln(1/DeltaR)` convention and `R = 1` — `ln z > ln z_cut - beta*u`; and
        `z = min(pT1,pT2)/(pT1+pT2) <= 1/2` by construction. So

            ln z  in  ( ln z_cut - beta*u ,  ln(1/2) ].

        The bound is made **cell-conditional**, not node-conditional: it is evaluated at
        the `u` in the cell that makes it LOOSEST,

            lo = min_{|u - cx| <= half_u} (ln z_cut - beta*u)
               = ln z_cut - beta*cx - |beta|*half_u,

        so every truth in the cell lies inside it. That is what keeps the coordinate
        likelihood a product of independent-given-cell factors: a bound that read the
        node's own drawn `u` would couple `ln z` to `du`, which this factorization
        cannot express (the per-node joint flow of `PLAN_UPDATES.md` WP1 is where that
        coupling belongs — plan §12's trigger).

        At the fielded `beta = 0` the two coincide exactly and the bound is the constant
        `(ln z_cut, ln 1/2]`; for `beta != 0` the residual slack is `|beta|*half_u`, and
        the WP-D support audit measures what leaks through it rather than assuming."""
        if not self.lnz_physical:
            return None
        lo = self._ln_zcut - self.lnz_beta * cx - abs(self.lnz_beta) * self.half_u
        # `hi` is materialised as a tensor rather than left a float: torch.clamp takes
        # two Tensors or two Numbers, never one of each, and every caller here clamps
        # against the pair.
        return lo, torch.full_like(lo, self._ln_half)

    def _coord_logprob(self, params: CoordParams, u, v, lnz, psi, cx, cy):
        p = params
        du = (u - cx).clamp(-self.half_u, self.half_u)
        dv = (v - cy).clamp(-self.half_v, self.half_v)
        ll = trunc_normal_logpdf(du, p.du_mean, p.du_sig, -self.half_u, self.half_u)
        ll = ll + trunc_normal_logpdf(dv, p.dv_mean, p.dv_sig, -self.half_v, self.half_v)
        ll = ll + self._lnz_logprob(p, lnz, cx)
        ll = ll + vonmises_logpdf(psi, p.mu, p.kappa)
        return ll

    # -- likelihood ----------------------------------------------------------
    def nll_terms(self, batch) -> dict[str, torch.Tensor]:
        """The three additive log-likelihood terms, per jet, before they are summed.

        `per_jet_nll` is `-(length_ll + split_ll + coord_ll)`; this returns the parts, so
        a report can say WHICH term moved. That distinction is not cosmetic when the
        geometry changes: the total with `continuous_coords=True` is a density on the
        (ln 1/DeltaR, ln kt) plane and is commensurable across `n_bins`, but `split_ll`
        alone is a probability over cells and shifts by `2*ln(n_bins_new/n_bins_old)` per
        emission for free (docs/PLAN_prod_test_v0.md check 1). Also returns `n_emissions`,
        since two of the three terms are per-emission and only comparable divided by it.
        """
        xf, nx = batch["xf"], batch["nx"]
        yc, ny = batch["yc"], batch["ny"]
        yraw = batch["yraw"]
        e = self.encode(xf, nx)
        out = self._decode_states(yc, e, self.xattn_kv(xf, nx))
        B, Lp1, _ = out.shape
        L = Lp1 - 1
        dev = yc.device

        eh = torch.cat([out, e.unsqueeze(1).expand(-1, Lp1, -1)], dim=-1)
        n = ny.unsqueeze(1)

        # length term: explicit categorical q(N|x) (v3) or the implicit continue/stop
        # product (default). The off-branch is unchanged, so -per_jet_nll matches today.
        if self.use_multiplicity_head:
            n_lp = F.log_softmax(self.n_head(e), dim=-1)  # (B, max+1)
            length_ll = n_lp.gather(-1, ny.clamp(max=self.max_emissions).unsqueeze(-1)).squeeze(-1)
        else:
            cont_logit = self.cont_head(eh).squeeze(-1)  # (B, L+1)
            idx = torch.arange(Lp1, device=dev).unsqueeze(0)
            cont_mask = (idx <= n).float()
            cont_tgt = (idx < n).float()
            cont_ll = -F.binary_cross_entropy_with_logits(cont_logit, cont_tgt, reduction="none")
            length_ll = (cont_ll * cont_mask).sum(1)

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

        return {"length_ll": length_ll, "split_ll": split_ll, "coord_ll": coord_ll,
                "n_emissions": ny.to(length_ll.dtype)}

    def per_jet_nll(self, batch) -> torch.Tensor:
        t = self.nll_terms(batch)
        return -(t["length_ll"] + t["split_ll"] + t["coord_ll"])

    def log_prob(self, batch) -> torch.Tensor:
        return -self.per_jet_nll(batch)

    # -- WP2: per-coordinate PIT ---------------------------------------------
    @torch.inference_mode()
    def coordinate_cdfs(self, batch) -> dict | None:
        """Teacher-forced probability-integral transform of the four true coordinates.

        The AR coordinate head is a product of closed-form 1-d densities, so its PIT is
        exact and lives in PHYSICAL coordinates: the truncated-normal CDF for the
        within-cell offsets (du, dv) — the same normalizer `trunc_normal_logpdf` divides
        by — the normal CDF for ln z, and the von Mises CDF for psi. v1
        (`continuous_coords=False`) has no coordinate density, so it returns None."""
        if not self.continuous_coords:
            return None
        xf, nx, yc, ny, yraw = (batch["xf"], batch["nx"], batch["yc"], batch["ny"], batch["yraw"])
        B, L = yc.shape
        if L == 0:
            empty = torch.zeros(B, 0, 4, device=yc.device)
            return {"names": _COORD_NAMES, "u": empty, "mask": empty[..., 0].bool(),
                    "space": "physical"}
        e = self.encode(xf, nx)
        out = self._decode_states(yc, e, self.xattn_kv(xf, nx))
        eh_t = torch.cat([out[:, :L, :], e.unsqueeze(1).expand(-1, L, -1)], dim=-1)
        p = self._coord_params(torch.cat([eh_t, self.y_embed(yc.clamp(min=0))], dim=-1))
        cx, cy = self.cell_cx[yc], self.cell_cy[yc]
        du = (yraw[..., 0] - cx).clamp(-self.half_u, self.half_u)
        dv = (yraw[..., 1] - cy).clamp(-self.half_v, self.half_v)
        # the SAME object the likelihood normalizes by, whichever ln z head is fielded
        lnz_pit = self._lnz_cdf(p, yraw[..., 2], cx)
        u = torch.stack(
            [
                trunc_normal_cdf(du, p.du_mean, p.du_sig, -self.half_u, self.half_u),
                trunc_normal_cdf(dv, p.dv_mean, p.dv_sig, -self.half_v, self.half_v),
                lnz_pit,
                vonmises_cdf(yraw[..., 3], p.mu, p.kappa),
            ],
            dim=-1,
        )
        mask = torch.arange(L, device=yc.device).unsqueeze(0) < ny.unsqueeze(1)
        return {"names": _COORD_NAMES, "u": u, "mask": mask, "space": "physical"}

    # -- single-jet / batched decoder steps ---------------------------------
    # These maintain their own GRU stepping rather than calling `_decode_states`, so
    # the cross-attention residual is mirrored here explicitly (WP3). `kv` is the
    # per-jet key/value pair from `xattn_kv`, threaded through by the decode entry
    # points; None (the default) is exactly the pre-WP3 arithmetic.
    def _step_core(self, tok: torch.Tensor, e: torch.Tensor, h, kv):
        inp = self.dec_in(torch.cat([self.y_embed(tok), e.unsqueeze(1)], dim=-1))
        out, h = self.decoder(inp, h)
        out = self._apply_xattn(out, kv)
        return torch.cat([out[:, -1, :], e], dim=-1), h

    def _step(self, tok: torch.Tensor, e: torch.Tensor, h, kv=None):
        hv, h = self._step_core(tok, e, h, kv)
        p_cont = torch.sigmoid(self.cont_head(hv)).item()
        logp_split = F.log_softmax(self.split_head(hv), dim=-1).squeeze(0)
        return p_cont, logp_split, h

    def _step_batched(self, tok: torch.Tensor, e: torch.Tensor, h, kv=None):
        """The SAMPLING step. `decode.continue_temperature` is applied here and only
        here, which is what makes it a decode-layer object: `_step` (beam search),
        `nll_terms` and `describe_sequence` read `cont_head` directly, so the trained
        likelihood and the MAP are untouched by it. The default 1.0 skips the branch
        entirely rather than dividing by one, so the off path is bit-identical."""
        hv, h = self._step_core(tok, e, h, kv)
        cont_logit = self.cont_head(hv)
        if self.continue_temperature != 1.0:
            cont_logit = cont_logit / self.continue_temperature
        p_cont = torch.sigmoid(cont_logit).squeeze(-1)
        split_logits = self.split_head(hv)
        return p_cont, split_logits, h

    def _step_cells(self, tok: torch.Tensor, e: torch.Tensor, h, kv=None):
        """cont_head-free batched decoder step (multiplicity-head model): returns
        ``(split_logits (K, n_cells), h)``. Length is set externally by q(N|x)."""
        hv, h = self._step_core(tok, e, h, kv)
        return self.split_head(hv), h

    @staticmethod
    def _bind(step_fn, kv):
        """Bind `kv` into a step so the generic `(tok, e, h)` signature the samplers
        and beam search expect is preserved."""
        if kv is None:
            return step_fn
        return lambda tok, e, h: step_fn(tok, e, h, kv)

    # -- contract: sample ----------------------------------------------------
    @torch.inference_mode()
    def sample(self, xf, nx, n, max_emissions: int = 25, cont_temperature: float = 1.0):
        self.eval()
        dev = xf.device
        e = self.encode(xf, nx).expand(n, -1).contiguous()
        h0 = self._init_hidden(e)
        kv = self.xattn_kv(xf, nx)  # once per jet, broadcast across the n draws
        if self.use_multiplicity_head:
            # first-class factorization: N_k ~ q(N|x), then decode exactly N_k cells.
            n_probs = F.softmax(self.recalibrated_n_logits(self.n_head(e[:1])),
                                dim=-1).squeeze(0)  # (max+1,)
            lengths = torch.multinomial(n_probs, n, replacement=True).clamp(max=max_emissions)
            return ancestral_sample_cells_fixed_length(
                self._bind(self._step_cells, kv), e, h0, self.start_token, lengths, dev,
                cont_temperature=cont_temperature,
            )
        return ancestral_sample_cells(
            self._bind(self._step_batched, kv), e, h0, self.start_token, n, dev,
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
        kv = self.xattn_kv(xf, nx)
        if self.use_multiplicity_head:
            return self._map_decode_fixed_length(
                e, h0, min_emissions=min_emissions, max_emissions=max_emissions, kv=kv
            )
        return beam_search_cells(
            self._bind(self._step, kv), e, h0, self.start_token, xf.device,
            beam_width=beam_width, topk_cells=topk_cells, max_emissions=max_emissions,
            min_emissions=min_emissions, length_penalty=length_penalty,
        )

    def _map_decode_fixed_length(self, e, h0, *, min_emissions: int, max_emissions: int,
                                 kv=None):
        """q(N|x)-head MAP: N* = clamp(argmax q(N|x), min_emissions, max_emissions),
        then greedily decode exactly N* cells (argmax split per step; no stop head).
        With the default `min_emissions=1` the estimate is never the empty tree."""
        dev = e.device
        n_lp = F.log_softmax(self.n_head(e), dim=-1).squeeze(0)
        n_star = int(n_lp.argmax().item())
        n_star = max(n_star, int(min_emissions))
        n_star = min(n_star, int(max_emissions), self.max_emissions)
        tok = torch.full((1, 1), self.start_token, dtype=torch.long, device=dev)
        h, cells = h0, []
        for _ in range(n_star):
            split_logits, h = self._step_cells(tok, e, h, kv)
            c = int(split_logits.argmax(dim=-1).item())
            cells.append(c)
            tok = torch.tensor([[c]], dtype=torch.long, device=dev)
        return cells

    # -- contract: exact skeleton enumeration (docs/PLAN_ModeMassAudit.md WP-2) ----
    @torch.inference_mode()
    def skeleton_search_spec(self, xf, nx):
        """The AR family's two skeleton factorizations, as one spec.

        Without the multiplicity head the length is the per-step continue/stop product,
        so the search is the plan's WP-1 verbatim on the SAME `_step` that
        `beam_search_cells` consumes — the audit adds no model code and touches no
        coordinate head. With it (`v3`/`v1_nhead`, where the G8 winner lives) the
        factorization is `q(N|x) prod_t P_split(c_t | h_t, e; N)`, so a fixed-length
        search runs per N off `_step_cells` and the merge happens on one heap.

        `continue_temperature` is deliberately NOT applied: it is a SAMPLING knob
        (`_step_batched`), and the audit enumerates the posterior the likelihood
        defines, exactly as the beam-search MAP does."""
        from ..inference.mode_audit import SkeletonSearchSpec

        self.eval()
        e = self.encode(xf, nx)
        kv = self.xattn_kv(xf, nx)
        h0 = self._init_hidden(e)
        if self.use_multiplicity_head:
            return SkeletonSearchSpec(
                kind="nhead", e=e, h0=h0, start_token=self.start_token,
                step_cells=self._bind(self._step_cells, kv),
                log_qn=F.log_softmax(self.recalibrated_n_logits(self.n_head(e)),
                                     dim=-1).squeeze(0),
                max_emissions=self.max_emissions, family="ar_junipr_nhead",
            )
        return SkeletonSearchSpec(
            kind="ar", e=e, h0=h0, start_token=self.start_token,
            step=self._bind(self._step, kv),
            max_emissions=self.max_emissions, family="ar_junipr",
        )

    @torch.inference_mode()
    def sample_coordinates(self, xf, nx, cells, *, generator=None):
        """A genuine DRAW from every coordinate head, teacher-forced on `cells`:

            du, dv ~ TruncNormal(mean, sigma) on (+-half_u, +-half_v) — the head's support
            ln z   ~ Normal(mean, sigma)
            psi    ~ vonMises(mu, kappa)

        Same replay as `describe_sequence` (encode -> `_decode_states` ->
        `_coord_params`), but sampled rather than moded. The distinction is the whole
        point: a per-jet argmax is not a draw, and for ln z and psi the mode IS the
        entire prediction, so a mode-based posterior-predictive series would carry
        exactly the shrinkage such a series exists to expose.

        v1 (`continuous_coords=False`) has no coordinate head, so it returns None and
        callers fall back to cell centres. Reproducibility is the global torch RNG
        (`train.trainer.seed_everything`) unless a `generator` is supplied.

        One jet's K draws go through `sample_coordinates_many` instead: this call
        re-runs `encode()` and `xattn_kv()` every time, which is ~57% of it at 20
        threads and produces the same tensor K times."""
        if not self.continuous_coords:
            return None
        dev = xf.device
        if not len(cells):
            return torch.zeros(0, 4, device=dev)
        p = self.coord_head_params(xf, nx, cells)
        yc = torch.tensor([int(c) for c in cells], dtype=torch.long, device=dev)
        du = trunc_normal_sample(p.du_mean, p.du_sig, -self.half_u, self.half_u,
                                 generator=generator)
        dv = trunc_normal_sample(p.dv_mean, p.dv_sig, -self.half_v, self.half_v,
                                 generator=generator)
        lnz = self._sample_lnz(p, self.cell_cx[yc], generator=generator)
        psi = vonmises_sample(p.mu, p.kappa, generator=generator)
        return torch.stack(
            [self.cell_cx[yc] + du, self.cell_cy[yc] + dv, lnz, psi], dim=-1
        )

    @torch.inference_mode()
    def sample_coordinates_many(self, xf, nx, draws, *, generator=None):
        """One jet's K draws' coordinates in ONE forward pass — the batched sibling of
        `sample_coordinates`, returning a `(L_k, 4)` tensor per draw.

        Same three densities, same teacher-forced replay; the only difference is that
        `encode()` and `xattn_kv()` run once instead of K times and the whole
        `(K, L_max)` block of cells goes through `_decode_states` / `_coord_params` /
        the samplers as one batch. Rows are padded to `L_max` with cell 0 and sliced
        back afterwards, so a padded position costs one wasted sample and reaches no
        caller.

        **This reorders RNG consumption** relative to the per-draw loop: the draws are
        from the same conditional and agree in distribution, but they are not the same
        numbers, so anything downstream shifts within Monte-Carlo noise. v1
        (`continuous_coords=False`) returns a list of None, which is what preserves
        every caller's `c is None -> no coordinate density` degradation path."""
        if not self.continuous_coords:
            return [None] * len(draws)
        dev = xf.device
        lens = [len(d) for d in draws]
        K = len(lens)
        if K == 0:
            return []
        empty = torch.zeros(0, 4, device=dev)
        L_max = max(lens)
        if L_max == 0:                       # every draw is the empty tree
            return [empty for _ in range(K)]
        self.eval()
        yc = torch.zeros(K, L_max, dtype=torch.long, device=dev)
        for k, d in enumerate(draws):
            if lens[k]:
                yc[k, : lens[k]] = torch.as_tensor([int(c) for c in d],
                                                   dtype=torch.long, device=dev)
        p = self._coord_params_padded(xf, nx, yc)
        du = trunc_normal_sample(p.du_mean, p.du_sig, -self.half_u, self.half_u,
                                 generator=generator)
        dv = trunc_normal_sample(p.dv_mean, p.dv_sig, -self.half_v, self.half_v,
                                 generator=generator)
        lnz = self._sample_lnz(p, self.cell_cx[yc], generator=generator)
        psi = vonmises_sample(p.mu, p.kappa, generator=generator)
        coords = torch.stack(
            [self.cell_cx[yc] + du, self.cell_cy[yc] + dv, lnz, psi], dim=-1
        )
        return [coords[k, : lens[k]] if lens[k] else empty for k in range(K)]

    def _sample_lnz(self, p: CoordParams, cx, *, generator=None):
        """One `ln z` draw per element: the unbounded Normal in `legacy` mode, the
        cell-conditional truncated normal in `physical` mode, and the spline-warped
        truncated normal when `lnz_head="spline"`.

        The samplers are paired with the densities of `_lnz_logprob` here, in one place,
        so a draw can never come from a distribution the likelihood does not normalize —
        which is exactly the failure `physical` mode exists to remove (v0's 0.88%
        soft-drop violations came from sampling a Normal whose support the grooming
        forbids). The spline path inherits that guarantee for free: it inverts the SAME
        composed CDF `_lnz_cdf` reports."""
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            return p.lnz_mean + p.lnz_sig * torch.randn(
                p.lnz_mean.shape, device=p.lnz_mean.device, dtype=p.lnz_mean.dtype,
                generator=generator,
            )
        lo, hi = bounds
        if p.lnz_spline is None:
            return trunc_normal_sample(p.lnz_mean, p.lnz_sig, lo, hi, generator=generator)
        return rq_interval_sample(lo, hi, p.lnz_spline, self.lnz_spline_bins,
                                  generator=generator)

    def _coord_params_padded(self, xf, nx, yc: torch.Tensor):
        """The eight `_coord_params` tensors, each `(B, L)`, for a `(B, L)` block of
        cell chains teacher-forced on ONE jet's conditioning.

        The body `coord_head_params` (B = 1) and `sample_coordinates_many` (B = K)
        share: encode once, decode the block, concatenate the cell embedding, read the
        heads. Factored rather than duplicated so the batched path cannot drift from
        the single-draw one."""
        B, L = yc.shape
        e = self.encode(xf, nx)
        if e.shape[0] != B:  # one jet's context, broadcast over the block's rows
            e = e.expand(B, -1)
        out = self._decode_states(yc, e, self.xattn_kv(xf, nx))
        eh = torch.cat([out, e.unsqueeze(1).expand(-1, L + 1, -1)], dim=-1)[:, :L, :]
        return self._coord_params(torch.cat([eh, self.y_embed(yc)], dim=-1))

    @torch.inference_mode()
    def coord_head_params(self, xf, nx, cells) -> CoordParams | None:
        """The coordinate-head parameters for `cells`, teacher-forced, each `(L,)` — the
        `CoordParams` `sample_coordinates` draws from.

        Public because the von Mises `kappa` is a diagnostic in its own right: the psi
        MODE that MAP/MBR report is near-arbitrary wherever kappa is small, so a psi
        panel can only be read beside its kappa distribution. None for v1."""
        if not self.continuous_coords:
            return None
        cells = [int(c) for c in cells]
        dev = xf.device
        if not cells:
            empty = torch.zeros(0, device=dev)
            lnz = (None, None) if self.lnz_spline else (empty, empty)
            spline = (torch.zeros(0, self.lnz_n_params, device=dev)
                      if self.lnz_spline else None)
            return CoordParams(empty, empty, empty, empty, *lnz, empty, empty, spline)
        self.eval()
        yc = torch.tensor([cells], dtype=torch.long, device=dev)
        return self._coord_params_padded(xf, nx, yc).apply(lambda t: t.squeeze(0))

    @torch.inference_mode()
    def describe_sequence(self, xf, nx, cells, coords=None, *, generator=None
                          ) -> LundPointEstimate:
        """Attach continuous coordinates and the per-node + total log-density to a
        primary cell sequence. The total is the full joint log-density **of the returned
        configuration** — that identity is the contract, and it is what forces the
        log-density to be re-evaluated whenever a coordinate is not the head's mode.

        Three coordinate sources (docs/PLAN_prod_test_v1.md WP-C):

        * `coords` given — an `(L, 4)` table the caller already drew. This is the MBR
          medoid path: the medoid IS a posterior sample, so its own coordinates are
          carried verbatim and `psi_identified` is `None` (mode identifiability is not
          a question about a draw).
        * head modes (`coords=None`, the staged MAP), **except** where the von Mises
          `kappa` falls below `decode.kappa_min_mode`. There the mode is the direction
          of a near-zero resultant — arbitrary — so a DRAW is substituted and the node
          is flagged `psi_identified=False`. v0 is the case in point: median
          kappa = 0.022 (peak/trough 1.04) yet MAP/MBR reported a psi resultant
          |R| = 0.69 against a truth of 0.045, a 17.5x pooled row that inflated both
          decode gmeans.
        * cell centres, for v1 (`continuous_coords=False`), which has no coordinate head.
        """
        self.eval()
        dev = xf.device
        L = len(cells)
        e = self.encode(xf, nx)
        yc = torch.tensor([list(cells)], dtype=torch.long, device=dev)
        out = self._decode_states(yc, e, self.xattn_kv(xf, nx))
        eh = torch.cat([out, e.unsqueeze(1).expand(-1, L + 1, -1)], dim=-1)
        if self.use_multiplicity_head:
            # length log-density = log q(N=L | x); no per-node continue/stop terms.
            n_lp = F.log_softmax(self.n_head(e), dim=-1).squeeze(0)
            length_logp = float(n_lp[min(L, self.max_emissions)])
            logp_cont = torch.zeros(L + 1, device=dev)
        else:
            cont_logit = self.cont_head(eh).squeeze(-1).squeeze(0)
            logp_cont = F.logsigmoid(cont_logit)
            length_logp = float(F.logsigmoid(-cont_logit)[L])  # log P(stop at L)

        nodes, total = [], 0.0
        if L > 0:
            eh_t = eh[:, :L, :]
            split_lp = F.log_softmax(self.split_head(eh_t), dim=-1).squeeze(0)
            chosen = split_lp.gather(-1, yc[0].unsqueeze(-1)).squeeze(-1)
            cx, cy = self.cell_cx[yc], self.cell_cy[yc]

            kappa_col = [None] * L
            psi_flag: list = [None] * L
            if self.continuous_coords:
                cell_emb = self.y_embed(yc)
                params = self._coord_params(torch.cat([eh_t, cell_emb], dim=-1))
                du_mean, dv_mean, mu, kappa = (params.du_mean, params.dv_mean,
                                               params.mu, params.kappa)
                kappa_col = [float(kappa[0, t]) for t in range(L)]
                if coords is not None:
                    c4 = torch.as_tensor(coords, dtype=cx.dtype, device=dev).reshape(1, L, 4)
                    u_mode, v_mode = c4[..., 0], c4[..., 1]
                    lnz_mean, mu = c4[..., 2], c4[..., 3]
                    src = "sample"                       # psi_flag stays None throughout
                else:
                    u_mode, v_mode = cx + du_mean, cy + dv_mean
                    # The mode of a TRUNCATED normal is its untruncated mean clamped into
                    # the support, so `physical` mode moves the reported ln z inside the
                    # grooming boundary by construction, not by a downstream repair. The
                    # spline head has no closed-form mode and reports its median instead
                    # (`_lnz_point`); both stay inside the support by construction.
                    lnz_mean = self._lnz_point(params, cx)
                    src = "mode"
                    # --- WP-C.2: the psi mode is reported only where it is identified ---
                    weak = kappa < self.kappa_min_mode
                    if bool(weak.any()):
                        g = self.decode_generator(dev) if generator is None else generator
                        drawn = vonmises_sample(mu, kappa, generator=g)
                        mu = torch.where(weak, drawn, mu)
                        psi_flag = [(not bool(weak[0, t])) for t in range(L)]
                    else:
                        psi_flag = [True] * L
                # Re-evaluated at whatever is actually reported, so `logprob` remains the
                # joint density of the returned configuration in all three branches.
                coord_per = self._coord_logprob(
                    params, u_mode, v_mode, lnz_mean, mu, cx, cy
                ).squeeze(0)
            else:
                u_mode, v_mode = cx, cy
                lnz_mean = torch.zeros_like(cx)
                mu = torch.zeros_like(cx)
                coord_per = torch.zeros(L, device=dev)
                src = "cell_center"

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
                        kappa=kappa_col[t], psi_identified=psi_flag[t],
                    )
                )
        else:
            src = "sample" if coords is not None else ("mode" if self.continuous_coords
                                                       else "cell_center")
        total += length_logp
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=L,
                                 coords_source=src)

    def describe_cells(self, xf, nx, cells, coords=None, *, generator=None
                       ) -> LundPointEstimate:
        """One posterior DRAW (the MBR medoid) -> LundPointEstimate, carrying its own
        sampled coordinates.

        This used to re-attach the head modes, which forfeited the one property that
        makes the medoid worth having: it is a genuine posterior sample, and a sample
        with its modes pasted back on is neither a sample nor the MAP. v0 measured the
        cost — a psi resultant |R| = 0.69 against a truth of 0.045, from a head whose
        median kappa is 0.022 — and `PLAN_prod_test_v1.md` WP-C.1 is the repair.

        `coords` given (the caller already drew them alongside the cells) is used
        verbatim; otherwise one draw is taken here. `map_estimate` is unaffected: it
        goes through `describe_sequence` with `coords=None`, which is still the staged
        mode decode."""
        cells = [int(c) for c in cells]
        if coords is None and cells and self.continuous_coords:
            # the decode stream, not the global one — see `PosteriorModel.decode_generator`
            g = self.decode_generator(xf.device) if generator is None else generator
            coords = self.sample_coordinates(xf, nx, cells, generator=g)
        return self.describe_sequence(xf, nx, cells, coords, generator=generator)

    @torch.inference_mode()
    def length_pmf(self, xf, nx, mults=None, n_samples: int = 500):
        """P(n|x): exact `softmax(n_head(e))` when the multiplicity head is on;
        otherwise the base sampler histogram (the implicit continue/stop belief)."""
        if not self.use_multiplicity_head:
            return super().length_pmf(xf, nx, mults=mults, n_samples=n_samples)
        self.eval()
        e = self.encode(xf, nx)
        return F.softmax(self.recalibrated_n_logits(self.n_head(e)),
                         dim=-1).squeeze(0).cpu().numpy()

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
