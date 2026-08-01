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

from ..distributions import (
    gauss_cdf,
    gauss_logpdf,
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

    def _coord_logprob(self, params, u, v, lnz, psi, cx, cy):
        du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, mu, kappa = params
        du = (u - cx).clamp(-self.half_u, self.half_u)
        dv = (v - cy).clamp(-self.half_v, self.half_v)
        ll = trunc_normal_logpdf(du, du_mean, du_sig, -self.half_u, self.half_u)
        ll = ll + trunc_normal_logpdf(dv, dv_mean, dv_sig, -self.half_v, self.half_v)
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            ll = ll + gauss_logpdf(lnz, lnz_mean, lnz_sig)
        else:
            lo, hi = bounds
            ll = ll + trunc_normal_logpdf(lnz.clamp(min=lo, max=hi), lnz_mean, lnz_sig, lo, hi)
        ll = ll + vonmises_logpdf(psi, mu, kappa)
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
        params = self._coord_params(torch.cat([eh_t, self.y_embed(yc.clamp(min=0))], dim=-1))
        du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, mu, kappa = params
        cx, cy = self.cell_cx[yc], self.cell_cy[yc]
        du = (yraw[..., 0] - cx).clamp(-self.half_u, self.half_u)
        dv = (yraw[..., 1] - cy).clamp(-self.half_v, self.half_v)
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            lnz_pit = gauss_cdf(yraw[..., 2], lnz_mean, lnz_sig)
        else:  # the SAME truncated normal the likelihood normalizes by, as for du/dv
            lo, hi = bounds
            lnz_pit = trunc_normal_cdf(
                torch.clamp(yraw[..., 2], min=lo, max=hi), lnz_mean, lnz_sig, lo, hi
            )
        u = torch.stack(
            [
                trunc_normal_cdf(du, du_mean, du_sig, -self.half_u, self.half_u),
                trunc_normal_cdf(dv, dv_mean, dv_sig, -self.half_v, self.half_v),
                lnz_pit,
                vonmises_cdf(yraw[..., 3], mu, kappa),
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
        hv, h = self._step_core(tok, e, h, kv)
        p_cont = torch.sigmoid(self.cont_head(hv)).squeeze(-1)
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
        du_m, dv_m, du_s, dv_s, lnz_m, lnz_s, mu, kappa = self.coord_head_params(xf, nx, cells)
        yc = torch.tensor([int(c) for c in cells], dtype=torch.long, device=dev)
        du = trunc_normal_sample(du_m, du_s, -self.half_u, self.half_u, generator=generator)
        dv = trunc_normal_sample(dv_m, dv_s, -self.half_v, self.half_v, generator=generator)
        lnz = self._sample_lnz(lnz_m, lnz_s, self.cell_cx[yc], generator=generator)
        psi = vonmises_sample(mu, kappa, generator=generator)
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
        du_m, dv_m, du_s, dv_s, lnz_m, lnz_s, mu, kappa = self._coord_params_padded(xf, nx, yc)
        du = trunc_normal_sample(du_m, du_s, -self.half_u, self.half_u, generator=generator)
        dv = trunc_normal_sample(dv_m, dv_s, -self.half_v, self.half_v, generator=generator)
        lnz = self._sample_lnz(lnz_m, lnz_s, self.cell_cx[yc], generator=generator)
        psi = vonmises_sample(mu, kappa, generator=generator)
        coords = torch.stack(
            [self.cell_cx[yc] + du, self.cell_cy[yc] + dv, lnz, psi], dim=-1
        )
        return [coords[k, : lens[k]] if lens[k] else empty for k in range(K)]

    def _sample_lnz(self, lnz_m, lnz_s, cx, *, generator=None):
        """One `ln z` draw per element: the unbounded Normal in `legacy` mode, the
        cell-conditional truncated normal in `physical` mode.

        The two samplers are paired with the two densities in `_coord_logprob` here, in
        one place, so a draw can never come from a distribution the likelihood does not
        normalize — which is exactly the failure `physical` mode exists to remove (v0's
        0.88% soft-drop violations came from sampling a Normal whose support the
        grooming forbids)."""
        bounds = self.lnz_bounds(cx)
        if bounds is None:
            return lnz_m + lnz_s * torch.randn(
                lnz_m.shape, device=lnz_m.device, dtype=lnz_m.dtype, generator=generator
            )
        lo, hi = bounds
        return trunc_normal_sample(lnz_m, lnz_s, lo, hi, generator=generator)

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
    def coord_head_params(self, xf, nx, cells):
        """The eight coordinate-head parameters for `cells`, teacher-forced, each `(L,)`
        — the `_coord_params` tuple `sample_coordinates` draws from.

        Public because the von Mises `kappa` is a diagnostic in its own right: the psi
        MODE that MAP/MBR report is near-arbitrary wherever kappa is small, so a psi
        panel can only be read beside its kappa distribution. None for v1."""
        if not self.continuous_coords:
            return None
        cells = [int(c) for c in cells]
        dev = xf.device
        if not cells:
            return tuple(torch.zeros(0, device=dev) for _ in range(8))
        self.eval()
        yc = torch.tensor([cells], dtype=torch.long, device=dev)
        return tuple(p.squeeze(0) for p in self._coord_params_padded(xf, nx, yc))

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

            if self.continuous_coords:
                cell_emb = self.y_embed(yc)
                params = self._coord_params(torch.cat([eh_t, cell_emb], dim=-1))
                du_mean, dv_mean, _, _, lnz_mean, _, mu, _ = params
                u_mode, v_mode = cx + du_mean, cy + dv_mean
                # The mode of a TRUNCATED normal is its untruncated mean clamped into the
                # support, so `physical` mode moves the reported ln z inside the grooming
                # boundary by construction rather than by a downstream repair.
                bounds = self.lnz_bounds(cx)
                lnz_mode = (lnz_mean if bounds is None
                            else torch.clamp(lnz_mean, min=bounds[0], max=bounds[1]))
                coord_per = self._coord_logprob(params, u_mode, v_mode, lnz_mode, mu, cx, cy).squeeze(0)
                lnz_mean = lnz_mode
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
        total += length_logp
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=L)

    def describe_cells(self, xf, nx, cells) -> LundPointEstimate:
        """MBR winner -> LundPointEstimate. AR attaches the head-mode continuous
        coordinates and the exact joint log-density (its staged decode), richer than
        the base cell-centre fallback."""
        return self.describe_sequence(xf, nx, cells)

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
