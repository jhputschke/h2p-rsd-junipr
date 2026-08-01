"""§5.5 edit transducer: hadron -> parton as a learned smearing + birth/death process
(docs/PLAN_EditTransducer.md).

The registry's fourth family, and the only one that does not generate `y` from scratch.
Every other family conditions on `x` through the encoder alone, so the decoder has to
*relearn* that `y ~ x` wherever hadronization is weak — which the closure suite already
prices: `dlund_identity` (treat `x` as the truth) beats `dlund_posterior_mode`. Here the
hadron tree is the **anchor** of the parton tree: the factorization of `q(y|x)` changes,
the target, the objective and the geometry do not.

    state (i, j): i hadron nodes consumed, j parton nodes emitted
      i <  nx : categorical {ADVANCE, EMIT}
      i == nx : categorical {STOP,    EMIT}      (trailing insertions)
      EMIT    : y_{j+1} ~ p_anch . f_shift(. | x_i)  +  (1 - p_anch) . f_free(.)

An ADVANCE with no anchored emit at that column is a **deletion**, an anchored emit is a
**kept, smeared** node, a free emit is an **insertion**. That is the RNN-T lattice
(Graves, arXiv:1211.3711), so `sum_y q(y|x) = 1` holds by construction and
`exact_likelihood = True` is earned rather than asserted. The alignment is latent and
marginalised by dynamic programming (`edit_dp`) — node-level parton<->hadron
correspondence is not observable (HOMER, arXiv:2410.06342; Assi et al.,
arXiv:2503.05667), so it must never be a supervised target. With `n_x, n_y <~ 25` the
`O(n_x . n_y)` forward recursion is exact and cheap, which is why this family is
attractive here and not in NLP, where the same lattices force heuristic surrogates
(Insertion Transformer, arXiv:1902.03249; Levenshtein Transformer, arXiv:1905.11006).

Two consequences worth naming. The multiplicity is `n_y = n_x - #del + #ins`, so length
is anchored at `|x|` and the open-ended continue/stop mechanism — the seat of the
marginal multiplicity bias and of MAP collapse — is removed *structurally*. And
marginalising the coordinates out of the same recursion gives `q(N|x)` exactly, with no
extra parameters, which is what `length_pmf` returns.

**Physics-form widths (the point of the exercise).** The smearing scale of local
parton-hadron duality (Azimov, Dokshitzer, Khoze & Troyan, Z. Phys. C 27 (1985) 65) runs
as `Lambda_eff / k_t`, so the residual width is parametrized

    sigma(i) = sigma_0 + Lambda_eff . exp(-ln k_t^(i))

with learnable `(sigma_0, Lambda_eff)` rather than as a free MLP output. That makes the
learned kernel directly confrontable with the shape-function expectation
(arXiv:1906.11843) instead of opaque, and regularizes the low-statistics tail.
`model.physics_width=false` swaps in the free-MLP head as the ablation.

**Two stages**, selected by `model.prefix_conditioning`:

* `edit_v1` (pair-HMM). Ops and emissions conditioned on `(i, s_i, e)` only — zero
  exposure bias anywhere, at the cost of conditionally independent shifts.
* `edit_v2` (transducer). A prediction network over the emitted **cell** prefix feeds the
  emission heads, restoring recoil correlation among the `y` nodes.

The op (stay/emit) head is prefix-free in **both** stages. That is deliberate and is what
the plan means by "teacher forcing enters the prefix only — never the length": it is
exactly the condition under which the coordinates marginalise out of the lattice, hence
the condition for `length_pmf` to stay *exact*. Conditioning the prefix on the emitted
cells (not their continuous coordinates) is the matching choice on the emission side: it
keeps `sample_coordinates`' constrained forward-backward exact too, since the cell chain
it is given determines every prefix state.

Nothing in `ar_junipr.py` is touched, so likelihood parity is unaffected by construction.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributions import (
    gauss_logpdf,
    trunc_normal_cdf,
    trunc_normal_logpdf,
    trunc_normal_sample,
    vonmises_logpdf,
    vonmises_sample,
    wrap_to_pi,
)
from ..encoders.base import build_encoder
from ..features import N_NODE_FEAT, configured_aux_names
from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate
from . import edit_dp
from .ar_junipr import _mlp  # the same n-layer MLP the v2 split_head / coord_head use
from .base import PosteriorModel, register_model

# A hard "no anchor here" logit. `-inf` would put `inf - inf` into logaddexp's backward;
# this underflows to p_anch = 0 exactly while every gradient stays finite, and `where`
# already blocks the head's own gradient at these positions.
_NO_ANCHOR = -1e9


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class _EmitParams(NamedTuple):
    """Emission-head outputs at a set of lattice states. Every field carries the same
    leading shape as the state grid it was evaluated on — `(B, n_col, T)` in the batched
    likelihood, `(K,)` in the sampler — so the target coordinates broadcast against it."""

    log_p_anch: torch.Tensor
    log_1m_anch: torch.Tensor
    # anchored (smeared) component, centred on the hadron node of this column
    mu_u: torch.Tensor
    mu_v: torch.Tensor
    sig_u: torch.Tensor
    sig_v: torch.Tensor
    mu_z: torch.Tensor
    sig_z: torch.Tensor
    mu_psi: torch.Tensor
    kappa: torch.Tensor
    # free (insertion) component: cell categorical + within-cell offsets
    cell_lp: torch.Tensor
    f_du_m: torch.Tensor
    f_dv_m: torch.Tensor
    f_du_s: torch.Tensor
    f_dv_s: torch.Tensor
    f_lz_m: torch.Tensor
    f_lz_s: torch.Tensor
    f_psi_m: torch.Tensor
    f_kappa: torch.Tensor


@register_model("edit_v1", "edit_v2", "edit")
class EditTransducer(PosteriorModel):
    # The lattice normalizes by construction (RNN-T), so this is structural, not a claim.
    exact_likelihood = True
    # The exact prefix-conditional CDF is available from the same recursion as a
    # responsibility-weighted mixture of trunc_normal_cdf / gauss_cdf / vonmises_cdf;
    # it lands once the DP is trusted (docs/PLAN_EditTransducer.md), not before.
    supports_coordinate_pit = False
    has_continuous_coords = True

    def __init__(self, cfg, geometry: Geometry):
        super().__init__()
        m = cfg.model
        self.geometry = geometry
        self.n_cells = geometry.n_cells
        self.n_bins = geometry.n_bins
        self.ctx_dim = int(m.ctx_dim)
        self.sigma_floor = float(m.sigma_floor)
        self.kappa_max = float(m.kappa_max)
        self.max_emissions = int(m.max_emissions)
        self.physics_width = bool(getattr(m, "physics_width", True))
        self.prefix_conditioning = bool(getattr(m, "prefix_conditioning", False))
        self.half_u, self.half_v = geometry.half_u, geometry.half_v
        self.lo_u, self.hi_u = (float(t) for t in geometry.ln_invdelta_range)
        self.lo_v, self.hi_v = (float(t) for t in geometry.ln_kt_range)

        emb = int(cfg.encoder.emb_dim)
        self.emb_dim = emb

        # ---- encoder e(x) + its PER-NODE states -----------------------------
        self.aux_feature_names = configured_aux_names(cfg.encoder)
        n_in = N_NODE_FEAT + len(self.aux_feature_names)
        self.encoder_net = build_encoder(cfg.encoder, self.ctx_dim, n_in)
        if not getattr(self.encoder_net, "returns_sequence", False):
            raise ValueError(
                f"model={cfg.model.name!r} anchors every parton node on a HADRON node, so it "
                f"needs the encoder's per-node states, but encoder={cfg.encoder.name!r} has "
                f"returns_sequence=False. Implement Encoder.forward_seq there, or use "
                f"encoder=gru|lundnet|deepsets."
            )
        self.state_proj = nn.Linear(int(self.encoder_net.seq_dim), self.ctx_dim)
        # The terminal column i = nx has no hadron node behind it; this learned state is
        # what the op head reads there (and it is where trailing insertions come from).
        self.end_state = nn.Parameter(torch.zeros(self.ctx_dim))

        # ---- prediction network over the emitted CELL prefix (edit_v2 only) --
        # Built only when on, so edit_v1's module list / state_dict carry nothing unused
        # (an embedding nothing reads would sit there collecting no gradient).
        if self.prefix_conditioning:
            self.y_embed = nn.Embedding(self.n_cells + 1, emb)  # +1 for START
            self.pred_gru = nn.GRU(emb, self.ctx_dim, batch_first=True)
            self.pred_h0 = nn.Linear(self.ctx_dim, self.ctx_dim)

        # ---- heads ----------------------------------------------------------
        # ops: prefix-free by design (see the module docstring) -> (stay, emit)
        self.op_head = _mlp(2 * self.ctx_dim, self.ctx_dim, 2, int(m.op_head_layers))
        emit_dim = 2 * self.ctx_dim + (self.ctx_dim if self.prefix_conditioning else 0)
        self.emit_dim = emit_dim
        # shift head: 4 mean displacements + the p_anch logit, plus 4 raw widths when the
        # physics form is ablated away.
        n_shift = 5 if self.physics_width else 9
        self.shift_head = _mlp(emit_dim, self.ctx_dim, n_shift, int(m.shift_head_layers))
        self.free_cell_head = _mlp(emit_dim, self.ctx_dim, self.n_cells,
                                   int(m.free_head_layers))
        # Unlike AR's coord_head this does NOT embed the cell: the head is evaluated at
        # every lattice state, and a per-(i, j, cell) offset head would be O(n_cells)
        # times the lattice. The cell categorical above carries the cell dependence.
        self.free_coord_head = _mlp(emit_dim, self.ctx_dim, 8, int(m.free_head_layers))

        # p_anch starts HIGH: the anchored and free components are only weakly
        # identifiable, and "most parton nodes have a hadron image" is the prior the
        # family exists to express (docs/PLAN_EditTransducer.md, risk 2).
        self.anch_bias = nn.Parameter(torch.tensor(2.0))

        # sigma = sigma_0 + Lambda_eff * exp(-ln kt), one (sigma_0, Lambda) pair per
        # coordinate, in the row order (ln 1/DeltaR, ln kt, ln z, psi). Small sigma_0 and
        # Lambda ~ 1 GeV is the shape-function expectation, and it is the initialization.
        init = torch.tensor(
            [
                [_inv_softplus(0.05), _inv_softplus(1.0)],   # ln 1/DeltaR
                [_inv_softplus(0.05), _inv_softplus(1.0)],   # ln kt
                [_inv_softplus(0.10), _inv_softplus(1.0)],   # ln z
                [_inv_softplus(0.30), _inv_softplus(1.0)],   # psi
            ]
        )
        self.width_raw = nn.Parameter(init)

        cx, cy = geometry.cell_center_tensors()
        self.register_buffer("cell_cx", cx)
        self.register_buffer("cell_cy", cy)

    @property
    def start_token(self) -> int:
        return self.n_cells

    # ------------------------------------------------------------------ encoder
    def _anchors(self, xf: torch.Tensor) -> torch.Tensor:
        """`(B, Mx, 4)` anchor coordinates `(ln 1/DeltaR, ln kt, ln z, psi)`.

        Read straight off `xf`, which `features.node_features` stores UNSTANDARDIZED as
        `(ln 1/DeltaR, ln kt, ln z, sin psi, cos psi)` — so no data-layer change is
        needed, and aux columns (docs/PLAN_Input.md) appending after index 4 are
        harmless."""
        u = xf[..., 0].clamp(self.lo_u, self.hi_u)
        v = xf[..., 1].clamp(self.lo_v, self.hi_v)
        return torch.stack([u, v, xf[..., 2], torch.atan2(xf[..., 3], xf[..., 4])], dim=-1)

    def _encode(self, xf: torch.Tensor, nx: torch.Tensor):
        """`(S, e, anchor, anchor_ok)` on the `(Mx + 1)`-column lattice.

        `S[:, i]` is the per-node encoder state of hadron node `i`, except at and beyond
        each jet's own terminal column `i = nx`, where it is the learned `end_state` — a
        jet padded to the batch maximum must still read its terminal column as terminal.
        `anchor_ok` is `i < nx`, the mask that forces `p_anch = 0` where there is no
        hadron node to anchor on (and hence makes an `nx == 0` jet reduce exactly to the
        free head)."""
        e = self.encoder_net(xf, nx)
        seq, _mask = self.encoder_net.forward_seq(xf, nx)
        B, Mx = xf.shape[0], xf.shape[1]
        end = self.end_state.view(1, 1, -1)
        S = torch.cat([self.state_proj(seq), end.expand(B, 1, self.ctx_dim)], dim=1)
        idx = torch.arange(Mx + 1, device=xf.device)[None, :]
        anchor_ok = idx < nx[:, None]                      # (B, Mx+1)
        S = torch.where(anchor_ok[..., None], S, end)
        anchor = self._anchors(xf)
        anchor = torch.cat([anchor, anchor.new_zeros(B, 1, 4)], dim=1)
        return S, e, anchor, anchor_ok

    def _op_logprobs(self, S: torch.Tensor, e: torch.Tensor):
        """`(log_stay, log_emit)`, each `(B, n_col)`. `log_stay` is ADVANCE at `i < nx` and
        STOP at `i == nx` — the SAME categorical slot, which is what closes the
        normalization without a bespoke argument."""
        op_in = torch.cat([S, e[:, None, :].expand(-1, S.shape[1], -1)], dim=-1)
        lp = F.log_softmax(self.op_head(op_in), dim=-1)
        return lp[..., 0], lp[..., 1]

    def _prefix_states(self, yc: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """`(B, L+1, ctx)` prediction-network states; column `j` has consumed `y_1..y_j`.

        RNN-T's trick is what keeps `edit_v2` cheap: `c_j` depends only on `j`, so it is
        `O(n_y)` GRU steps computed once and then joined with `s_i` across the lattice."""
        B, L = yc.shape
        start = torch.full((B, 1), self.start_token, dtype=torch.long, device=yc.device)
        tokens = torch.cat([start, yc.clamp(min=0)], dim=1)
        h0 = torch.tanh(self.pred_h0(e)).unsqueeze(0).contiguous()
        out, _ = self.pred_gru(self.y_embed(tokens), h0)
        return out

    def _emit_input(self, S: torch.Tensor, e: torch.Tensor, C):
        """`(B, n_col, T, emit_dim)`; `T == 1` (edit_v1: no prefix) or `T == len(C)`."""
        B, n_col, _ = S.shape
        T = 1 if C is None else C.shape[1]
        parts = [S[:, :, None, :].expand(B, n_col, T, self.ctx_dim)]
        if C is not None:
            parts.append(C[:, None, :, :].expand(B, n_col, T, self.ctx_dim))
        parts.append(e[:, None, None, :].expand(B, n_col, T, self.ctx_dim))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------- emission heads
    def _widths(self, ln_kt: torch.Tensor, raw):
        """The four residual widths at an anchor of hardness `ln_kt`.

        Physics form: `sigma = sigma_0 + Lambda_eff * exp(-ln kt)`, i.e. `Lambda_eff/k_t`
        — a shift of order `Lambda_eff` GeV in `k_t` is a shift of `Lambda_eff/k_t` in
        `ln k_t`, so `Lambda_eff` comes out in GeV and is directly comparable with the
        shape-function scale. `raw` is the free-MLP ablation's replacement."""
        if not self.physics_width:
            s = F.softplus(raw[..., 0]) + self.sigma_floor
            t = F.softplus(raw[..., 1]) + self.sigma_floor
            z = F.softplus(raw[..., 2]) + self.sigma_floor
            kap = F.softplus(raw[..., 3]).clamp(1e-3, self.kappa_max)
            return s, t, z, kap
        w0 = F.softplus(self.width_raw[:, 0])
        lam = F.softplus(self.width_raw[:, 1])
        inv_kt = torch.exp(-ln_kt.clamp(min=self.lo_v))
        sig = [(self.sigma_floor + w0[k] + lam[k] * inv_kt).clamp(max=10.0) for k in range(4)]
        kap = (1.0 / (sig[3] * sig[3])).clamp(1e-3, self.kappa_max)
        return sig[0], sig[1], sig[2], kap

    def _free_coord_params(self, raw):
        """AR's `_coord_params` on the free head: bounded within-cell offset means, a
        floored sigma each, a normal for ln z and a von Mises for psi."""
        du_m = self.half_u * torch.tanh(raw[..., 0])
        dv_m = self.half_v * torch.tanh(raw[..., 1])
        du_s = F.softplus(raw[..., 2]) + self.sigma_floor
        dv_s = F.softplus(raw[..., 3]) + self.sigma_floor
        lz_m = raw[..., 4]
        lz_s = F.softplus(raw[..., 5]) + self.sigma_floor
        a, b = raw[..., 6], raw[..., 7]
        kappa = torch.sqrt(a * a + b * b).clamp(1e-3, self.kappa_max)
        return du_m, dv_m, du_s, dv_s, lz_m, lz_s, torch.atan2(b, a), kappa

    def _emit_params(self, emit_in, anchor, anchor_ok) -> _EmitParams:
        """Both mixture components at a set of lattice states.

        `anchor` broadcasts against `emit_in`'s leading dims with a trailing 4; `anchor_ok`
        is the boolean "this column has a hadron node". Where it is False the anchored
        weight is driven to exactly zero, so the terminal column emits only insertions and
        an `nx == 0` jet is the pure free head."""
        p = self.shift_head(emit_in)
        a_u, a_v, a_z, a_psi = (anchor[..., k] for k in range(4))
        sig_u, sig_v, sig_z, kappa = self._widths(a_v, p[..., 5:] if not self.physics_width else None)
        mu_u = (a_u + p[..., 0]).clamp(self.lo_u, self.hi_u)
        mu_v = (a_v + p[..., 1]).clamp(self.lo_v, self.hi_v)
        mu_z = a_z + p[..., 2]
        mu_psi = a_psi + p[..., 3]
        logit = torch.where(anchor_ok, p[..., 4] + self.anch_bias,
                            torch.full_like(p[..., 4], _NO_ANCHOR))
        free = self._free_coord_params(self.free_coord_head(emit_in))
        return _EmitParams(
            log_p_anch=F.logsigmoid(logit),
            log_1m_anch=F.logsigmoid(-logit),
            mu_u=mu_u, mu_v=mu_v, sig_u=sig_u, sig_v=sig_v,
            mu_z=mu_z, sig_z=sig_z, mu_psi=mu_psi, kappa=kappa,
            cell_lp=F.log_softmax(self.free_cell_head(emit_in), dim=-1),
            f_du_m=free[0], f_dv_m=free[1], f_du_s=free[2], f_dv_s=free[3],
            f_lz_m=free[4], f_lz_s=free[5], f_psi_m=free[6], f_kappa=free[7],
        )

    # --------------------------------------------------------------- densities
    def _log_f_anch(self, p: _EmitParams, u, v, lz, psi):
        """The smeared (kept-node) component: a truncated normal on each plane
        coordinate — truncated to the GEOMETRY range, so the density is normalized on
        exactly the support the geometry defines — a normal on ln z, a von Mises on psi."""
        ll = trunc_normal_logpdf(u, p.mu_u, p.sig_u, self.lo_u, self.hi_u)
        ll = ll + trunc_normal_logpdf(v, p.mu_v, p.sig_v, self.lo_v, self.hi_v)
        ll = ll + gauss_logpdf(lz, p.mu_z, p.sig_z)
        return ll + vonmises_logpdf(psi, p.mu_psi, p.kappa)

    def _gather_cell_lp(self, cell_lp, cell):
        tgt = torch.broadcast_shapes(cell_lp.shape[:-1], cell.shape)
        return (
            cell_lp.expand(*tgt, self.n_cells)
            .gather(-1, cell.expand(*tgt).unsqueeze(-1))
            .squeeze(-1)
        )

    def _log_f_free(self, p: _EmitParams, cell, u, v, lz, psi):
        """The insertion component: cell categorical times within-cell offsets — the v2
        `split_head` / `coord_head` pattern, and a proper density on the same space the
        anchored component lives on."""
        cx, cy = self.cell_cx[cell], self.cell_cy[cell]
        du = (u - cx).clamp(-self.half_u, self.half_u)
        dv = (v - cy).clamp(-self.half_v, self.half_v)
        ll = self._gather_cell_lp(p.cell_lp, cell)
        ll = ll + trunc_normal_logpdf(du, p.f_du_m, p.f_du_s, -self.half_u, self.half_u)
        ll = ll + trunc_normal_logpdf(dv, p.f_dv_m, p.f_dv_s, -self.half_v, self.half_v)
        ll = ll + gauss_logpdf(lz, p.f_lz_m, p.f_lz_s)
        return ll + vonmises_logpdf(psi, p.f_psi_m, p.f_kappa)

    def _log_f_emit(self, p: _EmitParams, cell, u, v, lz, psi):
        return torch.logaddexp(
            p.log_p_anch + self._log_f_anch(p, u, v, lz, psi),
            p.log_1m_anch + self._log_f_free(p, cell, u, v, lz, psi),
        )

    def _targets(self, yraw: torch.Tensor):
        """The four target coordinates, clamped onto the geometry's support the same way
        `Geometry.to_cell` clips before discretising."""
        u = yraw[..., 0].clamp(self.lo_u, self.hi_u)
        v = yraw[..., 1].clamp(self.lo_v, self.hi_v)
        return u, v, yraw[..., 2], yraw[..., 3]

    def _lattice(self, batch):
        """Everything the DP needs for one batch: `(log_stay, log_emit, log_dens, params)`.

        `log_stay`/`log_emit` are `(B, n_col)` (prefix-free ops), `log_dens` is `(B, n_col, Ny)`
        — the log-density of `y_t` emitted from column `i`."""
        xf, nx, yc, yraw = batch["xf"], batch["nx"], batch["yc"], batch["yraw"]
        S, e, anchor, anchor_ok = self._encode(xf, nx)
        log_stay, log_emit = self._op_logprobs(S, e)
        B, n_col = log_stay.shape
        Ny = yc.shape[1]
        if Ny == 0:
            return log_stay, log_emit, log_stay.new_zeros(B, n_col, 0), None
        C = self._prefix_states(yc, e)[:, :Ny] if self.prefix_conditioning else None
        p = self._emit_params(
            self._emit_input(S, e, C), anchor[:, :, None, :], anchor_ok[:, :, None]
        )
        u, v, lz, psi = (t[:, None, :] for t in self._targets(yraw))
        dens = self._log_f_emit(p, yc.clamp(min=0)[:, None, :], u, v, lz, psi)
        return log_stay, log_emit, dens, p

    # --------------------------------------------------------------- likelihood
    def log_prob(self, batch) -> torch.Tensor:
        log_stay, log_emit, dens, _ = self._lattice(batch)
        B, n_col = log_stay.shape
        Ny = dens.shape[2]
        stay = log_stay[:, :, None].expand(B, n_col, Ny + 1)
        edge = log_emit[:, :, None] + dens
        nx = batch["nx"].clamp(max=n_col - 1)
        ny = batch["ny"].clamp(max=Ny)
        return edit_dp.forward_logsumexp(stay, edge, nx, ny)

    # ------------------------------------------------- physics diagnostics (DP)
    @torch.inference_mode()
    def alignment_posterior(self, batch) -> dict:
        """Forward-backward responsibilities `gamma(i, j)` — a posterior over alignments,
        obtained without ever supervising one.

        Returns `gamma_emit (B, n_col, Ny)` (which column emitted `y_t`), `r_anch (B, n_col, Ny)`
        (the anchored share of that emission), the per-column `anchor (B, n_col, 4)` and the
        target coordinates. This is the input to the decisive stage-1 diagnostic: bin the
        residual `y_t - anchor_i` weighted by `gamma_emit * r_anch` in `(k_t, R_g, N)` and
        fit `sigma = sigma_0 + Lambda_eff . exp(-ln k_t)`. **If the widths come out flat in
        `k_t`, the anchoring assumption is wrong** and stage 2 should not be built."""
        self.eval()
        log_stay, log_emit, dens, p = self._lattice(batch)
        B, n_col = log_stay.shape
        Ny = dens.shape[2]
        stay = log_stay[:, :, None].expand(B, n_col, Ny + 1)
        edge = log_emit[:, :, None] + dens
        nx = batch["nx"].clamp(max=n_col - 1)
        ny = batch["ny"].clamp(max=Ny)
        resp = edit_dp.forward_backward_responsibilities(stay, edge, nx, ny)
        xf = batch["xf"]
        anchor = torch.cat(
            [self._anchors(xf), xf.new_zeros(B, 1, 4)], dim=1
        )
        if Ny == 0 or p is None:
            r_anch = dens.new_zeros(B, n_col, 0)
        else:
            u, v, lz, psi = (t[:, None, :] for t in self._targets(batch["yraw"]))
            anch = p.log_p_anch + self._log_f_anch(p, u, v, lz, psi)
            r_anch = torch.exp((anch - dens).clamp(max=0.0))
        return {
            "gamma_emit": resp["gamma_emit"], "gamma_stay": resp["gamma_stay"],
            "r_anch": r_anch, "anchor": anchor, "log_z": resp["log_z"],
        }

    @torch.inference_mode()
    def edit_summary(self, batch) -> dict:
        """Per-jet `frac_anchored` / `delete_rate` / `insert_rate`, each `(B,)` numpy.

        Read off the alignment posterior via the family's own accounting identity
        `n_y = n_x - #del + #ins` with `#kept = E[# anchored emissions]`: the insertions
        are the emissions that were not anchored, and the deletions are the hadron nodes
        that anchored nothing. (The identity assumes at most one anchored emission per
        hadron column — one hadron node splitting into two anchored parton nodes would be
        counted as a deletion plus two keeps.) NaN where the rate is undefined, so a
        consumer averages with `nanmean` rather than reading a 0 that means "no jets"."""
        post = self.alignment_posterior(batch)
        n_anch = (post["gamma_emit"] * post["r_anch"]).sum(dim=(1, 2))
        nx = batch["nx"].to(n_anch.dtype)
        ny = batch["ny"].to(n_anch.dtype)
        nan = torch.full_like(n_anch, float("nan"))
        frac = torch.where(ny > 0, n_anch / ny.clamp(min=1.0), nan)
        delete = torch.where(nx > 0, ((nx - n_anch) / nx.clamp(min=1.0)).clamp(0.0, 1.0), nan)
        return {
            "frac_anchored": frac.cpu().numpy(),
            "insert_rate": (1.0 - frac).cpu().numpy(),
            "delete_rate": delete.cpu().numpy(),
        }

    def physics_width_params(self) -> dict | None:
        """`{coord: (sigma_0, Lambda_eff)}` — the learned shape-function width, in the
        units the physics is stated in (`Lambda_eff` in GeV). None under the free-MLP
        ablation, where there is no such scalar to read."""
        if not self.physics_width:
            return None
        w0 = F.softplus(self.width_raw[:, 0]).detach().cpu().tolist()
        lam = F.softplus(self.width_raw[:, 1]).detach().cpu().tolist()
        names = ("ln_invDelta", "ln_kt", "ln_z", "psi")
        return {n: (float(a), float(b)) for n, a, b in zip(names, w0, lam)}

    # ------------------------------------------------------- exact length marginal
    @torch.inference_mode()
    def length_pmf(self, xf, nx, mults=None, n_samples: int = 500) -> np.ndarray:
        """Exact `q(N | x)` from the structural DP — free, and conditioned on `|x|`.

        This is what `PLAN_MultHead.md`'s `n_head` learns, but with no parameters and no
        fit. `empty_gate` therefore reads an exact `q(N = 0 | x)`: the empty parton tree
        (16% of jets in the PYTHIA reference) is the delete-all path, which this family
        represents natively, so `PLAN_empty_parton_tree.md`'s decode-layer threshold
        becomes a diagnostic rather than a necessity. `learned_min_emissions` and
        `length_floor_quantile` compose unchanged, since they only consume this.

        `decode.length_temperature` / `length_tilt` are deliberately NOT applied. They
        recalibrate a fitted head; this belief is not fitted, and it is the marginal of the
        very lattice `sample` walks — tempering one without the other would decouple the
        two, which is the same reason they are a no-op for `ar_junipr_v1/v2`."""
        self.eval()
        S, e, _anchor, _ok = self._encode(xf, nx)
        log_stay, log_emit = self._op_logprobs(S, e)
        pmf = edit_dp.structural_length_pmf(
            log_stay, log_emit, nx.clamp(max=log_stay.shape[1] - 1), self.max_emissions
        )
        return pmf[0].cpu().numpy()

    # ------------------------------------------------------------------ sampling
    def _cells_from_coords(self, u, v):
        """`Geometry.to_cell`, tensorised — the same clip-then-truncate arithmetic, so a
        drawn coordinate always maps back to the cell reported beside it."""
        nb = self.n_bins
        fu = (u.clamp(self.lo_u, self.hi_u) - self.lo_u) / (self.hi_u - self.lo_u)
        fv = (v.clamp(self.lo_v, self.hi_v) - self.lo_v) / (self.hi_v - self.lo_v)
        ix = (fu * nb).long().clamp(0, nb - 1)
        iy = (fv * nb).long().clamp(0, nb - 1)
        return ix * nb + iy

    def _draw_emission(self, p: _EmitParams, *, generator=None):
        """One draw per element from the two-component mixture -> `(coords (K, 4), cell)`."""
        shape = p.log_p_anch.shape
        dev = p.log_p_anch.device
        take = torch.rand(shape, device=dev, generator=generator) < p.log_p_anch.exp()
        u_a = trunc_normal_sample(p.mu_u, p.sig_u, self.lo_u, self.hi_u, generator=generator)
        v_a = trunc_normal_sample(p.mu_v, p.sig_v, self.lo_v, self.hi_v, generator=generator)
        z_a = p.mu_z + p.sig_z * torch.randn(shape, device=dev, generator=generator)
        psi_a = vonmises_sample(p.mu_psi, p.kappa, generator=generator)

        probs = p.cell_lp.exp().reshape(-1, self.n_cells)
        cell_f = torch.multinomial(probs, 1, generator=generator).reshape(shape)
        du = trunc_normal_sample(p.f_du_m, p.f_du_s, -self.half_u, self.half_u,
                                 generator=generator)
        dv = trunc_normal_sample(p.f_dv_m, p.f_dv_s, -self.half_v, self.half_v,
                                 generator=generator)
        u_f = self.cell_cx[cell_f] + du
        v_f = self.cell_cy[cell_f] + dv
        z_f = p.f_lz_m + p.f_lz_s * torch.randn(shape, device=dev, generator=generator)
        psi_f = vonmises_sample(p.f_psi_m, p.f_kappa, generator=generator)

        u = torch.where(take, u_a, u_f)
        v = torch.where(take, v_a, v_f)
        lz = torch.where(take, z_a, z_f)
        psi = wrap_to_pi(torch.where(take, psi_a, psi_f))
        return torch.stack([u, v, lz, psi], dim=-1), self._cells_from_coords(u, v)

    @torch.inference_mode()
    def sample(self, xf, nx, n, max_emissions: int | None = None,
               cont_temperature: float = 1.0, **_kw):
        """`n` posterior draws for ONE jet: an ancestral walk over the lattice.

        All `n` draws advance in lockstep — the walk is at most `nx + max_emissions + 1`
        steps whatever `n` is — so a K-draw posterior costs a handful of vectorised steps
        rather than K python loops."""
        self.eval()
        dev = xf.device
        K = int(n)
        nx0 = int(nx[0])
        max_em = int(self.max_emissions if max_emissions is None else max_emissions)
        S, e, anchor, anchor_ok = self._encode(xf, nx)
        S, anchor, anchor_ok = S[:, : nx0 + 1], anchor[:, : nx0 + 1], anchor_ok[:, : nx0 + 1]
        log_stay, log_emit = self._op_logprobs(S, e)
        if cont_temperature != 1.0:  # exposure-bias knob, sampling-time only (as in AR)
            z = torch.stack([log_stay[0], log_emit[0]], dim=-1) / float(cont_temperature)
            lp = F.log_softmax(z, dim=-1)
            log_stay, log_emit = lp[None, :, 0], lp[None, :, 1]
        p_emit = log_emit[0].exp()                       # (n_col,)

        i = torch.zeros(K, dtype=torch.long, device=dev)
        alive = torch.ones(K, dtype=torch.bool, device=dev)
        n_emitted = torch.zeros(K, dtype=torch.long, device=dev)
        out: list[list[int]] = [[] for _ in range(K)]
        e_k = e.expand(K, -1)
        h = None
        if self.prefix_conditioning:
            h0 = torch.tanh(self.pred_h0(e_k)).unsqueeze(0).contiguous()
            tok = torch.full((K, 1), self.start_token, dtype=torch.long, device=dev)
            _, h = self.pred_gru(self.y_embed(tok), h0)

        for _ in range(nx0 + max_em + 1):
            if not bool(alive.any()):
                break
            emit = (torch.rand(K, device=dev) < p_emit[i]) & alive & (n_emitted < max_em)
            if bool(emit.any()):
                parts = [S[0, i]] + ([] if h is None else [h[0]]) + [e_k]
                p = self._emit_params(
                    torch.cat(parts, dim=-1), anchor[0, i], anchor_ok[0, i]
                )
                coords, cells = self._draw_emission(p)
                for k in emit.nonzero(as_tuple=False).flatten().tolist():
                    out[k].append(int(cells[k]))
                n_emitted = n_emitted + emit.long()
                if h is not None:
                    _, h_new = self.pred_gru(self.y_embed(cells)[:, None, :], h)
                    h = torch.where(emit[None, :, None], h_new, h)
            stay = alive & ~emit
            alive = alive & ~(stay & (i >= nx0))          # STAY at the terminal == STOP
            i = i + (stay & (i < nx0)).long()
        return out

    def sample_batch(self, xf, nx, n_samples, max_emissions: int = 25):
        return self.sample(xf, nx, n_samples, max_emissions=max_emissions)

    # ------------------------------------ coordinates given a drawn cell chain
    def _cell_bounds(self, cell):
        return (
            self.cell_cx[cell] - self.half_u, self.cell_cx[cell] + self.half_u,
            self.cell_cy[cell] - self.half_v, self.cell_cy[cell] + self.half_v,
        )

    def _log_cell_mass(self, p: _EmitParams, cell):
        """`(log mixture mass in `cell`, log anchored share)` — the emission likelihood
        with the coordinates integrated over the cell, which is what the CONSTRAINED
        lattice runs on."""
        u_lo, u_hi, v_lo, v_hi = self._cell_bounds(cell)
        mu = (trunc_normal_cdf(u_hi, p.mu_u, p.sig_u, self.lo_u, self.hi_u)
              - trunc_normal_cdf(u_lo, p.mu_u, p.sig_u, self.lo_u, self.hi_u)).clamp(min=1e-12)
        mv = (trunc_normal_cdf(v_hi, p.mu_v, p.sig_v, self.lo_v, self.hi_v)
              - trunc_normal_cdf(v_lo, p.mu_v, p.sig_v, self.lo_v, self.hi_v)).clamp(min=1e-12)
        anch = p.log_p_anch + torch.log(mu) + torch.log(mv)
        # the free component's within-cell offsets integrate to exactly 1 over their cell
        free = p.log_1m_anch + self._gather_cell_lp(p.cell_lp, cell)
        return torch.logaddexp(anch, free), anch

    @staticmethod
    def _index(p: _EmitParams, i_idx, t_idx) -> _EmitParams:
        """Pick the `(i, t)` lattice states out of a `(1, n_col, T, ...)` parameter block.

        Fields are not all the same `T`: the physics widths depend on the anchor alone, so
        they come back with `T == 1` even in `edit_v2`. Index each field on its own axis
        rather than assuming one shared prefix axis."""
        zero = torch.zeros_like(t_idx)
        return _EmitParams(*[f[0, i_idx, t_idx if f.shape[2] > 1 else zero] for f in p])

    @torch.inference_mode()
    def sample_coordinates(self, xf, nx, cells, *, generator=None):
        """`(L, 4)` coordinates drawn from `q(coords | cells, x)` — the one genuinely new
        inference routine in this family.

        The coordinates are NOT conditionally independent of the alignment given the cell
        chain, so this runs the **constrained** forward-backward over the paths consistent
        with those cells (emission weights = the mixture's mass in each cell), samples an
        alignment from that posterior, and only then draws each coordinate from the
        component the alignment implies, truncated to its cell. `O(n_x . n_y)`."""
        cells = [int(c) for c in cells]
        dev = xf.device
        if not cells:
            return torch.zeros(0, 4, device=dev)
        self.eval()
        L, nx0 = len(cells), int(nx[0])
        S, e, anchor, anchor_ok = self._encode(xf, nx)
        S, anchor, anchor_ok = S[:, : nx0 + 1], anchor[:, : nx0 + 1], anchor_ok[:, : nx0 + 1]
        log_stay, log_emit = self._op_logprobs(S, e)
        yc = torch.tensor([cells], dtype=torch.long, device=dev)
        C = self._prefix_states(yc, e)[:, :L] if self.prefix_conditioning else None
        p = self._emit_params(
            self._emit_input(S, e, C), anchor[:, :, None, :], anchor_ok[:, :, None]
        )
        cell_t = yc[:, None, :]                                   # (1, 1, L)
        log_mass, log_anch = self._log_cell_mass(p, cell_t)       # (1, n_col, L)
        n_col = log_stay.shape[1]
        stay = log_stay[:, :, None].expand(1, n_col, L + 1)
        edge = log_emit[:, :, None] + log_mass
        cols = edit_dp.sample_alignment(stay[0], edge[0], nx0, L, generator=generator)

        i_idx = torch.tensor(cols, dtype=torch.long, device=dev)
        t_idx = torch.arange(L, device=dev)
        q = self._index(p, i_idx, t_idx)
        cell = yc[0]
        r = torch.exp(
            (log_anch[0, i_idx, t_idx] - log_mass[0, i_idx, t_idx]).clamp(max=0.0)
        )
        take = torch.rand(L, device=dev, generator=generator) < r
        u_lo, u_hi, v_lo, v_hi = self._cell_bounds(cell)
        # the anchored component restricted to a cell is just the same normal truncated
        # to the cell's edges instead of the geometry's
        u_a = trunc_normal_sample(q.mu_u, q.sig_u, u_lo, u_hi, generator=generator)
        v_a = trunc_normal_sample(q.mu_v, q.sig_v, v_lo, v_hi, generator=generator)
        z_a = q.mu_z + q.sig_z * torch.randn(L, device=dev, generator=generator)
        psi_a = vonmises_sample(q.mu_psi, q.kappa, generator=generator)
        du = trunc_normal_sample(q.f_du_m, q.f_du_s, -self.half_u, self.half_u,
                                 generator=generator)
        dv = trunc_normal_sample(q.f_dv_m, q.f_dv_s, -self.half_v, self.half_v,
                                 generator=generator)
        z_f = q.f_lz_m + q.f_lz_s * torch.randn(L, device=dev, generator=generator)
        psi_f = vonmises_sample(q.f_psi_m, q.f_kappa, generator=generator)
        u = torch.where(take, u_a, self.cell_cx[cell] + du)
        v = torch.where(take, v_a, self.cell_cy[cell] + dv)
        lz = torch.where(take, z_a, z_f)
        psi = wrap_to_pi(torch.where(take, psi_a, psi_f))
        return torch.stack([u, v, lz, psi], dim=-1)

    # ------------------------------------------------------------ point estimate
    def _emission_mode(self, p: _EmitParams):
        """The modal emission at each lattice state: whichever component's own mode
        carries the larger MIXTURE density, with its cell and that density."""
        cell_a = self._cells_from_coords(p.mu_u, p.mu_v)
        f_a = self._log_f_emit(p, cell_a, p.mu_u, p.mu_v, p.mu_z, p.mu_psi)
        cell_f = p.cell_lp.argmax(dim=-1)
        u_f = self.cell_cx[cell_f] + p.f_du_m
        v_f = self.cell_cy[cell_f] + p.f_dv_m
        f_f = self._log_f_emit(p, cell_f, u_f, v_f, p.f_lz_m, p.f_psi_m)
        take = f_a >= f_f
        coords = torch.stack(
            [
                torch.where(take, p.mu_u, u_f),
                torch.where(take, p.mu_v, v_f),
                torch.where(take, p.mu_z, p.f_lz_m),
                wrap_to_pi(torch.where(take, p.mu_psi, p.f_psi_m)),
            ],
            dim=-1,
        )
        return coords, torch.where(take, cell_a, cell_f), torch.where(take, f_a, f_f)

    @torch.inference_mode()
    def map_estimate(self, xf, nx, **decode) -> LundPointEstimate:
        """Staged decode: `N* = argmax_n q(N=n|x)` from the exact length marginal, then
        the Viterbi alignment with modal emissions at exactly that length. A labelled
        **surrogate**, for two separate reasons worth keeping apart.

        The exact MAP is an argmax over a marginal-over-alignments and is intractable, so
        the shape is the best single alignment rather than the best `y` (the same honesty
        pattern as `Diffusion.exact_likelihood=False`).

        The length is taken from `q(N|x)` rather than from the joint argmax deliberately.
        A joint argmax over a variable-dimension DENSITY runs straight to `max_emissions`
        whenever the modal emission density exceeds the per-step op cost — with the
        physics widths that is the normal regime at high `k_t`, where the kernel is sharp
        — so the joint mode is a property of the decision rule, not of the fit. This is
        `ar_junipr_v3`'s staged decode (`_map_decode_fixed_length`) with an EXACT length
        marginal in place of a learned head, which is the one asset this family has that
        v3 does not. Collapse to `n = 0` is structurally suppressed on top: it needs
        ADVANCE at all `nx` columns AND a STOP, where the AR families need one stop draw.
        `min_emissions` still floors it, so the default decode never returns the empty
        tree, and `length_floor_quantile` composes through the same argument."""
        self.eval()
        dev = xf.device
        nx0 = int(nx[0])
        max_em = min(int(decode.get("max_emissions", self.max_emissions)), self.max_emissions)
        min_em = max(0, min(int(decode.get("min_emissions", 1)), max_em))
        S, e, anchor, anchor_ok = self._encode(xf, nx)
        S, anchor, anchor_ok = S[:, : nx0 + 1], anchor[:, : nx0 + 1], anchor_ok[:, : nx0 + 1]
        log_stay, log_emit = self._op_logprobs(S, e)
        n_col = log_stay.shape[1]

        pmf = edit_dp.structural_length_pmf(log_stay, log_emit, nx, max_em)[0]
        n_star = min_em + int(pmf[min_em:].argmax())
        if n_star == 0:
            return self._describe(xf, nx, [], torch.zeros(0, 4, device=dev))

        if not self.prefix_conditioning:
            emit_in = torch.cat([S[0], e.expand(n_col, -1)], dim=-1)
            p = self._emit_params(emit_in, anchor[0], anchor_ok[0])
            m_co, m_cell, m_lf = self._emission_mode(p)          # (n_col, 4), (n_col,), (n_col,)
            coords = m_co[:, None, :].expand(n_col, n_star, 4)
            cellsg = m_cell[:, None].expand(n_col, n_star)
            logf = m_lf[:, None].expand(n_col, n_star)
        else:
            # The prefix state depends on what was emitted, which the alignment has not
            # chosen yet, so roll it forward along the best-scoring emitting column at
            # each step — beam-1 on top of a surrogate, and exactly the v1 arithmetic
            # when the prefix is switched off.
            h = torch.tanh(self.pred_h0(e)).unsqueeze(0).contiguous()
            tok = torch.full((1, 1), self.start_token, dtype=torch.long, device=dev)
            _, h = self.pred_gru(self.y_embed(tok), h)
            co_t, ce_t, lf_t = [], [], []
            for _t in range(n_star):
                emit_in = torch.cat([S[0], h[0].expand(n_col, -1), e.expand(n_col, -1)], dim=-1)
                p = self._emit_params(emit_in, anchor[0], anchor_ok[0])
                m_co, m_cell, m_lf = self._emission_mode(p)
                co_t.append(m_co)
                ce_t.append(m_cell)
                lf_t.append(m_lf)
                best = int((log_emit[0] + m_lf).argmax())
                _, h = self.pred_gru(self.y_embed(m_cell[best].view(1, 1)), h)
            coords = torch.stack(co_t, dim=1)
            cellsg = torch.stack(ce_t, dim=1)
            logf = torch.stack(lf_t, dim=1)

        # min_n == n_star pins the terminal, so the Viterbi chooses the ALIGNMENT only.
        # one_per_column is the family's own semantics — a hadron node is kept-and-smeared
        # or deleted, never duplicated — and without it the argmax is degenerate (see
        # `edit_dp.viterbi_path`).
        stay = log_stay[0][:, None].expand(n_col, n_star + 1)
        edge = log_emit[0][:, None] + logf
        _score, cols = edit_dp.viterbi_path(stay, edge, nx0, min_n=n_star,
                                            one_per_column=True)
        cells = [int(cellsg[c, t]) for t, c in enumerate(cols)]
        coord = torch.stack([coords[c, t] for t, c in enumerate(cols)])
        return self._describe(xf, nx, cells, coord)

    def _describe(self, xf, nx, cells, coords) -> LundPointEstimate:
        """`(cells, coords)` -> `LundPointEstimate` with the family's EXACT joint
        log-density of that configuration.

        `logp_coord` per node is the alignment-posterior-weighted emission log-density
        `sum_i gamma(i, t) log f(y_t | i, t)`: with the alignment marginalised there is no
        exact per-node factorization of `log q(y|x)`, so the honest per-node number is an
        expectation under the same posterior the DP already computed. `logp_split` /
        `logp_cont` are 0 because this family has no separate cell or continue factor —
        the cell lives inside the emission density and the length inside the lattice."""
        dev = xf.device
        L = len(cells)
        yc = torch.tensor([[int(c) for c in cells]], dtype=torch.long, device=dev)
        yraw = coords[None].to(dev).float() if L else torch.zeros(1, 0, 4, device=dev)
        batch = {"xf": xf, "nx": nx, "yc": yc,
                 "ny": torch.tensor([L], device=dev), "yraw": yraw}
        log_stay, log_emit, dens, _p = self._lattice(batch)
        n_col = log_stay.shape[1]
        stay = log_stay[:, :, None].expand(1, n_col, L + 1)
        edge = log_emit[:, :, None] + dens
        nx_c = nx.clamp(max=n_col - 1)
        ny_c = torch.tensor([L], device=dev)
        total = float(edit_dp.forward_logsumexp(stay, edge, nx_c, ny_c)[0])
        per = torch.zeros(L, device=dev)
        if L:
            resp = edit_dp.forward_backward_responsibilities(stay, edge, nx_c, ny_c)
            per = (resp["gamma_emit"][0] * dens[0]).sum(0)   # gamma sums to 1 over i
        nodes = []
        for t in range(L):
            u, v, lz, ps = (float(yraw[0, t, k]) for k in range(4))
            nodes.append(
                LundNode(
                    depth=t, parent=t - 1, cell=int(cells[t]),
                    ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                    kt=math.exp(v), delta_R=math.exp(-u), z=math.exp(lz),
                    logp_split=0.0, logp_coord=float(per[t]), logp_cont=0.0,
                )
            )
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=L)

    @torch.inference_mode()
    def describe_cells(self, xf, nx, cells) -> LundPointEstimate:
        """MBR winner -> `LundPointEstimate`: a genuine draw from `q(coords | cells, x)`
        plus this family's exact joint log-density of it."""
        cells = [int(c) for c in cells]
        coords = self.sample_coordinates(xf, nx, cells)
        return self._describe(xf, nx, cells, coords)
