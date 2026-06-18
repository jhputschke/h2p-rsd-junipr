"""
Conditional RSD-JUNIPR v2: amortized per-jet hadronization-inversion posterior
q_phi(y | x), specialised to the PRIMARY branch -- WITH CONTINUOUS PER-NODE
COORDINATES.

WHAT V2 ADDS OVER conditional_rsd_junipr.py
-------------------------------------------
v1 modelled each primary splitting as a discrete Lund CELL over (ln 1/DeltaR,
ln kt) on the 10x10 grid, so its MAP point estimate carried only quantised
(cell-centre) coordinates and no ln z / psi. v2 keeps the categorical cell head
(it still drives the autoregressive backbone, beam search, and every cell-based
diagnostic) and ADDS a continuous coordinate head, conditioned on [h_t, e, the
chosen cell], that models the full node:

    p(node_t | h_t, e) = P_split(cell c_t)                       # coarse location
                       * TruncNormal(du | c_t) TruncNormal(dv | c_t)  # within-cell refinement
                       * Normal(ln z | c_t)                      # momentum fraction
                       * vonMises(psi | c_t)                     # azimuth (periodic)

where (du, dv) are the within-cell offsets of (ln 1/DeltaR, ln kt) from the cell
centre, truncated to the cell so the combined object is a PROPER density over the
continuous node (the truncation normaliser is included; it does not leak mass
outside the cell). psi is periodic, so it gets a von Mises -- a Gaussian on an
angle is wrong at the +-pi wrap. The factors are conditionally independent given
the cell (a modelling choice; a small joint flow is the next refinement).

The point estimate y_hat = argmax_y q_phi(y|x) is now a primary Lund tree whose
nodes carry GENUINE CONTINUOUS coordinates: the cell structure is found by beam
search (as in v1), then each node's continuous coordinates are the conditional
head modes (cell_centre + bounded-mean offset; the Normal/von-Mises means). This
is a staged MAP -- discrete structure then conditional continuous mode -- and the
reported log q is the FULL joint log-density of the returned configuration.

Everything else (encoder e(x), continue/stop head, the synthetic matched-pair
simulator, the closure/coverage diagnostics, MPS/CUDA device selection, and the
batched on-device posterior sampler sample_batch) is carried over from v1; see
its module docstring for the statistical setting and references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Lund-plane discretisation  (ln 1/DeltaR, ln kt) -> flat cell id
# ---------------------------------------------------------------------------
LN_INVDELTA_RANGE = (0.0, 6.0)     # ln(1/DeltaR); larger = more collinear
LN_KT_RANGE       = (0.0, 6.0)     # ln(kt/GeV);   lower edge = kt grooming floor
N_BINS  = 10
N_CELLS = N_BINS * N_BINS
START   = N_CELLS                  # extra embedding index used as the start token
CELL_WU = (LN_INVDELTA_RANGE[1] - LN_INVDELTA_RANGE[0]) / N_BINS   # cell width, ln 1/DeltaR
CELL_WV = (LN_KT_RANGE[1] - LN_KT_RANGE[0]) / N_BINS               # cell width, ln kt


def to_cell(ln_invdelta: float, ln_kt: float) -> int:
    x = float(np.clip(ln_invdelta, *LN_INVDELTA_RANGE))
    y = float(np.clip(ln_kt, *LN_KT_RANGE))
    ix = min(int((x - LN_INVDELTA_RANGE[0]) / (LN_INVDELTA_RANGE[1] - LN_INVDELTA_RANGE[0]) * N_BINS), N_BINS - 1)
    iy = min(int((y - LN_KT_RANGE[0]) / (LN_KT_RANGE[1] - LN_KT_RANGE[0]) * N_BINS), N_BINS - 1)
    return ix * N_BINS + iy


def cell_center(cell: int):
    ix, iy = divmod(cell, N_BINS)
    return LN_INVDELTA_RANGE[0] + (ix + 0.5) * CELL_WU, LN_KT_RANGE[0] + (iy + 0.5) * CELL_WV


# ===========================================================================
#  DATA
# ===========================================================================
N_NODE_FEAT = 5   # encoder/decoder INPUT features: (ln 1/DeltaR, ln kt, ln z, sin psi, cos psi)


def node_features(ln_invd, ln_kt, ln_z, psi) -> np.ndarray:
    """Continuous per-node feature matrix (n, 5) for the encoder/decoder inputs."""
    ln_invd = np.asarray(ln_invd, dtype=np.float32)
    ln_kt   = np.asarray(ln_kt,   dtype=np.float32)
    ln_z    = np.asarray(ln_z,    dtype=np.float32)
    psi     = np.asarray(psi,     dtype=np.float32)
    if ln_invd.size == 0:
        return np.zeros((0, N_NODE_FEAT), dtype=np.float32)
    return np.stack([ln_invd, ln_kt, ln_z, np.sin(psi), np.cos(psi)], axis=1).astype(np.float32)


def node_raw(ln_invd, ln_kt, ln_z, psi) -> np.ndarray:
    """Raw continuous TARGETS (n, 4) = (ln 1/DeltaR, ln kt, ln z, psi) for the
    continuous coordinate likelihood. psi is kept as a raw angle (not sin/cos)
    so the von Mises term can be evaluated directly."""
    arrs = [np.asarray(a, dtype=np.float32) for a in (ln_invd, ln_kt, ln_z, psi)]
    if arrs[0].size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(arrs, axis=1).astype(np.float32)


def seq_cells(ln_invd, ln_kt) -> np.ndarray:
    """Discretise a (ln 1/DeltaR, ln kt) sequence to flat Lund-cell ids."""
    return np.array([to_cell(a, b) for a, b in zip(ln_invd, ln_kt)], dtype=np.int64)


# ---------------------------------------------------------------------------
# (A) Real input: read the RNTuple written by write_lund_rntuple.cpp
# ---------------------------------------------------------------------------
def load_rntuple(path: str = "jets.root", ntuple: str = "Jets"):
    """Read matched (x, y) primary Lund sequences from a ROOT RNTuple via uproot.
    Returns a list of per-jet dicts or None if the file/tool is unavailable."""
    try:
        import uproot
        with uproot.open(path) as f:
            arr = f[ntuple].arrays(library="np")
    except Exception as exc:  # missing file, no RNTuple support, etc.
        print(f"[load_rntuple] could not read {path}:{ntuple} ({exc}); using synthetic data.")
        return None

    jets = []
    n = len(arr["weight"])
    for i in range(n):
        jets.append(dict(
            weight=float(arr["weight"][i]),
            x=(arr["x_lnInvDelta"][i], arr["x_lnkt"][i], arr["x_lnz"][i], arr["x_psi"][i]),
            y=(arr["y_lnInvDelta"][i], arr["y_lnkt"][i], arr["y_lnz"][i], arr["y_psi"][i]),
        ))
    print(f"[load_rntuple] read {len(jets)} jets from {path}:{ntuple}.")
    return jets


# ---------------------------------------------------------------------------
# (B) Synthetic stand-in: a matched-pair hadronization simulator
# ---------------------------------------------------------------------------
def _sample_parton_sequence(rng: np.random.Generator, max_emissions: int = 20):
    """A parton-level primary Lund sequence: angular ordered, kt drifting down.
    (Qualitative stand-in only; not a physical kinematic configuration.)"""
    n = min(max(1, int(rng.poisson(6))), max_emissions)
    ln_invd = 0.0
    li, lk, lz, ps = [], [], [], []
    for t in range(n):
        ln_invd += rng.exponential(0.8)                 # angular ordering: 1/DeltaR grows
        if ln_invd > LN_INVDELTA_RANGE[1]:
            break
        ln_kt = rng.normal(4.0 - 0.3 * t, 1.0)          # kt drifts down deeper in the shower
        if ln_kt < LN_KT_RANGE[0]:
            continue                                    # below floor -> removed by RSD
        li.append(ln_invd)
        lk.append(ln_kt)
        lz.append(-float(rng.exponential(0.7)) - 0.05)  # ln z < 0 (z < 1)
        ps.append(float(rng.uniform(-math.pi, math.pi)))
    if not li:
        li, lk, lz, ps = [0.5], [3.0], [-0.3], [0.0]
    return (np.array(li, np.float32), np.array(lk, np.float32),
            np.array(lz, np.float32), np.array(ps, np.float32))


def _hadronize(parton, rng: np.random.Generator):
    """Forward model y -> x: a kt-dependent smearing + soft migration. The
    smearing WIDTH grows as ln kt decreases (tight at high kt, loose near the
    floor) -- the property grooming exploits."""
    li, lk, lz, ps = parton
    xi, xk, xz, xp = [], [], [], []
    for a, b, c, d in zip(li, lk, lz, ps):
        sigma = 0.25 + 0.35 * max(0.0, 3.0 - b)         # wider smear at low kt
        p_drop = 0.05 + 0.25 * max(0.0, 2.0 - b)        # soft nodes migrate out of the groomed tree
        if rng.random() < p_drop:
            continue
        xi.append(a + rng.normal(0.0, sigma))
        xk.append(b + rng.normal(0.0, sigma))
        xz.append(min(-1e-3, c + rng.normal(0.0, sigma)))
        xp.append(((d + rng.normal(0.0, 0.3)) + math.pi) % (2 * math.pi) - math.pi)
    if rng.random() < 0.20:                             # spurious soft hadron-level declustering
        xi.append(float(rng.uniform(*LN_INVDELTA_RANGE)))
        xk.append(float(rng.uniform(LN_KT_RANGE[0], 2.0)))
        xz.append(-float(rng.exponential(0.7)) - 0.05)
        xp.append(float(rng.uniform(-math.pi, math.pi)))
    if not xi:                                          # never return empty
        xi, xk, xz, xp = [li[0]], [lk[0]], [lz[0]], [ps[0]]
    order = np.argsort(xi)                              # keep angular ordering
    xi = np.clip(np.array(xi, np.float32)[order], *LN_INVDELTA_RANGE)
    xk = np.clip(np.array(xk, np.float32)[order], *LN_KT_RANGE)
    xz = np.array(xz, np.float32)[order]
    xp = np.array(xp, np.float32)[order]
    return xi, xk, xz, xp


def synthetic_matched_dataset(n_jets: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    jets = []
    for _ in range(n_jets):
        y = _sample_parton_sequence(rng)
        x = _hadronize(y, rng)
        jets.append(dict(weight=1.0, x=x, y=y))
    return jets


# ---------------------------------------------------------------------------
# Torch Dataset / collate
# ---------------------------------------------------------------------------
class MatchedLundDataset(Dataset):
    def __init__(self, jets):
        self.items = []
        for j in jets:
            xf = node_features(*j["x"])                 # (nx, 5)  encoder input
            yf = node_features(*j["y"])                 # (ny, 5)  (unused by decoder; kept for parity)
            yc = seq_cells(j["y"][0], j["y"][1])        # (ny,)    discrete cell targets
            yr = node_raw(*j["y"])                      # (ny, 4)  continuous coordinate targets
            self.items.append(dict(
                xf=torch.tensor(xf), nx=len(xf),
                yf=torch.tensor(yf), yc=torch.tensor(yc),
                yraw=torch.tensor(yr), ny=len(yc),
                w=torch.tensor(float(j["weight"]), dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    B = len(batch)
    nx = torch.tensor([b["nx"] for b in batch], dtype=torch.long)
    ny = torch.tensor([b["ny"] for b in batch], dtype=torch.long)
    w  = torch.stack([b["w"] for b in batch])
    Mx, My = int(nx.max()), int(ny.max())

    xf = torch.zeros(B, Mx, N_NODE_FEAT)
    yf = torch.zeros(B, My, N_NODE_FEAT)
    yc = torch.zeros(B, My, dtype=torch.long)
    yraw = torch.zeros(B, My, 4)
    for i, b in enumerate(batch):
        xf[i, : b["nx"]] = b["xf"]
        yf[i, : b["ny"]] = b["yf"]
        yc[i, : b["ny"]] = b["yc"]
        yraw[i, : b["ny"]] = b["yraw"]
    return dict(xf=xf, nx=nx, yf=yf, yc=yc, yraw=yraw, ny=ny, w=w)


# ===========================================================================
#  CONTINUOUS DENSITY HELPERS  (device-safe: only elementwise ops, no torch.special)
# ===========================================================================
_LOG_2PI = math.log(2.0 * math.pi)


def _gauss_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * z * z - torch.log(sigma) - 0.5 * _LOG_2PI


def _std_normal_cdf(t):
    return 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))


def _trunc_normal_logpdf(x, mu, sigma, lo, hi):
    """log density of N(mu, sigma^2) TRUNCATED to [lo, hi] (scalars), at x in [lo,hi].
    Subtracting the in-interval mass makes the within-cell offset a proper density,
    so cell-prob x offset-density integrates to a proper density over (ln1/DR, lnkt)."""
    base = _gauss_logpdf(x, mu, sigma)
    Z = (_std_normal_cdf((hi - mu) / sigma) - _std_normal_cdf((lo - mu) / sigma)).clamp(min=1e-6)
    return base - torch.log(Z)


def _log_bessel_i0(x):
    """log I0(x) for x >= 0 via the Abramowitz & Stegun 9.8.1/9.8.2 polynomial
    approximations -- elementwise only, so it runs (and differentiates) on MPS,
    where torch.special.i0e support is not guaranteed. Both branches are kept
    finite everywhere so autograd through torch.where is clean."""
    t = x / 3.75
    t2 = t * t
    small = 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
            + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    log_small = torch.log(small)
    xl = x.clamp(min=3.75)                              # keep large branch finite for x<3.75
    u = 3.75 / xl
    large = (0.39894228 + u * (0.01328592 + u * (0.00225319 + u * (-0.00157565
             + u * (0.00916281 + u * (-0.02057706 + u * (0.02635537
             + u * (-0.01647633 + u * 0.00392377))))))))
    log_large = xl - 0.5 * torch.log(xl) + torch.log(large)
    return torch.where(x <= 3.75, log_small, log_large)


def _vonmises_logpdf(psi, mu, kappa):
    """log density of a von Mises (circular Gaussian) on psi in (-pi, pi]."""
    return kappa * torch.cos(psi - mu) - _LOG_2PI - _log_bessel_i0(kappa)


# ===========================================================================
#  MODEL
# ===========================================================================
@dataclass
class LundNode:
    """One primary splitting of the MAP groomed parton-shower configuration, with
    CONTINUOUS coordinates predicted by the model.

    The primary branch is a CATERPILLAR: at step `depth` the leading prong splits,
    emitting a softer prong with this node's kinematics. `parent` is the previous
    node (depth-1; -1 for the root split), so the object is already a tree the
    binary-JUNIPR extension can grow secondary branches on.
    """
    depth: int          # position along the primary chain (0 = first emission)
    parent: int         # depth of the parent node (-1 for the root split)
    cell: int           # flat Lund-cell id (coarse location)
    ln_invDelta: float  # continuous: cell_centre + bounded mean offset
    ln_kt: float        # continuous: cell_centre + bounded mean offset
    ln_z: float         # continuous: Normal-head mode
    psi: float          # continuous: von Mises mean (rad)
    kt: float           # exp(ln_kt)            [GeV]
    delta_R: float      # exp(-ln_invDelta)
    z: float            # exp(ln_z)             (momentum fraction)
    logp_split: float   # log P_split(cell | h_t, e)
    logp_coord: float   # log p(coords | cell, h_t, e) at the returned modes
    logp_cont: float    # log P_continue at this step


@dataclass
class LundPointEstimate:
    """The single most likely groomed parton-shower configuration y_hat =
    argmax_y q_phi(y | x), as a primary Lund tree of continuous-coordinate nodes,
    plus the FULL joint log-density log q_phi(y_hat | x) of the returned config."""
    nodes: list          # list[LundNode], in primary (emission) order
    logprob: float       # log q_phi(y_hat | x): conts + splits + coords + final stop
    multiplicity: int

    def pretty(self) -> str:
        head = (f"MAP groomed shower: {self.multiplicity} primary splittings, "
                f"log q(y_hat|x) = {self.logprob:.3f}")
        rows = [f"  [{n.depth}] kt={n.kt:6.2f} GeV  DeltaR={n.delta_R:5.3f}  z={n.z:5.3f}  "
                f"psi={n.psi:+5.2f}  (ln1/DR={n.ln_invDelta:4.2f}, lnkt={n.ln_kt:4.2f}, "
                f"lnz={n.ln_z:5.2f})  logP={n.logp_split + n.logp_coord:+.2f}"
                for n in self.nodes]
        return "\n".join([head, *rows]) if rows else head + "\n  (empty: MAP is immediate stop)"


class ConditionalPrimaryLundJUNIPR(nn.Module):
    """Encoder e(x) (bi-GRU over the hadron-level primary sequence) feeding the
    conditioned autoregressive JUNIPR decoder over y, with a discrete cell head
    AND a continuous coordinate head per node (P_mother dropped, primary branch)."""

    def __init__(self, n_cells: int = N_CELLS, emb_dim: int = 32,
                 enc_dim: int = 64, dec_dim: int = 64, ctx_dim: int = 64):
        super().__init__()
        self.n_cells = n_cells
        self.ctx_dim = ctx_dim
        self.half_u = CELL_WU / 2.0                              # within-cell offset bounds
        self.half_v = CELL_WV / 2.0

        # ---- Encoder e(x): continuous node features -> bi-GRU -> context -----
        self.x_feat = nn.Sequential(nn.Linear(N_NODE_FEAT, emb_dim), nn.ReLU(),
                                    nn.Linear(emb_dim, emb_dim))
        self.encoder = nn.GRU(emb_dim, enc_dim, batch_first=True, bidirectional=True)
        self.to_ctx = nn.Linear(2 * enc_dim + 1, ctx_dim)       # +1: hadron multiplicity
        self.drop = nn.Dropout(0.1)

        # ---- Decoder over y: cell-token embedding + context -> GRU -----------
        self.y_embed = nn.Embedding(n_cells + 1, emb_dim)       # +1 for START
        self.dec_in  = nn.Linear(emb_dim + ctx_dim, dec_dim)    # fuse token + e
        self.decoder = nn.GRU(dec_dim, dec_dim, batch_first=True)
        self.h0_proj = nn.Linear(ctx_dim, dec_dim)              # init hidden from e

        # ---- Heads -----------------------------------------------------------
        self.cont_head  = nn.Linear(dec_dim + ctx_dim, 1)       # continue/stop (single linear, as v1)
        self.split_head = nn.Sequential(nn.Linear(dec_dim + ctx_dim, dec_dim), nn.ReLU(),
                                        nn.Linear(dec_dim, n_cells))   # categorical cell
        # continuous coordinate head, conditioned on [h_t, e, chosen-cell embedding].
        # 8 outputs: du_raw, dv_raw, du_s, dv_s, lnz_mu, lnz_s, psi_a, psi_b.
        self.coord_head = nn.Sequential(
            nn.Linear(dec_dim + ctx_dim + emb_dim, dec_dim), nn.ReLU(),
            nn.Linear(dec_dim, 8))

        # cell-centre lookup buffers (move with .to(device) automatically)
        ix = torch.arange(n_cells) // N_BINS
        iy = torch.arange(n_cells) % N_BINS
        self.register_buffer("cell_cx", LN_INVDELTA_RANGE[0] + (ix + 0.5).float() * CELL_WU)
        self.register_buffer("cell_cy", LN_KT_RANGE[0] + (iy + 0.5).float() * CELL_WV)

    # -- encoder -------------------------------------------------------------
    def encode(self, xf: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """Mean-pool the bi-GRU outputs over valid x positions, append the hadron
        multiplicity, and project -> e  (B, ctx)."""
        B, Mx, _ = xf.shape
        out, _ = self.encoder(self.x_feat(xf))                  # (B, Mx, 2*enc)
        mask = (torch.arange(Mx, device=xf.device)[None, :] < nx[:, None]).float()
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        nx_feat = torch.log1p(nx.float()).unsqueeze(-1)
        return self.to_ctx(torch.cat([pooled, nx_feat], dim=-1))  # (B, ctx)

    # -- run the conditioned decoder over teacher-forced y tokens ------------
    def _decode_states(self, yc: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """GRU states h_0..h_L after consuming [START, y_0, ..., y_{L-1}]."""
        B, L = yc.shape
        start = torch.full((B, 1), self.n_cells, dtype=torch.long, device=yc.device)
        tokens = torch.cat([start, yc], dim=1)                  # (B, L+1)
        tok_emb = self.y_embed(tokens)                          # (B, L+1, emb)
        e_seq = e.unsqueeze(1).expand(-1, L + 1, -1)            # (B, L+1, ctx)
        inp = self.dec_in(torch.cat([tok_emb, e_seq], dim=-1))  # (B, L+1, dec)
        h0 = torch.tanh(self.h0_proj(e)).unsqueeze(0)           # (1, B, dec)
        out, _ = self.decoder(inp, h0)                          # (B, L+1, dec)
        return out

    # -- continuous coordinate head ------------------------------------------
    def _coord_params(self, coord_in: torch.Tensor):
        """Map [h_t, e, cell_emb] -> the continuous head's distribution parameters.
        Returns (du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, psi_mu, psi_kappa),
        each broadcasting over the leading dims of `coord_in`."""
        p = self.coord_head(coord_in)
        du_mean = self.half_u * torch.tanh(p[..., 0])           # bounded into the cell
        dv_mean = self.half_v * torch.tanh(p[..., 1])
        du_sig  = F.softplus(p[..., 2]) + 1e-2
        dv_sig  = F.softplus(p[..., 3]) + 1e-2
        lnz_mean = p[..., 4]
        lnz_sig  = F.softplus(p[..., 5]) + 1e-2
        a, b = p[..., 6], p[..., 7]
        kappa = torch.sqrt(a * a + b * b).clamp(1e-3, 50.0)     # von Mises concentration
        mu    = torch.atan2(b, a)                               # von Mises mean
        return du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, mu, kappa

    def _coord_logprob(self, params, u, v, lnz, psi, cx, cy):
        """log p(continuous coords | cell, h_t, e) for true/queried coords."""
        du_mean, dv_mean, du_sig, dv_sig, lnz_mean, lnz_sig, mu, kappa = params
        du = (u - cx).clamp(-self.half_u, self.half_u)
        dv = (v - cy).clamp(-self.half_v, self.half_v)
        ll  = _trunc_normal_logpdf(du, du_mean, du_sig, -self.half_u, self.half_u)
        ll += _trunc_normal_logpdf(dv, dv_mean, dv_sig, -self.half_v, self.half_v)
        ll += _gauss_logpdf(lnz, lnz_mean, lnz_sig)
        ll += _vonmises_logpdf(psi, mu, kappa)
        return ll

    def per_jet_nll(self, batch) -> torch.Tensor:
        """-log q_phi(y | x) per jet (B,), teacher forced over the true y, now
        including the continuous coordinate log-density."""
        xf, nx = batch["xf"], batch["nx"]
        yc, ny = batch["yc"], batch["ny"]
        yraw = batch["yraw"]                                    # (B, L, 4)
        e = self.encode(xf, nx)                                 # (B, ctx)
        out = self._decode_states(yc, e)                        # (B, L+1, dec)
        B, Lp1, _ = out.shape
        L = Lp1 - 1
        dev = yc.device

        eh = torch.cat([out, e.unsqueeze(1).expand(-1, Lp1, -1)], dim=-1)
        cont_logit = self.cont_head(eh).squeeze(-1)             # (B, L+1)

        idx = torch.arange(Lp1, device=dev).unsqueeze(0)
        n = ny.unsqueeze(1)
        cont_mask = (idx <= n).float()                          # positions 0..n valid
        cont_tgt  = (idx < n).float()                           # continue=1 at 0..n-1, stop=0 at n
        cont_ll = -F.binary_cross_entropy_with_logits(cont_logit, cont_tgt, reduction="none")
        cont_ll = (cont_ll * cont_mask).sum(1)                  # (B,)

        split_ll = torch.zeros(B, device=dev)
        coord_ll = torch.zeros(B, device=dev)
        if L > 0:
            eh_t = eh[:, :L, :]                                 # states h_0..h_{L-1}
            split_lp = F.log_softmax(self.split_head(eh_t), dim=-1)
            split_per = split_lp.gather(-1, yc.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, L)

            cell_emb = self.y_embed(yc.clamp(min=0))           # condition coords on the TRUE cell
            params = self._coord_params(torch.cat([eh_t, cell_emb], dim=-1))
            cx, cy = self.cell_cx[yc], self.cell_cy[yc]         # (B, L)
            coord_per = self._coord_logprob(params, yraw[..., 0], yraw[..., 1],
                                            yraw[..., 2], yraw[..., 3], cx, cy)  # (B, L)

            split_mask = (torch.arange(L, device=dev).unsqueeze(0) < n).float()
            split_ll = (split_per * split_mask).sum(1)
            coord_ll = (coord_per * split_mask).sum(1)

        return -(cont_ll + split_ll + coord_ll)                # (B,)

    def weighted_nll(self, batch) -> torch.Tensor:
        nll = self.per_jet_nll(batch)
        w = batch["w"]
        return (w * nll).sum() / w.sum().clamp(min=1e-8)

    # -- single-jet decoder step (for sampling / beam search) ----------------
    def _step(self, tok: torch.Tensor, e: torch.Tensor, h):
        inp = self.dec_in(torch.cat([self.y_embed(tok), e.unsqueeze(1)], dim=-1))
        out, h = self.decoder(inp, h)                           # (1, 1, dec)
        hv = torch.cat([out[:, -1, :], e], dim=-1)              # (1, dec+ctx)
        p_cont = torch.sigmoid(self.cont_head(hv)).item()
        logp_split = F.log_softmax(self.split_head(hv), dim=-1).squeeze(0)  # (N_CELLS,)
        return p_cont, logp_split, h

    @torch.inference_mode()
    def sample_batch(self, xf: torch.Tensor, nx: torch.Tensor,
                     n_samples: int, max_emissions: int = 25):
        """Draw `n_samples` posterior CELL chains y ~ q_phi(.|x) for ONE jet in
        parallel (single host sync at the end). Cell-level draws drive the
        closure/coverage diagnostics; the continuous coordinates are deterministic
        given the cell at the point estimate, so they are not sampled here (full
        continuous ancestral sampling -- truncated-normal + von Mises draws -- is
        a small add but unnecessary for the diagnostics or the MAP)."""
        self.eval()
        K = n_samples
        dev = xf.device
        e = self.encode(xf, nx).expand(K, -1).contiguous()      # (K, ctx)
        h = torch.tanh(self.h0_proj(e)).unsqueeze(0)            # (1, K, dec)
        tok = torch.full((K, 1), self.n_cells, dtype=torch.long, device=dev)
        alive = torch.ones(K, dtype=torch.bool, device=dev)
        cells   = torch.zeros(K, max_emissions, dtype=torch.long, device=dev)
        emitted = torch.zeros(K, max_emissions, dtype=torch.bool, device=dev)
        for t in range(max_emissions):
            inp = self.dec_in(torch.cat([self.y_embed(tok), e.unsqueeze(1)], dim=-1))
            out, h = self.decoder(inp, h)                       # (K, 1, dec)
            hv = torch.cat([out[:, -1, :], e], dim=-1)          # (K, dec+ctx)
            p_cont = torch.sigmoid(self.cont_head(hv)).squeeze(-1)          # (K,)
            cont = (torch.rand(K, device=dev) < p_cont) & alive
            draw = torch.multinomial(F.softmax(self.split_head(hv), dim=-1), 1).squeeze(-1)
            cells[:, t] = draw
            emitted[:, t] = cont
            alive = cont
            tok = draw.unsqueeze(1)
            if not bool(alive.any()):
                break
        cells_np   = cells.cpu().numpy()
        emitted_np = emitted.cpu().numpy()
        return [cells_np[k, emitted_np[k]].tolist() for k in range(K)]

    @torch.inference_mode()
    def map_decode(self, xf: torch.Tensor, nx: torch.Tensor,
                   beam_width: int = 8, topk_cells: int = 6, max_emissions: int = 25):
        """MAP cell structure argmax over (continue/stop, cell) by beam search.
        The continuous coordinates are attached afterwards at their conditional
        modes (staged MAP); the discrete cell choice dominates the structure."""
        self.eval()
        e = self.encode(xf, nx)
        h0 = torch.tanh(self.h0_proj(e)).unsqueeze(0)
        start = torch.full((1, 1), self.n_cells, dtype=torch.long, device=xf.device)

        active = [(0.0, [], h0, start)]
        finished = []
        for _ in range(max_emissions):
            cand = []
            for score, cells, h, tok in active:
                p_cont, logp_split, h_next = self._step(tok, e, h)
                p_cont = min(max(p_cont, 1e-8), 1 - 1e-8)
                finished.append((score + math.log(1 - p_cont), cells))         # STOP
                top = torch.topk(logp_split, k=min(topk_cells, logp_split.numel()))
                for lp, cell in zip(top.values.tolist(), top.indices.tolist()):
                    nt = torch.tensor([[cell]], dtype=torch.long, device=xf.device)
                    cand.append((score + math.log(p_cont) + lp, cells + [cell], h_next, nt))
            if not cand:
                break
            cand.sort(key=lambda b: b[0], reverse=True)
            active = cand[:beam_width]
        for score, cells, h, tok in active:
            p_cont, _, _ = self._step(tok, e, h)
            finished.append((score + math.log(min(max(1 - p_cont, 1e-8), 1.0)), cells))
        finished.sort(key=lambda b: b[0], reverse=True)
        return finished[0][1]                                   # best cell sequence

    # -- structured point estimate (with continuous coordinates) -------------
    @torch.inference_mode()
    def describe_sequence(self, xf: torch.Tensor, nx: torch.Tensor,
                          cells) -> LundPointEstimate:
        """Attach continuous coordinates (head modes) and the model's per-node +
        total log-density to a primary cell sequence. The total equals
        -per_jet_nll for (x, y_hat) when y_hat's continuous targets are these same
        modes -- i.e. it is the full joint log-density of the returned config."""
        self.eval()
        dev = xf.device
        L = len(cells)
        e = self.encode(xf, nx)                                 # (1, ctx)
        yc = torch.tensor([list(cells)], dtype=torch.long, device=dev)  # (1, L)
        out = self._decode_states(yc, e)                        # (1, L+1, dec)
        eh = torch.cat([out, e.unsqueeze(1).expand(-1, L + 1, -1)], dim=-1)
        cont_logit = self.cont_head(eh).squeeze(-1).squeeze(0)  # (L+1,)
        logp_cont = F.logsigmoid(cont_logit)                    # log P(continue)
        logp_stop = F.logsigmoid(-cont_logit)                   # log P(stop)

        nodes, total = [], 0.0
        if L > 0:
            eh_t = eh[:, :L, :]
            split_lp = F.log_softmax(self.split_head(eh_t), dim=-1).squeeze(0)  # (L, n_cells)
            chosen = split_lp.gather(-1, yc[0].unsqueeze(-1)).squeeze(-1)       # (L,)

            cell_emb = self.y_embed(yc)
            params = self._coord_params(torch.cat([eh_t, cell_emb], dim=-1))    # each (1, L)
            du_mean, dv_mean, _, _, lnz_mean, _, mu, _ = params
            cx, cy = self.cell_cx[yc], self.cell_cy[yc]         # (1, L)
            u_mode, v_mode = cx + du_mean, cy + dv_mean         # continuous coordinate modes
            coord_per = self._coord_logprob(params, u_mode, v_mode, lnz_mean, mu, cx, cy).squeeze(0)

            for t, c in enumerate(cells):
                c = int(c)
                lc, ls, lk = float(logp_cont[t]), float(chosen[t]), float(coord_per[t])
                total += lc + ls + lk
                u, v = float(u_mode[0, t]), float(v_mode[0, t])
                lz, ps = float(lnz_mean[0, t]), float(mu[0, t])
                nodes.append(LundNode(
                    depth=t, parent=t - 1, cell=c,
                    ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                    kt=math.exp(v), delta_R=math.exp(-u), z=math.exp(lz),
                    logp_split=ls, logp_coord=lk, logp_cont=lc))
        total += float(logp_stop[L])                            # final stop factor
        return LundPointEstimate(nodes=nodes, logprob=total, multiplicity=L)

    @torch.inference_mode()
    def map_tree(self, xf: torch.Tensor, nx: torch.Tensor, **beam_kwargs) -> LundPointEstimate:
        """The single most likely groomed parton-shower configuration as a
        structured primary Lund tree with continuous per-node coordinates:
        beam-search the cell structure, then attach the conditional coordinate
        modes and the full joint log-density."""
        cells = self.map_decode(xf, nx, **beam_kwargs)
        return self.describe_sequence(xf, nx, cells)


# ===========================================================================
#  CLOSURE / POSTERIOR DIAGNOSTICS
# ===========================================================================
def leading_emission_cell(cells):
    """Hardest (largest ln kt) primary emission cell -- the most perturbative,
    node-alignment-free observable. Returns its cell id or None."""
    if not cells:
        return None
    best, best_kt = cells[0], cell_center(cells[0])[1]
    for c in cells[1:]:
        kt = cell_center(c)[1]
        if kt > best_kt:
            best, best_kt = c, kt
    return best


def lund_distance(cell_a, cell_b):
    """Euclidean distance in (ln 1/DeltaR, ln kt) between two cell centres."""
    if cell_a is None or cell_b is None:
        return float("nan")
    ax, ay = cell_center(cell_a)
    bx, by = cell_center(cell_b)
    return math.hypot(ax - bx, ay - by)


def _tree_coords(obj):
    """Normalise a primary Lund tree to a list of (ln 1/DeltaR, ln kt, ln z, psi):
    a LundPointEstimate (model MAP) or an (n, 4) coordinate array / tensor (the
    plain-RSD hadron tree x, or the parton truth y)."""
    if isinstance(obj, LundPointEstimate):
        return [(n.ln_invDelta, n.ln_kt, n.ln_z, n.psi) for n in obj.nodes]
    arr = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    return [(float(u), float(v), float(lz), float(ps)) for u, v, lz, ps in arr]


def lund_tree_str(obj, title: str, ref=None) -> str:
    """Format a primary Lund tree as rows like LundPointEstimate.pretty, for the
    plain-RSD vs model vs truth comparison. A LundPointEstimate row carries the
    model per-node logP and its total log q in the title; an (n, 4) array row
    carries its Lund cell. When `ref` (the truth, as a LundPointEstimate or (n, 4)
    array) is given, each row also shows dLund = Euclidean (ln 1/DeltaR, ln kt)
    distance to the depth-aligned truth node -- the closeness-to-truth measure,
    smallest for hard splittings. (Per-node alignment is approximate when the
    multiplicities differ; the leading-emission distance is the alignment-free one.)"""
    is_pe = isinstance(obj, LundPointEstimate)
    coords = _tree_coords(obj)
    rcoords = _tree_coords(ref) if ref is not None else None
    head = (f"{title}: {len(coords)} primary splittings"
            + (f", log q(y_hat|x) = {obj.logprob:.3f}" if is_pe else ""))
    rows = []
    for t, (u, v, lz, ps) in enumerate(coords):
        if is_pe:
            n = obj.nodes[t]
            tail = f"  logP={n.logp_split + n.logp_coord:+.2f}"
        else:
            tail = f"  cell={to_cell(u, v):3d}"
        if rcoords is not None and t < len(rcoords):
            tail += f"  dLund={math.hypot(u - rcoords[t][0], v - rcoords[t][1]):.3f}"
        rows.append(f"  [{t}] kt={math.exp(v):6.2f} GeV  DeltaR={math.exp(-u):5.3f}  "
                    f"z={math.exp(lz):5.3f}  psi={ps:+5.2f}  "
                    f"(ln1/DR={u:4.2f}, lnkt={v:4.2f}, lnz={lz:5.2f}){tail}")
    return "\n".join([head, *rows]) if rows else head + "\n  (empty)"


# ===========================================================================
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ---- data: real RNTuple if present, else synthetic matched pairs --------
    jets = load_rntuple("jets.root", "Jets")
    if jets is None:
        jets = synthetic_matched_dataset(8000, seed=0)
    n_val = max(200, len(jets) // 10)
    train_jets, val_jets = jets[:-n_val], jets[-n_val:]

    train = MatchedLundDataset(train_jets)
    val   = MatchedLundDataset(val_jets)
    loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate, drop_last=True)

    model = ConditionalPrimaryLundJUNIPR().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    n_epochs = 20
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=3e-4)

    def move(batch):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    print(f"training on {len(train)} jets, validating on {len(val)} (device={device})")
    for epoch in range(n_epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            batch = move(batch)
            opt.zero_grad()
            loss = model.weighted_nll(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.inference_mode():
            vb = move(collate([val[i] for i in range(len(val))]))
            vnll = (vb["w"] * model.per_jet_nll(vb)).sum().item() / vb["w"].sum().item()
        print(f"epoch {epoch + 1:2d}   train NLL/jet = {tot / nb:8.3f}   val NLL/jet = {vnll:8.3f}")

    # ---- CLOSURE + CALIBRATION on held-out jets (cell-level, as v1) ---------
    K = 200
    d_id, d_mode = [], []
    n_id_bias, n_mean_bias = [], []
    covered = []
    n_closure = min(300, len(val))
    for i in range(n_closure):
        item = val[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        y_true = item["yc"].tolist()
        x_cells = seq_cells(val_jets[i]["x"][0], val_jets[i]["x"][1]).tolist()
        ny_true = len(y_true)

        draws = model.sample_batch(xf, nx, K)
        mults = np.array([len(d) for d in draws])
        lead = [c for c in (leading_emission_cell(d) for d in draws) if c is not None]
        ly = leading_emission_cell(y_true)
        if ly is None or not lead:
            continue

        cells_arr = np.array(lead)
        vals, counts = np.unique(cells_arr, return_counts=True)
        mode_cell = int(vals[counts.argmax()])
        d_mode.append(lund_distance(mode_cell, ly))
        d_id.append(lund_distance(leading_emission_cell(x_cells), ly))

        n_id_bias.append(len(x_cells) - ny_true)
        n_mean_bias.append(mults.mean() - ny_true)

        order = np.argsort(-counts)
        cum = np.cumsum(counts[order]) / counts.sum()
        k68 = int(np.searchsorted(cum, 0.68)) + 1
        hpd_set = set(int(c) for c in vals[order][:k68])
        covered.append(1.0 if ly in hpd_set else 0.0)

    d_id, d_mode = np.array(d_id), np.array(d_mode)
    print("\nclosure + calibration on held-out jets:")
    print(f"  mean multiplicity            :  true y = {np.mean([len(val[i]['yc']) for i in range(n_closure)]):.2f}"
          f"   hadron x = {np.mean([len(val_jets[i]['x'][0]) for i in range(n_closure)]):.2f}"
          f"   posterior = {np.mean([b + len(val[i]['yc']) for i, b in enumerate(n_mean_bias)]):.2f}")
    print(f"  leading-emission Lund distance to true y :  identity(x) = {np.nanmean(d_id):.3f}"
          f"   posterior-mode = {np.nanmean(d_mode):.3f}   (lower is better)")
    print(f"  multiplicity signed bias  <n - n_true>   :  identity(x) = {np.mean(n_id_bias):+.3f}"
          f"   posterior-mean = {np.mean(n_mean_bias):+.3f}   (closer to 0 is better)")
    print(f"  posterior 68% coverage of true leading cell = {np.mean(covered):.2f}"
          f"   (target ~0.68; <0.68 => over-confident)")

    # ---- POINT ESTIMATE: plain RSD (hadron x) vs model MAP vs truth (one jet) -
    item = val[0]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)
    draws = model.sample_batch(xf, nx, 500)
    mults = np.array([len(d) for d in draws])
    y_hat = model.map_tree(xf, nx)          # structured MAP with continuous coordinates

    # plain RSD: the hadron-level groomed tree x used directly as the estimate
    # (the identity(x) baseline) -- no inversion, just what RSD on hadrons gives.
    x_raw   = node_raw(*val_jets[0]["x"])
    y_truth = item["yraw"]
    lead_truth = leading_emission_cell(item["yc"].tolist())
    d_rsd = lund_distance(leading_emission_cell(seq_cells(*val_jets[0]["x"][:2]).tolist()), lead_truth)
    d_map = lund_distance(leading_emission_cell([n.cell for n in y_hat.nodes]), lead_truth)

    print("\nper-jet point estimate q_phi(y | x) for one validation jet:")
    print(f"  multiplicity:  truth y = {item['ny']}   model MAP = {y_hat.multiplicity}   "
          f"plain RSD (hadron x) = {len(x_raw)}   "
          f"posterior = {mults.mean():.2f} +/- {mults.std():.2f} "
          f"(68% CR [{np.percentile(mults,16):.0f}, {np.percentile(mults,84):.0f}])")
    print(f"  leading-emission Lund distance to truth:  plain RSD = {d_rsd:.3f}   "
          f"model MAP = {d_map:.3f}   (lower is better)")
    print("\n" + lund_tree_str(y_hat, "model MAP groomed shower", ref=y_truth))
    print("\n" + lund_tree_str(x_raw, "plain RSD groomed shower (hadron-level x)", ref=y_truth))
    print("\n" + lund_tree_str(y_truth, "true groomed shower"))


if __name__ == "__main__":
    main()
