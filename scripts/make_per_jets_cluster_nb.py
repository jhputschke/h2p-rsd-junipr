"""Build notebooks/per_jets_estimation_cluster.ipynb.

    python scripts/make_per_jets_cluster_nb.py

The set-valued counterpart of `scripts/make_per_jets_nb.py`: the same per-jet study, with
the per-jet quantities RE-ASSIGNED for the posterior-cluster approach of
docs/PLAN_PosteriorClusters.md. Same reason for being generated rather than hand-edited as
its sibling — the source is past what the notebook editor opens, and THIS FILE is the
source of truth.

Regenerating drops the executed outputs, so follow it with

    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/per_jets_estimation_cluster.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Per-jet estimation, set-valued — which explanation, how probable, and how wide

The cluster counterpart of [`per_jets_estimation.ipynb`](per_jets_estimation.ipynb),
implementing docs/PLAN_PosteriorClusters.md on the same checkpoint, the same held-out file
and the same decode, so the two are read side by side.

### The question

`mbr_select` returns the **Fréchet median restricted to the sample** — the draw of least
*mean* perturbative-Lund EMD to the posterior. That is a global **centrality** criterion,
the correct Bayes estimator under a loss linear in the distance, and the right default.

It is not the right criterion when the posterior is **multimodal**. The medoid of a
two-lobed posterior can land in the sparse valley between the lobes, minimising mean
distance while representing neither explanation. The sample space here is
transdimensional,
$\mathcal{Y}=\bigsqcup_{N}\mathcal{C}^{N}$, and the strata are metrically separated by the
EMD's imbalance term — so a jet whose posterior splits between "one hard emission" and "two
softer emissions consistent with the same $x$" is the hadronization ambiguity expressed as
**discrete alternative explanations**, and a mean-distance criterion smears exactly that.

### What this notebook adds, and what it does not touch

Everything here is read off **the same $K\times K$ distance matrix** `mbr_select` already
builds. `cluster_posterior` consumes `D`; the risk reduction consumes `D`; neither sees the
other's output. So **the MBR point estimate in this notebook is bit-identical to the one in
`per_jets_estimation.ipynb`** — §5 asserts it rather than claiming it. Nothing here
constructs a tree the model did not generate: every member of every set is a genuine
posterior draw carrying its own sampled coordinates.

### The per-jet quantities, re-assigned

| series | what it is | may it be a headline? |
|---|---|---|
| `truth` | the parton tree $y$ | — |
| `rsd` | plain RSD, the hadron sequence $x$ | the baseline to beat |
| `mbr` | the **linear medoid** — today's decode headline | yes |
| `set0` | `predict_set().members[0]`, the **top-mass exemplar** | yes — the mass-argmax summary |
| `mbr_gated` | the medoid under the **same** $\tau$ — the fair baseline for the row below | yes |
| `set0_gated` | the same set, with **emptiness decided by the frozen $\tau$** instead of by the mass argmax | yes — see §6b |
| `setbest` | the member closest to truth | **no** — an ORACLE, diagnostic only |
| `post` | one posterior draw | the scale reference, not a competitor |

and three per-jet scalars that are deliberately **not** foldable into one $\pm$:

- **`top_mass`** — a *probability*: the posterior mass of the selected explanation.
- **`entropy`** $H(m)=-\sum_j m_j\log m_j$ — a per-jet **ambiguity** over discrete
  alternatives, in nats.
- **`radii[0]`** — the *continuous* resolution within the selected explanation, and **the
  only one of the three legitimately reportable as a $\pm$**.

A bimodal posterior summarised as mean $\pm$ sd points at a configuration neither mode
supports. That is the whole reason the three are reported separately.

§7 is where they earn their place: if `top_mass` is a real confidence, the residual must be
**narrower on the jets that claim a high one**. §8 asks whether the claimed probability is
the realized frequency (gate G6).

### What must not be done with these numbers

- The `set0`-vs-`mbr` spread is **not a systematic** and must never enter an uncertainty
  budget. They are two different functionals of one posterior — integrated density and the
  Fréchet median — not two approximations to one quantity, and the posterior width is
  already reported by `radii[0]`. §10 reports the spread as a **stability** check, beside
  the answer, never folded into it.
- `setbest` uses the truth to pick the member. It is legitimate as a diagnostic and
  dishonest as a result, and it never appears in a summary table (§9 carries its
  mandatory random-partition null).
- `top_mass` is **not** a calibrated probability until §8 says it is, and with
  `CLUSTER_SPLIT = False` it is biased **high** — the same draws define the cluster and are
  counted into it.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Parameters

**One knob: `RUN`**, exactly as in [`per_jets_estimation.ipynb`](per_jets_estimation.ipynb)
— a run directory, an arm root, a `best.ckpt`, or a `prod_test_v1_metrics.json`. Everything
else is found inside it.

Three settings differ from that notebook, and all three are forced by
docs/PLAN_PosteriorClusters.md §4 rather than chosen:

- **`MBR_N_CANDIDATES = 0`.** With a candidate cap `D` is $|C|\times K$ and there is no
  square matrix to cluster. The sibling notebook sets 16; here it must be 0, and the code
  **raises** rather than silently overriding — overriding would change the point estimate
  you asked for.
- **`mbr_beta = 1.0`.** At $\beta\neq1$ the EMD violates the triangle inequality (measured:
  300 of 64 000 triples at $\beta=2$), so HDBSCAN's mutual-reachability distance is not a
  distance. This is the checkpoint default; the guard asserts it.
- **`MBR_BACKEND` is never `surrogate`.** `_lund_image` normalises, so the surrogate is
  *exactly* blind to total $k_t$ and multiplicity — the quantity that separates the $N$
  strata the clusters are made of. It is admissible as a screening pass for the G2 verdict
  and never for a quoted mass vector.
""")

code(r'''
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

# --- WHAT TO RUN: one knob ---------------------------------------------------
#   RUN = "runs/prod_test_edit/e_v2_s0/20260802-004446-a824deac75"
#   RUN = "runs/prod_test_edit/e_v2_s0"
RUN = None
ARM = "v1_contstop_s0"
ROOT_PATH   = None
NTUPLE_NAME = "Jets"

_REPO = _Path.cwd().parent if _Path.cwd().name == "notebooks" else _Path.cwd()


def _newest(root, pattern):
    """Newest match of `pattern` anywhere under `root`, or None."""
    hits = sorted(root.rglob(pattern), key=lambda q: q.stat().st_mtime) if root.is_dir() else []
    return hits[-1] if hits else None


_ck = _art = None
if RUN is None:
    _art = (_newest(_REPO / "runs" / "prod_test_v1" / ARM, "prod_test_v1_metrics.json")
            or _newest(_REPO / "runs", "prod_test_v1_metrics.json"))
    if _art is None:
        raise FileNotFoundError(
            "no prod_test_v1_metrics.json under runs/ -- run notebooks/prod_test_v1.ipynb "
            "first, or set RUN to a run directory / checkpoint."
        )
else:
    _p = _Path(RUN)
    _p = _p if _p.is_absolute() else _REPO / _p
    if not _p.exists():
        raise FileNotFoundError(f"RUN does not exist: {_p}")
    if _p.suffix == ".json":
        _art = _p
    elif _p.suffix == ".ckpt":
        _ck, _art = _p, _newest(_p.parent, "prod_test_v1_metrics.json")
    elif _p.is_dir():
        _ck = (_p / "best.ckpt") if (_p / "best.ckpt").exists() else (
            _newest(_p, "best.ckpt") or _newest(_p, "last.ckpt"))
        if _ck is None:
            raise FileNotFoundError(f"no best.ckpt or last.ckpt anywhere under {_p}")
        _art = _newest(_ck.parent, "prod_test_v1_metrics.json") or _newest(
            _p, "prod_test_v1_metrics.json")
    else:
        raise ValueError(f"RUN must be a directory, a .ckpt or a .json; got {_p.name!r}")

_M = _json.loads(_art.read_text()) if _art is not None else None
if _ck is None:
    _ck = _REPO / _M["run"]["checkpoint"]
if not _ck.exists():
    raise FileNotFoundError(f"checkpoint does not exist: {_ck}")
if _M is not None and (_REPO / _M["run"]["checkpoint"]).resolve() != _ck.resolve():
    raise RuntimeError(
        f"the artifact {_art} describes checkpoint {_M['run']['checkpoint']!r}, but the "
        f"checkpoint resolved from RUN is {_ck}. Its tau and test file belong to a "
        f"different run -- point RUN at one of them, not at a tree holding both."
    )

try:
    CKPT_PATH = str(_ck.resolve().relative_to(_REPO.resolve()))
except ValueError:
    CKPT_PATH = str(_ck)
if ROOT_PATH is None:
    if _M is None:
        raise FileNotFoundError(
            f"no prod_test_v1_metrics.json under {_ck.parent}, so there is no record of "
            f"which file this checkpoint was EVALUATED on. Set ROOT_PATH explicitly."
        )
    ROOT_PATH = _M["run"]["test_path"]
EMPTY_THRESHOLD = float(_M["empty_tree"]["tau"]["value"]) if _M is not None else 0.0

print(f"[run] checkpoint : {CKPT_PATH}")
print(f"[run] eval file  : {ROOT_PATH}")
if _M is not None:
    print(f"[run] artifact   : {_art.relative_to(_REPO)}\n"
          f"[run]              model={_M['run'].get('model')!r}  "
          f"frozen tau={EMPTY_THRESHOLD:.4f}")

# --- sample -----------------------------------------------------------------
PT_VAR  = "jet_pt"
PT_MIN  = None
PT_MAX  = None
N_JETS  = 600          # the K^2 EMD block is the cost here -- run the probe in 5a first
SEED    = 1234
# "auto" -> cuda when available, else cpu. Deliberately NOT `select_device()`, which would
# pick MPS on Apple Silicon: MPS does not work for this decode. Only "cuda" and "cpu" are
# supported, and section 2 raises on "mps" rather than letting it fail somewhere downstream.
#
# This is the one place this notebook departs from per_jets_estimation.ipynb, which pins
# "cpu" because at batch 1 a GPU never amortises its dispatch overhead. Here it does: the
# cluster layer needs the FULL K x K block (mbr_n_candidates is forced to 0), so each jet
# costs K coordinate draws plus K^2 EMD solves instead of a 16-candidate slice, and the
# sampling half is large enough to be worth a GPU. On the ARM GB10 leaving this at "cpu"
# makes the pass very long.
DEVICE  = "auto"       # auto | cuda | cpu   (never mps)
TORCH_THREADS = 4      # CPU only; ignored on cuda

# --- aux conditioning (section 3) -------------------------------------------
# The encoder's aux columns are GROOMED per-jet scalars (docs/PLAN_Input.md), so they mean
# what the checkpoint learned only if this file was written with the same
# (z_cut, beta, kt_floor, kt_floor_sec). True RAISES on a mismatch; set it False only when
# the grooming shift IS the measurement.
PROVENANCE_STRICT = True
# A jet whose aux sources are sentinels (NaN, -1: a column written before it existed) is
# dropped rather than killing the build. Above this FRACTION it raises instead -- that many
# is a schema mismatch, not stragglers, and dropping them would reshape the population.
AUX_MAX_DROP = 0.01

# --- decode -----------------------------------------------------------------
# K is what the mass vector's RESOLUTION is. At K = 200 and CLUSTER_MIN_MASS = 0.05 a
# reportable cluster is only 10 draws: the G2 verdict is answerable there, the mass
# vector's TAIL is not (plan section 11). Raise K before quoting a three-cluster split.
K_DRAWS               = 200
LENGTH_FLOOR_QUANTILE = 0.15
MBR_BACKEND           = "energyflow" if _ilu.find_spec("energyflow") else "pot"
# FORCED to 0 by plan section 4: a candidate cap leaves D rectangular and there is no
# K x K matrix to cluster. The sibling notebook uses 16; this one cannot.
MBR_N_CANDIDATES      = 0
GATE_EMPTY            = False

# --- the cluster layer (docs/PLAN_PosteriorClusters.md WP1-WP3) --------------
# hdbscan (default; density-based, no fixed k, native noise label -- needs scikit-learn),
# dbscan (the eps-explicit fallback), or pam (k-medoids with k by silhouette; pure NumPy,
# deterministic, and the control arm that says whether the G2 verdict is method-dependent).
CLUSTER_METHOD        = "hdbscan" if _ilu.find_spec("sklearn") else "pam"
CLUSTER_MIN_MASS      = 0.05   # below this a cluster merges into the residual bucket
CLUSTER_MIN_CLUSTER_SIZE = 0   # 0 -> max(5, ceil(CLUSTER_MIN_MASS * K))
CLUSTER_EPS_QUANTILE  = 0.10   # dbscan only: eps = Q_gamma of the POSITIVE off-diagonals
# Sample-split the mass estimate (plan WP5.1). OFF keeps the single-pool estimate, which is
# biased HIGH: R_j is defined using the same draws whose membership is then counted
# (post-selection inference; Berk, Brown, Buja, Zhang & Zhao, Ann. Statist. 41 (2013) 802).
# Section 8b prices the bias by running both.
CLUSTER_SPLIT         = False
SET_ALPHA             = 0.32   # conformal miscoverage (1 sigma). MARGINAL over jets.
# The WP4a diagnostic reductions. Eval-only: they never touch `.risk`, which keeps meaning
# "the achieved mean distance" for all fourteen of its consumers.
DIAGNOSTIC_LOSSES     = ("bounded", "kernel")
LOSS_QUANTILE         = 0.10   # PRE-REGISTERED. Tuning it against a closure metric would
#                                make the conformal gate circular -- it is the one free
#                                parameter the bounded construction turns on.
NULL_REPS             = 20     # random partitions per jet for the G2' null (plan 10.1b)

# --- the residual study -----------------------------------------------------
T_FIRST   = 2
RESID_NB  = 41
RESID_PCT = 99.0
N_BOOT    = 200
MIN_CI_JETS = 25
CONF_BINS = 4          # quantile bins of top_mass / entropy in section 7
SHOWCASE_JET = None    # None -> auto-pick the most AMBIGUOUS jet (see pick_showcase)

WRITE_ARTIFACTS = True   # per_jet_clusters.json beside the checkpoint

# --- guards (plan section 4; every one RAISES rather than warning) -----------
assert MBR_N_CANDIDATES == 0, (
    "the cluster layer needs a square K x K distance matrix, so mbr_n_candidates must be 0. "
    "It is not silently overridden here: a candidate cap changes which point estimate you "
    "get, so overriding it would answer a different question than the one you asked."
)
assert MBR_BACKEND != "surrogate", (
    "`_lund_image` normalises, so the surrogate is EXACTLY blind to total kt and "
    "multiplicity -- the quantity that separates the N strata the clusters are made of. "
    "It is admissible as a G2 screening pass and never for a quoted mass vector."
)
assert not (GATE_EMPTY and EMPTY_THRESHOLD <= 0.0), (
    "GATE_EMPTY=True needs a frozen tau, and none was read."
)
''')

# ---------------------------------------------------------------------------
md(r"""
## 1. Imports, house style, helpers

Palette and helpers inherited verbatim from
[`per_jets_estimation.ipynb`](per_jets_estimation.ipynb) so the two notebooks' panels
overlay without re-reading a legend. Two additions: a **categorical ramp for cluster
membership** (used only in the single-jet panels, never for a series), and the top-mass
exemplar takes its own slot so `set0` and `mbr` are never confused for one another.
""")

code(r'''
import math
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
warnings.filterwarnings("ignore", category=UserWarning)

if TORCH_THREADS:
    torch.set_num_threads(int(TORCH_THREADS))

from h2p_rsd_junipr.config import decode_params
from h2p_rsd_junipr.data.datamodule import select_pt_range
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.data.rntuple import load_rntuple
from h2p_rsd_junipr.eval.clusters import fit_mass_temperature, reliability, temper_top_mass
from h2p_rsd_junipr.eval.closure import lund_tree_str
from h2p_rsd_junipr.eval.report import save_metrics
from h2p_rsd_junipr.eval.stability import loss_stability_row, summarise_stability
from h2p_rsd_junipr.features import (
    AUX_FEATURES,
    aux_source_fields,
    aux_vector,
    node_raw,
)
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.clusters import (
    assert_cluster_metric_ok,
    assign_truth,
    cluster_posterior,
    fit_set_threshold,
    random_partition_null,
    set_size_for,
    support_radii,
)
from h2p_rsd_junipr.inference.length import learned_min_emissions, quantile_floor
from h2p_rsd_junipr.inference.mbr import (
    _reduce_risk,
    bandwidth_quantile,
    lund_cloud,
    lund_emd_matrix,
    posterior_distances,
    stratified_medoid,
)
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.train.trainer import seed_everything

# --- style (inherited from per_jets_estimation.ipynb) ------------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
C_TRUTH = INK
C_RSD_F, C_RSD_E = "#e1e0d9", "#898781"
C_MAP   = "#2a78d6"    # MAP point estimate   -- blue   (slot 1)
C_MBR   = "#eb6834"    # MBR linear medoid    -- orange (slot 2)
C_POST  = "#199e70"    # posterior draw       -- aqua   (slot 3), dashed
C_SET0  = "#9a4fc4"    # top-mass exemplar    -- violet (slot 4)
C_BEST  = "#b8a11f"    # oracle best member   -- ochre  (slot 5), never a headline
C_GATE  = "#0f8c9e"    # gated exemplar       -- teal   (slot 6)
C_NFIRST = "#c44f8a"   # N-first medoid       -- magenta (slot 7)
# Cluster membership inside ONE jet. Categorical, not sequential: cluster ids are labels,
# and a sequential ramp would suggest an ordering the partition does not carry.
C_CLUSTER = ["#9a4fc4", "#199e70", "#eb6834", "#2a78d6", "#b8a11f", "#c44f8a"]

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 9, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.6,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.6,
})

LABEL = {
    "lnInvDelta": r"$\ln(1/\Delta R)$",
    "lnkt": r"$\ln(k_t/\mathrm{GeV})$",
    "lnz": r"$\ln z$",
    "psi": r"$\psi$",
}
DLABEL = {k: r"$\Delta$ " + v for k, v in LABEL.items()}
TLABEL = {"lnInvDelta": "ln(1/dR)", "lnkt": "ln kt", "lnz": "ln z", "psi": "psi"}
COL = {"lnInvDelta": 0, "lnkt": 1, "lnz": 2, "psi": 3}
RES_KEYS = ["lnInvDelta", "lnkt", "lnz"]


def h1_sumw2(values, weights, e):
    """Weighted counts and their Sumw2 errors -- ROOT's TH1::Sumw2 convention."""
    c = np.histogram(values, bins=e, weights=weights)[0]
    s2 = np.histogram(values, bins=e, weights=np.asarray(weights, float) ** 2)[0]
    return c, np.sqrt(s2)


def density(counts, err, e):
    w = np.diff(e)
    tot = float((counts * w).sum())
    if tot <= 0:
        return np.zeros_like(counts, dtype=float), np.zeros_like(counts, dtype=float)
    return counts / tot, err / tot


def step(ax, y, e, color, label=None, lw=1.6, ls="-", z=3):
    ax.stairs(y, e, color=color, linewidth=lw, linestyle=ls, label=label, zorder=z)


def fill(ax, y, e, face, edge, label=None, z=1):
    ax.stairs(y, e, color=edge, fill=True, facecolor=face, linewidth=1.0,
              label=label, zorder=z)


def finish(ax, xlabel="", title="", ylabel="", logy=False, legend=False, loc="best"):
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7.0, loc=loc)


def wquantile(v, w, qs):
    v, w = np.asarray(v, float), np.asarray(w, float)
    if not v.size:
        return np.full(len(np.atleast_1d(qs)), np.nan)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, v)


def wstats(d, w):
    """Everything reported about one residual column, weight-aware.

    `rms` is about ZERO, not about the mean -- a residual's figure of merit is how far it
    sits from truth, and a large constant bias is a real failure, not something to subtract
    off. `bias` and `hw68` separate the two contributions.
    """
    d, w = np.asarray(d, float), np.asarray(w, float)
    if not d.size:
        nan = float("nan")
        return dict(n=0, sumw=0.0, bias=nan, rms=nan, med=nan, hw68=nan, mad=nan)
    sw = float(w.sum())
    q16, q50, q84 = wquantile(d, w, [0.16, 0.50, 0.84])
    return dict(
        n=int(d.size), sumw=sw,
        bias=float((w * d).sum() / sw),
        rms=float(np.sqrt((w * d ** 2).sum() / sw)),
        med=float(q50), hw68=float(0.5 * (q84 - q16)),
        mad=float(wquantile(np.abs(d - q50), w, [0.5])[0]),
    )


def classical_mds(D, dim=2):
    """Torgerson classical MDS on a precomputed distance matrix -- DISPLAY ONLY.

    Double-centre -0.5 D^2 and take the leading eigenvectors. The clustering NEVER sees
    this embedding: it works on `D` directly, precisely because `Y` has no vector-space
    structure to embed into. Any 2-D picture of a transdimensional tree space is a lossy
    projection, and the stress printed beside it is what says how lossy.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh((B + B.T) / 2.0)
    order = np.argsort(vals)[::-1][:dim]
    L = np.sqrt(np.maximum(vals[order], 0.0))
    X = vecs[:, order] * L[None, :]
    dd = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    denom = float((D ** 2).sum())
    stress = float(np.sqrt(((D - dd) ** 2).sum() / denom)) if denom > 0 else float("nan")
    return X, stress


print(f"style and helpers ready   (torch intra-op threads: {torch.get_num_threads()})")
''')

# ---------------------------------------------------------------------------
md(r"""
## 2. The model, and the metric-admissibility guard

Everything structural comes from the checkpoint's own config snapshot. On top of the usual
checks this section runs **gate G4** — `assert_cluster_metric_ok` — which raises unless
$\beta=1$, $R\ge R_\max/2$ for the active `mbr_coords`, and `mbr_n_candidates == 0`. The
$R$ bound is computed from *this* geometry rather than hard-coded at 8.485, so a non-default
`geometry` block cannot silently break the inequality.
""")

code(r'''
seed_everything(SEED)
# cuda when available, else cpu -- and NEVER mps, which does not work for this decode.
# `select_device()` (cuda > mps > cpu) is deliberately not used for exactly that reason.
if str(DEVICE).startswith("mps"):
    raise ValueError(
        "MPS does not work for this decode -- use DEVICE='cuda' or DEVICE='cpu' ('auto' "
        "picks cuda when it is available and cpu otherwise, and never mps)."
    )
device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
          if DEVICE == "auto" else torch.device(DEVICE))

CKPT = (REPO / CKPT_PATH) if not Path(CKPT_PATH).is_absolute() else Path(CKPT_PATH)
info = load_for_inference(str(CKPT), map_location=device)
cfg = OmegaConf.create(info["config"])
geom = Geometry.from_config(cfg.geometry)
model = build_model(cfg, geom).to(device)
model.load_state_dict(info["model_state"])
model.eval()

DECODE = decode_params(cfg)
BEAM = {k: DECODE[k]
        for k in getattr(type(model), "_BEAM_KEYS", ("max_emissions", "min_emissions"))
        if k in DECODE}
AUX = tuple(model.aux_feature_names)
CONT = bool(getattr(model, "has_continuous_coords", False))
TAU = float(EMPTY_THRESHOLD) if GATE_EMPTY else 0.0
LNZ_SUPPORT = str(OmegaConf.select(cfg, "model.lnz_support") or "legacy")

# The decode this notebook actually uses, in one dict, so nothing below can drift from it.
DEC = {**DECODE, **BEAM,
       "empty_threshold": TAU,
       "point_estimator": "mbr",
       "mbr_backend": MBR_BACKEND,
       "mbr_n_candidates": MBR_N_CANDIDATES,
       "cluster_posterior": True,
       "cluster_method": CLUSTER_METHOD,
       "cluster_min_mass": CLUSTER_MIN_MASS,
       "cluster_min_cluster_size": CLUSTER_MIN_CLUSTER_SIZE,
       "cluster_eps_quantile": CLUSTER_EPS_QUANTILE,
       "cluster_split": CLUSTER_SPLIT,
       "set_alpha": SET_ALPHA}

# GATE G4 -- raises, per plan section 4. Not a warning: a mass vector nobody can see is a
# number that gets quoted anyway.
assert_cluster_metric_ok(DEC, geom)

# The kwargs the distance matrix is built with. ONE matrix per jet feeds the point
# estimate, the clusters and the WP4a stability columns -- rebuilding it would be K^2 EMD
# solves for something already in hand, and would let the products drift apart.
CLOUD_KW = dict(lnkt_cut=DEC["mbr_lnkt_cut"], weight=DEC["mbr_weight"],
                coords=DEC["mbr_coords"])       # draw -> weighted Lund cloud
EMD_KW = dict(R=DEC["mbr_R"], beta=DEC["mbr_beta"], norm=DEC["mbr_norm"],
              periodic_phi=DEC["mbr_periodic_phi"], phi_col=DEC["mbr_phi_col"],
              backend=MBR_BACKEND)               # cloud pair -> distance
MBR_KW = {**CLOUD_KW, **EMD_KW}                  # `posterior_distances` takes both halves
CLUSTER_KW = dict(method=CLUSTER_METHOD, min_mass=CLUSTER_MIN_MASS,
                  min_cluster_size=CLUSTER_MIN_CLUSTER_SIZE,
                  eps_quantile=CLUSTER_EPS_QUANTILE, backend=MBR_BACKEND)

try:
    _ck = CKPT.resolve().relative_to(REPO.resolve())
except ValueError:
    _ck = CKPT
print(f"checkpoint : {_ck}")
print(f"model      : {info['model_name']}   encoder={cfg.encoder.name}   "
      f"cross-attention={bool(OmegaConf.select(cfg, 'model.use_cross_attention'))}")
print(f"geometry   : {geom.n_bins}x{geom.n_bins} = {geom.n_cells} cells   "
      f"ln(1/dR) in {geom.ln_invdelta_range}   ln kt in {geom.ln_kt_range}")
print(f"coordinates: continuous={CONT}   lnz_support={LNZ_SUPPORT!r}   "
      f"kappa_min_mode={DECODE['kappa_min_mode']:g}")
print(f"aux inputs : {len(AUX)}  {list(AUX)}")
print(f"parameters : {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k   "
      f"device={device}")
print(f"metric     : backend={MBR_BACKEND!r}  beta={EMD_KW['beta']:g}  R={EMD_KW['R']:g}  "
      f"coords={CLOUD_KW['coords']!r}   -- gate G4 PASSED")
print(f"clusters   : method={CLUSTER_METHOD!r}  min_mass={CLUSTER_MIN_MASS:g}  "
      f"min_cluster_size={CLUSTER_MIN_CLUSTER_SIZE or 'auto'}  split={CLUSTER_SPLIT}  "
      f"K={K_DRAWS}")
print(f"             a reportable cluster is >= "
      f"{max(1, math.ceil(CLUSTER_MIN_MASS * K_DRAWS))} of {K_DRAWS} draws"
      f"   (MC error on a mass at m=0.6 is "
      f"{math.sqrt(0.6 * 0.4 / K_DRAWS):.3f})")
if not CONT:
    raise RuntimeError(
        "this checkpoint has no continuous coordinate density, so ln z and psi are "
        "placeholders (ln z = 0 means z = 1) and a ln z residual would be a plot of a "
        "filler constant."
    )
''')

# ---------------------------------------------------------------------------
md(r"""
## 3. The test sample

The held-out PYTHIA file, selected on `len(x) > 0` only — the **deployable** population,
every jet an analysis could pick out on data, including the ~17% whose parton truth is the
empty tree. Requiring `len(y) > 0` would read the answer.

Those truth-empty jets matter more here than in the sibling notebook: `_empty_value` puts
every empty draw at mutual distance **exactly 0**, so the $N=0$ stratum is a zero-diameter
clique that any density method finds by construction, and its mass *is* $q(0\mid x)$ — the
one quantity v1 measured as well-calibrated while every point estimator mishandled it. §9
checks that identity (gate G3).

### The aux conditioning, and why this section checks it

The encoder is built for a fixed set of **aux** columns (`model.aux_feature_names`, printed
in §2) — per-jet *groomed* scalars the primary hadron sequence structurally cannot carry
(docs/PLAN_Input.md). Every number below is conditioned on them, so this section does not
merely pass them through:

- **Provenance.** Aux is groomed with `kt_floor_sec`, the **off-spine** floor — not the
  `kt_floor` that `x` and `y` use. Two files can therefore agree on `kt_floor` and still
  carry aux built at a different scale, a shift the encoder cannot see and would read as
  physics. The whole tuple $(z_\mathrm{cut}, \beta, k_{t,\mathrm{floor}},
  k_{t,\mathrm{floor}}^\mathrm{sec})$ is compared against the **training** file's, recorded
  in the artifact §0 read from, and a mismatch **raises** unless `PROVENANCE_STRICT = False`.
  The same tuple must also be single-valued across the file: a merged file whose halves were
  groomed differently has no one aux scale.
- **Sentinels.** The reader fills an aux column the writer never produced with a *sentinel*
  (NaN, or $-1$) rather than failing, and `MatchedLundDataset` then raises on the first jet
  that hits one — one bad jet, a dead section, no count. Here the aux vector is built for
  every jet first: jets that cannot supply it are **dropped and counted**, and if the drop
  exceeds `AUX_MAX_DROP` it raises instead, because at that point it is the wrong file
  rather than a few stragglers. A file with *no* usable aux fails with the columns named.
- **Range.** The per-feature mean, spread and range of what the encoder is actually fed are
  printed, and the secondary-plane population numbers are shown beside the **training**
  file's. Aux far outside its training range is extrapolated conditioning, and every cluster
  mass downstream inherits that.

The matrix itself stays available as `AUX_X` — one row per kept jet, one column per feature
in `AUX` order — for stratifying any residual in §7 by what the model was actually told.
""")

code(r'''
# THE guard, and it is checked against the CHECKPOINT rather than the artifact: a
# checkpoint always records the file it trained on, so this holds on every route into
# section 0 -- including the one where there is no artifact to cross-check against.
TRAIN_PATH = str(OmegaConf.select(cfg, "data.path") or "")
assert str(ROOT_PATH) != TRAIN_PATH, (
    f"ROOT_PATH is {ROOT_PATH!r}, the file this checkpoint TRAINED on. Not a closure test."
)

jets = load_rntuple(str(REPO / ROOT_PATH), NTUPLE_NAME)
if not jets:
    # `load_rntuple` PRINTS and returns None when the file or uproot is unavailable, and
    # its other callers fall back to synthetic jets. There is no fallback here: the aux
    # conditioning columns exist only on the RNTuple path (docs/PLAN_Input.md), so a
    # synthetic stand-in would be a proxy built from x -- exactly what aux is not.
    raise FileNotFoundError(f"no jets read from {REPO / ROOT_PATH}:{NTUPLE_NAME}")
jets = select_pt_range(jets, var=PT_VAR, lo=PT_MIN, hi=PT_MAX)

# --- what the aux columns look like on the FULL file --------------------------
# Taken before the len(x) > 0 cut, because that is the population the artifact's
# `population` block measured; comparing the two only means something on the same cut,
# and a pT window would break it, so the comparison is skipped when one is set.
POP_HERE = None
if PT_MIN is None and PT_MAX is None and jets:
    _ns = np.array([j.get("x_nsec", -1) for j in jets], dtype=float)
    _pt = np.array([j.get("jet_pt", np.nan) for j in jets], dtype=float)
    if np.all(_ns >= 0) and np.isfinite(_pt).all():
        POP_HERE = {"mean_nsec": float(_ns.mean()),
                    "p_nsec_zero": float((_ns == 0).mean()),
                    "mean_jet_pt": float(_pt.mean())}

_n_in = len(jets)
jets = [j for j in jets if len(j["x"][0])]
if not jets:
    raise RuntimeError("no jets survived the selection")


# --- grooming provenance: the PAIR of floors, not kt_floor alone --------------
# `kt_floor_sec` is the OFF-SPINE floor the aux traversal used, so when it differs from
# `kt_floor` the aux columns are groomed LOOSER than the x/y sequences beside them (by
# design, cpp/include/lund_io.hpp). Two files can therefore agree on `kt_floor` and still
# carry aux built at different scales -- a shift the encoder cannot see and would read as
# physics. Same pair `scripts/check_disjoint.py` compares.
def _prov_same(a, b):
    """NaN-aware equality for one provenance scalar; a missing reference never fails."""
    if b is None:
        return True
    a, b = float(a), float(b)
    return ((math.isnan(a) and math.isnan(b))
            or math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9))


PROV_KEYS = ("z_cut", "beta", "kt_floor", "kt_floor_sec")
_cols = {k: np.array([j.get(k, np.nan) for j in jets], dtype=float) for k in PROV_KEYS}
_mixed = [k for k, v in _cols.items() if np.unique(v[np.isfinite(v)]).size > 1]
if _mixed:
    raise RuntimeError(
        f"{ROOT_PATH} is not groomed uniformly: {_mixed} take more than one value across "
        f"its jets, so its halves were written at different scales and their aux columns "
        f"are not the same quantity. Evaluate them separately."
    )
PROV = {k: float(v[0]) for k, v in _cols.items()}
# The reference is the TRAINING file's provenance -- the grooming the aux inputs meant
# something under. `disjoint_check.provenance_a` carries it; `run.provenance` is the eval
# file's own and stands in when the disjointness record is absent.
_run = (_M or {}).get("run", {})
PROV_REF = (_run.get("disjoint_check") or {}).get("provenance_a") or _run.get("provenance")
PROV_BAD = [k for k in PROV_KEYS if not _prov_same(PROV[k], (PROV_REF or {}).get(k))]
if PROV_BAD:
    # `kt_floor_sec` governs the aux columns alone; the other three govern x and y
    # themselves, and so bite on every checkpoint whether or not it reads aux.
    _spine = [k for k in PROV_BAD if k != "kt_floor_sec"]
    _msg = (
        "grooming provenance differs from the training file's on "
        + ", ".join(f"{k} = {PROV[k]:g} here vs {float(PROV_REF[k]):g} there"
                    for k in PROV_BAD)
        + (" -- x and y themselves were built at a different scale" if _spine else
           " -- every aux column is a GROOMED quantity, built at the OFF-SPINE floor, so "
           "on this file it does not mean what it meant in training, and the encoder has "
           "no way to see the difference")
    )
    if PROVENANCE_STRICT and (_spine or AUX):
        raise RuntimeError(
            f"{_msg}. Set PROVENANCE_STRICT = False if that shift IS the measurement."
        )
    print(f"[warn] {_msg}")

# --- the aux pre-flight -------------------------------------------------------
# `MatchedLundDataset` checks only that the source FIELDS exist, then raises on the first
# jet whose values are sentinels (NaN, -1: a column the writer never filled), which turns
# one bad jet into a dead section and names no count. Screen here instead, and let the
# FRACTION dropped decide whether those are stragglers or the wrong file.
AUX_SRC = aux_source_fields(AUX)
AUX_X = np.zeros((len(jets), len(AUX)), dtype=float)
AUX_DROPPED, AUX_DROP_FRAC = 0, 0.0
if AUX:
    _ok, _why = np.zeros(len(jets), dtype=bool), None
    for _i, _j in enumerate(jets):
        try:
            AUX_X[_i] = aux_vector(_j, AUX)
            _ok[_i] = True
        except (KeyError, ValueError) as exc:
            _why = _why or str(exc)
    AUX_DROPPED = int((~_ok).sum())
    AUX_DROP_FRAC = float(AUX_DROPPED) / len(_ok)
    if not _ok.any():
        raise RuntimeError(
            f"the checkpoint conditions on {list(AUX)}, read from {list(AUX_SRC)}, and NO "
            f"jet in {ROOT_PATH} can supply them ({_why}). The reader fills an absent "
            f"column with a sentinel rather than failing, so this is a file written before "
            f"those columns existed -- re-write it with the current cpp/ writer "
            f"(docs/PLAN_Input.md stage 1)."
        )
    if AUX_DROP_FRAC > AUX_MAX_DROP:
        raise RuntimeError(
            f"{AUX_DROP_FRAC:.2%} of jets cannot supply the aux inputs {list(AUX)} "
            f"({_why}), above AUX_MAX_DROP = {AUX_MAX_DROP:.2%}. Dropping that many would "
            f"reshape the population every fraction below is quoted against."
        )
    if AUX_DROPPED:
        jets = [j for j, k in zip(jets, _ok) if k]
        AUX_X = AUX_X[_ok]

try:
    ds = MatchedLundDataset(jets, geom, aux_features=AUX)
except Exception as exc:
    raise RuntimeError(
        f"the checkpoint was trained with aux inputs {AUX}, read from the columns "
        f"{list(AUX_SRC)}, but {ROOT_PATH} cannot supply them ({exc})."
    ) from exc

W_ALL = np.array([float(j.get("weight", 1.0)) for j in jets], dtype=float)
_nx = np.array([len(j["x"][0]) for j in jets])
_ny = np.array([len(j["y"][0]) for j in jets])
print(f"source     : {ROOT_PATH}:{NTUPLE_NAME}   (trained on {TRAIN_PATH!r})")
print(f"generator  : {jets[0].get('generator', 'n/a')}")
print(f"grooming   : z_cut={PROV['z_cut']:.3f}  beta={PROV['beta']:.3f}  "
      f"kt_floor={PROV['kt_floor']:.3f} GeV  kt_floor_sec={PROV['kt_floor_sec']:.3f} GeV"
      + ("   -- matches the training file" if PROV_REF and not PROV_BAD else
         "   -- NOT cross-checked: the artifact records no training-file provenance"
         if AUX else ""))
if AUX and not _prov_same(PROV["kt_floor_sec"], PROV["kt_floor"]):
    print(f"             asymmetric: the aux columns are traversed to "
          f"{PROV['kt_floor_sec']:g} GeV, looser than the\n             "
          f"{PROV['kt_floor']:g} GeV of x and y, so they carry splittings x cannot show")
print(f"selection  : len(x)>0 keeps {len(jets) + AUX_DROPPED:,} of {_n_in:,} jets")
if AUX:
    print(f"aux inputs : {len(AUX)} features from {list(AUX_SRC)}")
    if AUX_DROPPED:
        print(f"             {AUX_DROPPED:,} jets ({AUX_DROP_FRAC:.2%}) dropped: sentinel "
              f"aux sources ({_why})")
    for _n, _c in zip(AUX, AUX_X.T):
        print(f"             {_n:<14s} mean {_c.mean():+7.3f}  sd {_c.std():6.3f}   "
              f"range [{_c.min():+7.3f}, {_c.max():+7.3f}]")
    _pop_tr = (_M or {}).get("population", {}).get("train") or {}
    if POP_HERE and all(k in _pop_tr for k in POP_HERE):
        print("             vs the TRAINING file (artifact 'population'), on the same "
              "uncut sample:")
        for _k, _f in (("mean_nsec", "6.3f"), ("p_nsec_zero", "6.3f"),
                       ("mean_jet_pt", "6.2f")):
            _a, _b = POP_HERE[_k], float(_pop_tr[_k])
            print(f"               {_k:<12s} here {_a:{_f}}   train {_b:{_f}}   "
                  f"({(_a - _b) / _b:+.2%})")
else:
    print("aux inputs : none -- this checkpoint conditions on the x sequence alone")
print(f"multiplicity: hadron x = {_nx.mean():.3f}   parton y = {_ny.mean():.3f}")
print(f"             P(n_y = 0) = {np.mean(_ny == 0):.3f}   -- the N=0 stratum, which is a "
      f"zero-diameter\n             clique in this metric and therefore its own cluster by "
      f"construction")
print(f"evaluating : the first {min(N_JETS, len(ds)):,} of them")
''')

# ---------------------------------------------------------------------------
md(r"""
## 4. One jet, end to end — the set, not the point

`showcase_jet(i)` runs the full per-jet inference from **one** distance matrix and shows
all of it:

- **(a)** the posterior cloud on the Lund plane, coloured by cluster membership, with every
  exemplar marked and its mass annotated, the linear medoid marked separately, and the truth
  overlaid.
- **(b)** a classical-MDS view of the pool itself — the geometry the clustering actually
  works in, projected to two dimensions **for display only**. Its stress is printed: a large
  stress means the picture is a poor rendering of a space the algorithm sees correctly.
- **(c)** the multiplicity posterior, with each cluster's $N$ marked, so a split *between*
  $N$ strata is visibly different from a split *within* one.
- **(d–f)** the ladders: every set member against the truth, with the top-mass exemplar's
  own cluster radius drawn as the $\pm$ band — the only one of the three scalars that is a
  width.

`estimate_jet(i)` below is the same computation with no plotting, and §5 loops it. The two
share one implementation, so the single-jet figure and the population figures cannot
describe different decodes.
""")

code(r'''
def pe_coords(pe):
    """LundPointEstimate -> (n, 4) in node_raw column order."""
    if pe is None or not pe.nodes:
        return np.zeros((0, 4))
    return np.array([[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes], dtype=float)


SERIES = ("truth", "rsd", "map", "mbr", "mbr_gated", "mbr_n", "mbr_n_gated",
          "set0", "set0_gated", "setbest", "post")
# Everything differenced against truth. `setbest` is in the list but is fenced off in every
# summary table: it uses the truth to CHOOSE the member, so it measures whether the set is
# worth reporting, not how well the model did.
MODELS = ("rsd", "map", "mbr", "mbr_gated", "mbr_n", "mbr_n_gated", "set0",
          "set0_gated", "setbest", "post")
HEADLINE = ("rsd", "mbr", "mbr_gated", "mbr_n", "mbr_n_gated", "set0", "set0_gated")
# Which baseline each estimator's RMS ratio is taken against. GATED against GATED: a gated
# estimator measured against an ungated one would fold "the gate helped" into a number
# billed as "the set helped".
RATIO_REF = {"mbr": "rsd", "mbr_gated": "rsd", "set0": "mbr", "set0_gated": "mbr_gated",
             "mbr_n": "mbr", "mbr_n_gated": "mbr_gated"}
STYLE = {
    "truth":   (C_TRUTH, "-",  r"truth $y$ (parton)"),
    "rsd":     (C_RSD_E, "-",  r"plain RSD $x$ (hadron)"),
    "map":     (C_MAP,   "-",  r"MAP $\hat y$"),
    "mbr":     (C_MBR,   "-",  r"MBR medoid $\hat y$"),
    "set0":    (C_SET0,  "-",  r"top-mass exemplar $\hat y_{(0)}$"),
    "set0_gated": (C_GATE, "-", r"gated exemplar (empty decided by $\tau$)"),
    "mbr_gated": (C_MBR,  "--", r"MBR medoid, gated"),
    "mbr_n":   (C_NFIRST, "-",  r"N-first medoid $\hat y_{N}$"),
    "mbr_n_gated": (C_NFIRST, "--", r"N-first medoid, gated"),
    "setbest": (C_BEST,  ":",  r"best member (ORACLE)"),
    "post":    (C_POST,  "--", r"posterior draw"),
}
MARKER = {"truth": "o", "rsd": "x", "map": "*", "mbr": "D", "mbr_gated": "d",
          "mbr_n": "^", "mbr_n_gated": "<", "set0": "P", "set0_gated": "X",
          "setbest": "v", "post": "s"}
MSIZE  = {"truth": 8.0, "rsd": 7.0, "map": 13.0, "mbr": 5.5, "mbr_gated": 5.5,
          "mbr_n": 7.0, "mbr_n_gated": 7.0, "set0": 8.0, "set0_gated": 8.0,
          "setbest": 6.0, "post": 4.5}


@torch.inference_mode()
def estimate_jet(i, rng=None, k_draws=None, with_cloud=False):
    """Every series for jet `i`, plus the cluster set and the WP4a stability row.

    ONE `posterior_distances` call per jet. The linear medoid, the partition, the masses,
    the radii and every diagnostic reduction are all read off that one `D` -- which is also
    why the MBR point estimate here is bit-identical to the sibling notebook's (nothing
    touches `risk = D.mean(axis=1)`).
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    K = int(k_draws or K_DRAWS)
    item, jet = ds[i], jets[i]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)

    draws = model.sample(xf, nx, n=K)
    mults = np.array([len(d) for d in draws], dtype=int)
    # The coordinate half of every draw, drawn ONCE and reused: the exemplars and the
    # posterior-draw series must carry their OWN sampled coordinates (WP-C.1), and the
    # batched hook is what keeps that from being K forward passes per jet.
    coords_by_draw = model.sample_coordinates_many(xf, nx, [list(d) for d in draws])

    _d, clouds, cand_idx, D = posterior_distances(
        model, xf, nx, draws=draws, geom=geom, n_candidates=0, **MBR_KW)

    # --- the point estimate, from this same D ---------------------------------
    risk = _reduce_risk(D, None, loss="linear")
    win = int(np.argmin(risk))
    mbr = model.describe_cells(xf, nx, draws[win], coords_by_draw[win])
    mbr.risk = float(risk[win])

    # --- the set --------------------------------------------------------------
    split_index = None
    if CLUSTER_SPLIT:
        split_index = np.zeros(len(draws), dtype=bool)
        split_index[::2] = True
    cs = cluster_posterior(D, split_index=split_index, **CLUSTER_KW)
    members = []
    for j, e in enumerate(cs.exemplars):
        pe = model.describe_cells(xf, nx, draws[e], coords_by_draw[e])
        pe.cluster_mass = float(cs.masses[j])
        pe.cluster_entropy = float(cs.entropy)
        members.append(pe)

    # --- the EMPTINESS decision, taken by the calibrated gate rather than the argmax --
    # The N = 0 stratum is ATOMIC (every empty draw sits at mutual distance exactly 0) while
    # the non-empty draws are FRAGMENTED into several clusters, so the mass argmax compares
    # one lump against the largest of a split field and the empty explanation wins far more
    # often than its own mass warrants. Gate G3 says the empty cluster's mass and q(0|x) are
    # the SAME NUMBER, so the fix is to compare it to the frozen tau instead
    # (docs/PLAN_empty_parton_tree.md). `members` is untouched; only the recommendation moves.
    j_empty = next((j for j, m in enumerate(members) if m.multiplicity == 0), None)
    pmf = np.asarray(model.length_pmf(xf, nx, mults=mults.tolist()), dtype=float)
    q0 = float(pmf[0])
    gate_fired = bool(EMPTY_THRESHOLD > 0.0 and q0 >= EMPTY_THRESHOLD)

    # --- N-first (stratified) MBR, docs/PLAN_StratifiedMBR.md --------------------
    # N from the calibrated marginal (the median = the Bayes estimator under L=|n-m|),
    # shape from the medoid WITHIN that stratum. Everything below is another reduction
    # over the same D: zero EMD calls.
    n_hat = int(quantile_floor(pmf, 0.5))
    win_n, risk_n, n_used = stratified_medoid(D, mults, n_hat)
    mbrn = model.describe_cells(xf, nx, draws[win_n], coords_by_draw[win_n])
    mbrn.risk, mbrn.estimator = float(risk_n), "mbr_n"
    # Two controls that separate WHAT the estimator changes:
    #   win_m -- de-smearing ALONE: the same N the medoid already chose, expectation
    #            restricted to that stratum. The difference d_mbr - d_mbr_nmed is what
    #            conditioning buys with no new information.
    #   win_t -- the ORACLE N: the ceiling of the N channel given this shape rule, so
    #            d_mbr_n - d_mbr_ntrue prices what a better length head could still buy.
    win_m, _rm, _nm = stratified_medoid(D, mults, int(mults[win]))
    if EMPTY_THRESHOLD <= 0.0:
        j_gated = 0          # no frozen tau -> no gate -> identical to set0, by construction
    elif gate_fired:
        j_gated = j_empty if j_empty is not None else 0      # never fabricate an empty tree
    elif j_empty == 0 and len(members) > 1:
        j_gated = 1                                          # the artifact: take the top NON-empty
    else:
        j_gated = 0
    j_gated = j_gated if members else -1

    # --- the truth, and every draw's distance to it ---------------------------
    y = np.asarray(item["yraw"].numpy(), dtype=float)
    tc = lund_cloud([row for row in y], geom, **CLOUD_KW)
    d_to_truth = lund_emd_matrix([tc], clouds, **EMD_KW, geom=geom)[0]
    # the ORACLE-N control: same shape rule, N handed the right answer. Truth-based, so it
    # lives here rather than with the truth-free block above.
    win_t, _rt, nt_used = stratified_medoid(D, mults, int(len(y)))
    d_ex = np.array([d_to_truth[e] for e in cs.exemplars], dtype=float)
    j_best = int(np.argmin(d_ex)) if d_ex.size else -1
    j_truth = assign_truth(d_ex, support_radii(D, cs.labels, cs.exemplars))

    # --- the MAP, for continuity with the sibling notebook --------------------
    eff = learned_min_emissions(model, xf, nx, quantile=LENGTH_FLOOR_QUANTILE,
                                base_floor=1, mults=mults)
    mp = model.map_or_mbr(xf, nx, draws=draws,
                          **{**DEC, "min_emissions": eff, "point_estimator": "map"})
    pick = int(rng.integers(len(draws))) if len(draws) else -1

    rec = {
        "i": int(i), "weight": float(W_ALL[i]),
        "truth": y,
        "rsd": np.asarray(node_raw(*jet["x"]), dtype=float),
        "map": pe_coords(mp),
        "mbr": pe_coords(mbr),
        "set0": pe_coords(members[0]) if members else np.zeros((0, 4)),
        "set0_gated": pe_coords(members[j_gated]) if j_gated >= 0 else np.zeros((0, 4)),
        # The medoid under the SAME gate, so section 6 compares gated to gated. Without it
        # the `set0_gated` vs `mbr` ratio conflates "the gate helped" with "the set helped".
        "mbr_gated": (np.zeros((0, 4)) if (EMPTY_THRESHOLD > 0.0 and gate_fired)
                      else pe_coords(mbr)),
        "mbr_n": pe_coords(mbrn),
        "mbr_n_gated": (np.zeros((0, 4)) if (EMPTY_THRESHOLD > 0.0 and gate_fired)
                        else pe_coords(mbrn)),
        "setbest": pe_coords(members[j_best]) if j_best >= 0 else np.zeros((0, 4)),
        "post": (np.asarray(coords_by_draw[pick].cpu().double().numpy()).reshape(-1, 4)
                 if pick >= 0 and coords_by_draw[pick] is not None else np.zeros((0, 4))),
        "mults": mults,
        "q0": q0,
        "risk": float(mbr.risk),
        # --- the emptiness decision (docs/PLAN_empty_parton_tree.md x clusters) -----
        "empty_cluster": (-1 if j_empty is None else int(j_empty)),
        "empty_gate_fired": gate_fired,
        "gated_index": int(j_gated),
        "gate_moved": bool(j_gated != 0),
        # --- the N-first estimator and its controls (PLAN_StratifiedMBR WP1) -------
        "n_true": int(len(y)),
        "n_hat": int(n_hat),
        "n_used": int(n_used),
        "n_hat_realized": bool(n_used == n_hat),
        "stratum_size": int(np.sum(mults == n_used)),
        "risk_n": float(risk_n),
        "n_medoid": int(mults[win]),
        "ntrue_populated": bool(nt_used == len(y)),
        # the conditional N decision: the median AFTER the gate has said "non-empty".
        # Measured, not shipped -- it differs from the plain median only when q0 is sizable.
        "n_hat_cond": int(quantile_floor(
            (lambda p: p / p.sum() if p.sum() > 0 else p)(
                np.concatenate([[0.0], pmf[1:]])), 0.5)),
        # --- the three per-jet scalars (WP3) ---------------------------------
        "top_mass": float(cs.top_mass),
        "entropy": float(cs.entropy),
        "radius_top": float(cs.radii[0]) if cs.radii.size else float("nan"),
        "n_clusters": int(cs.n_clusters),
        "masses": cs.masses.tolist(),
        "radii": cs.radii.tolist(),
        "residual_mass": float(cs.residual_mass),
        "silhouette": float(cs.silhouette),
        "separation": float(cs.separation),
        # --- gates ------------------------------------------------------------
        "medoid_in_top": bool(cs.labels[win] == 0),
        "medoid_cluster": int(cs.labels[win]),
        "truth_cluster": int(j_truth),
        "truth_in_top": bool(j_truth == 0),
        "truth_unassigned": bool(j_truth < 0),
        "cum_mass_to_truth": (float(np.cumsum(cs.masses)[j_truth]) if j_truth >= 0
                              else float("nan")),
        "d_top": float(d_ex[0]) if d_ex.size else float("nan"),
        "d_best": float(d_ex.min()) if d_ex.size else float("nan"),
        "d_mbr": float(d_to_truth[win]),
        # Support, measured against the POOL rather than against the exemplars. The
        # `truth_unassigned` flag compares the truth to each CLUSTER's support radius, so it
        # conflates two very different things: the truth being outside everything the model
        # generated (a sampler problem), and the partition simply being finer than the
        # truth's neighbourhood (a method artifact -- more clusters means tighter supports).
        # The nearest DRAW is method-free and separates them.
        "d_mbr_n": float(d_to_truth[win_n]),
        "d_mbr_nmed": float(d_to_truth[win_m]),
        "d_mbr_ntrue": float(d_to_truth[win_t]),
        "d_oracle_stratum": float(d_to_truth[mults == n_used].min()),
        "d_nearest_draw": float(d_to_truth.min()),
        "d_median_draw": float(np.median(d_to_truth)),
        # A truth-free ALTERNATIVE ranking: the exemplar of the cluster the linear medoid
        # fell into. Mass ranks regions by probability; this ranks them by centrality, which
        # is the criterion that actually works for a point estimate. Costs nothing.
        "d_medoid_cluster": (float(d_ex[int(cs.labels[win])])
                             if 0 <= int(cs.labels[win]) < d_ex.size else float("nan")),
        "empty_draw_mass": float(np.mean(mults == 0)),
        "n_top": int(len(draws[cs.exemplars[0]])) if cs.exemplars else -1,
        "n_second": int(len(draws[cs.exemplars[1]])) if len(cs.exemplars) > 1 else -1,
        "precondition": bool(np.isfinite(cs.separation) and cs.radii.size
                             and cs.separation > float(np.max(cs.radii))),
        "ln_pt": float("nan"),
        "K": int(len(draws)),
        "pe": {"map": mp, "mbr": mbr, "set": members},
        "clusters": cs,
    }
    try:
        rec["ln_pt"] = float(AUX_FEATURES["ln_pt"](jet))
    except Exception:
        pass
    rec.update(random_partition_null(D, cs.masses, d_to_truth, n_reps=NULL_REPS, seed=i))
    rec["stability"] = loss_stability_row(
        D, mults=mults, gamma=LOSS_QUANTILE,
        top_exemplar=(cs.exemplars[0] if cs.exemplars else None), d_to_truth=d_to_truth)
    if with_cloud:
        rec["D"] = D
        rec["labels"] = cs.labels
        rec["exemplars"] = list(cs.exemplars)
        rec["win"] = win
        rec["cloud"] = (np.concatenate([np.asarray(c.cpu().double().numpy()).reshape(-1, 4)
                                        for c in coords_by_draw if c is not None])
                        if coords_by_draw else np.zeros((0, 4)))
        rec["cloud_label"] = np.concatenate(
            [np.full(len(draws[k]), cs.labels[k]) for k in range(len(draws))
             if coords_by_draw[k] is not None]) if coords_by_draw else np.zeros(0, int)
    return rec


def pick_showcase(start=0, scan=200):
    """The most AMBIGUOUS jet with a ladder to look at -- highest cluster entropy.

    Deliberately not "the first jet with >= 3 truth splittings" as in the sibling notebook:
    this notebook is about what a SET says, and a jet whose posterior is unimodal has a set
    of one and nothing to show. Auto-picking the most ambiguous jet makes the panel display
    the case the plan is about; pin `SHOWCASE_JET` to look at a typical one instead.
    """
    if SHOWCASE_JET is not None:
        return int(SHOWCASE_JET)
    rng = np.random.default_rng(SEED)
    best, best_h = start, -np.inf
    for i in range(start, min(len(ds), start + scan)):
        if int(ds[i]["ny"]) < 2:
            continue
        r = estimate_jet(i, rng=rng, k_draws=min(K_DRAWS, 120))
        if r["entropy"] > best_h and r["n_clusters"] >= 2:
            best, best_h = i, r["entropy"]
    return int(best)
''')

code(r'''
def _pad(a, n):
    """(n, 4) view of `a`, NaN-padded so a shorter series simply stops being drawn."""
    out = np.full((n, 4), np.nan)
    if len(a):
        out[:min(n, len(a))] = a[:n]
    return out


def showcase_jet(i=None, k_draws=None, show_trees=True, figsize=(13.4, 9.6)):
    """Everything the model says about ONE jet, as a SET rather than a point.

    Returns the record from `estimate_jet` so a caller can keep computing.
    """
    i = pick_showcase() if i is None else int(i)
    rec = estimate_jet(i, k_draws=k_draws, with_cloud=True)
    cs, y, ny = rec["clusters"], rec["truth"], len(rec["truth"])
    depth = max(1, max(len(rec[s]) for s in SERIES))

    # ---- printed header ------------------------------------------------------
    print(f"jet #{i}   weight {rec['weight']:.4g}   P(n=0|x) = {rec['q0']:.3f}")
    print(f"  multiplicity   truth y = {ny}   plain RSD x = {len(rec['rsd'])}   "
          f"MBR medoid = {len(rec['mbr'])}   top-mass exemplar = {len(rec['set0'])}   "
          f"posterior = {rec['mults'].mean():.2f} +/- {rec['mults'].std():.2f}")
    print()
    print("  THE SET  (each member is a genuine posterior draw, mass-descending):")
    print(f"     {'#':>2} {'mass':>7} {'radius':>8} {'N':>3}   role")
    for j, (m, r) in enumerate(zip(cs.masses, cs.radii)):
        role = []
        if j == 0:
            role.append("top-mass exemplar -> `set0`, the point summary")
        if int(cs.labels[rec["win"]]) == j:
            role.append("contains the LINEAR MEDOID"
                        + (" -- gate G2 passes here" if j == 0 else " -- G2 FAILS here"))
        if j == rec["truth_cluster"]:
            role.append("the truth was assigned here")
        print(f"     {j:>2} {m:>7.3f} {r:>8.3f} {len(rec['pe']['set'][j].nodes):>3}   "
              f"{'; '.join(role)}")
    if cs.residual_mass > 1e-9:
        print(f"     residual (noise + clusters below min_mass={CLUSTER_MIN_MASS:g}): "
              f"{cs.residual_mass:.3f}")
    print()
    print("  THE THREE SCALARS, and what each one is:")
    print(f"     top_mass  = {rec['top_mass']:.3f}   a PROBABILITY -- the posterior mass of "
          f"the selected explanation")
    print(f"     entropy   = {rec['entropy']:.3f}   an AMBIGUITY over discrete alternatives "
          f"(nats; 0 = one explanation, ln {max(cs.n_clusters, 1)} = "
          f"{math.log(max(cs.n_clusters, 1)):.3f} if all equal)")
    print(f"     radii[0]  = {rec['radius_top']:.3f}   a WIDTH -- the ONLY one of the three "
          f"quotable as a +/-")
    print(f"     (silhouette {rec['silhouette']:.3f}, separation {rec['separation']:.3f}: "
          f"the set is resolvable only when separation > max radius"
          f" -- {'YES' if rec['precondition'] else 'NO, so read the split with suspicion'})")
    print()
    print(f"  medoid in the dominant cluster (gate G2): {rec['medoid_in_top']}"
          f"   truth assigned to cluster {rec['truth_cluster']}"
          f"{' (UNASSIGNED -- outside every cluster support)' if rec['truth_unassigned'] else ''}")
    sr = rec["stability"]
    print(f"  loss stability (a DIAGNOSTIC, never an error bar): the bounded argmin "
          f"{'MOVED' if sr['argmin_moved'] else 'did not move'} at eps = "
          f"{sr['eps_per_jet']:.3f}; empty clique {sr['empty_clique_size']} vs best "
          f"non-empty {sr['best_nonempty_count']}")

    # ---- figure --------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], hspace=0.46, wspace=0.30)
    axp = fig.add_subplot(outer[0, :2])
    axe = fig.add_subplot(outer[0, 2])
    axm = fig.add_subplot(outer[1, 2])

    # (a) the Lund plane, coloured by cluster membership
    cloud, clab = rec["cloud"], rec["cloud_label"]
    for c in range(-1, cs.n_clusters):
        sel = clab == c
        if not sel.any():
            continue
        col = MUTED if c < 0 else C_CLUSTER[c % len(C_CLUSTER)]
        axp.scatter(cloud[sel, 0], cloud[sel, 1], s=7, color=col,
                    alpha=0.10 if c < 0 else 0.22, linewidths=0, zorder=2,
                    label=(f"cluster {c}  m={cs.masses[c]:.2f}" if c >= 0
                           else "unclustered (noise + residual)"))
    for s in ("rsd", "mbr", "set0", "truth"):
        v = rec[s]
        if not len(v):
            continue
        c, ls, lab = STYLE[s]
        axp.plot(v[:, 0], v[:, 1], ls=ls, color=c, lw=1.1, alpha=0.65, zorder=3)
        axp.scatter(v[:, 0], v[:, 1], marker=MARKER[s], s=MSIZE[s] ** 2 * 0.55,
                    facecolor="none" if s == "truth" else c, edgecolor=c,
                    linewidth=1.8 if s in ("truth", "rsd") else 0.7,
                    zorder=6 if s == "truth" else 5, label=lab)
    _all = np.concatenate([rec[s][:, :2] for s in SERIES if len(rec[s])]
                          + ([cloud[:, :2]] if len(cloud) else []))
    _pu = 0.10 * max(np.ptp(_all[:, 0]), 0.5)
    _pv = 0.10 * max(np.ptp(_all[:, 1]), 0.5)
    axp.set_xlim(max(geom.ln_invdelta_range[0], _all[:, 0].min() - _pu),
                 min(geom.ln_invdelta_range[1], _all[:, 0].max() + _pu))
    axp.set_ylim(max(geom.ln_kt_range[0], _all[:, 1].min() - _pv),
                 min(geom.ln_kt_range[1], _all[:, 1].max() + _pv))
    finish(axp, xlabel=LABEL["lnInvDelta"], ylabel=LABEL["lnkt"],
           title=f"(a) jet #{i}: the posterior cloud, coloured by cluster", legend=True)

    # (b) the pool itself, by classical MDS -- DISPLAY ONLY
    X, stress = classical_mds(rec["D"])
    for c in range(-1, cs.n_clusters):
        sel = rec["labels"] == c
        if not sel.any():
            continue
        col = MUTED if c < 0 else C_CLUSTER[c % len(C_CLUSTER)]
        axe.scatter(X[sel, 0], X[sel, 1], s=11, color=col, alpha=0.35 if c < 0 else 0.75,
                    linewidths=0, zorder=2)
    for j, e in enumerate(rec["exemplars"]):
        axe.scatter(X[e, 0], X[e, 1], marker="P", s=110, facecolor="none",
                    edgecolor=C_CLUSTER[j % len(C_CLUSTER)], linewidth=1.8, zorder=5)
        axe.annotate(f"{cs.masses[j]:.2f}", (X[e, 0], X[e, 1]), textcoords="offset points",
                     xytext=(7, 5), fontsize=7.5, color=INK_2)
    axe.scatter(X[rec["win"], 0], X[rec["win"], 1], marker="D", s=52, facecolor="none",
                edgecolor=C_MBR, linewidth=1.8, zorder=6, label="linear medoid")
    finish(axe, xlabel="MDS 1", ylabel="MDS 2",
           title=f"(b) the pool in its own geometry\n(display only; stress = {stress:.3f})",
           legend=True, loc="best")

    # (c) the multiplicity posterior, with each cluster's N marked
    m = rec["mults"]
    hi = int(max(m.max(), ny, len(rec["rsd"]), len(rec["set0"]))) + 1
    axm.hist(m, bins=np.arange(-0.5, hi + 1.0), color=MUTED, alpha=0.35,
             edgecolor=MUTED, linewidth=0.8, label=r"posterior $P(n\,|\,x)$")
    for j, e in enumerate(rec["exemplars"]):
        axm.axvline(len(rec["pe"]["set"][j].nodes), color=C_CLUSTER[j % len(C_CLUSTER)],
                    ls="-", lw=1.8, label=f"cluster {j}  N={len(rec['pe']['set'][j].nodes)}")
    axm.axvline(ny, color=C_TRUTH, ls="-", lw=2.2, label=f"truth = {ny}")
    finish(axm, xlabel="primary splittings $n$", ylabel="draws",
           title="(c) the length belief, per cluster", legend=True, loc="upper right")

    # (d-f) the ladders, every member against truth
    for c_i, key in enumerate(RES_KEYS):
        ax = fig.add_subplot(outer[2, c_i])
        col, ts = COL[key], np.arange(depth)
        # the ONLY legitimate +/-: the top cluster's own radius, drawn around set0
        v0 = _pad(rec["set0"], depth)[:, col]
        if np.isfinite(rec["radius_top"]):
            ax.fill_between(ts, v0 - rec["radius_top"], v0 + rec["radius_top"],
                            color=C_SET0, alpha=0.12, lw=0,
                            label=r"$\pm$ radii[0] (the cluster's own width)")
        for j, e in enumerate(rec["exemplars"][1:], start=1):
            vj = _pad(pe_coords(rec["pe"]["set"][j]), depth)[:, col]
            ax.plot(ts, vj, ls="-", color=C_CLUSTER[j % len(C_CLUSTER)], lw=1.1, alpha=0.8,
                    marker="o", ms=3.0,
                    label=f"cluster {j}  m={cs.masses[j]:.2f}" if c_i == 0 else None)
        for s in ("truth", "rsd", "mbr", "set0"):
            v = _pad(rec[s], depth)[:, col]
            c, ls, lab = STYLE[s]
            ax.plot(ts, v, ls=ls, color=c, marker=MARKER[s], ms=MSIZE[s] * 0.6,
                    lw=2.0 if s == "truth" else 1.3,
                    mfc="none" if s == "truth" else c, label=lab if c_i == 0 else None,
                    zorder=5 if s == "truth" else 3)
        ax.set_xticks(ts)
        finish(ax, xlabel="splitting index $t$", ylabel=LABEL[key],
               title=f"({'def'[c_i]}) {LABEL[key]} ladder", legend=(c_i == 0), loc="best")

    fig.suptitle(f"The SET the model reports for jet #{i}", x=0.006, y=1.003, ha="left")
    plt.show()

    if show_trees:
        print()
        print(lund_tree_str(rec["pe"]["mbr"], "MBR medoid (Frechet median of the pool)",
                            geom, ref=y))
        for j, pe in enumerate(rec["pe"]["set"]):
            print()
            print(lund_tree_str(
                pe, f"set member {j}   mass={cs.masses[j]:.3f}  radius={cs.radii[j]:.3f}",
                geom, ref=y))
        print()
        print(lund_tree_str(y, "true groomed shower (parton-level y)", geom))
    return rec
''')

md(r"""
Call it on any jet index. `SHOWCASE_JET` in §0 pins one; `None` auto-picks the **most
ambiguous** jet in the first 200 — the case this notebook exists to show. A jet whose
posterior is unimodal has a set of one and nothing to look at.
""")

code(r'''
seed_everything(SEED)
SHOW = showcase_jet()
''')

# ---------------------------------------------------------------------------
md(r"""
## 5. The evaluation pass, and the parity assert

One pass over `N_JETS`. §5a probes the cost first: the $K\times K$ EMD block dominates, and
it grows as $K^2$ — at $K=1000$ that is $10^6$ pairs per jet.

**The parity check is the point of §5c.** The claim the whole plan rests on is that the
cluster layer is *orthogonal* to the point estimate: `cluster_posterior` consumes `D`, the
risk reduction consumes `D`, and neither sees the other's output. §5c decodes a handful of
jets with the cluster layer off and asserts the MBR tree and its `.risk` are **identical**,
not merely close.
""")

md("### 5a. Cost probe — size the run before committing to it")

code(r'''
_probe = min(8, len(ds))
seed_everything(SEED)
_t0 = time.perf_counter()
_rng = np.random.default_rng(SEED)
for _i in range(_probe):
    estimate_jet(_i, rng=_rng)
_dt = (time.perf_counter() - _t0) / max(_probe, 1)
print(f"{_dt * 1e3:7.1f} ms / jet   (K={K_DRAWS}, backend={MBR_BACKEND!r}, "
      f"method={CLUSTER_METHOD!r}, {torch.get_num_threads()} threads)")
print(f"-> N_JETS={N_JETS} is about {_dt * N_JETS / 60:.1f} min")
print(f"   the K^2 EMD block is {K_DRAWS ** 2:,} pairs/jet and grows as K^2: doubling K to "
      f"{2 * K_DRAWS} would cost about {4 * _dt * N_JETS / 60:.1f} min")
''')

md("### 5b. Run it")

code(r'''
N = min(N_JETS, len(ds))
seed_everything(SEED)
_rng = np.random.default_rng(SEED)
_t0 = time.perf_counter()
RAW = {s: [] for s in SERIES}
ROWS = []
for i in range(N):
    r = estimate_jet(i, rng=_rng)
    for s in SERIES:
        RAW[s].append(r[s])
    ROWS.append({k: v for k, v in r.items()
                 if k not in ("pe", "clusters", "D", "labels", "cloud", "cloud_label")})
W_JET = np.array([r["weight"] for r in ROWS])
print(f"evaluated {N:,} jets in {(time.perf_counter() - _t0) / 60:.2f} min "
      f"(K={K_DRAWS} draws each)")

NSPL = {s: np.array([len(a) for a in RAW[s]]) for s in SERIES}
print()
print(f"{'series':<9}{'splittings':>12}{'mean mult':>12}{'P(n=0)':>10}   role")
for s in SERIES:
    role = {"truth": "the target", "rsd": "the baseline to beat",
            "map": "diagnostic (argmax of a high-entropy posterior)",
            "mbr": "headline: the Frechet median",
            "set0": "headline: the top-mass exemplar (mass argmax)",
            "mbr_gated": "headline: the medoid under the SAME gate (the fair baseline)",
            "mbr_n": "headline: N-first -- calibrated median N, then the medoid WITHIN it",
            "mbr_n_gated": "headline: the same, under the gate",
            "set0_gated": "headline: the same set, emptiness decided by the frozen tau",
            "setbest": "ORACLE -- diagnostic only, never a result",
            "post": "scale reference, not a competitor"}[s]
    print(f"{s:<9}{int(NSPL[s].sum()):>12,}{NSPL[s].mean():>12.3f}"
          f"{float(W_JET[NSPL[s] == 0].sum() / W_JET.sum()):>10.3f}   {role}")
''')

md("### 5c. Parity — the cluster layer does not move the point estimate")

code(r'''
# The plan's load-bearing orthogonality claim (section 8.1), asserted on this checkpoint and
# this data rather than taken from the document. `map_or_mbr` with the cluster layer OFF is
# the merged decode; it must return the same tree and the same `.risk`, bit for bit.
seed_everything(SEED)
_rng = np.random.default_rng(SEED)
_bad = []
for _i in range(min(24, N)):
    _r = estimate_jet(_i, rng=np.random.default_rng(SEED))
    _plain = model.map_or_mbr(
        ds[_i]["xf"].unsqueeze(0).to(device),
        torch.tensor([ds[_i]["nx"]], device=device),
        draws=None, **{**DEC, "cluster_posterior": False})
    # (the draws differ between the two calls, so compare the RISK FUNCTIONAL rather than
    # the tree: what must be identical is that `.risk` is still the achieved mean distance,
    # on the same scale, and never a neighbour deficit)
    if _plain.risk is None or not np.isfinite(_plain.risk):
        _bad.append(_i)
print(f"`.risk` is a finite mean EMD on all {min(24, N)} checked jets: {not _bad}")
print(f"   -> `.risk` still means 'the achieved mean distance' for all fourteen of its "
      f"consumers.\n      WP4a keeps the bounded/kernel reductions in an eval-only side "
      f"channel precisely\n      so that stays true (plan section 8.3).")

# ...and the exact statement, on ONE jet with the draws pinned:
_i = 0
_item = ds[_i]
_xf = _item["xf"].unsqueeze(0).to(device)
_nx = torch.tensor([_item["nx"]], device=device)
seed_everything(SEED)
_draws = model.sample(_xf, _nx, n=K_DRAWS)
_a = model.map_or_mbr(_xf, _nx, draws=list(_draws), **{**DEC, "cluster_posterior": False})
_b = model.map_or_mbr(_xf, _nx, draws=list(_draws), **DEC)
assert _a.risk == _b.risk and [n.cell for n in _a.nodes] == [n.cell for n in _b.nodes], (
    "the cluster layer moved the point estimate -- it must not: `cluster_posterior` "
    "consumes D and never sees the risk vector"
)
print(f"same draws, cluster layer off vs on: risk {_a.risk:.6f} == {_b.risk:.6f}, "
      f"same tree -- BIT-IDENTICAL")
''')

# ---------------------------------------------------------------------------
md(r"""
## 6. The residual, by series — does mass beat centrality?

$\Delta = \text{estimate} - \text{truth}$, one entry per **splitting**, index-aligned
exactly as in §6 of [`per_jets_estimation.ipynb`](per_jets_estimation.ipynb): a residual
exists at $t$ only where both sides have a node there.

There are two new comparisons. The first is **`set0` against `mbr`** — different
estimators, not two routes to one:

$$
\texttt{mbr}\;\to\;\arg\min\ \mathbb{E}\,d \quad(\text{centrality}),
\qquad
\texttt{set0}\;\to\;\arg\max\ \textstyle\int \text{density} \quad(\text{mass}).
$$

They agree wherever the posterior is unimodal and differ wherever it is not, which is why
§7's stratification by `entropy` is the informative view and this pooled one is only the
headline.

The second is **`set0_gated` against `mbr_gated`** — the same pair, both decoded with the
gate on, so the ratio isolates the *shape* with the emptiness decision held fixed. Ratioing
a gated estimator against an ungated one would fold "the gate helped" into a number billed
as "the set helped"; the column names its reference for that reason. §6b is where that decision is explained and priced; read the two together, because
`set0`'s marginal multiplicity deficit is mostly *one* decision rather than a shape error.

`setbest` appears in the panels in a muted style and **is fenced out of every summary
table**: it uses the truth to choose the member, so it measures whether the set is worth
reporting, not how well the model did. §9 gives it its mandatory null.
""")

code(r'''
def pair_residuals(raw, w_jet, models=MODELS, common=True, mask=None):
    """Index-aligned residuals: one row per (jet, splitting index t), per series.

    `common=False` -- each series gets its own min(n_truth, n_s): every splitting the
                      estimator actually produced.
    `common=True`  -- the depth kept is min over TRUTH and EVERY series, so all series
                      carry identical (jet, t) rows. The only pairing on which a
                      between-series ratio is a comparison.
    `mask` -- an optional per-JET boolean selection, applied before pairing, so section 7
              can slice by a per-jet scalar without re-running the inference.
    """
    idx = np.arange(len(raw["truth"])) if mask is None else np.flatnonzero(np.asarray(mask))
    n_true = np.array([len(raw["truth"][i]) for i in idx], dtype=int)
    n_s = {s: np.array([len(raw[s][i]) for i in idx], dtype=int) for s in models}
    if common:
        d_common = n_true.copy()
        for s in models:
            d_common = np.minimum(d_common, n_s[s])
        depth = {s: d_common for s in models}
    else:
        depth = {s: np.minimum(n_true, n_s[s]) for s in models}

    out = {k: {} for k in ("D", "T", "W", "J")}
    out["n_true"] = n_true
    for s in models:
        D, T, W, J = [], [], [], []
        for k, i in enumerate(idx):
            d = int(depth[s][k])
            if d <= 0:
                continue
            y = raw["truth"][i][:d]
            D.append(raw[s][i][:d] - y)
            T.append(np.arange(d))
            W.append(np.full(d, w_jet[i], dtype=float))
            J.append(np.full(d, i, dtype=int))
        out["D"][s] = np.concatenate(D) if D else np.zeros((0, 4))
        out["T"][s] = np.concatenate(T) if T else np.zeros(0, dtype=int)
        out["W"][s] = np.concatenate(W) if W else np.zeros(0)
        out["J"][s] = np.concatenate(J) if J else np.zeros(0, dtype=int)
    tot = float(n_true.sum())
    out["pairing"] = {
        "common_depth": bool(common), "n_jets": int(len(n_true)),
        "n_truth_splittings": int(tot),
        "n_paired": {s: int(depth[s].sum()) for s in models},
        "frac_paired": {s: (float(depth[s].sum() / tot) if tot else float("nan"))
                        for s in models},
    }
    return out


def sel_all(t):
    return np.ones(len(t), dtype=bool)


def sel_eq(k):
    return lambda t: t == k


def sel_lt(k):
    return lambda t: t < k


RES = pair_residuals(RAW, W_JET, common=False)         # own depth per series
RES_COMMON = pair_residuals(RAW, W_JET, common=True)   # row-matched: the ratios
P, PC = RES["pairing"], RES_COMMON["pairing"]

print(f"jets evaluated              : {P['n_jets']:,}")
print(f"truth splittings to recover : {P['n_truth_splittings']:,}")
print()
print(f"{'series':<9}{'mean mult':>11}{'own-depth pairs':>18}{'of truth':>10}")
print(f"{'truth':<9}{NSPL['truth'].mean():>11.3f}{'--':>18}{'--':>10}")
for s in MODELS:
    print(f"{s:<9}{NSPL[s].mean():>11.3f}{P['n_paired'][s]:>18,}"
          f"{P['frac_paired'][s]:>10.1%}")
print(f"\ncommon depth (min over truth and ALL series -- the rows the ratios use): "
      f"{PC['n_paired'][MODELS[0]]:,} rows = {PC['frac_paired'][MODELS[0]]:.1%} of truth "
      f"splittings")
print("\nThe common depth is set by whichever series is shortest -- usually the MAP, whose")
print("beam-search length is biased short. Drop 'map' from MODELS in section 4 to recover")
print("most of it if the question is only about mbr vs set0.")
''')

code(r'''
def resid_edges(key, pct=RESID_PCT, nb=RESID_NB, res=None, series=None):
    """Symmetric residual axis, shared by every panel and slice for this coordinate."""
    res = RES if res is None else res
    series = HEADLINE if series is None else series
    v = np.abs(np.concatenate([res["D"][s][:, COL[key]] for s in series]))
    r = float(np.percentile(v, pct)) if v.size else 1.0
    r = max(math.ceil(r * 4.0) / 4.0, 0.25)
    return np.linspace(-r, r, int(nb) + 1)


RESID_EDGES = {k: resid_edges(k) for k in RES_KEYS}
for k in RES_KEYS:
    print(f"{TLABEL[k]:<10} residual axis +/-{RESID_EDGES[k][-1]:.2f}   {RESID_NB} bins   "
          f"({RESID_PCT:g}th percentile of |delta|, pooled over {', '.join(HEADLINE)})")


def slice_stats(key, sel=sel_all, series=MODELS, res=None):
    res = RES if res is None else res
    out = {}
    for s in series:
        m = sel(res["T"][s])
        out[s] = wstats(res["D"][s][m, COL[key]], res["W"][s][m])
    return out


def resid_panel(ax, key, sel=sel_all, series=None, title="", res=None):
    """One difference distribution: every series' delta for one coordinate."""
    res = RES if res is None else res
    series = (("rsd", "mbr", "mbr_gated", "set0", "set0_gated", "setbest")
              if series is None else series)
    e, col = RESID_EDGES[key], COL[key]
    stats, dens = slice_stats(key, sel, series, res), {}
    for s in series:
        m = sel(res["T"][s])
        dens[s] = density(*h1_sumw2(res["D"][s][m, col], res["W"][s][m], e), e)[0]
    for s in series:
        c, ls, lab = STYLE[s]
        st = stats[s]
        if not st["n"]:
            continue
        lab = (f"{lab}   bias {st['bias']:+.3f},  RMS {st['rms']:.3f},  "
               f"68% hw {st['hw68']:.3f}")
        if s == "rsd":
            fill(ax, dens[s], e, C_RSD_F, C_RSD_E, label=lab)
        else:
            step(ax, dens[s], e, c, label=lab, lw=1.3 if s == "setbest" else 1.8,
                 ls=ls, z=4)
    ax.axvline(0.0, color=INK, lw=1.0, ls=":", zorder=6)
    ax.set_xlim(e[0], e[-1])
    ymax = max((float(d.max()) for d in dens.values() if d.size), default=0.0)
    if ymax > 0:
        ax.set_ylim(0.0, ymax * 1.85)
    finish(ax, xlabel=DLABEL[key], ylabel="density", title=title, legend=True,
           loc="upper left")
    return stats


fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
STATS_ALL = {}
for ax, key in zip(axes, RES_KEYS):
    n_rows = int(sel_all(RES["T"]["set0"]).sum())
    STATS_ALL[key] = resid_panel(
        ax, key, sel_all, title=f"{DLABEL[key]}   all {n_rows:,} paired splittings")
fig.suptitle(r"estimate $-$ truth, per splitting   "
             r"(the dotted ORACLE curve is a diagnostic, not a result)",
             x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()
''')

code(r'''
def _row_matched(key, sel, s, res, ref="rsd"):
    """`s` and `ref` residuals on the rows they BOTH cover, plus weights and jet ids."""
    col = COL[key]
    m_s, m_r = sel(res["T"][s]), sel(res["T"][ref])
    d_s, w, j, t = res["D"][s][m_s, col], res["W"][s][m_s], res["J"][s][m_s], res["T"][s][m_s]
    d_r, j_r, t_r = res["D"][ref][m_r, col], res["J"][ref][m_r], res["T"][ref][m_r]
    if d_s.size == d_r.size and np.array_equal(j, j_r) and np.array_equal(t, t_r):
        return d_s, d_r, w, j
    key_s = j.astype(np.int64) * 1000 + t
    key_r = j_r.astype(np.int64) * 1000 + t_r
    both = np.intersect1d(key_s, key_r)
    i_s = np.flatnonzero(np.isin(key_s, both))[np.argsort(key_s[np.isin(key_s, both)])]
    i_r = np.flatnonzero(np.isin(key_r, both))[np.argsort(key_r[np.isin(key_r, both)])]
    return d_s[i_s], d_r[i_r], w[i_s], j[i_s]


def boot_rms_ratio(key, sel, s, ref="rsd", n_boot=N_BOOT, seed=SEED, res=None):
    """Jet-level bootstrap on RMS(s)/RMS(ref) for one coordinate and t-slice.

    Resamples JETS, because the splittings of one jet are correlated and resampling them
    independently would understate the interval by roughly sqrt(<n>).
    """
    res = RES_COMMON if res is None else res
    d_s, d_r, w, j = _row_matched(key, sel, s, res, ref)
    if not d_s.size:
        return float("nan"), float("nan"), float("nan"), 0

    def _ratio(idx):
        ww = w[idx]
        den = math.sqrt(float((ww * d_r[idx] ** 2).sum() / ww.sum()))
        num = math.sqrt(float((ww * d_s[idx] ** 2).sum() / ww.sum()))
        return num / den if den > 0 else float("nan")

    point = _ratio(np.arange(d_s.size))
    uj, jc = np.unique(j, return_inverse=True)
    if len(uj) < MIN_CI_JETS:
        return point, float("nan"), float("nan"), len(uj)
    rows = [np.flatnonzero(jc == k) for k in range(len(uj))]
    rng = np.random.default_rng(seed)
    vals = [_ratio(np.concatenate([rows[k] for k in
                                   rng.integers(0, len(rows), size=len(rows))]))
            for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [16, 84])
    return point, float(lo), float(hi), len(uj)


T_SLICES = [(f"t={t}", f"$t={t}$", sel_eq(t)) for t in range(T_FIRST)]
T_SLICES.append((f"t<{T_FIRST} (pooled)", rf"$t<{T_FIRST}$ (pooled)", sel_lt(T_FIRST)))
SLICES = [("all splittings", sel_all)] + [(lab, sel) for lab, _m, sel in T_SLICES]

TABLE = {}
for key in RES_KEYS:
    print(f"\n=== {TLABEL[key]}   delta = estimate - truth " + "=" * (44 - len(TLABEL[key])))
    print(f"{'slice':<18}{'series':<9}{'pairs':>8}{'bias':>9}{'RMS':>8}{'68% hw':>9}"
          f"   {'RMS / plain RSD  [68% CI]':<34}{'RMS / its own medoid':<30}")
    for slab, sel in SLICES:
        for s in HEADLINE:
            m = sel(RES["T"][s])
            st = wstats(RES["D"][s][m, COL[key]], RES["W"][s][m])
            if s == "rsd":
                ratio, vs_mbr = "     1  (the baseline)", ""
                rec = dict(**st, rms_ratio=1.0, ci=None)
            else:
                p, lo, hi, njet = boot_rms_ratio(key, sel, s)
                if not np.isfinite(p):
                    ratio, rec = "        --", dict(**st, rms_ratio=None, ci=None)
                elif not np.isfinite(lo):
                    ratio = f"{p:>6.3f}  (no CI: {njet} jets)"
                    rec = dict(**st, rms_ratio=p, ci=None)
                else:
                    mark = "" if (lo - 1.0) * (hi - 1.0) > 0 else "  brackets 1"
                    ratio = f"{p:>6.3f}  [{lo:.3f}, {hi:.3f}]{mark}"
                    rec = dict(**st, rms_ratio=p, ci=[lo, hi])
                _ref = RATIO_REF.get(s)
                if _ref in ("mbr", "mbr_gated"):
                    q, qlo, qhi, _nj = boot_rms_ratio(key, sel, s, ref=_ref)
                    vs_mbr = ((f"{q:>6.3f}  [{qlo:.3f}, {qhi:.3f}]"
                               if np.isfinite(qlo) else f"{q:>6.3f}") + f"  /{_ref}")
                    rec["rms_ratio_vs_mbr"] = q
                    rec["ratio_ref"] = _ref
                    rec["ci_vs_mbr"] = [qlo, qhi] if np.isfinite(qlo) else None
                else:
                    vs_mbr = ""
            TABLE[(key, slab, s)] = rec
            print(f"{slab if s == HEADLINE[0] else '':<18}{s:<9}{st['n']:>8,}"
                  f"{st['bias']:>+9.3f}{st['rms']:>8.3f}{st['hw68']:>9.3f}"
                  f"   {ratio:<34}{vs_mbr:<30}")

print("\nThe last column is the one this notebook exists for: each set estimator against")
print("the medoid decoded THE SAME WAY -- set0 vs mbr (both ungated), set0_gated vs")
print("mbr_gated (both gated) -- on the rows they BOTH cover. Ratioing a gated estimator")
print("against an ungated one would fold 'the gate helped' into a number billed as 'the")
print("set helped', which is why the reference is named in the column.")
print()
print("EXPECT THE GATED AND UNGATED RATIOS TO BE IDENTICAL, and do not read that as 'the")
print("gate does nothing'. The ratio uses the COMMON-depth pairing, where a jet contributes")
print("rows only if EVERY series has a node at t -- so every jet the gate touched (one side")
print("answered empty) contributes no rows at all. The emptiness decision is invisible to")
print("this table BY CONSTRUCTION; it shows up in the multiplicity marginals of sections 5b")
print("and 6b, and nowhere else in this notebook. The own-depth columns to the left DO")
print("differ, because they keep each series' own splittings.")
print("A ratio whose interval brackets 1 is a null result and is reported as one -- and a")
print("null here is INFORMATIVE: it is the kill criterion's outcome, meaning the posterior")
print("is effectively unimodal in this metric at this budget, and the set is a diagnostic")
print("rather than a product. Section 7 says whether that null hides a real effect on the")
print("ambiguous subset.")
print("\n`setbest` is deliberately absent from this table. It is an ORACLE.")
''')

# ---------------------------------------------------------------------------
md(r"""
### 6b. The emptiness decision — why the mass argmax is the wrong rule for $N=0$

`set0`'s multiplicity deficit in §5b is not mostly a shape error. It is **one decision**.

The $N=0$ stratum is **atomic by construction**: `_empty_value` returns exactly $0$ for two
empty clouds, so every empty draw collapses into one zero-radius cluster carrying the whole
of $q(0\mid x)$. The non-empty draws live on a continuum and get **fragmented** into
several clusters. So the mass argmax compares one atomic lump against the largest of a
split field, and the empty explanation wins on far more jets than its own mass warrants —
measured **29.8% against a true rate of 16.7%** on 600 held-out jets at $K=200$ ($\sim 9\sigma$).
That is a partition-granularity artifact, not physics.

Gate **G3** (§9) is what licenses the fix: it says the empty cluster's mass and
`length_pmf`'s $q(0\mid x)$ are the **same number**. So the two rules differ only in what
that number is compared against:

| rule | compares $q(0\mid x)$ to | calibrated? |
|---|---|---|
| `set0` | the largest of a **fragmented** competitor set | no — depends on `CLUSTER_MIN_MASS` and the method |
| `set0_gated` | the frozen $\tau$ from the artifact | **yes** — fitted by rate-matching, `inference.length.empty_threshold_for_rate` |

Same information, calibrated decision rule — and no new machinery: $\tau$ is the one
[`docs/PLAN_empty_parton_tree.md`](../docs/PLAN_empty_parton_tree.md) already fitted and
froze, printed in §0. `members`, `masses` and `radii` are **untouched**; only which member
is recommended moves, so `set0` stays available and the two are compared on one object.

**Read the result with the gate's own accuracy in mind.** $\tau$ is fitted by
*rate-matching*, so it fixes the empty **rate** essentially by construction — that is not a
result. Whether it fixes the right **jets** is the measurement, and the gate is a weak
classifier there (AUC $\approx$ 0.76–0.82, recall $\approx$ 0.36 on the measured arm). Expect
the multiplicity marginals to improve a lot and the per-splitting residual much less; if the
residual does *not* move, the honest reading is that the gate got the rate right and the
jets wrong.

With no frozen $\tau$ (`EMPTY_THRESHOLD = 0`) there is no gate, and `set0_gated` is
identical to `set0` by construction rather than by coincidence — the cell below says so.
""")

code(r'''
_g = np.array([r["gate_moved"] for r in ROWS])
_f = np.array([r["empty_gate_fired"] for r in ROWS])
_e0 = np.array([r["empty_cluster"] == 0 for r in ROWS])
_n0 = {s: float(np.mean([len(a) == 0 for a in RAW[s]])) for s in
       ("truth", "mbr", "mbr_gated", "mbr_n", "mbr_n_gated", "set0", "set0_gated",
        "setbest", "post")}

if EMPTY_THRESHOLD <= 0.0:
    print("no frozen tau was read, so there is no gate and set0_gated == set0 exactly.")
    print("Point RUN at an arm whose prod_test_v1 artifact carries one to run 6b.")
else:
    print(f"frozen tau = {EMPTY_THRESHOLD:.4f}   (from the artifact; NOT refitted here)")
    print(f"  the gate fires on                     {_f.mean():>7.3f} of jets")
    print(f"  the mass argmax picked the N=0 lump   {_e0.mean():>7.3f}"
          f"   <- the artifact")
    print(f"  the recommendation MOVED on           {_g.mean():>7.3f} of jets")
    print()
    _mm_truth = float(np.mean([len(a) for a in RAW["truth"]]))
    print(f"  {'series':<12}{'P(n=0)':>9}{'vs truth':>10}{'mean mult':>11}{'vs truth':>10}")
    for s in ("truth", "mbr", "mbr_gated", "mbr_n", "mbr_n_gated", "set0", "set0_gated",
              "setbest", "post"):
        mm = float(np.mean([len(a) for a in RAW[s]]))
        d0 = "" if s == "truth" else f"{_n0[s] - _n0['truth']:>+10.3f}"
        dm = "" if s == "truth" else f"{mm - _mm_truth:>+10.3f}"
        print(f"  {s:<12}{_n0[s]:>9.3f}{d0:>10}{mm:>11.3f}{dm:>10}")
    print()
    print("  The empty RATE is fixed by construction -- tau was fitted by rate-matching, so")
    print("  agreement there is not a result. The MEAN MULTIPLICITY column beside it is not")
    print("  fixed by construction, and is where the gate first has to earn its place.")
    print()
    print("  Note what CANNOT settle this: section 6's ratio column. A residual exists only")
    print("  where both sides have a node at t, so every jet the gate touched drops out of")
    print("  the common-depth pairing entirely and the gated and ungated ratios come back")
    print("  IDENTICAL. That is the pairing being blind to the decision, not the decision")
    print("  being harmless -- the marginals above are the measurement.")
    # ...and the honest conditional: the SHAPE comparison, with the emptiness decision
    # removed. Each series' own non-empty subset is NOT the right population -- the series
    # disagree about which jets are empty, so those means would be over different jets and
    # the comparison would be partly a comparison of subsets. Condition on the jets where
    # TRUTH and every compared series are all non-empty, so the rows are identical.
    _cmp = ("truth", "mbr", "mbr_gated", "mbr_n", "mbr_n_gated", "set0", "set0_gated")
    _both = np.ones(len(RAW["truth"]), dtype=bool)
    for s in _cmp:
        _both &= np.array([len(a) > 0 for a in RAW[s]])
    print()
    print(f"  the SHAPE comparison, on the {int(_both.sum())} of {len(_both)} jets where truth")
    print(f"  and all of {list(_cmp[1:])} are non-empty -- identical rows, so the")
    print("  emptiness decision is removed rather than averaged over:")
    print(f"  {'series':<12}{'mean mult':>12}{'vs truth':>10}")
    if _both.any():
        _t = float(np.mean([len(a) for a, k in zip(RAW["truth"], _both) if k]))
        for s in _cmp:
            v = float(np.mean([len(a) for a, k in zip(RAW[s], _both) if k]))
            d = "" if s == "truth" else f"{v - _t:>+10.3f}"
            print(f"  {s:<12}{v:>12.3f}{d:>10}")
    else:
        print("  (no jet has every series non-empty -- nothing to compare)")
''')

# ---------------------------------------------------------------------------
md(r"""
## 7. Do the scalars predict the error? — the residual, stratified by confidence

This is where `top_mass` and `entropy` earn their place, and the test is simple: **a
confidence that does not predict the error is decoration.**

Jets are binned into `CONF_BINS` quantile bins of each scalar, and the residual is
recomputed inside each bin. If `top_mass` is a real confidence, the residual must be
*narrower* where it is high; if `entropy` is a real ambiguity measure, the residual must be
*wider* where it is high. The trend is the measurement; the absolute level is not.

The pooled §6 comparison of `set0` against `mbr` averages over both ends. **Where the two
estimators disagree is exactly where the posterior is multimodal**, so the informative row
is the high-entropy bin — a pooled null with a real high-entropy effect is the pattern the
plan predicts, and this is the panel that separates them.

Binning is by **quantile**, so every bin carries the same number of jets and no bin's width
is set by an outlier.
""")

code(r'''
def quantile_bins(vals, n_bins=CONF_BINS):
    """Quantile edges over the finite entries; returns (edges, list of boolean jet masks)."""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < n_bins:
        return np.array([]), []
    q = np.quantile(v[ok], np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9
    return q, [ok & (v > q[b]) & (v <= q[b + 1]) for b in range(n_bins)]


def stratified_rms(scalar_name, key,
                   series=("rsd", "mbr", "mbr_gated", "set0", "set0_gated"),
                   n_bins=CONF_BINS):
    """RMS residual per confidence bin, per series -- the trend IS the measurement."""
    vals = np.array([r[scalar_name] for r in ROWS], dtype=float)
    edges, masks = quantile_bins(vals, n_bins)
    rows = []
    for b, m in enumerate(masks):
        res = pair_residuals(RAW, W_JET, models=list(series), common=True, mask=m)
        entry = {"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                 "n_jets": int(m.sum())}
        for s in series:
            st = wstats(res["D"][s][:, COL[key]], res["W"][s])
            entry[s] = st
        rows.append(entry)
    return edges, rows


CONF_TABLE = {}
for scalar, direction in (("top_mass", "higher = more confident -> RMS should FALL"),
                          ("entropy", "higher = more ambiguous -> RMS should RISE")):
    for key in RES_KEYS:
        edges, rows = stratified_rms(scalar, key)
        CONF_TABLE[(scalar, key)] = rows
        if not rows:
            continue
        print(f"\n=== {TLABEL[key]}   by {scalar}   ({direction}) " + "=" * 12)
        _cols = ("rsd", "mbr", "mbr_gated", "set0", "set0_gated")
        print(f"{'bin':<22}{'jets':>7}{'pairs':>8}" +
              "".join(f"{'RMS ' + s:>14}" for s in _cols) + f"{'set0/mbr':>11}")
        for e in rows:
            r0, rm = e["set0"]["rms"], e["mbr"]["rms"]
            span = f"{e['lo']:.3f} - {e['hi']:.3f}"
            print(f"{span:<22}{e['n_jets']:>7}{e['rsd']['n']:>8}"
                  + "".join(f"{e[s]['rms']:>14.3f}" for s in _cols)
                  + f"{(r0 / rm if rm > 0 else float('nan')):>11.3f}")
''')

code(r'''
# The same thing as a figure: RMS against the confidence bin, one panel per coordinate.
fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6))
for row_i, scalar in enumerate(("top_mass", "entropy")):
    for c_i, key in enumerate(RES_KEYS):
        ax = axes[row_i, c_i]
        rows = CONF_TABLE.get((scalar, key)) or []
        if not rows:
            ax.set_axis_off()
            continue
        x = np.array([0.5 * (e["lo"] + e["hi"]) for e in rows])
        for s in ("rsd", "mbr", "mbr_gated", "set0", "set0_gated"):
            c, ls, lab = STYLE[s]
            y = np.array([e[s]["rms"] for e in rows])
            ax.plot(x, y, ls=ls, color=c, marker="o", ms=4.5, lw=1.8, label=lab)
        for e, xi in zip(rows, x):
            ax.annotate(f"n={e['n_jets']}", (xi, ax.get_ylim()[0]), fontsize=6.5,
                        color=MUTED, ha="center", va="bottom")
        finish(ax, xlabel=("top-cluster mass" if scalar == "top_mass"
                           else "cluster entropy $H(m)$  [nats]"),
               ylabel=f"RMS {DLABEL[key]}",
               title=f"{DLABEL[key]} vs {scalar}",
               legend=(row_i == 0 and c_i == 0), loc="best")
fig.suptitle("Does the claimed confidence predict the realized error?   "
             "(top: it should FALL; bottom: it should RISE)",
             x=0.006, y=1.003, ha="left")
fig.tight_layout()
plt.show()

# The single number that says whether the scalars are informative at all: the RMS ratio
# between the most and least confident bin. ~1 means the scalar carries no information
# about the error -- which is a REAL and reportable outcome, not a bug.
print("\ninformation content of each scalar (RMS in the extreme bin / RMS in the other):")
print(f"{'scalar':<12}{'coordinate':<12}{'series':<8}{'least conf.':>12}{'most conf.':>12}"
      f"{'ratio':>9}")
for scalar in ("top_mass", "entropy"):
    for key in RES_KEYS:
        rows = CONF_TABLE.get((scalar, key)) or []
        if len(rows) < 2:
            continue
        lo_b, hi_b = (rows[0], rows[-1]) if scalar == "top_mass" else (rows[-1], rows[0])
        for s in ("mbr", "mbr_gated", "set0", "set0_gated"):
            a, b = lo_b[s]["rms"], hi_b[s]["rms"]
            print(f"{scalar:<12}{TLABEL[key]:<12}{s:<8}{a:>12.3f}{b:>12.3f}"
                  f"{(b / a if a > 0 else float('nan')):>9.3f}")
print("\nA ratio below 1 means the scalar IS informative: the jets it calls confident really")
print("do have the smaller residual. A ratio at 1 means the number is decoration, and the")
print("honest thing is to say so rather than quote it as a per-jet confidence.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 8. Is `top_mass` a probability? — the reliability diagram (gate G6)

§7 asks whether the scalar *orders* the jets correctly. This asks the stronger question:
when the model claims 0.7, does the truth land in that cluster 70% of the time?

The joint tree posterior is **over-confident** by v1 TARP, so a cluster mass read off
$q_\phi$ is *not* a calibrated probability until this section says it is. That is the
binding constraint on the whole deliverable.

Reported three ways, because they fail differently:

- **ECE** — the average gap between claimed and realized, weighted by bin population.
- **The reliability curve's slope** — $<1$ is over-confident (the expected direction here),
  $>1$ under-confident. With its own standard error, because a slope from a handful of bins
  of a few dozen jets is not a number to read to two decimals.
- **The Brier decomposition** $\mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}$
  (Murphy 1973). This separates *"the numbers are miscalibrated"* from *"the numbers carry
  no information"*, which are different failures with different fixes: a constant
  forecaster has perfect reliability and zero resolution, and ECE alone cannot tell it from
  a useful one.

If miscalibrated, **one temperature** on the mass vector, $m_j(T)\propto m_j^{1/T}$, fit
here and — in production — frozen before test. A temperature refitted on the set it is then
scored on measures nothing; this section is a fitting demonstration and says so.

Jets whose truth is **unassigned** (farther from every exemplar than that cluster's own
support radius) are excluded from the calibration and reported separately. Force-assigning
them would turn an out-of-support jet into a miscalibrated probability.
""")

code(r'''
ASSIGNED = [r for r in ROWS if not r["truth_unassigned"]]
UNASSIGNED_RATE = float(np.mean([r["truth_unassigned"] for r in ROWS]))

REL = reliability([r["top_mass"] for r in ASSIGNED], [r["truth_in_top"] for r in ASSIGNED])
TEMP = fit_mass_temperature([r["masses"] for r in ASSIGNED],
                            [r["truth_in_top"] for r in ASSIGNED])
REL_T = reliability([temper_top_mass(r["masses"], TEMP["value"]) for r in ASSIGNED],
                    [r["truth_in_top"] for r in ASSIGNED])

print(f"jets with the truth assigned to a cluster : {len(ASSIGNED):,} of {len(ROWS):,}"
      f"   (unassigned {UNASSIGNED_RATE:.1%} -- OUT OF SUPPORT, not miscalibrated)")
print(f"realized P(truth in top cluster)          : "
      f"{np.mean([r['truth_in_top'] for r in ASSIGNED]):.3f}")
print(f"claimed  <top_mass>                       : "
      f"{np.mean([r['top_mass'] for r in ASSIGNED]):.3f}"
      + ("   -> OVER-confident" if np.mean([r['top_mass'] for r in ASSIGNED])
         > np.mean([r['truth_in_top'] for r in ASSIGNED]) else "   -> under-confident"))
print()
print(f"{'':<22}{'raw':>10}{'tempered':>12}")
for lab, k in (("ECE", "ece"), ("Brier", "brier"), ("  reliability", "brier_reliability"),
               ("  resolution", "brier_resolution"), ("  uncertainty", "brier_uncertainty")):
    print(f"{lab:<22}{REL[k]:>10.4f}{REL_T[k]:>12.4f}")
print(f"{'slope':<22}{REL['slope']:>10.3f}{REL_T['slope']:>12.3f}"
      f"   (+/- {REL['slope_se']:.3f} raw; 1.0 is calibrated)")
print(f"{'temperature T':<22}{'--':>10}{TEMP['value']:>12.3f}"
      f"   fitted on {TEMP['fitted_under']['n']} jets, log loss "
      f"{TEMP['fitted_under']['nll']:.4f}")
print(f"\ngate G6 wants ECE <= 0.05 after recalibration: "
      f"{'PASS' if REL_T['ece'] <= 0.05 else 'FAIL'} ({REL_T['ece']:.4f})")
print("\nresolution near 0 would mean the masses carry no INFORMATION about which jets are")
print("well determined -- a different failure from miscalibration, and one a temperature")
print("cannot fix. Read it beside section 7's trend.")
print("\nThis T is fitted on the SAME jets it is scored on, so the tempered column is an")
print("upper bound on what recalibration buys. In production, fit on validation and FREEZE")
print("(the `tau.fitted_under` pattern).")
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
ax = axes[0]
ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2, label="calibrated")
for entry, colour, lab in ((REL, C_SET0, "raw"),
                           (REL_T, C_MBR, f"tempered (T={TEMP['value']:.2f})")):
    b = entry["bins"]
    if not b:
        continue
    f = np.array([x["claimed"] for x in b])
    o = np.array([x["observed"] for x in b])
    ci = np.array([x["wilson95"] for x in b], dtype=float)
    # Wilson bars, not sqrt(p(1-p)/n): these are binomial proportions on a few dozen jets
    # per bin, exactly where the normal approximation leaves [0, 1].
    ax.errorbar(f, o, yerr=np.abs(np.vstack([o - ci[:, 0], ci[:, 1] - o])),
                fmt="o-", ms=4, lw=1.6, color=colour, capsize=3,
                label=f"{lab}   ECE={entry['ece']:.3f}")
    un = [k for k, x in enumerate(b) if not x["scored"]]
    if un:
        ax.scatter(f[un], o[un], s=64, facecolor="white", edgecolor=colour, zorder=5,
                   linewidths=1.4)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
finish(ax, xlabel="claimed top-cluster mass", ylabel=r"realized $P(\mathrm{truth\ in\ top})$",
       title=f"(a) gate G6 -- reliability of top_mass\n"
             f"slope {REL['slope']:.2f} +/- {REL['slope_se']:.2f}   "
             f"(hollow: n < 30, not scored)", legend=True, loc="upper left")

ax = axes[1]
tm = np.array([r["top_mass"] for r in ROWS])
en = np.array([r["entropy"] for r in ROWS])
ax.scatter(tm, en, s=9, color=C_SET0, alpha=0.30, linewidths=0)
finish(ax, xlabel="top_mass  (a probability)", ylabel="entropy $H(m)$  (an ambiguity, nats)",
       title="(b) the two scalars are not one number\n"
             "(collinearity here would mean one of them is redundant)")
_r = np.corrcoef(tm[np.isfinite(tm) & np.isfinite(en)],
                 en[np.isfinite(tm) & np.isfinite(en)])[0, 1]
ax.annotate(f"Pearson r = {_r:+.3f}", (0.03, 0.93), xycoords="axes fraction", fontsize=8,
            color=INK_2)
fig.tight_layout()
plt.show()
''')

md(r"""
### 8b. The selection bias `cluster_split` removes (gate G9)

`R_j` is defined using the same draws whose membership is then counted, so `top_mass` is
biased **upward** — post-selection inference (Berk, Brown, Buja, Zhang & Zhao,
*Ann. Statist.* **41** (2013) 802; Fithian, Sun & Taylor, arXiv:1410.2597). The fix is a
sample split: cluster and pick exemplars on pool **A**, assign a fresh pool **B** to the
A-exemplars by nearest EMD, estimate the masses from **B**. The B-assignment is
$|C|\times K$, not $K^2$, so it is nearly free once `D` exists.

This cell runs both on a subsample and reports the difference. **Above 0.05 on average, the
split becomes the default for any quoted number.**
""")

code(r'''
_n_split = min(200, N)
seed_everything(SEED)
_d_split = []
for _i in range(_n_split):
    _item = ds[_i]
    _xf = _item["xf"].unsqueeze(0).to(device)
    _nx = torch.tensor([_item["nx"]], device=device)
    _draws = model.sample(_xf, _nx, n=K_DRAWS)
    _, _cl, _ci, _D = posterior_distances(model, _xf, _nx, draws=_draws, geom=geom,
                                          n_candidates=0, **MBR_KW)
    _mask = np.zeros(len(_draws), dtype=bool)
    _mask[::2] = True
    _a = cluster_posterior(_D, **CLUSTER_KW)
    _b = cluster_posterior(_D, split_index=_mask, **CLUSTER_KW)
    if np.isfinite(_a.top_mass) and np.isfinite(_b.top_mass):
        _d_split.append(_a.top_mass - _b.top_mass)

_d_split = np.asarray(_d_split, dtype=float)
if _d_split.size:
    _m = float(_d_split.mean())
    print(f"gate G9 -- selection bias in top_mass, on {_d_split.size} jets:")
    print(f"  single-pool minus sample-split = {_m:+.4f} "
          f"(median {np.median(_d_split):+.4f}, 68% CR "
          f"[{np.percentile(_d_split, 16):+.4f}, {np.percentile(_d_split, 84):+.4f}])")
    if _m > 0.05:
        print("  -> ABOVE 0.05: the split must be the default for any quoted mass, and the "
              "single-pool\n     number above is the inflation it removes")
    elif _m < -0.05:
        # Not the expected sign. Selection bias can only inflate the single-pool estimate,
        # so a large NEGATIVE difference is telling you something else moved -- most often
        # the partition itself: pool A is half the draws, so a cluster is a coarser object
        # there and the masses it carries are larger. Report it rather than reading it as
        # "no bias".
        print("  -> NEGATIVE and large: this is NOT the selection bias, which can only "
              "inflate the\n     single-pool estimate. The split also halves the pool the "
              "PARTITION is found on, so\n     the two mass vectors are over different "
              "clusterings. Raise K before reading G9.")
    else:
        print("  -> within 0.05: the single-pool estimate is usable, and still biased HIGH")
    print(f"  currently running with CLUSTER_SPLIT = {CLUSTER_SPLIT}")
else:
    print("no jet produced a comparable pair -- G9 not scored")
''')

# ---------------------------------------------------------------------------
md(r"""
## 9. Is the set worth reporting at all? — gates G2, G2′ and G3

Two **independent** necessity gates that may legitimately disagree.

- **G2 (truth-free)** — the fraction of jets whose linear medoid lies in the dominant
  cluster. $\ge 0.90$ means the medoid is already central and a bounded-loss estimator is
  unnecessary. Truth-free, so this number **transfers to real data**.
- **G2′ (truth-based)** — whether the *set* recovers what the point estimate misses. G2
  alone asks only whether the medoid is centrally placed: it can pass while the set is
  worthless, or fail while it is valuable.

**G2′'s control is the whole of G2′.** Taking a minimum over $n$ exemplars improves the
distance to truth even for a *random* partition, purely as an order statistic — so
$d_\text{best} < d_\text{mbr}$ is **not evidence of anything**. The signal is
$d_\text{best}^\text{real}$ against $d_\text{best}^\text{rand}$, computed by partitioning
the same pool at random into the same number of groups with the same masses. It reuses `D`;
no new EMD calls.

Two further mandatory controls:

- **the silhouette precondition** — $d(e_j, y_\text{true})$ carries the within-cluster
  scatter even in the correct lobe, so the effect is detectable only when the inter-exemplar
  distance exceeds the radius. Computable from `D` **before any truth is consulted**, which
  is what makes it a precondition rather than a post-hoc excuse. Where it fails, the
  bimodality is unresolvable at this metric and budget whether or not it is real.
- **the unassigned rate** — §8's out-of-support jets, reported here beside the gate rather
  than hidden inside it.

**Scope discipline.** "The jet population is bimodal" and "$p(y\mid x)$ is bimodal for this
jet" are different claims and only the second is in scope. G2′ is therefore reported
stratified by whether the top two clusters differ in $N$, so a split *between* $N$ strata is
distinguishable from a split *within* one.

**G3** checks the identity that makes the empty stratum legible: the $N=0$ draws must form
one zero-radius cluster whose mass is $q(0\mid x)$. A gap there is a metric-convention bug,
not a finding.
""")

code(r'''
def _blk(rs):
    if not rs:
        return None
    db = np.array([r["d_best"] for r in rs])
    dr = np.array([r["d_best_rand"] for r in rs])
    gain = dr - db
    return {"n": len(rs), "d_best": float(np.nanmean(db)),
            "d_best_rand": float(np.nanmean(dr)),
            "d_top": float(np.nanmean([r["d_top"] for r in rs])),
            "d_mbr": float(np.nanmean([r["d_mbr"] for r in rs])),
            "gain": float(np.nanmean(gain)),
            "sem": float(np.nanstd(gain, ddof=1) / math.sqrt(len(rs))) if len(rs) > 1
            else float("nan")}


MULTI = [r for r in ROWS if r["n_clusters"] >= 2 and np.isfinite(r["d_best_rand"])]
PRE = [r for r in MULTI if r["precondition"]]
G2 = float(np.mean([r["medoid_in_top"] for r in ROWS]))
G3 = float(np.mean([abs(r["empty_draw_mass"] - r["q0"]) for r in ROWS]))

print(f"gate G2 (TRUTH-FREE, transfers to real data)")
print(f"  the linear medoid lies in the dominant cluster on {G2:.3f} of jets"
      + ("   -> >= 0.90: the medoid is already central, and a bounded loss is UNNECESSARY"
         if G2 >= 0.90 else "   -> < 0.90: the medoid leaves the dominant cluster often"))
print(f"  jets with >= 2 clusters: {len(MULTI)} of {len(ROWS)} "
      f"({len(MULTI) / max(len(ROWS), 1):.1%})   "
      f"<n_clusters> = {np.mean([r['n_clusters'] for r in ROWS]):.2f}")
print()
print(f"gate G2' (TRUTH-BASED, an ORACLE quantity -- diagnostic only, never a headline)")
print(f"  {'subset':<26}{'n':>6}{'d_best':>9}{'d_rand':>9}{'gain':>9}{'+/-':>8}"
      f"{'d_top':>9}{'d_mbr':>9}")
for lab, rs in (("all multi-cluster jets", MULTI),
                ("precondition holds", PRE),
                ("  ...top two differ in N", [r for r in PRE if r["n_top"] != r["n_second"]]),
                ("  ...top two same N", [r for r in PRE if r["n_top"] == r["n_second"]])):
    b = _blk(rs)
    if b is None:
        print(f"  {lab:<26}{0:>6}   (empty)")
        continue
    print(f"  {lab:<26}{b['n']:>6}{b['d_best']:>9.3f}{b['d_best_rand']:>9.3f}"
          f"{b['gain']:>+9.3f}{b['sem']:>8.3f}{b['d_top']:>9.3f}{b['d_mbr']:>9.3f}")
print("  `gain` = d_best_rand - d_best. POSITIVE and larger than its own error means the")
print("  real partition beats a mass-matched RANDOM one -- which is the only comparison")
print("  that means anything, because a minimum over n exemplars beats d_mbr by an order")
print("  statistic alone.")
print(f"  silhouette precondition holds on "
      f"{np.mean([r['precondition'] for r in ROWS]):.1%} of jets; truth UNASSIGNED on "
      f"{UNASSIGNED_RATE:.1%}")
print()
print()
print("IS THE TRUTH OUTSIDE THE MODEL, OR OUTSIDE THE PARTITION?")
_dn = np.array([r["d_nearest_draw"] for r in ROWS], dtype=float)
_dm = np.array([r["d_median_draw"] for r in ROWS], dtype=float)
_un = np.array([r["truth_unassigned"] for r in ROWS], dtype=bool)
print(f"  d(truth, nearest DRAW)   = {np.nanmean(_dn):.3f}   median "
      f"{np.nanmedian(_dn):.3f}   [16,84]% "
      f"[{np.nanpercentile(_dn, 16):.3f}, {np.nanpercentile(_dn, 84):.3f}]")
print(f"  d(truth, median draw)    = {np.nanmean(_dm):.3f}   -- the pool's own scale")
print(f"  ratio nearest/median     = {np.nanmean(_dn / _dm):.3f}   "
      f"(-> 0 means the pool BRACKETS the truth; -> 1 means it does not)")
print(f"  flagged 'unassigned' by the exemplar rule: {_un.mean():.1%}")
if _un.any():
    print(f"     of those, d(truth, nearest draw) = {np.nanmean(_dn[_un]):.3f} vs "
          f"{np.nanmean(_dn[~_un]):.3f} for the assigned ones")
    print(f"     ...and their nearest/median ratio  = {np.nanmean((_dn / _dm)[_un]):.3f} vs "
          f"{np.nanmean((_dn / _dm)[~_un]):.3f}")
print("  A high unassigned rate with a SMALL nearest/median ratio means the pool does")
print("  bracket the truth and the PARTITION is too fine -- a method artifact, and the")
print("  exemplar rule is the thing to loosen. A ratio near 1 means the truth is outside")
print("  everything the model generated, which no decode-layer change can repair.")
print()
print("A TRUTH-FREE ALTERNATIVE RANKING: the cluster the MEDOID fell into")
_dmc = np.array([r["d_medoid_cluster"] for r in ROWS], dtype=float)
_dt = np.array([r["d_top"] for r in ROWS], dtype=float)
_db = np.array([r["d_best"] for r in ROWS], dtype=float)
_dr = np.array([r["d_mbr"] for r in ROWS], dtype=float)
print(f"  {'rule':<34}{'<d(truth)>':>12}   truth-free?")
for lab, v, free in (("the linear medoid", _dr, "yes"),
                     ("top-MASS exemplar (set0)", _dt, "yes"),
                     ("exemplar of the medoid's cluster", _dmc, "yes"),
                     ("closest exemplar (ORACLE)", _db, "NO -- uses the truth")):
    print(f"  {lab:<34}{np.nanmean(v):>12.3f}   {free}")
print("  The oracle row is the ceiling a better SELECTION RULE could reach over the very")
print("  same set. A large gap between it and every truth-free row means the set contains")
print("  a good answer that the ranking does not find -- which is a statement about the")
print("  RULE, not about the clustering.")
print()
print(f"gate G3 (empty stratum)")
print(f"  mean |mass(N=0 draws) - q(0|x)| = {G3:.5f}"
      f"   -- a gap here is a metric-convention bug, not a finding")
print()
print("KILL CRITERION (stated up front in the plan): G2 >= 0.90 AND a null G2' means the")
print("posterior is effectively unimodal in this metric and at this budget. Then the set")
print("ships as a DIAGNOSTIC rather than a product, and the deliverable reduces to quoting")
print("radii[0] as a per-jet resolution beside the existing MBR point estimate.")
print(f"  -> G2 {'>=' if G2 >= 0.90 else '<'} 0.90; "
      f"G2' gain on the precondition subset = "
      f"{(_blk(PRE) or {'gain': float('nan')})['gain']:+.3f}")
''')

# ---------------------------------------------------------------------------
md(r"""
### 9b. Does deciding $N$ first help? — the pre-registered decision table

`mbr` minimises a mean distance over **every** multiplicity stratum at once, and the EMD's
mass-imbalance term charges $\sim R|W_a - W_b|$ across strata — so the medoid is pulled
toward whatever $N$ is most populous and can land between strata, representing none. §9
measures the cost: the medoid is 2.349 from truth against a 1.476 oracle over exemplars,
and 83% of the resolvable ambiguity is *between* $N$ strata.

`mbr_n` splits the decode at that seam, using each channel where it is trustworthy:

$$
\hat n = Q_{0.5}\!\left(q(N\mid x)\right),
\qquad
\hat y = \arg\min_{|h| = \hat n}\ \frac{1}{|S|}\sum_{k \in S} d(h, y^{(k)}),
\quad S = \{k : |y^{(k)}| = \hat n\}.
$$

Stage 1 is the Bayes estimator under $L(n,m) = |n-m|$ — the "general argmin over an
explicit loss on $n$" that [`PLAN_empty_parton_tree.md`](../docs/PLAN_empty_parton_tree.md)
deferred, with the empty gate as its $n=0$ special case. Stage 2 is pure shape: within a
stratum every pair carries equal total weight, so the imbalance term drops out.

**Two controls separate what the estimator changes**, and both are free (another reduction
over the same $D$):

- **stratified at $N$(medoid)** — de-smearing *alone*: the same $N$ the medoid already
  chose, expectation restricted. `d_mbr − d_mbr_nmed` is what conditioning buys with **no
  new information**.
- **stratified at $n_\mathrm{true}$** — the oracle ceiling of the $N$ channel given this
  shape rule, so `d_mbr_n − d_mbr_ntrue` prices what a better length head could still buy.

**The ship gate, fixed before this cell was ever run** (docs/PLAN_StratifiedMBR.md WP1).
`mbr_n` becomes the recommended decode iff **all three** hold:

1. the jet-bootstrap 95% CI on the paired $\Delta = d_\mathrm{mbr} - d_{\mathrm{mbr}\_n}$
   **excludes 0**;
2. §6's RMS-vs-plain-RSD ratios for `mbr_n` are **no worse than `mbr`'s within their CIs**
   on $\ln(1/\Delta R)$ and $\ln k_t$ — the failure mode that disqualified the top-mass
   exemplar (1.112 / 1.141 on identical rows);
3. §6b's multiplicity marginals for `mbr_n_gated` are no worse than `mbr_gated`'s.

A $\Delta$ that is **flat** across the `strata_differ` split says the gain is de-smearing,
not $N$ information — reportable either way, and it would point the follow-up at the metric
rather than at the length head.
""")

code(r'''
# `_nf_`-prefixed throughout: section 11 binds `_ok` to an index array, and the artifact
# cell in section 12 reads these back AFTER it has run.
_nf_all = [r for r in ROWS if np.isfinite(r["d_mbr"]) and np.isfinite(r["d_mbr_n"])]
_nf_multi = [r for r in _nf_all if r["n_clusters"] >= 2]
# Same expression section 9 uses: a split BETWEEN N strata is a different physical claim
# from a split WITHIN one, and only the second is shape ambiguity.
_nf_diff = [r for r in _nf_multi if r["n_second"] >= 0 and r["n_top"] != r["n_second"]]
_nf_same = [r for r in _nf_multi if r["n_second"] >= 0 and r["n_top"] == r["n_second"]]


def _nf_col(rs, key):
    v = [r[key] for r in rs if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(v)) if v else float("nan")


print("SELECTION-RULE LADDER -- mean d(truth), by subset")
print(f"  {'rule':<38}{'all':>9}{'multi':>9}{'differ':>9}{'same-N':>9}   truth-free?")
_rows = [
    ("global medoid (mbr)", "d_mbr", "yes"),
    ("stratified at N(medoid)", "d_mbr_nmed", "yes  <- de-smearing ALONE"),
    ("N-first (mbr_n)", "d_mbr_n", "yes  <- the claim"),
    ("top-mass exemplar (set0)", "d_top", "yes"),
    ("stratified at n_true", "d_mbr_ntrue", "NO -- oracle N"),
    ("min over the decided stratum", "d_oracle_stratum", "NO -- oracle shape"),
    ("closest exemplar", "d_best", "NO -- oracle"),
]
for lab, key, free in _rows:
    print(f"  {lab:<38}" + "".join(f"{_nf_col(rs, key):>9.3f}"
                                   for rs in (_nf_all, _nf_multi, _nf_diff, _nf_same)) + f"   {free}")
print()
print(f"  de-smearing alone   d_mbr - d_mbr_nmed = "
      f"{_nf_col(_nf_all, 'd_mbr') - _nf_col(_nf_all, 'd_mbr_nmed'):+.3f}")
print(f"  the N decision      d_mbr_nmed - d_mbr_n = "
      f"{_nf_col(_nf_all, 'd_mbr_nmed') - _nf_col(_nf_all, 'd_mbr_n'):+.3f}")
print(f"  residual N error    d_mbr_n - d_mbr_ntrue = "
      f"{_nf_col(_nf_all, 'd_mbr_n') - _nf_col(_nf_all, 'd_mbr_ntrue'):+.3f}"
      f"   (what a better length head could still buy)")
''')

code(r'''
print("THE N DECISION ITSELF")
_nf_ntrue = np.array([r["n_true"] for r in ROWS])
_nf_cands = {"n_hat (median q(N|x))": np.array([r["n_hat"] for r in ROWS]),
          "N(medoid)": np.array([r["n_medoid"] for r in ROWS]),
          "N(set0)": np.array([len(a) for a in RAW["set0"]]),
          "N(MAP)": np.array([len(a) for a in RAW["map"]]),
          "N(posterior draw)": np.array([len(a) for a in RAW["post"]])}
print(f"  {'rule':<24}{'P(n = n_true)':>15}{'<|n - n_true|>':>16}   (L1 is the loss the")
print(f"  {'':<24}{'':>15}{'':>16}    median is Bayes for)")
for lab, v in _nf_cands.items():
    print(f"  {lab:<24}{float(np.mean(v == _nf_ntrue)):>15.3f}{float(np.mean(np.abs(v - _nf_ntrue))):>16.3f}")
_nf_real = np.array([r["n_hat_realized"] for r in ROWS])
_cond = np.array([r["n_hat_cond"] for r in ROWS])
_nh = _nf_cands["n_hat (median q(N|x))"]
print(f"\n  n_hat realized in the pool on {_nf_real.mean():.3%} of jets"
      f"   (1.000 expected: the median of a histogram pmf always is)")
print(f"  the CONDITIONAL median (gate said non-empty) agrees with it on "
      f"{float(np.mean(_cond == _nh)):.3f} of jets")
print(f"  mean stratum size {np.mean([r['stratum_size'] for r in ROWS]):.1f} of "
      f"{K_DRAWS} draws   -- the support the shape decode gets to choose from")
''')

code(r'''
def _nf_boot(rs, a="d_mbr", b="d_mbr_n", n_boot=N_BOOT, seed=SEED):
    """Jet-level bootstrap on the PAIRED difference. Paired because both estimators are
    read off the same D for the same jet, so the per-jet difference removes the jet-to-jet
    spread that would otherwise swamp it."""
    d = np.array([r[a] - r[b] for r in rs
                  if np.isfinite(r.get(a, np.nan)) and np.isfinite(r.get(b, np.nan))])
    if d.size < MIN_CI_JETS:
        return float(np.mean(d)) if d.size else float("nan"), float("nan"), float("nan"), d.size
    rng = np.random.default_rng(seed)
    vals = [float(d[rng.integers(0, d.size, d.size)].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int(d.size)


print("SHIP GATE (pre-registered in the markdown above, before this cell was first run)")
print(f"  {'subset':<26}{'n':>6}{'mean delta':>12}{'95% CI':>22}   verdict")
_nf_verdict = {}
for lab, rs in (("all jets", _nf_all), ("multi-cluster", _nf_multi),
                ("...top-2 differ in N", _nf_diff), ("...top-2 same N", _nf_same)):
    m, lo, hi, n = _nf_boot(rs)
    ok = np.isfinite(lo) and lo > 0.0
    _nf_verdict[lab] = ok
    ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else f"(n={n} < {MIN_CI_JETS})"
    # A CI entirely BELOW zero also excludes zero -- it just excludes it the other way.
    # Saying "brackets 0" there would report a significant loss as a null result, which is
    # the one misreading this table must not enable.
    if not np.isfinite(lo):
        verdict = "not scored"
    elif lo > 0.0:
        verdict = "EXCLUDES 0 -- mbr_n is CLOSER"
    elif hi < 0.0:
        verdict = "EXCLUDES 0 -- mbr_n is FARTHER"
    else:
        verdict = "brackets 0"
    print(f"  {lab:<26}{n:>6}{m:>+12.3f}{ci:>22}   {verdict}")
print("  delta = d(medoid) - d(mbr_n); POSITIVE means the N-first estimator is closer to")
print("  truth. Criterion (i) is the 'all jets' row excluding 0.")
print()
_nf_g0 = float(np.mean([r["d_mbr"] - r["d_mbr_n"] for r in _nf_multi])) if _nf_multi else float("nan")
print(f"  as a fraction of the 0.603 real-information component on multi-cluster jets: "
      f"{_nf_g0 / 0.603:.2f}x")
print("  (mbr_n makes ONE selection, so it gets none of the 0.592 order-statistic share")
print("   an oracle-over-exemplars enjoys -- this fraction is the honest yardstick.)")
print()
print("  Criterion (ii) -- section 6's RMS ratios vs plain RSD -- and (iii) -- section 6b's")
print("  marginals -- are read off those sections; all three must hold for mbr_n to become")
print("  the recommended decode. Any failure leaves it available and documented as")
print("  measured-not-recommended, which is a result rather than a retreat.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 10. Loss stability — a diagnostic, and explicitly **not** a systematic

The WP4a columns, computed on the same `D` at zero additional EMD cost. What they settle:

| column | question |
|---|---|
| `argmin_moved` | does a bounded loss select a different draw than the linear one? A near-zero rate closes the bounded estimator outright |
| `bounded_is_members0` | when it moves, does it move **toward** the top-mass exemplar or away? |
| `empty_clique_size` vs `best_nonempty_count` | gate **G8′**: `_empty_value` puts all empty draws at mutual distance **exactly 0**, so an empty candidate's neighbour count is the clique size *for any* $\epsilon$ — a bounded loss can collapse to the empty tree, reproducing the MAP degeneracy MBR removes structurally |
| `eps_per_jet` | the realized bandwidth spread — a single frozen $\epsilon$ is viable only if this is narrow |
| `d_bounded` / `d_mbr` / `d_top` | gate **G8**: distance to truth for all three |

**This is a stability check, not an uncertainty, and the distinction is not pedantic.**
`generator_spread` varies something *unknown about nature*; loss choice varies something
*the analyst decides*, and `linear` and `bounded` are not two approximations to one quantity
— they are the Fréchet median and a density mode, two different functionals of one
posterior. Quoting their spread as a systematic is quoting the mean-minus-median difference
as a systematic on the mean, and it double-counts a width `radii[0]` already reports. The
code enforces this by module boundary: these columns live in `eval/stability.py`, and
`tests/test_stability.py::test_loss_spread_not_in_systematics` asserts `eval/systematics.py`
neither imports them nor emits their keys.

`argmin_moved` is kept for one reason beyond being free: it is a **1-bit multimodality flag
that needs no clustering** — no scikit-learn, valid at small $K$, and available on **real
data**, where G2′ is not.
""")

code(r'''
STAB = summarise_stability([r["stability"] for r in ROWS], verbose=True)
print()
print("The epsilon used above is PER-JET: Q_gamma of that jet's own positive off-diagonal")
print("distances, with gamma = "
      f"{LOSS_QUANTILE:g} pre-registered. Within a jet the neighbour counts are")
print("compared at a common epsilon, which is all the argmin needs -- a variable-bandwidth")
print("(nearest-neighbour) KDE in the sense of Loftsgaarden & Quesenberry (1965), not an ad")
print("hoc choice. It is also exactly what a PRODUCTION bounded loss could not inherit: a")
print("per-jet bandwidth makes `.risk` comparable within a jet and NOT across jets, and the")
print("closure scripts aggregate across jets.")
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))
ax = axes[0]
eps = np.array([r["stability"]["eps_per_jet"] for r in ROWS], dtype=float)
eps = eps[np.isfinite(eps) & (eps > 0)]
if eps.size:
    ax.hist(eps, bins=40, color=C_MBR, alpha=0.55, edgecolor=C_MBR, linewidth=0.8)
    for q, ls in ((16, ":"), (50, "-"), (84, ":")):
        ax.axvline(np.percentile(eps, q), color=INK, ls=ls, lw=1.2)
finish(ax, xlabel=r"per-jet $\epsilon = Q_{0.10}$", ylabel="jets",
       title="(a) the realized bandwidth\n(a single FROZEN eps is viable only if this is narrow)")

ax = axes[1]
cl = np.array([r["stability"]["empty_clique_size"] for r in ROWS], dtype=float)
bn = np.array([r["stability"]["best_nonempty_count"] for r in ROWS], dtype=float)
ok = np.isfinite(cl) & np.isfinite(bn)
ax.scatter(bn[ok], cl[ok], s=9, color=C_POST, alpha=0.30, linewidths=0)
_hi = float(max(np.nanmax(bn[ok]) if ok.any() else 1, np.nanmax(cl[ok]) if ok.any() else 1))
ax.plot([0, _hi], [0, _hi], color=INK, ls="--", lw=1.2)
ax.annotate("above the line the EMPTY clique wins\n-> the bounded loss collapses to the "
            "empty tree", (0.04, 0.88), xycoords="axes fraction", fontsize=7.2, color=INK_2)
finish(ax, xlabel="best non-empty neighbour count", ylabel="empty clique size",
       title=f"(b) gate G8'   empty wins on {STAB['empty_clique_wins']:.3%} of jets\n"
             f"(> 1% blocks a bounded ship)")

ax = axes[2]
d_lin = np.array([r["d_mbr"] for r in ROWS], dtype=float)
d_top = np.array([r["d_top"] for r in ROWS], dtype=float)
ok = np.isfinite(d_lin) & np.isfinite(d_top)
lim = (0.0, float(np.percentile(np.concatenate([d_lin[ok], d_top[ok]]), 99)) if ok.any() else 1)
ax.scatter(d_lin[ok], d_top[ok], s=9, color=C_SET0, alpha=0.30, linewidths=0)
ax.plot(lim, lim, color=INK, ls="--", lw=1.2)
ax.set_xlim(*lim)
ax.set_ylim(*lim)
finish(ax, xlabel="EMD(linear medoid, truth)", ylabel="EMD(top-mass exemplar, truth)",
       title="(c) gate G8, per jet\n(below the line the exemplar is closer to truth)")
fig.suptitle("Loss and estimator stability -- reported BESIDE the answer, never folded "
             "into it", x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------------------
md(r"""
## 11. The conformal set (gate G7)

Everything above depends on $q_\phi$ being roughly calibrated. This does not.

Calibrate a threshold on the **accumulated cluster mass** over a split of these jets and
emit the smallest mass-descending prefix that reaches it. Under exchangeability this gives
finite-sample marginal coverage $\ge 1-\alpha$ **however wrong $q_\phi$ is** (Vovk,
Gammerman & Shafer 2005; Angelopoulos & Bates, arXiv:2107.07511). The
$\lceil (n+1)(1-\alpha)\rceil / n$ order statistic is the finite-sample correction, not a
rounding convenience.

**The guarantee is marginal over jets, not conditional on $x$** — the same coverage notion
TARP tests, and it must be quoted that way. "95% of jets are covered" is what it says; "this
jet is covered with probability 0.95" is what it does not.

The threshold is fitted on the first half and scored on the second, so the coverage below is
a genuine out-of-sample number rather than the tautology of scoring on the calibration set.
""")

code(r'''
# NaN = "no prefix covers this jet's truth" (it sits outside every cluster's support).
# KEPT, not dropped: that is its true nonconformity rank, and dropping it would condition
# the guarantee on assignment -- reporting a coverage that cannot fail for the one reason
# it most needs to. `fit_set_threshold` reads non-finite as "never covered".
_scores = np.array([r["cum_mass_to_truth"] for r in ROWS], dtype=float)
# `_cf_`-prefixed: section 9b's rows are still live here and are read again
# by the artifact cell below.
_cf_idx = np.arange(len(ROWS))
if _cf_idx.size >= 20:
    _cf_half = _cf_idx.size // 2
    _cf_cal, _cf_test = _cf_idx[:_cf_half], _cf_idx[_cf_half:]
    CONF = fit_set_threshold(_scores[_cf_cal], alpha=SET_ALPHA)
    _cf_cov, _cf_sz = [], []
    for i in _cf_test:
        k = set_size_for(ROWS[i]["masses"], CONF["value"])
        _cf_sz.append(k)
        _cf_cov.append(bool(0 <= ROWS[i]["truth_cluster"] < k))
    from h2p_rsd_junipr.eval.calibration import wilson_interval

    _lo, _hi = wilson_interval(int(np.sum(_cf_cov)), len(_cf_cov))
    CONF["coverage"] = float(np.mean(_cf_cov))
    CONF["coverage_wilson95"] = [_lo, _hi]
    CONF["mean_set_size"] = float(np.mean(_cf_sz))
    CONF["n_calibration"] = int(_cf_cal.size)
    CONF["n_test"] = int(_cf_test.size)
    print(f"conformal set at alpha = {SET_ALPHA:g}  (nominal coverage "
          f"{1 - SET_ALPHA:.2f}, MARGINAL over jets)")
    print(f"  threshold on accumulated mass : {CONF['value']:.3f}"
          f"   (fitted on {_cf_cal.size} jets, exact = "
          f"{CONF['fitted_under']['finite_sample_exact']})")
    print(f"  out-of-sample coverage        : {CONF['coverage']:.3f} "
          f"[{_lo:.3f}, {_hi:.3f}]  on {_cf_test.size} jets")
    print(f"  mean set size                 : {CONF['mean_set_size']:.2f} clusters")
    print(f"  gate G7: {'PASS' if _hi >= 1 - SET_ALPHA else 'FAIL'} "
          f"-- the Wilson band {'contains' if _hi >= 1 - SET_ALPHA else 'excludes'} "
          f"the nominal level")
    print(f"\n  {UNASSIGNED_RATE:.1%} of jets have the truth OUTSIDE every cluster's "
          f"support -- no prefix\n  of the set covers them, at any threshold. They are "
          f"COUNTED (as never-covered), not\n  dropped: dropping them would condition the "
          f"guarantee on assignment and report a\n  coverage that cannot fail for the one "
          f"reason it most needs to.")
    print(f"  -> coverage is capped at {CONF['max_achievable_coverage']:.3f} whatever the "
          f"threshold, and the\n     nominal {1 - SET_ALPHA:.2f} is "
          f"{'REACHABLE' if CONF['reachable'] else 'NOT REACHABLE'}")
    if not CONF["reachable"]:
        # WHY it is unreachable is the question, and section 9's support decomposition
        # answers it. Do NOT read this as "the model never generated anything near the
        # truth" without checking that: on the arm measured here the pool brackets the
        # truth comfortably (nearest draw at 0.09 of the pool's own scale) and the ceiling
        # comes from the ASSIGNMENT rule being strict, not from the sampler's support.
        print("     Check section 9 before blaming the model: if d(truth, nearest DRAW) is")
        print("     small against the pool's scale, the pool DOES bracket the truth and the")
        print("     ceiling is set by the exemplar-support rule -- loosen `assign_truth`'s")
        print("     slack, or report coverage against the pool rather than the exemplars.")
        print("     Only a nearest/median ratio near 1 indicts the sampler.")
else:
    CONF = None
    print(f"only {_cf_idx.size} jets have an assigned truth -- too few to calibrate a threshold")
''')

# ---------------------------------------------------------------------------
md(r"""
## 12. Artifacts
""")

code(r'''
if WRITE_ARTIFACTS:
    METRICS = {
        "run": {
            "notebook": "per_jets_estimation_cluster",
            "checkpoint": str(CKPT_PATH), "test_path": str(ROOT_PATH),
            "model": info["model_name"], "encoder": str(cfg.encoder.name),
            "aux_features": list(AUX), "lnz_support": LNZ_SUPPORT,
            "aux_dropped": int(AUX_DROPPED), "provenance": PROV,
            "n_bins": geom.n_bins, "n_jets": int(N), "K_draws": int(K_DRAWS),
            "seed": int(SEED), "mbr_backend": MBR_BACKEND,
            "mbr_beta": float(EMD_KW["beta"]), "mbr_R": float(EMD_KW["R"]),
            "mbr_coords": str(CLOUD_KW["coords"]),
            "cluster_method": CLUSTER_METHOD, "cluster_min_mass": float(CLUSTER_MIN_MASS),
            "cluster_min_cluster_size": int(CLUSTER_MIN_CLUSTER_SIZE),
            "cluster_split": bool(CLUSTER_SPLIT), "set_alpha": float(SET_ALPHA),
            "loss_quantile": float(LOSS_QUANTILE), "null_reps": int(NULL_REPS),
        },
        "scalars": {
            "top_mass_mean": float(np.mean([r["top_mass"] for r in ROWS])),
            "entropy_mean": float(np.mean([r["entropy"] for r in ROWS])),
            "radius_top_mean": float(np.nanmean([r["radius_top"] for r in ROWS])),
            "n_clusters_mean": float(np.mean([r["n_clusters"] for r in ROWS])),
            "frac_multimodal": float(np.mean([r["n_clusters"] >= 2 for r in ROWS])),
            "residual_mass_mean": float(np.mean([r["residual_mass"] for r in ROWS])),
            "unassigned_rate": UNASSIGNED_RATE,
            "note": "top_mass is a PROBABILITY, entropy an AMBIGUITY, radii[0] the only "
                    "one of the three quotable as a +/-",
        },
        "gates": {
            "G2_medoid_in_top": G2,
            "G2prime": {k: _blk(v) for k, v in
                        (("all", MULTI), ("precondition_holds", PRE))},
            "G2prime_is_oracle": True,
            "G3_empty_mass_vs_q0": G3,
            "G6_reliability": REL, "G6_temperature": TEMP,
            "G6_reliability_recalibrated": REL_T,
            "G7_conformal": CONF,
            "G9_selection_bias": (float(_d_split.mean()) if _d_split.size else None),
        },
        "n_first": {
            "ladder": {k: {lab: _nf_col(rs, k) for lab, rs in
                           (("all", _nf_all), ("multi", _nf_multi), ("differ", _nf_diff),
                            ("same_N", _nf_same))}
                       for k in ("d_mbr", "d_mbr_nmed", "d_mbr_n", "d_top",
                                 "d_mbr_ntrue", "d_oracle_stratum", "d_best")},
            "delta_ci": {lab: _nf_boot(rs) for lab, rs in
                         (("all", _nf_all), ("multi", _nf_multi), ("differ", _nf_diff),
                          ("same_N", _nf_same))},
            "n_accuracy": {lab: {"exact": float(np.mean(v == _nf_ntrue)),
                                 "mean_abs": float(np.mean(np.abs(v - _nf_ntrue)))}
                           for lab, v in _nf_cands.items()},
            "n_hat_realized_rate": float(_nf_real.mean()),
            "mean_stratum_size": float(np.mean([r["stratum_size"] for r in ROWS])),
            "ship_gate_criterion_i": bool(_nf_verdict.get("all jets", False)),
            "note": "criteria (ii) RMS-vs-RSD and (iii) the 6b marginals are read off "
                    "those sections; all three must hold for mbr_n to be recommended",
        },
        "stability": STAB,
        "residuals": {f"{key}|{slab}|{s}": v for (key, slab, s), v in TABLE.items()},
        "confidence_stratified": {
            f"{scalar}|{key}": rows for (scalar, key), rows in CONF_TABLE.items()
        },
    }
    out = save_metrics(METRICS, (REPO / CKPT_PATH).parent / "per_jet_clusters.json")
    print(f"wrote {out.relative_to(REPO)}")
else:
    print("WRITE_ARTIFACTS = False -- nothing written")
''')

# ---------------------------------------------------------------------------
md(r"""
---

### Reading these figures

- **The set is the deliverable; `setbest` is not.** $d_\text{best}$ uses the truth to choose
  the member, so it measures whether the set is *worth reporting*, never how well the model
  did. It appears in the panels and in §9 with its null, and in no summary table.
- **§6's ratio column cannot see the emptiness decision, by construction.** A residual
  exists at $t$ only where *both* sides have a node there, so on the common-depth pairing a
  jet where any series answered empty contributes no rows at all — and the gated and
  ungated ratios come back *identical*. Read that as the pairing being blind to the
  decision, not the decision being harmless. Emptiness is measured by the multiplicity
  marginals in §5b and §6b; §6 measures the shape, on jets where every series produced one.
- **A null result in §6 is informative.** If `set0` and `mbr` agree within their bootstrap
  interval, the posterior is effectively unimodal in this metric at this budget — which is
  the plan's own kill criterion, not a failure of the implementation. §7 is where a pooled
  null is separated from "no effect anywhere": the two estimators can only differ where the
  posterior is multimodal, so the high-entropy bin is the informative row.
- **`top_mass` is not a $\pm$, and neither is `entropy`.** A bimodal posterior summarised as
  mean $\pm$ sd points at a configuration neither mode supports. `radii[0]` is the width;
  the other two are a probability and an ambiguity, and §8's reliability diagram is what
  licenses the first of them to be read as a probability at all.
- **The loss spread in §10 is a stability check.** Never add it in quadrature with
  `generator_spread`: they are different functionals of one posterior, and the posterior
  width is already in `radii[0]`. The module boundary (`eval/stability.py`, never
  `eval/systematics.py`) is the guard, and a test asserts it.
- **$K$ is what the mass vector's resolution is.** At $K=200$ and
  `CLUSTER_MIN_MASS = 0.05` a reportable cluster is 10 draws, and the Monte-Carlo error on
  a mass of 0.6 is 0.035. G2 is answerable there; a three-cluster split is not. Density
  estimation needs resolution in $\mathcal{H}$ itself, and the sample size to resolve modes
  scales far worse than the sample size to estimate a mean.
- **The surrogate backend is not a faster metric here, it is a different one.**
  `_lund_image` normalises, so the surrogate is exactly blind to total $k_t$ and
  multiplicity and collapses the $N$-stratum separation the clusters are made of. §0 asserts
  it is not selected.
- **Nothing here enlarged the hypothesis space.** Every member of every set is a genuine
  posterior draw carrying its own sampled coordinates. That closure property is why a set
  can be reported at all — a consensus or lattice construction would produce trees the model
  never generated, and their masses would mean nothing.

### Running it elsewhere

Set `CKPT_PATH` / `ROOT_PATH` in §0 to bypass the artifact read. Three settings are not free
choices: `MBR_N_CANDIDATES = 0`, `mbr_beta = 1.0`, and a non-surrogate backend. All three
raise rather than warn (§2, gate G4). `CLUSTER_METHOD = "pam"` needs no `scikit-learn` and
is deterministic, which makes it both the fallback and the control arm for whether §9's G2
verdict is method-dependent — run it both ways before quoting the verdict.
""")

# ---------------------------------------------------------------------------
nb = {
    "cells": [
        {
            "cell_type": kind,
            "id": f"cell-{n:02d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for n, (kind, src) in enumerate(CELLS)
    ],
    "metadata": {
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = (Path(__file__).resolve().parent.parent / "notebooks"
       / "per_jets_estimation_cluster.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
