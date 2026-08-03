"""Build notebooks/per_jets_estimation.ipynb.

    python scripts/make_per_jets_nb.py

The notebook is ~1200 lines of source, which is past what the notebook editor can open,
so it is generated from here rather than hand-edited -- the same reason and the same
pattern as scripts/make_prod_closure_nb.py. THIS FILE is the source of truth: an edit made
straight to the .ipynb is lost the next time anyone regenerates.

Regenerating drops the executed outputs, so follow it with

    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/per_jets_estimation.ipynb
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
# Per-jet estimation — the residual to truth, splitting by splitting

The follow-up to §5 of [`inference_demo.ipynb`](inference_demo.ipynb), redone on the
**current** coordinates and the **extended** model: a $30\times30$ Lund-cell geometry, the
physical $\ln z$ support (`model.lnz_support="physical"`, so the head cannot leave the
soft-drop interval), and `ar_junipr_v4` with cross-attention and the nine groomed **aux**
conditioning columns of [`docs/PLAN_Input.md`](../docs/PLAN_Input.md).

It answers two questions, in this order.

1. **`showcase_jet(i)`** — everything the model says about *one* jet: the posterior cloud on
   the Lund plane, the multiplicity posterior, the MAP and MBR point estimates, the plain-RSD
   hadron sequence and the parton truth, the aux vector it was conditioned on, and the
   per-splitting residual ladder. One call, one figure, one printed tree table.

2. **The difference distribution** — $\Delta = \text{estimate} - \text{truth}$, one entry per
   *splitting*, pooled over jets, for the two Lund coordinates $\ln(1/\Delta R)$, $\ln k_t$
   and for $\ln z$. Plain RSD against the model, once over **all splittings** (§6) and once
   over the **first two** (§7). A model that is worth running has its residual narrower and
   more centred than the do-nothing baseline's.

Everything else — pooled marginals, the Lund-plane density, the $k_t$-cut spectrum, the
empty-tree rate — lives in
[`lund_distribution_closure_prod_test_v1.ipynb`](lund_distribution_closure_prod_test_v1.ipynb)
and is deliberately not repeated. This notebook runs on the same checkpoint, the same
held-out file and the same decode, so the two are directly comparable.

---

### What is differenced, and what it is aligned against

There is **no per-node $x\leftrightarrow y$ correspondence**: the hadron-level sequence and
the parton-level sequence are two separate declustering ladders, and nothing in the data
says which hadron splitting "is" which parton splitting. Every population number in
`eval/closure.py` is therefore alignment-*free* (multiplicity, the leading emission).

A residual needs an alignment, so this notebook uses **two**, and reports both rather than
picking one:

1. **Splitting index $t$** (§6–§8) — the position in the primary declustering sequence. Both
   ladders are built the same way and both march inward in angle ($\ln 1/\Delta R$ is
   non-decreasing in $t$ for ~97% of jets in this file), so "$t=0$" means the same thing —
   the widest-angle primary splitting that survived grooming — on both sides. This is the
   alignment `eval.closure.lund_tree_str(..., ref=)` already prints as its `dLund` column.
   It assumes the estimate got each splitting at the right *depth*.
2. **Kinematic matching** (§9) — depth-free: nodes matched one-to-one by proximity in the
   Lund plane (`scipy.optimize.linear_sum_assignment`). Uncapped it pairs exactly as many
   nodes as (1), so the two are a clean A/B on *which* nodes pair. Its own bias runs the
   other way: it minimises the distance it then reports, so it flatters whichever series
   brought the most nodes to choose from — here plain RSD, at ~30% more than truth. §9
   carries the unmatched ("spurious") rates beside every number for that reason.

Neither is the truth. A wide residual under (1) and a narrow one under (2) is the signature
of *right kinematics, wrong depth*; wide under both is genuinely wrong kinematics.

Two consequences follow for the index alignment, and both are measured rather than assumed:

- A residual exists at index $t$ only where **both** sides have a node there. A series that
  under-counts splittings contributes no pairs at large $t$, so the residual distribution is
  *conditioned on the pair existing*. §5 prints the pairing rate per series and §8 the pair
  count behind every number; §8a re-runs the whole table on the subset where all series are
  paired at once.
- Where the two ladders have different lengths, index alignment is a *choice*, not a
  measurement — the model may have got the same physical splitting right but at a different
  depth. That failure mode shows up here as a wide residual and in the closure notebook as a
  multiplicity bias; neither view alone separates them.

$\psi$ is deliberately absent from the difference panels. On the pinned `v1_contstop_s0`
arm the $\psi$ head's von Mises concentration has median $\kappa = 0.022$, and **99.9%** of
splittings sit below `decode.kappa_min_mode = 0.5` — the density is flat and its mode is
not an identified direction (Mardia & Jupp, *Directional Statistics*;
[`docs/PLAN_prod_test_v1.md`](../docs/PLAN_prod_test_v1.md) WP-C.2), so a $\Delta\psi$ panel
would mostly be plotting an arbitrary angle. `showcase_jet` reports it per node and prints
how many of them the decode had to draw rather than mode, which is the live check on that
claim for whatever checkpoint is loaded.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Parameters

**One knob: `RUN`.** Point it at a run directory and everything else is found inside —
the checkpoint, and the `prod_test_v1` artifact beside it that carries the held-out file
and the frozen empty-tree $\tau$. It also accepts an arm root, a `best.ckpt`, or an
artifact JSON directly, so any path you happen to have in hand works:

```python
RUN = "runs/prod_test_edit/e_v2_s0/20260802-004446-a824deac75"   # a run directory
RUN = "runs/prod_test_edit/e_v2_s0"                              # an arm root
RUN = "runs/prod_test_edit/e_v2_s0/20260802-004446-a824deac75/best.ckpt"
RUN = "runs/.../prod_test_v1/prod_test_v1_metrics.json"
RUN = None    # the newest prod_test_v1 artifact for ARM (the default, v1_contstop_s0)
```

The one thing that is **not** derived from the checkpoint is the evaluation file. A
checkpoint records the file it *trained* on (`data.path`), so taking the eval file from it
would silently turn a closure test into a report on the training set — it comes from the
artifact, or from `ROOT_PATH` when there is none. §3 asserts the two differ either way.

Everything below §0 is family-agnostic: it reads the geometry, the aux columns, the shape
decode and the length belief off whatever checkpoint is loaded, and §2/§3 print what they
found rather than assuming a family.
""")

code(r'''
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

# --- WHAT TO RUN: one knob ---------------------------------------------------
# A run directory, an arm root, a best.ckpt, or a prod_test_v1_metrics.json -- whichever
# path you have. Everything else is found inside it. None -> the newest prod_test_v1
# artifact for ARM below.
#   RUN = "runs/prod_test_edit/e_v2_s0/20260802-004446-a824deac75"
#   RUN = "runs/prod_test_edit/e_v2_s0"
RUN = None
# The arm docs/PROD_TEST_v1_RESULTS.md selected, and the one docs/PROD_TEST_edit_RESULTS.md
# §7 confirms still wins the head-to-head against both edit-transducer stages. Only read
# when RUN is None.
ARM = "v1_contstop_s0"
# The evaluation file. None -> from the artifact found beside the checkpoint. Set it only
# when there is none: a checkpoint records the file it TRAINED on, so this is the one thing
# that must never be derived from the checkpoint (see the assert in section 3).
ROOT_PATH   = None
NTUPLE_NAME = "Jets"

_REPO = _Path.cwd().parent if _Path.cwd().name == "notebooks" else _Path.cwd()


def _newest(root, pattern):
    """Newest match of `pattern` anywhere under `root`, or None."""
    hits = sorted(root.rglob(pattern), key=lambda q: q.stat().st_mtime) if root.is_dir() else []
    return hits[-1] if hits else None


_ck = _art = None
if RUN is None:
    # The arm's own directory FIRST, so mtime alone can never repoint this notebook at a
    # different checkpoint just because another notebook ran more recently.
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
        # a run dir has best.ckpt at its top; an ARM ROOT has it one level down, and there
        # the newest wins -- which is only well defined because a re-run writes a new stamp
        # directory rather than overwriting the old one.
        _ck = (_p / "best.ckpt") if (_p / "best.ckpt").exists() else (
            _newest(_p, "best.ckpt") or _newest(_p, "last.ckpt"))
        if _ck is None:
            raise FileNotFoundError(f"no best.ckpt or last.ckpt anywhere under {_p}")
        _art = _newest(_ck.parent, "prod_test_v1_metrics.json") or _newest(
            _p, "prod_test_v1_metrics.json")
    else:
        raise ValueError(
            f"RUN must be a directory, a .ckpt or a .json; got {_p.name!r}"
        )

_M = _json.loads(_art.read_text()) if _art is not None else None
if _ck is None:
    _ck = _REPO / _M["run"]["checkpoint"]
if not _ck.exists():
    raise FileNotFoundError(f"checkpoint does not exist: {_ck}")
# An artifact found NEXT TO a checkpoint but describing a different one is stale, and its
# tau and test file would be another run's. Fail rather than quietly mix two runs.
if _M is not None and (_REPO / _M["run"]["checkpoint"]).resolve() != _ck.resolve():
    raise RuntimeError(
        f"the artifact {_art} describes checkpoint {_M['run']['checkpoint']!r}, but the "
        f"checkpoint resolved from RUN is {_ck}. Its tau and test file belong to a "
        f"different run -- point RUN at one of them, not at a tree holding both."
    )

try:
    CKPT_PATH = str(_ck.resolve().relative_to(_REPO.resolve()))
except ValueError:
    CKPT_PATH = str(_ck)                       # a checkpoint outside the repo is legitimate
if ROOT_PATH is None:
    if _M is None:
        raise FileNotFoundError(
            f"no prod_test_v1_metrics.json under {_ck.parent}, so there is no record of "
            f"which file this checkpoint was EVALUATED on. Set ROOT_PATH explicitly -- it "
            f"is deliberately not taken from the checkpoint, which only knows the file it "
            f"was TRAINED on."
        )
    ROOT_PATH = _M["run"]["test_path"]
EMPTY_THRESHOLD = float(_M["empty_tree"]["tau"]["value"]) if _M is not None else 0.0

print(f"[run] checkpoint : {CKPT_PATH}")
print(f"[run] eval file  : {ROOT_PATH}")
if _M is not None:
    print(f"[run] artifact   : {_art.relative_to(_REPO)}\n"
          f"[run]              model={_M['run'].get('model')!r}  "
          f"frozen tau={EMPTY_THRESHOLD:.4f}")
else:
    print("[run] artifact   : none found -- EMPTY_THRESHOLD has no frozen value, so "
          "GATE_EMPTY must stay False")

# --- sample -----------------------------------------------------------------
PT_VAR  = "jet_pt"     # "jet_pt" (ungroomed) | "x_ptg" (groomed)
PT_MIN  = None         # half-open [PT_MIN, PT_MAX) GeV; both None -> every jet
PT_MAX  = None
N_JETS  = 2000         # jets in the evaluation pass (run the cost probe in 5a first)
SEED    = 1234
DEVICE  = "cpu"        # ~260k params decoded one jet at a time, so a GPU never amortises
#                        its dispatch overhead here. "auto"/"mps"/"cuda" work.
TORCH_THREADS = 4      # None -> torch's default (one per core). At batch 1 that can be
#                        slower; 4 is the measured choice for a sampling-dominated pass.

# --- decode -----------------------------------------------------------------
# K matches the artifact's, because for this family (use_multiplicity_head=false) the
# length belief P(n|x) IS the sampler histogram -- so K is what its resolution is.
K_DRAWS               = 200
LENGTH_FLOOR_QUANTILE = 0.15   # per-jet MAP floor at this quantile of P(n|x); 0.0 -> off
MBR_BACKEND           = "energyflow" if _ilu.find_spec("energyflow") else "pot"
MBR_N_CANDIDATES      = 16     # MBR candidate cap per jet (0 = all K draws)
# decode.empty_threshold: answer the EMPTY tree when q(N=0|x) >= tau, before any shape
# decode. OFF here, deliberately, and this is the one place this notebook departs from a
# production decode -- see the note in section 5. `GATE_EMPTY = True` applies the frozen
# tau printed above and the pairing table then prices it.
GATE_EMPTY            = False   # EMPTY_THRESHOLD is set above, from the artifact or 0.0

# --- the residual study -----------------------------------------------------
T_FIRST   = 2      # "the first N splittings" -- section 7 shows t = 0 .. T_FIRST-1
RESID_NB  = 41     # bins per residual panel; ODD so one bin is centred on zero
RESID_PCT = 99.0   # residual axes span +/- this percentile of |delta|, pooled
N_BOOT    = 200    # jet-level bootstrap resamples for the RMS-ratio CI
MIN_CI_JETS = 25   # below this many distinct jets in a slice, no CI is quoted
# --- the kinematic matching (section 9) -------------------------------------
# The cost the assignment minimises: the Lund plane, matching eval.closure.lund_distance.
# ln z is deliberately NOT in it -- a residual in a coordinate the matcher optimised is
# circular, and leaving ln z out makes its panel the one honest one in section 9.
MATCH_COST = ("lnInvDelta", "lnkt")
# None -> uncapped, which returns exactly min(n_truth, n_series) pairs, i.e. the SAME pair
# count as the own-depth index pairing -- that identity is what makes section 9 comparable
# to section 6, and it is asserted below rather than assumed. A float instead caps the match
# distance and drops the matched fraction sharply (~79% at 1.0, ~46% at 0.5 for plain RSD).
MATCH_RMAX = None
SHOWCASE_JET = None   # index for section 4; None -> auto-pick (see pick_showcase)

WRITE_ARTIFACTS = True   # per_jet_residuals.json beside the checkpoint

# --- guards -----------------------------------------------------------------
# The train/test guard lives in section 3, where the checkpoint's own `data.path` is
# available: that works with or without an artifact, unlike the artifact's train_path.
assert not (GATE_EMPTY and EMPTY_THRESHOLD <= 0.0), (
    "GATE_EMPTY=True needs a frozen tau, and none was read (CKPT_PATH/ROOT_PATH were set by "
    "hand). Point PROD_METRICS_PATH at the arm's artifact, or leave GATE_EMPTY False."
)
assert MBR_BACKEND != "surrogate", (
    "the surrogate is a different risk function, not a faster one -- never for a "
    "reported number"
)
''')

# ---------------------------------------------------------------------------
md(r"""
## 1. Imports, house style, helpers

Palette, `rcParams` and the histogram helpers are inherited verbatim from
[`lund_distribution_closure_v2.ipynb`](lund_distribution_closure_v2.ipynb) so its panels and
these overlay without re-reading a legend: truth is **ink**, plain RSD is a **grey fill**,
and the three model series take the first three slots of the validated categorical palette
(the only three that clear the all-pairs colour-vision gates).
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
# The kinematic matching of section 9. `linear_sum_assignment` is the globally optimal
# one-to-one assignment (Hungarian); inference.mbr is NOT a substitute -- it is fractional
# optimal transport with an imbalance penalty, which is a different object.
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
warnings.filterwarnings("ignore", category=UserWarning)

if TORCH_THREADS:
    torch.set_num_threads(int(TORCH_THREADS))

from h2p_rsd_junipr.config import decode_params
from h2p_rsd_junipr.data.datamodule import select_pt_range
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.data.rntuple import load_rntuple
from h2p_rsd_junipr.eval.closure import lund_tree_str
from h2p_rsd_junipr.eval.report import save_metrics
from h2p_rsd_junipr.features import AUX_FEATURES, node_raw
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import learned_min_emissions
from h2p_rsd_junipr.models.ar_junipr import ARJunipr
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.train.trainer import seed_everything, select_device

# --- style (inherited from lund_distribution_closure_v2.ipynb) ----------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
C_TRUTH = INK
C_RSD_F, C_RSD_E = "#e1e0d9", "#898781"
C_MAP   = "#2a78d6"    # MAP point estimate -- blue   (slot 1)
C_MBR   = "#eb6834"    # MBR point estimate -- orange (slot 2)
C_POST  = "#199e70"    # posterior draw     -- aqua   (slot 3), dashed

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
# The plain-text twin. `print` renders no mathtext, so a LaTeX label in a printed table is
# just backslashes -- every printed line below uses TLABEL and every axis uses LABEL.
TLABEL = {"lnInvDelta": "ln(1/dR)", "lnkt": "ln kt", "lnz": "ln z", "psi": "psi"}
COL = {"lnInvDelta": 0, "lnkt": 1, "lnz": 2, "psi": 3}   # node_raw column order
# The three coordinates this notebook differences: the two Lund-plane coordinates, and ln z.
RES_KEYS = ["lnInvDelta", "lnkt", "lnz"]


# --- weighted histogram / statistics helpers ---------------------------------
def h1_sumw2(values, weights, e):
    """Weighted counts and their Sumw2 errors -- ROOT's TH1::Sumw2 convention."""
    c = np.histogram(values, bins=e, weights=weights)[0]
    s2 = np.histogram(values, bins=e, weights=np.asarray(weights, float) ** 2)[0]
    return c, np.sqrt(s2)


def density(counts, err, e):
    """Unit-area normalisation with the error carried through it."""
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
    """Weighted quantiles by the (cumulative - half-weight) convention."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    if not v.size:
        return np.full(len(np.atleast_1d(qs)), np.nan)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, v)


def wstats(d, w):
    """Everything reported about one residual column, weight-aware.

    `rms` is about ZERO, not about the mean -- a residual's figure of merit is how far
    it sits from truth, and a large constant bias is a real failure, not something to
    subtract off. `bias` and `hw68` separate the two contributions.
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


print(f"style and helpers ready   (torch intra-op threads: {torch.get_num_threads()})")
''')

# ---------------------------------------------------------------------------
md(r"""
## 2. The model

Everything structural comes from the checkpoint's own config snapshot — geometry, encoder,
family, and the aux conditioning columns the encoder was *built* for. That last one is
load-bearing: the encoder's input width is fixed at build time, so a dataset built without
the aux columns does not warn, it mis-conditions (or raises a bare shape error at the first
decode).
""")

code(r'''
seed_everything(SEED)
device = select_device() if DEVICE == "auto" else torch.device(DEVICE)

CKPT = (REPO / CKPT_PATH) if not Path(CKPT_PATH).is_absolute() else Path(CKPT_PATH)
info = load_for_inference(str(CKPT), map_location=device)
cfg = OmegaConf.create(info["config"])
geom = Geometry.from_config(cfg.geometry)
model = build_model(cfg, geom).to(device)
model.load_state_dict(info["model_state"])
model.eval()

DECODE = decode_params(cfg)
# The shape-decode keys THIS family reads. `ar_junipr` publishes them as `_BEAM_KEYS`
# (it beam-searches); a family with a different shape decode -- `edit_*` takes the argmax
# of an exact q(N|x) and then a Viterbi alignment -- publishes none, and only the two
# length bounds are meaningful. Reading the list off the class rather than hard-coding
# ar_junipr's is what keeps the printed decode line honest for whichever is loaded.
BEAM = {k: DECODE[k]
        for k in getattr(type(model), "_BEAM_KEYS", ("max_emissions", "min_emissions"))
        if k in DECODE}
AUX = tuple(model.aux_feature_names)
CONT = bool(getattr(model, "has_continuous_coords", False))
TAU = float(EMPTY_THRESHOLD) if GATE_EMPTY else 0.0
LNZ_SUPPORT = str(OmegaConf.select(cfg, "model.lnz_support") or "legacy")

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
print(f"shape decode: {BEAM}   K={K_DRAWS}   MBR={MBR_BACKEND!r}"
      f"({MBR_N_CANDIDATES or 'all'})   length floor q={LENGTH_FLOOR_QUANTILE:g}")
print(f"empty gate : tau={TAU:g}" + ("" if GATE_EMPTY else
      f"   (OFF; frozen tau = {EMPTY_THRESHOLD:.4f} -- see section 5)"))
if not CONT:
    raise RuntimeError(
        "this checkpoint has no continuous coordinate density, so ln z and psi are "
        "placeholders (ln z = 0 means z = 1) and a ln z residual would be a plot of a "
        "filler constant. Point CKPT_PATH at a family with continuous_coords=true."
    )
''')

# ---------------------------------------------------------------------------
md(r"""
## 3. The test sample

The held-out PYTHIA file — a different generator seed from the one the checkpoint trained
on, with [`scripts/check_disjoint.py`](../scripts/check_disjoint.py)'s verdict recorded in
the artifact §0 read from.

The selection is `len(x) > 0` only: the **deployable** population, every jet an analysis
could pick out on data, including the ~17% whose parton truth is the empty tree. Requiring
`len(y) > 0` would read the answer. Those truth-empty jets contribute no residual (there is
nothing to difference against), but they stay in the sample so the pairing rates in §5 are
fractions of the real population rather than of a truth-selected one.
""")

code(r'''
# THE guard, and it is checked against the CHECKPOINT rather than the artifact: a
# checkpoint always records the file it trained on, so this holds on every route into
# section 0 -- including the one where there is no artifact to cross-check against.
TRAIN_PATH = str(OmegaConf.select(cfg, "data.path") or "")
assert str(ROOT_PATH) != TRAIN_PATH, (
    f"ROOT_PATH is {ROOT_PATH!r}, the file this checkpoint TRAINED on "
    f"(config snapshot data.path). That is not a closure test."
)

jets = load_rntuple(str(REPO / ROOT_PATH), NTUPLE_NAME)
jets = select_pt_range(jets, var=PT_VAR, lo=PT_MIN, hi=PT_MAX)

_n_in = len(jets)
jets = [j for j in jets if len(j["x"][0])]
if not jets:
    raise RuntimeError("no jets survived the selection")

try:
    ds = MatchedLundDataset(jets, geom, aux_features=AUX)
except Exception as exc:
    raise RuntimeError(
        f"the checkpoint was trained with aux inputs {AUX} but {ROOT_PATH} cannot supply "
        f"them ({exc}). Point ROOT_PATH at a file written with the aux columns."
    ) from exc

W_ALL = np.array([float(j.get("weight", 1.0)) for j in jets], dtype=float)
SD_KNOWN = "z_cut" in jets[0] and jets[0]["z_cut"] is not None
Z_CUT = float(jets[0]["z_cut"]) if SD_KNOWN else float("nan")
BETA = float(jets[0].get("beta", 0.0) or 0.0) if SD_KNOWN else float("nan")

_nx = np.array([len(j["x"][0]) for j in jets])
_ny = np.array([len(j["y"][0]) for j in jets])
print(f"source     : {ROOT_PATH}:{NTUPLE_NAME}   (trained on {TRAIN_PATH!r})")
print(f"generator  : {jets[0].get('generator', 'n/a')}")
if SD_KNOWN:
    print(f"grooming   : z_cut={Z_CUT:.3f}  beta={BETA:.3f}  "
          f"kt_floor={jets[0].get('kt_floor', float('nan')):.3f} GeV")
print(f"selection  : len(x)>0 keeps {len(jets):,} of {_n_in:,} jets")
print(f"multiplicity: hadron x = {_nx.mean():.3f}   parton y = {_ny.mean():.3f}   "
      f"x/y = {_nx.mean() / _ny.mean():.3f}")
print(f"             P(n_y = 0) = {np.mean(_ny == 0):.3f}   "
      f"(these jets have no truth splitting to difference against)")
print(f"evaluating : the first {min(N_JETS, len(ds)):,} of them")

# --- what P(n|x) MEANS for this checkpoint ----------------------------------
# The learned MAP floor and the empty gate both read `length_pmf`, and what that returns
# is family-dependent: an EXACT marginal for a family that has one (`edit_*`'s structural
# DP, a multiplicity head's softmax), or the histogram of the K draws for a family whose
# length belief IS its sampler. The difference matters -- a histogram has resolution 1/K
# and the floor quantile inherits it -- so it is PROBED rather than inferred from the
# config: feed the hook two different `mults` and see whether the answer moves.
_it0 = ds[0]
_xf0, _nx0 = _it0["xf"].unsqueeze(0).to(device), torch.tensor([_it0["nx"]], device=device)
with torch.inference_mode():
    _pa = np.asarray(model.length_pmf(_xf0, _nx0, mults=[0, 1]), dtype=float)
    _pb = np.asarray(model.length_pmf(_xf0, _nx0, mults=[4, 5]), dtype=float)
SAMPLER_PMF = not (_pa.shape == _pb.shape and np.allclose(_pa, _pb))
print()
if SAMPLER_PMF:
    print(f"P(n|x)     : the SAMPLER HISTOGRAM of the K={K_DRAWS} draws (resolution "
          f"1/{K_DRAWS}). The learned MAP\n             floor's quantile and any empty "
          f"gate inherit that resolution, and\n             decode.length_temperature / "
          f"length_tilt are inert -- tempering a histogram\n             without tempering "
          f"the sampler that made it would decouple the two.")
else:
    print("P(n|x)     : EXACT -- this family computes its own length marginal, so the "
          "learned MAP\n             floor and any empty gate read it directly rather than "
          "a K-sample estimate.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 4. One jet, end to end

`showcase_jet(i)` is the function this section exists to provide. It runs the full per-jet
inference — $K$ posterior draws, the MAP with its learned per-jet length floor, the MBR
(minimum expected perturbative-Lund EMD) point estimate, and one posterior draw with its
coordinates *sampled* rather than moded — then shows all of it against the parton truth and
the plain-RSD hadron sequence, and returns the record so it can be inspected further.

`estimate_jet(i)` below is the same computation with no plotting; §5 loops it over the
sample. The two share one implementation, so the single-jet figure and the population
figures cannot describe different decodes.
""")

code(r'''
def pe_coords(pe):
    """LundPointEstimate -> (n, 4) in node_raw column order."""
    if not pe.nodes:
        return np.zeros((0, 4))
    return np.array([[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes], dtype=float)


@torch.inference_mode()
def draw_coords(xf, nx, cells):
    """One posterior draw's CONTINUOUS coordinates, sampled rather than moded.

    `model.sample` returns cell chains only, and placing those at cell centres would
    leave ln z holding a filler constant -- so the contract hook `sample_coordinates` is
    what a posterior-predictive series has to go through.
    """
    if not len(cells):
        return np.zeros((0, 4))
    c = model.sample_coordinates(xf, nx, list(cells))
    return np.asarray(c.cpu().double().numpy(), dtype=float).reshape(-1, 4)


SERIES = ("truth", "rsd", "map", "mbr", "post")
MODELS = ("rsd", "map", "mbr", "post")   # everything differenced against truth
STYLE = {
    "truth": (C_TRUTH, "-",  r"truth $y$ (parton)"),
    "rsd":   (C_RSD_E, "-",  r"plain RSD $x$ (hadron)"),
    "map":   (C_MAP,   "-",  r"MAP $\hat y$"),
    "mbr":   (C_MBR,   "-",  r"MBR $\hat y$"),
    "post":  (C_POST,  "--", r"posterior draw"),
}
MARKER = {"truth": "o", "rsd": "x", "map": "*", "mbr": "D", "post": "s"}
MSIZE  = {"truth": 8.0, "rsd": 7.0, "map": 13.0, "mbr": 5.5, "post": 4.5}


@torch.inference_mode()
def estimate_jet(i, rng=None, k_draws=None, with_cloud=False):
    """Every series for jet `i`, as (n, 4) node_raw tables. No plotting, no printing.

    Returns the five series plus the raw posterior (multiplicities, and the full
    coordinate cloud when `with_cloud`), the point-estimate objects themselves, and the
    aux vector the encoder was conditioned on.
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    K = int(k_draws or K_DRAWS)
    item, jet = ds[i], jets[i]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)

    draws = model.sample(xf, nx, n=K)
    mults = np.array([len(d) for d in draws], dtype=int)

    # The learned per-jet floor: the alpha-quantile of the model's OWN length belief
    # P(n|x), reusing the draws above rather than sampling a second time.
    eff = learned_min_emissions(model, xf, nx, quantile=LENGTH_FLOOR_QUANTILE,
                                base_floor=1, mults=mults)
    dec = {**DECODE, **BEAM, "empty_threshold": TAU}
    mp = model.map_or_mbr(xf, nx, draws=draws,
                          **{**dec, "min_emissions": eff, "point_estimator": "map"})
    mbr = model.map_or_mbr(xf, nx, draws=draws,
                           **{**dec, "point_estimator": "mbr",
                              "mbr_backend": MBR_BACKEND,
                              "mbr_n_candidates": MBR_N_CANDIDATES})
    pick = draws[int(rng.integers(len(draws)))] if len(draws) else []

    rec = {
        "i": int(i), "weight": float(W_ALL[i]),
        "truth": np.asarray(item["yraw"].numpy(), dtype=float),
        "rsd": np.asarray(node_raw(*jet["x"]), dtype=float),
        "map": pe_coords(mp), "mbr": pe_coords(mbr),
        "post": draw_coords(xf, nx, pick),
        # q(N=0|x) through the contract hook, not `mean(mults == 0)`: for a family with an
        # exact length marginal that IS the exact number, and for one whose belief is the
        # sampler it reduces to the histogram -- so the reported value is always the same
        # quantity the empty gate would threshold.
        "mults": mults,
        "q0": float(model.length_pmf(xf, nx, mults=mults)[0]),
        "min_emissions": int(eff), "pe": {"map": mp, "mbr": mbr},
        "aux": {n: float(AUX_FEATURES[n](jet)) for n in AUX},
        "risk": float(mbr.risk) if mbr.risk is not None else float("nan"),
    }
    if with_cloud:
        cloud = model.sample_coordinates_many(xf, nx, [list(d) for d in draws if len(d)])
        rec["cloud"] = (np.concatenate([c.cpu().double().numpy().reshape(-1, 4)
                                        for c in cloud if c is not None])
                        if cloud else np.zeros((0, 4)))
    return rec


def pick_showcase(n_min=3, start=0):
    """First jet with at least `n_min` truth splittings -- a jet with a ladder to look at."""
    if SHOWCASE_JET is not None:
        return int(SHOWCASE_JET)
    for i in range(start, min(len(ds), 5000)):
        if int(ds[i]["ny"]) >= n_min:
            return i
    return int(np.argmax([int(ds[i]["ny"]) for i in range(min(len(ds), 5000))]))
''')

code(r'''
def _pad(a, n):
    """(n, 4) view of `a`, NaN-padded so a shorter series simply stops being drawn."""
    out = np.full((n, 4), np.nan)
    if len(a):
        out[:min(n, len(a))] = a[:n]
    return out


def showcase_jet(i=None, k_draws=None, show_trees=True, figsize=(13.4, 7.4)):
    """Everything the model says about ONE jet, beside the truth and plain RSD.

    Panels
      (a) the primary Lund plane: the posterior cloud (every node of every draw, with
          coordinates SAMPLED), the truth ladder, the plain-RSD ladder, and the MAP and
          MBR point estimates, each connected in declustering order.
      (b) the multiplicity posterior P(n|x), with truth / plain RSD / MAP / MBR marked.
      (c-e) the ladders themselves: ln(1/dR), ln kt and ln z against splitting index t,
          each over a strip of the per-splitting residual to truth -- the same quantity
          sections 6 and 7 pool over the whole sample.

    Returns the record from `estimate_jet` with a `resid` table added, so a caller can
    keep computing rather than re-running the inference.
    """
    i = pick_showcase() if i is None else int(i)
    rec = estimate_jet(i, k_draws=k_draws, with_cloud=True)
    y = rec["truth"]
    ny = len(y)
    depth = max(1, max(len(rec[s]) for s in SERIES))

    # per-splitting residuals for THIS jet, on the shared index alignment
    rec["resid"] = {s: (rec[s][:min(len(rec[s]), ny)] - y[:min(len(rec[s]), ny)])
                    for s in MODELS}

    # ---- printed header ------------------------------------------------------
    print(f"jet #{i}   weight {rec['weight']:.4g}   "
          f"P(n=0|x) = {rec['q0']:.3f}   MAP length floor = {rec['min_emissions']}")
    print(f"  multiplicity   truth y = {ny}   plain RSD x = {len(rec['rsd'])}   "
          f"MAP = {len(rec['map'])}   MBR = {len(rec['mbr'])}   "
          f"posterior = {rec['mults'].mean():.2f} +/- {rec['mults'].std():.2f} "
          f"(median {np.median(rec['mults']):.0f}, 68% CR "
          f"[{np.percentile(rec['mults'], 16):.0f}, {np.percentile(rec['mults'], 84):.0f}])")
    print(f"  MBR risk       {rec['risk']:.4f}   (mean expected Lund-EMD to the "
          f"posterior, {MBR_BACKEND!r} scale -- NOT a likelihood)")
    if rec["aux"]:
        print("  aux conditioning (per jet, broadcast onto every node of x):")
        _items = list(rec["aux"].items())
        for k in range(0, len(_items), 3):
            print("     " + "   ".join(f"{n:>13} = {v:+8.3f}" for n, v in _items[k:k + 3]))
    print("  mean |delta| to truth, over each series' own paired splittings:")
    print(f"     {'series':<10}{'pairs':>7}" + "".join(f"{TLABEL[k]:>12}" for k in RES_KEYS))
    for s in MODELS:
        r = rec["resid"][s]
        print(f"     {s:<10}{len(r):>7}" + "".join(
            f"{np.abs(r[:, COL[k]]).mean():>12.3f}" if len(r) else f"{'--':>12}"
            for k in RES_KEYS))

    # ---- figure --------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.30)
    axp = fig.add_subplot(outer[0, :2])
    axm = fig.add_subplot(outer[0, 2])

    # (a) the Lund plane
    cloud = rec["cloud"]
    if len(cloud):
        axp.scatter(cloud[:, 0], cloud[:, 1], s=7, color=C_POST, alpha=0.16,
                    linewidths=0, zorder=2,
                    label=f"posterior cloud ({len(cloud)} nodes / {len(rec['mults'])} draws)")
    for s in ("rsd", "map", "mbr", "truth"):
        v = rec[s]
        if not len(v):
            continue
        c, ls, lab = STYLE[s]
        axp.plot(v[:, 0], v[:, 1], ls=ls, color=c, lw=1.1, alpha=0.65, zorder=3)
        axp.scatter(v[:, 0], v[:, 1], marker=MARKER[s], s=MSIZE[s] ** 2 * 0.55,
                    facecolor="none" if s == "truth" else c, edgecolor=c,
                    linewidth=1.8 if s in ("truth", "rsd") else 0.7,
                    zorder=6 if s == "truth" else 5, label=lab)
    for t, (u, v) in enumerate(y[:, :2]):
        axp.annotate(f"{t}", (u, v), textcoords="offset points", xytext=(6, 5),
                     fontsize=7.5, color=INK_2)
    _all = np.concatenate([rec[s][:, :2] for s in SERIES if len(rec[s])]
                          + ([cloud[:, :2]] if len(cloud) else []))
    _pu = 0.10 * max(np.ptp(_all[:, 0]), 0.5)
    _pv = 0.10 * max(np.ptp(_all[:, 1]), 0.5)
    axp.set_xlim(max(geom.ln_invdelta_range[0], _all[:, 0].min() - _pu),
                 min(geom.ln_invdelta_range[1], _all[:, 0].max() + _pu))
    axp.set_ylim(max(geom.ln_kt_range[0], _all[:, 1].min() - _pv),
                 min(geom.ln_kt_range[1], _all[:, 1].max() + _pv))
    finish(axp, xlabel=LABEL["lnInvDelta"], ylabel=LABEL["lnkt"],
           title=f"(a) jet #{i} on the primary Lund plane   "
                 f"(numbers label the truth splitting index $t$)", legend=True, loc="best")

    # (b) the multiplicity posterior
    m = rec["mults"]
    hi = int(max(m.max(), ny, len(rec["rsd"]), len(rec["map"]), len(rec["mbr"]))) + 1
    axm.hist(m, bins=np.arange(-0.5, hi + 1.0), color=C_POST, alpha=0.5,
             edgecolor=C_POST, linewidth=0.8, label=r"posterior $P(n\,|\,x)$")
    for s, style in (("truth", (C_TRUTH, "-", 2.2)), ("rsd", (C_RSD_E, "-", 1.4)),
                     ("map", (C_MAP, "--", 1.6)), ("mbr", (C_MBR, ":", 1.8))):
        c, ls, lw = style
        axm.axvline(len(rec[s]), color=c, ls=ls, lw=lw,
                    label=f"{STYLE[s][2]} = {len(rec[s])}")
    finish(axm, xlabel="primary splittings $n$", ylabel="draws",
           title="(b) the length belief", legend=True, loc="upper right")

    # (c-e) the ladders, each over its residual strip
    for c_i, key in enumerate(RES_KEYS):
        inner = outer[1, c_i].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        rax = fig.add_subplot(inner[1], sharex=ax)
        ax.tick_params(labelbottom=False)
        col, ts = COL[key], np.arange(depth)
        for s in SERIES:
            v = _pad(rec[s], depth)[:, col]
            c, ls, lab = STYLE[s]
            ax.plot(ts, v, ls=ls, color=c, marker=MARKER[s], ms=MSIZE[s] * 0.6,
                    lw=2.0 if s == "truth" else 1.3,
                    mfc="none" if s == "truth" else c, label=lab,
                    zorder=5 if s == "truth" else 3)
            if s != "truth":
                rax.plot(ts, v - _pad(y, depth)[:, col], ls=ls, color=c,
                         marker=MARKER[s], ms=MSIZE[s] * 0.5, lw=1.1)
        rax.axhline(0.0, color=INK, lw=1.0)
        rax.set_ylabel(r"$-$ truth", fontsize=7.5)
        rax.set_xlabel("splitting index $t$")
        rax.set_xticks(ts)
        finish(ax, ylabel=LABEL[key], title=f"({'cde'[c_i]}) {LABEL[key]} ladder",
               legend=(c_i == 0), loc="best")

    fig.suptitle(f"Everything the model says about jet #{i}", x=0.006, y=1.005, ha="left")
    plt.show()

    if show_trees:
        # `dLund` in each row is the index-aligned Lund-plane distance to truth node t --
        # the same alignment sections 6 and 7 pool over.
        print()
        print(lund_tree_str(rec["pe"]["map"], "model MAP groomed shower", geom, ref=y))
        print()
        print(lund_tree_str(rec["pe"]["mbr"], "model MBR groomed shower (perturbative Lund)",
                            geom, ref=y))
        print()
        print(lund_tree_str(rec["rsd"], "plain RSD groomed shower (hadron-level x)",
                            geom, ref=y))
        print()
        print(lund_tree_str(y, "true groomed shower (parton-level y)", geom))
        if rec["pe"]["map"].n_psi_unidentified:
            print(f"\n* psi drawn, not moded, for "
                  f"{rec['pe']['map'].n_psi_unidentified} of "
                  f"{len(rec['map'])} MAP nodes: the von Mises concentration is below "
                  f"decode.kappa_min_mode={DECODE['kappa_min_mode']:g}, so the mode is "
                  f"not an identified direction.")
    return rec
''')

md(r"""
Call it on any jet index. `SHOWCASE_JET` in §0 pins one; `None` auto-picks the first jet with
at least three truth splittings, so there is a ladder to look at rather than a single point.
""")

code(r'''
seed_everything(SEED)
SHOW = showcase_jet()
''')

# ---------------------------------------------------------------------------
md(r"""
## 5. The evaluation pass, and the pairing it produces

One pass over `N_JETS`, `estimate_jet` per jet, then the residual table: one row per
$(\text{jet},\,t)$ at which both the truth and the series being differenced have a node.

**Two pairings, and which is used where.**

- **Own depth** — each series is paired against truth on $\min(n_\text{truth}, n_s)$,
  independently of the others. This keeps every splitting an estimator actually produced,
  and it is what §6 and §7 plot and what the descriptive columns of §8 report.
- **Common depth** — $\min$ over truth *and every* series, so all of them carry identical
  $(\text{jet}, t)$ rows. Fewer rows, but it is the only pairing on which "the model's
  residual is narrower than plain RSD's" is a statement about the estimators rather than
  partly about which splittings each one kept. §8's **RMS ratio and its bootstrap** are
  computed here, and §8a reports the full table on it.

Both are printed below. The gap between them is dominated by the **MAP**, whose beam-search
length is biased short — the joint mode of a high-entropy sequence posterior pays the split
head's entropy for every emission while "stop" costs roughly a constant, so the argmax
under-counts. On this sample that one series sets the common depth for all of them.

**Why the empty gate is off.** `decode.empty_threshold` decides *whether* a jet has any
parton splitting; this notebook measures *where* the splittings are. Applying the gate would
remove ~18% of jets from the paired sample for a reason unrelated to the question, and would
break the population match with
[`lund_distribution_closure_prod_test_v1.ipynb`](lund_distribution_closure_prod_test_v1.ipynb),
whose `eval_jets` is also ungated. Set `GATE_EMPTY = True` in §0 to apply the frozen $\tau$;
the pairing table below then prices it.
""")

md("### 5a. Cost probe — size the run before committing to it")

code(r'''
_probe = min(20, len(ds))
seed_everything(SEED)
_t0 = time.perf_counter()
_rng = np.random.default_rng(SEED)
for _i in range(_probe):
    estimate_jet(_i, rng=_rng)
_dt = (time.perf_counter() - _t0) / max(_probe, 1)
print(f"{_dt * 1e3:7.1f} ms / jet   (K={K_DRAWS}, MBR={MBR_BACKEND!r}, "
      f"candidates={MBR_N_CANDIDATES or 'all'}, {torch.get_num_threads()} threads)")
print(f"-> N_JETS={N_JETS} is about {_dt * N_JETS / 60:.1f} min")
''')

md("### 5b. Run it")

code(r'''
N = min(N_JETS, len(ds))
seed_everything(SEED)
_rng = np.random.default_rng(SEED)
_t0 = time.perf_counter()
RAW = {s: [] for s in SERIES}
W_JET, Q0 = [], []
for i in range(N):
    r = estimate_jet(i, rng=_rng)
    for s in SERIES:
        RAW[s].append(r[s])
    W_JET.append(r["weight"])
    Q0.append(r["q0"])
W_JET, Q0 = np.array(W_JET), np.array(Q0)
print(f"evaluated {N:,} jets in {(time.perf_counter() - _t0) / 60:.2f} min "
      f"(K={K_DRAWS} draws each)")

NSPL = {s: np.array([len(a) for a in RAW[s]]) for s in SERIES}
print()
print(f"{'series':<8}{'splittings':>12}{'mean mult':>12}{'P(n=0)':>10}")
for s in SERIES:
    print(f"{s:<8}{int(NSPL[s].sum()):>12,}{NSPL[s].mean():>12.3f}"
          f"{float(W_JET[NSPL[s] == 0].sum() / W_JET.sum()):>10.3f}")
''')

code(r'''
def pair_residuals(raw, w_jet, models=MODELS, common=True):
    """Index-aligned residuals: one row per (jet, splitting index t), per series.

    A residual exists at `t` only where both sides have a node there.

    `common=False` -- each series gets its own min(n_truth, n_s): every splitting the
                      estimator actually produced, at the cost that the series no longer
                      live on the same rows. Sections 6, 7 and the descriptive columns
                      of 8 use this.
    `common=True`  -- the depth kept for a jet is min over TRUTH and EVERY series, so
                      all series carry identical (jet, t) rows. The only pairing on which
                      a between-series ratio is a comparison; section 8's bootstrap and
                      section 8a use it.

    Either way the returned tables are keyed by series, so the two modes are the same
    data structure and nothing downstream has to know which one it holds.
    """
    n_true = np.array([len(a) for a in raw["truth"]], dtype=int)
    n_s = {s: np.array([len(a) for a in raw[s]], dtype=int) for s in models}
    if common:
        d_common = n_true.copy()
        for s in models:
            d_common = np.minimum(d_common, n_s[s])
        depth = {s: d_common for s in models}
    else:
        depth = {s: np.minimum(n_true, n_s[s]) for s in models}

    out = {k: {} for k in ("D", "T", "W", "J", "Y")}
    out["depth"] = depth
    out["n_true"] = n_true
    for s in models:
        D, T, W, J, Y = [], [], [], [], []
        for i, d in enumerate(depth[s]):
            d = int(d)
            if d <= 0:
                continue
            y = raw["truth"][i][:d]
            D.append(raw[s][i][:d] - y)
            T.append(np.arange(d))
            W.append(np.full(d, w_jet[i], dtype=float))
            J.append(np.full(d, i, dtype=int))
            Y.append(y)
        out["D"][s] = np.concatenate(D) if D else np.zeros((0, 4))
        out["T"][s] = np.concatenate(T) if T else np.zeros(0, dtype=int)
        out["W"][s] = np.concatenate(W) if W else np.zeros(0)
        out["J"][s] = np.concatenate(J) if J else np.zeros(0, dtype=int)
        out["Y"][s] = np.concatenate(Y) if Y else np.zeros((0, 4))
    if not any(len(out["T"][s]) for s in models):
        raise RuntimeError("no jet produced a paired splitting -- nothing to difference")

    # pairing rates, as fractions of the TRUTH splittings there were to recover
    tot = float(n_true.sum())
    out["pairing"] = {
        "common_depth": bool(common),
        "n_jets": int(len(n_true)),
        "n_jets_truth_nonempty": int((n_true > 0).sum()),
        "n_truth_splittings": int(tot),
        "n_jets_paired": {s: int((depth[s] > 0).sum()) for s in models},
        "n_paired": {s: int(depth[s].sum()) for s in models},
        "frac_paired": {s: (float(depth[s].sum() / tot) if tot else float("nan"))
                        for s in models},
        "frac_paired_own": {s: (float(np.minimum(n_true, n_s[s]).sum() / tot)
                                if tot else float("nan")) for s in models},
    }
    return out


# t-slice selectors. A selector maps a series' splitting-index array to a boolean mask,
# so the same slice can be applied to tables that do not share a row count.
def sel_all(t):
    return np.ones(len(t), dtype=bool)


def sel_eq(k):
    return lambda t: t == k


def sel_lt(k):
    return lambda t: t < k


def match_residuals(raw, w_jet, models=MODELS, cost=MATCH_COST, rmax=MATCH_RMAX):
    """Depth-free residuals: match nodes by proximity in the Lund plane, not by index.

    Index alignment assumes the estimate put each splitting at the right DEPTH; a tree
    that recovered the right physical splitting one rung up or down is scored as if it
    got the kinematics wrong. Here each jet's truth nodes and series nodes are matched
    one-to-one by `scipy.optimize.linear_sum_assignment` -- the globally optimal
    assignment, not a greedy nearest-neighbour sweep -- minimising total distance in
    `cost` (the Lund plane; ln z is deliberately outside it, see MATCH_COST).

    READ THE RESULT WITH ITS BIAS IN MIND. The assignment minimises the very distance
    the panels then report, so it flatters every series, and most the one with the most
    nodes to choose from -- which on this sample is plain RSD, carrying ~30% more
    splittings than truth. `frac_spurious` is what makes that visible: the fraction of a
    series' own nodes left with no truth partner, which the residual never penalises.

    Uncapped, `linear_sum_assignment` on a rectangular cost returns exactly
    min(n_truth, n_series) pairs -- the SAME count as own-depth index pairing, so the two
    views differ only in WHICH nodes are paired. `rmax` breaks that identity by dropping
    far-apart pairs, which is why it is off by default.

    Returns the same dict shape as `pair_residuals` (keyed by series), with `T` carrying
    the TRUTH node's index so section 7's t-slices keep meaning "the truth's t-th
    splitting" and every selector applies unchanged.
    """
    cols = [COL[k] for k in cost]
    n_true = np.array([len(a) for a in raw["truth"]], dtype=int)
    out = {k: {} for k in ("D", "T", "W", "J", "Y", "dist", "T_series")}
    matched, spurious, n_series = {}, {}, {}
    for s in models:
        D, T, W, J, Y, DIST, TS = [], [], [], [], [], [], []
        n_m = n_v = 0
        for i, (y, v) in enumerate(zip(raw["truth"], raw[s])):
            n_v += len(v)
            if not len(y) or not len(v):
                continue
            C = cdist(y[:, cols], v[:, cols])
            r, c = linear_sum_assignment(C)
            if rmax is not None:
                keep = C[r, c] <= float(rmax)
                r, c = r[keep], c[keep]
            if not len(r):
                continue
            n_m += len(r)
            D.append(raw[s][i][c] - y[r])
            T.append(r.astype(int))
            TS.append(c.astype(int))
            DIST.append(C[r, c])
            W.append(np.full(len(r), w_jet[i], dtype=float))
            J.append(np.full(len(r), i, dtype=int))
            Y.append(y[r])
        for key, buf, empty in (("D", D, np.zeros((0, 4))), ("T", T, np.zeros(0, int)),
                                ("W", W, np.zeros(0)), ("J", J, np.zeros(0, int)),
                                ("Y", Y, np.zeros((0, 4))), ("dist", DIST, np.zeros(0)),
                                ("T_series", TS, np.zeros(0, int))):
            out[key][s] = np.concatenate(buf) if buf else empty
        matched[s], spurious[s], n_series[s] = n_m, n_v - n_m, n_v
    tot = float(n_true.sum())
    out["n_true"] = n_true
    out["pairing"] = {
        "kinematic": True, "cost": list(cost), "rmax": rmax,
        "n_jets": int(len(n_true)),
        "n_truth_splittings": int(tot),
        "n_paired": {s: int(matched[s]) for s in models},
        # of the TRUTH splittings there were to recover...
        "frac_paired": {s: (matched[s] / tot if tot else float("nan")) for s in models},
        # ...and of the series' OWN nodes, how many found no truth partner at all
        "frac_spurious": {s: (spurious[s] / n_series[s] if n_series[s] else float("nan"))
                          for s in models},
        "mean_match_distance": {s: (float(np.average(out["dist"][s], weights=out["W"][s]))
                                    if len(out["dist"][s]) else float("nan"))
                                for s in models},
        "n_jets_paired": {s: int(len(np.unique(out["J"][s]))) for s in models},
    }
    return out


RES = pair_residuals(RAW, W_JET, common=False)         # headline: own depth per series
RES_COMMON = pair_residuals(RAW, W_JET, common=True)   # row-matched: ratios, section 8/8a
RES_MATCH = match_residuals(RAW, W_JET)                # depth-free, section 9
P, PC, PM = RES["pairing"], RES_COMMON["pairing"], RES_MATCH["pairing"]
_s0 = MODELS[0]

# The identity section 9 rests on, asserted rather than assumed: an UNCAPPED assignment
# pairs exactly as many nodes as own-depth index pairing, so the two views differ only in
# which nodes they pair. A cap breaks it, and then the comparison is against a smaller
# population and must say so.
if MATCH_RMAX is None:
    for s in MODELS:
        assert len(RES_MATCH["T"][s]) == len(RES["T"][s]), (
            f"uncapped matching gave {len(RES_MATCH['T'][s])} pairs for {s!r} against "
            f"{len(RES['T'][s])} from own-depth index pairing; they must agree"
        )

print(f"jets evaluated                    : {P['n_jets']:,}")
print(f"  with a non-empty parton truth   : {P['n_jets_truth_nonempty']:,} "
      f"({P['n_jets_truth_nonempty'] / P['n_jets']:.1%})")
print(f"truth splittings to recover       : {P['n_truth_splittings']:,}")
print()
print(f"{'series':<8}{'mean mult':>11}{'own-depth pairs':>18}{'of truth':>10}"
      f"{'jets w/ >=1 pair':>18}   |{'KINEMATIC pairs':>17}{'of truth':>10}"
      f"{'spurious':>10}{'<dist>':>9}")
print(f"{'truth':<8}{NSPL['truth'].mean():>11.3f}{'--':>18}{'--':>10}{'--':>18}"
      f"   |{'--':>17}{'--':>10}{'--':>10}{'--':>9}")
for s in MODELS:
    print(f"{s:<8}{NSPL[s].mean():>11.3f}{P['n_paired'][s]:>18,}"
          f"{P['frac_paired'][s]:>10.1%}{P['n_jets_paired'][s]:>18,}"
          f"   |{PM['n_paired'][s]:>17,}{PM['frac_paired'][s]:>10.1%}"
          f"{PM['frac_spurious'][s]:>10.1%}{PM['mean_match_distance'][s]:>9.3f}")
print(f"\ncommon depth (min over truth and ALL series, used for the ratios in section 8):"
      f"\n  {PC['n_paired'][_s0]:,} rows = {PC['frac_paired'][_s0]:.1%} of truth "
      f"splittings, on {PC['n_jets_paired'][_s0]:,} jets")
print(f"kinematic matching (section 9): cost={list(MATCH_COST)}, rmax={MATCH_RMAX}"
      + ("   -- same pair count as own depth, by the assert above"
         if MATCH_RMAX is None else "   -- CAPPED, so fewer pairs than own depth"))
print("  'spurious' is the fraction of a SERIES' OWN nodes with no truth partner. It is "
      "the rate\n  information index pairing conditions away, and the residual never "
      "penalises it -- read\n  section 9's RMS beside this column, never on its own.")
print()
for name, res in (("own depth   ", RES), ("common depth", RES_COMMON)):
    _T = res["T"][_s0]
    print(f"{name}: t = 0..{int(_T.max()) if len(_T) else 0}   "
          + "  ".join(f"t={t}: {int((res['T'][_s0] == t).sum()):,}" for t in range(T_FIRST))
          + f"   t>={T_FIRST}: {int((_T >= T_FIRST).sum()):,}")

# How much of the truth even LIVES beyond the first T_FIRST splittings. If this is small,
# sections 6 and 7 necessarily look alike, and that is a property of the sample rather
# than of the estimators -- so it is measured here rather than discovered from the figures.
_nt = RES["n_true"]
_deep = float(sum(int((_nt > t).sum()) for t in range(T_FIRST, int(_nt.max()) + 1))
              / max(_nt.sum(), 1))
print(f"\nof all truth splittings, {_deep:.1%} sit at t >= {T_FIRST} "
      f"(P(n_y<= {T_FIRST}) = {np.mean(_nt <= T_FIRST):.1%}) -- section 6 is therefore "
      f"largely\nsection 7 plus that tail, on THIS sample.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 6. The difference distribution — all splittings

$\Delta = \text{estimate} - \text{truth}$ for every paired splitting, unit-area normalised,
on axes shared with §7 so the two are read against each other without rescaling. The dotted
line at $\Delta = 0$ is perfect recovery; the legend carries each series' **bias**
($\langle\Delta\rangle$), **RMS** (about zero, not about the mean — a constant offset is a
real failure, not something to subtract), and the **68% half-width**, which separates the
two.

**Plain RSD is the number to beat.** Its residual is not noise: it is the hadronisation
correction itself, which is what the posterior exists to undo.
""")

code(r'''
def resid_edges(key, pct=RESID_PCT, nb=RESID_NB, res=None):
    """Symmetric residual axis, shared by every panel and slice for this coordinate.

    Spanned by a high percentile of |delta| POOLED over the compared series, so no one
    series sets the axis and sections 6 and 7 stack comparably. Entries beyond it fall
    outside the VIEW only -- every statistic below is computed on the unclipped rows.
    `nb` is a count of BINS and is odd, so one bin is centred on zero.
    """
    res = RES if res is None else res
    v = np.abs(np.concatenate([res["D"][s][:, COL[key]] for s in MODELS]))
    r = float(np.percentile(v, pct)) if v.size else 1.0
    r = max(math.ceil(r * 4.0) / 4.0, 0.25)
    return np.linspace(-r, r, int(nb) + 1)


RESID_EDGES = {k: resid_edges(k) for k in RES_KEYS}
for k in RES_KEYS:
    print(f"{TLABEL[k]:<10} residual axis +/-{RESID_EDGES[k][-1]:.2f}   "
          f"{RESID_NB} bins   ({RESID_PCT:g}th percentile of |delta|, pooled over "
          f"{', '.join(MODELS)})")


def slice_stats(key, sel=sel_all, series=MODELS, res=None):
    """`wstats` for one coordinate and one t-slice, per series."""
    res = RES if res is None else res
    out = {}
    for s in series:
        m = sel(res["T"][s])
        out[s] = wstats(res["D"][s][m, COL[key]], res["W"][s][m])
    return out


def resid_panel(ax, key, sel=sel_all, series=MODELS, title="", res=None):
    """One difference distribution: every series' delta for one coordinate.

    The per-series bias / RMS / 68% half-width go in the LEGEND rather than a corner
    annotation, so a panel is readable on its own; the y axis is given headroom for it
    instead of letting `loc="best"` drop the box onto the data.
    """
    res = RES if res is None else res
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
            step(ax, dens[s], e, c, label=lab, lw=1.8, ls=ls, z=4)
    ax.axvline(0.0, color=INK, lw=1.0, ls=":", zorder=6)
    ax.set_xlim(e[0], e[-1])
    ymax = max((float(d.max()) for d in dens.values() if d.size), default=0.0)
    if ymax > 0:
        ax.set_ylim(0.0, ymax * 1.78)      # headroom for the legend, not overlap
    finish(ax, xlabel=DLABEL[key], ylabel="density", title=title,
           legend=True, loc="upper left")
    return stats


fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
STATS_ALL = {}
for ax, key in zip(axes, RES_KEYS):
    n_rows = int(sel_all(RES["T"][MODELS[0]]).sum())
    STATS_ALL[key] = resid_panel(
        ax, key, sel_all, title=f"{DLABEL[key]}   all {n_rows:,} paired splittings")
fig.suptitle(r"estimate $-$ truth, per splitting, pooled over every splitting index",
             x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------------------
md(r"""
## 7. The difference distribution — the first two splittings

The same three coordinates, now sliced by splitting index: $t=0$, $t=1$, and the two pooled.

$t=0$ is the **widest-angle** primary splitting that survived grooming — and *usually*, but not
always, also the hardest. Declustering marches inward in angle, so $t=0$ is the widest for 98.4%
of multi-splitting truth trees; the hardest emission ($\max_t \ln k_t$) sits at $t=0$ for only
**80%** of them (74.6% for plain RSD), landing at $t=1$ in most of the rest. When the two differ
they are usually a near-tie: the $\ln k_t$ given up by quoting $t=0$ instead of the hardest has
median 0.000, mean 0.071 and p90 0.289 — a factor 1.34 in $k_t$ at p90.

So read $t=0$ as *the widest*, not as *the hardest*. The alignment-free observables of
`eval/closure.py` (`leading_emission_cell`) and the audit's `lnkt_lead` stratification use the
hardest instead, and the two are different observables that agree on four jets in five.

Every series should still be at its narrowest at $t=0$ — it is the most perturbative and the best
determined — and the gap between plain RSD and the model there is the cleanest single statement of
what the posterior buys. Disagreement that only appears at $t=1$ is the expected pattern (later
splittings are softer and the hadron→parton map is less constrained); disagreement already at
$t=0$ is not.
""")

code(r'''
# (plain label, mathtext label, selector) -- the plain one is for printed tables.
T_SLICES = [(f"t={t}", f"$t={t}$", sel_eq(t)) for t in range(T_FIRST)]
T_SLICES.append((f"t<{T_FIRST} (pooled)", rf"$t<{T_FIRST}$ (pooled)", sel_lt(T_FIRST)))

fig, axes = plt.subplots(len(T_SLICES), 3, figsize=(14.4, 4.0 * len(T_SLICES)))
axes = np.atleast_2d(axes)
STATS_T = {}
for r, (_lab, tlab, sel) in enumerate(T_SLICES):
    n_rows = int(sel(RES["T"][MODELS[0]]).sum())
    for c, key in enumerate(RES_KEYS):
        STATS_T[(_lab, key)] = resid_panel(
            axes[r, c], key, sel,
            title=f"{DLABEL[key]}   {tlab}   ({n_rows:,} splittings)")
fig.suptitle(rf"estimate $-$ truth for the first {T_FIRST} splittings",
             x=0.006, y=1.003, ha="left")
fig.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------------------
md(r"""
## 8. Summary — is the residual actually narrower than doing nothing?

The headline is the **RMS ratio** $\mathrm{RMS}(\hat y - y)\,/\,\mathrm{RMS}(x - y)$: below 1
means the estimate beat leaving the hadron-level jet alone. Its uncertainty is a **jet-level**
bootstrap — resampling jets, not splittings, because the splittings of one jet are
correlated and resampling them independently would understate the interval by roughly
$\sqrt{\langle n\rangle}$. A ratio whose interval brackets 1 is a null result and is
reported as one.

The descriptive columns come from the **own-depth** pairing (the population §6 and §7 plot);
the ratio column comes from the **common-depth** rows, where the model and plain RSD occupy
the same splittings and a ratio is therefore a comparison. §8a shows that subset in full so
the two can be read against each other.
""")

code(r'''
def _row_matched(key, sel, s, res):
    """`s` and `rsd` residuals on the rows they BOTH cover, plus weights and jet ids.

    For the common pairing that is every row, by construction. For the kinematic pairing
    the two series match different truth nodes, so the shared rows are the intersection on
    (jet, truth index) -- taken here rather than comparing two different populations and
    calling the quotient a ratio.
    """
    col = COL[key]
    m_s, m_r = sel(res["T"][s]), sel(res["T"]["rsd"])
    d_s, w, j, t = res["D"][s][m_s, col], res["W"][s][m_s], res["J"][s][m_s], res["T"][s][m_s]
    d_r, j_r, t_r = res["D"]["rsd"][m_r, col], res["J"]["rsd"][m_r], res["T"]["rsd"][m_r]
    if d_s.size == d_r.size and np.array_equal(j, j_r) and np.array_equal(t, t_r):
        return d_s, d_r, w, j                      # already row-for-row (common pairing)
    key_s = j.astype(np.int64) * 1000 + t
    key_r = j_r.astype(np.int64) * 1000 + t_r
    both = np.intersect1d(key_s, key_r)
    i_s = np.flatnonzero(np.isin(key_s, both))[np.argsort(key_s[np.isin(key_s, both)])]
    i_r = np.flatnonzero(np.isin(key_r, both))[np.argsort(key_r[np.isin(key_r, both)])]
    return d_s[i_s], d_r[i_r], w[i_s], j[i_s]


def boot_rms_ratio(key, sel, s, n_boot=N_BOOT, seed=SEED, res=None):
    """Jet-level bootstrap on RMS(s)/RMS(rsd) for one coordinate and t-slice.

    Computed on rows `s` and `rsd` BOTH cover -- a ratio between two different row sets is
    not a comparison. Resamples JETS, because the splittings of one jet are correlated and
    resampling them independently would understate the interval by roughly sqrt(<n>).
    """
    res = RES_COMMON if res is None else res
    d_s, d_r, w, j = _row_matched(key, sel, s, res)
    if not d_s.size:
        return float("nan"), float("nan"), float("nan"), 0

    def _ratio(idx):
        ww = w[idx]
        den = math.sqrt(float((ww * d_r[idx] ** 2).sum() / ww.sum()))
        num = math.sqrt(float((ww * d_s[idx] ** 2).sum() / ww.sum()))
        return num / den if den > 0 else float("nan")

    point = _ratio(np.arange(d_s.size))
    uj, jc = np.unique(j, return_inverse=True)
    if len(uj) < MIN_CI_JETS:      # too few jets for a bootstrap to mean anything
        return point, float("nan"), float("nan"), len(uj)
    rows = [np.flatnonzero(jc == k) for k in range(len(uj))]
    rng = np.random.default_rng(seed)
    vals = [_ratio(np.concatenate([rows[k] for k in
                                   rng.integers(0, len(rows), size=len(rows))]))
            for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [16, 84])
    return point, float(lo), float(hi), len(uj)


SLICES = [("all splittings", sel_all)] + [(lab, sel) for lab, _m, sel in T_SLICES]
TABLE = {}
for key in RES_KEYS:
    print(f"\n=== {TLABEL[key]}   delta = estimate - truth "
          + "=" * (48 - len(TLABEL[key])))
    print(f"{'slice':<18}{'series':<8}{'pairs':>8}{'bias':>9}{'RMS':>8}{'68% hw':>9}"
          f"{'median':>9}   {'RMS / plain RSD, row-matched  [68% CI]':<40}")
    for slab, sel in SLICES:
        for s in MODELS:
            m = sel(RES["T"][s])
            st = wstats(RES["D"][s][m, COL[key]], RES["W"][s][m])
            if s == "rsd":
                ratio, rec = "     1  (the baseline)", dict(**st, rms_ratio=1.0, ci=None)
            else:
                p, lo, hi, njet = boot_rms_ratio(key, sel, s)
                if not np.isfinite(p):
                    ratio, rec = "        --", dict(**st, rms_ratio=None, ci=None)
                elif not np.isfinite(lo):
                    ratio = f"{p:>6.3f}   (no CI: {njet} jets < MIN_CI_JETS)"
                    rec = dict(**st, rms_ratio=p, ci=None)
                else:
                    mark = "" if (lo - 1.0) * (hi - 1.0) > 0 else "   brackets 1"
                    ratio = f"{p:>6.3f}   [{lo:.3f}, {hi:.3f}]{mark}"
                    rec = dict(**st, rms_ratio=p, ci=[lo, hi])
            TABLE[(key, slab, s)] = rec
            print(f"{slab if s == MODELS[0] else '':<18}{s:<8}{st['n']:>8,}"
                  f"{st['bias']:>+9.3f}{st['rms']:>8.3f}{st['hw68']:>9.3f}"
                  f"{st['med']:>+9.3f}   {ratio:<40}")
print("\nThe first five columns are the OWN-depth pairing -- what each estimator did on")
print("every splitting it produced, and the population sections 6 and 7 plot. The ratio")
print("column is the COMMON-depth subset, the only rows on which plain RSD and the model")
print("occupy the same splittings; section 8a is that subset in full.")
print("\nRMS is about ZERO, not about each series' own mean: a constant offset is a real")
print("failure of a point estimate, not something to subtract off. `bias` and `68% hw`")
print("split the same total into its offset and its spread.")
''')

md(r"""
### 8a. The row-matched subset in full

The same table on the **common depth** — the rows where truth and every series have a node
at $t$, so all four are describing the same splittings. This is the subset the ratio column
above was computed on.

Compare it against §8 line by line. A series whose bias and RMS barely move between the two
is insensitive to which splittings survived; one that moves a lot is telling you its
residual depends on the depth it reached, and its ratio should be read with that in mind.
""")

code(r'''
for key in RES_KEYS:
    print(f"\n=== {TLABEL[key]}   common (row-matched) depth "
          + "=" * (48 - len(TLABEL[key])))
    print(f"{'slice':<18}{'series':<8}{'pairs':>8}{'bias':>9}{'RMS':>8}{'68% hw':>9}"
          f"{'median':>9}   {'RMS vs. the own-depth row above':<32}")
    for slab, sel in SLICES:
        for s in MODELS:
            m = sel(RES_COMMON["T"][s])
            st = wstats(RES_COMMON["D"][s][m, COL[key]], RES_COMMON["W"][s][m])
            base = TABLE[(key, slab, s)]["rms"]
            shift = (f"{st['rms'] / base:>6.3f}x" if np.isfinite(base) and base > 0
                     else "      --")
            print(f"{slab if s == MODELS[0] else '':<18}{s:<8}{st['n']:>8,}"
                  f"{st['bias']:>+9.3f}{st['rms']:>8.3f}{st['hw68']:>9.3f}"
                  f"{st['med']:>+9.3f}   {shift:<32}")
''')

# ---------------------------------------------------------------------------
md(r"""
## 9. Kinematic matching — the residual when depth is not assumed

Everything above pairs on **splitting index**, which assumes the estimate put each splitting
at the right *depth*. A tree that recovered the right physical splitting one rung up or down
is scored there as if it got the kinematics wrong. Here the pairing is depth-free: within
each jet, truth nodes and series nodes are matched **one-to-one by proximity in the Lund
plane** (`scipy.optimize.linear_sum_assignment` — the globally optimal assignment, not a
greedy sweep), and the residual is taken between matched partners.

Uncapped, that returns exactly $\min(n_\text{truth}, n_s)$ pairs — the **same count** as the
own-depth index pairing of §6, asserted in §5. So §9 against §6 is a clean A/B on *which*
nodes are paired, with nothing else moving.

**Read it with the bias it has.** The assignment minimises the very distance these panels
report, so it flatters every series — and most the one with the most nodes to choose from.
On this sample that is **plain RSD**, carrying ~30% more splittings than truth. Its
advantage is not free, and §5's `spurious` column is where it is paid: the fraction of a
series' own nodes left with no truth partner, which the residual never penalises. A series
can score well here by emitting many nodes and having the matcher keep the lucky ones.

Two things follow, and they are why this section is *additional* rather than a replacement:

- $\Delta\ln z$ is the **least circular panel**: `ln z` is deliberately not in the matching
  cost (`MATCH_COST`), so the matcher never optimised it. $\Delta\ln(1/\Delta R)$ and
  $\Delta\ln k_t$ are exactly what it minimised, and shrink most.
- The ratio column pairs each series against plain RSD on the truth nodes **both** matched,
  not on each one's own population — the intersection size is reported beside it.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
STATS_MATCH = {}
for ax, key in zip(axes, RES_KEYS):
    n_rows = int(sel_all(RES_MATCH["T"][MODELS[0]]).sum())
    STATS_MATCH[key] = resid_panel(
        ax, key, sel_all, res=RES_MATCH,
        title=f"{DLABEL[key]}   {n_rows:,} kinematically matched pairs")
fig.suptitle(r"estimate $-$ truth under KINEMATIC matching, all splittings   "
             r"(same axes as section 6)", x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(len(T_SLICES), 3, figsize=(14.4, 4.0 * len(T_SLICES)))
axes = np.atleast_2d(axes)
for r, (_lab, tlab, sel) in enumerate(T_SLICES):
    n_rows = int(sel(RES_MATCH["T"][MODELS[0]]).sum())
    for c, key in enumerate(RES_KEYS):
        resid_panel(axes[r, c], key, sel, res=RES_MATCH,
                    title=f"{DLABEL[key]}   {tlab}   ({n_rows:,} matched)")
fig.suptitle(rf"kinematic matching, the first {T_FIRST} splittings   "
             rf"($t$ indexes the TRUTH node, as in section 7)",
             x=0.006, y=1.003, ha="left")
fig.tight_layout()
plt.show()
''')

code(r'''
TABLE_MATCH = {}
for key in RES_KEYS:
    print(f"\n=== {TLABEL[key]}   KINEMATIC matching "
          + "=" * (55 - len(TLABEL[key])))
    print(f"{'slice':<18}{'series':<8}{'pairs':>8}{'bias':>9}{'RMS':>8}{'68% hw':>9}"
          f"{'median':>9}   {'vs. section 8 (index)':<22}{'RMS / plain RSD  [68% CI]':<38}")
    for slab, sel in SLICES:
        for s in MODELS:
            m = sel(RES_MATCH["T"][s])
            st = wstats(RES_MATCH["D"][s][m, COL[key]], RES_MATCH["W"][s][m])
            base = TABLE[(key, slab, s)]["rms"]
            shift = (f"{st['rms'] / base:>6.3f}x" if np.isfinite(base) and base > 0
                     else "      --")
            if s == "rsd":
                ratio, rec = "     1  (the baseline)", dict(**st, rms_ratio=1.0, ci=None)
            else:
                p, lo, hi, njet = boot_rms_ratio(key, sel, s, res=RES_MATCH)
                n_both = len(_row_matched(key, sel, s, RES_MATCH)[0])
                if not np.isfinite(p):
                    ratio, rec = "        --", dict(**st, rms_ratio=None, ci=None)
                elif not np.isfinite(lo):
                    ratio = f"{p:>6.3f}  (n={n_both}, no CI: {njet} jets)"
                    rec = dict(**st, rms_ratio=p, ci=None, n_both=n_both)
                else:
                    mark = "" if (lo - 1.0) * (hi - 1.0) > 0 else "  brackets 1"
                    ratio = f"{p:>6.3f}  [{lo:.3f}, {hi:.3f}] n={n_both}{mark}"
                    rec = dict(**st, rms_ratio=p, ci=[lo, hi], n_both=n_both)
            TABLE_MATCH[(key, slab, s)] = rec
            print(f"{slab if s == MODELS[0] else '':<18}{s:<8}{st['n']:>8,}"
                  f"{st['bias']:>+9.3f}{st['rms']:>8.3f}{st['hw68']:>9.3f}"
                  f"{st['med']:>+9.3f}   {shift:<22}{ratio:<38}")

print("\n'vs. section 8' is this RMS over the index-paired one. The TOTAL pair count is")
print("identical (the assert in section 5) -- but the per-t breakdown is NOT, because the")
print("assignment may leave a truth node unmatched and pair a later one instead, so a")
print("t-slice here is a slightly different population from the same slice in section 8.")
print("Below 1 everywhere is expected and is not a result: the matcher minimised these")
print("distances. What IS readable is which series gains most -- and that is whichever")
print("brought the most spare nodes to match with. Check section 5's 'spurious' column.")
print(f"\nln z is outside the matching cost ({list(MATCH_COST)}), so its panel is the one")
print("the assignment did not optimise.")
''')

code(r'''
if WRITE_ARTIFACTS:
    METRICS = {
        "run": {
            "notebook": "per_jets_estimation",
            "checkpoint": str(CKPT_PATH), "test_path": str(ROOT_PATH),
            "model": info["model_name"], "encoder": str(cfg.encoder.name),
            "aux_features": list(AUX), "lnz_support": LNZ_SUPPORT,
            "n_bins": geom.n_bins, "n_jets": int(N), "K_draws": int(K_DRAWS),
            "seed": int(SEED), "mbr_backend": MBR_BACKEND,
            "mbr_n_candidates": int(MBR_N_CANDIDATES),
            "length_floor_quantile": float(LENGTH_FLOOR_QUANTILE),
            "empty_gate": bool(GATE_EMPTY), "empty_threshold_applied": float(TAU),
            "empty_threshold_frozen": float(EMPTY_THRESHOLD),
        },
        "alignment": {
            "index_rule": "splitting index t; a residual exists only where truth and the "
                          "compared series have a node at t",
            "kinematic_rule": f"one-to-one linear_sum_assignment on {list(MATCH_COST)}, "
                              f"rmax={MATCH_RMAX}; MINIMISES the reported distance, so "
                              f"read it beside frac_spurious",
            "psi_excluded": "von Mises kappa below decode.kappa_min_mode for most "
                            "splittings -- the mode is not an identified direction",
            "own": P,
            "common": PC,
            "kinematic": PM,
            "stats_pairing": "own", "ratio_pairing": "common",
        },
        "residuals": {
            **{f"{key}|{slab}|{s}": v for (key, slab, s), v in TABLE.items()},
            **{f"{key}|{slab}|{s}|match": v for (key, slab, s), v in TABLE_MATCH.items()},
        },
    }
    out = save_metrics(METRICS, (REPO / CKPT_PATH).parent / "per_jet_residuals.json")
    print(f"wrote {out.relative_to(REPO)}")
else:
    print("WRITE_ARTIFACTS = False -- nothing written")
''')

# ---------------------------------------------------------------------------
md(r"""
---

### Reading these figures

- **The alignment is a choice, and it is the weakest link.** That is why there are two.
  §6–§8 index by depth and so score a right-splitting-at-the-wrong-rung as a kinematic
  error; §9 matches by kinematics and so cannot see depth at all, while flattering whichever
  series brought the most spare nodes. **Neither number stands alone.** Wide under index and
  narrow under matching ⇒ right kinematics, wrong depth. Wide under both ⇒ genuinely wrong
  kinematics. Narrow under matching *and* a high `spurious` rate in §5 ⇒ the series is
  buying its score with extra emissions the residual never charges it for.
- **§9 will show every series improving, and that is not a result.** The assignment
  minimises exactly the distance the panels then report. Only the *relative* movement is
  readable, and even that is confounded by node count. `ln z` is outside the matching cost,
  so its panel is the one the matcher did not optimise.
- **Conditioning on the pair existing is not neutral.** §5 prints, per series, what fraction
  of the truth splittings it managed to pair against. Where that fraction is well below 1 the
  residual is measured on the splittings the estimator *did* produce — the easy end of its
  own output — and the number is optimistic. The binding constraint on the common depth is
  the **MAP**, whose beam-search length is biased short; §6a of
  [`inference_demo.ipynb`](inference_demo.ipynb) and §6a of the closure notebook measure that
  bias directly. Dropping `map` from `MODELS` in §4 recovers most of the common depth if the
  question is only about MBR and the posterior.
- **§6 and §7 overlap by construction on this sample.** §5 prints the fraction of truth
  splittings living at $t\geq2$; where that is small, "all splittings" is "the first two"
  plus a thin tail, and the two figures *should* look alike. That is a property of a
  100 GeV groomed sample whose mean parton multiplicity is ~1.4, not a bug — on a harder
  sample, or with `T_FIRST` lowered, they separate.
- **The posterior-draw series is not a competitor.** Its residual should be *wider* than the
  point estimates' by construction: it carries the full posterior spread, while a point
  estimate is a summary of it. It is here as the scale reference — if a point estimate's
  residual is not comfortably narrower than a single draw's, the point estimate is not
  summarising anything.
- **RMS is about zero, deliberately.** A series with a large constant offset and a narrow
  spread is not a good estimator of the truth, and centring the RMS on its own mean would
  hide exactly that. The `bias` and `68% hw` columns split the total into those two parts.
- **$\ln z$ has a hard boundary.** With `lnz_support="physical"` the head is a truncated
  normal on the soft-drop interval $\ln z > \ln z_\mathrm{cut} - \beta \ln(1/\Delta R)$, so
  its residual is bounded by construction near the edge and the tails of
  $\Delta \ln z$ are not free to be symmetric. That is a feature — the `legacy` head could
  and did emit ungroomable $z$ — but it means the $\ln z$ panel is not comparable to one
  produced by a `legacy` checkpoint.

### Running it elsewhere

Set `CKPT_PATH` / `ROOT_PATH` in §0 to bypass the artifact read. The checkpoint must have a
continuous coordinate density (`has_continuous_coords`), which §2 asserts — without it
$\ln z$ is a filler constant and the third panel of every figure would be a plot of the
number 0.
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

out = Path(__file__).resolve().parent.parent / "notebooks" / "per_jets_estimation.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
