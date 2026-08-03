"""Build notebooks/per_jets_estimation_mode_mass_window.ipynb.

    python scripts/make_per_jets_window_nb.py

Same pattern and the same reason as scripts/make_per_jets_nb.py and
scripts/make_per_jets_mode_mass_nb.py: THIS FILE is the source of truth and an edit made
straight to the .ipynb is lost the next time anyone regenerates.

The third notebook of the per-jet family. `per_jets_estimation` measures the residual;
`per_jets_estimation_mode_mass` enumerates the top-k skeleton posterior and finds that
`M_1` is a readout of the cell grid. This one keeps ONLY the best skeleton and replaces
the cell with a SLIDING WINDOW, which is what makes the mode's claim resolution-labelled
and — the point of the notebook — testable: a region that holds mass `p` should contain
the truth a fraction `p` of the time.

§0 is deliberately the same RUN-resolution block as its two siblings, so all three
notebooks resolve a run directory to the same checkpoint and the same eval file.

Regenerating drops the executed outputs, so follow it with

    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/per_jets_estimation_mode_mass_window.ipynb
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
# The mode as a *region*: sliding-window skeletons, and whether the truth is in them

The third of the per-jet family, and the one that closes the loop.

- [`per_jets_estimation.ipynb`](per_jets_estimation.ipynb) measures the residual
  $\Delta = \text{estimate} - \text{truth}$, splitting by splitting.
- [`per_jets_estimation_mode_mass.ipynb`](per_jets_estimation_mode_mass.ipynb) enumerates the
  skeleton posterior exactly and finds that the mode's mass $M_1$ is largely a **readout of the
  $30\times30$ cell grid**: the coordinate head is truncation-saturated for ~92% of jets, the first
  splitting's 50% region spans ~18 cells, and $M_1 \propto r^2$ at the grid's own scale.
- **This one** keeps only the *best* skeleton and stops using cells to describe it. The mode
  becomes a **sliding box of half-width $r$**, its claim becomes a probability at a stated
  resolution, and that claim becomes **falsifiable**: does the truth land inside?

### Why a sliding window and not a coarser grid

Coarsening the cells into blocks trades one arbitrary discretization for a coarser one and keeps
its **origin** — a mode straddling a block boundary is split by the coarse grid exactly as it was
by the fine one. A window that slides has no phase, only a scale, and the scale is the one thing
here a physicist can actually choose: the non-perturbative smearing
$\sigma = \sigma_0 + \Lambda_{\rm eff}/k_t$, the width below which the coordinates cannot
concentrate at all.

So the object this notebook quotes is

$$M(r)\;=\;\max_{(u,v)}\ \int_{\lVert p - (u,v)\rVert_\infty < r} q_\phi(p \mid x)\ \mathrm{d}^2p ,$$

the largest mass in **any** box of half-width $r$, together with **the box itself**. Returning the
box is the whole difference: a cell-indexed mode can only be compared to the truth by exact
label match, which at 900 cells almost never happens and falls further with every refinement of
the grid. A box can be asked the question that matters — *is the truth in it?*

### The three things it measures

1. **§5 — the reliability diagram.** Per jet, the window's mass is a *prediction* and
   truth-in-window is the *outcome*. Binned by predicted mass, the observed frequency should lie
   on the diagonal. This is a coverage test of the mode's own neighbourhood, and it is the
   statement `per_jets_estimation_mode_mass` §7 was reaching for when it asked whether the mode is
   the truth — but grid-free, and answerable.
2. **§6 — correctness at a resolution.** $\mathrm{frac}(\text{truth} \in \text{box}(r))$ against
   the exact-cell-match number, side by side. The gap between them is the grid artefact in the
   *correctness* question, which is the same defect as in the dominance question and was left
   standing there.
3. **§7 — the residual of a region-based point estimate.** The **mass-weighted centroid inside
   the window** is a point estimate in its own right, so it goes into the same
   $\Delta = \text{estimate} - \text{truth}$ panels as plain RSD, MAP and MBR — beside the
   density's own **peak**, which is what "the most likely point" means and is measurably worse.

Everything is **exact**: the positional density is a block-wise mixture (each cell's truncated
normal is confined to its own cell), the window sums come from an integral image, and the
quadrature check is printed. Nothing here is sampled and nothing is a bound.

### Which splitting, exactly

**Node $t=0$ is the *widest-angle* splitting, not necessarily the hardest.** Declustering marches
inward in angle, so $t=0$ is the widest for 98.4% of multi-splitting truth trees — but the hardest
emission ($\max_t \ln k_t$) sits at $t=0$ for only **80%** of them (74.6% for plain RSD), landing at
$t=1$ in most of the rest. When they differ they are usually a near-tie: the $\ln k_t$ given up by
taking $t=0$ has median 0.000 and p90 0.289.

Those are two different observables, and this notebook measures **both**: the coverage test runs on
$t=0$ (the declustering-order node, which is what a teacher-forced prefix walks) *and* on the
hardest node of the truth, so the reader can see whether the calibration result survives the swap
rather than assume it. `eval/closure.py`'s `leading_emission_cell` and the audit's `lnkt_lead`
stratification both use the hardest; §7 of `per_jets_estimation.ipynb` slices on $t$, which is the
widest.

### What is deliberately *not* here

No top-$k$ enumeration, no $F(m)$ over ranks, no truth-rank table — those live in
[`per_jets_estimation_mode_mass.ipynb`](per_jets_estimation_mode_mass.ipynb) and are
same-geometry quantities. **Only the best skeleton**, and only through the window.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Parameters

**One knob: `RUN`.** The same block as the two sibling notebooks, so all three resolve a run
directory to the same checkpoint and the same held-out file. A run directory, an arm root, a
`best.ckpt` or a `prod_test_v1_metrics.json` all work.

The evaluation file is the one thing **not** derived from the checkpoint: a checkpoint records
the file it *trained* on, so taking the eval file from it would silently turn a closure test into
a report on the training set. §3 asserts the two differ.
""")

code(r'''
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

# --- WHAT TO RUN: one knob ---------------------------------------------------
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
        f"checkpoint resolved from RUN is {_ck}. Point RUN at one of them, not at a tree "
        f"holding both."
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

print(f"[run] checkpoint : {CKPT_PATH}")
print(f"[run] eval file  : {ROOT_PATH}")
if _M is not None:
    print(f"[run] artifact   : {_art.relative_to(_REPO)}   "
          f"model={_M['run'].get('model')!r}")

# --- sample -----------------------------------------------------------------
PT_VAR  = "jet_pt"
PT_MIN  = None
PT_MAX  = None
N_JETS  = 2000
SEED    = 1234
DEVICE  = "cpu"        # the density image is one batched pass per jet; a GPU never
#                        amortises its dispatch overhead at batch 1.
TORCH_THREADS = 4

# --- the sliding window -----------------------------------------------------
# The resolution grid, in ln units and deliberately NOT derived from the geometry: the
# whole point is a number that survives a change of n_bins.
RADII = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.70, 1.00)
# The headline resolution. 0.45 is where the median jet's window reaches half the mass
# (measured in per_jets_estimation_mode_mass section 6b) -- an even-odds region. It is a
# CHOICE, and the notebook reports the whole curve so the choice is visible.
R_STAR  = 0.45
SUB     = 7            # sub-pixels per cell per axis for the exact density image; the
#                        quadrature check in section 8 prices it (~2e-3 at 7).
N_REL_BINS = 8         # bins of the reliability diagram (section 5)
# Nodes beyond the first are conditioned on the mode's earlier cells, which is a real
# conditioning and not a marginal -- section 6 reports them separately for that reason.
MAX_NODES = 3

# --- reference series (the per_jets_estimation lineage) ---------------------
K_DRAWS               = 200
LENGTH_FLOOR_QUANTILE = 0.15
MBR_BACKEND           = "energyflow" if _ilu.find_spec("energyflow") else "pot"
MBR_N_CANDIDATES      = 16
WITH_REFERENCE        = True   # False -> window series only, and the pass is ~4x faster

# --- the residual study -----------------------------------------------------
RESID_NB  = 41
RESID_PCT = 99.0
N_BOOT    = 200
MIN_CI_JETS = 25
SHOWCASE_JET = None

WRITE_ARTIFACTS = True

assert 0.0 < R_STAR, "the headline resolution must be positive"
assert MBR_BACKEND != "surrogate", (
    "the surrogate is a different risk function, not a faster one -- never for a "
    "reported number"
)
''')

# ---------------------------------------------------------------------------
md(r"""
## 1. Imports, house style, helpers

Inherited verbatim from the sibling notebooks so the panels overlay without re-reading a legend:
truth is **ink**, plain RSD a **grey fill**, and the model series take the validated categorical
slots. The window series takes the fourth slot (magenta, `#b0499e`) — the same one the mode
skeleton carries in `per_jets_estimation_mode_mass`, because it *is* that estimate, read at a
resolution instead of at a cell.
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
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
warnings.filterwarnings("ignore", category=UserWarning)

if TORCH_THREADS:
    torch.set_num_threads(int(TORCH_THREADS))

from h2p_rsd_junipr.config import decode_params
from h2p_rsd_junipr.data.datamodule import select_pt_range
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.data.rntuple import load_rntuple
from h2p_rsd_junipr.eval.report import save_metrics
from h2p_rsd_junipr.features import node_raw
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import learned_min_emissions
from h2p_rsd_junipr.inference.mode_audit import (
    enumerate_skeletons,
    in_window,
    max_mass_window,
    node_density_image,
    window_centroid,
)
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.train.trainer import seed_everything, select_device

# --- style (inherited from per_jets_estimation.ipynb) -------------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
C_TRUTH = INK
C_RSD_F, C_RSD_E = "#e1e0d9", "#898781"
C_MAP   = "#2a78d6"    # MAP point estimate     -- blue    (slot 1)
C_MBR   = "#eb6834"    # MBR point estimate     -- orange  (slot 2)
C_POST  = "#199e70"    # posterior / density    -- aqua    (slot 3)
C_WIN   = "#b0499e"    # the WINDOW estimate    -- magenta (slot 4)

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
# A sequential ramp for the density panel: ONE hue, light -> dark, so magnitude reads as
# magnitude. Never a rainbow -- the eye reads hue as category, not as order.
CMAP_DENS = mpl.colors.LinearSegmentedColormap.from_list(
    "h2p_seq", ["#fcfcfb", "#cfe3f7", "#9ec5f4", "#5b9be8", "#2a78d6", "#184f95", "#0d366b"])
# The plane maps, inherited verbatim from lund_distribution_closure_v2.ipynb section 6 so
# the Lund panels of the three notebooks are readable against each other.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = mpl.colors.LinearSegmentedColormap.from_list("h2p_blue", SEQ_BLUE)
CMAP.set_bad(SURFACE)
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "h2p_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a3", "#d03b3b", "#7a1f1f"])
DIV.set_bad(SURFACE)

LABEL = {"lnInvDelta": r"$\ln(1/\Delta R)$", "lnkt": r"$\ln(k_t/\mathrm{GeV})$",
         "lnz": r"$\ln z$", "psi": r"$\psi$"}
DLABEL = {k: r"$\Delta$ " + v for k, v in LABEL.items()}
TLABEL = {"lnInvDelta": "ln(1/dR)", "lnkt": "ln kt", "lnz": "ln z", "psi": "psi"}
COL = {"lnInvDelta": 0, "lnkt": 1, "lnz": 2, "psi": 3}
RES_KEYS = ["lnInvDelta", "lnkt"]     # the window lives on the Lund PLANE; ln z is not
#                                       part of the region, so it is not differenced here.


def h1_sumw2(values, weights, e):
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
    ax.stairs(y, e, color=edge, fill=True, facecolor=face, linewidth=1.0, label=label,
              zorder=z)


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
    """`rms` is about ZERO, not about the mean: a constant offset is a real failure of a
    point estimate, not something to subtract off."""
    d, w = np.asarray(d, float), np.asarray(w, float)
    if not d.size:
        nan = float("nan")
        return dict(n=0, sumw=0.0, bias=nan, rms=nan, med=nan, hw68=nan)
    sw = float(w.sum())
    q16, q50, q84 = wquantile(d, w, [0.16, 0.50, 0.84])
    return dict(n=int(d.size), sumw=sw, bias=float((w * d).sum() / sw),
                rms=float(np.sqrt((w * d ** 2).sum() / sw)), med=float(q50),
                hw68=float(0.5 * (q84 - q16)))


def wilson(k, n, z=1.96):
    """Wilson score interval -- the honest error bar on every fraction below (Brown, Cai
    & DasGupta, *Statist. Sci.* 16 (2001) 101). The normal approximation is both too wide
    in the middle and nonsensical at the edges, which is exactly where a reliability
    diagram's end bins live."""
    n = int(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(k) / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


print(f"style and helpers ready   (torch intra-op threads: {torch.get_num_threads()})")
''')

# ---------------------------------------------------------------------------
md(r"""
## 2. The model, and the density this notebook is about

Everything structural comes from the checkpoint's own config snapshot. The one thing checked
explicitly is the **coordinate head**: without a continuous density there is no positional
posterior to slide a window over, and the notebook says so rather than sliding a window over
cell centres.

The density in question, for the first splitting, is

$$q_\phi(u, v \mid x,\ N\ge1) \;=\; \sum_c P_{\rm split}(c \mid h_0, e)\ \mathrm{TN}(u\mid c)\,\mathrm{TN}(v\mid c),$$

a mixture over all 900 cells whose components are each **confined to their own cell** (the head's
truncated normals are bounded by construction). That is what makes the image below exact rather
than sampled: a sub-grid inside every cell evaluates the whole thing, and the quadrature closes.
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
BEAM = {k: DECODE[k]
        for k in getattr(type(model), "_BEAM_KEYS", ("max_emissions", "min_emissions"))
        if k in DECODE}
AUX = tuple(model.aux_feature_names)

try:
    _ck = CKPT.resolve().relative_to(REPO.resolve())
except ValueError:
    _ck = CKPT
print(f"checkpoint : {_ck}")
print(f"model      : {info['model_name']}   encoder={cfg.encoder.name}")
print(f"geometry   : {geom.n_bins}x{geom.n_bins} = {geom.n_cells} cells   "
      f"cell {2 * geom.half_u:.3f} x {2 * geom.half_v:.3f} in ln units")
print(f"aux inputs : {len(AUX)}  {list(AUX)}")
print(f"parameters : {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k   "
      f"device={device}")
print(f"window     : radii {list(RADII)}   headline r = {R_STAR:g}   sub-grid {SUB}x{SUB}/cell")
print(f"             one pixel = {2 * geom.half_u / SUB:.4f} x {2 * geom.half_v / SUB:.4f} ln")
if not bool(getattr(model, "has_continuous_coords", False)):
    raise RuntimeError(
        "this checkpoint has no continuous coordinate density, so there is no positional "
        "posterior to slide a window over -- only cell centres, and a window over those "
        "would be measuring the grid again. Point RUN at a family with "
        "continuous_coords=true."
    )
''')

# ---------------------------------------------------------------------------
md(r"""
## 3. The test sample

The held-out PYTHIA file, selection `len(x) > 0` — the deployable population.

The coverage test of §5 runs on the jets whose parton truth has **at least one splitting**, and
that conditioning is not a convenience: the window's mass is
$P(\text{node }0 \in B \mid x,\ N\ge1)$, so the outcome it predicts is only defined where a node 0
exists. Jets with an empty truth are counted and reported, never silently dropped — whether the
jet has a splitting at all is the *multiplicity* question, and it is grid-free and answered by
$q(N\mid x)$ in the sibling notebook.
""")

code(r'''
TRAIN_PATH = str(OmegaConf.select(cfg, "data.path") or "")
assert str(ROOT_PATH) != TRAIN_PATH, (
    f"ROOT_PATH is {ROOT_PATH!r}, the file this checkpoint TRAINED on. Not a closure test."
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
        f"them ({exc})."
    ) from exc

W_ALL = np.array([float(j.get("weight", 1.0)) for j in jets], dtype=float)
_ny = np.array([len(j["y"][0]) for j in jets])
print(f"source      : {ROOT_PATH}:{NTUPLE_NAME}   (trained on {TRAIN_PATH!r})")
print(f"selection   : len(x)>0 keeps {len(jets):,} of {_n_in:,} jets")
print(f"P(n_y = 0)  : {np.mean(_ny == 0):.3f}   -- these have no node 0, so they carry no "
      f"coverage outcome\n              (see the length belief q(N|x) in "
      f"per_jets_estimation_mode_mass section 6a)")
print(f"evaluating  : the first {min(N_JETS, len(ds)):,}")
''')

# ---------------------------------------------------------------------------
md(r"""
## 4. One jet: the density, the window, and where the truth landed

`showcase_jet(i)` draws the thing the rest of the notebook counts. Panel (a) is the **exact**
positional density for the first splitting — not a scatter of draws — with the sliding window at
each radius drawn on top of it, the modal *cell* outlined for contrast, and the truth marked.

The gap between the cell outline and the window is the whole argument of this notebook family in
one picture: same posterior, two ways of saying where it is concentrated.
""")

code(r'''
@torch.inference_mode()
def jet_windows(i, radii=RADII, max_nodes=MAX_NODES):
    """The best skeleton, and the sliding window on each of its first `max_nodes` nodes.

    Node 0's density is unconditional given `x` (and `N >= 1`). Later nodes are
    teacher-forced on the MODE's earlier cells, so their masses are conditional on that
    prefix being right -- reported, and labelled, separately.
    """
    item, jet = ds[i], jets[i]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)
    spec = model.skeleton_search_spec(xf, nx)
    # ONLY the best skeleton: k=1 stops the search at the first completion.
    enum = enumerate_skeletons(model, xf, nx, k=1, budget=20000, spec=spec)
    cells = list(enum.top1_cells)

    y = np.asarray(item["yraw"].numpy(), dtype=float)
    nodes = []
    # node 0 always exists as a QUESTION ("where is the first splitting, given there is
    # one"), whether or not the mode skeleton chose to have one -- so the prefix chain
    # for the window is the mode's, and the depth is capped independently of its length.
    depth = min(max_nodes, max(1, len(cells)))
    for t in range(depth):
        grid = node_density_image(model, xf, nx, spec=spec, sub=SUB,
                                  prefix=[int(c) for c in cells[:t]])
        img, meta = grid
        wins = {float(r): max_mass_window(grid, float(r)) for r in radii}
        truth = y[t, :2] if t < len(y) else None
        # ...and the truth's HARDEST node, which is a different observable: for node 0's
        # window the two questions are "is the widest-angle truth in the box" and "is the
        # hardest truth in the box", and they are the same question for ~80% of jets.
        hard = (y[int(np.argmax(y[:, 1])), :2] if len(y) else None)
        nodes.append({
            "t": t,
            "windows": wins,
            # the point to quote is the mass-weighted centroid INSIDE the window; the
            # other two are kept so section 7 can price the choice rather than assert it
            "centroid": {r: list(window_centroid(grid, w)) for r, w in wins.items()},
            "box_centre": {r: list(box_centre(w)) for r, w in wins.items()},
            "peak": [float(x) for x in density_peak(grid)],
            "truth": (None if truth is None else [float(truth[0]), float(truth[1])]),
            "hit": {r: (None if truth is None else in_window(truth, w))
                    for r, w in wins.items()},
            "hit_hardest": {r: (None if hard is None else in_window(hard, w))
                            for r, w in wins.items()},
            "truth_hardest": (None if hard is None else
                              [float(hard[0]), float(hard[1])]),
            "truth_hardest_index": (None if not len(y) else int(np.argmax(y[:, 1]))),
            "quadrature": float((img * meta["pixel_area"]).sum()),
            "conditional_on": [int(c) for c in cells[:t]],
            "modal_cell": int(meta["modal_cell"]),
            "sigma_u": float(meta["du_sig"][meta["modal_cell"]]),
            "sigma_v": float(meta["dv_sig"][meta["modal_cell"]]),
            "_grid": grid,
        })
    return {"i": int(i), "cells": cells, "M1_fine": float(enum.m1),
            "certified_top1": bool(enum.certified_top1),
            "n_truth": int(item["ny"]), "truth": y, "nodes": nodes,
            "weight": float(W_ALL[i]),
            "rsd": np.asarray(node_raw(*jet["x"]), dtype=float)}


def box_centre(w):
    """The box's geometric centre. NOT the estimate this notebook quotes -- the window
    placement is only determined up to the pixel lattice, so its midpoint puts a grid
    back into the answer after the sliding window took one out."""
    return np.array([0.5 * (w["u_lo"] + w["u_hi"]), 0.5 * (w["v_lo"] + w["v_hi"])])


def density_peak(grid):
    """The positional posterior's own maximum -- what "the most likely point" means. In
    section 7 it is the worst of the three, which is the mode-vs-mean story at the
    coordinate level rather than the skeleton level."""
    img, meta = grid
    a = img.cpu().double().numpy()
    iu, iv = np.unravel_index(int(np.argmax(a)), a.shape)
    return np.array([meta["origin_u"] + (iu + 0.5) * meta["step_u"],
                     meta["origin_v"] + (iv + 0.5) * meta["step_v"]])


def pick_showcase(n_min=2, start=0):
    if SHOWCASE_JET is not None:
        return int(SHOWCASE_JET)
    for i in range(start, min(len(ds), 5000)):
        if int(ds[i]["ny"]) >= n_min:
            return i
    return 0
''')

code(r'''
def showcase_jet(i=None, figsize=(13.4, 4.6)):
    """The exact positional density for the first splitting, with the windows on it."""
    i = pick_showcase() if i is None else int(i)
    rec = jet_windows(i)
    node = rec["nodes"][0]
    img, meta = node["_grid"]
    dens = img.cpu().double().numpy()

    print(f"jet #{i}   best skeleton: "
          + (" -> ".join(str(c) for c in rec["cells"]) if rec["cells"] else "(empty)")
          + f"   its exact FINE mass M_1 = {rec['M1_fine']:.4f}"
          + ("" if rec["certified_top1"] else "   (top-1 not certified)"))
    print(f"  truth has {rec['n_truth']} splitting(s); plain RSD has {len(rec['rsd'])}")
    print(f"  quadrature check on the density: {node['quadrature']:.6f}  (target 1)")
    print()
    print(f"  the FIRST splitting, as a region -- exact, and unconditional given N>=1:")
    print(f"     {'r':>6}{'window mass':>14}{'box in ln(1/dR)':>22}{'box in ln kt':>20}"
          f"{'truth in it':>13}")
    for r in RADII:
        w = node["windows"][float(r)]
        h = node["hit"][float(r)]
        box_u = "[%.2f, %.2f]" % (w["u_lo"], w["u_hi"])
        box_v = "[%.2f, %.2f]" % (w["v_lo"], w["v_hi"])
        verdict = "--" if h is None else ("YES" if h else "no")
        print(f"     {r:>6.2f}{w['mass']:>14.4f}{box_u:>22}{box_v:>20}{verdict:>13}")
    print(f"  the head's own widths at the modal cell: sigma_u = {node['sigma_u']:.3f}, "
          f"sigma_v = {node['sigma_v']:.3f}   (half-cell {geom.half_u:.3f})")
    for nd in rec["nodes"][1:]:
        w = nd["windows"][float(R_STAR)]
        print(f"  node {nd['t']} at r = {R_STAR:g} -- CONDITIONAL on the mode's "
              f"{nd['conditional_on']}: mass {w['mass']:.3f}, "
              f"truth in it: {'--' if nd['hit'][float(R_STAR)] is None else nd['hit'][float(R_STAR)]}")

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    lo_u, hi_u = geom.ln_invdelta_range
    lo_v, hi_v = geom.ln_kt_range

    # (a) the exact density, with the windows and the modal CELL on it
    im = axes[0].imshow(dens.T, origin="lower", extent=[lo_u, hi_u, lo_v, hi_v],
                        cmap=CMAP_DENS, aspect="auto", interpolation="nearest", zorder=1)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02, label="density")
    ix, iy = divmod(node["modal_cell"], geom.n_bins)
    axes[0].add_patch(Rectangle((lo_u + ix * 2 * geom.half_u, lo_v + iy * 2 * geom.half_v),
                                2 * geom.half_u, 2 * geom.half_v, fill=False,
                                edgecolor=C_MBR, lw=1.6, zorder=4,
                                label=f"modal CELL (mass {rec['M1_fine']:.3f})"))
    for r, alpha in zip((R_STAR, RADII[-1]), (1.0, 0.55)):
        w = node["windows"][float(r)]
        axes[0].add_patch(Rectangle((w["u_lo"], w["v_lo"]), w["u_hi"] - w["u_lo"],
                                    w["v_hi"] - w["v_lo"], fill=False, edgecolor=C_WIN,
                                    lw=2.0, alpha=alpha, zorder=5,
                                    label=f"window r={r:g} (mass {w['mass']:.3f})"))
    if node["truth"] is not None:
        axes[0].scatter([node["truth"][0]], [node["truth"][1]], marker="o", s=90,
                        facecolor="none", edgecolor=C_TRUTH, linewidth=2.0, zorder=6,
                        label="truth, node 0")
    if len(rec["rsd"]):
        axes[0].scatter([rec["rsd"][0, 0]], [rec["rsd"][0, 1]], marker="x", s=60,
                        color=C_RSD_E, linewidth=1.8, zorder=6, label="plain RSD, node 0")
    finish(axes[0], xlabel=LABEL["lnInvDelta"], ylabel=LABEL["lnkt"],
           title=f"(a) jet #{i}: the EXACT positional posterior of the first splitting",
           legend=True, loc="upper right")

    # (b) this jet's M(r)
    ms = [node["windows"][float(r)]["mass"] for r in RADII]
    axes[1].plot(RADII, ms, color=C_WIN, lw=2.0, marker="P", ms=5, label=r"$M(r)$")
    axes[1].axvline(R_STAR, color=INK, ls=":", lw=1.2, label=f"$r = {R_STAR:g}$")
    axes[1].axvline(geom.half_u, color=C_MBR, ls=":", lw=1.2,
                    label=f"half a cell = {geom.half_u:.2f}")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.02)
    finish(axes[1], xlabel=r"$r$ [ln units]", ylabel="mass in the best box",
           title="(b) this jet's mass-vs-resolution", legend=True, loc="upper left")

    # (c) a zoom of (a) around the window, where the structure actually is
    w = node["windows"][float(R_STAR)]
    pad = 2.5 * R_STAR
    axes[2].imshow(dens.T, origin="lower", extent=[lo_u, hi_u, lo_v, hi_v],
                   cmap=CMAP_DENS, aspect="auto", interpolation="nearest", zorder=1)
    axes[2].add_patch(Rectangle((w["u_lo"], w["v_lo"]), w["u_hi"] - w["u_lo"],
                                w["v_hi"] - w["v_lo"], fill=False, edgecolor=C_WIN,
                                lw=2.0, zorder=5))
    if node["truth"] is not None:
        axes[2].scatter([node["truth"][0]], [node["truth"][1]], marker="o", s=90,
                        facecolor="none", edgecolor=C_TRUTH, linewidth=2.0, zorder=6)
    axes[2].set_xlim(max(lo_u, w["u_lo"] - pad), min(hi_u, w["u_hi"] + pad))
    axes[2].set_ylim(max(lo_v, w["v_lo"] - pad), min(hi_v, w["v_hi"] + pad))
    finish(axes[2], xlabel=LABEL["lnInvDelta"], ylabel=LABEL["lnkt"],
           title=f"(c) zoom on the $r={R_STAR:g}$ window")

    fig.suptitle(f"jet #{i}: the mode as a REGION, and whether the truth is in it",
                 x=0.006, y=1.02, ha="left")
    fig.tight_layout()
    plt.show()
    return rec
''')

code(r'''
seed_everything(SEED)
SHOW = showcase_jet()
''')

# ---------------------------------------------------------------------------
md(r"""
## 5. The pass, and the reliability diagram

One pass over `N_JETS`. Per jet the window at each radius gives a **prediction** — the mass it
claims — and the truth gives an **outcome** — whether it is inside. Nothing else is needed for the
central plot of this notebook.

A reliability diagram bins jets by the predicted mass and compares it with the observed frequency
in each bin. On the diagonal, the model's regions mean what they say. Below it, the regions are
over-confident (they claim more mass than they contain); above it, conservative.

This is a **coverage** statement, and it is worth being clear about what it is not: it does not say
the mode is *right*, only that the posterior's own width around the mode is honest. A model can be
perfectly calibrated and still put its regions in the wrong place — which is why §7's residual is
here too.

**One conditioning to keep straight.** The window's mass is
$P(\text{node }0 \in B \mid x,\ N\ge1)$ under the *model*, while the outcome is measured on the jets
whose *truth* has a node 0. Those are two different conditions, and the comparison is only clean to
the extent the model's length belief is calibrated — which on this checkpoint it is, to a striking
degree: `per_jets_estimation_mode_mass` §6a measures $q(N{=}0\mid x) = 0.172$ against a truth empty
rate of $0.171$. Any residual mismatch here is therefore a *length* effect, not a positional one,
and the length question has its own grid-free answer in that notebook.
""")

code(r'''
N = min(N_JETS, len(ds))
seed_everything(SEED)
_t0 = time.perf_counter()

REC = []
DENS_SUM = None          # sum_j q_j(u,v) for node 0 -- the MEAN posterior over jets, which
DENS_META = None         # section 6a compares against where the truth's node 0 actually is
for i in range(N):
    REC.append(jet_windows(i))
    _img, DENS_META = REC[-1]["nodes"][0]["_grid"]
    _a = _img.cpu().double().numpy()
    DENS_SUM = _a.copy() if DENS_SUM is None else DENS_SUM + _a
    REC[-1]["nodes"] = [{k: v for k, v in nd.items() if k != "_grid"}
                        for nd in REC[-1]["nodes"]]      # drop the tensors, keep the numbers
DENS_MEAN = DENS_SUM / max(N, 1)
print(f"windowed {N:,} jets in {(time.perf_counter() - _t0) / 60:.2f} min")

HAS_T0 = np.array([r["n_truth"] >= 1 for r in REC], dtype=bool)
MASS = {float(r): np.array([x["nodes"][0]["windows"][float(r)]["mass"] for x in REC])
        for r in RADII}
HIT = {float(r): np.array([bool(x["nodes"][0]["hit"][float(r)]) if x["nodes"][0]["hit"][float(r)]
                           is not None else False for x in REC]) for r in RADII}
# the same window, asked about the truth's HARDEST node instead of its widest-angle one
HIT_HARD = {float(r): np.array([bool(x["nodes"][0]["hit_hardest"][float(r)])
                                if x["nodes"][0]["hit_hardest"][float(r)] is not None
                                else False for x in REC]) for r in RADII}
FIRST_IS_HARDEST = np.array([x["nodes"][0].get("truth_hardest_index") == 0
                             for x in REC], dtype=bool)
QUAD = np.array([x["nodes"][0]["quadrature"] for x in REC])
M1_FINE = np.array([x["M1_fine"] for x in REC])
CELLS_EMPTY = np.array([not x["cells"] for x in REC], dtype=bool)

print(f"  jets with a truth node 0 (the coverage population): {HAS_T0.sum():,} "
      f"({HAS_T0.mean():.1%})")
print(f"  the best skeleton is the EMPTY tree for {CELLS_EMPTY.mean():.1%} of jets -- their "
      f"node-0 window is\n  still well defined ('where would the first splitting be, given "
      f"there is one'), so they stay in.")
print(f"  quadrature on the density: mean {QUAD.mean():.6f}, worst "
      f"{np.abs(QUAD - 1).max():.2e} from 1")
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
sel = HAS_T0

# (a) THE reliability diagram, pooled over every radius: predicted mass vs observed hit
_pred = np.concatenate([MASS[float(r)][sel] for r in RADII])
_obs = np.concatenate([HIT[float(r)][sel] for r in RADII])
edges = np.linspace(0.0, 1.0, N_REL_BINS + 1)
idx = np.clip(np.digitize(_pred, edges) - 1, 0, N_REL_BINS - 1)
xs, ys, los, his, ns = [], [], [], [], []
for b in range(N_REL_BINS):
    m = idx == b
    if m.sum() < 10:
        continue
    k, n = int(_obs[m].sum()), int(m.sum())
    lo, hi = wilson(k, n)
    xs.append(float(_pred[m].mean()))
    ys.append(k / n)
    los.append(lo)
    his.append(hi)
    ns.append(n)
axes[0].plot([0, 1], [0, 1], color=INK, ls=":", lw=1.4, label="calibrated")
axes[0].errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                 fmt="o", color=C_WIN, ms=5, lw=1.4, capsize=3,
                 label="observed (95% Wilson)")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
finish(axes[0], xlabel="mass the window claims", ylabel="fraction containing the truth",
       title="(a) do the regions mean what they say?", legend=True, loc="upper left")

# (b) coverage and claimed mass, both against r
_cov = [HIT[float(r)][sel].mean() for r in RADII]
_cl = [MASS[float(r)][sel].mean() for r in RADII]
_ci = [wilson(int(HIT[float(r)][sel].sum()), int(sel.sum())) for r in RADII]
axes[1].plot(RADII, _cl, color=C_WIN, lw=2.0, marker="P", ms=5,
             label="mean mass claimed")
axes[1].errorbar(RADII, _cov, yerr=[np.array(_cov) - np.array([c[0] for c in _ci]),
                                    np.array([c[1] for c in _ci]) - np.array(_cov)],
                 fmt="o", color=INK, ms=4, lw=1.2, capsize=3, label="truth inside")
axes[1].axvline(R_STAR, color=MUTED, ls=":", lw=1.2)
axes[1].axvline(geom.half_u, color=C_MBR, ls=":", lw=1.2,
                label=f"half a cell = {geom.half_u:.2f}")
axes[1].set_xscale("log")
axes[1].set_ylim(0, 1.02)
finish(axes[1], xlabel=r"$r$ [ln units]", ylabel="fraction",
       title="(b) claimed and achieved, against the resolution", legend=True,
       loc="upper left")

# (c) the residual of the two views at the headline r
axes[2].hist(MASS[float(R_STAR)][sel], bins=np.linspace(0, 1, 41), color=C_WIN, alpha=0.55,
             edgecolor=C_WIN, linewidth=0.8, label=rf"window mass at $r={R_STAR:g}$")
axes[2].hist(M1_FINE[sel], bins=np.linspace(0, 1, 41), histtype="step", color=C_MBR,
             linewidth=1.8, label=r"the CELL's mass $M_1$ (same jets)")
finish(axes[2], xlabel="mass", ylabel="jets",
       title="(c) the same mode, two resolutions", legend=True, loc="upper right")

fig.suptitle("the mode's region is a prediction; the truth is the outcome",
             x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

print(f"{'r':>7}{'mean mass claimed':>20}{'t=0 inside':>13}{'95% Wilson':>20}"
      f"{'diff':>8}{'HARDEST inside':>16}{'diff':>8}")
for r in RADII:
    c = HIT[float(r)][sel].mean()
    ch = HIT_HARD[float(r)][sel].mean()
    lo, hi = wilson(int(HIT[float(r)][sel].sum()), int(sel.sum()))
    cl = MASS[float(r)][sel].mean()
    ci = "[%.3f, %.3f]" % (lo, hi)
    print(f"{r:>7.2f}{cl:>20.3f}{c:>13.3f}{ci:>20}{c - cl:>+8.3f}{ch:>16.3f}"
          f"{ch - cl:>+8.3f}")
print(f"\nthe two right-hand columns ask the SAME window a different question: is the")
print(f"truth's widest-angle node in it (t=0, what the prefix walks) versus its HARDEST")
print(f"node (max ln kt, what the stratification uses). They coincide for "
      f"{FIRST_IS_HARDEST[sel].mean():.1%} of these")
print("jets, so the two columns track each other -- and where they part, the coverage")
print("claim is about t=0, because that is the node the density was built for.")
print(f"\non {int(sel.sum()):,} jets with a truth node 0. A difference consistent with zero")
print("means the window's mass is an honest probability -- the posterior's WIDTH around its")
print("mode is calibrated, whatever the mode's own cell mass looked like.")
print(f"\nfor reference, the same jets' CELL mass M_1: median {np.median(M1_FINE[sel]):.4f} "
      f"against the\nwindow's {np.median(MASS[float(R_STAR)][sel]):.4f} at r = {R_STAR:g}. "
      f"Same mode, same posterior, different\nresolution -- and only the second one can be "
      f"checked against the truth at all.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 6. Correctness, at a resolution instead of at a cell

`per_jets_estimation_mode_mass` §7 asks whether the mode skeleton *is* the truth, and answers with
an **exact cell-sequence match**: 1.2% on jets with a real splitting. That number carries the same
defect as the dominance number it sits beside — an exact match at 900 cells is nearly impossible by
construction, and it falls further with every refinement of the grid.

Here is the same question at a stated resolution: **is the truth inside the mode's box?** Below,
both, on the same jets. The exact-match column is the same-geometry quantity; the window column is
the one that transfers.

The second block reports nodes beyond the first. Those windows are **conditional on the mode's
earlier cells being right**, which they usually are not — so their coverage is expected to be worse,
and the drop is a measurement of how quickly the conditioning goes wrong rather than of the
window's honesty.
""")

code(r'''
sel = HAS_T0
CELL_MATCH = np.array([bool(x["cells"]) and x["n_truth"] >= 1
                       and int(x["cells"][0]) == int(geom.to_cell(x["truth"][0, 0],
                                                                 x["truth"][0, 1]))
                       for x in REC])

print(f"{'question':<52}{'fraction':>10}{'95% Wilson':>20}")


def _row(label, k, n):
    ci = "[%.3f, %.3f]" % wilson(k, n)
    print(f"{label:<52}{k / n:>10.3f}{ci:>20}")


_row("node 0 in the SAME CELL as the mode (same-geometry)",
     int(CELL_MATCH[sel].sum()), int(sel.sum()))
for r in RADII:
    _row("t=0 (widest) inside the mode window, r = %g" % r,
         int(HIT[float(r)][sel].sum()), int(sel.sum()))
_row("the HARDEST truth node inside it, r = %g" % R_STAR,
     int(HIT_HARD[float(R_STAR)][sel].sum()), int(sel.sum()))
_row("t=0 IS the hardest truth node (for reference)",
     int(FIRST_IS_HARDEST[sel].sum()), int(sel.sum()))
print(f"\nboth on the {int(sel.sum()):,} jets whose truth has a node 0. The first line is what")
print("an exact-label comparison can say; every line below it is the same comparison with a")
print("resolution named, and only those transfer to another n_bins.")

# deeper nodes: conditional on the mode's earlier cells
print(f"\nnodes beyond the first, at r = {R_STAR:g} -- CONDITIONAL on the mode's earlier "
      f"cells:")
print(f"   {'node':>6}{'jets':>8}{'mean mass claimed':>20}{'truth inside':>15}")
for t in range(MAX_NODES):
    have = np.array([len(x["nodes"]) > t and x["n_truth"] > t for x in REC])
    if have.sum() < MIN_CI_JETS:
        continue
    mm = np.array([x["nodes"][t]["windows"][float(R_STAR)]["mass"]
                   for x, h in zip(REC, have) if h])
    hh = np.array([bool(x["nodes"][t]["hit"][float(R_STAR)])
                   for x, h in zip(REC, have) if h])
    print(f"   {t:>6}{int(have.sum()):>8}{mm.mean():>20.3f}{hh.mean():>15.3f}")
print("   node 0 is unconditional given N>=1; the rest are conditioned on a prefix that is")
print("   usually wrong, so their gap is a measure of the PREFIX, not of the window.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 6a. The Lund plane: the *average posterior* against where the truth actually is

§5 asked a per-jet question — does this jet's window contain this jet's truth. Here is the
population version of the same check, and it needs no window at all.

If the model is calibrated, then averaging its **first-splitting posterior** over jets

$$\bar q(u,v) \;=\; \frac1{N}\sum_j q_\phi(u, v \mid x_j,\ N\ge1)$$

must reproduce the **marginal density of the truth's first splitting** on the same jets. That is a
statement about the whole Lund plane rather than about one region, and it can fail in ways coverage
cannot see: a posterior that is too wide in $\ln k_t$ and too narrow in $\ln(1/\Delta R)$ can still
contain the truth at the right rate.

Three things make this panel worth more than a scatter of point estimates:

- $\bar q$ is the **exact** average density, not a histogram of draws — no sampling noise at all,
  at the pixel resolution of §4's image.
- Every jet contributes its whole posterior, not one number, so the plane is populated even where
  no point estimate ever lands.
- The ratio is the same object the closure notebook plots for the *sampled* posterior
  ([`lund_distribution_closure_v2.ipynb`](lund_distribution_closure_v2.ipynb) §6), on the same
  binning and the same colour scale — so this is that check with the Monte-Carlo error removed.

The point-estimate planes (the window centroid, plain RSD's own node 0) sit beside it. They are a
different object — one point per jet against a density per jet — so they are shown as densities of
*where the estimate landed*, and compared with truth in the same ratio.
""")

code(r'''
U_LO, U_HI = geom.ln_invdelta_range
V_LO, V_HI = geom.ln_kt_range
NB = geom.n_bins                 # the model's own cells: the resolution it decides at
# The RATIO is binned coarser, and it has to be. There is one entry per JET here (node 0),
# not one per splitting, so at 30x30 a typical cell holds ~2 truth entries and a ratio
# there is Poisson noise painted in saturated colour. RATIO_NB divides n_bins, so a ratio
# cell is a whole number of model cells. Same reasoning as
# lund_distribution_closure_v2.ipynb section 6, one step further because the statistics
# are per-jet rather than per-splitting.
RATIO_NB = 10
RLO, RHI, N_MIN = 0.4, 2.5, 12
sel = HAS_T0
EDG = [np.linspace(U_LO, U_HI, NB + 1), np.linspace(V_LO, V_HI, NB + 1)]


def cell_area_at(nb):
    return (U_HI - U_LO) * (V_HI - V_LO) / nb ** 2


def q_bar_at(nb):
    """The EXACT mean posterior, rebinned from the pixel image to an nb x nb grid."""
    k = DENS_MEAN.shape[0] // nb
    m = DENS_MEAN.reshape(nb, k, nb, k).sum(axis=(1, 3)) * DENS_META["pixel_area"]
    return m / cell_area_at(nb)


def point_plane(points, weights, nb):
    """Density of WHERE a point estimate landed, and the raw counts behind it."""
    e = [np.linspace(U_LO, U_HI, nb + 1), np.linspace(V_LO, V_HI, nb + 1)]
    a = np.asarray(points, dtype=float)
    ok = np.isfinite(a).all(axis=1)
    w = np.asarray(weights)[ok]
    h = np.histogram2d(a[ok, 0], a[ok, 1], bins=e, weights=w)[0]
    n = np.histogram2d(a[ok, 0], a[ok, 1], bins=e)[0]
    return h / (w.sum() * cell_area_at(nb)), n


Q_BAR = q_bar_at(NB)
CELL_AREA = cell_area_at(NB)


W_T0 = np.array([r["weight"] for r in REC if r["n_truth"] >= 1])
TRUTH0 = np.array([r["truth"][0, :2] for r in REC if r["n_truth"] >= 1])
WIN0 = np.array([r["nodes"][0]["centroid"][float(R_STAR)] for r in REC if r["n_truth"] >= 1])
RSD0 = np.array([(r["rsd"][0, :2] if len(r["rsd"]) else [np.nan, np.nan])
                 for r in REC if r["n_truth"] >= 1])
P_TRUTH, N_TRUTH = point_plane(TRUTH0, W_T0, NB)
P_WIN, N_WIN = point_plane(WIN0, W_T0, NB)
P_RSD, N_RSD = point_plane(RSD0, W_T0, NB)
# ...and the same four planes at the coarser ratio binning
RB = RATIO_NB
REDG = [np.linspace(U_LO, U_HI, RB + 1), np.linspace(V_LO, V_HI, RB + 1)]
RQ = q_bar_at(RB)
RT, RTN = point_plane(TRUTH0, W_T0, RB)
RW, RWN = point_plane(WIN0, W_T0, RB)
RR, RRN = point_plane(RSD0, W_T0, RB)

PANELS = [(r"$\bar q$: the mean POSTERIOR (exact)", Q_BAR, N_TRUTH),
          ("truth, node 0", P_TRUTH, N_TRUTH),
          (rf"window centroid, $r={R_STAR:g}$", P_WIN, N_WIN),
          ("plain RSD, node 0", P_RSD, N_RSD)]
_pop = np.concatenate([P[P > 0] for _l, P, _n in PANELS if (P > 0).any()])
vmax = float(np.percentile(_pop, 99))
_hit = np.sum([P for _l, P, _n in PANELS], axis=0) > 0
_iu, _iv = np.flatnonzero(_hit.any(axis=1)), np.flatnonzero(_hit.any(axis=0))
XLIM = (EDG[0][max(_iu[0] - 1, 0)], EDG[0][min(_iu[-1] + 2, NB)])
YLIM = (EDG[1][max(_iv[0] - 1, 0)], EDG[1][min(_iv[-1] + 2, NB)])

fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.9), sharey=True)
for ax, (lab, P, _n) in zip(axes, PANELS):
    im = ax.pcolormesh(EDG[0], EDG[1], np.ma.masked_where(P <= 0, P).T, cmap=CMAP,
                       vmin=0.0, vmax=vmax, shading="flat", rasterized=True)
    ax.set_title(lab, fontsize=8)
    ax.set_xlabel(LABEL["lnInvDelta"])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(False)
axes[0].set_ylabel(LABEL["lnkt"])
fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012, label=r"density [per unit area]")
fig.suptitle(f"the first splitting on the Lund plane, {int(sel.sum()):,} jets   "
             f"(view cropped to the populated region)", x=0.06, y=1.03, ha="left")
plt.show()


def plane_ratio(num, den, n_den, n_num):
    """Gated on truth having enough entries to divide by; a bin where truth is empty but
    the prediction is not saturates the top rather than being blanked -- blanking would
    hide invented emissions."""
    out = np.full(num.shape, np.nan)
    ok = (den > 0) & (n_den >= N_MIN)
    out[ok] = num[ok] / den[ok]
    out[(den <= 0) & (n_num >= N_MIN)] = RHI
    return np.ma.masked_invalid(out)


RPANELS = [(PANELS[0][0], RQ, RTN), (PANELS[2][0], RW, RWN), (PANELS[3][0], RR, RRN)]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=True)
for ax, (lab, P, n) in zip(axes, RPANELS):
    R = plane_ratio(P, RT, RTN, n)
    im = ax.pcolormesh(REDG[0], REDG[1], R.T, cmap=DIV,
                       norm=mpl.colors.LogNorm(vmin=RLO, vmax=RHI), shading="flat",
                       rasterized=True)
    ax.set_title(f"{lab}  /  truth", fontsize=8)
    ax.set_xlabel(LABEL["lnInvDelta"])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(False)
axes[0].set_ylabel(LABEL["lnkt"])
fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012, label="ratio to truth")
fig.suptitle(f"ratio to the truth's own node-0 density, on {RB}x{RB} bins   "
             f"(grey = agrees, blue = too few, red = too many, blank = < {N_MIN} truth "
             f"entries)", x=0.05, y=1.03, ha="left")
plt.show()

ok = (RT > 0) & (RTN >= N_MIN)
_names = ["q_bar: the mean POSTERIOR (exact)", f"window centroid, r={R_STAR:g}",
          "plain RSD, node 0"]
print(f"{'plane':<34}{'total':>10}{'mean |log ratio to truth|':>28}{'bins':>7}")
for lab, P, _n in zip(_names, (RQ, RW, RR), (RTN, RWN, RRN)):
    lr = np.abs(np.log(P[ok] / RT[ok])) if ok.any() else np.array([np.nan])
    print(f"{lab:<34}{float(P.sum() * cell_area_at(RB)):>10.3f}"
          f"{float(lr.mean()):>28.3f}{int(ok.sum()):>7}")
print(f"{'truth, node 0 (the reference)':<34}"
      f"{float(RT.sum() * cell_area_at(RB)):>10.3f}{0.0:>28.3f}{int(ok.sum()):>7}")
print(f"\n(the ratio is binned {RB}x{RB} rather than {NB}x{NB}: one entry per JET means a "
      f"model cell holds\n~{float(RTN.sum()) / NB ** 2:.1f} truth entries, where a ratio is "
      f"Poisson noise rather than a measurement.)")
print("\n`total` is the integrated density: 1 for the mean posterior and for any series")
print("with exactly one entry per jet, so a departure there is a binning-window effect")
print("(mass outside the plotted square), not a normalisation error.")
print("\nThe first row is the one with no Monte-Carlo error in it: q_bar is the EXACT")
print("average of the per-jet posteriors, so its ratio to truth is a clean marginal")
print("calibration map for the first splitting. The point-estimate rows are a different")
print("object -- one point per jet rather than a density per jet -- and a point estimate")
print("is EXPECTED to be narrower than the distribution it summarises, so blue cores and")
print("red tails there are the shrinkage of a summary, not a miscalibration.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 7. The residual of a region-based point estimate

The densest box at scale $r$ gives a point estimate — but *which* point?

- The box's **geometric centre** is the obvious answer and the wrong one: the window placement is
  determined only up to the pixel lattice, and for a flat-topped density many placements tie. Quoting
  its midpoint would put a grid back into the answer after the sliding window took one out.
- The **mass-weighted centroid inside the box** has no such dependence. It is the conditional mean
  given the region — the minimiser of squared error restricted to the box — and it is what this
  notebook quotes.
- The density's **peak** is the third candidate, and it is what "the most likely point" actually
  means.

The three are printed together below so the choice is priced rather than asserted. On this
checkpoint the first two agree to well inside a pixel and their RMS differs by under 1% (the density
inside a box that size is near-symmetric), so the decision rests on the principle, not on the
number. The **peak**, though, is measurably worse than either — the mode-versus-mean story of the
plan's §3, now at the coordinate level rather than the skeleton level.

$\ln z$ is absent: the window lives on the Lund **plane**, so a $\Delta \ln z$ panel would be
reporting a coordinate the estimate never claimed. `per_jets_estimation.ipynb` has that panel for
the estimators that do claim it.
""")

code(r'''
SERIES = (["truth", "rsd", "win", "peak"]
          + (["map", "mbr"] if WITH_REFERENCE else []))
STYLE = {"truth": (C_TRUTH, "-", r"truth $y$ (parton)"),
         "rsd": (C_RSD_E, "-", r"plain RSD $x$ (hadron)"),
         "win": (C_WIN, "-.", rf"window centroid, $r={R_STAR:g}$"),
         "peak": (C_POST, ":", r"density peak (the actual mode)"),
         "map": (C_MAP, "-", r"MAP $\hat y$"),
         "mbr": (C_MBR, "-", r"MBR $\hat y$")}
MODELS = [s for s in SERIES if s != "truth"]

NODE0 = {s: [] for s in SERIES}
W_JET = []
if WITH_REFERENCE:
    seed_everything(SEED)
_t0 = time.perf_counter()
for i, rec in enumerate(REC):
    if rec["n_truth"] < 1:
        continue
    NODE0["truth"].append(rec["truth"][0, :2])
    NODE0["rsd"].append(rec["rsd"][0, :2] if len(rec["rsd"]) else np.full(2, np.nan))
    NODE0["win"].append(np.array(rec["nodes"][0]["centroid"][float(R_STAR)]))
    NODE0["peak"].append(np.array(rec["nodes"][0]["peak"]))
    W_JET.append(rec["weight"])
    if WITH_REFERENCE:
        item = ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        with torch.inference_mode():
            draws = model.sample(xf, nx, n=K_DRAWS)
            mults = np.array([len(d) for d in draws], dtype=int)
            eff = learned_min_emissions(model, xf, nx, quantile=LENGTH_FLOOR_QUANTILE,
                                        base_floor=1, mults=mults)
            dec = {**DECODE, **BEAM}
            mp = model.map_or_mbr(xf, nx, draws=draws,
                                  **{**dec, "min_emissions": eff, "point_estimator": "map"})
            mb = model.map_or_mbr(xf, nx, draws=draws,
                                  **{**dec, "point_estimator": "mbr",
                                     "mbr_backend": MBR_BACKEND,
                                     "mbr_n_candidates": MBR_N_CANDIDATES})
        NODE0["map"].append(np.array([mp.nodes[0].ln_invDelta, mp.nodes[0].ln_kt])
                            if mp.nodes else np.full(2, np.nan))
        NODE0["mbr"].append(np.array([mb.nodes[0].ln_invDelta, mb.nodes[0].ln_kt])
                            if mb.nodes else np.full(2, np.nan))
NODE0 = {s: np.asarray(v, dtype=float) for s, v in NODE0.items()}
W_JET = np.asarray(W_JET, dtype=float)
if WITH_REFERENCE:
    print(f"reference series (MAP / MBR) on {len(W_JET):,} jets in "
          f"{(time.perf_counter() - _t0) / 60:.2f} min")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
TABLE = {}
for ax, key in zip(axes, RES_KEYS):
    col = COL[key]
    d_all = np.concatenate([NODE0[s][:, col] - NODE0["truth"][:, col] for s in MODELS])
    d_all = d_all[np.isfinite(d_all)]
    rr = max(math.ceil(float(np.percentile(np.abs(d_all), RESID_PCT)) * 4) / 4, 0.25)
    e = np.linspace(-rr, rr, RESID_NB + 1)
    for s in MODELS:
        d = NODE0[s][:, col] - NODE0["truth"][:, col]
        ok = np.isfinite(d)
        st = wstats(d[ok], W_JET[ok])
        TABLE[(key, s)] = st
        lab = (f"{STYLE[s][2]}   bias {st['bias']:+.3f},  RMS {st['rms']:.3f},  "
               f"68% hw {st['hw68']:.3f}")
        y = density(*h1_sumw2(d[ok], W_JET[ok], e), e)[0]
        if s == "rsd":
            fill(ax, y, e, C_RSD_F, C_RSD_E, label=lab)
        else:
            step(ax, y, e, STYLE[s][0], label=lab, lw=1.8, ls=STYLE[s][1], z=4)
    ax.axvline(0.0, color=INK, lw=1.0, ls=":", zorder=6)
    ax.set_xlim(e[0], e[-1])
    finish(ax, xlabel=DLABEL[key], ylabel="density",
           title=f"{DLABEL[key]}   first splitting, {len(W_JET):,} jets",
           legend=True, loc="upper left")
fig.suptitle(r"estimate $-$ truth on the FIRST splitting, with the region-based estimate",
             x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

print(f"{'coordinate':<12}{'series':<8}{'jets':>8}{'bias':>9}{'RMS':>8}{'68% hw':>9}"
      f"{'RMS / plain RSD':>18}")
for key in RES_KEYS:
    base = TABLE[(key, "rsd")]["rms"]
    for s in MODELS:
        st = TABLE[(key, s)]
        ratio = st["rms"] / base if base > 0 else float("nan")
        print(f"{TLABEL[key] if s == MODELS[0] else '':<12}{s:<8}{st['n']:>8,}"
              f"{st['bias']:>+9.3f}{st['rms']:>8.3f}{st['hw68']:>9.3f}"
              + (f"{'1  (baseline)':>18}" if s == "rsd" else f"{ratio:>18.3f}"))
# the choice of POINT, priced rather than asserted
_bc = np.asarray([r["nodes"][0]["box_centre"][float(R_STAR)]
                  for r in REC if r["n_truth"] >= 1], dtype=float)
print()
print(f"{'which point in the window':<34}{'RMS ln(1/dR)':>14}{'RMS ln kt':>12}"
      f"{'median shift from the centroid':>32}")
for lab, arr in (("mass-weighted centroid (quoted)", NODE0["win"]),
                 ("the box's geometric centre", _bc),
                 ("the density's peak (the mode)", NODE0["peak"])):
    d = arr - NODE0["truth"]
    sh = np.median(np.abs(arr - NODE0["win"]), axis=0)
    print(f"{lab:<34}{np.sqrt((d[:, 0] ** 2).mean()):>14.3f}"
          f"{np.sqrt((d[:, 1] ** 2).mean()):>12.3f}"
          f"{'[%.3f, %.3f]' % (sh[0], sh[1]):>32}")
print(f"one pixel is {2 * geom.half_u / SUB:.3f} ln, so the first two are the same point to")
print("within the lattice -- the centroid is quoted because it does not DEPEND on that")
print("lattice, not because it wins. The peak is a different estimator and loses to both:")
print("the most likely point is not the best point, which is the same lesson MBR encodes")
print("at the level of whole trees.")
print("\nThe interest of the centroid row is not that it wins the residual -- it is that the")
print("region AROUND it carries a checked probability (section 5), which no other row does.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 8. Validity, and the artifact

Two arithmetic checks, neither a physics gate:

1. **Quadrature** — the density image integrates to 1. It is a Riemann sum on a `SUB×SUB` lattice
   per cell, so it closes to the lattice's own accuracy and not better; a large deviation means the
   image is wrong and every number above with it.
2. **Monotonicity** — a box that only grows can only gain mass, so $M(r)$ must be non-decreasing
   per jet.
""")

code(r'''
_bad_q = np.abs(QUAD - 1.0).max()
_mono = np.array([[x["nodes"][0]["windows"][float(r)]["mass"] for r in RADII] for x in REC])
_viol = int((np.diff(_mono, axis=1) < -1e-9).sum())
print(f"1. quadrature : worst |integral - 1| = {_bad_q:.3e}   "
      f"(SUB={SUB} lattice per cell)")
print(f"2. monotonic  : {_viol} violations of M(r) non-decreasing, over "
      f"{_mono.shape[0] * (_mono.shape[1] - 1):,} pairs")
assert _bad_q < 5e-3, "the density image does not integrate to 1 -- the mixture is wrong"
assert _viol == 0, "M(r) decreased with r -- the window search is wrong"

if WRITE_ARTIFACTS:
    sel = HAS_T0
    METRICS = {
        "run": {"notebook": "per_jets_estimation_mode_mass_window",
                "checkpoint": str(CKPT_PATH), "test_path": str(ROOT_PATH),
                "model": info["model_name"], "n_bins": geom.n_bins,
                "n_jets": int(N), "seed": int(SEED), "device": str(device),
                "radii": [float(r) for r in RADII], "r_star": float(R_STAR),
                "sub": int(SUB), "with_reference": bool(WITH_REFERENCE)},
        "coverage": {
            "n_jets_with_node0": int(sel.sum()),
            "note": "the window's mass is P(node 0 in the box | x, N>=1); the outcome is "
                    "whether the truth's node 0 is in it. Node 0 is the WIDEST-ANGLE "
                    "splitting; `frac_hardest_inside` asks the same window about the "
                    "truth's hardest node instead, and `frac_t0_is_hardest` says how "
                    "often those are the same node.",
            "by_radius": {f"{r:g}": {"mean_mass_claimed": float(MASS[float(r)][sel].mean()),
                                     "frac_truth_inside": float(HIT[float(r)][sel].mean()),
                                     "frac_hardest_inside": float(HIT_HARD[float(r)][sel].mean()),
                                     "wilson95": list(wilson(int(HIT[float(r)][sel].sum()),
                                                             int(sel.sum())))}
                          for r in RADII},
            "frac_t0_is_hardest": float(FIRST_IS_HARDEST[sel].mean()),
            "same_cell_match": float(CELL_MATCH[sel].mean()),
        },
        "resolution": {
            "M1_cell_median": float(np.median(M1_FINE[sel])),
            "window_mass_median_at_r_star": float(np.median(MASS[float(R_STAR)][sel])),
            "note": "same mode, same posterior; the first is a cell probability and scales "
                    "with n_bins, the second is a probability of a stated region and does "
                    "not.",
        },
        "residuals_node0": {f"{k}|{s}": v for (k, s), v in TABLE.items()},
        "validity": {"max_abs_quadrature_defect": float(_bad_q),
                     "monotonicity_violations": int(_viol)},
    }
    out = save_metrics(METRICS, (REPO / CKPT_PATH).parent / "mode_mass_window.json")
    print(f"\nwrote {out.relative_to(REPO)}")
else:
    print("\nWRITE_ARTIFACTS = False -- nothing written")
''')

# ---------------------------------------------------------------------------
md(r"""
---

### Reading this notebook

- **The window's mass is a claim; §5 checks it.** That is the whole reason to slide a box instead
  of reading a cell: a cell-indexed mode can only be compared to the truth by exact label match,
  which measures the grid. A box can be asked whether the truth is inside, and the answer is a
  coverage number that transfers to any `n_bins`.
- **Calibration is not correctness.** A perfectly calibrated model can put its regions in the
  wrong place — §5 would still sit on the diagonal. §7's residual is the check that they are in the
  *right* place, and the two have to be read together.
- **`t=0` is the widest-angle splitting, not the hardest.** They coincide for ~80% of
  multi-splitting jets, and §5 asks the same window about both so the difference is measured rather
  than assumed. Everything conditional here walks the declustering order, so `t=0` is the node the
  density is built for; `lnkt_lead` and `leading_emission_cell` elsewhere in the repo mean the
  hardest, and mixing the two names was a real error in the sibling notebook's §7 text.
- **Only node 0 is unconditional.** Its window mass is
  $P(\text{node }0 \in B \mid x,\ N\ge1)$, an honest marginal. Every later node is conditioned on
  the mode's earlier cells being right, which they mostly are not, so §6's deeper rows measure the
  prefix rather than the window. Do not quote them as coverage.
- **Jets with an empty parton truth carry no outcome**, so the coverage population is the
  truth-nonempty one. Whether a jet has a splitting at all is the multiplicity question — grid-free,
  and answered by $q(N\mid x)$ in
  [`per_jets_estimation_mode_mass.ipynb`](per_jets_estimation_mode_mass.ipynb) §6a. Splitting the
  two is deliberate: mixing "is there anything there" into "where is it" is what made the pooled
  numbers in that notebook uninformative.
- **`R_STAR` is a choice, and the curve is printed so you can see it being made.** 0.45 is where
  the median jet's window reaches half its mass. The physically motivated alternative is the
  non-perturbative width $\sigma_0 + \Lambda_{\rm eff}/k_t$ — on this checkpoint the coordinate
  head's own $\sigma$ is ~0.18–0.25, and §5(b) shows both readings at once.
- **Nothing here is sampled.** The density is exact (a block-wise mixture of bounded truncated
  normals), the window search is exact (integral image over every placement), and §8 prices the one
  approximation there is — the quadrature lattice.
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
    "metadata": {"language_info": {"name": "python", "pygments_lexer": "ipython3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = (Path(__file__).resolve().parent.parent / "notebooks"
       / "per_jets_estimation_mode_mass_window.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
