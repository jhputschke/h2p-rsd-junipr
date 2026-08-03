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
3. **§7 — the residual of a region-based point estimate.** The box's centre is a legitimate point
   estimate ("the centre of the densest region at scale $r$"), so it goes into the same
   $\Delta = \text{estimate} - \text{truth}$ panels as plain RSD, MAP and MBR.

Everything is **exact**: the positional density is a block-wise mixture (each cell's truncated
normal is confined to its own cell), the window sums come from an integral image, and the
quadrature check is printed. Nothing here is sampled and nothing is a bound.

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
        nodes.append({
            "t": t,
            "windows": wins,
            "truth": (None if truth is None else [float(truth[0]), float(truth[1])]),
            "hit": {r: (None if truth is None else in_window(truth, w))
                    for r, w in wins.items()},
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


def window_centre(w):
    """The box's centre -- the region-based POINT estimate."""
    return np.array([0.5 * (w["u_lo"] + w["u_hi"]), 0.5 * (w["v_lo"] + w["v_hi"])])


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
for i in range(N):
    REC.append(jet_windows(i))
    REC[-1]["nodes"] = [{k: v for k, v in nd.items() if k != "_grid"}
                        for nd in REC[-1]["nodes"]]      # drop the tensors, keep the numbers
print(f"windowed {N:,} jets in {(time.perf_counter() - _t0) / 60:.2f} min")

HAS_T0 = np.array([r["n_truth"] >= 1 for r in REC], dtype=bool)
MASS = {float(r): np.array([x["nodes"][0]["windows"][float(r)]["mass"] for x in REC])
        for r in RADII}
HIT = {float(r): np.array([bool(x["nodes"][0]["hit"][float(r)]) if x["nodes"][0]["hit"][float(r)]
                           is not None else False for x in REC]) for r in RADII}
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

print(f"{'r':>7}{'mean mass claimed':>20}{'truth inside':>15}{'95% Wilson':>20}"
      f"{'difference':>13}")
for r in RADII:
    c = HIT[float(r)][sel].mean()
    lo, hi = wilson(int(HIT[float(r)][sel].sum()), int(sel.sum()))
    cl = MASS[float(r)][sel].mean()
    ci = "[%.3f, %.3f]" % (lo, hi)
    print(f"{r:>7.2f}{cl:>20.3f}{c:>15.3f}{ci:>20}{c - cl:>+13.3f}")
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
    _row("node 0 inside the mode window, r = %g" % r,
         int(HIT[float(r)][sel].sum()), int(sel.sum()))
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
## 7. The residual of a region-based point estimate

The centre of the densest box at scale $r$ is a point estimate in its own right — "the middle of
where the posterior actually is" — and it belongs in the same $\Delta = \text{estimate} -
\text{truth}$ panels as plain RSD, the MAP and the MBR medoid, on the first splitting.

$\ln z$ is absent: the window lives on the Lund **plane**, so a $\Delta \ln z$ panel would be
reporting a coordinate the estimate never claimed. `per_jets_estimation.ipynb` has that panel for
the estimators that do claim it.
""")

code(r'''
SERIES = ["truth", "rsd", "win"] + (["map", "mbr"] if WITH_REFERENCE else [])
STYLE = {"truth": (C_TRUTH, "-", r"truth $y$ (parton)"),
         "rsd": (C_RSD_E, "-", r"plain RSD $x$ (hadron)"),
         "win": (C_WIN, "-.", rf"window centre, $r={R_STAR:g}$"),
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
    NODE0["win"].append(window_centre(rec["nodes"][0]["windows"][float(R_STAR)]))
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
print("\nThe window centre is a point estimate of the same kind as the others and is scored")
print("the same way. Its interest is not that it wins -- it is that the region AROUND it")
print("carries a checked probability (section 5), which none of the other rows does.")
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
                    "whether the truth's node 0 is in it. Node 0 only.",
            "by_radius": {f"{r:g}": {"mean_mass_claimed": float(MASS[float(r)][sel].mean()),
                                     "frac_truth_inside": float(HIT[float(r)][sel].mean()),
                                     "wilson95": list(wilson(int(HIT[float(r)][sel].sum()),
                                                             int(sel.sum())))}
                          for r in RADII},
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
