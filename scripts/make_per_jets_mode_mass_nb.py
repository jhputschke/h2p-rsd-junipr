"""Build notebooks/per_jets_estimation_mode_mass.ipynb.

    python scripts/make_per_jets_mode_mass_nb.py

Same pattern and the same reason as scripts/make_per_jets_nb.py: the notebook is past
what the notebook editor can open, so THIS FILE is the source of truth and an edit made
straight to the .ipynb is lost the next time anyone regenerates.

The notebook is `per_jets_estimation.ipynb` with the mode-mass audit
(docs/PLAN_ModeMassAudit.md) added as a first-class series: it carries the same per-jet
estimates (plain RSD, MAP, MBR, one posterior draw) and the same index-aligned residual
tables, plus the exactly-enumerated top-k skeleton posterior and the MODE-SKELETON
estimate that falls out of it.

Regenerating drops the executed outputs, so follow it with

    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/per_jets_estimation_mode_mass.ipynb
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
# Per-jet estimation, mode-mass edition — *is there a dominant parton skeleton, and is it the true one?*

The companion to [`per_jets_estimation.ipynb`](per_jets_estimation.ipynb), on the same
checkpoint, the same held-out file and the same decode, with one thing added: the posterior's
**skeleton** distribution is *enumerated exactly* rather than sampled or beam-searched.

The **skeleton** is $S = (N,\, c_0 \dots c_{N-1})$ — the multiplicity plus the *ordered* Lund-cell
sequence. Two facts make this notebook possible, and both are proved in
[`docs/PLAN_ModeMassAudit.md`](../docs/PLAN_ModeMassAudit.md) §1:

1. **The skeleton marginal is exact, with no integration.** The per-node coordinate factor is a
   proper density *given the cell*, so it integrates to 1 and drops out:
   $q_\phi(S\mid x) = \big[\prod_t P_{\rm cont}\,P_{\rm split}(c_t)\big]\,P_{\rm cont}({\rm stop})$
   is computable from the discrete heads alone.
2. **Prefix mass = subtree mass**, so a best-first (uniform-cost) search over the prefix tree pops
   completed skeletons in *exact descending mass order* — Dijkstra, not beam search. Every pruned
   branch's mass is accumulated in closed form, so
   $\sum_{i\le k} M_i + m_{\rm frontier} + m_{\rm pruned} = 1$ holds at every stop, and the
   coverage remainder is a **bound**, not an estimate.

So $M_1 > \tfrac12$ is a *certificate* of dominance, not evidence for it.

**One thing $M_1$ is not, and the notebook says so at every turn: grid-free.** A cell's
probability is $\approx$ density $\times$ area, so every $N\ge1$ skeleton's mass scales with the
cell area while $q(N{=}0\mid x)$ — the one skeleton that references no cell — does not. Refine
`n_bins` and both the *level* of $F(m)$ and the *identity* of the argmax change with nothing about
the model changing. Exactness is not invariance. So $F(m)$ is quoted here as the **same-geometry,
same-checkpoint** comparison it validly is, and §6a carries the grid-free companion: the
Lund-plane **area** the posterior actually occupies, in physical units, against the width the
coordinate head claims for itself.

It answers four questions, in this order, and **never merges the first two**:

1. **Does a dominant skeleton exist?** — the $M_1$ distribution and
   $F(m) = \mathrm{frac}(M_1 \ge m)$, overall and per stratum (§6), with §6a's
   resolution-free reading of the same posterior beside it.
2. **Is it the true one?** — the truth skeleton's exact mass and rank on the same jets (§7).
   Dominance and correctness are logically independent: a model can be sharply dominant and
   wrong, or diffuse and centred on the truth.
3. **How big is the region the posterior actually occupies, and how much mass sits in a region
   of a *stated* size?** — the grid-free pair, §6a (an area at fixed mass, in
   $\ln(1/\Delta R)\times\ln k_t$ units) and §6b ($M_1(r)$, a mass at fixed size, which is what
   turns the dominance sentence back into a quotable probability).
4. **Does dominance buy a better point estimate?** — the per-splitting residual
   $\Delta = \text{estimate} - \text{truth}$ of `per_jets_estimation.ipynb`, now with the
   **mode-skeleton** estimate beside MAP and MBR, and split by whether the jet *has* a dominant
   skeleton (§9–§10).

Nothing here is a gated intervention: the audit **reads** the posterior and never writes to the
estimator stack. There are no pass/fail thresholds, but the quoted quantities and strata are
pre-registered in the plan's §7, and the two *validity* checks (mass accounting, and the
empty-skeleton identity against the model's own $q(0\mid x)$) are in §11 — those are arithmetic,
and a nonzero defect means the search is wrong, not that the model is.

---

### What the mode-skeleton estimate is, and what it is not

Given the top-1 skeleton, its coordinates are attached by the **existing staged machinery** —
per-node conditional modes with the $\kappa$ gate respected — exactly as the MAP does once its beam
has chosen a cell sequence. The difference is *which* cell sequence: the MAP beam-searches the
**joint** argmax under a length floor, while this is the exact argmax of the **skeleton marginal**,
with its mass quoted.

The continuous coordinates are **never** claimed to concentrate. They cannot: the non-perturbative
width $\sigma = \sigma_0 + \Lambda_{\rm eff}/k_t$ is irreducible, which is why dominance is only
posed for the discrete structure and why §4 prints every node as *value $\pm$ posterior width*
rather than as a bare point.

$\psi$ is absent from the difference panels for the same reason as in
[`per_jets_estimation.ipynb`](per_jets_estimation.ipynb): on the pinned arm the von Mises
$\kappa$ has median 0.022 and 99.9% of splittings sit below `decode.kappa_min_mode`, so a
$\Delta\psi$ panel would mostly plot an arbitrary angle.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Parameters

**One knob: `RUN`.** Point it at a run directory and everything else is found inside — the
checkpoint, and the `prod_test_v1` artifact beside it that carries the held-out file and the
frozen empty-tree $\tau$. It also accepts an arm root, a `best.ckpt`, or an artifact JSON
directly, so any path you happen to have in hand works:

```python
RUN = "runs/prod_test_v1/v1_contstop_s0/20260801-173609-a08a175ae2"   # a run directory
RUN = "runs/prod_test_v1/v1_contstop_s0"                             # an arm root
RUN = "runs/.../best.ckpt"
RUN = "runs/.../prod_test_v1/prod_test_v1_metrics.json"
RUN = None    # the newest prod_test_v1 artifact for ARM (the default)
```

The one thing that is **not** derived from the checkpoint is the evaluation file. A checkpoint
records the file it *trained* on (`data.path`), so taking the eval file from it would silently
turn a closure test into a report on the training set — it comes from the artifact, or from
`ROOT_PATH` when there is none. §3 asserts the two differ either way.

The **audit block** below has the same defaults as `audit:` in the config schema
([`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md) §8a), so this notebook and
`h2p-rsd-junipr eval <ckpt> experiment.mode_audit=true` report the same numbers on the same jets.
""")

code(r'''
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

# --- WHAT TO RUN: one knob ---------------------------------------------------
# A run directory, an arm root, a best.ckpt, or a prod_test_v1_metrics.json -- whichever
# path you have. Everything else is found inside it. None -> the newest prod_test_v1
# artifact for ARM below.
RUN = None
# The arm docs/PROD_TEST_v1_RESULTS.md selected.
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
        raise ValueError(f"RUN must be a directory, a .ckpt or a .json; got {_p.name!r}")

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
DEVICE  = "cpu"        # the search is inherently SEQUENTIAL per jet, so a per-step
#                        accelerator sync dominates at dec_dim 64. "auto"/"mps"/"cuda" work.
TORCH_THREADS = 4      # None -> torch's default (one per core).

# --- decode -----------------------------------------------------------------
K_DRAWS               = 200
LENGTH_FLOOR_QUANTILE = 0.15   # per-jet MAP floor at this quantile of P(n|x); 0.0 -> off
MBR_BACKEND           = "energyflow" if _ilu.find_spec("energyflow") else "pot"
MBR_N_CANDIDATES      = 16     # MBR candidate cap per jet (0 = all K draws)
GATE_EMPTY            = False  # see section 5; EMPTY_THRESHOLD is set above

# --- the mode-mass audit (docs/PLAN_ModeMassAudit.md) -----------------------
# Defaults match the `audit:` config block, so this notebook and
# `h2p-rsd-junipr eval <ckpt> experiment.mode_audit=true` agree jet for jet.
AUDIT_K             = 32       # completions enumerated per jet, in EXACT descending order
AUDIT_BUDGET        = 20000    # expansion cap; a jet that hits it is `certified: false`
AUDIT_PRUNE_REL     = 1e-6     # drop children below prune_rel x the best mass found so far
AUDIT_TOPK_CHILDREN = 0        # cell children per expansion, 0 == every one of the 900.
#                                A cap costs CERTIFICATION, not correctness -- the dropped
#                                tail is accounted in closed form -- but on this geometry
#                                that is a bad trade: capping at 64 takes the certified
#                                fraction from 97% to 20% and saves 28% of the runtime.
AUDIT_MAX_FRONTIER  = 20000    # heap cap; evictions go to the pruned total, not to nothing
AUDIT_EPS_N         = 1e-4     # q(N|x) floor for the per-N searches (n_head / factorized)
# Pre-registered BEFORE the run (plan section 7): the F(m) grid and the dominance mark.
THRESHOLDS  = [0.3, 0.5, 0.7]
DOMINANT_M  = 0.5              # "a dominant skeleton exists" -- self-certifying above 1/2
# Draws used for the entropy estimate H_hat = -mean_k log q(S_k|x). Reused from the K
# posterior draws, so this costs scoring only, never a second sampling pass.
H_DRAWS     = 100

# --- the residual study (same knobs as per_jets_estimation.ipynb) ------------
T_FIRST   = 2      # "the first N splittings" -- section 9b shows t = 0 .. T_FIRST-1
RESID_NB  = 41     # bins per residual panel; ODD so one bin is centred on zero
RESID_PCT = 99.0   # residual axes span +/- this percentile of |delta|, pooled
N_BOOT    = 200    # jet-level bootstrap resamples for the RMS-ratio CI
MIN_CI_JETS = 25   # below this many distinct jets in a slice, no CI is quoted
SHOWCASE_JET = None   # index for section 4; None -> auto-pick (see pick_showcase)

WRITE_ARTIFACTS = True   # mode_audit_nb.json beside the checkpoint

# --- guards -----------------------------------------------------------------
assert not (GATE_EMPTY and EMPTY_THRESHOLD <= 0.0), (
    "GATE_EMPTY=True needs a frozen tau, and none was read. Point RUN at an arm with a "
    "prod_test_v1 artifact, or leave GATE_EMPTY False."
)
assert MBR_BACKEND != "surrogate", (
    "the surrogate is a different risk function, not a faster one -- never for a "
    "reported number"
)
assert 0.0 <= DOMINANT_M <= 1.0 and all(0.0 <= t <= 1.0 for t in THRESHOLDS)
''')

# ---------------------------------------------------------------------------
md(r"""
## 1. Imports, house style, helpers

Palette, `rcParams` and the histogram helpers are inherited verbatim from
[`per_jets_estimation.ipynb`](per_jets_estimation.ipynb) so the panels of the two overlay without
re-reading a legend: truth is **ink**, plain RSD is a **grey fill**, and the model series take the
validated categorical slots.

This notebook has a **fourth** model series (the mode skeleton), so the palette is re-validated as
a *set* rather than extended by eye: `#2a78d6, #eb6834, #199e70, #b0499e` clears every all-pairs
gate — lightness band, chroma floor, CVD separation (worst adjacent $\Delta E = 8.4$ under
protanopia), the normal-vision floor ($\Delta E \ge 18.5$) and 3:1 contrast against the surface.
Each series additionally carries its own marker and dash pattern, so identity is never colour
alone.
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

from h2p_rsd_junipr.config import audit_params, decode_params
from h2p_rsd_junipr.data.datamodule import select_pt_range
from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.data.rntuple import load_rntuple
from h2p_rsd_junipr.eval.closure import lund_tree_str
from h2p_rsd_junipr.eval.mode_audit import audit_jet, spearman, summarise_mode_audit
from h2p_rsd_junipr.eval.report import save_metrics
from h2p_rsd_junipr.eval.support import grooming_from_jets
from h2p_rsd_junipr.features import AUX_FEATURES, node_raw
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import learned_min_emissions
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.train.trainer import seed_everything, select_device

# --- style (inherited from per_jets_estimation.ipynb) -------------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
C_TRUTH = INK
C_RSD_F, C_RSD_E = "#e1e0d9", "#898781"
C_MAP   = "#2a78d6"    # MAP point estimate       -- blue    (slot 1)
C_MBR   = "#eb6834"    # MBR point estimate       -- orange  (slot 2)
C_POST  = "#199e70"    # posterior draw           -- aqua    (slot 3), dashed
C_MODE  = "#b0499e"    # MODE SKELETON estimate   -- magenta (slot 4)
# Mass-spectrum accents: the enumerated bars take the mode series' hue, the two
# accounting bands the neutrals, so nothing in section 4b competes with a data series.
C_FRONT, C_PRUNE = "#c3c2b7", "#e1e0d9"
# The plane maps, inherited verbatim from lund_distribution_closure_v2.ipynb so the two
# notebooks' Lund panels are readable against each other without re-learning a scale.
# Sequential = ONE hue light->dark (magnitude); diverging = two poles with a NEUTRAL grey
# midpoint (polarity), so "agrees with truth" reads as nothing at all.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = mpl.colors.LinearSegmentedColormap.from_list("h2p_blue", SEQ_BLUE)
CMAP.set_bad(SURFACE)   # empty bins recede to the surface instead of reading as data
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "h2p_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a3", "#d03b3b", "#7a1f1f"])
DIV.set_bad(SURFACE)

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


def wilson(k, n, z=1.96):
    """Wilson score interval -- the honest error bar on every fraction below.

    These are proportions on a few hundred jets per stratum, where `p +- z*sqrt(p(1-p)/n)`
    is both too wide in the middle and nonsensical at the edges (it can leave [0, 1] and
    collapses to zero width at p = 0 or 1, exactly where a near-empty Lund quadrant lands).
    Brown, Cai & DasGupta, *Statist. Sci.* 16 (2001) 101.
    """
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
## 2. The model

Everything structural comes from the checkpoint's own config snapshot — geometry, encoder, family,
and the aux conditioning columns the encoder was *built* for.

One extra line here: the **skeleton search strategy** the loaded family publishes. `ar` is the
per-step continue/stop product; `nhead` is the explicit $q(N\mid x)$ factorization, searched at
fixed length per $N$ and merged on one heap; `factorized` is `cinn`/`diffusion`, where the cells are
conditionally independent given $x$ and the top-$k$ is a lazy $k$-best over sorted categoricals. A
family with no adapter raises **by name** rather than returning something plottable — an empty
table for a family that was never searched would be worse than no table.
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
CONT = bool(getattr(model, "has_continuous_coords", False))
TAU = float(EMPTY_THRESHOLD) if GATE_EMPTY else 0.0
LNZ_SUPPORT = str(OmegaConf.select(cfg, "model.lnz_support") or "legacy")

# The audit's own parameters, assembled through the SAME helper the CLI uses, then
# overridden by the notebook knobs of section 0. Reading them back out of one dict is
# what keeps the artifact's `search` block honest about what actually ran.
AUDIT = {**audit_params(cfg), "k": AUDIT_K, "budget": AUDIT_BUDGET,
         "prune_rel": AUDIT_PRUNE_REL, "topk_children": AUDIT_TOPK_CHILDREN,
         "max_frontier": AUDIT_MAX_FRONTIER, "eps_n": AUDIT_EPS_N,
         "thresholds": list(THRESHOLDS)}

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
        "filler constant. The SKELETON audit would still be exact -- but sections 9-10 "
        "would not be. Point RUN at a family with continuous_coords=true."
    )
''')

# ---------------------------------------------------------------------------
md(r"""
## 3. The test sample

The held-out PYTHIA file — a different generator seed from the one the checkpoint trained on.

The selection is `len(x) > 0` only: the **deployable** population, every jet an analysis could pick
out on data, including the ~17% whose parton truth is the empty tree. Requiring `len(y) > 0` would
read the answer. Those truth-empty jets stay in: **the empty skeleton is a first-class row of the
enumeration** ($N=0$, mass $P_{\rm cont}(\mathrm{stop}\mid h_0,e)$ — the same $q(0\mid x)$ the
empty-tree analysis thresholds), so dropping them would remove exactly the jets where the
posterior's most dominant answer is often correct.

The grooming constants come from the **data**, not the config: the stratification axes $d_B$ and
$d_F$ are distances to boundaries the generator actually enforced.
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
GROOM = grooming_from_jets(jets)
Z_CUT, BETA, KT_FLOOR = GROOM["z_cut"], GROOM["beta"], GROOM["kt_floor"]

_nx = np.array([len(j["x"][0]) for j in jets])
_ny = np.array([len(j["y"][0]) for j in jets])
print(f"source     : {ROOT_PATH}:{NTUPLE_NAME}   (trained on {TRAIN_PATH!r})")
print(f"generator  : {jets[0].get('generator', 'n/a')}")
print(f"grooming   : z_cut={Z_CUT:.3f}  beta={BETA:.3f}  kt_floor={KT_FLOOR:.3f} GeV "
      f"(read from the DATA -- the audit's strata are distances to these)")
print(f"selection  : len(x)>0 keeps {len(jets):,} of {_n_in:,} jets")
print(f"multiplicity: hadron x = {_nx.mean():.3f}   parton y = {_ny.mean():.3f}   "
      f"x/y = {_nx.mean() / _ny.mean():.3f}")
print(f"             P(n_y = 0) = {np.mean(_ny == 0):.3f}   "
      f"(the EMPTY skeleton is enumerated, so these jets are audited like any other)")
print(f"evaluating : the first {min(N_JETS, len(ds)):,} of them")

# The search strategy this family publishes, probed rather than inferred from the config.
_it0 = ds[0]
_xf0, _nx0 = _it0["xf"].unsqueeze(0).to(device), torch.tensor([_it0["nx"]], device=device)
with torch.inference_mode():
    SEARCH_KIND = model.skeleton_search_spec(_xf0, _nx0).kind
print(f"\nskeleton search: kind={SEARCH_KIND!r}   k={AUDIT['k']}   "
      f"budget={AUDIT['budget']}   prune_rel={AUDIT['prune_rel']:g}   "
      f"topk_children={AUDIT['topk_children']}")
print("  'ar'         -- per-step continue/stop; the search steps the SAME hook the beam "
      "search does\n"
      "  'nhead'      -- explicit q(N|x); fixed-length search per N, merged on one heap\n"
      "  'factorized' -- cells independent given x; lazy k-best over sorted categoricals")
''')

# ---------------------------------------------------------------------------
md(r"""
## 4. One jet, end to end

`showcase_jet(i)` runs the full per-jet inference — $K$ posterior draws, the MAP with its learned
per-jet length floor, the MBR (minimum expected perturbative-Lund EMD) estimate, one posterior draw
with *sampled* coordinates — **and** the exact top-$k$ skeleton enumeration, the mode-skeleton
estimate built from it, the truth skeleton's own mass and rank, and the entropy estimate.

`estimate_jet(i)` is the same computation with no plotting; §5 loops it over the sample. The two
share one implementation, so the single-jet figure and the population figures cannot describe
different decodes — and the audit record is built by
`eval.mode_audit.audit_jet`, the same function `h2p-rsd-junipr eval … experiment.mode_audit=true`
calls, so this notebook and the CLI artifact cannot drift either.
""")

code(r'''
def pe_coords(pe):
    """LundPointEstimate -> (n, 4) in node_raw column order."""
    if pe is None or not pe.nodes:
        return np.zeros((0, 4))
    return np.array([[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes], dtype=float)


@torch.inference_mode()
def draw_coords(xf, nx, cells):
    """One posterior draw's CONTINUOUS coordinates, sampled rather than moded."""
    if not len(cells):
        return np.zeros((0, 4))
    c = model.sample_coordinates(xf, nx, list(cells))
    return np.asarray(c.cpu().double().numpy(), dtype=float).reshape(-1, 4)


@torch.inference_mode()
def mode_estimate(xf, nx, cells):
    """The MODE-SKELETON point estimate: the exact argmax of the skeleton marginal, with
    its coordinates attached by the EXISTING staged machinery.

    `describe_sequence` is the staged decode -- per-node conditional modes, with the psi
    mode replaced by a draw wherever the von Mises kappa falls below
    `decode.kappa_min_mode` and the node flagged. That is the same path `map_estimate`
    takes once its beam has chosen a cell sequence, which is exactly the point: the two
    estimates differ in WHICH skeleton, not in how coordinates are attached.

    A family without the staged hook falls back to the contract's `describe_cells`, whose
    coordinates are a DRAW -- reported as such by `coords_source`, never as a mode.
    """
    fn = getattr(model, "describe_sequence", None)
    return fn(xf, nx, list(cells)) if fn is not None else model.describe_cells(xf, nx, list(cells))


@torch.inference_mode()
def node_widths(xf, nx, cells):
    """Per-node posterior WIDTHS for a cell chain: (sigma_u, sigma_v, sigma_lnz, kappa).

    The coordinates never concentrate below the non-perturbative width, so a mode-skeleton
    node quoted as a bare number would be a claim the model does not make. `None` for a
    family with no closed-form coordinate head, and the printed table says so rather than
    inventing an error bar.
    """
    fn = getattr(model, "coord_head_params", None)
    if fn is None or not len(cells):
        return None
    p = fn(xf, nx, list(cells))
    if p is None:
        return None
    _du_m, _dv_m, du_s, dv_s, _lnz_m, lnz_s, _mu, kappa = p
    return np.stack([du_s.cpu().double().numpy(), dv_s.cpu().double().numpy(),
                     lnz_s.cpu().double().numpy(), kappa.cpu().double().numpy()], axis=-1)


SERIES = ("truth", "rsd", "map", "mbr", "mode", "post")
MODELS = ("rsd", "map", "mbr", "mode", "post")   # everything differenced against truth
STYLE = {
    "truth": (C_TRUTH, "-",   r"truth $y$ (parton)"),
    "rsd":   (C_RSD_E, "-",   r"plain RSD $x$ (hadron)"),
    "map":   (C_MAP,   "-",   r"MAP $\hat y$"),
    "mbr":   (C_MBR,   "-",   r"MBR $\hat y$"),
    "mode":  (C_MODE,  "-.",  r"mode skeleton $\hat S_1$"),
    "post":  (C_POST,  "--",  r"posterior draw"),
}
MARKER = {"truth": "o", "rsd": "x", "map": "*", "mbr": "D", "mode": "P", "post": "s"}
MSIZE  = {"truth": 8.0, "rsd": 7.0, "map": 13.0, "mbr": 5.5, "mode": 8.0, "post": 4.5}


@torch.inference_mode()
def estimate_jet(i, rng=None, k_draws=None, with_cloud=False):
    """Every series for jet `i` plus its audit record. No plotting, no printing.

    The audit record comes from `eval.mode_audit.audit_jet` -- the SAME builder the CLI
    runner loops -- and the posterior draws are handed to it, so the entropy estimate
    costs scoring only and never a second sampling pass.
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    K = int(k_draws or K_DRAWS)
    item, jet = ds[i], jets[i]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)

    draws = model.sample(xf, nx, n=K)
    mults = np.array([len(d) for d in draws], dtype=int)

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

    # --- the audit -----------------------------------------------------------
    rec_audit, enum = audit_jet(model, item, jet, geom, device, audit=AUDIT,
                                draws=draws[:H_DRAWS], groom=GROOM, index=int(i))
    mode_pe = mode_estimate(xf, nx, enum.top1_cells)

    rec = {
        "i": int(i), "weight": float(W_ALL[i]),
        "truth": np.asarray(item["yraw"].numpy(), dtype=float),
        "rsd": np.asarray(node_raw(*jet["x"]), dtype=float),
        "map": pe_coords(mp), "mbr": pe_coords(mbr), "mode": pe_coords(mode_pe),
        "post": draw_coords(xf, nx, pick),
        "mults": mults,
        "q0": float(model.length_pmf(xf, nx, mults=mults)[0]),
        "min_emissions": int(eff), "pe": {"map": mp, "mbr": mbr, "mode": mode_pe},
        "aux": {n: float(AUX_FEATURES[n](jet)) for n in AUX},
        "risk": float(mbr.risk) if mbr.risk is not None else float("nan"),
        "audit": rec_audit, "enum": enum,
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


def cells_str(cells):
    return "(empty)" if not len(cells) else " -> ".join(str(int(c)) for c in cells)


def showcase_jet(i=None, k_draws=None, show_trees=True, figsize=(13.4, 9.6)):
    """Everything the model says about ONE jet, with its skeleton posterior enumerated.

    Panels, in reading order
      (a) the primary Lund plane: the posterior cloud, the truth and plain-RSD ladders,
          and the MAP / MBR / MODE-SKELETON point estimates in declustering order.
      (b) the multiplicity posterior P(n|x), with every series' length marked.
      (c) the exact top-k skeleton mass spectrum, with the certified remainder
          (frontier + pruned) drawn as the bar the enumeration did NOT reach.
      (d) the cumulative coverage C_k and the bound on what it leaves out.
      (e-g) the ladders themselves over their residual-to-truth strips.

    Returns the record from `estimate_jet` with a `resid` table added.
    """
    i = pick_showcase() if i is None else int(i)
    rec = estimate_jet(i, k_draws=k_draws, with_cloud=True)
    a = rec["audit"]
    enum = rec["enum"]
    y = rec["truth"]
    ny = len(y)
    depth = max(1, max(len(rec[s]) for s in SERIES))

    rec["resid"] = {s: (rec[s][:min(len(rec[s]), ny)] - y[:min(len(rec[s]), ny)])
                    for s in MODELS}

    # ---- printed header ------------------------------------------------------
    print(f"jet #{i}   weight {rec['weight']:.4g}   "
          f"P(n=0|x) = {rec['q0']:.3f}   MAP length floor = {rec['min_emissions']}")
    print(f"  multiplicity   truth y = {ny}   plain RSD x = {len(rec['rsd'])}   "
          f"MAP = {len(rec['map'])}   MBR = {len(rec['mbr'])}   "
          f"mode skeleton = {len(rec['mode'])}   "
          f"posterior = {rec['mults'].mean():.2f} +/- {rec['mults'].std():.2f}")
    print(f"  MBR risk       {rec['risk']:.4f}   (mean expected Lund-EMD to the "
          f"posterior -- NOT a likelihood)")
    print()
    print("  --- the skeleton posterior, enumerated exactly ---")
    print(f"  M_1 = {a['M1']:.4f}   M_2 = {a['M2']:.4f}   "
          f"log(M_1/M_2) = {a['log_M1_M2']:.2f}")
    print(f"  top-1 skeleton cells: {cells_str(a['cells_top1'])}")
    print(f"  coverage C_k = {a['C_k']:.4f} over {a['n_enumerated']} skeletons; "
          f"remainder <= {a['remainder_bound']:.2e} "
          f"(frontier {a['frontier']:.2e} + pruned {a['pruned']:.2e})")
    print(f"  certified: top-k {a['certified']}   top-1 {a['certified_top1']}   "
          f"({a['n_expansions']} expansions)"
          + ("   [M_1 > 1/2 is self-certifying whatever the search pruned]"
             if a["M1"] > 0.5 else ""))
    print(f"  entropy H_hat = {a['H_hat']:.3f} nat over {a['n_entropy_draws']} draws "
          f"-> {a['eff_skeletons']:.2f} effective skeletons "
          f"(enumerated lower bound {a['H_enumerated_lower']:.3f})")
    print(f"  TRUTH skeleton: mass {a['M_truth']:.4g}   rank "
          + (f"{a['rank_truth']}" if a["rank_truth"] else f"> {a['n_enumerated']} (outside "
             f"the enumeration)")
          + f"   is the top-1: {a['top1_is_truth']}")
    print(f"  strata (from x, so an analysis can cut on them): "
          f"ln kt lead = {a['lnkt_lead']:+.2f}   d_boundary = {a['d_boundary']:+.2f}   "
          f"d_floor = {a['d_floor']:+.2f}   n_x = {a['n_x']}   region = {a['region']}")

    # the mode skeleton, node by node, as value +/- posterior width
    _item = ds[i]
    w = node_widths(_item["xf"].unsqueeze(0).to(device),
                    torch.tensor([_item["nx"]], device=device), a["cells_top1"])
    print()
    print("  the mode skeleton's kinematics -- CONDITIONAL summaries, never bare points:")
    print(f"     {'t':>2} {'cell':>6} {'ln(1/dR)':>18} {'ln kt':>18} {'ln z':>18} "
          f"{'psi':>10} {'kappa':>8}")
    for t, c in enumerate(a["cells_top1"]):
        v = rec["mode"][t] if t < len(rec["mode"]) else np.full(4, np.nan)
        if w is None:
            print(f"     {t:>2} {int(c):>6} {v[0]:>18.3f} {v[1]:>18.3f} {v[2]:>18.3f} "
                  f"{v[3]:>10.2f} {'n/a':>8}")
        else:
            print(f"     {t:>2} {int(c):>6} "
                  f"{v[0]:>11.3f} +/-{w[t, 0]:<5.3f} {v[1]:>11.3f} +/-{w[t, 1]:<5.3f} "
                  f"{v[2]:>11.3f} +/-{w[t, 2]:<5.3f} {v[3]:>10.2f} {w[t, 3]:>8.3f}")
    if w is not None and len(a["cells_top1"]):
        print(f"     (widths are the head's own sigmas; psi is quoted with kappa because "
              f"below kappa = {DECODE['kappa_min_mode']:g} its mode is not identified)")

    print()
    print("  mean |delta| to truth, over each series' own paired splittings:")
    print(f"     {'series':<10}{'pairs':>7}" + "".join(f"{TLABEL[k]:>12}" for k in RES_KEYS))
    for s in MODELS:
        r = rec["resid"][s]
        print(f"     {s:<10}{len(r):>7}" + "".join(
            f"{np.abs(r[:, COL[k]]).mean():>12.3f}" if len(r) else f"{'--':>12}"
            for k in RES_KEYS))

    # ---- figure --------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.95, 1.0], hspace=0.48, wspace=0.30)
    axp = fig.add_subplot(outer[0, :2])
    axm = fig.add_subplot(outer[0, 2])
    axs = fig.add_subplot(outer[1, :2])
    axn = fig.add_subplot(outer[1, 2])

    # (a) the Lund plane
    cloud = rec["cloud"]
    if len(cloud):
        axp.scatter(cloud[:, 0], cloud[:, 1], s=7, color=C_POST, alpha=0.16,
                    linewidths=0, zorder=2,
                    label=f"posterior cloud ({len(cloud)} nodes / {len(rec['mults'])} draws)")
    for s in ("rsd", "map", "mbr", "mode", "truth"):
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
    hi = int(max(m.max(), ny, len(rec["rsd"]), len(rec["map"]), len(rec["mbr"]),
                 len(rec["mode"]))) + 1
    axm.hist(m, bins=np.arange(-0.5, hi + 1.0), color=C_POST, alpha=0.5,
             edgecolor=C_POST, linewidth=0.8, label=r"posterior $P(n\,|\,x)$")
    for s, styl in (("truth", (C_TRUTH, "-", 2.2)), ("rsd", (C_RSD_E, "-", 1.4)),
                    ("map", (C_MAP, "--", 1.6)), ("mbr", (C_MBR, ":", 1.8)),
                    ("mode", (C_MODE, "-.", 1.8))):
        c, ls, lw = styl
        axm.axvline(len(rec[s]), color=c, ls=ls, lw=lw,
                    label=f"{STYLE[s][2]} = {len(rec[s])}")
    finish(axm, xlabel="primary splittings $n$", ylabel="draws",
           title="(b) the length belief", legend=True, loc="upper right")

    # (b') the enumerated mass spectrum -- the panel this notebook exists for
    masses = enum.masses
    n_show = min(len(masses), 20)
    xs = np.arange(n_show)
    axs.bar(xs, masses[:n_show], width=0.72, color=C_MODE, edgecolor="none",
            label=r"enumerated $M_i$ (exact, descending)")
    if a["remainder_bound"] > 0:
        axs.bar([n_show + 0.6], [a["remainder_bound"]], width=0.72, color=C_FRONT,
                edgecolor=AXIS, linewidth=0.8, hatch="//",
                label=f"certified remainder $\\leq$ {a['remainder_bound']:.1e}")
    axs.axhline(0.5, color=INK, lw=1.0, ls=":",
                label=r"$M_1 > 1/2$: dominance, self-certifying")
    truth_rank = a["rank_truth"]
    if 1 <= truth_rank <= n_show:
        axs.scatter([truth_rank - 1], [masses[truth_rank - 1]], marker="o", s=70,
                    facecolor="none", edgecolor=C_TRUTH, linewidth=1.8, zorder=6,
                    label=f"the TRUTH skeleton (rank {truth_rank})")
    axs.set_yscale("log")
    axs.set_xticks(list(xs) + ([n_show + 0.6] if a["remainder_bound"] > 0 else []))
    axs.set_xticklabels([str(v + 1) for v in xs] + (["rest"] if a["remainder_bound"] > 0 else []),
                        fontsize=7)
    finish(axs, xlabel="skeleton rank $i$", ylabel=r"$M_i = q_\phi(S_i\,|\,x)$",
           title=f"(c) the skeleton posterior, exactly enumerated   "
                 f"$C_k$ = {a['C_k']:.3f} over {a['n_enumerated']} skeletons",
           legend=True, loc="upper right")

    # (b'') the cumulative coverage
    cum = np.cumsum(masses)
    axn.plot(np.arange(1, len(cum) + 1), cum, color=C_MODE, marker="P", ms=3.4, lw=1.4)
    axn.axhline(1.0, color=INK, lw=1.0, ls=":")
    axn.set_ylim(0.0, 1.04)
    finish(axn, xlabel="skeletons kept $k$", ylabel=r"coverage $C_k$",
           title=f"(d) $1 - C_k \\leq$ {a['remainder_bound']:.1e}")

    # (e-g) the ladders, each over its residual strip
    for c_i, key in enumerate(RES_KEYS):
        inner = outer[2, c_i].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
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
        finish(ax, ylabel=LABEL[key], title=f"({'efg'[c_i]}) {LABEL[key]} ladder",
               legend=(c_i == 0), loc="best")

    fig.suptitle(f"Everything the model says about jet #{i}, including how much of the "
                 f"posterior its top skeleton holds", x=0.006, y=1.004, ha="left")
    plt.show()

    if show_trees:
        print()
        print(lund_tree_str(rec["pe"]["mode"], "model MODE-SKELETON groomed shower "
                           f"(M_1 = {a['M1']:.3f})", geom, ref=y))
        print()
        print(lund_tree_str(rec["pe"]["map"], "model MAP groomed shower", geom, ref=y))
        print()
        print(lund_tree_str(rec["pe"]["mbr"], "model MBR groomed shower (perturbative Lund)",
                            geom, ref=y))
        print()
        print(lund_tree_str(y, "true groomed shower (parton-level y)", geom))
        if rec["pe"]["mode"].n_psi_unidentified:
            print(f"\n* psi drawn, not moded, for "
                  f"{rec['pe']['mode'].n_psi_unidentified} of {len(rec['mode'])} "
                  f"mode-skeleton nodes: the von Mises concentration is below "
                  f"decode.kappa_min_mode={DECODE['kappa_min_mode']:g}, so the mode is not "
                  f"an identified direction.")
    return rec
''')

md(r"""
Call it on any jet index. `SHOWCASE_JET` in §0 pins one; `None` auto-picks the first jet with at
least three truth splittings, so there is a ladder to look at rather than a single point.
""")

code(r'''
seed_everything(SEED)
SHOW = showcase_jet()
''')

# ---------------------------------------------------------------------------
md(r"""
## 5. The evaluation pass

One pass over `N_JETS`: `estimate_jet` per jet, producing both the residual series (§9–§10) and the
audit records (§6–§8). Nothing is sampled twice — the same $K$ draws feed the MAP floor, the MBR
risk, the posterior-draw series and the entropy estimate.

**Why the empty gate is off.** `decode.empty_threshold` decides *whether* a jet has any parton
splitting; this notebook measures *which* skeleton the posterior puts its mass on, and the empty
skeleton is one of the candidates. Applying the gate would pre-empt exactly the decision the
enumeration is measuring. Set `GATE_EMPTY = True` in §0 to apply the frozen $\tau$ to the MAP/MBR
series; the enumeration is untouched by it either way.
""")

md(r"""
### 5a. Cost probe — size the run before committing to it

The audit adds one best-first search and $\le$ `H_DRAWS` teacher-forced scorings per jet on top of
the `per_jets_estimation` cost. Both are sequential single-jet GRU stepping, so the probe below is
the honest estimate — read it before setting `N_JETS`.
""")

code(r'''
_probe = min(20, len(ds))
seed_everything(SEED)
_t0 = time.perf_counter()
_rng = np.random.default_rng(SEED)
_exp = []
for _i in range(_probe):
    _r = estimate_jet(_i, rng=_rng)
    _exp.append(_r["audit"]["n_expansions"])
_dt = (time.perf_counter() - _t0) / max(_probe, 1)
print(f"{_dt * 1e3:7.1f} ms / jet   (K={K_DRAWS}, MBR={MBR_BACKEND!r}, "
      f"audit k={AUDIT['k']}, H_DRAWS={H_DRAWS}, {torch.get_num_threads()} threads)")
print(f"-> N_JETS={N_JETS} is about {_dt * N_JETS / 60:.1f} min")
print(f"search expansions per jet: median {np.median(_exp):.0f}   max {max(_exp)}   "
      f"(budget {AUDIT['budget']}) -- mass concentration is WHY typical jets terminate "
      f"early, so this number is itself a preview of the answer")
''')

md(r"""
### 5b. Run it
""")

code(r'''
N = min(N_JETS, len(ds))
seed_everything(SEED)
_rng = np.random.default_rng(SEED)
_t0 = time.perf_counter()
RAW = {s: [] for s in SERIES}
W_JET, Q0, AUDITS = [], [], []
for i in range(N):
    r = estimate_jet(i, rng=_rng)
    for s in SERIES:
        RAW[s].append(r[s])
    W_JET.append(r["weight"])
    Q0.append(r["q0"])
    AUDITS.append(r["audit"])
W_JET, Q0 = np.array(W_JET), np.array(Q0)
print(f"evaluated {N:,} jets in {(time.perf_counter() - _t0) / 60:.2f} min "
      f"(K={K_DRAWS} draws each, audit k={AUDIT['k']})")

NSPL = {s: np.array([len(a) for a in RAW[s]]) for s in SERIES}
print()
print(f"{'series':<8}{'splittings':>12}{'mean mult':>12}{'P(n=0)':>10}")
for s in SERIES:
    print(f"{s:<8}{int(NSPL[s].sum()):>12,}{NSPL[s].mean():>12.3f}"
          f"{float(W_JET[NSPL[s] == 0].sum() / W_JET.sum()):>10.3f}")

# The run-level summary, computed by the SAME function that writes the CLI artifact --
# so a number quoted from this notebook and one read out of `mode_audit.json` cannot
# disagree.
SUMMARY = summarise_mode_audit(AUDITS, thresholds=THRESHOLDS, audit=AUDIT,
                               kind=SEARCH_KIND, groom=GROOM, K=H_DRAWS, verbose=True)
''')

# ---------------------------------------------------------------------------
md(r"""
## 6. Question one — *does* a dominant skeleton exist?

$M_1$ is the exact mass of the single most likely skeleton. $F(m) = \mathrm{frac}(M_1 \ge m)$ is the
fraction of jets whose posterior puts at least $m$ of its mass on one discrete configuration, quoted
at the pre-registered $m \in \{0.3, 0.5, 0.7\}$ with 95% Wilson intervals.

Two things to keep straight while reading it.

- **$M_1 > 1/2$ is self-certifying.** The total mass is 1, so nothing else can be larger, whatever
  the search pruned. Below $1/2$ the reported $M_1$ is a *lower* bound on the true top-1 mass (the
  search always keeps the highest-mass child of every expansion), so $F(m)$ is a **lower bound** on
  the true fraction — never an over-claim.
- **The `certified` rate travels with every number.** A jet that exhausted the expansion budget
  before $k$ completions carries `certified = False`; its $M_1$ is still a valid lower bound, but its
  *ranking* claim is not, so the rate is printed beside every fraction rather than in a footnote.
- **Every number in this section is a *same-geometry* number.** A cell's mass is $\approx$ density
  $\times$ area, so $F(m)$ scales with `n_bins`; it compares checkpoints at one geometry and says
  nothing grid-free. §6a is the companion that does — read it before quoting anything here.

The strata are the axes on which $\sigma = \sigma_0 + \Lambda_{\rm eff}/k_t$ predicts the
dominant/fragmented mixture to separate — all three read off $x$, so a data analysis can make the
same cut.
""")

code(r'''
M1   = np.array([a["M1"] for a in AUDITS])
M2   = np.array([a["M2"] for a in AUDITS])
CERT = np.array([a["certified"] for a in AUDITS], dtype=bool)
CERT1 = np.array([a["certified_top1"] for a in AUDITS], dtype=bool)
CK   = np.array([a["C_k"] for a in AUDITS])
LNKT = np.array([a["lnkt_lead"] for a in AUDITS])
D_B  = np.array([a["d_boundary"] for a in AUDITS])
D_F  = np.array([a["d_floor"] for a in AUDITS])
N_X  = np.array([a["n_x"] for a in AUDITS], dtype=float)
REGION = np.array([a["region"] or "none" for a in AUDITS])
RANK = np.array([a["rank_truth"] for a in AUDITS], dtype=int)
M_TRUTH = np.array([a["M_truth"] for a in AUDITS])
TOP1_TRUE = np.array([a["top1_is_truth"] for a in AUDITS], dtype=bool)
H_HAT = np.array([a["H_hat"] for a in AUDITS])
EFF = np.array([a["eff_skeletons"] for a in AUDITS])
DOMINANT = M1 >= DOMINANT_M
# The grid-FREE half of q(S|x) = q(N|x) q(cells|N,x). Only the second factor carries the
# cell-area scaling, so these three numbers mean the same thing at any n_bins.
QN0 = np.array([a.get("qN_0", np.nan) for a in AUDITS])
QN1 = np.array([a.get("qN_1", np.nan) for a in AUDITS])
QN2 = np.array([a.get("qN_ge2", np.nan) for a in AUDITS])
# ...and section 6a's region, per jet.
HPD50 = np.array([a.get("hpd_area_50", np.nan) for a in AUDITS])
HPD90 = np.array([a.get("hpd_area_90", np.nan) for a in AUDITS])
HPD_CELLS = np.array([a.get("hpd_cells_50", np.nan) for a in AUDITS])
HPD_SIG = np.array([a.get("hpd_over_sigma_box_50", np.nan) for a in AUDITS])
SIG_U = np.array([a.get("sigma_u", np.nan) for a in AUDITS])
SIG_V = np.array([a.get("sigma_v", np.nan) for a in AUDITS])
SATURATED = np.array([bool(a.get("truncation_saturated", False)) for a in AUDITS])
# The one candidate skeleton that is special: the EMPTY tree. "The posterior is sure
# there is nothing there" and "the posterior is sure WHICH splitting is there" are both
# dominance, and they are different physical claims -- so they are separated everywhere
# below rather than pooled into one F(m).
EMPTY_TOP1 = np.array([len(a["cells_top1"]) == 0 for a in AUDITS], dtype=bool)
EMPTY_TRUTH = np.array([a["n_truth"] == 0 for a in AUDITS], dtype=bool)

# the pre-registered strata, one boolean mask each
def _hi(v):
    finite = v[np.isfinite(v)]
    med = float(np.median(finite)) if finite.size else float("nan")
    return ((v >= med) | ~np.isfinite(v)), med


HI_KT, MED_KT = _hi(LNKT)
HI_B, MED_B = _hi(D_B)
HI_F, MED_F = _hi(D_F)
MIX = HI_KT & HI_B & HI_F      # the "perturbative" stratum of the plan's section 7.4
# Strata marked (truth) are DECOMPOSITIONS, not cuts: they use the parton answer and no
# analysis could make them on data. Everything else is computable from x alone.
STRATA = [
    ("all jets", np.ones(len(M1), dtype=bool)),
    ("mode is a splitting", ~EMPTY_TOP1),
    ("mode is the empty tree", EMPTY_TOP1),
    (f"ln kt lead >= {MED_KT:.2f}", HI_KT),
    (f"ln kt lead <  {MED_KT:.2f}", ~HI_KT),
    (f"d_boundary >= {MED_B:.2f}", HI_B),
    (f"d_floor    >= {MED_F:.2f}", HI_F),
    ("perturbative (all three)", MIX),
    ("its complement", ~MIX),
    ("truth non-empty (truth)", ~EMPTY_TRUTH),
    ("truth empty (truth)", EMPTY_TRUTH),
]
# Panel (b) draws the strata that carry an argument; the rest go in as thin references.
STRATUM_STYLE = {
    "all jets": (INK, 2.0, "-"),
    "mode is a splitting": (C_MODE, 2.0, "-"),
    "mode is the empty tree": (C_MODE, 1.4, "--"),
    "perturbative (all three)": (C_MAP, 2.0, "-"),
    "its complement": (C_MBR, 2.0, "-"),
}

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

# (a) the M1 distribution
e_m1 = np.linspace(0.0, 1.0, 41)
axes[0].hist(M1, bins=e_m1, color=C_MODE, alpha=0.55, edgecolor=C_MODE, linewidth=0.9,
             label=f"all {len(M1):,} jets")
axes[0].hist(M1[MIX], bins=e_m1, histtype="step", color=C_MAP, linewidth=1.8,
             label=f"perturbative stratum ({MIX.sum():,})")
axes[0].axvline(DOMINANT_M, color=INK, ls=":", lw=1.2,
                label=rf"$M_1 = {DOMINANT_M:g}$ (self-certifying above)")
finish(axes[0], xlabel=r"$M_1 = q_\phi(S_1\,|\,x)$", ylabel="jets",
       title=f"(a) the top-1 skeleton mass   median {np.median(M1):.3f}, "
             f"p90 {np.percentile(M1, 90):.3f}", legend=True, loc="upper center")

# (b) F(m), the survival curve, overall and per stratum
ms = np.linspace(0.0, 1.0, 101)
for lab, sel in STRATA:
    if sel.sum() < MIN_CI_JETS or lab.endswith("(truth)"):
        continue
    c, lw, ls = STRATUM_STYLE.get(lab, (MUTED, 1.0, ":"))
    axes[1].plot(ms, [(M1[sel] >= m).mean() for m in ms], color=c, lw=lw, ls=ls,
                 label=f"{lab}  (n={int(sel.sum()):,})")
for t in THRESHOLDS:
    axes[1].axvline(t, color=GRID, lw=0.8, zorder=0)
axes[1].set_ylim(0.0, 1.02)
finish(axes[1], xlabel=r"$m$", ylabel=r"$F(m) = \mathrm{frac}(M_1 \geq m)$",
       title="(b) how often the posterior concentrates", legend=True, loc="upper right")

# (c) M1 against the leading emission's hardness -- the predicted trend
axes[2].scatter(LNKT, M1, s=6, color=C_MODE, alpha=0.28, linewidths=0)
_ok = np.isfinite(LNKT)
if _ok.sum() > 20:
    _q = np.quantile(LNKT[_ok], np.linspace(0, 1, 9))
    _c, _mid = [], []
    for lo, hi in zip(_q[:-1], _q[1:]):
        sel = _ok & (LNKT >= lo) & (LNKT <= hi)
        if sel.sum():
            _c.append(np.median(M1[sel]))
            _mid.append(0.5 * (lo + hi))
    axes[2].plot(_mid, _c, color=INK, lw=1.8, marker="o", ms=4, label="median per octile")
axes[2].axhline(DOMINANT_M, color=INK, ls=":", lw=1.0)
axes[2].set_ylim(0.0, 1.02)
finish(axes[2], xlabel=r"leading-emission $\ln(k_t/\mathrm{GeV})$ of $x$", ylabel=r"$M_1$",
       title=r"(c) dominance against hardness   "
             rf"(Spearman {SUMMARY['spearman_M1_vs']['lnkt_lead']:+.3f}, predicted $+$)",
       legend=True, loc="lower right")

fig.suptitle(r"Question one: does a dominant parton skeleton exist? "
             r"($M_1$ is EXACT, not an estimate)", x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

print(f"{'stratum':<28}{'jets':>7}{'M1 p50':>9}{'M1 p90':>9}"
      + "".join(f"{('F(' + f'{t:g}' + ')'):>20}" for t in THRESHOLDS)
      + f"{'certified':>11}")
for lab, sel in STRATA:
    n = int(sel.sum())
    if not n:
        continue
    cells = ""
    for t in THRESHOLDS:
        k = int((M1[sel] >= t).sum())
        lo, hi = wilson(k, n)
        cells += f"{k / n:>8.3f} [{lo:.2f},{hi:.2f}]"
    print(f"{lab:<28}{n:>7,}{np.median(M1[sel]):>9.3f}"
          f"{np.percentile(M1[sel], 90):>9.3f}{cells}{CERT[sel].mean():>11.1%}")
print("\n95% Wilson intervals. `certified` is the fraction whose whole top-k list is "
      "proved exact;\nan uncertified M_1 is still a LOWER bound, so every F(m) above is a "
      "lower bound too.")
print(f"M_1 > 1/2 needs no certificate at all: {int((M1 > 0.5).sum()):,} jets "
      f"({(M1 > 0.5).mean():.1%}) are dominant by proof.")
print(f"\ncoverage: median C_k = {np.median(CK):.4f} at k = {AUDIT['k']}   "
      f"(median certified remainder "
      f"{np.median([a['remainder_bound'] for a in AUDITS]):.2e})")

# WHICH skeleton wins matters as much as how much mass it holds, and one candidate is
# special: the EMPTY tree. No point estimator under the default decode can produce it
# (the argmax over n lands at 0 essentially never, and MBR's imbalance term prices an
# empty cloud at near-maximal risk), so if the posterior's mode is frequently empty, that
# is a fact about the model that the MAP/MBR tables structurally cannot show.
_lo, _hi = wilson(int(EMPTY_TOP1.sum()), len(EMPTY_TOP1))
print(f"\nthe EMPTY skeleton is the posterior's mode for {EMPTY_TOP1.mean():.1%} of jets "
      f"[{_lo:.3f}, {_hi:.3f}]\n  against a truth empty rate of {EMPTY_TRUTH.mean():.1%}; "
      f"of the jets whose mode is empty, "
      f"{(EMPTY_TRUTH & EMPTY_TOP1).sum() / max(EMPTY_TOP1.sum(), 1):.1%} really are.")
print(f"  Read that as a statement about the GRID, not about emptiness. The empty "
      f"skeleton is the only\n  configuration the cell grid does not slice: its mass is "
      f"cell-free, while every N>=1 skeleton's\n  is ~ density x area. Section 6a "
      f"measures the slicing directly.")
print(f"    - the model does NOT believe the jet is empty. Its own length belief here is "
      f"q(N=0) = {np.nanmean(QN0):.3f}\n      against q(N>=1) = "
      f"{1.0 - np.nanmean(QN0):.3f}"
      + (f"; q(N=1) > q(N=0) for {np.nanmean(QN1 > QN0):.1%} of jets."
         if np.isfinite(QN1).any() else ".")
      + " That factor of q(S|x) = q(N|x) q(cells|N,x)\n      is grid-free, and it says "
        "the opposite of what the mode does.")
print(f"    - 'the posterior concentrates on ONE configuration' and 'the posterior "
      f"concentrates on a\n      REGION' are different claims. On the "
      f"{int((~EMPTY_TOP1).sum()):,} jets whose mode IS a splitting, "
      f"F({DOMINANT_M:g}) = {(M1[~EMPTY_TOP1] >= DOMINANT_M).mean():.3f} "
      f"and median M_1 = {np.median(M1[~EMPTY_TOP1]):.3f};\n      on the rest, "
      f"F({DOMINANT_M:g}) = {(M1[EMPTY_TOP1] >= DOMINANT_M).mean():.3f}, median "
      f"{np.median(M1[EMPTY_TOP1]):.3f}. The strata table above splits them.")
print(f"    - the mode SERIES of section 9 contributes no splitting on those jets, so its "
      f"residual is\n      measured on the complement -- selected, not typical.")
''')

# ---------------------------------------------------------------------------
md(r"""
### 6a. The same posterior, measured without the grid

§6 asked *how much mass sits on one cell*. That is $\approx$ density $\times$ cell area, so it is
as much a statement about `n_bins` as about the model. Here is the grid-free question about the
same object: **how large a region does the first splitting's posterior actually occupy?**

The density over the first node's position is the mixture
$\sum_c P_{\rm split}(c\mid h_0,e)\,\mathrm{TN}(du\mid c)\,\mathrm{TN}(dv\mid c)$, and every
component is supported on *its own cell and nowhere else* — the coordinate head's truncated
normals are bounded by construction. So the mixture is block-wise, a sub-grid inside each cell
evaluates it exactly, and the $\alpha$-highest-density region is read off by sorting. The area
comes out in $\ln(1/\Delta R)\times\ln k_t$ units and has a limit as the grid refines.

Three numbers make it readable:

- $\sqrt{\text{area}}$ — an effective **linear** scale, directly comparable to the cell width, to
  the residual widths of §9, and to the non-perturbative smearing $\sigma_0+\Lambda_{\rm eff}/k_t$.
- **area / $\pm1\sigma$ box** of the coordinate head at the modal cell — is the spread wider than
  the width the model claims for itself?
- **truncation saturation**: is the head's $\sigma$ *larger than half a cell*? If it is, the
  within-cell density is nearly uniform, the head cannot express its own width inside a cell, and
  the model is carrying its coordinate uncertainty in the **cell distribution** instead. In that
  regime a small $M_1$ says *the grid is finer than the model's resolution* — it is not evidence
  that the posterior is fragmented, and §6's headline must not be read as if it were.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

_ok = np.isfinite(HPD50)
if not _ok.any():
    print("no positional region available -- this family has no continuous coordinate "
          "head, so section 6a does not apply")
else:
    # (a) the region, as a linear scale, against the two yardsticks that matter
    _lin = np.sqrt(HPD50[_ok])
    axes[0].hist(_lin, bins=40, color=C_MODE, alpha=0.6, edgecolor=C_MODE, linewidth=0.8,
                 label=f"50% region, median {np.median(_lin):.2f}")
    axes[0].axvline(2 * geom.half_u, color=INK, ls=":", lw=1.4,
                    label=f"one cell = {2 * geom.half_u:.2f}")
    axes[0].axvline(np.nanmedian(2 * SIG_U), color=C_MAP, ls="--", lw=1.4,
                    label=rf"head's $2\sigma_u$ = {np.nanmedian(2 * SIG_U):.2f}")
    finish(axes[0], xlabel=r"$\sqrt{\mathrm{area}}$   [ln units]", ylabel="jets",
           title="(a) how wide the first splitting's posterior is", legend=True,
           loc="upper right")

    # (b) how many CELLS that region spans -- the bridge between 6 and 6a
    axes[1].hist(HPD_CELLS[_ok], bins=40, color=C_MAP, alpha=0.6, edgecolor=C_MAP,
                 linewidth=0.8)
    axes[1].axvline(1.0, color=INK, ls=":", lw=1.4, label="one cell")
    axes[1].axvline(np.median(HPD_CELLS[_ok]), color=C_MODE, lw=1.6,
                    label=f"median {np.median(HPD_CELLS[_ok]):.0f} cells")
    axes[1].set_xscale("log")
    finish(axes[1], xlabel="cells spanned by the 50% region", ylabel="jets",
           title=f"(b) the grid slices it into this many pieces   "
                 f"(of {geom.n_cells})", legend=True, loc="upper right")

    # (c) M_1 against that count -- the claim that F(m) is a resolution readout
    axes[2].scatter(HPD_CELLS[_ok], M1[_ok], s=7, color=C_MODE, alpha=0.3, linewidths=0)
    _x = np.logspace(np.log10(max(np.nanmin(HPD_CELLS[_ok]), 0.5)),
                     np.log10(np.nanmax(HPD_CELLS[_ok])), 50)
    axes[2].plot(_x, 1.0 / _x, color=INK, lw=1.4, ls="--",
                 label=r"$M_1 \propto 1/\mathrm{cells}$")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    finish(axes[2], xlabel="cells spanned by the 50% region", ylabel=r"$M_1$",
           title="(c) the mode mass IS a resolution readout", legend=True,
           loc="lower left")

fig.suptitle(r"6a. the same posterior with the grid divided out — these numbers survive "
             r"a change of $n_\mathrm{bins}$", x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

if _ok.any():
    print(f"{'quantity':<44}{'median':>12}{'p90':>12}")
    for lab, v in (("50% region area  [ln^2]", HPD50),
                   ("  its linear scale sqrt(area)  [ln]", np.sqrt(HPD50)),
                   ("90% region area  [ln^2]", HPD90),
                   ("cells spanned by the 50% region", HPD_CELLS),
                   ("area / the head's own +-1 sigma box", HPD_SIG)):
        f = v[np.isfinite(v)]
        print(f"{lab:<44}{np.median(f):>12.3f}{np.percentile(f, 90):>12.3f}")
    print(f"{'head width sigma_u at the modal cell':<44}"
          f"{np.nanmedian(SIG_U):>12.3f}{np.nanpercentile(SIG_U, 90):>12.3f}")
    print(f"{'head width sigma_v at the modal cell':<44}"
          f"{np.nanmedian(SIG_V):>12.3f}{np.nanpercentile(SIG_V, 90):>12.3f}")
    print(f"\ncell width {2 * geom.half_u:.3f} x {2 * geom.half_v:.3f}, "
          f"half-cell {geom.half_u:.3f}")
    print(f"TRUNCATION-SATURATED (sigma > half-cell on BOTH axes) for "
          f"{SATURATED.mean():.1%} of jets.")
    if SATURATED.mean() > 0.5:
        print("  -> the head wants to be wider than a cell and the truncation forbids it,")
        print("     so the within-cell density is nearly uniform and the model carries its")
        print("     coordinate uncertainty in the CELL distribution. At this geometry the")
        print("     grid is FINER than the model's own resolution, and section 6's small")
        print("     M_1 is that fact, not a fragmented posterior.")
    print(f"\nthe grid-free length belief, for comparison: q(N=0) = {np.nanmean(QN0):.3f}   "
          f"q(N=1) = {np.nanmean(QN1):.3f}   q(N>=2) = {np.nanmean(QN2):.3f}")
    print(f"  q(N=1|x) > q(N=0|x) for {np.nanmean(QN1 > QN0):.1%} of jets -- the model "
          f"believes there IS a splitting;")
    print(f"  it just spreads that belief over ~{np.median(HPD_CELLS[_ok]):.0f} cells, "
          f"which is why no single skeleton holds much.")
''')

# ---------------------------------------------------------------------------
md(r"""
### 6b. $M_1(r)$ — the mode mass with its resolution *named*

§6a fixed the mass and reported the size. Here is the same content read the other way, which
is the one that gives back a quotable probability: **the largest mass the posterior puts in any
box of half-width $r$**,

$$M_1(r)\;=\;\max_{(u,v)}\ \int_{\lVert p-(u,v)\rVert_\infty<r} q(p\mid x)\,\mathrm{d}^2p .$$

Two properties earn it its place. It is a **probability of a stated event** — "the leading
splitting lies within $\pm r$ of here" — so the dominance sentence survives. And the window
**slides**, so no partition origin enters: coarsening the grid into blocks would not do this, because
a blob straddling a block boundary is split by the coarse grid exactly as it was by the fine one.
Only the *scale* is a choice, and it is a choice a physicist can defend.

Read the curve, not a point of it:

- $M_1(r)\propto r^2$ at small $r$ — the regime where the number is measuring the resolution
  element (density $\times$ area) and nothing about the model. **`M₁` as §6 reports it lives here.**
- a **knee** at the posterior's own scale;
- saturation at 1 when the box swallows the plane.

Two vertical marks make the argument: $r=$ half a cell (what §6's $F(m)$ is built on) and
$r=\sigma$ of the coordinate head (what the model claims for itself). The gap between the two
readings *is* the grid artefact.

The last panel is the sequence-level version of the same move — the enumerated skeletons regrouped
by coarse cell label. That one can only be a **lower bound**: summing fine skeletons that share a
coarse label does not factorise for $N\ge2$, because the decoder state depends on the fine cell, so
the futures differ within a block (the label-sum problem, NP-hard in general). It tightens as $k$
grows, and a coarse mass above $\tfrac12$ is dominant *by proof* regardless, since coarse labels
partition the space.
""")

code(r'''
from h2p_rsd_junipr.inference.mode_audit import RESOLUTION_RADII

CURVES = np.array([a["m1_curve"] for a in AUDITS if a.get("m1_curve")], dtype=float)
R_GRID = np.array(RESOLUTION_RADII, dtype=float)
M1_RCELL = np.array([a.get("m1_at_r_cell", np.nan) for a in AUDITS])
M1_RSIG = np.array([a.get("m1_at_r_sigma", np.nan) for a in AUDITS])
R_SIG = np.array([a.get("r_sigma", np.nan) for a in AUDITS])
COARSE_GAIN = np.array([a.get("coarse_gain", np.nan) for a in AUDITS])
COARSE_CERT = np.array([bool(a.get("coarse_certified", False)) for a in AUDITS])

if not len(CURVES):
    print("no positional density -- section 6b does not apply to this family")
else:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    med = np.median(CURVES, axis=0)
    lo, hi = np.percentile(CURVES, [16, 84], axis=0)
    r_cell = float(geom.half_u)
    r_sig = float(np.nanmedian(R_SIG))

    # (a) the curve itself, log-log, with the r^2 slope that IS the artefact
    axes[0].fill_between(R_GRID, lo, hi, color=C_MODE, alpha=0.18, linewidth=0,
                         label="16-84% of jets")
    axes[0].plot(R_GRID, med, color=C_MODE, lw=2.0, marker="P", ms=4,
                 label=r"median $M_1(r)$")
    _ref = med[2] * (R_GRID / R_GRID[2]) ** 2
    axes[0].plot(R_GRID, np.minimum(_ref, 1.0), color=INK, ls="--", lw=1.2,
                 label=r"$\propto r^2$ (pure resolution)")
    axes[0].axvline(r_cell, color=C_MAP, ls=":", lw=1.6,
                    label=f"half a cell = {r_cell:.2f}")
    axes[0].axvline(r_sig, color=C_MBR, ls=":", lw=1.6,
                    label=rf"head's $\sigma$ = {r_sig:.2f}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    finish(axes[0], xlabel=r"$r$   [ln units]", ylabel=r"$M_1(r)$",
           title="(a) the mode mass against the resolution it is quoted at",
           legend=True, loc="upper left")

    # (b) the PRE-REGISTERED thresholds, as functions of r
    for t, c in zip(THRESHOLDS, (MUTED, INK, C_MAP)):
        axes[1].plot(R_GRID, (CURVES >= t).mean(axis=0), color=c, lw=1.8, marker="o",
                     ms=3.4, label=rf"$F({t:g})$ at resolution $r$")
    axes[1].axvline(r_cell, color=C_MAP, ls=":", lw=1.6)
    axes[1].axvline(r_sig, color=C_MBR, ls=":", lw=1.6)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.02)
    finish(axes[1], xlabel=r"$r$   [ln units]", ylabel=r"$\mathrm{frac}(M_1(r) \geq m)$",
           title="(b) section 6's headline, as a function of the resolution",
           legend=True, loc="upper left")

    # (c) the sequence-level coarsening, on the jets where it can do anything
    _ne = (~EMPTY_TOP1) & np.isfinite(COARSE_GAIN)
    if _ne.sum():
        axes[2].hist(COARSE_GAIN[_ne], bins=30, color=C_MODE, alpha=0.6,
                     edgecolor=C_MODE, linewidth=0.8)
        axes[2].axvline(1.0, color=INK, ls=":", lw=1.4, label="no gain")
        axes[2].axvline(np.median(COARSE_GAIN[_ne]), color=C_MAP, lw=1.6,
                        label=f"median {np.median(COARSE_GAIN[_ne]):.2f}x")
    finish(axes[2], xlabel=r"coarse $M_1$ / fine $M_1$", ylabel="jets",
           title=f"(c) whole-sequence {int(AUDITS[0].get('coarse_block', 3))}x"
                 f"{int(AUDITS[0].get('coarse_block', 3))} coarsening, lower bound   "
                 f"({int(_ne.sum())} jets with a non-empty mode)",
           legend=bool(_ne.sum()), loc="upper right")

    fig.suptitle(r"6b. name the resolution and the mode mass comes back as a quotable "
                 r"probability", x=0.006, y=1.005, ha="left")
    fig.tight_layout()
    plt.show()

    print(f"{'r [ln]':>9}" + "".join(f"{r:>8.3f}" for r in R_GRID))
    print(f"{'M_1(r)':>9}" + "".join(f"{v:>8.3f}" for v in med))
    for t in THRESHOLDS:
        print(f"{'F(' + f'{t:g}' + ')':>9}"
              + "".join(f"{v:>8.3f}" for v in (CURVES >= t).mean(axis=0)))
    print()
    print(f"at r = half a cell   ({r_cell:.3f}): median M_1 = "
          f"{np.nanmedian(M1_RCELL):.4f}   <- the resolution section 6 inherited")
    print(f"at r = head's sigma  ({r_sig:.3f}): median M_1 = "
          f"{np.nanmedian(M1_RSIG):.4f}   <- the resolution the model claims")
    _i50 = int(np.argmin(np.abs(med - 0.5)))
    print(f"the median jet reaches M_1 = 0.5 at r ~ {R_GRID[_i50]:.2f} ln, i.e. the "
          f"leading splitting is\nlocalised to about +/-{R_GRID[_i50]:.2f} in "
          f"(ln 1/dR, ln kt) with even odds. THAT is the honest\ndominance statement, and "
          f"it is the same number at any n_bins.")
    if _ne.sum():
        print(f"\nwhole-sequence coarsening ({int(AUDITS[0].get('coarse_block', 3))}x"
              f"{int(AUDITS[0].get('coarse_block', 3))} blocks): median "
              f"{np.median(COARSE_GAIN[_ne]):.2f}x the fine M_1 on the {int(_ne.sum())} "
              f"jets whose\n  mode is a splitting; certified coarse-dominant (> 1/2, so "
              f"proof regardless of what the\n  search left unexplored) for "
              f"{COARSE_CERT.mean():.1%} of all jets. This is a LOWER bound and it\n  "
              f"tightens with k -- the enumeration covers C_k = {np.median(CK):.2f} of "
              f"the mass, so most of a\n  coarse label's members were never enumerated.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 7. Question two — is the dominant skeleton the *true* one?

Independent of §6, and deliberately in its own section. The truth skeleton's mass
$M_{\rm truth} = q_\phi(S_{\rm truth}\mid x)$ is computed by the same teacher-forced scorer the
search uses, so it lives on the same scale as $M_1$ and the two are directly comparable; its **rank**
comes from the enumeration.

Read the two questions together only in the four-way table at the end: dominant-and-right,
dominant-and-wrong, diffuse-and-right, diffuse-and-wrong. A model that is mostly *dominant and
wrong* is a worse instrument than one that is mostly diffuse, and only that cross shows it.

`rank = 0` means the truth was **outside** the enumerated top-$k$ — a statement about $k$ and about
the truth's mass, not a proof that it is impossible.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

# (a) truth mass against top-1 mass
axes[0].scatter(M1, M_TRUTH, s=7, color=C_MODE, alpha=0.3, linewidths=0)
_lim = [0.0, 1.0]
axes[0].plot(_lim, _lim, color=INK, lw=1.0, ls=":", label=r"$M_{\rm truth} = M_1$ (the top-1 IS the truth)")
axes[0].set_xlim(0, 1.02)
axes[0].set_ylim(0, 1.02)
finish(axes[0], xlabel=r"$M_1$", ylabel=r"$M_{\rm truth}$",
       title=f"(a) the truth's own mass   "
             f"(on the diagonal for {TOP1_TRUE.mean():.1%} of jets)",
       legend=True, loc="upper left")

# (b) the rank of the truth
_maxr = int(AUDIT["k"])
_bins = np.arange(0.5, min(_maxr, 20) + 1.5)
axes[1].hist(RANK[RANK >= 1], bins=_bins, color=C_MODE, alpha=0.6, edgecolor=C_MODE,
             linewidth=0.8, label="rank inside the enumeration")
_out = int((RANK == 0).sum())
axes[1].axhline(0, color=AXIS, lw=0.8)
finish(axes[1], xlabel=r"rank of $S_{\rm truth}$", ylabel="jets",
       title=f"(b) where the truth sits   "
             f"({_out:,} jets = {_out / len(RANK):.1%} outside the top-{_maxr})",
       legend=True, loc="upper right")

# (c) correctness against dominance -- the cross, as fractions with Wilson bars
cats = [("dominant & true", DOMINANT & TOP1_TRUE),
        ("dominant, wrong", DOMINANT & ~TOP1_TRUE),
        ("diffuse & true", ~DOMINANT & TOP1_TRUE),
        ("diffuse, wrong", ~DOMINANT & ~TOP1_TRUE)]
vals = [float(sel.mean()) for _l, sel in cats]
errs = np.array([[v - wilson(int(sel.sum()), len(M1))[0],
                  wilson(int(sel.sum()), len(M1))[1] - v]
                 for v, (_l, sel) in zip(vals, cats)]).T
axes[2].bar(range(4), vals, width=0.66, color=[C_MODE, C_MBR, C_POST, MUTED],
            edgecolor="none")
axes[2].errorbar(range(4), vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=1.0,
                 capsize=3)
axes[2].set_xticks(range(4))
axes[2].set_xticklabels([lab for lab, _s in cats], fontsize=7.5)
for xi, v in enumerate(vals):
    axes[2].annotate(f"{v:.1%}", (xi, v), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=7.5, color=INK_2)
finish(axes[2], ylabel="fraction of jets",
       title=rf"(c) the cross   (dominant $\equiv M_1 \geq {DOMINANT_M:g}$)")

fig.suptitle(r"Question two: is it the TRUE skeleton? — logically independent of question one",
             x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

print(f"{'stratum':<28}{'jets':>7}{'truth=top1':>22}{'in top-3':>22}{'in top-10':>22}"
      f"{'med M_truth':>13}")
for lab, sel in STRATA:
    n = int(sel.sum())
    if not n:
        continue
    row = ""
    for mask in (TOP1_TRUE[sel], ((RANK[sel] >= 1) & (RANK[sel] <= 3)),
                 ((RANK[sel] >= 1) & (RANK[sel] <= 10))):
        k = int(mask.sum())
        lo, hi = wilson(k, n)
        row += f"{k / n:>10.3f} [{lo:.2f},{hi:.2f}]"
    print(f"{lab:<28}{n:>7,}{row}{np.median(M_TRUTH[sel]):>13.4g}")
print("\nDominance (section 6) and correctness (here) are independent statements. A jet can")
print("hold 90% of its mass on one skeleton that is not the truth -- panel (c) is the only")
print("place the two are read together, and it is a cross, not a single score.")
if EMPTY_TRUTH.any():
    print(f"\nthe empty tree, both ways round -- read these two lines together or neither:")
    print(f"  of the {int(EMPTY_TRUTH.sum()):,} jets whose parton truth IS empty, the empty "
          f"skeleton is the top-1 for {TOP1_TRUE[EMPTY_TRUTH].mean():.1%}\n    (median "
          f"q(0|x) = {np.median(M_TRUTH[EMPTY_TRUTH]):.3f}) -- the answer no point estimator "
          f"under the default decode can give.")
    print(f"  of the {int((~EMPTY_TRUTH).sum()):,} jets with a REAL parton splitting, the "
          f"top-1 skeleton is the truth for {TOP1_TRUE[~EMPTY_TRUTH].mean():.1%}\n    "
          f"(median M_truth = {np.median(M_TRUTH[~EMPTY_TRUTH]):.2e}, truth in the top-10 "
          f"for {(((RANK >= 1) & (RANK <= 10))[~EMPTY_TRUTH]).mean():.1%}).")
    print(f"  The pooled 'truth = top-1' rate of {TOP1_TRUE.mean():.1%} is a MIXTURE of "
          f"those two, and quoting it\n  alone would credit the model on the empty class "
          f"for a discrete decision the non-empty\n  class never gets right at the same "
          f"rate. Neither number is wrong; the pooled one is\n  uninformative.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 8. How many skeletons is the posterior really spread over?

$\hat H = -\frac1K\sum_k \log q_\phi(S^{(k)}\mid x)$ over posterior draws is an **unbiased**
estimator of the skeleton-marginal entropy $H(S\mid x)$ — the draws come from $q_\phi(S\mid x)$
itself — and $e^{\hat H}$ is the *effective number of skeletons* (the typical-set reading; Cover &
Thomas, ch. 3). The enumerated $-\sum_i M_i\log M_i$ is its certified **lower** bound: the tail the
enumeration did not reach can only add entropy.

The Spearman correlations are the plan's §7.3 pre-registered signs. They are quoted as rank
correlations rather than fits because the prediction is about *ordering* — more perturbative, more
dominant — which is what survives the monotone reparametrisations ($k_t$ vs $\ln k_t$) the physics
does not fix.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

_h = H_HAT[np.isfinite(H_HAT)]
axes[0].hist(_h, bins=40, color=C_MODE, alpha=0.6, edgecolor=C_MODE, linewidth=0.8)
axes[0].axvline(np.median(_h), color=INK, lw=1.4, ls="-",
                label=f"median {np.median(_h):.3f} nat")
finish(axes[0], xlabel=r"$\hat H$  [nat]", ylabel="jets",
       title=rf"(a) skeleton entropy, {H_DRAWS} draws/jet", legend=True, loc="upper right")

_e = EFF[np.isfinite(EFF)]
axes[1].hist(_e, bins=np.linspace(1.0, max(2.0, np.percentile(_e, 99)), 40),
             color=C_MAP, alpha=0.6, edgecolor=C_MAP, linewidth=0.8)
axes[1].axvline(np.median(_e), color=INK, lw=1.4,
                label=f"median {np.median(_e):.2f} skeletons")
finish(axes[1], xlabel=r"$e^{\hat H}$  (effective skeletons)", ylabel="jets",
       title="(b) how many configurations the posterior really holds", legend=True,
       loc="upper right")

axes[2].scatter(EFF, M1, s=6, color=C_MODE, alpha=0.25, linewidths=0)
axes[2].set_xscale("log")
axes[2].axhline(DOMINANT_M, color=INK, ls=":", lw=1.0)
finish(axes[2], xlabel=r"$e^{\hat H}$", ylabel=r"$M_1$",
       title=r"(c) the two views of the same concentration")

fig.suptitle("How spread out is the skeleton posterior?", x=0.006, y=1.005, ha="left")
fig.tight_layout()
plt.show()

sp = SUMMARY["spearman_M1_vs"]
print(f"{'Spearman(M_1, .)':<22}{'value':>9}{'predicted':>11}   verdict")
for k in ("lnkt_lead", "d_boundary", "d_floor", "n_x"):
    v, want = sp[k], sp["predicted_sign"][k]
    agree = ("--" if not np.isfinite(v) else
             "as predicted" if (v > 0) == (want == "+") else "AGAINST the prediction")
    print(f"{k:<22}{v:>+9.3f}{want:>11}   {agree}")
print("\nsigma = sigma_0 + Lambda_eff/kt predicts + for the three hardness/distance axes")
print("(a harder, further-from-the-boundary jet is less smeared, so more concentrated) and")
print("- for n_x (more hadron structure, more skeletons compatible with it).")

# These four are MARGINAL correlations, and n_x is a confounder of the other three: a jet
# with a harder leading emission tends to have more of them, and multiplicity moves M_1
# hard in the opposite direction. A marginal sign that contradicts the prediction is
# therefore not yet evidence against it -- so the same correlation is re-read at FIXED
# n_x, which is the comparison the physics argument actually makes.
print(f"\nthe same correlations AT FIXED n_x (the confounder), where the prediction lives:")
print(f"  {'n_x':>6}{'jets':>7}" + "".join(f"{k:>14}" for k in
                                           ("lnkt_lead", "d_boundary", "d_floor")))
for _v in sorted(set(N_X.astype(int)))[:6]:
    _sel = N_X == _v
    if _sel.sum() < MIN_CI_JETS:
        continue
    print(f"  {_v:>6}{int(_sel.sum()):>7,}"
          + "".join(f"{spearman(M1[_sel], arr[_sel]):>+14.3f}"
                    for arr in (LNKT, D_B, D_F)))
print("  (a marginal sign that flips inside every n_x bin was the multiplicity confound;")
print("   one that survives the split is a statement about the smearing.)")

mx = SUMMARY["mixture"]
_fs = mx["stratum"]["F"][f"{DOMINANT_M:g}"]
_fc = mx["complement"]["F"][f"{DOMINANT_M:g}"]
print(f"\nthe mixture statement (plan section 7.4), quoted as pre-registered:")
print(f"  F({DOMINANT_M:g}) = {_fs['frac']:.3f} [{_fs['wilson95'][0]:.3f}, "
      f"{_fs['wilson95'][1]:.3f}] on the perturbative stratum ({_fs['n']:,} jets)")
print(f"  F({DOMINANT_M:g}) = {_fc['frac']:.3f} [{_fc['wilson95'][0]:.3f}, "
      f"{_fc['wilson95'][1]:.3f}] on its complement      ({_fc['n']:,} jets)")
print(f"  entropy: median e^H_hat = {np.median(_e):.2f} overall, "
      f"{np.median(EFF[MIX][np.isfinite(EFF[MIX])]):.2f} on the perturbative stratum")
''')

# ---------------------------------------------------------------------------
md(r"""
## 9. The per-jet residuals — with the mode skeleton as a fourth estimator

Everything from here is `per_jets_estimation.ipynb` §5–§8, unchanged, with `mode` added to the
series list: $\Delta = \text{estimate} - \text{truth}$, one entry per *splitting*, paired on the
**splitting index** $t$ (both ladders march inward in angle, so $t$ means the same thing on both
sides).

Two pairings, and which is used where:

- **Own depth** — each series against truth on $\min(n_{\rm truth}, n_s)$, independently of the
  others. Every splitting an estimator actually produced; §9a/§9b plot this and the descriptive
  columns of §10 report it.
- **Common depth** — $\min$ over truth *and every* series, so all of them carry identical
  $(\text{jet}, t)$ rows. The only pairing on which a between-series ratio is a comparison; §10's
  bootstrap uses it.

The depth-free kinematic matching of `per_jets_estimation.ipynb` §9 is deliberately **not** repeated
here — it is orthogonal to the question this notebook asks, and it is measured there on the same
checkpoint and the same file.
""")

code(r'''
def pair_residuals(raw, w_jet, models=MODELS, common=True):
    """Index-aligned residuals: one row per (jet, splitting index t), per series.

    A residual exists at `t` only where both sides have a node there.

    `common=False` -- each series gets its own min(n_truth, n_s): every splitting the
                      estimator actually produced, at the cost that the series no longer
                      live on the same rows.
    `common=True`  -- the depth kept for a jet is min over TRUTH and EVERY series, so all
                      series carry identical (jet, t) rows. The only pairing on which a
                      between-series ratio is a comparison.
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


RES = pair_residuals(RAW, W_JET, common=False)         # headline: own depth per series
RES_COMMON = pair_residuals(RAW, W_JET, common=True)   # row-matched: ratios, section 10
P, PC = RES["pairing"], RES_COMMON["pairing"]
_s0 = MODELS[0]

print(f"jets evaluated                    : {P['n_jets']:,}")
print(f"  with a non-empty parton truth   : {P['n_jets_truth_nonempty']:,} "
      f"({P['n_jets_truth_nonempty'] / P['n_jets']:.1%})")
print(f"truth splittings to recover       : {P['n_truth_splittings']:,}")
print()
print(f"{'series':<8}{'mean mult':>11}{'own-depth pairs':>18}{'of truth':>10}"
      f"{'jets w/ >=1 pair':>18}")
print(f"{'truth':<8}{NSPL['truth'].mean():>11.3f}{'--':>18}{'--':>10}{'--':>18}")
for s in MODELS:
    print(f"{s:<8}{NSPL[s].mean():>11.3f}{P['n_paired'][s]:>18,}"
          f"{P['frac_paired'][s]:>10.1%}{P['n_jets_paired'][s]:>18,}")
print(f"\ncommon depth (min over truth and ALL series, used for the ratios in section 10):"
      f"\n  {PC['n_paired'][_s0]:,} rows = {PC['frac_paired'][_s0]:.1%} of truth splittings, "
      f"on {PC['n_jets_paired'][_s0]:,} jets")
print("\nA residual exists only where BOTH sides have a node at t, so every distribution")
print("below is conditioned on the pair existing. Where a series' `of truth` fraction is")
print("well below 1 it was scored on the splittings it DID produce -- the easy end of its")
print("own output -- and the number is optimistic.")
print(f"\nTHE `mode` ROW IS THE EXTREME CASE and must be read with section 6 open: its")
print(f"skeleton is EMPTY for {EMPTY_TOP1.mean():.1%} of jets, so it contributes no row "
      f"there at all and pairs\nagainst only {P['frac_paired']['mode']:.1%} of the truth "
      f"splittings. Its residual is the residual OF THE\nJETS WHOSE MODE IS NON-EMPTY, "
      f"which is a different and easier population -- not a\nnarrower estimator. The "
      f"`rsd` row restricted to the same jets is the control for it,\nand section 9c "
      f"makes that restriction explicit.")
# Which series sets the common depth, counted rather than asserted: without this the
# ratio column of section 10 silently describes whatever the shortest series allowed.
_binds = {s: int(((RES_COMMON["depth"][s] == NSPL[s]) & (NSPL[s] < RES["n_true"])).sum())
          for s in MODELS}
print(f"\ncommon depth is set by: "
      + "   ".join(f"{s} {_binds[s]:,}" for s in MODELS)
      + "  (jets where that series is the binding one)")
''')

md(r"""
### 9a. All splittings

$\Delta$ for every paired splitting, unit-area normalised. The dotted line at $\Delta = 0$ is
perfect recovery; the legend carries each series' **bias** ($\langle\Delta\rangle$), **RMS** (about
zero, not about the mean — a constant offset is a real failure, not something to subtract) and the
**68% half-width**, which separates the two.

**Plain RSD is the number to beat.** Its residual is not noise: it is the hadronisation correction
itself, which is what the posterior exists to undo. The **posterior draw** is not a competitor
either way — it carries the full posterior spread, so a point estimate whose residual is not
comfortably narrower than a single draw's is not summarising anything.
""")

code(r'''
def resid_edges(key, pct=RESID_PCT, nb=RESID_NB, res=None):
    """Symmetric residual axis, shared by every panel and slice for this coordinate."""
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


def slice_stats(key, sel=sel_all, series=MODELS, res=None, jet_mask=None):
    """`wstats` for one coordinate and one t-slice, per series.

    `jet_mask` (per JET, not per row) additionally restricts to a jet subset -- that is
    how section 9c conditions the residual on the audit's own verdict without building a
    second residual table.
    """
    res = RES if res is None else res
    out = {}
    for s in series:
        m = sel(res["T"][s])
        if jet_mask is not None:
            m = m & jet_mask[res["J"][s]]
        out[s] = wstats(res["D"][s][m, COL[key]], res["W"][s][m])
    return out


def resid_panel(ax, key, sel=sel_all, series=MODELS, title="", res=None, jet_mask=None):
    """One difference distribution: every series' delta for one coordinate."""
    res = RES if res is None else res
    e, col = RESID_EDGES[key], COL[key]
    stats, dens = slice_stats(key, sel, series, res, jet_mask), {}
    for s in series:
        m = sel(res["T"][s])
        if jet_mask is not None:
            m = m & jet_mask[res["J"][s]]
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
        ax.set_ylim(0.0, ymax * 1.92)      # headroom for the legend, not overlap
    finish(ax, xlabel=DLABEL[key], ylabel="density", title=title, legend=True,
           loc="upper left")
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

md(r"""
### 9b. The first two splittings

$t=0$ is the **widest-angle** primary splitting that survived grooming — and *usually*, but not
always, also the hardest. Declustering marches inward in angle, so $t=0$ is the widest for 98.4% of
multi-splitting truth trees; the hardest emission ($\max_t \ln k_t$) sits at $t=0$ for only **80%**
of them, at $t=1$ for most of the rest, and when the two differ they are usually a near-tie
(the $\ln k_t$ given up has median 0.000, p90 0.289).

That distinction matters here more than in the sibling notebook, because §6–§8 stratify on
`lnkt_lead`, which is the **hardest** emission of $x$, while this section slices on $t$, which is
the **widest**. They are different observables that agree on four jets in five.

Every series should still be at its narrowest at $t=0$ — it is the most perturbative and the best
determined — and the gap between plain RSD and the model there is the cleanest single statement of
what the posterior buys.
""")

code(r'''
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

md(r"""
### 9c. The residual where the posterior *is* dominant

The cross the two halves of this notebook exist to make: the same residual, split by whether the
jet's own posterior concentrates ($M_1 \ge$ `DOMINANT_M`) or not. The split is on a quantity
available **at inference time, on data** — no truth is used to define it — so a narrower residual on
the dominant side is a usable statement: *the model knows which of its own answers to trust.*

If the two sides sit on top of each other, $M_1$ carries no information about accuracy, and that is
just as much a result: it says the concentration of the posterior and the correctness of its mode
are unrelated for this model, and the dominance numbers of §6 should not be read as a quality
signal.
""")

code(r'''
DOM_JET = DOMINANT.copy()      # per-JET mask, indexed by the same i the residual rows carry
fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
for r, (mask, lab) in enumerate([(DOM_JET, f"dominant  ($M_1 \\geq {DOMINANT_M:g}$)"),
                                 (~DOM_JET, f"diffuse  ($M_1 < {DOMINANT_M:g}$)")]):
    for c, key in enumerate(RES_KEYS):
        n_rows = int((sel_all(RES["T"][MODELS[0]])
                      & mask[RES["J"][MODELS[0]]]).sum())
        resid_panel(axes[r, c], key, sel_all, jet_mask=mask,
                    title=f"{DLABEL[key]}   {lab}   ({int(mask.sum()):,} jets, "
                          f"{n_rows:,} splittings)")
fig.suptitle(r"the same residual, split by whether the posterior has a dominant skeleton "
             r"(a cut available ON DATA)", x=0.006, y=1.003, ha="left")
fig.tight_layout()
plt.show()

print(f"{'coordinate':<12}{'series':<8}"
      f"{'RMS | dominant':>17}{'RMS | diffuse':>16}{'ratio':>9}   "
      f"{'68% hw dom':>12}{'68% hw dif':>12}")
DOM_TABLE = {}
for key in RES_KEYS:
    for s in MODELS:
        a_ = slice_stats(key, sel_all, [s], jet_mask=DOM_JET)[s]
        b_ = slice_stats(key, sel_all, [s], jet_mask=~DOM_JET)[s]
        ratio = a_["rms"] / b_["rms"] if b_["rms"] > 0 else float("nan")
        DOM_TABLE[(key, s)] = {"dominant": a_, "diffuse": b_, "rms_ratio": float(ratio)}
        print(f"{TLABEL[key] if s == MODELS[0] else '':<12}{s:<8}"
              f"{a_['rms']:>17.3f}{b_['rms']:>16.3f}{ratio:>9.3f}   "
              f"{a_['hw68']:>12.3f}{b_['hw68']:>12.3f}")
print("\nratio < 1 means the estimator is genuinely more accurate on the jets whose")
print("posterior concentrates -- i.e. M_1 is a usable per-jet confidence. Note the two")
print("populations are different jets, so this is a conditional statement about the")
print("SAMPLE, not a paired comparison: the dominant subset is also the more")
print("perturbative one (section 8's Spearman), and an easier population is expected to")
print("be easier for plain RSD too. Read the `rsd` row as the control -- the model rows")
print("only mean something to the extent they beat it.")
''')

# ---------------------------------------------------------------------------
md(r"""
### 9d. The primary Lund plane, and where each estimate puts its splittings

$\rho(\ln 1/\Delta R,\ \ln k_t)$ — weighted splittings **per jet per unit Lund area**, the same
observable and the same binning as §6 of
[`lund_distribution_closure_v2.ipynb`](lund_distribution_closure_v2.ipynb), so the two overlay
without re-learning a scale. At fixed coupling the primary Lund density is approximately flat
($\rho \approx 2\alpha_s C_F/\pi$) — that plateau is what the coordinates are for — with the
running coupling tilting it toward small $k_t$ and the grooming cutting hard edges into it.

The new series here is **`mode`**, the top-1 skeleton. Two ratio rows, because it under-produces
splittings for a reason that has nothing to do with *where* it puts them:

- **rate ratio** — $\rho_s/\rho_{\rm truth}$ as measured. The `mode` row is suppressed almost
  everywhere, and it should be: its skeleton is the empty tree for ~78% of jets (§6), so it
  contributes no splitting at all on those. This ratio is the multiplicity deficit and the
  placement error multiplied together.
- **shape ratio** — both planes normalised to unit total first, which divides the multiplicity out
  and asks the separate question: *given that it emits, does it emit in the right places?*

Reading only the first would blame the mode's placement for a length effect; reading only the
second would hide that the length effect exists. The pair is the honest presentation, and it is the
same rate-versus-shape split the closure notebook makes for the multiplicity bias.
""")

code(r'''
U_LO, U_HI = geom.ln_invdelta_range
V_LO, V_HI = geom.ln_kt_range
PLANE_NB = geom.n_bins          # the density map: the model's own cells, so its granularity
#                                 shows rather than hides
# The RATIO is binned coarser, and it has to be: this sample has ~1.4 truth splittings per
# jet, so at 30x30 a cell holds ~3 of them and a ratio there is Poisson noise painted in
# saturated colour. RATIO_NB divides n_bins, so a ratio bin is a whole number of model
# cells. Same reasoning as lund_distribution_closure_v2.ipynb section 6, which rebins for
# the same reason on a larger sample.
RATIO_NB = 10
RLO, RHI, N_MIN = 0.4, 2.5, 12  # ratio scale, and the truth count a bin needs to divide by


def pack(series):
    """Splitting-level (coords, weights) for one series -- one row per node, carrying its
    jet's weight, which is the level a per-jet-per-area density is defined on."""
    v = [a for a in RAW[series] if len(a)]
    w = [np.full(len(a), W_JET[i]) for i, a in enumerate(RAW[series]) if len(a)]
    return (np.concatenate(v) if v else np.zeros((0, 4)),
            np.concatenate(w) if w else np.zeros(0))


def plane(series, nb):
    """(density, raw counts) on an nb x nb grid: splittings per jet per unit Lund area."""
    v, w = pack(series)
    e = [np.linspace(U_LO, U_HI, nb + 1), np.linspace(V_LO, V_HI, nb + 1)]
    h = np.histogram2d(v[:, 0], v[:, 1], bins=e, weights=w)[0]
    n = np.histogram2d(v[:, 0], v[:, 1], bins=e)[0]
    area = (U_HI - U_LO) * (V_HI - V_LO) / nb ** 2
    return h / (W_JET.sum() * area), n, e


PLANES, PCOUNT = {}, {}
for s in SERIES:
    PLANES[s], PCOUNT[s], PE = plane(s, PLANE_NB)
# A single hot bin would flatten the whole ramp on a shared linear scale, so the top of
# the scale is a high percentile of the populated bins, not the maximum.
_pop = np.concatenate([P[P > 0] for P in PLANES.values() if (P > 0).any()])
vmax = float(np.percentile(_pop, 99)) if _pop.size else 1.0
# View limits only -- the binning stays on the model's full window, but a 100 GeV sample
# populates a corner of it and plotting the whole square would shrink every structure.
_hit = np.sum(list(PLANES.values()), axis=0) > 0
_iu, _iv = np.flatnonzero(_hit.any(axis=1)), np.flatnonzero(_hit.any(axis=0))
XLIM = (PE[0][max(_iu[0] - 1, 0)], PE[0][min(_iu[-1] + 2, PLANE_NB)])
YLIM = (PE[1][max(_iv[0] - 1, 0)], PE[1][min(_iv[-1] + 2, PLANE_NB)])

fig, axes = plt.subplots(1, len(SERIES), figsize=(3.2 * len(SERIES), 3.9), sharey=True)
for ax, s in zip(np.atleast_1d(axes), SERIES):
    P = np.ma.masked_where(PLANES[s] <= 0, PLANES[s])
    im = ax.pcolormesh(PE[0], PE[1], P.T, cmap=CMAP, vmin=0.0, vmax=vmax,
                       shading="flat", rasterized=True)
    ax.set_title(f"{STYLE[s][2]}\n{int(NSPL[s].sum()):,} splittings", fontsize=8)
    ax.set_xlabel(LABEL["lnInvDelta"])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(False)
np.atleast_1d(axes)[0].set_ylabel(LABEL["lnkt"])
fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012,
             label=r"$\rho$  [splittings / jet / unit area]")
fig.suptitle(f"the primary Lund plane   (view cropped to the populated region; binning is "
             f"the full {geom.ln_invdelta_range} x {geom.ln_kt_range} window)",
             x=0.07, y=1.03, ha="left")
plt.show()


def plane_ratio(num, den, n_den, n_num):
    """Ratio map, gated on truth having enough entries to divide by.

    A bin where TRUTH is empty but the prediction is not is a real disagreement (ratio ->
    infinity), so it saturates the top of the scale rather than being blanked -- blanking
    would hide invented emissions."""
    out = np.full(num.shape, np.nan)
    ok = (den > 0) & (n_den >= N_MIN)
    out[ok] = num[ok] / den[ok]
    out[(den <= 0) & (n_num >= N_MIN)] = RHI
    return np.ma.masked_invalid(out)


RPLANES, RCOUNT = {}, {}
for _s in SERIES:
    RPLANES[_s], RCOUNT[_s], RE = plane(_s, RATIO_NB)

fig, axes = plt.subplots(2, len(MODELS), figsize=(3.2 * len(MODELS), 7.4), sharey=True)
axes = np.atleast_2d(axes)
for j, s in enumerate(MODELS):
    # row 0: the RATE ratio -- multiplicity and placement multiplied together
    R = plane_ratio(RPLANES[s], RPLANES["truth"], RCOUNT["truth"], RCOUNT[s])
    im = axes[0, j].pcolormesh(RE[0], RE[1], R.T, cmap=DIV,
                               norm=mpl.colors.LogNorm(vmin=RLO, vmax=RHI),
                               shading="flat", rasterized=True)
    axes[0, j].set_title(f"{STYLE[s][2]} / truth\nRATE", fontsize=8)
    # row 1: the SHAPE ratio -- each plane normalised to unit total first
    a, b = RPLANES[s], RPLANES["truth"]
    sa, sb = a.sum(), b.sum()
    Rs = plane_ratio(a / sa if sa > 0 else a, b / sb if sb > 0 else b,
                     RCOUNT["truth"], RCOUNT[s])
    axes[1, j].pcolormesh(RE[0], RE[1], Rs.T, cmap=DIV,
                          norm=mpl.colors.LogNorm(vmin=RLO, vmax=RHI),
                          shading="flat", rasterized=True)
    axes[1, j].set_title(f"{STYLE[s][2]} / truth\nSHAPE (unit-normalised)", fontsize=8)
    for r in (0, 1):
        axes[r, j].set_xlim(*XLIM)
        axes[r, j].set_ylim(*YLIM)
        axes[r, j].grid(False)
        axes[r, j].set_xlabel(LABEL["lnInvDelta"])
for r in (0, 1):
    axes[r, 0].set_ylabel(LABEL["lnkt"])
fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012, label="ratio to truth")
fig.suptitle(f"where each estimate over- and under-populates the plane, on {RATIO_NB}x"
             f"{RATIO_NB} bins\n(grey = agrees, blue = too few, red = too many, blank = "
             f"fewer than {N_MIN} truth splittings)", x=0.06, y=1.04, ha="left")
plt.show()

print(f"{'series':<8}{'splittings':>12}{'per jet':>10}{'rate / truth':>14}"
      f"{'mean |log ratio|, shape':>26}")
_lt = RPLANES["truth"]
for s in SERIES:
    per = NSPL[s].sum() / max(len(W_JET), 1)
    rate = NSPL[s].sum() / max(NSPL["truth"].sum(), 1)
    a, b = RPLANES[s], _lt
    ok = (a > 0) & (b > 0) & (RCOUNT["truth"] >= N_MIN)
    sa, sb = a.sum(), b.sum()
    shp = (float(np.mean(np.abs(np.log((a[ok] / sa) / (b[ok] / sb))))) if ok.any() and sa > 0
           else float("nan"))
    print(f"{s:<8}{int(NSPL[s].sum()):>12,}{per:>10.3f}"
          + (f"{'--':>14}" if s == "truth" else f"{rate:>14.3f}")
          + (f"{'--':>26}" if s == "truth" else f"{shp:>26.3f}"))
print("\n`rate / truth` is the multiplicity ratio -- for `mode` it is dominated by the "
      f"{EMPTY_TOP1.mean():.0%} of jets\nwhose best skeleton is EMPTY, which contribute no "
      "splitting anywhere on the plane.")
print("`mean |log ratio|, shape` is the placement error with the multiplicity divided out:")
print("0 would mean the series populates the plane in exactly truth's proportions. Read the")
print("two together -- a series can have the right rate in the wrong places, or the right")
print("places at the wrong rate, and the two rows of the figure separate them.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 10. Summary — is any of these estimates narrower than doing nothing?

The headline is the **RMS ratio** $\mathrm{RMS}(\hat y - y)/\mathrm{RMS}(x - y)$: below 1 means the
estimate beat leaving the hadron-level jet alone. Its uncertainty is a **jet-level** bootstrap —
resampling jets, not splittings, because the splittings of one jet are correlated and resampling
them independently would understate the interval by roughly $\sqrt{\langle n\rangle}$. A ratio whose
interval brackets 1 is a null result and is reported as one.

The descriptive columns come from the **own-depth** pairing; the ratio column from the
**common-depth** rows, where every series occupies the same splittings and a ratio is therefore a
comparison.

**The common depth here is not the companion notebook's.** It is the minimum over truth and *all
five* series, and the `mode` series — empty for a large fraction of jets (§6) — binds it hard. So
the ratio column below is internally consistent but is computed on many fewer rows than
[`per_jets_estimation.ipynb`](per_jets_estimation.ipynb) §8, which is the reference for the
RSD/MAP/MBR trio. §9 prints which series binds, so the cost is visible rather than inferred.
""")

code(r'''
def _row_matched(key, sel, s, res):
    """`s` and `rsd` residuals on the rows they BOTH cover, plus weights and jet ids."""
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
    """Jet-level bootstrap on RMS(s)/RMS(rsd) for one coordinate and t-slice."""
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
print("\nRMS is about ZERO, not about each series' own mean: a constant offset is a real")
print("failure of a point estimate, not something to subtract off. `bias` and `68% hw`")
print("split the same total into its offset and its spread.")
print("\nThe `mode` row is the mode-SKELETON estimate: the exact argmax of the skeleton")
print("marginal with staged coordinates. It differs from `map` only in WHICH cell")
print("sequence, so the two rows together price the beam search's length bias against an")
print("exact marginal argmax -- and neither is MBR, which optimises a different loss.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 11. Validity checks

Arithmetic, not physics. These are the plan's §8 T2/T4/T5 patterns run on the loaded checkpoint
rather than on a tiny synthetic model, and a failure means the *search* is wrong — no statement
about the model would survive it.

1. **Mass accounting** — $\sum_i M_i + m_{\rm frontier} + m_{\rm pruned} = 1$ per jet. This is the
   identity that turns coverage into a bound.
2. **The empty-skeleton identity** — the enumerated $M_{N=0}$ against the model's own $q(0\mid x)$
   reached through `describe_cells`, a code path that never touches the incremental step the search
   uses.
3. **Sampled-frequency consistency** — on a handful of jets, the top-1 skeleton's *frequency* among
   fresh posterior draws against its enumerated mass. The search and the sampler must describe the
   same distribution.
""")

code(r'''
_defect = np.array([abs(a["mass_defect"]) for a in AUDITS])
_q0dev = np.array([abs(a["q0_enum"] - a["q0_model"]) for a in AUDITS])
print(f"1. mass accounting     : max |sum M_i + frontier + pruned - 1| = {_defect.max():.3e}"
      f"   (median {np.median(_defect):.1e})")
print(f"2. empty-skeleton q(0|x): max |enumerated - describe_cells| = {_q0dev.max():.3e}"
      f"   (median {np.median(_q0dev):.1e})")
# float32 heads: a product of softmaxes closes to ~1e-7 per factor, not to float64.
assert _defect.max() < 1e-5, "the mass-accounting identity failed -- the search is wrong"
assert _q0dev.max() < 1e-5, "the empty skeleton's mass disagrees with the model's q(0|x)"

# 3. the sampled-frequency check, on the first few jets with a non-degenerate M_1
_n_check, _n_draws = 6, 4000
print(f"\n3. sampled-frequency consistency, {_n_draws} fresh draws on {_n_check} jets:")
print(f"   {'jet':>5}{'M_1 (exact)':>14}{'frequency':>12}{'|diff|':>10}{'4 sigma':>10}"
      f"   verdict")
seed_everything(SEED + 1)
_checked = 0
for _i in range(len(AUDITS)):
    if _checked >= _n_check:
        break
    a = AUDITS[_i]
    if a["M1"] < 0.02:            # too rare for 4000 draws to resolve
        continue
    _item = ds[_i]
    _xf = _item["xf"].unsqueeze(0).to(device)
    _nxj = torch.tensor([_item["nx"]], device=device)
    with torch.inference_mode():
        _dr = model.sample(_xf, _nxj, n=_n_draws)
    _top = [int(c) for c in a["cells_top1"]]
    _freq = sum(1 for d in _dr if [int(c) for c in d] == _top) / _n_draws
    _sig = 4.0 * math.sqrt(max(a["M1"] * (1 - a["M1"]), 1e-12) / _n_draws) + 1.0 / _n_draws
    _ok = abs(_freq - a["M1"]) <= _sig
    print(f"   {_i:>5}{a['M1']:>14.4f}{_freq:>12.4f}{abs(_freq - a['M1']):>10.4f}"
          f"{_sig:>10.4f}   {'ok' if _ok else 'OUTSIDE THE BAND'}")
    _checked += 1
print("   (a normal binomial band at n = 4000, where np > 5 comfortably; the sampler's")
print("    own continue/cell temperatures are at their no-op defaults for this check)")
''')

# ---------------------------------------------------------------------------
code(r'''
if WRITE_ARTIFACTS:
    METRICS = {
        "run": {
            "notebook": "per_jets_estimation_mode_mass",
            "checkpoint": str(CKPT_PATH), "test_path": str(ROOT_PATH),
            "model": info["model_name"], "encoder": str(cfg.encoder.name),
            "aux_features": list(AUX), "lnz_support": LNZ_SUPPORT,
            "n_bins": geom.n_bins, "n_jets": int(N), "K_draws": int(K_DRAWS),
            "seed": int(SEED), "device": str(device), "mbr_backend": MBR_BACKEND,
            "mbr_n_candidates": int(MBR_N_CANDIDATES),
            "length_floor_quantile": float(LENGTH_FLOOR_QUANTILE),
            "empty_gate": bool(GATE_EMPTY), "empty_threshold_applied": float(TAU),
            "dominant_threshold": float(DOMINANT_M),
        },
        # The audit block, verbatim from `summarise_mode_audit` -- the same function the
        # CLI's mode_audit.json goes through, so the two artifacts are comparable field
        # by field (and section 7.5's cross-family delta is a subtraction).
        "mode_audit": SUMMARY,
        "residuals": {
            "alignment": {"index_rule": "splitting index t; a residual exists only where "
                                        "truth and the compared series have a node at t",
                          "own": P, "common": PC,
                          "stats_pairing": "own", "ratio_pairing": "common"},
            **{f"{key}|{slab}|{s}": v for (key, slab, s), v in TABLE.items()},
        },
        "residuals_by_dominance": {
            f"{key}|{s}": v for (key, s), v in DOM_TABLE.items()
        },
    }
    out = save_metrics(METRICS, (REPO / CKPT_PATH).parent / "mode_audit_nb.json")
    print(f"wrote {out.relative_to(REPO)}")
    print("  (the CLI writes the same audit block to mode_audit.json via "
          "`eval <ckpt> experiment.mode_audit=true`; this file adds the residual tables)")
else:
    print("WRITE_ARTIFACTS = False -- nothing written")
''')

# ---------------------------------------------------------------------------
md(r"""
---

### Reading these figures

- **$M_1$ is exact; the *ranking* is what needs a certificate.** The mass of any enumerated
  skeleton is a product of head probabilities and is exact whatever the search did. What pruning can
  cost is the claim that nothing bigger was missed — which is why `certified` sits beside every
  fraction, why an uncertified $M_1$ is reported as a lower bound, and why the pre-registered
  dominance mark is $1/2$: above it, the claim is a proof.
- **Exact is not invariant, and this is the one that will mislead you.** $M_1$ is a probability of
  a *cell*, hence $\approx$ density $\times$ area: refine `n_bins` and every $N\ge1$ mass falls
  while $q(N{=}0\mid x)$ does not, so both the level of $F(m)$ and which skeleton wins are set by
  the grid. A skeleton bundles two different things — the multiplicity and the *order*, which are
  genuinely discrete and grid-free, and the cell labels, which are a discretization of a continuum.
  Dominance is a well-posed probability question only for the first. §6a measures the second the way
  it has to be measured: as an **area**. On this checkpoint the first splitting's 50% region spans
  ~17 cells, the coordinate head is truncation-saturated for ~90% of jets, and $M_1$ tracks
  $1/\text{cells}$ — so §6's small numbers are the grid talking, not a fragmented posterior.
- **Dominance is not correctness.** §6 and §7 answer different questions and are never combined
  except in the explicit four-way cross. A model that is *dominant and wrong* is worse than one that
  is diffuse; only that panel distinguishes them.
- **The empty skeleton is a real answer, not a failure mode — and its win is a grid effect.** It
  carries mass $P_{\rm cont}(\mathrm{stop}\mid h_0,e)$ like any other, and on this sample the parton
  truth really is empty for ~17% of jets. Neither the MAP nor MBR can produce it under the default
  decode (the argmax over $n$ lands at 0 essentially never; MBR's imbalance term prices an empty
  cloud at near-maximal risk) — so §7's empty-truth line measures something the point-estimate table
  structurally cannot. But it is *also* the one configuration the cell grid does not slice, which is
  why it wins the argmax far more often than the model believes the jet is empty: §6a prints
  $q(N{=}1\mid x) > q(N{=}0\mid x)$ for the large majority of jets. "The mode is empty" and "the
  posterior thinks there is nothing there" are not the same statement, and only the second would be
  a claim about the physics.
- **The coordinates never concentrate, and §4 says so on every line.** Dominance is a statement
  about the discrete skeleton only. The non-perturbative width $\sigma_0 + \Lambda_{\rm eff}/k_t$ is
  irreducible, so a mode-skeleton node is quoted as value $\pm$ the head's own width, with $\kappa$
  beside $\psi$ because below `decode.kappa_min_mode` its mode is not an identified direction.
- **§9c is a conditional statement about a population, not a paired test.** The dominant subset is
  also the more perturbative one, and an easier population is easier for plain RSD too — which is
  exactly why the `rsd` row is in that table. The model rows are only informative to the extent they
  move *relative* to it.
- **The audit changes nothing.** No decode-layer behaviour depends on it: MBR remains the headline
  estimator whatever these numbers say (Kumar & Byrne, HLT-NAACL 2004; Eikema & Aziz,
  arXiv:2005.10283). A majority skeleton, where one exists, is an **additional** quotable — "this
  jet's parton structure is $S_1$ with probability $M_1$" — not a replacement for a decision rule
  fitted to a loss.

### Running it elsewhere

Set `RUN` in §0 to any run directory, arm root, `best.ckpt` or artifact JSON. The checkpoint must
have a continuous coordinate density (§2 asserts it) for §9–§10 to mean anything; the skeleton audit
of §6–§8 needs only the discrete heads and a family with a `skeleton_search_spec` adapter —
`ar_junipr_*`, `cinn` and `diffusion` have one, and anything else raises by name rather than
quietly reporting a beam-search approximation.

For the depth-free kinematic matching, the pairing-bias discussion and the spurious-node rates, see
[`per_jets_estimation.ipynb`](per_jets_estimation.ipynb) §9 — same checkpoint, same file, same
decode.
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
       / "per_jets_estimation_mode_mass.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
