"""The Lund distribution closure as a script: PDF figures + a Markdown report.

Same analysis as [`notebooks/lund_distribution_closure_prod_test_v1.ipynb`](../notebooks/lund_distribution_closure_prod_test_v1.ipynb)
— which is `lund_distribution_closure_v2.ipynb` pointed at the held-out file with the
five production-test constants read from the run's own artifact — with the output
inverted: every panel goes to a **PDF** under `<out>/figures/`, and the prose, the
tables and the run-evaluated conclusions go to **`<out>/report.md`**, which references
each figure by number and by path. Nothing is left as cell output, so the whole thing
survives a kernel restart, a `git pull` and an nbstripout smudge.

Default mode is the production test: the checkpoint, the held-out file, the FROZEN
empty-tree `tau` and the fitted `(temperature, tilt)` come from the newest
`runs/prod_test_v*/*/prod_test_v*/prod_test_v*_metrics.json` (v1's or v0's — the report
records which), so they cannot disagree
with the section-6 fit that produced them — including the scale check that a `tau`
fitted on the raw head is not applied to a recalibrated one. `--no-prod-metrics` runs
the plain v2 configuration instead (rate-matched tau, checkpoint's own T/tilt), which
is the circular setting the production test exists to avoid; the report says so.

    PYTHONPATH=src python scripts/lund_closure_report.py --jets 2000
    PYTHONPATH=src python scripts/lund_closure_report.py --probe 20      # cost, then exit
    PYTHONPATH=src python scripts/lund_closure_report.py --no-prod-metrics \
        --ckpt runs/calibration_v2_walkthrough/ar_junipr_v4/best.ckpt \
        --root cpp/test_data/jets.root --jets 500 --png

ON DUPLICATION. This is a second implementation of numbers the notebook also defines,
which is the failure mode `scripts/make_prod_closure_nb.py` and `tests/test_prod_closure_nb.py`
exist to prevent for the two notebooks (docs/PLAN_prod_test_v0.md §7). It is deliberate
here — a script is not a notebook and cannot be generated from one — so the drift check
is made cheap instead: this writes `dist_closure_metrics.json` in exactly the schema
the notebook writes, so

    jq -S .headline a/dist_closure_metrics.json   # notebook
    jq -S .headline b/dist_closure_metrics.json   # this script, same ckpt/file/K/backend

is the whole comparison. It is a comparison to within MONTE-CARLO NOISE, not a bitwise
one: the notebook's §4a cost probe draws on the torch RNG before its main pass, so the
same seed gives a different draw stream here, and every quantity built on K posterior
draws moves at the level of its own sampling error. Ratios that disagree by more than
that are drift, and where they do, the notebook is the definition.

Measured against the committed `dist_closure_metrics.json` of the prod_test_v0 run
(4000 jets, K=120, `energyflow`, seed 1234), every quantity that does NOT depend on the
draw stream came back bit-identical — truth, plain RSD, MAP (the beam search is
deterministic), all 96 noise floors, the scoreable counts, the gate's tau/precision/
recall and the q(0|x) AUC. Only the two draw-dependent series moved: MBR by ~5% and the
posterior by ~20% on the geometric-mean ratios, the latter because it is one draw per
jet and several of its rows sit a hair above their own floor. That is the signature of
a faithful port; a discrepancy in the MAP or the floors would not be.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # before pyplot: this script never has a display

import matplotlib as mpl                     # noqa: E402
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import torch                                 # noqa: E402
from omegaconf import OmegaConf              # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from h2p_rsd_junipr.config import decode_params                     # noqa: E402
from h2p_rsd_junipr.data.datamodule import select_pt_range          # noqa: E402
from h2p_rsd_junipr.data.dataset import MatchedLundDataset          # noqa: E402
from h2p_rsd_junipr.data.rntuple import load_rntuple                # noqa: E402
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset  # noqa: E402
from h2p_rsd_junipr.eval.calibration import REGION_LABELS, cell_region  # noqa: E402
from h2p_rsd_junipr.eval.report import save_metrics                 # noqa: E402
from h2p_rsd_junipr.features import node_raw                        # noqa: E402
from h2p_rsd_junipr.geometry import Geometry                        # noqa: E402
from h2p_rsd_junipr.inference.length import (                       # noqa: E402
    empty_threshold_for_rate,
    learned_min_emissions,
)
from h2p_rsd_junipr.models.ar_junipr import ARJunipr                # noqa: E402
from h2p_rsd_junipr.models.base import build_model                  # noqa: E402
from h2p_rsd_junipr.train.checkpoint import load_for_inference      # noqa: E402
from h2p_rsd_junipr.train.trainer import seed_everything, select_device  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=UserWarning)

# --- style: identical roles to the notebook ----------------------------------
# truth = ink (the reference every distance is measured against), plain RSD = grey fill
# (the do-nothing backdrop). Those two carry no hue, so the three MODEL series get the
# first three slots of the validated categorical palette -- the only three that clear the
# all-pairs colour-vision gates, which is what overlaid step histograms need.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
C_TRUTH = INK
C_RSD_F, C_RSD_E = "#e1e0d9", "#898781"
C_MAP, C_MBR, C_POST = "#2a78d6", "#eb6834", "#199e70"

SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP = mpl.colors.LinearSegmentedColormap.from_list("h2p_blue", SEQ_BLUE)
CMAP.set_bad(SURFACE)
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "h2p_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a3", "#d03b3b", "#7a1f1f"]
)
DIV.set_bad(SURFACE)

mpl.rcParams.update({
    # savefig.dpi is 300 rather than the notebook's 120: the Lund-plane meshes are
    # rasterized (a vector mesh of 900 cells bloats the PDF for no visible gain), and
    # 120 dpi rasters look soft the moment anyone zooms the PDF. Everything else in
    # these figures stays vector either way.
    "figure.dpi": 120, "savefig.dpi": 300,
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
    "pdf.fonttype": 42,   # embed TrueType, not Type 3: the maths renders in any viewer
})

# --- binning: hist_lund_rntuple.cpp, coarsened by REBIN ----------------------
BINS = {
    "lnInvDelta": (100, 0.0, 10.0),
    "lnkt": (120, -4.0, 8.0),
    "lnz": (100, -10.0, 0.0),
    "psi": (100, -np.pi, np.pi),
    "mult": (51, -0.5, 50.5),
}
LABEL = {
    "lnInvDelta": r"$\ln(1/\Delta R)$",
    "lnkt": r"$\ln(k_t/\mathrm{GeV})$",
    "lnz": r"$\ln z$",
    "psi": r"$\psi$",
    "mult": "primary splittings / jet",
}
COL = {"lnInvDelta": 0, "lnkt": 1, "lnz": 2, "psi": 3}   # node_raw column order
REBIN = {"lnInvDelta": 2, "lnkt": 2, "lnz": 1, "psi": 4, "mult": 1}
SERIES = ("truth", "rsd", "map", "mbr", "post")
MODELS = ("map", "mbr", "post")     # the series that get scored against plain RSD
T_SLICES = (0, 1, 2, 3)   # per-splitting-index panels and rows; a "4+" pool is appended
T_LADDER = 10             # ladder profiles run t = 0 .. T_LADDER-1
MET = ("w1", "ks", "chi2")
METRICS_LABELS = [("w1", "W1"), ("ks", "KS"), ("chi2", "chi2/ndf")]
CIRC = {"psi": (-np.pi, np.pi)}     # observables that live on a circle, not a line


# ============================================================================
# 0. Parameters
# ============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("what to run on")
    g.add_argument("--prod-metrics", default=None, metavar="PATH",
                   help="prod_test_v*_metrics.json to take the five production settings "
                        "from (default: newest under runs/prod_test_v*/)")
    g.add_argument("--no-prod-metrics", action="store_true",
                   help="ignore the artifact and use the v2 defaults: tau rate-matched on "
                        "THIS sample (circular) and the checkpoint's own (T, tilt)")
    g.add_argument("--ckpt", default=None,
                   help="checkpoint, repo-relative (default: from the artifact, or the "
                        "newest runs/**/best.ckpt)")
    g.add_argument("--root", default=None,
                   help="ROOT file to evaluate on (default: from the artifact, else "
                        "cpp/test_data/jets.root; missing -> synthetic fallback)")
    g.add_argument("--ntuple", default="Jets")
    g.add_argument("--jets", type=int, default=2000, help="jets in the evaluation pass")
    g.add_argument("--draws", type=int, default=120,
                   help="posterior draws per jet, shared by floor / MBR / posterior")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--device", default="cpu",
                   help="keep cpu: batch-1 decode never amortises GPU dispatch "
                        "(docs/PLAN_prod_test_speedup_mac.md)")

    g = p.add_argument_group("sample selection")
    g.add_argument("--pt-var", default="jet_pt", choices=["jet_pt", "x_ptg"])
    g.add_argument("--pt-min", type=float, default=None)
    g.add_argument("--pt-max", type=float, default=None)
    g.add_argument("--require-truth-splitting", action="store_true",
                   help="also require len(y)>0, reproducing v1's population. That reads "
                        "the answer, so it is not a selection any analysis can make")

    g = p.add_argument_group("decode")
    g.add_argument("--mbr-backend", default="pot",
                   choices=["pot", "energyflow", "surrogate"],
                   help="'energyflow' is the same metric ~6x faster; 'surrogate' is a "
                        "DIFFERENT risk function and may not be reported")
    g.add_argument("--mbr-candidates", type=int, default=16)
    g.add_argument("--length-floor-quantile", type=float, default=0.15)
    g.add_argument("--no-map-allow-empty", action="store_true",
                   help="skip the unfloored-MAP control row of section 3")
    g.add_argument("--empty-threshold", type=float, default=None,
                   help="override tau (default: frozen, from the artifact)")
    g.add_argument("--length-temperature", type=float, default=None)
    g.add_argument("--length-tilt", type=float, default=None)

    g = p.add_argument_group("binning, scoreability, output")
    g.add_argument("--plane-bins", type=int, default=30,
                   help="Lund-plane bins per axis; must be a multiple of geometry.n_bins")
    g.add_argument("--n-boot", type=int, default=24,
                   help="jet-level bootstrap resamples behind each row's noise floor")
    g.add_argument("--floor-pct", type=float, default=95)
    g.add_argument("--out", default=None,
                   help="output directory (default: <checkpoint dir>/lund_closure_report)")
    g.add_argument("--png", action="store_true",
                   help="also write a PNG of every figure and embed it in the report, so "
                        "the Markdown renders inline; the PDF stays the reference copy")
    g.add_argument("--probe", type=int, default=0, metavar="N",
                   help="time N jets, print the projected cost of --jets, and exit")
    g.add_argument("--no-artifacts", action="store_true",
                   help="skip dist_closure_metrics.json (the drift check against the "
                        "notebook goes through it, so this is rarely what you want)")
    return p.parse_args(argv)


def newest(pattern: str):
    found = sorted(REPO.glob(pattern), key=lambda q: q.stat().st_mtime)
    return found[-1] if found else None


def resolve_settings(a):
    """The five production-test settings, and where each one came from.

    Read from the run's OWN artifact rather than pasted in, so the frozen tau and the
    fitted (T, tilt) cannot disagree with the fit that produced them. Every guard the
    notebook's parameter cell asserts is asserted here, including the one that matters
    most: a tau is a QUANTILE of q(0|x), so applying one fitted on the raw head to a
    recalibrated head leaves the ranking untouched and the cut in the wrong place.
    """
    src = {}
    prod = None
    if not a.no_prod_metrics:
        mp = Path(a.prod_metrics) if a.prod_metrics else None
        if mp is not None and not mp.is_absolute():
            mp = REPO / mp
        if mp is None:
            # v* so a prod_test_v1 artifact is found too; newest wins, and the path is
            # recorded in the report, so which regime produced it is never a guess.
            mp = newest("runs/prod_test_v*/*/prod_test_v*/prod_test_v*_metrics.json")
        if mp is None:
            raise SystemExit(
                "no prod_test_v*_metrics.json under runs/. This script takes "
                "its checkpoint, its test file, the frozen empty-tree tau and the fitted "
                "length recalibration from that artifact — run notebooks/prod_test_v1.ipynb "
                "first, or pass --prod-metrics / --no-prod-metrics."
            )
        prod = json.loads(Path(mp).read_text())
        # The PRIMARY record of each value, not a summary block duplicating it: a second
        # copy is a second thing that can be stale.
        try:
            src = {
                "ckpt": prod["run"]["checkpoint"],
                "root": prod["run"]["test_path"],
                "empty_threshold": prod["empty_tree"]["tau"]["value"],
                "length_temperature": prod["empty_tree"]["recalibration"]["T"],
                "length_tilt": prod["empty_tree"]["recalibration"]["tilt"],
            }
        except KeyError as e:
            raise SystemExit(
                f"{mp} has no {e} — it is not a prod_test metrics file, or predates "
                f"the section-6 recalibration. Re-run notebooks/prod_test_v1.ipynb."
            ) from None
    s = {
        # repo-relative when it can be, so the report and the metrics JSON name the same
        # artifact whatever directory the report ends up in
        "prod_metrics_path": (str(Path(mp).resolve().relative_to(REPO))
                              if prod and Path(mp).resolve().is_relative_to(REPO)
                              else (str(Path(mp).resolve()) if prod else None)),
        "ckpt": a.ckpt or src.get("ckpt"),
        "root": a.root or src.get("root") or "cpp/test_data/jets.root",
        "empty_threshold": (a.empty_threshold if a.empty_threshold is not None
                            else src.get("empty_threshold")),
        "length_temperature": (a.length_temperature if a.length_temperature is not None
                               else src.get("length_temperature")),
        "length_tilt": (a.length_tilt if a.length_tilt is not None
                        else src.get("length_tilt")),
    }
    for k in ("empty_threshold", "length_temperature", "length_tilt"):
        if s[k] is not None:
            s[k] = float(s[k])

    # Checked against the file this run will ACTUALLY read, not the artifact's, so a
    # `--root` override cannot walk past it.
    if prod is not None and str(s["root"]) == str(prod["run"].get("train_path")):
        raise SystemExit(
            "the eval file is the file this checkpoint TRAINED on — that is not a "
            "closure test"
        )

    # THE scale check. Only meaningful when tau came from the artifact AND the notebook
    # is applying the same (T, tilt) it was fitted under -- an override of either breaks
    # the pairing, so say which one and stop.
    if prod is not None and a.empty_threshold is None:
        under = prod["empty_tree"]["tau"].get("fitted_under")
        if under is None:
            raise SystemExit(
                "this artifact records no scale for its tau (a prod_test run predating the "
                "fix). Re-run notebooks/prod_test_v1.ipynb: a tau without its scale cannot "
                "be applied."
            )
        if not (abs(float(under["length_temperature"]) - s["length_temperature"]) < 1e-9
                and abs(float(under["length_tilt"]) - s["length_tilt"]) < 1e-9):
            raise SystemExit(
                f"EMPTY_THRESHOLD was fitted at (T, tilt) = "
                f"({under['length_temperature']}, {under['length_tilt']}) but this run "
                f"applies ({s['length_temperature']}, {s['length_tilt']}). A tau is a "
                f"quantile of q(0|x); on a different scale it cuts in the wrong place."
            )

    if a.mbr_backend == "surrogate":
        print("[warn] MBR_BACKEND='surrogate' is a DIFFERENT risk function — fine for "
              "iterating on the figures, never for a reported number.")
    if a.require_truth_splitting:
        print("[warn] --require-truth-splitting selects on the answer: it removes exactly "
              "the jets whose truth is the empty tree, which is the blind spot v2 exists "
              "to close.")
    return s, prod


# ============================================================================
# 1. Helpers — binning, histograms, panels (as in the notebook)
# ============================================================================
def edges(key):
    """The app's edges coarsened by REBIN[key] -- a strict subset of them."""
    n, lo, hi = BINS[key]
    r = int(REBIN.get(key, 1))
    if n % r:
        raise ValueError(f"REBIN[{key}]={r} does not divide the app's {n} bins")
    return np.linspace(lo, hi, n // r + 1)


def h1_sumw2(values, weights, e, w2=None):
    """Weighted counts and their Sumw2 errors -- ROOT's TH1::Sumw2 convention.

    `w2` overrides the sum-of-squares term. It exists for the bootstrap, where an entry
    standing in for `c` independent copies contributes `c*w**2`, not `(c*w)**2`.
    """
    c = np.histogram(values, bins=e, weights=weights)[0]
    sq = np.asarray(weights, float) ** 2 if w2 is None else w2
    s2 = np.histogram(values, bins=e, weights=sq)[0]
    return c, np.sqrt(s2)


def density(counts, err, e):
    """Unit-area normalisation with the error propagated through it."""
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


def zoom(ax, ys, e, pad=0.03):
    """Keep the app's binning, view only the populated range (a view limit only)."""
    tot = np.sum(np.atleast_2d(np.asarray(ys, dtype=float)), axis=0)
    hit = np.flatnonzero(tot > 0)
    if hit.size == 0:
        return
    lo, hi = e[hit[0]], e[hit[-1] + 1]
    m = pad * (hi - lo)
    ax.set_xlim(lo - m, hi + m)


def finish(ax, xlabel="", title="", ylabel="", logy=False, legend=False):
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7.5, loc="best")


def ratio_axes(n_cols, n_rows=1, w=4.1, h=3.4):
    """A grid of (main, ratio-to-truth) panel pairs -- the standard HEP layout.

    Nested gridspecs, because the two gaps are not the same gap: a main panel and its
    ratio strip must sit flush, while consecutive ROWS need room for the next title.
    """
    fig = plt.figure(figsize=(w * n_cols, h * n_rows))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.46, wspace=0.28)
    pairs = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.07)
            a = fig.add_subplot(inner[0])
            b = fig.add_subplot(inner[1], sharex=a)
            a.tick_params(labelbottom=False)
            b.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
            b.set_ylim(0.0, 2.0)
            b.set_ylabel("/ truth", fontsize=7.5)
            row.append((a, b))
        pairs.append(row)
    return fig, pairs


def ratio_of(num, den):
    """Bin-wise ratio, masked where truth has no support (0/0 is not 1)."""
    out = np.full_like(np.asarray(num, dtype=float), np.nan)
    ok = np.asarray(den, dtype=float) > 0
    out[ok] = np.asarray(num, dtype=float)[ok] / np.asarray(den, dtype=float)[ok]
    return out


def fig_legend(fig, ax, title, ncols=5, top=0.80):
    """One legend per figure, in reserved space -- never on top of the data."""
    h, lab = ax.get_legend_handles_labels()
    fig.subplots_adjust(top=top)
    fig.suptitle(title, x=0.006, y=1.005, ha="left")
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.006, 0.955),
               ncols=ncols, fontsize=8, frameon=False)


def md_table(header, rows):
    w = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) for i in range(len(header))]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(header)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def fmt(x, p=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{p}g}"


def pct(x, p=2):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{p}f}%"


def gmean(xs):
    xs = np.array([x for x in xs if np.isfinite(x) and x > 0], dtype=float)
    return float(np.exp(np.mean(np.log(xs)))) if xs.size else float("nan")


class Figures:
    """Numbered PDF figures + the captions the report references them by.

    A figure is written the moment it is built and then closed, so a long run never
    holds more than one open, and a crash in section 9 still leaves sections 1-8 on
    disk in their final form.
    """

    def __init__(self, out_dir: Path, png: bool):
        self.dir = out_dir / "figures"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.png = png
        self.items = []

    def add(self, fig, slug, caption):
        n = len(self.items) + 1
        stem = f"fig{n:02d}_{slug}"
        pdf = self.dir / f"{stem}.pdf"
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        png = None
        if self.png:
            png = self.dir / f"{stem}.png"
            fig.savefig(png, format="png", bbox_inches="tight")
        plt.close(fig)
        self.items.append({"n": n, "slug": slug, "caption": caption,
                           "pdf": pdf, "png": png})
        print(f"  figure {n}: {pdf.relative_to(self.dir.parent)}")
        return n

    def ref(self, n):
        return f"[Figure {n}](figures/{self.items[n - 1]['pdf'].name})"

    def block(self, n):
        """The figure's own block in the report: preview (if any), link, caption."""
        it = self.items[n - 1]
        lines = []
        if it["png"] is not None:
            lines.append(f"![Figure {it['n']}](figures/{it['png'].name})")
            lines.append("")
        lines.append(f"**Figure {it['n']}** — {it['caption']}  \n"
                     f"[`figures/{it['pdf'].name}`](figures/{it['pdf'].name})")
        return "\n".join(lines)

    def index(self):
        return md_table(
            ["#", "file", "what it shows"],
            [[it["n"], f"[`figures/{it['pdf'].name}`](figures/{it['pdf'].name})",
              it["caption"]] for it in self.items])


# ============================================================================
# 2. Model, sample, evaluation pass
# ============================================================================
def load_model(a, s):
    seed_everything(a.seed)
    device = select_device() if a.device == "auto" else torch.device(a.device)
    ckpt = (REPO / s["ckpt"]) if s["ckpt"] else newest("runs/**/best.ckpt")
    if ckpt is None:
        ckpt = newest("runs/**/last.ckpt")
    if ckpt is None:
        raise SystemExit(f"no checkpoint under {REPO / 'runs'} -- train one first")

    info = load_for_inference(ckpt)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(device)
    model.load_state_dict(info["model_state"])
    model.eval()

    decode = decode_params(cfg)
    # The CLI/artifact values win over the snapshot when set, mirroring how
    # `cli.py::cmd_eval` re-applies a lifted `decode.length_temperature`. Setting the
    # model attributes is what makes them reach `length_pmf` AND `sample`; writing them
    # back into `decode` is what makes them appear in the metrics JSON.
    if s["length_temperature"] is not None:
        decode["length_temperature"] = float(s["length_temperature"])
    if s["length_tilt"] is not None:
        decode["length_tilt"] = float(s["length_tilt"])
    model.length_temperature = float(decode["length_temperature"])
    model.length_tilt = float(decode["length_tilt"])
    beam = {k: decode[k] for k in ARJunipr._BEAM_KEYS if k in decode}
    aux = tuple(model.aux_feature_names)
    # The contract flag: every family that can draw ln z / psi sets it, and only
    # ar_junipr_v1 does not. When False the nodes carry ln_z = 0, psi = 0 PLACEHOLDERS,
    # so those panels must not plot them as predictions.
    cont = bool(getattr(model, "has_continuous_coords", False))

    if a.plane_bins % geom.n_bins:
        raise SystemExit(
            f"--plane-bins {a.plane_bins} must be a multiple of geometry.n_bins "
            f"({geom.n_bins}) so the Lund-plane edges stay a strict subset of the "
            f"model's own cells"
        )
    return dict(device=device, ckpt=ckpt, info=info, cfg=cfg, geom=geom, model=model,
                decode=decode, beam=beam, aux=aux, cont=cont)


def load_sample(a, s, m):
    """The test file, the deployable selection, and the grooming provenance."""
    root = s["root"]
    jets = load_rntuple(str(REPO / root), a.ntuple) if root else None
    source = "rntuple"
    if jets is None:
        source = "synthetic"
        print("no ROOT file -> synthetic fallback (no grooming provenance, weights 1.0)")
        jets = synthetic_matched_dataset(max(4 * a.jets, 2000), seed=a.seed)

    jets = select_pt_range(jets, var=a.pt_var, lo=a.pt_min, hi=a.pt_max)

    # len(x)>0 drops jets with no conditioning information at all, and is a cut any
    # analysis can make on data. len(y)>0 reads the parton truth, so it is NOT.
    n_in = len(jets)
    n_x0 = sum(1 for j in jets if not len(j["x"][0]))
    n_y0_of_x = sum(1 for j in jets if len(j["x"][0]) and not len(j["y"][0]))
    jets = [j for j in jets
            if len(j["x"][0]) and (len(j["y"][0]) or not a.require_truth_splitting)]
    if not jets:
        raise SystemExit("no jets survived the selection")

    try:
        ds = MatchedLundDataset(jets, m["geom"], aux_features=m["aux"])
    except Exception as exc:   # aux_vector rejects the absent-column sentinels
        raise SystemExit(
            f"the checkpoint was trained with aux inputs {m['aux']} but {root} cannot "
            f"supply them ({exc}). Point --root at a file written with the aux columns, "
            f"e.g. cpp/test_data/jets_aux.root."
        ) from exc

    w_all = np.array([float(j.get("weight", 1.0)) for j in jets], dtype=float)
    sd_known = "z_cut" in jets[0] and jets[0]["z_cut"] is not None
    z_cut = float(jets[0]["z_cut"]) if sd_known else float("nan")
    beta = float(jets[0].get("beta", 0.0) or 0.0) if sd_known else float("nan")
    # Global ln z floor implied by soft drop over the whole angular window:
    #   z > z_cut (dR/R0)^beta   <=>   ln z > ln z_cut - beta * ln(1/dR)
    lnz_floor = (math.log(z_cut) - beta * m["geom"].ln_invdelta_range[1]
                 if sd_known else -10.0)
    return dict(jets=jets, ds=ds, source=source, w_all=w_all, n_in=n_in, n_x0=n_x0,
                n_y0_of_x=n_y0_of_x, sd_known=sd_known, z_cut=z_cut, beta=beta,
                lnz_floor=lnz_floor,
                mean_x=float(np.mean([len(j["x"][0]) for j in jets])),
                mean_y=float(np.mean([len(j["y"][0]) for j in jets])),
                p_y0=float(np.mean([len(j["y"][0]) == 0 for j in jets])))


def pe_coords(pe):
    """LundPointEstimate -> (n, 4) in node_raw column order."""
    if not pe.nodes:
        return np.zeros((0, 4))
    return np.array([[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes],
                    dtype=float)


@torch.inference_mode()
def draw_coords(model, xf, nx, cells):
    """One posterior draw's CONTINUOUS coordinates, sampled rather than moded.

    `model.sample_coordinates` is the contract hook: `sample` returns cell chains only,
    and placing those at cell centres would leave ln z and psi holding a filler
    constant. Families without a coordinate density return None, and the cell-centre
    fallback is then the honest answer -- flagged by `cont`, not silently plotted.
    """
    if not len(cells):
        return np.zeros((0, 4))
    c = model.sample_coordinates(xf, nx, list(cells))
    if c is None:
        return pe_coords(model.describe_cells(xf, nx, cells))
    return np.asarray(c.cpu().double().numpy(), dtype=float).reshape(-1, 4)


@torch.inference_mode()
def ar_kappa(model, xf, nx, cells):
    """Per-node von Mises concentration -- the psi caveat panel only, AR-only."""
    L = len(cells)
    if L == 0 or not isinstance(model, ARJunipr) or not model.continuous_coords:
        return np.zeros(0)
    dev = xf.device
    e = model.encode(xf, nx)
    yc = torch.tensor([[int(c) for c in cells]], dtype=torch.long, device=dev)
    out = model._decode_states(yc, e, model.xattn_kv(xf, nx))
    eh = torch.cat([out, e.unsqueeze(1).expand(-1, L + 1, -1)], dim=-1)[:, :L, :]
    *_, kappa = model._coord_params(torch.cat([eh, model.y_embed(yc)], dim=-1))
    # .cpu() BEFORE .double(): MPS has no float64, so casting on-device raises.
    return kappa.squeeze(0).cpu().double().numpy()


def eval_jets(index, a, m, d):
    """The single pass. One set of K draws per jet feeds THREE consumers -- the learned
    length floor, the MBR risk minimisation and the posterior series -- so nothing is
    sampled twice. Also records `n_map0`, the MAP re-decoded with the floor lifted."""
    model, device, ds, jets = m["model"], m["device"], d["ds"], d["jets"]
    rng = np.random.default_rng(a.seed)
    raw = {s: [] for s in SERIES}
    wjet, kappas, risks, n_map0 = [], [], [], []
    for i in index:
        item = ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)

        draws = model.sample(xf, nx, n=a.draws)
        mults = np.array([len(x) for x in draws], dtype=int)

        # learned per-jet MAP floor, reusing the draws above (no second sample)
        eff = learned_min_emissions(model, xf, nx, quantile=a.length_floor_quantile,
                                    base_floor=1, mults=mults)
        mp = model.map_estimate(xf, nx, **{**m["beam"], "min_emissions": eff})
        # control: the same beam search with the >=1 floor lifted, so the empty tree is
        # reachable at all.
        n_map0.append(
            model.map_estimate(xf, nx, **{**m["beam"], "min_emissions": 0}).multiplicity
            if not a.no_map_allow_empty else -1
        )
        # MBR: the drawn tree of least expected perturbative-Lund EMD to the posterior
        mbr = model.map_or_mbr(xf, nx, draws=draws,
                               **{**m["decode"], "point_estimator": "mbr",
                                  "mbr_backend": a.mbr_backend,
                                  "mbr_n_candidates": a.mbr_candidates})
        # posterior predictive: one draw per jet, coordinates sampled where possible
        pick = draws[int(rng.integers(len(draws)))] if draws else []
        pv = draw_coords(model, xf, nx, pick)

        raw["truth"].append(np.asarray(item["yraw"].numpy(), dtype=float))
        raw["rsd"].append(np.asarray(node_raw(*jets[i]["x"]), dtype=float))
        raw["map"].append(pe_coords(mp))
        raw["mbr"].append(pe_coords(mbr))
        raw["post"].append(pv)
        wjet.append(d["w_all"][i])
        kappas.append(ar_kappa(model, xf, nx, pick))
        risks.append(mbr.risk if mbr.risk is not None else np.nan)
    return (raw, np.array(wjet), kappas, np.array(risks, dtype=float),
            np.array(n_map0, dtype=int))


def pack(arrays, weights):
    """Flatten per-jet (n,4) arrays into one splitting-level table.

    Every splitting carries its jet's weight, its splitting index t and its jet id -- so
    slicing by splitting index downstream is a boolean mask, not a re-loop.
    """
    keep = [(i, x, w) for i, (x, w) in enumerate(zip(arrays, weights)) if len(x)]
    if not keep:
        return {"v": np.zeros((0, 4)), "w": np.zeros(0), "t": np.zeros(0, int),
                "jet": np.zeros(0, int)}
    return {
        "v": np.concatenate([x for _, x, _ in keep]),
        "w": np.concatenate([np.full(len(x), w, dtype=float) for _, x, w in keep]),
        "t": np.concatenate([np.arange(len(x)) for _, x, _ in keep]),
        # index into the EVALUATED set, so the same id means the same jet in every
        # series even though different series drop different jets
        "jet": np.concatenate([np.full(len(x), i, dtype=int) for i, x, _ in keep]),
    }


# ============================================================================
# 3. Distances
# ============================================================================
def _wecdf(x, w, grid):
    """Weighted ECDF of (x, w) evaluated on `grid`."""
    o = np.argsort(np.asarray(x, float))
    xs = np.asarray(x, float)[o]
    cw = np.cumsum(np.asarray(w, float)[o])
    cw = cw / cw[-1]
    idx = np.searchsorted(xs, grid, side="right")
    return np.where(idx > 0, cw[np.clip(idx - 1, 0, None)], 0.0)


def w1(a, wa, b, wb):
    """Wasserstein-1. Weight-aware."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(wasserstein_distance(a, b, wa, wb))


def w1_circular(a, wa, b, wb, lo, hi, n=2048):
    """Wasserstein-1 on a CIRCLE -- the right distance for psi.

    Linear W1 measures the +pi/-pi wrap as a 2*pi transport, so two identical azimuthal
    distributions read as maximally far apart purely because of where the branch cut
    falls. On the circle the optimal plan may rotate:
        W1 = min_theta Int |F_a - F_b - theta| dx
    (Delon, Salomon & Sobolevski 2010), and on a uniform grid the minimiser is the
    median of the CDF difference.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    L = hi - lo
    g = lo + (np.arange(n) + 0.5) * L / n
    dd = _wecdf(a, wa, g) - _wecdf(b, wb, g)
    return float(np.mean(np.abs(dd - np.median(dd))) * L)


def ks_w(a, wa, b, wb):
    """Two-sample KS statistic off the WEIGHTED ECDFs.

    `scipy.stats.ks_2samp` ignores weights; with per-jet ROOT weights in play that would
    silently answer a different question.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    grid = np.union1d(np.asarray(a, float), np.asarray(b, float))
    return float(np.max(np.abs(_wecdf(a, wa, grid) - _wecdf(b, wb, grid))))


def kuiper_w(a, wa, b, wb):
    """Kuiper's V = sup(F_a - F_b) + sup(F_b - F_a) -- the circular KS.

    KS depends on where the circle was cut open; Kuiper's statistic is invariant under
    rotation, which is what an azimuth needs. Its scale differs from KS (V in [0,2]), so
    psi rows are only ever compared against psi rows.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    grid = np.union1d(np.asarray(a, float), np.asarray(b, float))
    dd = _wecdf(a, wa, grid) - _wecdf(b, wb, grid)
    return float(max(dd.max(), 0.0) + max((-dd).max(), 0.0))


def chi2_ndf(a, wa, b, wb, e, w2a=None, w2b=None):
    """Shape chi2/ndf on shared bins, unit-area normalised, Sumw2 errors.

    Bins where NEITHER distribution has an error are dropped, not counted -- an
    empty-empty bin is agreement about nothing and would deflate the statistic.
    ndf = (bins used) - 1; the one constraint is the shared normalisation.
    """
    pa, sa = density(*h1_sumw2(a, wa, e, w2a), e)
    pb, sb = density(*h1_sumw2(b, wb, e, w2b), e)
    var = sa ** 2 + sb ** 2
    use = var > 0
    ndf = int(use.sum()) - 1
    if ndf < 1:
        return float("nan")
    return float(np.sum((pa[use] - pb[use]) ** 2 / var[use]) / ndf)


def dists(a, wa, b, wb, e, circular, w2a=None, w2b=None):
    if circular is None:
        d = {"w1": w1(a, wa, b, wb), "ks": ks_w(a, wa, b, wb)}
    else:
        d = {"w1": w1_circular(a, wa, b, wb, *circular), "ks": kuiper_w(a, wa, b, wb)}
    d["chi2"] = chi2_ndf(a, wa, b, wb, e, w2a, w2b)
    return d


def noise_floor(v, w, j, e, rng, n_boot, floor_pct, circular=None):
    """The distance you measure between TRUTH and ITSELF at this sample size.

    Two independent bootstrap resamples of truth, scored with the same three distances.
    Whatever comes back is pure sampling noise, so a plain-RSD distance at or below it
    carries no information -- and an improvement ratio dividing by it is meaningless,
    not merely imprecise.

    The bootstrap resamples JETS, not splittings: splittings within a jet share a weight
    and correlated kinematics, so a splitting-level resample would treat them as
    independent and understate the floor.
    """
    if len(v) < 8:
        return {m: float("nan") for m in MET}
    _, jc = np.unique(j, return_inverse=True)   # jet id -> 0..nb-1
    nb = int(jc.max()) + 1
    p = np.full(nb, 1.0 / nb)
    acc = {m: [] for m in MET}
    w2 = w ** 2
    for _ in range(n_boot):
        # A bootstrap resample IS a multinomial reweighting of the blocks, and W1/KS are
        # linear in the weights -- so the resample is a weight vector, not an index
        # shuffle. chi2 is the exception: a jet drawn c times stands for c INDEPENDENT
        # entries, contributing c*w**2 and not (c*w)**2, so its sum-of-squares term is
        # passed explicitly. Folding it into the weight would inflate the error bars,
        # deflate the null chi2, and wave through rows that are really inside the noise.
        ca, cb = rng.multinomial(nb, p)[jc], rng.multinomial(nb, p)[jc]
        wa, wb = w * ca, w * cb
        if wa.sum() <= 0 or wb.sum() <= 0:
            continue
        dd = dists(v, wa, v, wb, e, circular, w2a=ca * w2, w2b=cb * w2)
        for m in MET:
            acc[m].append(dd[m])
    return {m: (float(np.nanpercentile(acc[m], floor_pct))
                if len(acc[m]) and np.isfinite(acc[m]).any() else float("nan"))
            for m in MET}


def compare(vals, wts, jids, e, rng, n_boot, floor_pct, circular=None):
    """All three distances of every model series against truth, the ratios, and the
    noise floor that decides whether those ratios mean anything."""
    out = {}
    ref_v, ref_w = vals["truth"], wts["truth"]
    base = {}
    for s in ("rsd",) + MODELS:
        if s not in vals or len(vals[s]) < 2:
            continue
        dd = dists(vals[s], wts[s], ref_v, ref_w, e, circular)
        if s == "rsd":
            base = dd
        out[s] = dd
    for s, dd in out.items():
        for m in MET:
            b = base.get(m, float("nan"))
            dd[m + "_r"] = (dd[m] / b) if (np.isfinite(b) and b > 0) else float("nan")
    floor = noise_floor(ref_v, ref_w, jids["truth"], e, rng, n_boot, floor_pct, circular)
    # scoreable == plain RSD is measurably further from truth than truth is from itself.
    ok = {m: bool(np.isfinite(floor[m]) and np.isfinite(base.get(m, float("nan")))
                  and base[m] > floor[m]) for m in MET}
    return {"series": out, "floor": floor, "scoreable": ok}


# ============================================================================
# 4. The figures
# ============================================================================
def figure_lund_planes(figs, R, a):
    """Section 5: the plane itself, and the ratio map on the model's own cells."""
    geom, POOL, w_jet = R["geom"], R["POOL"], R["w_jet"]
    U_LO, U_HI = geom.ln_invdelta_range
    V_LO, V_HI = geom.ln_kt_range

    def plane_edges(nb):
        return [np.linspace(U_LO, U_HI, nb + 1), np.linspace(V_LO, V_HI, nb + 1)]

    def plane(s, e, nb):
        """Splittings per jet per unit Lund area, plus the raw counts behind it."""
        p = POOL[s]
        h = np.histogram2d(p["v"][:, 0], p["v"][:, 1], bins=e, weights=p["w"])[0]
        n = np.histogram2d(p["v"][:, 0], p["v"][:, 1], bins=e)[0]
        area = (U_HI - U_LO) * (V_HI - V_LO) / nb ** 2
        return h / (w_jet.sum() * area), n

    nb = a.plane_bins
    PE = plane_edges(nb)
    PLANES, PCOUNT = {}, {}
    for s in SERIES:
        PLANES[s], PCOUNT[s] = plane(s, PE, nb)
    # A single hot bin would flatten the whole ramp on a shared linear scale, so the top
    # of the scale is a high percentile of the populated bins, not the maximum.
    vmax = float(np.percentile(np.concatenate([P[P > 0] for P in PLANES.values()]), 99))

    # View limits only -- the binning stays on the model's full window, but a 100 GeV
    # sample populates a corner of it.
    hit = np.sum([P for P in PLANES.values()], axis=0) > 0
    iu, iv = np.flatnonzero(hit.any(axis=1)), np.flatnonzero(hit.any(axis=0))
    xlim = (PE[0][max(iu[0] - 1, 0)], PE[0][min(iu[-1] + 2, nb)])
    ylim = (PE[1][max(iv[0] - 1, 0)], PE[1][min(iv[-1] + 2, nb)])

    fig, axes = plt.subplots(1, 5, figsize=(19.0, 3.9), sharey=True)
    for ax, s in zip(axes, SERIES):
        P = np.ma.masked_where(PLANES[s] <= 0, PLANES[s])
        im = ax.pcolormesh(PE[0], PE[1], P.T, cmap=CMAP, vmin=0.0, vmax=vmax,
                           shading="flat", rasterized=True)
        ax.set_title(R["STYLE"][s][2])
        ax.set_xlabel(LABEL["lnInvDelta"])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)
    axes[0].set_ylabel(LABEL["lnkt"])
    fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012,
                 label=r"$\rho$  [splittings / jet / unit area]")
    fig.suptitle("Primary Lund plane density  (view cropped to the populated region; "
                 f"binning is the full {geom.ln_invdelta_range} x {geom.ln_kt_range} "
                 "window)", x=0.09, ha="left")
    n1 = figs.add(fig, "lund_plane",
                  r"Primary Lund plane density $\rho(\ln 1/\Delta R, \ln k_t)$, "
                  "weighted splittings per jet per unit area, one panel per series. "
                  "MAP and MBR are striped because they place nodes at the coordinate "
                  "head's mode inside a discrete cell; the posterior samples the "
                  "offsets, which is why only it looks smooth.")

    # The ratio map is rebinned to the GEOMETRY's own cells: at plane_bins=30 there is
    # barely one splitting per bin, so a ratio there is Poisson noise in saturated
    # colour. The model's cell grid is both the coarser binning the statistics support
    # and the resolution at which the model actually makes decisions.
    RLO, RHI, N_MIN = 0.4, 2.5, 12
    RE = plane_edges(geom.n_bins)
    RPLANE, RCOUNT = {}, {}
    for s in SERIES:
        RPLANE[s], RCOUNT[s] = plane(s, RE, geom.n_bins)

    def plane_ratio(num, den, n_den, n_num):
        """Ratio map, gated on truth having enough entries to divide by. A bin where
        TRUTH is empty but the prediction is not is a real disagreement, so it saturates
        the top of the scale rather than being blanked -- blanking would hide invented
        emissions."""
        out = np.full(num.shape, np.nan)
        ok = (den > 0) & (n_den >= N_MIN)
        out[ok] = num[ok] / den[ok]
        out[(den <= 0) & (n_num >= N_MIN)] = RHI
        return np.ma.masked_invalid(out)

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.9), sharey=True)
    for ax, s in zip(axes, ("rsd", "map", "mbr", "post")):
        Rm = plane_ratio(RPLANE[s], RPLANE["truth"], RCOUNT["truth"], RCOUNT[s])
        im = ax.pcolormesh(RE[0], RE[1], Rm.T, cmap=DIV,
                           norm=mpl.colors.LogNorm(vmin=RLO, vmax=RHI),
                           shading="flat", rasterized=True)
        ax.set_title(f"{R['STYLE'][s][2]}  /  truth")
        ax.set_xlabel(LABEL["lnInvDelta"])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)
    axes[0].set_ylabel(LABEL["lnkt"])
    fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012, label="ratio to truth")
    fig.suptitle(f"Where each estimate over- and under-populates the plane, on the "
                 f"model's own {geom.n_bins}x{geom.n_bins} cells   (grey = agrees, "
                 f"blue = too few, red = too many, blank = fewer than {N_MIN} truth "
                 f"splittings)", x=0.06, ha="left")
    n2 = figs.add(fig, "lund_plane_ratio",
                  "Ratio to truth of the same density, rebinned to the geometry's own "
                  f"{geom.n_bins}x{geom.n_bins} cells. Grey is agreement, blue too few "
                  "splittings, red too many; blank means fewer than "
                  f"{N_MIN} truth splittings to divide by.")
    return n1, n2


def marginal_panel(ax, rax, key, R, series=SERIES, t_mask=None, title=""):
    e = edges(key)
    dens = {}
    for s in series:
        p = R["POOL"][s]
        m = np.ones(len(p["v"]), dtype=bool) if t_mask is None else t_mask(p)
        c, err = h1_sumw2(p["v"][m, COL[key]], p["w"][m], e)
        dens[s] = density(c, err, e)[0]
    if "rsd" in dens:
        fill(ax, dens["rsd"], e, C_RSD_F, C_RSD_E, label=R["STYLE"]["rsd"][2])
    for s in series:
        if s == "rsd":
            continue
        col, ls, lab = R["STYLE"][s]
        step(ax, dens[s], e, col, label=lab, lw=2.2 if s == "truth" else 1.7,
             ls=ls, z=5 if s == "truth" else 3)
        if s != "truth":
            step(rax, ratio_of(dens[s], dens["truth"]), e, col, lw=1.4, ls=ls)
    if "rsd" in dens:
        step(rax, ratio_of(dens["rsd"], dens["truth"]), e, C_RSD_E, lw=1.2)
    zoom(ax, list(dens.values()), e)
    rax.set_xlim(ax.get_xlim())
    finish(ax, title=title, ylabel="density")
    rax.set_xlabel(LABEL[key])
    return dens


def figure_marginals(figs, R):
    """Section 6: pooled coordinate marginals, and the kappa caveat panel."""
    cont, KEYS = R["cont"], R["KEYS"]
    fig, pairs = ratio_axes(4, 1, w=4.3, h=3.5)
    for (ax, rax), key in zip(pairs[0], ["lnInvDelta", "lnkt", "lnz", "psi"]):
        # Only a family with NO coordinate density (ar_junipr_v1) leaves ln z / psi as
        # placeholders; plotting a filler constant as a prediction would be a lie, so
        # those series are dropped instead.
        ser = SERIES if (cont or key in ("lnInvDelta", "lnkt")) else ("truth", "rsd")
        marginal_panel(ax, rax, key, R, series=ser,
                       title=LABEL[key] + ("" if cont or key in ("lnInvDelta", "lnkt")
                                           else "   (no coordinate density)"))
    if not cont:
        for c in (2, 3):
            pairs[0][c][0].text(0.03, 0.97, "model series dropped: this family\nhas no "
                                "coordinate density (ln z, psi unset)", ha="left",
                                va="top", transform=pairs[0][c][0].transAxes,
                                fontsize=7.5, color=MUTED,
                                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))
    fig_legend(fig, pairs[0][0][0],
               "Groomed-observable marginals, pooled over every splitting and jet")
    n = figs.add(fig, "marginals_pooled",
                 "Coordinate marginals pooled over every splitting and jet, unit-area "
                 "normalised on the C++ app's edges coarsened by REBIN, with the ratio "
                 "to truth under each panel. These are DENSITIES: the ~20-30% excess "
                 "rate of hadron-level splittings is divided out here on purpose, and "
                 "lives instead in the rate table and the $k_t$-cut spectrum.")

    n_k = None
    KAPPA = R["KAPPA"]
    if cont and len(KAPPA):
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        ax.hist(KAPPA, bins=np.linspace(0, max(2.0, np.percentile(KAPPA, 99)), 40),
                color=C_POST, alpha=0.85)
        ax.axvline(1.0, color=MUTED, lw=1.0, ls="--")
        q = np.mean(KAPPA < 1.0)
        finish(ax, xlabel=r"von Mises $\kappa$ of the $\psi$ head", ylabel="splittings",
               title=rf"$\psi$ is informative only where $\kappa$ is large "
                     rf"({100 * q:.0f}% of splittings have $\kappa<1$)")
        n_k = figs.add(fig, "psi_kappa",
                       r"Concentration $\kappa$ of the von Mises $\psi$ head. As "
                       r"$\kappa \to 0$ the conditional becomes uniform and its MODE — "
                       r"which is what MAP and MBR report — carries no information, so "
                       r"this panel says how much to discount the $\psi$ marginal above.")
    return n, n_k


def figure_marginals_by_t(figs, R, t_slices):
    """Section 7: the same marginals, split by splitting index."""
    t_labels = [f"$t={t}$" for t in t_slices] + [f"$t\\geq{t_slices[-1] + 1}$"]
    t_masks = [(lambda p, t=t: p["t"] == t) for t in t_slices] + \
              [(lambda p: p["t"] > t_slices[-1])]
    keys_t = R["KEYS"] if R["cont"] else ["lnInvDelta", "lnkt"]

    fig, pairs = ratio_axes(len(t_masks), len(keys_t), w=3.5, h=3.1)
    for row, key in zip(pairs, keys_t):
        for (ax, rax), tm, tl in zip(row, t_masks, t_labels):
            marginal_panel(ax, rax, key, R, t_mask=tm, title=f"{LABEL[key]}   {tl}")
    fig_legend(fig, pairs[0][0][0], "Coordinate marginals split by splitting index",
               top=0.90 if len(keys_t) > 2 else 0.84)
    return figs.add(fig, "marginals_by_splitting_index",
                    "The same marginals, one column per splitting index. $t=0$ is the "
                    "hardest and best determined and every series should agree there; "
                    "disagreement that GROWS with $t$ is the expected signature, "
                    "disagreement already at $t=0$ is not.")


def figure_ladder(figs, R, t_ladder):
    """Section 8: mean coordinate vs splitting index, and the survival curve."""
    POOL, NSPL, w_jet, STYLE = R["POOL"], R["NSPL"], R["w_jet"], R["STYLE"]

    def profile(s, col):
        mu, lo, hi = [], [], []
        p = POOL[s]
        for t in range(t_ladder):
            m = p["t"] == t
            if m.sum() < 3:
                mu.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            v, w = p["v"][m, col], p["w"][m]
            o = np.argsort(v)
            cw = np.cumsum(w[o]) / w[o].sum()
            mu.append(float(np.average(v, weights=w)))
            lo.append(float(v[o][np.searchsorted(cw, 0.16)]))
            hi.append(float(v[o][np.searchsorted(cw, 0.84)]))
        return np.array(mu), np.array(lo), np.array(hi)

    ts = np.arange(t_ladder)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.6))
    for ax, (col, key) in zip(axes[:2], [(0, "lnInvDelta"), (1, "lnkt")]):
        for s in SERIES:
            c, ls, lab = STYLE[s]
            mu, lo, hi = profile(s, col)
            ax.fill_between(ts, lo, hi, color=c, alpha=0.10, lw=0, zorder=1)
            ax.plot(ts, mu, color=c, ls=ls, marker="o", ms=3.4,
                    lw=2.2 if s == "truth" else 1.7, label=lab, zorder=4)
        finish(ax, xlabel="splitting index $t$", ylabel="weighted mean " + LABEL[key],
               title=LABEL[key] + " ladder")

    for s in SERIES:
        c, ls, lab = STYLE[s]
        surv = [float(w_jet[NSPL[s] > t].sum() / w_jet.sum()) for t in ts]
        axes[2].plot(ts, surv, color=c, ls=ls, marker="o", ms=3.4,
                     lw=2.2 if s == "truth" else 1.7, label=lab)
    axes[2].set_yscale("log")
    finish(axes[2], xlabel="splitting index $t$",
           ylabel="fraction of jets reaching $t$", title="survival")
    # Trim to the depths any series actually reaches -- t_ladder is an upper bound.
    t_max = max((int(NSPL[s].max()) for s in SERIES), default=1)
    for ax in axes:
        ax.set_xlim(-0.3, min(t_ladder - 1, t_max) + 0.3)
    fig.tight_layout()
    fig_legend(fig, axes[0],
               "Does the predicted sequence march inward the way QCD demands?", top=0.78)
    return figs.add(fig, "ladder_profiles",
                    r"Weighted mean $\ln 1/\Delta R$ and $\ln k_t$ against splitting "
                    r"index (band = weighted 16–84 percentile), plus the survival "
                    r"curve. Angular ordering requires the declustering to march "
                    r"inward: $\ln 1/\Delta R$ rising, $\ln k_t$ falling. A FLAT ladder "
                    r"means the pooled marginals were learned without the sequence — "
                    r"which the pooled panels cannot detect.")


def figure_kt_cut(figs, R):
    """Section 9: N(ln kt > c) per jet, the one panel that sees the RATE."""
    POOL, w_jet, STYLE, geom = R["POOL"], R["w_jet"], R["STYLE"], R["geom"]
    V_LO, V_HI = geom.ln_kt_range
    cuts = np.linspace(V_LO, V_HI, 25)

    def n_above(s):
        p, out = POOL[s], []
        wsum = w_jet.sum()
        for c in cuts:
            m = p["v"][:, 1] > c
            out.append(float(p["w"][m].sum() / wsum))
        return np.array(out)

    ncut = {s: n_above(s) for s in SERIES}
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.10})
    for s in SERIES:
        c, ls, lab = STYLE[s]
        axes[0].plot(cuts, ncut[s], color=c, ls=ls, lw=2.2 if s == "truth" else 1.7,
                     label=lab)
        if s != "truth":
            axes[1].plot(cuts, ratio_of(ncut[s], ncut["truth"]), color=c, ls=ls, lw=1.4)
    axes[1].axhline(1.0, color=MUTED, lw=0.8)
    axes[1].set_ylim(0.0, 2.0)
    axes[0].set_yscale("log")
    axes[0].tick_params(labelbottom=False)
    finish(axes[0], ylabel=r"$\langle N(\ln k_t > c)\rangle$ per jet",
           title=r"$k_t$-cut multiplicity spectrum", legend=True)
    finish(axes[1], xlabel=r"cut $c$ on $\ln(k_t/\mathrm{GeV})$", ylabel="/ truth")
    return figs.add(fig, "kt_cut_spectrum",
                    r"Emissions per jet above a $\ln k_t$ cut, weight-averaged, as the "
                    r"cut sweeps the geometry's range. The panel to keep if only one "
                    r"survives: multiplicity says HOW MANY, the marginals say WHERE, "
                    r"this says both at once — a model that gets the count right at the "
                    r"wrong hardness shows up as the right endpoint with the wrong "
                    r"slope.")


def figure_leading(figs, R):
    """Section 10: the hardest-kt splitting of each jet, and the multiplicity."""
    raw, w_jet, NSPL, STYLE, cont = R["raw"], R["w_jet"], R["NSPL"], R["STYLE"], R["cont"]

    def leading(s):
        """(n_jets, 4) coords of each jet's hardest-kt splitting, + its weight."""
        rows, ws = [], []
        for x, w in zip(raw[s], w_jet):
            if len(x):
                rows.append(x[int(np.argmax(x[:, 1]))])
                ws.append(w)
        return (np.array(rows) if rows else np.zeros((0, 4))), np.array(ws)

    LEAD = {s: leading(s) for s in SERIES}
    lead_keys = ["lnkt", "lnInvDelta"] + (["lnz"] if cont else [])
    fig, pairs = ratio_axes(len(lead_keys) + 1, 1, w=4.2, h=3.4)
    for (ax, rax), key in zip(pairs[0], lead_keys):
        e = edges(key)
        d = {}
        for s in SERIES:
            v, w = LEAD[s]
            d[s] = density(*h1_sumw2(v[:, COL[key]], w, e), e)[0]
        fill(ax, d["rsd"], e, C_RSD_F, C_RSD_E, label=STYLE["rsd"][2])
        for s in SERIES:
            if s == "rsd":
                continue
            c, ls, lab = STYLE[s]
            step(ax, d[s], e, c, label=lab, lw=2.2 if s == "truth" else 1.7, ls=ls)
            if s != "truth":
                step(rax, ratio_of(d[s], d["truth"]), e, c, lw=1.4, ls=ls)
        step(rax, ratio_of(d["rsd"], d["truth"]), e, C_RSD_E, lw=1.2)
        zoom(ax, list(d.values()), e)
        rax.set_xlim(ax.get_xlim())
        finish(ax, title="leading emission: " + LABEL[key], ylabel="density")
        rax.set_xlabel(LABEL[key])

    ax, rax = pairs[0][-1]
    e = edges("mult")
    dm = {s: density(*h1_sumw2(NSPL[s], w_jet, e), e)[0] for s in SERIES}
    fill(ax, dm["rsd"], e, C_RSD_F, C_RSD_E)
    for s in SERIES:
        if s == "rsd":
            continue
        c, ls, _ = STYLE[s]
        step(ax, dm[s], e, c, lw=2.2 if s == "truth" else 1.7, ls=ls)
        if s != "truth":
            step(rax, ratio_of(dm[s], dm["truth"]), e, c, lw=1.4, ls=ls)
    step(rax, ratio_of(dm["rsd"], dm["truth"]), e, C_RSD_E, lw=1.2)
    zoom(ax, list(dm.values()), e)
    rax.set_xlim(ax.get_xlim())
    finish(ax, title="multiplicity", ylabel="density")
    rax.set_xlabel(LABEL["mult"])
    fig_legend(fig, pairs[0][0][0], "Leading emission and multiplicity")
    n = figs.add(fig, "leading_and_multiplicity",
                 "The hardest-$k_t$ splitting of each jet, and the multiplicity "
                 "distribution. The leading panels are CONDITIONAL on the series having "
                 "any splitting at all, and that condition differs per series — truth is "
                 "empty for a sizeable fraction of jets while a floored MAP never is — "
                 "so read them with the empty-tree table, not as a like-for-like "
                 "comparison.")
    return n, LEAD


# ============================================================================
# 5. The report
# ============================================================================
def build_report(a, s, R, figs, tables, cmdline):
    """Every number in here is computed on THIS run.

    The notebook's prose carried hard-coded numbers from the checkpoint it was written
    against, and several stopped being true when the model changed. A conclusion that
    cannot restate itself is a conclusion that goes stale, so the verdicts below are
    assembled from the run's own results and the prose around them is only the part
    that is checkpoint-independent.
    """
    m, d, geom = R["m"], R["d"], R["geom"]
    info, cfg, decode = m["info"], m["cfg"], m["decode"]
    ET, SUPPORT, NSPL = R["EMPTY_TREE"], R["SUPPORT"], R["NSPL"]
    STYLE, cont, KAPPA = R["STYLE"], R["cont"], R["KAPPA"]
    U_LO, U_HI = geom.ln_invdelta_range
    V_LO, V_HI = geom.ln_kt_range
    L = []

    def P(*lines):
        L.extend(lines)
        L.append("")

    # --- header ---------------------------------------------------------------
    P("# Lund distribution closure — "
      + ("production test v0" if s["prod_metrics_path"] else "v2 configuration"),
      "",
      "Does the model move the hadron-level Lund distribution toward parton level, and "
      "by how much against the do-nothing baseline of using the hadron sequence "
      "unchanged? Every distance in this report divides by **plain RSD**, so its own "
      "ratio is exactly 1.0 and the plumbing checks itself.")

    if s["prod_metrics_path"]:
        # relative to the REPORT, not to the repo: the report can land anywhere
        link = os.path.relpath(REPO / s["prod_metrics_path"], R["out_dir"])
        P(f"The checkpoint, the held-out file, the **frozen** empty-tree $\\tau$ and the "
          f"fitted length recalibration are read from "
          f"[`{s['prod_metrics_path']}`]({link}) rather than pasted "
          f"in, so they cannot disagree with the fit that produced them. $\\tau$ is a "
          f"quantile of $q(0\\mid x)$, so it is checked against the scale it was fitted "
          f"under — a $\\tau$ fitted on the raw head and applied to a recalibrated one "
          f"leaves the ranking untouched and the cut in the wrong place.")
    else:
        P("**No production artifact was used** (`--no-prod-metrics`), so $\\tau$ below is "
          "rate-matched on the very sample it is reported on unless `--empty-threshold` "
          "was passed. That reproduces the fitted rate by construction: it measures the "
          "quantile function, not the model. Treat the empty-rate row as a diagnostic, "
          "not a held-out result.")

    try:   # a checkpoint outside the repo is legitimate; just print it whole
        ck = m["ckpt"].resolve().relative_to(REPO.resolve())
    except ValueError:
        ck = m["ckpt"]
    run_rows = [
        ["checkpoint", f"`{ck}`"],
        ["model", f"{info['model_name']} + {cfg.encoder.name} "
                  f"({sum(p.numel() for p in m['model'].parameters()):,} params)"],
        ["epoch / best val NLL", f"{info['epoch']} / {info['best_val_nll']:.4f}"],
        ["geometry", f"n_bins={geom.n_bins} ({geom.n_cells} cells), "
                     f"ln(1/dR) in {geom.ln_invdelta_range}, ln kt in {geom.ln_kt_range}"],
        ["aux inputs", f"`{list(m['aux'])}`" if m["aux"] else "(none)"],
        ["exact NLL", str(m["model"].exact_likelihood)],
        ["data", f"{R['d']['source']} `{s['root']}` "
                 f"({R['d']['jets'][0].get('generator', 'n/a')})"],
        ["jets evaluated", f"{R['N']:,} (of {len(d['jets']):,} kept from "
                           f"{d['n_in']:,} in the file)"],
        ["selection", "len(x)>0 AND len(y)>0 — **v1 population, uses the truth**"
                      if a.require_truth_splitting
                      else "len(x)>0 — the deployable population, no truth in the cut"],
        ["draws / jet", f"K = {a.draws}"],
        ["point estimators", f"MAP (beam, learned floor at quantile "
                             f"{a.length_floor_quantile}) and MBR "
                             f"(`{a.mbr_backend}`, {a.mbr_candidates} candidates)"],
        ["length recalibration", f"T = {decode['length_temperature']:.4f}, "
                                 f"tilt = {decode['length_tilt']:+.4f}"
                                 + ("  (identity)" if (decode["length_temperature"],
                                                       decode["length_tilt"]) == (1.0, 0.0)
                                    else "  — reaches `length_pmf` AND `sample`, so the "
                                         "posterior series, the empty rate and $\\tau$ "
                                         "all move with it")],
        ["empty-tree gate", (f"tau = {ET['gate']['tau']:.4f}"
                             + (" (rate-matched on THIS sample — circular)"
                                if ET["gate"]["rate_matched_on_this_sample"]
                                else " (frozen, fitted on held-out jets)"))
                            if ET["gate"] else "off"],
        ["device / seed", f"{m['device']} / {a.seed}"],
        ["runtime", f"{R['runtime_min']:.1f} min "
                    f"({1e3 * R['runtime_min'] * 60 / max(R['N'], 1):.0f} ms/jet)"],
    ]
    P("## Run", "", md_table(["", ""], run_rows))
    P(f"Reproduce with:", "", "```", cmdline, "```")

    # --- selection ------------------------------------------------------------
    P("## 1. The sample and the selection")
    P(f"`len(x) > 0` is a cut any analysis can make on data. `len(y) > 0` reads the "
      f"parton truth, so it is not — and it removes exactly the jets whose correct "
      f"answer is the empty tree. Of {d['n_in']:,} jets in the file, {d['n_x0']:,} have "
      f"no hadron-level splitting and are dropped; **{d['n_y0_of_x']:,} have no "
      f"parton-level splitting and are "
      + ("also dropped, reproducing v1's population"
         if a.require_truth_splitting else "KEPT") + "**.")
    P(md_table(["quantity", "value"], [
        ["jets kept by the selection", f"{len(d['jets']):,}"],
        ["jets evaluated", f"{R['N']:,}"],
        # The evaluated jets, not the whole selection: every number below is computed on
        # them, and a weighted sample's effective N is what its error bars scale with.
        ["sum of weights (evaluated)", f"{R['w_jet'].sum():.6g}"],
        ["effective N (evaluated)",
         f"{R['w_jet'].sum() ** 2 / (R['w_jet'] ** 2).sum():,.0f}"],
        ["mean hadron multiplicity $x$", f"{d['mean_x']:.3f}"],
        ["mean parton multiplicity $y$", f"{d['mean_y']:.3f}"],
        ["hadron / parton", f"{d['mean_x'] / max(d['mean_y'], 1e-9):.3f}"],
        ["$P(n_y = 0)$", pct(d["p_y0"], 1)],
        ["grooming", (f"z_cut = {d['z_cut']:.3f}, beta = {d['beta']:.3f} "
                      f"=> ln z > {d['lnz_floor']:.3f}") if d["sd_known"]
                     else "no provenance in this sample — soft-drop checks disabled"],
    ]))
    if a.require_truth_splitting:
        P("> **Warning.** `--require-truth-splitting` is a truth-level cut with no data "
          "analogue. It removes the jets where hadronisation created every splitting, so "
          "the hadron/parton excess above understates the real one.")

    # --- empty tree -----------------------------------------------------------
    P("## 2. The empty parton tree")
    P(f"On this population **{pct(ET['p_truth'], 1)}** of jets have no parton-level "
      f"primary splitting: hadronisation manufactured every splitting you see at hadron "
      f"level, and the correct prediction is *nothing*. Two questions — does the model "
      f"reproduce the **rate**, and does the estimator get those jets **right, one at a "
      f"time**. The second turns out to be a property of the decode, not of the model.")
    P(tables["empty"])
    P(f"`decode.min_emissions = {ET['min_emissions']}` forbids the MAP from returning an "
      f"empty tree at all, so its row is zero by construction; the *no floor* control "
      f"row is the same beam search with that lifted. The **MBR row moves with the risk "
      f"function** (`{ET['mbr_backend']}` here) and the others do not: the "
      f"perturbative-Lund EMD charges an imbalance penalty for unmatched weight, and an "
      f"empty cloud is nothing but unmatched weight, so its risk against every non-empty "
      f"draw is near-maximal. Do not carry this column from one backend to another.")

    q0 = ET["q0"]
    if q0["auc"] is not None:
        P(f"**The information is there and the decode is what discards it.** Before any "
          f"decode touches it, $q(0\\mid x)$ averages {q0['mean_truth_empty']:.4f} on "
          f"truth-empty jets against {q0['mean_truth_nonempty']:.4f} on the rest — "
          f"AUC **{q0['auc']:.3f}**. Its mean is {q0['mean']:.4f} against a true rate of "
          f"{ET['p_truth']:.4f}, i.e. the head is "
          + (f"under-confident by {q0['underconfidence']:.2f}x"
             if q0["underconfidence"] > 1 else
             f"over-confident by {1 / max(q0['underconfidence'], 1e-9):.2f}x")
          + ". "
          + ("The scale is right; nothing to recalibrate."
             if abs(q0["underconfidence"] - 1.0) < 0.15 else
             "The RANKING is fine and the SCALE is not — which SBC/PIT cannot see, since "
             "they rank against the sampler's own draws, so a uniformly squashed "
             "$q(N\\mid x)$ still passes. The gate works anyway because it thresholds "
             "the ranking, which a monotone squash leaves alone."))

    nf = ET.get("map_no_floor")
    if nf:
        P("**Floor or argmax?**  "
          + (f"Lifting the floor changes essentially nothing "
             f"({pct(nf['p_pred'])} predicted empty), so the floor was *not* the binding "
             f"constraint. With a multiplicity head the MAP is $\\arg\\max_n q(n\\mid x)$ "
             f"and the peak lands at $n=0$ essentially never, however much mass sits "
             f"there. That is a mode artifact, and only a different point estimator "
             f"fixes it."
             if nf["p_pred"] < 0.01 * ET["p_truth"] else
             f"Lifting the floor recovers {pct(nf['recall'], 1)} of the truth-empty "
             f"jets, so here the floor **is** costing real accuracy and the fix belongs "
             f"in `decode`."))
    g = ET.get("gate")
    if g and g["precision"] is not None:
        P(f"**The gate** — decide the empty tree when $q(0\\mid x) \\geq \\tau$, before "
          f"any shape decode — runs at $\\tau = {g['tau']:.4f}$ and scores precision "
          f"**{g['precision']:.3f}**, recall **{g['recall']:.3f}**"
          + (", rate-matched on this sample, so its $P(n=0)$ reproduces truth by "
             "construction — fit $\\tau$ on held-out jets before quoting it"
             if g["rate_matched_on_this_sample"] else
             ", with $\\tau$ frozen from the training-file val split, so this row is a "
             "genuine held-out result")
          + ". Unlike every other row here it is **backend-independent**: the gate never "
            "touches the MBR risk. The population is right; the per-jet call is not "
            "solved.")

    # --- support --------------------------------------------------------------
    P("## 3. Support and validity — what the model can and cannot produce")
    P("Three numbers decide how much of any distance below is even the model's fault. "
      "`Geometry.to_cell` **clips** rather than drops, so truth outside the geometry's "
      "ranges was piled into edge cells during training and the model can never emit "
      "there at all — whatever fraction of truth lies outside is an irreducible floor on "
      "every distance, which is why section 6 scores on the fiducial window. The groomer "
      "enforces $z > z_\\mathrm{cut}(\\Delta R/R_0)^\\beta$, so truth and plain RSD "
      "violate it exactly zero times; the coordinate head models $\\ln z$ with an "
      "*unbounded* normal and has no idea the boundary exists, so a non-zero number in "
      "that column is a real physics failure, not a plotting artefact.")
    P(tables["support"])
    P(f"Fiducial window: $\\ln(1/\\Delta R) \\in [{U_LO}, {U_HI}]$, "
      f"$\\ln k_t \\in [{V_LO}, {V_HI}]$"
      + (f", $\\ln z > {d['lnz_floor']:.3f}$" if d["sd_known"] else "") + ".")

    viol = {x: SUPPORT[x]["sd_violation"] for x in SERIES}
    floor_v = max(viol["truth"], viol["rsd"]) if np.isfinite(viol["truth"]) else np.nan
    if np.isfinite(floor_v):
        bad = [x for x in MODELS if np.isfinite(viol[x]) and viol[x] - floor_v > 1e-9]
        P("**Soft-drop verdict.** " + (
            "Every model series sits at the truth/RSD floor — no unphysical mass below "
            "the grooming boundary."
            if not bad else
            "Unphysical mass below the grooming boundary in: "
            + ", ".join(f"`{x}` ({pct(viol[x])} against a floor of {pct(floor_v)})"
                        for x in bad)
            + ". Unlike everything else in this report that is **not** a decode ceiling a "
              "better decision rule could lift — the $\\ln z$ head is an unbounded normal "
              "and knows nothing about the boundary."))

    # --- figures --------------------------------------------------------------
    P("## 4. The primary Lund plane")
    P("$\\rho(\\ln 1/\\Delta R, \\ln k_t)$ — weighted splittings per jet per unit area, "
      "on the model's own window at "
      f"{a.plane_bins} bins per axis (a multiple of `geometry.n_bins`, so the model's "
      "cell granularity shows rather than hides). At fixed coupling the primary Lund "
      "density is approximately **flat**, $\\rho \\approx 2\\alpha_s C_F/\\pi$; the "
      "running coupling tilts it toward small $k_t$ and the soft-drop and $k_t$-floor "
      "conditions cut hard edges into it. Structure the eye can see here beats any "
      "scalar in section 6: reproducing the plateau *and* the edges is doing QCD, "
      "reproducing only the marginals may just be matching one-dimensional shapes.")
    P(figs.block(R["fig_plane"]))
    P(figs.block(R["fig_plane_ratio"]))

    P("## 5. Marginals, ladders and rates")
    P("Weighted, **unit-area normalised**, on the C++ app's edges coarsened by `REBIN` — "
      "so any panel here overlays `lund_rntuple_histograms.ipynb` bin for bin.")
    P("> These are **densities, not counts**, so they will look like better $x$-vs-$y$ "
      "agreement than the same observables plotted as weighted counts. Hadronisation "
      "changes the *rate* of primary splittings a lot and their *shape* very little "
      f"(here: {d['mean_x'] / max(d['mean_y'], 1e-9):.2f}x as many hadron-level "
      "splittings as parton-level), and normalising to unit area divides that offset out "
      "on purpose — it is what makes $W_1$, KS and $\\chi^2$ *shape* distances. The rate "
      "has not gone missing; it is the $k_t$-cut spectrum and the rate table below.")
    P(figs.block(R["fig_marginals"]))
    if R["fig_kappa"]:
        P(f"**Read $\\psi$ with the $\\kappa$ panel.** The point estimate's $\\psi$ is "
          f"the von Mises *mode*; as $\\kappa \\to 0$ the conditional becomes uniform and "
          f"its mode carries essentially no information. Median $\\kappa$ = "
          f"{np.median(KAPPA):.3f}, and {100 * np.mean(KAPPA < 1.0):.0f}% of splittings "
          f"sit below 1. $\\psi$ *should* be uniform by azimuthal symmetry, so small "
          f"$\\kappa$ is the head getting the physics right — and the MAP/MBR $\\psi$ "
          f"marginal can be sharply structured even when the posterior is perfectly flat "
          f"and correct.")
        P(figs.block(R["fig_kappa"]))
    P(figs.block(R["fig_by_t"]))
    P(figs.block(R["fig_ladder"]))
    P(figs.block(R["fig_ktcut"]))
    P(figs.block(R["fig_leading"]))
    P("Jets contributing a leading emission (the rest have no splitting at all): "
      + ", ".join(f"`{x}` {R['n_lead'][x]:,}" for x in SERIES)
      + f" of {R['N']:,}.")

    # --- distances ------------------------------------------------------------
    P("## 6. Distribution distances")
    P("Three distances, deliberately redundant, so no single statistic drives the "
      "verdict: **$W_1$** (binning-free, carries the observable's units, sensitive to "
      "shifts, weight-aware), **KS** (binning-free, sensitive to shape disagreement "
      "anywhere, computed off the *weighted* ECDFs since `scipy.stats.ks_2samp` would "
      "silently drop the jet weights), and **$\\chi^2/\\mathrm{ndf}$** (bin-wise, Sumw2 "
      "errors on both histograms, the one that depends on the binning).")
    P("**$\\psi$ is a circle, not a line**, so its rows (marked `(circ)`) use the "
      "rotation-invariant twins: circular $W_1$, whose transport plan may wrap, and "
      "**Kuiper's** $V$ in place of KS. The linear versions charge a full $2\\pi$ of "
      "transport for mass either side of the branch cut, so two identical azimuthal "
      "distributions can score as maximally far apart. Kuiper's $V$ lives on $[0,2]$ "
      "rather than $[0,1]$, which is harmless because $\\psi$ rows are only compared "
      "against $\\psi$ rows.")
    P(f"**Not every row can be won.** An improvement ratio divides by $d(x,y)$, and when "
      f"that baseline is itself at the level of statistical noise the ratio is not a hard "
      f"test — it is meaningless, and every estimator scores above 1 no matter how good "
      f"it is. So each row carries a **noise floor**: the same three distances measured "
      f"between two independent bootstrap resamples of *truth against itself* at that "
      f"row's own sample size ({a.n_boot} resamples, {a.floor_pct:g}th percentile). The "
      f"bootstrap resamples **jets, not splittings** — splittings within a jet share a "
      f"weight and correlated kinematics. A row is **scoreable** only when $d(x,y)$ "
      f"exceeds its floor; the rest are printed with their distances and their floor, "
      f"marked `[n/s]`, ratio columns blanked, and excluded from the headline.")
    P("Everything is computed on the **fiducial window** of section 3: scoring the model "
      "on truth it structurally cannot reach measures the geometry, not the model.")

    P("### Headline")
    P("Geometric mean of the improvement ratio across every **scoreable** observable "
      "(geometric, because ratios compose multiplicatively and one 10x outlier should "
      "not dominate), and how many of them each estimator actually beat plain RSD on. "
      "**Lower is better; below 1 means better than plain RSD.**")
    P(tables["headline"])
    P(f"{len(R['ROWS'])} shape observables + {len(R['RATE_ROWS'])} rate observables were "
      f"compared against truth. Scoreable: "
      + ", ".join(f"{lab} {sum(1 for _, dd in R['ROWS'] if dd['scoreable'][mm])}/"
                  f"{len(R['ROWS'])}" for mm, lab in METRICS_LABELS)
      + f", rate {sum(1 for r in R['RATE_ROWS'] if r[3])}/{len(R['RATE_ROWS'])}.")

    for mm, lab in METRICS_LABELS:
        n_ns = sum(1 for _, dd in R["ROWS"] if not dd["scoreable"][mm])
        P(f"### {lab}")
        P(f"Distance to truth; lower is better. {n_ns}/{len(R['ROWS'])} rows are "
          f"`[n/s]` — plain RSD is inside the noise floor there.")
        P(tables[mm])

    P("### Rate")
    P("All three distances above run on unit-area histograms, so they are blind to "
      "overall normalisation by construction — they ask *is this the right shape?*, "
      "never *is this the right number of emissions?*. That axis is scored here on its "
      "own terms: emissions per jet above a $k_t$ cut, and the relative deviation from "
      "truth, with the same lower-is-better / ratio-against-RSD convention. This is the "
      "table that can disagree with the other three, and the one to read alongside the "
      f"$k_t$-cut spectrum ({figs.ref(R['fig_ktcut'])}).")
    P(tables["rate"])

    # --- reading the results --------------------------------------------------
    P("## 7. Reading the results")
    hl = R["headline"]
    # Between MAP and MBR only: `post` is the calibrated COMPARATOR, not a competing
    # answer -- it draws one sample per jet, so it is not a point estimate at all and
    # calling it the winner would recommend an estimator nobody can deploy. NaN sorts
    # unpredictably, so a series with nothing scoreable is pushed back explicitly rather
    # than being allowed to win by accident.
    best = min(("map", "mbr"), key=lambda x: (hl[x]["w1"]["gmean_ratio"]
                                              if np.isfinite(hl[x]["w1"]["gmean_ratio"])
                                              else np.inf))
    P("**The ratio is the verdict, the panels are the diagnosis.** A geometric-mean ratio "
      "below 1 with most observables beaten means the model genuinely moves the "
      "hadron-level distribution toward parton level. Above 1 means plain RSD was already "
      "better and the model is adding noise.")
    verdict = []
    for x in MODELS:
        r = hl[x]["w1"]["gmean_ratio"]
        verdict.append(f"`{x}` {fmt(r, 3)} ({hl[x]['w1']['n_better']}/"
                       f"{hl[x]['w1']['n_scored']} rows beaten)")
    P("On $W_1$ over the scoreable rows: " + ", ".join(verdict)
      + f". Of the two deployable **point estimators**, `{best}` is ahead on this run "
      + f"({fmt(hl[best]['w1']['gmean_ratio'], 3)}). `post` is not a competitor but the "
      + "calibrated comparator — it is one draw per jet, so it carries the model's own "
      + "spread rather than an argmax's shrinkage. Judge the *model* by it and the "
      + "*estimators* by their gap to it.")

    if cont and len(KAPPA):
        psi = [dd for n, dd in R["ROWS"] if n.startswith("psi (pooled)")]
        if psi and psi[0]["series"]:
            dd = psi[0]["series"]
            P(f"**$\\psi$ carries into the aggregates.** At median $\\kappa$ = "
              f"{np.median(KAPPA):.3f} the density is nearly flat, so its mode carries no "
              f"information: MAP and MBR take the mode, the posterior samples. In the "
              f"pooled $\\psi$ row that costs map {fmt(dd['map']['w1_r'], 3)}x, mbr "
              f"{fmt(dd['mbr']['w1_r'], 3)}x, post {fmt(dd['post']['w1_r'], 3)}x ($W_1$ "
              f"ratio vs plain RSD)"
              + (" — and that row **is** scoreable, so it is inflating the MAP/MBR "
                 "geometric means above. Read those aggregates knowing $\\psi$ is in them."
                 if psi[0]["scoreable"]["w1"] else
                 " — and that row is not scoreable, so it stays out of the aggregates."))

    quads = [(n, dd) for n, dd in R["ROWS"] if n.startswith("lnkt [")]
    if quads:
        rows = []
        for n, dd in quads:
            # `.get`, not `[...]`: a quadrant can be EMPTY rather than merely
            # unscoreable. u = ln(1/dR) >= ln(1/R), so at R = 0.4 nothing exists below
            # u = 0.92 and a mid-range split can leave `narrow_hard` with no emissions at
            # all -- a fact about the geometry, not a missing result.
            if not dd["series"]:
                st = "EMPTY — unreachable at this $R$, not a model result"
            elif dd["scoreable"].get("w1"):
                st = "scored"
            else:
                st = "n/s — RSD already inside the noise floor"
            rows.append([n, fmt(dd["series"].get("map", {}).get("w1_r", float("nan")), 3),
                         fmt(dd["series"].get("mbr", {}).get("w1_r", float("nan")), 3),
                         fmt(dd["series"].get("post", {}).get("w1_r", float("nan")), 3),
                         st])
        P("**Where the model earns its keep.** Plain RSD is a strong baseline for hard "
          "emissions — hadron level already tracks parton level there — so the `*_hard` "
          "quadrants often fall inside the noise floor rather than scoring near 1. The "
          "pooled row hides this.")
        P(md_table(["quadrant ($\\ln k_t$)", "map/rsd", "mbr/rsd", "post/rsd",
                    "scoreability"], rows))

    P("**What is not a model failure.**")
    P("0. **An `[n/s]` row.** Plain RSD is already within truth-vs-truth noise there, so "
      "hadronisation left no shape difference to recover. Expect $\\ln z$ and $\\psi$ "
      "rows to fall out this way — the first is squeezed between two hard walls "
      "($\\ln(0.5/z_\\mathrm{cut})$ nats apart), the second is uniform by symmetry at "
      "both levels. Raising `--jets` lowers the floors and brings marginal rows back into "
      "scope; it will never rescue a row where the two levels genuinely agree.\n"
      "1. **Mode shrinkage.** MAP and MBR are per-jet argmaxes; their populations are "
      "narrower than truth by construction. The `post` series is the calibrated "
      "comparator — judge the model by it and the *estimators* by their gap to it.\n"
      "2. **The $\\psi$ panel for MAP/MBR** is the von Mises mode, near-arbitrary at "
      "small $\\kappa$. A property of the estimator, not a defect.\n"
      "3. **Out-of-window truth.** The model cannot emit outside the geometry; the "
      "section-3 fraction is an irreducible floor, which is why the distances use the "
      "fiducial window.\n"
      "4. **A family with no coordinate density** (`ar_junipr_v1` only) leaves $\\ln z$ "
      "and $\\psi$ as placeholders, so those panels drop the model series rather than "
      "plotting a filler constant as a prediction.")
    P("**One thing that *is* a model failure:** a soft-drop violation fraction above the "
      "truth/RSD floor — see the verdict in section 3.")
    if not a.require_truth_splitting:
        P("**Do not compare these numbers to v1's.** v1 additionally requires "
          "`len(y) > 0`, so this population adds back exactly the jets whose answer is "
          "the empty tree, and every distance, ratio and rate moves. v1's numbers are not "
          "superseded so much as *scoped*: they describe jets that have parton-level "
          "substructure, which is a legitimate question, just not the one an analysis "
          "gets to ask.")

    # --- figure index + artifacts ---------------------------------------------
    P("## Figures")
    P(figs.index())
    P("## Artifacts")
    P(md_table(["file", "what it is"], R["artifacts"]))
    return "\n".join(L).rstrip() + "\n"


# ============================================================================
# 6. main
# ============================================================================
def main(argv=None):
    a = parse_args(argv)
    s, prod = resolve_settings(a)

    m = load_model(a, s)
    geom, model, device = m["geom"], m["model"], m["device"]
    print(f"checkpoint : {m['ckpt']}")
    print(f"model      : {m['info']['model_name']}  +  {m['cfg'].encoder.name}   "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"epoch      : {m['info']['epoch']}   best val NLL {m['info']['best_val_nll']:.4f}")
    print(f"device     : {device}   aux: {m['aux'] if m['aux'] else '(none)'}")
    if (model.length_temperature, model.length_tilt) != (1.0, 0.0):
        print(f"[decode] length head recalibrated: T = {model.length_temperature:.4f}, "
              f"tilt = {model.length_tilt:+.4f} -- the posterior series, the empty rate "
              f"and TAU all move with these.")
    if not m["cont"]:
        print("NOTE: this family has no continuous coordinate head -- point estimates sit "
              "at cell centres with ln z = 0, psi = 0 PLACEHOLDERS; those panels drop the "
              "model series.")

    d = load_sample(a, s, m)
    print(f"source     : {d['source']}  {s['root']}")
    print(f"jets       : {len(d['jets']):,} of {d['n_in']:,}   "
          f"(dropped {d['n_x0']:,} with no hadron splitting; {d['n_y0_of_x']:,} with no "
          f"PARTON splitting are "
          + ("dropped too)" if a.require_truth_splitting else "KEPT)"))
    print(f"mean mult  : hadron x {d['mean_x']:.3f}   parton y {d['mean_y']:.3f}   "
          f"P(n_y=0) = {100 * d['p_y0']:.1f}%")

    # --- cost probe -----------------------------------------------------------
    if a.probe:
        n = min(a.probe, len(d["ds"]))
        t0 = time.perf_counter()
        eval_jets(range(n), a, m, d)
        dt = (time.perf_counter() - t0) / max(n, 1)
        print(f"\n{dt * 1e3:7.1f} ms / jet   (K={a.draws}, MBR backend={a.mbr_backend!r}, "
              f"candidates={a.mbr_candidates or 'all'})")
        print(f"-> --jets {a.jets} would take about {dt * a.jets / 60:.1f} min")
        if a.mbr_backend == "pot":
            print("   too slow? --mbr-backend energyflow is the SAME metric ~6x faster "
                  "(identical winner on 99.3% of jets).")
        return 0

    # --- the evaluation pass --------------------------------------------------
    N = min(a.jets, len(d["ds"]))
    t0 = time.perf_counter()
    raw, w_jet, kappas, mbr_risk, n_map_unfloored = eval_jets(range(N), a, m, d)
    runtime_min = (time.perf_counter() - t0) / 60
    print(f"\nevaluated {N} jets in {runtime_min:.1f} min (K={a.draws} draws each)")

    POOL = {x: pack(raw[x], w_jet) for x in SERIES}
    NSPL = {x: np.array([len(v) for v in raw[x]]) for x in SERIES}
    KAPPA = np.concatenate(kappas) if any(len(k) for k in kappas) else np.zeros(0)
    cont = m["cont"]
    STYLE = {
        "truth": (C_TRUTH, "-", r"truth $y$ (parton)"),
        "rsd":   (C_RSD_E, "-", r"plain RSD $x$ (hadron)"),
        "map":   (C_MAP,   "-", r"MAP $\hat y$"),
        "mbr":   (C_MBR,   "-", r"MBR $\hat y$"),
        "post":  (C_POST, "--", "posterior (1 draw/jet)" if cont
                                else "posterior (cells sampled; no coordinate density)"),
    }
    print(f"\n{'series':<8}{'splittings':>12}{'mean mult':>12}{'mean ln kt':>12}")
    for x in SERIES:
        v = POOL[x]["v"]
        mk = float(np.average(v[:, 1], weights=POOL[x]["w"])) if len(v) else float("nan")
        print(f"{x:<8}{len(v):>12,}{NSPL[x].mean():>12.2f}{mk:>12.3f}")

    R = dict(m=m, d=d, geom=geom, raw=raw, w_jet=w_jet, POOL=POOL, NSPL=NSPL,
             KAPPA=KAPPA, STYLE=STYLE, cont=cont, N=N, runtime_min=runtime_min,
             KEYS=["lnInvDelta", "lnkt", "lnz", "psi"])

    # --- section 2: the empty tree -------------------------------------------
    IS0 = {x: (NSPL[x] == 0) for x in SERIES}
    t_empty = IS0["truth"]
    wsum = w_jet.sum()
    P_TRUE0 = float(w_jet[t_empty].sum() / wsum)
    # q(N=0|x) -- the belief itself, before any decode touches it.
    q0 = np.array([float(model.length_pmf(d["ds"][i]["xf"].unsqueeze(0).to(device),
                                          torch.tensor([d["ds"][i]["nx"]], device=device))[0])
                   for i in range(N)])
    TAU = (empty_threshold_for_rate([np.array([v, 1.0 - v]) for v in q0], P_TRUE0)
           if s["empty_threshold"] is None else float(s["empty_threshold"]))
    GATED = (q0 >= TAU) if TAU > 0 else np.zeros(N, dtype=bool)

    def p0(mask):
        return float(w_jet[mask].sum() / wsum)

    def recall0(mask):
        """P(n_hat=0) on the truth-empty jets. None with no truth-empty jets: dividing
        by an empty class is not an overprediction, it is no question."""
        return (float(w_jet[mask & t_empty].sum() / w_jet[t_empty].sum())
                if t_empty.any() else None)

    auc_q0 = None
    if t_empty.any() and (~t_empty).any():
        r = np.argsort(np.argsort(q0)) + 1          # AUC by rank statistic, no sklearn
        auc_q0 = float((r[t_empty].mean() - (t_empty.sum() + 1) / 2) / (~t_empty).sum())

    has_nofloor = (not a.no_map_allow_empty) and bool((n_map_unfloored >= 0).all())
    EMPTY_TREE = {
        "p_truth": P_TRUE0,
        "n_truth_empty": int(t_empty.sum()),
        "min_emissions": int(m["decode"]["min_emissions"]),
        "map_allow_empty": bool(not a.no_map_allow_empty),
        # The MBR row, alone among these, moves with the risk function, so the backend
        # has to travel with the numbers or they cannot be compared across runs.
        "mbr_backend": a.mbr_backend,
        "p_pred": {x: p0(IS0[x]) for x in SERIES if x != "truth"},
        "recall": {x: recall0(IS0[x]) for x in SERIES if x != "truth"},
        "map_no_floor": ({"p_pred": p0(n_map_unfloored == 0),
                          "recall": recall0(n_map_unfloored == 0)}
                         if has_nofloor else None),
        "gate": ({"tau": float(TAU),
                  "rate_matched_on_this_sample": s["empty_threshold"] is None,
                  "p_pred": p0(GATED),
                  "recall": recall0(GATED),
                  "precision": (float(w_jet[GATED & t_empty].sum() / w_jet[GATED].sum())
                                if w_jet[GATED].sum() > 0 else None)}
                 if TAU > 0 else None),
        "q0": {"mean": float(q0.mean()),
               "mean_truth_empty": float(q0[t_empty].mean()) if t_empty.any() else None,
               "mean_truth_nonempty": (float(q0[~t_empty].mean())
                                       if (~t_empty).any() else None),
               "auc": auc_q0,
               "underconfidence": (float(P_TRUE0 / max(q0.mean(), 1e-9))
                                   if P_TRUE0 > 0 else None)},
    }
    R["EMPTY_TREE"] = EMPTY_TREE

    et_rows = [["truth", pct(P_TRUE0), "--", "--"]]

    def et_row(label, p_, rec):
        et_rows.append([label, pct(p_),
                        f"{p_ / P_TRUE0:.2f}x" if P_TRUE0 > 0 else "--",
                        "--" if rec is None else pct(rec, 1)])

    for x in SERIES:
        if x != "truth":
            et_row(f"`{x}`", EMPTY_TREE["p_pred"][x], EMPTY_TREE["recall"][x])
    if EMPTY_TREE["map_no_floor"]:
        et_row("`map` (no floor)", EMPTY_TREE["map_no_floor"]["p_pred"],
               EMPTY_TREE["map_no_floor"]["recall"])
    if EMPTY_TREE["gate"]:
        et_row(f"gate $\\tau$={TAU:.3f}", EMPTY_TREE["gate"]["p_pred"],
               EMPTY_TREE["gate"]["recall"])
    tables = {"empty": md_table(
        ["series", "$P(n=0)$", "vs truth", "$P(\\hat n=0)$ on truth-empty jets"], et_rows)}
    print(f"\nempty tree : truth {pct(P_TRUE0, 1)}   "
          f"post {pct(EMPTY_TREE['p_pred']['post'], 1)}   "
          f"gate tau={TAU:.4f} -> {pct(EMPTY_TREE['gate']['p_pred'], 1) if EMPTY_TREE['gate'] else 'off'}"
          + (f"   q0 AUC {auc_q0:.3f}" if auc_q0 is not None else ""))

    # --- section 3: support ---------------------------------------------------
    U_LO, U_HI = geom.ln_invdelta_range
    V_LO, V_HI = geom.ln_kt_range

    def support_row(v, w):
        if not len(v):
            return dict(out_of_window=np.nan, sd_violation=np.nan,
                        ktfloor_violation=np.nan)
        tot = w.sum()
        oow = ((v[:, 0] < U_LO) | (v[:, 0] > U_HI) | (v[:, 1] < V_LO) | (v[:, 1] > V_HI))
        kt = v[:, 1] < V_LO
        if d["sd_known"]:
            sd = v[:, 2] <= (math.log(d["z_cut"]) - d["beta"] * v[:, 0])
            sd_f = float(w[sd].sum() / tot)
        else:
            sd_f = float("nan")
        return dict(out_of_window=float(w[oow].sum() / tot), sd_violation=sd_f,
                    ktfloor_violation=float(w[kt].sum() / tot))

    SUPPORT = {x: support_row(POOL[x]["v"], POOL[x]["w"]) for x in SERIES}
    R["SUPPORT"] = SUPPORT
    tables["support"] = md_table(
        ["series", "out of window", "soft-drop violation", "$k_t$-floor violation"],
        [[f"`{x}`", pct(SUPPORT[x]["out_of_window"], 3),
          pct(SUPPORT[x]["sd_violation"], 3), pct(SUPPORT[x]["ktfloor_violation"], 3)]
         for x in SERIES])

    # --- figures --------------------------------------------------------------
    out_dir = Path(a.out) if a.out else (m["ckpt"].resolve().parent / "lund_closure_report")
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    R["out_dir"] = out_dir
    figs = Figures(out_dir, a.png)
    print(f"\nwriting figures to {out_dir}/figures")
    R["fig_plane"], R["fig_plane_ratio"] = figure_lund_planes(figs, R, a)
    R["fig_marginals"], R["fig_kappa"] = figure_marginals(figs, R)
    R["fig_by_t"] = figure_marginals_by_t(figs, R, T_SLICES)
    R["fig_ladder"] = figure_ladder(figs, R, T_LADDER)
    R["fig_ktcut"] = figure_kt_cut(figs, R)
    R["fig_leading"], LEAD = figure_leading(figs, R)
    R["n_lead"] = {x: len(LEAD[x][0]) for x in SERIES}

    # --- section 6: distances -------------------------------------------------
    RNG = np.random.default_rng(a.seed + 1)

    def fid_window(key):
        """Fiducial edges: the plotting edges restricted to the model's own support."""
        e = edges(key)
        if key == "lnInvDelta":
            lo, hi = U_LO, U_HI
        elif key == "lnkt":
            lo, hi = V_LO, V_HI
        elif key == "lnz":
            lo, hi = d["lnz_floor"], 0.0
        else:
            lo, hi = e[0], e[-1]
        sub = e[(e >= lo - 1e-9) & (e <= hi + 1e-9)]
        return sub if sub.size >= 3 else np.linspace(lo, hi, 26)

    def obs_splitting(key, t_mask=None):
        """A splitting-level observable, clipped to its fiducial window."""
        e = fid_window(key)
        lo, hi = e[0], e[-1]
        vals, wts, jids = {}, {}, {}
        for x in SERIES:
            p = POOL[x]
            mm = np.ones(len(p["v"]), bool) if t_mask is None else t_mask(p)
            v = p["v"][mm, COL[key]]
            keep = (v >= lo) & (v <= hi)
            vals[x], wts[x] = v[keep], p["w"][mm][keep]
            jids[x] = p["jet"][mm][keep]      # bootstrap blocks
        return vals, wts, jids, e

    def obs_jet(getter, e):
        """A per-jet observable. One entry per jet, so every bootstrap block is a
        singleton and the block bootstrap reduces to the ordinary one."""
        vals, wts, jids = {}, {}, {}
        for x in SERIES:
            v, w = getter(x)
            vals[x], wts[x], jids[x] = v, w, np.arange(len(v))
        return vals, wts, jids, e

    ROWS = []

    def add(name, vals, wts, jids, e, key=None):
        ROWS.append((name, compare(vals, wts, jids, e, RNG, a.n_boot, a.floor_pct,
                                   circular=CIRC.get(key))))

    for key in (R["KEYS"] if cont else ["lnInvDelta", "lnkt"]):
        mark = " (circ)" if key in CIRC else ""
        add(f"{key} (pooled){mark}", *obs_splitting(key), key=key)
        for t in T_SLICES:
            add(f"{key} t={t}{mark}",
                *obs_splitting(key, (lambda p, t=t: p["t"] == t)), key=key)
        add(f"{key} t>={T_SLICES[-1] + 1}{mark}",
            *obs_splitting(key, lambda p: p["t"] > T_SLICES[-1]), key=key)

    add("multiplicity", *obs_jet(lambda x: (NSPL[x].astype(float), w_jet), edges("mult")))
    for key in (["lnkt", "lnInvDelta"] + (["lnz"] if cont else [])):
        add(f"leading {key}",
            *obs_jet(lambda x, k=key: (LEAD[x][0][:, COL[k]], LEAD[x][1]), fid_window(key)))

    # Lund quadrants: plain RSD is already close to truth for HARD emissions, so a
    # pooled number hides where the model actually earns its keep.
    def quad_mask(label):
        def f(p):
            cells = np.array([geom.to_cell(u, v) for u, v in p["v"][:, :2]], dtype=int)
            return np.array([cell_region(int(c), geom) == label for c in cells])
        return f

    for lab in REGION_LABELS:
        add(f"lnkt [{lab}]", *obs_splitting("lnkt", quad_mask(lab)))

    # --- rates, scored separately --------------------------------------------
    def rate_per_jet(x, c, mult=None):
        """Emissions per jet above a kt cut. `mult` is a per-jet repeat count, so a
        bootstrap draw can weight a jet by how many times it was drawn -- a
        set-membership test would silently deduplicate and bias the rate down."""
        p = POOL[x]
        mm = p["v"][:, 1] > c
        if mult is None:
            num, den = p["w"][mm].sum(), w_jet.sum()
        else:
            num = (p["w"][mm] * mult[p["jet"][mm]]).sum()
            den = (w_jet * mult).sum()
        return float(num / den) if den > 0 else float("nan")

    RATE_ROWS = []
    for c in (0.0, 1.0, 2.0, 3.0):
        per_jet = {x: rate_per_jet(x, c) for x in SERIES}
        ref = per_jet["truth"]
        boot = [rate_per_jet("truth", c,
                             np.bincount(RNG.integers(0, N, N), minlength=N).astype(float))
                for _ in range(a.n_boot)]
        floor = (float(np.nanpercentile(np.abs(np.array(boot) / ref - 1.0), a.floor_pct))
                 if ref > 0 else float("nan"))
        base = abs(per_jet["rsd"] / ref - 1.0) if ref > 0 else float("nan")
        ok = bool(np.isfinite(floor) and np.isfinite(base) and base > floor)
        dd = {}
        for x in ("rsd",) + MODELS:
            dev = abs(per_jet[x] / ref - 1.0) if ref > 0 else float("nan")
            dd[x] = {"n_per_jet": per_jet[x], "rel_dev": dev,
                     "rel_dev_r": (dev / base) if (np.isfinite(base) and base > 0)
                     else float("nan")}
        RATE_ROWS.append((f"N(ln kt > {c:g})", ref, floor, ok, dd))
    R["ROWS"], R["RATE_ROWS"] = ROWS, RATE_ROWS

    def scored(x, mm):
        """Ratios from SCOREABLE rows only -- the rest divide by noise."""
        return [dd["series"][x][mm + "_r"] for _, dd in ROWS
                if dd["scoreable"][mm] and x in dd["series"]]

    head_rows = []
    for x in MODELS:
        row = [f"`{x}`"]
        for mm, _ in METRICS_LABELS:
            ratios = scored(x, mm)
            won = sum(1 for v in ratios if np.isfinite(v) and v < 1.0)
            tot = sum(1 for v in ratios if np.isfinite(v))
            row += [fmt(gmean(ratios), 3), f"{won}/{tot}"]
        rate_r = [ddd[x]["rel_dev_r"] for _, _, _, ok, ddd in RATE_ROWS if ok and x in ddd]
        row += [fmt(gmean(rate_r), 3),
                f"{sum(1 for v in rate_r if np.isfinite(v) and v < 1.0)}/"
                f"{sum(1 for v in rate_r if np.isfinite(v))}"]
        head_rows.append(row)
    tables["headline"] = md_table(
        ["estimator", "W1 ratio", "W1 wins", "KS ratio", "KS wins",
         "chi2 ratio", "chi2 wins", "rate ratio", "rate wins"], head_rows)

    for mm, _ in METRICS_LABELS:
        rows = []
        for name, dd in ROWS:
            ser, ok = dd["series"], dd["scoreable"][mm]
            cells = [name if ok else name + "  [n/s]"]
            cells += [fmt(ser[x][mm]) if x in ser else "--" for x in ("rsd",) + MODELS]
            cells += [fmt(dd["floor"][mm])]
            cells += [(fmt(ser[x][mm + "_r"], 3) if x in ser else "--") if ok else "n/s"
                      for x in MODELS]
            rows.append(cells)
        tables[mm] = md_table(
            ["observable", "rsd", "map", "mbr", "post", "noise floor",
             "map/rsd", "mbr/rsd", "post/rsd"], rows)

    tables["rate"] = md_table(
        ["observable", "truth", "rsd", "map", "mbr", "post", "noise floor",
         "map/rsd", "mbr/rsd", "post/rsd"],
        [[name if ok else name + "  [n/s]", fmt(ref, 3)]
         + [fmt(dd[x]["n_per_jet"], 3) for x in ("rsd",) + MODELS]
         + [fmt(floor, 3)]
         + [(fmt(dd[x]["rel_dev_r"], 3) if ok else "n/s") for x in MODELS]
         for name, ref, floor, ok, dd in RATE_ROWS])

    R["headline"] = {
        x: {mm: {"gmean_ratio": gmean(scored(x, mm)),
                 "n_better": int(sum(1 for v in scored(x, mm)
                                     if np.isfinite(v) and v < 1.0)),
                 "n_scored": int(sum(1 for v in scored(x, mm) if np.isfinite(v)))}
            for mm, _ in METRICS_LABELS}
        for x in MODELS
    }
    print("\nHEADLINE (geometric-mean improvement ratio over scoreable rows)\n")
    print(tables["headline"])

    # --- artifacts ------------------------------------------------------------
    R["artifacts"] = [["[`report.md`](report.md)", "this report"]]
    if not a.no_artifacts:
        metrics = {
            "model": m["info"]["model_name"],
            "encoder": str(m["cfg"].encoder.name),
            "checkpoint": str(m["ckpt"]),
            "aux_features": list(m["aux"]),
            "continuous_coords": bool(cont),
            "selection": {"require_truth_splitting": bool(a.require_truth_splitting),
                          "population": ("len(x)>0 and len(y)>0 (v1, truth-selected)"
                                         if a.require_truth_splitting
                                         else "len(x)>0 (deployable)"),
                          "n_in": int(d["n_in"]), "n_kept": int(len(d["jets"])),
                          "p_truth_empty": float(np.mean(NSPL["truth"] == 0)),
                          "map_allow_empty": bool(not a.no_map_allow_empty)},
            "empty_tree": EMPTY_TREE,
            "data": {
                "source": d["source"], "path": str(s["root"]),
                "generator": str(d["jets"][0].get("generator", "n/a")),
                "z_cut": d["z_cut"], "beta": d["beta"],
                "kt_floor": float(d["jets"][0].get("kt_floor", float("nan"))),
                # The OFF-SPINE floor: two files agreeing on kt_floor can still be
                # different aux samples, so recording only kt_floor makes an
                # asymmetric-file run indistinguishable from a symmetric one.
                "kt_floor_sec": float(d["jets"][0].get("kt_floor_sec", float("nan"))),
                "pt_var": a.pt_var, "pt_min": a.pt_min, "pt_max": a.pt_max,
                "n_eval_jets": int(N), "sum_w": float(w_jet.sum()),
                "eff_n": float(w_jet.sum() ** 2 / (w_jet ** 2).sum()),
            },
            "decode": {**m["decode"], "point_estimator": "map+mbr",
                       "mbr_backend": a.mbr_backend,
                       "mbr_n_candidates": a.mbr_candidates,
                       "length_floor_quantile": a.length_floor_quantile},
            "n_draws": int(a.draws),
            "fiducial_window": {"ln_invdelta": list(geom.ln_invdelta_range),
                                "ln_kt": list(geom.ln_kt_range),
                                "ln_z_floor": d["lnz_floor"]},
            "support": SUPPORT,
            "mean_multiplicity": {x: float(NSPL[x].mean()) for x in SERIES},
            "mbr_risk_mean": (float(np.nanmean(mbr_risk))
                              if np.isfinite(mbr_risk).any() else None),
            "scoreability": {"n_boot": int(a.n_boot), "floor_percentile": float(a.floor_pct),
                             "n_rows": len(ROWS),
                             "n_scoreable": {mm: int(sum(1 for _, dd in ROWS
                                                         if dd["scoreable"][mm]))
                                             for mm, _ in METRICS_LABELS}},
            "headline": R["headline"],
            "observables": {name: dd for name, dd in ROWS},
            "rates": {name: {"truth_per_jet": ref, "noise_floor": floor,
                             "scoreable": ok, **dd}
                      for name, ref, floor, ok, dd in RATE_ROWS},
            "report": {"figures": [f"figures/{it['pdf'].name}" for it in figs.items],
                       "source": "scripts/lund_closure_report.py",
                       "prod_metrics": s["prod_metrics_path"]},
        }
        p = save_metrics(metrics, out_dir / "dist_closure_metrics.json")
        print(f"wrote {p}")
        R["artifacts"].append(
            ["[`dist_closure_metrics.json`](dist_closure_metrics.json)",
             "the same schema the notebook writes — diff the `headline` block against a "
             "notebook run on the same checkpoint, file and seed to check the two have "
             "not drifted"])
    R["artifacts"].append(
        ["[`figures/`](figures/)",
         f"{len(figs.items)} figures as PDF"
         + (" (and a PNG of each, for viewers that cannot inline a PDF)" if a.png else "")
         + " — indexed above"])

    cmdline = "PYTHONPATH=src python " + " ".join(sys.argv)
    report = build_report(a, s, R, figs, tables, cmdline)
    rp = out_dir / "report.md"
    rp.write_text(report)
    print(f"wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
