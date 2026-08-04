"""Build notebooks/inference_demo_cluster.ipynb.

    python scripts/make_inference_demo_cluster_nb.py

The set-valued companion to `notebooks/inference_demo.ipynb`, implementing §10.5 of
docs/PLAN_PosteriorClusters.md: a single-jet panel showing the posterior pool projected by
classical MDS on `D` (display only -- the clustering never sees the embedding), exemplars
marked, masses annotated, truth overlaid.

Deliberately SHORT and standalone: it is the "what does a set-valued prediction look like"
demo, not the measurement pass. The population gates live in
`notebooks/per_jets_estimation_cluster.ipynb` and in `eval/clusters.py`.

Regenerating drops the executed outputs, so follow it with

    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/inference_demo_cluster.ipynb
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
# Set-valued inference — what the posterior actually says about one jet

The companion to [`inference_demo.ipynb`](inference_demo.ipynb) for
docs/PLAN_PosteriorClusters.md. Standalone: it loads the newest checkpoint, falls back to
the synthetic simulator when no ROOT file is given, and needs nothing that notebook does
not.

`inference_demo.ipynb` §5 shows the MAP and the posterior spread for one jet. This shows
what neither of those can: **the discrete alternative explanations the posterior is split
between**, each with its own probability.

The sample space is transdimensional,
$\mathcal{Y}=\bigsqcup_{N}\mathcal{C}^{N}$, and the strata are metrically separated by the
perturbative-Lund EMD's imbalance term. A jet whose posterior splits between "one hard
emission" and "two softer emissions consistent with the same $x$" is the hadronization
ambiguity expressed as **alternative shower histories**, and both a mode and a mean-distance
medoid smear exactly that: the medoid of a two-lobed posterior can land in the sparse valley
between the lobes, minimising mean distance while representing neither explanation.

**Nothing here enlarges the hypothesis space.** Every member of the set is a genuine
posterior draw carrying its own sampled coordinates. That closure property is what makes
the masses mean something — a consensus or lattice construction would produce trees the
model never generated, and a probability attached to one of those is a probability of
nothing.

Three panels and a printed table:

- **(a)** the posterior cloud on the Lund plane, coloured by cluster, exemplars marked.
- **(b)** the pool in **its own geometry** — classical MDS on `D`, *display only*: the
  clustering works on `D` directly because $\mathcal{Y}$ has no vector-space structure to
  embed into. The stress printed beside it says how lossy the picture is.
- **(c)** the length belief, with each cluster's $N$ marked — so a split *between* $N$
  strata is visibly different from a split *within* one.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Parameters
""")

code(r'''
# --- inputs -----------------------------------------------------------------
CKPT_PATH   = None      # path to a best.ckpt; None -> newest runs/**/best.ckpt
ROOT_PATH   = None      # path to a test jets.root; None -> synthetic test data
NTUPLE_NAME = "Jets"

# --- test sample ------------------------------------------------------------
N_TEST_JETS = 600
SEED        = 1234
DEVICE      = "cpu"

# --- inference knobs --------------------------------------------------------
# K is what the mass vector's RESOLUTION is: at CLUSTER_MIN_MASS = 0.05 a reportable
# cluster is 5% of K draws, and the Monte-Carlo error on a mass of 0.6 is
# sqrt(0.6*0.4/K). Density estimation needs resolution in the sample space itself, and
# the sample size to resolve MODES scales far worse than the one to estimate a mean.
N_POSTERIOR       = 400
N_SUMMARY         = 60    # jets in the small aggregate readout of section 5
MBR_BACKEND       = "pot"  # "pot" | "energyflow".  NEVER "surrogate" -- see the assert below
# FORCED to 0: with a candidate cap D is |C| x K and there is no square matrix to cluster.
MBR_N_CANDIDATES  = 0
CLUSTER_METHOD    = "hdbscan"   # hdbscan | dbscan (need scikit-learn) | pam (pure NumPy)
CLUSTER_MIN_MASS  = 0.05
SHOWCASE_JET      = None   # None -> auto-pick the most AMBIGUOUS jet

assert MBR_N_CANDIDATES == 0, "the cluster layer needs a square K x K distance matrix"
assert MBR_BACKEND != "surrogate", (
    "`_lund_image` normalises, so the surrogate is exactly blind to total kt and "
    "multiplicity -- the quantity that separates the N strata the clusters are made of. "
    "It is a screening pass for a verdict, never a quoted mass vector."
)
''')

# ---------------------------------------------------------------------------
md(r"""
## 1. Imports, helpers & house style
""")

code(r'''
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from h2p_rsd_junipr.data.dataset import MatchedLundDataset
from h2p_rsd_junipr.data.rntuple import load_rntuple
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.eval.closure import lund_tree_str
from h2p_rsd_junipr.features import node_raw
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.clusters import assert_cluster_metric_ok
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.train.trainer import seed_everything, select_device

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "figure.titlesize": 13,
})

C_TRUTH = "#2ca02c"    # truth y                (green)
C_MBR   = "#d62728"    # the linear medoid      (red)
C_RSD   = "#7f7f7f"    # plain-RSD x baseline   (grey)
# Cluster membership INSIDE one jet. Categorical, not sequential: cluster ids are labels
# and a sequential ramp would suggest an ordering the partition does not carry.
C_CLUSTER = ["#9a4fc4", "#1f77b4", "#ff7f0e", "#17becf", "#bcbd22", "#e377c2"]


def repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists():
            return cand
    return p


def find_latest_checkpoint(runs_dir: Path) -> Path | None:
    cks = sorted(runs_dir.rglob("best.ckpt"), key=lambda q: q.stat().st_mtime)
    if not cks:
        cks = sorted(runs_dir.rglob("last.ckpt"), key=lambda q: q.stat().st_mtime)
    return cks[-1] if cks else None


def classical_mds(D, dim=2):
    """Torgerson classical MDS on a precomputed distance matrix -- DISPLAY ONLY.

    Double-centre -0.5 D^2 and take the leading eigenvectors. The clustering NEVER sees
    this embedding: it works on `D` directly, precisely because the tree space has no
    vector-space structure to embed into. Any 2-D picture of it is a lossy projection, and
    the returned stress is what says how lossy.
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


REPO = repo_root()
print("repo root:", REPO)
''')

# ---------------------------------------------------------------------------
md(r"""
## 2. The model, and gate G4

Rebuilt from the checkpoint's own config snapshot, then `assert_cluster_metric_ok` — which
**raises** (never warns) unless the metric settings make `D` a metric:

1. **$\beta = 1$.** At $\beta\neq1$ the perturbative-Lund EMD violates the triangle
   inequality — measured, 300 of 64 000 triples at $\beta=2$ — and HDBSCAN's
   mutual-reachability construction assumes a metric.
2. **$R \ge R_\max/2$** for the active `mbr_coords` (Komiske, Metodiev & Thaler,
   *Phys. Rev. Lett.* **123** (2019) 041801). Computed from *this* geometry, so a
   non-default `geometry` block cannot silently break it.
3. **`mbr_n_candidates == 0`**, or `D` is rectangular and there is nothing to cluster.
""")

code(r'''
ckpt = Path(CKPT_PATH) if CKPT_PATH else find_latest_checkpoint(REPO / "runs")
assert ckpt is not None and ckpt.exists(), (
    "No checkpoint found. Train one with `h2p-rsd-junipr train ...` or set CKPT_PATH."
)
device = select_device() if DEVICE == "auto" else torch.device(DEVICE)
seed_everything(SEED, deterministic=True)

info  = load_for_inference(str(ckpt), map_location=device)
cfg   = OmegaConf.create(info["config"])
geom  = Geometry.from_config(cfg.geometry)
model = build_model(cfg, geom).to(device)
model.load_state_dict(info["model_state"])
model.eval()

from h2p_rsd_junipr.config import decode_params

DEC = {**decode_params(cfg),
       "point_estimator": "mbr",
       "mbr_backend": MBR_BACKEND,
       "mbr_n_candidates": MBR_N_CANDIDATES,
       "cluster_posterior": True,
       "cluster_method": CLUSTER_METHOD,
       "cluster_min_mass": CLUSTER_MIN_MASS}
assert_cluster_metric_ok(DEC, geom)          # GATE G4 -- raises, never warns

print(f"checkpoint : {ckpt.relative_to(REPO)}")
print(f"model      : {info['model_name']}   encoder={cfg.encoder.name}")
print(f"geometry   : n_bins={geom.n_bins}  n_cells={geom.n_cells}")
print(f"metric     : backend={MBR_BACKEND!r}  beta={DEC['mbr_beta']:g}  R={DEC['mbr_R']:g}"
      f"  coords={DEC['mbr_coords']!r}   -- gate G4 PASSED")
print(f"clusters   : method={CLUSTER_METHOD!r}  min_mass={CLUSTER_MIN_MASS:g}  "
      f"K={N_POSTERIOR}")
print(f"             a reportable cluster is >= "
      f"{max(1, math.ceil(CLUSTER_MIN_MASS * N_POSTERIOR))} of {N_POSTERIOR} draws; "
      f"MC error on a mass of 0.6 is {math.sqrt(0.6 * 0.4 / N_POSTERIOR):.3f}")
''')

# ---------------------------------------------------------------------------
md(r"""
## 3. The test data — ROOT file **or** synthetic

`len(x) > 0` only, exactly as `inference_demo.ipynb`. Truth-empty jets are **kept**: the
$N=0$ draws sit at mutual distance exactly 0 (`inference.mbr._empty_value`), so the empty
stratum is a zero-diameter clique any density method finds by construction, and its cluster
mass *is* $q(0\mid x)$. Dropping those jets would hide the one part of this posterior that is
already known to be well calibrated.
""")

code(r'''
jets, source_desc = None, None
if ROOT_PATH:
    rp = Path(ROOT_PATH)
    rp = rp if rp.is_absolute() else (REPO / rp)
    jets = load_rntuple(str(rp), NTUPLE_NAME)
    if jets:
        source_desc = f"ROOT RNTuple  {rp.name}:{NTUPLE_NAME}"
if jets is None:
    jets = synthetic_matched_dataset(N_TEST_JETS, seed=SEED)
    source_desc = f"synthetic matched simulator  (n={N_TEST_JETS}, seed={SEED})"

n_raw = len(jets)
jets = [j for j in jets if len(np.asarray(j["x"][0])) >= 1]
AUX = tuple(model.aux_feature_names)
try:
    ds = MatchedLundDataset(jets, geom, aux_features=AUX)
except Exception as exc:
    raise RuntimeError(
        f"this checkpoint was trained with aux inputs {AUX}, which "
        f"{ROOT_PATH or 'the synthetic generator'} cannot supply ({exc})."
    ) from exc

mult_y = np.array([len(j["y"][0]) for j in jets])
print(f"source     : {source_desc}")
print(f"jets       : {len(jets)} of {n_raw} kept (len(x) > 0)")
print(f"mean mult. : parton truth y = {mult_y.mean():.2f}   "
      f"P(n_y = 0) = {np.mean(mult_y == 0):.3f}")
assert len(ds) > 10, "need >10 matched jets"
''')

# ---------------------------------------------------------------------------
md(r"""
## 4. One jet, as a set

`predict_set` returns one `LundPointEstimate` per posterior cluster, mass-descending, each
a genuine draw. Beside it, `map_or_mbr` returns the linear medoid from the **same** distance
matrix — and the medoid's position among the clusters is gate **G2**: if it already lies in
the dominant one, a density-mode estimator has nothing to add.

The three scalars are printed separately and deliberately not folded into one $\pm$:

- **`top_mass`** — a *probability*: the posterior mass of the selected explanation.
- **`entropy`** — an *ambiguity* over discrete alternatives, in nats.
- **`radii[0]`** — the *width* of the selected explanation, and the only one of the three
  that is a $\pm$ at all.
""")

code(r'''
def pick_showcase(scan=40):
    """The most AMBIGUOUS jet in the first `scan` -- the case a SET has something to say
    about. A unimodal posterior has a set of one and nothing to show."""
    if SHOWCASE_JET is not None:
        return int(SHOWCASE_JET)
    best, best_h = 0, -np.inf
    for i in range(min(scan, len(ds))):
        item = ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        torch.manual_seed(SEED)
        ps = model.predict_set(xf, nx, **DEC)
        if ps is not None and len(ps) >= 2 and ps.entropy > best_h:
            best, best_h = i, ps.entropy
    return int(best)


jet_i = pick_showcase()
item = ds[jet_i]
xf = item["xf"].unsqueeze(0).to(device)
nx = torch.tensor([item["nx"]], device=device)

torch.manual_seed(SEED)
draws = model.sample_batch(xf, nx, N_POSTERIOR)
coords_by_draw = model.sample_coordinates_many(xf, nx, [list(d) for d in draws])

# ONE distance matrix. The point estimate and the set are both read off it -- which is why
# `predict_set` cannot move `map_or_mbr`'s answer: the partition never sees the risk vector.
from h2p_rsd_junipr.inference.mbr import mbr_kwargs_from_decode, posterior_distances

_mk = mbr_kwargs_from_decode(DEC)
_mk.pop("n_samples"), _mk.pop("resample_to_qn"), _mk.pop("n_candidates")
_d, clouds, cand_idx, D = posterior_distances(
    model, xf, nx, draws=draws, geom=geom, n_candidates=0, **_mk)

mbr = model.map_or_mbr(xf, nx, draws=draws, coords_by_draw=coords_by_draw, **DEC)
ps = model.predict_set(xf, nx, draws=draws, coords_by_draw=coords_by_draw, D=D, **DEC)
cs = ps.clusters

y_truth = item["yraw"].numpy()
x_raw = node_raw(*jets[jet_i]["x"])
mult = np.array([len(d) for d in draws])
win = int(np.argmin(D.mean(axis=1)))          # the linear medoid's draw index

print(f"showcase jet #{jet_i}   truth multiplicity = {int(item['ny'])}   "
      f"plain RSD = {len(x_raw)}   posterior = {mult.mean():.2f} +/- {mult.std():.2f}")
print()
print(f"THE SET  ({len(ps)} explanation(s) over {N_POSTERIOR} draws, mass-descending):")
print(f"   {'#':>2} {'mass':>7} {'radius':>8} {'N':>3} {'log q(y|x)':>12}   role")
for j, m in enumerate(ps.members):
    role = []
    if j == 0:
        role.append("the point summary -- argmax of INTEGRATED density")
    if int(cs.labels[win]) == j:
        role.append("contains the LINEAR MEDOID" + (" -- gate G2 passes for this jet"
                                                    if j == 0 else " -- G2 FAILS here"))
    print(f"   {j:>2} {ps.masses[j]:>7.3f} {ps.radii[j]:>8.3f} {m.multiplicity:>3} "
          f"{m.logprob:>12.3f}   {'; '.join(role)}")
if cs.residual_mass > 1e-9:
    print(f"   residual (noise + clusters below min_mass={CLUSTER_MIN_MASS:g}): "
          f"{cs.residual_mass:.3f}")
print()
print("THE THREE SCALARS -- three different things, never one +/-:")
print(f"   top_mass = {ps.top_mass:.3f}   a PROBABILITY (not yet calibrated: the joint "
      f"tree posterior is over-confident by v1 TARP)")
print(f"   entropy  = {ps.entropy:.3f}   an AMBIGUITY over discrete alternatives, in nats")
print(f"   radii[0] = {ps.radii[0]:.3f}   a WIDTH -- the only one of the three quotable "
      f"as a +/-")
print(f"   silhouette {cs.silhouette:.3f}, separation {cs.separation:.3f}: the split is "
      f"resolvable only when separation > max radius "
      f"({'YES' if cs.separation > float(np.max(cs.radii)) else 'NO -- read it with suspicion'})")
print()
_g2 = (" -- the DOMINANT one, so a density-mode estimator adds nothing here"
       if cs.labels[win] == 0 else
       " -- NOT the dominant one: this is the medoid-in-the-valley case")
print(f"gate G2: the linear medoid lies in cluster {int(cs.labels[win])}{_g2}")
print(f"MBR risk = {mbr.risk:.4f}   (the achieved mean EMD; NOT a likelihood, and NOT "
      f"changed by taking a set)")
''')

code(r'''
fig = plt.figure(figsize=(14.0, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.28)

# --- (a) the posterior cloud on the Lund plane, coloured by cluster ----------
ax = fig.add_subplot(gs[0, 0])
cloud, clab = [], []
for k, c in enumerate(coords_by_draw):
    if c is None or not len(draws[k]):
        continue
    arr = np.asarray(c.cpu().double().numpy()).reshape(-1, 4)
    cloud.append(arr)
    clab.append(np.full(len(arr), cs.labels[k]))
cloud = np.concatenate(cloud) if cloud else np.zeros((0, 4))
clab = np.concatenate(clab) if len(clab) else np.zeros(0, dtype=int)
for c in range(-1, len(ps)):
    sel = clab == c
    if not sel.any():
        continue
    col = C_RSD if c < 0 else C_CLUSTER[c % len(C_CLUSTER)]
    ax.scatter(cloud[sel, 0], cloud[sel, 1], s=8, color=col,
               alpha=0.10 if c < 0 else 0.22, linewidths=0, zorder=2,
               label=(f"cluster {c}   m = {ps.masses[c]:.2f}" if c >= 0
                      else "unclustered"))
for j, m in enumerate(ps.members):
    v = np.array([[n.ln_invDelta, n.ln_kt] for n in m.nodes])
    if not len(v):
        continue
    col = C_CLUSTER[j % len(C_CLUSTER)]
    ax.plot(v[:, 0], v[:, 1], "-", color=col, lw=1.4, zorder=4)
    ax.scatter(v[:, 0], v[:, 1], marker="P", s=90, color=col, edgecolor="white",
               linewidth=0.8, zorder=5)
v = np.array([[n.ln_invDelta, n.ln_kt] for n in mbr.nodes])
if len(v):
    ax.scatter(v[:, 0], v[:, 1], marker="D", s=46, facecolor="none", edgecolor=C_MBR,
               linewidth=1.8, zorder=6, label="linear medoid")
if len(y_truth):
    ax.plot(y_truth[:, 0], y_truth[:, 1], "-", color=C_TRUTH, lw=1.2, alpha=0.7, zorder=6)
    ax.scatter(y_truth[:, 0], y_truth[:, 1], marker="o", s=90, facecolor="none",
               edgecolor=C_TRUTH, linewidth=2.0, zorder=7, label="truth $y$")
ax.set_xlabel(r"$\ln(1/\Delta R)$")
ax.set_ylabel(r"$\ln(k_t/\mathrm{GeV})$")
ax.set_title(f"(a) jet #{jet_i}: the posterior, by explanation")
ax.legend(fontsize=8, loc="best")

# --- (b) the pool in ITS OWN geometry, by classical MDS ----------------------
ax = fig.add_subplot(gs[0, 1])
X, stress = classical_mds(D)
for c in range(-1, len(ps)):
    sel = cs.labels == c
    if not sel.any():
        continue
    col = C_RSD if c < 0 else C_CLUSTER[c % len(C_CLUSTER)]
    ax.scatter(X[sel, 0], X[sel, 1], s=12, color=col, alpha=0.35 if c < 0 else 0.75,
               linewidths=0, zorder=2)
for j, e in enumerate(cs.exemplars):
    ax.scatter(X[e, 0], X[e, 1], marker="P", s=130, facecolor="none",
               edgecolor=C_CLUSTER[j % len(C_CLUSTER)], linewidth=2.0, zorder=5)
    ax.annotate(f"{ps.masses[j]:.2f}", (X[e, 0], X[e, 1]), textcoords="offset points",
                xytext=(8, 5), fontsize=9)
ax.scatter(X[win, 0], X[win, 1], marker="D", s=60, facecolor="none", edgecolor=C_MBR,
           linewidth=2.0, zorder=6, label="linear medoid")
ax.set_xlabel("MDS 1")
ax.set_ylabel("MDS 2")
ax.set_title(f"(b) the pool in its own geometry\n(display only; stress = {stress:.3f})")
ax.legend(fontsize=8, loc="best")

# --- (c) the length belief, with each cluster's N ----------------------------
ax = fig.add_subplot(gs[0, 2])
hi = int(max(mult.max(), int(item["ny"]))) + 1
ax.hist(mult, bins=np.arange(-0.5, hi + 1.0), color=C_RSD, alpha=0.35, edgecolor=C_RSD,
        linewidth=0.8, label=r"posterior $P(n\,|\,x)$")
for j, m in enumerate(ps.members):
    ax.axvline(m.multiplicity, color=C_CLUSTER[j % len(C_CLUSTER)], lw=2.0,
               label=f"cluster {j}: $N$ = {m.multiplicity}")
ax.axvline(int(item["ny"]), color=C_TRUTH, lw=2.4, label=f"truth $N$ = {int(item['ny'])}")
ax.set_xlabel("primary splittings $n$")
ax.set_ylabel("draws")
ax.set_title("(c) the length belief, per explanation")
ax.legend(fontsize=8, loc="best")

fig.suptitle(f"Jet #{jet_i}: {len(ps)} explanation(s), "
             f"top mass {ps.top_mass:.2f}, $H$ = {ps.entropy:.2f} nats", y=1.04)
plt.show()
''')

code(r'''
# The same jet as text. Every member is a genuine drawn tree, so `log q(y|x)` is the
# model's own density OF WHAT IS PRINTED -- not of a nearby object.
print(lund_tree_str(mbr, "the linear medoid (Frechet median of the pool)", geom,
                    ref=y_truth))
for j, m in enumerate(ps.members):
    print()
    print(lund_tree_str(
        m, f"explanation {j}   mass = {ps.masses[j]:.3f}   radius = {ps.radii[j]:.3f}",
        geom, ref=y_truth))
print()
print(lund_tree_str(y_truth, "the truth (parton-level y)", geom))
print()
print(lund_tree_str(x_raw, "plain RSD (hadron-level x)", geom, ref=y_truth))
''')

# ---------------------------------------------------------------------------
md(r"""
## 5. How often is there more than one explanation?

A small aggregate readout — enough to see whether this posterior is multimodal at all, and
to give gate **G2** a number. The full measurement pass, with G2′'s mass-matched
random-partition null, the reliability diagram and the conformal set, lives in
[`per_jets_estimation_cluster.ipynb`](per_jets_estimation_cluster.ipynb) and in
`eval/clusters.py`.

**G2 $\ge 0.90$ closes the case for a density-mode estimator**: the medoid is already inside
the dominant cluster, so a bounded loss would be solving a problem this posterior does not
have. It is a truth-free number, which is why it — unlike everything in §9 of the companion
notebook — transfers to real data.
""")

code(r'''
torch.manual_seed(SEED)
n_sum = min(N_SUMMARY, len(ds))
rows = []
for i in range(n_sum):
    it = ds[i]
    _xf = it["xf"].unsqueeze(0).to(device)
    _nx = torch.tensor([it["nx"]], device=device)
    _dr = model.sample_batch(_xf, _nx, N_POSTERIOR)
    if not _dr:
        continue
    # ONE distance matrix per jet, feeding both the medoid and the partition -- which is
    # what makes "the medoid is in cluster j" a statement about one object rather than two.
    _dd, _cl_, _ci, _D = posterior_distances(
        model, _xf, _nx, draws=_dr, geom=geom, n_candidates=0, **_mk)
    if not _ci:
        continue
    _p = model.predict_set(_xf, _nx, draws=_dr, D=_D, **DEC)
    if _p is None or not len(_p):
        continue
    _win = int(np.argmin(_D.mean(axis=1)))
    rows.append({"n_clusters": len(_p), "top_mass": _p.top_mass, "entropy": _p.entropy,
                 "radius": float(_p.radii[0]),
                 "medoid_in_top": bool(_p.clusters.labels[_win] == 0)})

n_cl = np.array([r["n_clusters"] for r in rows])
tm = np.array([r["top_mass"] for r in rows])
en = np.array([r["entropy"] for r in rows])
rd = np.array([r["radius"] for r in rows])
g2 = float(np.mean([r["medoid_in_top"] for r in rows])) if rows else float("nan")
print(f"{len(rows)} jets, K = {N_POSTERIOR} draws each")
print(f"  <n_clusters>  = {n_cl.mean():.2f}    more than one explanation on "
      f"{np.mean(n_cl >= 2):.1%} of jets")
print(f"  <top_mass>    = {tm.mean():.3f}   [16, 84]% = "
      f"[{np.percentile(tm, 16):.3f}, {np.percentile(tm, 84):.3f}]")
print(f"  <entropy>     = {en.mean():.3f} nats")
print(f"  <radii[0]>    = {rd.mean():.3f}   -- the per-jet resolution, the ONE quotable +/-")
print()
_verdict = (">= 0.90: the medoid is already central, so a density-mode estimator would be "
            "solving a\n           problem this posterior does not have"
            if g2 >= 0.90 else
            "< 0.90: the medoid leaves the dominant cluster often enough to be worth "
            "measuring\n           properly -- see per_jets_estimation_cluster.ipynb "
            "section 9")
print(f"  gate G2 (TRUTH-FREE, so it transfers to real data): the linear medoid lies in "
      f"the\n           dominant cluster on {g2:.3f} of jets   -> {_verdict}")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, v, lab, col in ((axes[0], n_cl, "explanations per jet", C_CLUSTER[0]),
                        (axes[1], tm, "top-cluster mass (a probability)", C_CLUSTER[1]),
                        (axes[2], en, "entropy $H(m)$  (nats)", C_CLUSTER[2])):
    if lab.startswith("explanations"):
        ax.hist(v, bins=np.arange(0.5, v.max() + 1.5), color=col, alpha=0.75,
                edgecolor="white")
    else:
        ax.hist(v, bins=24, color=col, alpha=0.75, edgecolor="white")
    ax.set_xlabel(lab)
    ax.set_ylabel("jets")
fig.suptitle("How ambiguous is this posterior, jet by jet?", y=1.03)
plt.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------------------
md(r"""
---

### Reading these figures

- **Panel (b) is a picture, not the computation.** The clustering runs on `D` directly —
  the tree space has no vector-space structure to embed into, which is why every method here
  is distance-matrix-only. A large MDS stress means the *picture* is a poor rendering of a
  space the algorithm sees correctly; it is not evidence against the partition.
- **The masses are not yet calibrated probabilities.** The joint tree posterior is
  over-confident by v1 TARP, and with the sample split off (`decode.cluster_split = false`)
  `top_mass` is additionally biased **high** — the same draws define the cluster and are
  then counted into it. `per_jets_estimation_cluster.ipynb` §8 measures both.
- **`entropy` is not a width.** A bimodal posterior summarised as mean $\pm$ sd points at a
  configuration neither mode supports. The width of the selected explanation is `radii[0]`,
  and it is the only one of the three scalars that belongs after a $\pm$.
- **`predict_set` does not move the point estimate.** `cluster_posterior` consumes `D` and
  never sees the risk vector, so `map_or_mbr` returns the same tree and the same `.risk`
  whether or not a set was taken. That orthogonality is what lets the cluster layer ship at
  stock MBR settings.
- **The $N = 0$ stratum is a cluster by construction.** All empty draws sit at mutual
  distance exactly 0 and at a large constant distance from every non-empty draw, so it
  appears as its own zero-radius cluster whose mass is $q(0\mid x)$ — the one quantity
  already measured as well calibrated while every point estimator mishandled it.
- **A set of one is a real answer.** If most jets come back with a single explanation, the
  posterior is effectively unimodal in this metric at this budget: the set ships as a
  diagnostic rather than a product, and the deliverable reduces to quoting `radii[0]` as a
  per-jet resolution beside the existing MBR point estimate. That is the plan's own kill
  criterion, and reaching it is a result.
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
       / "inference_demo_cluster.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
