"""Regenerate the figures embedded in docs/PROD_TEST_v0_RESULTS.md.

`runs/` is gitignored, so a summary that pointed at the run directory would render
blank for everyone who did not produce it. This copies the calibration figures the
assessment already wrote into `docs/figures/prod_test_v0/` and builds the two
purpose-made ones the summary argues from:

  encoder_fix.png   what the encoder padding fix moved, pre vs post, on the metrics
                    whose sign or verdict it changed
  psi_mode.png      why the MAP/MBR psi panels look nothing like the posterior's:
                    the head drives kappa -> 0, so its MODE is numerical residue

    python scripts/make_prod_test_figures.py                     # newest fixed run
    python scripts/make_prod_test_figures.py --run runs/prod_test_v0/<id>

The copied figures come from the run directory, so they always describe the run named on
the command line. The pre-fix column of `encoder_fix.png` is **hard-coded** in `RATIOS` /
`ABSOLUTE` below rather than read from `runs/prod_test_v0_pre_encoder_fix/`: that archive
is local and gitignored, so reading it would make the figure unreproducible for everyone
else, and those numbers are a fixed historical record rather than something to recompute.
Update them by hand if the pre-fix run is ever repeated.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "prod_test_v0"

COPY = ("calibration_pit_coords.png", "calibration_tarp.png", "calibration_by_region.png")

# Ratio metrics: 1.0 is the threshold in every case, and log scale makes a 2x
# improvement and a 2x degradation the same visual distance — which is the honest
# rendering for a quantity that composes multiplicatively.
RATIOS = [
    ("medoid / identity\n(leading emission)", 1.112, 0.902),
    ("geo-median / identity\n(off grid)", 1.111, 0.871),
    ("soft ln$k_t$ third\nmedoid / identity", 1.520, 0.955),
    ("posterior W1\nvs plain RSD", 0.977, 0.414),
    ("TARP max dev\n/ null floor", 0.087 / 0.0785, 0.0367 / 0.0785),
    ("uncalibrated $q(0|x)$\n/ truth rate", 3.10, 0.983),
]
# Absolute metrics, each with its own target — not ratios, so they get their own panel
# rather than being silently rescaled onto someone else's threshold.
ABSOLUTE = [
    ("leading-cell\n68% coverage", 0.455, 0.538, 0.68),
    ("$q(0|x)$ AUC", 0.724, 0.823, None),
]

def _latest_run(root: Path) -> Path:
    cands = sorted(root.glob("*/prod_test_v0/prod_test_v0_metrics.json"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no prod_test_v0_metrics.json under {root} — run the notebook first")
    # .../<id>/prod_test_v0/prod_test_v0_metrics.json -> .../<id>, which is where
    # best.ckpt lives; one .parent short lands in the artifact dir instead
    return cands[-1].parent.parent


def encoder_fix_figure(plt, np, path: Path) -> None:
    """Two panels: ratios against their common 1.0 threshold on a log axis, and the
    absolute metrics against their own targets. Putting a 3.1 next to a 0.455 on one
    linear axis makes the largest number the loudest rather than the most important."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.0),
                                 gridspec_kw={"width_ratios": [3, 1]})
    w = 0.38
    grey, blue = "#B0B0B0", "#4C78A8"

    x = np.arange(len(RATIOS))
    ax.bar(x - w / 2, [r[1] for r in RATIOS], w, color=grey, edgecolor="none",
           label="pre-fix (padding read as signal)")
    ax.bar(x + w / 2, [r[2] for r in RATIOS], w, color=blue, edgecolor="none",
           label="fixed (`encoder.mask_padding`)")
    ax.axhline(1.0, color="#E45756", lw=1.4, ls="--", zorder=3, label="threshold / target")
    for xi, (_l, a, b) in zip(x, RATIOS):
        for off, v in ((-w / 2, a), (w / 2, b)):
            ax.annotate(f"{v:.3g}", (xi + off, v), ha="center",
                        va="bottom" if v >= 1 else "top",
                        xytext=(0, 2 if v >= 1 else -2), textcoords="offset points",
                        fontsize=7.5, color="#333333")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in RATIOS], fontsize=7.5)
    ax.set_ylabel("ratio (log scale; 1.0 = threshold)")
    ax.set_ylim(0.28, 4.2)
    ax.set_title("Ratios — below 1 is better", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")

    xb = np.arange(len(ABSOLUTE))
    bx.bar(xb - w / 2, [r[1] for r in ABSOLUTE], w, color=grey, edgecolor="none")
    bx.bar(xb + w / 2, [r[2] for r in ABSOLUTE], w, color=blue, edgecolor="none")
    for xi, (_l, a, b, tgt) in zip(xb, ABSOLUTE):
        if tgt is not None:
            bx.plot([xi - 0.5, xi + 0.5], [tgt, tgt], color="#E45756", lw=1.4, ls="--",
                    zorder=3)
        for off, v in ((-w / 2, a), (w / 2, b)):
            bx.annotate(f"{v:.3g}", (xi + off, v), ha="center", va="bottom",
                        fontsize=7.5, color="#333333")
    bx.set_xticks(xb)
    bx.set_xticklabels([r[0] for r in ABSOLUTE], fontsize=7.5)
    bx.set_ylim(0, 1.0)
    bx.set_ylabel("value")
    bx.set_title("Absolute — higher is better", fontsize=9.5)

    fig.suptitle("What the encoder padding fix moved — identical data and preset, "
                 "retrained 2x2 grid", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def psi_figure(plt, np, path: Path, run: Path) -> None:
    """Truth / sampled / mode psi on one axis, plus the kappa spectrum that explains it."""
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(REPO / "src"))
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.data.rntuple import load_rntuple
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference

    dev = torch.device("cpu")
    info = load_for_inference(run / "best.ckpt", map_location=dev)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(dev)
    model.load_state_dict(info["model_state"])
    model.eval()

    jets = load_rntuple(str(REPO / "data" / "jet_aux_asym_test.root"), "Jets")[:1500]
    ds = MatchedLundDataset(jets, geom, tuple(model.aux_feature_names))
    mode, drawn, truth, kappa = [], [], [], []
    torch.manual_seed(0)
    with torch.inference_mode():
        for i in range(len(ds)):
            it = ds[i]
            if int(it["ny"]) == 0:
                continue
            xf, nx = it["xf"].unsqueeze(0), torch.tensor([it["nx"]])
            cells = it["yc"].tolist()
            mode.extend(n.psi for n in model.describe_sequence(xf, nx, cells).nodes)
            c = model.sample_coordinates(xf, nx, cells)
            drawn.extend(c[:, 3].tolist())
            truth.extend(it["yraw"][:, 3].tolist())
            e = model.encode(xf, nx)
            out = model._decode_states(it["yc"].unsqueeze(0), e, model.xattn_kv(xf, nx))
            eh = torch.cat([out[:, :len(cells)],
                            e.unsqueeze(1).expand(-1, len(cells), -1)], dim=-1)
            p = model._coord_params(
                torch.cat([eh, model.y_embed(it["yc"].unsqueeze(0))], dim=-1))
            kappa.extend(p[7].flatten().tolist())

    mode, drawn, truth = map(lambda a: np.asarray(a), (mode, drawn, truth))
    kappa = np.asarray(kappa)

    def R(a):
        return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 3.6),
                                 gridspec_kw={"width_ratios": [1.55, 1]})
    bins = np.linspace(-np.pi, np.pi, 25)
    for a, lab, col, ls in ((truth, "truth", "#333333", "-"),
                            ("", "", "", ""),
                            (drawn, "posterior (sampled)", "#4C78A8", "-"),
                            (mode, "MAP / MBR (mode)", "#E45756", "-")):
        if lab == "":
            continue
        ax.hist(a, bins=bins, histtype="step", density=True, lw=1.9, color=col, ls=ls,
                label=f"{lab}   |R| = {R(a):.3f}")
    ax.axhline(1 / (2 * np.pi), color="#999999", lw=1.0, ls=":", label="uniform")
    ax.set_xlabel(r"$\psi$")
    ax.set_ylabel("density")
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.set_title(r"$\psi$ is uniform, and the posterior reproduces it —"
                 "\n" r"the MAP/MBR mode does not", fontsize=9.5)

    bx.hist(kappa, bins=np.linspace(0, max(0.3, float(np.percentile(kappa, 99))), 40),
            color="#4C78A8", alpha=0.85)
    bx.set_xlabel(r"von Mises $\kappa$ of the $\psi$ head")
    bx.set_ylabel("splittings")
    bx.set_title(rf"median $\kappa$ = {np.median(kappa):.3f}: the density is flat,"
                 "\n" rf"so its mode is numerical residue", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"    psi: |R| truth {R(truth):.3f}  sampled {R(drawn):.3f}  mode {R(mode):.3f}"
          f"   median kappa {np.median(kappa):.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="a runs/prod_test_v0/<id> directory")
    ap.add_argument("--skip-psi", action="store_true",
                    help="skip the figure that needs the checkpoint and the ROOT file")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run = Path(args.run) if args.run else _latest_run(REPO / "runs" / "prod_test_v0")
    run = run if run.is_absolute() else REPO / run
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[figures] run {run.relative_to(REPO)}  ->  {OUT.relative_to(REPO)}")

    for name in COPY:
        src = run / "prod_test_v0" / name          # notebook tier: more jets than the CLI's
        if not src.exists():
            src = run / name
        if src.exists():
            shutil.copyfile(src, OUT / name)
            print(f"    copied {name}")
        else:
            print(f"    MISSING {name} — run notebooks/prod_test_v0.ipynb")

    encoder_fix_figure(plt, np, OUT / "encoder_fix.png")
    print("    built encoder_fix.png")
    if not args.skip_psi:
        psi_figure(plt, np, OUT / "psi_mode.png", run)
        print("    built psi_mode.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
