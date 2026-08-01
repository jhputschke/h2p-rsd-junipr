"""Evaluation artifacts: the metrics record and the calibration figures.

`h2p-rsd-junipr eval` prints its numbers; this writes them next to the checkpoint
so a run directory carries its own evidence (`eval_metrics.json` plus PNGs). JSON
is always written — it is the machine-readable input to the WP4 A/B table. Figures
need matplotlib, which is NOT a package dependency, so their absence degrades to a
one-line note instead of an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()  # np.bool_ is neither of the other two, and json rejects it
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def inert_decode_keys(model, decode: dict) -> list[dict]:
    """Which `decode.*` knobs did NOT reach the numbers in this report, and why.

    `metrics["decode"]` is a faithful dump of the decode config, which makes it a
    faithful record of settings that had no effect — a reader comparing two runs on
    `beam_width` would be comparing a knob neither run consulted
    (docs/PLAN_prod_test_v0.md check 11). Each entry is `{key, value, reason}`; an empty
    list means every knob was live.

    Deliberately derived from the model's own flags and the call sites in `cmd_eval`,
    not hand-maintained per preset: `use_multiplicity_head` alone decides whether the
    beam keys mean anything, and only the code knows that."""
    inert: list[dict] = []

    def add(key, reason):
        if key in decode:
            inert.append({"key": key, "value": decode[key], "reason": reason})

    # `sample_batch(xf, nx, n_samples, max_emissions=25)` takes no `cont_temperature`
    # and does not forward the decode `max_emissions`, so every posterior draw behind
    # run_closure / run_calibration / run_tarp is at T=1, capped at the signature default.
    add("cont_temperature", "PosteriorModel.sample_batch takes no cont_temperature; "
                            "every posterior draw here is at T=1")
    add("max_emissions", "sample_batch uses its own signature default (25); the decode "
                         "value reaches map_estimate but not the draws")
    # closure/calibration take their draw count from experiment.n_closure_samples (K).
    add("n_posterior_samples", "closure/calibration draw experiment.n_closure_samples "
                               "(K) instead")

    if bool(getattr(model, "use_multiplicity_head", False)):
        why = ("use_multiplicity_head=true routes map_decode to _map_decode_fixed_length "
               "(greedy argmax per step); beam_search_cells is never called")
        for k in ("beam_width", "topk_cells", "length_penalty"):
            add(k, why)

    if str(decode.get("point_estimator", "map")) != "mbr":
        why = "point_estimator != 'mbr'; no OT backend is imported"
        for k in sorted(k for k in decode if k.startswith("mbr_")):
            add(k, why)

    # `length_floor_quantile` is applied by print_point_estimate (and the serving API),
    # not by run_closure's map_or_mbr — so the closure table is unfloored either way.
    if float(decode.get("length_floor_quantile", 0.0)) > 0.0:
        add("length_floor_quantile",
            "applied by print_point_estimate only; run_closure's map_or_mbr does not "
            "floor, so the closure/calibration numbers are unfloored")
    return inert


def save_metrics(metrics: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=False) + "\n")
    return path


def _bar_uniform(ax, entry, title):
    """One rank/PIT histogram with the Uniform(0,1) expectation drawn on it."""
    hist = np.asarray(entry["hist"], dtype=float)
    edges = np.asarray(entry["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax.bar(centers, hist, width=width * 0.92, color="#4C78A8", edgecolor="none")
    ax.axhline(hist.sum() / max(len(hist), 1), color="#E45756", lw=1.4, ls="--",
               label="uniform")
    ax.set_title(f"{title}\nKS={entry['ks']:.3f}  mean={entry['mean']:.3f}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_calibration(metrics: dict, out_dir: Path) -> list[Path]:
    """Write the WP2 figures that exist in `metrics`; returns the paths written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not installed; wrote metrics JSON only "
              "(pip install matplotlib for the calibration figures).")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    pits = metrics.get("pit_coords")
    if pits:
        names = list(pits["names"])
        fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 2.8))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            _bar_uniform(ax, pits["coords"][name], name)
        axes[0].set_ylabel("emissions")
        fig.suptitle(f"per-coordinate PIT ({pits['space']} space)", fontsize=10)
        fig.tight_layout()
        p = out_dir / "calibration_pit_coords.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    t = metrics.get("tarp")
    if t:
        fig, ax = plt.subplots(figsize=(4.0, 3.6))
        alpha = np.asarray(t["alpha"], dtype=float)
        ecp = np.asarray(t["ecp"], dtype=float)
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1.2, label="calibrated")
        ax.plot(alpha, ecp, color="#4C78A8", lw=2.0, marker="o", ms=3,
                label=f"TARP ({t['reference']} refs)")
        ax.fill_between(alpha, alpha, ecp, color="#4C78A8", alpha=0.15)
        ax.set_xlabel("credibility level $\\alpha$")
        ax.set_ylabel("expected coverage ECP($\\alpha$)")
        # The floor belongs in the title: a sup-norm CDF deviation is a KS statistic, and
        # without its ~1.36/sqrt(n) null value any nonzero max dev reads as a defect.
        floor = t.get("tarp_null_floor95")
        sub = (f"\n95% null floor {floor:.3f} at n = {t['n_jets']}"
               if floor is not None else "")
        ax.set_title(f"TARP  max dev = {t['tarp_max_dev']:.3f}{sub}", fontsize=9)
        if floor is not None:
            ax.fill_between(alpha, alpha - floor, alpha + floor, color="#999999",
                            alpha=0.12, lw=0, label="95% null band")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        p = out_dir / "calibration_tarp.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    reg = metrics.get("by_region")
    if reg:
        labels = list(reg)
        fig, ax = plt.subplots(figsize=(1.55 * max(len(labels), 3) + 1.6, 3.6))
        cov = np.array([reg[k]["coverage_68"] for k in labels], dtype=float)
        x = np.arange(len(labels))
        # A bare bar against a 0.68 line invites reading every gap as a failure. These
        # are binomial proportions on a few dozen jets per quadrant, so the Wilson
        # interval and the count are what make the comparison meaningful; a quadrant
        # below the stated minimum-n is drawn hollow and NOT scored.
        scored = np.array([bool(reg[k].get("scored", True)) for k in labels])
        ci = np.array([reg[k].get("coverage_68_ci", [np.nan, np.nan]) for k in labels],
                      dtype=float)
        err = np.abs(np.vstack([cov - ci[:, 0], ci[:, 1] - cov])) if ci.size else None
        ax.bar(x[scored], cov[scored], color="#54A24B", width=0.6, label="scored")
        if (~scored).any():
            ax.bar(x[~scored], cov[~scored], facecolor="none", edgecolor="#54A24B",
                   width=0.6, hatch="///",
                   label=f"n < {metrics.get('region_min_n', '?')}, not scored")
        if err is not None and np.isfinite(err).any():
            ax.errorbar(x, cov, yerr=err, fmt="none", ecolor="#333333", elinewidth=1.2,
                        capsize=4, label="95% Wilson")
        for xi, k in zip(x, labels):
            # boxed: a short bar puts its own top right where the count would sit
            ax.annotate(f"n={reg[k].get('n_coverage', reg[k]['n_jets'])}",
                        (xi, 0.02), ha="center", va="bottom", fontsize=7, color="#333333",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=0.85))
        ax.axhline(0.68, color="#E45756", ls="--", lw=1.4, label="target 0.68")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("leading-cell 68% coverage")
        ax.set_title("region-stratified coverage", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / "calibration_by_region.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)
    return written
