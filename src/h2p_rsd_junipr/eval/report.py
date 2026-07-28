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
        ax.set_title(f"TARP  max dev = {t['tarp_max_dev']:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        p = out_dir / "calibration_tarp.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    reg = metrics.get("by_region")
    if reg:
        labels = list(reg)
        fig, ax = plt.subplots(figsize=(1.35 * max(len(labels), 3) + 1.6, 3.4))
        cov = [reg[k]["coverage_68"] for k in labels]
        ax.bar(np.arange(len(labels)), cov, color="#54A24B", width=0.6)
        ax.axhline(0.68, color="#E45756", ls="--", lw=1.4, label="target 0.68")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylabel("leading-cell 68% coverage")
        ax.set_title("region-stratified coverage", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "calibration_by_region.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)
    return written
