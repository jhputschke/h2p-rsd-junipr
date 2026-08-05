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
    # `continue_temperature` is NOT in that list: it rides on the model (set by
    # build_model, re-applied by cmd_eval) and so reaches every draw — but only for a
    # family that has a continue/stop head to temper at all.
    if not hasattr(model, "cont_head"):
        add("continue_temperature",
            "this family has an explicit q(N|x) head and takes no per-step continue "
            "decision; length_temperature/length_tilt is its length knob")
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
        # The cluster layer reads the MBR distance matrix, so without an MBR decode there
        # is no D and every cluster_* knob is inert for the same reason.
        for k in sorted(k for k in decode if k.startswith("cluster_")) + ["set_alpha"]:
            add(k, "point_estimator != 'mbr'; the cluster layer has no distance matrix "
                   "to read")
    elif not bool(decode.get("cluster_posterior", False)):
        for k in sorted(k for k in decode if k.startswith("cluster_")
                        and k != "cluster_posterior") + ["set_alpha"]:
            add(k, "decode.cluster_posterior=false; no cluster labelling is built")

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

    # WP-D.3: the region x coordinate PIT cross, as a heat map of KS distances. This is
    # the instrument gate G5 reads — it says WHICH coordinate fails in WHICH quadrant —
    # and a four-by-four table of numbers is not something a reader scans for a pattern.
    cross = metrics.get("pit_coords_by_region")
    if cross:
        coords = list(cross)
        regions = sorted({r for v in cross.values() for r in v})
        ks = np.array([[cross[c].get(r, {}).get("ks", np.nan) for r in regions]
                       for c in coords], dtype=float)
        ns = np.array([[cross[c].get(r, {}).get("n", 0) for r in regions]
                       for c in coords], dtype=float)
        scored = np.array([[cross[c].get(r, {}).get("scored", False) for r in regions]
                           for c in coords], dtype=bool)
        # Colour by KS / its OWN 95% critical value, never by raw KS. The regions differ
        # in count by ~50x, so on a raw scale the smallest region is always the darkest:
        # on the base arm, psi x wide_hard (0.175, n=52) renders as the worst cell while
        # sitting at 0.93x its critical value — a PASS — and ln_z x wide_soft (0.057,
        # n=2671) renders mild at 2.2x, a real failure. A figure that draws the eye to the
        # passing cell is worse than no figure.
        with np.errstate(invalid="ignore", divide="ignore"):
            crit = 1.36 / np.sqrt(np.maximum(ns, 1.0))
            ratio = ks / crit
        fig, ax = plt.subplots(figsize=(1.6 * len(regions) + 2.6, 0.9 * len(coords) + 2.2))
        # centred at 1.0 = the criterion, so blue passes and red fails on sight
        im = ax.imshow(np.where(scored, ratio, np.nan), cmap="RdBu_r", aspect="auto",
                       vmin=0.0, vmax=2.0)
        for i in range(len(coords)):
            for j in range(len(regions)):
                if not np.isfinite(ks[i, j]):
                    continue
                # An unscored cell is shown but greyed: it is a measurement nobody should
                # act on, and hiding it would read as "that quadrant is fine".
                txt = (f"{ratio[i, j]:.2f}x\nKS {ks[i, j]:.3f}\nn={int(ns[i, j])}")
                ax.text(j, i, txt + ("" if scored[i, j] else "\n(not scored)"),
                        ha="center", va="center", fontsize=7,
                        color="#333333" if not scored[i, j] else
                        ("white" if abs(ratio[i, j] - 1.0) > 0.75 else "black"))
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=20, fontsize=8)
        ax.set_yticks(range(len(coords)))
        ax.set_yticklabels(coords, fontsize=9)
        ax.set_title("PIT KS / its own 95% critical value, by coordinate x Lund quadrant\n"
                     "(>1 fails; the critical value is 1.36/sqrt(n), so it differs per cell)",
                     fontsize=9)
        cb = fig.colorbar(im, ax=ax, label="KS / (1.36/sqrt(n))   — 1.0 is the criterion")
        cb.ax.axhline(1.0, color="black", lw=1.4)
        fig.tight_layout()
        p = out_dir / "calibration_pit_by_region.png"
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
        # Prefer the band recomputed at THIS run's (n, alpha grid) over the asymptotic
        # 1.36/sqrt(n): the plot is where "inside the band" gets read off, so it must be
        # the band the gate uses (docs/PLAN_prod_test_v1.md WP-D.2).
        band = t.get("null_band")
        floor = band["p95"] if band else t.get("tarp_null_floor95")
        src = (f"MC null, {band['n_reps']} reps" if band else "1.36/sqrt(n)")
        sub = (f"\n95% null band {floor:.3f} at n = {t['n_jets']} ({src})"
               + ("" if not band else
                  f" — {'quotable' if band['floor_ok'] else 'NOT quotable: floor >= 0.05'}")
               if floor is not None else "")
        ax.set_title(f"TARP  max dev = {t['tarp_max_dev']:.3f}{sub}", fontsize=9)
        if floor is not None:
            ax.fill_between(alpha, alpha - floor, alpha + floor, color="#999999",
                            alpha=0.12, lw=0, label=f"95% null band ({src})")
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


def plot_clusters(metrics: dict, out_dir: Path) -> list[Path]:
    """The WP6 posterior-cluster figures: the `top_mass` reliability diagram (gate G6) and
    the conformal coverage against nominal (gate G7).

    `metrics` is the `clusters` block of an eval artifact. Same contract as
    `plot_calibration`: matplotlib is optional, and its absence degrades to a note."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not installed; wrote the cluster metrics JSON only.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    rel = metrics.get("G6_reliability")
    if rel and rel.get("bins"):
        relT = metrics.get("G6_reliability_recalibrated") or {}
        T = (metrics.get("G6_temperature") or {}).get("value", 1.0)
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1.2, label="calibrated")
        for entry, colour, label in ((rel, "#4C78A8", "raw"),
                                     (relT, "#E45756", f"tempered (T={T:.2f})")):
            bins = entry.get("bins") or []
            if not bins:
                continue
            f = np.array([b["claimed"] for b in bins])
            o = np.array([b["observed"] for b in bins])
            ci = np.array([b["wilson95"] for b in bins], dtype=float)
            # Wilson bars, not sqrt(p(1-p)/n): these are binomial proportions on a few
            # dozen jets per bin, exactly where the normal approximation leaves [0, 1].
            ax.errorbar(f, o, yerr=np.abs(np.vstack([o - ci[:, 0], ci[:, 1] - o])),
                        fmt="o-", ms=4, lw=1.6, color=colour, capsize=3,
                        label=f"{label}  ECE={entry.get('ece', float('nan')):.3f}")
            # An unscored bin is drawn hollow: a point on 7 jets must not read like one on 200.
            un = [i for i, b in enumerate(bins) if not b.get("scored", True)]
            if un:
                ax.scatter(f[un], o[un], s=64, facecolor="white", edgecolor=colour,
                           zorder=5, linewidths=1.4)
        ax.set_xlabel("claimed top-cluster mass")
        ax.set_ylabel("realized P(truth in top cluster)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"gate G6 — reliability of `top_mass`\nslope = {rel.get('slope', float('nan')):.2f} "
                     f"+/- {rel.get('slope_se', float('nan')):.2f} "
                     f"(hollow: n < {metrics.get('region_min_n', 30)}, not scored)",
                     fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        p = out_dir / "clusters_reliability.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    conf = metrics.get("G7_conformal")
    if conf:
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        cov, ci = conf["coverage"], conf["coverage_wilson95"]
        ax.barh([0], [cov], color="#54A24B", height=0.45)
        ax.errorbar([cov], [0], xerr=[[cov - ci[0]], [ci[1] - cov]], fmt="none",
                    ecolor="#333333", elinewidth=1.4, capsize=5)
        ax.axvline(conf["nominal"], color="#E45756", ls="--", lw=1.4,
                   label=f"nominal {conf['nominal']:.2f}")
        ax.set_yticks([])
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("empirical set coverage")
        ax.set_title(f"gate G7 — conformal coverage at threshold {conf['value']:.3f}\n"
                     f"mean set size {conf['mean_set_size']:.2f}; MARGINAL over jets, "
                     f"not conditional on x", fontsize=9)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        p = out_dir / "clusters_conformal_coverage.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)
    return written
