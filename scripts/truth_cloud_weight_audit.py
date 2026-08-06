"""A3 — the `_truth_cloud` `kt`-weight mismatch, measured. docs/PLAN_next_steps.md A3.

`eval/clusters.py::_truth_cloud` builds the truth from the continuous `yraw` rows while
every DRAW cloud was built from cell centres. At the default `mbr_weight="kt"` that weights
the truth by `exp(v_continuous)` and the draws by `exp(v_cell_centre)` — a per-point
mismatch of `exp(±half_v)`, `exp(±0.1) ≈ [0.905, 1.105]` at the fielded `n_bins = 30`, plus
a systematic Jensen inflation of the truth cloud's total mass, which the EMD charges at
`R·|ΔW|` with `R = 8.485`. **`d_top`, `d_best`, `d_mbr`, `d_nearest_draw` and gates
G2′/G6/G7 sit on it today** (`PLAN_z_aware.md` §4/WP-3's inset, left orphaned by §11.3).

WP-3's threading fixes it as a side effect: under `decode.mbr_cloud_source="coords"` the
draws are placed at their own continuous coordinates, so both sides of the EMD are finally
in the same representation and `W_truth/W_draw` is 1 by construction.

**The plan says do not assume the effect is small** — report `W_truth/W_draw` and `R·|ΔW|`
against the typical `d`, then re-read G2′, G6 and G7 and say whether any verdict moved.
That is exactly what this script does, on ONE pass:

    draws once  ->  coordinates once  ->  run_cluster_diagnostics twice,
                                          differing ONLY in `mbr_cloud_source`

so the two gate tables are paired jet-by-jet on byte-identical draws. Nothing here is a
new gate: G2′/G6/G7 are `PLAN_PosteriorClusters.md`'s and are re-read, not re-written.

**A caveat that has to be stated rather than buried.** Flipping `cloud_source` removes the
weight mismatch AND de-quantizes `(u, v)`, and those are not separable — the `kt` weight IS
`exp(v)`, so putting the draws in the truth's representation is what fixes the weights.
`W_ratio` isolates the mass half of it; the gate deltas do not, and are reported as "the
effect of placing the draws at their own coordinates", which is the honest description.

Run:
    python scripts/truth_cloud_weight_audit.py --fast          # smoke: 12 jets, K=16
    python scripts/truth_cloud_weight_audit.py                 # 600 jets, K=200
    python scripts/truth_cloud_weight_audit.py --analyze runs/truth_cloud_audit/<stamp>

Output: the printed tables plus `runs/truth_cloud_audit/<stamp>/audit.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# The published cluster tier (`per_jet_clusters.json`'s own `run` block), quoted rather
# than re-chosen: 600 jets, K = 200, seed 1234, energyflow, hdbscan, min_mass 0.05,
# min_cluster_size auto, R = 8.485, 20 null partitions per jet. Changing any of these means
# the numbers below are not comparable to the gate table they are meant to re-read.
TIER = dict(n_jets=600, k_draws=200, seed=1234, null_reps=20)
DEFAULT_ARM = "runs/prod_test_v1/v1_contstop_s0"      # the FIELDED family
COMMITTED = "per_jet_clusters.json"


def _resolve(arm: str) -> tuple[Path, Path | None]:
    root = Path(arm) if Path(arm).is_absolute() else REPO / arm
    hits = sorted(root.glob("*/best.ckpt")) or (
        [root / "best.ckpt"] if (root / "best.ckpt").is_file() else [])
    if not hits:
        raise FileNotFoundError(f"no best.ckpt under {root}")
    ckpt = hits[-1]
    committed = ckpt.parent / COMMITTED
    return ckpt, (committed if committed.is_file() else None)


# ---------------------------------------------------------------------------
def run(arm: str, *, n_jets: int, k_draws: int, device: str, backend: str,
        null_reps: int, seed: int) -> dict:
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.clusters import run_cluster_diagnostics
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.inference.mbr import coords_for_draws
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    ckpt, committed_path = _resolve(arm)
    seed_everything(int(seed))
    dev = torch.device(device)
    info = load_for_inference(str(ckpt), map_location=dev)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(dev)
    model.load_state_dict(info["model_state"])
    model.eval()

    committed = json.loads(committed_path.read_text()) if committed_path else None
    test_path = (committed or {}).get("run", {}).get("test_path") or cfg.data.path
    cfg.data.path = str(REPO / test_path if not Path(test_path).is_absolute() else test_path)
    dm = LundDataModule(cfg, geom).setup()
    dm.train_jets, dm.val_jets = [], dm.jets
    _, val_ds = dm.datasets()

    dec = dict(decode_params(cfg))
    dec.update({"point_estimator": "mbr", "mbr_backend": backend,
                "mbr_n_candidates": 0, "cluster_posterior": True})

    # --- ONE sampling pass, shared by both arms --------------------------------------
    # This is what makes the comparison a comparison of representations and nothing else:
    # the two `run_cluster_diagnostics` calls see byte-identical draws AND byte-identical
    # coordinates, so every difference below is the ground metric.
    n_jets = min(int(n_jets), len(val_ds))
    t0 = time.time()
    draws_by_jet, coords_by_jet = [], []
    with torch.inference_mode():
        for i in range(n_jets):
            item = val_ds[i]
            xf = item["xf"].unsqueeze(0).to(dev)
            nx = torch.tensor([item["nx"]], device=dev)
            d = model.sample_batch(xf, nx, int(k_draws))
            draws_by_jet.append(d)
            coords_by_jet.append(coords_for_draws(model, xf, nx, d))
    t_sample = time.time() - t0

    out = {
        "arm": arm, "checkpoint": str(ckpt.relative_to(REPO)),
        "committed_artifact": (str(committed_path.relative_to(REPO))
                               if committed_path else None),
        "committed_gates": (committed or {}).get("gates"),
        "committed_run": (committed or {}).get("run"),
        "device": str(dev), "backend": backend,
        "tier": {"n_jets": int(n_jets), "K": int(k_draws), "seed": int(seed),
                 "null_reps": int(null_reps)},
        "data": {"path": str(cfg.data.path), "fingerprint": dm.fingerprint},
        "seconds_sampling": round(t_sample, 1),
        "arms": {},
    }
    for source in ("cells", "coords"):
        t1 = time.time()
        m = run_cluster_diagnostics(
            model, val_ds, dm.val_jets, geom, dev, K=int(k_draws), n_jets=n_jets,
            decode={**dec, "mbr_cloud_source": source}, verbose=False,
            draws_by_jet=draws_by_jet, coords_by_jet=coords_by_jet,
            null_reps=int(null_reps),
        )
        m["seconds"] = round(time.time() - t1, 1)
        out["arms"][source] = m
        print(f"[a3] {source:>6}  W_truth/W_as_drawn = "
              f"{m['weight_audit']['W_truth_over_W_truth_as_drawn']:.4f}"
              f"  R|dW| = {m['weight_audit']['R_dW_mean']:.4f}"
              f"  G2' gain = {m['G2prime']['all']['gain']:+.4f}"
              f"  ({m['seconds']:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# Analysis — the paired per-jet deltas, and the gate re-read
# ---------------------------------------------------------------------------
def _paired(a_rows, b_rows, key) -> dict:
    """`coords - cells`, jet by jet, with a paired BCa 95% interval."""
    from mbr_zaware_ab import bca_bootstrap

    a = {int(r["jet"]): float(r.get(key, np.nan)) for r in a_rows}
    b = {int(r["jet"]): float(r.get(key, np.nan)) for r in b_rows}
    d = np.array([a[j] - b[j] for j in sorted(set(a) & set(b))], dtype=float)
    st = bca_bootstrap(d[np.isfinite(d)])
    st["excludes_0"] = bool(np.isfinite(st["ci95"][0])
                            and (st["ci95"][0] > 0 or st["ci95"][1] < 0))
    return st


def analyse(rec: dict) -> dict:
    cells, coords = rec["arms"]["cells"], rec["arms"]["coords"]
    rc, rd = cells["per_jet"], coords["per_jet"]
    keys = ("d_top", "d_best", "d_mbr", "d_nearest_draw", "d_best_rand",
            "top_mass", "entropy", "radius_top", "pool_bound")
    out = {
        "weight_audit": {"cells": cells["weight_audit"], "coords": coords["weight_audit"]},
        "paired": {k: _paired(rd, rc, k) for k in keys},
        "rates": {
            k: {"cells": cells.get(k), "coords": coords.get(k)}
            for k in ("G2_medoid_in_top", "truth_in_top_rate", "unassigned_rate",
                      "pool_covered_rate", "n_clusters_mean", "frac_multimodal",
                      "top_mass_mean", "entropy_mean", "residual_mass_mean",
                      "silhouette_mean", "G3_empty_mass_vs_q0")
        },
        "gates": {},
    }
    for name, m in (("cells", cells), ("coords", coords)):
        g2p = m["G2prime"]
        out["gates"][name] = {
            "G2_medoid_in_top": m["G2_medoid_in_top"],
            "G2_pass_wp4_closed": m["G2_pass_wp4_closed"],
            "G2prime_all": {k: g2p["all"][k] for k in ("n", "d_best", "d_best_rand",
                                                       "d_top", "d_mbr", "gain",
                                                       "gain_sem")},
            "G2prime_precondition": {k: g2p["precondition_holds"][k]
                                     for k in ("n", "d_best", "d_best_rand", "gain",
                                               "gain_sem")},
            "precondition_rate": g2p["precondition_rate"],
            "G3_empty_mass_vs_q0": m["G3_empty_mass_vs_q0"],
            "G6_ece_raw": m["G6_reliability"]["ece"],
            "G6_ece_recal": m["G6_reliability_recalibrated"]["ece"],
            "G6_temperature": m["G6_temperature"]["value"],
            "G6_slope": m["G6_reliability"]["slope"],
            "G6_brier_resolution": m["G6_reliability"]["brier_resolution"],
            "G6_pass": m["G6_pass"],
            "G7_coverage": (m["G7_conformal"] or {}).get("coverage"),
            "G7_wilson95": (m["G7_conformal"] or {}).get("coverage_wilson95"),
            "G7_nominal": (m["G7_conformal"] or {}).get("nominal"),
            "G7_pass": (m["G7_conformal"] or {}).get("pass"),
            "G7_mean_set_size": (m["G7_conformal"] or {}).get("mean_set_size"),
        }
    # Did any VERDICT move? The gates are boolean by construction; a number moving inside
    # a verdict is reported and is not a verdict change.
    moved = [k for k in ("G2_pass_wp4_closed", "G6_pass", "G7_pass")
             if out["gates"]["cells"][k] != out["gates"]["coords"][k]]
    # G2' is scored by sign-and-significance of the gain rather than by a stored boolean.
    def _g2p_sig(side, label):
        e = out["gates"][side][label]
        return bool(np.isfinite(e["gain_sem"]) and e["gain"] > 2.0 * e["gain_sem"])

    for label in ("G2prime_all", "G2prime_precondition"):
        if _g2p_sig("cells", label) != _g2p_sig("coords", label):
            moved.append(label)
    out["verdicts_moved"] = moved
    out["any_verdict_moved"] = bool(moved)
    return out


def _ci(s):
    return f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def print_report(rec: dict, res: dict) -> None:
    w = res["weight_audit"]
    print("\n" + "=" * 100)
    print("A3 — the truth/draw kt-weight mismatch, and what it costs")
    print("    The SAME truth tree through both representations, so the genuine "
          "multiplicity imbalance\n    is divided out rather than mixed in.")
    print(f"    {'cloud source':>14} {'W_truth':>11} {'as drawn':>11} {'ratio':>9}"
          f" {'R|dW|':>10} {'/<d_mbr>':>10} {'/<d_near>':>10} {'/real imbal.':>13}")
    for name in ("cells", "coords"):
        e = w[name]
        print(f"    {name:>14} {e['W_truth_mean']:>11.3f} "
              f"{e['W_truth_as_drawn_mean']:>11.3f} "
              f"{e['W_truth_over_W_truth_as_drawn']:>9.4f} {e['R_dW_mean']:>10.4f}"
              f" {e['R_dW_over_d_mbr']:>9.1%}"
              f" {e['R_dW_over_d_nearest_draw']:>9.1%}"
              f" {e['R_dW_over_R_dW_physical']:>12.1%}")
    print("    'cells' is the FIELDED path and is the mismatch; 'coords' is 1.0000 by "
          "construction.")
    print(f"    for scale: the REAL (multiplicity) imbalance charge is "
          f"{w['cells']['R_dW_physical_mean']:.4f}, <d_mbr> is "
          f"{res['gates']['cells']['G2prime_all']['d_mbr']:.4f} and <d_nearest_draw> is "
          f"{rec['arms']['cells']['d_nearest_draw_mean']:.4f}.")

    print("\nper-jet paired deltas, coords - cells (BCa 95%; the EMD rows are what the "
          "gates sit on):")
    for k, s in res["paired"].items():
        tag = "moved" if s["excludes_0"] else "straddles 0"
        print(f"    {k:>18} n={s['n']:>4}  {_ci(s):>28}   {tag}")

    print("\nthe PARTITION itself, because it is what actually moves the gates:")
    print(f"    {'quantity':>22} {'cells (fielded)':>18} {'coords':>18}")
    for k in ("n_clusters_mean", "frac_multimodal", "top_mass_mean", "entropy_mean",
              "residual_mass_mean", "unassigned_rate", "pool_covered_rate",
              "silhouette_mean"):
        e = res["rates"][k]
        print(f"    {k:>22} {e['cells']:>18.4f} {e['coords']:>18.4f}")

    print("\ngate re-read (docs/PLAN_PosteriorClusters.md G2 / G2' / G3 / G6 / G7):")
    g = res["gates"]
    rows = [
        ("G2 medoid-in-top", "G2_medoid_in_top", "{:.4f}"),
        ("G2 pass (>=0.90)", "G2_pass_wp4_closed", "{}"),
        ("G3 |mass - q0|", "G3_empty_mass_vs_q0", "{:.5f}"),
        ("G6 ECE raw", "G6_ece_raw", "{:.4f}"),
        ("G6 ECE recalibrated", "G6_ece_recal", "{:.4f}"),
        ("G6 temperature", "G6_temperature", "{:.3f}"),
        ("G6 slope", "G6_slope", "{:.3f}"),
        ("G6 Brier resolution", "G6_brier_resolution", "{:.4f}"),
        ("G6 pass (ECE<=0.05)", "G6_pass", "{}"),
        ("G7 coverage", "G7_coverage", "{:.4f}"),
        ("G7 mean set size", "G7_mean_set_size", "{:.3f}"),
        ("G7 pass", "G7_pass", "{}"),
        ("precondition rate", "precondition_rate", "{:.4f}"),
    ]
    print(f"    {'gate':>24} {'cells (fielded)':>18} {'coords':>18}")
    for label, key, fmt in rows:
        a, b = g["cells"][key], g["coords"][key]
        fa = fmt.format(a) if a is not None else "n/a"
        fb = fmt.format(b) if b is not None else "n/a"
        print(f"    {label:>24} {fa:>18} {fb:>18}")
    for label in ("G2prime_all", "G2prime_precondition"):
        a, b = g["cells"][label], g["coords"][label]
        print(f"    {label:>24} n={a['n']:<4} gain {a['gain']:+.4f} +/- {a['gain_sem']:.4f}"
              f"   |  n={b['n']:<4} gain {b['gain']:+.4f} +/- {b['gain_sem']:.4f}")

    c = rec.get("committed_gates") or {}
    if c:
        print("\ninformative (NOT a gate): the committed notebook artifact, for scale.")
        print(f"    G2 {c.get('G2_medoid_in_top')}   G2' all gain "
              f"{(c.get('G2prime') or {}).get('all', {}).get('gain')}"
              f"   G6 ECE {(c.get('G6_reliability') or {}).get('ece')}")
        print("    It is a different runner and a different RNG stream, so a difference "
              "here is not a\n    failure — the A3 verdict rests on the PAIRED columns "
              "above, which share their draws.")

    print("\n" + "=" * 100)
    print(f"VERDICT: {'A GATE VERDICT MOVED — ' + ', '.join(res['verdicts_moved'])}"
          if res["any_verdict_moved"] else
          "VERDICT: no G2 / G2' / G6 / G7 verdict moves when the draws are placed at "
          "their own coordinates.")
    print("=" * 100)
    print("Read the two halves separately — they are NOT the same size and only the first\n"
          "is A3's question:")
    print("  * the WEIGHT mismatch is the `ratio` and `R|dW|` rows above, isolated on the\n"
          "    same tree so the real multiplicity imbalance is divided out;")
    print("  * a gate that moves moves because the whole PARTITION changed — cell-centre\n"
          "    draws are quantised, so many coincide exactly and hdbscan finds tight lumps;\n"
          "    continuous draws are all distinct. That is de-quantization, not the weight\n"
          "    defect, and the two cannot be separated by flipping this switch (the kt\n"
          "    weight IS exp(v), so fixing the weights IS placing the draws continuously).")
    print("  * G6 in particular must be read with its Brier RESOLUTION, not its ECE alone:\n"
          "    a near-constant forecaster is trivially calibrated and carries no\n"
          "    information, which is a different failure from miscalibration.")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--n-jets", type=int, default=TIER["n_jets"])
    ap.add_argument("--k-draws", type=int, default=TIER["k_draws"])
    ap.add_argument("--seed", type=int, default=TIER["seed"])
    ap.add_argument("--null-reps", type=int, default=TIER["null_reps"])
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu (never mps)")
    ap.add_argument("--backend", default="", help="default: energyflow if importable")
    ap.add_argument("--out", default="")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)

    if args.analyze:
        out_dir = Path(args.analyze)
        rec = json.loads((out_dir / "raw.json").read_text())
    else:
        if args.fast:
            args.n_jets, args.k_draws, args.null_reps = 12, 16, 3
        if args.device == "auto":
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.backend:
            import importlib.util as ilu

            args.backend = "energyflow" if ilu.find_spec("energyflow") else "pot"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else REPO / "runs" / "truth_cloud_audit" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[a3] {args.arm}  {args.n_jets} jets, K={args.k_draws}, "
              f"{args.backend} on {args.device} -> {out_dir}")
        rec = run(args.arm, n_jets=args.n_jets, k_draws=args.k_draws, device=args.device,
                  backend=args.backend, null_reps=args.null_reps, seed=args.seed)
        (out_dir / "raw.json").write_text(json.dumps(rec, indent=1) + "\n")

    res = analyse(rec)
    res["out_dir"] = str(out_dir)
    res["tier"] = rec["tier"]
    res["arm"] = rec["arm"]
    print_report(rec, res)
    (out_dir / "audit.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[a3] wrote {out_dir / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
