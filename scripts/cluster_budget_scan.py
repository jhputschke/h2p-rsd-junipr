"""B4 — un-confound G6's cross-K row. docs/PLAN_next_steps.md B4, SUMMARY §4.1(5).

`cluster_min_cluster_size = 0` resolves to `max(5, ceil(min_mass * K))`, so the fielded
default is **10** at `K = 200` and **50** at `K = 1000`. The two committed cluster
artifacts therefore differ in the budget *and* in the clustering granularity at once, and
`SUMMARY` §2.2 records the consequence:

> *"the 'unresolvable half' was partly budget; **but** `min_cluster_size ∝ K` confounds G6
> across K (residual mass 0.284→0.362, unassigned 35.7%→43.5% while the pool support
> improved) — G6's cross-K row is unscored"*

**Four cells, not the three the plan item names**, and the reason is a knob interaction the
plan item does not mention. "Hold the granularity fixed" is ambiguous because the cluster
layer has *two* granularity knobs that scale differently with `K`: `min_cluster_size` is a
COUNT and `min_mass` is a FRACTION. Pinning only the count at `K = 1000` asks HDBSCAN for
many small clusters and then lets `min_mass = 0.05` (= 50 draws there) fold nearly all of
them into the residual bucket — the partition collapses to about one reportable cluster.
That cell is run and reported, and a fourth holds **every** granularity knob at its
`K = 200` value in absolute draws:

    K200_mcs10         200 draws, mcs 10, min_mass 0.05 (= 10 draws)  -- committed K=200
    K1000_mcs50       1000 draws, mcs 50, min_mass 0.05 (= 50 draws)  -- committed K=1000
    K1000_mcs10       1000 draws, mcs 10, min_mass 0.05 (= 50 draws)  -- B4 LITERALLY
    K1000_mcs10_mm01  1000 draws, mcs 10, min_mass 0.01 (= 10 draws)  -- the clean arm

so the confounded comparison decomposes:

    K200_mcs10       -> K1000_mcs10_mm01   the effect of MORE DRAWS, granularity fixed
    K1000_mcs10_mm01 -> K1000_mcs50        the effect of COARSER CLUSTERING, draws fixed
    K200_mcs10       -> K1000_mcs50        what the committed pair actually measured

**Nested by construction.** One sampling pass of `K = 1000` per jet; the `K = 200` cell is
its first 200 draws. So the cells are paired jet-by-jet *and* the small pool is a subsample
of the large one — a stronger design than two independent runs, and free.

This is a re-read of `PLAN_PosteriorClusters.md`'s G2/G2'/G3/G5/G6/G7, not a new gate.

Run:
    python scripts/cluster_budget_scan.py --fast            # smoke: 8 jets, K=64/16
    python scripts/cluster_budget_scan.py                   # 600 jets, K=1000
    python scripts/cluster_budget_scan.py --analyze runs/cluster_budget/<stamp>
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

from truth_cloud_weight_audit import _resolve  # noqa: E402 -- same arm-resolution rule

# The published cluster tier, quoted rather than re-chosen (`per_jet_clusters.json`'s own
# `run` block): 600 jets, seed 1234, energyflow, hdbscan, min_mass 0.05, 20 null reps.
TIER = dict(n_jets=600, k_big=1000, k_small=200, seed=1234, null_reps=20)
DEFAULT_ARM = "runs/prod_test_v1/v1_contstop_s0"
# (label, K, min_cluster_size, min_mass). `mcs` is PINNED on every cell — never `0` —
# because `0` is exactly the K-dependence this scan exists to remove.
#
# **Why there are FOUR cells and not the three SUMMARY §4.1(5) asks for.** "Hold the
# clustering granularity fixed" has two knobs, and they scale differently:
#
#   `cluster_min_cluster_size` is a COUNT      -> pinning it at 10 makes the density
#                                                 smoothing 5x finer at K = 1000
#   `cluster_min_mass`         is a FRACTION   -> 0.05 is 10 draws at K = 200 and
#                                                 50 draws at K = 1000
#
# So `mcs = 10, min_mass = 0.05` at K = 1000 asks HDBSCAN for many small clusters and then
# folds nearly all of them into the residual bucket, and the partition collapses to ~1
# reportable cluster. That is the plan item taken literally, and it is reported as such —
# but it does not isolate anything, so the fourth cell holds **every** granularity knob at
# its K = 200 value in ABSOLUTE draw counts (`min_mass = 10/1000 = 0.01`). That is the
# comparison that separates "more draws" from "coarser clustering".
CELLS = (
    ("K200_mcs10", 200, 10, 0.05),      # the committed K=200 tier (mcs = 5% of K)
    ("K1000_mcs50", 1000, 50, 0.05),    # the committed K=1000 tier (mcs = 5% of K)
    ("K1000_mcs10", 1000, 10, 0.05),    # B4 literally — degenerate, and reported so
    ("K1000_mcs10_mm01", 1000, 10, 0.01),   # every knob fixed in ABSOLUTE draws
)


def run(arm: str, *, n_jets: int, k_big: int, k_small: int, device: str, backend: str,
        null_reps: int, seed: int, cells) -> dict:
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.clusters import run_cluster_diagnostics
    from h2p_rsd_junipr.geometry import Geometry
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

    # ONE sampling pass at the LARGE budget; the small cell is its prefix, so the two
    # pools are nested and every comparison below is paired at the jet.
    n_jets = min(int(n_jets), len(val_ds))
    t0 = time.time()
    draws_big = []
    with torch.inference_mode():
        for i in range(n_jets):
            item = val_ds[i]
            draws_big.append(model.sample_batch(
                item["xf"].unsqueeze(0).to(dev),
                torch.tensor([item["nx"]], device=dev), int(k_big)))
    t_sample = time.time() - t0

    out = {
        "arm": arm, "checkpoint": str(ckpt.relative_to(REPO)), "device": str(dev),
        "backend": backend,
        "committed_artifact": (str(committed_path.relative_to(REPO))
                               if committed_path else None),
        "committed_gates": (committed or {}).get("gates"),
        "tier": {"n_jets": int(n_jets), "K_big": int(k_big), "K_small": int(k_small),
                 "seed": int(seed), "null_reps": int(null_reps),
                 "cells": [list(c) for c in cells]},
        "data": {"path": str(cfg.data.path), "fingerprint": dm.fingerprint},
        "seconds_sampling": round(t_sample, 1),
        "cells": {},
    }
    for label, K, mcs, mm in cells:
        dbj = [d[:K] for d in draws_big]
        t1 = time.time()
        m = run_cluster_diagnostics(
            model, val_ds, dm.val_jets, geom, dev, K=K, n_jets=n_jets, decode=dec,
            verbose=False, draws_by_jet=dbj, null_reps=int(null_reps),
            cluster_kwargs={"min_cluster_size": int(mcs), "min_mass": float(mm)},
        )
        m["seconds"] = round(time.time() - t1, 1)
        out["cells"][label] = m
        print(f"[b4] {label:>17}  <n_clusters> = {m['n_clusters_mean']:.2f}"
              f"  <top_mass> = {m['top_mass_mean']:.3f}  <H> = {m['entropy_mean']:.3f}"
              f"  G6 ECE(T) = {m['G6_reliability_recalibrated']['ece']:.4f}"
              f"  ({m['seconds']:.0f}s)")
    return out


# ---------------------------------------------------------------------------
def _paired(rec, a, b, key) -> dict:
    """`a - b`, jet by jet, paired BCa 95%."""
    from mbr_zaware_ab import bca_bootstrap

    ra = {int(r["jet"]): float(r.get(key, np.nan)) for r in rec["cells"][a]["per_jet"]}
    rb = {int(r["jet"]): float(r.get(key, np.nan)) for r in rec["cells"][b]["per_jet"]}
    d = np.array([ra[j] - rb[j] for j in sorted(set(ra) & set(rb))], dtype=float)
    st = bca_bootstrap(d[np.isfinite(d)])
    st["excludes_0"] = bool(np.isfinite(st["ci95"][0])
                            and (st["ci95"][0] > 0 or st["ci95"][1] < 0))
    return st


SCALARS = ("top_mass", "entropy", "radius_top", "n_clusters", "residual_mass",
           "d_mbr", "d_best", "d_top", "d_nearest_draw", "pool_bound")
GATES = (("G2_medoid_in_top", "{:.4f}"), ("truth_in_top_rate", "{:.4f}"),
         ("unassigned_rate", "{:.4f}"), ("pool_covered_rate", "{:.4f}"),
         ("n_clusters_mean", "{:.3f}"), ("top_mass_mean", "{:.4f}"),
         ("entropy_mean", "{:.4f}"), ("residual_mass_mean", "{:.4f}"),
         ("silhouette_mean", "{:.4f}"), ("top_mass_mc_error", "{:.5f}"),
         ("G3_empty_mass_vs_q0", "{:.5f}"))
# Each contrast changes ONE thing. The first is the one that answers B4.
CONTRASTS = (
    ("more draws, granularity FIXED in absolute draws", "K1000_mcs10_mm01", "K200_mcs10"),
    ("coarser clustering, draws fixed", "K1000_mcs50", "K1000_mcs10_mm01"),
    ("the committed pair (CONFOUNDED)", "K1000_mcs50", "K200_mcs10"),
    ("B4 taken literally (mcs pinned, min_mass not)", "K1000_mcs10", "K200_mcs10"),
)


def analyse(rec: dict) -> dict:
    cells = rec["cells"]
    present = [c[0] for c in CELLS if c[0] in cells]
    out = {"cells": {}, "contrasts": {}, "tier": rec["tier"], "arm": rec["arm"]}
    for c in present:
        m = cells[c]
        out["cells"][c] = {
            **{k: m.get(k) for k, _ in GATES},
            "G2_pass_wp4_closed": m["G2_pass_wp4_closed"],
            "G2prime_gain": m["G2prime"]["all"]["gain"],
            "G2prime_gain_sem": m["G2prime"]["all"]["gain_sem"],
            "G2prime_n": m["G2prime"]["all"]["n"],
            "precondition_rate": m["G2prime"]["precondition_rate"],
            "G6_ece_raw": m["G6_reliability"]["ece"],
            "G6_ece_recal": m["G6_reliability_recalibrated"]["ece"],
            "G6_temperature": m["G6_temperature"]["value"],
            "G6_slope": m["G6_reliability"]["slope"],
            "G6_brier_resolution": m["G6_reliability"]["brier_resolution"],
            "G6_pass": m["G6_pass"],
            "G7_coverage": (m["G7_conformal"] or {}).get("coverage"),
            "G7_mean_set_size": (m["G7_conformal"] or {}).get("mean_set_size"),
            "G7_pass": (m["G7_conformal"] or {}).get("pass"),
            "weight_audit": m.get("weight_audit"),
            "seconds": m.get("seconds"),
        }
    for label, a, b in CONTRASTS:
        if a in cells and b in cells:
            out["contrasts"][label] = {"a": a, "b": b,
                                       "paired": {k: _paired(rec, a, b, k)
                                                  for k in SCALARS}}
    # Does the cross-K comparison survive holding granularity fixed? That is the whole
    # question: a quantity that moves under "more draws" AND under "coarser clustering" in
    # the same direction was never attributable in the committed pair.
    verdict = {}
    if all(lbl in out["contrasts"] for lbl, _, _ in CONTRASTS[:3]):
        d = out["contrasts"]["more draws, granularity FIXED in absolute draws"]
        g = out["contrasts"]["coarser clustering, draws fixed"]
        for k in SCALARS:
            dd, gg = d["paired"][k], g["paired"][k]
            verdict[k] = {
                "draws": dd["mean"], "draws_sig": dd["excludes_0"],
                "granularity": gg["mean"], "granularity_sig": gg["excludes_0"],
                # attributable == the budget moves it and the granularity does not
                "attributable_to_draws": bool(dd["excludes_0"] and not gg["excludes_0"]),
                "confounded": bool(dd["excludes_0"] and gg["excludes_0"]),
            }
    out["attribution"] = verdict
    return out


def _ci(s):
    return f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def print_report(res: dict) -> None:
    cells = res["cells"]
    names = list(cells)
    print("\n" + "=" * 108)
    print("B4 — the cluster budget scan: does K move the gates, or does min_cluster_size?")
    print(f"    {'quantity':>26} " + " ".join(f"{n:>17}" for n in names))
    for k, fmt in GATES:
        print(f"    {k:>26} " + " ".join(
            f"{fmt.format(cells[n][k]) if cells[n][k] is not None else 'n/a':>17}"
            for n in names))
    for k, fmt in (("G2_pass_wp4_closed", "{}"), ("precondition_rate", "{:.4f}"),
                   ("G2prime_gain", "{:+.4f}"), ("G2prime_gain_sem", "{:.4f}"),
                   ("G6_ece_raw", "{:.4f}"), ("G6_ece_recal", "{:.4f}"),
                   ("G6_temperature", "{:.3f}"), ("G6_slope", "{:.3f}"),
                   ("G6_brier_resolution", "{:.4f}"), ("G6_pass", "{}"),
                   ("G7_coverage", "{:.4f}"), ("G7_mean_set_size", "{:.3f}"),
                   ("G7_pass", "{}")):
        print(f"    {k:>26} " + " ".join(
            f"{fmt.format(cells[n][k]) if cells[n][k] is not None else 'n/a':>17}"
            for n in names))

    for label, cont in res["contrasts"].items():
        print(f"\npaired delta — {label}   ({cont['a']} - {cont['b']})")
        print(f"    {'quantity':>18} {'n':>5} {'mean':>10} {'95% CI':>24}")
        for k, s in cont["paired"].items():
            tag = "moved" if s["excludes_0"] else "straddles 0"
            print(f"    {k:>18} {s['n']:>5} {s['mean']:>+10.4f}"
                  f" [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  {tag}")

    att = res.get("attribution") or {}
    if att:
        print("\nATTRIBUTION — what the committed cross-K comparison could and could not say")
        print(f"    {'quantity':>18} {'more draws':>14} {'coarser mcs':>14}  reading")
        for k, e in att.items():
            if e["attributable_to_draws"]:
                reading = "attributable to the BUDGET"
            elif e["confounded"]:
                reading = "CONFOUNDED — both move it"
            elif e["granularity_sig"]:
                reading = "attributable to GRANULARITY"
            else:
                reading = "neither resolves it"
            print(f"    {k:>18} {e['draws']:>+14.4f} {e['granularity']:>+14.4f}  {reading}")
    print("=" * 108)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--n-jets", type=int, default=TIER["n_jets"])
    ap.add_argument("--k-big", type=int, default=TIER["k_big"])
    ap.add_argument("--k-small", type=int, default=TIER["k_small"])
    ap.add_argument("--seed", type=int, default=TIER["seed"])
    ap.add_argument("--null-reps", type=int, default=TIER["null_reps"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)

    cells = CELLS
    if args.analyze:
        out_dir = Path(args.analyze)
        rec = json.loads((out_dir / "raw.json").read_text())
    else:
        if args.fast:
            args.n_jets, args.k_big, args.k_small, args.null_reps = 8, 64, 16, 3
            cells = (("K200_mcs10", 16, 5, 0.30), ("K1000_mcs50", 64, 12, 0.30),
                     ("K1000_mcs10", 64, 5, 0.30), ("K1000_mcs10_mm01", 64, 5, 0.075))
        if args.device == "auto":
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.backend:
            import importlib.util as ilu

            args.backend = "energyflow" if ilu.find_spec("energyflow") else "pot"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else REPO / "runs" / "cluster_budget" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[b4] {args.arm}  {args.n_jets} jets, K={args.k_big}/{args.k_small}, "
              f"{args.backend} on {args.device} -> {out_dir}")
        rec = run(args.arm, n_jets=args.n_jets, k_big=args.k_big, k_small=args.k_small,
                  device=args.device, backend=args.backend, null_reps=args.null_reps,
                  seed=args.seed, cells=cells)
        (out_dir / "raw.json").write_text(json.dumps(rec, indent=1) + "\n")

    res = analyse(rec)
    res["out_dir"] = str(out_dir)
    print_report(res)
    (out_dir / "budget.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[b4] wrote {out_dir / 'budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
