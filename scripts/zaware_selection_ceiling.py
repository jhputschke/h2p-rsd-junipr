"""Is the `ln z`-blind SELECTION worth fixing? — docs/PLAN_z_aware.md §12.

WP-0 answered a different question and killed its own premise: there is no `d(MBR)`
regression to explain (§11.1). What it never asked is whether the EMD ground metric should
see `ln z` **at all**, and §11.5's within-arm paired numbers make that a live question on
their own:

    MBR winner's ln z   -  identity(x)'s ln z   :  -0.008..+0.008,  0/8 CIs exclude 0
    geo-median's ln z   -  MBR winner's ln z    :  -0.047..-0.071,  8/8 CIs exclude 0

The MBR winner's `ln z` is a **single draw** from `q(ln z | cells, x)` — it carries the full
posterior variance and is scored by a distance, which is exactly the mismatch `medoid_cell`
already fixed one coordinate over ("the mode is the estimator for a loss nobody is
measuring"). So ~0.065 of `|d ln z|`, about 16% of the 0.41 baseline, sits between what the
decode reports and what the same posterior already knows.

**How much of that can a selection restricted to the drawn pool actually recover, and does
it cost the `(u, v)` half?** That is what this script measures, and the answer decides
whether WP-3's plumbing is worth building.

Three selections per jet off byte-identical draws AND coordinates, differing only in the
ground metric — so the comparison is a comparison of metrics and nothing else:

    cells-2D   cell centres,     gdim 2   -- the FIELDED selection
    cont-2D    coordinate rows,  gdim 2   -- de-quantization alone
    cont-3D    coordinate rows,  gdim 3   -- de-quantization + ln z

`cont-2D` is not optional: "+lnz vs cells" changes two things at once, and without it a win
is unattributable (§4/WP-4, restated as §12.1's attribution clause).

Plus two free reference points that bracket what any pool-restricted rule could reach:

    pool-medoid(ln z)   the draw minimising mean |d ln z| to the other draws -- the best
                        `H = {pool}` can do on the ln z axis ALONE, i.e. the real ceiling
    free-median(ln z)   the unrestricted L1 Bayes point -- the 0.065 figure above

B1/B2/B3 are §12.2 and were committed before this file existed. Nothing here is tuned
against anything it produces: `R`, `K`, `mbr_n_candidates` and `lnkt_cut` are the fielded
values.

Run:
    python scripts/zaware_selection_ceiling.py --fast              # smoke, 12 jets, 1 arm
    python scripts/zaware_selection_ceiling.py --only spline_s0    # one arm
    python scripts/zaware_selection_ceiling.py --analyze runs/zaware_sel/<stamp>

Output: the printed tables plus `runs/zaware_sel/<stamp>/{<arm>.json, ceiling.json}`.
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

from mbr_zaware_ab import (  # noqa: E402  -- the sibling runner owns the shared machinery
    ARM_ROOT,
    DECODE_TIER,
    DEFAULT_TEST,
    bca_bootstrap,
    resolve_arm,
)

SELECTIONS = ("cells_2d", "cont_2d", "cont_3d")
SPLINE_ARMS = ("spline_s0", "spline_s1", "spline_s2", "contstop_spline_s0")

# §12.2, fixed before any arm ran. Repeated here so the printer cannot drift from the doc.
B1_MIN_GAIN = 0.020      # `dlnz` must improve by at least this much...
B2_MAX_LOSS = 0.020      # ...without the (u, v) half degrading by more than this
B3_MIN_MOVED = 0.05      # ...and the selection has to actually move


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------
def run_arm(arm: str, *, test_file: str, n_jets: int, k_draws: int, device: str) -> dict:
    """One pass: draws once, coordinates once, three selections off both.

    The draw order mirrors `cli.py`'s `eval` for the same reason WP-0's runner does — the
    global RNG stream is consumed identically, so `cells-2D` here reproduces the fielded
    number rather than merely resembling it (a sanity row is printed for that)."""
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params, load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.closure import _leading_coords, leading_emission_cell
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.inference.clusters import ground_diameter
    from h2p_rsd_junipr.inference.mbr import lund_cloud, lund_emd_matrix
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    ckpt, _ = resolve_arm(arm)
    seed_everything(int(load_config([]).trainer.seed))
    dev = torch.device(device)
    info = load_for_inference(str(ckpt), map_location=dev)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(dev)
    model.load_state_dict(info["model_state"])
    cfg.data.path = str(test_file)
    dm = LundDataModule(cfg, geom).setup()
    dm.train_jets, dm.val_jets = [], dm.jets
    _, val_ds = dm.datasets()
    model.eval()

    dec = dict(decode_params(cfg))
    dec.update(DECODE_TIER)
    n_cand = int(dec["mbr_n_candidates"])
    emd_kw = dict(R=float(dec["mbr_R"]), beta=float(dec["mbr_beta"]),
                  norm=bool(dec["mbr_norm"]), backend="pot")
    cloud_kw = dict(lnkt_cut=dec["mbr_lnkt_cut"], weight=str(dec["mbr_weight"]))
    # Guard, asserted rather than assumed (§12.2): the EMD is a metric only when
    # R >= half the maximum ground distance, and that maximum grows with gdim.
    guards = {c: {"ground_diameter": ground_diameter(geom, c),
                  "kmt_bound": ground_diameter(geom, c) / 2.0,
                  "R": float(dec["mbr_R"]),
                  "ok": float(dec["mbr_R"]) >= ground_diameter(geom, c) / 2.0}
              for c in ("lnDR_lnkt", "+lnz")}
    if not all(g["ok"] for g in guards.values()):
        raise ValueError(f"R={dec['mbr_R']} fails the KMT bound at one of the gdims: {guards}")

    t0 = time.time()
    rows: list[dict] = []
    n_jets = min(int(n_jets), len(val_ds))
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(dev)
        nx = torch.tensor([item["nx"]], device=dev)
        y_true = item["yc"].tolist()
        ly = leading_emission_cell(y_true, geom)
        draws = model.sample_batch(xf, nx, int(k_draws))
        row = {"jet": int(i), "ny_true": len(y_true), "kept": False}
        rows.append(row)
        if ly is None or not draws:
            continue
        y_lead = _leading_coords(item["yraw"].numpy(), 3)
        if y_lead is None:
            continue

        # Coordinates ONCE, index-aligned with `draws`. `sample_coordinates_many` pads to
        # L_max and slices back, so an empty draw returns an honest (0, 4) table and the
        # alignment survives without filtering.
        coords = model.sample_coordinates_many(xf, nx, [list(d) for d in draws])
        if coords and coords[0] is None:
            continue                       # no coordinate density (ar_junipr_v1)
        tables = [c.detach().cpu().double().numpy().reshape(-1, 4) for c in coords]

        clouds = {
            "cells_2d": [lund_cloud(d, geom, coords="lnDR_lnkt", **cloud_kw) for d in draws],
            "cont_2d": [lund_cloud(t, geom, coords="lnDR_lnkt", **cloud_kw) for t in tables],
            "cont_3d": [lund_cloud(t, geom, coords="+lnz", **cloud_kw) for t in tables],
        }
        cand = list(range(min(n_cand, len(draws)))) if n_cand else list(range(len(draws)))
        win = {}
        for name, cl in clouds.items():
            D = lund_emd_matrix([cl[j] for j in cand], cl, geom=geom, **emd_kw)
            win[name] = int(cand[int(np.argmin(D.mean(axis=1)))])

        # Every selection scored by the SAME ruler on the SAME coordinates: only which
        # draw won differs, so a difference is a difference of metrics.
        lead = [_leading_coords(t, 3) for t in tables]
        for name, k in win.items():
            r = lead[k]
            if r is None:
                row[f"dlnz_{name}"] = row[f"dlund_{name}"] = float("nan")
                continue
            row[f"dlnz_{name}"] = float(abs(r[2] - y_lead[2]))
            row[f"dlund_{name}"] = float(np.linalg.norm(r[:2] - y_lead[:2]))
            row[f"cell_{name}"] = int(leading_emission_cell(
                [int(c) for c in draws[k]], geom) or -1)
        row["winner_moved_cont3d"] = bool(win["cont_3d"] != win["cells_2d"])
        row["winner_moved_cont2d"] = bool(win["cont_2d"] != win["cells_2d"])
        row["leading_cell_moved_cont3d"] = bool(
            row.get("cell_cont_3d", -1) != row.get("cell_cells_2d", -2))

        # --- the two ceilings, pure numpy over the SAME leading emissions -------------
        lz = np.array([r[2] for r in lead if r is not None], dtype=float)
        if lz.size >= 2:
            # `H = {pool}`: the draw whose ln z minimises mean |d ln z| to the others.
            risk = np.abs(lz[:, None] - lz[None, :]).mean(axis=1)
            row["dlnz_pool_medoid"] = float(abs(lz[int(np.argmin(risk))] - y_lead[2]))
            # unrestricted L1 Bayes point
            row["dlnz_free_median"] = float(abs(float(np.median(lz)) - y_lead[2]))
            row["lnz_pool_iqr"] = float(np.subtract(*np.percentile(lz, [75, 25])))
        row["kept"] = True

    return {
        "arm": arm, "checkpoint": str(ckpt.relative_to(REPO)), "device": str(dev),
        "tier": {"closure_jets": int(n_jets), "n_closure_samples": int(k_draws),
                 "mbr_n_candidates": n_cand, **{k: v for k, v in emd_kw.items()
                                                if k != "backend"}},
        "guards": guards,
        "data": {"path": str(test_file), "fingerprint": dm.fingerprint},
        "seconds": round(time.time() - t0, 1),
        "per_jet": rows,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _col(rows, key):
    return np.array([r.get(key, np.nan) for r in rows if r.get("kept")], dtype=float)


def _paired(rows, a, b):
    d = _col(rows, a) - _col(rows, b)
    return bca_bootstrap(d[np.isfinite(d)])


def analyse(arms: dict) -> dict:
    out = {"arms": {}, "b1": {}, "b2": {}, "b3": {}}
    for name, rec in arms.items():
        rows = [r for r in rec["per_jet"] if r.get("kept")]
        block = {
            "n": len(rows),
            "absolute": {k: float(np.nanmean(_col(rows, k))) for k in (
                "dlnz_cells_2d", "dlnz_cont_2d", "dlnz_cont_3d",
                "dlnz_pool_medoid", "dlnz_free_median",
                "dlund_cells_2d", "dlund_cont_2d", "dlund_cont_3d")},
            # B1 / B2 — the two paired deltas the verdict reads
            "d_lnz_3d_vs_cells": _paired(rows, "dlnz_cont_3d", "dlnz_cells_2d"),
            "d_lnz_2d_vs_cells": _paired(rows, "dlnz_cont_2d", "dlnz_cells_2d"),
            "d_lund_3d_vs_cells": _paired(rows, "dlund_cont_3d", "dlund_cells_2d"),
            "d_lund_2d_vs_cells": _paired(rows, "dlund_cont_2d", "dlund_cells_2d"),
            # the ceilings, as paired deltas against the fielded selection
            "d_lnz_poolmedoid_vs_cells": _paired(rows, "dlnz_pool_medoid", "dlnz_cells_2d"),
            "d_lnz_freemedian_vs_cells": _paired(rows, "dlnz_free_median", "dlnz_cells_2d"),
            "winner_moved_cont3d": float(np.mean([r["winner_moved_cont3d"] for r in rows])),
            "winner_moved_cont2d": float(np.mean([r["winner_moved_cont2d"] for r in rows])),
            "leading_cell_moved_cont3d": float(
                np.mean([r["leading_cell_moved_cont3d"] for r in rows])),
            "guards": rec.get("guards"),
        }
        out["arms"][name] = block

    def tally(pred, arms_subset):
        hits = [a for a in arms_subset if a in out["arms"] and pred(out["arms"][a])]
        return {"n": len(hits), "of": len([a for a in arms_subset if a in out["arms"]]),
                "arms": hits}

    spl = [a for a in SPLINE_ARMS if a in out["arms"]]
    out["b1"] = {
        "rule": f"d(dlnz) <= -{B1_MIN_GAIN} on >= 3/4 spline arms, CI excluding 0 on >= 3/4",
        "gain": tally(lambda b: b["d_lnz_3d_vs_cells"]["mean"] <= -B1_MIN_GAIN, spl),
        "significant": tally(lambda b: b["d_lnz_3d_vs_cells"]["ci95"][1] < 0, spl),
    }
    out["b1"]["pass"] = (out["b1"]["gain"]["n"] >= 3 and out["b1"]["significant"]["n"] >= 3)
    out["b2"] = {
        "rule": f"d(dlund 2-D) does NOT exceed +{B2_MAX_LOSS} with CI excluding 0 on >= 2/4",
        "violations": tally(lambda b: (b["d_lund_3d_vs_cells"]["mean"] > B2_MAX_LOSS
                                       and b["d_lund_3d_vs_cells"]["ci95"][0] > 0), spl),
    }
    out["b2"]["pass"] = out["b2"]["violations"]["n"] < 2
    out["b3"] = {
        "rule": f"winner_moved_rate >= {B3_MIN_MOVED} on >= 3/4",
        "moved": tally(lambda b: b["winner_moved_cont3d"] >= B3_MIN_MOVED, spl),
    }
    out["b3"]["pass"] = out["b3"]["moved"]["n"] >= 3
    out["verdict"] = ("BUILD" if (out["b1"]["pass"] and out["b2"]["pass"] and out["b3"]["pass"])
                      else "DON'T BUILD")
    return out


def _ci(s):
    return f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def print_report(res: dict) -> None:
    print("\n" + "=" * 100)
    print("ABSOLUTE |d ln z| of the selected tree's leading emission, and the two ceilings")
    print(f"    {'arm':>20} {'n':>5} {'cells-2D':>10} {'cont-2D':>10} {'cont-3D':>10}"
          f" {'pool-medoid':>12} {'free-median':>12}")
    for a, b in res["arms"].items():
        x = b["absolute"]
        print(f"    {a:>20} {b['n']:>5} {x['dlnz_cells_2d']:>10.4f} {x['dlnz_cont_2d']:>10.4f}"
              f" {x['dlnz_cont_3d']:>10.4f} {x['dlnz_pool_medoid']:>12.4f}"
              f" {x['dlnz_free_median']:>12.4f}")

    print("\nB1 — does a `+lnz` ground metric recover it?  d(dlnz) = cont-3D - cells-2D,"
          " paired BCa 95%")
    print(f"    {'arm':>20} {'cont-3D - cells':>26} {'cont-2D - cells (attribution)':>32}")
    for a, b in res["arms"].items():
        print(f"    {a:>20} {_ci(b['d_lnz_3d_vs_cells']):>26}"
              f" {_ci(b['d_lnz_2d_vs_cells']):>32}")

    print("\n    ...against what a pool-restricted rule COULD reach on the ln z axis alone:")
    print(f"    {'arm':>20} {'pool-medoid - cells':>26} {'free-median - cells':>32}")
    for a, b in res["arms"].items():
        print(f"    {a:>20} {_ci(b['d_lnz_poolmedoid_vs_cells']):>26}"
              f" {_ci(b['d_lnz_freemedian_vs_cells']):>32}")

    print("\nB2 — is the (u, v) half bought with ln z?  d(dlund 2-D) = cont-3D - cells-2D")
    print(f"    {'arm':>20} {'cont-3D - cells':>26} {'cont-2D - cells':>32}")
    for a, b in res["arms"].items():
        print(f"    {a:>20} {_ci(b['d_lund_3d_vs_cells']):>26}"
              f" {_ci(b['d_lund_2d_vs_cells']):>32}")

    print("\nB3 — can the selection move at all?")
    print(f"    {'arm':>20} {'winner moved (3D)':>19} {'winner moved (2D)':>19}"
          f" {'leading cell moved':>20}")
    for a, b in res["arms"].items():
        print(f"    {a:>20} {b['winner_moved_cont3d']:>18.1%} {b['winner_moved_cont2d']:>18.1%}"
              f" {b['leading_cell_moved_cont3d']:>19.1%}")

    g = next(iter(res["arms"].values()))["guards"] if res["arms"] else None
    if g:
        print("\nguards — the EMD is a metric only when R >= half the ground diameter:")
        for c, e in g.items():
            print(f"    {c:>12}  diameter {e['ground_diameter']:.4f}"
                  f"   KMT bound {e['kmt_bound']:.4f}   R = {e['R']:.4f}"
                  f"   {'OK' if e['ok'] else 'FAIL'}")

    print("\n" + "=" * 100)
    for k in ("b1", "b2", "b3"):
        blk = res[k]
        detail = "  ".join(f"{n}: {v['n']}/{v['of']}" for n, v in blk.items()
                           if isinstance(v, dict) and "of" in v)
        print(f"{k.upper()}  {'PASS' if blk['pass'] else 'FAIL'}   {blk['rule']}\n"
              f"      {detail}")
    print(f"\nVERDICT: {res['verdict']}   (docs/PLAN_z_aware.md §12.2, fixed before the run)")
    print("=" * 100)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--test-file", default=DEFAULT_TEST)
    ap.add_argument("--n-jets", type=int, default=1000)
    ap.add_argument("--k-draws", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)

    if args.analyze:
        out_dir = Path(args.analyze)
    else:
        if args.fast:
            args.n_jets, args.k_draws = 12, 16
            args.only = args.only or "spline_s0"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else REPO / "runs" / "zaware_sel" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARM_ROOT)
        print(f"[sel] {len(names)} arm(s) -> {out_dir}  ({args.n_jets} jets, K={args.k_draws})")
        for name in names:
            rec = run_arm(name, test_file=args.test_file, n_jets=args.n_jets,
                          k_draws=args.k_draws, device=args.device)
            (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
            kept = [r for r in rec["per_jet"] if r.get("kept")]
            print(f"[sel] {name:>20}  n={len(kept):>4}  "
                  f"dlnz cells={np.nanmean(_col(kept, 'dlnz_cells_2d')):.4f} "
                  f"cont3d={np.nanmean(_col(kept, 'dlnz_cont_3d')):.4f}  ({rec['seconds']:.0f}s)")

    arms = {n: json.loads((out_dir / f"{n}.json").read_text())
            for n in ARM_ROOT if (out_dir / f"{n}.json").is_file()}
    if not arms:
        print(f"no arm JSON under {out_dir}", file=sys.stderr)
        return 2
    res = analyse(arms)
    res["out_dir"] = str(out_dir)
    print_report(res)
    (out_dir / "ceiling.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[sel] wrote {out_dir / 'ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
