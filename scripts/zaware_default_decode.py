"""§14 — should `+lnz` become the DEFAULT decode? docs/PLAN_z_aware.md §14, PLAN_next_steps A4.

§13 answered *build it* and A1 built it. This answers the separate question §13.4 clause 3
left open: does it go **on by default**, or keep shipping the way `mbr_n` and
`dv_head="spline"` ship — measured, available, documented, off.

**D1/D2/D3 and the three-way verdict are §14.3 and were committed before this file
existed** (`fa45c9f`). Nothing here is tuned against anything it produces.

Two things make it a test rather than a ratification of §13.3's post-hoc +0.0042, and §14.1
says both out loud:

  * a **disjoint population** — jets `[1000, 2000)` of a 97 018-jet file. §11, §12 and §13
    all scored `[0, 1000)`;
  * the **fielded code path** — `run_closure` -> `map_or_mbr` -> `mbr_select` ->
    `describe_cells`, which did not exist when §13 ran (§13 scored the coordinate table
    directly, inside `zaware_selection_ceiling.py`).

Per arm, ONE pass: seed -> draws -> coordinates -> **two `run_closure` calls off
byte-identical draws AND coordinates**, differing only in `(mbr_cloud_source, mbr_coords)`.
That makes the comparison paired WITHIN an arm, which is strictly stronger than §11's
across-arm pairing: same model, same jets, same pool, only the ground metric differs.

Run:
    python scripts/zaware_default_decode.py --fast                 # smoke, 12 jets
    python scripts/zaware_default_decode.py --only spline_s0       # one arm
    python scripts/zaware_default_decode.py                        # the measurement
    python scripts/zaware_default_decode.py --analyze runs/zaware_default/<stamp>

Output: the printed tables plus `runs/zaware_default/<stamp>/{<arm>.json, default.json}`.
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

from mbr_zaware_ab import (  # noqa: E402  -- the WP-0 runner owns the shared machinery
    ARM_ROOT,
    DECODE_TIER,
    DEFAULT_TEST,
    SPLINE_ARMS_ALL,
    bca_bootstrap,
    resolve_arm,
)

# §14.2, fixed before any arm ran. Repeated here so the printer cannot drift from the doc.
JET_OFFSET = 1000        # the DISJOINT slice: §11/§12/§13 all scored [0, 1000)
N_JETS = 1000
K_DRAWS = 200
D1A_MAX_UPPER = 0.010    # §11.1's measured pooled resolution of this exact ruler
D1B_MAX_ARMS = 2         # §12.2's B2 rate (25%) on twice the arms
D2_MIN_GAIN = 0.020      # §12.2's B1 bar, unchanged
D2_MIN_ARMS = 6
D3_MAX_MULT_BIAS = 0.05  # emissions
D3_MAX_P_EMPTY = 0.02
D3_MIN_ARMS = 7

# The two decodes. Everything else is the fielded tier (`DECODE_TIER`).
SELECTIONS = {
    "fielded": {"mbr_cloud_source": "cells", "mbr_coords": "lnDR_lnkt"},
    "lnz": {"mbr_cloud_source": "coords", "mbr_coords": "+lnz"},
}
# Scored per arm. The first is D1's primary read; `dlund_identity` and
# `dlund_posterior_medoid` are CONTROLS — both are built from the same draws with no EMD,
# so both must be exactly 0 or the two sides did not share their pool.
SERIES = ("dlund_mbr", "dlnz_mbr", "dlund_mbr_cont", "dlund3_mbr_cont",
          "dlund_identity", "dlund_posterior_medoid")
SCALARS = ("mult_bias_mbr", "p_empty_pred", "dlund_mbr", "dlnz_mbr", "n_kept_leading")


# ---------------------------------------------------------------------------
def run_arm(arm: str, *, test_file: str, offset: int, n_jets: int, k_draws: int,
            device: str) -> dict:
    """One pass: draws once, coordinates once, two `run_closure` calls off both."""
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params, load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.closure import run_closure
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.inference.clusters import ground_diameter
    from h2p_rsd_junipr.inference.mbr import coords_for_draws
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    ckpt, _committed = resolve_arm(arm)
    seed_everything(int(load_config([]).trainer.seed))
    dev = torch.device(device)
    info = load_for_inference(str(ckpt), map_location=dev)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(dev)
    model.load_state_dict(info["model_state"])
    cfg.data.path = str(test_file)
    dm = LundDataModule(cfg, geom).setup()
    # THE DISJOINT SLICE. Everything §11/§12/§13 published lives on `[0, offset)`, so no
    # threshold in §14.3 can have been fitted to a jet scored here.
    dm.train_jets = []
    dm.val_jets = dm.jets[int(offset):int(offset) + int(n_jets)]
    _, val_ds = dm.datasets()

    dec = dict(decode_params(cfg))
    dec.update(DECODE_TIER)
    model.length_temperature = float(dec["length_temperature"])
    model.length_tilt = float(dec["length_tilt"])
    model.continue_temperature = float(dec["continue_temperature"])
    model.kappa_min_mode = float(dec["kappa_min_mode"])
    model.eval()

    guards = {c: {"ground_diameter": ground_diameter(geom, c),
                  "kmt_bound": ground_diameter(geom, c) / 2.0,
                  "R": float(dec["mbr_R"]),
                  "ok": float(dec["mbr_R"]) >= ground_diameter(geom, c) / 2.0}
              for c in ("lnDR_lnkt", "+lnz")}
    if not all(g["ok"] for g in guards.values()):
        raise ValueError(f"R={dec['mbr_R']} fails the KMT bound at one gdim: {guards}")

    # --- ONE sampling pass, shared by BOTH decodes -----------------------------------
    t0 = time.time()
    n_scored = min(int(n_jets), len(val_ds))
    draws_by_jet, coords_by_jet = [], []
    with torch.inference_mode():
        for i in range(n_scored):
            item = val_ds[i]
            xf = item["xf"].unsqueeze(0).to(dev)
            nx = torch.tensor([item["nx"]], device=dev)
            d = model.sample_batch(xf, nx, int(k_draws))
            draws_by_jet.append(d)
            coords_by_jet.append(coords_for_draws(model, xf, nx, d))
    t_sample = time.time() - t0

    out = {
        "arm": arm, "checkpoint": str(ckpt.relative_to(REPO)), "device": str(dev),
        "tier": {"jet_offset": int(offset), "closure_jets": int(n_scored),
                 "n_closure_samples": int(k_draws), **DECODE_TIER},
        "guards": guards,
        "data": {"path": str(test_file), "fingerprint": dm.fingerprint,
                 "n_eval_jets": len(dm.jets)},
        "seconds_sampling": round(t_sample, 1),
        "closure": {},
    }
    for name, over in SELECTIONS.items():
        t1 = time.time()
        m = run_closure(model, val_ds, dm.val_jets, geom, dev, K=int(k_draws),
                        n_closure=n_scored, decode={**dec, **over}, continuous=True,
                        per_jet=True, verbose=False, draws_by_jet=draws_by_jet,
                        coords_by_jet=coords_by_jet)
        m["seconds"] = round(time.time() - t1, 1)
        out["closure"][name] = m
        print(f"[a4] {arm:>20} {name:>8}  dlund_mbr = {m.get('dlund_mbr', float('nan')):.4f}"
              f"  dlnz_mbr = {m.get('dlnz_mbr', float('nan')):.4f}  ({m['seconds']:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# Analysis — every delta is WITHIN an arm, on byte-identical draws and coordinates
# ---------------------------------------------------------------------------
def _by_jet(rows, key) -> dict:
    return {int(r["jet"]): float(r.get(key, np.nan)) for r in rows}


def _paired(rec, key) -> dict:
    a = _by_jet(rec["closure"]["lnz"]["per_jet"], key)
    b = _by_jet(rec["closure"]["fielded"]["per_jet"], key)
    d = np.array([a[j] - b[j] for j in sorted(set(a) & set(b))], dtype=float)
    st = bca_bootstrap(d[np.isfinite(d)])
    st["excludes_0"] = bool(np.isfinite(st["ci95"][0])
                            and (st["ci95"][0] > 0 or st["ci95"][1] < 0))
    st["sig_positive"] = bool(st["excludes_0"] and st["ci95"][0] > 0)
    st["sig_negative"] = bool(st["excludes_0"] and st["ci95"][1] < 0)
    return st, d[np.isfinite(d)]


def _moved_rate(rec, key) -> float:
    a = {int(r["jet"]): r.get(key) for r in rec["closure"]["lnz"]["per_jet"]}
    b = {int(r["jet"]): r.get(key) for r in rec["closure"]["fielded"]["per_jet"]}
    keys = [j for j in sorted(set(a) & set(b)) if a[j] is not None and b[j] is not None]
    return float(np.mean([a[j] != b[j] for j in keys])) if keys else float("nan")


def analyse(arms: dict) -> dict:
    per_arm, pooled = {}, {k: [] for k in SERIES}
    for name, rec in arms.items():
        block = {"series": {}, "scalars": {}}
        for k in SERIES:
            st, d = _paired(rec, k)
            block["series"][k] = st
            pooled[k].append(d)
        for k in SCALARS:
            f = rec["closure"]["fielded"].get(k)
            z = rec["closure"]["lnz"].get(k)
            block["scalars"][k] = {"fielded": f, "lnz": z,
                                   "delta": (float(z) - float(f))
                                   if (f is not None and z is not None) else None}
        block["leading_cell_moved_rate"] = _moved_rate(rec, "mbr_leading_cell")
        block["n_hat_moved_rate"] = _moved_rate(rec, "n_hat")
        block["guards"] = rec.get("guards")
        per_arm[name] = block

    out = {"arms": per_arm,
           "pooled": {k: bca_bootstrap(np.concatenate(v) if v else np.zeros(0))
                      for k, v in pooled.items()}}
    names = list(per_arm)

    # --- D1: the PRIMARY read, `dlund_mbr` -------------------------------------------
    p = out["pooled"]["dlund_mbr"]
    sig_up = [a for a in names if per_arm[a]["series"]["dlund_mbr"]["sig_positive"]]
    out["d1"] = {
        "rule": (f"pooled 95% CI upper bound < +{D1A_MAX_UPPER} (§11.1's measured "
                 f"resolution of this ruler) AND CI excludes 0 upward on "
                 f"<= {D1B_MAX_ARMS} of 8"),
        "pooled_mean": p["mean"], "pooled_ci95": p["ci95"], "pooled_n": p["n"],
        "d1a_pass": bool(np.isfinite(p["ci95"][1]) and p["ci95"][1] < D1A_MAX_UPPER),
        "n_sig_positive": len(sig_up), "arms_sig_positive": sig_up,
        "d1b_pass": bool(len(sig_up) <= D1B_MAX_ARMS),
    }
    out["d1"]["pass"] = bool(out["d1"]["d1a_pass"] and out["d1"]["d1b_pass"])

    # --- D2: the gain has to survive the fielded pipeline ----------------------------
    gain = [a for a in names
            if per_arm[a]["series"]["dlnz_mbr"]["mean"] <= -D2_MIN_GAIN
            and per_arm[a]["series"]["dlnz_mbr"]["sig_negative"]]
    out["d2"] = {
        "rule": (f"d(dlnz_mbr) <= -{D2_MIN_GAIN} with the CI excluding 0 on "
                 f">= {D2_MIN_ARMS} of 8 (§12.2's B1 bar)"),
        "n": len(gain), "arms": gain,
        "pooled_mean": out["pooled"]["dlnz_mbr"]["mean"],
        "pooled_ci95": out["pooled"]["dlnz_mbr"]["ci95"],
        "pass": bool(len(gain) >= D2_MIN_ARMS),
    }

    # --- D3: nothing else on the fielded row moves -----------------------------------
    ok = []
    for a in names:
        s = per_arm[a]["scalars"]
        mb, pe = s["mult_bias_mbr"]["delta"], s["p_empty_pred"]["delta"]
        ok.append(mb is not None and pe is not None
                  and abs(mb) < D3_MAX_MULT_BIAS and abs(pe) < D3_MAX_P_EMPTY)
    ctrl = {k: {"max_abs": float(np.nanmax(np.abs([per_arm[a]["series"][k]["mean"]
                                                   for a in names]))),
                "exact": bool(all(per_arm[a]["series"][k]["mean"] == 0.0 for a in names))}
            for k in ("dlund_identity", "dlund_posterior_medoid")}
    out["d3"] = {
        "rule": (f"|d mult_bias_mbr| < {D3_MAX_MULT_BIAS} and |d p_empty_pred| < "
                 f"{D3_MAX_P_EMPTY} on >= {D3_MIN_ARMS} of 8; the two no-EMD controls "
                 f"exactly 0"),
        "n_ok": int(sum(ok)), "controls": ctrl,
        "pass": bool(sum(ok) >= D3_MIN_ARMS and all(c["exact"] for c in ctrl.values())),
    }

    # --- the three-way verdict, §14.3 -------------------------------------------------
    # D1b/D2/D3 all count arms out of EIGHT, so a partial set cannot satisfy them and would
    # otherwise report a verdict it has not measured. NOT SCORED is the honest answer.
    out["n_arms"] = len(names)
    out["complete"] = bool(len(names) == len(ARM_ROOT))
    if not out["complete"]:
        out["verdict"] = "NOT SCORED"
        out["not_scored_reason"] = (
            f"{len(names)} of {len(ARM_ROOT)} arms present; D1b/D2/D3 are counts out of "
            f"eight and a partial set cannot meet them")
        return out
    if not out["d2"]["pass"]:
        verdict = "RECONSIDER-THE-BUILD"
    elif not out["d1"]["pass"]:
        verdict = "AVAILABLE-NOT-DEFAULT"
    elif not out["d3"]["pass"]:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "DEFAULT"
    out["verdict"] = verdict
    out["spline_arms"] = [a for a in names if a in SPLINE_ARMS_ALL]
    return out


def _ci(s):
    return f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def print_report(res: dict) -> None:
    print("\n" + "=" * 104)
    print("§14 — is `+lnz` fit to be the DEFAULT decode?  delta = lnz - fielded, per jet, "
          "paired WITHIN each arm")
    for key, title in (
        ("dlund_mbr", "D1 PRIMARY — the fielded cell-centre headline"),
        ("dlnz_mbr", "D2 — the gain: |d ln z| of the selected tree's leading emission"),
        ("dlund_mbr_cont", "context — the winner's own (u, v), off the grid (§12.2's B2 ruler)"),
        ("dlund3_mbr_cont", "context — the same emission with ln z restored"),
        ("dlund_identity", "CONTROL — model-independent; must be exactly 0"),
        ("dlund_posterior_medoid", "CONTROL — same draws, no EMD; must be exactly 0"),
    ):
        print(f"\n{title}   [{key}]")
        print(f"    {'arm':>20} {'n':>5} {'mean':>9} {'95% CI':>24}   verdict")
        for a, b in res["arms"].items():
            s = b["series"][key]
            tag = ("worse (CI > 0)" if s["sig_positive"] else
                   "better (CI < 0)" if s["sig_negative"] else "straddles 0")
            print(f"    {a:>20} {s['n']:>5} {s['mean']:>+9.4f}"
                  f" [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]   {tag}")
        p = res["pooled"][key]
        print(f"    {'pooled':>20} {p['n']:>5} {p['mean']:>+9.4f}"
              f" [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")

    print("\nD3 — the rest of the fielded row, and the guards")
    print(f"    {'arm':>20} {'d mult_bias':>12} {'d p_empty':>10} {'lead cell moved':>16}"
          f" {'n_hat moved':>12}")
    for a, b in res["arms"].items():
        s = b["scalars"]
        mb = s["mult_bias_mbr"]["delta"]
        pe = s["p_empty_pred"]["delta"]
        print(f"    {a:>20} {mb:>+12.4f} {pe:>+10.4f} {b['leading_cell_moved_rate']:>15.1%}"
              f" {b['n_hat_moved_rate']:>11.1%}")

    g = next(iter(res["arms"].values()))["guards"]
    if g:
        print("\n    guards — the EMD is a metric only when R >= half the ground diameter:")
        for c, e in g.items():
            print(f"      {c:>12}  diameter {e['ground_diameter']:.4f}"
                  f"   KMT bound {e['kmt_bound']:.4f}   R = {e['R']:.4f}"
                  f"   {'OK' if e['ok'] else 'FAIL'}")

    print("\n" + "=" * 104)
    for k in ("d1", "d2", "d3"):
        b = res[k]
        print(f"{k.upper()}  {'PASS' if b['pass'] else 'FAIL'}   {b['rule']}")
        if k == "d1":
            print(f"      pooled {b['pooled_mean']:+.4f} "
                  f"[{b['pooled_ci95'][0]:+.4f}, {b['pooled_ci95'][1]:+.4f}] "
                  f"(n = {b['pooled_n']})  ->  D1a "
                  f"{'PASS' if b['d1a_pass'] else 'FAIL'};   CI excludes 0 upward on "
                  f"{b['n_sig_positive']}/8  ->  D1b "
                  f"{'PASS' if b['d1b_pass'] else 'FAIL'}")
        if k == "d2":
            print(f"      {b['n']}/8 arms clear the bar significantly;  pooled "
                  f"{b['pooled_mean']:+.4f} "
                  f"[{b['pooled_ci95'][0]:+.4f}, {b['pooled_ci95'][1]:+.4f}]")
        if k == "d3":
            print(f"      {b['n_ok']}/8 arms inside both bounds;  controls "
                  + ", ".join(f"{n} max|d| = {c['max_abs']:.2e}"
                              for n, c in b["controls"].items()))
    print(f"\nVERDICT: {res['verdict']}"
          "   (docs/PLAN_z_aware.md §14.3, fixed before the run in fa45c9f)")
    if not res.get("complete", True):
        print(f"         {res['not_scored_reason']}")
    print("=" * 104)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--test-file", default=DEFAULT_TEST)
    ap.add_argument("--offset", type=int, default=JET_OFFSET)
    ap.add_argument("--n-jets", type=int, default=N_JETS)
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default and the standing rule); cuda is a different RNG "
                         "stream and different float kernels")
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
        out_dir = Path(args.out) if args.out else REPO / "runs" / "zaware_default" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARM_ROOT)
        print(f"[a4] {len(names)} arm(s) -> {out_dir}  (jets "
              f"[{args.offset}, {args.offset + args.n_jets}), K = {args.k_draws})")
        for name in names:
            rec = run_arm(name, test_file=args.test_file, offset=args.offset,
                          n_jets=args.n_jets, k_draws=args.k_draws, device=args.device)
            (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")

    arms = {n: json.loads((out_dir / f"{n}.json").read_text())
            for n in ARM_ROOT if (out_dir / f"{n}.json").is_file()}
    if not arms:
        print(f"no arm JSON under {out_dir}", file=sys.stderr)
        return 2
    res = analyse(arms)
    res["out_dir"] = str(out_dir)
    res["arms_present"] = sorted(arms)
    print_report(res)
    (out_dir / "default.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[a4] wrote {out_dir / 'default.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
