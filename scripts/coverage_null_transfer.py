"""B3 — is `coverage_68`'s null family-independent? docs/PLAN_next_steps.md B3, SUMMARY §4.1(3).

`PLAN_StratifiedMBR.md` §1c measured the statistic's own null — model as truth, the same
`K`-draw HPD — on the **continue/stop** arm `v1_contstop_s0`:

    coverage_68   0.546 [0.502, 0.589]   on   502 jets
    its own null  0.553 [0.543, 0.563]   on 8 841 pseudo-truths

so a *perfect* model scores 0.553 at `K = 200`, not 0.68, and the observed deficit is the
estimator rather than the posterior. That became a stop-sign — *"reading `coverage_68`
against 0.68"* — on the strength of **one arm of one family**.

This transfers it to the **explicit-`q(N|x)`** family, which is the interesting direction:
that family is the one whose joint posterior genuinely **is** too narrow (all six arms fail
TARP/G7, `PROD_TEST_v1_RESULTS.md`), so if the null were a property of the model rather
than of the estimator, this is where it should move. It should not: the HPD-68 built from
`K` draws cannot contain a cell of probability below `1/K` whatever generated the draws.

**Scope note, stated because it is larger than the plan asked for.** SUMMARY §4.1(3) says
"one `eval` run". This runs **three** `v1_base` seeds, so the transfer carries a seed band
instead of a single number, plus a fourth arm — `v1_contstop_s0` — as a **positive
control**: does this runner reproduce §1c's 0.546 / 0.553? A null quoted from a runner that
was never checked against the number it is extending is not a transfer.

It does **not** write beside the checkpoint. `h2p-rsd-junipr eval` would overwrite each
arm's committed `eval_metrics.json`, and §1c had to preserve its own artifact by hand for
exactly that reason.

Run:
    python scripts/coverage_null_transfer.py --fast        # smoke: 16 jets, K=32
    python scripts/coverage_null_transfer.py               # 600 jets, K=200, 4 arms
    python scripts/coverage_null_transfer.py --analyze runs/coverage_null/<stamp>
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

# §1c's tier, quoted rather than re-chosen. Changing any of it means this is not a
# transfer of that number.
TIER = dict(n_jets=600, k_draws=200, null_reps=20)
DECODE = {"point_estimator": "mbr", "mbr_backend": "pot", "mbr_n_candidates": 64,
          "min_emissions": 0}
DEFAULT_TEST = "data/jet_aux_asym_test.root"
# The reference §1c reported, so the control row is scored against the record rather than
# against a number typed into this file from memory.
REFERENCE = {"arm": "v1_contstop_s0", "coverage_68": 0.546, "coverage_68_null": 0.553,
             "null_ci": [0.543, 0.563], "source": "docs/PLAN_StratifiedMBR.md §1c WP-4"}
ARMS = {
    # the TRANSFER: explicit q(N|x), the family whose joint posterior IS too narrow
    "v1_base_s0": ("runs/prod_test_v1", "explicit q(N|x)"),
    "v1_base_s1": ("runs/prod_test_v1", "explicit q(N|x)"),
    "v1_base_s2": ("runs/prod_test_v1", "explicit q(N|x)"),
    # the POSITIVE CONTROL: the arm §1c measured
    "v1_contstop_s0": ("runs/prod_test_v1", "continue/stop (the §1c arm)"),
}
G_REPRO_TOL = 0.02   # absolute, on a proportion whose own Wilson half-width is ~0.02


def run_arm(arm: str, *, test_file: str, n_jets: int, k_draws: int, null_reps: int,
            device: str) -> dict:
    """`run_closure` then `run_calibration`, in cli.py's order.

    The order matters and is not cosmetic: `run_closure` consumes the global RNG stream
    before `run_calibration` starts, so a script that skipped it would hand the calibration
    block different draws than a real `eval` does. Same reason WP-0's runner mirrors the
    call order (`PLAN_z_aware.md` §11.1's G-repro landed at 0.00% because of it)."""
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params, load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.calibration import run_calibration
    from h2p_rsd_junipr.eval.closure import run_closure
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.inference.mbr import mbr_kwargs_from_decode
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    root = REPO / ARMS[arm][0] / arm
    hits = sorted(root.glob("*/best.ckpt"))
    if not hits:
        raise FileNotFoundError(f"no best.ckpt under {root}")
    ckpt = hits[-1]

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

    dec = dict(decode_params(cfg))
    dec.update(DECODE)
    model.length_temperature = float(dec["length_temperature"])
    model.length_tilt = float(dec["length_tilt"])
    model.continue_temperature = float(dec["continue_temperature"])
    model.kappa_min_mode = float(dec["kappa_min_mode"])
    model.eval()

    t0 = time.time()
    closure = run_closure(model, val_ds, dm.val_jets, geom, dev, K=int(k_draws),
                          n_closure=int(n_jets), decode=dec, verbose=False)
    calib = run_calibration(model, val_ds, geom, dev, K=int(k_draws), n_jets=int(n_jets),
                            verbose=False, mbr_kwargs=mbr_kwargs_from_decode(dec),
                            coverage_null_reps=int(null_reps))
    return {
        "arm": arm, "family": ARMS[arm][1],
        "checkpoint": str(ckpt.relative_to(REPO)), "device": str(dev),
        "has_multiplicity_head": bool(hasattr(model, "n_head")),
        "tier": {"closure_jets": int(n_jets), "n_closure_samples": int(k_draws),
                 "coverage_null_reps": int(null_reps), **DECODE},
        "data": {"path": str(test_file), "fingerprint": dm.fingerprint},
        "seconds": round(time.time() - t0, 1),
        "coverage": {k: calib.get(k) for k in
                     ("coverage_68", "coverage_68_ci", "n_coverage",
                      "coverage_68_null", "coverage_68_null_ci", "n_coverage_null",
                      "coverage_68_vs_null", "coverage_68_null_explains_deficit",
                      "coverage_68_vs_null_ci",
                      "coverage_68_null_explains_deficit_paired")},
        "sbc": {k: calib.get(k) for k in
                ("sbc_chi2_uniform", "sbc_chi2_dof", "sbc_chi2_crit95",
                 "sbc_rank_mean", "pit_mean")},
        "closure": {k: closure.get(k) for k in
                    ("dlund_mbr", "dlund_posterior_medoid", "coverage_68",
                     "n_kept_leading", "n_jets_scored")},
    }


# ---------------------------------------------------------------------------
def point_in_ci_false_reject_rate(n_obs: int, n_null: int, p: float, reps: int = 20_000,
                                  seed: int = 20260806) -> float:
    """How often "is the observation inside the null's interval" rejects a PERFECT model.

    Simulated, not asserted — the repo's own rule (`SUMMARY` §5, *simulate the reference,
    never assume it*), and it is the rule the test being checked here failed to follow.
    Draw `coverage_68` from `Binomial(n_obs, p)` and the null from `Binomial(n_null, p)`,
    i.e. from the SAME `p`, then apply the test. Anything above 5% is over-rejection."""
    from h2p_rsd_junipr.eval.calibration import wilson_interval

    rng = np.random.default_rng(int(seed))
    k_obs = rng.binomial(int(n_obs), float(p), size=int(reps))
    k_nul = rng.binomial(int(n_null), float(p), size=int(reps))
    miss = 0
    for a, b in zip(k_obs, k_nul):
        lo, hi = wilson_interval(int(b), int(n_null))
        if not (lo <= a / n_obs <= hi):
            miss += 1
    return float(miss) / int(reps)


def analyse(arms: dict) -> dict:
    from h2p_rsd_junipr.eval.calibration import wilson_diff_interval

    rows = []
    for name, rec in arms.items():
        c = dict(rec["coverage"])
        # Recompute the paired test here too, so an artifact produced before
        # `wilson_diff_interval` shipped is still scored the right way.
        k_obs = int(round(c["coverage_68"] * c["n_coverage"]))
        k_nul = int(round(c["coverage_68_null"] * c["n_coverage_null"]))
        lo, hi = wilson_diff_interval(k_obs, c["n_coverage"], k_nul, c["n_coverage_null"])
        c["coverage_68_vs_null_ci"] = [lo, hi]
        c["coverage_68_null_explains_deficit_paired"] = bool(lo <= 0.0 <= hi)
        rows.append({
            "arm": name, "family": rec["family"],
            "explicit_qn": bool(rec["has_multiplicity_head"]),
            **c,
        })
    expl = [r for r in rows if r["explicit_qn"]]
    ctrl = [r for r in rows if not r["explicit_qn"]]
    nulls = np.array([r["coverage_68_null"] for r in expl], dtype=float)
    n_nul = np.array([r["n_coverage_null"] for r in expl], dtype=float)

    # THE TRANSFER CLAIM, and it is about the NULL: is it ~0.553 on this family too?
    # Pooled over the seeds, then compared with the reference as a DIFFERENCE of two
    # proportions — not by asking whether a point lands in an interval.
    k_pool = float(np.sum(nulls * n_nul))
    n_pool = float(n_nul.sum())
    ref_n = 8841  # §1c's own pseudo-truth count, recorded with its number
    p_lo, p_hi = wilson_diff_interval(k_pool, n_pool,
                                      REFERENCE["coverage_68_null"] * ref_n, ref_n)
    out = {
        "rows": rows,
        "reference": REFERENCE,
        "explicit_qn": {
            "n_arms": len(expl),
            "null_mean": float(nulls.mean()) if nulls.size else float("nan"),
            "null_min": float(nulls.min()) if nulls.size else float("nan"),
            "null_max": float(nulls.max()) if nulls.size else float("nan"),
            "null_spread": float(nulls.max() - nulls.min()) if nulls.size else float("nan"),
            "null_pooled": float(k_pool / n_pool) if n_pool else float("nan"),
            "n_pooled": int(n_pool),
            "pooled_vs_reference": float(k_pool / n_pool - REFERENCE["coverage_68_null"])
            if n_pool else float("nan"),
            "pooled_vs_reference_ci": [p_lo, p_hi],
            "null_agrees_with_reference": bool(p_lo <= 0.0 <= p_hi),
            # ...and, per arm, whether its OWN deficit is explained by its OWN null.
            "deficit_explained_paired": [
                bool(r["coverage_68_null_explains_deficit_paired"]) for r in expl],
            # the strict-and-wrong scoring, kept so the difference is visible
            "deficit_explained_point_in_ci": [
                bool(r["coverage_68_null_explains_deficit"]) for r in expl],
        },
    }
    e = out["explicit_qn"]
    e["n_deficit_explained_paired"] = int(sum(e["deficit_explained_paired"]))
    e["n_deficit_explained_point_in_ci"] = int(sum(e["deficit_explained_point_in_ci"]))
    if expl:
        r0 = expl[0]
        e["point_in_ci_false_reject_rate"] = point_in_ci_false_reject_rate(
            int(r0["n_coverage"]), int(r0["n_coverage_null"]),
            float(REFERENCE["coverage_68_null"]))
    if ctrl:
        r = ctrl[0]
        out["control"] = {
            "arm": r["arm"],
            "coverage_68": r["coverage_68"], "recorded": REFERENCE["coverage_68"],
            "coverage_68_null": r["coverage_68_null"],
            "recorded_null": REFERENCE["coverage_68_null"],
            "abs_diff": abs(float(r["coverage_68_null"]) - REFERENCE["coverage_68_null"]),
            "tol": G_REPRO_TOL,
        }
        out["control"]["repro_ok"] = bool(out["control"]["abs_diff"] <= G_REPRO_TOL)
    n = len(expl)
    out["verdict"] = (
        "NOT SCORED" if not n else
        "FAMILY-INDEPENDENT" if e["null_agrees_with_reference"]
        and e["n_deficit_explained_paired"] == n
        else "TRANSFERS, WITH A RESIDUAL" if e["null_agrees_with_reference"]
        else "NOT CONFIRMED")
    return out


def print_report(res: dict) -> None:
    print("\n" + "=" * 104)
    print("B3 — `coverage_68` against its OWN null, transferred to the explicit-q(N|x) family")
    print(f"    {'arm':>20} {'family':>28} {'coverage_68':>22} {'its null':>22} "
          f"{'difference (Newcombe 95%)':>28}")
    for r in res["rows"]:
        cov = (f"{r['coverage_68']:.3f} [{r['coverage_68_ci'][0]:.3f}, "
               f"{r['coverage_68_ci'][1]:.3f}]")
        nul = (f"{r['coverage_68_null']:.3f} [{r['coverage_68_null_ci'][0]:.3f}, "
               f"{r['coverage_68_null_ci'][1]:.3f}]")
        d = r["coverage_68"] - r["coverage_68_null"]
        dd = (f"{d:+.3f} [{r['coverage_68_vs_null_ci'][0]:+.3f}, "
              f"{r['coverage_68_vs_null_ci'][1]:+.3f}]")
        tag = "ok" if r["coverage_68_null_explains_deficit_paired"] else "BELOW"
        print(f"    {r['arm']:>20} {r['family']:>28} {cov:>22} {nul:>22} "
              f"{dd:>22} {tag:>5}")
        print(f"    {'':>20} {'':>28} {'on ' + str(r['n_coverage']) + ' jets':>22} "
              f"{'on ' + str(r['n_coverage_null']) + ' pseudo-truths':>22}")
    ref = res["reference"]
    print(f"\n    reference ({ref['source']}): coverage_68 = {ref['coverage_68']:.3f}, "
          f"null = {ref['coverage_68_null']:.3f} "
          f"[{ref['null_ci'][0]:.3f}, {ref['null_ci'][1]:.3f}]")
    c = res.get("control")
    if c:
        print(f"    POSITIVE CONTROL — {c['arm']} re-measured by THIS runner: null "
              f"{c['coverage_68_null']:.4f} vs the recorded {c['recorded_null']:.3f}"
              f"   |diff| = {c['abs_diff']:.4f}   "
              f"{'OK' if c['repro_ok'] else 'FAIL'} (tol {c['tol']:.2f})")
    e = res["explicit_qn"]
    print(f"\n    THE TRANSFER — explicit-q(N|x) nulls: {e['null_min']:.4f} .. "
          f"{e['null_max']:.4f} (spread {e['null_spread']:.4f}) over {e['n_arms']} seeds;"
          f"\n    pooled {e['null_pooled']:.4f} on {e['n_pooled']:,} pseudo-truths, vs the "
          f"reference {ref['coverage_68_null']:.3f}: {e['pooled_vs_reference']:+.4f} "
          f"[{e['pooled_vs_reference_ci'][0]:+.4f}, {e['pooled_vs_reference_ci'][1]:+.4f}]"
          f"  ->  {'AGREES' if e['null_agrees_with_reference'] else 'DIFFERS'}")
    print(f"    own deficit explained (Newcombe, both errors priced) on "
          f"{e['n_deficit_explained_paired']}/{e['n_arms']}")
    if "point_in_ci_false_reject_rate" in e:
        print(f"    [the strict 'is the observation inside the null's interval' test says "
              f"{e['n_deficit_explained_point_in_ci']}/{e['n_arms']}, and it is WRONG:\n"
              f"     simulated on a PERFECT model at these sample sizes it rejects "
              f"{e['point_in_ci_false_reject_rate']:.1%} of the time, because it discards "
              f"the observation's\n     own error, which is the larger of the two]")
    print("\n" + "=" * 104)
    print(f"VERDICT: {res['verdict']}")
    print("    The HPD-68 is built from K draws and cannot contain a cell of probability")
    print("    below 1/K, whatever generated them — so the null is a property of the")
    print("    ESTIMATOR and K, not of the family. Always quote coverage_68 with its K.")
    print("=" * 104)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--test-file", default=DEFAULT_TEST)
    ap.add_argument("--n-jets", type=int, default=TIER["n_jets"])
    ap.add_argument("--k-draws", type=int, default=TIER["k_draws"])
    ap.add_argument("--null-reps", type=int, default=TIER["null_reps"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)

    if args.analyze:
        out_dir = Path(args.analyze)
    else:
        if args.fast:
            args.n_jets, args.k_draws, args.null_reps = 16, 32, 5
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else REPO / "runs" / "coverage_null" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARMS)
        print(f"[b3] {len(names)} arm(s) -> {out_dir}  ({args.n_jets} jets, "
              f"K = {args.k_draws}, {args.null_reps} pseudo-truths/jet)")
        for name in names:
            rec = run_arm(name, test_file=args.test_file, n_jets=args.n_jets,
                          k_draws=args.k_draws, null_reps=args.null_reps,
                          device=args.device)
            (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
            c = rec["coverage"]
            print(f"[b3] {name:>20}  coverage_68 = {c['coverage_68']:.4f}"
                  f"   null = {c['coverage_68_null']:.4f}"
                  f"   explained = {c['coverage_68_null_explains_deficit']}"
                  f"   ({rec['seconds']:.0f}s)")

    arms = {n: json.loads((out_dir / f"{n}.json").read_text())
            for n in ARMS if (out_dir / f"{n}.json").is_file()}
    if not arms:
        print(f"no arm JSON under {out_dir}", file=sys.stderr)
        return 2
    res = analyse(arms)
    res["out_dir"] = str(out_dir)
    print_report(res)
    (out_dir / "coverage_null.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[b3] wrote {out_dir / 'coverage_null.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
