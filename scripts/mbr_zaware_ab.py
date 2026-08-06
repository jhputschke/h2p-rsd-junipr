"""Is §2.5's `d(MBR)` regression even resolved? — docs/PLAN_z_aware.md WP-0.

`PLAN_lnz_spline_head.md` §6.2 records that the RQ-spline `ln z` head improved held-out
NLL and `pit_ks_max` on 4/4 arms while `dlund_mbr` got **worse** on 4/4 (+0.0047, +0.0113,
+0.0091, +0.0031), and offers an explanation: "the MBR metric runs on `lnDR_lnkt` and
cannot see `ln z`". `PLAN_z_aware.md` exists to test that sentence rather than assert it,
and its §3 turned up two facts that reorder the work:

  * the pairing is **exact** — `dlund_identity` is a model-independent function of the
    jets and is identical to all printed digits within every pair, so the same jets are
    scored on both sides. A per-jet PAIRED analysis is legitimate, and every number in
    §2.5 is an **unpaired mean**;
  * only `dlund_mbr` is 4/4. Every other estimator built from the SAME draws — the
    leading-cell medoid, the leading-cell mode, the continuous geometric median — is
    mixed-signed, i.e. the cell posterior did not degrade.

So the first question is not *is the regression an artifact* but **is there a regression
to explain**. That is what this script measures, and its verdict gates everything the plan
would build afterwards (WP-3's coordinate threading and WP-4's 3x3 selection grid).

    G-repro   re-measured `dlund_mbr` within 0.5% of the committed value on 8/8 arms
    G-pair    per-jet `dlund_identity` identical within each pair on 4/4
    G-exists  ESTABLISHED iff `delta > 0` with the paired 95% CI excluding 0 on >= 3/4

All three, and the effect-size note they are read against, are §4 and §5 of the plan and
were fixed **before** this file existed. If G-exists fails the verdict is
INCONCLUSIVE-BY-CONSTRUCTION: +0.005 is inside its own per-jet noise, and an explanation
is not owed for a number that is not resolved.

The `ln z`-aware ruler rows (`dlund3_mbr_cont`, `dlnz_mbr`, `dlund_mbr_cont` — WP-1) come
out of the same pass for free, so they are reported here too. **They are the C1/C2/C3
statistics of §5 measured at the FIELDED `cells-2D` selection**, which is the only
selection that exists before WP-3; C4 is a difference-in-differences across selection arms
and cannot be computed here. This script therefore never prints a CONFIRM/FALSIFY verdict
for §5 — that belongs to WP-4.

A runner, not a reader: it runs models. `scripts/lnz_spline_gates.py` stays a pure JSON
reader, and merging the two would put a multi-hour decode behind a table printer
(precedents: `scripts/n_ceiling_probe.py`, `scripts/probe_map_collapse.py`).

Run:
    python scripts/mbr_zaware_ab.py --fast                  # ~2 min smoke, 12 jets
    python scripts/mbr_zaware_ab.py                         # the measurement, 8 arms
    python scripts/mbr_zaware_ab.py --only spline_s0        # one arm (parallelise by hand)
    python scripts/mbr_zaware_ab.py --analyze runs/zaware_wp0/<stamp>   # verdict only

Output: the printed tables plus `runs/zaware_wp0/<stamp>/{<arm>.json, wp0.json}`.
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
if str(REPO / "src") not in sys.path:  # editable installs already have it; be explicit
    sys.path.insert(0, str(REPO / "src"))

# The four pairs, and there will be exactly four: `spline_s3/s4/s5` exist, `v1_base_s3/s4/s5`
# do not, so a two-sided sign test on the arms floors at p = 0.125 whatever is run. More
# power has to come from JETS, which is why the per-jet rows below are the whole point.
PAIRS = (
    ("spline_s0", "v1_base_s0"),
    ("spline_s1", "v1_base_s1"),
    ("spline_s2", "v1_base_s2"),
    ("contstop_spline_s0", "v1_contstop_s0"),
)
ARM_ROOT = {
    **{a: "runs/lnz_spline" for a, _ in PAIRS},
    **{c: "runs/prod_test_v1" for _, c in PAIRS},
}
# The four `lnz_head="spline"` arms, named once so a sibling script does not re-derive the
# split from `PAIRS` and get it subtly wrong.
SPLINE_ARMS_ALL = tuple(a for a, _ in PAIRS)
DEFAULT_TEST = "data/jet_aux_asym_test.root"

# The published decode tier (`scripts/eval_prod_test_v1.sh` pass B), quoted rather than
# re-chosen: 300 jets, K = 200, 64 candidates, `pot`, floor-free. Changing any of these
# means this is not a re-measurement of the committed number.
TIER = dict(closure_jets=300, n_closure_samples=200)
DECODE_TIER = {
    "point_estimator": "mbr", "mbr_backend": "pot", "mbr_n_candidates": 64,
    "min_emissions": 0,
}
# G-repro's tolerance, and the precedent for it: the ceiling probe's sanity row re-measured
# `d(medoid)` 2.3489 -> 2.3495 (0.03%) on the same jets with fresh draws. A statistic
# computed from K draws carries K-draw noise; 0.5% is that noise, not a fitting margin.
G_REPRO_TOL = 0.005
N_BOOT = 10_000
BOOT_SEED = 20260806

# Every per-jet series this script pairs. The first is the one under test; the rest are the
# controls that make §3's reading checkable (`dlund_identity` certifies the pairing, and the
# same-draw estimators say whether the CELL posterior moved) and the WP-1 ruler rows.
SERIES = (
    "dlund_mbr",
    "dlund_identity",
    "dlund_posterior_medoid",
    "dlund_posterior_mode",
    "dlund_mbr_cont",
    "dlund3_mbr_cont",
    "dlnz_mbr",
    "dlund_posterior_mode_cont",
    "dlund_posterior_geomedian_cont",
    "dlund3_posterior_geomedian_cont",
    "dlnz_posterior_geomedian",
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bca_bootstrap(delta, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05
                  ) -> dict:
    """Mean of a paired per-jet difference with a **BCa** 95% interval (Efron 1987).

    Paired at the JET, which is the unit that was sampled — the same 247 jets are scored
    on both sides of every pair, certified by `dlund_identity`. An unpaired interval would
    price the spread BETWEEN jets (the `dlund_*` distributions are wide and skewed) rather
    than the difference between arms, and that spread is what made §2.5's +0.005 look like
    noise beside a 0.018 seed spread.

    Bias-corrected and accelerated rather than plain percentile because the per-jet delta
    is skewed: `d` is a non-negative distance, so its difference has a long tail on the
    side where one arm lands far away. `z0` corrects the median bias, `a` the
    variance-vs-mean trend, from the jackknife. Both reduce to the percentile interval
    when the statistic is unbiased and symmetric, so nothing is lost where it did not
    matter."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    out = {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": int(n),
           "n_pos": 0, "method": "bca"}
    if n < 2:
        return out
    out["mean"] = float(d.mean())
    out["n_pos"] = int((d > 0).sum())
    rng = np.random.default_rng(int(seed))
    boot = d[rng.integers(0, n, size=(int(n_boot), n))].mean(axis=1)
    from scipy.stats import norm

    prop = float((boot < out["mean"]).mean())
    if prop <= 0.0 or prop >= 1.0:  # degenerate: every resample on one side
        out["method"] = "percentile"
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["ci95"] = [float(lo), float(hi)]
        return out
    z0 = float(norm.ppf(prop))
    # jackknife acceleration; the leave-one-out mean is closed-form, so this is O(n)
    jack = (d.sum() - d) / (n - 1)
    dev = jack.mean() - jack
    denom = 6.0 * (float((dev**2).sum()) ** 1.5)
    a = float((dev**3).sum()) / denom if denom > 0 else 0.0
    out["z0"], out["a"] = z0, a
    qs = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        adj = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        qs.append(float(np.clip(norm.cdf(adj), 1e-6, 1 - 1e-6)))
    lo, hi = np.percentile(boot, [100 * qs[0], 100 * qs[1]])
    out["ci95"] = [float(lo), float(hi)]
    return out


def sign_test_p(n_pos: int, n: int) -> float:
    """Two-sided exact sign test. On four pairs its floor is 0.125, which is why the
    per-jet rows above exist and why the arm-level count is quoted beside them, never
    alone."""
    from math import comb

    if n == 0:
        return float("nan")
    k = max(int(n_pos), n - int(n_pos))
    tail = sum(comb(n, j) for j in range(k, n + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------
def resolve_arm(arm: str) -> tuple[Path, Path | None]:
    """`(best.ckpt, eval_metrics_decode.json)` for a named arm.

    The committed decode-tier artifact is what G-repro compares against, so it is looked
    up rather than quoted: a re-measurement scored against a number typed into a plan is
    not a re-measurement."""
    root = REPO / ARM_ROOT.get(arm, "runs")
    hits = sorted((root / arm).glob("*/best.ckpt")) if (root / arm).is_dir() else []
    if not hits:
        raise FileNotFoundError(f"no best.ckpt under {root / arm} — is the arm trained?")
    ckpt = hits[-1]
    committed = ckpt.parent / "eval_metrics_decode.json"
    return ckpt, (committed if committed.is_file() else None)


def run_arm(arm: str, *, test_file: str, n_jets: int, k_draws: int, device: str) -> dict:
    """One decode pass over the published tier, with `run_closure(per_jet=True)`.

    The call order mirrors `cli.py`'s `eval` — seed, build, load, datamodule, closure — so
    the global RNG stream is consumed the same way and the re-measurement lands within
    G-repro's tolerance of the committed number rather than merely near it.

    `cpu` by default and that is a whole-grid decision, not a per-arm one: cuda is a
    different RNG stream *and* different float kernels, so a half-and-half set would be a
    silent ranking hazard (`scripts/eval_prod_test_v1.sh`'s standing rule)."""
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params, load_config
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.closure import run_closure
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    ckpt, committed_path = resolve_arm(arm)
    # `h2p-rsd-junipr eval` seeds from the CLI-composed config, not the checkpoint
    # snapshot, and no arm's eval passed `trainer.seed=` — so every arm decoded at 0.
    seed_everything(int(load_config([]).trainer.seed))
    dev = torch.device(device)
    info = load_for_inference(str(ckpt), map_location=dev)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(dev)
    model.load_state_dict(info["model_state"])
    # An explicitly named eval sample is a TEST set: report on every jet in it, exactly as
    # `cli.py` does when `data.path` is overridden. Keeping the 90/10 split would score a
    # tenth of the file — and a *different* tenth as soon as its length changed.
    cfg.data.path = str(test_file)
    dm = LundDataModule(cfg, geom).setup()
    dm.train_jets, dm.val_jets = [], dm.jets
    _, val_ds = dm.datasets()

    decode = dict(decode_params(cfg))
    decode.update(DECODE_TIER)
    model.length_temperature = float(decode["length_temperature"])
    model.length_tilt = float(decode["length_tilt"])
    model.continue_temperature = float(decode["continue_temperature"])
    model.kappa_min_mode = float(decode["kappa_min_mode"])
    model.eval()

    t0 = time.time()
    closure = run_closure(
        model, val_ds, dm.val_jets, geom, dev, K=int(k_draws), n_closure=int(n_jets),
        decode=decode, continuous=True, per_jet=True, verbose=False,
    )
    committed = None
    if committed_path is not None:
        rec = json.loads(committed_path.read_text())
        committed = {k: rec["closure"].get(k) for k in
                     ("dlund_mbr", "dlund_identity", "dlund_posterior_medoid",
                      "n_kept_leading", "n_jets_scored")}
        committed["experiment"] = rec.get("experiment")
        committed["decode"] = rec.get("decode")
    return {
        "arm": arm,
        "checkpoint": str(ckpt.relative_to(REPO)),
        "committed_artifact": (str(committed_path.relative_to(REPO))
                               if committed_path else None),
        "committed": committed,
        "device": str(dev),
        "data": {"path": str(test_file), "fingerprint": dm.fingerprint,
                 "n_eval_jets": len(dm.val_jets)},
        "tier": {"closure_jets": int(n_jets), "n_closure_samples": int(k_draws),
                 **DECODE_TIER},
        "seconds": round(time.time() - t0, 1),
        "closure": closure,
    }


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------
def _by_jet(rows, key) -> dict:
    return {int(r["jet"]): float(r.get(key, np.nan)) for r in rows}


def paired_delta(a_rows, b_rows, key) -> np.ndarray:
    """`spline - control`, jet by jet, over the jets BOTH arms scored for `key`.

    Pairing is by jet index and nothing else: the two arms are different models, so their
    posterior draws cannot be shared and the only thing that transfers is which jet is
    which. G-pair is what certifies that the index means the same jet on both sides."""
    a, b = _by_jet(a_rows, key), _by_jet(b_rows, key)
    keys = sorted(set(a) & set(b))
    d = np.array([a[j] - b[j] for j in keys], dtype=float)
    return d[np.isfinite(d)]


def g_repro(arms: dict) -> dict:
    """Did this script re-measure the committed `dlund_mbr`? 8/8 within 0.5%.

    **Scored only where the tiers match.** The committed artifact is a 300-jet decode, so
    the 1000-jet escalation scores a different population (839 kept jets, not 247) and a
    disagreement there is the population, not the pipeline — reporting it as a failure
    would be reporting the escalation as a bug. Comparable rows are scored; the rest are
    marked `n/a` and the gate says how many it could score."""
    rows = []
    for name, rec in arms.items():
        got = float(rec["closure"].get("dlund_mbr", np.nan))
        com = rec.get("committed") or {}
        ref = com.get("dlund_mbr")
        ref_tier = ((com.get("experiment") or {}).get("closure_jets")
                    if com.get("experiment") else None)
        same_tier = ref_tier is not None and int(ref_tier) == int(rec["tier"]["closure_jets"])
        rel = (abs(got - float(ref)) / abs(float(ref))
               if (ref and same_tier) else float("nan"))
        rows.append({"arm": name, "recorded": ref, "remeasured": got, "rel": rel,
                     "same_tier": bool(same_tier),
                     "recorded_tier": ref_tier, "tier": rec["tier"]["closure_jets"],
                     "ok": bool(np.isfinite(rel) and rel <= G_REPRO_TOL),
                     "n_kept_recorded": com.get("n_kept_leading"),
                     "n_kept_remeasured": rec["closure"].get("n_kept_leading")})
    scored = [r for r in rows if r["same_tier"]]
    return {"rows": rows, "n_ok": sum(r["ok"] for r in scored), "n": len(scored),
            "n_arms": len(rows), "scored": bool(scored),
            "pass": bool(scored) and all(r["ok"] for r in scored), "tol": G_REPRO_TOL}


def g_pair(arms: dict) -> dict:
    """Is the pairing exact? `dlund_identity` is a model-independent function of the jets,
    so it must agree within a pair to floating-point — if it does not, the two arms are
    not scoring the same jets and no paired statistic below means anything."""
    rows = []
    for spl, ctl in PAIRS:
        if spl not in arms or ctl not in arms:
            continue
        a = _by_jet(arms[spl]["closure"]["per_jet"], "dlund_identity")
        b = _by_jet(arms[ctl]["closure"]["per_jet"], "dlund_identity")
        keys = sorted(set(a) & set(b))
        d = np.array([a[j] - b[j] for j in keys], dtype=float)
        d = d[np.isfinite(d)]
        worst = float(np.abs(d).max()) if d.size else float("nan")
        rows.append({"pair": f"{spl} - {ctl}", "n": int(d.size), "max_abs_diff": worst,
                     "ok": bool(d.size and worst <= 1e-9)})
    return {"rows": rows, "n_ok": sum(r["ok"] for r in rows), "n": len(rows),
            "pass": bool(rows) and all(r["ok"] for r in rows)}


def paired_table(arms: dict, key: str) -> dict:
    """Per-pair paired BCa bootstrap of one series, plus the arm-level sign test."""
    rows = []
    pooled = []
    for spl, ctl in PAIRS:
        if spl not in arms or ctl not in arms:
            continue
        d = paired_delta(arms[spl]["closure"]["per_jet"],
                         arms[ctl]["closure"]["per_jet"], key)
        st = bca_bootstrap(d)
        st["pair"] = f"{spl} - {ctl}"
        st["excludes_0"] = bool(np.isfinite(st["ci95"][0])
                                and (st["ci95"][0] > 0 or st["ci95"][1] < 0))
        st["sig_positive"] = bool(st["excludes_0"] and st["ci95"][0] > 0)
        st["sig_negative"] = bool(st["excludes_0"] and st["ci95"][1] < 0)
        rows.append(st)
        pooled.append(d)
    n_pos_arms = sum(1 for r in rows if r["mean"] > 0)
    return {
        "key": key, "rows": rows,
        "n_pairs": len(rows),
        "n_arms_positive": n_pos_arms,
        "sign_test_p": sign_test_p(n_pos_arms, len(rows)),
        "n_sig_positive": sum(r["sig_positive"] for r in rows),
        "n_sig_negative": sum(r["sig_negative"] for r in rows),
        # Pooling concatenates the per-jet deltas of all four pairs. It is reported, never
        # used as the verdict: the four pairs are four different model comparisons, so the
        # pool is a mixture and its CI answers "is the average intervention effect
        # nonzero", which is a weaker question than the pre-registered per-pair rule.
        "pooled": bca_bootstrap(np.concatenate(pooled) if pooled else np.zeros(0)),
    }


def analyse(arms: dict) -> dict:
    """G-repro, G-pair, G-exists and the free WP-1 ruler rows."""
    tables = {k: paired_table(arms, k) for k in SERIES}
    mbr = tables["dlund_mbr"]
    repro, pair = g_repro(arms), g_pair(arms)
    established = bool(mbr["n_sig_positive"] >= 3)
    verdict = "ESTABLISHED" if established else "INCONCLUSIVE-BY-CONSTRUCTION"
    return {
        "g_repro": repro,
        "g_pair": pair,
        "g_exists": {
            "rule": ("delta > 0 with the paired 95% CI excluding 0 on >= 3 of 4 pairs "
                     "(docs/PLAN_z_aware.md WP-0, fixed before any arm ran)"),
            "n_sig_positive": mbr["n_sig_positive"],
            "n_arms_positive": mbr["n_arms_positive"],
            "sign_test_p": mbr["sign_test_p"],
            "verdict": verdict,
            # G-repro is `pass` where it could be scored; at an escalated tier it is
            # `not scored` and the 300-jet run is what certifies the pipeline.
            "gates_ok": bool((repro["pass"] or not repro["scored"]) and pair["pass"]),
            "g_repro_scored": bool(repro["scored"]),
        },
        "tables": tables,
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        # The tier these numbers live at. Recorded because a reader (and
        # `lnz_spline_gates.print_dmbr_band`) must be able to quote the analysis that
        # matches the artifact column it sits beside, rather than whichever ran last.
        "tier": {"closure_jets": int(next(iter(arms.values()))["tier"]["closure_jets"]),
                 "n_closure_samples": int(
                     next(iter(arms.values()))["tier"]["n_closure_samples"])} if arms else None,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_report(res: dict) -> None:
    r = res["g_repro"]
    print(f"\nG-repro — re-measured `dlund_mbr` vs the committed artifact "
          f"(tolerance {100 * r['tol']:.1f}%):")
    print(f"    {'arm':>20} {'recorded':>10} {'re-measured':>12} {'rel':>8} {'n_kept':>12}  ok")
    for row in r["rows"]:
        rec = f"{row['recorded']:.4f}" if row["recorded"] is not None else "n/a"
        nk = f"{row['n_kept_recorded']}/{row['n_kept_remeasured']}"
        rel = f"{row['rel']:.2%}" if np.isfinite(row["rel"]) else "n/a"
        tag = ("OK" if row["ok"] else "FAIL") if row["same_tier"] else (
            f"n/a ({row['recorded_tier']}-jet artifact vs {row['tier']}-jet run)")
        print(f"    {row['arm']:>20} {rec:>10} {row['remeasured']:>12.4f}"
              f" {rel:>8} {nk:>12}  {tag}")
    if r["scored"]:
        print(f"    -> {r['n_ok']}/{r['n']}  {'PASS' if r['pass'] else 'FAIL'}")
    else:
        print("    -> NOT SCORED: no arm was run at the tier its committed artifact "
              "records. G-repro is a\n       statement about this pipeline, and it is "
              "made at the published tier.")

    p = res["g_pair"]
    print("\nG-pair — `dlund_identity` is model-independent, so it must agree within a pair:")
    for row in p["rows"]:
        print(f"    {row['pair']:>40}  n = {row['n']:>3}"
              f"  max|diff| = {row['max_abs_diff']:.2e}  {'OK' if row['ok'] else 'FAIL'}")
    print(f"    -> {p['n_ok']}/{p['n']}  {'PASS' if p['pass'] else 'FAIL'}")

    for key, title in (
        ("dlund_mbr", "G-exists — THE question: is there a regression to explain?"),
        ("dlund_identity", "control: the pairing certificate (must be exactly 0)"),
        ("dlund_posterior_medoid", "control: the same draws, a free one-node estimator"),
        ("dlund_posterior_mode", "control: the same draws, the modal cell"),
        ("dlund_mbr_cont", "ruler C3: the MBR winner's own (u, v), off the cell grid"),
        ("dlund3_mbr_cont", "ruler C1: the same emission with `ln z` restored"),
        ("dlnz_mbr", "ruler C2: `|d ln z|` alone — the axis the spline moved"),
        ("dlund_posterior_geomedian_cont", "ruler: continuous geometric median, 2-D"),
        ("dlund3_posterior_geomedian_cont", "ruler: continuous geometric median, 3-D"),
        ("dlnz_posterior_geomedian", "ruler: continuous geometric median, `ln z` alone"),
    ):
        t = res["tables"].get(key)
        if t is None or not t["rows"]:
            continue
        print(f"\n{title}\n  delta = spline - control, per jet, paired BCa "
              f"({res['n_boot']} resamples), key = {key}:")
        print(f"    {'pair':>40} {'n':>4} {'mean':>9} {'95% CI':>22}   verdict")
        for row in t["rows"]:
            ci = f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
            tag = ("worse (CI > 0)" if row["sig_positive"] else
                   "better (CI < 0)" if row["sig_negative"] else "straddles 0")
            print(f"    {row['pair']:>40} {row['n']:>4} {row['mean']:>+9.4f} {ci:>22}   {tag}")
        pooled = t["pooled"]
        print(f"    {'pooled (reported, not the rule)':>40} {pooled['n']:>4}"
              f" {pooled['mean']:>+9.4f}"
              f" [{pooled['ci95'][0]:+.4f}, {pooled['ci95'][1]:+.4f}]")
        print(f"    -> positive on {t['n_arms_positive']}/{t['n_pairs']} arms"
              f" (sign test p = {t['sign_test_p']:.3f}, floor 0.125 at this n);"
              f" CI excludes 0 upward on {t['n_sig_positive']}, downward on"
              f" {t['n_sig_negative']}")

    g = res["g_exists"]
    print("\n" + "=" * 78)
    print(f"G-exists: {g['verdict']}")
    print(f"  rule (pre-registered): {g['rule']}")
    print(f"  measured: CI excludes 0 upward on {g['n_sig_positive']}/4 pairs;"
          f" delta > 0 on {g['n_arms_positive']}/4 arms")
    if not g["gates_ok"]:
        print("  WARNING: G-repro and/or G-pair did not pass — read nothing above until "
              "they do.")
    if g["verdict"] != "ESTABLISHED":
        print("  Consequence, fixed in advance (docs/PLAN_z_aware.md WP-0): +0.005 is "
              "inside its own\n  per-jet noise, so the §2.5 explanation is untestable — "
              "the phenomenon it explains is\n  not resolved. Escalate ONCE to 1000 jets; "
              "if still unresolved, rewrite the SUMMARY\n  sentence as \"d(MBR) is "
              "unchanged within its own per-jet noise\". WP-1..WP-3 ship\n  regardless; "
              "WP-3/WP-4 do not.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_arms(out_dir: Path) -> dict:
    arms = {}
    for name in ARM_ROOT:
        p = out_dir / f"{name}.json"
        if p.is_file():
            arms[name] = json.loads(p.read_text())
    return arms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated arm names (default: all 8)")
    ap.add_argument("--test-file", default=DEFAULT_TEST)
    ap.add_argument("--n-jets", type=int, default=TIER["closure_jets"])
    ap.add_argument("--k-draws", type=int, default=TIER["n_closure_samples"])
    ap.add_argument("--device", default="cpu",
                    help="cpu (default and the standing rule); cuda is a different RNG "
                         "stream and different float kernels")
    ap.add_argument("--out", default="", help="output directory (default: a fresh stamp)")
    ap.add_argument("--analyze", default="",
                    help="skip the decode passes; read <dir>/<arm>.json and print the verdict")
    ap.add_argument("--fast", action="store_true",
                    help="12 jets, K=16, one pair — a smoke test, not a measurement")
    args = ap.parse_args(argv)

    if args.analyze:
        out_dir = Path(args.analyze)
        arms = load_arms(out_dir)
        if not arms:
            print(f"no arm JSON under {out_dir}", file=sys.stderr)
            return 2
    else:
        if args.fast:
            args.n_jets, args.k_draws = 12, 16
            args.only = args.only or "spline_s0,v1_base_s0"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else REPO / "runs" / "zaware_wp0" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARM_ROOT)
        print(f"[wp0] {len(names)} arm(s) -> {out_dir}"
              f"  ({args.n_jets} jets, K = {args.k_draws}, {args.device})")
        for name in names:
            t0 = time.time()
            rec = run_arm(name, test_file=args.test_file, n_jets=args.n_jets,
                          k_draws=args.k_draws, device=args.device)
            (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
            c = rec["closure"]
            print(f"[wp0] {name:>20}  dlund_mbr = {c.get('dlund_mbr', float('nan')):.4f}"
                  f"  dlund3_mbr_cont = {c.get('dlund3_mbr_cont', float('nan')):.4f}"
                  f"  dlnz_mbr = {c.get('dlnz_mbr', float('nan')):.4f}"
                  f"  ({time.time() - t0:.0f} s)")
        arms = load_arms(out_dir)

    res = analyse(arms)
    res["arms_present"] = sorted(arms)
    res["out_dir"] = str(out_dir)
    print_report(res)
    (out_dir / "wp0.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[wp0] wrote {out_dir / 'wp0.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
