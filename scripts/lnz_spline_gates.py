"""Score the spline work packages (docs/PLAN_lnz_spline_head.md §3, §6, §7).

Three questions, three verdicts, one script — because they are read off the same arms and
splitting them would let the seed sets drift apart.

**G3 (ln z)** — the original pre-registered gate:

    CLOSES iff  the `ln z` PIT KS is below its own 1.36/sqrt(n) critical value on EVERY
                seed AND the `ln_z x wide_soft` bulk cell is below 1.0x its own.

Scored over the `spline_s*` arms. §7.2a added seeds 3-5 for a specific reason: three seeds
cannot distinguish "one marginal seed" from "a 1-in-3 failure rate", and the first run
landed 2/3 with the third missing by 4%.

**G3-dv (§7.1)** — the same gate on the coordinate that became binding once `ln z` was
fixed. Scored over the `dvspline_*` arms, which carry BOTH splines, so it requires `dv`
*and* `ln z` to pass: fixing one by breaking the other is not a result.

**K sensitivity (§7.2b)** — `spline_k16_s2` against `spline_s2`: is the marginal seed's 4%
miss expressiveness or variance? Reported, never used to pick K. Tuning `K` against G3
across all seeds would make the gate circular, exactly as a closure-tuned bandwidth would
have made G7 circular.

Guards ride along on every arm, because a PIT improvement can be bought with something
else: the support audit must stay at 0.0000%, held-out NLL must not worsen beyond the
control's seed spread, and TARP / `pit_ks_max` / `d(MBR)` are reported beside the verdicts.

**`d(MBR)` is never printed without its band.** The guards table's column is four *unpaired*
means, and §6.2 read a conclusion off it that `PLAN_z_aware.md` WP-0 later withdrew: paired
per jet, every CI contains 0. `print_dmbr_band` therefore attaches the paired analysis from
`runs/zaware_wp0/*/wp0.json` — or says out loud that it is missing.

Controls are the arms that differ by exactly one thing — `v1_base_s*` for the `ln z`
spline, and the `ln z`-spline arm itself for the `dv` spline, so each row prices its own
intervention rather than the accumulated stack.

Run (after `run_lnz_spline.sh` and `eval_prod_test_v1.sh --run-root runs/lnz_spline`):

    python scripts/lnz_spline_gates.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from h2p_rsd_junipr.eval.report import save_metrics  # noqa: E402

# arm -> (control arm, control root key, group). The control is the arm differing by
# exactly ONE thing, so every row prices its own intervention: `v1_base_s*` for the ln z
# spline, and the ln z-spline arm itself for the dv spline stacked on top of it.
ARMS: dict[str, tuple[str | None, str, str]] = {
    "spline_s0": ("v1_base_s0", "control", "lnz"),
    "spline_s1": ("v1_base_s1", "control", "lnz"),
    "spline_s2": ("v1_base_s2", "control", "lnz"),
    "spline_s3": (None, "control", "lnz"),
    "spline_s4": (None, "control", "lnz"),
    "spline_s5": (None, "control", "lnz"),
    "dvspline_s0": ("spline_s0", "run", "dv"),
    "dvspline_s1": ("spline_s1", "run", "dv"),
    "dvspline_s2": ("spline_s2", "run", "dv"),
    "spline_k16_s2": ("spline_s2", "run", "k16"),
    # §10 (docs/PLAN_next_steps.md B1): three paired seeds on the FIELDED continue/stop
    # family, so §6.4's TARP finding stops resting on one arm. `v1_contstop_s2` is the
    # control that had to be trained for it.
    "contstop_spline_s0": ("v1_contstop_s0", "control", "transfer"),
    "contstop_spline_s1": ("v1_contstop_s1", "control", "transfer"),
    "contstop_spline_s2": ("v1_contstop_s2", "control", "transfer"),
    # §8.5(1): the conditioning experiment. Control is the SAME seed's ln z-spline arm, so
    # the row prices the cell-centre input alone.
    "cellctr_s0": ("spline_s0", "run", "cellctr"),
    "cellctr_s1": ("spline_s1", "run", "cellctr"),
    "cellctr_s2": ("spline_s2", "run", "cellctr"),
    # §8.5(2): does the dv spline pay once the conditioning is fixed? Control is the
    # cell-centre arm WITHOUT the dv spline, again isolating one change.
    "cellctr_dvspline_s0": ("cellctr_s0", "run", "cellctr_dv"),
}
COORDS = ("du", "dv", "ln_z", "psi")
# The tier the guards table's `d(MBR)` column is measured at (`eval_prod_test_v1.sh` pass
# B). `print_dmbr_band` quotes the paired analysis run at the SAME tier, so the band and
# the mean beside it are statements about one population.
DECODE_TIER_JETS = 300


def _get(d, path, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def find_metrics(root: Path, arm: str) -> Path | None:
    hits = sorted((root / arm).rglob("eval_metrics.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def critical(n) -> float:
    """The KS statistic's own 95% critical value. It differs per cell because n does —
    which is why the region cross is read as a RATIO and never as a raw KS."""
    return 1.36 / math.sqrt(float(n)) if n else float("nan")


def best_val_nll(root: Path, arm: str) -> float | None:
    """The arm's best held-out NLL/jet, from its TRAINING log — `eval_metrics.json` does
    not carry it. Every arm compared here is an `lnz_support="physical"` density on the
    same `(u, v, ln z, psi)` box, so the comparison is legitimate; the one that would not
    be is across `lnz_support`, which nothing here does."""
    log = root / "logs" / f"{arm}.log"
    if not log.is_file():
        return None
    hits = re.findall(r"best val NLL/jet = ([0-9.]+?)\.?(?:\s|$)", log.read_text())
    return float(hits[-1]) if hits else None


def read_arm(path: Path) -> dict:
    m = json.loads(path.read_text())
    cal = m.get("calibration", {})
    post = _get(m, "support_audit.posterior", {}) or {}
    out = {"path": str(path)}
    for c in COORDS:
        e = _get(cal, f"pit_coords.coords.{c}", {}) or {}
        out[f"{c}_ks"] = e.get("ks")
        out[f"{c}_ratio"] = (e["ks"] / critical(e["n"])) if e.get("n") else None
    for c in ("ln_z", "dv"):
        cell = _get(cal, f"pit_coords_by_region.{c}.wide_soft", {}) or {}
        out[f"{c}_bulk_ratio"] = (cell["ks"] / critical(cell["n"])) if cell.get("n") else None
    out.update({
        "pit_ks_max": cal.get("pit_coords_ks_max"),
        "tarp_max_dev": _get(cal, "tarp.tarp_max_dev"),
        "tarp_null_p95": _get(cal, "tarp.null_band.p95"),
        "tarp_passes_g7": _get(cal, "tarp.tarp_passes_g7"),
        "soft_drop_viol": post.get("soft_drop"),
        "z_above_half_viol": post.get("z_above_half"),
        "dlund_mbr": _get(m, "closure.dlund_mbr"),
    })
    return out


def fmt(x, spec=".4f", dash="—"):
    return dash if x is None else format(x, spec)


def verdict(rows: dict, arms: list[str], clauses: list[tuple[str, str]]) -> dict:
    """`n/N` per clause, and CLOSED only when every clause holds on every scored arm."""
    scored = [a for a in arms if a in rows]
    tally = {}
    for label, key in clauses:
        vals = [rows[a]["spline"].get(key) for a in scored]
        tally[label] = (sum(1 for v in vals if v is not None and v < 1.0), len(scored))
    closes = bool(scored) and all(n == d for n, d in tally.values())
    return {
        "arms_scored": scored,
        "clauses": {k: f"{n}/{d}" for k, (n, d) in tally.items()},
        "verdict": ("CLOSED" if closes else
                    "PARTIAL" if any(n for n, _ in tally.values()) else "NOT CLOSED"),
        "closes": closes,
    }


def print_dmbr_band(zaware_root: Path) -> dict | None:
    """The `d(MBR)` column above, but PAIRED and with its band.

    The guards table prints four unpaired means, and that column is the one number in this
    document that a reader has already been misled by: §6.2 read "+0.005, worse on 4/4" off
    it and attached a mechanism to it. `PLAN_z_aware.md` WP-0 then paired it per jet — which
    the exact `dlund_identity` pairing always permitted — and every CI contains 0, with the
    4/4 signing itself gone at 1000 jets.

    So this block is not decoration: it is the rule that the column may never again be
    quoted without its resolution. It reads `runs/zaware_wp0/*/wp0.json` and picks the run at
    the tier the column above is measured at (`DECODE_TIER_JETS`), newest first — NOT simply
    the newest, because the 1000-jet escalation scores a different population and a band from
    one population beside a mean from another is worse than no band. With no artifact at all
    it says so plainly rather than printing the bare means alone."""
    hits = sorted(zaware_root.glob("*/wp0.json"), key=lambda q: q.stat().st_mtime)
    if not hits:
        print("d(MBR) band: NOT MEASURED here — the column above is four UNPAIRED means, "
              "which is\n  how docs/PLAN_lnz_spline_head.md §6.2 reached a conclusion that "
              "was later withdrawn.\n  Run `python scripts/mbr_zaware_ab.py` for the paired "
              "per-jet CI (docs/PLAN_z_aware.md §11).")
        return None
    # The tier that MATCHES the column above, not whichever ran last: the guards table is
    # the 300-jet decode tier, and the 1000-jet escalation scores a different population.
    def _tier(path):
        return _get(json.loads(path.read_text()), "tier.closure_jets")

    chosen = next((h for h in reversed(hits) if _tier(h) == DECODE_TIER_JETS), hits[-1])
    others = [h.parent.name for h in hits if h != chosen]
    rec = json.loads(chosen.read_text())
    t = rec.get("tables", {}).get("dlund_mbr", {})
    if not t.get("rows"):
        return None
    n_jets = "?"
    print(f"\nd(MBR), PAIRED per jet (docs/PLAN_z_aware.md §11; {chosen.parent.name},"
          f" {_get(rec, 'tier.closure_jets')}-jet tier):")
    print(f"    {'pair':>40} {'n':>5} {'mean':>9} {'paired BCa 95%':>22}")
    for row in t["rows"]:
        ci = f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
        print(f"    {row['pair']:>40} {row['n']:>5} {row['mean']:>+9.4f} {ci:>22}")
    p = t["pooled"]
    print(f"    {'pooled':>40} {p['n']:>5} {p['mean']:>+9.4f}"
          f" [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
    print(f"    -> CI excludes 0 upward on {t['n_sig_positive']}/{t['n_pairs']} pairs."
          f"  {'d(MBR) is UNCHANGED within its own per-jet noise.' if t['n_sig_positive'] < 3 else 'A regression is ESTABLISHED.'}")
    # the ruler that would have explained a regression, had there been one
    for key, label in (("dlund_mbr_cont", "the winner's own (u, v), off the grid"),
                       ("dlund3_mbr_cont", "...with ln z restored"),
                       ("dlnz_mbr", "...|d ln z| alone")):
        tt = rec.get("tables", {}).get(key)
        if tt and tt.get("pooled", {}).get("n"):
            q = tt["pooled"]
            n_jets = q["n"]
            print(f"    {label:>40} {q['n']:>5} {q['mean']:>+9.4f}"
                  f" [{q['ci95'][0]:+.4f}, {q['ci95'][1]:+.4f}]")
    print(f"    (a ln z-aware ruler over the same {n_jets} paired jets — eval/closure.py's"
          " dlund3_* / dlnz_* series)")
    if others:
        print(f"    other wp0 runs present, NOT quoted here: {', '.join(others)}")
    return {"source": str(chosen), "tier": rec.get("tier"), "other_runs": others,
            "dlund_mbr": t,
            **{k: rec["tables"][k] for k in ("dlund_mbr_cont", "dlund3_mbr_cont", "dlnz_mbr")
               if k in rec.get("tables", {})}}


# §10 / PLAN_next_steps.md B1 — does the spline's TARP gain reach the FIELDED family?
# T1/T2 were committed in 7932089, before contstop_spline_s1/s2 and v1_contstop_s2 were
# trained. The bar is the weakest improving instance of the effect being transferred
# (§6.4's s1, -0.0085), which is also about one control seed spread there.
CONTSTOP_PAIRS = ("contstop_spline_s0", "contstop_spline_s1", "contstop_spline_s2")
T1_MIN_MEAN_GAIN = -0.0085


def contstop_tarp_transfer(rows: dict) -> dict | None:
    """§10.3's T1, scored. Paired delta(tarp_max_dev) = spline - control, per seed."""
    pairs = []
    for arm in CONTSTOP_PAIRS:
        r = rows.get(arm)
        if not r or not r.get("control"):
            continue
        a, b = r["spline"].get("tarp_max_dev"), r["control"].get("tarp_max_dev")
        if a is None or b is None:
            continue
        pairs.append({
            "arm": arm, "control_arm": r["control_arm"],
            "spline": float(a), "control": float(b), "delta": float(a) - float(b),
            "null_p95": r["spline"].get("tarp_null_p95"),
            "spline_passes_g7": r["spline"].get("tarp_passes_g7"),
            "control_passes_g7": r["control"].get("tarp_passes_g7"),
            "nll_spline": r["spline"].get("nll"), "nll_control": r["control"].get("nll"),
        })
    if not pairs:
        return None
    d = [p["delta"] for p in pairs]
    n_neg = sum(1 for x in d if x < 0)
    mean = sum(d) / len(d)
    # The sanity clause: the val NLL must not REGRESS on any pair (a seed where it does is
    # a training failure to investigate, not a TARP result to report).
    nll_ok = all(p["nll_spline"] is not None and p["nll_control"] is not None
                 and p["nll_spline"] <= p["nll_control"] for p in pairs)
    complete = len(pairs) == len(CONTSTOP_PAIRS)
    t1 = bool(complete and n_neg == len(pairs) and mean <= T1_MIN_MEAN_GAIN)
    return {
        "rule": (f"T1 TRANSFERS iff delta < 0 on {len(CONTSTOP_PAIRS)}/"
                 f"{len(CONTSTOP_PAIRS)} AND mean delta <= {T1_MIN_MEAN_GAIN} "
                 f"(docs/PLAN_lnz_spline_head.md §10.3, fixed before the arms)"),
        "pairs": pairs, "n_pairs": len(pairs), "complete": complete,
        "n_negative": n_neg, "mean_delta": mean,
        "min_delta": min(d), "max_delta": max(d),
        "nll_never_regresses": nll_ok,
        "verdict": ("NOT SCORED" if not complete else
                    "TRANSFERS" if t1 else "DOES NOT TRANSFER"),
    }


def print_contstop_transfer(blk: dict | None) -> None:
    if not blk:
        return
    print("\n" + "=" * 100)
    print("§10 (B1) — does the spline's TARP gain reach the FIELDED continue/stop family?")
    print(f"{'seed pair':<44}{'control':>10}{'+spline':>10}{'delta':>10}"
          f"{'G7 ctl->spl':>14}")
    for p in blk["pairs"]:
        g7 = (f"{'yes' if p['control_passes_g7'] else 'no'}->"
              f"{'yes' if p['spline_passes_g7'] else 'no'}")
        print(f"{p['arm'] + ' vs ' + p['control_arm']:<44}{p['control']:>10.4f}"
              f"{p['spline']:>10.4f}{p['delta']:>+10.4f}{g7:>14}")
    p95 = next((p["null_p95"] for p in blk["pairs"] if p["null_p95"]), None)
    print(f"{'':44}{'':10}{'':10}{'mean':>10}{'':>0} {blk['mean_delta']:+.4f}"
          + (f"     (MC null p95 = {p95:.4f})" if p95 else ""))
    print(f"  improves on {blk['n_negative']}/{blk['n_pairs']};  "
          f"val NLL never regresses: {blk['nll_never_regresses']}")
    print(f"  {blk['rule']}")
    print(f"  VERDICT: {blk['verdict']}")
    print("=" * 100)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-root", default="runs/lnz_spline")
    p.add_argument("--control-root", default="runs/prod_test_v1")
    p.add_argument("--zaware-root", default="runs/zaware_wp0",
                   help="where scripts/mbr_zaware_ab.py wrote its paired d(MBR) analysis")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    root, croot = REPO / args.run_root, REPO / args.control_root
    rows: dict[str, dict] = {}
    for arm, (control, where, group) in ARMS.items():
        a_path = find_metrics(root, arm)
        if a_path is None:
            continue
        spline = read_arm(a_path)
        spline["nll"] = best_val_nll(root, arm)
        ctrl = None
        if control:
            c_root = croot if where == "control" else root
            c_path = find_metrics(c_root, control)
            if c_path:
                ctrl = read_arm(c_path)
                ctrl["nll"] = best_val_nll(c_root, control)
        rows[arm] = {"control_arm": control, "group": group, "spline": spline,
                     "control": ctrl}
    if not rows:
        print("[gates] nothing to score: run the grid and the eval first.")
        return 2

    # ---- the per-coordinate PIT, which is what every gate here reads ----------
    print("\n" + "=" * 100)
    print("per-coordinate PIT, as a ratio to each coordinate's OWN critical value (>1 fails)")
    print(f"{'arm':<20}{'group':<10}" + "".join(f"{c:>9}" for c in COORDS)
          + f"{'ln_z bulk':>11}{'dv bulk':>10}")
    print("-" * 100)
    for arm, r in rows.items():
        s = r["spline"]
        print(f"{arm:<20}{r['group']:<10}"
              + "".join(f"{fmt(s[f'{c}_ratio'], '.2f'):>8}x" for c in COORDS)
              + f"{fmt(s['ln_z_bulk_ratio'], '.2f'):>10}x{fmt(s['dv_bulk_ratio'], '.2f'):>9}x")
    print("=" * 100)

    lnz_arms = [a for a, r in rows.items() if r["group"] == "lnz"]
    dv_arms = [a for a, r in rows.items() if r["group"] == "dv"]
    g3 = verdict(rows, lnz_arms, [("ln z marginal", "ln_z_ratio"),
                                  ("ln_z x wide_soft", "ln_z_bulk_ratio")])
    g3dv = verdict(rows, dv_arms, [("dv marginal", "dv_ratio"),
                                   ("dv x wide_soft", "dv_bulk_ratio"),
                                   ("ln z marginal (must not regress)", "ln_z_ratio")])
    for name, v in (("G3    (ln z spline)", g3), ("G3-dv (§7.1, dv spline)", g3dv)):
        if not v["arms_scored"]:
            continue
        print(f"\n{name}: **{v['verdict']}**  on {len(v['arms_scored'])} seeds "
              f"({', '.join(v['arms_scored'])})")
        for k, frac in v["clauses"].items():
            print(f"     {k:<36} below its critical value on {frac}")

    # ---- §7.2b: K sensitivity, reported and never used to choose K ------------
    k16 = rows.get("spline_k16_s2")
    if k16 and k16["control"]:
        a, b = k16["spline"], k16["control"]
        print("\n§7.2b  K sensitivity on the marginal seed (K=16 vs K=8, seed 2):")
        print(f"     ln z  {fmt(b['ln_z_ratio'], '.2f')}x -> {fmt(a['ln_z_ratio'], '.2f')}x"
              f"     bulk  {fmt(b['ln_z_bulk_ratio'], '.2f')}x -> "
              f"{fmt(a['ln_z_bulk_ratio'], '.2f')}x"
              f"     NLL  {fmt(b['nll'], '.3f')} -> {fmt(a['nll'], '.3f')}")
        print("     (reported, NOT used to select K — a K tuned against G3 makes the gate "
              "circular)")

    # ---- the guards ----------------------------------------------------------
    print("\n" + "=" * 100)
    print("guards — what a PIT improvement must not have been bought with")
    print(f"{'arm':<20}{'soft-drop':>11}{'z>1/2':>9}{'pit_ks_max':>12}{'TARP':>9}"
          f"{'G7':>6}{'val NLL':>10}{'d(MBR)':>9}")
    print("-" * 100)
    for arm, r in rows.items():
        for label, blk in ((arm, r["spline"]), (f"  vs {r['control_arm']}", r["control"])):
            if blk is None:
                continue
            print(f"{label:<20}{fmt(blk['soft_drop_viol'], '.5f'):>11}"
                  f"{fmt(blk['z_above_half_viol'], '.5f'):>9}{fmt(blk['pit_ks_max']):>12}"
                  f"{fmt(blk['tarp_max_dev'], '.4f'):>9}"
                  f"{('yes' if blk['tarp_passes_g7'] else 'no'):>6}"
                  f"{fmt(blk.get('nll'), '.3f'):>10}{fmt(blk['dlund_mbr'], '.4f'):>9}")
    print("=" * 100)
    transfer = contstop_tarp_transfer(rows)
    print_contstop_transfer(transfer)
    zaware = print_dmbr_band(REPO / args.zaware_root)
    leaked = [a for a, r in rows.items()
              if (r["spline"]["soft_drop_viol"] or 0) > 0
              or (r["spline"]["z_above_half_viol"] or 0) > 0]
    print("support guard: " + ("HELD — 0.0000% on every arm" if not leaked
                               else f"LEAKED on {leaked}"))

    out = Path(args.out) if args.out else root / "lnz_spline_gates.json"
    save_metrics({"plan": "docs/PLAN_lnz_spline_head.md",
                  "G3": g3, "G3_dv": g3dv, "d_mbr_paired": zaware,
                  "contstop_tarp_transfer": transfer,
                  "support_guard_held": not leaked, "arms": rows}, out)
    print(f"\n[gates] wrote {out}")
    return 0 if (g3["closes"] and g3dv["closes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
