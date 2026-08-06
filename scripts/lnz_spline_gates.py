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
    "contstop_spline_s0": ("v1_contstop_s0", "control", "transfer"),
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-root", default="runs/lnz_spline")
    p.add_argument("--control-root", default="runs/prod_test_v1")
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
    leaked = [a for a, r in rows.items()
              if (r["spline"]["soft_drop_viol"] or 0) > 0
              or (r["spline"]["z_above_half_viol"] or 0) > 0]
    print("support guard: " + ("HELD — 0.0000% on every arm" if not leaked
                               else f"LEAKED on {leaked}"))

    out = Path(args.out) if args.out else root / "lnz_spline_gates.json"
    save_metrics({"plan": "docs/PLAN_lnz_spline_head.md",
                  "G3": g3, "G3_dv": g3dv,
                  "support_guard_held": not leaked, "arms": rows}, out)
    print(f"\n[gates] wrote {out}")
    return 0 if (g3["closes"] and g3dv["closes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
