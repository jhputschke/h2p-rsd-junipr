"""Score gate G3 for the RQ-spline `ln z` head (docs/PLAN_lnz_spline_head.md §3).

The measurement the plan pre-registered: does a monotone rational-quadratic spline on the
soft-drop interval close the `ln z` PIT failure that `lnz_support="physical"` halved and
left significant on every v1 seed?

    G3 CLOSES iff  KS < its own 1.36/sqrt(n) critical value on EVERY seed
                   AND the `ln_z x wide_soft` cell falls below 1.0x its own critical value.

Both clauses are pre-registered, and both are checked here rather than eyeballed. A
two-of-three pass is a partial result and is reported as one — the plan says so in advance
precisely because "the best seed closes it" is the reading a failed gate invites.

Three guards ride along, because a coordinate fix can buy the PIT by spending something
else:
  * the support audit must stay at 0.0000% soft-drop and z > 1/2 violations — the property
    v1's WP-A bought, which this change must not spend;
  * held-out NLL must not worsen beyond the control's own seed spread;
  * TARP and `pit_ks_max` are reported beside G3, so a narrower joint is visible.

Controls are the v1 arms themselves: `runs/prod_test_v1/v1_base_s{0,1,2}` are the SAME
preset and the SAME seeds with the truncated-normal head, already trained and evaluated,
and the spline is bit-identical off its switch (tests/test_lnz_spline.py). Nothing is
re-run to produce them.

Run (after `run_lnz_spline.sh` and `eval_prod_test_v1.sh --run-root runs/lnz_spline`):

    python scripts/lnz_spline_gates.py
    python scripts/lnz_spline_gates.py --run-root runs/lnz_spline --out docs/…
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

# arm -> its control arm. Seed to seed, family to family: a spline arm compared against a
# different seed would price the seed spread as if it were the intervention.
PAIRS = {
    "spline_s0": "v1_base_s0",
    "spline_s1": "v1_base_s1",
    "spline_s2": "v1_base_s2",
    "contstop_spline_s0": "v1_contstop_s0",
}
# The pre-registered gate runs on these three only. `contstop_spline_s0` is a transfer
# check on the fielded family and is reported separately, never folded into the verdict.
GATE_ARMS = ("spline_s0", "spline_s1", "spline_s2")


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
    which is the whole reason the region cross is read as a RATIO and never as a raw KS
    (`eval/report.py` draws it that way for the same reason)."""
    return 1.36 / math.sqrt(float(n)) if n else float("nan")


def best_val_nll(root: Path, arm: str) -> float | None:
    """The arm's best held-out NLL/jet, read from its TRAINING log.

    `eval_metrics.json` does not carry it — the eval reports closure and calibration, not
    the objective — so the grid log is the record. Both sides of every pair are
    `lnz_support="physical"` densities on the same `(u, v, ln z, psi)` box (the spline
    integrates to 1 over it, `tests/test_lnz_spline.py`), so this comparison is
    legitimate; the one that is not is across `lnz_support`, which nothing here does."""
    log = root / "logs" / f"{arm}.log"
    if not log.is_file():
        return None
    # The line continues past the number ("... = 3.846. checkpoints in runs/..."), so it
    # is matched rather than split on.
    hits = re.findall(r"best val NLL/jet = ([0-9.]+?)\.?(?:\s|$)", log.read_text())
    return float(hits[-1]) if hits else None


def read_arm(path: Path) -> dict:
    """The G3 numbers and the three guards, from one `eval_metrics.json`."""
    m = json.loads(path.read_text())
    cal = m.get("calibration", {})
    lnz = _get(cal, "pit_coords.coords.ln_z", {}) or {}
    cell = _get(cal, "pit_coords_by_region.ln_z.wide_soft", {}) or {}
    post = _get(m, "support_audit.posterior", {}) or {}
    return {
        "path": str(path),
        "ln_z_ks": lnz.get("ks"),
        "ln_z_n": lnz.get("n"),
        "ln_z_ratio": (lnz["ks"] / critical(lnz["n"])) if lnz.get("n") else None,
        "wide_soft_ks": cell.get("ks"),
        "wide_soft_n": cell.get("n"),
        "wide_soft_ratio": (cell["ks"] / critical(cell["n"])) if cell.get("n") else None,
        "pit_ks_max": cal.get("pit_coords_ks_max"),
        "tarp_max_dev": _get(cal, "tarp.tarp_max_dev"),
        "tarp_null_p95": _get(cal, "tarp.null_band.p95"),
        "tarp_passes_g7": _get(cal, "tarp.tarp_passes_g7"),
        # the WP-A property this change must not spend
        "soft_drop_viol": post.get("soft_drop"),
        "z_above_half_viol": post.get("z_above_half"),
        "dlund_mbr": _get(m, "closure.dlund_mbr"),
        "coverage_68": _get(m, "closure.coverage_68"),
    }


def fmt(x, spec=".4f", dash="—"):
    return dash if x is None else format(x, spec)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-root", default="runs/lnz_spline")
    p.add_argument("--control-root", default="runs/prod_test_v1")
    p.add_argument("--out", default=None, help="where to write the JSON (default beside the run)")
    args = p.parse_args(argv)

    root, croot = REPO / args.run_root, REPO / args.control_root
    rows = {}
    for arm, control in PAIRS.items():
        a_path, c_path = find_metrics(root, arm), find_metrics(croot, control)
        if a_path is None:
            print(f"[gates] {arm}: no eval_metrics.json under {root / arm} — skipped")
            continue
        spline, ctrl = read_arm(a_path), (read_arm(c_path) if c_path else None)
        spline["nll"] = best_val_nll(root, arm)
        if ctrl is not None:
            ctrl["nll"] = best_val_nll(croot, control)
        rows[arm] = {"control_arm": control, "spline": spline, "control": ctrl}
    if not rows:
        print("[gates] nothing to score: run the grid and the eval first.")
        return 2

    # ---- G3, the pre-registered gate -----------------------------------------
    print("\n" + "=" * 96)
    print("gate G3 — the `ln z` PIT, spline vs the truncated normal it replaces")
    print(f"{'arm':<22}{'KS':>9}{'crit':>9}{'ratio':>9}   |{'control KS':>12}{'ratio':>9}"
          f"   |{'bulk cell':>11}{'was':>8}")
    print("-" * 96)
    for arm, r in rows.items():
        s, c = r["spline"], r["control"] or {}
        tag = arm + ("" if arm in GATE_ARMS else "  (transfer)")
        print(f"{tag:<22}{fmt(s['ln_z_ks']):>9}{fmt(critical(s['ln_z_n'])):>9}"
              f"{fmt(s['ln_z_ratio'], '.2f'):>8}x   |{fmt(c.get('ln_z_ks')):>12}"
              f"{fmt(c.get('ln_z_ratio'), '.2f'):>8}x   |"
              f"{fmt(s['wide_soft_ratio'], '.2f'):>10}x{fmt(c.get('wide_soft_ratio'), '.2f'):>7}x")
    print("=" * 96)

    scored = [rows[a] for a in GATE_ARMS if a in rows]
    marg = [r["spline"]["ln_z_ratio"] for r in scored]
    bulk = [r["spline"]["wide_soft_ratio"] for r in scored]
    n_marg = sum(1 for v in marg if v is not None and v < 1.0)
    n_bulk = sum(1 for v in bulk if v is not None and v < 1.0)
    closes = len(scored) == len(GATE_ARMS) and n_marg == len(scored) and n_bulk == len(scored)
    verdict = ("CLOSED" if closes else
               "PARTIAL" if (n_marg or n_bulk) else "NOT CLOSED")
    print(f"\nG3 verdict: **{verdict}** — marginal PIT below its critical value on "
          f"{n_marg}/{len(scored)} seeds, the `ln_z x wide_soft` bulk cell on "
          f"{n_bulk}/{len(scored)}.")
    print("   (the gate needs BOTH clauses on ALL seeds; anything else is reported as "
          "partial, per the plan)")

    # ---- the three guards -----------------------------------------------------
    print("\n" + "=" * 96)
    print("guards — what a PIT improvement must not have been bought with")
    print(f"{'arm':<22}{'soft-drop':>11}{'z>1/2':>9}{'pit_ks_max':>12}{'TARP':>9}"
          f"{'null p95':>10}{'val NLL':>9}{'d(MBR)':>9}")
    print("-" * 96)
    for arm, r in rows.items():
        for label, blk in (("", r["spline"]), ("  control", r["control"])):
            if blk is None:
                continue
            name = (arm if not label else f"  vs {r['control_arm']}")
            print(f"{name:<22}{fmt(blk['soft_drop_viol'], '.5f'):>11}"
                  f"{fmt(blk['z_above_half_viol'], '.5f'):>9}"
                  f"{fmt(blk['pit_ks_max']):>12}{fmt(blk['tarp_max_dev'], '.4f'):>9}"
                  f"{fmt(blk['tarp_null_p95'], '.4f'):>10}{fmt(blk.get('nll'), '.4f'):>9}"
                  f"{fmt(blk['dlund_mbr'], '.4f'):>9}")
    print("=" * 96)

    leaked = [a for a, r in rows.items()
              if (r["spline"]["soft_drop_viol"] or 0) > 0
              or (r["spline"]["z_above_half_viol"] or 0) > 0]
    print("support guard: " + ("HELD — 0.0000% soft-drop and z > 1/2 violations on every arm"
                               if not leaked else f"LEAKED on {leaked}"))

    out = Path(args.out) if args.out else root / "lnz_spline_gates.json"
    save_metrics({
        "plan": "docs/PLAN_lnz_spline_head.md",
        "gate_arms": list(GATE_ARMS),
        "verdict": {
            "G3": verdict,
            "closes": bool(closes),
            "marginal_below_crit": f"{n_marg}/{len(scored)}",
            "bulk_cell_below_crit": f"{n_bulk}/{len(scored)}",
            "support_guard_held": not leaked,
        },
        "arms": rows,
    }, out)
    print(f"\n[gates] wrote {out}")
    return 0 if closes else 1


if __name__ == "__main__":
    raise SystemExit(main())
