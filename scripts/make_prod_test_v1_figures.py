"""Copy the figures `docs/PROD_TEST_v1_RESULTS.md` argues from into `docs/figures/`.

`runs/` is gitignored — the same reason `make_prod_test_figures.py` exists for v0. A
results document that pointed at the run directory would render blank for everyone who
did not produce the run, and the v1 grid is 11 arms of it.

Copies, per named arm, the calibration figures the assessment already wrote:

  calibration_pit_coords.png      per-coordinate PIT, the G3 instrument
  calibration_pit_by_region.png   the region x coordinate cross, coloured by KS / its own
                                  critical value — the G5 attribution instrument
  calibration_tarp.png            TARP against its RECOMPUTED null band, the G7 instrument
  calibration_by_region.png       region-stratified coverage

The arms are chosen to be the ones the prose actually compares:

  v1_base_s0        the pre-registered base arm (explicit q(N|x), physical ln z)
  v1_legacy_lnz_s0  the WP-A attribution arm — the same thing with the unbounded head
  v1_contstop_s0    the G8 winner (implicit continue/stop), which passes G7

    python scripts/make_prod_test_v1_figures.py
    python scripts/make_prod_test_v1_figures.py --run-root runs/prod_test_v1

Figures are copied from the run directory, so they always describe the run named here —
and they are redrawn from the MERGED record by the eval driver, so they describe the
2000-jet calibration tier rather than the 300-jet decode pass that runs after it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "prod_test_v1"

ARMS = ("v1_base_s0", "v1_legacy_lnz_s0", "v1_contstop_s0")
COPY = ("calibration_pit_coords.png", "calibration_pit_by_region.png",
        "calibration_tarp.png", "calibration_by_region.png")


def newest_stamp(arm_dir: Path) -> Path | None:
    stamps = [d for d in sorted(arm_dir.iterdir()) if (d / "eval_metrics.json").is_file()]
    return stamps[-1] if stamps else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="runs/prod_test_v1")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    if not root.is_absolute():
        root = REPO / root
    if not root.is_dir():
        print(f"no such run root: {root}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    n, missing = 0, []
    for arm in ARMS:
        arm_dir = root / arm
        stamp = newest_stamp(arm_dir) if arm_dir.is_dir() else None
        if stamp is None:
            missing.append(arm)
            continue
        for name in COPY:
            src = stamp / name
            if not src.is_file():
                missing.append(f"{arm}/{name}")
                continue
            dst = OUT / f"{arm}__{name}"
            shutil.copy2(src, dst)
            print(f"[figures] {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
            n += 1
    print(f"[figures] copied {n} file(s) into {OUT.relative_to(REPO)}")
    if missing:
        # Loud, not silent: a results document with a missing figure renders a broken
        # image, and the whole point of this script is that nobody else can regenerate it.
        print(f"[figures] MISSING: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
