"""Minimal sweep runner (§2.1 / §8): replaces Hydra's `--multirun` with an
explicit grid looped over `load_config`. Each point trains a fresh run; extend the
`submit` hook to dispatch to SLURM.

Example:
    python scripts/sweep.py model=ar_junipr_v2 \
        optim.lr=1e-3,2e-3,3e-3 geometry.n_bins=10,16
"""

from __future__ import annotations

import itertools
import subprocess
import sys


def parse_grid(argv):
    """Split tokens into fixed `k=v` and swept `k=v1,v2,...` (comma-separated)."""
    fixed, swept = [], {}
    for tok in argv:
        key, val = tok.split("=", 1)
        if "," in val:
            swept[key] = val.split(",")
        else:
            fixed.append(tok)
    return fixed, swept


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fixed, swept = parse_grid(argv)
    keys = list(swept)
    combos = list(itertools.product(*[swept[k] for k in keys])) or [()]
    print(f"[sweep] {len(combos)} run(s) over {keys}")
    for combo in combos:
        overrides = fixed + [f"{k}={v}" for k, v in zip(keys, combo)]
        print(f"[sweep] === {overrides} ===")
        rc = subprocess.call([sys.executable, "-m", "h2p_rsd_junipr.cli", "train", *overrides])
        if rc != 0:
            print(f"[sweep] run failed (rc={rc}) for {overrides}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
