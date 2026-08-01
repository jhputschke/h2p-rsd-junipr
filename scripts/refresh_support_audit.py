"""Re-run the support audit for every evaluated arm, under ONE convention.

    python scripts/refresh_support_audit.py [--run-root runs/prod_test_v1] [--dry-run]

The boundary convention (`eval.support.EDGE_TOL`) changed mid-grid: a truncated sampler
clamps to its own bound, and a bound is only representable to float32, so a strict
comparison counted draws sitting exactly ON the soft-drop cut as crossings of it — 8 in
575 525 on the first trained arm, every one of them arithmetic. Arms evaluated before the
fix carry the old numbers, and a gate table that mixes two conventions is worse than
either.

This recomputes only the audit — one sampling pass per arm, cheap next to a full eval —
and patches `support_audit` in each arm's `eval_metrics.json` (and in the pass-A file it
came from), leaving every other block untouched. It records `audit_refreshed_at_edge_tol`
so the file says which convention produced its numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def refresh(stamp_dir: Path, *, dry_run=False) -> dict | None:
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.support import EDGE_TOL, run_support_audit
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    merged = stamp_dir / "eval_metrics.json"
    if not merged.is_file():
        return None
    m = json.loads(merged.read_text())
    if float(m.get("audit_refreshed_at_edge_tol", -1)) == EDGE_TOL:
        print(f"[refresh] {stamp_dir.parent.name}: already at EDGE_TOL={EDGE_TOL:g}")
        return m
    old = (m.get("support_audit") or {}).get("posterior") or {}

    info = load_for_inference(str(stamp_dir / "best.ckpt"), map_location="cpu")
    cfg = OmegaConf.create(info["config"])
    # the audit must run on the jets the eval reported on, not the training file
    cfg.data.path = m["data"]["path"]
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom)
    model.load_state_dict(info["model_state"])
    model.eval()
    dm = LundDataModule(cfg, geom).setup()
    dm.train_jets, dm.val_jets = [], dm.jets
    _, val_ds = dm.datasets()

    # same seed and same tier as the pass that produced the file it patches
    seed_everything(int(OmegaConf.select(cfg, "trainer.seed") or 0), True)
    exp = m.get("experiment", {})
    audit = run_support_audit(
        model, val_ds, dm.val_jets, geom, torch.device("cpu"),
        n_jets=int(exp.get("closure_jets", 2000)),
        K=int(exp.get("n_closure_samples", 200)), verbose=False,
    )
    new = audit["posterior"] or {}
    print(f"[refresh] {stamp_dir.parent.name}: soft_drop "
          f"{old.get('soft_drop', float('nan')):.5%} -> {new.get('soft_drop', float('nan')):.5%}"
          f"   z>1/2 {old.get('z_above_half', float('nan')):.5%} -> "
          f"{new.get('z_above_half', float('nan')):.5%}"
          f"   on the bound {new.get('frac_at_boundary', float('nan')):.5%}"
          f"   verdict {'PASS' if audit['passes'] else 'FAIL'}")
    if dry_run:
        return m
    m["support_audit"] = audit
    m["audit_refreshed_at_edge_tol"] = EDGE_TOL
    merged.write_text(json.dumps(m, indent=2) + "\n")
    calib = stamp_dir / "eval_metrics_calib.json"
    if calib.is_file():
        c = json.loads(calib.read_text())
        c["support_audit"] = audit
        c["audit_refreshed_at_edge_tol"] = EDGE_TOL
        calib.write_text(json.dumps(c, indent=2) + "\n")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="runs/prod_test_v1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    if not root.is_absolute():
        root = REPO / root
    n = 0
    for arm in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
        for stamp in sorted(arm.iterdir()):
            if (stamp / "eval_metrics.json").is_file():
                refresh(stamp, dry_run=a.dry_run)
                n += 1
    print(f"[refresh] {n} arm(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
