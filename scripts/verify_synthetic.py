"""End-to-end verification on the SAME synthetic data as conditional_rsd_junipr_v2.py.

Reproduces the v2 script's `main()` through the refactored package: identical
synthetic dataset (seed 0, 8000 jets, trailing 800-jet val split), the same
ar_junipr_v2 model + AdamW/cosine schedule for 20 epochs, then the closure +
calibration + point-estimate diagnostics. Prints a PASS/FAIL against the v2
reference bands (scripts/baseline_v2_reference.txt).

Run:  python scripts/verify_synthetic.py
"""

from __future__ import annotations

from pathlib import Path

from h2p_rsd_junipr.config import load_config, save_config
from h2p_rsd_junipr.data.datamodule import LundDataModule
from h2p_rsd_junipr.eval.calibration import run_calibration
from h2p_rsd_junipr.eval.closure import print_point_estimate, run_closure
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.train.logging import CSVJSONLLogger
from h2p_rsd_junipr.train.trainer import Trainer, build_components, seed_everything, select_device

# v2 reference (from scripts/baseline_v2_reference.txt): final val NLL/jet ~ 20.71,
# posterior mean multiplicity ~ 6.14, leading-cell 68% coverage ~ 0.68.
REF = dict(val_nll=20.71, mult_posterior=6.14, coverage=0.68)
BANDS = dict(val_nll=(0.0, 23.0), mult_posterior=(4.5, 8.0), coverage=(0.45, 0.88))


def main() -> int:
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru"])  # defaults: 8000 jets, 20 epochs
    seed_everything(cfg.trainer.seed, cfg.trainer.deterministic)
    device = select_device()
    geom = Geometry.from_config(cfg.geometry)

    dm = LundDataModule(cfg, geom).setup()
    train_ds, val_ds = dm.datasets()
    print(
        f"training on {len(train_ds)} jets, validating on {len(val_ds)} "
        f"(device={device})  [fingerprint={dm.fingerprint}]"
    )

    run_dir = Path("runs") / "verify_synthetic"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger = CSVJSONLLogger(run_dir)
    model, opt, sched = build_components(cfg, geom, device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k")

    trainer = Trainer(model, opt, sched, dm.loaders(), cfg, logger, device, run_dir, dm.fingerprint)
    best_val = trainer.fit()
    logger.close()

    model.eval()
    closure = run_closure(
        model, val_ds, dm.val_jets, geom, device,
        K=cfg.experiment.n_closure_samples, n_closure=cfg.experiment.closure_jets,
    )
    run_calibration(model, val_ds, geom, device,
                    K=cfg.experiment.n_closure_samples, n_jets=cfg.experiment.closure_jets)
    print_point_estimate(model, val_ds, dm.val_jets, geom, device)

    got = dict(
        val_nll=best_val,
        mult_posterior=closure["mean_mult_posterior"],
        coverage=closure["coverage_68"],
    )
    print("\n" + "=" * 64)
    print("VERIFICATION vs v2 reference (bands are run-to-run tolerant):")
    ok = True
    for k, (lo, hi) in BANDS.items():
        passed = lo <= got[k] <= hi
        ok = ok and passed
        print(f"  {k:16s} got={got[k]:7.3f}  ref={REF[k]:7.3f}  band=[{lo},{hi}]  "
              f"{'PASS' if passed else 'FAIL'}")
    print("=" * 64)
    print("SYNTHETIC VERIFICATION " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
