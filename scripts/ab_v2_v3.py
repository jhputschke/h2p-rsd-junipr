"""The v2-vs-v3 A/B: what the multiplicity head changed, and which decode knobs are
still doing work (docs/PLAN_UPDATES.md WP4).

v3 (`use_multiplicity_head=true`) makes the length a first-class categorical q(N|x)
and kills the two v2 length pathologies at source. Three decode knobs existed only
to patch those pathologies:

  * `min_emissions`          — the hard MAP floor against the empty-tree collapse
  * `length_floor_quantile`  — the learned per-jet floor against the residual under-count
  * `mbr_resample_to_qn`     — the decode-layer fix for the biased MBR candidate pool

Under v3 all three are EXPECTED to be near-no-ops. This script measures that instead
of assuming it, and answers:

  (a) is the empty-tree collapse gone at `min_emissions=0` under v3?
      (expected yes — the argmax is over q(N|x), which is not brevity-biased)
  (b) is `mbr_resample_to_qn` a no-op under v3?
      (expected yes — `length_pmf` is exact there, so the weights are ~1)
  (c) what multiplicity bias of ancestral draws SURVIVES v3?
      (whatever remains is coordinate-level exposure bias, which v3 does not touch)

The comparison is gated on the WP2 suite — per-coordinate PITs and TARP — and NOT on
SBC-N: v3 trains q(N|x) by direct NLL on N, so SBC-N would certify it near-
tautologically. That is the whole reason WP2 landed first.

Each arm is trained ONCE and evaluated at every decode cell (decode knobs are
inference-time only), so the grid costs 2 trainings, not 2 x |grid| trainings.

Run:
  python scripts/ab_v2_v3.py --preset presets/ab_v2_v3.yaml --out runs/ab_v2_v3
  python scripts/ab_v2_v3.py --fast          # CI tier: tiny data, 1 epoch, no MBR
  python scripts/ab_v2_v3.py --out runs/ab data.n_jets=20000 trainer.max_epochs=40

Any extra `key=value` token is passed through to `load_config` for both arms.
Writes `ab_table.md` (the docs table) and `ab_results.json` (machine-readable) to --out.
"""

from __future__ import annotations

import importlib.util
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

from h2p_rsd_junipr.config import config_hash, decode_params, experiment_params, load_config
from h2p_rsd_junipr.data.datamodule import LundDataModule
from h2p_rsd_junipr.data.stats import check_multiplicity_support
from h2p_rsd_junipr.eval.calibration import run_calibration
from h2p_rsd_junipr.eval.closure import run_closure
from h2p_rsd_junipr.eval.report import save_metrics
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.inference.length import learned_min_emissions
from h2p_rsd_junipr.inference.mbr import mbr_kwargs_from_decode
from h2p_rsd_junipr.train.logging import CSVJSONLLogger
from h2p_rsd_junipr.train.trainer import (
    Trainer,
    build_components,
    seed_everything,
    select_device,
)

ARMS = ("ar_junipr_v2", "ar_junipr_v3")
FLAGS = {"--preset", "--out", "--fast", "--arms", "--n-eval"}
POT_OK = importlib.util.find_spec("ot") is not None


def _parse(argv):
    flags, overrides = {}, []
    it = iter(argv)
    for tok in it:
        if tok == "--fast":
            flags["--fast"] = True
        elif tok in FLAGS:
            flags[tok] = next(it)
        elif tok.startswith("--"):
            raise SystemExit(f"unknown flag {tok!r}; known: {sorted(FLAGS)}")
        else:
            overrides.append(tok)
    return flags, overrides


# ---------------------------------------------------------------------------
# the decode grid
# ---------------------------------------------------------------------------
def decode_grid(with_mbr: bool):
    """{map, mbr} x {min_emissions 0,1} x {length_floor_quantile 0.0,0.5}
    x {mbr_resample_to_qn false,true}, pruned of the cells that are duplicates.

    The floors only steer the MAP and the q(N|x) reweighting only steers MBR, so the
    full 2^4 product contains 8 redundant cells. Pruning them is not a coverage cut:
    the dropped cells are bit-identical to kept ones."""
    cells = []
    for est, mn, alpha, resample in product(
        ("map", "mbr"), (0, 1), (0.0, 0.5), (False, True)
    ):
        if est == "map" and resample:
            continue                       # mbr_resample_to_qn is inert for the MAP
        if est == "mbr" and (mn, alpha) != (0, 0.0):
            continue                       # MBR is floor-free by construction
        if not (with_mbr or est == "map"):
            continue
        cells.append({"point_estimator": est, "min_emissions": mn,
                      "length_floor_quantile": alpha, "mbr_resample_to_qn": resample})
    return cells


def cell_label(c) -> str:
    if c["point_estimator"] == "mbr":
        return f"mbr, resample_to_qn={str(c['mbr_resample_to_qn']).lower()}"
    return f"map, min_emissions={c['min_emissions']}, floor_q={c['length_floor_quantile']}"


# ---------------------------------------------------------------------------
# train / evaluate one arm
# ---------------------------------------------------------------------------
def train_arm(model_name, base_argv, out_dir, preset):
    argv = list(base_argv)
    if preset:
        argv = [f"base={preset}", *argv]
    cfg = load_config([*argv, f"model={model_name}"])
    seed_everything(cfg.trainer.seed, cfg.trainer.deterministic)
    device = select_device()
    geometry = Geometry.from_config(cfg.geometry)
    dm = LundDataModule(cfg, geometry).setup()
    check_multiplicity_support(dm.jets, cfg)     # WP4 guard, before any training time

    run_dir = Path(out_dir) / f"{model_name}-{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVJSONLLogger(run_dir, tensorboard=False)
    model, opt, sched = build_components(cfg, geometry, device)
    print(f"\n[ab] training {model_name}  ({sum(p.numel() for p in model.parameters())/1e3:.1f}k "
          f"params, {len(dm.train_jets)} train jets) -> {run_dir}")
    trainer = Trainer(model, opt, sched, dm.loaders(), cfg, logger, device, run_dir,
                      dm.fingerprint)
    best = trainer.fit()
    logger.close()
    return {"cfg": cfg, "model": trainer.model, "device": device, "geometry": geometry,
            "dm": dm, "best_val_nll": float(best), "run_dir": run_dir}


def map_zero_fraction(model, val_ds, device, decode, n_jets):
    """Fraction of jets whose point estimate is the empty tree — the headline (a).

    Reproduces the eval-path floor logic exactly, including the learned per-jet floor,
    so the number answers "what would a user actually get" and not "what does the raw
    argmax do"."""
    n_jets = min(int(n_jets), len(val_ds))
    zeros, mults = 0, []
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        dec = dict(decode)
        draws = None
        alpha = float(dec.get("length_floor_quantile", 0.0))
        if alpha > 0.0 or dec.get("point_estimator") == "mbr":
            draws = model.sample_batch(xf, nx, int(dec.get("n_posterior_samples", 200)))
        if alpha > 0.0:
            dec["min_emissions"] = learned_min_emissions(
                model, xf, nx, quantile=alpha, base_floor=int(dec.get("min_emissions", 1)),
                mults=[len(d) for d in draws],
            )
        pe = model.map_or_mbr(xf, nx, draws=draws, **dec)
        zeros += int(pe.multiplicity == 0)
        mults.append(pe.multiplicity)
    return {"map0_frac": zeros / max(n_jets, 1),
            "point_mult_mean": float(np.mean(mults)) if mults else float("nan")}


def qn_weight_spread(model, val_ds, device, n_jets=50, K=200, seed=0):
    """Answer (b): how far the MBR `q(N|x)` importance weights are from 1 — measured
    against the finite-K null, which is the only way to read the number.

    `w_k = q(N=|y_k| | x) / p_emp(N=|y_k|)` compares an EXACT head against a histogram
    of K draws, so even a perfect sampler gives `w != 1` by Monte-Carlo noise alone
    (O(1/sqrt(K)) per bin). Quoting the raw spread would therefore make a genuine
    no-op look like a live correction. The null is obtained by re-drawing K
    multiplicities directly from the exact pmf and running the identical statistic;
    `excess = observed - null` is what is attributable to real sampler bias.

    For `ar_junipr_v2` there is no exact head — `length_pmf` reuses the very same
    draws — so `p_emp == pmf`, the weights are exactly 1, and both numbers are 0."""
    from h2p_rsd_junipr.inference.mbr import _qn_importance_weights

    def _spread(mults, pmf):
        m = np.asarray(mults, dtype=int)
        p_emp = np.bincount(m, minlength=pmf.size) / max(m.size, 1)
        w = np.array([(pmf[n] / p_emp[n]) if p_emp[n] > 0 else 0.0 for n in m])
        if w.sum() <= 0:
            return float("nan")
        w = w / w.mean()
        return float(np.abs(w - 1.0).mean())

    # Only meaningful where `length_pmf` is an EXACT head independent of the draws.
    # Without one (ar_junipr_v2) `length_pmf` *is* the draw histogram: the weights are
    # identically 1 by construction and re-drawing from that histogram would produce a
    # "null" that compares the histogram to itself — a number with no interpretation.
    exact_head = bool(getattr(model, "use_multiplicity_head", True))
    rng = np.random.default_rng(seed)
    n_jets = min(int(n_jets), len(val_ds))
    obs, null = [], []
    for i in range(n_jets):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        draws = model.sample_batch(xf, nx, K)
        mults = [len(d) for d in draws]
        w = _qn_importance_weights(model, xf, nx, draws)
        if w.size and w.sum() > 0:
            w = w / w.mean()
            obs.append(float(np.abs(w - 1.0).mean()))
        if exact_head:
            pmf = np.asarray(model.length_pmf(xf, nx, mults=mults), dtype=float)
            if pmf.sum() > 0:
                pmf = pmf / pmf.sum()
                null.append(_spread(rng.choice(pmf.size, size=K, p=pmf), pmf))
    o = float(np.nanmean(obs)) if obs else float("nan")
    n = float(np.nanmean(null)) if null else float("nan")
    return {"observed": o, "finite_K_null": n, "excess": (o - n) if exact_head else 0.0,
            "exact_head": exact_head, "K": int(K)}


def eval_cell(arm, cell, n_eval, exp):
    cfg, model, device, geometry = arm["cfg"], arm["model"], arm["device"], arm["geometry"]
    _, val_ds = arm["dm"].datasets()
    decode = {**decode_params(cfg), **cell}
    model.eval()
    closure = run_closure(model, val_ds, arm["dm"].val_jets, geometry, device,
                          K=exp["n_closure_samples"], n_closure=n_eval, decode=decode,
                          verbose=False)
    calib = run_calibration(model, val_ds, geometry, device, K=exp["n_closure_samples"],
                            n_jets=n_eval, verbose=False,
                            pit_coords=exp["pit_coords"],
                            stratify_regions=exp["stratify_regions"],
                            tarp=exp["tarp"], tarp_refs=exp["tarp_refs"],
                            tarp_reference=exp["tarp_reference"],
                            mbr_kwargs=mbr_kwargs_from_decode(decode))
    row = {
        "arm": str(cfg.model.name),
        "cell": cell_label(cell),
        "decode": cell,
        "val_nll": arm["best_val_nll"],
        **map_zero_fraction(model, val_ds, device, decode, n_eval),
        "mult_bias_posterior": closure["mult_bias_posterior"],
        "mult_bias_posterior_median": closure["mult_bias_posterior_median"],
        "mult_bias_by_N": closure["mult_bias_by_N"],
        "dlund_posterior_mode": closure["dlund_posterior_mode"],
        "coverage_68": calib["coverage_68"],
        "sbc_chi2": calib["sbc_chi2_uniform"],
        "pit_coords_ks_max": calib.get("pit_coords_ks_max", float("nan")),
        "tarp_max_dev": calib.get("tarp_max_dev", float("nan")),
    }
    if "mult_bias_mbr" in closure:
        row["mult_bias_mbr"] = closure["mult_bias_mbr"]
    return row


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _fmt(v, spec=".3f"):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return format(v, spec)


def markdown_table(rows, extras) -> str:
    head = ("| arm | decode cell | val NLL | MAP=0 frac | ⟨n−n_true⟩ mean | ⟨n−n_true⟩ median "
            "| cov68 | PIT KS max | TARP max dev |")
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| `{r['arm']}` | {r['cell']} | {_fmt(r['val_nll'], '.2f')} "
            f"| {_fmt(r['map0_frac'], '.3f')} | {_fmt(r['mult_bias_posterior'], '+.3f')} "
            f"| {_fmt(r['mult_bias_posterior_median'], '+.3f')} | {_fmt(r['coverage_68'], '.2f')} "
            f"| {_fmt(r['pit_coords_ks_max'])} | {_fmt(r['tarp_max_dev'])} |"
        )
    out = ["# v2 vs v3 A/B (docs/PLAN_UPDATES.md WP4)", "",
           "Decode knobs are inference-time only, so each arm is trained once and",
           "evaluated at every cell. Gated on the WP2 suite (per-coordinate PITs, TARP),",
           "not on SBC-N — which v3 optimizes directly.", "", *lines, "",
           "## (b) is `mbr_resample_to_qn` a no-op?", "",
           "| arm | observed mean \\|w−1\\| | finite-K null | excess |", "|---|---:|---:|---:|"]
    for arm, s in extras["qn_weight_spread"].items():
        null = _fmt(s["finite_K_null"], ".4f") if s.get("exact_head", True) else "n/a"
        excess = _fmt(s["excess"], "+.4f") if s.get("exact_head", True) else "0 (exact)"
        out.append(f"| `{arm}` | {_fmt(s['observed'], '.4f')} | {null} | {excess} |")
    out += ["",
            "`w_k = q(N=|y_k||x) / p_emp(N=|y_k|)` compares an exact head against a",
            "K-draw histogram, so `w != 1` at O(1/sqrt(K)) even for a perfect sampler.",
            "**Read the excess, not the observed value**: excess ~ 0 means the reweighting",
            "is a no-op up to Monte-Carlo noise. `ar_junipr_v2` has no exact head —",
            "`length_pmf` there IS the draw histogram, so the weights are identically 1 and",
            "the null does not apply (a histogram compared against itself)."]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    flags, overrides = _parse(list(sys.argv[1:] if argv is None else argv))
    preset = flags.get("--preset")
    out_dir = Path(flags.get("--out", "runs/ab_v2_v3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = (flags.get("--arms") or ",".join(ARMS)).split(",")
    fast = bool(flags.get("--fast"))
    if fast:  # CI tier: everything small, and no OT backend needed
        overrides = ["data.n_jets=400", "data.min_val=64", "trainer.max_epochs=1",
                     "trainer.batch_size=32", "experiment.closure_jets=15",
                     "experiment.n_closure_samples=20", "experiment.tarp_refs=20",
                     *overrides]
        preset = preset or None
    n_eval = int(flags.get("--n-eval", 0) or 0)

    trained = {name: train_arm(name, overrides, out_dir, preset) for name in arms}
    exp = experiment_params(next(iter(trained.values()))["cfg"])
    if fast:
        exp = {**exp, "pit_coords": True, "stratify_regions": True, "tarp": POT_OK}
    n_eval = n_eval or exp["closure_jets"]
    grid = decode_grid(with_mbr=POT_OK and not fast)
    if not POT_OK:
        print("[ab] POT not installed — skipping the MBR cells (pip install 'pot>=0.9').")

    rows = []
    for name, arm in trained.items():
        for cell in grid:
            print(f"[ab] evaluating {name}: {cell_label(cell)}")
            rows.append(eval_cell(arm, cell, n_eval, exp))
    extras = {"qn_weight_spread": {
        name: qn_weight_spread(arm["model"], arm["dm"].datasets()[1], arm["device"],
                               n_jets=min(50, n_eval))
        for name, arm in trained.items()
    }}

    save_metrics({"rows": rows, **extras}, out_dir / "ab_results.json")
    table = markdown_table(rows, extras)
    (out_dir / "ab_table.md").write_text(table)
    print("\n" + table)
    print(f"[ab] wrote {out_dir/'ab_table.md'} and {out_dir/'ab_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
