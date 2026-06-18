"""Probe the MAP=0 (empty-tree) collapse and what reduces it.

The MAP point estimate argmax_y q_phi(y|x) collapses to the unphysical empty tree
for a sizeable fraction of jets (the brevity bias of an un-normalized argmax over a
high-entropy categorical head). This script trains variants and reports, per variant,
the headline `map0_frac` (fraction of jets with MAP multiplicity 0) alongside NLL/jet
and the multiplicity bias of the MAP / posterior-mean / posterior-median estimators.

It sweeps the cartesian product of:
  --epochs            training length (does more training shrink the *underlying* pressure?)
  --label-smoothing   AR split-head label smoothing (cell_label_smoothing; AR only)
  --min-emissions     the decode-time MAP floor (the structural fix; eval-only)

min_emissions is applied at eval, so each (epochs x label_smoothing) model is trained
once and evaluated at every floor. Baseline (epochs=20, ls=0, min_emissions=0) reproduces
the collapse; min_emissions=1 must drive map0_frac -> 0 with NLL within band.

Run:
  python scripts/probe_map_collapse.py model=ar_junipr_v2 encoder=gru \
      --epochs 20 --label-smoothing 0.0,0.1 --min-emissions 0,1 --n-eval 200
  python scripts/probe_map_collapse.py model=cinn      encoder=gru --min-emissions 0,1
  python scripts/probe_map_collapse.py model=diffusion encoder=gru --min-emissions 0,1

Any extra OmegaConf override (e.g. data.n_jets=2000, trainer.batch_size=128) is passed
through to load_config, so the probe can be made fast for a smoke check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.datamodule import LundDataModule
from h2p_rsd_junipr.data.dataset import collate
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.train.logging import CSVJSONLLogger
from h2p_rsd_junipr.train.trainer import (
    Trainer, build_components, seed_everything, select_device,
)

PROBE_FLAGS = {"--epochs", "--label-smoothing", "--min-emissions", "--n-eval", "--posterior-k"}


def _parse(argv):
    """Split argv into probe flags (--flag a,b,c) and OmegaConf overrides (key=value)."""
    flags, overrides = {}, []
    it = iter(argv)
    for tok in it:
        if tok in PROBE_FLAGS:
            flags[tok] = next(it)
        elif tok.startswith("--"):
            raise SystemExit(f"unknown flag {tok!r}; known: {sorted(PROBE_FLAGS)}")
        else:
            overrides.append(tok)
    return flags, overrides


def _floats(s):
    return [float(x) for x in str(s).split(",")]


def _ints(s):
    return [int(x) for x in str(s).split(",")]


def _model_name(overrides):
    for tok in overrides:
        if tok.startswith("model="):
            return tok.split("=", 1)[1]
    return "ar_junipr_v2"


def train_variant(base_overrides, epochs, ls, is_ar):
    overrides = list(base_overrides) + [f"trainer.max_epochs={epochs}"]
    if is_ar:
        overrides.append(f"model.cell_label_smoothing={ls}")
    cfg = load_config(overrides)
    seed_everything(cfg.trainer.seed, cfg.trainer.deterministic)
    device = select_device()
    geom = Geometry.from_config(cfg.geometry)
    dm = LundDataModule(cfg, geom).setup()
    tag = f"{cfg.model.name}_ep{epochs}_ls{ls}".replace(".", "p")
    run_dir = Path("runs") / "probe_map_collapse" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVJSONLLogger(run_dir, tensorboard=False)
    model, opt, sched = build_components(cfg, geom, device)
    trainer = Trainer(model, opt, sched, dm.loaders(), cfg, logger, device, run_dir, dm.fingerprint)
    best_val = trainer.fit()
    logger.close()
    model.eval()
    return model, dm, geom, device, best_val


@torch.inference_mode()
def evaluate(model, dm, geom, device, n_eval, K, min_emissions_list):
    _, val_ds = dm.datasets()
    B = min(n_eval, len(val_ds))

    # NLL/jet (batched; independent of the MAP floor)
    batch = collate([val_ds[i] for i in range(B)])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    nll = float((-model.log_prob(batch)).mean())

    # sample once per jet; run the MAP at each floor
    n_true, post_mean, post_median = [], [], []
    map_by_me = {me: [] for me in min_emissions_list}
    for i in range(B):
        item = val_ds[i]
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([item["nx"]], device=device)
        mults = np.array([len(d) for d in model.sample_batch(xf, nx, K)])
        n_true.append(int(item["ny"]))
        post_mean.append(float(mults.mean()))
        post_median.append(float(np.median(mults)))
        for me in min_emissions_list:
            map_by_me[me].append(model.map_estimate(xf, nx, min_emissions=me).multiplicity)

    n_true = np.array(n_true, float)
    rows = []
    for me in min_emissions_list:
        nm = np.array(map_by_me[me], float)
        rows.append(dict(
            min_emissions=me,
            map0_frac=float(np.mean(nm == 0)),
            nll=nll,
            bias_map=float(np.mean(nm - n_true)),
            bias_mean=float(np.mean(np.array(post_mean) - n_true)),
            bias_median=float(np.mean(np.array(post_median) - n_true)),
        ))
    return rows


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    flags, overrides = _parse(argv)
    model_name = _model_name(overrides)
    is_ar = model_name.startswith("ar_junipr")

    epochs_list = _ints(flags.get("--epochs", "20"))
    ls_list = _floats(flags.get("--label-smoothing", "0.0")) if is_ar else [0.0]
    me_list = _ints(flags.get("--min-emissions", "0,1"))
    n_eval = int(flags.get("--n-eval", "200"))
    K = int(flags.get("--posterior-k", "200"))

    print(f"probe: model={model_name}  epochs={epochs_list}  "
          f"label_smoothing={ls_list if is_ar else '(n/a)'}  min_emissions={me_list}  "
          f"n_eval={n_eval}  K={K}")

    table = []
    for epochs in epochs_list:
        for ls in ls_list:
            print(f"\n[train] {model_name}  epochs={epochs}  label_smoothing={ls} ...")
            model, dm, geom, device, best = train_variant(overrides, epochs, ls, is_ar)
            print(f"[train] best val NLL/jet = {best:.3f}")
            for row in evaluate(model, dm, geom, device, n_eval, K, me_list):
                row.update(model=model_name, epochs=epochs, ls=ls)
                table.append(row)

    # ---- summary table ----------------------------------------------------
    hdr = ["model", "epochs", "ls", "min_emiss", "map0_frac", "nll/jet",
           "bias_map", "bias_mean", "bias_median"]
    print("\n" + "=" * 92)
    print("  ".join(f"{h:>11s}" for h in hdr))
    print("-" * 92)
    for r in table:
        print("  ".join([
            f"{r['model']:>11s}", f"{r['epochs']:>11d}", f"{r['ls']:>11.3g}",
            f"{r['min_emissions']:>11d}", f"{r['map0_frac']:>11.3f}", f"{r['nll']:>11.3f}",
            f"{r['bias_map']:>+11.3f}", f"{r['bias_mean']:>+11.3f}", f"{r['bias_median']:>+11.3f}",
        ]))
    print("=" * 92)

    floored = [r for r in table if r["min_emissions"] >= 1]
    ok = all(r["map0_frac"] == 0.0 for r in floored) if floored else False
    print(f"min_emissions>=1 drives MAP=0 to zero: {'YES' if ok else 'NO'}  "
          f"(posterior median is the recommended multiplicity point estimate)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
