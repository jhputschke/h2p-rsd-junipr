# Usage guide: training, checkpoints, resume, inference

A practical, copy-pasteable walkthrough of the full workflow. For the physics see
[`README_PHYSICS.md`](README_PHYSICS.md); for the design see
[`PRODUCTION-PLAN-v4.md`](PRODUCTION-PLAN-v4.md).

- [0. Mental model](#0-mental-model)
- [1. Train](#1-train)
- [2. Checkpoints and the "best" model](#2-checkpoints-and-the-best-model)
- [3. Resume a pre-empted run](#3-resume-a-pre-empted-run)
- [4. Evaluate / inference testing](#4-evaluate--inference-testing)
- [5. Programmatic inference](#5-programmatic-inference)
- [6. Export and serve](#6-export-and-serve)
- [7. Sweeps](#7-sweeps)
- [8. Troubleshooting](#8-troubleshooting)

---

## 0. Mental model

One command, `h2p-rsd-junipr <subcommand> [overrides]`, drives everything. Config
is OmegaConf with dotted CLI overrides that are **type-checked against the schema**
(a typo like `optim.lrr=1e-3` fails at load, not three hours in).

Every training run writes a self-contained **run directory**

```
runs/<YYYYMMDD-HHMMSS>-<cfg-hash>/
├── config.yaml      # the fully-resolved config for this run (provenance)
├── best.ckpt        # lowest val-NLL checkpoint seen so far
├── last.ckpt        # most recent epoch (for resume)
├── metrics.csv      # step,epoch,train_nll,val_nll,lr  (one row per epoch)
└── metrics.jsonl    # same, line-delimited JSON
```

`<cfg-hash>` is a 10-char hash of the resolved config, so the directory name itself
encodes the configuration. Override the root with `run_root=/path/to/runs`.

Install first (editable):

```bash
pip install -e .            # core
pip install -e ".[dev]"     # + pytest/ruff/black/mypy for the test suite
```

---

## 1. Train

### Basic

```bash
h2p-rsd-junipr train model=ar_junipr_v2 encoder=gru trainer.max_epochs=20
```

With no `data=` override this uses the built-in **synthetic** matched-pair
simulator (8000 jets, seed 0) — no ROOT file needed, ideal for a first run. Output:

```
[train] model=ar_junipr_v2 encoder=gru device=mps run_dir=runs/20260617-220350-181e80b291
[train] 1300 train / 200 val jets (fingerprint=4c728ecb4470)
[train] 111.5k parameters
epoch  1   train NLL/jet =   35.721   val NLL/jet =   34.053
epoch  2   train NLL/jet =   32.216   val NLL/jet =   32.162
epoch  3   train NLL/jet =   30.669   val NLL/jet =   31.289
[train] done. best val NLL/jet = 31.289. checkpoints in runs/20260617-...
```

The objective is the per-jet **weighted negative log-likelihood**
`-(w · log q_φ(y|x)).sum() / w.sum()`; "NLL/jet" is that quantity averaged over the
split. The device is auto-selected (CUDA → MPS → CPU).

### Train on real data (from the C++ generator)

```bash
./cpp/build/pythia_driver 1000000 jets.root 1 cpp/cards/pp_dijet.cmnd   # generate
h2p-rsd-junipr train data=rntuple data.path=jets.root model=ar_junipr_v2
```

When the RNTuple carries `event` ids the split is **by event** (jets of one event
never straddle train/val); otherwise it is a deterministic trailing split.

### Pick the model and encoder (drop-in, no code change)

```bash
h2p-rsd-junipr train model=ar_junipr_v1            # discrete cells only (no continuous coords)
h2p-rsd-junipr train model=cinn   encoder=lundnet  # conditional flow + graph encoder
h2p-rsd-junipr train model=diffusion encoder=deepsets
```

Models: `ar_junipr_v2` (recommended), `ar_junipr_v1`, `cinn`, `diffusion`.
Encoders: `gru`, `lundnet`, `deepsets`.

### Key knobs

| Override | Default | Meaning |
|---|---|---|
| `trainer.max_epochs` | 20 | epochs |
| `trainer.batch_size` | 64 | minibatch |
| `trainer.seed` | 0 | global seed (torch/numpy/python) |
| `trainer.fast_dev_run` | false | ~2-step smoke path (CI); also `trainer=fast_dev` |
| `trainer.amp` | false | mixed precision (off — model is overhead-bound) |
| `trainer.compile` | false | `torch.compile(mode="reduce-overhead")` |
| `trainer.num_workers` | 0 | DataLoader workers |
| `optim.lr` / `optim.weight_decay` | 2e-3 / 3e-4 | AdamW |
| `optim.scheduler` / `optim.eta_min` | cosine / 3e-4 | LR schedule |
| `optim.grad_clip` | 1.0 | gradient-norm clip |
| `data.n_jets` / `data.seed` | 8000 / 0 | synthetic dataset |
| `geometry.n_bins` | 10 | Lund-cell grid (n_cells = n_bins²) |
| `run_root` | runs | where run dirs go |

Full list: the dataclass schemas in `src/h2p_rsd_junipr/config.py`.

### Quick smoke (seconds)

```bash
h2p-rsd-junipr train trainer=fast_dev model=ar_junipr_v2   # ~2 steps, asserts the loop runs
```

---

## 2. Checkpoints and the "best" model

After **every epoch** the trainer writes `last.ckpt`. Whenever the validation NLL
improves it **also** writes `best.ckpt` — so `best.ckpt` is always the lowest-val-NLL
model seen, and `last.ckpt` is the most recent epoch (use it to resume).

```
fit():
    last.ckpt  ← saved every epoch
    best.ckpt  ← saved when val_nll < best_val_so_far
```

A checkpoint is a **complete, resumable** snapshot (not just weights):

```python
{
  "format_version": 2,
  "model":     {"name": "ar_junipr_v2", "state_dict": ...},
  "config":    {...},                 # full resolved run config
  "optimizer": ..., "scheduler": ..., "scaler": ..., "ema": ...,
  "epoch": int, "global_step": int, "best_val_nll": float,
  "rng": {"torch": ..., "numpy": ..., "python": ..., "torch_cuda": ...},
  "git_sha": ..., "config_hash": ..., "data_fingerprint": ...,
}
```

Two ways to load it (see `src/h2p_rsd_junipr/train/checkpoint.py`):

- `load_for_inference(path)` → `{model_state, config, model_name}` — weights + config
  only, ignores optimiser state. Use this for eval/serving.
- `load_checkpoint(path)` + `restore_into(...)` — full state incl. RNG, for exact
  resume (the trainer does this for you).

---

## 3. Resume a pre-empted run

If a run is interrupted (pre-emption, crash, Ctrl-C), continue it from `last.ckpt`:

```bash
h2p-rsd-junipr train model=ar_junipr_v2 data.n_jets=1500 data.min_val=200 \
    trainer.max_epochs=8 trainer.seed=0 \
    trainer.resume_from=runs/<interrupted-run>/last.ckpt
```

The trainer rebuilds the model/optimiser/scheduler **from the checkpoint's own
config snapshot**, restores all state (weights, optimiser, scheduler, scaler, RNG,
epoch, step, best-val), and continues. A run pre-empted after epoch 3 of 8 picks up
at epoch 4 and reproduces the loss curve:

```
# phase 1 (interrupted at epoch 3 of 8)
epoch 1   train NLL/jet =   35.721   val NLL/jet =   34.053
epoch 2   train NLL/jet =   32.115   val NLL/jet =   31.949
epoch 3   train NLL/jet =   30.189   val NLL/jet =   30.507   ← crash here

# phase 2 (resume) — continues from epoch 4
epoch  4   train NLL/jet =   28.865   val NLL/jet =   29.084
epoch  5   train NLL/jet =   27.818   val NLL/jet =   28.458
epoch  6   train NLL/jet =   27.436   val NLL/jet =   28.337
epoch  7   train NLL/jet =   27.080   val NLL/jet =   27.987
epoch  8   train NLL/jet =   26.924   val NLL/jet =   27.882
```

(The uninterrupted 8-epoch run ends at val 27.828; resume reproduces it up to MPS
atomic non-determinism.)

**Semantics worth knowing:**

- **Same architecture only.** A `config_hash` mismatch on resume is a *hard error* —
  no silent architecture drift. Pass the same `model=`/`encoder=`/`geometry.*`
  overrides you trained with (the model is rebuilt from the snapshot, so it stays
  consistent; the data loaders come from your current overrides, so keep `data.*` the
  same too).
- **Resume continues to the original `max_epochs`.** Because changing `max_epochs`
  changes the hash, you cannot *extend* a finished run by resuming — to train longer,
  start a fresh run. Resume is for finishing an *interrupted* run.
- **A new run dir** is created for the continued portion; the original is untouched.

The exact save→resume round-trip (weights + forward parity) is covered by
`tests/test_checkpoint.py::test_save_resume_roundtrip`.

---

## 4. Evaluate / inference testing

### Closure + calibration + point estimate on a checkpoint

```bash
h2p-rsd-junipr eval runs/<id>/best.ckpt
```

This rebuilds the model from the checkpoint, regenerates the **same held-out
val set** (from the snapshotted config), and runs the §8 validation suite:

```
closure + calibration on held-out jets:
  mean multiplicity            :  true y = 5.30   hadron x = 5.15   posterior = 5.54
  leading-emission Lund distance to true y :  identity(x) = 0.558   posterior-mode = 1.745
  multiplicity signed bias  <n - n_true>   :  identity(x) = -0.150   posterior-mean = +0.245
  posterior 68% coverage of true leading cell = 0.42   (target ~0.68; <0.68 => over-confident)

posterior calibration (SBC / PIT / coverage):
  SBC rank-uniformity chi^2 (10 bins) = 20.00   SBC mean rank = 0.437   PIT mean = 0.518
  leading-cell 68% coverage = 0.37   (target ~0.68)

per-jet point estimate q_phi(y | x) for one validation jet:
  multiplicity:  truth y = 9   model MAP = 0   plain RSD (hadron x) = 7   posterior = 6.70 +/- 2.20
  ...
```

(Numbers above are from a deliberately under-trained 3-epoch demo; a fully trained
`ar_junipr_v2` reaches val NLL ≈ 20.7 and ~0.68 coverage — see
`scripts/verify_synthetic_result.txt`.)

What the metrics mean:
- **leading-emission Lund distance** — how close the MAP's hardest splitting is to
  the truth's (node-alignment-free); lower is better.
- **multiplicity bias** `⟨n − n_true⟩` — over/under-counting of splittings; closer to 0
  is better.
- **68% coverage / SBC / PIT** — calibration. Coverage well below 0.68 means
  over-confident (too-narrow) posteriors; SBC χ² near 0 and PIT mean near 0.5 mean
  well-calibrated.

Tune cost vs. precision with `experiment.closure_jets` and
`experiment.n_closure_samples`.

### Reproduce the v2 reference (acceptance tests)

```bash
python scripts/verify_parity.py      # bit-for-bit per_jet_nll vs the original v2 script
python scripts/verify_synthetic.py   # full 20-epoch train + closure on the v2 synthetic data
```

### Unit + integration tests

```bash
pytest -q                            # 36 tests: geometry, densities, models, checkpoint, data, config
pytest tests/test_train_integration.py   # fast_dev train for ar_junipr_v1/v2/cinn
```

---

## 5. Programmatic inference

The contract every family implements is `log_prob` / `sample` / `map_estimate`.
Load a checkpoint and call them directly:

```python
import numpy as np, torch
from omegaconf import OmegaConf
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset   # or data.rntuple.load_rntuple

# 1. rebuild the exact model from the checkpoint's snapshotted config
info  = load_for_inference("runs/<id>/best.ckpt", map_location="cpu")
cfg   = OmegaConf.create(info["config"])
geom  = Geometry.from_config(cfg.geometry)
model = build_model(cfg, geom)
model.load_state_dict(info["model_state"]); model.eval()

# 2. one jet (here synthetic; in practice from load_rntuple)
ds   = MatchedLundDataset(synthetic_matched_dataset(50, seed=123), geom)
item = ds[0]
xf, nx = item["xf"].unsqueeze(0), torch.tensor([item["nx"]])

# 3a. MAP point estimate: beam search + conditional coordinate modes
mp = model.map_estimate(xf, nx)
print(mp.pretty())                       # human-readable Lund tree
print(mp.multiplicity, mp.logprob)       # n splittings, log q(y_hat|x)
for n in mp.nodes:
    print(n.cell, n.kt, n.delta_R, n.z, n.psi)   # per-node continuous coords

# 3b. posterior draws: ancestral sampling -> multiplicity band
draws = model.sample(xf, nx, n=500)      # list of cell-id chains
m = np.array([len(d) for d in draws])
print(f"mult mean={m.mean():.2f}  68% CR={np.percentile(m,[16,84])}")

# 3c. exact per-jet log-likelihood on a collated batch
batch = collate([ds[0], ds[1], ds[2]])
print(model.log_prob(batch))             # (B,) tensor of log q_phi(y|x)
```

`map_estimate` returns a `LundPointEstimate` (`.nodes`, `.multiplicity`,
`.logprob`, `.pretty()`); each `LundNode` carries the cell id and the continuous
`(ln 1/ΔR, ln k_t, ln z, ψ)` plus derived `kt`, `delta_R`, `z`.

---

## 6. Export and serve

**Export the encoder** to TorchScript (autoregressive sampling/beam search stay a
Python loop — only the encoder is scripted, and it is `allclose`-verified against
eager):

```bash
h2p-rsd-junipr export runs/<id>/best.ckpt encoder_scripted.pt
# [export] scripted encoder -> encoder_scripted.pt (allclose-verified)
```

**Serve** a FastAPI endpoint (needs the `[serve]` extra: `pip install -e ".[serve]"`):

```bash
h2p-rsd-junipr serve runs/<id>/best.ckpt 127.0.0.1 8000
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "lnInvDelta":[0.3,1.3], "lnkt":[4.7,4.4], "lnz":[-1.1,-0.2], "psi":[-3.0,-2.8]
}'
# -> {"map_multiplicity":..,"map_logprob":..,"map_nodes":[...],
#     "posterior_mult_mean":..,"posterior_mult_68CR":[..,..]}
```

---

## 7. Sweeps

A dependency-free replacement for Hydra `--multirun` — loop `train` over an explicit
grid (comma-separated values are swept):

```bash
python scripts/sweep.py model=ar_junipr_v2 optim.lr=1e-3,2e-3,3e-3 geometry.n_bins=10,16
```

Each grid point is a fresh `train` run with its own run dir. Extend `submit()` in
`scripts/sweep.py` to dispatch to SLURM.

---

## 8. Troubleshooting

- **`config_hash mismatch on resume`** — you changed the architecture/config between
  the checkpoint and the resume command. Resume with the same `model=`/`encoder=`/
  `geometry.*`/`trainer.max_epochs` you trained with.
- **Empty train split / `IndexError`** — `data.min_val` exceeds `data.n_jets`. Lower
  `data.min_val` (e.g. `data.min_val=32`) for tiny runs.
- **Run-to-run numbers differ slightly** — MPS/CUDA atomics are non-deterministic
  even with a fixed seed. The bit-exact check is `scripts/verify_parity.py` (weight
  copy), not end-to-end training; closure numbers are reported with tolerance bands.
- **Unknown config key rejected at load** — that is OmegaConf struct-mode catching a
  typo; check the field name against `config.py`.
- **`rsd-junipr: command not found`** — the console script is `h2p-rsd-junipr`; or run
  `python -m h2p_rsd_junipr.cli ...`.
- **Slow / want a quick check** — `trainer=fast_dev` (≈2 steps) or reduce
  `data.n_jets` and `trainer.max_epochs`.
