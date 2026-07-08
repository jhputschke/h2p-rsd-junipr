# Usage guide: training, checkpoints, resume, inference

A practical, copy-pasteable walkthrough of the full workflow. For a field-by-field
explanation of **every config knob** (including the inference/decode parameters) see
[`CONFIGURATION.md`](CONFIGURATION.md); for the physics see
[`README_PHYSICS.md`](README_PHYSICS.md); for the design see
[`PRODUCTION-PLAN-v4.md`](PRODUCTION-PLAN-v4.md).

- [0. Mental model](#0-mental-model)
- [1. Train](#1-train)
- [2. Checkpoints and the "best" model](#2-checkpoints-and-the-best-model)
- [3. Resume a pre-empted run](#3-resume-a-pre-empted-run)
- [4. Evaluate / inference testing](#4-evaluate--inference-testing)
- [5. Inference in your own script or notebook](#5-inference-in-your-own-script-or-notebook)
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

Full list with per-field explanations: [`CONFIGURATION.md`](CONFIGURATION.md) (the
dataclass schemas in `src/h2p_rsd_junipr/config.py` are the source of truth).

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
  multiplicity signed bias  <n - n_true>   :  identity(x) = -0.150   posterior-mean = +0.245   posterior-median = +0.180
  posterior 68% coverage of true leading cell = 0.42   (target ~0.68; <0.68 => over-confident)

posterior calibration (SBC / PIT / coverage):
  SBC rank-uniformity chi^2 (10 bins) = 20.00   SBC mean rank = 0.437   PIT mean = 0.518
  leading-cell 68% coverage = 0.37   (target ~0.68)

per-jet point estimate q_phi(y | x) for one validation jet:
  multiplicity:  truth y = 9   model MAP = 8   plain RSD (hadron x) = 7   posterior = 6.70 +/- 2.20 (median 7, ...)
  ...
```

(Numbers above are from a deliberately under-trained 3-epoch demo; a fully trained
`ar_junipr_v2` reaches val NLL ≈ 20.7 and ~0.68 coverage — see
`scripts/verify_synthetic_result.txt`.)

> **MAP multiplicity floor.** The MAP is the *joint mode* `argmax_y q_φ(y|x)`; for a
> discrete autoregressive posterior it is length-biased and, un-floored, collapses to
> the **unphysical empty tree** (0 splittings) for a large fraction of jets. The
> decoder enforces `decode.min_emissions` (default **1**) so the MAP always has ≥1
> splitting; `decode.length_penalty` (GNMT `score/len**α`, default 0) further counters
> the brevity bias. `decode.length_floor_quantile` (default **0.0** = off) is the
> *learned, per-jet* version of that floor: it raises the MAP length to the
> `α`-quantile of the model's own length belief `P(n|x)` —
> `eff = max(min_emissions, ⌊Q_α(P(n|x))⌋)` — cutting the residual under-count while
> keeping `n=0` at 0% (the floor only ever raises the bound; `α=0` is bit-for-bit
> today's behavior, `α→median` ≈ a length-conditioned MAP). For a *count*, still prefer
> the **posterior median** — the MAP is the wrong summary for multiplicity. See
> `notebooks/inference_demo.ipynb` §6a and `scripts/probe_map_collapse.py`.

> **MBR point estimator.** `decode.point_estimator=mbr` swaps the joint-mode MAP for the
> **minimum-Bayes-risk** tree — the drawn tree of least expected perturbative-Lund EMD to
> the posterior — which is *floor-free*: it never collapses to the empty tree even with
> `decode.min_emissions=0` (an empty cloud pays the full mass-imbalance penalty). `eval`
> adds an `MBR (<backend>)` series to the closure panels and the per-jet print, so you can
> compare `dLund-to-truth` and `⟨n − n_true⟩` against the MAP:
> ```bash
> h2p-rsd-junipr eval runs/<id>/best.ckpt decode.point_estimator=mbr decode.mbr_backend=pot decode.min_emissions=0
> ```
> Needs the `[mbr]` extra (`pot`); `decode.mbr_backend=energyflow` (the `[energyflow]` extra)
> selects the same tree. MBR closure is O(K²) EMD solves per jet — shrink it with
> `experiment.closure_jets` / `decode.mbr_n_candidates`. Off by default; the `map` path is
> unchanged and imports no OT backend.

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

## 5. Inference in your own script or notebook

You don't need the CLI to run the model — `import h2p_rsd_junipr` and call it
directly. The package installs as `pip install -e .`; nothing here needs the
`[serve]` or `[track]` extras (only `numpy`/`torch`; add `matplotlib` for the
plots in §5.4).

The contract every model family implements is **`log_prob`** (exact per-jet
`log q_φ(y|x)`), **`sample`** (posterior draws), and **`map_estimate`** (the MAP
Lund tree). Everything below is built on those three.

### 5.1 Load a trained model once

Two equivalent ways. The shortest is the serving helper (no FastAPI needed — it is
only imported when you start the web server):

```python
import torch
from h2p_rsd_junipr.serving.api import load_service_model
from h2p_rsd_junipr.train.trainer import select_device     # cuda > mps > cpu

device = select_device()                                   # or torch.device("cpu")
model, geom = load_service_model("runs/<id>/best.ckpt", device)
# model is in eval() mode; `geom` is the Lund-plane geometry (cell <-> coords)
```

Equivalently, the explicit form (useful if you want the config too):

```python
from omegaconf import OmegaConf
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference

info  = load_for_inference("runs/<id>/best.ckpt", map_location=device)
cfg   = OmegaConf.create(info["config"])                   # the run's full config
geom  = Geometry.from_config(cfg.geometry)
model = build_model(cfg, geom).to(device)
model.load_state_dict(info["model_state"]); model.eval()
```

### 5.2 Infer from a single hadron-level Lund sequence

If you already have a jet's groomed **hadron-level** primary Lund sequence as four
arrays `(ln 1/ΔR, ln k_t, ln z, ψ)`, the one-call helper returns the MAP tree plus a
posterior summary:

```python
from h2p_rsd_junipr.serving.api import predict

x = {"lnInvDelta": [0.29, 1.29, 4.26, 4.29, 5.21],   # your jet's hadron-level x
     "lnkt":       [4.70, 4.42, 3.63, 2.17, 3.93],
     "lnz":        [-1.10, -0.16, -0.90, -1.90, -0.31],
     "psi":        [-3.07, -2.80, -0.31,  2.95, -2.38]}

out = predict(model, geom, device, x)
print(out["map_multiplicity"], out["map_logprob"])
print(out["posterior_mult_mean"], out["posterior_mult_68CR"])
for node in out["map_nodes"]:
    print(node["cell"], node["ln_invDelta"], node["ln_kt"], node["ln_z"], node["psi"])
```

To drive the model yourself (full control over `n` samples, beam width, etc.), build
the input tensor with `node_features` and call the contract methods:

```python
import numpy as np, torch
from h2p_rsd_junipr.features import node_features

xf = torch.tensor(node_features(x["lnInvDelta"], x["lnkt"], x["lnz"], x["psi"]))
xf = xf.unsqueeze(0).to(device)                # (1, n_nodes, 5)
nx = torch.tensor([xf.shape[1]], device=device)

# MAP groomed parton tree: beam search + conditional coordinate modes.
# min_emissions (default 1) floors the multiplicity so the MAP is never the unphysical
# empty tree; length_penalty (GNMT score/len**alpha, default 0) counters the brevity
# bias. Both default from cfg.decode; pass explicitly to override.
mp = model.map_estimate(xf, nx, min_emissions=1, length_penalty=0.0)   # LundPointEstimate
print(mp.pretty())                             # human-readable tree
print("multiplicity:", mp.multiplicity, " log q(y_hat|x):", mp.logprob)
for n in mp.nodes:                             # each node carries continuous coords
    print(n.cell, n.kt, n.delta_R, n.z, n.psi)

# Posterior draws -> multiplicity band (ancestral sampling; returns cell-id chains)
draws = model.sample(xf, nx, n=500)
mult  = np.array([len(d) for d in draws])
print(f"posterior multiplicity: median={np.median(mult):.0f} mean={mult.mean():.2f} "
      f"68% CR=[{np.percentile(mult,16):.0f}, {np.percentile(mult,84):.0f}]")
# the posterior median is the recommended multiplicity point estimate (the MAP is the
# length-biased joint mode; see the MAP floor note in §4)
```

**Learned per-jet floor.** To floor the MAP at a low quantile of the model's own length
belief `P(n|x)` instead of the hard `min_emissions` constant, compute the per-jet floor
with `learned_min_emissions` (reusing the draws you already took — no second sample) and
feed it back as `min_emissions`:

```python
from h2p_rsd_junipr.inference.length import learned_min_emissions

mult = np.array([len(d) for d in draws])                 # the draws from above
eff  = learned_min_emissions(model, xf, nx, quantile=0.15, base_floor=1, mults=mult)
mp_floored = model.map_estimate(xf, nx, min_emissions=eff)   # eff = max(1, ⌊Q_0.15(P(n|x))⌋)
print("learned floor:", eff, " MAP multiplicity:", mp_floored.multiplicity)
```

`quantile=0` short-circuits to `base_floor` (today's behavior). For cINN/diffusion the
`P(n|x)` is read exactly from the multiplicity head (the `mults` are ignored); for AR it
is the histogram of `mults`. Or, end-to-end, just set
`decode.length_floor_quantile=0.15` and the eval CLI / `predict` / `serve` path applies
it automatically (the posterior draws those code paths already take are reused).

**MBR point estimate (perturbative-Lund, floor-free).** Instead of the joint-mode MAP,
select the drawn tree of least *expected* perturbative-Lund EMD to the posterior
(`decode.point_estimator="mbr"`). It reuses the same draws, returns the same
`LundPointEstimate`, and — because an empty cloud pays the full mass-imbalance penalty —
never collapses to the empty tree **with no floor** (`min_emissions=0`), unlike the MAP.
It needs the optional `[mbr]` extra (the `pot` backend); `energyflow` is a separate,
independently importable extra:

```bash
pip install -e ".[mbr]"          # default `pot` backend (self-contained, lazy-imported)
pip install -e ".[energyflow]"   # optional reference EMD backend (needs a working wasserstein)
```

```python
# MBR reusing your own draws (no second sample), via the base dispatch:
draws = model.sample(xf, nx, n=200)                        # the draws from above
mbr = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="pot")
print("MBR multiplicity:", mbr.multiplicity, " risk:", mbr.risk)   # .risk is a score, NOT an NLL
print(mbr.pretty())

# the reference backend gives the SAME selected tree (its value differs by a 1/R scale):
mbr_ef = model.map_or_mbr(xf, nx, draws=draws, point_estimator="mbr", mbr_backend="energyflow")
assert [n.cell for n in mbr_ef.nodes] == [n.cell for n in mbr.nodes]
```

`pot` and `energyflow` agree on the *argmin* (the selected tree) but not the numeric scale
— EnergyFlow normalises ground distances by `R`, so pick **one backend per analysis** for
comparable `risk` numbers. Tune the metric with `decode.mbr_R` (length↔kinematics
trade-off, ≈ Lund-plane diameter), `decode.mbr_lnkt_cut` (perturbative support; `null`
inherits the geometry cut), and `decode.mbr_weight`/`mbr_coords`/`mbr_beta`. See
[`CONFIGURATION.md` §10](CONFIGURATION.md#10-inference-knobs-in-depth--the-map-floor-mincut--quantile-floor)
for every knob and [`README_PHYSICS.md` §3](README_PHYSICS.md) for the physics.

### 5.3 Batch inference over a `jets.root` file

Read the RNTuple your C++ stage produced, build the dataset, and run the model on
many jets at once. `log_prob` is vectorised over the batch; `map_estimate`/`sample`
are per-jet:

```python
import numpy as np, torch
from h2p_rsd_junipr.data.rntuple import load_rntuple
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate

jets = load_rntuple("jets.root", "Jets")       # list of per-jet dicts (x, y, weight, ...)
ds   = MatchedLundDataset(jets, geom)          # uses the model's geometry for cell targets

# (a) exact per-jet log-likelihood, batched
batch = collate([ds[i] for i in range(min(256, len(ds)))])
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
with torch.inference_mode():
    logq = model.log_prob(batch)               # (B,) tensor of log q_phi(y|x)
print("mean log q(y|x) =", float(logq.mean()))

# (b) MAP tree + posterior multiplicity for each jet
records = []
for i in range(min(100, len(ds))):
    item = ds[i]
    xf = item["xf"].unsqueeze(0).to(device)
    nx = torch.tensor([item["nx"]], device=device)
    mp = model.map_estimate(xf, nx)
    mult = np.array([len(d) for d in model.sample(xf, nx, n=200)])
    records.append({"jet": i, "map_mult": mp.multiplicity, "map_logq": mp.logprob,
                    "post_mult_mean": float(mult.mean()),
                    "post_mult_median": float(np.median(mult))})
# records -> pandas.DataFrame(records) for analysis
```

(If you only need the hadron-level `x` and not the matched truth `y`, build the
encoder input straight from the arrays with `node_features`, as in §5.2 — `y` is
only required for `log_prob`.)

### 5.4 In a Jupyter notebook (with plots)

The same calls work in a notebook; add `matplotlib` (`pip install matplotlib`) to
visualise the posterior. This plots the posterior multiplicity distribution and the
posterior draws on the Lund plane, with the MAP tree overlaid:

```python
import numpy as np, torch, matplotlib.pyplot as plt
from h2p_rsd_junipr.features import node_features

xf = torch.tensor(node_features(x["lnInvDelta"], x["lnkt"], x["lnz"], x["psi"])).unsqueeze(0).to(device)
nx = torch.tensor([xf.shape[1]], device=device)

draws = model.sample(xf, nx, n=2000)           # posterior cell-id chains
mp    = model.map_estimate(xf, nx)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# posterior multiplicity
mult = np.array([len(d) for d in draws])
ax1.hist(mult, bins=range(0, mult.max() + 2), align="left", alpha=0.8)
ax1.axvline(mp.multiplicity, color="k", ls="--", label=f"MAP n={mp.multiplicity}")
ax1.set_xlabel("posterior multiplicity"); ax1.set_ylabel("draws"); ax1.legend()

# posterior draws on the Lund plane (cell centres), MAP nodes overlaid
pts = np.array([geom.cell_center(c) for d in draws for c in d])   # (N, 2): (ln1/ΔR, ln kt)
if len(pts):
    ax2.hist2d(pts[:, 0], pts[:, 1], bins=geom.n_bins,
               range=[list(geom.ln_invdelta_range), list(geom.ln_kt_range)], cmap="Blues")
ax2.scatter([n.ln_invDelta for n in mp.nodes], [n.ln_kt for n in mp.nodes],
            c="red", marker="*", s=160, label="MAP")
ax2.set_xlabel(r"$\ln 1/\Delta R$"); ax2.set_ylabel(r"$\ln k_t$"); ax2.legend()
plt.tight_layout(); plt.show()

print(mp.pretty())                             # the MAP Lund tree, as text
```

Notebook tips: pick the device once with `select_device()` (CPU is fine and fully
deterministic for inference); wrap heavy loops in `torch.inference_mode()`; and
remember a freshly `build_model`'d-but-unloaded model is random — always
`load_state_dict` (or use `load_service_model`) before trusting the output.

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
#     "posterior_mult_mean":..,"posterior_mult_median":..,"posterior_mult_68CR":[..,..]}
# (map_multiplicity >= decode.min_emissions; the service reads the checkpoint's decode config)
# When the checkpoint's decode has point_estimator=mbr, the response additionally carries
# "mbr_risk" and "mbr_backend" (the point estimate is then the MBR tree, floor-free).
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
