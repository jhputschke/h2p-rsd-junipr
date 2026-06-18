# Configuration reference — every knob, what it means, when to turn it

This is the complete field-by-field reference for the config schema. It complements
[`USAGE.md`](USAGE.md) (which shows *how to run* the workflow) by explaining *what each
parameter does* — including the inference-time **decode** knobs (`min_emissions`,
`length_floor_quantile`, `length_penalty`, `cont_temperature`) that shape the MAP and
the posterior. For the physics behind these choices see
[`README_PHYSICS.md`](README_PHYSICS.md); the schema source of truth is
[`src/h2p_rsd_junipr/config.py`](../src/h2p_rsd_junipr/config.py).

- [0. How configuration works](#0-how-configuration-works)
- [1. `geometry` — the Lund-plane grid](#1-geometry--the-lund-plane-grid)
- [2. `data` — dataset & split](#2-data--dataset--split)
- [3. `encoder` — the context network e(x)](#3-encoder--the-context-network-ex)
- [4. `model` — the posterior family](#4-model--the-posterior-family)
- [5. `optim` — optimizer & schedule](#5-optim--optimizer--schedule)
- [6. `trainer` — the training loop](#6-trainer--the-training-loop)
- [7. `decode` — inference / MAP / posterior knobs](#7-decode--inference--map--posterior-knobs)
- [8. `experiment` — evaluation suite](#8-experiment--evaluation-suite)
- [9. Top-level fields](#9-top-level-fields)
- [10. Inference knobs in depth — the MAP floor, mincut & quantile floor](#10-inference-knobs-in-depth--the-map-floor-mincut--quantile-floor)
- [11. Defined-but-not-wired fields](#11-defined-but-not-wired-fields)

---

## 0. How configuration works

Config is **OmegaConf** (no Hydra). Two kinds of CLI token:

- **Group selectors** `group=name` pick a YAML file from `configs/<group>/<name>.yaml`
  and (for `model`/`encoder`) bind a specific schema. Groups:
  `geometry data model encoder optim trainer decode experiment`.
- **Dotted overrides** `a.b.c=value` set an individual field. Everything is
  **type-checked against the schema** and **unknown keys are rejected at load** (a typo
  like `optim.lrr=1e-3` fails immediately, not three hours in).

```bash
h2p-rsd-junipr train model=cinn encoder=lundnet \
    encoder.num_layers=3 geometry.n_bins=16 optim.lr=1e-3 trainer.max_epochs=100
```

Defaults live in the dataclasses in `config.py` and the per-group YAML in `configs/`.
The resolved config is hashed (`config_hash`, 10 chars) into the run-dir name and stored
in every checkpoint, so the architecture is reproducible and resume refuses silent drift.
Adding/altering any field changes the hash for *new* runs; old checkpoints still load via
the tolerant `decode_params()` / `OmegaConf.select` backfill.

> **Three different `max_emissions`.** They are independent caps — don't confuse them:
> `data.max_emissions` (synthetic *truth* length cap), `model.max_emissions` (cINN/diffusion
> multiplicity-head width), and `decode.max_emissions` (beam/sample length cap). Each is
> documented in its own section.

---

## 1. `geometry` — the Lund-plane grid

The primary Lund plane `(ln 1/ΔR, ln k_t)` is discretised into an `n_bins × n_bins` grid
of **cells**; the cell id is the discrete target the models predict, and the within-cell
offset bounds for the v2 continuous head are *derived* from this grid (never set
independently). Source: [`geometry.py`](../src/h2p_rsd_junipr/geometry.py).

| Field | Default | Meaning |
|---|---|---|
| `ln_invdelta_range` | `[0.0, 6.0]` | min/max of `ln 1/ΔR` (inverse opening angle) spanned by the grid |
| `ln_kt_range` | `[0.0, 6.0]` | min/max of `ln k_t` (the perturbative ordering variable) |
| `n_bins` | `10` | bins per axis ⇒ **`n_cells = n_bins²`** (100 by default) |

Increasing `n_bins` sharpens the discretised likelihood but widens the categorical cell
head (output width `= n_cells`) and shrinks each cell, so the within-cell continuous head
has to do less work. The grid bounds also act as a perturbative window: anything outside
`ln_kt_range` is clipped, keeping inference in the hadronization-robust band.

---

## 2. `data` — dataset & split

Source selection, the deterministic train/val split, and the data fingerprint that ties a
run to its data. Sources: [`datamodule.py`](../src/h2p_rsd_junipr/data/datamodule.py).

| Field | Default | Meaning |
|---|---|---|
| `source` | `"synthetic"` | `synthetic` (built-in simulator) or `rntuple` (a `jets.root` file) |
| `path` | `"jets.root"` | RNTuple path when `source=rntuple` |
| `ntuple` | `"Jets"` | RNTuple name inside the ROOT file |
| `n_jets` | `8000` | number of jets the **synthetic** simulator generates |
| `seed` | `0` | synthetic-generation **and** split seed (deterministic either way) |
| `val_fraction` | `0.1` | target validation fraction; `n_val = max(min_val, ⌊len/round(1/val_fraction)⌋)` |
| `min_val` | `200` | floor on the validation count (lower it for tiny runs, e.g. `min_val=32`) |
| `cache_dir` | `null` | if set, preprocessed tensors are cached at `<cache_dir>/jets_<fingerprint>.pt` |
| `max_emissions` | `20` | cap on the synthetic **truth** parton-sequence length (simulator only) |

**Split semantics.** When the data carries `event` ids (real RNTuples), the split is **by
event** so jets of one event never straddle train/val; otherwise it is a deterministic
*trailing* split (`jets[:-n_val]` / `jets[-n_val:]`), matching the original v2 script.
`min_val` wins when `n_jets` is small — set it below `n_jets` or you get an empty train
split.

---

## 3. `encoder` — the context network e(x)

Any encoder maps the hadron-level primary Lund sequence to a context vector `e(x)` of
width `model.ctx_dim`, and **any encoder pairs with any model family**. Select with
`encoder=gru|lundnet|deepsets`. `emb_dim` and `hidden_dim` are shared conventions across
all three; the rest are per-encoder. Sources:
[`encoders/`](../src/h2p_rsd_junipr/encoders/).

Common fields:

| Field | Default | Meaning |
|---|---|---|
| `emb_dim` | `32` | node-feature embedding width (in AR, **also** the decoder y-token embedding width) |
| `hidden_dim` | `64` | the encoder's internal hidden width (the output is projected to `ctx_dim`) |
| `num_layers` | varies | encoder depth (the "encoder depth" knob) |
| `dropout` | `0.1` | dropout inside the encoder |

Per-encoder:

| Encoder | Extra field | Default | Meaning |
|---|---|---|---|
| `gru` | `bidirectional` | `True` | bi-GRU over the sequence; `num_layers` default `1` |
| `lundnet` | `k` | `4` | EdgeConv neighbourhood size (LundNet graph net); `num_layers` default `3` |
| `deepsets` | — | — | permutation-invariant Deep Sets; `num_layers` default `2` |

---

## 4. `model` — the posterior family

The polymorphic group: `model=ar_junipr_v2|ar_junipr_v1|cinn|diffusion` binds a specific
schema. All families expose the same `log_prob`/`sample`/`map_estimate` contract.
`ctx_dim` is the context width the encoder must produce (the encoder is built with this as
its output dim). Sources: [`models/`](../src/h2p_rsd_junipr/models/).

### `ar_junipr_v2` (recommended) / `ar_junipr_v1`

Autoregressive RSD-JUNIPR: a 3-head decoder (continue/stop, cell, and — v2 only —
continuous coordinates) over the parton tree.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width `e(x)` (encoder output dim) |
| `dec_dim` | `64` | GRU decoder hidden width |
| `dec_layers` | `1` | decoder depth (builds `h0` of shape `(dec_layers, B, dec_dim)`) |
| `split_head_layers` | `2` | depth of the cell (split) head MLP |
| `coord_head_layers` | `2` | depth of the continuous-coordinate head MLP |
| `continuous_coords` | `True` | **v2** ⇒ `True` (adds the within-cell coordinate density); **v1** ⇒ `False` (discrete cells only) |
| `sigma_floor` | `1e-2` | floor added to the predicted std of the truncated-normal / normal coordinate densities (stability) |
| `kappa_max` | `50.0` | cap on the von Mises concentration κ for the periodic ψ coordinate |
| `cell_label_smoothing` | `0.0` | label smoothing on the split-head target; `0.0` keeps likelihood parity (a probe knob for the MAP collapse) |

`ar_junipr_v2` vs `ar_junipr_v1` is exactly `continuous_coords` True vs False — v1 drops
the coordinate density and is the categorical-cell-only backbone.

### `cinn` — conditional normalizing flow

`P(n|e)·∏P(cell|e)·∏p_flow(coords|e,cell)` with a RealNVP over the 4 coordinates.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width |
| `n_blocks` | `6` | RealNVP affine-coupling blocks |
| `hidden_dim` | `64` | coupling-network hidden width |
| `max_emissions` | `25` | multiplicity-head width (categorical over `n=0..max_emissions`) |
| `sigma_floor`, `kappa_max` | `1e-2`, `50.0` | carried for schema symmetry; **not used** by the RealNVP flow (see §11) |

### `diffusion` — conditional diffusion

Categorical `n`/cell heads + a variance-preserving diffusion over the 4 coordinates.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width |
| `hidden_dim` | `64` | denoiser MLP width |
| `n_steps` | `50` | diffusion (DDPM) steps for the reverse process |
| `max_emissions` | `25` | multiplicity-head width (categorical over `n=0..max_emissions`) |

---

## 5. `optim` — optimizer & schedule

AdamW + an optional cosine schedule. Source:
[`trainer.py`](../src/h2p_rsd_junipr/train/trainer.py).

| Field | Default | Meaning |
|---|---|---|
| `lr` | `2e-3` | AdamW learning rate |
| `weight_decay` | `3e-4` | AdamW weight decay |
| `scheduler` | `"cosine"` | `cosine` (CosineAnnealingLR over `trainer.max_epochs`) or `none`/`null` (constant LR) |
| `eta_min` | `3e-4` | cosine floor LR (ignored when `scheduler=none`) |
| `grad_clip` | `1.0` | gradient-norm clip |

---

## 6. `trainer` — the training loop

The framework-free epoch loop: weighted NLL, scheduling, checkpointing.

| Field | Default | Meaning |
|---|---|---|
| `max_epochs` | `20` | training epochs (also the cosine `T_max`) |
| `batch_size` | `64` | minibatch size |
| `seed` | `0` | global seed (torch / numpy / python) |
| `amp` | `False` | mixed precision; **off** because the model is overhead-bound |
| `compile` | `False` | `torch.compile(mode="reduce-overhead")` |
| `fast_dev_run` | `False` | ~2-step smoke path (CI); also via `trainer=fast_dev` |
| `num_workers` | `0` | DataLoader worker processes |
| `resume_from` | `null` | path to a `last.ckpt` to resume an interrupted run |
| `deterministic` | `True` | set cuDNN deterministic / disable benchmark |
| `ema_decay` | `null` | **reserved — not currently wired into the loop** (see §11) |

Device is auto-selected (CUDA → MPS → CPU); there is no device knob. Resume requires the
same architecture (a `config_hash` mismatch is a hard error) and continues to the original
`max_epochs` — changing `max_epochs` changes the hash, so you cannot *extend* a finished
run by resuming (start a fresh run instead). See [`USAGE.md` §3](USAGE.md#3-resume-a-pre-empted-run).

---

## 7. `decode` — inference / MAP / posterior knobs

**These are inference-time only** — they never touch the trained likelihood, so you can
A/B them on a fixed checkpoint. At `eval`, the checkpoint's snapshot decode is the default,
but an explicit CLI `decode.*` override wins (e.g. `eval <ckpt> decode.length_floor_quantile=0.9`).
The serving layer reads the checkpoint's decode config. Source:
[`point_estimate.py`](../src/h2p_rsd_junipr/inference/point_estimate.py),
[`sampling.py`](../src/h2p_rsd_junipr/inference/sampling.py),
[`length.py`](../src/h2p_rsd_junipr/inference/length.py).

| Field | Default | Affects | Meaning |
|---|---|---|---|
| `beam_width` | `8` | MAP | beam size for the MAP cell-structure search |
| `topk_cells` | `6` | MAP | candidate cells expanded per beam step |
| `max_emissions` | `25` | MAP + posterior | hard cap on decoded / sampled tree length |
| `n_posterior_samples` | `500` | posterior | default number of posterior draws |
| `cont_temperature` | `1.0` | posterior | softmax temperature on the cell logits at **sampling** time (exposure-bias remedy); `>1` flattens, `<1` sharpens |
| `min_emissions` | `1` | MAP | **hard floor** on MAP length — the "mincut" (never the unphysical empty tree) |
| `length_penalty` | `0.0` | MAP | GNMT `score/len**α` at final beam rank; counters the brevity bias; `0` = off |
| `length_floor_quantile` | `0.0` | MAP | **learned per-jet floor** at the α-quantile of `P(n|x)`; `0` = off |

`min_emissions`, `length_penalty`, and `length_floor_quantile` are explained in depth in
§10 — they are the knobs that decide what multiplicity the **point estimate** reports.

---

## 8. `experiment` — evaluation suite

Controls the §8 closure / calibration / systematic run (`h2p-rsd-junipr eval`).

| Field | Default | Meaning |
|---|---|---|
| `name` | `"default"` | experiment label |
| `closure_jets` | `300` | held-out jets evaluated in the closure/calibration loop |
| `n_closure_samples` | `200` | posterior draws **per jet** inside that loop |
| `generator_b` | `null` | a second generator/checkpoint for the PYTHIA-vs-HERWIG systematic (the dominant uncertainty) |

Trade cost vs. precision with `closure_jets` and `n_closure_samples`.

---

## 9. Top-level fields

| Field | Default | Meaning |
|---|---|---|
| `run_name` | `"${model.name}_${encoder.name}"` | interpolated run label (e.g. `cinn_lundnet`) |
| `run_root` | `"runs"` | where run directories are written |

---

## 10. Inference knobs in depth — the MAP floor, mincut & quantile floor

The **MAP** is the *joint mode* `ŷ = argmax_y q_φ(y|x)`. For a discrete autoregressive
posterior it is **length-biased low**: every emission pays the cell head's categorical
entropy while "stop" costs a roughly fixed amount, so for higher-multiplicity jets the
single most-probable explicit tree scores *below* the empty tree, and the un-floored
argmax collapses to **0 splittings** — unphysical (a groomed jet has ≥1 primary
splitting). Three decode knobs address this, from bluntest to most learned:

### `min_emissions` — the hard floor ("mincut")

A constant lower bound on the returned MAP length. The beam search never records a STOP
shorter than `min_emissions`, so the MAP always has ≥ `min_emissions` splittings (cINN /
diffusion clamp their multiplicity head identically: `n* = max(argmax P(n|x), min_emissions)`).
Default **1** removes the empty-tree collapse. `min_emissions=0` reproduces the raw,
un-floored argmax (and may collapse).

### `length_penalty` — GNMT length normalization

Ranks finished beam hypotheses by `score / len**α` instead of raw `score`, so longer
trees are not unfairly penalized by the un-normalized sum of log-probs. `α=0` is the raw
score (default); larger `α` favors longer trees. Pruning *within* a step stays on raw
score (candidates there share a length).

### `length_floor_quantile` — the learned, per-jet floor (the "quantile floor")

`min_emissions` is a single global constant; this generalizes it into a **per-jet** bound
read from the model's *own* learned length belief `P(n|x)`. That belief is unbiased in
length (unlike the joint argmax), so flooring the MAP at a low quantile of it transfers
the belief into the point estimate and cuts the residual under-count. The effective floor
passed to the decoder is

```
eff = max(min_emissions, ⌊Q_α(P(n|x))⌋)
```

where `Q_α` is the smallest `n` with `cdf(n) ≥ α` — i.e. α picks a point on the per-jet
**CDF** of `P(n|x)`, and the MAP is forced to emit at least that many splittings. `P(n|x)`
comes from the cINN/diffusion multiplicity head **exactly**, and from the histogram of
posterior draws for the AR model (reusing draws the caller already took — no double-sample).

#### What different α values mean

Take one jet whose belief is `P(n|x) = [0, .4, .1, .1, .1, .1, .1, .1]` (mode at `n=1`, a
long right tail). Its CDF is `[0, .4, .5, .6, .7, .8, .9, 1.0]`, so α walks *up* the CDF:

| α | `Q_α` (first `n` with cdf ≥ α) | `eff = max(1, Q_α)` | MAP length |
|---|---|---|---|
| `0.0` | — (short-circuits; pmf never read) | 1 | **off** → unchanged |
| `0.3` | 1 | 1 | unchanged |
| `0.5` | 2 | 2 | ≥ 2 |
| `0.6` | 3 | 3 | ≥ 3 |
| `0.9` | 6 | 6 | ≥ 6 |
| `0.99` | 7 | 7 | ≥ 7 |

#### The dial, in words

- **`α = 0.0` (default, off).** Short-circuits before reading `P(n|x)` → `eff = min_emissions`
  (the plain hard floor). Bit-for-bit today's MAP.
- **`α` small (≈0.05–0.2).** Floors at the *low tail* of the belief — only raises the MAP
  for jets whose pessimistic length estimate already exceeds the hard floor. Rescues the
  worst under-counts, leaves confident jets alone. Gentle, conservative.
- **`α = 0.5`.** Floors at the per-jet **median** of `P(n|x)`: "the MAP should count at
  least as many splittings as the model's median draw." A strong, well-motivated correction.
- **`α` high (≈0.8–0.95).** Floors near the upper belief; the MAP is pushed toward the high
  end of plausible multiplicities — approaching a **length-conditioned MAP** (the MAP cell
  structure, length pinned high).
- **`α → 1.0`.** Floors at (near) the top of the support. Aggressive; bounded above by
  `max_emissions` / `n_cells` (a floor past those is absorbed by the beam's degenerate
  fallback, so it never crashes).

#### Two properties worth internalizing

1. **It only ever *raises* the count** (`eff = max(min_emissions, …)`), so the `n ≥ 1`
   guarantee holds and `n=0` stays 0% at every α. For a fixed jet, MAP multiplicity is
   non-decreasing in α.
2. **It is genuinely per-jet.** A jet with a *narrow, confident* `P(n|x)` barely moves as α
   changes; a jet with a *broad* belief moves a lot. A global constant can't adapt — this
   does.

A worked A/B on a trained checkpoint (one jet whose `P(n|x)` has median 6, tail to 8):

```bash
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.length_floor_quantile=0.0   # MAP = 6
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.length_floor_quantile=0.5   # MAP = 6 (≈ median)
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.length_floor_quantile=0.9   # MAP = 7 (floored up)
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.length_floor_quantile=0.99  # MAP = 8 (floored up more)
```

#### How to choose it

- Leave **`0.0`** unless the MAP is *under*-counting. The diagnostic is the closure
  multiplicity bias `⟨n − n_true⟩` for the MAP (from `eval`): if it is negative, raising α
  reduces it.
- **`0.1–0.5`** is the useful range for cutting a residual under-count; `0.5` (median) is
  the natural middle ground.
- **It cuts both ways:** if a model already *over*-counts (e.g. an under-trained checkpoint
  where MAP = 6 vs truth = 5), raising α makes it **worse**. Tune against the closure bias,
  don't set it blind.

In a script you can compute the floor yourself and reuse your draws (see
[`USAGE.md` §5.2](USAGE.md#52-infer-from-a-single-hadron-level-lund-sequence)):

```python
from h2p_rsd_junipr.inference.length import learned_min_emissions

draws = model.sample(xf, nx, n=500)
mult  = [len(d) for d in draws]
eff   = learned_min_emissions(model, xf, nx, quantile=0.15, base_floor=1, mults=mult)
mp    = model.map_estimate(xf, nx, min_emissions=eff)   # eff = max(1, ⌊Q_0.15(P(n|x))⌋)
```

> **For a count, still prefer the posterior median.** The MAP — even floored — is the
> length-biased joint mode; the posterior median sidesteps both the mode collapse and the
> mean's tail sensitivity. The floor knobs make the MAP *usable*, not optimal, as a count.
> Quantified in `notebooks/inference_demo.ipynb` §6a and `scripts/probe_map_collapse.py`.

### `cont_temperature` — posterior sampling temperature

Applies a softmax temperature to the cell logits during **ancestral sampling** only (the
posterior, not the MAP). It is the documented exposure-bias remedy for an over-counted
posterior multiplicity: `>1` flattens the cell distribution (more diverse draws), `<1`
sharpens it. It never touches the trained likelihood.

---

## 11. Defined-but-not-wired fields

For honesty, these schema fields exist but are **not consumed** by the current code:

- **`trainer.ema_decay`** — the checkpoint format reserves an `ema` slot, but the training
  loop never builds or passes an EMA model, so this value has no effect today.
- **`cinn.sigma_floor` / `cinn.kappa_max`** — carried for symmetry with the AR coordinate
  head, but the cINN's RealNVP flow does not use truncated-normal/von-Mises densities, so
  these are inert for `model=cinn`.

Everything else listed above is live. The schema is the source of truth
([`config.py`](../src/h2p_rsd_junipr/config.py)); when in doubt, grep the field name.
