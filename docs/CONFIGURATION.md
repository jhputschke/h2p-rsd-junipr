# Configuration reference — every knob, what it means, when to turn it

This is the complete field-by-field reference for the config schema. It complements
[`USAGE.md`](USAGE.md) (which shows *how to run* the workflow) by explaining *what each
parameter does* — including the inference-time **decode** knobs (`min_emissions`,
`length_floor_quantile`, `length_penalty`, `cont_temperature`) that shape the MAP and
the posterior, and the **`point_estimator` / `mbr_*`** knobs that select and configure the
minimum-Bayes-risk (MBR) point estimate. For the physics behind these choices see
[`README_PHYSICS.md`](README_PHYSICS.md); the schema source of truth is
[`src/h2p_rsd_junipr/config.py`](../src/h2p_rsd_junipr/config.py).

> **New in the post-review update** ([`PLAN_UPDATES.md`](PLAN_UPDATES.md)): the exact-likelihood
> `model=cfm` family and the `exact_likelihood` flag (§4), the calibration suite v2 switches
> `experiment.pit_coords` / `.stratify_regions` / `.tarp` (§8), the `model.max_emissions`
> support guard (§4), the v3 knob-semantics table (§7), and cross-attention conditioning
> `model.use_cross_attention` / `model=ar_junipr_v4` (§4). Everything defaults off or absent,
> so existing runs and numbers are unchanged.

- [0. How configuration works](#0-how-configuration-works)
- [1. `geometry` — the Lund-plane grid](#1-geometry--the-lund-plane-grid)
- [2. `data` — dataset & split](#2-data--dataset--split)
- [3. `encoder` — the context network e(x)](#3-encoder--the-context-network-ex)
- [4. `model` — the posterior family](#4-model--the-posterior-family)
- [5. `optim` — optimizer & schedule](#5-optim--optimizer--schedule)
- [6. `trainer` — the training loop](#6-trainer--the-training-loop)
- [7. `decode` — inference / MAP / posterior knobs](#7-decode--inference--map--posterior-knobs)
- [8. `experiment` — evaluation suite](#8-experiment--evaluation-suite)
- [8a. `audit` — the mode-mass audit's search](#8a-audit--the-mode-mass-audits-search)
- [9. Top-level fields](#9-top-level-fields)
- [10. Inference knobs in depth — the MAP floor, mincut & quantile floor](#10-inference-knobs-in-depth--the-map-floor-mincut--quantile-floor)
- [11. Defined-but-not-wired fields](#11-defined-but-not-wired-fields)

---

## 0. How configuration works

Config is **OmegaConf** (no Hydra). Two kinds of CLI token:

- **Group selectors** `group=name` pick a YAML file from `configs/<group>/<name>.yaml`
  and (for `model`/`encoder`) bind a specific schema. Groups:
  `geometry data model encoder optim trainer decode experiment`. A name with no matching
  file is a **hard error** (`FileNotFoundError`, listing what is available) — silently
  falling back to the schema defaults would turn `model=ar_junipr_v3` into a plain v2.
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

### Custom setups without a long CLI chain

Two ways to keep a run configuration in YAML instead of retyping overrides.

**Your own group file.** Every group YAML is a *patch* on the dataclass defaults, not a full
config — write only the fields you change, drop it in `configs/<group>/`, and select it:

```yaml
# configs/decode/mbr_study.yaml
point_estimator: mbr
mbr_backend: pot
mbr_lnkt_cut: ${geometry.ln_kt_range[0]}   # interpolation across groups works
beam_width: 16
```
```bash
h2p-rsd-junipr train decode=mbr_study decode.beam_width=4   # CLI still wins per field
```

The file may only contain that group's own fields (`optim:` inside an `experiment/` file is
rejected), lists replace wholesale, and interpolations resolve *after* the CLI, so
`data.max_emissions=12` propagates into a `${data.max_emissions}` reference. `model` and
`encoder` are polymorphic: a new file there also needs an entry in `MODEL_SCHEMA` /
`ENCODER_SCHEMA` (and, for a model, a name in `@register_model`).

**A custom top-level config, `base=<path>`.** Selects several groups at once — the
cross-group setup a single group file cannot express:

```yaml
# presets/mbr_study.yaml — only what differs from configs/config.yaml
defaults:                  # picks WHICH group files to load
  model: ar_junipr_v3
  encoder: lundnet
  decode: mbr_study        # resolved from presets/decode/ first, then configs/decode/

model:                     # sets VALUES — a patch, like a group file, no file needed
  dec_dim: 128
optim:
  lr: 1.0e-3
geometry:                  # a group you did not re-select can be tuned too
  n_bins: 16
run_root: runs/mbr_study
```
```bash
h2p-rsd-junipr train base=presets/mbr_study.yaml optim.lr=1e-3
```

That exact pair ships in the repo — [`presets/mbr_study.yaml`](../presets/mbr_study.yaml) with
its group file [`presets/decode/mbr_study.yaml`](../presets/decode/mbr_study.yaml) — as a
working template to copy.

It is layered **over** `configs/config.yaml`, so unlisted groups keep the repo default, and
its directory becomes a group-file root searched **before** `configs/` — a `presets/<group>/`
subdir can add new names or shadow existing ones while everything else is inherited. Only the
**first** match is loaded, so a shadowing file replaces the repo file of the same name rather
than merging on top of it.

Every top-level block other than `defaults:` is deep-merged into the config as a **value
override**, schema-checked like everything else (`optim: {lrr: …}` fails at load, and
`dec_dim` under a `model: cinn` preset fails as `Key 'dec_dim' not in 'CINNConfig'`). Untouched
fields keep the group file's value, so the block above still gets `optim.weight_decay` from
`configs/optim/default.yaml`. A `defaults:` block is optional. CLI tokens still win. Precedence,
per field:

```
dataclass default → the winning <group>/<name>.yaml (base dir before configs/)
                  → inline block in the base file → CLI a.b=v
```

> **Pick families through `defaults:`, never inline.** `model: {name: cinn}` in a base file is
> silently ignored — `cfg.model.name` is re-set from the selector after the merge. An inline
> `encoder: {name: deepsets}` is worse: it changes which class `build_encoder` constructs while
> the schema stays bound to the *selector's* encoder, so you get one encoder's knobs on another's
> implementation. Selector picks the family; inline blocks tune values.

> **Three different `max_emissions`.** They are independent caps — don't confuse them:
> `data.max_emissions` (synthetic *truth* length cap), `model.max_emissions` (multiplicity-head
> width for cINN / diffusion / `cfm` / `ar_junipr_v3+`, and the one the §4 **support guard**
> checks the data against — for `edit_v1/v2` it is only the exact-`q(N|x)` readout width and
> the guard does not apply), and `decode.max_emissions` (beam/sample length cap). Each is
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
| `pt_var` | `"jet_pt"` | which pT the window below cuts on: `jet_pt` (ungroomed) or `x_ptg` (groomed) |
| `pt_min` | `null` | GeV, **inclusive** lower edge of the jet-pT window; `null` == unbounded |
| `pt_max` | `null` | GeV, **exclusive** upper edge of the jet-pT window; `null` == unbounded |

**Split semantics.** When the data carries `event` ids (real RNTuples), the split is **by
event** so jets of one event never straddle train/val; otherwise it is a deterministic
*trailing* split (`jets[:-n_val]` / `jets[-n_val:]`), matching the original v2 script.
`min_val` wins when `n_jets` is small — set it below `n_jets` or you get an empty train
split.

### `pt_min` / `pt_max` — training on one slice of the jet spectrum

Keeps only the jets with `pt_min <= pt < pt_max`, applied to the list `load_rntuple`
returns, *before* the fingerprint and *before* the train/val split — so both splits see
the same window and the run's data hash records it.

```bash
# a 100-150 GeV ungroomed slice (27.9% of cpp/test_data/jets_aux.root)
h2p-rsd-junipr train data=rntuple data.path=cpp/test_data/jets_aux.root \
    data.pt_min=100 data.pt_max=150

# the same cut on GROOMED momentum instead
h2p-rsd-junipr train data=rntuple data.path=cpp/test_data/jets_aux.root \
    data.pt_var=x_ptg data.pt_min=50
```

- **Both bounds `null` is the off path, and off is byte-identical**: no jet is dropped,
  the same list object flows through, and the fingerprint hashes exactly as it did before
  the knob existed. The window is mixed into the hash only when it is active — two
  different windows can leave the same jet count with the same leading jets, which the
  length + content-sample terms alone would not tell apart.
- **The upper edge is exclusive**, so adjacent windows (`[40,60)`, `[60,100)`) tile a
  sample without any jet landing in both.
- **`pt_var` accepts `jet_pt`/`pt` or `x_ptg`/`ptg`/`pt_g`** and nothing else; a typo
  names the alternatives rather than silently matching no column.
- **At `eval` the window comes from the checkpoint's config snapshot**, so a model
  trained on a slice is evaluated on that same slice unless you override it explicitly.
- **A jet with no finite `pt_var` is dropped** — synthetic jets carry no pT and an
  RNTuple written before the aux columns stores the NaN sentinel
  ([rntuple.py](../src/h2p_rsd_junipr/data/rntuple.py)). If *every* jet is unset the
  loader raises and names the column, rather than handing back an empty dataset. A window
  that keeps 0 jets, or too few to leave a non-empty train split after `min_val`, also
  raises and names the knob to move.

> **What this is for.** [PLAN_jet_xsection.md](PLAN_jet_xsection.md) §2 measures that the
> training sample is `pTHat`-sculpted rather than steeply falling, and §3 that the
> conditional genuinely moves with pT (⟨n_y⟩ at fixed `n_x` swings 73% across the range).
> A window makes the single-slice experiment — train in one pT band, evaluate in it or
> across it — possible without regenerating anything. It is a **selection**, not a
> reweighting: it narrows the sample, it does not restore a physical spectrum (§6).

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
| `num_layers` | varies | encoder depth (the "encoder depth" knob); for `gru`, raising it is measured to be neutral-to-harmful — see below |
| `dropout` | `0.1` | dropout inside the encoder |
| `aux_features` | `[]` | groomed per-jet conditioning scalars appended to every node of `xf` (below) |

Per-encoder:

| Encoder | Extra field | Default | Meaning |
|---|---|---|---|
| `gru` | `bidirectional` | `True` | bi-GRU over the sequence; `num_layers` default `1` |
| `lundnet` | `k` | `4` | EdgeConv neighbourhood size (LundNet graph net); `num_layers` default `3` |
| `deepsets` | — | — | permutation-invariant Deep Sets; `num_layers` default `2` |

### `encoder.num_layers` — why the `gru` default is `1`

> **Measured, and it is not a free knob: depth 2 is noise, depth 3+ is a regression.**
> `model=ar_junipr_v3 encoder=gru`, 15 epochs, 3 seeds per point, on **PYTHIA**
> (`cpp/test_data/jets.root`, `z_cut=0.1`, 1 GeV `k_t` floor) with **`aux_features=[]`** —
> i.e. the tightly-groomed sample and the aux-off input width of 5 columns per node. Both
> qualifiers matter; see the caveats below. Best val NLL/jet, and the *paired* per-seed
> differences against `num_layers=1`:
>
> | `num_layers` | encoder params | model params | mean | seed spread | paired Δ vs `1` |
> |---|---:|---:|---:|---:|---|
> | `1` (default) | 47.2k | 117.2k | **4.5973** | 0.035 | — |
> | `2` | 121.7k | 191.7k | 4.5825 | 0.040 | −0.032, **+0.027**, −0.039 |
> | `3` | 196.2k | 266.2k | 4.6859 | **0.193** | −0.023, +0.134, +0.154 |
> | `4` | 270.7k | 340.7k | 4.7846 | 0.038 | +0.171, +0.228, +0.163 |
> | `1`, `hidden_dim=128` | 142.2k | 212.2k | 4.5918 | **0.020** | +0.004, +0.002, −0.022 |
>
> Read against the **0.029-nat seed spread** of the aux A/B (`runs/aux_input_ab/ab_summary.json`
> — `runs/` is gitignored, so that file is local to whoever ran it), which is
> the noise floor for this data/epoch budget. `num_layers=2` improves the mean by 0.015 —
> half the floor — and its paired differences *change sign* across seeds, so it buys nothing
> for +64% model size. `3` is 0.089 worse **and** its seed spread explodes to 0.193, 5× the
> spread at `1`: an encoder that has become a seed lottery rather than a better one. `4` is
> 0.187 worse with all three paired differences positive — no seed ambiguity left.
>
> **It is recurrent depth that hurts, not capacity.** The last row spends *more* parameters
> than `num_layers=2` on width instead of layers and comes out flat, with the tightest spread
> in the scan. Extra capacity here is merely wasted; stacking it as recurrent layers is
> destructive.
>
> Same mechanism as the `ar_junipr_v4` cross-attention result below: at this grooming the
> encoder input is a **median of 2 nodes** (mean 1.74, p99 4, 6.9% empty), so layer 2+ has no
> temporal structure left to compose, and
> [`encoders/gru.py`](../src/h2p_rsd_junipr/encoders/gru.py) mean-pools the result to one
> vector regardless.
>
> **Revisit this under looser grooming.** Sequence length is set by `z_cut` / `β` / the `k_t`
> floors, and it is the *only* reason depth fails here — lower the floor or `z_cut` and the
> mean multiplicity rises until stacked layers have something to compose (`data=synthetic`,
> mean ~6, is the existing long-sequence reference: it is where cross-attention won by 3.8
> nats while losing on this sample). So do not carry `num_layers=1` over as a settled default
> to a looser-groomed file; re-run the scan there. Same for turning `aux_features` on: the
> aux columns widen every node's input, which is more per-node structure to combine and a
> case this scan does not cover.
>
> Two further scoping caveats. The LR schedule and 15-epoch budget were held **fixed** across
> arms, so part of the `3`/`4` degradation is optimization difficulty, not pure capacity
> waste — a retuned schedule would likely recover some of it, but the ceiling is still `1`,
> so parity is the best case for 2–3× the parameters. And this measures the **`gru`** knob
> only: `lundnet` layers are graph message-passing rounds and `deepsets` layers are not
> recurrent over position at all, so neither default (`3`, `2`) is implicated.
>
> **Regenerating the table.** No run directory is committed (`runs/` is gitignored), but the
> input is — `cpp/test_data/jets.root` is in the repo, so the scan reproduces from a clean
> clone. The 12 depth runs are one `sweep.py` grid, and the width control is a second:
>
> ```sh
> PYTHONPATH=src python scripts/sweep.py \
>     model=ar_junipr_v3 encoder=gru data=rntuple data.path=cpp/test_data/jets.root \
>     trainer.max_epochs=15 encoder.num_layers=1,2,3,4 trainer.seed=0,1,2
>
> PYTHONPATH=src python scripts/sweep.py \
>     model=ar_junipr_v3 encoder=gru data=rntuple data.path=cpp/test_data/jets.root \
>     trainer.max_epochs=15 encoder.num_layers=1 encoder.hidden_dim=128 trainer.seed=0,1,2
> ```
>
> Each point lands in its own `runs/<stamp>-<config_hash>/`; the table's entry is
> `min(val_nll)` over that run's `metrics.csv`, i.e. the same `best val NLL/jet` the trainer
> prints on exit. Add `run_root=<dir>` to keep a scan's runs together. ~3 min per run on an
> M-series `mps` device, so ~45 min for all 15.

### `aux_features` — groomed all-branch conditioning

The encoder input is the **primary** Lund sequence only: everything inside the softer
prongs — the secondary Lund planes — is discarded at write time, so two
conditioning-relevant quantities can never be functions of `x`. `aux_features` opts them
back in (see [`PLAN_Input.md`](PLAN_Input.md)):

| Name | Value | Why it is not a function of `x` |
|---|---|---|
| `ln_mg_pt` | `ln(max(x_mg, 1e-3) / jet_pt)` | every primary node is recorded **massless**; the subjet masses making up `m_g` live in the discarded prongs |
| `ln_ptg_pt` | `ln(min(x_ptg / jet_pt, 1))` | how much **momentum** grooming removed; `x` records `z` only at the *kept* splittings, so the dropped fraction is unreconstructable |
| `nsec` | `log1p(x_nsec)` | grooming-passing splittings on **non-primary** branches; secondary-plane density carries quark/gluon information (arXiv:2112.09140) |
| `ln_pt` | `ln(jet_pt / 100)` | the scale anchor — already written per jet, never previously read |
| `abs_eta` | `abs(jet_eta) / 2` | at fixed `pt` the q/g fraction varies strongly with rapidity; `x` is entirely intra-jet |
| `has_sec` | `1` if `x_nsec > 0` else `0` | presence indicator gating the four below (see *undefined vs zero*) |
| `ln_kt_sec` | `log1p(x_kt_sec_max)` | hardest **off-spine** splitting — separates one hard secondary prong from several soft ones at equal `n_sec` |
| `ln_kt_sec_sum` | `log1p(x_kt_sec_sum)` | total off-spine hardness; differs from the above only when `n_sec > 1` |
| `sec_depth` | `log1p(x_sec_attach)` | which primary node the hardest secondary hangs off (`0` = widest-angle) |

**Why `ln_ptg_pt` and not the mass drop `ln(m_g/m)`.** The encoder already sees
`ln(m_g/pt)`, so adding `ln(m_g/m)` would be an invertible reparameterization handing it
`ln(m/pt)` — the **ungroomed** mass, exactly what the grooming-first design excludes.
`ln(pt_g/pt)` combined with `ln_pt` instead yields `ln(pt_g)`, a groomed quantity;
nothing ungroomed becomes reconstructable. Measured (medians, 3 000 events, MPI off → on):
`pt` shifts **+0.4 %** as a normalizer where `m` shifts **+9.7 %**, and at ratio level
`pt_g/pt` shifts **−1.6 %** against `m_g/pt` **+6.1 %** and `m_g/m` **−5.3 %** — i.e. the
new feature is the most UE-robust of the three. (That test probes UE only; hadronization
robustness is criterion (iv), still blocked on WP5.)

> **`pt_g/pt` ≈ 0.4, not ≈ 0.95.** This is the *pipeline*-groomed momentum: recursive
> Soft Drop with **no iteration limit** and the `k_t` floor applied, so a collinear-but-hard
> prong (e.g. `z = 0.4` at `ΔR = 0.02` → `k_t = 0.8 GeV`) is discarded where textbook mMDT
> would stop and keep it. Same predicate as `m_g` and as the persisted sequences — one
> grooming definition per file. Consequence worth knowing: the quantity is governed by
> drops near the 1 GeV floor, i.e. the NP boundary, so it is more NP- than UE-sensitive.

**Undefined vs zero.** `x_kt_sec_max` / `x_kt_sec_sum` / `x_sec_attach` are meaningless
when `x_nsec == 0` (82.6 % of the reference sample). The C++ side writes `0`, and the
Python registry maps them to exactly `0` while `has_sec` goes to `0` — so the encoder can
**gate** them rather than read `0` as a measurement. `log1p` is chosen precisely so the
neutral point is `0` and any real value is bounded away from it
(`k_t ≥ k_t^floor ⇒ log1p(k_t) ≥ log(1 + k_t^floor)`). Ship the indicator whenever you
ship any of the three. Their absent-column sentinel is `-1`, not `0`, for the same reason.

All three are **groomed**, so they keep the NP/UE suppression that motivates the pipeline
and stay usable in a heavy-ion environment. Ungroomed observables (constituent
multiplicity, ungroomed mass, girth) are deliberately excluded: IRC-unsafe,
background-sensitive conditioning contradicts the grooming-first design.

```bash
h2p-rsd-junipr train model=ar_junipr_v3 encoder=gru \
    data=rntuple data.path=cpp/test_data/jets_aux.root \
    encoder.aux_features='[ln_mg_pt,nsec,ln_pt]'
```

- **Mechanism.** The scalars are appended as **constant per-node columns of `xf`**, so they
  reach every consumer (`log_prob`, closure, calibration, MBR, serving) through the
  existing `(xf, nx)` plumbing. The only model-side change is the encoder's input width.
- **Parity.** `[]` (the default) is byte-identical: same module list, same `state_dict`,
  same `log_prob`. A checkpoint config predating the field rebuilds as the plain model.
- **Data requirement.** The sources (`jet_pt`, `x_mg`, `x_nsec`) come from the C++ writer.
  A pre-`PLAN_Input` `jets.root` reads them as sentinels (NaN / `-1`) and the dataset
  **raises** rather than training on NaNs. `data.source=synthetic` raises too — the
  synthetic generator has no secondary planes, and any proxy would be a function of `x`,
  faking the very information gain this feature exists to measure.
- **Known limitation.** A jet with `nx == 0` (empty groomed hadron tree; ~7 % of the
  reference PYTHIA sample) has no rows to broadcast onto and carries **no** aux signal.
  `LundDataModule.setup` prints that fraction whenever aux is on.
- **Serving.** A model built with aux requires `aux` in the request body — a dict of the
  raw source columns. The response echoes `aux_features`.
- **A/B coverage.** The measured result below is for the original triple
  `[ln_mg_pt, nsec, ln_pt]` only. `ln_ptg_pt`, `abs_eta` and the four secondary-plane
  features are implemented, tested and available but **have not been through an A/B** —
  treat them as untested until one is run.
- **Status: measured, and NOT adopted.** On `cpp/test_data/jets_aux.root`
  (`ar_junipr_v3 + gru`, 15 epochs, 3 seeds) the held-out NLL/jet goes 4.6136 ± 0.0205 →
  4.5848 ± 0.0202: a −0.029 nat gain against a 0.029 seed spread, with one of three seeds
  going the wrong way. Calibration and closure are unchanged within noise, so nothing is
  broken — there is just no measurable gain to adopt at this grooming working point,
  where **82.6 % of jets have `x_nsec == 0`**. The `n_sec = 2–3` stratum does gain
  −0.100 nats/jet, so the effect is real where the structure exists; raise `⟨n_sec⟩`
  (looser `z_cut`, lower `k_t` floor, higher-`p_T` sample) before re-judging. Full A/B,
  ablation and exit-criteria table:
  [`notebooks/aux_input_ab.ipynb`](../notebooks/aux_input_ab.ipynb); criterion (iv)
  (generator-B / fragmentation-prior spread) is separately blocked on
  `PLAN_UPDATES.md` WP5.

---

## 4. `model` — the posterior family

The polymorphic group:
`model=ar_junipr_v2|ar_junipr_v1|ar_junipr_v3|ar_junipr_v4|cinn|diffusion|cfm|edit_v1|edit_v2`
binds a specific schema. All families expose the same
`log_prob`/`sample`/`map_estimate` contract.
`ctx_dim` is the context width the encoder must produce (the encoder is built with this as
its output dim). Sources: [`models/`](../src/h2p_rsd_junipr/models/).

### Is `log_prob` a density? — `exact_likelihood`

Every family exposes `log_prob`, but they do not all mean the same thing by it. The class
attribute `exact_likelihood` says which:

| Family | `exact_likelihood` | What `log_prob` returns |
|---|---|---|
| `ar_junipr_v1/v2/v3/v4` | `True` | exact: categorical terms + closed-form coordinate densities |
| `cinn` | `True` | exact: change of variables through the RealNVP |
| `cfm` | `True` | exact: probability-flow ODE with an exact 4-VJP divergence |
| `edit_v1/v2` | `True` | exact: the RNN-T lattice normalizes by construction; the alignment is marginalized, not approximated |
| `diffusion` | **`False`** | a denoising-score-matching **surrogate** with an unknown offset |

**Only compare NLLs, and only form likelihood ratios, across the `True` rows.** `train`,
`eval`, and `serve` each print a one-line warning when they report a number from a
surrogate family, so this cannot go unnoticed — but nothing stops you plotting two
incomparable numbers side by side, so it is on you. `diffusion` is kept as the registry's
cheap-sampler baseline; `cfm` is the exact-likelihood member of the same continuous-time
family and is what you want for model selection.

### `ar_junipr_v2` (recommended) / `ar_junipr_v1` / `ar_junipr_v3` / `ar_junipr_v4`

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
| `use_multiplicity_head` | `False` | **v3** ⇒ `True`: promote length to a first-class categorical `q(N\|x)` head (drops the continue/stop head); `False` keeps the implicit per-step continue/stop length model (bit-parity with today) |
| `max_emissions` | `25` | multiplicity-head width (categorical over `n=0..max_emissions`); only used when `use_multiplicity_head=True`. **Guarded**: see "the support guard" below |
| `use_cross_attention` | `False` | **v4** ⇒ `True`: the decoder attends to the encoder's *per-node* hadron states instead of only the pooled `e(x)`; `False` is byte-identical to today |
| `xattn_heads` | `4` | attention heads when `use_cross_attention=True`; must divide `dec_dim` |
| `lnz_support` | `"legacy"` | `legacy` = the unbounded Normal on `ln z`; `physical` = a truncated normal on the interval the grooming actually leaves. See below |
| `lnz_zcut` | `0.1` | the file's soft-drop `z_cut`; read only when `lnz_support="physical"` |
| `lnz_beta` | `0.0` | the file's soft-drop `β`; read only when `lnz_support="physical"` |
| `lnz_head` | `"truncnorm"` | the SHAPE `ln z` may take on that interval: `truncnorm` = the two-parameter truncated normal; `spline` = a monotone rational-quadratic spline composed on its CDF. Requires `lnz_support="physical"`. See below |
| `lnz_spline_bins` | `8` | spline pieces `K`; read only when `lnz_head="spline"`. Costs `3K−1` extra coordinate-head outputs per node (23 at `K=8`) |
| `dv_head` | `"truncnorm"` | the same switch on `dv`, the within-cell `ln k_t` offset — the coordinate that became binding once `ln z` was fixed. **Measured and NOT recommended**: it fails its own pre-registered gate on 3/3 seeds. Bit-identical off. See below |
| `dv_spline_bins` | `8` | spline pieces for the `dv` spline; read only when `dv_head="spline"` |
| `coord_cell_center` | `False` | append the cell's continuous centre `(c_x, c_y)`, affinely mapped onto `[−1, 1]` by the geometry's own ranges, to the coordinate head's input — so the head is told *where* the cell is and not only *which* cell it is. `False` is bit-identical: no extra input, no extra buffer. See below |

**`ar_junipr_v2` vs `ar_junipr_v1`** is exactly `continuous_coords` True vs False — v1 drops
the coordinate density and is the categorical-cell-only backbone.

#### `lnz_support` — the one coordinate head that was on the wrong space

The coordinate likelihood is `TN(du)·TN(dv)·N(ln z)·vM(ψ)`. Three of those four are on
bounded or periodic supports; `ln z` was a plain Normal on all of ℝ. It should not have
been: Soft Drop keeps a splitting only if `z > z_cut (ΔR/R)^β`, and
`z = min(p_{T1},p_{T2})/(p_{T1}+p_{T2}) ≤ ½` by construction, so

    ln z ∈ ( ln z_cut − β·ln(1/ΔR),  ln ½ ].

On the fielded files (`z_cut = 0.1`, `β = 0`) that is `[−2.3026, −0.6931]`, width `ln 5`,
and **both endpoints are attained** — the truth is flush against both walls. A Normal there
necessarily leaks, and production test v0 measured both halves of the leak: PIT KS 0.066
against a 0.016 critical value, and 0.88% of sampled emissions below the `z_cut` boundary
the training data never crosses.

`"physical"` puts the same truncated normal on `ln z` that `du`/`dv` already use. The
sampler, the PIT and the reported mode all inherit the truncation, so the violation rate is
zero *by construction* rather than by a downstream repair.

`"legacy"` is the default and is bit-identical: no new parameter, no new buffer (the bounds
are built from the existing `cell_cx`), and `scripts/verify_parity.py` still matches the
reference v2 script bit-for-bit.

The bound is **cell-conditional, not node-conditional** — evaluated at the `u` inside the
cell that makes it loosest, `lo = ln z_cut − β·c_x − |β|·half_u`. That keeps the coordinate
likelihood a product of independent-given-cell factors; a bound reading the node's own drawn
`u` would couple `ln z` to `du`, which this factorization cannot express. At `β = 0` the two
coincide exactly; for `β ≠ 0` the residual slack is `|β|·half_u`, and the WP-D support audit
*measures* what leaks through it rather than assuming.

`(lnz_zcut, lnz_beta)` are config fields because `build_model` sees only the config, but
they are properties of the **file**. `data.stats.check_lnz_support` runs before training and
checks two things: that the declared pair matches the jets' own grooming record, and that
every truth `ln z` lies inside the resulting interval. The second catches a convention error
— a sign on `β`, an `R ≠ 1` — that matching scalars cannot.

> ⚠️ **NLL is not comparable across the head change.** A different coordinate normalization
> shifts NLL/jet by a constant that has nothing to do with fit quality. Numbers from a
> `physical` run are comparable only to other `physical` runs; the bridge to the older
> record is a `legacy` arm trained on the same data. Never put a `physical` NLL and a
> `legacy` NLL in the same column.
>
> **This extends across families, not just across arms.** The edit transducer carries the
> same three fields (§4, `edit_v1` / `edit_v2` below), and the production-test-edit run
> compares its held-out NLL directly with `ar_junipr_v4`'s continue/stop arm. Both are
> `exact_likelihood = True` and both are densities on the same `(u, v, ln z, ψ)` space, so
> the comparison is legitimate — but only once both sides declare the **same**
> `lnz_support`. Otherwise the delta is the head, not the fit, and it is a large delta:
> `v1_legacy_lnz` came in at 4.0703 against a `v1_base` band of 3.9036–3.9237 — ~0.15 nat,
> on nothing but the head. [`tests/test_nll_comparability.py`](../tests/test_nll_comparability.py)
> pins the cross-family claim; `scripts/prod_test_edit_gates.py` refuses to rank a
> mismatched pair.

#### `lnz_head` — the shape on that interval, once the support is right

`lnz_support` fixed *where* `ln z` lives. Production test v1 then measured that the two
failures were separate: putting `ln z` on its interval removed **every** support violation
(0.83% → 0.0000%) and still left the PIT at **1.05–2.07×** its critical value on three
seeds, concentrated at **2.16×** in the `wide_soft` quadrant that holds 94% of emissions. A
truncation cannot fix a mismatch *inside* the interval — the truncated normal has two free
numbers per node and the residual is the shape beyond them.

`"spline"` puts a monotone rational-quadratic spline `S: [0,1] → [0,1]` (Durkan, Bekasov,
Murray & Papamakarios, [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)) on that
interval through the affine map `t = (x − lo)/(hi − lo)`:

    F(x) = S(t),   p(x) = S′(t)/(hi − lo),   x = lo + (hi − lo)·S⁻¹(u).

The **support closure is kept exactly** — `t` is an affine bijection of the interval onto
`[0,1]` and `S` maps `[0,1]` onto itself, so no draw can leave, by construction rather than
by a clamp — and the **PIT and the sampler come out of one object**, so they cannot drift.
At the raw parameters' zero the widths and heights are uniform and the knot derivatives are
1, so `S` is the identity and the density is **uniform on the interval**: the
maximum-entropy starting point.

> ⚠️ **The base is fixed, and that is a measured decision.** The first implementation
> warped the *truncated normal's* CDF instead, `F(x) = S(F_TN(x))`, so that `truncnorm`
> would be the identity special case. That parameterization is **non-identifiable**: once
> `S` carries the shape, any `(μ, σ)` leaving `F_TN` roughly linear on the interval gives
> the same density, so the pair drifts along a flat direction. It did, and it broke — on
> seed 2 of the first 3-seed run `lnz_mean` reached **−533** against an interval of
> `[−2.303, −0.693]`, `lnz_sig` reached **85**, `F_TN` saturated to 0 or 1 on **100%** of
> emissions, the gradient through `S` died, and val NLL went 4.19 → 19.2 at epoch 4 and
> never recovered. Seeds 0 and 1 were on the same flat direction and had merely not walked
> as far. An affine base has no parameter to run away.
> [`tests/test_lnz_spline.py`](../tests/test_lnz_spline.py) pins the contract: exactly one
> of the two ln z parameterizations is live, and the spline arm carries no learnable base.

The spline **replaces** `(mean, sigma)` rather than adding to them, so the coordinate head
is `6 + (3K−1)` wide — 29 at `K = 8` against `truncnorm`'s 8 — and no output is ever dead.
That is 1 365 parameters, ~1.0% of the model, and nothing else moves. `"truncnorm"` is the
default and is bit-identical: same head width, same `state_dict`, same likelihood, same
PIT, same draws.

`spline` requires `lnz_support="physical"` and **raises** otherwise — on `legacy` there is
no bounded interval to put a spline on, so the pairing is a configuration error rather than
a silently different model.

> The same NLL-comparability warning above applies with full force here: a spline arm's
> NLL is comparable to a `truncnorm` arm's only because both are densities on the same
> space with the same `lnz_support`. Compare seed to seed, and never across `lnz_support`.

#### `dv_head` — the same fix on the coordinate the residual moved to (measured, **not** recommended)

Splining `ln z` relocated the defect rather than removing it: with `lnz_head="spline"` the
`ln z` PIT falls to 0.47–1.04× critical while **`dv`** — the within-cell `ln k_t` offset —
fails on every seed (1.10× / 1.04× / 1.12×). `dv_head` is the same switch on that
coordinate, with `dv_spline_bins` its `K`. The layout keeps the all-truncnorm case
byte-identical: the offset block is `[du_mean, dv_mean, du_sig, dv_sig]` when off and
`[du_mean, du_sig] + <3K−1 spline>` when on, so slots 0–3 never change meaning and an older
checkpoint still reads correctly. `du` deliberately has **no** such flag: it has no measured
defect (0 of 6 seeds).

**It does not work, and the falsification is why the field is documented at all.** The
motivation was a tilt budget: a truncated normal on `[−h, h]` has log-density slope
`μ/σ²`, so it can tilt across the cell by at most `2hμ/σ²`, and `μ` is *clamped* to `±h` by
the `h·tanh` parameterization — a wide `σ` spends tilt authority that cannot be bought
back. Measured on the trained arms, `dv` runs at `σ = 2.6h` and achieves a tilt of 0.158
against a data requirement of ~0.173, while `du` runs at `1.7h` and achieves 0.258. That
predicted a spline — flat and tilted at once — would fix it. It did not:

| gate G3-dv (`dvspline_s*` vs its own same-seed `spline_s*` control) | s0 | s1 | s2 |
|---|---:|---:|---:|
| `dv` PIT KS, × critical | 1.10 → **1.22** | 1.04 → **1.12** | 1.12 → **1.02** |
| `dv × wide_soft` (the bulk cell) | 1.01 → **1.23** | 0.94 → **1.14** | 1.03 → **1.05** |
| val NLL, Δ | **+0.016** | +0.000 | **+0.028** |
| TARP max dev (G7) | 0.0215 → **0.0430** | 0.0265 → **0.0560** | 0.0400 → 0.0315 |

0/3 on both clauses, with seeds 0 and 1 going from *passing* G7 to failing it: worse on two
seeds, unchanged on the third, and never better where it was supposed to be. Re-running the
per-cell diagnostic on the spline arm then said *why*: the mean-PIT pattern is **identical** under both
density families — `.523/.489/.512/.484/…` (truncnorm) against `.525/.492/.513/.488/…`
(spline) — so strictly more within-cell freedom moved the bias not at all. The defect is a
per-cell **location** bias: a limit on what the head can *predict from its conditioning*,
not on what its density can *express*. That is what `coord_cell_center` below tests, and it
is why the escalation to a joint coordinate density (§7.3 of the plan) was **not** taken.

The field ships **measured and not recommended** — the same status `decode.point_estimator="mbr_n"`
carries — rather than removed: it is bit-identical off, and a later change to the head's
conditioning may make the extra flexibility pay where today it only adds variance. Full
record in [`PLAN_lnz_spline_head.md`](PLAN_lnz_spline_head.md) §7.1/§8.1 and
[`SUMMARY_Model_Status.md`](SUMMARY_Model_Status.md) §2.6; the diagnostic is
[`scripts/offset_head_diagnostic.py`](../scripts/offset_head_diagnostic.py) and the gate
printer [`scripts/lnz_spline_gates.py`](../scripts/lnz_spline_gates.py).

#### `coord_cell_center` — the head is told *which* cell, never *where* it is

The coordinate head's input is `[decoder state | e(x) | cell embedding]`, and the embedding
is a free vector per **categorical** id. Nothing in it says that cell 437 and cell 438 are
neighbours, so every cell's within-cell tilt has to be learned from its own emissions
alone — which is exactly the shape of a per-cell location bias that more *output*
flexibility cannot touch, i.e. what `dv_head` measured. `coord_cell_center=true` appends
the cell's continuous centre `(c_x, c_y)`, affinely mapped onto `[−1, 1]` by the geometry's
own `ln_invdelta_range` / `ln_kt_range`, to that input. Two columns in one `torch.cat`,
~130 parameters.

The map is **fixed**, not data-dependent — it is built from the geometry, never from
sample statistics — so a checkpoint means the same thing on a new sample, the same rule
[`features.py`](../src/h2p_rsd_junipr/features.py) standardization follows. It is kept
as plain floats rather than a buffer, so with the switch off the `state_dict` is
byte-identical and old checkpoints load strictly.

> **Status: open experiment, and its reading is pre-registered.** The arms
> (`cellctr_s{0,1,2}` against `spline_s{0,1,2}`, same seed, `ln z` spline on both sides so
> the row prices the conditioning alone) are defined in
> [`scripts/run_lnz_spline.sh`](../scripts/run_lnz_spline.sh); the verdict rule is written
> down in [`PLAN_lnz_spline_head.md`](PLAN_lnz_spline_head.md) §9.2 **before** the numbers
> existed, because the previous hypothesis in that document was elegant, wrong, and
> tempting to reinterpret afterwards. The statistic is the RMS of `mean PIT − 0.5` over
> populated `ln k_t` cells — the quantity §8.1 showed to be identical under two density
> families. CONFIRMED = `dv` below 1.0× on all three seeds ⇒ field it beside
> `lnz_head="spline"`; PARTIAL = the RMS drops by more than a third without clearing 1.0×
> ⇒ *more* conditioning (cell width, neighbour occupancies), not a different density;
> DEAD = RMS within ±20% of the control ⇒ the conditioning hypothesis is falsified and the
> joint coordinate density becomes the live next step. Guards unchanged: support at
> 0.0000%, NLL not worse beyond the control's seed spread, TARP and `pit_ks_max` reported
> beside the verdict — a conditioning fix that buys `dv` by spending `ln z` is not a fix.

**`ar_junipr_v3`** is the v2 backbone with `use_multiplicity_head=True`: it factorizes
`q(y|x) = q(N|x)·q(y|N,x)` with a dedicated categorical multiplicity head (the same head cINN
and diffusion carry) instead of the implicit per-step continue/stop product. This makes the
length a calibrated, low-dimensional marginal — killing the short-sequence MAP collapse at its
source and giving `length_pmf` an *exact* `softmax(n_head(e))` (no sampler histogram). The kinematics
(cells + coordinates) are the unchanged autoregressive JUNIPR heads, run for exactly `N` steps.
The switch is a plain bool, so `model=ar_junipr_v3` and `model=ar_junipr_v2 model.use_multiplicity_head=true`
are equivalent, and `False` (the default) leaves the model byte-identical to today (old AR
checkpoints load unchanged). See [`docs/PLAN_MultHead.md`](PLAN_MultHead.md) and README_PHYSICS §"Length as a first-class factor".

**The support guard.** A categorical `q(N|x)` has *finite* support `N = 0..max_emissions`;
the v2 continue/stop head had none. A truth sequence past the support is clamped into the
last bin, so it receives the **wrong likelihood** — silently, with no signature in the loss
curve. `train` therefore checks `P_data(N > model.max_emissions)` against the data actually
loaded and **hard-errors above `1e-3`** (warns above `1e-4`); `eval` reports it without
refusing. The message quotes the offending `z_cut` / `β` / `k_t`-floor and the bound to
raise `max_emissions` to. Grooming parameters move this tail, so it is checked per run, not
once at design time. Source: [`data/stats.py`](../src/h2p_rsd_junipr/data/stats.py).

**`ar_junipr_v4`** is v3 plus `use_cross_attention=true`. v1–v3 hand the decoder one pooled
`ctx_dim` vector, tiled at every step: every hadron-level node reaches the parton-level
decoder only through that vector — the classic fixed-length bottleneck, and the reason
LundNet's graph structure is flattened before the decoder sees it. v4 lets the decoder
additionally cross-attend to the encoder's per-node states
(`Encoder.forward_seq`; `gru`, `lundnet` and `deepsets` all provide them — an encoder that
does not is a hard config error, never a silent fallback). The attention is applied as a
**residual**, so no head's input width changes: with the switch off the module list and
`state_dict` are byte-identical and existing checkpoints load strictly. Attention is over
`x` only, so the autoregressive factorization over `y` stays causal and the sampling/beam
paths inherit the change unmodified.

> **Compare at matched parameter count** — cross-attention adds `kv_proj` + `xattn`
> (~25k params at `dec_dim=64`), so shrink `dec_dim` to compensate (52 matches v3 to +1.1%)
> before drawing any conclusion.
>
> **And expect the answer to depend on the data.** Measured, `encoder=gru`, v4 at
> `dec_dim=52` vs v3 at `dec_dim=64`:
>
> | data | mean hadron multiplicity | v3 val NLL/jet | v4 val NLL/jet |
> |---|---:|---:|---:|
> | synthetic (15 epochs) | ~6 | 21.68 | **17.85** |
> | `cpp/test_data/jets.root` (12 epochs) | 1.74 | **4.61** | 4.64 |
>
> Not a contradiction — the mechanism showing itself. What cross-attention removes is the
> cost of *pooling a sequence into one vector*, and the PYTHIA sample above is groomed
> tightly enough (`z_cut=0.1`, `k_t` floor 1 GeV) that the mean hadron sequence is under two
> emissions, with 6.9% of jets having none at all. There is no bottleneck to remove there,
> so the capacity `dec_dim` gave up to pay for the attention is simply lost. **An ablation
> on a generator whose statistics differ from your data can point the wrong way.** Adoption
> for physics runs goes through the WP4 A/B on the data you will actually use.

### `cinn` — conditional normalizing flow

`P(n|e)·∏P(cell|e)·∏p_flow(coords|e,cell)` with a RealNVP over the 4 coordinates.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width |
| `n_blocks` | `6` | RealNVP affine-coupling blocks |
| `hidden_dim` | `64` | coupling-network hidden width |
| `max_emissions` | `25` | multiplicity-head width (categorical over `n=0..max_emissions`) |
| `sigma_floor`, `kappa_max` | `1e-2`, `50.0` | carried for schema symmetry; **not used** by the RealNVP flow (see §11) |

### `diffusion` — conditional diffusion (surrogate likelihood)

Categorical `n`/cell heads + a variance-preserving diffusion over the 4 coordinates.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width |
| `hidden_dim` | `64` | denoiser MLP width |
| `n_steps` | `50` | diffusion (DDPM) steps for the reverse process |
| `max_emissions` | `25` | multiplicity-head width (categorical over `n=0..max_emissions`) |

> **`exact_likelihood = False`.** The coordinate term of `log_prob` is the
> denoising-score-matching residual used as a *proxy* — not the ELBO, not the
> probability-flow-ODE likelihood — so it carries an unknown, context-dependent offset.
> Its NLL is a relative score **within this family only**, and its log-ratios are not
> likelihood ratios. Use `cfm` when you need the density. `diffusion` remains the
> registry's cheap-sampler baseline.

### `cfm` — conditional flow matching (exact ODE likelihood)

The exact-likelihood member of the continuous-time family (Lipman et al., ICLR 2023,
arXiv:2210.02747; FMPE, arXiv:2305.17161; probability-flow ODE, arXiv:2011.13456). Same
factorization as `cinn` — `q(N|x)·∏q(cell|x)·∏p_cfm(coords|x,cell)` — with the coordinate
density given by a conditional vector field.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width |
| `hidden_dim` | `64` | vector-field MLP width |
| `n_ode_steps` | `32` | fixed-step ODE steps, for both the likelihood and sampling |
| `ode_solver` | `"rk4"` | `rk4` (4 field evals/step) or `heun` (2 — ~2× cheaper, same object) |
| `max_emissions` | `25` | multiplicity-head width |
| `time_features` | `16` | Fourier time features fed to the field |
| `sigma_min` | `1e-3` | OT-path terminal width (Lipman Eq. 20) |
| `cfm_map` | `"ode_mode"` | MAP coordinates: `ode_mode` (push the base mode through the ODE — cheap, deterministic) or `ascent` (gradient ascent on the exact density — the true conditional mode, costs `ascent_steps` ODE likelihood evaluations) |

Two things worth knowing before using it:

- **Training minimizes a different objective than it reports.** The vector field is trained
  by flow-matching regression (no ODE in the loop); the ODE runs only at evaluation. So the
  logged `train_nll` is that regression objective while `val_nll` is the exact NLL — the two
  are on different scales, and `train` says so once at startup. This is what lets `log_prob`
  stay an honest density instead of becoming a training proxy.
- **`n_ode_steps` is an accuracy knob, `ode_solver` is a cost knob.** Both solvers integrate
  the same ODE and agree to `1e-4` at fine steps; the divergence is exact (4 vector-Jacobian
  products for a 4-dimensional state, so no Hutchinson estimator and no stochastic
  likelihood). Coordinates live on the physical support via fixed tanh-box / angle-wrap
  bijections with closed-form log-Jacobians, so the density integrates to 1 *on the box* —
  the property a discretized grid head cannot have.

> **Known limitation.** Unlike the AR von Mises head, the ψ map is not periodic across the
> branch cut at ±π: the density is exactly normalized on `(-π, π)` but the seam is not
> closed structurally (the model can learn to match across it, but nothing enforces it).
> Fixing that needs Riemannian flow matching, deliberately out of scope.

### `edit_v1` / `edit_v2` — the edit transducer (latent alignment)

The one family that does not generate `y` from scratch
([`PLAN_EditTransducer.md`](PLAN_EditTransducer.md)). The hadron tree is the **anchor** of
the parton tree: each parton node is a smeared copy of a hadron node (**kept**), a fresh
draw (**insertion**), or a hadron node with no image at all (**deletion**). Which is which
is a **latent** variable, marginalized by an `O(n_x·n_y)` forward recursion — node-level
parton↔hadron correspondence is not observable, so it is never a supervised target.

    i <  nx : {ADVANCE, EMIT}      i == nx : {STOP, EMIT}   (trailing insertions)
    EMIT    : y_{j+1} ~ p_anch·f_shift(·|x_i) + (1 − p_anch)·f_free(·)

This is the RNN-T lattice (Graves, arXiv:1211.3711), so `Σ_y q(y|x) = 1` holds by
construction and `exact_likelihood = True` is structural rather than asserted. Requires an
encoder with `returns_sequence = true` (`gru`, `lundnet`, `deepsets` all qualify) — the
anchors are the **per-node** states, not the pooled `e(x)`.

| Field | Default | Meaning |
|---|---|---|
| `ctx_dim` | `64` | context width; also the width the per-node states are projected to |
| `op_head_layers` | `2` | depth of the STAY/EMIT head |
| `shift_head_layers` | `2` | depth of the anchored-emission head (displacements + `p_anch`) |
| `free_head_layers` | `2` | depth of the free-emission (insertion) heads |
| `sigma_floor` | `1e-2` | floor on every width |
| `kappa_max` | `50.0` | von Mises concentration ceiling |
| `max_emissions` | `25` | readout width of the exact `q(N\|x)` and the sampler's cap — **not** a likelihood support (see below) |
| `physics_width` | `true` | `σ = σ₀ + Λ_eff·exp(−ln k_t)`; `false` = free-MLP ablation |
| `prefix_conditioning` | `false` | `false` = `edit_v1`, `true` = `edit_v2` |
| `lnz_support` | `"legacy"` | `legacy` = the unbounded Normal on `ln z` in **both** mixture components; `physical` = the truncated normal on the interval the grooming leaves. See below |
| `lnz_zcut` | `0.1` | the file's soft-drop `z_cut`; read only when `lnz_support="physical"` |
| `lnz_beta` | `0.0` | the file's soft-drop `β`; read only when `lnz_support="physical"` |

#### `lnz_support` on this family — the same field, one bound tighter

Same name, same semantics and the same `data.stats.check_lnz_support` guard as the AR
families above ([`PLAN_prod_test_edit.md`](PLAN_prod_test_edit.md) WP-E). The plane
coordinates were already truncated to the geometry range in *both* mixture components;
`ln z` was the one left on an unbounded Normal, so `legacy` reproduces the v0 support
failure by construction (~0.81% below the soft-drop wall, ~3.98% above `z = ½`).

One difference from the AR implementation, and it is a tightening. The bound is read at
the node's **own** `u`, `lo(u) = ln z_cut − β·u`, not at the loosest `u` in its cell.
This factorization supports that: the emission density is `f(u, v)·f(ln z | u)·f(ψ)`, and
a `u`-dependent `ln z` factor is exactly what such a product can express, whereas the AR
coordinate head — a product of factors independent *given the cell* — cannot. So this is
the exact Soft Drop boundary, the same expression the guard verifies the truth against,
with no `|β|·half_u` slack to audit. At the fielded `β = 0` the two conventions coincide.

`_log_cell_mass` is deliberately **unchanged** by the switch, and that is a fact rather
than an oversight: a cell is a box in `(u, v)` only, and `∫ f(ln z | u) d ln z = 1` for
every `u`. The constrained forward–backward behind `sample_coordinates` therefore draws
bit-identical alignments under both supports; only the `ln z` inside the chosen component
differs. `tests/test_edit_lnz_support.py` asserts this rather than trusting it.

> ⚠️ **NLL is not comparable across the head change — in either direction, and across
> families too.** A truncated `ln z` concentrates its mass on a 1.61-wide interval and
> *gains* NLL against a Normal on ℝ for reasons that have nothing to do with fit quality.
> That applies to `edit` vs `edit` and, in the cross-family A/B this field exists for
> (`edit_v1` vs the AR continue/stop arm), to `edit` vs `ar_junipr_*`. Both families are
> `exact_likelihood = True` and both normalize over the same `(u, v, ln z, ψ)` space, so
> their NLL/jet **is** comparable — *provided both sides declare the same `lnz_support`*.
> `tests/test_nll_comparability.py` is where that is asserted: it MC-integrates the two
> emission densities over one common box and checks both give 1, and checks that a
> `legacy` head does not. `scripts/prod_test_edit_gates.py` enforces it mechanically —
> a mismatched pair prints `!` and refuses to rank.

Three properties that follow from the factorization rather than from a knob:

- **Length is anchored at `|x|`.** `n_y = n_x − #del + #ins`, so the open-ended
  continue/stop mechanism — the documented seat of the marginal multiplicity bias and of
  MAP collapse — is removed *structurally*. Marginalizing the coordinates out of the same
  recursion gives `q(N|x)` **exactly, with no extra parameters**, and that is what
  `length_pmf` returns. `empty_gate` therefore reads an exact `q(N=0|x)`: the empty parton
  tree is the delete-all path, represented natively. `length_floor_quantile` and
  `learned_min_emissions` compose unchanged.
- **The width is a physics form, not an MLP output.** `σ = σ₀ + Λ_eff/k_t` is the
  shape-function scaling of local parton–hadron duality, so `Λ_eff` comes out in GeV and
  the learned kernel is directly confrontable with the expectation (arXiv:1906.11843)
  instead of opaque. `model.physics_width_params()` reads it back. This is the family's
  falsifiable claim: **if the residual widths are flat in `k_t`, the anchoring assumption
  is wrong.**
- **The alignment posterior is a free diagnostic.** `eval` reports `frac_anchored`,
  `insert_rate` and `delete_rate` off the forward–backward responsibilities — an emergent
  alignment, obtained without ever supervising one. `model.alignment_posterior(batch)`
  returns the per-`(i, j)` responsibilities for binning residual widths.

> **`model.max_emissions` is inert in the likelihood here**, unlike `cinn`/`cfm`/`ar_junipr_v3+`.
> The length model is the open-ended STOP/EMIT lattice, so a truth longer than
> `max_emissions` is merely improbable, not mis-normalized — which is why the §4 support
> guard does not apply to this family (`data/stats.py:model_support`). What the cap does
> bound is the `length_pmf` array (renormalized over `n ≤ max_emissions`) and the sampler.

> **`decode.point_estimator=mbr` is the recommended estimator for this family.**
> `map_estimate` is a labelled **surrogate** twice over: the shape is the best single
> alignment (the exact MAP is an argmax of a marginal-over-alignments, and is intractable),
> and the length is `argmax_n q(N=n|x)` from the exact marginal rather than the joint
> argmax. The latter is not a shortcut — a joint argmax over a variable-dimension *density*
> runs straight to `max_emissions` whenever the modal emission density beats the per-step op
> cost, which with sharp kernels is the normal regime at high `k_t`. It is the same staged
> decode `ar_junipr_v3` uses, with an exact marginal in place of a learned head.

> **`edit_v2` is gated on a stage-1 result, not on taste — and the gate has been checked
> once.** The premise it rests on is that the smearing really is a `Λ_eff/k_t` kernel; if
> the residual widths are flat in `k_t` there is nothing for a richer emission model to
> sharpen. On `cpp/test_data/jets.root` the fit lands at `Λ_eff = 1.29 GeV`, `R² = 1.000`
> (see [`PLAN_EditTransducer.md`](PLAN_EditTransducer.md)), so `edit_v2` ships. That result
> is **sample-dependent**: re-run the check on your own selection before quoting anything
> from v2, by binning `model.alignment_posterior(batch)`'s residuals in `ln k_t` and fitting
> `σ = σ₀ + Λ_eff·exp(−ln k_t)`. Do it with `model.physics_width=false`, or you are reading
> back the form you imposed.
>
> The v2 prediction network runs over the emitted **cell** prefix and feeds the *emission*
> heads only; the op head stays prefix-free in both stages, which is exactly the condition
> for `length_pmf` to remain exact ("teacher forcing enters the prefix only — never the
> length"). Memory scales as `batch·(n_x+1)·n_y·n_cells`, so drop `trainer.batch_size` on a
> long-tailed sample.
>
> **Nothing has adjudicated v1 vs v2 yet.** Held-out NLL is the arbiter and it has not been
> run at equal budget — v2 carries ~40% more parameters, so a lower NLL on its own says
> little.

> **`supports_coordinate_pit = False` in both stages.** The exact prefix-conditional CDF is
> available from the same recursion as a responsibility-weighted mixture of
> `trunc_normal_cdf` / `gauss_cdf` / `vonmises_cdf`; it lands once the DP is trusted, not
> before. The WP2 coordinate-PIT panel therefore reports nothing for this family today.

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
A/B them on a fixed checkpoint. At `eval`, the checkpoint's snapshot decode is the default
and anything this invocation names explicitly wins over it — `decode=<name>`, a `base=`
preset's `defaults:`/inline block, or a dotted `decode.*` token, in the §0 precedence order
with the CLI last (e.g. `eval <ckpt> decode.length_floor_quantile=0.9`). Only the fields
actually named move; the rest stay as the checkpoint left them, and every move is printed
and recorded in `eval_metrics.json`. The same applies to `data` — see
[USAGE §4](USAGE.md#evaluating-on-a-different-sample-a-held-out-test-set) for evaluating a
checkpoint on a held-out test file. `geometry` and `encoder` are **not** liftable this way:
they set tensor widths and the model contract.
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
| `cont_temperature` | `1.0` | posterior | softmax temperature on the **cell** logits at **sampling** time; `>1` flattens, `<1` sharpens. A historical name — it tempers *which* cell is emitted, not *whether* one is. For the length knob see `continue_temperature` |
| `continue_temperature` | `1.0` | posterior | temperature on the **continue/stop** logit at sampling: `p_cont = σ(logit/T)`. `>1` pulls `p_cont` toward ½, which *lengthens* trees where the head is confident to stop and *shortens* them where it is confident to continue. Sampling only; `1.0` = off and bit-identical. **No-op** for families with an explicit `q(N\|x)` head |
| `kappa_min_mode` | `0.5` | point estimate | below this von Mises concentration the ψ **mode** is not identified, so a point estimate reports a **draw** there and flags the node `psi_identified=False`. `0.5` is peak/trough ≈ e; `0.0` disables the gate |
| `min_emissions` | `1` | MAP | **hard floor** on MAP length — the "mincut" (never the unphysical empty tree) |
| `length_penalty` | `0.0` | MAP | GNMT `score/len**α` at final beam rank; counters the brevity bias; `0` = off |
| `length_floor_quantile` | `0.0` | MAP | **learned per-jet floor** at the α-quantile of `P(n|x)`; `0` = off |
| `length_temperature` | `1.0` | posterior + point estimate | post-hoc scalar temperature on the multiplicity logits; `1` = off |
| `length_tilt` | `0.0` | posterior + point estimate | companion term **linear in n** on the same logits — what actually moves mass between short and long trees; `0` = off |
| `empty_threshold` | `0.0` | point estimate | **emptiness ceiling**: answer the empty tree when `q(N=0\|x) >= τ`, before any shape decode; `0` = off |
| `point_estimator` | `"map"` | point estimate | `map` (beam-search joint mode), `mbr` (minimum-Bayes-risk tree; §10 MBR) or `mbr_n` (**N-first**: N from the calibrated `q(N\|x)` median, shape from the medoid *within* that stratum; §10) |
| `mbr_backend` | `"pot"` | MBR | OT backend: `pot` (default, self-contained) / `energyflow` (reference) / `surrogate` (fast χ²) |
| `mbr_n_candidates` | `0` | MBR | `0` = every draw is a candidate; `k>0` = only the first `k` (asymmetric MBR, faster) |
| `mbr_lnkt_cut` | `null` | MBR | drop emissions below this `ln k_t`; `null` inherits `geometry.ln_kt_range[0]` (the region cut) |
| `mbr_weight` | `"kt"` | MBR | Lund-cloud point weights: `kt` (IRC-safe) / `z` / `unit` |
| `mbr_coords` | `"lnDR_lnkt"` | MBR | ground-metric columns: `lnDR_lnkt` / `+lnz` / `+psi` (gdim 2/3/4; `+psi` engages periodicity) |
| `mbr_R` | `8.485` | MBR | mass-imbalance penalty radius ≈ Lund-plane diameter (scale it with `geometry`) |
| `mbr_beta` | `1.0` | MBR | ground-distance exponent; `1.0` = KMT 1-Wasserstein EMD |
| `mbr_norm` | `False` | MBR | energyflow weight normalisation; **off** keeps the imbalance term (empty-tree-never-wins) |
| `mbr_periodic_phi` | `False` | MBR | wrap the ψ column (only with `mbr_coords=+psi`) |
| `mbr_phi_col` | `-1` | MBR | ψ column index for the periodic wrap; `-1` = last coordinate |
| `mbr_resample_to_qn` | `False` | MBR | reweight the candidate/support pool to the calibrated `q(N\|x)` marginal (decode-layer exposure-bias fix; `False` = plain mean risk) |
| `cluster_posterior` | `False` | set-valued | build the `K×K` cluster labelling beside the point estimate. **Raises** unless `mbr_n_candidates == 0`, `mbr_beta == 1.0` and `mbr_R ≥ R_max/2` — see the guards below |
| `cluster_method` | `"hdbscan"` | set-valued | `hdbscan` (density, no fixed *k*, native noise label) / `dbscan` (ε-explicit) / `pam` (*k*-medoids, *k* by silhouette; pure NumPy and deterministic) |
| `cluster_min_cluster_size` | `0` | set-valued | `hdbscan`/`dbscan` control; `0` ⇒ `max(5, ceil(cluster_min_mass × K))` over the pool actually clustered |
| `cluster_eps_quantile` | `0.10` | set-valued | **`dbscan` only**: `ε = Q_γ` of the *positive* off-diagonal distances. Backend- and `R`-invariant by construction, which is why it is a quantile |
| `cluster_min_mass` | `0.05` | set-valued | clusters below this merge into a residual bucket, so the reported mass vector stays short |
| `cluster_split` | `False` | set-valued | sample-split the mass estimate: cluster on pool A, estimate masses from a fresh pool B. **Off keeps the single-pool estimate, which is biased HIGH** |
| `set_alpha` | `0.32` | set-valued | conformal miscoverage for `predict_set` (1 σ). The guarantee is **marginal over jets**, not conditional on *x* |

`min_emissions`, `length_penalty`, and `length_floor_quantile` are explained in depth in
§10; `point_estimator` / `mbr_*` — the whole second point-estimate family — are covered in
the §10 MBR subsection, and `cluster_*` / `set_alpha` in the §10 posterior-cluster
subsection. All are inference-time only, so you can A/B them on a fixed checkpoint.

### `empty_threshold` — the one decision the other knobs cannot express

At parton level **~17% of jets have no primary splitting at all**: hadronisation
manufactured every splitting you see at hadron level, and the correct answer is nothing.
Under the default decode no point estimator ever says so, for two unrelated reasons — and
neither is a fitting failure:

- **MAP.** With a multiplicity head the estimate is `argmax_n q(n|x)`. A model can hold 16%
  of its length mass at `n=0` and still never put its *peak* there, because that mass has to
  beat `n=1` and `n=2` outright. **Lifting `min_emissions` to 0 changes nothing** — the clamp
  was never the binding constraint.
- **MBR.** Mode-free and floor-free, so it *could* return the empty tree, but the
  perturbative-Lund EMD charges `mbr_R` for unmatched weight and an empty cloud is nothing
  but unmatched weight. Its risk is near-maximal: not merely unlikely, close to the worst
  answer available.

`min_emissions` and `length_floor_quantile` clamp the output *length*; they cannot
distinguish "this jet's structure is spurious" from "this jet is short". `empty_threshold`
is the missing decision — a **ceiling** where `length_floor_quantile` is a floor, taken on
the model's own belief before either shape decode runs
([`models/base.py::map_or_mbr`](../src/h2p_rsd_junipr/models/base.py)).

Fit it, never guess it:

```python
from h2p_rsd_junipr.inference.length import empty_threshold_for_rate
tau = empty_threshold_for_rate(pmfs_heldout, rate=0.17)   # then freeze
```

**It thresholds the ranking, not the scale.** Measured on the walkthrough `ar_junipr_v3`:
`q(0|x)` separates the classes at **AUC 0.76** while its mean is 0.085 against a true rate
of 0.161 — the head is under-confident by **1.9×**. A quantile threshold is invariant to
that monotone squash, which is why the gate works without recalibrating first. Note the
existing suite does **not** flag the under-confidence: SBC ranks against the sampler's own
draws, so a uniformly squashed `q(N|x)` passes.

Held-out (fit on one half of the val split, scored on the other): predicted rate 0.172
against a truth 0.159, **recall 0.36, precision 0.33**. The population becomes right and the
per-jet call goes from impossible to a third correct — it is not a solved classification
problem, so report both. τ is a *quantile*, hence sample-dependent: re-fit per pT window,
and never carry one across a selection change.

Two consequences worth internalising. The gate is **backend-independent** — it never touches
the MBR risk, unlike the empty-tree column itself, which reads ~0.2% under `pot` and ~57%
under `surrogate`. And it **changes what a point estimate can be**: downstream consumers
that assume `multiplicity >= 1` must tolerate an empty `LundPointEstimate`
(`leading_emission_cell` already returns `None`). Full analysis:
[`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md).

### `length_temperature` + `length_tilt` — recalibrating the length head

`softmax(z/T + tilt·n)` on the multiplicity logits, both fitted post-hoc on held-out jets.
No retraining, no new weights. This is a **decode-layer** transform: it moves `length_pmf`
and the `N` drawn by `sample`, and deliberately never touches `log_prob` or the `logprob` a
point estimate reports — those are the trained likelihood. Applied in one place,
`PosteriorModel.recalibrated_n_logits`, so the belief and the draws cannot disagree. A
no-op for families with no `n_head` (`ar_junipr_v1/v2`), where the length belief *is* the
sampler histogram; `build_model` warns rather than half-applying it.

```python
from h2p_rsd_junipr.inference.length import fit_length_recalibration
T, tilt = fit_length_recalibration(pmfs_heldout, n_true_heldout)   # then freeze
```

**Why two parameters and not one.** A temperature is symmetric about the mode, so it can
only pull a non-modal class toward uniform or toward zero. `q(0|x) = 0.085` sits above
uniform (`1/26 = 0.038`) and below the mode, so flattening pushes it *down*: no scalar `T`
reaches the truth rate, and sweeping `[0.1, 20]` tops out at 0.125 against 0.161. The
measured error is a **monotone ramp across n** (empirical/predicted = 1.90, 0.96, 0.93,
0.80, 0.68 at `n = 0..4`), and a ramp needs a term linear in `n`. Measured on the
walkthrough `ar_junipr_v3`, fit on half the val split and scored on the other:

| | `T=1` | scalar only | **+tilt** | truth |
|---|---|---|---|---|
| NLL of `N` | 1.2133 | 1.2125 | **1.1810** | |
| mean `q(0\|x)` | 0.0846 | 0.0910 | **0.1428** | 0.1610 |
| max `q(0\|x)` | 0.4261 | 0.4166 | **0.5089** | |
| emp/pred at `n=0` | 1.90 | 1.77 | **1.13** | |

It fixes the length *mean* as well, not only the `n=0` cell: `E[N]` goes 1.628 → **1.429**
against a truth of 1.435 on held-out jets. Watch one interaction, though —
`run_closure`'s `mean_mult_posterior` and `mult_bias_posterior` are computed over the
truth-**non**-empty jets only (the `ly is None` continue), and on that subset a tilt toward
`n=0` reads as a *worse* bias even while the full-population mean improves. Quote the
`p_empty_*` row and the multiplicity row together, or the two look contradictory.

Fitted `T = 1.21`, `tilt = −0.31`. `with_tilt=False` restricts the fit to the scalar case
and is kept as the honest baseline. What it buys: the **cost-based** `empty_threshold` form
(`τ = c/(1+c)`) becomes usable, since the belief now reaches 0.5; the posterior series in
the closure notebooks moves from 0.085 to 0.143 empty against a truth 0.161. What it does
**not** buy is gate quality — recall 0.360 → 0.368, precision 0.333 → 0.340, AUC flat —
because a monotone-in-n transform largely preserves the cross-jet *ranking* of `q(0|x)`.
That is the same reason `empty_threshold` works without recalibrating first.

### v3 semantics — which of these knobs are still live

Three of the decode knobs exist **only** to patch the two length pathologies of the v2
implicit continue/stop head. `ar_junipr_v3` (and `cinn` / `diffusion` / `cfm`, which
always had one) removes those pathologies at source by making the length a first-class
categorical `q(N|x)`. So under v3 the same knob names mean different things:

| Knob | Status under v3 | Why |
|---|---|---|
| `max_emissions` (model) | **live — and now load-bearing** | It is the *categorical support* `N=0..max`. A truth past it is clamped into the last bin and gets the wrong likelihood, so `train` hard-errors when `P_data(N > max_emissions) > 1e-3` (warns above `1e-4`). It was merely a decode cap under v2. |
| `cont_temperature` | **live** | Still the softmax temperature on the *cell* logits at sampling time. v3 changed the length model, not the kinematics. |
| `decode.max_emissions` | **live** | Still the decode-time length cap; clamps `N ~ q(N|x)` draws. |
| `min_emissions` | **legacy-v2** | It floored a brevity-biased *joint* argmax. Under v3 the MAP length is `argmax q(N|x)`, which is not brevity-biased, so the floor is expected to be inert — [`scripts/ab_v2_v3.py`](../scripts/ab_v2_v3.py) measures the `MAP=0` fraction at `min_emissions=0` to confirm it per checkpoint. |
| `length_floor_quantile` | **legacy-v2** | Same reason: it transferred the model's length belief into a mode that had lost it. Under v3 the mode *is* read off that belief. |
| `mbr_resample_to_qn` | **legacy-v2 (measure it)** | It matched the MBR support's multiplicity marginal to the calibrated `q(N|x)`. Under v3 the draws already come from `q(N|x)`, so the weights should be 1 up to Monte-Carlo noise. **Read the *excess* over the finite-K null**, not the raw weight spread: `w_k` compares an exact head against a K-draw histogram, so `w ≠ 1` at `O(1/√K)` even for a perfect sampler. The A/B script reports both columns. |
| `length_penalty` | **legacy-v2** | GNMT normalization of a beam score; v3's greedy fixed-length decode has no beam to rank. |

**Do not read "legacy" as "removed".** Every knob still works and still defaults to the
same value, so v2 checkpoints and v2 runs are unaffected. The claim is only that under
v3 they are expected to be *no-ops*, and that expectation is measured per checkpoint by
the A/B table rather than assumed.

```bash
# reproduce the table (2 trainings, then every decode cell on each):
python scripts/ab_v2_v3.py --preset presets/ab_v2_v3.yaml --out runs/ab_v2_v3
python scripts/ab_v2_v3.py --fast          # CI tier: tiny data, 1 epoch
```

The A/B is gated on the **WP2 suite** — per-coordinate PITs and TARP — and deliberately
**not** on SBC-N: v3 trains `q(N|x)` by direct NLL on `N`, so an SBC-on-N comparison
would certify v3 near-tautologically (see §8 and
[`eval/calibration.py`](../src/h2p_rsd_junipr/eval/calibration.py)).

### The deferred "feed N into the decoder" extension — the decision rule

`docs/PLAN_MultHead.md` deferred conditioning the *kinematics* decoder on the drawn `N`
(so `q(y|N,x)` sees its own length). It stays deferred. Recorded here so it is not
re-litigated: **implement it only if** the WP2 diagnostics show miscalibration of
`q(y|N,x)` that is *systematically* `N`- or region-dependent, at a magnitude comparable
to the quoted generator systematic. Concretely, all three must hold on a trained
checkpoint:

1. `pit_coords` KS rises monotonically with the emission index (`by_emission_index`) or
   differs across `by_region` strata by more than the KS 95% critical value
   `1.36/√n` at that sample size — i.e. the miscalibration is *structured*, not noise;
2. `tarp_max_dev` degrades measurably when TARP is restricted to high-`N` jets relative
   to the pooled curve;
3. the resulting spread is of order the `generator_spread` figure
   ([`eval/systematics.py`](../src/h2p_rsd_junipr/eval/systematics.py)) — below that it is
   not the dominant uncertainty and buys nothing.

If only (1) holds, the cheaper remedies are `cont_temperature` and the cross-attention
of §4 (`use_cross_attention`), both of which address coordinate-level exposure bias
without changing the factorization.

---

## 8. `experiment` — evaluation suite

Controls the §8 closure / calibration / systematic run (`h2p-rsd-junipr eval`).

| Field | Default | Meaning |
|---|---|---|
| `name` | `"default"` | experiment label |
| `closure_jets` | `300` | held-out jets evaluated in the closure/calibration loop |
| `n_closure_samples` | `200` | posterior draws **per jet** inside that loop |
| `generator_b` | `null` | a second generator/checkpoint for the PYTHIA-vs-HERWIG systematic (the dominant uncertainty) |
| `pit_coords` | `False` | per-coordinate PITs (calibration suite v2) |
| `stratify_regions` | `False` | bin every metric by the leading emission's Lund quadrant |
| `tarp` | `False` | TARP expected-coverage curve on tree-valued posteriors |
| `tarp_refs` | `100` | size of the TARP reference pool |
| `tarp_reference` | `"pooled"` | `pooled` (posterior draws of other jets) or `prior` (their truth trees) |
| `closure_continuous` | `False` | leading-emission distances **off** the cell grid, via `sample_coordinates` |
| `exposure_diagnostic` | `False` | the WP-B block: `<N>` on **both** populations, SBC-on-N against its own simulated null, and the teacher-forced vs on-policy continue probability by depth |
| `support_audit` | `False` | window / soft-drop / `z>½` / `k_t`-floor violation rates of the sampled posterior, **scored** against a hard zero (gate G2) |
| `tarp_null_reps` | `0` | Monte-Carlo reps for the TARP null band at this run's own `(n_jets, α grid)`; `0` keeps only the asymptotic `1.36/√n` floor |
| `tarp_stratify` | `False` | TARP additionally per Lund quadrant |
| `coverage_null_reps` | `0` | pseudo-truths **per jet** for `coverage_68`'s own null: extra held-out draws from the same posterior, scored through the identical `K`-draw HPD construction. `0` = off; ~20 is plenty. Without it `coverage_68` has no reference it can actually be read against (see below) |
| `mode_audit` | `False` | exact top-k **skeleton** enumeration with dominance certificates → `mode_audit.json` (§8a) |
| `cluster_diagnostics` | `False` | the posterior-cluster measurement pass → `metrics["clusters"]`: per-jet `n_clusters` / `top_mass` / `entropy`, gates **G2** (medoid-in-dominant-cluster), **G2′** (the oracle-set diagnostic with its mass-matched random-partition null and silhouette precondition), **G3**, **G6** (reliability + Brier decomposition), **G7** (conformal coverage) and the WP4a loss-stability columns. Needs `decode.point_estimator=mbr` (or `mbr_n`) — there is no distance matrix otherwise, and it says so and skips rather than emitting a table of NaN |

Trade cost vs. precision with `closure_jets` and `n_closure_samples`.

## 8a. `audit` — the mode-mass audit's search

Read only when `experiment.mode_audit=true`. It answers *does the posterior concentrate on
a single discrete parton configuration, and when it does, is that the true one* — as
**certificates** rather than estimates, because the skeleton marginal is exact (the
coordinate factors integrate to 1 given the cell) and the best-first search over the prefix
tree pops completions in exact descending mass order. See
[`PLAN_ModeMassAudit.md`](PLAN_ModeMassAudit.md) and the physics reading in
[`README_PHYSICS.md`](README_PHYSICS.md).

| Field | Default | Meaning |
|---|---|---|
| `k` | `64` | completions enumerated per jet, in exact descending mass order |
| `budget` | `20000` | expansion cap per jet; a jet that hits it before `k` completions is `certified: false` and its rate is itself reported |
| `prune_rel` | `1e-6` | drop a child below `prune_rel ×` the best completion mass found so far (× 1 before the first completion, i.e. an absolute floor). Pruned mass is accumulated **exactly**, so this costs certification, never correctness |
| `topk_children` | `0` | cell children per expansion; `0` = every cell. A cap is a bad trade on the fielded 30×30 geometry — measured at `k=32`, capping at 64 took the certified fraction from 97% to 20% and saved 28% of the runtime. Cap only when frontier memory binds |
| `max_frontier` | `20000` | heap cap; the lowest-mass entries are evicted into the pruned accounting rather than dropped |
| `eps_n` | `1e-4` | `q(N|x)` floor for the per-`N` searches of the `n_head` / `factorized` families; the dropped mass goes to the pruned total |
| `thresholds` | `[0.3, 0.5, 0.7]` | the pre-registered `F(m) = frac(M_1 ≥ m)` grid |
| `coarse_block` | `3` | block size for the sequence-level coarsening in `resolution.coarse_sequence`; `1` is the identity |
| `n_jets` | `0` | jets audited; `0` → `experiment.closure_jets`, so the audit reports on the same population as the rest of the suite |

```bash
h2p-rsd-junipr eval runs/.../best.ckpt experiment.mode_audit=true \
    experiment.closure_jets=2000 audit.k=32
```

Four things about this block that are **not** conventions:

- **It never writes to the estimator stack.** The audit reads the posterior; no MAP, MBR,
  floor or NLL moves because of it, and `min_emissions` deliberately does **not** apply —
  the enumeration is unconstrained and the empty skeleton is a first-class row, for the
  same reason a sampling floor would distort SBC/PIT.
- **`M₁ > ½` needs no certificate.** The total mass is 1 by construction, so nothing can
  exceed a half-mass mode whatever the search pruned. Below ½ the reported `M₁` is a
  **lower** bound on the true top-1 mass, so every `F(m)` is a lower bound too — never an
  over-claim.
- **`M₁` is resolution-relative — exactness is not invariance.** A cell's probability is
  `≈ density × area`, so every `N ≥ 1` skeleton's mass scales with the cell area while
  `q(N=0|x)`, the one skeleton that references no cell, does not. Refine `geometry.n_bins`
  and both the *level* of `F(m)` and the *identity* of the argmax change with nothing about
  the model changing (measured: a 9× coarser cell took the best one-splitting skeleton from
  0.015 to 0.098). Conditioning on `N` does not repair it — at fixed `N` the area factors
  are shared, so ratios survive but absolute masses still scale. So `F(m)` is a
  **same-geometry, same-checkpoint** comparison — which is exactly what the plan's §7.5
  cross-family delta needs — and never a grid-free claim that a dominant parton skeleton
  exists. The artifact's `resolution` block is the companion that *is* grid-free: the
  Lund-plane **area** the first splitting's posterior occupies (in `ln(1/ΔR) × ln k_t`
  units), its linear scale `√area`, the ratio to the coordinate head's own `±1σ` box, the
  grid-free length belief `q(N|x)`, and `frac_truncation_saturated` — the fraction of jets
  where the head's `σ` exceeds a half-cell, so it cannot express its own width *inside* a
  cell and carries that width in the cell distribution instead. Where that fraction is
  high, a small `M₁` says the grid is finer than the model's resolution, not that the
  posterior is fragmented. **Read `resolution` before quoting any mode mass.**
- **Naming the resolution gives the probability back.** `resolution.m1_of_r` is `M₁(r)`:
  the largest mass the posterior puts in *any* box of half-width `r`. The window **slides**,
  so unlike a coarsened grid it carries no partition origin — a mode straddling a block
  boundary would be split by a coarse grid exactly as it was by the fine one. `M₁(r) ∝ r²`
  at small `r` is the regime where the number measures the resolution element and nothing
  else, and that is where the grid's own `M₁` sits (`m1_at_r_cell`); the knee is the
  posterior's own scale; `m1_at_r_sigma` is the same reading at the width the head claims.
  This is what restores the quotable sentence — *"the leading splitting lies within ±r of
  here with probability p"* — with `r` stated. `resolution.coarse_sequence` is the
  sequence-level analogue, necessarily a **lower bound**: summing fine skeletons that share
  a coarse label does not factorise for `N ≥ 2` (the decoder state depends on the fine
  cell), so it aggregates what the search enumerated and tightens with `audit.k`. A coarse
  mass above ½ is dominant by proof regardless, since coarse labels partition the space.
  `audit.coarse_block` sets the block size (default 3).
- **Two validity checks, not gates.** The artifact reports the per-jet mass-accounting
  defect (`Σᵢ Mᵢ + frontier + pruned − 1`, float-exact) and the enumerated `M(N=0)` against
  the model's own `q(0|x)` reached through a different code path. A nonzero defect means
  the *search* is wrong; it says nothing about the model.
- **The empty class is never pooled away.** "The posterior is sure there is *nothing*
  there" and "the posterior is sure *which* splitting is there" are different physical
  claims, so `by_class` reports `F(m)` and the truth-rank fractions separately for
  `top1_is_empty` / `top1_is_a_splitting` (an inference-time cut) and
  `truth_is_empty` / `truth_is_a_splitting` (a decomposition that uses the answer, and is
  labelled as one). The pooled `overall` block is a mixture of the two.

The audit is family-agnostic through `PosteriorModel.skeleton_search_spec()`: `ar_junipr_*`
map to `ar` (per-step continue/stop) or `nhead` (explicit `q(N|x)`, fixed-length search per
`N` merged on one heap), `cinn`/`diffusion` to `factorized` (cells independent given `x`).
A family with no adapter raises **by name** rather than reporting a beam-search
approximation as if it were exact.

### The four references that are not what they look like

Each of these was a *reference* that failed, not a model that did — and each cost a
conclusion (three in production test v0, the fourth in the stratified-MBR campaign) before
it was found. They are grouped because the mistake is the same one four times: quoting a
statistic against a null it does not have.

**SBC-on-N has no χ²(9) null.** `N` is discrete and, at the fielded grooming, takes a
handful of values, so its mid-rank statistic lands on a handful of atoms and cannot be
uniform on `[0,1]` **for any model**. v0 read χ² = 107 against the χ²(9) 95% point of
16.90 and concluded the multiplicity was broken. With `exposure_diagnostic=True` the same
statistic is quoted against its own simulated null — the truth redrawn from `q(N|x)`
itself, same jets, same discreteness — which on the v0 checkpoint puts the observed value
at the **88th percentile of its own null**, i.e. consistent with calibrated. The χ²(9)
reference is still reported, labelled as the continuous-rank one it is.

**TARP's `1.36/√n` floor is asymptotic, and at n = 300 it is 0.079.** A band that wide
cannot detect a 5% miscalibration, so "max dev 0.037, inside the band" was a statement
about the sample size. `tarp_null_reps>0` recomputes the band by Monte Carlo at the run's
own `(n, α grid)` and reports `floor_ok` — whether the band is tight enough for the
statistic to be quotable at all. At n = 300 the recomputed 95% point is 0.073 (**not**
quotable); at n = 2000 it is 0.028.

**`coverage_68`'s target is not 0.68, and it never was.** The leading-cell HPD-68 is built
from the `K` posterior draws themselves, so it cannot contain a cell of probability
`< 1/K` that a genuine draw still visits: the statistic **under-covers for a perfect
model**, and by an amount set by `K`, not by the posterior's width.
`coverage_null_reps=M` measures exactly that — `M` extra held-out draws per jet, scored as
pseudo-truths through the identical construction — and reports `coverage_68_null`,
`coverage_68_vs_null` and `coverage_68_null_explains_deficit` beside the raw number. On the
fielded checkpoint at `K = 200` the null is **0.553** [0.543, 0.563] on 8 841 pseudo-truths
against an observed **0.546**: inside its own interval, so the "0.55 vs 0.68 ⇒ the
posterior is too narrow" reading that stood through v1 was the *estimator*, not the model.
Never quote `coverage_68` without its `K`. The draws are taken inside `torch.random.fork_rng`,
so switching the diagnostic on cannot move any other number in the run — a switch that
perturbs the statistic it exists to explain would be worse than no switch. TARP is
unaffected (it carries its own MC null), so the joint-narrowness case now rests on TARP and
the PIT cross alone. Record: [`PLAN_StratifiedMBR.md`](PLAN_StratifiedMBR.md) WP4 and
[`SUMMARY_Model_Status.md`](SUMMARY_Model_Status.md) §2.3.

**`mean_mult_posterior` used to be a mean over a different set of jets than
`mean_mult_true`, and the truth-nonempty version of it is biased by construction.**
Selecting jets by `N_true ≥ 1` and comparing them to `E_q[N|x]` is regression to the mean:
the deficit is negative even for a perfect posterior. The closure metrics now report the
**full-population** pair (`mean_mult_posterior`, `mean_mult_ratio` — what gate G4 reads)
and the truth-selected pair beside it, flagged. On the v0 checkpoint the two read 0.977
and 0.871 respectively; only the first is a measurement of the model.

### The mandatory validation — and why SBC-on-N is not it

The v1 statistic is the SBC rank of the **multiplicity** `n`. That is a real test for the
implicit continue/stop length model (`ar_junipr_v2`) — but `ar_junipr_v3` trains `q(N|x)` by
direct NLL on `N`, so SBC-on-N certifies *the very marginal the model optimizes*, near
tautologically. **A v2-vs-v3 comparison judged on SBC-N is biased toward v3 by
construction.** That is why the WP4 A/B is gated on the three switches below, not on SBC-N.

All three default **off**, so the reported metric dict is bit-for-bit the pre-v2 dict until
you opt in and existing tables do not move. Source:
[`eval/calibration.py`](../src/h2p_rsd_junipr/eval/calibration.py).

**`pit_coords` — the kinematics, coordinate by coordinate.** Teacher-forces the truth and
evaluates each coordinate's conditional CDF at it. For the AR heads that transform is exact
and *physical* — truncated-normal for the within-cell offsets `du, dv` (the same normalizer
the likelihood divides by), normal for `ln z`, von Mises for `ψ`. For `cinn` / `cfm` it is
the flow's **base space** (a base dimension is not one Lund coordinate, but every base
marginal is exactly `N(0,1)` under a calibrated flow, so each histogram is still a genuine
per-dimension test); the report tags which space it is in so the physical reading is never
implied. `diffusion` and `ar_junipr_v1` have no exact coordinate density and **opt out**
rather than fake one.

Read the shape, not just the number: a **U-shaped** histogram (mass at 0 and 1) means the
head is *over-confident* — too narrow for the data it sees; a **dome** means over-dispersed.
The headline is the KS distance to `Uniform(0,1)`, whose 95% critical value is `1.36/√n` at
the emission count printed beside it. Two breakdowns come with it: `by_emission_index`
(is the *late* emission calibrated, or only the first? — the exposure-bias signature) and,
with `stratify_regions`, `by_region`.

**`stratify_regions` — where calibration holds.** Bins SBC, PIT and coverage by the Lund
quadrant of the leading emission (`wide_soft`, `wide_hard`, `narrow_soft`, `narrow_hard`;
`wide/narrow` split on `ln 1/ΔR`, `soft/hard` on `ln k_t`). Calibration that only holds on
average over the plane cannot pass this — which is the precondition for any *localized*
claim, heavy-ion included.

**`tarp` — the whole tree, in the physics metric.** TARP expected coverage (Lemos et al.,
ICML 2023, arXiv:2302.03026) on tree-valued posteriors, with distance the perturbative-Lund
EMD the MBR estimator already minimizes (so it inherits your `mbr_*` metric configuration).
Per jet, draw a reference tree `r` and compute the credibility level

```
f = (1/K) [ #{k : d(r, y_k) < d(r, y_true)} + ½ #{k : d(r, y_k) = d(r, y_true)} ]
```

(the half-weight on ties is the mid-rank convention the SBC statistic uses; without it the
discrete cell chains tie often enough to fake over-dispersion). Under a calibrated posterior
each `f` is uniform, so `ECP(α) = α`. Reported: the curve, `tarp_max_dev = max|ECP(α) − α|`,
and `ecp_at` — the quotable form, "at 90% credibility the posterior actually covered X%".
**The sign is the diagnosis**: ECP below the diagonal ⇒ over-confident, above ⇒
over-dispersed. This is a *joint* test, which neither SBC-on-N nor the per-coordinate
marginals can be.

`tarp_reference` picks the reference distribution — TARP's guarantee holds for any whose
support covers the posterior's, and the two differ in variance and in which failure they
are most sensitive to, so the choice is recorded in the output. Cost is `n_jets × (K+1)` EMD
solves and it needs the `pot` extra.

```bash
h2p-rsd-junipr eval runs/<id>/best.ckpt \
    experiment.pit_coords=true experiment.stratify_regions=true experiment.tarp=true
```

`eval` writes `eval_metrics.json` plus `calibration_pit_coords.png`,
`calibration_tarp.png` and `calibration_by_region.png` beside the checkpoint (figures need
matplotlib — not a core dependency, but the `[plots]` extra; without it you still get the JSON).

### `closure_continuous` — the leading-emission metric off the grid

`run_closure` scores the leading emission by `lund_distance` between **cell centres**. At the
default geometry a cell is `(6-0)/10 = 0.6` wide and the distances themselves are ~0.6, so that
metric is largely measuring the grid: it cannot resolve what the model is doing, and it can make
the model look worse than plain RSD when it is in fact better.

`closure_continuous=true` repeats the comparison with no quantisation — each draw's leading
emission is placed by [`sample_coordinates`](../src/h2p_rsd_junipr/models/base.py) and summarised
by [`geometric_median`](../src/h2p_rsd_junipr/eval/closure.py) (the L1 Bayes point, which unlike
the cell medoid is not restricted to the drawn support). Four keys are added:
`dlund_identity_cont`, `dlund_posterior_mode_cont`, `dlund_posterior_geomedian_cont` and
`n_continuous_jets`.

```bash
h2p-rsd-junipr eval runs/<id>/best.ckpt experiment.closure_continuous=true
```

```
  leading-emission Lund distance to true y :  identity(x) = 0.694   posterior-mode = 0.714
     posterior-medoid = 0.642   (lower is better; medoid is the loss-matched estimator, ...)
  the same, OFF the cell grid (336 jets) :  identity(x) = 0.643   posterior-mode = 0.714
     posterior-geo-median = 0.594
      (cells are ~0.60 wide, so the cell-level row above is quantisation-limited)
```

**Cost** is `closure_jets × n_closure_samples` forward passes — one `sample_coordinates` call
per draw per jet — which is why it is opt-in. Like the WP2 switches it only *adds* keys, so
existing tables do not move. A family with no coordinate density (`ar_junipr_v1`) returns `None`
from the hook and the `*_cont` keys come back **NaN** rather than absent: "asked, unavailable"
is a different fact from "never asked".

> **`dlund_posterior_medoid` needs no switch.** It is reported on every eval beside
> `dlund_posterior_mode`, because it is pure numpy over cells already drawn. The mode minimises
> expected 0-1 loss while the score is a distance, so it is optimal for a loss nobody measures;
> the medoid is the argmin over the same support of the quantity actually reported, and under the
> model's own posterior it cannot do worse. Judge an arm on the medoid — see
> [PLAN_ProductionAssessment §9.1](PLAN_ProductionAssessment.md) — and use
> [`scripts/leading_estimators.py`](../scripts/leading_estimators.py) for the oracle, per-jet win
> rates, `ln kt` stratification and a paired bootstrap.

---

## 9. Top-level fields

| Field | Default | Meaning |
|---|---|---|
| `run_name` | `"${model.name}_${encoder.name}"` | interpolated run label (e.g. `cinn_lundnet`) |
| `run_root` | `"runs"` | where run directories are written |

---

## 10. Inference knobs in depth — the MAP floor, mincut & quantile floor

> **Read this section as the `ar_junipr_v2` story.** Everything below describes the
> pathology of a *joint* argmax over an implicit continue/stop length model, and the three
> knobs built to patch it. Under `ar_junipr_v3` (and `cinn` / `diffusion` / `cfm`) the
> length is a categorical `q(N|x)` and the pathology is gone at source, so these knobs are
> expected to be no-ops — see "v3 semantics" in §7 and the A/B table it points at.

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

### `point_estimator` — MAP vs MBR (the perturbative-Lund minimum-Bayes-risk tree)

`point_estimator="map"` (default) is everything above: the beam-search joint mode, made
usable by the floors. `point_estimator="mbr"` selects a *different decision rule* on the
**same posterior draws** — the drawn tree of least **expected perturbative-Lund distance**
to the posterior (minimum Bayes risk; Kumar & Byrne 2004, Eikema & Aziz 2020/2022):

```
ŷ_MBR = argmin_{h ∈ C} (1/K) Σ_k d(h, y⁽ᵏ⁾),   C ⊆ {y⁽¹⁾…y⁽ᴷ⁾} ~ q_φ(·|x)
```

where `d` is the Energy Mover's Distance between the two draws as weighted Lund-plane point
clouds (Komiske, Metodiev & Thaler, *PRL* **123** (2019) 041801, arXiv:1902.02346). It reuses
the draws the caller already takes (no extra sampling), returns the **same `LundPointEstimate`
type** as the MAP (a drop-in for every consumer), and additionally reports `.risk` — the
achieved mean distance, a decision-theoretic score, **not** a likelihood (never feed it to
anything expecting an NLL).

**Why it exists — no floor needed.** The empty tree has large expected distance to typical
non-empty draws (it pays the full mass-imbalance penalty `R·|Σw|`), so MBR **cannot select it**
when the posterior is non-empty-dominated — the brevity bias is removed *structurally* rather
than clamped. On a trained checkpoint, floor-free (`min_emissions=0`) MAP collapses to `n=0`
for a large fraction of jets while MBR stays at 0% — the property the `min_emissions` /
`length_floor_quantile` floors had to *enforce*. (Conversely, if a jet's draws are genuinely
mostly empty, MBR picks a short/empty tree — that is *correct*, it reflects the honest
posterior, unlike a floor that manufactures emissions.)

**The metric knobs** (`mbr_lnkt_cut`, `mbr_weight`, `mbr_coords`, `mbr_R`, `mbr_beta`) shape
`d`:

- `mbr_lnkt_cut` is the **metric support** — emissions below it are dropped, so hadronization-
  region jitter cannot dominate the risk. `null` inherits `geometry.ln_kt_range[0]` (the region
  cut) rather than hard-coding a second physics constant.
- `mbr_weight="kt"` weights each Lund point by its transverse momentum (IRC-safe, the KMT
  choice); `z` / `unit` are alternatives.
- `mbr_coords` chooses which columns enter the ground metric (`+psi` engages the periodic
  wrap via `mbr_periodic_phi` / `mbr_phi_col`).
- `mbr_R` sets the **length ↔ kinematics trade-off**: large `R` penalises multiplicity mismatch
  heavily (MBR tracks the count); small `R` favours kinematic agreement of the shared hard
  emissions. Default ≈ the Lund-plane diameter — **scale it with `geometry`** if you change the
  ranges. Check closure-metric stability across `R` (unequal-mass EMD is known to depend on it).
- `mbr_beta=1.0` is the true 1-Wasserstein EMD; `2.0` an energy-distance-like variant.

**The two backends** (`mbr_backend`) implement the *same* mathematical object; the choice is
provenance and batching, not semantics:

- **`pot`** (default) builds the augmented cost by hand — pad the smaller cloud with a sink
  particle of weight `|Σw − Σw'|` at ground distance `R`, then `ot.emd2`. Fewest dependencies
  (`pip install -e ".[mbr]"`), the imbalance term written out to match the equation exactly.
- **`energyflow`** calls the reference `energyflow.emd.emd`/`emds` (`pip install -e
  ".[energyflow]"`). EnergyFlow normalises ground distances by `R` internally, so for `beta=1`
  its value equals the `pot` value **divided by `R`** — the two agree on the **argmin** but not
  the numeric scale. **Pick one backend per analysis** for comparable `risk` numbers; the value
  is reported with its `mbr_backend` tag. (`mbr_norm=True` rescales weights to unit sum, which
  *removes* the imbalance term and the empty-tree-never-wins guarantee — off by default.)
- **`surrogate`** is a fully vectorised binned Lund-image χ² (no OT) — a fast ranker/pre-filter.

Both `ot` and `energyflow` are **lazy, per-backend imports**, so `point_estimator="map"` (the
default) pulls neither and likelihood **parity stays dependency-free**. Reproducing the KMT
collider-event EMD *verbatim* (hadronic coordinates, their `R`, `beta`, `norm`, `periodic_phi`)
is a configuration you dial in through the `mbr_*` knobs — the defaults here are tuned for the
Lund-plane application, not pinned to the paper.

**`mbr_resample_to_qn` — the q(N|x) exposure-bias correction.** MBR candidates are ancestral
draws, so the candidate pool inherits the sampler's marginal-multiplicity bias. With
`mbr_resample_to_qn=True`, each support draw is importance-weighted by
`q(N=|y^(k)| | x) / p_emp(N=|y^(k)|)` — the model's calibrated `length_pmf` over the draws'
empirical multiplicity histogram — so the Monte-Carlo risk expectation matches the calibrated
`q(N|x)` marginal. This is a **decode-layer** fix only: the trained likelihood is untouched
(unlike minimum-risk / sequence fine-tuning, which would distort the ratio you want to preserve).
It is most meaningful with a calibrated head (`ar_junipr_v3`, cINN, diffusion, where `length_pmf`
is exact); for the implicit-length `ar_junipr_v2` the weights collapse to uniform (a no-op),
since `length_pmf` there is the same draw histogram. Off by default. To *measure* whether the
bias survives into MBR before switching it on, the closure suite prints the signed multiplicity
bias of the MBR estimate **stratified by true N** (see USAGE closure output).

```bash
# A/B the point estimator on a fixed checkpoint (floor-free, to see the collapse contrast)
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.point_estimator=map  decode.min_emissions=0
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.point_estimator=mbr  decode.mbr_backend=pot decode.min_emissions=0
h2p-rsd-junipr eval runs/<id>/best.ckpt decode.point_estimator=mbr  decode.mbr_backend=energyflow  # same tree
```

### `cont_temperature` — posterior sampling temperature

Applies a softmax temperature to the cell logits during **ancestral sampling** only (the
posterior, not the MAP). It is the documented exposure-bias remedy for an over-counted
posterior multiplicity: `>1` flattens the cell distribution (more diverse draws), `<1`
sharpens it. It never touches the trained likelihood.

> **`cont_temperature` and `continue_temperature` are different knobs.** The first is a
> softmax temperature on the **cell** logits — it changes *which* emission is drawn. The
> second (below) is a temperature on the **continue/stop** logit — it changes *how many*
> are drawn. The names are close because the first predates the second; the code paths
> share nothing.

### `continue_temperature` — the length knob for the continue/stop family

`p_cont = σ(logit / T)` at sampling only. `T > 1` pulls every continue probability toward
½: where the head is confident to **stop** that lengthens trees, and where it is confident
to **continue** it shortens them, so the direction of a fit depends on which side of ½ the
head sits — do not assume "hotter is longer".

It is an inference-layer object and follows the `tau` pattern from production test v0: fit
on **training-val** jets with `inference.length.fit_continue_temperature`, record the value
*and* its `fitted_under` description in the artifact, then apply it **frozen** to the test
file. That bookkeeping is not ceremony — v0's `tau` was fitted on one scale and applied on
another, which left the ranking intact and the cut in the wrong place, and the empty rate
went from 1.07× to 2.83× off.

```python
from h2p_rsd_junipr.inference.length import fit_continue_temperature

def mean_n(T):                       # measured on TRAINING-VAL jets
    model.continue_temperature = T
    return float(np.mean([len(d) for jet in val for d in model.sample(*jet, 200)]))

T, info = fit_continue_temperature(mean_n, target_mean_n=val_truth_mean_N)
assert info["bracketed"], info        # no root in [0.2, 5] => the knob cannot get there
```

It is a **no-op** for `ar_junipr_v3`/`v4`, `cinn`, `diffusion` and `cfm`: they draw `N`
from an explicit `q(N|x)` and take no per-step continue decision. `build_model` says so out
loud, and `eval_metrics.json` lists it under `decode_inert`. Their length knobs are
`length_temperature` / `length_tilt`.

### `kappa_min_mode` — when the ψ mode is not a prediction

The azimuth head is a von Mises `vM(μ, κ)`. As `κ → 0` the density is **flat**, and its
mode is the direction of a near-zero resultant (Mardia & Jupp, *Directional Statistics*,
Wiley 2000) — a number the head does not actually determine. Production test v0 measured a
median `κ = 0.022` (peak/trough 1.04) while the point estimate reported a ψ resultant
`|R| = 0.69` against a truth of 0.045: a 17.5× pooled row, entirely manufactured by
reporting an unidentified mode as though it were a prediction.

Below `kappa_min_mode` the point estimate carries a **draw** from that same von Mises and
flags the node `psi_identified = False`; above it, the mode, flagged `True`. The default
`0.5` is peak/trough `e^{2κ} ≈ e`. `0.0` restores the ungated mode (the pinned reference
path for the parity harness). The per-node `κ` is recorded on every node, because a ψ panel
is unreadable without it.

Two consequences worth stating:

- **A point estimate with unidentified nodes is stochastic in ψ.** That is the honest
  representation of "flat density", not a defect. The draw comes from a generator private
  to the decode layer (`PosteriorModel.decode_generator`), so it never advances the global
  RNG — taking a point estimate cannot change which posterior draws a later section gets.
- `eval_metrics.json` reports `closure.psi.frac_psi_unidentified` beside the resultants.

### The MBR medoid carries its own coordinates

`mbr_select` returns a genuine posterior **sample**, and it now keeps that sample's own
continuous coordinates for **all four** of them (`coords_source = "sample"`). It used to
re-attach the head modes, which forfeited exactly the property that makes a medoid worth
reporting. `map_estimate` is unchanged — it is still the staged mode decode — but it is a
**diagnostic**, not a headline: the argmax of a high-entropy sequence posterior is an
estimator for a loss nobody is measuring (Stahlberg & Byrne, arXiv:1908.10090; Eikema &
Aziz, arXiv:2005.10283). The decode headline is MBR; the population headline is the
decode-free posterior series.

### `point_estimator="mbr_n"` — decide N first, then the shape

`mbr` minimises a mean distance over **every** multiplicity stratum at once. The
perturbative-Lund EMD carries a mass-imbalance term, so a draw of multiplicity `m` pays
`~R|W_a − W_b|` against every draw of a different `m` — the medoid is pulled toward
whatever `N` is most populous, and can land between strata representing none of them. On
the 600-jet `K=200` arm that leaves it **2.349** from truth while the closest cluster
exemplar is **1.476**, with **83% of the resolvable posterior ambiguity being between-N**.

`mbr_n` splits the decode at that seam, using each channel where it is trustworthy:

```
    n_hat  = Q_0.5( q(N|x) )                      # calibrated on this family
    y_hat  = argmin_{h : |h| = n_hat}  mean_{k : |y_k| = n_hat} d(h, y_k)
```

Stage 1 is the Bayes estimator under `L(n, m) = |n − m|` — which is why this is also the
"general argmin over an explicit loss on n" that
[`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md) deferred, with the empty gate as
its `n = 0` special case. Stage 2 is pure shape: within a stratum every pair has equal
total weight, so the imbalance term drops out of the reduction entirely.

**Cost: zero additional EMD calls** — it is another reduction over the `D` that `mbr`
already builds. It requires `mbr_n_candidates == 0` (restricting both the candidates and
the expectation to one stratum needs the row and column indices to agree) and raises
otherwise.

**Composition with the empty gate.** `decode.empty_threshold` is stage 0 and runs *before*
dispatch in `map_or_mbr`, so it is not duplicated inside the estimator. The interaction is
benign: any sensible τ is below 0.5, so "the gate did not fire" implies `q(0|x) < 0.5` and
the median cannot be 0 on the gated path. With the gate off, a median of 0 honestly returns
the empty tree at risk exactly 0 (the empty clique has zero diameter).

**`.risk` is the WITHIN-STRATUM mean** — the achieved risk of the decision that produced
the tree, which is the only meaning `.risk` has. It is a *different number* from `mbr`'s
global mean, and `LundPointEstimate.estimator == "mbr_n"` is the provenance that keeps the
two from being averaged together. `mbr_resample_to_qn` composes as an exact no-op:
`_qn_importance_weights` assigns one weight per multiplicity, constant within a stratum —
this estimator is the exact form of the correction that knob approximates.

**When the median is unrealised** (an explicit-`q(N|x)` family whose exact softmax median
falls on a multiplicity the finite pool lacks): the nearest populated stratum by
`|n − n_hat|`, ties to the larger pool mass then the smaller `n`. It never raises (an
unrealised median is a runtime state, not a misconfiguration) and never falls back to the
global medoid, which would reintroduce the smearing on exactly the most N-ambiguous jets.
For a continue/stop family the question does not arise: `length_pmf` *is* the draw
histogram, so the median is realised by construction.

**Status: measured, and NOT recommended — `mbr` stays the default.** On 600 held-out jets
at `K=200` it is *significantly farther* from truth than the plain medoid: Δ = −0.083, 95%
CI [−0.128, −0.039]. Its residual RMS is fine (1.003 / 0.999 / 1.024 vs the medoid) and its
multiplicity marginals are the best of any estimator — the shape is right, the *selection*
is worse. Both components lose: restricting the expectation costs −0.043 (cross-stratum
distances are inflated by the imbalance term but are not noise), and the calibrated median
picks N no better than the medoid already does (0.448 vs 0.443 exact).

The same measurement prices what would help: stratifying at the **true** N gives 1.661
against 2.349, so `q(N|x)` is **calibrated but not sharp** — right rate and ranking, wrong
about which N on more than half the jets. Use this knob to reproduce that measurement, not
as a production decode. Full table in
[`PLAN_StratifiedMBR.md`](PLAN_StratifiedMBR.md) §1a.

### `cluster_posterior` — a set of explanations instead of one tree

`mbr_select` returns the **Fréchet median restricted to the sample**: the draw of least
*mean* EMD to the posterior. That is a *centrality* criterion and the right default. It is
the wrong criterion when the posterior is **multimodal** — the medoid of a two-lobed
posterior can land in the sparse valley between the lobes, minimising mean distance while
representing neither explanation. The sample space is transdimensional,
`Y = ⊔_N C^N`, and the strata are metrically separated by the EMD's imbalance term, so
"one hard emission" and "two softer emissions consistent with the same `x`" are genuinely
*alternative shower histories* rather than two ends of one continuum.

`decode.cluster_posterior=true` clusters the **same** `K×K` matrix `mbr_select` already
builds and hands back one genuine posterior draw per cluster, with its mass:

```python
ps = model.predict_set(xf, nx, draws=draws, **decode)   # a sibling of map_or_mbr
ps.members[0]   # the top-mass exemplar — a LundPointEstimate, coords_source="sample"
ps.masses       # posterior mass per cluster, mass-descending
ps.radii        # mean within-cluster EMD to the exemplar
ps.top_mass, ps.entropy
```

**The point estimate does not move.** `cluster_posterior` consumes `D`; `_reduce_risk`
consumes `D`; neither sees the other's output, so `labels`, `exemplars`, `masses`, `radii`,
`top_mass` and `entropy` are *bit-identical* across every risk reduction, and `map_or_mbr`
returns the same tree and the same `.risk` whether or not a set was also taken. That
orthogonality is what lets the cluster layer ship at stock MBR settings, and
`tests/test_clusters.py::test_losses_do_not_move_clusters` asserts it rather than trusting
the argument.

**Three per-jet numbers, and they are not one ±.** A bimodal posterior summarised as
mean ± sd points at a configuration neither mode supports, so the three are reported
separately and `LundPointEstimate` carries the first two (`cluster_mass`,
`cluster_entropy`, `None` on every other path):

| quantity | what it is | quotable as ±? |
|---|---|---|
| `top_mass` | a **probability** — the posterior mass of the selected explanation | no |
| `entropy` `H(m) = −Σ m log m` | an **ambiguity** over discrete alternatives, in nats | no |
| `radii[0]` | the **width** of the selected explanation | **yes**, and only this one |

`top_mass` is *not* a calibrated probability out of the box: the joint tree posterior is
over-confident by v1 TARP, and with `cluster_split=false` it is additionally biased **high**
because the same draws define the cluster and are then counted into it (post-selection
inference; Berk, Brown, Buja, Zhang & Zhao, *Ann. Statist.* **41** (2013) 802). Turn on
`experiment.cluster_diagnostics` to measure both — reliability diagram, ECE, the Brier
decomposition and the split-vs-no-split difference — before quoting it.

**Three guards, and every one raises rather than warns** (a mass vector nobody can see is a
number that gets quoted anyway). All three are measured facts, not conventions:

1. **`mbr_n_candidates` must be 0.** With a candidate cap `D` is `|C|×K` and there is no
   pairwise matrix over the posterior to cluster. Not silently overridden: the cap changes
   which point estimate you get, so overriding it would answer a different question.
2. **`mbr_beta` must be 1.0.** At β ≠ 1 the perturbative-Lund EMD violates the triangle
   inequality — 300 violations in 64 000 measured triples at β = 2, against 0 at β = 1 —
   and HDBSCAN's mutual-reachability construction assumes a metric.
3. **`mbr_R ≥ R_max/2`** for the active `mbr_coords`, KMT's condition for the EMD to be a
   metric. Computed from *your* `geometry` block, not hard-coded at 8.485, so a non-default
   range cannot silently break it. At the default `[0, 6]²` the diagonal is `6√2 = 8.485`,
   exactly `mbr_R`'s default; `+lnz` and `+psi` raise it to ≈ 9.9 and ≈ 11.7, still leaving
   margin.

**`empty_threshold` reaches the set too, and means the same thing.** The `N = 0` stratum is
*atomic* — every empty draw sits at mutual distance exactly 0, so it forms one zero-radius
cluster carrying the whole of `q(0|x)` — while the non-empty draws are *fragmented* into
several clusters. The mass argmax therefore compares one lump against the largest of a split
field, and the empty explanation wins on far more jets than its own mass warrants: measured
**29.8% against a true rate of 16.7%** on 600 held-out jets at `K = 200`. Setting
`decode.empty_threshold` hands that decision to the calibrated gate instead
([`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md)); gate G3 says the two numbers are
the *same number*, so this changes only what it is compared against. Only
`PosteriorSetEstimate.point` moves — `members`, `masses`, `radii` and the conformal prefix
are untouched, and `members[0]` keeps meaning "the top-mass exemplar". Unlike `map_or_mbr`
the gate does not short-circuit the decode here: the set still carries the empty
explanation, because a rejected alternative is still a reported alternative. `0.0` (the
default) is off and bit-identical.

**`mbr_backend="surrogate"` is refused** for a cluster mass vector. Not principally for its
two triangle violations in 64 000 — `_lund_image` **normalises**, so the surrogate is
*exactly* blind to total `k_t` and multiplicity and collapses the very `N`-stratum
separation that makes the clusters physical. It stays admissible as a cheap screening pass
for the medoid-in-dominant-cluster verdict, which is robust to that collapse, via an
explicit `screening_only=True` at the Python level; there is no config route to it.

**Dependency.** `cluster_method` `hdbscan`/`dbscan` need `scikit-learn ≥ 1.3`, added under
the existing **`[mbr]` extra** — the `point_estimator="map"` path must import nothing new,
and both are lazy-imported inside their branch. `cluster_method="pam"` is pure NumPy and
deterministic, which makes it both the no-dependency fallback and the control arm for
whether the gate-G2 verdict is method-dependent.

**`K` is what the mass vector's resolution is.** At the default `n_posterior_samples=500`
and `cluster_min_mass=0.05` a reportable cluster is 25 draws, and the Monte-Carlo error on
a mass of 0.6 is `√(0.6·0.4/500) ≈ 0.022`. Whether the medoid sits in the dominant cluster
is answerable there; *how many* clusters there are is not. Density estimation needs
resolution in the sample space itself, and the sample size to resolve **modes** scales far
worse than the one to estimate a mean — so raise `K` before quoting a three-cluster split,
and remember the `K²` EMD block grows with it.

**The bounded-loss reduction is deliberately not a config field.** `mbr_select` accepts an
eval-only `diagnostic_losses=("bounded", "kernel")` side channel that returns the alternative
argmins beside an *unchanged* `LundPointEstimate`. It is kept out of `DecodeConfig` because
`.risk` is documented as "the achieved mean distance" and has fourteen consumers, five of
which aggregate it across jets; under a bounded loss it silently becomes a dimensionless
neighbour deficit, and none of them break loudly. The columns those diagnostics produce live
in [`eval/stability.py`](../src/h2p_rsd_junipr/eval/stability.py) — **not**
`eval/systematics.py`, and the module boundary is the guard: the linear-vs-bounded spread is
a *stability* check, not a systematic, because the two are different functionals of one
posterior (a Fréchet median and a density mode) rather than two approximations to one
quantity, and the posterior width is already reported by `radii[0]`.
`tests/test_stability.py::test_loss_spread_not_in_systematics` asserts the boundary.

**ε is pre-registered, not tuned.** Where a bandwidth is needed (`dbscan`, and the bounded
diagnostics) it is `Q_γ` of the *positive* off-diagonal distances with γ fixed in advance.
Tuning it against a closure metric is forbidden: it is the one free parameter the
construction turns on, and a closure-tuned bandwidth makes the conformal gate circular. The
quantile form is also what makes it invariant to the `mbr_norm` / `energyflow` `1/R`
convention. Note what it excludes: `_empty_value` returns exactly 0 for two empty clouds, so
the `N = 0` clique is invisible to the bandwidth rule while remaining decisive in the
neighbour tally — that is the hazard the `empty_clique_size` column exists to measure.

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
