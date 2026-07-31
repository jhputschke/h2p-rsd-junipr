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
> checks the data against), and `decode.max_emissions` (beam/sample length cap). Each is
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
`model=ar_junipr_v2|ar_junipr_v1|ar_junipr_v3|ar_junipr_v4|cinn|diffusion|cfm` binds a
specific schema. All families expose the same `log_prob`/`sample`/`map_estimate` contract.
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

**`ar_junipr_v2` vs `ar_junipr_v1`** is exactly `continuous_coords` True vs False — v1 drops
the coordinate density and is the categorical-cell-only backbone.

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
| `cont_temperature` | `1.0` | posterior | softmax temperature on the cell logits at **sampling** time (exposure-bias remedy); `>1` flattens, `<1` sharpens |
| `min_emissions` | `1` | MAP | **hard floor** on MAP length — the "mincut" (never the unphysical empty tree) |
| `length_penalty` | `0.0` | MAP | GNMT `score/len**α` at final beam rank; counters the brevity bias; `0` = off |
| `length_floor_quantile` | `0.0` | MAP | **learned per-jet floor** at the α-quantile of `P(n|x)`; `0` = off |
| `point_estimator` | `"map"` | point estimate | `map` (beam-search joint mode) or `mbr` (minimum-Bayes-risk tree; §10 MBR) |
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

`min_emissions`, `length_penalty`, and `length_floor_quantile` are explained in depth in
§10; `point_estimator` / `mbr_*` — the whole second point-estimate family — are covered in
the §10 MBR subsection. All are inference-time only, so you can A/B them on a fixed checkpoint.

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

Trade cost vs. precision with `closure_jets` and `n_closure_samples`.

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
