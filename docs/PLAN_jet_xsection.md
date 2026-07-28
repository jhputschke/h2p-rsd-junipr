# Does the steeply falling jet cross-section bias h2p-rsd-junipr training?

## Context

The model learns an amortized posterior `q_φ(y|x) ≈ p(y|x)` — parton-level groomed
Lund tree `y` given the hadron-level one `x`. Jet production cross-sections fall
steeply with `p_T`, so the natural worry is that the training sample's `p_T`
composition biases the learned posterior. This document records the analysis, the
measurements that back it, and the minimal work needed to make the effect
observable and correctable.

All numbers below were measured on `cpp/test_data/jets_aux.root` (54 007 jets,
25 000 events, PYTHIA-8.3 Monash, `pp_dijet.cmnd`).

---

## Findings

### 1. In principle there is no bias to the conditional — this is covariate shift, not confounding

Training minimizes `E_{p_train(x)}[ KL( p(y|x) ‖ q(y|x) ) ]`
([trainer.py:121-122](../src/h2p_rsd_junipr/train/trainer.py#L121-L122)). Hadronization is
applied per jet, so `p(y|x)` is a property of the physics and is invariant under any
reweighting of the *marginal* `p(x)`. The minimizer is `p(y|x)` regardless of the
spectrum. In the infinite-data / infinite-capacity limit the cross-section shape is
irrelevant.

That is the reassuring half. The rest is where it stops being true.

### 2. The sample is not steeply falling — it is pT-hat sculpted, and matches nothing physical

[`pp_dijet.cmnd:13`](../cpp/cards/pp_dijet.cmnd#L13) sets `PhaseSpace:pTHatMin = 100.`,
one unbounded slice, and `weight ≡ 1.0` for **every** jet (verified: `np.unique(weight) == [1.]`).
Accepted jets nonetheless span 20–826 GeV because `HadronJet:ptMin = 20` admits
subleading/ISR jets. The result peaks at 100–150 GeV:

| pT bin (GeV) | jets | train frac | physical frac (`pT^-5`) | over/under |
|---|---|---|---|---|
| 20–40   | 11 774 | 0.218 | 0.975 | **0.22×** |
| 40–60   |  7 007 | 0.130 | 0.021 | 6.2× |
| 60–100  | 14 653 | 0.271 | 0.0033 | 83× |
| 100–150 | 15 086 | 0.279 | 0.0003 | 820× |
| 150–250 |  4 858 | 0.090 | ~1e-5 | 7 300× |
| 250–900 |    629 | 0.012 | ~1e-7 | 1.3e5× |

So the model is not trained on a steeply falling spectrum at all. The per-jet
`weight` channel is fully plumbed — written at [lund_writer.cpp:17](../cpp/src/lund_writer.cpp#L17),
read at [rntuple.py:77](../src/h2p_rsd_junipr/data/rntuple.py#L77), tensorized at
[dataset.py:77](../src/h2p_rsd_junipr/data/dataset.py#L77), applied at
[trainer.py:122](../src/h2p_rsd_junipr/train/trainer.py#L122) — and is a **no-op**, because
`HardQCD:all` with a single hard `pTHatMin` is unweighted. There is no slice
stitching, no `xsec`/`pThat`/`mcChannelNumber` column, and no reweighting anywhere.

### 3. The conditional genuinely depends on pT — multiplicity does not screen it off

`⟨n_y⟩` at **fixed** hadron multiplicity `n_x`, per pT bin:

| n_x | 20–40 | 40–60 | 60–100 | 100–150 | 150–250 | 250–900 |
|---|---|---|---|---|---|---|
| 0 | 1.199 | 1.241 | 1.237 | 1.207 | 1.208 | 1.114 |
| 1 | 0.901 | 1.015 | 1.068 | 1.071 | 1.094 | 1.074 |
| 2 | 1.164 | 1.401 | 1.573 | 1.640 | 1.731 | 1.847 |
| 3 | 1.343 | 1.688 | 1.961 | 2.136 | 2.253 | 2.324 |
| 4 |   —   | 1.933 | 2.269 | 2.436 | 2.759 | 2.909 |

At `n_x = 3` the mean parton multiplicity swings **73%** across the pT range. The
model must read the jet scale out of the kinematics; it cannot get it from length.

### 4. It can — jet pT is analytically recoverable from the first Lund node

For the first primary emission `ln k_t = ln z + ln p_T + ln ΔR`, hence

```
ln p_T  =  x_lnkt[0] − x_lnz[0] + x_lnInvDelta[0]
```

Measured against the true `jet_pt`: **corr = 0.990, residual σ = 0.088 in ln pT**
(the residual is grooming removing momentum from the parent). The encoder consumes
*raw continuous* features with no clipping
([dataset.py:62](../src/h2p_rsd_junipr/data/dataset.py#L62) →
[features.node_features:194-204](../src/h2p_rsd_junipr/features.py#L194-L204)), so this
linear combination is available in the first Linear layer.

This also explains the failed aux A/B: `ln_pt` gave ΔNLL = −0.029 against a seed
spread of 0.029 (`runs/aux_input_ab/ab_summary.json`) — not because pT is
irrelevant, but because **it is already in the input**.

### 5. The one structural blind spot looks benign

For the 6.9% of jets with `n_x = 0` the encoder sees an empty sequence and cannot
know pT at all — and aux conditioning cannot rescue it either
([features.py:217](../src/h2p_rsd_junipr/features.py#L217), "no rows to carry the aux
signal"). But the `n_x = 0` row of the table above is flat in pT
(1.199 → 1.114), so the conditional there is close to pT-independent. First moment
only; worth confirming on the full distribution, but this is not the leak.

### 6. Where the shift actually costs: capacity allocation, and it tilts the wrong way

- **High pT is starved.** 1.2% of jets above 250 GeV, 0.13% above 400 GeV — and
  these have the longest Lund sequences and the widest perturbative band
  (`[ln k_t^floor, ln(p_T R/2)]` grows like `ln p_T`). Hardest part of the task,
  least data.
- **Low pT is starved *relative to any physical application*.** The relative
  hadronization correction is largest there (`⟨n_y⟩/⟨n_x⟩` = 0.79 at 20–40 GeV vs
  0.88 at 250+), and a physically weighted sample is >97% 20–40 GeV.
- **Length weighting compounds it.** `loss = Σ w·NLL / Σ w` with per-jet NLL a sum
  over emissions, so long sequences dominate the gradient. Measured gradient share
  ÷ jet share: 0.84× at 20–40 GeV rising monotonically to 1.17× at 250+. Mild, but
  it tilts *further* toward high pT.
- **Not fixable post-hoc.** Importance-reweighting this sample to `pT^-5` leaves
  `N_eff = 6 125` of 54 007 (**11%**), with almost all weight in the thinnest bin.
  Matching a physical spectrum requires generating low-`pTHatMin` slices.

### 7. Two latent landmines that fire the moment weights become non-trivial

- [`trainer.py:122`](../src/h2p_rsd_junipr/train/trainer.py#L122) normalizes **per batch**:
  `(w*nll).sum() / w.sum()`. With real slice cross-section weights (spanning ~10^6),
  a batch containing one high-weight jet is normalized by that jet and every other
  jet contributes nothing — the effective learning rate becomes a random variable.
  Correct at `w ≡ 1`; pathological otherwise.
- **`eval/` ignores `w` entirely** — no `batch["w"]` in `closure.py`,
  `calibration.py`, or `systematics.py`. Train/val NLL would be weighted while every
  closure/PIT/SBC/TARP number stays unweighted.

### 8. Geometry bounds are tuned to this card (currently fine, silently fragile)

`ln_kt_range = [0,6]`, `ln_invdelta_range = [0,6]`
([configs/geometry/default.yaml](../configs/geometry/default.yaml)) with hard clipping in
[`Geometry.to_cell:58-65`](../src/h2p_rsd_junipr/geometry.py#L58-L65). Measured clipping in
this sample: **0.0% on all four bounds** — `ktFloor = 1.0 GeV` pins the lower `ln k_t`
edge at exactly 0 and `R = 0.4` keeps `ln 1/ΔR ≥ 0.22`. No current bias. But
[`PLAN_Input.md:208-215`](PLAN_Input.md) contemplates `R = 0.8` /
`ktFloor = 0.2`, which breaks both lower edges, and a genuinely high-pT slice starts
clipping the top. Nothing checks this.

### 9. The measurement gap — none of the above is currently observable

No metric anywhere is binned in jet pT. Stratification exists, but only by
leading-emission Lund quadrant
([`cell_region`, calibration.py:46-55](../src/h2p_rsd_junipr/eval/calibration.py#L46-L55))
and by truth multiplicity ([`closure.py:23`](../src/h2p_rsd_junipr/eval/closure.py#L23)).
`MatchedLundDataset` does not even carry `jet_pt` into the item dict
([dataset.py:69-79](../src/h2p_rsd_junipr/data/dataset.py#L69-L79)), so pT-binning is not
possible downstream today. Every mechanism argued in §4 is plausible and untested.

---

## Bottom line

The steeply falling cross-section is **not** a bias source for a well-specified
conditional model, and this conditional is well-specified and already carries pT in
its inputs. The real exposures are, in priority order:

1. Nothing is binned in pT, so a pT-dependent failure is invisible.
2. The training spectrum matches nothing physical, and cannot be reweighted to
   match (N_eff = 11%).
3. The weighted loss and the unweighted eval are correct only because `w ≡ 1`.

**Make it measurable first. Then decide whether the spectrum needs fixing.**

---

## Scope

**Stage A only — make it measurable.** Fixed physical pT bin edges.

Stages B and C are written out below in full so they can be picked up cold, but are
explicitly **not** being implemented in this pass. Read them in order — B1 gates C1,
and B2 gates B3.

| | | |
|---|---|---|
| **A** | make it measurable | **this pass** |
| **B** | make the weight channel correct (no-ops at `w ≡ 1`) | deferred |
| **C** | fix the spectrum (multi-slice generation) | deferred; needs B1 |

Guiding constraint throughout: every addition is **off by default** and the existing
`eval_metrics.json` must stay key-for-key identical until opted in — the same
discipline `stratify_regions` / `pit_coords` / `tarp` already follow
([config.py:216-222](../src/h2p_rsd_junipr/config.py#L216-L222)).

---

## Work — Stage A

### A1. Carry `jet_pt` through the dataset

`MatchedLundDataset` currently drops it
([dataset.py:69-79](../src/h2p_rsd_junipr/data/dataset.py#L69-L79)), and this is the
blocker: `run_calibration` / `coordinate_pits` / `run_tarp` receive **only** `val_ds`
([calibration.py:96,199,293](../src/h2p_rsd_junipr/eval/calibration.py#L96)), so they have
no other route to the jet scale. (`run_closure` also gets `val_jets`
([closure.py:80](../src/h2p_rsd_junipr/eval/closure.py#L80)) and could read
`val_jets[i]["jet_pt"]` directly, but one uniform source is better than two.)

- Add `jet_pt=torch.tensor(float(j.get("jet_pt", float("nan"))))` to the item dict.
  `.get` with a NaN default matters: synthetic jets carry no pT
  ([synthetic.py:86](../src/h2p_rsd_junipr/data/synthetic.py#L86)) and must not raise —
  they simply fall outside every bin.
- Add the matching `jet_pt` stack to `collate`
  ([dataset.py:88-111](../src/h2p_rsd_junipr/data/dataset.py#L88-L111)) as a `(B,)` tensor.
  Additive; no model reads it.
- `rntuple.py` already loads the column ([rntuple.py:83,104](../src/h2p_rsd_junipr/data/rntuple.py#L83)),
  so nothing changes in the reader.

### A2. `pt_bin` — a pT stratification axis, mirroring `cell_region`

Add next to [`cell_region`](../src/h2p_rsd_junipr/eval/calibration.py#L46-L55), same shape
(`-> str | None`, `None` meaning "not classifiable", which is how NaN synthetic pT
falls out for free):

```python
PT_EDGES  = (20.0, 40.0, 60.0, 100.0, 150.0, 250.0)   # GeV; fixed, never data-derived
PT_LABELS = ("pt20_40", "pt40_60", "pt60_100", "pt100_150", "pt150_250", "pt250_inf")
```

Fixed edges, not quantiles, for the reason already stated in
[features.py:25-31](../src/h2p_rsd_junipr/features.py#L25-L31): a data-derived binning
silently re-scales when the spectrum changes, and these numbers exist precisely to be
compared across samples and checkpoints.

Wire as a `by_pt` block, alongside the existing `by_region`:
- `run_calibration` — the stratification loop at
  [calibration.py:357-384](../src/h2p_rsd_junipr/eval/calibration.py#L357-L384)
- `coordinate_pits` — [calibration.py:129-151](../src/h2p_rsd_junipr/eval/calibration.py#L129-L151),
  reusing `_uniformity_report`
- `run_closure` — a second loop beside `mult_bias_by_N`
  ([closure.py:159-175](../src/h2p_rsd_junipr/eval/closure.py#L159-L175)), producing
  `mult_bias_by_pt` and per-bin `dlund_posterior_mode`. This is the panel that would
  actually show §3/§6 if it were real.

Gate on a new `experiment.stratify_pt: bool = False` — added in **both** places the
existing flags live: the dataclass at
[config.py:211-222](../src/h2p_rsd_junipr/config.py#L211-L222) and the tolerant-defaults
mirror at [config.py:403-407](../src/h2p_rsd_junipr/config.py#L403-L407) (the latter is what
lets old checkpoint snapshots rebuild).

Also add a per-bin `n_jets` count to every `by_pt` entry — with 1.2% of jets above
250 GeV, a reader must be able to see when a bin is too thin to interpret.

### A3. Spectrum + clipping guard at setup

New `spectrum_stats(jets)` / `check_spectrum(jets, geometry, cfg, *, strict, verbose)`
in [`data/stats.py`](../src/h2p_rsd_junipr/data/stats.py), modeled directly on
[`check_multiplicity_support`](../src/h2p_rsd_junipr/data/stats.py#L65-L102) — same
`strict` / `verbose` signature, same "message names the knob to change" style, same
call sites ([cli.py:79](../src/h2p_rsd_junipr/cli.py#L79) strict for train,
[cli.py:125](../src/h2p_rsd_junipr/cli.py#L125) `strict=False` for eval).

Two things reported, one enforced:
- **Reported**: jet-pT quantiles (1/25/50/75/99/max) and the per-bin counts over
  `PT_EDGES`. Pure visibility — today nothing prints the spectrum at all.
- **Enforced**: fraction of `x` and `y` nodes that `Geometry.to_cell` clips at any of
  the four bounds. Currently **0.0%**; error above ~1e-3, warn above ~1e-4, matching
  the existing `SUPPORT_TAIL_ERROR` / `SUPPORT_TAIL_WARN` thresholds. The message
  must name `geometry.ln_kt_range` / `geometry.ln_invdelta_range` and the recorded
  `kt_floor` — reuse [`_grooming_context`](../src/h2p_rsd_junipr/data/stats.py#L42-L48),
  which already pulls `z_cut`/`beta`/`kt_floor` off the jet and degrades gracefully on
  synthetic data.

This is the §8 landmine: the `R = 0.8` / `ktFloor = 0.2` variant in
[PLAN_Input.md:208-215](PLAN_Input.md) breaks both lower edges and would
otherwise clip silently.

### A4. Document the finding

Cross-reference this document from [README_PHYSICS.md](README_PHYSICS.md) — specifically
the `ln p_T = x_lnkt[0] − x_lnz[0] + x_lnInvDelta[0]` identity and its measured 0.088
residual, since that is the load-bearing reason the design is sound and it is currently
written down nowhere else. Add a line to [notebooks/README.md](../notebooks/README.md)
if a spectrum panel is added there.

---

## Work — Stage B (DEFERRED, not in this pass)

Written out in full so it can be picked up cold. B1 and B2 are strict **no-ops at
`w ≡ 1`** — they change nothing about the current results and exist purely so the
weight channel is correct before it ever carries a real cross-section. Both should
land *before* any multi-slice sample is generated (Stage C), not after.

### B1. Fix the weighted-loss normalization (§7)

[trainer.py:122](../src/h2p_rsd_junipr/train/trainer.py#L122) currently divides by the
**batch's** weight sum:

```python
loss = (batch["w"] * nll).sum() / batch["w"].sum().clamp(min=1e-8)
```

With real slice weights this makes the effective learning rate a random variable: a
batch containing one jet from a low-`pTHatMin` slice is normalized by that jet, so
every other jet in the batch contributes ~nothing to the gradient. Replace the divisor
with a **fixed dataset-level mean weight**, so the batch retains its natural size
weighting:

```python
loss = (batch["w"] * nll).sum() / (self.mean_w * batch["w"].numel())
```

Implementation notes:
- Compute `self.mean_w` once in
  [`Trainer.__init__`](../src/h2p_rsd_junipr/train/trainer.py#L72-L82) from
  `self.train_loader.dataset` — `MatchedLundDataset.items` each hold a scalar `w`
  ([dataset.py:77](../src/h2p_rsd_junipr/data/dataset.py#L77)). Doing it inside
  `__init__` keeps the call signature at [cli.py:90](../src/h2p_rsd_junipr/cli.py#L90)
  unchanged **and** covers [`Trainer.resume`](../src/h2p_rsd_junipr/cli.py#L82-L84),
  which has its own construction path. Guard `mean_w > 0`.
- It is deterministic from the dataset, so it needs **no** checkpoint field — resume
  recomputes the identical value. Don't add it to `save_checkpoint`.
- `_validate` ([trainer.py:135-146](../src/h2p_rsd_junipr/train/trainer.py#L135-L146))
  should keep its `num/den` form: val NLL/jet is a *reported mean*, not a gradient, so
  the weighted mean is the right estimator there. Only the training divisor changes.
- At `w ≡ 1`, `mean_w = 1` and `w.sum() == w.numel()`, so the loss is bit-identical.

Also log **`N_eff = (Σw)² / Σw²`** over the train set, once at setup — the single
number that says whether the weighting has silently destroyed the sample. Print it
next to the existing jet counts at [cli.py:74-75](../src/h2p_rsd_junipr/cli.py#L74-L75)
and add it to the epoch log dict at
[trainer.py:91-95](../src/h2p_rsd_junipr/train/trainer.py#L91-L95) if it is ever made
per-epoch. For the current sample it prints `N_eff = 54007` exactly.

### B2. Honor `w` in eval (§7)

`eval/` contains no `batch["w"]` anywhere. Once weights are real, train/val NLL would
be weighted while every closure/PIT/SBC/TARP number stays unweighted — a silent
train/eval inconsistency, and one that would first show up as an unexplained
disagreement between the loss curve and the calibration plots.

Two tiers, and they are **not** equally cheap:

- **Easy — `closure.py`.** It accumulates plain Python lists (`d_id`, `d_mode`,
  `n_mean_bias`, `n_median_bias`, `covered`, `true_ns`) and reduces them with
  `np.mean` / `np.nanmean`
  ([closure.py:140-175](../src/h2p_rsd_junipr/eval/closure.py#L140-L175)). Accumulate a
  parallel `w_kept` list inside the same loop (it already skips jets at
  [closure.py:113-114](../src/h2p_rsd_junipr/eval/closure.py#L113-L114), so alignment
  matters) and swap the reductions for `np.average(v, weights=w)`. The
  `mult_bias_by_N` and new `mult_bias_by_pt` loops take `w_kept[sel]` the same way.
- **Harder — `calibration.py`.** `_uniformity_report`
  ([calibration.py:80-90](../src/h2p_rsd_junipr/eval/calibration.py#L80-L90)) is
  straightforward (`np.histogram(..., weights=w)`, weighted `mean`), but
  **`_ks_uniform` ([calibration.py:61-71](../src/h2p_rsd_junipr/eval/calibration.py#L61-L71))
  is not** — the KS statistic needs a weighted ECDF, and its usual `1.36/√n` critical
  value no longer applies; the honest substitute is `1.36/√N_eff`. Do not let this be
  a silent one-line change: either implement the weighted ECDF and switch the quoted
  critical value to use `N_eff`, or leave KS unweighted and **say so in the metric
  dict** (e.g. a `ks_unweighted: true` flag) rather than letting a reader assume it
  matches the weighted histogram beside it.

Same principle as B1: at `w ≡ 1` every number must be unchanged, which is the
regression test.

### B3. Reweighting stress test

An analysis probe, not library code. New `scripts/spectrum_stress.py`, following the
existing probe pattern (`scripts/probe_map_collapse.py`, `scripts/ab_v2_v3.py`,
writing a JSON summary under `runs/<name>/` the way
`runs/aux_input_ab/ab_summary.json` does).

What it does: load a trained checkpoint and the val set, then re-score the **same**
jets under importance weights `w_i ∝ pT_i^{-n}` for `n ∈ {0, 4.5, 5.5}` — `n = 0` is
the current unweighted baseline and must reproduce `eval_metrics.json` exactly — and
report how the headline numbers (val NLL/jet, `mult_bias_posterior`,
`dlund_posterior_mode`, `coverage_68`) move as the spectrum is tilted toward physical.

This answers *"would a physical spectrum change the conclusion?"* without regenerating
anything. It depends on B2, since the whole point is weighted metrics.

**The caveat must be printed by the script itself, not just documented here:**
reweighting this sample to `pT^-5` leaves `N_eff = 6 125` of 54 007 (**11%**), with
almost all the weight in the 20–40 GeV bin — the thinnest region of the Lund-plane
coverage. So the test is *indicative of the direction and rough size* of the shift and
is **not** a substitute for generating low-`pTHatMin` slices. Have the script emit
`N_eff` per `n` alongside every metric so no reader can quote a number without it.

---

## Work — Stage C (DEFERRED, larger; separate decision)

### C1. Multi-slice generation + stitching (§2)

The actual fix for §2, and the only way to get a training spectrum that matches a
physical application. Roughly:

- Generate several `pTHatMin` slices from
  [`pp_dijet.cmnd`](../cpp/cards/pp_dijet.cmnd) (the parameter is
  [line 13](../cpp/cards/pp_dijet.cmnd#L13), also hardcoded as a default at
  [pythia_driver.cpp:43](../cpp/apps/pythia_driver.cpp#L43)), each written to its own
  `jets.root`.
- Write the per-slice `σ / N_gen` into the existing `weight` column
  ([lund_writer.cpp:17](../cpp/src/lund_writer.cpp#L17), filled from
  `pythia.info.weight()` at
  [pythia_driver.cpp:87](../cpp/apps/pythia_driver.cpp#L87)) — the channel already
  exists end-to-end, which is the one part of this that is free.
- Add a stitching path to
  [`LundDataModule.setup`](../src/h2p_rsd_junipr/data/datamodule.py#L55-L73), which today
  makes exactly one `load_rntuple(d.path, d.ntuple)` call with no glob and no
  concatenation. `DataConfig.path` is a single string
  ([config.py:41](../src/h2p_rsd_junipr/config.py#L41)) and would need to accept a list.
- `_fingerprint` ([datamodule.py:24-40](../src/h2p_rsd_junipr/data/datamodule.py#L24-L40))
  must hash **all** source paths, or two different slice combinations collide to the
  same fingerprint and the preprocessed-tensor cache serves the wrong data.
- Overlapping slices double-count in the overlap region; either use exclusive
  `pTHatMin`/`pTHatMax` windows or accept the standard slice-stitching bookkeeping.

**Ordering constraint: B1 must land first.** Slice cross-sections span ~10⁶, and the
per-batch normalization pathology in §7 bites the moment the first weighted sample is
loaded — the symptom would be a training run that simply fails to converge, with
nothing in the loss curve explaining why.

---

## Verification

- `pytest` — full suite green. `tests/test_data.py` and `tests/test_config.py` pin
  current shapes and config keys, so A1/A2 must not disturb them.
- **Byte-identical default path.** Run `train` + `eval` on a fixed seed *before* the
  change, save `eval_metrics.json`, repeat after with `stratify_pt` off, and `diff`.
  Must be identical key-for-key — this is the acceptance test for A2.
- **Opt-in path.** Then:
  ```
  PYTHONPATH=src <conda python> -m h2p_rsd_junipr.cli train \
      data=rntuple data.path=cpp/test_data/jets_aux.root trainer=fast_dev
  PYTHONPATH=src <conda python> -m h2p_rsd_junipr.cli eval runs/<...>/best.ckpt \
      experiment.stratify_pt=true
  ```
  Confirm `by_pt` / `mult_bias_by_pt` appear, per-bin `n_jets` sums to the number of
  jets actually scored, and the bin populations track the §2 table (≈22 / 13 / 27 /
  28 / 9 / 1 %).
- **Guard fires.** `geometry.ln_kt_range=[0,2]` must raise, with a message naming
  `ln_kt_range` and the recorded `kt_floor`. `geometry.ln_kt_range=[0,6]` (the
  default) must stay silent — measured clipping is 0.0%.
- **Synthetic path unbroken.** `data=synthetic` with `stratify_pt=true` must run and
  simply report every jet as unbinned, not raise — this is what the NaN default in A1
  buys.
- **Sanity check against the analysis.** The `mult_bias_by_pt` panel should reproduce
  the direction of §3 (`⟨n_y⟩` rising with pT at fixed `n_x`); if the model has learned
  the scale from node 0 as argued in §4, the *bias* should be roughly flat across bins
  even though the underlying `⟨n_y⟩` is not. **That contrast is the actual experiment
  this whole stage exists to run.**

Run everything with `PYTHONPATH=src` and the `fno_env_mlx` conda python
(`/opt/homebrew/Caskroom/miniconda/base/envs/fno_env_mlx/bin/python`) — the package is
not pip-installed.
