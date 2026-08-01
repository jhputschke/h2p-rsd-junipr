# Production test v1 — results

**Status: in progress.** The grid specified by
[`PLAN_prod_test_v1.md`](PLAN_prod_test_v1.md) §8 is running; this document holds the
results as they land. The plan holds the design and the rationale, and every pass
criterion in it was fixed before the grid started — read the plan for *why* each gate
exists. Companion to [`PROD_TEST_v0_RESULTS.md`](PROD_TEST_v0_RESULTS.md).

Regenerate the gate tables with:

```bash
python scripts/prod_test_v1_gates.py --run-root runs/prod_test_v1
```

| | |
|---|---|
| grid | `scripts/run_prod_test_v1.sh` — 11 trainings, `presets/prod_test_v1.yaml` + one override each |
| base arm | `ar_junipr_v4` + `lundnet`, aux(9), `n_bins = 30`, 60 epochs, **`lnz_support: physical`** |
| train | `data/jet_aux_asym.root` — 495 071 jets, seed 1 |
| test | `data/jet_aux_asym_test.root` — 97 018 jets, seed 2 |
| grooming | `z_cut = 0.1`, `beta = 0`, `kt_floor = 1.0`, `kt_floor_sec = 0.2` |
| environment | conda `js_fno`, torch 2.11+cu130, one NVIDIA GB10 |

> ⚠️ **NLL is not comparable to v0, or to the `v1_legacy_lnz` arm.** `lnz_support:
> physical` changes the coordinate normalization, which shifts NLL/jet by a constant that
> has nothing to do with fit quality. v1 NLLs are comparable only within v1 and only
> across arms sharing the same `ln z` head. The `v1_legacy_lnz` arm exists as the bridge
> to the v0 record, and it is an *attribution* arm, not a candidate.

---

## 0. Two corrections to the plan, made before the grid ran

Both were found while implementing, both are recorded in
[`PLAN_prod_test_v1.md`](PLAN_prod_test_v1.md)'s status block, and neither changes a gate.

1. **`decode.cont_temperature` was already taken.** The plan's §5.2 proposes that name for
   a scalar on the *continue* logit; the field has existed since WP2 as the softmax
   temperature on the *cell* logits. The new knob ships as
   **`decode.continue_temperature`**; `cont_temperature` keeps its meaning.
2. **`v1_base` and `v1_nhead` were the same model.** `configs/model/ar_junipr_v4.yaml`
   already sets `use_multiplicity_head: true`, so the plan's two G8 arms differ in
   nothing. G8's stated rationale — SBC-N must not decide, *because* the explicit head is
   calibrated on it by construction — only has content when one arm has the head and the
   other does not. The missing arm is the **implicit continue/stop** one, fielded as
   **`v1_contstop`**. G8 is `v1_base` (explicit) vs `v1_contstop` (implicit). It is also
   the only arm on which `continue_temperature` is not a no-op.

---

## 1. Before the grid: two of v0's five findings were the measurement

The plan's WP-B.1 requires the multiplicity diagnostic to run **before** any remedy is
defaulted. It did, on the v0 checkpoint, and it retired the remedy — the ⟨N⟩ deficit and
the SBC-on-N failure are both artefacts of the reference each was quoted against, not
properties of the model. This is the single most consequential result of v1 so far, and it
cost no training at all.

*(Numbers: [§1.1](#11-the-n-deficit-was-a-selection-effect-and-a-mispairing) and
[§1.2](#12-sbc-on-n-had-no-χ²9-null) below, both measured on 2 000 held-out jets from
`data/jet_aux_asym_test.root` with the committed v0 checkpoint.)*

### 1.1 The ⟨N⟩ deficit was a selection effect and a mispairing

| population | jets | ⟨N⟩ truth | ⟨N⟩ posterior | ratio | signed bias |
|---|---:|---:|---:|---:|---:|
| **full (what G4 reads)** | 2 000 | 1.4170 | 1.3845 | **0.9771** | −0.0325 |
| truth `N ≥ 1` (selected on truth) | 1 676 | 1.6909 | 1.4727 | 0.8709 | −0.2183 |

The posterior mean is the exact `E_q[N|x]`, which is what `sample` draws from. Two things
were wrong with v0's "1.15 vs 1.40":

- **The metric paired the wrong jets.** `mean_mult_posterior` was
  `mean(b + len(val_ds[i]["yc"]) for i, b in enumerate(n_mean_bias))`, where `i` indexes
  the *kept* jets and `val_ds[i]` the *unfiltered* dataset — so each kept jet's bias was
  added to a different jet's truth multiplicity.
- **The population it used is biased low by construction.** Selecting jets by `N_true ≥ 1`
  and comparing them to `E_q[N|x]` is regression to the mean: the deficit is negative even
  for a perfectly calibrated posterior. That is the whole of the second row above.

The full-population ratio **0.977 is inside G4's `[0.95, 1.05]`**, with no temperature
applied. `decode.continue_temperature` was implemented anyway — §3 of the plan
pre-registers it and the `v1_contstop` arm is the one family it acts on — and is reported
as the null check the plan says it becomes when the untempered sampler passes.

### 1.2 SBC-on-N had no χ²(9) null

| statistic | value |
|---|---:|
| observed χ² (10 bins, exact mid-rank, 2 000 jets) | 215.6 |
| **its own null** (truth redrawn from `q(N\|x)`, 200 reps): mean | 190.7 |
| the same null, 95% point | 225.1 |
| **where the observed value sits in its own null** | **88th percentile** |
| the reference v0 quoted, χ²(9) 95% point | 16.90 |

`N` takes 7 distinct values on this sample. A mid-rank statistic on a discrete quantity
lands on a handful of atoms and **cannot** be uniform on `[0,1]` for any model, so χ²(9) —
the null for a *continuous* rank — is the wrong reference by construction, not by a
subtlety. Against the only defensible reference, the multiplicity marginal is consistent
with calibrated.

The same failure mode had a third instance, fixed at the same time: TARP's `1.36/√n` floor
is asymptotic, and at v0's `n = 300` the correctly recomputed 95% null point is **0.073**.
A band that wide cannot resolve a 5% miscalibration, so "max dev 0.037, inside the band"
was a statement about the sample size. At `n = 2000` the recomputed point is 0.028. Gate
G7 now requires the band's own floor to be below 0.05 before the statistic is quoted.

---

## 2. Grid arms

*Pending — filled by `scripts/prod_test_v1_gates.py` when the grid completes.*

## 3. Gates G1–G8

*Pending.*

| # | gate | verdict | numbers |
|---|---|---|---|
| G1 | acceptance | | |
| G2 | support | | |
| G3 | `ln z` PIT | | |
| G4 | N marginal | | |
| G5 | `narrow_soft` | | |
| G6 | decode | | |
| G7 | TARP | | |
| G8 | family A/B | | |

## 4. Retroactive pass on the v0 checkpoint

The training-free items — `continue_temperature`, the WP-C estimator repairs, and the
whole WP-D assessment block — run on the committed v0 checkpoint
`runs/prod_test_v0/20260731-212800-8209a78a33/best.ckpt` with no retraining, which is what
makes §1 above a statement about the *same* model v0 reported on. Per the plan's §13.4
this is an **addendum** to the v0 record, not a revision of it: every number in
[`PROD_TEST_v0_RESULTS.md`](PROD_TEST_v0_RESULTS.md) stands as the record of what that
run's suite reported.

*Pending the full 2 000-jet pass.*

## 5. What is still broken

*Pending.*

## 6. What is not measured

- **PYTHIA vs HERWIG.** Still no `herwig_driver`; the train/test deltas remain the noise
  floor stand-in, exactly as in v0 §10.
- **`v1_wide` (`dec_dim = 128`).** The plan lists it as optional; it is not in the grid.
- **The per-node joint coordinate density.** Deferred with a trigger (plan §12): G3
  failing on the truncated head, or the region × coordinate PITs showing the
  independence-given-cell approximation binding.
