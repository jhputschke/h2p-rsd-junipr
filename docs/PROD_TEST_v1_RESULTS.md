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

## 1. Before the grid: the v0 diagnosis was upside down

The plan's WP-B.1 requires the multiplicity diagnostic to run **before** any remedy is
defaulted. It did — on the committed v0 checkpoint, 2 000 held-out jets from
`data/jet_aux_asym_test.root`, no retraining — and it retired the remedy. Together with
the recomputed TARP band (WP-D.2) it also **inverts** v0 §9's central reading.

v0 concluded: *"TARP passing while SBC-on-N fails localises the residue to the
multiplicity rather than the tree shape."* Both halves of that sentence were artefacts of
the null each statistic was quoted against. Measured against nulls recomputed at this
run's own size:

| | v0 said | v1 measures | why it changed |
|---|---|---|---|
| SBC-on-N | χ² 107 vs "crit 16.90" ⇒ **fails** | 88th percentile of its own null ⇒ **consistent with calibrated** | χ²(9) is the null for a *continuous* rank; `N` takes 7 values here |
| TARP | max dev 0.037 vs floor 0.079 at n = 300 ⇒ **passes** | max dev 0.046 vs a recomputed 0.027 at n = 2000 ⇒ **fails**, over-confident | `1.36/√n` is asymptotic, and at n = 300 the band was too wide to resolve anything |

**The residue is in the tree shape, not the multiplicity — the opposite of the v0
reading.** The multiplicity marginal is calibrated on every statistic that has a null;
the joint tree posterior is measurably too narrow (`ECP(0.68) = 0.635` against 0.68,
signed deviation −0.016), and the leading-cell coverage says the same thing
(0.53, 95% Wilson [0.51, 0.56] on 1 676 jets).

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

### 1.3 TARP, at the tier where its band can resolve anything

TARP's `1.36/√n` floor is asymptotic. Recomputed by Monte Carlo at the run's own
`(n, α grid)`:

| n_jets | recomputed 95% null | analytic `1.36/√n` | quotable (floor < 0.05)? |
|---:|---:|---:|---|
| 300 | 0.073 | 0.079 | **no** — cannot resolve a 5% miscalibration |
| 2 000 | 0.027 | 0.030 | yes |

At 2 000 jets with 200 references, the v0 checkpoint gives **max dev 0.046 against a 0.027
band — a real deviation**, signed −0.016, `ECP(0.68) = 0.635`. Over-confident: the tree
posterior is too narrow. v0's "0.037 inside 0.079" at 300 jets was a statement about the
sample size, and gate G7 now refuses to quote the statistic until the band's own floor is
below 0.05.

Stratified, the deviation is where the jets are: `wide_soft` 0.035 against its own 0.032
band on 1 555 jets; `narrow_soft` 0.098 against 0.123 on 96 (inside, but a band that wide
says little); `wide_hard` 25 jets, not scored.

### 1.4 The support error is 5× bigger than v0 measured

v0 found 0.88% of sampled emissions below the soft-drop boundary. It never checked the
*other* wall. At 2 000 jets / 553 184 sampled emissions:

| series | out of window | soft drop | **`z > ½`** | `k_t` floor | verdict |
|---|---:|---:|---:|---:|---|
| truth | 0.00000% | 0.00000% | 0.00000% | 0.00000% | control passes |
| posterior (`lnz_support: legacy`) | 0.00000% | 0.83263% | **3.94317%** | 0.00000% | **FAIL** |

`z = min(p_{T1},p_{T2})/(p_{T1}{+}p_{T2}) ≤ ½` by construction, so 3.94% of the posterior's
emissions are not soft prongs at all. This is what an unbounded Normal on a bounded
coordinate does, and it is the whole case for WP-A: under `lnz_support: physical` both
columns are zero by construction rather than by luck.

### 1.5 Where the `ln z` failure lives

The region × coordinate PIT cross, ranked by **KS over its own 95% critical value**
(regions differ in count by 50×, so raw KS would name the smallest region every time):

| coordinate × region | KS | n | crit | KS/crit |
|---|---:|---:|---:|---:|
| `ln_z` × `wide_soft` | 0.077 | 2 671 | 0.026 | **2.91×** |
| `dv` × `wide_soft` | 0.032 | 2 671 | 0.026 | 1.21× |
| `dv` × `wide_hard` | 0.152 | 52 | 0.189 | 0.81× |
| `ln_z` × `narrow_soft` | 0.083 | 111 | 0.129 | 0.64× |

Two things follow. The `ln z` misfit is **not** localized to a quadrant — it is in the
quadrant that holds 94% of the emissions, which is what a support error looks like rather
than a conditioning failure. And `narrow_soft` — v0's worst region, coverage 0.38
[0.28, 0.47] on 96 jets here — has **no** coordinate PIT failure at all (worst cell 0.64×
its critical value). Its coverage deficit is therefore not attributable to a miscalibrated
coordinate there, which is exactly the kind of documented attribution gate G5 allows.

---

## 2. Is the test valid?

The plan's §9 carries v0 §1's validity checks unchanged, and they carry because **v1 uses
the same two files**: `data/jet_aux_asym.root` (seed 1) to train, `data/jet_aux_asym_test.root`
(seed 2) to report on.

- **Seeds did not collide.** `runs/prod_test_v0/disjoint.json`: 0 overlapping jets of
  20 000 compared under the `full` fingerprint (sequences + jet four-vector), and 0 under
  the sequence-only one. `passed: true`.
- **The test file is asymmetric.** `kt_floor = 1.0`, `kt_floor_sec = 0.2` on both files, so
  the secondary-plane aux columns are groomed looser than the sequences beside them — which
  is the only reason the aux arms can see this file at all.
- **Same physics.** `(z_cut, beta, kt_floor, kt_floor_sec, generator)` are identical
  between the two files, so the assessment measures generalisation rather than covariate
  shift.
- **The `ln z` support declaration is verified against the data, not assumed.** Before any
  arm trains, `data.stats.check_lnz_support` confirms the declared `(z_cut, beta) = (0.1, 0)`
  matches the file's own grooming record *and* that all **700 330** truth emissions lie
  inside `[ln 0.1, ln ½] = [−2.3026, −0.6931]`. Both endpoints are attained in the data,
  which is what makes the bounded head the right one rather than merely a tighter one.

## 3. Grid arms

*Pending — filled by `scripts/prod_test_v1_gates.py` when the grid completes.*

## 4. Gates G1–G8

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

## 5. Retroactive pass on the v0 checkpoint

The training-free items — the WP-C estimator repairs and the whole WP-D assessment block —
run on the committed v0 checkpoint
`runs/prod_test_v0/20260731-212800-8209a78a33/best.ckpt` with **no retraining**, which is
what makes §1 a statement about the *same* model v0 reported on. Per the plan's §13.4 this
is an **addendum**, not a revision: every number in
[`PROD_TEST_v0_RESULTS.md`](PROD_TEST_v0_RESULTS.md) stands as the record of what that
run's suite reported. Artifacts: `eval_metrics_calib.json` (2 000-jet calibration tier),
`eval_metrics_decode.json` (300-jet decode tier), merged into `eval_metrics.json`.

### 5.1 The ψ pathology was entirely in the decode rule

Each row against the uniform floor `√π / 2√n` for its own node count — because `|R|` is a
norm and is positive under isotropy too:

| series | nodes | \|R\| | uniform E\|R\| | Rayleigh p |
|---|---:|---:|---:|---:|
| truth | 427 | 0.0781 | 0.0429 | 0.074 |
| **point estimate (repaired MBR)** | 325 | **0.0360** | 0.0492 | **0.656** |
| posterior draws | 73 156 | 0.0055 | 0.0033 | 0.111 |

v0's decode reported `|R| = 0.69` against a truth of 0.045 — a 15× anisotropy it
manufactured by attaching the *mode* of a von Mises whose median κ is 0.022. The repaired
decode reports 0.036, **below its own uniform floor**, Rayleigh p = 0.66: no preferred
azimuth at all, which is what the physics says there should not be. `frac_psi_unidentified`
is 0.0% because the MBR estimate now carries sampled coordinates, where mode
identifiability is not a question that applies.

### 5.2 Acceptance and decode, on the same checkpoint

| quantity | value | criterion |
|---|---:|---|
| medoid / identity, cell tier | **0.929** | G1: < 1 |
| geo-median / identity, off-grid tier | **0.927** | G1: < 1, agreeing in sign — it does |
| repaired MBR / identity | **0.943** | G6: < 1 |
| ψ \|R\| point / truth | 0.46× | G6: within 2× — but both rows are at their uniform floors, so the ratio is noise; the substantive result is 4.1 |

So on the v0 checkpoint, with only the decode layer repaired: **G1 and G6 pass**, G4
passes (§1.1), and **G2, G3 and G7 fail** — the support error (§1.4), the `ln z` PIT
(§1.5), and the tree-level over-confidence (§1.3). Every one of the three failures is
something v1's grid is designed to move, and G2/G3 are precisely what WP-A addresses.

## 6. What is still broken

*Pending.*

## 7. What is not measured

- **PYTHIA vs HERWIG.** Still no `herwig_driver`; the train/test deltas remain the noise
  floor stand-in, exactly as in v0 §10.
- **`v1_wide` (`dec_dim = 128`).** The plan lists it as optional; it is not in the grid.
- **The per-node joint coordinate density.** Deferred with a trigger (plan §12): G3
  failing on the truncated head, or the region × coordinate PITs showing the
  independence-given-cell approximation binding.
