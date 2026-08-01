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

**What the seed band covers.** The arms vary `trainer.seed`, which sets weight
initialisation and batch order; `data.seed` stays 0, so **every arm sees the same
train/val split** (fingerprint `ca253883e8a6`, 445 563 train / 49 508 val). That is v0's
convention unchanged — and it is the right one here, because the aux A/B (`v1_base` vs
`v1_ctrl`) is only paired if both arms are fitted to the same jets. The band therefore
measures initialisation and ordering variance, **not** split variance, and no claim below
extends to the latter.

| arm | varies | seeds | best val NLL/jet | band |
|---|---|---:|---:|---:|
| `v1_base` | — (v4 + lundnet + aux(9) + physical `ln z`) | 3 | **3.9169** | [3.9036, 3.9237] |
| `v1_ctrl` | `aux_features = [ln_pt, abs_eta]` | 3 | **3.9215** | [3.9124, 3.9386] |
| `v1_contstop` | implicit continue/stop (no `q(N\|x)` head) | 2 | **3.8283** | [3.7799, 3.8768] |
| `v1_legacy_lnz` | `lnz_support = legacy` | 1 | 4.0703 ! | — |
| `v1_gru` / `v1_deepsets` | `encoder` | 1 each | *training* | — |

`!` — not comparable to the rows above it. A different `ln z` normalization shifts NLL/jet
by a constant unrelated to fit quality; the 4.0703 is *identical* to the v0 checkpoint's,
which is the point of the arm. The `v1_contstop` row **is** comparable: both
factorizations are normalized densities over the same space and share the physical `ln z`
head.

### 3.1 Aux isolation — the secondary-plane columns buy nothing measurable

The question plan §8 poses is *isolation*, not existence: v0 §5 already established that
aux conditioning helps at all (aux vs no-aux). v1 asks what the **secondary-plane and
groomed-mass** columns add over pure jet kinematics.

| quantity | `v1_base` (9 columns) | `v1_ctrl` (`ln_pt`, `abs_eta`) | delta | clears the spread? |
|---|---|---|---:|---|
| best val NLL/jet | 3.9169 [3.9036, 3.9237] | 3.9215 [3.9124, 3.9386] | −0.0045 | no |
| `ln z` PIT KS | 0.0423 [0.0270, 0.0529] | 0.0466 [0.0382, 0.0592] | −0.0043 | no |
| `pit_ks_max` | 0.0441 [0.0324, 0.0529] | 0.0466 [0.0382, 0.0592] | −0.0025 | no |
| TARP max dev | 0.0367 [0.0335, 0.0415] | 0.0360 [0.0340, 0.0395] | +0.0007 | no |
| `coverage_68` | 0.5255 [0.5179, 0.5400] | 0.5366 [0.5280, 0.5507] | −0.0111 | no |
| medoid/identity | 0.9312 [0.9240, 0.9373] | 0.9386 [0.9309, 0.9504] | −0.0073 | no |

**Null on every deciding metric.** Every delta is a fraction of the seed spread it sits
in — and two of them (`ln z` PIT KS, `pit_ks_max`) *changed sign* when the third `v1_ctrl`
seed landed, which is as direct a demonstration as one gets that they measure the seed
rather than the columns. This is v0 §5's warning realized in the other direction: that A/B
failed because a −0.029 nat delta *was* the 0.029 spread, and it is why this plan required
three seeds per arm.

Two claims, and only the second is new: aux conditioning helps (v0), **and `ln_pt` +
`abs_eta` appear to carry the whole of that gain** (v1). The seven secondary-plane and
groomed-mass columns are not adding to it.

Consequently plan §12's WP3 trigger — *"`v1_base` vs `v1_ctrl` shows the secondary columns
carry the aux gain"* — **does not fire.** Full-tree LundNet with secondary-plane sequences
stays deferred, and this is evidence for leaving it there rather than an absence of
evidence. Two caveats bound that: the arms are matched at fixed capacity, so this says the
columns add nothing *as fed to this encoder*, not that the secondary planes are
uninformative; and the aux columns are groomed at `kt_floor_sec = 0.2` against the
sequences' 1.0, which is the asymmetry that makes them non-degenerate at all.

### 3.2 Encoder probe — `lundnet` is not the best of the three

v0 §10 flagged this as open: `lundnet` is the pairing
[`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §4 *names* for the A4 arm,
not a measured winner. One seed each, so the only yardstick is the `v1_base` band at the
same configuration.

| quantity | `v1_base` (lundnet, 3 seeds) | `v1_gru` (1) | `v1_deepsets` (1) | outside the band |
|---|---|---:|---:|---|
| best val NLL/jet | 3.9169 [3.9036, 3.9237] | **3.8988** | **3.8855** | both |
| `ln z` PIT KS | 0.0423 [0.0270, 0.0529] | 0.0537 | 0.0513 | `gru` (worse) |
| TARP max dev | 0.0367 [0.0335, 0.0415] | **0.0275** | 0.0360 | `gru` (better) |
| `coverage_68` | 0.5255 [0.5179, 0.5400] | **0.5519** | 0.5328 | `gru` (better) |
| medoid/identity | 0.9312 [0.9240, 0.9373] | 0.9254 | 0.9253 | no |

**Both alternatives beat `lundnet`'s three-seed NLL band**, and `gru` additionally lands
its TARP at 0.0275 — *exactly* its recomputed null, the only explicit-`q(N|x)` arm in the
grid that does not fail G7 outright — with the best leading-cell coverage of any arm here.
It pays for it on the `ln z` PIT, where it is the worst.

**This is a probe, not a result, and the distinction is the whole point of §3.1.** One seed
cannot be compared to a band on equal terms: `lundnet`'s own band is 0.020 wide and `gru`
sits 0.005 outside it. What the row licenses is "worth a proper multi-seed A/B", which is
exactly what v0 §10 asked for and what this grid was not sized to deliver — the plan
budgets one training per encoder. It does *not* license changing the fielded pairing.

Two things make it more interesting than a tie, though. The encoder was never a suspect in
§6's defect, and yet `gru` moves TARP and coverage — the two instruments that read that
defect — in the same direction that `v1_contstop` moves them (§4.8). And it does so while
being *worse* on the coordinate PIT, so it is not simply a better model.

## 4. Gates G1–G8

Verdicts on `v1_base`, evaluated on **all three seeds** — a gate evaluated on one seed is
a gate evaluated on one draw from the band, and G3's value spans 0.027–0.053 against a
0.0255 criterion, close enough that "which seed" could have decided it. It does not: every
verdict below is unanimous across the three.

| # | gate | verdict | the number |
|---|---|---|---|
| G1 | acceptance | **PASS** | medoid/identity 0.924–0.937, geo-median/identity 0.925–0.932, agreeing in sign |
| G2 | support | **PASS** | 0.0000% on all four boundaries, all three seeds |
| G3 | `ln z` PIT | **FAIL** | KS 0.0529 / 0.0270 / 0.0471 vs crit 0.0255 |
| G4 | N marginal | **FAIL** | ⟨N⟩ ratio **passes** (1.008 / 0.994 / 0.987); regional coverage fails |
| G5 | `narrow_soft` | **ATTRIBUTED** | no coordinate there exceeds its own critical value |
| G6 | decode | **PASS** | MBR/identity 0.936–0.965; the ψ clause is underpowered, see §4.6 |
| G7 | TARP | **FAIL** | 0.0415 / 0.0350 / 0.0335 vs a recomputed null of 0.0275 |
| G8 | family A/B | **implicit continue/stop wins** | held-out NLL −0.124 nat and TARP 0.021 vs 0.037, both clearing the seed spread; PIT, coverage and medoid tie |

### 4.1 G2 — support: the one gate WP-A was built to move, and it moves completely

| series | out of window | soft drop | `z > ½` | `k_t` floor | on the bound |
|---|---:|---:|---:|---:|---:|
| truth (control) | 0.00000% | 0.00000% | 0.00000% | 0.00000% | — |
| `v1_legacy_lnz_s0` | 0.00000% | **0.83263%** | **3.94300%** | 0.00000% | — |
| `v1_base` (3 seeds) | 0.00000% | **0.00000%** | **0.00000%** | 0.00000% | 0.012–0.030% |

The `legacy` arm reproduces the v0 checkpoint's rates to five significant figures — it is
the same configuration on the same data with the same seed, so it should, and the fact
that it does is what licenses reading the difference as attributable to `lnz_support`
alone. Every violation is gone under the bounded head, and 0.012–0.030% of draws sit
*exactly on* a bound: that is the truncation doing its job, and it is reported rather than
being allowed to look like a violation (see `eval.support.EDGE_TOL`).

### 4.2 G3 — `ln z` PIT: halved, not closed

| arm | KS | ×crit (0.0255) |
|---|---:|---:|
| v0 checkpoint / `v1_legacy_lnz_s0` | 0.0734 | 2.88× |
| `v1_base_s0` | 0.0529 | 2.07× |
| `v1_base_s1` | 0.0270 | 1.05× |
| `v1_base_s2` | 0.0471 | 1.84× |

**G3 fails.** Putting `ln z` on its correct support removes the *support* error entirely
(§4.1) and roughly halves the *calibration* error, but the residual is still significant on
every seed — the best of the three misses by 5%. So the two failures were never the same
failure: the leak was the support, and what is left is a shape mismatch inside the
interval, which a truncation cannot fix.

This is the pre-registered trigger in plan §4.4: a monotone rational-quadratic spline on
the same interval (Durkan et al., arXiv:1906.04032), **as a follow-up plan, not a mid-run
change**. No such change was made.

The region × coordinate cross localises it: `ln_z × wide_soft` at **2.16×** its critical
value on 2 671 emissions, with every other scored cell below 1.02×. `wide_soft` holds 94%
of the emissions, so this is a mismatch in the bulk rather than in a corner.

### 4.3 G4 — N marginal: the ⟨N⟩ clause passes, the coverage clause fails

`⟨N⟩_post/⟨N⟩_truth` = 1.0075 / 0.9939 / 0.9866, comfortably inside `[0.95, 1.05]`, with
no temperature applied — confirming §1.1 on the retrained arms. The gate nonetheless fails
on its regional clause: leading-cell 68% coverage is 0.518–0.540 against a target of 0.68,
and no scoreable region is Wilson-consistent with it.

SBC-on-N against its own null sits at the 95th percentile on seed 0 — exactly on the line,
and with 200 null reps that percentile carries ±1.5%, so it is genuinely marginal rather
than a clean pass or fail. It is reported as marginal; the rep count was not raised
mid-run to resolve it.

### 4.4 G5 — `narrow_soft`: attributed

Coverage **0.344–0.479** across the three seeds (seed 0: 0.479 [0.38, 0.58] on 96 jets),
against v0's 0.375 [0.28, 0.47]. The band straddles the v0 value, so this quadrant is
**not** measurably improved — a single-seed read of 0.479 would have said it was. It stays
well below the 0.68 target on every seed. The region × coordinate cross shows **no coordinate exceeding its own
critical value in that quadrant** (worst 0.69×), so the deficit is not a coordinate
miscalibration there. That is the documented mechanistic attribution the gate allows, and
it points the follow-up at the tree-level width (§4.7), not at a coordinate head.

### 4.5 G1 — acceptance: carried

Medoid/identity 0.924–0.937 on the cell tier and geo-median/identity 0.925–0.932 off the
grid, both below 1 and agreeing in sign on every seed. The arm beats the identity baseline, which
is the criterion v0 established and the one thing that has to hold before any of the rest
means anything.

### 4.6 G6 — decode: passes, with the ψ clause underpowered

Repaired MBR/identity **0.936–0.965, below 1 on every seed**. The ψ clause cannot be evaluated as written: `|R|` is
a norm and is positive under isotropy too, so a ratio of two resultants is a measurement
only when both are resolved above their own uniform floors. Point estimate 0.0401 against
a floor of 0.0449 (Rayleigh p = 0.53) — *below* its floor, i.e. no anisotropy at all. The
gate's substance — that the decode does not manufacture anisotropy the posterior does not
have — is met; its stated 2× ratio is not measurable at this sample size.

### 4.7 G7 — TARP: fails, and it is the finding

| arm | max dev | recomputed null 95% | signed |
|---|---:|---:|---:|
| v0 checkpoint | 0.046 | 0.027 | −0.016 |
| `v1_base_s0` | 0.0415 | 0.0275 | — |
| `v1_base_s1` | 0.0350 | 0.0275 | — |
| `v1_base_s2` | 0.0335 | 0.0275 | — |

Every seed exceeds its band. WP-A moved it slightly (0.046 → 0.033–0.042) and did not close
it — which is what §1 predicted: the residue is a **width** problem in the joint tree
posterior, and fixing a support error was never going to reach it. Together with the
coverage clause of G4 and the attribution in G5, three independent instruments say the
same thing about the same defect.

§4.8 identifies what causes it.

### 4.8 G8 — and the defect turns out to be the multiplicity factorization

| quantity | `v1_base` explicit `q(N\|x)` (3 seeds) | `v1_contstop` implicit (2 seeds) | delta | clears the spread? |
|---|---|---|---:|---|
| **best val NLL/jet** | 3.9169 [3.9036, 3.9237] | **3.7927** [3.7799, 3.8054] | **+0.1242** | **yes** |
| **TARP max dev** | 0.0367 [0.0335, 0.0415] | **0.0212** [0.0200, 0.0225] | **+0.0154** | **yes** |
| `ln z` PIT KS | 0.0423 [0.0270, 0.0529] | 0.0398 [0.0315, 0.0482] | +0.0025 | no |
| `pit_ks_max` | 0.0441 [0.0324, 0.0529] | 0.0398 [0.0315, 0.0482] | +0.0043 | no |
| `coverage_68` | 0.5255 [0.5179, 0.5400] | 0.5307 [0.5304, 0.5310] | −0.0053 | no |
| medoid/identity | 0.9312 [0.9240, 0.9373] | 0.9307 [0.9286, 0.9327] | +0.0006 | no |

**The implicit continue/stop factorization wins on two of the four deciding metrics and
ties on the rest.** NLL is comparable here — both are normalized densities over the same
space, sharing the physical `ln z` head — unlike across the WP-A head change. SBC-on-N is
reported and does not decide, per the gate's own rule.

And G7 is not close. Across the whole grid:

| family | arms | TARP max dev | ECP(0.68) | signed bias | G7 |
|---|---:|---|---|---|---|
| explicit `q(N\|x)` | 6 | 0.0335–0.0465 | 0.635–0.652 | −0.005 … −0.020 | **FAIL ×6** |
| implicit continue/stop | 2 | **0.0200, 0.0225** | **0.665** | −0.0002, −0.0036 | **PASS ×2** |

Those six span three seeds, two aux configurations *and* both `ln z` heads; the two that
pass differ from `v1_base` in the multiplicity factorization and nothing else. So the
defect §6 describes — the joint tree posterior being too narrow — **is attributable to the
explicit `q(N|x)` head**, and the signed bias going from −0.020 to −0.0002 says it is not
merely reduced but essentially removed.

A mechanism consistent with this, offered as a hypothesis rather than a result: the
explicit factorization draws `N ~ q(N|x)` and *then* decodes exactly `N` cells, so length
is independent of shape given `x`. The continue/stop model lets the prefix decide when to
stop, coupling the two. TARP is a *joint* tree statistic, and it is precisely the joint
that an independence assumption would narrow. That is a testable claim and this run does
not test it.

**This inverts the expectation the plan carried into G8.** The gate's rationale was
protective of the explicit head — SBC-N must not decide *because* the head is calibrated on
it by construction — and the concern was that the A/B would be biased in its favour. The
deciding metrics say the opposite, which is the outcome a pre-registered rule exists to
make reportable.

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

**One defect, seen by three instruments.** The joint tree posterior is too narrow.

| instrument | reading | what it is not |
|---|---|---|
| G7 TARP | 0.0335–0.0415 vs a 0.0275 null, signed negative | not the multiplicity: SBC-on-N is consistent with its own null |
| G4 coverage | leading-cell 68% coverage 0.518–0.540 | not the ⟨N⟩ marginal: that ratio is 0.987–1.008 |
| G5 attribution | `narrow_soft` 0.344–0.479, with **no** coordinate above its critical value there | not a coordinate head in that quadrant |

Each of the three could individually be explained away; together they cannot. Note what
the list does *not* contain: any statistic about the multiplicity *marginal*. v0 read the
residue as a multiplicity failure because the two statistics that would have said
otherwise were quoted against nulls they do not have (§1).

**And §4.8 attributes it.** Every arm carrying the explicit `q(N|x)` head fails G7 — six
of them, spanning three seeds, two aux configurations and both `ln z` heads — while both
arms differing only in the multiplicity **factorization** pass, with the signed TARP bias
going from −0.020 to −0.0002. So the defect is neither the coordinate heads nor the
encoder nor the support: it is how length is attached to shape. That is a third distinct
reading of "multiplicity", and it is worth separating from the two v0 conflated: the
*marginal* `q(N|x)` is calibrated (§1.1, §1.2), and the *factorization* that marginal
belongs to is what narrows the joint.

**`ln z` is still miscalibrated inside its own support** — 1.05–2.07× its critical value
after WP-A, concentrated in `wide_soft` at 2.16× on 94% of the emissions. The support half
of this failure is closed; the shape half is not.

### The triggers that have fired

Plan §12 defers three things *with stated triggers*. Two have now fired, and they are
recorded here rather than acted on, because acting on them mid-run is exactly what a
pre-registered plan exists to prevent:

1. **Per-node joint coordinate density** (cINN coords / CFM, `PLAN_UPDATES.md` WP1).
   Trigger: *"G3 fails on the truncated head."* **Fired** (§4.2). The deeper structural
   reason the plan gives is worth restating now that the number agrees with it: under the
   `LundGenerator` conventions `ln z = u + v − ln p_T,sum` holds *exactly*, so
   coordinate-independence-given-cell is violated by a kinematic identity — and a
   truncation cannot repair a factorization error, which is precisely the residual shape.
2. **Monotone rational-quadratic spline on the same interval** (Durkan et al.,
   arXiv:1906.04032), plan §4.4's pre-authorized escalation for exactly this outcome.
   **Fired.** It is the cheaper of the two and does not disturb the factorization.
3. **Full-tree LundNet with secondary-plane sequences** (WP3; Dreyer & Qu,
   arXiv:2012.08526). Trigger: *"`v1_base` vs `v1_ctrl` shows the secondary columns carry
   the aux gain."* — see §3; **pending the third `v1_ctrl` seed.**

Nothing in this run was changed in response to any of them.

## 7. Verdict

**The arm passes acceptance and support; the run's main product is an attribution.**

Pre-registered gates on `v1_base`, unanimous across three seeds: **G1 PASS, G2 PASS,
G3 FAIL, G4 FAIL, G5 ATTRIBUTED, G6 PASS, G7 FAIL**, and G8 decided against the fielded
family.

Four things this run establishes that v0 could not:

1. **WP-A works, completely, on the thing it was built for.** Every support violation is
   gone — 0.83% below soft drop and 3.94% above `z = ½` become 0.0000% — and the
   `v1_legacy_lnz` arm reproduces the v0 rates to five significant figures, so the
   difference is attributable to `lnz_support` and nothing else.
2. **The `ln z` failure was two failures.** The support half is closed; the shape half
   (1.05–2.07× its critical value, concentrated in the quadrant holding 94% of emissions)
   is not, and a truncation cannot close it. Plan §4.4's spline escalation and §12's
   joint-density trigger have both fired.
3. **v0's central reading was inverted, and the real defect is now located.** The joint
   tree posterior is too narrow — read by TARP, by leading-cell coverage, and by the
   `narrow_soft` attribution — and it is **the multiplicity factorization**, not the
   coordinate heads, the encoder, or the support: all six explicit-`q(N|x)` arms fail G7,
   both continue/stop arms pass, and the signed bias goes from −0.020 to −0.0002. The
   `q(N|x)` *marginal* is calibrated; the factorization it sits in is what narrows the joint.
4. **Two open questions from v0 are answered, one negatively.** The aux *isolation* is null
   on every deciding metric — `ln_pt` + `abs_eta` carry the whole of the aux gain — so
   WP3's trigger does not fire. The encoder probe says `lundnet` is not the best of the
   three and deserves the multi-seed A/B this grid was not sized to run.

**What should change in the fielded configuration.** On the pre-registered deciding
metrics, G8 favours the implicit continue/stop factorization: better held-out NLL by
0.124 nat and a TARP that passes where every explicit-head arm fails, with coordinate
PITs, coverage and the decode metric all tied. That is a recommendation this run supports;
it is not a recommendation the run *validated end to end*, since `v1_contstop` was fielded
as a two-seed comparison arm rather than as a candidate deliverable.

**What has not changed.** The acceptance criterion — the posterior beats the identity
baseline on both estimators and both tiers — holds on every arm in the grid.

## 8. What is not measured

- **PYTHIA vs HERWIG.** Still no `herwig_driver`; the train/test deltas remain the noise
  floor stand-in, exactly as in v0 §10. This remains the largest unquantified systematic.
- **`v1_wide` (`dec_dim = 128`).** The plan lists it as optional; it is not in the grid, so
  the v0 §8 capacity question — whether the residual under-coverage argues for more decoder
  capacity — is still open.
- **Split variance.** The seed band covers initialisation and batch order at a fixed
  train/val split (§3). Nothing here bounds what a different split would do.
- **Whether the spline or the joint flow actually closes G3.** Both triggers have fired;
  neither remedy has been tried. That the support fix halved the failure is evidence about
  the *cause*, not about either cure.
