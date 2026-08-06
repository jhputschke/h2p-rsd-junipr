# PLAN — the RQ-spline `ln z` head

Status: **implemented, trained and measured, twice.** §6 is the `ln z` spline's result and
§8 is the follow-up grid's.

- **`lnz_head="spline"` works and is worth fielding.** Over six seeds (§8.2) the `ln z` PIT
  falls from 1.05–2.07× its critical value to **0.47–1.04×**, five of six below the line;
  held-out NLL improves on every arm to a band (3.834–3.862) entirely below the control's
  (3.904–3.924); the support closure is kept exactly. Gate G3 is formally **PARTIAL**
  because one seed of six misses by 4%.
- **`dv_head="spline"` does NOT work** (§8.1). G3-dv is **0/3**: the spline leaves `dv`
  the same or worse, costs a little NLL, and regresses TARP on two of three seeds. It ships
  measured-and-not-recommended, like `mbr_n`.
- **The reason is the useful part.** The `dv` residual is a per-cell *location* bias that a
  truncated normal and a spline reproduce **identically, cell by cell**. It is a limit on
  what the head can predict from its conditioning, not on what its output family can
  express — so more density freedom cannot fix it, and neither would the joint-density
  escalation (§8.4).

- **Giving the head the cell's continuous centre changes nothing either** (§9.3).
  Pre-registered verdict **DEAD**: `dv` is 0/3 below its critical value and the per-cell
  bias RMS moves by less than 4%. Two mechanisms proposed for `dv`, two falsified by the
  experiments that tested them.
- **The `d(MBR)` reservation is WITHDRAWN** (§6.2, 2026-08-06). It was four *unpaired*
  means; paired per jet the 95% CI contains 0 on 4/4 pairs, and at 1000 jets the 4/4 signing
  itself is gone (pooled −0.0014 [−0.0089, +0.0062] over 3273 jets). `PLAN_z_aware.md` §11.
  So the spline carries **one** open item — G3 formally PARTIAL — not two.

**Result in one line:** the `ln z` spline paid; every attempt to fix `dv` **inside** the
per-coordinate factorization has failed — more density freedom and better conditioning both
left the per-cell bias where it was — and that is what promotes §7.3, the joint coordinate
density, from deferred to indicated (§9.4), reversing §8.4 on evidence that changed.

---

## 1. The trigger, already fired

`PROD_TEST_v1_RESULTS.md` §4.2 and §6 record it:

| arm | `ln z` PIT KS | × critical (0.0255) |
|---|---:|---:|
| v0 checkpoint / `v1_legacy_lnz_s0` | 0.0734 | 2.88× |
| `v1_base_s0` | 0.0529 | 2.07× |
| `v1_base_s1` | 0.0270 | 1.05× |
| `v1_base_s2` | 0.0471 | 1.84× |

**G3 fails on every seed**, and the region × coordinate cross localises the residual to
`ln_z × wide_soft` at **2.16×** its own critical value on **2 671 emissions** — the quadrant
holding **94%** of them — with every other scored cell below 1.02×. So this is a mismatch in
the bulk, not in a corner.

`lnz_support="physical"` (v1 WP-A) already closed the *support* half completely: 0.83%
below soft drop and 3.94% above `z = ½` both become 0.0000%. What is left is a **shape**
mismatch *inside* the interval, and — the sentence the escalation rests on — **a truncation
cannot fix a shape mismatch inside the interval.** The truncated normal has two free
numbers per node; the residual is the third and fourth moments of a distribution that is
not Gaussian on `(ln z_cut, ln ½]`.

The escalation was pre-authorized in `PLAN_prod_test_v1.md` §4.4 for exactly this outcome,
recorded as fired in `PROD_TEST_v1_RESULTS.md` §6, and deliberately **not** acted on
mid-run. This document is where it gets built.

---

## 2. The change, as built

A **monotone rational-quadratic spline** (Durkan, Bekasov, Murray & Papamakarios,
*Neural Spline Flows*, arXiv:1906.04032) on the same soft-drop interval the truncated
normal already uses, reached through the affine map `t = (x − lo)/(hi − lo)`:

    F(x) = S(t),   p(x) = S′(t)/(hi − lo),   x = lo + (hi − lo)·S⁻¹(u).

- **Config:** `model.lnz_head: "truncnorm" | "spline"`, default `"truncnorm"`. Same
  `getattr`-tolerant read as `lnz_support` / `cell_label_smoothing`, so a checkpoint config
  snapshot predating the field rebuilds as the truncated-normal model rather than crashing.
  `model.lnz_spline_bins: int = 8` alongside it, read only in `"spline"` mode.
- **Off path bit-identical.** The spline lives beside `_coord_logprob`'s
  `trunc_normal_logpdf` branch, `_sample_lnz`'s `trunc_normal_sample` branch and
  `coordinate_cdfs`'s `trunc_normal_cdf` branch — collected into four `_lnz_*` dispatch
  methods, one flag, and the `"truncnorm"` route untouched. **Verified**: same `state_dict`
  keys and values, head width still 8, and `log_prob` / PIT / `sample` /
  `sample_coordinates` / `describe_sequence` all agree to 0.0;
  `scripts/verify_parity.py` still reproduces the reference v2 script bit-for-bit.
- **The interval is unchanged.** `lnz_bounds(cx)` stays exactly as it is — cell-conditional,
  evaluated at the loosest `u` in the cell — so the factorization is not disturbed and the
  support guarantee v1 bought is preserved by construction. The head emits `3K − 1` numbers
  per node (widths, heights, internal derivatives) **instead of** `(mean, sigma)`, so the
  coordinate head is `6 + (3K−1)` = 29 wide at `K = 8` against `truncnorm`'s 8, with no
  dead outputs.
- **Why the spline first and not the joint density:** it is strictly cheaper, it changes
  one factor rather than the factorization, and it leaves every consumer of
  `has_continuous_coords` alone. If it closes G3 the structural escalation is not needed.

### 2a. The base is FIXED — a design decision the first run overturned

The obvious construction is to warp the *truncated normal's* CDF, `F(x) = S(F_TN(x))`, so
that `lnz_head="truncnorm"` is the spline's identity special case and training starts at
today's density. That was built first, and it **diverges**.

It is **non-identifiable**. Once `S` carries the shape, any `(μ, σ)` that leaves `F_TN`
roughly linear on the interval gives the same composed density, so the pair is free to
drift along a flat direction — and nothing in the objective pulls it back. Measured on
seed 2 of the first 3-seed run, at epoch 13:

| | diverged arm (s2) | healthy arm (s1) |
|---|---:|---:|
| `lnz_mean` median (interval is [−2.303, −0.693]) | **−533** | −1.22 |
| `lnz_sig` median | **85** | 0.28 |
| emissions with `F_TN(x)` saturated at 0 or 1 | **100%** | 0% |
| val NLL | **20.85** | 3.97 |

Val NLL went 4.19 → 19.2 at **epoch 4** and never recovered; seeds 0 and 1 were on the
same flat direction and had merely not walked as far, so this was a latent failure of all
three arms rather than one bad seed.

The fix removes the redundancy at the root instead of bounding its symptom: the base is the
interval's **affine** map, which has no parameters at all, and the spline is the whole
density. `truncnorm` remains available as its own path — that is what the parity flag is
for — it is simply no longer *nested* inside the spline. Identity initialization then means
the **uniform** density on the interval, which is the maximum-entropy starting point and a
stable one. `tests/test_lnz_spline.py::test_the_spline_replaces_the_base_rather_than_warping_a_learnable_one`
pins the contract on the head's own output so the redundancy cannot reappear.

## 3. The measurement, pre-registered

- 3-seed training at the **v1 budget**, `v1_base` config with `model.lnz_head=spline`
  as the only difference — same data, same epochs, same encoder, seeds 0/1/2.
- **G3 re-test** on the recorded reference: the per-seed KS against its own 0.0255 critical
  value, and the `ln_z × wide_soft` cell against its 2.16×. The gate closes iff KS < crit on
  **every** seed *and* the bulk cell falls below 1.0×. A two-of-three pass is a partial
  result and is reported as one.
- **Guards that must not move:** the support audit stays at 0.0000% violations (a spline
  that leaks outside `(ln z_cut, ln ½]` is a regression, not a trade); held-out NLL must not
  worsen beyond seed spread; TARP and `pit_ks_max` reported beside G3 so a coordinate fix
  that narrows the joint is visible.
- Report into `PROD_TEST_v1_RESULTS.md` §4.2's table as a fourth block of rows, so the
  before/after is read on one page.

## 4. If it does not close G3

The **per-node joint coordinate density** (cINN-coords / CFM-coords, `PLAN_UPDATES.md` WP1)
is the follow-up, and it is structurally motivated rather than a second guess: under the
`LundGenerator` conventions

    ln z = u + v − ln p_T,sum

holds **exactly**, so independence-given-cell is violated by a *kinematic identity*, and no
per-coordinate head — spline or otherwise — can express that coupling. The spline is
therefore the test of "is the marginal shape wrong?", and its failure is the evidence that
the *factorization* is what is wrong. That trigger has also already fired
(`PROD_TEST_v1_RESULTS.md` §6.1); this ordering just spends the cheap one first.

## 5. Verification

- `pytest tests/` green (**857**, was 842 — the 15 new spline tests); ruff clean on touched
  files; `scripts/verify_parity.py` still reproduces the reference v2 script bit-for-bit.
- A new test asserting the `"truncnorm"` path is bit-identical with the flag absent and with
  it set explicitly (the `lnz_support` legacy-parity test is the template). **Verified**:
  same `state_dict` keys and values, head width still 8, and `log_prob` / PIT / `sample` /
  `sample_coordinates` / `describe_sequence` all 0.0 apart.
- A spline-mode round-trip test: `cdf(sample(...))` uniform on a synthetic head, and the
  log-density integrating to 1 over `(lo, hi]` by quadrature.

---

# §6. RESULT — G3 is PARTIAL: the shape error is mostly gone, and it moved

Run 2026-08-05. Training `bash scripts/run_lnz_spline.sh` (4 arms, ~60 min at concurrency
4); evaluation `bash scripts/eval_prod_test_v1.sh --run-root runs/lnz_spline --device cpu`
— the SAME two-tier command the v1 campaign used, so the PIT numbers are produced by the
same code path as the numbers they are compared against. Scored by
`python scripts/lnz_spline_gates.py`; artifact `runs/lnz_spline/lnz_spline_gates.json`.

Controls are the v1 arms themselves — `v1_base_s{0,1,2}`, same preset, same seeds, same
data, truncated-normal head — re-read from their own `eval_metrics.json`. They reproduce
their documented 2.07× / 1.05× / 1.84× and the 2.16× bulk cell exactly, so the comparison
is against the right record.

## 6.1 The gate

| arm | `ln z` KS | ×crit (0.0255) | was | KS p | `ln_z × wide_soft` | was |
|---|---:|---:|---:|---:|---:|---:|
| `spline_s0` | 0.0120 | **0.47×** | 2.07× | 0.805 | **0.48×** | 2.16× |
| `spline_s1` | 0.0163 | **0.64×** | 1.05× | 0.440 | **0.75×** | 1.17× |
| `spline_s2` | 0.0266 | **1.04×** | 1.84× | 0.035 | **1.04×** | 1.91× |
| `contstop_spline_s0` *(transfer, not in the gate)* | 0.0268 | 1.05× | 1.89× | 0.033 | 1.08× | 1.95× |

**G3: PARTIAL** — both pre-registered clauses hold on **2 of 3** seeds. Seed 2 misses by
4%, at p = 0.035 against a 0.05 threshold; it is a marginal failure and is reported as one.
The plan wrote that rule down in advance for exactly this outcome: a 2–4× improvement on
every seed is the reading that invites "close enough", and the gate says all three.

The improvement itself is unambiguous and larger than the change v1's WP-A produced: every
seed improves by a factor 1.6–4.4 on the marginal and 1.6–4.5 on the bulk quadrant, and the
transfer arm shows the fix carries to the **fielded** continue/stop family at the same size.

## 6.2 The guards — nothing was bought with something else

| arm | soft-drop | z > ½ | `pit_ks_max` | val NLL | Δ NLL | TARP | passes G7 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `spline_s0` | 0.00000% | 0.00000% | 0.0282 | 3.846 | **−0.078** | 0.0215 | **yes** |
| *v1_base_s0* | 0.00000% | 0.00000% | 0.0529 | 3.924 | — | 0.0415 | no |
| `spline_s1` | 0.00000% | 0.00000% | 0.0264 | 3.861 | **−0.043** | 0.0265 | **yes** |
| *v1_base_s1* | 0.00000% | 0.00000% | 0.0324 | 3.904 | — | 0.0350 | no |
| `spline_s2` | 0.00000% | 0.00000% | 0.0287 | 3.860 | **−0.064** | 0.0400 | no |
| *v1_base_s2* | 0.00000% | 0.00000% | 0.0471 | 3.924 | — | 0.0335 | no |
| `contstop_spline_s0` | 0.00000% | 0.00000% | 0.0268 | 3.739 | **−0.041** | 0.0255 | yes |
| *v1_contstop_s0* | 0.00000% | 0.00000% | 0.0482 | 3.780 | — | 0.0200 | yes |

- **Support: held.** 0.00000% below soft drop and above `z = ½` on every arm, as the
  construction guarantees. The property v1's WP-A bought was not spent.
- **NLL: improved on every arm**, by 0.041–0.078 nat against a control seed spread of
  **0.020** (3.904–3.924). This is the first change in this line of work to move the
  likelihood *and* the calibration in the same direction — the aux expansion, the encoder
  swaps and every decode rule either moved one or neither.
- **`pit_ks_max`: improved on every arm** (0.0529 → 0.0282, 0.0324 → 0.0264, 0.0471 →
  0.0287, 0.0482 → 0.0268), so the worst coordinate got better even though it is no longer
  `ln z` (§6.3).
- ~~**d(MBR) to truth: marginally WORSE on all four arms**~~ — **WITHDRAWN, 2026-08-06.**
  The row read +0.0047, +0.0113, +0.0091, +0.0031 against a control seed spread of 0.018,
  and was stated rather than dismissed because it was consistently signed on 4 of 4. Those
  are **unpaired means**, and `PLAN_z_aware.md` WP-0 has since run the paired analysis the
  exact pairing always permitted (`dlund_identity` is identical within every pair):

  > **the per-jet paired 95% CI contains 0 on 4/4 pairs**, each ~±0.03 wide against an
  > effect of +0.005; and at the pre-declared 1000-jet escalation the 4/4 sign consistency
  > *itself* does not survive — 3/4, with `spline_s0` significantly **better** (−0.0211
  > [−0.0357, −0.0065]) and the pool over 3273 paired jets at **−0.0014 [−0.0089, +0.0062]**.

  So the correct statement is **`d(MBR)` is unchanged within its own per-jet noise**, and
  the caveat this bullet carried is discharged. The mechanism named below is still literally
  true — the MBR metric runs on `mbr_coords="lnDR_lnkt"` and does not read `ln z`, so a
  better `ln z` density can only perturb which trees are drawn — but it was explaining a
  number that is not resolved. `PLAN_z_aware.md` §11.1 has the tables;
  §11.2 adds that the *other* half of the sentence also fails: under a ruler that does see
  `ln z`, the MBR winner's own `|Δ ln z|` is flat (+0.0003 [−0.0109, +0.0117]), so there was
  no hidden improvement for a blind ruler to miss either.

## 6.3 The unplanned finding: the residual moved to `dv`

With `ln z` fixed, the binding coordinate is no longer `ln z`:

| coordinate | s0 | s1 | s2 |
|---|---:|---:|---:|
| `du` | 0.77× | 0.68× | 0.61× |
| **`dv`** | **1.10×** | **1.04×** | **1.12×** |
| `ln z` | 0.47× | 0.64× | **1.04×** |
| `psi` | 0.44× | 0.83× | 0.61× |

`dv` — the within-cell `ln kt` offset, also a truncated normal — now fails on **all three**
seeds, and `dv × wide_soft` sits at 1.01× / 0.94× / 1.03× in the same bulk quadrant. It was
always there; `ln z`'s 2.07× was simply larger and `pit_ks_max` reported that instead. The
diagnosis that motivated this work package — "a truncation cannot fix a shape mismatch
inside the interval" — applies verbatim to `du`/`dv`, which are truncated normals on
`[−half_u, half_u]` and `[−half_v, half_v]`. **§7.1 is the direct consequence.**

## 6.4 A second unplanned finding: TARP moved, mostly the right way

`v1_base` is the explicit-`q(N|x)` family, which failed G7 on **all six** of v1's arms — the
evidence behind v1's central attribution that the joint narrowness is the multiplicity
factorization. With the spline, seeds 0 and 1 **cross below the null band** (0.0415 → 0.0215
and 0.0350 → 0.0265 against a p95 of 0.0275) and now pass G7; seed 2 moves the wrong way
(0.0335 → 0.0400) and still fails.

Read carefully, and **not** as an overturning of v1's attribution: that rested on six
explicit arms failing while both continue/stop arms passed, and the continue/stop family
still passes here either way. What this does establish is that the **coordinate density was
contributing to the joint narrowness too** — the factorization was not the only cause. Two
of three seeds with one contrary seed is suggestive, not a verdict, and it is recorded as
such. A 3-seed continue/stop arm would be the way to settle it.

## 6.5 What this closes, and what it does not

**Closes.** The `ln z` *support-plus-shape* story, as far as a per-coordinate head can take
it. `lnz_head="spline"` is worth fielding on its own numbers — better likelihood, better
marginal PIT, better `pit_ks_max`, unchanged support — independently of whether the gate's
third seed cooperates.

**And, as of 2026-08-06, the `d(MBR)` caveat too** (§6.2, withdrawn). `PLAN_z_aware.md`
WP-0's paired analysis leaves the difference at −0.0014 [−0.0089, +0.0062] over 3273 paired
jets, so the second of the two reservations attached to fielding the spline is discharged
against a measurement rather than argued away. The first (G3 formally PARTIAL, 5/6 seeds by
§8.2) stands.

**Does not close.** G3, formally: seed 2 sits at 1.04×. And the marginal PIT was never the
whole question — the kinematic identity `ln z = u + v − ln p_T,sum` still makes
independence-given-cell false, and §6.4's seed 2 says the joint is not fixed.

**Does not fire the structural escalation yet, and that is a change of plan.** §4 said the
joint coordinate density is the follow-up "if the spline does not close G3". It did not
close it — but the reason is now measurable and it is *not* the factorization: the largest
remaining per-coordinate defect is `dv` at 1.04–1.12×, which is the same fixable shape
mismatch on a different coordinate. Spending the cheap fix there first is the same logic
that put the spline before the joint density in the first place, and §7 records it.

---

# §7. What to do next, in the order the measurement supports

## 7.1 Spline the `dv` head — the same fix, on what is now the binding coordinate

**Trigger: fired** (§6.3). `dv` fails on all three seeds at 1.04–1.12× and `dv × wide_soft`
sits at ~1.0× in the bulk quadrant, while `ln z` is now 0.47–1.04×. The argument is
identical to the one that authorized this work package: `dv` is a two-parameter truncated
normal being asked to carry a shape it cannot.

**`dv` only — `du` is NOT justified by the evidence**, and the distinction matters because
"spline the offsets" is the lazy generalization. Across all six measurements (three control
arms and three spline arms):

| coordinate | ratio to its own critical value | fails |
|---|---|---:|
| `dv` | **1.13 ± 0.08** | 6 / 6 |
| `psi` | 0.73 ± 0.26 | 1 / 6 |
| `du` | **0.73 ± 0.09** | 0 / 6 |

`du` and `dv` are ~4σ apart, both stable across seeds *and* across the `ln z` change. And
`dv` was **already failing before the spline** — on seed 1 it was already the worst
coordinate at 1.27×, i.e. it *was* that arm's `pit_ks_max`. So `ln z` and `dv` are two
independent pre-existing defects, not a chain in which each fix exposes the next; fixing
`dv` should leave `du` where it is. Splining `du` as well would be spending head width on a
coordinate with no measured defect — do it only if a later run puts it near its critical
value.

Two caveats on that recommendation. `du`'s run-to-run scatter is ±0.09, so a retrain could
land it in ~0.6–0.9; reaching 1.0 is ~3σ, unlikely rather than impossible. And a *passing*
KS is not proof of correctness — at n = 2834 the test has limited power, so `du` at 0.73×
may still hide a smaller defect. If a coordinate other than `dv` becomes the binding one it
is most likely **`psi`**, which posted a genuine 1.28× failure on control seed 2 and whose
scatter is 3× the others' (that reads as noise-dominated rather than systematically wrong,
and `psi` is a von Mises on a periodic domain, so it needs a circular treatment, not this one).

**Why `dv` and not `du`, mechanistically: not established.** Three natural explanations were
tested against the truth sample and two are ruled out — the marginal log-density gradient is
the same on both axes (0.174 vs 0.181 per 0.2-wide cell) and the within-cell offset shapes
are the same and near-uniform on both (|skew| 0.087 vs 0.066, excess kurtosis −1.14 vs
−1.19). The one structural asymmetry found is that **13.3% of emissions sit in the `ln kt`
cell touching the `kt_floor` cut, against 0.0% for the angular axis** — the `kt` axis has a
hard support boundary inside its populated region and the angular axis does not. That is a
hypothesis, not a finding: testing it means stratifying `dv`'s PIT by `ln kt` cell, which is
a cheap diagnostic and should be run *before* 7.1, because if the defect is an edge effect
then a spline is the wrong fix and the right one is the same `lnz_support` move applied to
`ln kt` — a support correction, not a shape one.

Cheap, because the machinery exists and is generic: `rq_interval_{logpdf,cdf,icdf,sample}`
already take an arbitrary `(lo, hi)`, and for `du`/`dv` those bounds are the *constant*
`±half_u` / `±half_v` rather than cell-conditional — strictly simpler than the case already
built. The work is a `model.offset_head: "truncnorm" | "spline"` flag with the same
head-width arithmetic (`+2 × (3K−1) − 4` outputs), the same four dispatch points, and the
same parity discipline.

Pre-register the same gate before running it: KS below critical on **every** seed for both
`du` and `dv`, the bulk quadrant below 1.0×, support and NLL guards unchanged — and expect
the residual to move again, since that is what happened here.

## 7.2 Settle seed 2 rather than arguing about it

Two cheap options, and they answer different questions:
- **More seeds.** 3 seeds cannot distinguish "one marginal seed" from "a 1-in-3 failure
  rate". Seeds 3–5 at the same budget (~1 hour) would; the gate criterion stays as written.
- **More spline capacity.** `lnz_spline_bins` is a config field and was never tuned — `K = 8`
  was chosen, not fitted. `K = 16` on seed 2 alone is a 15-minute test of whether the
  marginal miss is expressiveness or variance. Do **not** tune `K` against G3 across all
  seeds afterwards; that would make the gate circular, exactly as `LOSS_QUANTILE` would have.

## 7.3 The joint coordinate density — still the structural escalation, now better targeted

Unchanged in motivation (`ln z = u + v − ln p_T,sum` holds exactly, so no per-coordinate
head can express the coupling) and **no longer the immediate next step**, because §6.3 says
the largest measurable defect is still a per-coordinate one. The right sequencing is 7.1
first: if splining `du`/`dv` closes the marginal PITs everywhere and TARP *still* fails,
that is a clean, isolated statement that the failure is the factorization — which is what
the joint density is for, and a much sharper trigger than the one available today.

## 7.4 Field the flag, or wait?

`lnz_head="spline"` improves NLL, the marginal PIT, and `pit_ks_max` on every seed, keeps
the support closure exactly, and costs 1 365 parameters. The one thing it does not improve
is `d(MBR)`, which is marginally worse on 4 of 4 arms (§6.2) — and which cannot see `ln z`
at all. A defensible reading is to field it for the **density** products (the decode-free
posterior series, calibration, anything reading `log_prob`) and to keep the recommended
point estimate under review until 7.1 lands, since that is the number the point estimate
actually moves on. Default stays `truncnorm` either way; this is a config decision, not a
code one.

---

# §8. RESULT — §7.1 fails, §7.2 settles, and §7.3's trigger has to be restated

Run 2026-08-05, second grid: `dvspline_s{0,1,2}` (§7.1), `spline_s{3,4,5}` (§7.2a) and
`spline_k16_s2` (§7.2b), same preset and budget, evaluated with the same two-tier command.
Artifacts: `runs/lnz_spline/lnz_spline_gates.json`, `offset_head_diagnostic.json`.

## 8.1 §7.1 — the `dv` spline does NOT work, and it falsifies the prediction that led to it

Each arm against its own control, the `ln z`-spline arm of the **same seed**, so the row
prices the `dv` change alone:

| seed | `dv` truncnorm | `dv` **spline** | `dv × wide_soft` | was | Δ NLL | TARP | was |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.10× | **1.22×** | **1.23×** | 1.01× | +0.016 | 0.0430 | 0.0215 |
| 1 | 1.04× | **1.12×** | **1.14×** | 0.94× | +0.000 | 0.0560 | 0.0265 |
| 2 | 1.12× | **1.02×** | **1.05×** | 1.03× | +0.028 | 0.0315 | 0.0400 |

**G3-dv: 0/3 on both clauses.** The spline did not fix `dv` — it is *worse* on two seeds
and unchanged on the third, the bulk cell is worse on all three, NLL is flat-to-worse, and
TARP/G7 regress on two of three (seeds 0 and 1 go from passing G7 to failing it). There is
no reading of this table on which the `dv` spline should be fielded.

**The prediction it was built on is falsified, and the falsification is the useful part.**
§7.1 argued from a *tilt budget*: a truncated normal on `[-h, h]` tilts its log-density
across the cell by `2h·μ/σ²` with `μ` clamped to `±h`, so `dv` (running at σ = 2.6h) could
achieve 0.158 against a measured requirement of ~0.173 while `du` (at 1.7h) achieved 0.258.
The arithmetic is right and the conclusion drawn from it was wrong.

The direct measurement says why, and it was already on the page. `offset_head_diagnostic.py`
measurement 1 showed **no individual `ln kt` cell failing** — the aggregate exceeds its
critical value only because small per-cell mean-PIT biases fail to cancel. Re-running that
diagnostic on the spline arm settles it:

| `ln kt` cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean PIT, **truncnorm** | .523 | .489 | .512 | .484 | .517 | .476 | .525 | .499 | .534 | .573 | .531 |
| mean PIT, **spline** | .525 | .492 | .513 | .488 | .520 | .480 | .529 | .513 | .541 | .581 | .545 |

**Identical, cell by cell.** Strictly more within-cell freedom moved the bias not at all —
if anything slightly further from 0.5 everywhere. So the defect is not a within-cell shape
the family cannot express: it is that the head predicts a slightly wrong within-cell
*location*, and it does so identically under two different density families. The limit is
what the head can **know** from `(decoder state, e(x), cell embedding)`, not what it can
express. A more flexible output that must be predicted from the same conditioning adds
variance without addressing the bias — which is exactly the measured table above.

It is not a data artifact either: the truth's marginal spectrum falls smoothly and
monotonically on both axes (11 consecutive negative per-cell log-slopes on `ln kt`, 10 on
`ln 1/ΔR`, **zero sign changes**), so there is no periodic structure at the cell scale for
the model to be chasing.

**Methodological note, recorded because it cost a training grid.** Measurement 3 was an
elegant mechanism and measurement 1 was a direct observation, and they disagreed. I weighted
the mechanism. The rule that would have caught it: *a mechanism that contradicts a direct
measurement of the same object is a hypothesis about something else.* The per-cell PIT was
already saying the within-cell shapes were fine.

## 8.2 §7.2a — six seeds: G3 is 5/6, and seed 2 is the lone failure

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `ln z` ×crit | 0.47× | 0.64× | **1.04×** | 0.72× | 0.90× | 0.64× |

Mean 0.74×, one failure, and it is the same seed 2 that missed in the first grid. So the
question §7.2a was posed to answer — "one marginal seed, or a 1-in-3 failure rate?" — comes
back **1 in 6**, not 1 in 3. G3 stays formally **PARTIAL** because the gate says *every*
seed, and that verdict is unchanged. But the thing a reader should take from it moved: the
`ln z` spline puts five of six seeds comfortably under the line (0.47–0.90×) against a
control that failed on all three of the seeds it was run on.

Held-out NLL over the six: **3.834–3.862**, a band that is both tighter than and entirely
below the `v1_base` control band of 3.904–3.924.

## 8.3 §7.2b — K=16 is inconclusive, and confounded by construction

| | K=8 (seed 2) | K=16 (seed 2) |
|---|---:|---:|
| `ln z` ×crit | 1.042× | **0.797×** |
| `ln_z × wide_soft` | 1.040× | 0.864× |
| `psi` ×crit | 0.613× | **1.489×** |
| `pit_ks_max` | 0.0287 | **0.0380** |
| val NLL | 3.860 | 3.862 |

Read literally this "fixes" seed 2. It should not be read literally, for two reasons, and
neither is a judgement call:

1. **It is confounded with a re-draw.** Changing `K` changes the head's output width, hence
   its parameter count, hence its initialization — so `spline_k16_s2` is not seed 2 with
   more spline capacity, it is *a different draw* that also has more spline capacity. The
   arm cannot separate the two.
2. **It broke `psi`**, from 0.613× to 1.489× — a coordinate the spline does not touch — and
   `pit_ks_max` got worse overall. A change that improves the targeted coordinate while
   introducing a new failure elsewhere on an untouched coordinate is the signature of
   run-to-run variance, not of capacity.

Combined with §8.2's 5/6, the parsimonious reading is that **seed 2 is a variance draw and
`K = 8` is adequate**. `K` is therefore left at 8 and was not selected against G3 — doing so
would have made the gate circular, which is why the arm was pre-registered as reported-only.

## 8.4 Is §7.3 — the joint coordinate density — the right next step? **No, and not yet necessary.**

It is the pre-authorized structural escalation and its motivation is real, so the answer
needs the reasons rather than a verdict.

**What would make it right.** §4 wrote the trigger as *"the spline closes the marginal PIT
but TARP still fails"* — i.e. every per-coordinate density is demonstrably correct and the
*joint* is still too narrow. That is a clean isolation: it leaves the factorization as the
only thing standing. The kinematic identity `ln z = u + v − ln p_T,sum` then names the
mechanism exactly, since it makes independence-given-cell false by construction.

**Why that is not the situation.** The marginal PIT does **not** close: `dv` fails on every
arm measured, 10 of 10, under *both* density families (1.02–1.22×). So a TARP failure today
cannot be attributed to the factorization — it is confounded with a per-coordinate marginal
that is still wrong. Running the expensive structural change now would produce a result
nobody can read.

**And the measured defect is not the one a joint density fixes.** §8.1 established that
`dv`'s residual is a per-cell *location* bias, reproduced identically by a truncated normal
and by a spline with strictly more freedom. That is a limit on what the head can **predict
from its conditioning**, not on what its output family can **express**. A joint density
changes the family — it lets the coordinates depend on one another given the cell — while
reading the *same* `(decoder state, e(x), cell embedding)`. It is not targeted at a
conditioning limit, and on the evidence of §8.1 it would inherit the same bias.

**A caveat against my own reading, stated because it is the strongest counter-argument.** A
marginal PIT is only uniform if the model's *marginal-given-cell* is right, and the true
marginal-given-cell is obtained by integrating the true joint — so a factorized model can be
forced into a wrong marginal by a correlation it cannot represent. On that reading `dv`'s
failure *is* evidence for §7.3. What makes me not take it: the bias is stable across seeds
and across two density families to the third decimal, which looks like a deterministic
shortfall in what the head is told rather than an averaging artifact. That is a hypothesis,
and §8.5(1) is the experiment that separates the two.

**So: not next, and not necessary yet.** It stays the escalation of record, with its trigger
**restated** so it cannot fire on a confounded run:

> §7.3 fires when every per-coordinate marginal PIT is below its critical value on every
> seed **and** TARP still exceeds its own MC null band. Until the marginals close, a TARP
> failure is not attributable and the joint density is not the indicated fix.

## 8.5 What to do instead, in order

1. **Test the conditioning hypothesis — cheap, decisive, and it targets the measured
   defect.** §8.1 says the head cannot predict the correct within-cell location from what it
   is given. The single most likely reason is that the cell enters only as a *learned
   embedding* of a categorical id, so nothing tells the head that cell 437 and cell 438 are
   neighbours; every cell's within-cell tilt must be learned independently, from its own
   emissions. Feed the cell's **continuous centre `(c_x, c_y)`** into the coordinate head
   alongside the embedding — a two-column change to one `torch.cat`, one flag, one training
   grid — and re-read the per-cell mean-PIT table of §8.1. If the bias shrinks, the
   diagnosis is confirmed and the fix is nearly free. If it does not, the conditioning
   hypothesis is dead and §7.3's counter-argument above becomes the live one, which is
   exactly the fork worth buying.
2. **Then re-run the `dv` question**, whichever way (1) goes. Note that `dv_head="spline"`
   should be considered **measured and not recommended** in the meantime — the same status
   `mbr_n` carries — rather than removed: it is bit-identical off, and a later change to the
   conditioning may make the extra flexibility pay where today it only adds variance.
3. **Field `lnz_head="spline"` on its own numbers.** Six seeds, 5/6 below critical, NLL
   better than the control band on every one, support unchanged. It does not depend on any
   of the above — **and since 2026-08-06 it does not carry the `d(MBR)` caveat either**
   (§6.2, withdrawn on `PLAN_z_aware.md` §11's paired measurement).
4. ~~**Only then §7.3**, under the restated trigger.~~ — **superseded by §9.4**: the
   restated trigger is withdrawn and §7.3 is indicated. §9.5/§9.5a carry its relationship
   to `PLAN_z_aware.md`, the order the two should run in, and the fact that the two
   `PLAN_z_aware.md` steps ahead of §7.3 are now done.

---

# §9. §8.5(1) — the conditioning experiment, and its PRE-REGISTERED reading

Written before the arms finished training, for the reason §8.1 exists: the last hypothesis
in this document was elegant, wrong, and the temptation after the fact was to reinterpret it
rather than to record that it failed. What follows is fixed in advance.

## 9.1 The change

`model.coord_cell_center=true` appends the cell's continuous centre, affinely mapped onto
`[-1, 1]` by the geometry's own ranges, to the coordinate head's input. Today that input is
`[decoder state | e(x) | cell embedding]`, and the embedding is a free vector per
**categorical** id: the head is told *which* cell it is in and nothing about *where* that
cell is. Neighbouring cells share no parameters, so every cell's within-cell tilt has to be
learned from its own emissions alone. That is the shape of a defect that more *output*
flexibility cannot touch — which is what §8.1 measured.

Arms: `cellctr_s{0,1,2}` against `spline_s{0,1,2}` (same seed, `ln z` spline both sides, so
the row prices the conditioning alone), plus `cellctr_dvspline_s0` against `cellctr_s0` for
§8.5(2) — does the `dv` spline pay once the conditioning is fixed?

## 9.2 What each outcome means — decided in advance

The hypothesis makes a specific prediction: the per-cell mean-PIT deviations shrink.

- **CONFIRMED** — `dv`'s marginal PIT falls below 1.0× on **all three** seeds. The
  diagnosis is right, the fix is nearly free (two inputs, +130 parameters), and it should
  be fielded alongside `lnz_head="spline"`.
- **PARTIAL** — the per-cell bias measurably shrinks (RMS of `mean PIT − 0.5` across cells
  down by more than a third) but `dv` does not clear 1.0× on every seed. The mechanism is
  real and incomplete; the follow-up is more conditioning (the cell's *width*, or the
  neighbouring cells' occupancies), not a different density.
- **DEAD** — the per-cell bias is unchanged (RMS within ±20% of the control) and `dv` stays
  at ≥ 1.0×. Then the conditioning hypothesis is **falsified**, and the counter-argument of
  §8.4 becomes the live one: a factorized model *can* be forced into a wrong marginal by a
  correlation it cannot represent, and the joint coordinate density (§7.3) is the indicated
  next step rather than a deferred one.

The RMS of `mean PIT − 0.5` over populated `ln kt` cells is the statistic, because that is
the quantity §8.1 showed to be identical under two density families — it is the thing this
change is supposed to move, and the one that stayed put when the density was made more
flexible.

**Guards unchanged**: support at 0.0000%, NLL not worse beyond the control's seed spread,
and TARP / `pit_ks_max` reported beside the verdict. A conditioning fix that buys `dv` by
spending `ln z` or the joint is not a fix.

## 9.3 RESULT — **DEAD**. The conditioning hypothesis is falsified.

Run 2026-08-05. Arms `cellctr_s{0,1,2}` against `spline_s{0,1,2}` (same seed, `ln z` spline
both sides), plus `cellctr_dvspline_s0`. Artifacts: `runs/lnz_spline/lnz_spline_gates.json`,
`offset_head_diagnostic_cellctr.json`.

**Primary criterion — `dv` below 1.0× on all three seeds: 0/3.**

| seed | control | + cell centre | `ln z` | was | `du` | `psi` | Δ NLL | Δ d(MBR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.10× | **1.08×** | 0.82× | 0.47× | 0.73× | 0.75× | −0.001 | −0.019 |
| 1 | 1.04× | **1.12×** | 0.92× | 0.64× | 0.81× | 0.42× | +0.002 | +0.002 |
| 2 | 1.12× | **1.13×** | 1.05× | 1.04× | 0.76× | 0.90× | +0.003 | +0.009 |
| `cellctr_dvspline_s0` | 1.08×¹ | 1.03× | 0.94× | 0.82×¹ | 0.72× | 0.54× | +0.018 | +0.009 |

¹ control is `cellctr_s0`, so this row prices the `dv` spline **on top of** the conditioning.

**Secondary criterion — the per-cell bias RMS, the statistic §9.2 named in advance because
§8.1 showed it is identical under two density families: within ±4%, far inside the ±20%
DEAD band.**

| seed | control | + cell centre | change |
|---|---:|---:|---:|
| 0 | 0.0300 | 0.0295 | −1.7% |
| 1 | 0.0302 | 0.0303 | +0.3% |
| 2 | 0.0303 | 0.0314 | +3.6% |

Held-out NLL is unmoved (3.845 / 3.863 / 3.863 against 3.846 / 3.861 / 3.860) and the
support guard holds at 0.0000% on every arm. **Telling the head where the cells are changed
nothing.** The per-cell location bias is not caused by the cell arriving as a bare
categorical id.

**Two mechanisms proposed for `dv`, two falsified by the experiments that tested them.** The
tilt budget (§8.1) and now the conditioning. The honest summary is that the *existence* of
the `dv` defect is solid — it fails on every arm of every configuration measured, 13 of 13 —
and its *cause* is not established by anything in this document.

**An instrument caveat, recorded because a reader comparing the two tables will see it.**
`offset_head_diagnostic.py` and the `eval` pipeline disagree on the absolute `dv` ratio by
0.03–0.13 (e.g. `cellctr_s0` reads 0.95× in the diagnostic and 1.08× in the gate) because
they select marginally different 2000-jet populations. The verdict does not turn on it: the
*paired* control-vs-arm comparison has the same sign and magnitude under both instruments,
and the bias RMS above is computed on identical populations within one instrument. The
**gate** numbers are the quotable ones, being from the same pipeline as every other number
in this work package.

## 9.4 What this promotes — §7.3 is now the indicated next step

§9.2 fixed this consequence in advance, and it is the opposite of §8.4's conclusion two
experiments ago. That is what the evidence did, so it is what the document says.

The argument §8.4 put on the record *against* its own reading is now the surviving one:

> A marginal PIT is only uniform if the model's *marginal-given-cell* is right, and the true
> marginal-given-cell is obtained by integrating the true joint — so a factorized model can
> be forced into a wrong marginal by a correlation it cannot represent.

Two independent attempts to fix `dv` **inside** the factorization have now failed: more
density freedom (§8.1) and better conditioning (§9.3). Neither moved the per-cell bias by
more than a few percent. What remains is the possibility the factorization itself cannot
represent the right marginal — and the kinematic identity `ln z = u + v − ln p_T,sum` says
independence-given-cell is false by construction, so this is a mechanism with a derivation
rather than another hypothesis of mine.

**The restated trigger of §8.4 is therefore not met and should not be waited for.** It asked
for the marginals to close before firing §7.3 — but the two cheap ways of closing `dv` are
now spent, and the reason a marginal *cannot* close is precisely what §7.3 addresses.
Waiting for a condition that the diagnosis says is unreachable inside the current
factorization is not caution, it is a deadlock. §7.3 fires.

**What it must be run with, so the result is readable.** The confound §8.4 worried about is
real and has to be handled by design rather than by hope:
- 3 seeds at the v1 budget, `lnz_head="spline"` on both sides so the row prices the
  factorization alone;
- **pre-registered before the run**: does `dv`'s marginal PIT fall below 1.0× on every seed?
  That is the question two per-coordinate fixes failed, and it is the one a joint density is
  claimed to answer. TARP is the *secondary* read, not the primary — a joint density that
  fixes TARP while leaving `dv` at 1.1× has not explained the defect;
- the same guards: support at 0.0000%, NLL within the control's seed spread, and `d(MBR)`
  reported — **with its band**, which `PLAN_z_aware.md` §11 has now measured: paired per jet
  this ruler resolves to about ±0.01 over 3273 jets, so a `d(MBR)` difference smaller than
  that is not a finding. Its `dlund3_*_cont` / `dlnz_*` companions come out of the same
  `experiment.closure_continuous=true` pass for free and should be quoted beside it.

**One caveat against firing, stated so the decision is made with it in view.** `dv`'s failure
is small in absolute terms — KS 0.026–0.029 against a critical 0.0255, i.e. 2–13% over — and
it is the *only* coordinate still failing, with `du`, `ln z` and `psi` now all comfortably
under. A joint coordinate density is the most expensive change in this line of work.

**And that caveat is now stronger, not weaker** (2026-08-06). It was written conditionally —
"if the `ln z`-aware decode metric of `PLAN_z_aware.md` shows the `d(MBR)` regression is an
artifact" — and the conditional resolved in the direction that strengthens it, by a route
neither branch anticipated: there is no regression at all (its §11.1; the paired CI contains
0 on 4/4 and the 4/4 signing does not survive 1000 jets). So the fielded product **is** in
good shape on every axis except a 10% miss on one coordinate's marginal, and "run the most
expensive structural change in this line of work for that" is a judgement call about how much
the joint posterior's correctness is worth, not a conclusion the data forces.

## 9.5 How §7.3 relates to `PLAN_z_aware.md`, and which runs first

Recorded here so the ownership split and the ordering do not have to be re-derived. The two
documents look adjacent — both have `ln z` in the title, both were prompted by findings in
§6 — and they change **different layers**, by design rather than by accident.

`PLAN_z_aware.md` §2 separates three layers of `ln z`-blindness and claims two:

| layer | what is blind | owner |
|---|---|---|
| **(a) selection** | the EMD ground metric spans `(u, v)`; `mbr_coords="+lnz"` appends a **constant-zero** column and changes no distance, and `mbr_weight="z"` is silently identical to `unit` | `PLAN_z_aware.md` |
| **(b) scoring** | `dlund_mbr` is Euclidean between *cell centres*; the continuous block slices `[..., :2]` and carries no MBR row at all | `PLAN_z_aware.md` |
| **(c) structural** | the cell grid discretizes `(u, v)` only, and this family keeps `ln z` conditionally independent of `du` by construction, so `ln z = u + v − ln p_T,sum` is documented in prose and encoded nowhere | **§7.3 (this document)** |

Its §9 states the boundary from the other side: the structural layer is out of scope there.
So one fixes the **ruler** and the other fixes the **model**, and neither substitutes for
the other — a decode metric cannot move a PIT (that is computed from the model's own CDF),
and a joint density cannot make a blind metric see.

**The ordering, and it is not symmetric.**

1. **`PLAN_z_aware.md` WP-0 first.** It asks *is there a regression to explain* before any
   plumbing, costs one decode pass per arm, and may dissolve the question outright — in
   which case §6.2's `d(MBR)` caveat evaporates and `lnz_head="spline"` is unambiguously
   fieldable.
2. **Then its WP-2**, the silent-zero guard, **regardless of WP-0's verdict.** A knob that
   looks switchable and is not is a trap for whoever next reaches for the obvious fix.
3. **Then §7.3.**
4. **Then WP-3/WP-4**, only if WP-0 found a phenomenon.

Three reasons for that order: z_aware is decode-time on existing checkpoints while §7.3
needs a new density *and* retraining; WP-0 is nearly free and can remove one of the two open
concerns about the spline; and the two rest on different kinds of evidence — z_aware on a
documented code fact (the knob is inert), §7.3 on the *third* proposed mechanism for `dv`
after §8.1 and §9.3 falsified the first two. Spending the cheap and certain one first is the
risk-adjusted choice.

**The nuance that stops this from becoming a dependency.** §7.3's **primary** pre-registered
read is `dv`'s marginal PIT, which is computed from the model's own CDF and does not consult
`mbr_coords` at all. So `PLAN_z_aware.md` is **not** a prerequisite for §7.3's verdict — only
for reading its decode-side consequences. Running the two in parallel is defensible; they
converge at exactly one place, §7.3's `d(MBR)` row, which should not be read until the ruler
question is settled. That is the same trap §6.2 fell into.

### 9.5a UPDATE, 2026-08-06 — steps 1 and 2 are done, and §7.3 is now next

`PLAN_z_aware.md` §11 records the outcome; the consequences for *this* document are three:

1. **Step 1 (its WP-0) ran and found no phenomenon.** §6.2's `d(MBR)` bullet is withdrawn
   above, so the ordering's stated payoff — "in which case §6.2's `d(MBR)` caveat evaporates
   and `lnz_head="spline"` is unambiguously fieldable" — is the outcome that occurred.
2. **Step 2 (its WP-2) shipped**, so the inert-knob trap is closed: `mbr_coords="+lnz"` and
   `mbr_weight="z"` now raise on a cell chain instead of silently measuring 2-D numbers under
   a 3-D label. Its WP-1 ruler shipped with it. **Its WP-3/WP-4 did not run** — they were
   gated on WP-0 — so `+lnz` is loud-and-unavailable rather than functional, which is the
   correct state for a knob nothing currently wants.
3. **§7.3 is therefore step 3, i.e. next**, unchanged in scope and with one reservation
   lifted: **§7.3's `d(MBR)` row may now be read straight.** The resolution of that ruler is
   measured — ±0.01 pooled over 3273 paired jets — so the row is quotable with a band instead
   of a caveat. Everything else about §7.3 (the pre-registered `dv`-PIT primary read, the
   3 seeds, the cost, and §9.4's caveat against firing at all) is untouched: nothing in
   `PLAN_z_aware.md` §11 bears on the factorization.

---

# §10. PRE-REGISTRATION — does the spline's TARP gain transfer to the FIELDED family?

Written 2026-08-06 on branch `nextStepsTrackAB`, **before `contstop_spline_s1`,
`contstop_spline_s2` and `v1_contstop_s2` are trained.** This is `PLAN_next_steps.md` **B1**
and `SUMMARY_Model_Status.md` §4.2(4); §6.4 is the finding it settles, and §6.4's own last
line asked for exactly this arm.

## 10.1 The question, and why one arm cannot answer it

§6.4 measured, on the **explicit-`q(N|x)`** family:

| seed | control | + spline | Δ | vs the MC null p95 = 0.0275 |
|---|---:|---:|---:|---|
| s0 | 0.0415 | **0.0215** | **−0.0200** | crosses below — now passes G7 |
| s1 | 0.0350 | **0.0265** | **−0.0085** | crosses below — now passes G7 |
| s2 | 0.0335 | 0.0400 | **+0.0065** | still fails |

Two of three, with a contrary seed. §6.4 recorded that as *suggestive, not a verdict*.

**But the fielded family is continue/stop**, and there it has exactly **one** arm, which
moved the *other* way:

| seed | control | + spline | Δ | |
|---|---:|---:|---:|---|
| s0 | 0.0200 | 0.0255 | **+0.0055** | both pass; the spline is *worse* |

One seed, going the wrong way, on the family that is actually shipped. That is the state
this pre-registration exists to resolve, and it is a genuinely open question in both
directions: the coordinate density is shared between the families, so the gain either
transfers or it was a property of a posterior that was too narrow to begin with.

## 10.2 The arms

Three trainings, all `presets/prod_test_v1.yaml` — the same preset, data, geometry,
encoder, aux columns, epochs and batch size as every arm above:

| arm | overrides | role |
|---|---|---|
| `contstop_spline_s1` | `trainer.seed=1 model.use_multiplicity_head=false model.lnz_head=spline` | spline, seed 1 |
| `contstop_spline_s2` | `trainer.seed=2 model.use_multiplicity_head=false model.lnz_head=spline` | spline, seed 2 |
| `v1_contstop_s2` | `trainer.seed=2 model.use_multiplicity_head=false` | the **missing control** |

`contstop_spline_s0`, `v1_contstop_s0` and `v1_contstop_s1` already exist, so this
completes three paired seeds. Evaluated with `scripts/eval_prod_test_v1.sh` — the standing
two-pass command, unchanged — so TARP is computed at the same tier as every number above:
**2000 jets, 200 references, `pooled`, 4000 MC null reps, p95 = 0.0275, cpu**.

## 10.3 The verdict rule — fixed before the arms are trained

Primary read: **paired Δ(`tarp_max_dev`) = spline − control**, seed by seed, on the three
continue/stop pairs.

- **T1 — TRANSFERS** iff Δ < 0 on **3 of 3** seeds **and** the mean Δ is **≤ −0.0085**.
- **T2 — DOES NOT TRANSFER** otherwise.

*Where −0.0085 comes from, and it is not this measurement.* It is the **weakest improving
instance of the effect being transferred** — s1's −0.0085 in the table above, measured in
§6.4 and committed long before this section. A transfer has to be at least as large as the
smallest instance of the thing it transfers, or it is not the same effect. It is also
comparable to the explicit family's own control seed spread (0.0415/0.0350/0.0335, range
0.0080), so it is about one seed-to-seed fluctuation — the scale below which three seeds
cannot distinguish an effect from a draw.

*Why 3/3 and not 2/3.* A two-sided sign test on three pairs floors at **p = 0.25**, so the
sign alone can never be evidence here — `SUMMARY` §5's standing lesson, which cost this
repo a withdrawn conclusion one campaign ago. The sign clause is therefore a *precondition*
and the effect-size clause is what carries the verdict; neither is sufficient alone.

**Both outcomes are useful, and what each implies is fixed now:**

- **T1 (TRANSFERS)** → §6.4's finding is a property of the coordinate density, not of the
  family. `SUMMARY` §2.5's TARP row is upgraded from "suggestive" to a statement, and
  `lnz_head="spline"` gains a second axis on the fielded family.
- **T2 (DOES NOT TRANSFER)** → §6.4's TARP row must be quoted as **family-specific** from
  now on: the coordinate density helps a joint posterior that is too narrow (the explicit
  family, which fails G7 on all six arms) and does nothing measurable for one that already
  passes. That is not a mark against the spline — it was fielded on NLL, PIT and support,
  none of which this touches — and it *removes* a claim the roadmap is currently carrying.

**Reported beside the verdict, and NOT scored** (so the arm cannot be re-read into a
different verdict afterwards): `tarp_exceeds_null` and `tarp_exceeds_null_mc` per arm
against the same 4000-rep band; val NLL; `pit_ks_max`; the `ln z` and `dv` PIT KS ratios;
and G3's clause on the three new seeds. One clause is a **sanity gate rather than a
read**: the val NLL must not *regress* on any of the three pairs — the spline improved it
by −0.041 on `contstop_spline_s0` and on 4/4 arms overall, and a seed where it does not is
a training failure to investigate, not a TARP result to report.

**Anti-circularity:** the preset, the seeds, the eval tier and the TARP null band are the
standing ones and are chosen against nothing this produces. No fourth seed is trained if
the first three disagree — three is the number `SUMMARY` §4.2(4) asked for, and adding
seeds until a pattern appears is the failure mode `PLAN_z_aware.md` §11 was written about.
