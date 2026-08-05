# PLAN — the RQ-spline `ln z` head

Status: **implemented, trained and measured.** The verdict is §6 below: gate G3 is
**PARTIAL** — the `ln z` PIT falls from 1.05–2.07× its critical value to 0.47–1.04×, held-out
NLL improves on every seed, the support closure is kept exactly, and two of three seeds
close both pre-registered clauses while the third sits marginally over the line. The
residual has **moved to `dv`**, which is now the binding coordinate on all three seeds.

**Result in one line:** the spline does most of what it was authorized to do — it is the
first change in this line of work to improve the likelihood *and* the calibration together —
but it does not close G3 outright, and the next lever it exposes is the *same fix applied to
the within-cell offsets* rather than the structural escalation.

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
- **d(MBR) to truth: marginally WORSE on all four arms** (+0.0047, +0.0113, +0.0091,
  +0.0031 against a control seed spread of 0.018). Small and inside the spread in
  magnitude — but **consistently signed on 4 of 4**, so it is stated rather than dismissed.
  The mechanism is indirect and worth naming: the MBR metric runs on
  `mbr_coords="lnDR_lnkt"` and does not read `ln z` at all, so a better `ln z` density
  cannot help it and can only perturb which trees are drawn.

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
