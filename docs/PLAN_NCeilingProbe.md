# PLAN — the N-information ceiling probe, the consolidated conclusion, and the spline-head to-do

Status: **implemented and run.** WP-A landed as `scripts/n_ceiling_probe.py` and its verdict
is §A below; WP-B is `docs/PLAN_lnz_spline_head.md`; WP-C is `PLAN_StratifiedMBR.md` §1d and
`SUMMARY_Model_Status.md`. Follows directly from `PLAN_StratifiedMBR.md` §1a–§1c and decides
where model-improvement effort goes next.

**Result in one line:** the discriminative probe reaches **0.4550** against the generative
posterior median's **0.4583** on identical jets (paired p = 0.91) while beating both trivial
predictors ($p \le 0.006$) — *no evidence of headroom in the length channel*, so the ~0.63
EMD oracle-N lever is **real and out of reach**, and the remaining model-side lever is the
`ln z` shape (WP-B).

---

## 1. Context — where improvement lives now

The accumulated measurements leave exactly two levers, both on the **decoder/head** side.
The encoder is triple-cleared — v1's attribution ("it is the multiplicity factorization,
not the coordinate heads, the encoder, or the support"), the null aux A/B (`ln_pt +
abs_eta` carry the whole aux gain; the full-tree-LundNet trigger did **not** fire), and
the flat encoder probe. The decode layer is exhausted — three selection rules (mass
argmax, medoid's-cluster, N-first) all lost to the plain medoid, at two budgets.

1. **The N channel.** The oracle-N decode reaches **1.67** against the medoid's **2.33**
   (~0.7 EMD — the largest lever measured anywhere in this line of work), but `q(N|x)`'s
   median is right on only **0.448** of jets, identical at K=200 and K=1000: calibrated
   but not sharp. **Open question, never measured: is 0.448 an information ceiling of
   `x`, or an extraction failure of the generative length model?** This gates whether any
   architecture/training work on N can pay.
2. **The ln z shape.** `ln_z × wide_soft` PIT at **2.16×** its critical value on 2 671
   emissions — the quadrant holding 94% of them. The RQ-spline escalation fired in
   `PROD_TEST_v1_RESULTS.md` §4.2 and was never executed.

Decision (user): **implement the ceiling probe now (WP-A); record the spline head as the
future to-do (WP-B, doc only); consolidate the conclusion (WP-C).**

---

## WP-A — the N-information ceiling probe (`scripts/n_ceiling_probe.py`)

A **discriminative** predictor of `n_true` from `(x, aux)` — a far easier task than the
generative posterior, so its accuracy is a *lower bound* on the information `x` carries
about N. Precedent for a standalone probe script: `scripts/probe_map_collapse.py`.

**Data.**
- Train: `data/jet_aux_asym.root` (the checkpoint's training file), `len(x[0]) > 0`
  population, target `n_true = len(j["y"][0])` clipped to a `7+` bucket.
- Test: the **first 600 jets with `len(x[0]) > 0`** of `data/jet_aux_asym_test.root` —
  the exact population the 0.448 was measured on, for direct comparability.

**Features** (per jet, existing helpers only):
- `n_x`; per-node summary stats of `features.node_raw(*j["x"])` — mean/std/min/max of the
  four coordinates, leading and subleading `ln kt`, `sum(exp(ln kt))`.
- The 9 aux columns via `features.AUX_FEATURES`, with drop-and-count sentinel handling
  (the `AUX_MAX_DROP` pattern of the cluster notebook §3).
- **Two variants**: with aux, and x-only. The aux A/B was null for *NLL*, but the
  `n_sec = 2–3` stratum carried signal, so aux may matter for N specifically — this is a
  free second measurement.

**Model.** `sklearn.ensemble.HistGradientBoostingClassifier` (sklearn is already in the
`[mbr]` extra), multiclass over `n = 0..7+`; from `predict_proba` report both the argmax
(exact accuracy) and the distribution **median** (the L1-Bayes answer, comparable to
`n_hat`).

**Baselines** (all already measured): posterior median **0.448** exact / **0.615** L1;
`N(medoid)` 0.452 / 0.615; raw `n_x`; the majority class.

**The EMD payoff row.** One fresh pass over the 600 test jets (cuda, ~2 min): K=200
draws, `D` from `inference.mbr.posterior_distances`, then
`stratified_medoid(D, mults, n̂)` for n̂ ∈ {classifier, posterior median, true N} beside
the plain medoid, `d(truth)` via `lund_emd_matrix` against the truth cloud (template:
`estimate_jet` in `scripts/make_per_jets_cluster_nb.py`). Paired jet-bootstrap CI on Δ vs
the medoid.

**Pre-registered reading, printed by the script before its numbers:**
- accuracy **> 0.448** with a binomial CI excluding it ⇒ the length channel
  **under-extracts**; the `mbr_n` machinery exploits a sharper n̂ immediately, and
  length-model work is justified;
- accuracy **≈ 0.448** ⇒ *no evidence of headroom with these features* — a lower bound,
  not proof of a ceiling — and the set layer's calibrated ambiguity is the right product
  for N;
- either way the EMD row prices what the measured n̂ buys against the 1.67 oracle.

**Output**: printed table + `runs/n_ceiling_probe/<stamp>/n_ceiling_probe.json` via
`eval.report.save_metrics`. `--fast` smoke flag (small subsample). Ruff-clean, functions
importable; no unit tests, per the probe-script precedent.

---

## WP-B — the spline ln z head, recorded as the next model to-do (doc only)

New stub `docs/PLAN_lnz_spline_head.md` (one PLAN per work item), status
**proposed — pre-authorized escalation, not yet implemented**, kept short:

- The trigger already fired: G3 fails on the truncated head on every seed
  (`PROD_TEST_v1_RESULTS.md` §4.2, §6), residual `ln_z × wide_soft` at 2.16× critical on
  the bulk quadrant; "a truncation cannot fix a shape mismatch inside the interval".
- The escalation as pre-authorized: a monotone rational-quadratic spline (Durkan et al.,
  arXiv:1906.04032) on the same soft-drop interval; `model.lnz_head = "truncnorm" |
  "spline"` with a bit-identical off path; 3-seed training at the v1 budget; G3 PIT
  re-test against the recorded 1.05–2.07× numbers.
- The joint-coordinate-density escalation stays the follow-up if the spline does not
  close G3 (`ln z = u + v − ln p_T,sum` holds exactly, so independence-given-cell is
  violated by a kinematic identity — a per-coordinate head cannot express that).
- Cross-referenced from `PLAN_StratifiedMBR.md`'s conclusion section.

---

## WP-C — the consolidated conclusion

Append **§1d "Where improvement lives now"** to `docs/PLAN_StratifiedMBR.md`:

- **Settled**: decode layer exhausted (three selection rules lost, two budgets); encoder
  cleared (three independent lines); `coverage_68` corrected (the deficit is the
  statistic — a perfect model scores 0.553 at K=200); G7's ceiling is the reporting rule
  (0.617 exemplar vs 0.793 pool on the same sets); the set layer's real product is
  calibrated ambiguity, quoted at the K=1000 tier.
- **The two levers with their prices**: the N channel (~0.7 EMD, gated on WP-A's
  verdict) and the ln z shape (2.16×, WP-B's to-do).
- **Explicit stop-signs** so effort is not respent: selection rules over the existing
  posterior, aux-column expansion, encoder swaps.
- WP-A's result recorded here when it lands.

---

## Order of execution

1. WP-A script; `--fast` smoke, then the full run; verdict recorded.
2. WP-B + WP-C docs (WP-C carries WP-A's numbers).
3. One commit per WP; branch `nCeilingProbe`.

## Verification

- `python scripts/n_ceiling_probe.py --fast` runs end-to-end in ~1 min; the full run
  produces the table and the JSON artifact.
- Sanity row: the probe's posterior-median baseline reproduces 0.448 on the same test
  population as `per_jet_clusters.json`. *(Met, and it taught something — §A.4: the medoid
  reproduces to 0.03%, while 0.448 itself re-measures as 0.458 because it is a statistic
  computed from K draws.)*
- `pytest tests/` stays green (842 baseline); ruff clean on new/touched files.

---

# §A. RESULT — the length channel does not under-extract

Run 2026-08-05, `python scripts/n_ceiling_probe.py`, artifact
`runs/n_ceiling_probe/20260805-122832/n_ceiling_probe.json`. Train: the **460 594** jets of
`data/jet_aux_asym.root` with `len(x) > 0`; test: the **first 600** such jets of
`data/jet_aux_asym_test.root`; **0 aux-dropped on either side**, so nothing about the
population was reshaped by the screen. 29 features (20 sequence summaries + the 9 aux
columns), `HistGradientBoostingClassifier` over `n = 0 … 7+`, early stopping on an internal
split — **nothing was tuned against the test population**, which is what keeps the ceiling
verdict from being circular.

## A.1 The accuracy table

| predictor | exact | 95% Wilson | mean \|Δn\| |
|---|---:|---:|---:|
| **probe, x + aux — distribution median** | **0.4550** | [0.416, 0.495] | 0.620 |
| probe, x + aux — argmax | 0.4500 | [0.411, 0.490] | 0.638 |
| probe, x only — median | 0.4433 | [0.404, 0.483] | 0.628 |
| probe, x only — argmax | 0.4450 | [0.406, 0.485] | 0.632 |
| **`q(N\|x)` posterior median** — the reference | **0.4583** | [0.419, 0.498] | 0.608 |
| `n_x`, the hadron multiplicity | 0.3767 | [0.339, 0.416] | 0.800 |
| majority class (`n = 1`) | 0.3950 | [0.357, 0.435] | 0.775 |

Both point rules are reported because they answer different questions: the **median** is the
L1-Bayes answer and is the one comparable to `n_hat` (literally the same `quantile_floor`
applied to a different belief); the **argmax** is the 0-1-loss answer and is the more
generous reading of "how much can a classifier know". Neither exceeds the generative
model's 0.4583, and the argmax — the rule with the freest hand — does worse than the median.

## A.2 Verdict — `no_evidence_of_headroom` (the pre-registered second branch)

The probe's 95% interval **contains** the posterior median's value, and the **paired**
McNemar test on identical jets is **36 vs 38** discordant, **p = 0.91**. As beliefs about N
on these 600 jets, the discriminative probe and the generative length model are
indistinguishable. The interval [0.416, 0.495] also contains the *recorded* 0.4483, so the
verdict does not depend on which of the two references is used — which matters, because the
recorded value carries the draw noise §A.4 measures.

**The probe is a working instrument**, which is what makes the tie informative rather than
vacuous — a null from a probe that cannot learn measures nothing:

| paired against | probe-only right | other-only right | p |
|---|---:|---:|---:|
| majority class | 101 | 65 | **0.0064** |
| `n_x` | 119 | 72 | **0.00083** |
| the x-only probe | 25 | 18 | 0.36 |

So the features do carry real multiplicity information beyond the trivial predictors, the
probe extracts it, and it still lands exactly on `q(N|x)`.

**The aux A/B is null for N as well** — the free second measurement the plan asked for.
`x + aux` 0.4550 vs `x only` 0.4433, paired p = 0.36. This is now the *third* independent
line clearing the aux columns (v1's isolation A/B, the `n_sec` stratum, this), and it is the
sharpest, because N is the one quantity aux was specifically suspected to help with.

## A.2b Was the probe simply starved? No — the learning curve is flat

The second control, and the one that decides whether the null is about `x` or about the
training budget. Independent fits on nested subsamples, scored on the same 600 jets:

| n_train | median | argmax |
|---:|---:|---:|
| 23 029 | 0.4817 | 0.4633 |
| 57 574 | 0.4517 | 0.4583 |
| 115 148 | 0.4183 | 0.4367 |
| 230 297 | 0.4400 | 0.4317 |
| **460 594** | 0.4467 | 0.4400 |

**Twenty times the training data buys nothing** — there is no trend, and the last step is
+0.007. A probe that was still climbing would have made the tie unreadable; this one is
saturated well before the full sample, so the tie is a statement about the information in
`x`.

It also supplies the second half of the honest error bar. There are two independent noise
sources and a null result has to price both:

| source | size |
|---|---:|
| test-set noise — the Wilson interval on 600 jets | width **0.079** |
| fit-to-fit variability — the range across the five curve fits (0.418–0.482) | **0.063** |

They are **comparable**, so neither dominates and the headline is not an artifact of either.
Taken together, **this test separates beliefs about N down to about 0.08, not to a third
decimal**. `q(N|x)`'s 0.4583 sits deep inside that of the probe's 0.4550, which is why the
tie is read as a tie — and, symmetrically, why a future probe claiming to beat 0.458 has to
clear ~0.08, not merely tip a Wilson interval.

## A.3 The EMD payoff — the lever is real, and out of reach

600 jets, K = 200, `energyflow` backend, one `K × K` matrix per jet feeding every row:

| n̂ feeding `stratified_medoid` | d(truth) | Δ vs medoid | 95% CI (paired jet-bootstrap) |
|---|---:|---:|---:|
| plain medoid — no N decision at all | 2.349 | — | — |
| probe (x + aux) median | 2.411 | **−0.062** | [−0.123, −0.001] |
| probe (x only) median | 2.430 | **−0.080** | [−0.147, −0.017] |
| **true N (oracle)** | **1.721** | **+0.629** | [+0.514, +0.752] |

Read together with §1a: the oracle keeps its ~0.63 EMD, so the lever is **not** an artifact
of the earlier run — it is the largest single number in this line of work and it survives a
fresh posterior pass. But **the best measurable n̂ loses to the plain medoid**, by the same
margin `mbr_n`'s calibrated median did (−0.083 recorded, −0.062 here, CIs overlapping).

That is the mechanism worth stating plainly: at ~45% accuracy, *deciding* N is worse than
*not* deciding it. `stratified_medoid` restricts both the candidate set and the expectation
to one stratum, so a correct n̂ removes the EMD's mass-imbalance smearing and a wrong one
throws away every draw that could have rescued the estimate. At p = 0.45 the 55% pay more
than the 45% win. The oracle's +0.63 and the measured −0.06 are therefore not a gap to be
closed by a better decode — they are separated by information that is not in `x`.

## A.4 The sanity row

Same jets, fresh draws, against what `per_jet_clusters.json` recorded:

| quantity | recorded | re-measured |
|---|---:|---:|
| posterior-median exact accuracy | 0.4483 | 0.4583 |
| d(truth) of the plain medoid | 2.3489 | **2.3495** |
| d(truth) at oracle N | 1.6613 | 1.7209 |

The medoid reproduces to **0.03%**, which is what says this script is looking at the same
jets under the same decode. The 0.448 → 0.458 shift is the statistic's own Monte-Carlo
noise: for a continue/stop family `length_pmf` **is** the histogram of the K draws, so the
median flips on jets whose belief straddles two multiplicities — 6 jets of 600 here. **Quote
0.448 with a ±0.01 draw-noise band, not as a four-decimal constant** (the oracle row moves
more, 1.66 → 1.72, because a rarely-populated true-N stratum is the most draw-sensitive
quantity in the ladder — still ~1 bootstrap SE).

## A.5 What this closes, and what it does not

**Closes.** Length-model work as a priced lever. There is no measured headroom to buy, so
none of the three options §4.2(2) of `SUMMARY_Model_Status.md` listed — a discriminatively
trained auxiliary N-head at decode time, a recalibrated continue head on richer summaries,
capacity in the continue/stop path — has a target to aim at. The first of the three is in
fact exactly what this probe *is*, run at decode time in §A.3, and it loses.

**Does not close.** This is a **lower bound**, and it is stated as one. The probe reads
fixed-length summaries of `x` plus nine per-jet scalars; a sequence-level discriminative
model could in principle read more. What it does establish is that the *generative* length
model is not leaving anything on the table that this class of features can see — and that
`q(N|x)`'s residual ambiguity behaves like hadronization physics rather than like an
extraction failure.

**What would reopen it.** One measurement, cheap and specific: the same LundNet encoder
already trained here, with a *classification* head on `n_true`, beating 0.458 significantly
on these jets. That is the sequence-level version of this probe and the only remaining way
the "extraction failure" hypothesis survives. Until someone runs it, the N channel is
**closed as a lever and stands as a product**: the calibrated ambiguity of the set layer is
the right thing to ship for N, and `radii[0]`/`entropy` already report it.
