# SUMMARY — model status, findings, and where improvement lives

Last updated: 2026-08-05, branch `nCeilingProbe`. This document consolidates the state of
the hadron→parton posterior — what has been tested, what fell, what stands, and what to do
next — so the conclusions do not have to be reassembled from eight plan documents. Every
number links back to a primary record; nothing here is a new measurement.

Primary records: `PROD_TEST_v0_RESULTS.md`, `PROD_TEST_v1_RESULTS.md`,
`PROD_TEST_edit_RESULTS.md`, `PLAN_PosteriorClusters.md` (implementation notes),
`PLAN_StratifiedMBR.md` §1a–§1d, `PLAN_NCeilingProbe.md` §A, and the run artifacts under
`runs/prod_test_v1/v1_contstop_s0/…/` (`per_jet_clusters.json`,
`per_jet_clusters_K1000.json`, `eval_metrics_wp34.json`) and
`runs/n_ceiling_probe/20260805-122832/n_ceiling_probe.json`.

---

## 1. The framework, end to end

```mermaid
flowchart TD
    subgraph GEN["Data generation (cpp/)"]
        P["PYTHIA 8.3 + FastJet + fjcontrib LundPlane<br/>pythia_driver.cpp"] --> R["jets.root RNTuple<br/>hadron x + matched parton y (primary Lund sequences),<br/>9 groomed aux scalars, grooming provenance"]
    end

    subgraph DATA["Data layer"]
        R --> DM["LundDataModule / MatchedLundDataset<br/>Geometry (30x30 Lund cells), aux vectors,<br/>pT windows, train/val split + fingerprint"]
    end

    subgraph MODEL["Model registry (one contract: log_prob / sample / map_estimate)"]
        DM --> ENC["Encoders: gru | lundnet | deepsets<br/>(+ optional cross-attention, aux conditioning)"]
        ENC --> FAM["Families:<br/>ar_junipr v1–v4 (autoregressive; v1/v2 continue-stop,<br/>v3/v4 optional explicit q(N|x) head)<br/>cinn | diffusion | cfm | edit_v1/v2 (transducer)"]
    end

    subgraph DECODE["Decode layer (inference-time only, every switch default-off)"]
        FAM --> GATE["stage 0: empty gate<br/>q(N=0|x) >= frozen tau"]
        GATE --> PE["point estimate:<br/>map (beam/argmax) | mbr (Frechet medoid over K draws)<br/>| mbr_n (N-first; measured, NOT recommended)"]
        FAM --> DRAWS["K posterior draws + coordinates<br/>-> K x K perturbative-Lund EMD matrix D"]
        DRAWS --> PE
        DRAWS --> SET["predict_set: cluster D (hdbscan|dbscan|pam)<br/>-> exemplars + masses + top_mass / entropy / radii"]
    end

    subgraph EVAL["Evaluation suite (h2p-rsd-junipr eval)"]
        PE --> EV["closure (dlund, mult bias, empty recall)<br/>calibration (SBC, PIT, TARP+null, coverage_68+null)<br/>support audit | exposure | mode audit<br/>cluster diagnostics (G2..G8') | stability | systematics"]
        SET --> EV
        EV --> ART["eval_metrics.json + figures + plan-doc gate tables"]
    end
```

The decode decision flow for a single jet, as currently recommended:

```mermaid
flowchart LR
    X["jet x"] --> Q["q(N=0|x) >= tau?"]
    Q -- yes --> E["empty tree<br/>(estimator = empty_gate)"]
    Q -- no --> M["MBR medoid over K draws<br/>(the recommended point estimate)"]
    X --> S["predict_set (same D)<br/>top_mass = calibrated probability<br/>entropy = ambiguity, radii[0] = the one honest ±"]
```

---

## 2. What has been established, by test campaign

### 2.1 Production test v0 → v1 (`PROD_TEST_v1_RESULTS.md`)

| finding | evidence | status |
|---|---|---|
| `lnz_support="physical"` closes the support leak completely | 0.83% below-soft-drop and 3.94% `z>½` → 0.0000% on all seeds | **closed** |
| The residual ln z failure is a *shape* mismatch, not support | `ln_z × wide_soft` PIT at **2.16×** its critical value on 2 671 emissions (94% of the bulk); best seed still misses by 5% | **open — the spline escalation fired and was never built** |
| SBC-on-N "failure" was a wrong reference | χ² 215.6 vs "crit 16.90" — but N takes 7 values; against its own simulated null: 88th percentile ⇒ consistent with calibrated | corrected in v1 |
| The joint tree posterior is too narrow, and it is the **multiplicity factorization** | TARP at n=2000 with MC null: all six explicit-`q(N\|x)` arms fail G7, both continue/stop arms pass | **continue/stop is the fielded family** |
| Family A/B: continue/stop wins | held-out NLL −0.124 nat, TARP 0.021 vs 0.037, both clearing seed spread | decided |
| Aux isolation is null | `ln_pt + abs_eta` carry the whole aux gain; 9-column delta inside seed spread; jets that *cannot* receive aux "gain" as much | **encoder-input expansion not justified**; full-tree-LundNet trigger did **not** fire |
| Encoder probe (lundnet/gru/deepsets) | one seed each; NLL differences small, gru worse PITs | no evidence the encoder is the bottleneck |
| Empty parton tree (~17% of jets) needs its own decision | `q(0\|x)` AUC 0.824, scale ratio 1.026, Brier reliability 1.5e-4; τ = 0.3506 frozen by rate-matching; recall 0.46 / precision 0.44 | **gate implemented and calibrated**; per-jet accuracy is the limit, not the scale |
| Edit transducer (`e_v1/e_v2`) | loses the head-to-head to `v1_contstop` on the readable axes (TARP band 3.4× the reference) | not fielded |

### 2.2 Posterior clusters (`PLAN_PosteriorClusters.md`, 600 jets, K=200 and K=1000)

| gate | result | reading |
|---|---|---|
| G2 medoid-in-dominant-cluster | 0.468 (hdbscan) / 0.648 (pam) — verdicts agree | the medoid often leaves the dominant cluster; **kill criterion does not fire** |
| G2′ set value vs mass-matched random-partition null | **+0.603 ± 0.048** (K=200) → **+0.676 ± 0.049** (K=1000) | the partition carries real information (~12σ); survives the silhouette precondition |
| G3 empty stratum mass = q(0\|x) | 0.00000, both K | exact by construction — the N=0 clique **is** the calibrated empty probability |
| G6 reliability of `top_mass` | ECE 0.197→**0.040** at T=0.83, slope 0.97, Brier resolution 0.079 (K=200) | **calibrated AND informative** — the scalars predict the set's error (0.73/0.79 most/least-confident RMS ratios) and not the medoid's |
| G7 conformal coverage, exemplar rule | 0.617 [0.577, 0.655] vs nominal 0.68 — **unreachable** | the ceiling is the **reporting rule**: same sets at the pool's own resolution cover **0.793** (see §2.3) |
| G8′ empty-clique dominance | bounded argmin collapses to empty on **24.5%** vs a 1% ceiling | WP4b (bounded loss) stays closed, independently confirmed |
| the mass argmax as a point estimate | `set0` 11–14% worse RMS than the medoid, CIs excluding 1; over-selects empty **0.455** (hdbscan) vs true 0.167 — a granularity artifact (the N=0 stratum is atomic, the rest fragments) | **`members[0]` is not a tree to ship**; the frozen-τ gate fixes the emptiness completely (0.455→0.130) and moves nothing else |
| K=1000 arm | silhouette precondition 46.2%→**66.3%**, `<n_clusters>` 4.89→3.95, G2′ up | the "unresolvable half" was partly budget; **but** `min_cluster_size ∝ K` confounds G6 across K (residual mass 0.284→0.362, unassigned 35.7%→43.5% while the pool support *improved*) — G6's cross-K row is unscored |

### 2.3 Stratified MBR + the two audits (`PLAN_StratifiedMBR.md` §1a–§1c)

| finding | evidence | consequence |
|---|---|---|
| **`mbr_n` (N-first decode) fails its own pre-registered gate** | Δ = d(medoid) − d(mbr_n) = **−0.083** [−0.128, −0.039] at K=200; **−0.084** [−0.135, −0.030] at K=1000 | significantly *farther* from truth; ships as `point_estimator="mbr_n"`, documented measured-not-recommended |
| Both components lose independently | de-smearing alone −0.043 (cross-stratum distances are inflated but **not noise**); the calibrated median picks N no better than the medoid (0.448 vs 0.443 exact) | both premises of the estimator were wrong |
| **The N channel is the biggest priced lever** | stratified at *true* N: **1.67** vs medoid **2.33** (~0.7 EMD); `P(median = n_true) = 0.4483` at K=200 **and** K=1000, identical to 4 decimals | `q(N\|x)` is **calibrated but not sharp** — and §2.4 now shows the residual is information ***x* lacks**, not information the model or the decode fails to use |
| **`coverage_68` was never evidence of over-confidence** | its own null (model as truth, same K-draw HPD): **0.553** [0.543, 0.563] on 8 841 pseudo-truths; observed 0.546 sits inside it | a perfect model scores 0.553 at K=200, not 0.68 — the deficit is the estimator (HPD from K draws misses cells with p<1/K). Correction note added to `PROD_TEST_v1_RESULTS.md`; **TARP is unaffected** (own MC null), so joint-narrowness now rests on TARP + the PIT cross |
| G7's ceiling is the reporting rule, measured | same sets: exemplar rule 0.617, pool-resolution rule **0.793**, nominal 0.68 | the model's support is fine (`d(truth, nearest draw)` = 0.09 of the pool's scale); a residual 20.7% support gap is real and stated |
| Three selection rules lost to plain centrality | mass argmax 2.72, medoid's-cluster 2.31, N-first 2.43 — vs medoid 2.33 | **the decode layer is exhausted** |

### 2.4 The N-information ceiling probe (`PLAN_NCeilingProbe.md` §A, `scripts/n_ceiling_probe.py`)

The one measurement that decided where the *next* effort goes. A **discriminative**
predictor of `n_true` from `(x, aux)` — 460 594 training jets, 29 features, scored on the
same 600 test jets as everything above. Because predicting a label is far easier than
carrying a correct generative posterior, its accuracy is a **lower bound** on the
multiplicity information `x` carries.

| predictor | exact | 95% Wilson | mean \|Δn\| |
|---|---:|---:|---:|
| probe, x + aux (median) | **0.4550** | [0.416, 0.495] | 0.620 |
| probe, x only (median) | 0.4433 | [0.404, 0.483] | 0.628 |
| **`q(N\|x)` posterior median** | **0.4583** | [0.419, 0.498] | 0.608 |
| `n_x` | 0.3767 | [0.339, 0.416] | 0.800 |
| majority class | 0.3950 | [0.357, 0.435] | 0.775 |

| finding | evidence | consequence |
|---|---|---|
| **The length channel does not under-extract** | probe 0.4550 vs posterior median 0.4583; paired McNemar **36 vs 38, p = 0.91** | **no evidence of headroom** — the residual N ambiguity behaves like hadronization physics, not like a modelling failure |
| **…and the probe is a working instrument**, so the tie is informative | beats the majority class **p = 0.0064** (101 vs 65) and `n_x` **p = 0.00083** (119 vs 72) | a null from a probe that could not learn would measure nothing; this one learns real structure and still lands on `q(N\|x)` |
| **…and it was not starved** | learning curve flat over a **20×** range of training data (23 k → 461 k jets: 0.482 → 0.447, last step +0.007) | the tie is about the information in `x`, not the training budget. Fit-to-fit range 0.063 vs a 0.079 Wilson width — **comparable**, so the test resolves N beliefs to ~0.08 and no smaller; that is the bar a future claim of a win must clear |
| **Aux is null for N too** | x+aux 0.4550 vs x-only 0.4433, paired **p = 0.36** | the *third* independent line clearing the aux columns, on the one quantity they were suspected to help |
| **The oracle lever is real and unreachable** | oracle-N **1.721**, medoid **2.349**, Δ **+0.629** [+0.514, +0.752]; the probe's own n̂ fed to `stratified_medoid` gives **−0.062** [−0.123, −0.001] | at ~45% accuracy, *deciding* N is worse than *not* deciding it — the 55% wrong pay more than the 45% right win |
| **Sanity: same jets, same decode** | d(medoid) recorded 2.3489 → re-measured **2.3495** (0.03%) | the comparison is against the right population; the 0.448 → 0.458 shift is the statistic's own K-draw MC noise (6 jets of 600) |

---

## 3. The overall conclusion

**The encoder is not the bottleneck** (three independent lines: v1's attribution, the null
aux A/B, the flat encoder probe), **the decode layer is now exhausted** (three
selection rules lost to the plain medoid at two budgets; the gate composition works; the
bounded loss is structurally unsafe), **and the length channel is not under-extracting**
(§2.4: a discriminative probe on `(x, aux)` ties `q(N|x)` on identical jets while beating
both trivial predictors). Which leaves the coordinate density as the one place a model
change still has a measured target.

**The recommended per-jet product today:**
- point estimate: **the MBR medoid**, with the frozen-τ empty gate (`decode.empty_threshold`);
- uncertainty: **the cluster set** — `top_mass` as a calibrated probability (after one
  temperature), `entropy` as the per-jet ambiguity, `radii[0]` as the one honest ±,
  quoted at the **K=1000 tier**;
- population-level: the decode-free posterior series, as before.

**Of the two levers that were open, one has now been decided:**

| lever | size | status |
|---|---|---|
| **N channel** | ~0.63 EMD (oracle-N 1.72 vs medoid 2.35) | **CLOSED as a lever** (§2.4). The discriminative probe ties `q(N\|x)` on identical jets (p = 0.91) while beating both trivial predictors and sitting on a flat learning curve — no evidence that `x` carries more N information than the model already extracts. The lever is *real* (the oracle keeps its +0.63) and *unreachable*: at 45% accuracy an N decision is worse than none. It is now a **product**, not a target — the calibrated ambiguity of the set layer is the honest way to report N. |
| **ln z shape** | 2.16× critical PIT in the quadrant holding 94% of emissions | **OPEN — and now the only priced model-side lever.** The RQ-spline head, pre-authorized in v1 §4.4, fired, never built; written up as `PLAN_lnz_spline_head.md`. Then the joint coordinate density if the spline does not close it (`ln z = u + v − ln p_T,sum` holds exactly, so independence-given-cell is violated by a kinematic identity). |

**Explicit stop-signs** (measured dead ends — do not respend effort here):
- new selection rules over the existing posterior (three straight losses);
- aux-column expansion (isolation null; the full-tree-LundNet trigger did not fire; and now
  null for N specifically, §2.4);
- encoder swaps (probe flat, v1 attribution elsewhere);
- **a better length head, or any decode fed by one** — the ceiling probe found no sharper n̂
  to feed it, and a 45%-accurate n̂ loses to no N decision at all (§2.4);
- the bounded/kernel MBR loss as a product (G8′ fails at 24.5% vs a 1% ceiling);
- reading `coverage_68` against 0.68 (its null is 0.553 at K=200 — always quote it with K).

---

## 4. Next steps — what the ceiling probe changed

The probe's verdict re-orders everything below it. Two consequences drive the list:

- **Stop trying to make the point estimate better through N.** The lever is real and
  unreachable, and every route to it that could be tried at the decode or head layer has now
  been tried and lost. What is left is either a *different coordinate density* (§4.2) or
  *more information at the input* (§4.4) — not a smarter use of what is already there.
- **Start treating the N ambiguity as the product.** ~55% of jets genuinely do not have a
  determined parton multiplicity given `x`. A single tree cannot say that; the set can, and
  G6 already says the set's confidence scalars are calibrated and informative. §4.3 is about
  making the set say it *well*.

### 4.1 Diagnostics (cheap, decisive first)

1. ~~**The N-information ceiling probe**~~ — **done** (§2.4, `PLAN_NCeilingProbe.md` §A).
   Verdict: no evidence of headroom.
2. **The sequence-level N probe — the one measurement that could reopen the N lever.**
   The probe of §2.4 reads fixed-length *summaries*, so its null is a lower bound over that
   feature class. The sharper version costs almost nothing: take the already-trained LundNet
   encoder, freeze it, put a `n_true` classification head on the pooled embedding, score on
   the same 600 jets. Reopening the length channel means clearing **0.458 by more than the
   0.063 fit-to-fit spread** §2.4 measured — not merely the Wilson interval; failing to
   clear it converts "no evidence of headroom" into a much stronger statement, since the
   encoder sees the whole sequence the summaries compress. Pre-register the same paired
   McNemar, the same trivial-predictor controls, and the same learning curve before running
   it.
3. **Transfer the `coverage_68` null to one explicit-`q(N|x)` arm** (where TARP *does*
   fail) — one `eval` run with `experiment.coverage_null_reps=20`. Confirms the null
   lands near 0.553 there too, so the correction is family-independent.
4. **Pin `cluster_min_cluster_size=10` at K=1000** — one notebook run. Separates "more
   draws" from "coarser clustering" and un-confounds G6's cross-K comparison.
5. **Per-jet rows in the cluster artifact**, so gate G5's paired criterion becomes
   computable instead of falling back to "quote the K=1000 tier".

### 4.2 Model extensions (pre-authorized, in order)

1. **RQ-spline ln z head** — `model.lnz_head = "truncnorm" | "spline"` (Durkan et al.,
   arXiv:1906.04032) on the same soft-drop interval, bit-identical off path, 3-seed
   training at the v1 budget, G3 PIT re-test against the recorded 1.05–2.07× numbers.
   The cheaper of the two fired escalations, it does not disturb the factorization, and it
   is now **the only priced model-side lever left**. Plan: `PLAN_lnz_spline_head.md`.
2. ~~**Length-channel improvement**~~ — **withdrawn.** It was gated on 4.1(1) showing
   headroom; it did not. All three options are dead for a specific reason and each is worth
   recording so none is re-proposed: the *decode-time auxiliary N-head* **is** the probe of
   §2.4 and it loses to the plain medoid by −0.062 [−0.123, −0.001]; a *recalibrated
   continue head on richer summaries* has nothing to recalibrate toward, since `q(N|x)` is
   already calibrated (G4 ratio 0.977, SBC-on-N at the 88th percentile of its own null) and
   the probe says it is also as *sharp* as these features allow; *capacity in the
   continue/stop path* is an extraction fix for an extraction failure that was not found.

### 4.3 Turn the irreducible N ambiguity into a better product

Not a new point estimate — the medoid stays, and "new selection rules" remains a stop-sign.
These change what is **reported alongside** it, which is where the probe says the value is.

1. **Stratify the set by N, instead of clustering through it.** 83% of the resolvable
   posterior ambiguity is between-N; `q(N|x)` is calibrated; gate G3 shows the empty
   stratum's cluster mass **is** `q(0|x)` exactly. Yet the partition is currently produced
   by hdbscan on `D`, which makes the N = 0 stratum atomic and *fragments* the non-empty
   continuum — the measured granularity artifact (empty selected 29.8% vs a true 16.7%),
   residual mass 0.284, unassigned 35.7%. Partitioning by multiplicity **first** and
   clustering only *within* a stratum gives a top-level partition that is exact by
   construction, carries calibrated masses, and removes the `min_cluster_size ∝ K` confound
   that left G6's cross-K row unscored. Report `q(N|x)` as the top-level mass vector and the
   within-stratum exemplars beneath it.
2. **Quote the N ambiguity explicitly per jet.** `entropy` currently mixes between-N and
   within-N ambiguity into one number. Splitting it — `H[q(N|x)]` beside the mean
   within-stratum entropy — tells a user *which kind* of ambiguity a jet has, and the first
   term is the one the probe just showed is irreducible.
3. **Re-label the oracle row wherever it appears.** `d_mbr_ntrue` = 1.72 is an *ambiguity
   scale*, not a target: after §2.4 it is measurably not achievable from `x`. Any future
   estimator claiming improvement must beat **2.35 without knowing N**.
4. **Lean on the population-level product.** Per-jet N is irreducible; population-level
   posterior series average over exactly that ambiguity, so they are where this model's
   information actually converts into a measurement. Prioritise the decode-free series and
   its systematics over further per-jet point-estimate work.

### 4.4 If more N information is genuinely wanted, it has to come from the input

The probe's real message: a better *head* cannot invent information that is not in `x`. The
only remaining routes add information, and both are data-side rather than model-side, so
both cost a regeneration + retrain and should be costed before being started.

1. **Lower the primary `kt_floor`.** `x` is traversed to 1.0 GeV while the aux columns
   already use an off-spine floor of 0.2 GeV — a factor-5 asymmetry that is deliberate and
   documented, and that throws away exactly the soft emissions whose count correlates with
   the parton multiplicity. A `kt_floor` scan (1.0 → 0.5 → 0.2 GeV) with the §2.4 probe
   re-run at each point measures the information gain **before** any model is retrained:
   the probe is the cheap instrument for exactly this question, and it is now built.
   NP/UE contamination rises as the floor drops, so the scan measures a trade, not a free
   lunch — which is why it is a measurement and not a change.
2. **Full-tree LundNet with secondary-plane sequences** (Dreyer & Qu, arXiv:2012.08526) —
   **low prior, and the reason is now sharper.** Its trigger did not fire on the aux
   isolation, and §2.4 adds that the nine *scalar* summaries of the secondary planes carry
   no N information either (p = 0.36). The sequences might carry what the scalars do not,
   but the evidence points the other way; if it is tried, the §2.4 probe fed secondary-plane
   summaries is the cheap pre-test to run first.

### 4.5 Exploring new models (structural, only if the above stall)

1. **Per-node joint coordinate density** (cINN-coords / CFM-coords, `PLAN_UPDATES.md`
   WP1) — the second fired escalation; structurally motivated by the exact kinematic
   identity, and the fallback if the spline closes the marginal PIT but TARP still fails.
2. **Not currently justified** (triggers measured false, listed so they are not
   re-proposed by default): the edit transducer as the fielded family (lost the
   head-to-head); consensus/lattice MBR or any decode that leaves `H = {pool}`; any
   architecture whose stated benefit is a sharper `q(N|x)` (§2.4).

---

## 5. Methodological notes that keep paying for themselves

Recorded because each one caught a wrong conclusion this cycle:

- **Simulate the reference, never assume it.** SBC-on-N (χ²(9) vs a 7-valued N) and
  `coverage_68` (0.68 vs an HPD built from K draws) were the *same error twice* in one
  suite; both flipped when scored against their own simulated nulls.
- **Pre-register the gate before running the cell.** `mbr_n`'s ship gate was written in
  the notebook markdown above the code; the estimator failed it and the failure is a
  clean result instead of a negotiation.
- **Pair the comparison or don't call it one.** `set0` looked ~40% better than the medoid
  on each series' own subset and is 11–14% *worse* on identical rows.
- **A verdict label must cover both tails**: a CI entirely below zero "excludes 0" too —
  the gate printer briefly reported a significant loss as a null result.
- **Method-stable statements need method-stable rules**: the exemplar support rule swung
  35.7%↔8.2% with the clustering method on identical jets; the pool-resolution rule
  cannot move by construction.
- **Preserve the artifact a conclusion was read from** (`per_jet_clusters_K{K}.json`,
  `eval_metrics_wp34.json`, backups before overwriting) — twice this cycle an artifact
  nearly masqueraded as another run's evidence.
- **A null result needs a positive control — two, in fact.** "The probe ties the model" is
  only a statement about the *data* if the probe can learn at all, and if it was not merely
  starved. `n_ceiling_probe.py` therefore reports the paired tests against the majority class
  and `n_x` (p = 0.0064 / 0.00083) *and* the learning curve over a 20× range of training data
  (flat) beside the headline. Without those rows the headline p = 0.91 would have been
  unreadable in both directions.
- **A single fit's accuracy is not its uncertainty.** Re-fitting the same probe on nested
  subsamples spans 0.418–0.482 (range 0.063) — *comparable to* the 0.079-wide Wilson interval
  on the test set, not negligible beside it. A null that rests on "these two numbers are
  close" has to price both sources, or it is quoting to three decimals a test that resolves
  to one.
- **A statistic computed from K draws carries K-draw noise — quote the band, not the
  decimals.** The 0.4483 that this whole work package was built around re-measured as 0.4583
  on the same 600 jets with fresh draws, because for a continue/stop family `length_pmf`
  *is* the draw histogram and its median flips on jets whose belief straddles two
  multiplicities. Same lesson as `coverage_68`, one layer up: the *estimator* has a
  sampling distribution too.
