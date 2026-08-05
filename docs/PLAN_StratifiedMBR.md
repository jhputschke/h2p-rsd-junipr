# PLAN — N-first (stratified) MBR, and three measurement work packages

Status: **complete. WP-1 FAILS its own ship gate and is not recommended**; WP-2, WP-3 and
WP-4 landed. The estimator ships as an available `decode.point_estimator` value, documented
as measured-not-recommended. §1a is the WP-1 verdict, §1b the K=1000 arm — which confirms
the verdict is budget-independent and turns up a hyperparameter-scaling artifact that
contaminates gate G6.
Follows directly from the 600-jet / K=200 gate run recorded in
`PLAN_PosteriorClusters.md` (implementation-note tables) and answers the open question of
`PLAN_empty_parton_tree.md` ("should the gate be a general argmin over an explicit loss on
n?" — yes: L=|n−m| gives the posterior median, and this plan builds the estimator around
it).

> **Line anchors** taken at commit `1697de8`. Re-verify before editing.

---

## 1. The measured facts this plan is built on

All from the 600-jet, K=200 run on `v1_contstop_s0` (hdbscan, energyflow, cuda), recorded
in `PLAN_PosteriorClusters.md`:

1. **The partition carries real information no truth-free rule extracts.** Mean
   perturbative-Lund EMD to truth by selection rule: global medoid **2.349**, top-mass
   exemplar 2.715, medoid's-cluster exemplar 2.310, closest exemplar (oracle) **1.476**.
   G2′'s mass-matched random-partition null decomposes the oracle advantage on
   multi-cluster jets (2.029 → 0.834) into **0.592 order statistic** (any partition gets
   it) and **0.603 genuine information**.
2. **The ambiguity is mostly multiplicity.** 83% of the resolvable cluster splits differ
   in N.
3. **This family's `q(N|x)` is calibrated** (v1: G4 ratio 0.977; SBC-on-N at the 88th
   percentile of its own null; `q(0|x)` AUC 0.824, scale ratio 1.026), and the posterior
   **median of N** is the repo's recommended count estimator.
4. **The EMD's mass-imbalance term smears the medoid across N strata** — it minimizes
   mean distance to draws of *every* N, so it can sit in the wrong stratum or between
   strata. This is the mechanism behind (1).

Hypothesis: deciding N by the calibrated marginal and the shape by the **conditional
medoid within that stratum** removes exactly the cross-stratum smearing, and should
recover part of the 0.603 information component. Two structural facts make the estimator
clean:

- the median of a histogram pmf is always **realized** (`quantile_floor` returns the
  smallest n with cdf ≥ α, which forces `pmf[n] > 0`), so on the continue/stop family the
  stratum is non-empty by construction;
- `_qn_importance_weights` is constant within a stratum, so `mbr_resample_to_qn` composes
  as an **exact no-op** — the stratified estimator is the exact form of the correction
  that knob approximates.

---

## 1a. RESULT — `mbr_n` is significantly WORSE than the medoid

600 held-out jets, K = 200, `v1_contstop_s0`, hdbscan, energyflow on cuda.

**Ship-gate criterion (i) fails, and in the wrong direction.** `Δ = d(medoid) − d(mbr_n)`,
jet-level paired bootstrap:

| subset | n | mean Δ | 95% CI | verdict |
|---|---:|---:|---|---|
| all jets | 600 | **−0.083** | [−0.128, −0.039] | excludes 0 — `mbr_n` is **farther** |
| multi-cluster | 450 | −0.080 | [−0.127, −0.030] | excludes 0 — farther |
| …top-2 differ in N | 390 | −0.067 | [−0.116, −0.019] | excludes 0 — farther |
| …top-2 same N | 60 | −0.163 | [−0.344, +0.015] | brackets 0 |

Criteria (ii) and (iii) *pass* — RMS-vs-medoid 1.003 / 0.999 / 1.024 on ln(1/ΔR), ln kt,
ln z, and the gated marginals are the best of any estimator (`mbr_n` mean multiplicity
1.407 against truth's 1.442). The shape is fine; the *selection* is worse. All three must
hold, so the gate fails.

**Both components are negative — the hypothesis fails in two independent places:**

| | Δ |
|---|---:|
| de-smearing alone (same N as the medoid, expectation restricted) | −0.043 |
| the N decision (calibrated median vs the medoid's own N) | −0.040 |

1. **Restricting the expectation hurts.** The cross-stratum distances are inflated by the
   imbalance term, but they are **not noise** — they still carry shape information, and
   dropping them costs more than the smearing did. The effective support also falls from
   200 draws to a mean stratum of 92. The premise that the imbalance term is a pure
   nuisance to be conditioned away is wrong.
2. **The calibrated median is no better at picking N than the medoid already was:**
   `P(n̂ = n_true) = 0.448` against `P(N_medoid = n_true) = 0.443`
   (mean `|n − n_true|` 0.618 vs 0.630). The premise that the medoid picks N badly and a
   calibrated marginal would fix it is also wrong.

### What the run does establish — and it is the more useful result

| rule | mean d(truth) | truth-free? |
|---|---:|---|
| global medoid | 2.349 | yes |
| stratified at N(medoid) | 2.392 | yes |
| **`mbr_n`** | 2.432 | yes |
| top-mass exemplar | 2.715 | yes |
| **stratified at n_true** | **1.661** | **no — oracle N** |
| closest exemplar | 1.476 | no — oracle |

**Knowing the true N would buy 0.688 of the medoid's 2.349** — the largest single lever
measured anywhere in this line of work, and larger than the whole 0.603 information
component the cluster partition carries. No truth-free rule reaches it, because `q(N|x)` is
right on only 45% of jets.

That sharpens, and partly corrects, the reading recorded in
[`PLAN_PosteriorClusters.md`](PLAN_PosteriorClusters.md): `q(N|x)` is **calibrated but not
sharp**. It has the right rate and the right ranking (G4 ratio 0.977, `q(0|x)` AUC 0.824)
while being wrong about *which* N on more than half the jets. The 0.770 residual is
information the model does not have — not information the decode is failing to use.

**So the next lever is more information about N — conditioning, architecture, training —
not a better decision rule over the existing posterior.** Three decode-layer selection
rules have now been measured against the medoid (mass argmax, medoid's-cluster, N-first)
and all three lose. That is a sufficient basis to stop looking there.

**What stays.** `point_estimator="mbr_n"` remains available and documented as
measured-not-recommended; `mbr` stays the recommended decode. The negative result is worth
more than the estimator would have been: it converts "the medoid smears across strata" from
a plausible mechanism into a measured non-problem, and it prices the N channel.

---

## WP-1 — stratified MBR (`decode.point_estimator="mbr_n"`)

### Core: two siblings in `inference/mbr.py` — NOT a flag on `mbr_select`

`mbr_select` stays byte-untouched: its `.risk` contract ("achieved mean distance over all
K draws") and the G1 bit-identity tests stay structurally unthreatened. The repo's idiom
for "a different estimator over the same D" is a sibling entry point
(`mbr_select` / `mbr_cluster_set`).

```python
def stratified_medoid(D, mults, n_hat, *, w=None) -> tuple[int, float, int]:
    """Within-stratum Frechet medoid: argmin over rows of D[stratum, stratum],
    stratum = {k: mults[k] == n_used}. Returns (win_idx GLOBAL, risk = within-stratum
    (weighted) mean distance, n_used). Square D required (raises naming
    mbr_n_candidates). Zero EMD calls. Pure numpy — CI-fast-tier testable."""

def mbr_select_stratified(model, xf, nx, *, draws=None, geom, ..., n_quantile=0.5,
                          D=None) -> LundPointEstimate:
    # kwargs mirror mbr_select exactly so map_or_mbr can pass **mbr_kwargs_from_decode
    # unchanged; D= + draws= accepted together (the mbr_cluster_set pattern);
    # n_candidates accepted and raises if nonzero.
```

- `n_hat = quantile_floor(length_pmf(..., mults=reused), 0.5)` (`inference/length.py`,
  reused unchanged). The empty gate is NOT re-implemented here: `map_or_mbr` runs it
  before dispatch; a median of 0 with the gate off honestly returns the empty medoid
  (risk exactly 0.0).
- **Fallback when `n_hat` is unrealized** (explicit-head families only): nearest
  populated stratum by `|n − n_hat|`, ties → larger pool mass, then smaller n. Never
  raise (a legitimate runtime state, not misconfiguration); never the global medoid
  (that silently reverts to the estimator this one replaces on exactly the most
  N-ambiguous jets). `n_used` is returned so callers record when the guard fired.
- `.risk` = the **within-stratum** mean — the achieved risk of the decision that produced
  this tree; `estimator = "mbr_n"` on every return path is the provenance that
  disambiguates it. Degenerate branches mirror `mbr_select`'s (zero draws → empty tree,
  risk 0.0; singleton stratum → that draw, risk 0.0).
- `point_estimate.py`: `_ESTIMATOR_LABEL["mbr_n"] = "MBR-N (stratified)"`; empty-footer
  `why["mbr_n"] = "the posterior median multiplicity is 0 — the N decision itself
  answered empty"`; one line on the `risk` field comment.
  `clusters.py: assert_ancestral_draws` branch extended to `est in ("mbr", "mbr_n")` so
  the rejection message is accurate.

### Productionization (same change — user decision)

- `map_or_mbr` (`models/base.py`): one `elif point_estimator == "mbr_n"` →
  `mbr_select_stratified(..., **mbr_kwargs_from_decode(decode))`, after the existing
  empty gate. `"map"`/`"mbr"` branches untouched. **No new config field** — a new *value*
  on the existing `point_estimator`, `n_quantile` hard-coded at 0.5: no
  `_DECODE_DEFAULTS` change, no config-hash movement.
- **The six `point_estimator == "mbr"` call sites become `in ("mbr", "mbr_n")`**
  (verified): `eval/closure.py:165,548` (`want_mbr` — the closure MBR series applies to
  either), `eval/report.py:72` (inert-key marking), `serving/api.py:70` (`is_mbr`
  draw-reuse), `cli.py` cluster-diagnostics guard. One-line comment at each.
  `predict_set` is unaffected.
- `docs/CONFIGURATION.md`: `point_estimator` row gains the third value + a short §10
  subsection (what: two-stage decode, calibrated N then conditional medoid; why: fact 2
  and fact 4 above; the `.risk` semantics note). `PLAN_empty_parton_tree.md`: the open
  question is now implemented — cross-reference.

### Measurement (`scripts/make_per_jets_cluster_nb.py`, same change)

In `estimate_jet` (hoist `pmf` — `q0` is currently read inline): `n_hat`,
`stratified_medoid(D, mults, n_hat)`, plus two zero-EMD controls:

- `win_m = stratified_medoid(D, mults, N(medoid))` — **de-smearing alone** (same N
  information as the medoid, expectation restricted);
- `win_t = stratified_medoid(D, mults, n_true)` — oracle ceiling of the N channel given
  this shape rule;
- `d_oracle_stratum = d_to_truth[mults == n_used].min()` — ceiling of the shape rule
  given this N decision.

New series `mbr_n` (headline; `RATIO_REF["mbr_n"] = "mbr"`) and `mbr_n_gated`
(`RATIO_REF = "mbr_gated"`, gated-vs-gated per the §6 discipline); both in `HEADLINE` and
the §5b/§6b marginals. New per-jet columns: `n_true, n_hat, n_used, n_hat_realized,
stratum_size, risk_n, d_mbr_n, d_mbr_nmed, d_mbr_ntrue, ntrue_populated,
d_oracle_stratum`, and `n_hat_cond` (median of pmf with `pmf[0]` zeroed — measured, not
shipped).

**New §9b — the decision table** (pre-registered in the markdown before execution):

1. *Selection-rule ladder* (mean d(truth); columns: all | multi-cluster | strata_differ |
   same-N): global medoid → stratified-at-N(medoid) → **mbr_n** → stratified-at-n_true
   (oracle) → min-over-decided-stratum (oracle) → closest exemplar (oracle, 1.476). The
   differences that matter: `d_mbr − d_mbr_nmed` (de-smearing alone) and
   `d_mbr_nmed − d_mbr_n` (what the calibrated median adds — the claim under test);
   `d_mbr_n − d_mbr_ntrue` prices what a better length head could still buy.
2. *The N decision*: `P(n̂ = n_true)` vs `P(N_medoid = n_true)` vs `P(N_set0 = n_true)`
   vs `P(N_map = n_true)`; mean `|n − n_true|` (L1 — the loss the median is Bayes for);
   split by gate-fired × truth-empty; `n_hat_realized` rate.
3. **Ship gate**: paired per-jet `Δ = d_mbr − d_mbr_n`, jet-bootstrap 95% CI, overall +
   multi-cluster + strata_differ, quoted as a fraction of the 0.603 information
   component. **`mbr_n` becomes the recommended decode iff** (i) the CI on Δ excludes 0;
   (ii) §6's RMS-vs-RSD ratios for `mbr_n` are no worse than `mbr`'s within their CIs on
   ln(1/ΔR) and ln kt (the failure mode that killed `set0`: 1.112/1.141 on identical
   rows); (iii) §6b marginals for `mbr_n_gated` are no worse than `mbr_gated`'s. A Δ flat
   across the strata_differ split ⇒ the gain is de-smearing, not N information —
   reportable either way. If the gate fails, the config value stays (documented as
   measured-not-recommended) and the results note records the verdict.

`eval/clusters.py: run_cluster_diagnostics`: per-jet `d_mbr_n`, `n_hat`, `n_hat_correct`,
`n_medoid_correct` (`D`, `mults`, `dt` already in hand) + a row in `summarise_clusters`'s
printed ranking.

### Tests (`tests/test_mbr.py`, new banner `# --- N-first (stratified) MBR ---`)

1. `stratified_medoid` vs a hand-computed 6×6 `D`, `mults = [0,0,2,2,2,3]`: sub-block row
   means by arithmetic; a mult-3 row with the smallest *global* mean is never selected at
   `n_hat = 2`; `max|Δ| == 0.0` vs the literal `sub.mean(axis=1)`; a
   constant-within-stratum `w` changes nothing.
2. Nearest-populated fallback: `n_hat = 1` → mass tie-break → `n_used = 2`; `n_hat = 5`
   → 3; equal mass → smaller n.
3. Single-stratum parity: all draws the same N ⇒ same tree, bit-identical `.risk` as
   `mbr_select` (the structural anchor: stratification is a no-op when there is nothing
   to stratify).
4. `mbr_select` bit-identical before/after a stratified call (the predict_set pattern).
5. Label + risk: `estimator == "mbr_n"`, `pretty()` label, empty-dominated pool →
   multiplicity 0 / risk ≈ 0 / still `"mbr_n"`, `assert_ancestral_draws` rejects it.
6. Supplied `coords_by_draw` reproduced exactly; winner multiplicity == `n_used`.
7. Guards raise: `n_candidates=3` (match `mbr_n_candidates`), `D=` without `draws`.
8. `map_or_mbr(point_estimator="mbr_n")` dispatch + gate composition (τ fires ⇒
   `empty_gate` label wins); the six-site update spot-checked via `run_closure` metric
   keys under `"mbr_n"`.

Tests 1–2 are pure numpy (CI fast tier); model-facing ones POT-gated per existing style.

---

## 1b. RESULT — WP-2, the K=1000 arm (gate G5)

600 jets at K = 1000, same arm and decode, **13.2 min** on cuda (the K² block is 25× the
K = 200 one). Artifact: `per_jet_clusters_K1000.json`.

**The conclusions are budget-independent. That is the point of the arm.**

| | K=200 | K=1000 |
|---|---:|---:|
| `mbr_n` Δ, all jets | −0.083 [−0.128, −0.039] | **−0.084 [−0.135, −0.030]** |
| `P(n̂ = n_true)` | 0.4483 | **0.4483** |
| stratified at n_true (the N lever) | 1.661 | **1.668** |
| global medoid | 2.349 | 2.329 |
| G2 medoid-in-dominant | 0.468 | 0.472 |

`mbr_n`'s failure is not a small-sample artifact, and the 0.745 N-channel headroom is a
property of the model rather than of the budget — `P(n̂ = n_true)` is identical to four
decimal places at 5× the draws. **`q(N|x)` is calibrated but not sharp, and no amount of
sampling fixes that.**

**What K does buy — the silhouette question is answered:**

| | K=200 | K=1000 |
|---|---:|---:|
| silhouette precondition holds | 46.2% | **66.3%** |
| G2′ gain vs the null (all multi) | +0.603 ± 0.048 | **+0.676 ± 0.049** |
| `<n_clusters>` | 4.89 | 3.95 |
| d(truth, nearest draw) / pool scale | 0.092 | **0.074** |
| MC error on a mass at m = 0.6 | 0.035 | 0.015 |

So the "unresolvable half" at K = 200 was **partly budget**: two thirds of jets have
resolvable structure at K = 1000, the partition carries *more* real information, and it
fragments less (fewer, better-supported clusters). The set-valued layer is better at
K = 1000 on every measure that describes the partition itself.

### An unexpected finding: the cluster hyperparameters do not transfer across K

`cluster_min_cluster_size = 0` resolves to `max(5, ceil(0.05·K))` — **10 draws at K = 200,
50 at K = 1000**. HDBSCAN's notion of a cluster therefore demands a much denser region as K
grows, and more of the pool falls to noise:

| | K=200 | K=1000 |
|---|---:|---:|
| residual (noise + sub-threshold) mass | 0.284 | **0.362** |
| truth unassigned by the exemplar rule | 35.7% | **43.5%** |
| G6 ECE (tempered) | 0.0395 | 0.0526 |
| G6 slope | 1.12 | 0.43 |
| G6 Brier resolution | 0.0785 | 0.0267 |

**The G6 degradation is confounded and must not be read as "the masses got worse".** G6 is
computed on the *assigned* jets only, and the assignment rule rejects 43.5% at K = 1000
against 35.7% at K = 200 — while the pool actually brackets the truth **better**
(nearest-draw ratio 0.074 vs 0.092). More of the posterior is labelled noise, so more truths
sit near an unclustered draw and are ruled outside every exemplar's support; the surviving
subset G6 scores is more selected, and its calibration degrades for that reason.

This is the same exemplar-rule artifact WP-3 was built for, now shown to contaminate **G6**
as well as G7 — and to get *worse* with budget, which is the opposite of what a budget arm
is usually expected to show. Two consequences:

- **`coverage_pool` (WP-3) is the right instrument at any K**, and the K = 1000 arm is the
  strongest evidence for it: the model's support improved while the exemplar rule's verdict
  got worse.
- **A K comparison at fixed `cluster_min_mass` is not a clean control.** Re-running
  K = 1000 with `cluster_min_cluster_size` pinned to 10 (the K = 200 absolute value) would
  separate "more draws" from "coarser clustering". Not done here, and G6's cross-K row
  should be treated as unscored until it is.

**G5 verdict.** The per-jet paired criterion ("`top_mass` and `entropy` agree within their
binomial error on ≥ 90% of jets") is **not computable from these artifacts** — the notebook
records aggregates, not per-jet rows. What is measurable is that the aggregate scalars move
in ways fully explained by the granularity change above (`entropy` 1.417 → 1.143 with
`<n_clusters>` 4.89 → 3.95), so the plan's fallback applies: **quote the K = 1000 tier**,
and do not mix tiers in one table.

---

## WP-2 — K=1000 budget arm (gate G5)

- Notebook §12: artifact filename stays `per_jet_clusters.json` at the default K and
  becomes `per_jet_clusters_K{K}.json` otherwise, so the K=200 artifact is never
  clobbered.
- One executed run at `K_DRAWS=1000, N_JETS=600` (~35–40 min: the EMD block grows ×25,
  sampling on cuda).
- Comparison (cell or small script reading both artifacts): silhouette-precondition rate
  vs 46.2%; the G5 criterion (`top_mass`/`entropy` agree within their binomial error on
  ≥ 90% of jets); G2′ gain + the oracle-gap decomposition at K=1000; `mbr_n` vs medoid at
  K=1000.

## 1c. RESULT — WP-3 and WP-4 on a full eval run

600 jets, K = 200, `v1_contstop_s0`, via `h2p-rsd-junipr eval` with
`experiment.cluster_diagnostics=true experiment.coverage_null_reps=20`. Artifact:
`eval_metrics_wp34.json` (the arm's own `eval_metrics.json` was preserved).

### WP-4 — `coverage_68` is NOT evidence of over-confidence. The deficit is the statistic.

| | value | on |
|---|---:|---|
| `coverage_68` | **0.546** [0.502, 0.589] | 502 jets |
| **its own null** (model as truth, same K-draw HPD) | **0.553** [0.543, 0.563] | 8 841 pseudo-truths |
| difference | −0.007 | — |

**The observed coverage sits inside the null's interval**, and the null is tight. A
*perfect* model scores **0.553** under this construction, not 0.68: the empirical HPD-68
built from K = 200 draws cannot contain cells of probability below 1/200, and a calibrated
truth still lands in them. The entire 0.68 → 0.55 gap is the estimator.

**This corrects a conclusion carried in several places**, including
`PROD_TEST_v1_RESULTS.md` §1 ("the joint tree posterior is measurably too narrow … *and the
leading-cell coverage says the same thing*"), its G4 regional clause, and my own earlier
readings in this session. It is the same class of error v1 itself caught in SBC-on-N — a
mid-rank statistic on a 7-valued discrete N scored against a *continuous* χ²(9) null — and
it went unnoticed for the same reason: the reference was assumed rather than simulated.

**What still stands.** TARP is a separate statistic with its own MC null recomputed at
n = 2000, and its finding — the explicit-`q(N|x)` arms are too narrow, both continue/stop
arms pass — is untouched, as is the `narrow_soft` PIT attribution. What falls away is the
*corroboration* leading-cell coverage was supplying, which means the joint-narrowness case
now rests on TARP and the PIT cross alone.

**Scope of the null.** It is drawn at the same K as the statistic it explains — it must
share the estimator's handicap — so it is specific to `n_closure_samples = 200` and does
not transfer to another budget. Quote it with its K.

### WP-3 — same sets, two coverage rules, 0.617 vs 0.793

| rule | coverage |
|---|---:|
| exemplar support (`assign_truth`) | **0.617** [0.577, 0.655] |
| pool resolution (`pool_covered`) | **0.793** |
| nominal (α = 0.32) | 0.680 |

The conformal set's mean size is 4.94 against `<n_clusters>` ≈ 5, so the emitted prefix is
essentially the full set — the two rows are the *same sets* judged two ways. Under the
exemplar rule G7's ceiling is 0.617 (= 1 − the 38.3% unassigned rate) and the nominal 0.68
is unreachable; at the pool's own nearest-neighbour scale the same sets cover **0.793**, and
0.68 becomes reachable with margin.

That converts §1's inference into a direct measurement: **G7's failure is the reporting
rule, not the model's support.** Supporting numbers: `pool_bound` = 0.630 against a mean
nearest-draw distance of 0.294, so the bound is ~2× the pool's own resolution — loose enough
to be fair, tight enough that 20.7% of truths still fall outside it. The residual 0.207 is a
real support gap and is not explained away.

Both rules stay reported. They measure different things — "is the truth inside the region
this exemplar represents" versus "did the pool put a draw near the truth at all" — and the
exemplar rule's failure remains on the record rather than being tuned away.

---

## WP-3 — pool-based coverage beside G7

- `inference/clusters.py: pool_coverage_bound(D)` — the pool's own resolution scale
  (95th percentile of per-draw nearest-neighbour distance; references pool geometry,
  never the partition — method-stable by construction, which is the property the
  exemplar rule lacks: 35.7% "unassigned" under hdbscan vs 8.2% under pam on the same
  jets). Truth covered by the emitted prefix iff `min(d_to_truth over draws of the
  emitted clusters) <= bound`. Zero new EMD.
- `eval/clusters.py` + notebook §11: `coverage_pool` reported **beside** the exemplar
  rule with the explicit note: a pre-registered alternative *definition*, not a slack
  tune — the exemplar rule and its hdbscan-granularity failure stay on the record.
- Test: pam vs hdbscan on the same synthetic two-lobe pool give the same pool-coverage
  verdict (the motivating property).

## WP-4 — `coverage_68` self-consistency null

Motivation: leading-cell 68% coverage reads 0.527 vs 0.68 on every arm while TARP passes
for the continue/stop family — the same posterior cannot be both. v1 already caught one
discreteness trap in exactly this territory (SBC on a 7-valued N against a continuous
χ²(9) null); an empirical HPD set built from K=200 draws over 900 cells deserves the same
audit before being read as over-confidence.

- `eval/calibration.py`, behind `experiment.coverage_null_reps: int = 0` (pattern:
  `tarp_null_reps` — config.py + `_EXPERIMENT_DEFAULTS` + `experiment_params`). When
  > 0: per jet draw `K + M` (M = coverage_null_reps); build the empirical HPD-68 from
  the K exactly as production does (`run_calibration`'s leading-cell block); score the M
  held-out draws as pseudo-truths; report `coverage_68_null` ± binomial error beside
  `coverage_68`.
- Interpretation stated in the metric dict: null ≈ 0.53 ⇒ the deficit is the statistic
  (the empirical HPD from K draws misses cells with p < 1/K that a calibrated truth still
  lands in) and G4's regional clause needs this null as its reference; null ≈ 0.68 ⇒ the
  over-confidence is real and the coordinate heads are the target.
- Tests: a synthetic categorical where the model IS the truth → `coverage_68` ≈
  `coverage_68_null`; a deliberately narrowed model → `coverage_68` drops while the null
  stays. Default-off parity: only adds keys when on (the `tarp` convention).

---

## Order of execution

1. WP-1 source + productionization + tests + docs (one commit).
2. WP-1 notebook wiring; 600-jet run at K=200 → the §9b decision table; verdict recorded
   in `PLAN_PosteriorClusters.md`'s results note (commit).
3. WP-3 + WP-4 (independent, small).
4. WP-2 K=1000 run (long, last) → comparison recorded.

## Verification

- `pytest tests/` green after each commit (818 baseline); ruff clean on touched files;
  `scripts/verify_parity.py` unchanged.
- G1: the `"map"` and `"mbr"` paths bit-identical (tests 3–4 above + the existing parity
  suite).
- The 600-jet K=200 run is WP-1's decision instrument; the K=1000 run is WP-2's.
- Gate verdicts (ship/no-ship for `mbr_n`, G5, pool coverage, coverage null) appended to
  `docs/PLAN_PosteriorClusters.md`.
