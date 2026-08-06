# PLAN — next steps, ordered

Status: **an execution order, not a new decision.** Written 2026-08-06 on branch
`zAwareMetric`, after `PLAN_z_aware.md` §11 (WP-0: no `d(MBR)` regression) and §13 (§12's
BUILD verdict for the `ln z`-aware selection).

**What this document is.** The work that is decided-but-unbuilt, in the order it should be
done, with the file-level specs and the pre-registration requirements attached — so whoever
picks it up next does not have to re-derive them from eight plan documents.

**What this document is not.** A source of truth for any *conclusion*. Every item names the
document that owns it, and when the two disagree **the owning document wins**. Nothing here
is a new measurement or a new decision; the same contract `SUMMARY_Model_Status.md` holds
itself to, one layer up.

---

## 1. The order, in one table

> **Executed 2026-08-06 on branch `nextStepsTrackAB` — A1–A4 then B4, B3, B5, B1.**
> The `state` column below is now the OUTCOME; **§8 records what each one found** and names
> the owning document's result section, which is where the numbers live.

| # | item | owner | state | result |
|---|---|---|---|---|
| **A1** | **WP-3 — thread the coordinates** | `PLAN_z_aware.md` §4/WP-3, §13.4 | **BUILT** | `decode.mbr_cloud_source`; bit-identical off, verified |
| **A2** | WP-3's test clauses | `PLAN_z_aware.md` §6 | **SHIPPED** | +24 tests across four files |
| **A3** | `_truth_cloud` `kt`-weight mismatch + its measurement | `PLAN_z_aware.md` §4/WP-3 inset | **FIXED + MEASURED** | `PLAN_PosteriorClusters.md` §18 — small in mass, 9.6% of `<d_mbr>` in charge |
| **A4** | Should `+lnz` become the **default** decode? | `PLAN_z_aware.md` §14 (pre-reg), §15 | **AVAILABLE-NOT-DEFAULT** | D1 fails, D2/D3 pass; the headline pays 2.7× the post-hoc estimate |
| **B4** | Pin `cluster_min_cluster_size=10` at K=1000 | `SUMMARY` §4.1(5) | **RUN — goal unreachable** | `PLAN_PosteriorClusters.md` §19: the partition is not budget-invariant either way |
| **B3** | Transfer the `coverage_68` null to an explicit-`q(N\|x)` arm | `SUMMARY` §4.1(3) | **TRANSFERS** | `PLAN_StratifiedMBR.md` §1e — pooled 0.5504 vs 0.553; and a shipped test was wrong |
| **B5** | Per-jet rows in the cluster artifact | `SUMMARY` §4.1(6) | **SHIPPED** | `METRICS["per_jet"]`, keyed by the jet-file index |
| **B1** | 3-seed continue/stop spline arm | `SUMMARY` §4.2(4), `PLAN_lnz_spline_head.md` §10 (pre-reg), §11 | **RUN** | see §8 |
| **B2** | Sequence-level N probe | `SUMMARY` §4.1(2) | decided, unrun | ~1 day |
| **B6** | §7.3 joint coordinate density | `PLAN_lnz_spline_head.md` §7.3, §9.4 | **needs an explicit decision** | new density + retrain |
| **B7** | Turn the N ambiguity into a better product | `SUMMARY` §4.3 | open — **and B4 is now an argument for it** | — |
| **B8** | `kt_floor` scan (1.0 → 0.5 → 0.2 GeV) | `SUMMARY` §4.4(1) | open | regeneration |

**A1–A3 were one piece of work** — the same diff — and A3 was a defect three live gates sat
on. **Two of the runs did not come back with what the plan expected**, and both are why the
pre-registrations were written first: A4's cost to the fielded headline is nearly 3× the
post-hoc estimate a later-written rule would have been scored against, and B4's goal turns
out to be unreachable by the route the plan named. Both are in §8.

---

## 2. Track A — what §13's BUILD verdict created

### A1 — WP-3, the coordinate threading

**Why, in one line:** putting `ln z` in the EMD ground metric recovers **47–70% (mean ≈59%)**
of the measured `ln z` ceiling, −0.026…−0.047 with **8/8 paired CIs excluding 0**, and
de-quantization alone delivers nothing (0/8), so the gain is attributable to `ln z`
specifically (`PLAN_z_aware.md` §13.1). The knob `mbr_coords="+lnz"` currently **raises**
(WP-2), which was the honest state while nothing wanted it. Something does now.

The spec is `PLAN_z_aware.md` §4/WP-3 verbatim; reproduced here as a checklist:

| file | change |
|---|---|
| `config.py` | `decode.mbr_cloud_source: str = "cells"` (`"cells"` or `"coords"`), beside `mbr_coords`, plus the tolerant backfill dict. **Explicit, not implicit** — `make_per_jets_cluster_nb.py` and `make_inference_demo_cluster_nb.py` already pass `coords_by_draw` for winner decoration, so an implicit "coordinates supplied ⇒ use them" would silently move their numbers. `inert_decode_keys` picks it up free via the `mbr_` prefix |
| `inference/mbr.py` | new `coords_for_draws(model, xf, nx, draws)`: one batched `sample_coordinates_many`, **unfiltered and index-aligned**, float64, raising **by family name** when `has_continuous_coords` is False. `posterior_distances` gains `coords_by_draw` / `cloud_source` and **never draws** — it raises when it needs coordinates and has none. The draw happens **once**, in `mbr_select` / `mbr_select_stratified` / `mbr_cluster_set`, inside an `if needs_coords:` block, and the same array feeds both the clouds and `describe_cells(..., win_coords)` — so the tree shown sits at the coordinates its cloud was built from and a double draw is structurally impossible. **Keep the 4-tuple return**: six sites unpack it, two inside notebook-generator string literals |
| `eval/closure.py` | `run_closure(..., coords_by_jet=None)`; take the per-jet coordinates once and reuse them for both `map_or_mbr` and the `cont_ok` block |
| `eval/clusters.py` | `run_cluster_diagnostics(..., coords_by_jet=None)`, forwarded into `posterior_distances`. `_truth_cloud` needs no code change, but its docstring must record that under `"coords"` both sides finally are continuous |
| `eval/stability.py` | **no change** — it consumes `D` alone; listing it would produce a diff that changes nothing |

**The hazard, and it is the largest one in this plan.** `run_closure`'s existing coordinate
call is *filtered* (`[list(d) for d in draws if len(d)]`), and `sample_coordinates_many` pads
to `L_max` over the list it is given — so unfiltering changes the block shape, reorders RNG
consumption, and moves `dlund_*_cont` and the `psi` block **on the default path**
(`PLAN_z_aware.md` §7.1). The unfiltered call must live strictly behind the switch.

**Definition of done:**

1. `mbr_coords="+lnz"` and `mbr_weight="z"` work under `mbr_cloud_source="coords"` and still
   raise under `"cells"`.
2. **Bit-identical with the switch off**, verified the way WP-1 was: run the closure metric
   dict before and after on a fixed synthetic setup and diff it — the diff must contain
   additions only. *(Reset both RNG streams between runs: `torch.manual_seed` and the model's
   private `_decode_generators`; `decode_generator` advances per call and persists.)*
3. A2's tests pass; the full suite is green; `scripts/verify_parity.py` still reproduces the
   v2 reference bit-for-bit.
4. `tests/test_notebooks.py` green — two of the six unpack sites are inside notebook-generator
   string literals.

**It is not a "new selection rule".** `SUMMARY` §3 lists that as a stop-sign, on three
straight losses (mass argmax 2.72, medoid's-cluster 2.31, N-first 2.43 against the medoid's
2.33). This is the **same Fréchet medoid under a different ground metric**, and unlike those
three it **won its pre-registered gate** (`PLAN_z_aware.md` §12.2 → §13.2). The distinction is
recorded so neither the stop-sign nor this exception is applied by analogy later.

### A2 — the WP-5 clauses that could not be written before A1

From `PLAN_z_aware.md` §6, minus the parts already shipped:

- `tests/test_mbr.py` — `posterior_distances` raises naming `coords_for_draws`;
  `coords_for_draws` is index-aligned and unfiltered over an empty draw, and raises **by
  family name** on `ar_junipr_v1`; coordinates are drawn **exactly once** (monkeypatch count);
  the winner's reported nodes are the rows its cloud used; `coords_by_draw` supplied under
  `"cells"` is bit-identical; `"coords"` de-quantizes to the supplied `(u, v)`.
- `tests/test_config.py` — add `mbr_cloud_source` to the exact-set assertion (**it fails
  otherwise**); assert the default and the old-snapshot backfill.
- `tests/test_clusters.py` — `assert_cluster_metric_ok` is coords-dependent (an `R` passing at
  2-D failing at `+lnz`); truth and draws are in the same representation under `+lnz` (both
  column 2 non-constant); `|W_truth − W_draw| / W ≈ 0` under `"coords"` and demonstrably
  nonzero under `"cells"` — **pinning both the defect and its fix** (A3).
- `tests/test_shared_draws.py` — `run_closure` reuses supplied coordinates (monkeypatch the
  sampler to raise); **with the switch off the existing filtered call is still taken**, which
  is what pins the bit-identity hazard above.

### A3 — the `_truth_cloud` `kt`-weight mismatch

**A live defect, not a hypothetical.** `_truth_cloud` builds the truth from continuous `yraw`
rows while every draw cloud is built from cell centres, so under the default
`mbr_weight="kt"` the truth's point weights are `exp(v_continuous)` and the draws' are
`exp(v_cell_centre)` — a per-point mismatch of `exp(±0.1) ≈ [0.905, 1.105]` at the fielded
`n_bins = 30`, plus a systematic Jensen inflation of the truth cloud's total mass, which the
EMD charges at `R·|ΔW|` with `R = 8.485`.

**`d_top`, `d_best`, `d_mbr`, `d_nearest_draw` and gates G2′/G6/G7 sit on it today.** It does
not touch `dlund_mbr` (a plain Euclidean distance, no EMD).

It is fixed as a **side effect** of A1's threading — which is exactly why it was orphaned when
WP-3 was cancelled (`PLAN_z_aware.md` §11.3) and has an owner again now.

**Do not assume the effect is small.** Report `W_truth/W_draw` and `R·|ΔW|` against the
typical `d`, then re-read G2′, G6 and G7 and say in `PLAN_PosteriorClusters.md` whether any
verdict moved. Under `mbr_weight="z"` the same mismatch would be catastrophic.

### A4 — should `+lnz` become the **default** decode? *(needs its own pre-registration)*

**Not answered by §13, deliberately.** §12.2's B2 was written against the **continuous**
`(u, v)` ruler and passed 0/4 violations — but the *fielded* headline is `dlund_mbr`, which
compares leading-emission **cell centres**, and it was not covered. Measured post-hoc
(`PLAN_z_aware.md` §13.3):

> Δ`dlund_mbr` = cont-3D − cells-2D: **+0.0042 [+0.0004, +0.0081]** pooled (n = 6560), 1/8
> significant per arm. `cont-2D` alone: +0.0012 [−0.0021, +0.0046], not significant.

That is a gap in the §12 pre-registration, reported rather than resolved by choosing a
flattering ruler. Until it is settled, **`+lnz` ships available-not-default**, the way `mbr_n`
and `dv_head="spline"` ship — with the difference that this one won its gate.

**Before running anything:** pre-register a rule with `dlund_mbr` as the **primary** read, a
threshold justified against something other than this measurement, and a stated position on
the trade (the gain is ≈10× the cost in absolute terms and 8/8-significant against 1/8; but
one is a diagnostic and the other is the fielded headline). Note also that `dlund_mbr` already
loses to the free one-node `dlund_posterior_medoid` by ≈0.020 on all eight arms
(`PLAN_z_aware.md` §3, reading 4), so +0.0042 is a fifth of a gap the decode already carries.

---

## 3. Track B — the standing roadmap

Unchanged by this session. Each item's case, gates and prior evidence live in the owner
document; only the ordering rationale is here.

- **B4 — pin `cluster_min_cluster_size=10` at K=1000** (`SUMMARY` §4.1(5)). One notebook run.
  Separates "more draws" from "coarser clustering" and **un-confounds G6's cross-K row**,
  which is unscored today. First because it is a single run that converts an unscored gate
  into a scored one.
- **B3 — transfer the `coverage_68` null to one explicit-`q(N|x)` arm** (`SUMMARY` §4.1(3)).
  One `eval` with `experiment.coverage_null_reps=20`. Confirms the 0.553 correction is
  family-independent.
- **B5 — per-jet rows in the cluster artifact** (`SUMMARY` §4.1(6)), so gate G5's paired
  criterion becomes computable instead of falling back to "quote the K=1000 tier". The shape
  precedent now exists on **both** halves of the suite: `eval/clusters.py`'s `metrics["per_jet"]`
  and, since WP-1, `run_closure(per_jet=True)`.
- **B1 — a 3-seed continue/stop spline arm** (`SUMMARY` §4.2(4)). Two of three
  explicit-`q(N|x)` seeds crossed below the TARP null band once the coordinate density
  improved and one moved the other way, while the **fielded** family has a single arm. The
  cheapest way to turn a suggestive result into a statement.
- **B2 — the sequence-level N probe** (`SUMMARY` §4.1(2)). The one measurement that could
  reopen the N lever: freeze the trained LundNet encoder, put an `n_true` head on the pooled
  embedding, score the same 600 jets. **Pre-register the bar before running** — clearing 0.458
  by more than the **0.063 fit-to-fit spread**, not merely the Wilson interval — plus the same
  paired McNemar, the same trivial-predictor controls and the same learning curve.
- **B7 — turn the N ambiguity into a product** (`SUMMARY` §4.3): stratify the set by N instead
  of clustering through it; split `entropy` into between-N and within-N; re-label the oracle
  row (`d_mbr_ntrue` = 1.72 is an *ambiguity scale*, not a target).
- **B8 — the `kt_floor` scan** (`SUMMARY` §4.4(1)): 1.0 → 0.5 → 0.2 GeV with the §2.4 probe
  re-run at each point, measuring the information gain **before** any retrain. NP/UE
  contamination rises as the floor drops, so it prices a trade, not a free lunch.

---

## 4. Two decisions nobody has made yet

Listed separately because they are **judgement calls**, and the plans deliberately stop short
of making them.

### 4.1 B6 — the joint coordinate density (`PLAN_lnz_spline_head.md` §7.3)

**For:** `dv` fails on **13 of 13** arms across two density families and with/without
cell-centre conditioning; both cheap fixes inside the per-coordinate factorization are spent
(§8.1 the spline, §9.3 the conditioning); and the kinematic identity
`ln z = u + v − ln p_T,sum` makes independence-given-cell false **by construction**, so this
is a mechanism with a derivation rather than a third hypothesis. §9.4 promoted it from
deferred to **indicated** as a *pre-registered consequence* of §8.5(1)'s DEAD verdict — that
promotion binds.

**Against, and it got heavier on 2026-08-06:** §9.4's caveat was written *conditionally*,
pointing at a measurement that had not run — "if the `ln z`-aware decode metric shows the
`d(MBR)` regression is an artifact, the fielded product is in good shape on every axis except
a 10% miss on one coordinate's marginal." The condition resolved, by a route neither branch
anticipated: **there is no regression at all** (`PLAN_z_aware.md` §11.1). So the caveat is now
*live* rather than hypothetical, and `lnz_head="spline"` carries **one** open reservation (G3
formally PARTIAL at 5/6 seeds) rather than two. `dv`'s miss is KS 0.026–0.029 against a 0.0255
critical — 2–13% over — and it is the only failing coordinate.

**The ordering (`PLAN_z_aware.md` §10) does not decide this.** `SUMMARY` §5 already records
why: *"Pre-registration binds the verdict, not the next question."* Steps 1 and 2 of that
ordering are done, so the pointer is at §7.3 — but arriving there by momentum is the exact
failure mode that note was written about.

**What "decided" would look like:** a short written call stating the price (a new density
*and* retraining, 3 seeds), the pre-registered **primary** read — *does `dv`'s marginal PIT
fall below 1.0× on every seed?* **not TARP**, since a joint density that fixes TARP while
leaving `dv` at 1.1× has not explained the defect — what it changes about the fielded product
if it works, and whether a 2–13% miss on one coordinate's marginal is worth that.

**A cheaper pre-test worth considering first** *(a suggestion, not repo doctrine)*: the case
rests on a correlation the factorization cannot represent, and that correlation is measurable
in the data directly — condition on cell, look at the residual dependence between `dv` and
`ln z` that the identity predicts. If it is small at the fielded `n_bins = 30`, a joint density
cannot buy much, and that is learned for the cost of a diagnostic instead of a training grid.
Same "cheap and certain before expensive and hypothesised" logic that made WP-0 pay off.

### 4.2 A4 — whether `+lnz` becomes the default decode

See §2/A4. Building it (A1) and defaulting to it are separate calls; A1 does not presuppose A4.

---

## 5. Stop-signs — do not re-propose

Carried verbatim from `SUMMARY` §3, because a roadmap document is exactly where a measured
dead end gets quietly re-proposed:

- **new selection rules over the existing posterior** — three straight losses. *(A1 is a new
  ground **metric** under the same rule, and it won its pre-registered gate. That distinction
  is the exception; it is not a licence to reopen the rules.)*
- **aux-column expansion** — isolation null, the full-tree-LundNet trigger did not fire, and
  null for N specifically.
- **encoder swaps** — probe flat, v1's attribution elsewhere.
- **a better length head, or any decode fed by one** — the ceiling probe found no sharper `n̂`,
  and a 45%-accurate `n̂` loses to no N decision at all.
- **the bounded/kernel MBR loss as a product** — G8′ fails at 24.5% against a 1% ceiling.
- **reading `coverage_68` against 0.68** — its null is 0.553 at K=200; always quote it with K.
- **`mbr_coords="+lnz"` as a fix for the `d(MBR)` regression** — *new, 2026-08-06.* There is no
  regression (`PLAN_z_aware.md` §11). A1 is justified on a different measurement entirely
  (§13), and conflating the two is how §5's verdict rule would get mis-scored later.

---

## 6. The discipline that applies to everything above

Not new; collected because each item here is a place it would be tempting to skip. Full
versions with the incident that produced each one are in `SUMMARY` §5.

1. **Pre-register the gate — and the consequence — before the cell runs.** §12's B1/B2/B3 were
   committed (`e5e3d38`) before the arms; the result (`e9ea938`) is a result because of it.
2. **Pair the comparison or don't call it one.** Twice now: `set0`, and §2.5's `d(MBR)`. An
   *available* pairing left untaken is a claim quoted to three decimals from a test that
   resolves to one.
3. **A consistent sign across a handful of arms is not a small effect — it is an untested
   one.** 4/4 floors at p = 0.125, and that particular 4/4 was 3/4 at 1000 jets with one arm
   significantly the other way.
4. **Simulate the reference, never assume it.**
5. **Quote the band, not the decimals** — a statistic computed from K draws carries K-draw
   noise, and so does the estimator itself.
6. **A small change in an objective is not a small change in its argmin.** *New, from §13.2:*
   §5 predicted `winner_moved_rate < 0.05` as the *expected* outcome; it measured ~60%.
7. **Check that a proposed explanation has something to explain** — §2.5's mechanism was
   literally correct about the code and was accounting for a number that does not survive its
   own noise.
8. **Preserve the artifact a conclusion was read from.** `runs/` is gitignored, so the
   `zaware_wp0` / `zaware_sel` artifacts are local-only and regenerable from
   `scripts/mbr_zaware_ab.py` and `scripts/zaware_selection_ceiling.py`. If a conclusion has to
   survive this machine, copy the JSON somewhere durable.

---

## 7. Where each item's truth lives

| topic | owning document |
|---|---|
| the `d(MBR)` null, the `ln z` ruler, WP-2's guard, WP-3's spec, §12/§13's BUILD verdict | `PLAN_z_aware.md` |
| the RQ-spline `ln z` head, `dv`, the conditioning experiment, §7.3 | `PLAN_lnz_spline_head.md` |
| the N-information ceiling | `PLAN_NCeilingProbe.md` |
| cluster gates G2–G8′ | `PLAN_PosteriorClusters.md` |
| `mbr_n` and the stratified decode | `PLAN_StratifiedMBR.md` |
| the consolidated status, the roadmap's *rationale*, the stop-signs | `SUMMARY_Model_Status.md` |
| **the order and the file-level specs** | **this document** |

When this document and an owner disagree, **the owner wins** — and this one gets fixed.

---

# §8. RESULTS — what the execution found, 2026-08-06

Branch `nextStepsTrackAB`. **This section is a pointer, not a record**: every number below
lives in the owning document's own result section, and when the two disagree the owner
wins (§7). What is here is the outcome and the one thing worth carrying forward.

## 8.1 A1 / A2 — WP-3 is built

`decode.mbr_cloud_source` (`"cells"` | `"coords"`) ships. `coords_for_draws` draws the
jet's coordinate table **once**, unfiltered and index-aligned, and the same array feeds
both the clouds and the winner's `describe_cells` — so the tree that comes back sits at the
coordinates its own cloud was built from and a double draw is structurally impossible.
`posterior_distances` never draws and raises naming `coords_for_draws`. The 4-tuple return
is preserved; all six unpack sites, including the two inside notebook-generator string
literals, are untouched.

**Definition of done, all four clauses:**

1. `+lnz` / `+psi` / `mbr_weight="z"` work under `"coords"` and still raise under `"cells"`.
2. **Bit-identical with the switch off**, verified the way WP-1 was:
   `scripts/zaware_wp3_bitidentity.py` dumps the closure *and* cluster metric dicts from a
   reset RNG state — **both** streams, since `decode_generator` persists — and diffs them
   against a pristine HEAD worktree. **1569 → 1641 leaf keys; ADDED 72, REMOVED 0,
   MOVED 0.**
3. A2's clauses pass (+24 tests: `test_mbr.py` ×13, `test_clusters.py` ×4,
   `test_shared_draws.py` ×4, `test_config.py`, `test_calibration_v2.py` ×2); the full
   suite is green; `scripts/verify_parity.py` reproduces the v2 reference at
   `max |Δ| = 0.000e+00`.
4. `tests/test_notebooks.py` green.

One incidental fix the raise needed: `build_model` now stamps `model.model_name`. One class
serves `ar_junipr_v1..v4` and only v1 lacks a coordinate density, so `type(model).__name__`
cannot name the family — and `skeleton_search_spec`'s existing `getattr(self,
'model_name', '?')` had been printing a literal `?` since it was written.

## 8.2 A3 — the weight mismatch is small in mass and not small in charge

**`PLAN_PosteriorClusters.md` §18.** Isolated on the *same* tree through both
representations, so the genuine multiplicity imbalance is divided out:
`W_truth / W_truth_as_drawn` = **1.0030**, buying a spurious `|ΔW|` of **0.213** —
**9.6% of `<d_mbr>`** and **72% of `<d_nearest_draw>`**. The plan's *"do not assume the
effect is small"* was right.

`d_mbr` does **not** move when the draws are placed at their own coordinates
(+0.0024 [−0.0185, +0.0241], 600 paired jets), so the recommended per-jet product is
untouched. G2, G6 and G7 verdicts all flip to *pass*, and §18.3 declines to bank them: the
cause is de-quantization, which cannot be separated from the weight fix because the `kt`
weight **is** `exp(v)`, and G6's ECE only improves because the forecaster becomes nearly
constant (Brier resolution 0.0623 → 0.0109).

**Carry forward:** the committed G2/G6/G7 numbers are conditional on the cell-centre
representation and that was never stated. `metrics["clusters"]["config"]` now carries
`mbr_cloud_source`, and `metrics["clusters"]["weight_audit"]` prices the residual on every
run.

*Two corrections to my own first pass, both material and both caught before the write-up:*
the first audit divided `W_truth` by the mean **draw** weight, mixing the physical
multiplicity imbalance into the representation defect (it read 0.53 for a quantity that is
1.003); and it charged `R·|ΔW|` while the run used `energyflow`, whose distances are `1/R`
of `pot`'s — an 8.485× overstatement. `_weight_audit` now takes the backend's own ground
scale, the convention `_empty_value` already used.

## 8.3 A4 — AVAILABLE-NOT-DEFAULT, on a rule fixed before the run

**`PLAN_z_aware.md` §14 (pre-registration, `fa45c9f`) → §15 (result).** 8 arms × 1000 jets
× 2 decodes off byte-identical draws and coordinates, on jets **`[1000, 2000)`** — disjoint
from everything §11/§12/§13 scored — through the **fielded** `run_closure` path A1 built.

| | rule | measured | |
|---|---|---|---|
| **D1** *(primary, `dlund_mbr`)* | pooled CI upper < +0.010 **and** ≤ 2/8 arms significantly worse | +0.0114 **[+0.0075, +0.0152]**; **4/8** | **FAIL** |
| **D2** | Δ`dlnz_mbr` ≤ −0.020, CI excluding 0, on ≥ 6/8 | **7/8**; pooled −0.0289 [−0.0349, −0.0230] | **PASS** |
| **D3** | nothing else moves; both no-EMD controls exactly 0 | 8/8; controls `max|Δ| = 0.00e+00` | **PASS** |

**The single most useful number in this whole execution is the D1 one.** §13.3's post-hoc
estimate, computed from stored cell ids on jets `[0, 1000)`, was **+0.0042 [+0.0004,
+0.0081]**. Measured on a disjoint slice through the code a user actually runs, the cost is
**+0.0114** — low by a factor of nearly three. Any rule with a threshold near +0.005 would
have been scored against an estimate that was wrong in the direction that flatters the
answer. §14.1's two design choices — a disjoint population and the real code path — are
the entire reason that is visible.

D2 passing was not a formality either: §13 scored the coordinate table directly, while this
goes through `describe_cells`, the empty gate and the `min_emissions` floor. **A1 ships a
knob that does what §13 said it does.**

## 8.4 B4 — the goal is unreachable by the route the plan named

**`PLAN_PosteriorClusters.md` §19.** "Hold the granularity fixed" is two instructions:
`cluster_min_cluster_size` is a **count** and `cluster_min_mass` is a **fraction**. Pinning
only the count at `K = 1000` gives **1.2** reportable clusters per jet (79% of the mass in
the residual bucket); pinning both in absolute draws gives **30.1**; the committed
fraction convention gives 4.1; the `K = 200` target is 4.9. All four were run.

The decomposition resolves exactly three quantities — `d_mbr`, `d_nearest_draw`,
`pool_bound` — and they are the three that are **not** statements about the partition. The
granularity contrast moves them by exactly **0.0000**, which is the design's own check.
Every partition statistic is confounded, and the two effects run in **opposite directions
at comparable size**, so the committed cross-K pair's small deltas are a near-cancellation
rather than a null.

**So §2.2's "G6's cross-K row is unscored" stands — now with a reason rather than a
caveat**, and one more run of B4 as written would not change it. It becomes an independent
argument for `SUMMARY` §4.3(1) (stratify by `N` first): that partition's top level is
`q(N|x)`, which does not move with `K` at all.

**A clean by-product**, and the first statement in this repo of what `K` buys separated
from the partition: going 200 → 1000 draws improves `d_mbr` by **−0.0426 [−0.0695,
−0.0194]**, `d_nearest_draw` by −0.0681 and `pool_bound` by −0.2672. About 7% of the N
lever, bought with compute rather than information.

## 8.5 B3 — the null transfers, and the test that read it was wrong

**`PLAN_StratifiedMBR.md` §1e.** Positive control first: this runner re-measures §1c's
`v1_contstop_s0` at `coverage_68` 0.5478 (recorded 0.546) and null 0.5476 (recorded 0.553).

Pooled over three `v1_base` seeds the explicit-`q(N|x)` null is **0.5504** on 26 334
pseudo-truths against §1c's 0.553: **−0.0026 [−0.0145, +0.0094]**. **The correction is
family-independent**, as the mechanism predicts, and the stop-sign now rests on two
families and four arms instead of one and one. One residual is reported rather than
smoothed: `v1_base_s1` is −0.067 [−0.112, −0.022] below its own null.

**And it found a defect in shipped code.** `coverage_68_null_explains_deficit` asks whether
the observation lies inside the *null's* interval, discarding the observation's own error —
the larger of the two by ~4×. **Simulated** at the fielded sample sizes on a model drawn
from the null itself, it rejects **64.8%** of the time. Fixed additively:
`wilson_diff_interval` (Newcombe 1998, method 10) plus `coverage_68_vs_null_ci` and
`coverage_68_null_explains_deficit_paired`. The old key keeps its value; its note now says
what it actually tests.

## 8.6 B5 — per-jet rows in the cluster artifact

`METRICS["per_jet"]` in `scripts/make_per_jets_cluster_nb.py`, keyed by `i` = the index
into the jet file, so two budget tiers of the same file pair exactly. Curated rather than
`ROWS` verbatim — a row also carries eight estimators' coordinate arrays and the cluster
object, and no gate reads them. Verified end-to-end by executing the notebook at a reduced
tier: 40 rows, 44 keys each.

Two things landed with it. The artifact name now carries `min_cluster_size` when it differs
from the K-dependent auto value, and `run.cluster_min_cluster_size_effective` records what
`0` resolved to — that resolution *is* B4's confound. And the three budget knobs are
overridable from the environment (`PJC_N_JETS`, `PJC_K_DRAWS`, `PJC_MIN_CLUSTER_SIZE`,
`PJC_SEED`), so a budget arm is a command rather than an edit to a generated notebook.
