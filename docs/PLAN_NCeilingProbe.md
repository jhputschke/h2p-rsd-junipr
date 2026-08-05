# PLAN — the N-information ceiling probe, the consolidated conclusion, and the spline-head to-do

Status: **proposed** (approved for implementation). Follows directly from
`PLAN_StratifiedMBR.md` §1a–§1c and decides where model-improvement effort goes next.

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
3. One commit per WP; push `stratifiedMBR`.

## Verification

- `python scripts/n_ceiling_probe.py --fast` runs end-to-end in ~1 min; the full run
  produces the table and the JSON artifact.
- Sanity row: the probe's posterior-median baseline reproduces 0.448 on the same test
  population as `per_jet_clusters.json`.
- `pytest tests/` stays green (842 baseline); ruff clean on new/touched files.
