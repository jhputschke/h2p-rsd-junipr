# notebooks/

Validation + physics figures, nbstripped in git (run `bash setup_nbstripout.sh`
once per clone; `.pre-commit-config.yaml` carries the same hook) and rendered in
the docs:

- **lund_rntuple_histograms.ipynb** — the notebook twin of
  [`cpp/apps/hist_lund_rntuple.cpp`](../cpp/apps/hist_lund_rntuple.cpp): reads a
  `jets.root` RNTuple with `uproot` and fills the **same histograms, bin for bin** —
  same edges, same per-jet `weight`, same `x_nsec > 0` gate on the secondary-`k_t`
  columns. Opens on the two **primary Lund planes** (hadron `x` vs parton `y`, shared
  colour scale), then the jet kinematics, the primary multiplicity, the four Lund
  observables pooled, the same four **split by splitting index** (the first two;
  `NSPLIT_SHOW` raises it), and every aux conditioning column. The last section is a
  parity check against the C++ app's output file — currently 13 histograms, worst bin
  difference 0. Runs in seconds on `cpp/test_data/jets_aux.root`; needs no ROOT install.

- **aux_input_ab.ipynb** — the A/B for [`docs/PLAN_Input.md`](../docs/PLAN_Input.md):
  does conditioning on **groomed all-branch** scalars — the pipeline-groomed mass `m_g`,
  the secondary-plane splitting count `n_sec`, and the jet scale — buy anything the
  primary-only hadron sequence cannot supply? Runs on `cpp/test_data/jets_aux.root`
  (the same card and the same 25 000 events as `jets.root`, plus the two new columns, so
  the comparison is paired). Shows why the aux columns are not functions of `x`; verifies
  the `aux_features=[]` off path is byte-identical; then trains `ar_junipr_v3 + gru` with
  and without aux over **three seeds** and reads the held-out NLL delta against the seed
  spread. Breaks the gain down by secondary activity and groomed mass (including the
  `n_x = 0` jets that structurally cannot benefit), checks closure / MBR risk / SBC /
  per-coordinate PITs **stratified in aux bins**, ablates the three features
  individually, and closes against the plan's four exit criteria — including the one
  (generator-B prior spread) that is blocked on `PLAN_UPDATES.md` WP5 and is reported as
  blocked rather than skipped. **Outcome: the gate fails and aux stays opt-in** — the
  −0.029 nat/jet gain is exactly the seed spread, one of three seeds goes the wrong way,
  and the `n_x = 0` jets that *cannot* receive aux "gain" just as much, which is what
  shows the aggregate is noise. The signal that does survive is in the `n_sec = 2–3`
  stratum (−0.100 nats/jet); 82.6 % of this sample has `n_sec = 0`, so there is little
  else to find here. ~35 min on MPS for the 15 trainings; every run is cached, so a
  re-run is instant.

- **calibration_v2_walkthrough.ipynb** — the post-review calibration suite and the new
  model families ([`docs/PLAN_UPDATES.md`](../docs/PLAN_UPDATES.md) WP1–WP4), worked
  end to end on the **real PYTHIA 8.3 data** in `cpp/test_data/jets.root` rather than the
  synthetic generator. Trains `ar_junipr_v3` on it, then walks through: why SBC-on-N
  cannot referee a v2-vs-v3 comparison; per-coordinate PITs (with the U-shape /
  over-confidence signature demonstrated by deliberately narrowing the head); region-
  stratified coverage on the Lund plane; TARP expected coverage on tree-valued
  posteriors, with a deliberately over-confident control so the curve's *sign* is
  readable; the `exact_likelihood` contrast between `cfm` and `diffusion`; the
  multiplicity-support guard on real grooming parameters; and the v4 cross-attention
  A/B. Runs top to bottom on CPU/MPS in a few minutes.

- **inference_demo.ipynb** — standalone end-to-end evaluation: load a `best.ckpt`,
  take a `jets.root` test file **or** generate synthetic matched test data, and show
  posterior + MAP point-estimate performance vs truth (single-jet posterior & Lund
  plane, multiplicity / leading-emission recovery, coordinate marginals, SBC / PIT /
  coverage). Runs with no arguments. Combines the three notebooks below into one.
  §6 compares five multiplicity estimators (MAP, **MAP with the learned per-jet
  floor**, posterior mean, **posterior median**, and **MBR (perturbative Lund)**); with
  the `decode.min_emissions=1` floor the "MAP = 0" mode-collapse fraction now reads ~0%,
  and the learned floor (`LENGTH_FLOOR_QUANTILE`, the `α`-quantile of `P(n|x)`) raises the
  MAP toward the truth to cut the residual under-count while keeping `n=0` at 0% (the
  median is still the recommended count estimator). The **MBR** estimator is mode-free and
  *floor-free* — it picks the drawn tree of least expected perturbative-Lund EMD to the
  posterior, so it never collapses to `n=0` even with no floor; a one-line `MBR_BACKEND`
  toggle (`pot` default, `energyflow` optional) runs the demo without `energyflow`
  installed. See `scripts/probe_map_collapse.py` for the floor/training sweep.

- **lund_distribution_closure.ipynb** — the **population** counterpart to
  `inference_demo.ipynb`. Where `eval/closure.py` and `eval/calibration.py` ask per-jet
  questions, this asks whether the predicted *ensemble* of primary-Lund splittings looks
  like the parton-level ensemble — and whether it looks more like it than **plain RSD**
  (the file's own hadron-level `x_*` branches) already does. Five series on shared axes:
  truth `y`, plain RSD `x`, the **MAP** and **MBR** point estimates, and a
  posterior-predictive draw whose *continuous coordinates are sampled*, not moded — the
  only honest comparator for point estimates, since a per-jet argmax is narrower than
  truth by construction. Figures: the primary Lund plane with ratio-to-truth maps; the
  four coordinate marginals pooled **and split by splitting index**; the angular-ordering
  ladder profiles; the `k_t`-cut multiplicity spectrum `N(k_t > c)`; leading-emission
  kinematics and multiplicity. Each observable is scored with **W1, KS and χ²/ndf**
  (weight-aware throughout; ψ uses circular W1 and Kuiper's V), and the headline is the
  **improvement ratio** `d(ŷ,y)/d(x,y)` — below 1 when the model beat doing nothing.
  Reports the out-of-window, soft-drop-violation and `k_t`-floor-violation fractions up
  front, since the model cannot emit outside the geometry window and its unbounded `ln z`
  head knows nothing about the grooming boundary. Writes `dist_closure_metrics.json` and
  `dist_closure_table.md` beside the checkpoint, which is the runtime harness
  [`docs/PLAN_ProductionAssessment.md`](../docs/PLAN_ProductionAssessment.md) §7/§10 needs
  per pT window. ~4 min for 2000 jets with the `pot` MBR backend; a cost probe cell sizes
  the run before you commit to it. **Read the population caveat under v2 below before
  quoting any number from this one.**

- **lund_distribution_closure_v2.ipynb** — the same study on the population you could
  actually select on data. **Prefer this one; see the note below for why both exist.**
  Adds §5a, the observable v1 structurally could not have: the **empty-tree rate**
  `P(n = 0)` per series, plus a `MAP_ALLOW_EMPTY` control that re-decodes the MAP with the
  length floor lifted. Everything else is identical, and
  `REQUIRE_TRUTH_SPLITTING = True` reproduces v1's population exactly.

- **prod_test_v0.ipynb** — the end-to-end production test of
  [`docs/PLAN_prod_test_v0.md`](../docs/PLAN_prod_test_v0.md), and the only notebook
  here that reports on a genuinely **independent file**: `data/jet_aux_asym_test.root`,
  a different PYTHIA seed from the one the checkpoint trained on, with
  [`scripts/check_disjoint.py`](../scripts/check_disjoint.py) asserting the two streams
  never collided. Four nested jet tiers over one frozen shuffled index list keep every
  comparison paired while bounding a per-jet cost that spans four orders of magnitude.
  What it carries that no other notebook does: the **aux ablation with a seed band**
  (aux on/off × seed 0/1 — a single pair cannot conclude, and the previous A/B failed
  exactly there), the per-term NLL table with a **`10-bin-comparable?` column** (the
  total is a density on the plane and *is* comparable across `n_bins`; `split_ll` alone
  shifts by `2·ln 3`), **split-head occupancy and effective rank** (900 cells behind a
  `Linear(64, 900)`), and a `q(0|x)` **reliability diagram with the Brier
  reliability/resolution decomposition** — the diagnostic SBC/PIT structurally cannot
  provide, since SBC ranks against the sampler's own draws. Both the empty-gate `tau`
  and the `(temperature, tilt)` length recalibration are fitted on the **training
  file's** val split and applied frozen, because `empty_threshold_for_rate` is a
  quantile and would otherwise reproduce its own fitted rate by construction. §7 is a
  deliberate *pointer* to `lund_distribution_closure_v2.ipynb` rather than a second
  implementation of the headline distances, with a staleness guard that refuses to
  quote a `dist_closure_metrics.json` describing a different run.

- **closure.ipynb** — leading-emission Lund distance, multiplicity bias, MAP vs
  plain-RSD vs truth trees (§8 closure).
- **calibration.ipynb** — SBC rank histogram, PIT, coverage (Talts et al.,
  arXiv:1804.06788).
- **posterior_walkthrough.ipynb** — a per-jet posterior: MAP tree + 68% credible
  region + multiplicity distribution.
- **generator_systematic.ipynb** — PYTHIA-trained vs HERWIG-trained MAP/posterior
  spread (§8); the inter-model spread is the dominant systematic.

### Why there are two distribution-closure notebooks

v1 (and `inference_demo.ipynb`) select jets with at least one primary splitting **at both
levels**:

```python
jets = [j for j in jets if len(j["x"][0]) and len(j["y"][0])]
```

The second condition reads the **parton** sequence — the thing being predicted. No analysis
can apply it to data. On `cpp/test_data/jets.root` it discards 8 631 jets, **17.2% of
everything selectable on data**, and they are not a random 17%: every one is a jet whose
correct answer is the *empty tree*, where hadronisation manufactured all the visible
structure from a parton jet with no splitting surviving grooming.

| population | selection | jets | mean `n_x` | mean `n_y` | `x/y` | `P(n_y=0)` |
|---|---|---|---|---|---|---|
| train | none — `LundDataModule` filters nothing | 54 007 | 1.744 | 1.420 | 1.228 | 16.0% |
| **deploy** | `len(x)>0` — **all you can apply on data** (v2) | 50 290 | 1.873 | 1.436 | 1.305 | 17.2% |
| v1 eval | `len(x)>0 and len(y)>0` | 41 659 | 1.973 | 1.733 | 1.139 | 0.0% |

Three consequences, all of which flatter the result:

1. **The rate gap is understated by half.** Plain RSD over-counts primary splittings by
   30% on the deployable population, not the 14% v1 reports.
2. **Training never applied the cut.** `MatchedLundDataset` builds `n_y = 0` items and
   `log_prob` scores them, so the model learned these jets; v1 just never tested it on them.
3. **It silently rigs the MAP-vs-MBR comparison** by removing exactly the jets where MAP
   structurally fails and MBR most clearly wins.

What v2 then exposes on the walkthrough `ar_junipr_v3` checkpoint: the model holds real
information about which jets are empty (mean `q(N=0|x)` = 0.16 on truth-empty jets against
0.08 on the rest, AUC 0.77), and **no point estimator under the default decode can use
it** — both read `P(n̂ = 0) ≈ 0%` against a truth rate of 17.4%, for two unrelated reasons:

- **MAP** — *not* the `decode.min_emissions = 1` floor. Lifting the floor changes nothing,
  because with a multiplicity head the MAP is `argmax q(n|x)`, and the peak lands at 0
  essentially never however much mass sits there. The same mode-vs-distribution effect the
  rest of the notebook is about, one level up — applied to the *length*.
- **MBR** — mode-free and floor-free, so it *could* answer "nothing", but the
  perturbative-Lund EMD charges an imbalance penalty (`mbr_R`) for unmatched weight, and an
  empty cloud is entirely unmatched weight. Its risk is near-maximal, so it is close to the
  worst answer available rather than a cheap one.

Only the posterior draws produce empty trees at all (9.7% against truth's 17.4%). Getting
these jets right needs a decision rule that can express emptiness — thresholding
`q(0|x)` — not a different floor.

**This one column is backend-dependent, and the shape panels are not.** On identical draws
`pot` and `energyflow` both give `P(n̂=0) ≈ 0.2%` and recover 0% of the truth-empty jets,
while `surrogate` gives 57% and 82% — a normalised binned-image χ² does not punish an empty
image the way an EMD with an imbalance term does. Neither is "right"; they are different
risk functions, and this is where they diverge hardest. Quote §5a together with the
`MBR_BACKEND` it was run under.

Relatedly, `beam_search_cells` calls the empty tree "unphysical (a groomed jet has >=1
primary splitting)". At parton level that is false for 17.2% of jets in this file. The
floor is a real fix for MAP length collapse, but its stated premise does not hold for the
target distribution — a decode-layer question for the package, which is why v2 measures the
cost rather than working around it.

**Which to use.** v2 for anything you report or compare against data. v1 remains valid but
narrower — it answers "given a jet with parton-level substructure, is the predicted
substructure right?", which is a real question, just not one an analysis gets to ask. Their
numbers are **not comparable**: different populations, so every distance, ratio and rate
moves.

Generate the underlying metrics with `h2p-rsd-junipr eval <ckpt>` (writes a JSON
metrics record + the CSV/JSONL training curves in the run dir).
