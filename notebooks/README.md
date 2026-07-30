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
- **closure.ipynb** — leading-emission Lund distance, multiplicity bias, MAP vs
  plain-RSD vs truth trees (§8 closure).
- **calibration.ipynb** — SBC rank histogram, PIT, coverage (Talts et al.,
  arXiv:1804.06788).
- **posterior_walkthrough.ipynb** — a per-jet posterior: MAP tree + 68% credible
  region + multiplicity distribution.
- **generator_systematic.ipynb** — PYTHIA-trained vs HERWIG-trained MAP/posterior
  spread (§8); the inter-model spread is the dominant systematic.

Generate the underlying metrics with `h2p-rsd-junipr eval <ckpt>` (writes a JSON
metrics record + the CSV/JSONL training curves in the run dir).
