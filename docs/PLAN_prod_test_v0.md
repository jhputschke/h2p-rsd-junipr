# PLAN_prod_test_v0 — one end-to-end production test

*Status: **RUN. Results in [§Results](#results-2026-07-31) below.*** Unlike the other
`PLAN_*.md` files this one describes a **run**, not a code change — keep the status line
current. The four sibling plans it flagged as carrying stale "proposed" headers
(`PLAN_MBR_PerturbativeLund`, `PLAN_MultHead`, `PLAN_NsplitMinCut`, `PLAN_QuantileMinCut`)
have been corrected to state what shipped.

## Context

Every result in this repo so far comes from one 25 000-event PYTHIA file
([`cpp/test_data/jets.root`](../cpp/test_data/) / `jets_aux.root`, 54 007 jets) at
`n_bins: 10`, evaluated on its own 10% val split. Three findings have accumulated that this
setup cannot settle:

1. **The aux conditioning A/B failed, but on the wrong file.**
   [`PLAN_Input.md`](PLAN_Input.md) measured −0.029 nat/jet against a 0.029 seed spread —
   noise. The reason is visible in the same doc: 82.6% of that sample has `x_nsec == 0`, so
   the five secondary-plane features are constant-zero four times in five. The asymmetric
   floor (`SoftDrop:ktFloorSec`) exists precisely to fix that (`⟨x_nsec⟩` 0.25 → 2.21, zero
   fraction 80.6% → 20.5%) **while leaving the x/y sequences bit-for-bit unchanged**. It has
   never been generated.
2. **The leading-emission metric was quantisation-limited.** At `n_bins: 10` a cell is 0.6
   wide and the distances are ~0.6. `n_bins: 30` takes the cell to 0.2 and is the direct fix.
3. **There is no test split.** [`LundDataModule`](../src/h2p_rsd_junipr/data/datamodule.py)
   produces train/val only; an independent file plus the G1 `data`-lifting mechanism is the
   entire story ([`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §2.4), and no
   such file exists.

This plan runs one arm end to end on data built for the question: generate an
asymmetric-floor train/test pair, train the SOTA AR arm at 30 bins, and assess it on the
held-out file with a single notebook. It is a **production test of the pipeline**, not a
model ranking — one arm, one seed.

## Prerequisites — the empty-tree fix is *not* a blocker

Required: the leading-emission estimator work (commit `c45a593`: `medoid_cell`,
`geometric_median`, `dlund_posterior_medoid`, `experiment.closure_continuous`). Steps ii and
iii read those keys directly.

[`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md)'s gate **landed** in `25e9dac`
(`decode.empty_threshold`, `empty_gate` / `empty_threshold_for_rate`, `run_closure`'s
`p_empty_*` keys, §5a's gated row), so §6 below can be written in full. It was never a hard
prerequisite, and the coupling is recorded here because it is the general case — a
decode-layer change never blocks generation or training:

| | needs the fix? |
|---|---|
| Step i — generation | No. Pure C++ run. |
| Step ii — training | No. `decode.empty_threshold` is a *decode-time* knob defaulting to `0.0`; nothing touches the loss or the architecture. `best.ckpt` is byte-identical either way. |
| §6 `p_empty_true`, `q(0\|x)` AUC, reliability curve, measured `p_empty_pred ≈ 0%` | No — all reachable today via `model.length_pmf`. |
| §6 the **gated** row (`tau` fitted, `p_empty_pred` matching truth) | **Yes** — the only item that does. |

Because the gate is decode-only it applies to an already-trained checkpoint: fit `tau` on
the training run's val split and re-run one cell. The expensive step imposes no ordering
constraint. **Recommended order:** start step i immediately — it is the long pole and depends
on nothing.

Measured on the 10-bin walkthrough checkpoint, which is what §6 should reproduce at 30 bins:
`q(0|x)` AUC 0.760, under-confident 1.90×, held-out predicted rate 0.172 against truth 0.159
at recall 0.36 / precision 0.33.

The two are also physically orthogonal: the asymmetric floor leaves the x/y sequences
bit-for-bit unchanged so `P(n_y=0)` is identical to the symmetric file, and `n_bins` is a
*cell* change while emptiness is a *length* property. Neither knob moves the empty-tree
numbers. One consequence: check 7 below (auditing consumers that assume `multiplicity >= 1`)
is only exercisable once the gate can fire, so it belongs to the fix's own verification, not
to this test.

**Out of scope, deliberately:** the PYTHIA-vs-HERWIG generator systematic.
[`PLAN_Input.md`](PLAN_Input.md) says a looser off-spine floor must be validated with it
"not with the NLL alone", and WP5 has no `herwig_driver`. Recorded as the known gap, not
silently skipped.

## Deliverables

| # | Artifact | Note |
|---|---|---|
| 1 | this document | write it first, so the run is reproducible from the repo alone |
| 2 | `data/jet_aux_asym.root`, `data/jet_aux_asym_test.root` | gitignored by `*.root`; not committed |
| 3 | `presets/prod_test_v0.yaml` | new; `presets/` has no production presets today |
| 4 | `runs/prod_test_v0/…/{best.ckpt, metrics.csv, eval_metrics.json, *.png}` | |
| 5 | `notebooks/prod_test_v0.ipynb` + `prod_test_v0_metrics.json` | nbstripped before commit |
| 6 | one-line fix to [`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md):334 | it wrongly calls `jets_aux.root` asymmetric; that file predates the `kt_floor_sec` column |

---

## Step 0 — build and verify the C++ side

```bash
conda activate js_fno                     # ROOT / FastJet / fjcontrib / PYTHIA
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
ctest --test-dir cpp/build                # test_lund_io sweeps kt_floor x kt_floor_sec
```

The asymmetric floor is **already fully implemented** — `GroomParams::secondaryFloor()` in
[`cpp/include/lund_io.hpp`](../cpp/include/lund_io.hpp), the per-node spine/off-spine choice
in `groomRecurse` ([`cpp/src/lund_io.cpp`](../cpp/src/lund_io.cpp)), the `kt_floor_sec`
provenance field in [`cpp/src/lund_writer.cpp`](../cpp/src/lund_writer.cpp), and the card
[`cpp/cards/pp_dijet_asym_floor.cmnd`](../cpp/cards/pp_dijet_asym_floor.cmnd). No C++ work is
needed.

---

## Step i — the two ROOT files

`pythia_driver <nEvents> <out.root> <seed> <card>`; the CLI seed overrides the card's.

```bash
mkdir -p data
./cpp/build/pythia_driver 230000 data/jet_aux_asym.root      1 cpp/cards/pp_dijet_asym_floor.cmnd
./cpp/build/pythia_driver  45000 data/jet_aux_asym_test.root 2 cpp/cards/pp_dijet_asym_floor.cmnd
./cpp/build/read_lund_rntuple data/jet_aux_asym_test.root Jets   # expect kt_floor_sec: 0.2
```

**Sizing.** The reference file is 54 007 jets from **25 000** events — 2.16 jets/event, not
one jet per event. The target here is 9× the current jet count, so the event counts above
deliver ~497k jets train and ~97k test. Scale both if you want a different jet budget; the
per-cell statistics argument below is about jets, not events.

At 30 bins the 900 cells receive ~850k train emissions, i.e. ~940/cell on average — parity
with what 100 cells received at the old file size. The Lund density is peaked, so that
average overstates the sparse corners; §3 of the notebook is what measures the real
occupancy.

Three things this must satisfy, all checkable from `read_lund_rntuple`:

- `kt_floor = 1.0`, `kt_floor_sec = 0.2` — the file really is asymmetric.
- Distinct seeds (1 / 2). **Same card for both** — the support guard reads the stamped
  provenance, and mixing cards makes the test file a different physics sample.
- Both files land in `data/`, which `.gitignore`'s `*.root` already excludes (only
  `cpp/test_data/jets.root` and `jets_aux.root` are negated). Do **not** add negations — a
  500k-jet RNTuple does not belong in git.

---

## Step ii — preset and training

New file **`presets/prod_test_v0.yaml`**. `presets/` currently holds only `ab_v2_v3.yaml`
and `mbr_study.yaml`; the `t*`/`f*` ladder in
[`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §5 was never built, so this is
written from scratch following `mbr_study.yaml`'s shape.

```yaml
# presets/prod_test_v0.yaml — the end-to-end production test arm.
defaults:
  model: ar_junipr_v4        # v3 + decoder cross-attention (the A4 "SOTA AR" arm)
  encoder: lundnet           # returns_sequence=True; the documented v4 pairing
  data: rntuple
  optim: default
  trainer: default
  decode: default
  experiment: closure

geometry:
  n_bins: 30                 # 900 cells; cell 0.6 -> 0.2, half_u/half_v 0.3 -> 0.1

data:
  path: data/jet_aux_asym.root
  # min_val is deliberately left at its default: n_val = max(min_val, len//10), and
  # len//10 ~= 50k dominates any sane min_val here, so setting it would be a no-op.

encoder:
  num_layers: 5
  aux_features: [ln_mg_pt, ln_ptg_pt, ln_pt, abs_eta,
                 nsec, has_sec, ln_kt_sec, ln_kt_sec_sum, sec_depth]

model:
  dec_dim: 64
  dec_layers: 2
  xattn_heads: 4             # must divide dec_dim
  sigma_floor: 0.005         # see note below — half_u/half_v shrank 3x

trainer:
  max_epochs: 60
  batch_size: 256

run_root: runs/prod_test_v0
```

```bash
h2p-rsd-junipr train base=presets/prod_test_v0.yaml
```

**Encoder choice — `lundnet`, with the honest caveat.** It is the pairing
[`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §4 names for A4, and
cross-attention needs `returns_sequence=True`. But: all three encoders set that flag, no
encoder A/B exists in the repo, `lundnet`'s `k: 4` is dead code (the shipped version is a
chain EdgeConv over the primary chain, not the graph net), and the mean hadron sequence here
is 1.78 nodes — a chain graph over two nodes has almost nothing to message-pass. Take
`lundnet` as the documented arm, not as a measured winner. Adding `encoder=gru` and
`encoder=deepsets` arms is +2 trainings and would close
[`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §7's open encoder probe.

**Five things about this preset that are not obvious:**

- **`beam_width` and `topk_cells` are inert here.** With `use_multiplicity_head: true`,
  `map_decode` routes to `_map_decode_fixed_length`
  ([`ar_junipr.py`](../src/h2p_rsd_junipr/models/ar_junipr.py):404) — greedy argmax per step;
  `beam_search_cells` is never called. Do not "raise `topk_cells` for 900 cells"; it would do
  nothing. The MAP's quality at 30 bins rests on the split head and `q(N|x)`.
- **`sigma_floor`.** `half_u/half_v` go 0.3 → 0.1, so the truncated-normal support narrows 3×
  while the default `sigma_floor: 0.01` does not. Lowering it to 0.005 keeps the floor from
  binding; the per-coordinate PIT on `du`/`dv` is what tells you whether it was needed.
- **`mbr_R: 8.485` needs no change.** It is `6√2`, the diagonal of the `[0,6]²` Lund box set
  by the *ranges*, not the bins. `configs/decode/default.yaml`'s comment "scale with
  geometry" invites the opposite conclusion; [`CONFIGURATION.md`](CONFIGURATION.md) §10 is
  the correct one.
- **Early stopping is not wired.** [`train/callbacks.py`](../src/h2p_rsd_junipr/train/callbacks.py)
  defines `EarlyStopping` but `Trainer.fit` never references it and there is no `patience`
  field. `max_epochs: 60` runs 60 epochs; "early stop on val NLL" is delivered by `best.ckpt`
  selection (`val_nll < best_val`) only — model selection, not compute savings. Resume also
  cannot extend a finished run (`config_hash` must match and it continues to the *original*
  `max_epochs`), so pick 60 deliberately.
- **Wall clock.** ~497k jets / batch 256 ≈ 1940 steps/epoch × 60 ≈ 116k steps. Budget
  accordingly and prefer CUDA if available; `select_device()` is automatic with no knob.

---

## Step iii — `notebooks/prod_test_v0.ipynb`

One notebook, run against `data/jet_aux_asym_test.root`, following
[`lund_distribution_closure_v2.ipynb`](../notebooks/lund_distribution_closure_v2.ipynb)'s
house style: cell 0 title, `## 0. Parameters` with a single all-caps constants cell,
everything structural read off the checkpoint snapshot and `model.aux_feature_names`,
`WRITE_ARTIFACTS` → `save_metrics`.

**Four nested jet tiers, one frozen shuffled index list**, so every comparison is paired and
the cost per jet (which spans four orders of magnitude) stays bounded:

| tier | jets | runs |
|---|---|---|
| `POP` | all ~97k | test NLL, `q(0\|x)`, cell occupancy, marginals, provenance — batched forward passes, chunked at `bs=256` |
| `PIT` | 5 000 × 4 disjoint chunks | `coordinate_pits` — **it collates every jet into one padded batch** ([`calibration.py`](../src/h2p_rsd_junipr/eval/calibration.py):116); at 97k that is multiple GB and will OOM. The 4 chunks double as an MC-error estimate |
| `SAMP` | 2 000 | `run_calibration`, `run_closure(continuous=False)`, `collect()` |
| `HEAVY` | 300 | `run_tarp`, `run_closure(point_estimator="mbr")`, `run_closure(continuous=True)` |

Three ordering constraints, each of which silently corrupts output if ignored:

- **`plot_calibration` must be the last figure call.**
  [`eval/report.py`](../src/h2p_rsd_junipr/eval/report.py):57 does `matplotlib.use("Agg")`
  inside it; in a live kernel that switches the backend for the rest of the session and every
  later `plt.show()` renders blank.
- **Re-seed (`seed_everything(SEED)`) at the top of each sampling section.** `sample` and
  `sample_coordinates` ride the *global* torch RNG with no `generator` argument, so a cell
  re-run in isolation gives different numbers than the same cell run in order.
- **Write artifacts to `CKPT.parent / "prod_test_v0"`, not `CKPT.parent`.**
  `plot_calibration` writes fixed filenames that would clobber what `h2p-rsd-junipr eval`
  wrote.

Import `collect()` from [`scripts/leading_estimators.py`](../scripts/leading_estimators.py)
via `importlib` rather than shelling out — its `main()` builds `LundDataModule` from the
*checkpoint's* `cfg.data.path` and would silently score the training file's val split.
`collect()` itself is sample-agnostic.

| § | Section | Reuses |
|---|---|---|
| 0 | Parameters | `CKPT_PATH`, `ROOT_PATH`, `N_JETS`, `K_DRAWS`, `SEED`, `DEVICE="cpu"`, `MBR_BACKEND`, `WRITE_ARTIFACTS` |
| 1 | Load model + test file; **assert provenance** `kt_floor_sec != kt_floor`, geometry `n_bins==30`, aux names match the checkpoint | `load_for_inference`, `build_model`, `load_rntuple`, `MatchedLundDataset(jets, geom, model.aux_feature_names)` |
| 2 | Population at 30 bins: multiplicity, `P(n_y=0)`, aux-column distributions, **train-vs-test agreement** | `check_multiplicity_support(..., strict=False)`, `multiplicity_stats` |
| 3 | **Split-head occupancy** — how many of 900 cells the data and the model actually populate | `geom.seq_cells`, `model.sample_batch` |
| 4 | Calibration: SBC/PIT, per-coordinate PITs incl. `by_emission_index`, region strata, TARP | `run_calibration(..., pit_coords=True, stratify_regions=True, tarp=True)`, `plot_calibration` |
| 5 | Closure + leading emission vs plain RSD — **medoid**, continuous row, `ln kt` thirds | `run_closure(..., continuous=True)`, then `collect()` |
| 6 | The empty tree: `p_empty_true` vs `p_empty_pred`, rate-matched `tau`, `q(0\|x)` AUC + **reliability curve** | `empty_threshold_for_rate`, `empty_gate`, `model.length_pmf` |
| 7 | Distribution closure — **run `lund_distribution_closure_v2.ipynb` pointed at the test file**, do not re-implement W1/KS/χ² | that notebook, unchanged |
| 8 | Support + validity: out-of-window, soft-drop-violation, `k_t`-floor-violation fractions | as in closure_v2 §5 |
| 9 | Summary table + `save_metrics(... "prod_test_v0_metrics.json")` | `save_metrics` |

§7 is deliberately a pointer, not a re-implementation: that notebook already computes the
improvement ratio with bootstrap noise floors and scoreability gating, and duplicating it
would create a second definition of the headline number — this repo has already been burned
by two closure populations drifting apart. Two constants to change there: `ROOT_PATH`
(hard-defaulted to `cpp/test_data/jets.root`) and `CKPT_PATH`. Its `PLANE_NB = 30` is
documented as "a multiple of `geometry.n_bins`"; at `n_bins: 30` that becomes exactly 1× and
the Lund plane shows the model's own cell granularity, which is what you want here — raise it
to 60 only for sub-cell resolution. Guard against staleness: refuse to quote
`dist_closure_metrics.json` if its `data.path` / `checkpoint` / `n_eval_jets` disagree with
this notebook's.

The CLI counterpart, which the notebook should print so the run is reproducible without it:

```bash
h2p-rsd-junipr eval runs/prod_test_v0/<id>/best.ckpt \
    data=rntuple data.path=data/jet_aux_asym_test.root \
    experiment.pit_coords=true experiment.stratify_regions=true \
    experiment.tarp=true experiment.closure_continuous=true
```

Naming `data` lifts it over the snapshot and evaluates the **whole** file rather than a 10%
split ([`cli.py`](../src/h2p_rsd_junipr/cli.py):204). `geometry` and `encoder` are
deliberately not liftable.

---

## Checks to add that none of the three steps otherwise covers

1. **Which NLL terms survive the geometry change.** It is tempting to declare a 30-bin NLL
   simply incomparable to the 10-bin `4.61`. That is too strong.
   [`distributions.py`](../src/h2p_rsd_junipr/distributions.py)`::trunc_normal_logpdf` states
   the position:

   > "Subtracting the in-interval mass makes the within-cell offset a proper density, so
   > cell-prob × offset-density integrates to a proper density over (ln 1/ΔR, ln kt)."

   So with `continuous_coords: true` the **total** NLL is a density on the plane and *is*
   dimensionally commensurable across `n_bins`. What is not:
   - `split_ll` **alone** — shifts by `2·ln(30/10) = 2.197` nat/emission;
   - any `ar_junipr_v1` NLL (`continuous_coords: false`, purely discrete) — same shift;
   - any run with `cell_label_smoothing > 0` (verify the snapshot has the 0.0 default).

   The real confound on the comparable total is that a finer grid is a *strictly richer
   density class*, so a lower 30-bin total is evidence of better resolution, not of a better
   conditional. Emit a per-term table with a `10-bin-comparable?` column rather than a
   blanket warning.

2. **The aux ablation — the actual scientific question.** The whole point of the asymmetric
   floor is that the secondary-plane features stop being constant-zero. Train a second arm
   with `encoder.aux_features=[]` and compare held-out NLL, stratified by `n_sec` (the
   symmetric-file A/B's only surviving signal was the `n_sec = 2–3` stratum at −0.100 nat).
   Without this arm the file's reason for existing goes unmeasured.

   **Report the `nx == 0` stratum separately** — this is the control that exposed the last
   A/B as noise. Aux rides as *constant per-node columns of `xf`*, so a jet with an empty
   groomed hadron tree receives no aux signal at all; `LundDataModule._report_aux_coverage`
   already prints that fraction (~6.9% of the reference sample). If those jets "gain" as much
   as the rest, the aggregate delta is seed noise, whatever its sign.

   **The ablation needs a seed band or it cannot conclude.** The previous A/B failed because
   −0.029 nat *was* the 0.029 seed spread; a single aux-on/aux-off pair here reproduces that
   ambiguity exactly. Minimum honest configuration is two seeds of at least one arm
   (`trainer.seed=0,1`) to establish the band — 3 trainings rather than 2, 4 for a clean
   two-by-two. [`eval/systematics.py`](../src/h2p_rsd_junipr/eval/systematics.py)`::generator_spread(model_a, model_b, val_ds, ...)`
   takes two *models*, not two files, so it computes the per-jet spread between seeds
   directly. **If the budget allows only one training, drop the ablation entirely** and
   report the arm on its own — an underpowered A/B is worse than none, because it will be
   quoted.

   A cheaper inference-only substitute, valid for a weaker claim: permutation importance over
   the 9 aux columns on the trained model (permute each column across jets, measure the NLL
   rise). That measures *use*, not *value*, and cannot re-open the adoption verdict.

3. **`q(0|x)` reliability.** [`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md) F5
   measures it under-confident by ~2× (`E[q(0|x)] = 9.2%` vs true 18.2%) and notes SBC/PIT
   **do not catch it**, because SBC ranks against the sampler's own draws. A reliability
   diagram on `P(N=0)`, plus a Brier score with its reliability/resolution decomposition, is
   the missing diagnostic — the one scalar that separates "the ranking is good" from "the
   scale is wrong".

   **Recalibrate the head first, and fit both on the training file's val split.**
   `decode.length_temperature` + `decode.length_tilt` (`fit_length_recalibration`) landed
   with the gate. On the 10-bin walkthrough they take mean `q(0|x)` from 0.085 to 0.143
   against a truth 0.161, the NLL of `N` from 1.2133 to 1.1810, and the `n=0`
   empirical/predicted ratio from 1.90 to 1.13. **A scalar temperature alone cannot do
   this** — it is symmetric about the mode, so it pulls `q(0|x)` *down* toward `1/26`; the
   measured error is a monotone ramp across `n` and needs the tilt. §6 must report the
   uncalibrated numbers too, or the recalibration hides the defect it corrects.

   Two things to re-measure at 30 bins rather than carry over: the fitted `(T, tilt)` are
   sample- and geometry-dependent, and the ramp itself may differ once the split head has
   900 classes. Note the recalibration also moves the **posterior series** in §7, since it
   reaches `sample` — so `dist_closure_metrics.json` must record `(T, tilt)` or its empty
   rate is unattributable.

   **Fit `tau` on the training file's val split, then apply it frozen to the test file.**
   `empty_threshold_for_rate` is a quantile of `q(0|x)` and reproduces its fitted rate by
   construction, so fitting and reporting on the same jets makes §6 circular — it would
   measure the quantile function, not the model. This is the one place the empty-tree plan
   leaves the protocol implicit, and it is exactly what an independent test file is for.
   `tau` is sample-dependent, so re-fit per pT window if
   [`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §7's windows are swept.

4. **Train/test agreement.** The two files differ only by seed, so any disagreement in the
   multiplicity marginal or the Lund plane is a generation bug, not physics. Report it as the
   same-generator different-seed **statistical noise floor** against which any future
   generator spread must be read — that is the substitute for the blocked WP5 systematic.

5. **`x_mg` / `x_ptg` are redefined by the asymmetric floor.**
   [`PLAN_Input.md`](PLAN_Input.md) trap 1: under `ktFloorSec` they are the complement of a
   tree the model never sees. `ln_mg_pt` and `ln_ptg_pt` are still legitimate conditioning,
   but they do not mean what they mean in the symmetric file — do not compare their learned
   effect across the two.

6. **`_lund_image(pts, w, geom, nb=10)`** in
   [`inference/mbr.py`](../src/h2p_rsd_junipr/inference/mbr.py):204 hard-codes a 10×10 image
   and does not follow `geom.n_bins`. Harmless at 10 bins, silently stale at 30 — the
   surrogate would bin at 0.6 while the model decides at 0.2. Affects only
   `mbr_backend="surrogate"`. Either wire it to `geom.n_bins` or document that the surrogate
   is fixed-resolution.

7. **Empty-tree consumers.** [`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md) warns
   that anything assuming `multiplicity >= 1` must tolerate the empty tree once the gate can
   fire. `leading_emission_cell` already returns `None`; audit the rest before enabling `tau`.

8. **Quote `MBR_BACKEND` with §6.** The empty-tree column is the one observable that swings
   with the backend (`pot` ≈ 0.2%, `surrogate` ≈ 57%). No other panel does.

9. **`run_closure`'s own population is truth-selected — the bug v2 exists to condemn, still
   live in the library.** [`closure.py`](../src/h2p_rsd_junipr/eval/closure.py):171 does
   `if ly is None or not lead: continue`, so **every `dlund_*` number is conditioned on the
   truth having ≥1 emission**, and `eval_metrics.json` records no kept-jet count. Report the
   kept fraction and label the verdict `p(leading | n_y > 0)`. This also means
   **`p_empty_true` must be validated**: if the empty-tree plan's edit lands *after* that
   `continue`, the key is identically 0 by construction. Compute the truth empty rate
   independently in §6 and assert the two agree — a reading of exactly `0.0` is the tell.

10. **The 900-way head is rank-limited by construction.**
    `split_head = _mlp(dec_dim+ctx_dim=128, 64, n_cells=900, 2)` ends in `Linear(64, 900)`,
    so the logit map has **rank ≤ 64 over a 900-dim output** — a constraint with no analogue
    at 100 cells. Report the effective rank of that weight, plus how many of the 900 cells
    the posterior ever emits versus how many truth occupies. Emitting 300 where truth
    occupies 420 is under-covering the support, and it argues for widening `dec_dim` or
    adding a `split_head` layer rather than for more data.

11. **`cont_temperature` never reaches sampling.**
    `ARJunipr.sample_batch(xf, nx, n_samples, max_emissions=25)` takes no `cont_temperature`
    and uses its own signature default for `max_emissions`, so every `run_closure` /
    `run_calibration` number is drawn at `T=1` and capped at 25 whatever `decode` says.
    Record `beam_width`, `topk_cells` and `cont_temperature` in the artifact under an explicit
    `decode_inert` list, or the JSON will faithfully report knobs that did nothing.

12. **Uncertainty on the coverage numbers.** `coverage_68` and its region strata are binomial
    proportions on ~300 jets (per region, sometimes ~40), and `plot_calibration` draws a bare
    0.68 line. [`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §9.4 ("no Lund
    quadrant fails the band") is a coin flip without Wilson intervals, per-region counts, and
    a stated minimum-n below which a region is not scored. Related: `cell_region` splits at
    `u = 3.0` while `R = 0.4` makes `u < ln(1/R) ≈ 0.92` unreachable, so a `narrow_*` quadrant
    may be structurally near-empty. Quote `tarp_max_dev` against its null floor
    (`≈1.36/√n_jets` ≈ 0.078 at 300 jets) and `sbc_chi2_uniform` against the χ²(9) 95% point
    (16.9), neither of which anything does today.

13. **`eval/` ignores the per-jet `weight`.** `run_closure` and `run_calibration` average
    unweighted over `val_ds`, while
    [`lund_distribution_closure_v2.ipynb`](../notebooks/lund_distribution_closure_v2.ipynb)
    weights everything. Print `np.unique(weights).size` and `n_eff`; if the weights are
    non-trivial, every `eval/` number is an unweighted average of a weighted sample and must
    be labelled so.

14. **Carry the decode-free comparator everywhere.** Put the posterior-predictive `P(n̂=0)`,
    multiplicity and leading position in every table beside MAP/MBR, so a reviewer can
    separate a model failure from a decode ceiling — the central lesson of closure v2's
    "Reading the results".

---

## Results (2026-07-31)

Executed as written, with one deliberate widening: the aux ablation of check 2 was run as
the **full two-by-two** (aux on/off × `trainer.seed` 0/1, four 60-epoch trainings) rather
than dropped. At ~30 s/epoch on one GB10 the whole grid cost about two hours, so the
"budget allows only one training" escape hatch never applied. Two seeds turned out to be
the *minimum* rather than the sufficient configuration — see the ablation section for why
the band a two-seed grid buys is too loose to be the load-bearing statistic.

**Inputs.** `pythia_driver` produced 495 071 train jets from 230 000 events (seed 1, 3 min)
and 97 018 test jets from 45 000 events (seed 2, 36 s), both from
`pp_dijet_asym_floor.cmnd`. `read_lund_rntuple` confirms `kt_floor = 1.0`,
`kt_floor_sec = 0.2` on both. The new
[`scripts/check_disjoint.py`](../scripts/check_disjoint.py) found **0 shared jets of
20 000 compared** and identical provenance tuples: the seeds did not collide.

On that script — the plan's "hash the (x, y) buffers" is necessary but not sufficient
here. The mean groomed sequence is ~1.8 nodes, so hashing the sequences alone leaves only
**7%** of jets identifying (1 349 of 20 000 at a ≥3-emission bar); every empty-tree jet
hashes like every other. The script therefore compares two sets and reports both: `full`
(sequences **plus** the jet four-vector, covering every jet read) and `seq` (the plan's
definition, restricted to long jets). Both are empty.

**The asymmetric floor did what it was built to do.** Over all 97 018 test jets
`⟨x_nsec⟩ = 2.10` with a **21.9%** zero fraction, against 0.25 and 82.6% on the symmetric
reference. The five secondary-plane features are no longer constant-zero four times in
five.

**Train/test agreement — the substitute for the blocked generator systematic.** The two
files differ only by seed, so their disagreement is the same-generator noise floor. Every
marginal agrees to better than 1.5% relative: `⟨n_y⟩` +0.08%, `P(n_y = 0)` +0.11%,
`⟨n_x⟩` −0.34%, `P(n_x = 0)` +1.47%, `⟨x_nsec⟩` −0.50%, `⟨jet p_T⟩` −0.33%. Any future
PYTHIA-vs-HERWIG spread is only a finding where it exceeds these.

**The aux ablation — the headline scientific result.** Held-out on the full 97 018-jet
test file, four arms at 60 epochs each:

| arm | aux | seed | held-out NLL/jet | best epoch |
|---|---|---|---|---|
| `aux_s0` (the headline arm) | on | 0 | **4.0864** | 49 |
| `aux_s1` | on | 1 | 4.1056 | 47 |
| `noaux_s0` | off | 0 | 4.1391 | 45 |
| `noaux_s1` | off | 1 | 4.1687 | 50 |

aux ON − aux OFF = **−0.0579 ± 0.0071 nat/jet** (paired SEM over 97 018 jets); seed band
(max within-arm spread) **0.0296**, so `|delta| / band = 1.96`.

**Read that ratio carefully — this is where the previous A/B went wrong, and it is easy to
repeat.** The delta is well determined: the paired SEM bounds it at ±0.007. The *band* is
not: it is `max` over two arms of a **two-seed range**, a statistic with roughly 60%
relative uncertainty and no error bar of its own. A third seed could move it by half its
size in either direction, and a factor of 1.96 is no margin at all against something that
loose. An earlier pass of this same comparison on a 40 000-jet subsample — noise only, the
same four checkpoints — returned band 0.0575 and delta −0.0560, i.e. **the opposite
verdict**, purely because the per-jet NLL has an SD near 5 and the band is a difference of
differences. The notebook now evaluates the ablation on the whole file for exactly this
reason, and prints an explicit caution whenever the ratio is under 3 — as it does here.

**The control is what actually carries the result, and it needs no band at all.** Aux
rides as *constant per-node columns of `xf`*, so a jet with an empty groomed hadron tree
(`nx == 0`) receives no aux signal whatsoever — a gain there is arithmetically impossible
to attribute to aux:

| stratum | jets | aux on | aux off | delta |
|---|---|---|---|---|
| **`nx == 0` (control)** | 6 856 | 4.7867 | 4.7732 | **+0.0135** |
| `nx > 0` | 90 162 | 4.0435 | 4.1068 | **−0.0633** |
| `n_sec == 0` | 21 241 | 3.5507 | 3.6096 | −0.0589 |
| `n_sec = 1` | 25 445 | 3.4647 | 3.5249 | −0.0602 |
| `n_sec = 2–3` | 31 128 | 4.2498 | 4.3033 | −0.0534 |
| `n_sec ≥ 4` | 19 204 | 5.2862 | 5.3472 | −0.0610 |

The control moves the **wrong way** (+0.0135, aux-on marginally worse) while every stratum
that can carry aux gains ~0.06 — a 0.077 separation, more than 2.5× the seed band, between
two groups of the same jets under the same four checkpoints. That is not something seed
spread produces: a seed effect moves both. So the reading is **aux conditioning helps** —
the finding the exercise existed to obtain, and one that could not have been obtained on
`cpp/test_data/jets_aux.root`, which (deliverable 6) is not an asymmetric-floor file at
all. But it rests on the control and a tight paired SEM, **not** on clearing a two-seed
band; a third seed per arm is what would make the band itself quotable.

**What this ablation does *not* isolate.** The preset's `aux_features` is nine columns of
three different kinds, and the A/B measures all nine at once:

| columns | kind | defined for |
|---|---|---|
| `nsec`, `has_sec`, `ln_kt_sec`, `ln_kt_sec_sum`, `sec_depth` | secondary plane — what `ktFloorSec` unlocks | non-degenerate only when `n_sec > 0` |
| `ln_mg_pt`, `ln_ptg_pt` | groomed mass / momentum — **also redefined by the floor** (check 5) | every jet with a node |
| `ln_pt`, `abs_eta` | jet kinematics, floor-independent | every jet with a node |

So the `n_sec == 0` stratum — 21 241 jets with no secondary plane at all — still gains
0.0588, because four of the nine columns are defined for it regardless. The honest claim
is therefore **"the nine-column aux set helps"**, not "the secondary-plane features help";
and the gain being flat across `n_sec` (−0.054 to −0.065, with none of the symmetric
file's `n_sec = 2–3` peak) is consistent with a large share coming from the always-defined
four.

Isolating the part `SoftDrop:ktFloorSec` was actually built for needs a third arm, and the
right control is **`[ln_pt, abs_eta]`** — *not* all four non-secondary columns, since
`ln_mg_pt`/`ln_ptg_pt` are themselves complements of the off-spine tree and carry floor
information of their own. That is +2 trainings for a two-seed band (~1 h at this scale),
and together with the third seed the band wants, it is the obvious next run.

**Which NLL numbers survive the geometry change** (check 1). `nll_terms` now returns the
decomposition, so the report says which term moved rather than warning about all of them:

| term | value | unit | 10-bin-comparable? |
|---|---|---|---|
| total | 4.0864 | per jet | **yes** — a density on the (ln 1/ΔR, ln k_t) plane |
| length `−ln q(N\|x)` | 1.1271 | per jet | **yes** — references no cell grid |
| split `−ln q(cell)` | 3.8982 | per emission | **no** — shifts by `2·ln 3 = 2.197` |
| coord `−ln p(du,dv,ln z,ψ)` | −1.8079 | per emission | **no** — pays that shift back |
| split + coord | 2.0903 | per emission | **yes** — the product is a density |

over 137 353 emissions in 97 018 jets (1.416 per jet).

`cell_label_smoothing` is 0.0, so the split term is a genuine log-probability.
[`tests/test_nll_terms.py`](../tests/test_nll_terms.py) pins the `2·ln 3` shift and the
invariance of the sum on an untrained model, where both are exactly computable.
`per_jet_nll` is now `-(length + split + coord)` over that decomposition rather than an
independent expression, so the parts cannot drift from the whole;
`scripts/verify_parity.py` still reports **max |Δ| = 0.0** against the original v2 script.

**Acceptance: FAILED on the pooled criterion — but see the stratification below, which
says the pooled number is the wrong summary.** From the notebook's tiers —
cell level on 2 000 jets (1 667 with a truth leading emission, i.e. an 83.4% kept
fraction, so every `dlund_*` is `p(leading | n_y > 0)` per check 9), off-grid on 300:

| estimator | cell (2 000 jets) | off-grid (300 jets) |
|---|---|---|
| identity(x) | 0.651 | 0.646 |
| posterior mode | 0.799 | 0.786 |
| **medoid / geo-median** | **0.696** | **0.670** |

Ratios **1.069** and **1.037** — the arm **loses to identity**, and the two estimators
**agree in sign**, which per the amended §9 rules out "quantisation-limited". The CLI's
300-jet run agrees on the verdict (1.112 / 1.111) but not on the size, which is the
sample-size lesson again: quote the 2 000-jet row.

The 30-bin geometry is visibly doing its job — cell and off-grid now differ by 0.026 where
the plan expected the 0.6-wide cell to dominate the score entirely.
`lund_distribution_closure_v2.ipynb` on the same file concurs from the population side:
the decode-free posterior series is a wash (W1 gmean ratio 0.977, 7/14 wins) while MAP and
MBR are 2.9× and 1.9× **worse** than plain RSD.

**But the failure is not uniform, and this is the run's most actionable result.**
Stratified by the truth's leading `ln k_t` (thirds, 556/555/556 jets):

| truth leading `ln k_t` | jets | identity | medoid | medoid / identity |
|---|---|---|---|---|
| **[0.00, 0.79] — soft** | 556 | 0.709 | 1.078 | **1.520** |
| [0.79, 1.53] — middle | 555 | 0.656 | 0.523 | **0.798** |
| [1.53, 3.79] — hard | 556 | 0.598 | 0.477 | **0.798** |

The model **beats identity by 20% on two thirds of the data** and loses by 52% on the
soft third — and that one third is enough to drag the pooled ratio to 1.064. So "the arm
loses to identity" is true but badly incomplete: the arm is genuinely better than any
function of `x` over most of the Lund plane and collapses in the soft/wide-angle corner.

That is a **localized under-conditioning** finding, and three independent measurements now
point at the same corner: the soft `ln k_t` third here, the two failing coverage quadrants
(`wide_soft` 0.48, `narrow_soft` 0.17 — both `*_soft`), and `narrow_soft`'s SBC χ² of 126
against a 16.9 reference. A calibrated, fully conditional posterior beats any function of
`x` everywhere, so losing there is a real defect rather than a decode artefact — and it is
a far more specific target for the next iteration than "improve the model".

**Calibration is over-confident, and now says so with its error bars** (check 12). Every
statistic is quoted against its null for the first time:

- leading-cell 68% coverage **0.46**, 95% Wilson **[0.43, 0.48]** on 1 667 jets — 0.68 is
  far outside, so this is a real failure, not a small-sample wobble;
- `sbc_chi2_uniform` **359.4** against the χ²(9) 95% point **16.90** — decisively
  non-uniform;
- `tarp_max_dev` **0.097** against the null floor `1.36/√300 = 0.079` — above it, and the
  sign says over-confident (ECP(0.68) = 0.591);
- per-coordinate PIT over 4 disjoint 5 000-jet chunks: `ln z` KS **0.047 ± 0.004** against
  a 0.016 critical value is the one failing coordinate; `du` (0.015 ± 0.005),
  `dv` (0.012 ± 0.003) and `ψ` (0.013 ± 0.003) all pass.

That last line answers a preset question directly: **`sigma_floor: 0.005` was the right
call.** `du`/`dv` are the within-cell offsets whose support narrowed 3× with `n_bins`, and
a binding floor shows up as a U-shaped, over-confident PIT. They are the two
best-calibrated coordinates. (The CLI's 300-jet run flagged `dv` as miscalibrated at
KS 0.078; the 20 000-jet chunked measurement says 0.012. Same lesson as the seed band —
the small sample was the problem, not the model.)

`by_emission_index` shows the exposure-bias signature plainly: KS on `ln z` runs
0.052 → 0.049 → 0.050 → 0.143 → 0.229 across `t = 0…4`, i.e. calibration degrades on the
model's own generated prefix — though the last two bins hold 97 and 9 emissions, so read
them as a direction, not a measurement.

Region stratification is where the interval machinery earns itself. `wide_hard` reads
coverage 0.45 on **22 jets**, interval [0.27, 0.65] — wide enough to be consistent with
almost anything, so it is reported with `scored: false` and drawn hollow rather than
counted. The two scoreable quadrants both fail, and they fail *differently*:
`wide_soft` 0.48 [0.45, 0.50] on 1 533 jets, `narrow_soft` **0.17 [0.11, 0.25]** on 112.
The collinear corner is far worse than the average, which the pooled 0.46 hides entirely —
and it is the same corner where the `ln k_t`-stratified leading-emission ratio is worst.
[`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §9.4's "no Lund quadrant
fails the band" is now answerable, and the answer is that both scoreable quadrants do.

**The empty tree: the ranking is fine, the scale was wrong, and only the tilt fixes it**
(check 3). `q(0|x)` AUC **0.724** — the model separates the two classes perfectly well.
What was broken is the *scale*, which SBC/PIT structurally cannot see because SBC ranks
against the sampler's own draws. The Brier decomposition is what exposes it, and the
recalibration — `(T, tilt)` fitted on the **training file's** val split (20 000 jets) and
applied frozen to the test file — is what repairs it:

| variant | mean `q(0\|x)` | truth | emp/pred | NLL of N | AUC | Brier | **reliability** |
|---|---|---|---|---|---|---|---|
| uncalibrated | 0.0518 | 0.1605 | **3.10×** | 1.3361 | 0.724 | 0.1400 | 0.0170 |
| temperature only (`T = 1.340`) | 0.0716 | 0.1605 | 2.24× | 1.3165 | 0.725 | 0.1348 | 0.0119 |
| **temperature + tilt** (`T = 1.372`, `tilt = −0.511`) | 0.1497 | 0.1605 | **1.07×** | 1.2217 | 0.720 | 0.1251 | **0.0018** |

This confirms the plan's claim exactly: **a scalar temperature cannot close the gap.** It
is symmetric about the mode, so it pulls `q(0|x)` down toward `1/(max_emissions+1)`; it
recovers only a third of the deficit (3.10× → 2.24×) while the tilt takes it to 1.07×. The
reliability term falls 9× while the AUC does not move — recalibration fixes the scale, not
the ranking, which is precisely what the decomposition said was broken. The 10-bin
walkthrough measured 1.90× under-confidence; at 30 bins on an independent file it is
3.10×, so the numbers were right to re-measure rather than carry over.

The gate transfers. `tau = 0.1034`, fitted to reproduce the training-val empty rate
(0.1663) and applied **frozen**, yields `p_empty_pred = 0.164` against a test truth of
**0.161** — a rate error of 2%, at recall 0.283 / precision 0.277, MBR backend `pot`.
Because `empty_threshold_for_rate` is a quantile of `q(0|x)` it would reproduce its own
fitted rate by construction on the jets it was fitted to; that this holds on a *different
file* is the measurement, and it is the one thing an independent test file was needed for.

**900 cells were affordable; the head's rank is the thing to watch** (check 10). Truth
occupies **275 of 900** cells (30.6%) over 137 353 emissions; the posterior emits 273,
covers 255 of the truth-occupied ones and **misses 20**, while emitting 18 the truth never
visits. So 30 bins did not starve the head — but the support is under-covered at the
margin, and `split_head` ends in `Linear(64, 900)`, a rank bound of 64 with no analogue at
100 cells. Its measured effective rank (spectral entropy) is **37.9 of 64**, with 40
singular values holding 99% of the energy. The bound is not yet saturated, so the
under-coverage argues for more capacity in the *decoder* before more cells — not for more
data.

**§7 stayed a pointer, and its staleness guard did real work.** The notebook does not
re-implement the population-level W1/KS/χ² — it accepts
`dist_closure_metrics.json` only after checking that its `data.path` names the test file
and its `checkpoint` sits in *this* run directory (not merely that the basename is
`best.ckpt`). It also compares the artifact's recorded `(length_temperature,
length_tilt) = (1.0, 0.0)` against §6's fitted `(1.372, −0.511)` and says so.

That comparison exposed a gap the plan had assumed away. The plan requires
`dist_closure_metrics.json` to record `(T, tilt)` "or its empty rate is unattributable" —
but **closure_v2 had no such knob**: it read them from the checkpoint snapshot, so it
could only ever record the identity, however they had been fitted. It now takes
`LENGTH_TEMPERATURE` / `LENGTH_TILT` constants (`None` → whatever the snapshot carries, so
the default path is unchanged), applies them to the model the way `cmd_eval` applies a
lifted override, and writes them into `DECODE` so they reach the artifact.

The size of what that was hiding, measured on 400 test jets:

| | mean `q(0\|x)` | **sampled `P(n̂ = 0)`** |
|---|---|---|
| identity `(1.0, 0.0)` — what the artifact recorded | 0.0521 | **0.0465** |
| recalibrated `(1.372, −0.511)` | 0.1509 | **0.1559** |
| truth | — | 0.1675 |

The recalibration reaches `sample`, not just `length_pmf`, so closure_v2's posterior empty
rate moves by a factor of **3.4** — from badly under-producing empty trees to nearly
matching truth. That is one of its headline observables, and until now nothing in the
artifact said which of the two numbers it was.

§7 therefore prints the **complete** set of constants closure_v2 needs, each tagged
`CHANGE` or `already right` with its provenance, rather than the two the plan named. The
other one the plan missed: `EMPTY_THRESHOLD` defaults to `None`, which rate-matches `tau`
on the very sample it reports on — circular in exactly the way §6 is careful not to be.
The frozen `tau = 0.10344` is what makes it a measurement.

**Support and validity** (§8). Truth and identity(x) both sit at exactly 0 out-of-window,
0 soft-drop violation, 0 `k_t`-floor violation over 137 353 and 168 521 emissions — the
generator enforces the fiducial window, so the floor is a hard zero rather than a small
number. Against that floor the posterior's **0.61% soft-drop violation rate** is entirely
the model's: it occasionally places an emission below the `z_cut` boundary the training
data never crosses. Small, but it is a support error rather than a calibration error, and
nothing else in the suite would have surfaced it.

**What did not need re-litigating.** `p_empty_true` reads 0.167 rather than exactly 0.0,
confirming the empty-tree accounting still sits *before* `run_closure`'s leading-emission
`continue` (check 9); the notebook asserts this against an independently computed truth
rate. The support guard is silent: over the whole test file `max N = 6` against
`model.max_emissions = 25`, so `P(N > support) = 0` exactly. Weights are constant
(`n_eff = n = 97 018`), so `eval/`'s unweighted averaging is exact here and check 13's
caveat does not bite — but the notebook prints it either way.

**Check 7 was in scope after all, and it found two things.** The plan defers the
empty-tree consumer audit on the grounds that it "is only exercisable once the gate can
fire" — but §6 fits a `tau` and fires it, so the audit was run here. Driving every
consumer with a `tau` that fires on every jet turned up two that could not answer
"nothing":

- **`print_point_estimate` bypassed the gate.** It called `map_estimate` directly, which
  structurally cannot return the empty tree. With the gate on it therefore printed a
  non-empty MAP for the very jets `run_closure`'s `p_empty_pred` had just counted as
  empty — one `eval`, two contradictory answers. Now routed through `map_or_mbr`.
- **`generator_spread` took no decode at all.** It called `map_estimate()` with no
  arguments, so the quantity the module bills as "the dominant systematic" was measured
  under signature defaults rather than the run's configured decode, and could never see
  the empty tree. It now accepts `decode=` (applied identically to both models, since a
  spread between differently-decoded models is meaningless), records
  `point_estimator` / `empty_threshold` / `min_emissions` in its output, and returns
  **NaN** rather than 0 for the leading-emission spread when neither model has a leading
  emission to compare — a 0 there would read as perfect agreement.

`serving.predict` and `run_closure` already handled it correctly.
[`tests/test_empty_tree_gate.py`](../tests/test_empty_tree_gate.py) now pins all four.

**A trap the notebook hit, worth recording.** §6 has to rebuild the *training* run's val
split to fit `tau` and `(T, tilt)` on it, which means handing `LundDataModule` the
checkpoint's own `cfg.data.path`. That path is stored **repo-relative**, and a notebook
kernel runs in `notebooks/` — so `load_rntuple` missed the file. It does not raise when
that happens: it prints a note, returns `None`, and the datamodule **silently substitutes
synthetic jets**. The run only failed because this checkpoint conditions on aux and the
synthetic generator has no secondary planes; an aux-free checkpoint would have sailed
through and reported a recalibration fitted on synthetic data as "the training file's val
split". §6 now resolves the path against the repo root and hard-asserts both
`generator != "synthetic"` and a jet count matching §1's independent read.

Remaining honest gaps, unchanged and stated rather than skipped:

- **PYTHIA vs HERWIG** — no `herwig_driver` in WP5, so the generator systematic is not
  measured. The §2 train/test deltas above are the same-generator, different-seed noise
  floor it would have to exceed.
- **The encoder A/B** — `lundnet` is the pairing
  [`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §4 names for A4, not a
  measured winner. Its `k: 4` is dead code and the mean hadron sequence is 1.74 nodes, so
  there is very little for a chain EdgeConv to message-pass over. `encoder=gru` and
  `encoder=deepsets` arms are +2 trainings (~1 h at this scale) and would close §7's open
  probe.
- **v3 vs v4** — not run. The plan flags cross-attention as likely a wash on real PYTHIA
  for the same multiplicity reason, and this test deliberately fields one architecture.
  Nothing here confirms or refutes it; the four arms differ only in aux and seed.

## Verification

- `ctest --test-dir cpp/build` green; `read_lund_rntuple` shows `kt_floor_sec: 0.2` on both
  files and jet counts ≈ 497k / 97k.
- `pytest tests/ -q` green after any source change.
- Training: `best.ckpt` written, `metrics.csv` val curve monotone-ish, support guard silent
  (it reads `max_emissions`, not `n_cells`, so 30 bins does not affect it).
- The `eval` command above runs clean and writes `eval_metrics.json` + three PNGs.
- Notebook runs top to bottom on the test file and writes `prod_test_v0_metrics.json`.
- Acceptance, per [`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §9 as
  amended: beats identity on `dlund_posterior_medoid` **and** on
  `dlund_posterior_geomedian_cont`; if the two disagree in sign the arm is
  quantisation-limited, not better or worse.

## Risks that would invalidate the test

- **Seed collision — the single failure that voids everything.** Both files come from the
  same binary and the same card; only the seed differs, and `pythia_driver` does
  `Random:seed = seed % 900000000`. If the streams overlap, "held-out" is false and every
  number in the notebook is a training number. **Hard assert**: hash the `(x, y)` buffers of
  ~20k jets from each file and require an empty intersection. Nothing in the repo checks this
  today.
- **Different cards.** If the two files' `(z_cut, beta, kt_floor, kt_floor_sec, generator)`
  tuples differ, the assessment measures covariate shift, not generalisation. Hard assert
  they are equal — and separately assert the two data fingerprints *differ*.
- **`MBR_BACKEND="surrogate"` used for any reported number.** Beyond being a different risk
  function, `_lund_image`'s hard-coded `nb=10` bins at 0.6 while the model now decides at 0.2,
  so at 30 bins the surrogate is coarser than the model's own resolution — it happened to
  match at 10 bins. Hard assert `!= "surrogate"`.
- **`AUX == ()` on the loaded checkpoint.** Then the asymmetric file is inert by construction
  (it leaves x/y bit-for-bit unchanged) and the entire framing of this test is void. Warn
  loudly at load.
- **v4 may be a wash.** Cross-attention gave 17.85 vs 21.68 on synthetic but 4.64 vs 4.61 on
  real PYTHIA, because mean hadron multiplicity is 1.74 — there is no fixed-length bottleneck
  to remove. The asymmetric floor does **not** change this: it leaves the sequences
  untouched, so the multiplicity stays ~1.78. Expect v4 ≈ v3 and treat any delta inside the
  seed band as undecided ([`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md)
  §9.7).
- **900 cells on a peaked density.** Even at ~850k emissions the Lund density is
  concentrated; many of the 900 cells will be near-empty. §3's occupancy check is what tells
  you whether 30 bins was affordable, and `model.cell_label_smoothing` (default 0.0) is the
  knob if the tail cells are starving the head.
- **One seed, one arm.** No seed band, so no delta is rankable. This is a pipeline test.
