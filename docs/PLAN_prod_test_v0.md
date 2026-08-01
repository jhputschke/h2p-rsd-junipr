# PLAN_prod_test_v0 — one end-to-end production test

*Status: **RUN (twice — see [§Results](#results-2026-08-01)).*** Unlike the other
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

## Results (2026-08-01)

Executed as written, with one deliberate widening: the aux ablation of check 2 was run as
the **full two-by-two** (aux on/off × `trainer.seed` 0/1, four 60-epoch trainings) rather
than dropped. At ~30 s/epoch on one GB10 the whole grid cost about two hours, so the
"budget allows only one training" escape hatch never applied. Two seeds turned out to be
the *minimum* rather than the sufficient configuration — see the ablation section for why
the band a two-seed grid buys is too loose to be the load-bearing statistic.

**The run was performed twice**, and the second time is the one reported. The first pass
exposed a defect in the encoders that the assessment itself could not have found — see
[§The padding bug](#the-padding-bug-and-what-it-was-hiding) — and every number below comes
from arms retrained with it fixed. The first pass is archived at
`runs/prod_test_v0_pre_encoder_fix/` and quoted only for the comparison that prices the
bug.

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

### The padding bug, and what it was hiding

Running the generated closure notebook against the first pass turned up a `q(0|x)` that
disagreed with the artifact by 3×. The cause was not in the gate: **`gru` and `lundnet`
both read the zero-padding `collate` adds**, so `e(x)` was a function of the batch
composition.

* `gru` ran a **bidirectional** GRU over the padded tensor, so the backward pass started
  in the padding and swept through it into the real nodes.
* `lundnet`'s chain EdgeConv self-looped at `h[:, -1]` — the last row of the *padded*
  tensor rather than each jet's last real node — so a jet's final node took its neighbour
  from padding and its edge feature became `−h`. On this sample ~40% of jets have
  `nx = 1`, and for those that node *is* the jet.

Wrong on its own terms, and worse in consequence: training is always batched while
single-jet inference (`sample`, `map_estimate`, `length_pmf`) has `Mx = nx` and no padding
at all, so **the model was decoded from a context distribution it had never been trained
on**. On the first-pass checkpoint the same 256 jets gave mean `q(0|x)` = 0.053 decoded
one at a time and 0.155 in a batch, max per-jet difference 0.59 *on a probability*.

Fixed behind `encoder.mask_padding` (default true; `deepsets` was already correct, and a
pre-field snapshot backfills to the legacy path so it still evaluates as it was trained).
The reference script has the same defect, so `scripts/verify_parity.py` and
`tests/test_parity.py` pin `mask_padding=false` and say why — parity measures the refactor
against the reference *as it is*, and is back to max |Δ| = 0.0.
[`tests/test_encoder_padding.py`](../tests/test_encoder_padding.py) pins batch-invariance
for all three encoders, including that `per_jet_nll` no longer depends on how the eval
loop chunked the file.

**What it was hiding.** Retraining the identical grid with the fix changed the verdict of
the production test:

| | pre-fix | **fixed** |
|---|---|---|
| `dlund_posterior_medoid` / identity | 1.112 | **0.902** |
| `dlund_posterior_geomedian_cont` / identity | 1.111 | **0.871** |
| leading-emission ratio, **soft** `ln k_t` third | **1.520** | **0.955** |
| leading-cell 68% coverage (2 000-jet tier) | 0.455 | **0.538** |
| SBC χ² (same tier) | 359.4 | **107.0** |
| TARP max dev vs 0.079 null floor | 0.087 | **0.037** |
| `q(0\|x)` AUC | 0.724 | **0.823** |
| uncalibrated `q(0\|x)` / truth | **3.10×** | **0.98×** |
| fitted recalibration `(T, tilt)` | (1.372, −0.511) | **(1.010, +0.011)** |
| posterior W1 gmean vs plain RSD | 0.977 (7/14) | **0.414 (10/14)** |
| per-jet NLL SD across arms | 3.9 – 6.9 | **3.3 – 3.6** |

Three of the first pass's headline conclusions were artifacts of it:

1. **"The arm loses to identity."** It does not; it beats identity on both estimators.
2. **"Localized under-conditioning in the soft/wide-angle corner."** The soft-third ratio
   was 1.520 and is now 0.955 — all three `ln k_t` thirds beat identity. The mechanism is
   consistent: soft/wide-angle emissions dominate the `nx = 1` jets whose whole context
   was corrupted.
3. **"The length head is ~3× under-confident and needs the tilt."** After the fix the head
   is calibrated essentially out of the box (ratio 0.983, reliability 0.0001), and
   `fit_length_recalibration` returns the identity to within 1%. The ~2× under-confidence
   [`PLAN_empty_parton_tree.md`](PLAN_empty_parton_tree.md) F5 documented is at least
   partly this bug rather than the fit.

This is what an end-to-end production test is *for*. None of the unit tests, the parity
check, or the training curves could see it: every one of them lives on one side of the
train/inference boundary.

### The aux ablation

Held-out on the full 97 018-jet test file, four arms at 60 epochs each, fixed encoder:

| arm | aux | seed | held-out NLL/jet | per-jet SD |
|---|---|---|---|---|
| `aux_s0` (the headline arm) | on | 0 | **4.1063** | 3.51 |
| `aux_s1` | on | 1 | 4.1096 | 3.31 |
| `noaux_s0` | off | 0 | 4.1829 | 3.63 |
| `noaux_s1` | off | 1 | 4.1550 | 3.39 |

aux ON − aux OFF = **−0.0610 ± 0.0050 nat/jet** (paired SEM over 97 018 jets); seed band
(max within-arm spread) **0.0279**, so `|delta| / band = 2.18`. The pre-fix pass gave
−0.0588 ± 0.0072 against a 0.0282 band — the same answer, with the SEM ~30% tighter now
that a noise source has been removed from the likelihood.

**Read that ratio carefully.** The delta is well determined: the paired SEM bounds it at
±0.005. The *band* is not — it is `max` over two arms of a **two-seed range**, a statistic
with roughly 60% relative uncertainty and no error bar of its own. A third seed could move
it by half its size. An earlier pass on a 40 000-jet subsample of the *same* four
checkpoints returned band 0.0575 and delta −0.0560, i.e. **the opposite verdict**, purely
because the per-jet NLL has a wide spread and the band is a difference of differences. The
notebook now evaluates the ablation on the whole file for exactly this reason and prints a
caution whenever the ratio is under 3, as it does here.

**The strata are what carry the result.** Aux rides as *constant per-node columns of
`xf`*, so a jet with an empty groomed hadron tree (`nx == 0`) receives no aux signal
whatsoever:

| stratum | jets | aux on | aux off | delta | vs band |
|---|---|---|---|---|---|
| **`nx == 0` (control)** | 6 856 | 4.7669 | 4.7873 | **−0.0204** | **inside** |
| `nx > 0` | 90 162 | 4.0579 | 4.1219 | **−0.0641** | beyond |
| `n_sec == 0` | 21 241 | 3.5625 | 3.6222 | −0.0597 | beyond |
| `n_sec = 1` | 25 445 | 3.4706 | 3.5374 | −0.0667 | beyond |
| `n_sec = 2–3` | 31 128 | 4.2789 | 4.3348 | −0.0559 | beyond |
| `n_sec ≥ 4` | 19 204 | 5.2787 | 5.3418 | −0.0631 | beyond |

The control sits **inside** the seed band — consistent with no effect, which is what it
must be, since those jets carry no aux signal to exploit — while every stratum that *can*
carry aux is 2× the band outside it. (The aux-on and aux-off arms differ in input width
and therefore in initialisation, so some drift on the control is expected; the point is
that it is small and band-consistent, not that it is zero. The pre-fix pass had the
control at +0.0135, and reading that sign as evidence was over-reaching — it moved to
−0.0204 under the fix.)

So: **aux conditioning helps**, on a file where the secondary-plane features are not
degenerate. That could not have been established on `cpp/test_data/jets_aux.root`, which
(deliverable 6) is not an asymmetric-floor file at all. The claim rests on the control and
a ±0.005 paired SEM, **not** on clearing a two-seed band; a third seed per arm is what
would make the band itself quotable.

**What this ablation does *not* isolate.** The preset's `aux_features` is nine columns of
three different kinds, and the A/B moves all nine at once:

| columns | kind | defined for |
|---|---|---|
| `nsec`, `has_sec`, `ln_kt_sec`, `ln_kt_sec_sum`, `sec_depth` | secondary plane — what `ktFloorSec` unlocks | non-degenerate only when `n_sec > 0` |
| `ln_mg_pt`, `ln_ptg_pt` | groomed mass / momentum — **also redefined by the floor** (check 5) | every jet with a node |
| `ln_pt`, `abs_eta` | jet kinematics, floor-independent | every jet with a node |

So the `n_sec == 0` stratum still gains 0.0597, because four of the nine columns are
defined for it regardless. The supported claim is **"the nine-column aux set helps"**, not
"the secondary-plane features help"; the gain being flat across `n_sec` (−0.056 to −0.067,
with none of the symmetric file's `n_sec = 2–3` peak) is consistent with a large share
coming from the always-defined four. Isolating what `SoftDrop:ktFloorSec` actually buys
needs a third arm, and the right control is **`[ln_pt, abs_eta]`** — *not* all four
non-secondary columns, since `ln_mg_pt`/`ln_ptg_pt` are themselves complements of the
off-spine tree. That is +2 trainings (~1 h), and together with the third seed the band
wants, it is the obvious next run.

### Which NLL numbers survive the geometry change (check 1)

`nll_terms` returns the decomposition, so the report says which term moved rather than
warning about all of them:

| term | value | unit | 10-bin-comparable? |
|---|---|---|---|
| total | 4.1063 | per jet | **yes** — a density on the (ln 1/ΔR, ln k_t) plane |
| length `−ln q(N\|x)` | 1.1325 | per jet | **yes** — references no cell grid |
| split `−ln q(cell)` | 3.9172 | per emission | **no** — shifts by `2·ln 3 = 2.197` |
| coord `−ln p(du,dv,ln z,ψ)` | −1.8167 | per emission | **no** — pays that shift back |
| split + coord | 2.1005 | per emission | **yes** — the product is a density |

over 137 353 emissions in 97 018 jets (1.416 per jet). `cell_label_smoothing` is 0.0, so
the split term is a genuine log-probability.
[`tests/test_nll_terms.py`](../tests/test_nll_terms.py) pins the `2·ln 3` shift and the
invariance of the sum on an untrained model, where both are exactly computable.
`per_jet_nll` is now built from that decomposition rather than being an independent
expression, so the parts cannot drift from the whole.

The caveat that survives: a finer grid is a **strictly richer density class**, so a lower
30-bin total is evidence of better resolution, not of a better conditional.

### Acceptance: PASS

Cell level on 2 000 jets (1 667 with a truth leading emission — an 83.4% kept fraction, so
every `dlund_*` is `p(leading | n_y > 0)` per check 9), off-grid on 300:

| estimator | cell (2 000 jets) | off-grid (300 jets) |
|---|---|---|
| identity(x) | 0.651 | 0.647 |
| posterior mode | 0.679 | — |
| **medoid / geo-median** | **0.587 (0.902)** | **0.564 (0.871)** |

**Beats identity on both**, agreeing in sign — the criterion
[`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §9 as amended asks for. And
the gain is not localised any more:

| truth leading `ln k_t` | jets | identity | medoid | ratio |
|---|---|---|---|---|
| soft [0.00, 0.79] | 556 | 0.709 | 0.677 | **0.955** |
| middle [0.79, 1.53] | 555 | 0.656 | 0.556 | **0.847** |
| hard [1.53, 3.79] | 556 | 0.598 | 0.532 | **0.889** |

All three thirds beat identity. The population view agrees: `dist_closure_metrics.json`
puts the decode-free posterior series at a **0.414** W1 geometric-mean ratio against plain
RSD, winning 10 of 14 scoreable observables (KS 0.419, 10/13; χ² 0.315, 8/10). MAP and MBR
remain worse than plain RSD (1.83 and 1.28), so the decode is still the weak link — but
the density underneath it is now clearly better than the identity baseline.

### Calibration: improved, and the residue is the model's own

Every statistic quoted against its null (check 12), 2 000-jet tier:

- leading-cell 68% coverage **0.538**, 95% Wilson **[0.514, 0.562]** on 1 667 jets — 0.68
  is still outside, so residual **over-confidence** is real (pre-fix: 0.455);
- `sbc_chi2_uniform` **107.0** against the χ²(9) 95% point **16.90** — still non-uniform
  (pre-fix: 359.4);
- `tarp_max_dev` **0.037** against the null floor `1.36/√300 = 0.079` — now **below** it,
  i.e. consistent with a calibrated posterior, and `ECP(0.68) = 0.674` against a target of
  0.68 (pre-fix: 0.097, above the floor, ECP(0.68) = 0.591);
- per-coordinate PIT over 4 disjoint 5 000-jet chunks: **`ln z` is the one failing
  coordinate**, KS **0.066 ± 0.008** against a 0.016 critical value — and it got *worse*
  than pre-fix (0.047). `du` (0.013), `dv` (0.011) and `ψ` (0.012) all pass comfortably.

That `du`/`dv` result answers a preset question directly: **`sigma_floor: 0.005` was the
right call.** They are the within-cell offsets whose support narrowed 3× with `n_bins`,
and a binding floor shows up as a U-shaped, over-confident PIT. They are the two
best-calibrated coordinates.

The tension worth naming: TARP — a *joint* test over the whole tree in the physics metric
— now passes, while SBC-on-N and the leading-cell coverage do not. Those measure the
length marginal and a single cell; TARP measures the tree. Read together they say the
posterior's *shape* is close and its *multiplicity* is not, which the closure table
corroborates: `mean_mult_posterior` = 1.15 against a truth of 1.40, so the posterior now
**under**-produces emissions (pre-fix it over-produced, 1.53).

Region stratification is where the interval machinery earns itself. `wide_hard` reads
coverage 0.545 on **22 jets**, interval [0.35, 0.73] — consistent with almost anything, so
it is reported `scored: false` and drawn hollow rather than counted. The two scoreable
quadrants both still fail, and they fail differently: `wide_soft` 0.551 [0.53, 0.58] on
1 533 jets, `narrow_soft` **0.357 [0.27, 0.45]** on 112. The collinear corner remains the
worst region even after the fix — improved from 0.125, but the only place where the
residual defect is large.

### The empty tree, and the tau scale fix

`q(0|x)` AUC **0.823** — the model separates the two classes well. And after the encoder
fix the *scale* is right too, which it very much was not before:

| variant | mean `q(0\|x)` | truth | emp/pred | NLL of N | AUC | Brier | reliability |
|---|---|---|---|---|---|---|---|
| uncalibrated | 0.1633 | 0.1605 | **0.983** | 1.1351 | 0.823 | 0.1092 | **0.0001** |
| temperature only | 0.1639 | 0.1605 | 0.979 | 1.1350 | 0.823 | 0.1092 | 0.0001 |
| temperature + tilt | 0.1619 | 0.1605 | 0.991 | 1.1347 | 0.823 | 0.1092 | 0.0001 |

Every column is flat to four decimals across the three rows — the recalibration has
literally nothing to do.

`fit_length_recalibration` returns `(T, tilt) = (1.010, +0.011)` — the identity to within
1%. There is nothing left for it to correct. Compare the pre-fix pass, where the
uncalibrated head was 3.10× off, the fit returned `(1.372, −0.511)`, and the tilt was
*essential* (temperature alone recovered only a third of the gap). **That conclusion did
not survive the encoder fix**, and the plan's check 3 should be read accordingly: the
machinery is right and the diagnosis was right for the model it was measuring, but the
defect it was correcting was mostly upstream.

**The gate still transfers.** `tau = 0.3177`, fitted to reproduce the training-val empty
rate and applied **frozen** to the test file, gives `p_empty_pred = 0.171` against a truth
of **0.161** — ratio 1.067, recall 0.464, precision 0.435, MBR backend `pot`.

`tau` is now fitted on the **scale it is applied to**, which the first pass got wrong.
`empty_threshold_for_rate` returns a *quantile* of `q(0|x)`: scale-free in its ranking,
but a specific number on a specific distribution. §6 fitted it on the raw head — its
`length_pmfs` helper zeroes `(T, tilt)` so the recalibration fit sees unmodified logits —
while every deployed decode carries `(T, tilt)`. Self-consistent inside §6, and broken the
moment the frozen tau met a recalibrated head. Measured on the pre-fix checkpoint, where
the recalibration was large:

| | rate | recall | precision |
|---|---|---|---|
| raw tau on raw `q0` | 1.07× | 0.468 | 0.439 |
| raw tau on **recalibrated** `q0` | **2.83×** | 0.868 | 0.306 |
| recalibrated tau on recalibrated `q0` | 1.07× | 0.468 | 0.439 |

Same ranking throughout; only the cut moved. This is `length.py`'s documented "re-fit per
pT window" hazard reached through *recalibration* instead of selection. §6 now fits on the
recalibrated pmfs and prints the mismatch beside the result; the artifact records
`tau.fitted_under = {length_temperature, length_tilt}`, and
`lund_distribution_closure_prod_test_v0.ipynb` **asserts** the pair it applies matches, and
refuses an artifact that records no scale at all. On *this* checkpoint the two taus nearly
coincide (0.3177 vs 0.3208) precisely because the recalibration is now the identity — the
fix costs nothing here and prevents the general case.

### 900 cells were affordable; the head's rank is what to watch (check 10)

Truth occupies **275 of 900** cells (30.6%) over 137 353 emissions; the posterior emits
271, covers 256 of the truth-occupied ones and **misses 19**, while emitting a handful the
truth never visits. So 30 bins did not starve the head. `split_head` ends in
`Linear(64, 900)` — a rank bound of 64 with no analogue at 100 cells — and its measured
effective rank (spectral entropy) is **37.2 of 64**. The bound is not saturated, so the
residual under-coverage argues for more capacity in the *decoder* before more cells, not
for more data.

### §7 and §8

**§7 stayed a pointer**, and its staleness guard did real work. The notebook does not
re-implement the population-level W1/KS/χ²; it accepts `dist_closure_metrics.json` only
after checking that its `data.path` names the test file and its `checkpoint` sits in *this*
run directory (not merely that the basename is `best.ckpt`), and it compares the artifact's
recorded `(T, tilt)` against §6's fit. In this run §6 executed *before* the closure
notebook had written its artifact, so §7 correctly reported it absent — the numbers quoted
above come from the artifact written afterwards, and a re-run of §6 would accept it. That
ordering is the one wrinkle worth automating next.

The variant that produces it is **generated**, not hand-edited:
[`scripts/make_prod_closure_nb.py`](../scripts/make_prod_closure_nb.py) builds
`lund_distribution_closure_prod_test_v0.ipynb` from v2 with every cell except the title and
section 0 copied byte-for-byte, and reads its five settings from
`prod_test_v0_metrics.json` at runtime. A hand-edited copy would have been a second
definition of the same headline ratios.

**§8, support and validity.** Truth and identity(x) sit at exactly 0 out-of-window, 0
soft-drop violation, 0 `k_t`-floor violation over 137 353 and 168 521 emissions — the
generator enforces the fiducial window, so the floor is a hard zero. Against that, the
posterior's **0.88%** soft-drop violation rate is entirely the model's: it places emissions
below the `z_cut` boundary the training data never crosses. Small, but it is a *support*
error rather than a calibration error, and nothing else in the suite surfaces it.

### What did not need re-litigating

`p_empty_true` reads 0.177 rather than exactly 0.0, confirming the empty-tree accounting
still sits *before* `run_closure`'s leading-emission `continue` (check 9); the notebook
asserts this against an independently computed truth rate. The support guard is silent:
`max N = 6` against `model.max_emissions = 25`, so `P(N > support) = 0` exactly. Weights
are constant (`n_eff = n = 97 018`), so `eval/`'s unweighted averaging is exact here and
check 13's caveat does not bite — the notebook prints it either way.

**Check 7 was in scope after all**, and it found two consumers that could not answer
"nothing" once a `tau` exists: `print_point_estimate` called `map_estimate` directly and so
printed a non-empty MAP for the very jets `run_closure` had just counted as empty, and
`generator_spread` called `map_estimate()` with no arguments at all, measuring "the
dominant systematic" under signature defaults rather than the run's decode. Both fixed;
`serving.predict` and `run_closure` were already correct.
[`tests/test_empty_tree_gate.py`](../tests/test_empty_tree_gate.py) pins all four.

### Remaining honest gaps

- **`ln z` calibration** — the one failing coordinate (KS 0.066 vs 0.016), and it got worse
  under the fix. The most concrete open defect.
- **Residual over-confidence in the length marginal** — coverage 0.538 [0.514, 0.562] and
  SBC χ² 107 vs 16.9, with `mean_mult_posterior` = 1.15 against truth 1.40. TARP passing
  while these fail localises it to the multiplicity rather than the tree shape.
- **`narrow_soft`** — 0.357 [0.27, 0.45] coverage on 112 jets, still the worst quadrant.
- **PYTHIA vs HERWIG** — no `herwig_driver` in WP5, so the generator systematic is not
  measured. The train/test deltas above are the noise floor it would have to exceed.
- **The encoder A/B** — `lundnet` is the pairing
  [`PLAN_ProductionAssessment.md`](PLAN_ProductionAssessment.md) §4 names for A4, not a
  measured winner. Now that padding no longer contaminates either, `encoder=gru` and
  `encoder=deepsets` arms are +2 trainings and would close §7's open probe — and are more
  interesting than before, since the fix changed what all three encoders compute.
- **v3 vs v4** — not run; this test deliberately fields one architecture.

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
