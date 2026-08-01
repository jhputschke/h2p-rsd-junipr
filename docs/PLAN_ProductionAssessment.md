# PLAN — Production assessment of the posterior models

A fixed protocol for answering one question on real data: **which `(model, encoder, decode)`
triple do we ship, with what uncertainty, over what domain of validity?**

The repo trains seven families but has a verdict on one. `ar_junipr` is *verified* in the
[README](../README.md#verification-this-is-the-acceptance-test) sense — weight-level parity
([`scripts/verify_parity.py`](../scripts/verify_parity.py)) plus a train+closure run
([`scripts/verify_synthetic.py`](../scripts/verify_synthetic.py)) — but on **synthetic** data.
`cinn`, `diffusion` and `cfm` carry the full shared contract and are exercised by `tests/`, yet
none has a closure or calibration verdict on data beyond the short demo in
[`calibration_v2_walkthrough.ipynb`](../notebooks/calibration_v2_walkthrough.ipynb). The two
existing study drivers — [`scripts/ab_v2_v3.py`](../scripts/ab_v2_v3.py) and
[`scripts/probe_map_collapse.py`](../scripts/probe_map_collapse.py) — each answer one narrow
question well. What is missing is the protocol that spans them.

Everything below is expressed as `base=` presets so a number can be re-derived from a filename
rather than from a shell history.

**Status: enabling fixes landed; the assessment itself is not started.** §8 G1 and G2 are
implemented ([`cli.py`](../src/h2p_rsd_junipr/cli.py), `config.explicit_group_keys`,
[`tests/test_eval_overrides.py`](../tests/test_eval_overrides.py)), so the held-out test set,
the pT-window axis and preset-driven decode cells are all runnable. No presets, driver or runs
exist yet.

---

## 1. Scope, and what was deliberately left out

| Axis | Decision |
|---|---|
| Assessment covers | statistical quality (NLL, closure, calibration) **plus robustness / prior dependence** |
| Data | real PYTHIA, with an **independent test sample** that model selection never sees |
| Breadth | **all seven families, single seed** (7 trainings) |
| Runtime / latency / throughput / serving | **out of scope** — see §8 G3 |

The single-seed choice is a budget decision with a consequence that must be stated wherever a
number is quoted: [`aux_input_ab.ipynb`](../notebooks/aux_input_ab.ipynb) measured a
−0.029 nat/jet effect against a 0.029 nat/jet three-seed spread and correctly called it noise.
**At one seed, any cross-arm NLL delta inside roughly that band is undecided and is reported as
undecided, not ranked.** Decode cells (§6) and robustness slices (§7) are evaluated on a *fixed*
checkpoint and are paired comparisons, so they do not inherit this limitation — which is why the
protocol spends its breadth there rather than on more arms.

---

## 2. Four properties of the code that fix the protocol's shape

Each of these silently changes what a reported number means, so the protocol is fixed in advance
rather than discovered mid-run.

1. ~~**`eval` with a checkpoint rebuilds the checkpoint's own val split and ignores CLI
   `data.*`.**~~ **Fixed (G1).** `eval` now lifts explicitly named `data` fields over the
   snapshot, treats a named sample as a test set (every jet, not a re-split), reads it once, and
   records the fingerprint / jet count / overrides in `eval_metrics.json`.
2. ~~**`eval` ignores preset `decode:` blocks and `decode=<group>` selectors.**~~ **Fixed (G2).**
   `decode` binds through the full composition surface at eval, CLI last. The §6 ladder can be
   run either as literal tokens or as preset files.
3. **`experiment.*` comes from the CLI-composed config, never the checkpoint** — unchanged, and
   now the deliberate third case rather than an accident: `data` and `decode` default to the
   checkpoint and are liftable, `experiment` is the eval suite's own configuration.
   `geometry` / `encoder` are pinned to the checkpoint and are **not** liftable.
4. **There is no test split.** [`LundDataModule`](../src/h2p_rsd_junipr/data/datamodule.py)
   produces train/val only. An independent `jets_test.root` plus G1 is the entire mechanism; no
   schema change was needed.

Machinery this protocol reuses rather than reinvents:
[`run_closure`](../src/h2p_rsd_junipr/eval/closure.py),
[`run_calibration`](../src/h2p_rsd_junipr/eval/calibration.py) (with `coordinate_pits` and
`run_tarp`), [`check_multiplicity_support`](../src/h2p_rsd_junipr/data/stats.py),
[`learned_min_emissions`](../src/h2p_rsd_junipr/inference/length.py),
[`mbr_select`](../src/h2p_rsd_junipr/inference/mbr.py),
[`select_pt_range`](../src/h2p_rsd_junipr/data/datamodule.py),
[`save_metrics` / `plot_calibration`](../src/h2p_rsd_junipr/eval/report.py),
[`medoid_cell` / `geometric_median`](../src/h2p_rsd_junipr/eval/closure.py) with
[`scripts/leading_estimators.py`](../scripts/leading_estimators.py) behind the §9.1 criterion,
and the arm / cell / markdown-table idiom of [`scripts/ab_v2_v3.py`](../scripts/ab_v2_v3.py).

---

## 3. Reproducibility spine

Held constant across every arm: `trainer.seed=0`, `data.seed=0`, `trainer.deterministic=true`,
one geometry, one optimizer block, one data file.

Recorded automatically: the run dir is `runs/<stamp>-<config_hash>` carrying its resolved
`config.yaml`; the data fingerprint
([`_fingerprint`](../src/h2p_rsd_junipr/data/datamodule.py)) is printed at train and stored in
the checkpoint, so the run↔data link is explicit. The assessment as a whole is pinned to a git
SHA plus the two RNTuple fingerprints.

> **Bands, not bit-equality.** MPS/CUDA atomics are non-deterministic even at fixed seed
> ([USAGE §8](USAGE.md)). Every number here is quoted with a tolerance band. The bit-exact check
> in this repo is `verify_parity.py` (weight copy), never end-to-end training.

**Pre-flight — all three must pass before any assessment number is quoted:**

```bash
pytest -q
python scripts/verify_parity.py       # bit-for-bit NLL parity vs the v2 script
python scripts/verify_synthetic.py    # train + closure on the v2 synthetic data
```

### Data generation

```bash
conda activate js_fno                 # ROOT / FastJet / fjcontrib / PYTHIA live here
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
ctest --test-dir cpp/build

./cpp/build/pythia_driver 500000 data/jets_train.root 1 cpp/cards/pp_dijet.cmnd
./cpp/build/pythia_driver 200000 data/jets_test.root  2 cpp/cards/pp_dijet.cmnd   # independent seed
./cpp/build/read_lund_rntuple data/jets_test.root Jets                            # jets != events
```

Both files **must** come from the same card — the grooming provenance `(z_cut, beta, kt_floor,
kt_floor_sec, generator)` is stamped into the RNTuple and the support guard (§7.5) reads it.
The dry run of the whole harness uses the committed `cpp/test_data/jets.root` (25k events) so
nothing long is generated before the pipeline is known to work.

---

## 4. The training ladder — simplest to state-of-the-art

Encoder held at `gru` across A1–A4 so that decoder changes are attributable; the encoder swap is
a separate probe (§7.6). One seed each.

| Arm | Preset | Selectors | What it adds over the rung below | Verified today? |
|---|---|---|---|---|
| **A0** | — (free) | identity(x) | the plain-RSD hadron sequence used *as* the parton estimate; zero parameters. Already reported by `run_closure` as `dlund_identity` / `mult_bias_identity` on every eval (and `dlund_identity_cont` under `experiment.closure_continuous`). Compare against `dlund_posterior_medoid`, **not** `dlund_posterior_mode` — see §9.1 | n/a — the floor every arm must beat |
| **A1** | `t1_v1_cells.yaml` | `model=ar_junipr_v1` | discrete Lund cells only (`continuous_coords=false`) | — |
| **A2** | `t2_v2_reference.yaml` | `model=ar_junipr_v2` | continuous within-cell coordinates | ✅ parity + closure (synthetic) |
| **A3** | `t3_v3_multhead.yaml` | `model=ar_junipr_v3` | first-class `q(N\|x)`: `q(y\|x) = q(N\|x)·q(y\|N,x)` | — |
| **A4** | `t4_v4_xattn.yaml` | `model=ar_junipr_v4`, `encoder=lundnet` | decoder cross-attention over per-node hadron states — the SOTA AR arm | — |
| **A5** | `f_cinn.yaml` | `model=cinn` | conditional normalizing flow, exact likelihood | — |
| **A6** | `f_cfm.yaml` | `model=cfm` | flow matching, **exact** probability-flow-ODE likelihood | density verified; posterior not |
| **A7** | `f_diffusion.yaml` | `model=diffusion` | **surrogate** `log_prob` — segregated in every table | — |

Two rules that make the table honest:

- **A4 gets a parameter-matched twin.** [CONFIGURATION §4](CONFIGURATION.md#4-model--the-posterior-family)
  warns that a like-for-like v3/v4 comparison needs `dec_dim` shrunk to match parameter counts
  (`xattn_heads` must divide `dec_dim`). A4 runs at `dec_dim=64` **and** at the matched width;
  `n_params` — printed by `train` — is a table column, not a footnote. Whether cross-attention
  helps is known to depend on hadron sequence length: a large win on the synthetic generator, a
  wash on tightly-groomed PYTHIA. This assessment is the measurement.
- **A7 never shares an NLL column with A1–A6.** `exact_likelihood=False` means its `log_prob` is
  a training surrogate, not a normalized density; `train` and `eval` both say so out loud and the
  tables must too. Its sampling-based metrics (SBC / PIT / coverage / TARP) remain valid and are
  reported normally.

---

## 5. Preset layout

Uses the `base=` mechanism documented in
[CONFIGURATION §0](CONFIGURATION.md#custom-setups-without-a-long-cli-chain): a preset's own
directory becomes a group-file root searched **before** `configs/`, so shared blocks live in
group files and each tier file stays ~10 lines with no duplication.

`model` and `encoder` group files are deliberately **not** added — those groups are polymorphic
and a new file there would also need a `MODEL_SCHEMA` / `ENCODER_SCHEMA` entry. Families are
picked by selector and tuned by inline value patch, exactly as
[`presets/mbr_study.yaml`](../presets/mbr_study.yaml) does.

```
presets/production/
  data/assess.yaml             # rntuple, data/jets_train.root
  trainer/assess.yaml          # schedule, seed, determinism
  optim/assess.yaml
  experiment/e0_smoke.yaml     #  25 x  20, WP2 switches off
  experiment/e1_standard.yaml  # 300 x 200, + pit_coords + stratify_regions
  experiment/e2_full.yaml      # 300 x 500, + tarp
  decode/d0..d6.yaml           # the §6 cells; selectable at eval as `decode=d4_mbr` (G2)
  t0_smoke.yaml  t1_v1_cells.yaml  t2_v2_reference.yaml
  t3_v3_multhead.yaml  t4_v4_xattn.yaml
  f_cinn.yaml  f_cfm.yaml  f_diffusion.yaml
```

### Shared group files

```yaml
# presets/production/data/assess.yaml
source: rntuple
path: data/jets_train.root
ntuple: Jets
seed: 0
val_fraction: 0.1
min_val: 2000
cache_dir: null
pt_var: jet_pt
pt_min: null          # inclusive spectrum for training; the windows are an EVAL axis (§7.2)
pt_max: null
```

```yaml
# presets/production/trainer/assess.yaml
max_epochs: 40
batch_size: 128
seed: 0
amp: false
compile: false
ema_decay: null
num_workers: 0
deterministic: true
```

```yaml
# presets/production/optim/assess.yaml
lr: 1.0e-3
weight_decay: 3.0e-4
scheduler: cosine
eta_min: 1.0e-4
grad_clip: 1.0
```

```yaml
# presets/production/experiment/e1_standard.yaml
name: assess_standard
closure_jets: 300
n_closure_samples: 200
pit_coords: true
stratify_regions: true
tarp: false
```

```yaml
# presets/production/experiment/e2_full.yaml
name: assess_full
closure_jets: 300        # NOT scaled with the sample: TARP is n_jets x (K+1) EMD solves
n_closure_samples: 500
pit_coords: true
stratify_regions: true
tarp: true
tarp_refs: 200
tarp_reference: pooled   # a second pass with `prior` is part of §7
```

`e0_smoke.yaml` is the same shape at `closure_jets: 25`, `n_closure_samples: 20`, every WP2
switch `false`.

### Tier files

Each is a `defaults:` block plus the per-arm value patch. Two representative bodies:

```yaml
# presets/production/t2_v2_reference.yaml — the verified reference, on real data
defaults:
  model: ar_junipr_v2
  encoder: gru
  data: assess
  trainer: assess
  optim: assess
  decode: default
  experiment: e1_standard
run_root: runs/assessment/t2_v2
```

```yaml
# presets/production/t4_v4_xattn.yaml — the SOTA AR arm
defaults:
  model: ar_junipr_v4        # v3 backbone + decoder cross-attention
  encoder: lundnet           # returns_sequence=True, required by cross-attention
  data: assess
  trainer: assess
  optim: assess
  decode: default
  experiment: e1_standard
model:
  dec_dim: 64                # xattn_heads (4) must divide dec_dim; the param-matched twin
  dec_layers: 2              # overrides dec_dim on the CLI (§4)
  max_emissions: 25          # raise if the §7.5 support guard fires on real grooming
encoder:
  num_layers: 5
run_root: runs/assessment/t4_v4
```

```bash
h2p-rsd-junipr train base=presets/production/t4_v4_xattn.yaml
```

`f_cinn.yaml` / `f_cfm.yaml` / `f_diffusion.yaml` follow the same shape with their own family
selector; note their value patches may only contain that family's own fields (`dec_dim` under a
`model: cinn` preset fails at load as `Key 'dec_dim' not in 'CINNConfig'` — which is the config
system working, not a bug).

---

## 6. The inference / decode ladder

Decode is **inference-time only**, so every cell runs on the already-trained arms: the grid costs
7 trainings, not 7 × |grid|. This is the `ab_v2_v3.py` insight, and it is what makes a single-seed
budget buy a wide decode study.

Since G2, a cell can be given either way — as the literal tokens below, or as the matching
`presets/production/decode/*.yaml` selected with `decode=d4_mbr` (or through a `base=` preset).
The tokens are what the driver passes; the files are what a human reproduces a row with. Either
way only the named fields move and the rest stay as the checkpoint left them.

| Cell | Tokens appended to `h2p-rsd-junipr eval <ckpt>` | What it isolates |
|---|---|---|
| **D0** | `decode.point_estimator=map decode.min_emissions=0` | the raw joint mode — exposes the empty-tree collapse (`map0_frac`) |
| **D1** | `decode.min_emissions=1` | the shipped hard floor (repo default) |
| **D2** | `decode.min_emissions=1 decode.length_penalty=0.6` | GNMT `score/len**α` against the brevity bias |
| **D3** | `decode.min_emissions=1 decode.length_floor_quantile=0.5` | the learned per-jet floor `max(min_emissions, Q_α(P(n\|x)))` |
| **D4** | `decode.point_estimator=mbr decode.mbr_backend=pot decode.min_emissions=0` | floor-free MBR under the perturbative-Lund EMD |
| **D5** | D4 `+ decode.mbr_resample_to_qn=true` | q(N\|x) reweighting of the candidate pool — expected **inert** for A1/A2 (no exact head), live for A3–A7 |
| **D6** | D4 `+ decode.mbr_coords=+lnz decode.beam_width=16 decode.topk_cells=10` | richer ground metric, wider beam |
| **T** | `decode.cont_temperature=0.9 / 1.0 / 1.1` | sampling temperature as an exposure-bias probe |

Reported alongside, per arm: the **five-estimator multiplicity comparison** from
[`inference_demo.ipynb`](../notebooks/inference_demo.ipynb) §6 — MAP, MAP+learned floor,
posterior mean, posterior median, MBR — because the MAP is the wrong summary for a *count* and no
table here should imply otherwise. The per-N stratified bias table that `run_closure` always
prints is the headline for whether a bias is real or an averaging artifact.

**Cost.** MBR closure is O(K²) EMD solves per jet (shrink with `decode.mbr_n_candidates`);
TARP is `n_jets × (K+1)` solves. Both need the `[mbr]` extra (`pot`). This is why `e2_full.yaml`
holds `closure_jets` at 300 instead of scaling it with the larger sample.

---

## 7. The assessment matrix

Every arm at E1; the arms that clear §9 also at E2. Headline numbers are computed on
`data/jets_test.root` — never on the split used for model selection (needs G1).

1. **Held-out test.** Train/select on `jets_train.root`, report on `jets_test.root`.
2. **pT-window transfer.** `data.pt_var=jet_pt` with `[100,150) / [150,250) / [250,∞)` GeV. The
   kinematic analogue of region stratification: does calibration hold *locally on the spectrum*,
   or only on average? Uses the existing half-open window in
   [`select_pt_range`](../src/h2p_rsd_junipr/data/datamodule.py), so adjacent windows tile the
   sample without double-counting.
3. **Grooming / aux shift.** The same checkpoint on an asymmetric-floor file
   (`SoftDrop:ktFloorSec`) vs `jets.root`. **`cpp/test_data/jets_aux.root` is NOT that file** —
   it predates the `kt_floor_sec` column entirely (`read_lund_rntuple` prints no
   `kt_floor_sec` line for it) and is symmetric at 1 GeV; it adds the aux *columns*, not the
   asymmetric *floor*. Generate one with
   [`cpp/cards/pp_dijet_asym_floor.cmnd`](../cpp/cards/pp_dijet_asym_floor.cmnd), as
   [`PLAN_prod_test_v0.md`](PLAN_prod_test_v0.md) step i does. **Caveat stated up front:** the
   asymmetric floor leaves the `x`/`y` sequences bit-for-bit unchanged and only redefines
   `x_mg` / `x_ptg`, so this axis is **inert for the default `aux_features=[]` arms** and bites
   only once aux conditioning is on.
4. **Aux conditioning.** `encoder.aux_features=[]` vs `[ln_mg_pt,nsec,ln_pt]` on the best arm.
   One seed cannot re-open the adoption verdict, so the protocol *reports the existing three-seed
   finding* (−0.029 nat/jet against a 0.029 spread — gate failed, aux stays opt-in) and re-checks
   only the stratum that survived it: `n_sec = 2–3`, −0.100 nat/jet.
5. **Multiplicity-support guard.** `P_data(N > model.max_emissions)` recorded for every
   categorical-head arm (A3–A7) against the real grooming. Hard error above 1e-3, warning above
   1e-4. It is **strict at train and non-strict at eval**, so it must be read from the train log,
   not inferred from the eval output. Fix by raising `model.max_emissions` — not by quietly
   accepting the clamp, which mis-normalizes exactly the tail the physics cares about.
6. **Encoder probe.** `gru | lundnet | deepsets` on the best AR arm only (+2 trainings), labelled
   a probe rather than a rung of the ladder.
7. **Generator systematic — reported as BLOCKED.**
   [`generator_spread`](../src/h2p_rsd_junipr/eval/systematics.py) and
   `configs/experiment/pythia_vs_herwig.yaml` presuppose a generator-B RNTuple that nothing
   produces: `herwig_driver.cpp` is [PLAN_UPDATES.md](PLAN_UPDATES.md) WP5, not started. Following
   the precedent set by `aux_input_ab.ipynb`, this is reported as **blocked, never silently
   skipped** — it is the dominant systematic and its absence is the single largest caveat on any
   conclusion here. The cheap partial substitute this protocol *does* specify: a **same-generator,
   different-seed spread** using `jets_test.root`, which measures the statistical noise floor
   against which any future generator spread must be read.

---

## 8. Fixes the protocol needs

- **G1 — `eval` honours an explicitly named `data` group over the checkpoint config. ✅ landed.**
  `config.explicit_group_keys` answers "did this invocation *name* the field?", which
  `load_config` cannot: it always returns a fully populated group seeded from
  `configs/config.yaml`, so an unrequested group is indistinguishable from a requested default.
  `cli._lift_onto_snapshot` copies only the named fields onto the snapshot, and an explicitly
  named sample is evaluated **whole** — a test set, not a corpus to re-split. The fingerprint,
  jet count, scope and the overrides themselves land in `eval_metrics.json`. The discarded first
  datamodule load is gone, so the sample is read once.
  **Restricted to `data` and `decode` on purpose** — `geometry` and `encoder` change tensor
  widths and the model contract and stay pinned to the checkpoint.
- **G2 — decode binds from a preset at eval. ✅ landed.** The same mechanism, so `decode=<name>`,
  a `base=` preset's `defaults:`/inline block and dotted tokens all work, CLI last. The §6
  ladder is runnable either way.
- Both are covered by [`tests/test_eval_overrides.py`](../tests/test_eval_overrides.py),
  including the negative case (a CLI `geometry.n_bins` must not reach the model) and the
  precedence case (a dotted token outranks the preset that selected the group file).
- **G3 — no runtime harness.** Nothing in `src/`, `scripts/` or `tests/` measures latency,
  throughput or memory. Out of scope by decision (§1) and recorded here so this assessment is not
  mistaken for a production-readiness sign-off. A later pass would cover: batched `log_prob`
  throughput, K=200 sampling latency, MAP beam vs O(K²) MBR cost, the TorchScript export
  (`export_encoder_torchscript`, already allclose-verified) and the FastAPI `/predict` path.

---

## 9. Exit criteria — what makes an arm shippable

1. **Beats A0 identity** on `dlund_posterior_medoid` and on `|mult_bias|`. An arm that does not
   beat copying the hadron tree is not a model, whatever its NLL.

   **Not `dlund_posterior_mode`** — that key gates on the wrong estimator. The modal leading
   cell minimises expected 0-1 loss, but the score is `lund_distance`, so the mode is optimal
   for a loss nobody measures; the medoid ([`medoid_cell`](../src/h2p_rsd_junipr/eval/closure.py))
   is the argmin over the same support of the quantity actually reported. Measured on the
   walkthrough `ar_junipr_v3` (2000 val jets, K=200): mode **1.030×** identity, medoid
   **0.944×** — the criterion flips on the estimator alone. The mode is still reported, for
   continuity with tables written before this change; it is not what an arm is judged on.

   **Quote the un-quantised row too.** At the default geometry a cell is ~0.6 wide and these
   distances are ~0.6, so the cell-level metric is largely measuring the grid. Run with
   `experiment.closure_continuous=true` and quote `dlund_posterior_geomedian_cont` against
   `dlund_identity_cont` beside the cell numbers; on the same checkpoint that pair reads
   **0.905×** (95% CI [0.882, 0.928], paired bootstrap). An arm whose two rows disagree in
   *sign* is quantisation-limited, not better or worse than A0.

   Stratify by leading `ln kt` before concluding — `scripts/leading_estimators.py` prints the
   thirds. On the walkthrough checkpoint the model wins the hard and middle thirds and *loses*
   the soft one (1.088×), which is a localized under-conditioning finding, not a pooled verdict.
2. `coverage_68` inside a stated band around 0.68; `sbc_rank_mean` and `pit_mean` near 0.5.
3. `pit_coords_ks_max` below the `1.36/√n` critical value at the printed emission count, **and
   the histogram shape recorded** — U-shaped ⇒ over-confident, dome ⇒ over-dispersed — including
   the `by_emission_index` breakdown, which is where exposure bias shows. The report's `space`
   tag is quoted with it: only `ar_junipr` is `physical`; `cinn`/`cfm` report `latent`.
4. **Region-stratified: no Lund quadrant fails the coverage band.** Calibration that holds only
   on average over the plane does not pass — this is the precondition for any localized claim.
5. `tarp_max_dev` small **and its sign quoted**, with `ecp_at` in the quotable form ("at 90%
   credibility the posterior actually covered X%").
6. Support guard under 1e-4 (§7.5); `map0_frac` ≈ 0 at the shipped decode cell.
7. NLL comparisons confined to `exact_likelihood=True` families, with any delta inside the seed
   band declared **undecided** rather than ranked (§1).

---

## 10. Reporting

`runs/assessment/` mirroring the layout of `runs/aux_input_ab/`: one run dir per arm with its
`config.yaml`, `metrics.csv` / `metrics.jsonl` curves, `eval_metrics.json`, and the three
calibration figures written by `plot_calibration` (`calibration_pit_coords.png`,
`calibration_tarp.png`, `calibration_by_region.png`; figures need the `[plots]` extra, the JSON
does not).

Top level: `assessment_table.md` + `assessment.json`, in the shape of `ab_table.md`. Columns:

| arm | n_params | val NLL | **test NLL** | decode cell | map0_frac | ⟨n−n_true⟩ mean | ⟨n−n_true⟩ median | cov68 | PIT KS max | TARP max dev |

with a per-arm limitations line, and the seed band from §1 printed above the table so no reader
has to go looking for it.

---

## 11. Non-goals

Runtime, latency, throughput and serving (§8 G3). The PYTHIA-vs-HERWIG generator systematic
(WP5 — reported blocked, §7.7). Multi-seed re-adjudication of the aux gate (§7.4). Any change to
a model family, a loss, or the likelihood: this protocol measures what is implemented, it does not
extend it.
