# PLAN — Production test edit: the edit transducer against the v1 winner

**Status: RUN AND COMPLETE.** All 9 arms of the §7 grid trained and evaluated, the reference
re-evaluated per WP-F.1, and gates E1–E9 applied. Results:
[`PROD_TEST_edit_RESULTS.md`](PROD_TEST_edit_RESULTS.md), with machine-produced numbers in
[`PROD_TEST_edit_TABLES.md`](PROD_TEST_edit_TABLES.md).

**Outcome in one line: the edit factorization loses** — E4 (TARP, the deciding gate), E5
(coverage) and E6 (NLL) all fail unanimously across three seeds, with bands that do not
overlap the reference's, at a *smaller* parameter budget. **E7 passes**, so the family's
falsifiable physics claim survives: Λ_eff = 0.631 GeV at R² = 0.949, read off the arm never
told the functional form. The mechanism explains the loss — `frac_anchored ≈ 0.19`, so the
anchoring term reaches a fifth of the emissions and an ordinary insertion head does the rest.
Two deviations from §12 are recorded in the results document's §0, not here.

What landed in the tree:

| WP | where |
|---|---|
| **WP-E** the `ln z` support port | `config.py` `EditTransducerConfig`, `models/edit.py` (`lnz_bounds` / `_log_lnz` / `_draw_lnz` / `_mode_lnz`), `configs/model/edit_v{1,2}.yaml` |
| **WP-F.1** reference re-eval | `scripts/eval_prod_test_v1.sh --run-root runs/prod_test_v1 --device cpu --only v1_contstop_s0,v1_contstop_s1` (no code change needed) |
| **WP-F.2** gates | `scripts/prod_test_edit_gates.py` |
| **WP-F.3** notebook | `notebooks/prod_test_v1.ipynb` §0/§6/§8 |
| **WP-G** anchoring | `scripts/edit_anchoring_diagnostic.py` |
| §7 grid | `presets/prod_test_edit.yaml`, `scripts/run_prod_test_edit.sh` |
| §9 tests | `tests/test_edit_lnz_support.py`, `tests/test_nll_comparability.py`, extensions to `tests/test_edit_model.py` / `tests/test_models.py` |

Companion to [`PLAN_prod_test_v0.md`](PLAN_prod_test_v0.md)
and [`PLAN_prod_test_v1.md`](PLAN_prod_test_v1.md); as there, **this plan holds the design
and the rationale, the results document will hold only numbers.** Every pass criterion
below is fixed before the grid runs.

**Framing.** v1 localized the residual defect and attributed it: the joint tree posterior
is too narrow, and the cause is the multiplicity *factorization*. v1 could not fix it —
it could only name the arm that does better. This run asks whether a **third**
factorization, designed against that defect before it was measured, beats the arm v1
picked.

---

## 1. Context — what v1 left open

v1 ended with an attribution, not a fix
([`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md) §6–§7). Three instruments say the
joint tree posterior is too narrow, and §4.8 attributes it: all six explicit-`q(N|x)` arms
fail G7 (TARP 0.0335–0.0465 against a recomputed 0.0275 null), spanning three seeds, two
aux configurations and both `ln z` heads; both implicit continue/stop arms pass (0.0200,
0.0225), with the signed bias going from −0.020 to −0.0002. `v1_contstop` also wins
held-out NLL by 0.124 nat. v1's own verdict on that: *"a recommendation this run supports;
not a recommendation the run validated end to end,"* since `v1_contstop` was fielded as a
two-seed comparison arm rather than as a candidate deliverable.

The edit transducer ([`PLAN_EditTransducer.md`](PLAN_EditTransducer.md)) is a third
factorization of the same length/shape coupling, and it was designed against exactly this
defect before v1 measured it: `n_y = n_x − #del + #ins`, so length is anchored at `|x|` and
the open-ended continue/stop mechanism is removed **structurally**, not recalibrated
downstream. Its `q(N|x)` is the terminal value of a structural DP — exact, parameter-free,
explicitly conditioned on `|x|`.

So the question is narrow and directed: **does the edit factorization beat the v1 winner on
the metrics that decided v1?** The reference is `v1_contstop_s0` (with `s1` for its band),
not `v1_base`.

| v1 finding | edit's structural claim | decided by |
|---|---|---|
| G7 FAIL on every explicit-`q(N\|x)` arm; PASS on continue/stop (§4.8) | length anchored at `\|x\|`, coupled to shape through the latent alignment | gate **E4** |
| G4's coverage clause fails, 0.518–0.540 against 0.68 (§4.3) | same | gate **E5** |
| continue/stop wins held-out NLL by 0.124 nat (§4.8) | a different, also-exact density on the same space | gate **E6** |
| the `q(N\|x)` *marginal* is calibrated; the *factorization* narrows the joint (§6) | exact structural `q(N\|x)`, no fitted head | gate **E3** |
| `v1_contstop`'s `length_pmf` costs 52.19 ms/jet, forcing `N_FIT` 20 000 → 5 000 (§4.9) | the structural DP is `O(n_x·n_max)` and exact | WP-F.3 |
| G3 still fails: `ln z` shape *inside* its support, 1.05–2.07× crit (§4.2) | **not readable on this family** | gate **E9**, `n/a` by construction |

## 2. Key facts (verified against the code, not assumed)

- **The edit family's `ln z` is an unbounded Normal in both mixture components** —
  `gauss_logpdf` at [edit.py:353](../src/h2p_rsd_junipr/models/edit.py) (anchored) and
  edit.py:374 (free), with matching `randn` draws at edit.py:534, 545, 686, 692 — and
  [`EditTransducerConfig`](../src/h2p_rsd_junipr/config.py) (config.py:197) has no
  `lnz_support` field. So the family reproduces the v0 / `v1_legacy_lnz` support failure
  (0.81% below soft drop, 3.98% above `z = ½`; results §4.1) **by construction**, and its
  NLL is not comparable to `v1_contstop_s0`, which is `lnz_support: physical`. This is the
  one blocking item, and it is WP-E.
- **The plane coordinates are already truncated.** `trunc_normal_logpdf` bounds `u` and `v`
  to the geometry range in both components (edit.py:351–352, 372–373). Only `ln z` was left
  unbounded, so the port is a `ln z`-only change.
- **`_log_cell_mass` needs no change.** It integrates `u`/`v` over the cell; the `ln z` and
  `ψ` factors integrate to 1 and never appear (edit.py:618). That stays true for a
  `u`-dependent bound, since `∫ p(z|u) dz = 1` for every `u` — so the constrained
  forward–backward behind `sample_coordinates` is untouched by WP-E, and that is a fact to
  assert in a test rather than a hope.
- **`check_lnz_support` is already family-agnostic.** It reads `model.lnz_support` through
  `OmegaConf.select` (stats.py:131) and verifies the declared `(z_cut, β)` against the
  file's own grooming record *and* that every truth emission lies inside the interval.
  Adding the three config fields buys the WP-A guard for free.
- **`supports_coordinate_pit = False`** (edit.py:134), and `coordinate_pits` returns `None`
  when the family cannot provide a transform, so the suite degrades gracefully
  (calibration.py:150–152). `scripts/prod_test_v1_gates.py` already enforces *"a gate whose
  input is missing is `n/a`, never `pass`."* The blind spot is therefore safe — but it is
  real, and it must be named up front (E9).
- **`length_pmf` is the exact structural DP** (edit.py:494), so §4.9's 71× penalty does not
  apply and `N_FIT` can go back to 20 000. `decode.length_temperature` / `length_tilt` are
  deliberately no-ops for this family, and `empty_gate` reads an exact `q(N = 0 | x)`.
- **The alignment diagnostics are already wired.** `alignment_posterior`, `edit_summary` and
  `physics_width_params` exist, and `closure.py` picks `frac_anchored` / `delete_rate` /
  `insert_rate` up family-gated on `getattr(model, "edit_summary", None)` (closure.py:168).
- **`edit_v2`'s prefix head is the memory item.** With prefix conditioning on, the free-cell
  head is evaluated at `(B, n_col, Ny, n_cells)`; at `n_bins = 30` that is 900 cells across
  the whole lattice. `edit_v1` collapses to `T = 1` and is ~`Ny` times cheaper.
- **The harness takes `--run-root`,** and `scripts/eval_prod_test_v1.sh` now also takes
  `--device` (default `cpu`), with the standing rule that the device is a **whole-grid**
  decision because cpu and cuda are a different RNG stream and different float kernels. The
  v1 artifacts predate that flag and carry no `device` key; they were produced under the
  hardcoded `CUDA_VISIBLE_DEVICES=""`, i.e. cpu. This run pins `--device cpu` everywhere
  and re-evaluates the reference so the whole comparison is one device, recorded.
- **`v1_contstop`'s published band is two seeds.** Its coverage interval [0.5304, 0.5310] is
  narrow because it is 2 draws, not because it is stable.

**The reference numbers this run is measured against** (results §4.8, §4.9; `v1_contstop`
two-seed grid-tier bands unless marked):

| quantity | `v1_contstop` |
|---|---|
| best val NLL/jet | 3.7927 [3.7799, 3.8054] |
| TARP max dev (against a 0.0275 null) | 0.0212 [0.0200, 0.0225] |
| `coverage_68` | 0.5307 [0.5304, 0.5310] |
| medoid/identity | 0.9307 [0.9286, 0.9327] |
| `ln z` PIT KS / `pit_ks_max` | 0.0398 [0.0315, 0.0482] — **not readable on edit**, see E9 |
| deep pass, `s0`, 97 018 jets | medoid/identity 0.901, geo-median/identity 0.872, NLL/jet 3.8143, coverage 0.541 [0.517, 0.565], `q(0\|x)` AUC 0.827 |

## 3. Design decisions

- **One code change, and it is a support correction, not an experiment.** WP-E ports v1's
  WP-A into the edit family. Support correctness needs no A/B to be adopted; the
  `e_v1_legacy_lnz` arm exists for *attribution* — reproduce the failure under identical
  data — exactly as `v1_legacy_lnz` did.
- **WP-E is what makes E6 quotable at all.** A truncated `ln z` concentrates its mass on a
  1.61-wide interval and gains NLL relative to a Normal on `ℝ` for reasons that have nothing
  to do with fit quality. Comparing edit's NLL to `v1_contstop`'s without the port would
  measure the head, not the factorization. The gate script's existing `!`-and-footnote rule
  enforces this mechanically.
- **The blind spot is declared up front, not discovered in the results.** The edit family
  has no `coordinate_cdfs`, so G3, `pit_ks_max` and the region × coordinate cross are
  unavailable. E9 records this as a *known incompleteness of the comparison*. Landing
  `coordinate_cdfs` is a §11 trigger, deliberately not this run's work.
- **`edit_v2` is trained but conditionally quoted.** `PLAN_EditTransducer.md`'s verification
  4 is a stage gate: *"if the widths are flat in `k_t`, the anchoring assumption is wrong —
  stop, and do not build stage 2."* Its `Λ_eff = 1.29 GeV, R² = 1.000` is a 6-epoch fit on
  the 54k-jet test file, not production. E7 re-runs it at production scale and pre-registers
  that a flat fit demotes every `edit_v2` number to null context.
- **The `physics_width = false` arm is the readout, not an ablation for its own sake.**
  Quoting `Λ_eff` from the arm that was *told* the functional form restates the
  parametrization; the free-MLP arm is the independent measurement. This is the rule
  `PLAN_EditTransducer.md` already set for itself, carried into production.
- **Capacity is a confound, reported rather than assumed away.** `ctx_dim = 64` on edit and
  `dec_dim = 64` on `ar_junipr_v4` are not the same parameter budget. Every table prints
  parameter counts, and v1 §3.1's "matched at fixed capacity" caveat carries.
- **One device, one support convention, one code path.** The reference arms are
  re-evaluated rather than quoted, so no delta in this document mixes two evaluation passes.
- **No mid-run changes.** A trigger that fires is recorded in the results document and acted
  on in a follow-up plan. That is what a pre-registered plan is for.

## 4. WP-E — the `ln z` support port (blocking)

1. **`config.py`, `EditTransducerConfig`** (config.py:197): add `lnz_support: str =
   "legacy"`, `lnz_zcut: float = 0.1`, `lnz_beta: float = 0.0` — the **same names and
   semantics** as `ARJuniprConfig`, so `data.stats.check_lnz_support` works unchanged.
   `"legacy"` is bit-identical to today.
2. **`models/edit.py`**: bounds `lo_z(u) = ln z_cut − β·u`, `hi_z = ln ½`, evaluated at the
   **emitted** `u` (constant at the fielded `β = 0`; implement the general cell-conditional
   form now so `β ≠ 0` files need no code change, per WP-A's precedent). Sites:
   - `_log_f_anch` (:353) and `_log_f_free` (:374) — `gauss_logpdf` → `trunc_normal_logpdf`
   - `_draw_emission` (:534, :545) and `sample_coordinates` (:686, :692) — `randn` →
     `trunc_normal_sample`
   - `_emission_mode` (:701) — the mode must be clamped into the support
   - `_log_cell_mass` (:618) — **unchanged**; state why in a comment (the `ln z` factor
     integrates to 1 for every `u`, so the constrained lattice is untouched)
3. **`configs/model/edit_v1.yaml` / `edit_v2.yaml`**: the three fields, with the header-
   comment convention those files already use.
4. **Parity:** no new parameters, so `state_dict` is unchanged, and `"legacy"` reproduces
   today's `log_prob` bit-for-bit. `scripts/verify_parity.py` pins the AR families and must
   be *asserted* unchanged, not assumed — if it moves, something outside
   `edit.py`/`config.py` was edited.

## 5. WP-F — harness (no new tiering, no new eval code)

1. **Re-evaluate the reference**, no retraining:
   `bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_v1 --device cpu --only v1_contstop_s0,v1_contstop_s1`,
   then `python scripts/refresh_support_audit.py` on both run roots. This puts every number
   in the head-to-head on one code path, one `EDGE_TOL` convention and one recorded device —
   and incidentally backfills the `device` key the pre-flag artifacts lack.
2. **`scripts/prod_test_edit_gates.py`** — imports `gate_g1/g2/g4/g5/g6/g7` verbatim from
   `scripts/prod_test_v1_gates.py` and adds the family A/B in `gate_g8`'s shape with
   `name_a="v1_contstop"`, `name_b="e_v1"`. It carries the two rules that script already
   enforces, plus a third: **a gate whose instrument the family does not implement is `n/a`
   with a named reason**, so E9 cannot read as a pass. Adds a parameter-count column.
   Writes `docs/PROD_TEST_edit_TABLES.md`.
3. **Notebook.** Point `notebooks/prod_test_v1.ipynb` at `e_v1_s0` via `CKPT_PATH`, restore
   `N_FIT` to 20 000, and **measure and report ms/jet for `length_pmf`** against §4.9's
   52.19 (exact DP vs a 500-draw sampling pass per jet). Read the four-column support audit
   from `eval/support.py`, not the notebook's three-column §8 table — results §4.9 records
   that gap, and it is precisely the `z > ½` wall a `legacy` arm would fail.

## 6. WP-G — the anchoring diagnostic at production scale

New `scripts/edit_anchoring_diagnostic.py`, writing `anchoring_diagnostic.json` beside each
checkpoint. All of it comes off `model.alignment_posterior`:

1. responsibility-weighted residual widths binned in `ln k_t` (and `R_g`, `N`), fit to
   `σ = σ_0 + Λ_eff·exp(−ln k_t)` — **quoted from `e_v1_freewidth`**, the arm not told the form
2. `frac_anchored` / `delete_rate` / `insert_rate` (already in closure) — the 6-epoch
   reference is `frac_anchored = 0.20`, and mixture identifiability (risk 2) is live
3. deletion rate vs `ln k_t` — should track the sub-floor fragmentation population
4. free-emission (insertion) rate vs distance to the grooming boundary
5. crossing-pair count in sampled alignments — the monotonicity audit (risk 1)
6. the `n_x = 0` rate on the production file, which bounds how much of it the anchoring
   mechanism can act on at all

## 7. The grid

`presets/prod_test_edit.yaml` — identical to [`presets/prod_test_v1.yaml`](../presets/prod_test_v1.yaml)
except `model: edit_v1` and the edit block, so geometry (`n_bins = 30`), data files, encoder
(`lundnet`), aux(9), epochs (60) and batch size (256) are held fixed against the reference
arm. `scripts/run_prod_test_edit.sh` follows `run_prod_test_v1.sh`'s
one-override-per-arm shape.

| arm | varies | seeds | trainings |
|---|---|---|---|
| `e_v1` | — (edit_v1 + lundnet + aux(9) + physical `ln z`) | 0, 1, 2 | 3 |
| `e_v2` | `prefix_conditioning = true` | 0, 1, 2 | 3 |
| `e_v1_legacy_lnz` | `lnz_support = legacy` | 0 | 1 |
| `e_v1_freewidth` | `physics_width = false` | 0 | 1 |
| `e_v1_gru` | `encoder = gru` | 0 | 1 |

**9 trainings**, one overnight grid. Reference: the existing
`runs/prod_test_v1/v1_contstop_s0` and `_s1` checkpoints, re-evaluated per WP-F.1 — no
retraining. Train `data/jet_aux_asym.root` (seed 1), test `data/jet_aux_asym_test.root`
(seed 2); conda `js_fno`, one NVIDIA GB10. Seeds are `trainer.seed` at fixed `data.seed = 0`,
so the band measures initialisation and batch ordering, **not** split variance (v1 §3).

The `gru` probe is one training and licenses only "worth a proper multi-seed A/B" — v1
§3.2's discipline, carried unchanged. It is here because v1 found `gru` moved TARP and
coverage in the same direction `v1_contstop` did, so leaving it unmeasured in a run about
TARP would be a gap.

## 8. Pre-registered gates

Evaluated on the seed-2 test file at the two tiers `scripts/eval_prod_test_v1.sh` defines
(calibration: 2 000 jets, `tarp_refs = 200`; decode: 300 jets, MBR, `min_emissions = 0`),
with unanimity across `e_v1`'s three seeds required. Coverage intervals are 95% Wilson;
regions with n < 30 are reported `scored: false`.

| # | gate | criterion |
|---|---|---|
| **E1** | acceptance (carries G1) | medoid **and** geo-median beat identity on both tiers, agreeing in sign, on every seed. Nothing below means anything if this fails. |
| **E2** | support (carries G2) | sampled out-of-window / soft-drop / `z > ½` / `k_t`-floor rates ≡ 0 on every `physical` edit arm; any nonzero value is a bug, not a finding. `e_v1_legacy_lnz` reproduces the ~0.81% / ~3.98% failure (attribution). |
| **E3** | multiplicity | ⟨N⟩_post/⟨N⟩_truth ∈ [0.95, 1.05] on the **full** population, never the `N ≥ 1` selection (v1 §1.1); SBC-on-N reported as a percentile of **its own MC null** (≥ 200 reps), never against χ²(9) (v1 §1.2); `q(0\|x)` AUC against 0.827; and the exact `length_pmf` agrees with the sampled multiplicity histogram at production scale. |
| **E4** | **TARP — the deciding gate** | `e_v1` max dev inside the null recomputed at this run's own (n, refs, α grid), with the band's floor < 0.05, on every seed. A/B clause: the three-seed band must **clear** `v1_contstop`'s [0.0200, 0.0225] to claim an improvement; overlapping bands are a tie and are reported as one. |
| **E5** | coverage | leading-cell 68% coverage Wilson-consistent with 0.68 in every scoreable region to pass; the A/B delta against 0.5307 [0.5304, 0.5310] must clear the pooled seed spread. |
| **E6** | held-out NLL | `e_v1` vs `v1_contstop` 3.7927 [3.7799, 3.8054]; the delta must clear the pooled spread. **Quotable only if E2 passes** (both arms on the physical `ln z` head) *and* the §9 normalization audit passes. Both families are `exact_likelihood = True`, so the number means the same thing on both sides. |
| **E7** | anchoring (stage gate) | responsibility-weighted residual widths fall monotonically in `ln k_t` and fit `σ = σ_0 + Λ_eff·exp(−ln k_t)` with **Λ_eff ∈ [0.2, 5] GeV and R² ≥ 0.9**, read from `e_v1_freewidth`. If they are flat in `k_t`, the anchoring premise fails on production data and **every `edit_v2` number is reported as null context, not as a family result**. `frac_anchored` reported beside it. |
| **E8** | `edit_v1` vs `edit_v2` | decided on held-out NLL + TARP + coverage at matched seeds, encoder, batch size and epochs, with parameter counts printed. Coordinate PITs cannot decide — neither stage has them. Conditional on E7. |
| **E9** | coordinate PIT | **`n/a` by construction, pre-registered.** No `coordinate_cdfs` on this family, so G3, `pit_ks_max` and the region × coordinate cross are unavailable. The one v1 gate still open — `ln z` shape *inside* its support, 1.05–2.07× crit — **cannot be read here**, and the head-to-head is incomplete on that axis. Recorded as a blind spot, never as a pass. |

**Validity checks carried unchanged from v0 §1 / v1 §2:** seed disjointness under the
`full` fingerprint, asymmetry verification of the test file, the train/test same-generator
noise-floor table as the systematic stand-in, and `check_lnz_support` against the file's own
grooming record before any arm trains.

## 9. Tests

- **New `tests/test_edit_lnz_support.py`:** MC normalization of the four-coordinate emission
  density ≈ 1 in `physical` mode, **each mixture component separately**; `"legacy"`
  bit-identical `log_prob` (parity guard); zero violations of *both* walls at 10⁵ draws from
  `sample` **and** from `sample_coordinates`; general-β bounds against hand-computed values;
  `_log_cell_mass` provably unchanged by the port.
- **New `tests/test_nll_comparability.py`** — what licenses E6: on one frozen context,
  MC-integrate the `ar_junipr_v4`/contstop emission density and the `edit_v1` emission
  density over the same `(u, v, ln z, ψ)` box; both ≈ 1, and both report NLL under the same
  per-jet / per-emission convention.
- **Extend `tests/test_edit_model.py`:** `map_estimate` and `sample_coordinates` respect the
  support under `physical`; `describe_cells(...).logprob == log_prob(...)` still holds.
- **Extend `tests/test_models.py`:** add `["model=edit_v2", "encoder=gru"]` beside the
  existing `edit_v1` entry, so stage 2 satisfies the generic contract suite from day one.
- **Parity:** `python scripts/verify_parity.py` + `pytest tests/test_parity.py -q`
  unchanged — asserted, not assumed.
- **Smoke before the grid:** `bash scripts/run_prod_test_edit.sh --smoke`, specifically to
  **measure `edit_v2`'s peak memory** at `n_bins = 30`.

## 10. Docs

- `docs/CONFIGURATION.md` §4: `lnz_support` on the edit family; extend the boxed
  NLL-comparability note to cover the cross-family A/B.
- `docs/PLAN_EditTransducer.md`: status block gains the WP-E port and the production-scale
  E7 result — its current `Λ_eff = 1.29 GeV` is a 6-epoch test-file number and already says so.
- `docs/PROD_TEST_edit_RESULTS.md`: skeleton with E1–E9 as its section spine, mirroring v1's
  structure; `docs/PROD_TEST_edit_TABLES.md` is the gate script's generated output,
  committed beside it so the prose can be checked against machine-produced numbers.

## 11. Non-goals (deferred, with triggers)

- **`coordinate_cdfs` for the edit family** — the exact prefix-conditional CDF as a
  responsibility-weighted mixture of `trunc_normal_cdf` / `gauss_cdf` / `vonmises_cdf` off
  the same forward recursion. Trigger: E4 or E6 favours edit, making it a fielding
  candidate — the blind spot must be closed before it can be quoted as a deliverable.
- **The monotone rational-quadratic spline** (Durkan et al., arXiv:1906.04032), v1 §4.4's
  fired trigger. It belongs to the AR family's follow-up; adopting it inside edit
  mid-comparison would confound the family A/B.
- **Per-node joint coordinate density** (cINN coords / CFM, `PLAN_UPDATES.md` WP1) — v1's
  other fired trigger, same reasoning.
- **Full-tree edit DP** (tree-edit alignment, `O(n_x² n_y²)`-class) — different machinery, a
  separate plan, as `PLAN_EditTransducer.md` already scopes it.
- **Scheduled sampling or any training-side exposure-bias remedy** — improper objective
  (Huszár, arXiv:1511.05101). `edit_v1` has zero exposure bias structurally in any case.
- **HERWIG driver / fragmentation-variation weights.** Still absent; the train/test deltas
  remain the noise-floor stand-in and the largest unquantified systematic (v0 §10, v1 §8).
- **Split variance.** `data.seed` stays 0; nothing here bounds what a different train/val
  split would do.

## 12. Verification

1. `pytest tests/test_edit_lnz_support.py tests/test_nll_comparability.py tests/test_edit_dp.py tests/test_edit_model.py tests/test_models.py -q`
2. `python scripts/verify_parity.py` + `pytest tests/test_parity.py -q` — unchanged.
3. `bash scripts/run_prod_test_edit.sh --smoke` — end-to-end on every arm, plus the
   `edit_v2` memory measurement.
4. `bash scripts/run_prod_test_edit.sh` — the §7 grid.
5. `bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_edit --device cpu`, then the
   same with `--run-root runs/prod_test_v1 --device cpu --only v1_contstop_s0,v1_contstop_s1`,
   then `python scripts/refresh_support_audit.py` on both roots.
6. `python scripts/edit_anchoring_diagnostic.py` (WP-G) on every edit arm.
7. `python scripts/prod_test_edit_gates.py --out docs/PROD_TEST_edit_TABLES.md`.
8. `notebooks/prod_test_v1.ipynb` pointed at `e_v1_s0`, `N_FIT = 20 000`, on the full
   97 018-jet test file — the deep pass, matching what results §4.9 did for `v1_contstop_s0`.
9. Fill `docs/PROD_TEST_edit_RESULTS.md` against E1–E9.

Run in the conda `js_fno` environment.

## 13. Risks

- **Parameter-count confound.** `ctx_dim = 64` (edit) and `dec_dim = 64` (AR) are not the
  same budget, and edit's free-cell head is a `Linear(ctx, 900)` evaluated across the whole
  lattice. Counts are printed in every table; if they differ by more than ~2×, the family
  claim is *stated as confounded*, per v1 §3.1's caveat.
- **`edit_v2` memory** scales as `batch × (n_x+1) × n_y × n_cells`, measured in the smoke
  pass. If it forces a batch-size drop, drop it for `e_v1` too so E8 stays matched, and
  record that this then differs from `v1_contstop`'s 256 — an optimization difference, not a
  density difference, but it belongs in the results rather than in a footnote nobody reads.
- **Mixture identifiability** (`PLAN_EditTransducer.md` risk 2): `frac_anchored` was 0.20 at
  6 epochs. Collapse toward the free head makes edit an expensive AR model. E7 reports it;
  that outcome is a null result and is written as one.
- **The two-seed reference.** `v1_contstop`'s bands are narrow because they are 2 draws. A
  three-seed edit band clearing them is not decisive on its own, and E4/E5 say so.
- **E9 is a real gap, not a formality.** Any recommendation to field edit must state that
  the comparison never saw the `ln z` shape failure v1 left open.
- **`n_x = 0` jets** (6.9% of the PYTHIA reference) reduce exactly to the free head —
  covered by an existing test, but the production rate bounds the anchoring mechanism's
  reach and is reported (WP-G.6).
- **E7 could fail.** `Λ_eff = 1.29 GeV, R² = 1.000` is 6 epochs on a 54k-jet file. A flat
  production fit is an *informative* failure — it says the anchoring premise does not hold on
  this selection — and this plan pre-commits to reporting it as such rather than retuning
  into a pass.
- **Device drift.** `scripts/eval_prod_test_v1.sh`'s standing rule is that the device is a
  whole-grid decision; a half-cpu/half-cuda comparison is a silent ranking hazard. WP-F.1
  pins `--device cpu` for the edit grid *and* the reference re-eval for exactly that reason.

## References

Graves (RNN-T), arXiv:1211.3711 · Graves et al. (CTC), ICML 2006 · Chan et al. (Imputer),
arXiv:2002.08926 · Stern et al. (Insertion Transformer), arXiv:1902.03249 · Gu et al.
(Levenshtein Transformer), arXiv:1905.11006 · Azimov, Dokshitzer, Khoze & Troyan, *Z. Phys.
C* **27** (1985) 65 · Korchemsky & Sterman, hep-ph/9902341 · Hoang, Mateu, Pathak, Stewart
et al., arXiv:1906.11843 · Dreyer, Salam & Soyez, arXiv:1807.04758 · Larkoski, Marzani,
Soyez & Thaler, arXiv:1402.2657 · Dreyer, Necib, Soyez & Thaler (RSD), arXiv:1804.03657 ·
Talts et al. (SBC), arXiv:1804.06788 · Lemos et al. (TARP), arXiv:2205.03910 · Eikema &
Aziz, arXiv:2005.10283 · Kumar & Byrne, HLT-NAACL 2004 · Huszár, arXiv:1511.05101 · Durkan
et al., arXiv:1906.04032 · Dreyer & Qu (LundNet), arXiv:2012.08526 · Bierlich et al.
(HOMER), arXiv:2410.06342 · Brown, Cai & DasGupta, Statist. Sci. **16** (2001) 101.
