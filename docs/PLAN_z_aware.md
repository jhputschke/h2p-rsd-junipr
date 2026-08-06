# PLAN — a `ln z`-aware decode metric, and whether the spline's `d(MBR)` regression is an artifact of not having one

Status: **proposed, not yet run.** Follows directly from `PLAN_lnz_spline_head.md` §6.2 and
`SUMMARY_Model_Status.md` §2.5, which record a finding this document exists to *test* rather
than assert: the RQ-spline `ln z` head improved held-out NLL and `pit_ks_max` on 4/4 arms
while `d(MBR)` got **worse** on 4/4, and the explanation offered was "the MBR metric runs on
`lnDR_lnkt` and cannot see `ln z`".

**Result in one line:** *(to be filled by WP-0/WP-4 — the verdict rule is §5 and is fixed
before any arm runs.)*

**Pre-registration note.** §5 is written now, with no number from it in hand. The reason is
`PLAN_lnz_spline_head.md` §8.1: the last mechanism in that line of work was elegant, wrong,
and tempting to reinterpret after the fact. A verdict rule written after the numbers is not
a verdict rule.

---

## 1. Context — the question, and why it is not the obvious one

The trigger is a question about what the model conditions on: *does it only learn on
`ln k_t` and `ΔR`, not `z`?*

**On the density, no — `ln z` is fully learned.** It is encoder input column 2
(`features.py:202`, `N_NODE_FEAT = 5`), a mandatory RNTuple field (`data/rntuple.py:29-32`),
a target column `yraw[..., 2]` (`features.py:224-231`), and it carries its own conditional
density — Normal, truncated Normal, or RQ spline (`models/ar_junipr.py:419-427`) — whose
log-density enters `coord_ll` with **unit weight** and no detach, mask or scaling
(`ar_junipr.py:499` → `:559` → `:566`). Every family except `ar_junipr_v1` does the same.
That is exactly why `ln z` could fail its PIT at 2.16× critical and why splining it moved
the NLL on every seed.

**On everything downstream, yes — and in three layers.** This is the real content of the
question, and it is what makes the §2.5 finding hard to read.

---

## 2. The three layers of `ln z`-blindness

**(a) Selection.** `decode.mbr_coords` defaults to `"lnDR_lnkt"` (`config.py:352`), so the
EMD ground metric spans `(u, v)` only. Everything built on that matrix inherits it: the MBR
medoid, `mbr_select_stratified`, the cluster partition and its masses/radii/exemplars, TARP,
`eval/stability.py`, and every `dlund_*`.

> **And the knob that looks like the fix is inert.** `lund_cloud` hard-codes
> `lz = ps = 0.0` for cell-chain draws (`inference/mbr.py:108-111`), and
> `posterior_distances` builds every cloud straight from those chains
> (`mbr.py:759-761`) — `coords_by_draw` exists on `mbr_select` but only decorates the
> *winner* afterwards (`mbr.py:823-825`). So `mbr_coords="+lnz"` appends a constant-zero
> third column and changes **no distance**. `mbr_weight="z"` is silently identical to
> `unit` for the same reason (`mbr.py:122-123`). The knob is not merely off by default; it
> cannot be switched on. It is nevertheless *half*-live: `ground_diameter` is already
> coords-aware (`inference/clusters.py:66-84`), so `+lnz` moves an admissibility check
> governing a matrix it does not otherwise touch.

**(b) Scoring.** `dlund_mbr` — the number that regressed — is the Euclidean distance between
leading-emission **cell centres** (`eval/closure.py:39-45`, used at `:289`). It is blind to
`ln z` *and* to the within-cell `du`/`dv` offsets. The continuous block is no better: it
draws real coordinates but `_leading_coords` slices `a[..., :2]` (`closure.py:92-97`), and
**it carries no MBR row at all** — the only `*_cont` keys in the artifacts are `identity`,
`posterior_mode` and `posterior_geomedian`. So the ruler that scored the spline could not
register what the spline improved, on either axis.

**(c) Structural.** The Lund cell grid discretizes `(u, v)` only (`geometry.py:53-60`), so
`ln z` never influences *which* cell is emitted — it only refines a coordinate inside an
already-chosen cell. And the AR family deliberately evaluates the soft-drop bound at the
cell's loosest `u` (`ar_junipr.py:470-482`), keeping `ln z` conditionally independent of
`du`, so the exact identity `ln z = u + v − ln p_{T,sum}` is documented in prose and encoded
nowhere. **Out of scope here** — `PLAN_lnz_spline_head.md` §7.3 owns it and nothing in this
document fires its trigger.

A consequence worth writing down regardless of the verdict: `sample_batch` returns **cell
chains**, so the `ln z` head reaches the fielded metric *only through training* — a shared
trunk whose gradient changes the cell posterior. "The metric cannot see `ln z`" is therefore
literally true, and any residual `dlund_mbr` difference between a spline arm and its control
is a statement about the **cell** posterior.

---

## 3. What the existing artifacts already say — free, no compute

Paired deltas (spline − same-seed control) read off the eight committed `eval_metrics.json`
files. Decode tier throughout: **300 jets scored, 247 with a leading emission**, `K = 200`,
`mbr_n_candidates = 64`, `pot`, `R = 8.485`, `β = 1`, `weight = kt`, `min_emissions = 0`,
`mbr_coords = lnDR_lnkt`, `n_bins = 30`.

| Δ = spline − control | s0 | s1 | s2 | contstop | signed |
|---|---:|---:|---:|---:|---|
| `dlund_mbr` | **+0.0047** | **+0.0113** | **+0.0091** | **+0.0031** | **4/4** |
| `dlund_posterior_medoid` | −0.0088 | −0.0025 | +0.0014 | +0.0018 | 2/4 |
| `dlund_posterior_mode` | +0.0045 | −0.0049 | −0.0238 | +0.0011 | 2/4 |
| `dlund_posterior_mode_cont` | −0.0007 | −0.0040 | −0.0307 | +0.0007 | 1/4 |
| `dlund_posterior_geomedian_cont` | −0.0087 | +0.0010 | +0.0043 | −0.0039 | 2/4 |
| `dlund_identity` / `_cont` | **+0.0000** | **+0.0000** | **+0.0000** | **+0.0000** | — |

Four readings, and they reorder the work:

1. **The pairing is exact, and certified.** `dlund_identity` is identical to all printed
   digits within every pair. It is a model-independent function of the jets, so the same 247
   jets are scored on both sides of every pair. **A per-jet paired analysis is therefore
   legitimate — and none has ever been run.** Every number in §2.5 is an unpaired mean.
2. **The cell posterior did not get worse.** Every *other* estimator built from the **same
   draws** — leading-cell medoid, leading-cell mode, continuous geometric median — is
   mixed-signed. The effect is confined to *which of the K draws the EMD medoid picks*.
3. **4/4 on four pairs has a floor of p = 0.125.** A two-sided sign test cannot do better at
   this `n`, and it cannot be strengthened with seeds: `spline_s3/s4/s5` exist,
   `v1_base_s3/s4/s5` do not, so there are exactly four pairs and there will be four. **More
   power has to come from jets.**
4. **`dlund_mbr` loses to `dlund_posterior_medoid` on all eight arms**, controls included
   (0.594–0.609 against 0.567–0.586; mean gap ≈ 0.020). The MBR medoid is ~0.02 worse than a
   free one-node baseline on this ruler — **4× the effect under debate**. Any statement about
   +0.005 must be read beside that.

**So the first question is not "is the regression an artifact" but "is there a regression to
explain".** That is WP-0: one decode pass per arm and a paired bootstrap, before any
plumbing.

---

## 4. The work packages

### WP-0 — Is there a phenomenon? *(gates it all; needs only WP-1's free per-jet rows)*

Per-jet paired BCa bootstrap of `Δ(dlund_mbr)` over the 247 jets, four pairs, 10 000
resamples, fixed seed, at the published tier.

- **G-repro** — re-measured `dlund_mbr` within 0.5% of the committed value on 8/8. (Precedent:
  the ceiling probe's sanity row, 2.3489 → 2.3495.)
- **G-pair** — `dlund_identity` identical within each pair on 4/4.
- **G-exists** — the regression is **ESTABLISHED** iff `Δ > 0` with the paired 95% CI
  excluding 0 on **≥ 3 of 4** pairs.

> If **G-exists fails**, the verdict is **INCONCLUSIVE-BY-CONSTRUCTION**: +0.005 is inside
> its own per-jet noise, and the explanation in §2.5 is untestable because the phenomenon it
> explains is not resolved. Escalate **once** to 1000 jets, same everything. If still
> unresolved, **rewrite** the `SUMMARY` sentence as "d(MBR) is unchanged within its own
> per-jet noise" — an explanation is not owed for a number that is not resolved. WP-1–WP-3
> still ship; they fix latent defects and are not conditional on this.

### WP-1 — A `ln z`-aware ruler *(free: no metric change, no new model calls)*

`hat = model.map_or_mbr(...)` (`closure.py:217`) already carries **drawn** coordinates
(`describe_cells`, `coords_source="sample"`), so `hat.nodes[t].ln_z` is a genuine draw from
`q(ln z | cells, x)` **today**. The ruler half needs none of WP-3.

In `eval/closure.py`:

- `_leading_coords(arr, ncols=2)` — add the parameter, default 2, so every existing caller is
  unchanged *by construction*; add `_leading_row(arr)` so `(u, v, ln z)` come off one
  `argmax(a[:, 1])` rather than three selections.
- In the `cont_ok` block (`:255-279`), for identity / mode / geometric-median **and MBR (new
  — the series is absent from the continuous block entirely)**: `dlund_*_cont` (2-D
  continuous), `dlund3_*_cont` (3-D over `u, v, ln z`), `dlnz_*` (`|Δ ln z|` alone).
  `geometric_median` (`:67-81`) is already dimension-agnostic.
- **Additive only.** `decode_headline` stays `dlund_mbr` (`:352`); every pre-existing key
  keeps its exact meaning and value. Same discipline as `coverage_null_reps`' `fork_rng`
  (`eval/calibration.py:493-515`): a switch that perturbs the statistic it exists to explain
  is worse than no switch.
- **Unavailable ⇒ `NaN`, never `0`.** For `ar_junipr_v1`, or any `coords_source ==
  "cell_center"`, the new keys are NaN — the "asked, unavailable" convention already at
  `:358-367`. A placeholder `ln_z = 0` means `z = 1` and must never be scored
  (`models/base.py:214-218`).
- `run_closure(..., per_jet=False)` → `metrics["per_jet"]` when True (shape precedent:
  `eval/clusters.py:370`), off by default so `eval_metrics.json` stays byte-identical.
  **This is what WP-0 consumes.**

### WP-2 — Make the silent-zero path impossible

In `lund_cloud` (`mbr.py:96-128`), with

    need = max(_COORD_GDIM[coords], 3 if weight == "z" else 2)

an `int`/`np.integer` cell element, or a row shorter than `need`, **raises** when
`need > 2` — naming the fix by knob and by function. An empty draw still yields an honestly
empty cloud and fabricates nothing. Add `needs_continuous_coords(coords, weight)` so
`report.py`, the tests and the scripts ask one function instead of re-deriving the rule.
Raise-not-warn matches `assert_cluster_metric_ok` and the `lnz_head='spline'` +
`lnz_support='legacy'` guard (`ar_junipr.py:161-170`).

> This **intentionally breaks `tests/test_mbr.py:103-108`**
> (`test_lund_cloud_empty_and_coords_gdim`), which asserts that a **cell chain** under
> `+lnz` yields `(2, 3)`. It passes today *only because the knob is inert* — it is the test
> that would have caught this had it been written against a coordinate table.

### WP-3 — Thread the coordinates (make `+lnz` / `+psi` / `weight="z"` functional)

- **`config.py`** — `decode.mbr_cloud_source: str = "cells"` (`"cells"` | `"coords"`), beside
  `mbr_coords`, plus the tolerant backfill dict. **Explicit, not implicit.**
  `make_per_jets_cluster_nb.py` and `make_inference_demo_cluster_nb.py` already pass
  `coords_by_draw` for winner decoration, so an implicit "coordinates were supplied ⇒ use
  them" would silently move those notebooks' numbers; it also makes the continuous-2-D arm of
  §4/WP-4 inexpressible. It lands in `metrics["decode"]` and the config hash, and
  `inert_decode_keys` picks it up free via the `mbr_` prefix.
- **`mbr.py`** — new `coords_for_draws(model, xf, nx, draws)`: one batched
  `sample_coordinates_many`, **unfiltered and index-aligned**, float64, raising with the
  family name when `has_continuous_coords` is False (the `skeleton_search_spec` precedent).
  `posterior_distances` gains `coords_by_draw` / `cloud_source` and **never draws** — it
  raises when it needs coordinates and has none. The draw happens once, in `mbr_select` /
  `mbr_select_stratified` / `mbr_cluster_set`, inside an `if needs_coords:` block, and the
  same array feeds both the clouds and `describe_cells(..., win_coords)` — so the tree shown
  is placed at the coordinates its cloud was built from, and a double draw is structurally
  impossible. **Keep the 4-tuple return**: six sites unpack it, two inside notebook-generator
  string literals.
- **`eval/closure.py`** — `run_closure(..., coords_by_jet=None)`; when needed, take the
  unfiltered per-jet coordinates once and reuse them for both `map_or_mbr` and the `cont_ok`
  block.
- **`eval/clusters.py`** — `run_cluster_diagnostics(..., coords_by_jet=None)`, forwarded into
  `posterior_distances`. `_truth_cloud` is already continuous — no change, but its docstring
  must record that under `"coords"` both sides finally are.
- **`eval/stability.py`** — **no change.** It consumes `D` alone; listing it would produce a
  diff that changes nothing.

> **A latent defect in the current default, fixed as a side effect.** `_truth_cloud`
> (`eval/clusters.py:217-224`) builds the truth from continuous `yraw` rows while every draw
> cloud is built from cell centres. Under the default `mbr_weight="kt"` the truth's point
> weights are `exp(v_continuous)` and the draws' are `exp(v_cell_centre)` — a per-point
> mismatch of `exp(±half_v) = exp(±0.1) ≈ [0.905, 1.105]` at the fielded `n_bins = 30`
> (larger on a coarser grid), plus a systematic inflation of the truth cloud's total mass by
> Jensen, which the EMD charges at `R·|ΔW|` with `R = 8.485`. **`d_top`, `d_best`, `d_mbr`,
> `d_nearest_draw` and gates G2′/G6/G7 sit on that mismatch today.** It does not touch
> `dlund_mbr` (a plain Euclidean distance, no EMD), so it is *not* the regression under
> investigation — but report `W_truth/W_draw` and `R·|ΔW|` against the typical `d` rather
> than assume it small. Under `mbr_weight="z"` the same mismatch is catastrophic.

### WP-4 — The measurement *(only if G-exists passes)*

`scripts/mbr_zaware_ab.py` — a **runner** (it runs models; `lnz_spline_gates.py` is a pure
JSON reader, and merging would put a multi-hour decode behind a table printer). Precedents:
`n_ceiling_probe.py`, `leading_estimators.py`. Plus a ~15-line reader block in
`lnz_spline_gates.py` so its guards table can never again quote `d(MBR)` without its z-aware
companion and its band.

**It is a 3×3, not a 2×2.** `+lnz` also **de-quantizes** `(u, v)` — a cloud built from
coordinate rows carries continuous `u, v`, not cell centres — so "cells-2D vs +lnz" changes
two things at once and is unattributable. Selection arms: `cells-2D` / `cont-2D` /
`cont-3D`. Rulers (free, all three come out of every pass): 2-D cell / 2-D continuous / 3-D.

**Arms.** Four pairs: `runs/lnz_spline/{spline_s0,s1,s2, contstop_spline_s0}` against
`runs/prod_test_v1/{v1_base_s0,s1,s2, v1_contstop_s0}`. `spline_s3/s4/s5` give a seed band
only (no control exists). `dvspline_*` is a **secondary block, reported and never scored** —
the `dv` spline is already measured-and-not-recommended, and mixing it prices two
interventions at once.

**Pairing discipline.** Per arm, once, in order: seed → `draws_by_jet` (one `sample_batch`
per jet, jets in file order) → `coords_by_jet` (one batched `coords_for_draws` per jet,
unfiltered) → three `run_closure` calls differing **only** in
`(mbr_cloud_source, mbr_coords)`. Within an arm the three selections see byte-identical
draws *and* coordinates. Across arms the models differ, so draws cannot be shared — pairing
is by jet index, certified by `dlund_identity`.

**Held fixed on every arm:** `R = 8.485`, `K = 200`, `mbr_n_candidates = 64`, `pot`, cpu
(cuda is a different RNG stream *and* different float kernels — `eval_prod_test_v1.sh`'s
standing rule). Stay in `mbr_select`: at `n_candidates = 64` the matrix is 64×200 and
`assert_cluster_metric_ok` rejects the cluster path, so routing through it would stop the
measurement being about the number that regressed.

---

## 5. The verdict rule — fixed before the run

**Pre-registered effect size.** At the fielded `z_cut = 0.1, β = 0` the `ln z` support is
`[ln 0.1, ln 0.5]`, width `ln 5 = 1.609`, against a Lund-plane diagonal of `6√2 = 8.485`.
Adding `ln z` can change a ground distance by **at most 19% of the diameter**, and the typical
`|Δ ln z|` between two draws of one jet is a fraction of that. **Every Δ is expected to be
O(0.01) on a distance of O(0.6) — the same order as the quantity under test.** This is a
small-effect measurement by construction, which is why every number carries a paired CI and
why `n = 300` may be underpowered.

**CONFIRM — "the `d(MBR)` regression is an artifact of a z-blind metric" — iff all of:**

- **C1** `Δ(dlund3_mbr_cont) ≤ 0` on ≥ 3/4, pooled paired CI containing or below 0. *No
  regression survives a ruler that sees `ln z`.*
- **C2** `Δ(dlnz_mbr) < 0` on 4/4, CI excluding 0 on ≥ 3/4. **Load-bearing**: without it,
  "the ruler cannot see the improvement" describes an improvement that is not there.
- **C3** `Δ(dlund_mbr_cont) > 0` with CI excluding 0 on ≤ 1/4. *The `(u,v)` half is not
  significantly worse.*
- **C4** difference-in-differences `[Δ₃D(cont-3D) − Δ₃D(cont-2D)]` negative on ≥ 3/4 **and**
  `winner_moved_rate ≥ 0.05`. *A z-aware selection helps the spline arms more than the
  controls.*

**FALSIFY iff any of:**

- **F1** `Δ(dlund3_mbr_cont) > 0`, CI excluding 0, on ≥ 3/4 — the spline arms genuinely place
  the leading emission worse under a ruler that sees everything.
- **F2** `Δ(dlnz_mbr) ≥ 0` on ≥ 2/4 — there is no hidden `ln z` gain to miss.
- **F3** `Δ(dlund_mbr_cont) > 0`, CI excluding 0, on ≥ 3/4 — the regression lives in
  coordinates the fielded ruler already measures, i.e. it is real and visible.

**INCONCLUSIVE** otherwise, and specifically when C1–C3 point different ways or every CI
straddles 0.

**NOT-SCORED clause, declared now:** if `winner_moved_rate < 0.05` on ≥ 3/4 arms, **C4 is not
scored** — a selection that cannot move cannot explain anything — and the verdict rests on
C1–C3. Given the effect size above this is the *expected* outcome, and it is stated here
rather than discovered afterwards. (Same move as G2′'s silhouette precondition.)

**Guards printed beside the verdict**, because a metric improvement can be bought with
something else: support audit at 0.00000% (the coordinates now enter the distances, so a
leaked `z` would too), `ground_diameter` vs `R` (8.485 against 9.849/2 = 4.925 under `+lnz`,
asserted via `assert_cluster_metric_ok`, never assumed), G-repro, `winner_moved_rate`,
`leading_cell_moved_rate` against the **controls'** own rates, and `W_truth/W_draw`.

**Anti-circularity.** No threshold, `R`, `K` or bandwidth is tuned against any of these
numbers. An `R`-sensitivity arm at the 3-D diameter is **reported and never used to select** —
the same rule that kept `K = 8` on the spline and `γ = 0.10` on the cluster bandwidth.

---

## 6. WP-5 — tests

`tests/test_mbr.py` — rewrite `test_lund_cloud_empty_and_coords_gdim` into three (cell chain
+ 2-D ⇒ `(2,2)`; coordinate table + each `coords` ⇒ `(m,g)` with the real columns; cell chain
+ `+lnz`/`+psi` ⇒ **raises**), the docstring stating the old form passed only because the
knob was inert. Add: `weight="z"` raises on a cell chain and equals `exp(lnz)` on a table
(the "silently identical to `unit`" regression); a short row raises rather than pads;
**`test_lnz_is_no_longer_inert`** — draws with identical cells but differing `ln z` give
`|D|.max() == 0` at 2-D and a *different winner* at `+lnz`; supplying `coords_by_draw` under
`"cells"` is bit-identical; `"coords"` de-quantizes to the supplied `(u,v)`;
`posterior_distances` raises naming `coords_for_draws`; `coords_for_draws` is
index-aligned/unfiltered over an empty draw and raises **by family name** on `ar_junipr_v1`;
coordinates are drawn **exactly once** (monkeypatch count) and the winner's reported nodes
are the rows its cloud used.

`tests/test_clusters.py` — `assert_cluster_metric_ok` is coords-dependent (an `R` passing at
2-D failing at `+lnz`); truth and draws are in the same representation under `+lnz` (both
column 2 non-constant); `|W_truth − W_draw|/W` ≈ 0 under `"coords"` and demonstrably nonzero
under `"cells"` — pinning both the defect and its fix.

`tests/test_config.py` — add `mbr_cloud_source` to the exact-set assertion (it fails
otherwise) and assert the default plus the old-snapshot backfill.

`tests/test_shared_draws.py` — `run_closure` reuses supplied coordinates (monkeypatch the
sampler to raise); **with the switch off the existing filtered call is still taken**, which
is what pins the bit-identity hazard of §7.1.

New ruler tests — `_leading_coords` default `ncols=2`; the new keys are NaN not 0 without
coordinates; a hand-computed 3-D value (hypot of the three components, `dlnz` the third
alone); `decode_headline` unmoved and every pre-existing key intact.

## 7. Risks

1. **Bit-identity break via the unfiltered coordinate call — the largest.**
   `run_closure`'s existing call is *filtered* (`[list(d) for d in draws if len(d)]`,
   `closure.py:260`); `ar_junipr.sample_coordinates_many` pads to `L_max` over the list it is
   given, so unfiltering changes the block shape, reorders RNG consumption, and moves
   `dlund_*_cont` and the psi block on the **default** path. Mitigation: the unfiltered call
   lives strictly behind the switch, pinned by a test. `sample_coordinates_many`'s own
   docstring already warns that draws are not bit-comparable across such a change.
2. **Underpower.** Four pairs, p ≥ 0.125 on the sign test, effect O(0.01) on O(0.6).
   Mitigation: per-jet paired bootstrap (the dominant power gain, and free), the 1000-jet
   escalation declared in advance, and INCONCLUSIVE as a legitimate outcome.
3. **The experiment is powerless by physics** (`winner_moved_rate ≈ 0`). Mitigated by
   pre-registering it as a precondition and by the NOT-SCORED clause.
4. **De-quantization confound** — handled by the third selection arm; dropping it makes
   "+lnz vs cells" two changes.
5. **`R` semantics under a 3-D ground metric** — `R = 8.485` clears KMT's bound at both
   diameters, but the imbalance term's *relative* weight shifts. Fixed on every arm;
   sensitivity reported, never used to select.
6. **TARP under `+lnz` raises between WP-2 and the (out-of-scope) reference-pool fix.** That
   is loud-and-correct, but land them together or state the window. No config on disk sets
   `mbr_coords` to anything but `lnDR_lnkt` (`configs/decode/default.yaml:16`,
   `presets/decode/mbr_study.yaml:10`), so no committed artifact is affected — though any
   past run that *did* set it produced 2-D numbers under a 3-D label.

## 8. Cost

| item | per jet | per arm (300 jets) |
|---|---|---|
| WP-1 ruler | ~10 numpy ops | negligible |
| coordinate draw | 1 batched `sample_coordinates_many` | the call `closure_continuous` already makes |
| 3-D ground metric | `_ground` over 3 columns not 2; OT solve unchanged | est. +5–10% per pass |
| one decode pass | 64 × 200 = 12 800 `ot.emd2` solves | ≈ 20–30 min, cpu |

**WP-0** ≈ one pass-set over 8 arms ≈ 40–60 min at concurrency 6. **WP-4** = 24 passes ≈ 2–3 h.
The 1000-jet escalation is ~3.3×. All inside a day.

## 9. Out of scope — stated so it is not assumed

- **The structural layer** (§2c): a `ln z`-aware cell grid, or the per-node joint coordinate
  density. `PLAN_lnz_spline_head.md` §7.3 carries the restated trigger and nothing here
  fires it.
- **Retraining.** Everything is decode-time on existing checkpoints.
- **TARP under `+lnz`** — `_tarp_reference_pool` (`eval/calibration.py:206-228`) would have to
  carry reference coordinates, and the truth would move from `yc` to `yraw`, a deliberate
  change in what is being tested. Separate work package if wanted.
- **Any claim about the model from this measurement.** WP-4 prices a *metric* and a *ruler*.
  If the regression turns out real (F1/F3), that is a finding about the decode layer, not a
  reason to un-field `lnz_head="spline"` — which was fielded on NLL, PIT and support, on
  numbers this document does not touch.
