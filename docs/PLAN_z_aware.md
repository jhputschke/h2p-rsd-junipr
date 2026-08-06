# PLAN — a `ln z`-aware decode metric, and whether the spline's `d(MBR)` regression is an artifact of not having one

Status: **WP-0 run and closed; WP-1, WP-2 and WP-5 shipped; WP-3/WP-4 NOT RUN** (gated on
WP-0, which did not fire). Branch `zAwareMetric`, 2026-08-06 — see **§11**. Follows directly
from `PLAN_lnz_spline_head.md` §6.2 and `SUMMARY_Model_Status.md` §2.5, which record a
finding this document exists to *test* rather than assert: the RQ-spline `ln z` head improved
held-out NLL and `pit_ks_max` on 4/4 arms while `d(MBR)` got **worse** on 4/4, and the
explanation offered was "the MBR metric runs on `lnDR_lnkt` and cannot see `ln z`".

**Result in one line:** *there is no regression to explain* — the per-jet paired CI contains
0 on 4/4 pairs at the published tier, and at the pre-declared 1000-jet escalation the 4/4
sign consistency itself does not survive (3/4, with one pair significantly **better**, pooled
−0.0014 [−0.0089, +0.0062]). **INCONCLUSIVE-BY-CONSTRUCTION**, twice, which by §4's own
pre-registered consequence rewrites the `SUMMARY` sentence rather than explaining it (§11).

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

> Every file:line below is the code **as it stood when this was written**, and is left
> unedited because it is the evidence the plan was built on. Layers (a) and (b) have since
> changed — see §11.3 — so the line numbers no longer resolve; the file and function names
> still do.

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

> **RUN — see §11.1.** G-repro 8/8 (exactly), G-pair 4/4, **G-exists FAILS at both tiers**.
> The escalation below was fired once, as declared, and the phenomenon did not resolve. The
> text of this work package is left as it was written.

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

> **SHIPPED** — `eval/closure.py`, pinned by `tests/test_zaware_ruler.py`. §11.3.

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

> **SHIPPED** — `lund_cloud` raises; `needs_continuous_coords` / `cloud_columns_needed`
> are the shared predicate. `report.py` was deliberately left alone — §11.3 says why.

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

> **NOT RUN** — gated on WP-0 by §10, which did not fire. The `_truth_cloud` `kt`-weight
> mismatch called out below is therefore **still open**: it is fixed as a side effect of
> this threading, and nothing else here touches it (§11.3).

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

> **NOT RUN** — G-exists did not pass (§11.1). `scripts/mbr_zaware_ab.py` exists and ran
> WP-0; the 3×3 selection grid it would add needs WP-3 first.

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

> **NOT SCORED.** WP-4 never ran: §10's ordering gates it on WP-0, and WP-0 found no
> phenomenon (§11.1). C1/C2/C3 are nevertheless *measurable* from WP-0's own pass, since
> WP-1's three rulers are free, and §11.2 reports them at the fielded `cells-2D` selection —
> **as statistics, not as a verdict.** C4 is a difference-in-differences across selection
> arms that WP-3 has not built, and scoring three of four pre-registered criteria is the move
> pre-registration exists to prevent. This rule stays as written, available unchanged if the
> question is reopened.

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

> **SHIPPED for what shipped.** `tests/test_mbr.py` is rewritten as described; the new
> ruler tests live in `tests/test_zaware_ruler.py`. The clauses below that test WP-3's
> plumbing (`coords_for_draws`, `mbr_cloud_source`, `test_config.py`'s exact set,
> `test_clusters.py`'s `W_truth/W_draw`) are **not** written, because that code does not
> exist. §11.3.

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
  density. `PLAN_lnz_spline_head.md` §7.3 owns it and nothing here fires it. **Note its
  status changed on 2026-08-05**: that document's §9.4 *withdrew* the restated trigger this
  bullet used to point at and promoted §7.3 from deferred to **indicated**, after a second
  per-coordinate fix failed. §10 below records how the two documents divide the work and
  which runs first.
- **Retraining.** Everything is decode-time on existing checkpoints.
- **TARP under `+lnz`** — `_tarp_reference_pool` (`eval/calibration.py:206-228`) would have to
  carry reference coordinates, and the truth would move from `yc` to `yraw`, a deliberate
  change in what is being tested. Separate work package if wanted.
- **Any claim about the model from this measurement.** WP-4 prices a *metric* and a *ruler*.
  If the regression turns out real (F1/F3), that is a finding about the decode layer, not a
  reason to un-field `lnz_head="spline"` — which was fielded on NLL, PIT and support, on
  numbers this document does not touch.

---

## 10. Relationship to `PLAN_lnz_spline_head.md` §7.3 — the split, and the order

Recorded so it does not have to be re-derived. The two documents look adjacent and change
**different layers**; §2's three-layer split is exactly the boundary:

| layer | blind because | owner |
|---|---|---|
| **(a) selection** | `mbr_coords="+lnz"` appends a constant-zero column; `mbr_weight="z"` is silently `unit` | **this document** |
| **(b) scoring** | `dlund_mbr` compares cell centres; the `*_cont` block slices `[..., :2]` and has no MBR row | **this document** |
| **(c) structural** | the grid discretizes `(u, v)`; `ln z` is conditionally independent of `du` by construction, so the identity is encoded nowhere | `PLAN_lnz_spline_head.md` §7.3 |

**This document fixes the ruler; §7.3 fixes the model.** Neither substitutes for the other:
nothing here can move a coordinate's PIT — that is computed from the model's own CDF and
never consults `mbr_coords` — and no joint density can make a blind metric see.

**Order: WP-0 → WP-2 → §7.3 → WP-3/WP-4.**

> **Where this stands (2026-08-06).** WP-0 **done** and WP-2 **done**, so the pointer is at
> **§7.3**. WP-3/WP-4 are **cancelled by this ordering's own condition** — WP-0 found no
> phenomenon (§11.1) — rather than merely deferred. The first bullet below turned out to be
> the operative one: the question dissolved.

- **WP-0 first** because it is nearly free and may dissolve the question entirely. If there
  is no regression to explain, `PLAN_lnz_spline_head.md` §6.2's `d(MBR)` caveat evaporates
  and one of the two open concerns about `lnz_head="spline"` closes at the cost of one
  decode pass per arm.
- **WP-2 next regardless of WP-0's verdict.** The silent-zero path is a live trap for
  whoever next reaches for `mbr_coords="+lnz"` expecting it to do something. Fixing a knob
  that cannot be switched on is worth doing on its own account.
- **§7.3 next.** It is the expensive one — a new density *and* retraining, against this
  document's decode-time-on-existing-checkpoints — and its case rests on the *third*
  proposed mechanism for `dv` after two were falsified, whereas this document's case rests
  on a documented code fact. Cheap and certain before expensive and hypothesised.
- **WP-3/WP-4 last**, and only if WP-0 found a phenomenon.

**Not a hard dependency, and the one place they touch.** §7.3's *primary* pre-registered
read is `dv`'s marginal PIT, which is ruler-independent — so this document is not a
prerequisite for §7.3's verdict, and running the two in parallel is defensible. They meet at
exactly one row: §7.3's `d(MBR)`, which should not be read until the ruler question here is
settled. `PLAN_lnz_spline_head.md` §6.2 read that row against a blind ruler and had to carry
a caveat for it; §9.5 there records the same split from the other side.

---

# §11. RESULT — there is no regression to explain

Run 2026-08-06 on branch `zAwareMetric`. Runner: `scripts/mbr_zaware_ab.py`. Artifacts:
`runs/zaware_wp0/full-20260806-131513/` (published tier) and
`runs/zaware_wp0/esc1000-20260806-132207/` (the escalation §4 declared in advance).

**No rule above was edited after the numbers arrived.** §4's gates and §5's criteria are
verbatim as pre-registered; the only additions to those sections are the status banners that
point here, and they change no clause. The result lives in this section, which is additive.

## 11.1 WP-0 — the gates

**G-repro: PASS, 8/8, and exactly.** Every arm re-measured its committed `dlund_mbr` to all
four printed digits (0.00% on all eight, 247/247 kept jets), which is stronger than the 0.5%
the gate asked for and stronger than the ceiling probe's own sanity row (0.03%). The reason
is worth recording: `run_arm` mirrors `cli.py`'s call order — seed, build, load, datamodule,
closure — so the global RNG stream is consumed the same way rather than merely re-seeded.
**A re-measurement this exact means every number below is about the models, not the harness.**

**G-pair: PASS, 4/4.** `max |Δ dlund_identity| = 0` within every pair, on 226 jets of the 247
at 300 and 769 of the 839 at 1000. The remainder are jets with an **empty hadron `x`** —
checked against the file rather than assumed: all 21 of the 300-jet tier's have
`len(x) == 0`, so `identity(x)` has no leading emission and the distance is NaN on both
sides. The pairing §3 asserted is exact, so the per-jet analysis is legitimate.

**Why the paired `n` is 241–242, not 247.** `dlund_mbr` exists only where the MBR estimate is
non-empty; 3 of the 247 kept jets decode to the empty tree on the spline side and a few more
on the control side, and a paired delta needs both. Those jets are **dropped, not zero-filled**
— an empty estimate has no leading emission to score, and the alternative would be inventing
one. `n` is printed on every row so the population is never implicit.

**G-exists: FAILS. Verdict INCONCLUSIVE-BY-CONSTRUCTION.**

| Δ(`dlund_mbr`) = spline − control | n | 300 jets, paired BCa 95% | n | 1000 jets, paired BCa 95% |
|---|---:|---|---:|---|
| `spline_s0` − `v1_base_s0` | 242 | **+0.0039** [−0.0222, +0.0309] | 822 | **−0.0211** [−0.0357, **−0.0065**] |
| `spline_s1` − `v1_base_s1` | 241 | **+0.0061** [−0.0178, +0.0312] | 817 | **+0.0113** [−0.0029, +0.0262] |
| `spline_s2` − `v1_base_s2` | 241 | **+0.0139** [−0.0120, +0.0402] | 817 | **+0.0040** [−0.0108, +0.0191] |
| `contstop_spline_s0` − `v1_contstop_s0` | 241 | **+0.0032** [−0.0259, +0.0321] | 817 | **+0.0001** [−0.0155, +0.0148] |
| pooled *(reported, never the rule)* | 965 | +0.0068 [−0.0068, +0.0197] | 3273 | −0.0014 [−0.0089, +0.0062] |
| | | **0/4 CIs exclude 0**; positive 4/4 | | **0/4 exclude 0 upward**, 1/4 downward; positive 3/4 |

Three readings, and the second is the one that settles it.

1. **The +0.005 is inside its own per-jet noise.** Every 300-jet CI is ~±0.03 wide — six
   times the effect. §5's pre-registered effect-size note said exactly this would happen
   (`ln z` can move a ground distance by at most 19% of the diameter, so every Δ is O(0.01)
   on a distance of O(0.6)), and it is why the note was written before the run.
2. **The 4/4 sign consistency — the entire basis of §2.5's claim — does not survive more
   jets.** At 1000 it is 3/4, and the arm that flips does not merely flip: `spline_s0` is
   **significantly better**, −0.0211 [−0.0357, −0.0065]. A pattern whose only evidence was
   "4 out of 4, p = 0.125 at its floor" and which reverses on one of four arms when the
   sample triples is a sampling artifact, not a phenomenon.
3. **Pooled over 3273 paired jets the difference is −0.0014 [−0.0089, +0.0062]** — bounded
   inside ±0.01, i.e. inside a fiftieth of the 0.6 it is a difference of. Whatever the
   spline does to the fielded decode metric, it is smaller than this experiment can see and
   smaller than the 0.020 gap by which `dlund_mbr` already loses to the free one-node
   `dlund_posterior_medoid` (§3, reading 4) on *both* families.

**The controls behave as §3 predicted**, which is what makes the null readable rather than
merely quiet: `dlund_posterior_medoid` is 2/4 positive at both tiers (pooled −0.0001
[−0.0049, +0.0047] at 1000) and `dlund_posterior_mode` 2/4. No estimator built from the same
draws moved. The cell posterior did not degrade — and now neither did the MBR selection off
it.

**Consequence, taken from §4 rather than invented here.** The escalation was fired once, as
declared; it did not resolve the phenomenon; so §2.5's sentence is **rewritten**, not
explained. An explanation is not owed for a number that is not resolved. Done in
`SUMMARY_Model_Status.md` §2.5 and `PLAN_lnz_spline_head.md` §6.2.

## 11.2 The free ruler rows — and why they do NOT constitute §5's verdict

WP-1's three rulers come out of every pass, so the C1/C2/C3 *statistics* are in hand at the
**fielded `cells-2D` selection**. They are reported because they are free and because they
bear on the explanation's second half; they are **not** scored against §5, because §5 is
WP-4's verdict rule, WP-4 is a 3×3 over *selection* arms that WP-3 has not built, and C4 is a
difference-in-differences across those arms. Scoring three of four pre-registered criteria
and declaring a verdict is the move pre-registration exists to prevent.

| pooled Δ, 1000 jets (n = 3273) | mean | paired BCa 95% | arms positive |
|---|---:|---|---:|
| `dlund_mbr` — cell centres (the fielded headline) | −0.0014 | [−0.0089, +0.0062] | 3/4 |
| `dlund_mbr_cont` — the winner's own `(u, v)`, off the grid | −0.0019 | [−0.0097, +0.0058] | 3/4 |
| `dlund3_mbr_cont` — the same emission with `ln z` restored | −0.0016 | [−0.0122, +0.0092] | 3/4 |
| `dlnz_mbr` — `\|Δ ln z\|` alone | +0.0003 | [−0.0109, +0.0117] | 2/4 |

**The load-bearing half of §2.5's sentence is the half that fails.** C2 is the criterion §5
marked *load-bearing*: "without it, 'the ruler cannot see the improvement' describes an
improvement that is not there." At 300 jets Δ(`dlnz_mbr`) is positive on 4/4 — the wrong
sign for C2 and the firing condition of F2 — and at 1000 jets it is flat at +0.0003
[−0.0109, +0.0117]. **The MBR winner's own drawn `ln z` is no closer to the truth under the
spline than under its control.** The head's `ln z` marginal is unambiguously better (KS
0.47–1.04× critical against 1.05–2.16×, `SUMMARY` §2.5); that gain simply does not reach the
*selected tree's leading emission*, which §2 already predicted for a structural reason —
`sample_batch` returns **cell chains**, so the head reaches the fielded decode only through
training, via a shared trunk that moves the cell posterior.

So the ruler was indeed blind, and building it changed nothing, because there was nothing on
the other side of it to see. That is a more useful outcome than a confirmation would have
been: it removes the residual `d(MBR)` caveat from `lnz_head="spline"` on *both* halves at
once — no regression to explain, and no hidden gain being missed.

**One thing to put on the record now rather than rediscover later.** F2 — "`Δ(dlnz_mbr) ≥ 0`
on ≥ 2/4" — is the one §5 criterion that needs **no selection arm at all**: it is a statement
about the ruler applied to whatever tree was selected, and the fielded selection is a
selection. Its condition is met at **both** tiers (4/4 at 300, 2/4 at 1000). C2 is its
mirror and fails at both. So if the question is ever reopened, **§5's CONFIRM branch is
already unreachable** — C2 is load-bearing and cannot be recovered by building WP-3 — and
what WP-4 could still decide is only whether the outcome is FALSIFY or INCONCLUSIVE, on
criteria (C1/C3/F1/F3) whose CIs all straddle 0 today. Stating that is not the same as
scoring the rule: the verdict stays unscored, and this paragraph is the reason a future
reader should not expect much from running it.

## 11.3 What shipped, and what did not

| WP | status | note |
|---|---|---|
| **WP-0** | **run, closed** | §11.1. `scripts/mbr_zaware_ab.py`, two tiers, 16 decode passes |
| **WP-1** | **shipped** | `eval/closure.py`: `_leading_row`, `_leading_coords(ncols=2)`, `dlund3_*_cont`, `dlnz_*`, the MBR row the continuous block never had, `run_closure(per_jet=)`. Additive only — verified by re-running the closure metric dict before and after and diffing: nothing but additions |
| **WP-2** | **shipped** | `lund_cloud` raises on a cell chain (or a short row) whenever the config reads more than two columns; `needs_continuous_coords` / `cloud_columns_needed` are the one place the rule lives |
| **WP-5** | **shipped** | `tests/test_mbr.py` — the old `test_lund_cloud_empty_and_coords_gdim` is replaced by **7** test functions (12 cases) including `test_lnz_is_no_longer_inert_in_the_distance`; new `tests/test_zaware_ruler.py`, **12** functions |
| **WP-3** | **NOT RUN** | gated on G-exists by §10's ordering, and it did not fire |
| **WP-4** | **NOT RUN** | same gate. §5 is therefore unscored and stays pre-registered, available if the question is ever reopened |

**One thing taken from WP-4 anyway.** WP-4 asked for "a ~15-line reader block in
`lnz_spline_gates.py` so its guards table can never again quote `d(MBR)` without its z-aware
companion and its band." WP-4 did not run, but that requirement became *more* pressing, not
less: the guards table prints four unpaired means, and §6.2 of `PLAN_lnz_spline_head.md` read
a since-withdrawn conclusion off exactly that column. `print_dmbr_band` is therefore in, fed
by WP-0's own artifact, and it selects the run at the **matching tier** rather than the newest
— the 1000-jet escalation scores a different population and must not be quoted beside a
300-jet column. With no artifact present it says so out loud instead of printing the bare
means alone.

**Two things WP-2 did *not* do, stated so they are not assumed.** `report.py` is untouched:
the plan expected it to ask `needs_continuous_coords`, but `inert_decode_keys` never claimed
`mbr_coords` was inert under an MBR decode, so there was no wrong entry to fix — and adding a
*fatal* row to a list titled "decode knobs that did NOT reach these numbers" would trade one
confusion for another. And the `_truth_cloud` `kt`-weight mismatch of §4/WP-3 — truth weighted
by `exp(v_continuous)` against draws weighted by `exp(v_cell_centre)`, which
`d_top`/`d_best`/`d_mbr` and gates G2′/G6/G7 sit on today — **is still open**, because it is
fixed as a side effect of WP-3's threading and WP-3 did not run. It is a real defect and it is
unrelated to the question this document answered.

## 11.4 What this closes

- **`PLAN_lnz_spline_head.md` §6.2's `d(MBR)` caveat is discharged.** One of the two open
  concerns about `lnz_head="spline"` is closed, at the cost of 16 decode passes and no model
  change. The other (G3 formally PARTIAL at 5/6 seeds) is untouched.
- **`SUMMARY_Model_Status.md` §4.1(4) is done** — and its answer to "is the regression even
  resolved" is *no*, which was one of the two outcomes it named.
- **`mbr_coords="+lnz"` can no longer be silently inert.** It raises. Whether it should be
  made to *work* is now a question with no measurement behind it, which is a fair place for
  it to sit: nothing in §11 wants it.
- **§7.3 of `PLAN_lnz_spline_head.md` is unaffected and is next**, exactly as §10 ordered.
  Its primary read is `dv`'s marginal PIT, which never consulted `mbr_coords`; the one row
  the two documents shared — its `d(MBR)` — may now be read straight, with §11.1's band
  (±0.01 pooled over 3273 jets) as the resolution of that ruler rather than a caveat.
- **But it also strengthens §9.4's caveat *against* firing §7.3 at all**, which is worth
  saying because it points the other way from the bullet above. That caveat was written
  conditionally — "if the `ln z`-aware decode metric shows the `d(MBR)` regression is an
  artifact, the fielded product is in good shape on every axis except a 10% miss on one
  coordinate's marginal" — and the condition resolved in the direction that strengthens it,
  by a route neither branch anticipated. The decision to spend the most expensive change in
  this line of work on `dv`'s 2–13% marginal miss is now being made against a product with
  **one** blemish rather than two. That is a judgement call, not a conclusion the data
  forces, and §11 makes it a slightly harder one.

**Recorded in:** `SUMMARY_Model_Status.md` §2.7 (new), §2.5 (row withdrawn), §3, §4.1(4),
§4.5(1) and §5; `PLAN_lnz_spline_head.md` §6.2 (withdrawn), §6.5, §8.5, §9.4 and §9.5a;
`CONFIGURATION.md` (the new closure keys, and the raise); `USAGE.md`.

---

# §12. PRE-REGISTRATION — is the `ln z`-blind **selection** worth fixing?

Written 2026-08-06, **before the arms it reads run**. This is a *different question* from §5
and must not be confused with it.

**§5 asked** whether `d(MBR)`'s regression was an artifact of a z-blind metric. WP-0 killed
its premise: there is no regression (§11.1). §5 stays unscored and untouched.

**§12 asks** something WP-0 did not address and could not have: the decode reports the MBR
winner's `ln z`, which is a **single draw** from `q(ln z | cells, x)`, where a centrality-based
estimate off the *same* draws is measurably closer to the truth. Measured within-arm, paired,
1000 jets:

| within-arm paired Δ | result |
|---|---|
| MBR winner's `ln z` − identity(x)'s `ln z` | −0.008…+0.008, **0/8 CIs exclude 0** |
| posterior geo-median's `ln z` − MBR winner's `ln z` | **−0.047 … −0.071, 8/8 CIs exclude 0** |

So `≈0.065` of `|Δ ln z|` — about 16% of the 0.41 baseline — sits between what the decode
reports and what the same posterior already knows. The question is how much of that a
selection **restricted to the drawn pool** (`H = {pool}`, the standing discipline) can
actually recover, and whether it costs the `(u, v)` half.

This is the same shape of argument `medoid_cell` already won one coordinate over — *"the mode
is the estimator for a loss nobody is measuring"*, mode 1.030× identity vs medoid 0.944× — and
it is **family-independent**: it is a decode fix, not a spline result. Nothing here reopens
§2.5.

## 12.1 The measurement

Per arm, ONE pass, in order: seed → `draws_by_jet` → `coords_by_jet` (one batched
`sample_coordinates_many`, index-aligned) → **three selections off byte-identical draws and
coordinates, differing only in the ground metric**:

| arm | cloud source | gdim | isolates |
|---|---|---|---|
| `cells-2D` | cell centres | 2 | **the fielded selection** |
| `cont-2D` | coordinate rows | 2 | de-quantization alone |
| `cont-3D` | coordinate rows | 3 (`+lnz`) | de-quantization **+** `ln z` |

`cont-2D` is not optional: "+lnz vs cells" changes *two* things at once (§4/WP-4), and without
it a win is unattributable.

Two reference points bound what any pool-restricted rule could reach, both free numpy:

- **`pool-medoid(ln z)`** — the draw minimising mean `|Δ ln z|` to the other draws. The best a
  selection obeying `H = {pool}` can do **on the `ln z` axis alone**, i.e. the realistic ceiling.
- **`free-median(ln z)`** — the unrestricted L1 Bayes point. The 0.065 figure above.

Held fixed at the fielded values: `K = 200`, `mbr_n_candidates = 64`, `pot`, `R = 8.485`,
`β = 1`, `weight = kt`, `lnkt_cut` inherited, cpu. 1000 jets, 8 arms.

## 12.2 The verdict rule — fixed before the run

**BUILD WP-3 iff all three hold:**

- **B1** `Δ(dlnz) = cont-3D − cells-2D` is **≤ −0.020** on ≥ 3/4 spline arms, with the paired
  95% CI excluding 0 on ≥ 3/4. *(0.020 is ≈5% of the 0.41 baseline and 2× this ruler's own
  pooled resolution from §11.1; the decode-layer precedent for "worth fielding" is the
  medoid's ~8% over the mode.)*
- **B2** `Δ(dlund_cont 2-D) = cont-3D − cells-2D` does **not** exceed **+0.020** with CI
  excluding 0 on ≥ 2/4. *The `(u, v)` half must not be bought with `ln z`* — C3's spirit,
  and the reason `dlund_mbr` stays the headline.
- **B3** `winner_moved_rate ≥ 0.05` on ≥ 3/4. *A selection that cannot move cannot deliver
  anything* — the §5 NOT-SCORED clause, restated as a precondition rather than an excuse.

**DON'T BUILD otherwise.** If B1 fails specifically, the finding is that the 0.065 lever is
**real but unreachable by this mechanism** — the same shape as §2.4's oracle-N, and it should
be recorded as a stop-sign so the obvious fix is not re-proposed.

**Attribution clause, declared now:** if `cont-2D` alone accounts for most of the Δ, the
effect is **de-quantization**, not `ln z`, and `+lnz` is not what earned it. Reported either
way, and it changes what gets built.

**Guards printed beside the verdict:** `ground_diameter` vs `R` at both dimensions (8.485
against 4.243 and 4.925 — clears, asserted not assumed), the pool-restricted and free `ln z`
ceilings, `winner_moved_rate`, `leading_cell_moved_rate`, and the absolute `|Δ ln z|` of every
arm so the deltas are readable against their own scale.

**Anti-circularity:** `R`, `K`, `mbr_n_candidates` and `lnkt_cut` are the fielded values and
are not tuned against any number this produces.

---

# §13. RESULT §12 — **BUILD.** The `+lnz` selection recovers ~60% of the ceiling

Run 2026-08-06, branch `zAwareMetric`. Runner: `scripts/zaware_selection_ceiling.py`,
8 arms × 1000 jets × 3 selections off byte-identical draws and coordinates, ~22 min/arm.
Artifact: `runs/zaware_sel/full-20260806-143047/ceiling.json`. §12.2's B1/B2/B3 were
committed (`e5e3d38`) before this file existed.

## 13.1 The ceiling, and how much of it is reachable

Absolute `|Δ ln z|` of the selected tree's leading emission, 839 kept jets per arm:

| arm | cells-2D *(fielded)* | cont-2D | **cont-3D** | pool-medoid | free-median |
|---|---:|---:|---:|---:|---:|
| `spline_s0` | 0.4203 | 0.4186 | **0.3735** | 0.3475 | 0.3474 |
| `spline_s1` | 0.4017 | 0.4115 | **0.3754** | 0.3481 | 0.3481 |
| `spline_s2` | 0.4186 | 0.4168 | **0.3722** | 0.3451 | 0.3449 |
| `contstop_spline_s0` | 0.4169 | 0.4056 | **0.3723** | 0.3456 | 0.3457 |
| `v1_base_s0` | 0.4076 | 0.4119 | **0.3716** | 0.3464 | 0.3464 |
| `v1_base_s1` | 0.4038 | 0.4024 | **0.3710** | 0.3421 | 0.3423 |
| `v1_base_s2` | 0.4150 | 0.4125 | **0.3737** | 0.3482 | 0.3482 |
| `v1_contstop_s0` | 0.3946 | 0.3958 | **0.3608** | 0.3468 | 0.3464 |

**The pool restriction costs nothing.** `pool-medoid` equals `free-median` to four decimals
on 8/8 arms — at `K = 200` the drawn pool already realises its own `ln z` median, so
`H = {pool}` is free on this axis. The ceiling is therefore the full **−0.065**, not some
fraction of it.

**`+lnz` recovers 47–70% of it (mean ≈ 59%).** Paired BCa, per jet:

| Δ = selection − `cells-2D` | `cont-3D` (B1) | `cont-2D` *(attribution)* | `pool-medoid` *(ceiling)* |
|---|---|---|---|
| `spline_s0` | **−0.0472** [−0.0637, −0.0312] | −0.0017 [−0.0170, +0.0139] | −0.0735 [−0.0907, −0.0567] |
| `spline_s1` | **−0.0256** [−0.0440, −0.0078] | +0.0102 [−0.0087, +0.0281] | −0.0549 [−0.0734, −0.0367] |
| `spline_s2` | **−0.0464** [−0.0638, −0.0299] | −0.0018 [−0.0191, +0.0156] | −0.0747 [−0.0925, −0.0573] |
| `contstop_spline_s0` | **−0.0444** [−0.0617, −0.0266] | −0.0111 [−0.0275, +0.0052] | −0.0719 [−0.0906, −0.0535] |
| `v1_base_s0` | **−0.0352** [−0.0529, −0.0175] | +0.0042 [−0.0124, +0.0211] | −0.0627 [−0.0806, −0.0450] |
| `v1_base_s1` | **−0.0313** [−0.0480, −0.0147] | −0.0014 [−0.0178, +0.0145] | −0.0627 [−0.0800, −0.0453] |
| `v1_base_s2` | **−0.0413** [−0.0589, −0.0237] | −0.0025 [−0.0218, +0.0165] | −0.0688 [−0.0867, −0.0509] |
| `v1_contstop_s0` | **−0.0347** [−0.0528, −0.0176] | +0.0007 [−0.0157, +0.0173] | −0.0499 [−0.0686, −0.0324] |
| | **8/8 CIs exclude 0** | **0/8 exclude 0** | 8/8 exclude 0 |

**The attribution clause resolves cleanly: it is `ln z`, not de-quantization.** `cont-2D`
uses exactly the same continuous coordinate rows and delivers **nothing** on this axis —
0/8 significant, signs mixed. The entire gain arrives with the third column. That is why
§12.1 insisted on the third arm; a 2×2 would have credited it to the wrong change.

## 13.2 B1 / B2 / B3

| | rule | measured | |
|---|---|---|---|
| **B1** | `Δ(dlnz) ≤ −0.020` on ≥3/4 spline arms, CI excluding 0 on ≥3/4 | gain **4/4**, significant **4/4** (8/8 over all arms) | **PASS** |
| **B2** | `Δ(dlund` 2-D cont`)` not > +0.020 with CI excluding 0 on ≥2/4 | violations **0/4**; range −0.0176…+0.0082, only one arm significant and it is *better* | **PASS** |
| **B3** | `winner_moved_rate ≥ 0.05` on ≥3/4 | **58.8–62.5%** on 8/8 | **PASS** |

Guards, asserted not assumed: `ground_diameter` 8.4853 (2-D) and 9.8489 (`+lnz`), KMT bounds
4.2426 and 4.9244, `R = 8.485` clears both. `leading_cell_moved_rate` 44–49%.

**§5's NOT-SCORED clause predicted the opposite of B3** — "given the effect size above this
is the *expected* outcome", i.e. `winner_moved_rate < 0.05`. Measured: **~60%, twelve times
the threshold.** The effect-size reasoning was right about the *distance* (adding `ln z`
moves a ground distance by at most 19% of the diameter) and wrong about the *argmin*: the
risk landscape over candidates is nearly flat, so a small perturbation of the objective
relocates the minimiser often. Worth recording as a general lesson — **a small change in an
objective is not a small change in its argmin.**

## 13.3 The complication, and it is one I did not pre-register

B2 was written against the **continuous** `(u, v)` ruler, and by that rule there is no cost.
But the *fielded* headline is `dlund_mbr`, which compares leading-emission **cell centres**,
and it was not in B2. Computed post-hoc from the stored cell ids (a pure function of the
data, no re-run):

| Δ `dlund_mbr` = selection − `cells-2D` | `cont-3D` | `cont-2D` |
|---|---|---|
| pooled, n = 6560 | **+0.0042 [+0.0004, +0.0081]** | +0.0012 [−0.0021, +0.0046] |
| per arm | 6/8 positive, **1/8 significant** (`spline_s0` +0.0146 [+0.0038, +0.0251]) | 0/8 significant |

So the fielded cell-centre headline degrades by **+0.0042** — small, but the pooled CI
excludes 0. **This is a gap in my own pre-registration, not a reason to overturn the
verdict**, and it is stated here rather than resolved by picking whichever ruler flatters
the answer. Read it against three things:

- the **gain is ~10× the cost** in absolute terms (−0.040 on `ln z` against +0.0042 on the
  plane), and 8/8-significant against 1/8;
- `dlund_mbr` already loses to the free one-node `dlund_posterior_medoid` by ≈0.020 on all
  eight arms (§3, reading 4), so +0.0042 is a fifth of a gap the decode already carries;
- the cell ruler is quantisation-limited by construction (cells are 0.2 wide against
  distances of 0.6) — but *that* argument is only admissible if it is made in advance, and
  it was not, so it is offered as context and not as a defence.

## 13.4 Verdict and what to build

**BUILD** — B1, B2 and B3 all pass on the rule fixed before the run, the gain is
attributable to `ln z` specifically, and the pool restriction costs nothing.

Concretely, and bounded by §13.3:

1. **Build WP-3** (`decode.mbr_cloud_source`, `coords_for_draws`, the threading through
   `posterior_distances` / `run_closure` / `run_cluster_diagnostics`) exactly as §4/WP-3
   specifies, **default off and bit-identical off** — the standing discipline, and risk #1
   (the unfiltered coordinate call reordering RNG) is real and must stay behind the switch.
2. **`mbr_coords="+lnz"` becomes functional rather than fatal.** It currently raises (WP-2),
   which was the honest state while nothing wanted it; something does now.
3. **Do NOT make it the default decode on this evidence.** §13.3's +0.0042 on the fielded
   headline is the open question, and the arm that would settle it is a `dlund_mbr`-primary
   re-run with its own pre-registered rule. Ship it the way `mbr_n` and `dv_head="spline"`
   ship: measured, available, documented — with the difference that this one *won* its gate.
4. **Second payoff, independent of all the above:** WP-3's threading is what fixes the
   `_truth_cloud` `kt`-weight mismatch (§4/WP-3's inset), which `d_top`, `d_best`, `d_mbr`,
   `d_nearest_draw` and gates G2′/G6/G7 sit on today and which §11.3 left orphaned.

**What this does not say.** Nothing here reopens §2.5 or bears on the spline: the gain is
the same size on control arms as on spline arms (−0.031…−0.041 vs −0.026…−0.047), because it
is a **decode** fix. And it does not revive §5 — that rule was about whether a z-blind metric
explained a regression that does not exist, and it stays unscored.

---

# §14. PRE-REGISTRATION — should `+lnz` become the **default** decode?

Written 2026-08-06 on branch `nextStepsTrackAB`, **before the arms it reads exist**, and
before `scripts/zaware_default_decode.py` was written. It is the arm `§13.4` clause 3
called for and `PLAN_next_steps.md` A4 owns.

**§13 asked** whether the `ln z`-blind selection was worth *building*. Answer: BUILD, and
it is built (A1). **§14 asks** the separate question §13 deliberately did not: should it be
**on by default**, or continue to ship the way `mbr_n` and `dv_head="spline"` ship —
measured, available, documented, off.

## 14.1 What §13 left open, stated exactly

§12.2's **B2** — "the `(u, v)` half must not be bought with `ln z`" — was written against
the **continuous** `(u, v)` ruler and passed 0/4 violations. The *fielded* headline is
`dlund_mbr`, which compares leading-emission **cell centres**, and it was not in B2.
Measured post-hoc from the stored cell ids (§13.3):

> Δ`dlund_mbr` = cont-3D − cells-2D: **+0.0042 [+0.0004, +0.0081]** pooled (n = 6560),
> 6/8 positive, **1/8 significant** per arm.

**That number is known, and this pre-registration is written knowing it.** Pretending
otherwise would be worse than saying so. Two things are therefore done differently, and
both are what make §14 a test rather than a ratification:

1. **A disjoint population.** §11, §12 and §13 all scored jets `[0, 1000)` of
   `data/jet_aux_asym_test.root`. §14 scores jets **`[1000, 2000)`** of the same file —
   97 018 jets long, so the slice is free and has never been decoded by anything in this
   repo. Nothing below is fitted to jets it will be scored on.
2. **The fielded code path, not a bespoke script.** §13 scored the *coordinate table's*
   leading emission inside `scripts/zaware_selection_ceiling.py`. §14 goes through
   `run_closure` — `map_or_mbr` → `mbr_select` → `describe_cells` → `leading_emission_cell`
   on `hat.nodes` — i.e. the decode a user gets. That path did not exist when §13 ran; A1
   built it. If the §13 gain does not survive it, that is a finding about the shipped knob.

## 14.2 The measurement

Per arm, ONE pass, in order: seed → `draws_by_jet` (one `sample_batch` per jet) →
`coords_by_jet` (one batched `coords_for_draws` per jet, **unfiltered**) → **two
`run_closure` calls off byte-identical draws and coordinates**, differing only in

| arm | `mbr_cloud_source` | `mbr_coords` | |
|---|---|---|---|
| `fielded` | `cells` | `lnDR_lnkt` | the default decode |
| `lnz` | `coords` | `+lnz` | the candidate |

Sharing the draws makes the comparison paired **within** an arm, which is strictly stronger
than §11's across-arm pairing: the two sides are the same model on the same jets with the
same pool, so the only difference is the ground metric. `dlund_identity` is
model-independent and must be identical to floating point within each arm — the pairing
certificate, asserted rather than assumed.

**Arms:** the same eight as §11/§13 — `runs/lnz_spline/{spline_s0,s1,s2,
contstop_spline_s0}` and `runs/prod_test_v1/{v1_base_s0,s1,s2, v1_contstop_s0}` — so the
family-independence §13 found is checkable rather than assumed.

**Held fixed at the fielded values,** quoted from `scripts/eval_prod_test_v1.sh` pass B and
not re-chosen: `K = 200`, `mbr_n_candidates = 64`, `mbr_R = 8.485`, `β = 1`, `weight = kt`,
`min_emissions = 0`, `pot`, **cpu** (cuda is a different RNG stream *and* different float
kernels). 1000 jets per arm. Paired BCa 95%, 10 000 resamples, seed 20260806 —
`mbr_zaware_ab.bca_bootstrap`, unchanged.

## 14.3 The verdict rule — fixed before the run

**D1 — the primary read, and it is `dlund_mbr`.** Δ = `lnz` − `fielded`, per jet, paired.

- **D1a** the **pooled** 95% CI's **upper bound is below +0.010**;
- **D1b** the per-arm CI excludes 0 **upward** on **≤ 2 of 8** arms.

*Where +0.010 comes from, and it is not this measurement.* §11.1 measured the pooled paired
resolution of **this exact ruler** at **±0.01 over 3273 paired jets** — the largest
degradation `dlund_mbr` cannot distinguish from zero. That number was produced on
2026-08-06 for a different question (spline vs control), before §12 was written and before
any selection arm existed. *Where ≤ 2 of 8 comes from:* §12.2's B2 allowed violations on
fewer than 2 of 4 arms, i.e. a **25% rate**; 2 of 8 is the same rate on twice the arms, and
B2 was committed (`e5e3d38`) before §13's numbers existed.

**D2 — the gain has to survive the fielded pipeline.** Δ(`dlnz_mbr`) **≤ −0.020** with the
per-arm CI excluding 0 on **≥ 6 of 8**. Same −0.020 as §12.2's B1, same justification (≈5%
of the 0.41 baseline, 2× the ruler's own resolution), committed before §13. This is a real
risk and not a formality: §13 scored the coordinate table directly, while `run_closure`
scores `hat.nodes` after `describe_cells`, through the empty gate and the `min_emissions`
floor.

**D3 — nothing else on the fielded row moves.** `mult_bias_mbr` changes by less than 0.05
emissions and `p_empty_pred` by less than 0.02 on ≥ 7 of 8 arms. A decode that buys `ln z`
by shifting the multiplicity or the emptiness rate has changed a different product.
`dlund_posterior_medoid` and `dlund_identity` are **controls**: both are built from the
same draws without any EMD, so both must be **exactly** unchanged, and a nonzero delta
there means the two sides did not share their pool.

### The verdict, three-way and stated now

| | condition | verdict |
|---|---|---|
| **DEFAULT** | D1 **and** D2 **and** D3 | flip `decode.mbr_cloud_source` to `"coords"` and `decode.mbr_coords` to `"+lnz"` in `configs/decode/default.yaml`, and say so in `USAGE.md` / `CONFIGURATION.md` |
| **AVAILABLE-NOT-DEFAULT** | D2 **and not** D1 | keep §13.4 clause 3: the gain is real and is bought with a **measurable** loss on the deliverable. Ships as it does today |
| **RECONSIDER-THE-BUILD** | **not** D2 | the §13 gain does not reach the fielded pipeline or does not reproduce on fresh jets. A1 then ships a knob that does not do what §13 said it does, and that has to be recorded in §13 rather than in a footnote |
| **INCONCLUSIVE** | D2 holds, D1 holds, D3 fails | the trade is not the one priced. Report and stop; do **not** re-cut D3 |

### The position on the trade, written before the numbers

Required by `PLAN_next_steps.md` A4, and it is the content of D1. **The fielded headline
may not measurably degrade at the resolution of the instrument that defines it.** Inside
±0.010 the headline is unchanged *as far as `dlund_mbr` can tell*, and then the `ln z` gain
— an order of magnitude larger and 8/8-significant in §13 — is what decides. Outside it,
the headline is measurably worse and a diagnostic improvement does not buy that, however
much larger it is: `dlund_mbr` is what the product is quoted on.

Two facts are put beside the verdict rather than into it, because both are context and
neither is a threshold: the gain is ≈10× the cost in absolute terms; and `dlund_mbr`
**already** loses to the free one-node `dlund_posterior_medoid` by ≈0.020 on all eight arms
(§3, reading 4), so a +0.0042 would be a fifth of a gap the decode already carries. That
second fact is an argument for tolerating the cost and it is deliberately **not** encoded
in D1 — it would have been invented after §13.3, which is exactly the move §13.3 declined
to make.

**Guards printed beside the verdict:** `ground_diameter` vs `R` at both gdims (8.485
against 4.243 and 4.925 — asserted, not assumed), `winner_moved_rate`,
`leading_cell_moved_rate`, `n` on every row, the `dlund_identity` /
`dlund_posterior_medoid` controls, and the jet slice actually scored.

**Anti-circularity:** `R`, `K`, `mbr_n_candidates` and `lnkt_cut` are the fielded values and
are tuned against nothing. The two thresholds (+0.010, −0.020) both predate §13. No arm,
tier or key below is chosen after seeing a §14 number.
