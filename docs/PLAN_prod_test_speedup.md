# PLAN_prod_test_speedup — why the prod-test notebooks cost more than the training they assess

*Status: **IMPLEMENTED, 2026-08-01.** Diagnosis measured 2026-07-31 (this box, under
load); the implementation and its re-measurement are in
[What was implemented](#what-was-implemented-2026-08-01) at the foot of this document.
Two of the plan's conclusions did not survive re-measurement on an **idle** box, and a
**~60 min cost this document dismissed** turned out to be the largest one in the
notebook; all three are recorded there rather than quietly dropped.*

Covers both notebooks of the production test: [Part A](#part-a--prod_test_v0ipynb) is
`prod_test_v0.ipynb` (~109 min → ~10 min), [Part B](#part-b--lund_distribution_closure_prod_test_v0ipynb)
is `lund_distribution_closure_prod_test_v0.ipynb` (~19 → ~8 min). They share one cause —
batch-1 CPU decode — but their dominant costs are **different stages**, and only one of
the Part A fixes moves Part B.

The Part A changes ship as a **new notebook**,
[`notebooks/prod_test_v1.ipynb`](../notebooks/prod_test_v1.ipynb); `prod_test_v0.ipynb`
stays as the record of how the committed v0 artifact was produced. Part B ships as a
second generated variant, `lund_distribution_closure_prod_test_v1.ipynb`.

## Context

[`notebooks/prod_test_v0.ipynb`](../notebooks/prod_test_v0.ipynb) takes substantially
longer than the 60-epoch training run it assesses, which is backwards from the usual
intuition and reads as a bug. It is neither a bug nor an illusion: training and the
notebook do **structurally different amounts of work per jet, on different hardware**.
Training is one teacher-forced forward+backward per jet per epoch, batched at 256, on
the GPU. The notebook is ~1.4M **batch-size-1** forward passes on the CPU, most of them
autoregressive draws, and it re-does several of them.

Measured on this box (20-core Grace CPU + GB10 GPU) with four `h2p-rsd-junipr train`
jobs and a `pytest` run live — i.e. under the conditions the notebook is actually run in:

| operation | cost per jet |
|---|---|
| teacher-forced `nll_terms`, batch 256 — **what training does** | **0.041 ms** |
| `length_pmf`, batched at 256 | 0.010 ms |
| `length_pmf`, batch of 1 (§6, as written) | 20 ms |
| `sample_batch(K=200)`, batch of 1 (§3/§4/§5) | 123 ms |
| `run_closure(continuous=False)` | 230 ms |
| `run_closure(continuous=True)` | **6,800–7,200 ms** |

Training: 495,071 jets × 60 epochs at batch 256 on the GPU ≈ **60 s/epoch, ~1 h total**
(33 epochs in 33 min, four runs sharing one GPU). The notebook's single-jet sections
project to **~109 min** — for the *evaluation*.

Four multiplicative causes, all measured:

1. **Batch 1 on CPU vs batch 256 on GPU.** The model is 265k parameters; at batch 1
   essentially all wall-clock is fixed per-op overhead (Python dispatch + oneDNN
   fork/join), not arithmetic. `DEVICE = "cpu"` in cell 2 is defensible for the decode
   loop — the loop is the problem, not the device.
2. **K=200 draws per jet, re-sampled in four independent passes** over the same 2000-jet
   tier (§3 occupancy, §4 `run_calibration`, §5 `run_closure`, §5 `collect()`). No
   section shares draws with another.
3. **The continuous closure is ~30× the cell closure and is run twice.**
   `sample_coordinates` is called once *per draw* (≈130–185 non-empty of 200) per jet,
   and every call re-runs `encode()` **and** `xattn_kv()` from scratch on the same jet —
   57% of the call at 20 threads, producing an identical tensor 184 times
   ([`ar_junipr.py:488-490`](../src/h2p_rsd_junipr/models/ar_junipr.py#L488-L490)).
   §5-continuous (36 min) + `collect()`'s continuous pass (31 min) ≈ 67 of the 109 min.
4. **torch's default thread count actively hurts.** A fresh kernel takes all 20 cores; at
   these tensor sizes the barrier dominates. Measured: `sample_batch` 93 ms/jet at 20
   threads vs 23 ms at 1–4 threads; `run_closure(continuous)` 6.8 s/jet at 20 vs 3.4 s at
   1. On top of that the four trainings and the pytest are eating ~11 of the 20 cores.

**Not the problem, so do not chase it:** `load_rntuple` 0.6 s (test) / 2.8 s (train),
`MatchedLundDataset.__init__` 3.3 s for 97k jets, `LundDataModule.setup()` 3.0 s,
`coordinate_pits` 0.25 s per 5000-jet chunk, and the whole §2b/§2c batched NLL work
(5 × 97,018 jets) at **0.4 min** — which is the proof that the batched path is fine and
only the single-jet path is not.

> **Correction (2026-08-01).** That last claim is the one thing in this diagnosis that
> was wrong, and it was wrong by ~60 minutes: 0.4 min is what §2b/§2c cost with the
> dataset built once per chunk, which is what the timing above measured — but *not* what
> the notebook does. See
> [the batched sections were O(B²)](#the-batched-sections-were-ob2-and-that-was-the-biggest-cost-of-all).

Target: **~109 min → ~10 min.** Changes that move reported numbers within Monte-Carlo
noise are accepted (decision recorded 2026-07-31); the run will not be bit-comparable to
earlier artifacts and must say so.

## Where the time goes, as written

| section | projected |
|---|---|
| §3 occupancy, 2000 jets | 4.1 min |
| §4 `run_calibration`, 2000 jets | 4.1 min |
| §4 TARP, 300 jets | 1.0 min |
| §5 `run_closure` cell, 2000 jets | 7.7 min |
| §5 `run_closure` continuous, 300 jets | 35.9 min |
| §5 `collect()`, 2000 cell + 300 continuous | 38.8 min |
| §6 `length_pmfs`, 40,000 jets | 13.3 min |
| §8 support, 300 jets | 4.1 min |
| **total** | **109 min** |

## Part A — `prod_test_v0.ipynb`

### 1. Thread cap in §0 — one line, ~2× everywhere

In [`prod_test_v0.ipynb`](../notebooks/prod_test_v0.ipynb) cell 2, beside `DEVICE`:

```python
TORCH_THREADS = 4   # 20 (the default) is SLOWER: at batch 1 every op pays a 20-way
#                     fork/join barrier on a few hundred elements. Measured 4x on
#                     sample_batch, 2x on the continuous closure.
torch.set_num_threads(TORCH_THREADS)
```

`torch` is imported in cell 4, so either move the call there or import torch in cell 2.
Keep the "measured, not folklore" note in the comment so nobody reverts it.

### 2. Batch the coordinate draws — the big one, ~67 min → ~2 min

Add a batched sibling to the per-draw hook so one jet's K draws cost one forward pass
instead of ~184.

* [`models/base.py:126`](../src/h2p_rsd_junipr/models/base.py#L126) — add
  `sample_coordinates_many(self, xf, nx, draws) -> list[torch.Tensor | None]` next to
  `sample_coordinates`, with a default implementation that loops the existing hook.
  `cfm`, `cinn` and `diffusion` then inherit today's behaviour unchanged.
* [`models/ar_junipr.py:440`](../src/h2p_rsd_junipr/models/ar_junipr.py#L440) — override
  it: compute `e = self.encode(xf, nx)` and `kv = self.xattn_kv(xf, nx)` **once**, pad
  the draws to `L_max` into a `(K, L_max)` cell tensor, run `_decode_states` at batch K
  (reuse `_apply_xattn`'s existing K-broadcast at
  [`ar_junipr.py:183-185`](../src/h2p_rsd_junipr/models/ar_junipr.py#L183-L185)), one
  `_coord_params` call, then `trunc_normal_sample` / `vonmises_sample` over the whole
  `(K, L_max)` block, and slice each row back to its own length. Factor the shared body
  with `coord_head_params` rather than duplicating it.
* Callers switch to the batched hook:
  [`closure.py:201-212`](../src/h2p_rsd_junipr/eval/closure.py#L201-L212) and
  [`leading_estimators.py:103-112`](../scripts/leading_estimators.py#L103-L112). Both
  keep their `c is None → cont_ok = False` degradation path, which the list-of-`None`
  return preserves.

This reorders RNG consumption, so the `*_cont` numbers shift within MC noise. Say so in
the `run_closure` docstring, which currently advertises the per-draw cost.

### 3. Batch `length_pmf` in §6 — 13.3 min → ~5 s

`length_pmf` for the multiplicity-head model is `softmax(n_head(encode(x)))`
([`ar_junipr.py:560-569`](../src/h2p_rsd_junipr/models/ar_junipr.py#L560-L569)) — a pure
batched op called 40,000 times at batch 1. Rewrite the notebook's `length_pmfs` helper
(cell 32) to chunk with the existing `collate` at `POP_BATCH`, exactly as `nll_terms_over`
(cell 14) already does, and call `model.n_head` on the batched encoding. Keep the
`length_temperature` / `length_tilt` zeroing wrapper as is.

### 4. Share one sampling pass across §3/§4/§5 — ~24 min → ~8 min

Build the K=200 draws for the `SAMP` tier **once** in a new cell after cell 7, and have
§3 (occupancy), §4 (`run_calibration`), §5 (`run_closure` cell) and §5 (`collect()`)
consume them:

* `run_closure` re-samples internally
  ([`closure.py:176`](../src/h2p_rsd_junipr/eval/closure.py#L176)); `mbr_select` already
  has the `draws=` reuse pattern
  ([`mbr.py:556`](../src/h2p_rsd_junipr/inference/mbr.py#L556)) — copy it. Add an optional
  `draws_by_jet=None` argument to `run_closure`, `run_calibration` and `collect()`,
  defaulting to today's behaviour so `h2p-rsd-junipr eval` is untouched.
* §8's posterior pool (cell 40) re-samples the HEAVY tier at `K_DRAWS // 10`; give it a
  slice of the shared draws instead (~4.1 → ~0.2 min).

The comparisons become exactly paired across sections, which is a small scientific
improvement, but the run is **not bit-comparable to earlier artifacts** — record a
`shared_draws: true` flag in `METRICS["run"]` so a future reader can tell.

### 5. Re-cost the header comment

Cell 2's "About an hour end to end … ~1.4M forward passes" is the only cost statement in
the notebook and will be wrong after this. Replace it with the measured budget table above
and the new total.

## Part B — `lund_distribution_closure_prod_test_v0.ipynb`

The sibling notebook
([`lund_distribution_closure_prod_test_v0.ipynb`](../notebooks/lund_distribution_closure_prod_test_v0.ipynb),
generated by [`scripts/make_prod_closure_nb.py`](../scripts/make_prod_closure_nb.py) from
`lund_distribution_closure_v2.ipynb`) is in the same batch-1 CPU regime but its cost sits
in **different stages**, so most of Part A does not transfer. Measured on the same box,
per jet through `eval_jets` at the notebook's own settings (`K_DRAWS=120`,
`MBR_N_CANDIDATES=16`):

| stage | 20 threads, `pot` | 4 threads, `pot` | 4 threads, `energyflow` |
|---|---|---|---|
| `map_estimate` ×2 | 198.6 ms | 106.9 ms | 106.9 ms |
| MBR | 144.2 ms | 102.5 ms | **31.3 ms** |
| `sample(K=120)` | 122.3 ms | 51.2 ms | 54.3 ms |
| `sample_coordinates` ×1 | 42.8 ms | 15.3 ms | 20.9 ms |
| `ar_kappa` | 39.1 ms | 14.7 ms | 20.9 ms |
| length floor + `length_pmf` | 35.2 ms | 15.7 ms | 16.3 ms |
| **total at `N_JETS=2000`** | **19.4 min** | **10.2 min** | **8.4 min** |

What transfers from Part A:

* **Step 1 (thread cap) — yes, and it is the single biggest win here: 19.4 → 10.2 min.**
  Same one line, same reason.
* **Step 3 (batched `length_pmf`) — yes, negligible.** Cell 16's per-jet list
  comprehension over `model.length_pmf` is 2.4% (~28 s over 2000 jets, ~0.02 s batched).
  Free once the helper exists; do it for consistency, not for the time.
* **Step 2 (batched coordinates) — no.** This notebook draws **one** posterior sample per
  jet, so `sample_coordinates` is called once per jet, not once per draw. Batching across
  *draws* has nothing to bite on; batching across *jets* (the follow-up) would.
* **Step 4 (shared draws) — no.** `eval_jets` already shares one `draws` across the length
  floor, MBR and the posterior series. It is already a single pass.

Two changes specific to this notebook, neither of which needs code in `src/`:

* **Set `MBR_BACKEND = "energyflow"`** (installed on this box): MBR 102 → 31 ms/jet. The
  notebook's own §0 documents it as *the same number, not an approximation* — identical
  MBR tree on 99.3% of jets, 100% on multiplicity, the remainder being solver tie-breaks —
  and the generated variant's assertion only forbids `"surrogate"`, so the guard already
  permits it. Record the backend in `dist_closure_metrics.json` as it already does.
* **Fix the stale cost comment in §0.** It claims *"MBR is ~96% of the runtime"* and quotes
  113 ms/jet for `pot`. On this hardware MBR is 12–34% and `map_estimate` is the larger
  line, so `MBR_N_CANDIDATES` / `K_DRAWS` — advertised there as "THE speed knob" — point at
  the wrong stage. Replace with the table above. Because the notebook is generated, this
  edit belongs in [`scripts/make_prod_closure_nb.py`](../scripts/make_prod_closure_nb.py)
  and/or `lund_distribution_closure_v2.ipynb`, **not** in the generated file — the header
  of cell 2 says so explicitly.

Left alone deliberately: `map_estimate` is called twice per jet (the floored MAP plus the
`MAP_ALLOW_EMPTY = True` control) and is now the dominant stage at 34–43%. Both calls are
load-bearing — the control is what prices the length floor in §6 — so this is a
batch-the-beam-search problem, not something to delete. See the follow-up.

## Follow-up (not in this pass)

Recorded so it is not lost; each is a separate piece of work.

* **Batch the AR sampler across jets.** With `use_multiplicity_head=True` the sampler is
  already fixed-length
  ([`sampling.py:60`](../src/h2p_rsd_junipr/inference/sampling.py#L60)), so `jets × draws`
  can pad to one `L_max` and decode as a single batch. This is the last 10² factor: it
  would take the whole per-jet tier from minutes to seconds and make `DEVICE = "cuda"`
  finally pay off.
* **A `PosteriorCache` per tier** — `{jet: (draws, coords, pmf)}` computed once and handed
  to every section, replacing the ad-hoc `draws_by_jet` threading from step 4.
* **Drop the host sync in `vonmises_sample`.** The `bool(done.all())` per iteration
  ([`distributions.py:222`](../src/h2p_rsd_junipr/distributions.py#L222)) costs more than
  the extra iterations once the tensor is `(K, L)` rather than `(1,)`; sync every 8
  iterations, or run a fixed count.
* **Batch `map_estimate` / `beam_search_cells` across jets.** Part B's dominant stage
  (34–43%, and 2 calls per jet). The beam is `beam_width=8` over `topk_cells=6` at batch 1
  ([`point_estimate.py:74`](../src/h2p_rsd_junipr/inference/point_estimate.py#L74)); the
  jets in a tier are independent, so `jets × beams` is one padded batch.
* **`lund_emd_matrix`'s `pot` backend is a Python double loop of `ot.emd2`**
  ([`mbr.py:501-506`](../src/h2p_rsd_junipr/inference/mbr.py#L501-L506)) at ~991 µs/solve,
  while the `energyflow` path already batches the whole non-empty block through one
  OpenMP-parallel `emds` call ([`_matrix_ef`](../src/h2p_rsd_junipr/inference/mbr.py#L438)).
  Part B's immediate lever is therefore configuration, not code. It still matters for
  `pot`-only hosts: TARP needs one row per jet so it is tolerable (1 min), but a full K×K
  MBR matrix measures **39.6 s/jet** — turning on `decode.point_estimator=mbr` for
  `prod_test_v0`'s 2000-jet tier would cost ~22 h. Either batch the `pot` loop the way
  `_matrix_ef` does, or make `mbr_n_candidates` mandatory when `point_estimator=mbr`.

## Verification

1. **Thread cap:** re-run cell 2 + §3 and confirm the occupancy loop drops from ~4 min to
   ~2 min. No numbers change — confirm by comparing `METRICS["occupancy"]` before/after.
2. **Batched coordinates:** a test beside [`tests/`](../tests/) asserting
   `sample_coordinates_many(xf, nx, draws)` returns per-row tensors of the same shapes as
   the per-draw loop, and that under a fixed `generator` the marginal means/σ agree with
   the loop to within MC error over ~2000 draws. Exactness is not claimed — the RNG stream
   differs by design.
3. **Batched `length_pmf`:** assert bit-identical `pmf_test` against the batch-1 loop on
   500 jets. This one **is** exact — no RNG involved.
4. **Shared draws:** check `CAL["coverage_68"]`, `CLO["dlund_posterior_medoid"]` and
   `LEAD["cell_medoid"]` land inside their previously quoted MC bands, not that they match.
   The four PIT chunks' spread (§4) is the on-hand scale for "within noise".
5. `python -m pytest tests/` green, and `h2p-rsd-junipr eval <ckpt>` still runs — every new
   argument must be optional.
6. **Part B backend switch:** run `eval_jets` on ~200 jets under both `pot` and
   `energyflow` and confirm the MBR trees agree on ≳99% of jets and 100% on multiplicity,
   as the notebook's §0 claims. If they do not, keep `pot` and say why — the claim is what
   makes the switch free.
7. **Part B regeneration:** after editing `lund_distribution_closure_v2.ipynb` /
   `make_prod_closure_nb.py`, re-run `python scripts/make_prod_closure_nb.py` and
   `pytest tests/test_prod_closure_nb.py`, which is the guard that the generated notebook
   stays byte-identical to v2 outside its title and section 0.
8. **End to end:** run both notebooks top to bottom on an **idle** box (the four trainings
   and the pytest currently eat ~11 of 20 cores) and record the wall-clock per section
   against the tables above.

## What was implemented (2026-08-01)

Everything above, on the same box with **nothing else running** — which is itself a
finding: two of the plan's conclusions are artifacts of the load it was measured under,
and both are corrected below rather than deleted, because the loaded numbers are the ones
that apply when this notebook is run the way it usually is.

### Code

| where | change |
|---|---|
| [`models/base.py`](../src/h2p_rsd_junipr/models/base.py) | `sample_coordinates_many(xf, nx, draws) -> list`, defaulting to a loop over the per-draw hook, so `cfm` / `cinn` / `diffusion` are bit-identical to today |
| [`models/ar_junipr.py`](../src/h2p_rsd_junipr/models/ar_junipr.py) | overrides it: encode + `xattn_kv` once, pad to `(K, L_max)`, one `_coord_params` call, one call to each sampler. `_coord_params_padded` is the shared body `coord_head_params` now also uses, so the two paths cannot drift |
| [`eval/closure.py`](../src/h2p_rsd_junipr/eval/closure.py) | `run_closure(..., draws_by_jet=None)`; the continuous branch calls the batched hook |
| [`eval/calibration.py`](../src/h2p_rsd_junipr/eval/calibration.py) | `run_calibration(..., draws_by_jet=None)` |
| [`scripts/leading_estimators.py`](../scripts/leading_estimators.py) | `collect(..., draws_by_jet=None)`; the batched hook |
| [`scripts/make_prod_closure_nb.py`](../scripts/make_prod_closure_nb.py) | emits **two** variants (`v0`, `v1`) from one v2 source |
| [`notebooks/prod_test_v1.ipynb`](../notebooks/prod_test_v1.ipynb) | new: steps 1–5 of Part A, plus the O(B²) fix below |
| [`notebooks/lund_distribution_closure_v2.ipynb`](../notebooks/lund_distribution_closure_v2.ipynb) | §0 re-costed, `TORCH_THREADS` knob, batched `q(0|x)` |

New tests: `tests/test_batched_coordinates.py`, `tests/test_shared_draws.py`, plus
`test_length_pmf_batches_consistently` in `tests/test_multiplicity_head.py`,
`test_no_dataset_is_rebuilt_once_per_item` in `tests/test_notebooks.py`, and both
variants parameterised through `tests/test_prod_closure_nb.py`.

### Re-measured, idle box, `ar_junipr_v4` + `lundnet`, K = 200

| stage | v0 path | v1 path |
|---|---|---|
| `sample_batch(K=200)`, 4 threads | 49 ms/jet | *(unchanged; now drawn once)* |
| one jet's K coordinate draws | **2 528 ms/jet** | **22 ms/jet** |
| `run_closure(continuous=True)`, given draws | — | 67 ms/jet |
| `run_closure(cell)`, given draws | — | 38 ms/jet |
| `length_pmf` per jet | 6.6 ms/jet | 0.06 ms/jet |
| §8 support pool per jet | — | 22 ms/jet |

Projected at the notebook's own tiers: one shared sampling pass 1.6 min, §5-continuous
0.3 min (12.9 min on the v0 path), §5-cell 1.3 min, §6 2.5 s (4.4 min at batch 1), §8 7 s.

### The batched sections were O(B²), and that was the biggest cost of all

Not a decode problem, not in the plan, and larger than everything the plan targeted. Both
batched helpers of `prod_test_v0.ipynb` — `nll_terms_over` (§2b) and `per_jet_nll_of`
(§2c) — built their chunk like this:

```python
b = collate([MatchedLundDataset(chunk, geom, AUX)[k] for k in range(len(chunk))])
```

The constructor is *inside* the comprehension, so the whole 256-jet dataset is rebuilt
once per `k`: **B datasets of B jets**. Measured on this box, per jet through that helper:

| | ms/jet |
|---|---|
| as written (dataset rebuilt per item) | **7.49** |
| dataset hoisted out of the comprehension | **0.036** |
| — of which the `nll_terms` forward itself | 0.24 (CPU) / 0.10 (GPU) |

At `POP_BATCH = 256` that is a 256× multiplier on the dataset build, and across the
ablation's five arms over all 97,018 jets it is **~60 min** — more than the continuous
closure and the four re-sampling passes put together. The first end-to-end v1 run was
still in §2c at 45 minutes, which is what exposed it.

This is exactly the failure mode the diagnosis above could not see: it timed
`nll_terms` at batch 256 (0.041 ms/jet) and `MatchedLundDataset.__init__` (34 µs/jet)
*separately*, both correct, and concluded that "the batched path is fine". The batched
path is fine. The loop wrapping it was not, and the only way to catch that is to time the
notebook's own code rather than the primitives it calls.

Fixed by hoisting the dataset (`ds_chunk = ...`) in §2b and §2c of **both** notebooks.
v0 is otherwise frozen, but this one is applied there too: the dataset is deterministic
and no RNG is involved, so building it once instead of B times cannot change a number —
it is the one edit that costs a reader nothing and saves them an hour.
`tests/test_notebooks.py::test_no_dataset_is_rebuilt_once_per_item` walks every
notebook's AST for the shape so it cannot come back.

### Where the plan was wrong, measured

* **Step 1's magnitude.** "~2× everywhere" holds under the load it was measured under, not
  on an idle box. Idle, capping to 4 threads costs `run_closure` (38 vs 30 ms/jet) while
  buying `sample_batch` (49 vs 77) — **net ~10%** for `prod_test_v1`, whose cost is the
  sampler. The cap is kept, with those numbers in the cell-2 comment.
* **Part B's step 1 reverses sign.** `lund_distribution_closure_v2`'s largest stage is
  `map_estimate`'s beam search, which *likes* 20 threads (45.5 ms/jet vs 83.1 at 4), so
  idle the cap makes that notebook **~10% slower** (8.9 vs 8.1 min at `N_JETS=2000`), not
  2× faster. It therefore ships as `TORCH_THREADS = None` — an off-by-default knob with
  both measurements beside it — instead of a hard cap. Same knob, opposite answer in two
  notebooks: it depends on which stage dominates.
* **The backend switch is free for the tree, not for the risk.** `energyflow` picked a
  bit-identical MBR tree on **100%** of 200 held-out jets (the plan quotes 99.3%), and is
  3.5× on the MBR stage / 1.55× on the whole pass. But EnergyFlow reports the distance on
  its R-normalised scale, so `mbr_risk_mean` comes out **1/R = 1/8.485** of `pot`'s — a
  constant factor, which is exactly why the selection is unaffected. `dist_closure_metrics.json`
  already records `mbr_backend` beside the risk. This is also why the **v0** variant keeps
  `pot`: its committed artifact records a POT-scale risk.

### Verification, as run

1. **Thread cap** — measured per stage rather than by re-running §3; see above. And no
   number changes with it: `run_calibration` on 300 held-out jets at K=200 returns
   **bit-identical** `coverage_68` and `sbc_chi2` at 4 and at 20 threads, over five seeds.
   One caveat found by the artifact diff below: a *teacher-forced* mean can differ in its
   last digits, because a multi-threaded reduction adds in a different order. Two of the
   389 numeric keys did — both `pit_mean`s, at a relative **7 × 10⁻¹¹**.
2. **Batched coordinates** — `tests/test_batched_coordinates.py`: per-row shapes, cells
   land in their own cell, marginals agree with the loop to within 5 standard errors over
   2 000 draws, `None` per row for `ar_junipr_v1`, and the contract default is bit-identical.
3. **Batched `length_pmf`** — bit-identical, asserted twice: in
   `tests/test_multiplicity_head.py` and in the notebook itself on 500 jets.
4. **Shared draws** — `tests/test_shared_draws.py` proves nothing re-samples (the helpers
   run with `sample_batch` monkeypatched to raise) and that handing over the same draws
   reproduces the cell-level metrics exactly; `prod_test_v1` §9 prints its headline numbers
   beside the v0 artifact's whenever one is on disk.

   The band to read those against had to be measured, since none was on record: over **17
   independent draw streams** on the same 247 scored jets, `coverage_68` is
   **0.53 ± 0.03** (range 0.490–0.579) and `sbc_chi2_uniform` is **27 ± 5** (range
   20.1–35.3).

   In the event the band was not needed, which is the more interesting result. Comparing
   the two artifacts key by key — **389 numeric keys, 345 bit-identical**, 2 differing at
   7 × 10⁻¹¹ (thread-order, above) and 32 being the coordinate rows — the v1 run came back
   **bit-identical on every cell-level number** — `dlund_identity` / `_mode` / `_medoid`, `coverage_68` (0.5381),
   `sbc_chi2` (107.0), `mult_bias_posterior`, `posterior_cells_emitted` (271),
   `tarp_max_dev` (0.0367), all at Δ = 0.0000. The reason is that v0 called
   `seed_everything(SEED)` before each of its four sampling passes and walked the same
   tier in the same order, so the four passes were drawing **the same draws**: sharing
   them is an exact refactor, and v0 was paying three times over for a coincidence
   nobody had checked. Only the two sections that interleave coordinate draws moved —
   `dlund_posterior_geomedian_cont` +0.88% and `collect()`'s cell medoid ratio −0.19%,
   with the small-sample rows built on them (the `*_oracle`s, §8's `sd_violation` on
   300 jets × 20 draws) moving up to ~6% — which is `sample_coordinates_many` reordering
   the RNG, exactly as advertised. The headline `nll.total_per_jet`,
   `aux_ablation.delta_nat_per_jet`, `empty_tree.tau.value` and the fitted `(T, tilt)` are
   all identical to the last bit.
5. `python -m pytest tests/` green (476 passed, 1 skipped); every new argument is optional
   and `h2p-rsd-junipr eval` is untouched — re-run in full on the production-test
   checkpoint (97k jets, `pit_coords`/`stratify_regions`/`tarp`/`closure_continuous` all
   on). Its draw-free rows come back identical to the run recorded in
   `runs/prod_test_v0/eval_cli.log`; the draw-dependent ones move inside the band above,
   as they must — the continuous closure no longer consumes the RNG the same way, so
   everything downstream of it sees a different stream. And with an identical stream the
   two versions agree bit-for-bit: `run_calibration` at a fixed seed returns the same
   numbers before and after the change, which is what says the plumbing is plumbing.
6. **Part B backend switch** — 200 jets, K=120, 16 candidates: identical tree 100%,
   identical multiplicity 100%, MBR 98.0 → 22.0 ms/jet.
7. **Part B regeneration** — both variants regenerated; `tests/test_prod_closure_nb.py`
   pins each to v2 outside the title and parameter cells.
8. **End to end** — `prod_test_v1.ipynb` executed top to bottom on the idle box:
   **5.90 min**, no errors, `ACCEPTANCE: PASS`, artifact written to
   `<ckpt>/prod_test_v1/`. Per cell, the six that carry the run:

   | cell | s |
   |---|---|
   | §1 the one shared sampling pass | 85.8 |
   | §2c aux ablation, 5 arms × 97k jets | 81.0 |
   | §5 `run_closure` cell tier, 2000 jets | 74.8 |
   | §4 TARP, 300 jets | 22.6 |
   | §2b held-out NLL, 97k jets | 20.0 |
   | §5 `run_closure` continuous, 300 jets | 17.4 |

   Everything else is under 18 s, including the whole of §6 (the `(T, tilt)` fit is now
   the largest thing in it at 17 s, which it never was before). The target was ~10 min.
