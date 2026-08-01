# PLAN_prod_test_speedup — why the prod-test notebook costs more than the training it assesses

*Status: **PROPOSED.** Diagnosis is measured (2026-07-31, this box); no code changed yet.*

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

## Plan

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
* **`lund_emd_matrix`'s `pot` backend is a Python double loop of `ot.emd2`**
  ([`mbr.py:501-506`](../src/h2p_rsd_junipr/inference/mbr.py#L501-L506)) at ~991 µs/solve.
  TARP needs only one row per jet so it is tolerable (1 min), but a full K×K MBR matrix
  measures **39.6 s/jet** — turning on `decode.point_estimator=mbr` for the 2000-jet tier
  would cost ~22 h. Either batch it the way `_matrix_ef` does, or make `mbr_n_candidates`
  mandatory when `point_estimator=mbr`.

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
6. **End to end:** run the notebook top to bottom on an **idle** box (the four trainings
   and the pytest currently eat ~11 of 20 cores) and record the wall-clock per section
   against the table above.
