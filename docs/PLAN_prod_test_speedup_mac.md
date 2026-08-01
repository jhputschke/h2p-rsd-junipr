# PLAN_prod_test_speedup_mac — the same notebook, measured on Apple silicon

*Status: **MEASURED, 2026-07-31**, on an M3 Max. No code changed. Addendum to
[`PLAN_prod_test_speedup.md`](PLAN_prod_test_speedup.md), which it supersedes **for this
box only** — the original's numbers are correct for the box they were taken on.*

## Context

[`PLAN_prod_test_speedup.md`](PLAN_prod_test_speedup.md) diagnoses
[`notebooks/prod_test_v0.ipynb`](../notebooks/prod_test_v0.ipynb) at **~109 min** — longer
than the training it assesses — and proposes four fixes. Every figure in it was taken on
the Linux GB10 box (20-core Grace CPU + GB10 GPU) **with four trainings and a pytest run
live**, i.e. on ~9 of 20 free cores.

This addendum answers two questions that document cannot: does the diagnosis hold on an
Apple-silicon Mac, and is MPS a way out of it. Measured, not argued:

* **The premise does not hold here.** The same single-jet sections project to **~4.2 min**
  on this Mac, against 109 min on the Grace box. The notebook is no longer the expensive
  half of anything.
* **MPS is 10–30× *slower* than CPU** on every path the notebook actually uses. It is not
  a fallback; `DEVICE = "cpu"` in cell 2 is not merely "defensible" on a Mac, it is the
  only sane setting.
* **The four fixes remain individually correct**, but three of them buy seconds here
  instead of tens of minutes, and step 1's stated *mechanism* does not exist in this torch
  build at all.

Read the two tables as *different boxes under different load*, not as a controlled A/B —
and note which rows differ. The batched rows are at parity (`nll_terms` 0.041 vs 0.035
ms/jet; batched `length_pmf` 0.010 vs 0.010). Only the batch-1 rows blow up, and the ratio
tracks batch size: ~80× at batch 1, ~10× on the K=200 decode loop, ~1.2× at batch 256.
Essentially none of the gap is arithmetic; all of it is per-op overhead.

The probable mechanism is that an **oversubscribed OpenMP barrier is a scheduler
round-trip rather than a spin**. At batch 1 every op forks a 20-way parallel region over a
few hundred elements; with the cores already claimed, the barrier waits on a *descheduled*
thread — a Linux scheduler quantum (ms), not a spin (µs). That is how one `encode` reaches
20 ms, and it is a cliff rather than a proportional penalty. The oversubscription was also
worse than "11 of 20 cores" suggests: nothing in this repo caps threads, so each of the
four concurrent trainings was itself claiming 20 intra-op threads — order 100 OpenMP
threads on 20 cores. The original plan's own 4× thread-cap result (93 → 23 ms) against
this box's 1.3× is consistent with that reading: capping to 4 there does not merely shrink
the barrier, it drops below the free-core count and removes the cliff.

If that is right, the original's 123 ms/jet and 109 min are substantially an artifact of
what else was running, and an idle Grace box at 4 threads would land far closer to this
one. Untested — the check is one number: batch-1 `length_pmf` at 20 / 4 / 1 threads on an
idle box.

## The box

```
torch 2.11.0.dev20251221, arm64          Apple M3 Max — 12 P + 4 E cores, 64 GB
torch.get_num_threads()      : 12        <- the default is the PERF-CORE count, not 16
ATen parallel backend        : OpenMP    omp_get_max_threads() : 12
MKLDNN                       : not found <- there is no oneDNN on this platform
torch.backends.mps.is_available() : True
```

That third line matters: the original plan's step 1 attributes the thread-count penalty to
"Python dispatch + oneDNN fork/join". There is no oneDNN here. The penalty is real anyway
— it is the plain OpenMP fork/join — but the explanation has to be rewritten rather than
copied, or the next person will look for a library that isn't installed.

## Measurements

Checkpoint `runs/prod_test_v0_pre_encoder_fix/20260731-142021-3629b89a37/best.ckpt`
(265k params, `use_multiplicity_head=True`, `continuous_coords=True`), test file
`data/jet_aux_asym_test.root` (97,018 jets), box otherwise idle. `K = 200`.

### Per operation, ms

| operation | cpu thr=12 | thr=4 | thr=1 | **mps** | Grace, contended |
|---|---|---|---|---|---|
| `sample(K=200)`, batch 1 | 12.8 | 9.8 | 12.0 | **129.5** | 123 |
| `sample_coordinates`, per draw | 1.04 | 0.85 | 0.80 | **24.6** | — |
| `length_pmf`, batch 1 | 0.25 | 0.25 | 0.26 | **3.26** | 20 |
| `length_pmf`, batched at 256 (per jet) | 0.0204 | 0.0101 | 0.0068 | 0.0298 | 0.010 |
| `nll_terms`, batch 256 (per jet) | 0.0663 | 0.0349 | 0.0313 | 0.0382 | 0.041 |

The batched rows are the control: at batch 256 this Mac and that Grace box are within
~1.6× of each other, and MPS finally draws level. Every order of magnitude in the table
lives in the **batch-1** rows. That is the original plan's thesis, and it survives here —
only its arithmetic doesn't.

### Per section, ms/jet

| section | cpu thr=12 | thr=4 | thr=2 |
|---|---|---|---|
| §3 occupancy loop | 18.1 | 10.3 | 11.0 |
| §4 `run_calibration` | 13.0 | 10.9 | 11.6 |
| §5 `run_closure(continuous=False)` | 14.6 | 12.4 | 13.0 |
| §5 `run_closure(continuous=True)` | 153.4 | 142.8 | 141.1 |
| §4 `run_tarp` | — | 175.3 | — |

`run_tarp` was timed on 4 jets only; treat it as an order of magnitude, not a figure.
The thr=4 / thr=2 / thr=1 columns differ by less than run-to-run noise on several rows —
the honest statement is **flat between 1 and 4 threads**, not "4 is the optimum".

### Projected notebook budget, `TORCH_THREADS = 4`

| section | jets | projected |
|---|---|---|
| §3 occupancy | 2000 | 21 s |
| §4 `run_calibration` | 2000 | 22 s |
| §4 TARP | 300 | 53 s |
| §5 `run_closure` cell | 2000 | 25 s |
| §5 `run_closure` continuous | 300 | 43 s |
| §5 `collect()` | 2000 cell + 300 cont | 75 s |
| §6 `length_pmfs` | 40,000 | 10 s |
| §8 support | 300 | ~1 s (scaled, not measured) |
| **total** | | **~4.2 min** |

At the stock 12 threads the same total is ~4.7 min. `collect()` was measured as a mixed
20-cell + 4-continuous run (44.5 ms per cell-jet including the continuous work); the split
above backs out ~16.5 ms/jet cell and ~140 ms/jet continuous, consistent with the
`run_closure` rows. §8 was not timed.

Compare the original's table: 109 min, of which 67 min was the continuous closure. Here
the continuous work is **~2 min of a ~4 min run**.

## Verdict, step by step

### 1. Thread cap — *keep it, rewrite the comment*

Direction right, magnitude and mechanism wrong for this box. The default is 12, not 20;
the fork/join is OpenMP, not oneDNN; and capping buys **1.1–1.8× per section, ~12% on the
notebook total**, not "~2× everywhere". The proposed comment text would be folklore here
— exactly what its last line asks nobody to do.

Two things the original does **not** claim, which hold on this Mac and are the better
argument for the change:

* the optimum is **flat from 1 to 4 threads**, so the cap is safe rather than tuned;
* it also speeds the **batched** §2/§2b/§2c path 2.1× (`nll_terms` 0.066 → 0.035 ms/jet).
  There is no batched-vs-single-jet tradeoff to worry about — the cap is a win on both
  sides, which is not obvious a priori and is the only reason a *global*
  `torch.set_num_threads` is defensible at all.

### 2. Batch the coordinate draws — *applies unchanged, still the largest item*

The per-draw `sample_coordinates` loop is **~91% of the continuous closure** here
(142.8 ms/jet continuous vs 12.4 ms/jet cell), which is the same structural finding the
original reports, and the fix in
[`models/ar_junipr.py:440`](../src/h2p_rsd_junipr/models/ar_junipr.py#L440) /
[`models/base.py:126`](../src/h2p_rsd_junipr/models/base.py#L126) is unchanged.

It is worth **~1.5 min** on this Mac, not ~65 min. As performance work it no longer pays
for its own risk; as a piece of engineering it is still the right shape.

### 3. Batch `length_pmf` — *applies unchanged, still exact*

0.25 → 0.007 ms/jet, ~35×, and still bit-identical (no RNG). That is **~10 s** saved, not
13.3 min. One trap found while measuring: the batched helper must run inside
`torch.inference_mode()` — `model.encode` outside it returns a tensor that requires grad
and the `.numpy()` at the end raises. The per-jet `length_pmf`
([`ar_junipr.py:560`](../src/h2p_rsd_junipr/models/ar_junipr.py#L560)) carries its own
decorator, so the notebook's cell-32 helper never had to think about this.

### 4. Share one sampling pass across §3/§4/§5 — *applies, on different grounds*

Worth ~1 min here. Its surviving justification is the one that was always
box-independent: the comparisons become **exactly paired** across sections. Do it for
that, or not at all — and it still costs the `shared_draws: true` flag in
`METRICS["run"]` and the loss of bit-comparability with earlier artifacts, which is a
real price to pay for one minute.

### 5. Re-cost the header comment — *applies, with more urgency*

Cell 2's "About an hour end to end … ~1.4M forward passes" is wrong on this Mac by an
order of magnitude **in the other direction**, which is the more damaging error: it tells
a reader not to run the notebook when they easily could.

### Follow-ups — one does not transfer

> "…it would take the whole per-jet tier from minutes to seconds and make `DEVICE = "cuda"`
> finally pay off."

On this Mac, with `mps` in place of `cuda`, that is **not** supported by measurement. Even
at batch 256 MPS is a wash or worse (`nll_terms` 0.038 vs 0.035 ms/jet; batched
`length_pmf` 0.030 vs 0.010). Cross-jet batching is worth doing here as a **CPU-side** win;
the payoff would have to be re-measured before any claim about the accelerator.

The other three follow-ups — the `PosteriorCache`, the `vonmises_sample` host sync
([`distributions.py:222`](../src/h2p_rsd_junipr/distributions.py#L222)), and
`lund_emd_matrix`'s `pot` double loop
([`mbr.py:501-506`](../src/h2p_rsd_junipr/inference/mbr.py#L501-L506)) — transfer
unchanged; the last is numpy/POT and never touched a device on either box.

## MPS

**It runs.** `sample`, `sample_coordinates`, `length_pmf` and `nll_terms` all executed on
`mps` without error. In particular `torch.multinomial`
([`inference/sampling.py:44`](../src/h2p_rsd_junipr/inference/sampling.py#L44),
[`sampling.py:89`](../src/h2p_rsd_junipr/inference/sampling.py#L89),
[`ar_junipr.py:388`](../src/h2p_rsd_junipr/models/ar_junipr.py#L388)) is not a blocker on
torch 2.11, and the float64 casts are already ordered `.cpu()` **before** `.double()`
([`eval/closure.py:210`](../src/h2p_rsd_junipr/eval/closure.py#L210),
[`scripts/leading_estimators.py:110`](../scripts/leading_estimators.py#L110)), which is the
one thing that would have raised outright.

**It is just slow**, and structurally so. These paths are launch-latency-bound and carry
host syncs *inside* their loops, so the accelerator is stalled on the interpreter:

| location | sync | frequency |
|---|---|---|
| [`sampling.py:49`](../src/h2p_rsd_junipr/inference/sampling.py#L49) | `bool(alive.any())` | per AR timestep |
| [`sampling.py:77`](../src/h2p_rsd_junipr/inference/sampling.py#L77) | `torch.as_tensor(list(lengths))` | **K scalar extractions per jet** |
| [`distributions.py:222`](../src/h2p_rsd_junipr/distributions.py#L222) | `bool(done.all())` | per rejection iteration, per draw |
| [`ar_junipr.py:352`](../src/h2p_rsd_junipr/models/ar_junipr.py#L352) | `.item()` on `p_cont` | per beam, per step |

This corroborates and extends the note already carried in
[`notebooks/lund_distribution_closure_prod_test_v0.ipynb`](../notebooks/lund_distribution_closure_prod_test_v0.ipynb)
cell 2 — *"Measured on M-series MPS: every torch stage 10-15× SLOWER than CPU"* — which
was written about the closure notebook and turns out to describe the prod-test paths too.

**Recommendation: keep `DEVICE = "cpu"`.** Do not add an `"auto"` or `"mps"` branch to
[`prod_test_v0.ipynb`](../notebooks/prod_test_v0.ipynb) cell 2; it would only give a future
reader a way to make the notebook 10× slower by accident.

One latent trap, if anyone does batch these paths and revisits MPS: the original plan's
verification item 2 proposes comparing `sample_coordinates_many` against the loop **under
a fixed `generator`**. A CPU `torch.Generator` against an MPS model raises
`Expected a 'mps' device type for generator`
([`ar_junipr.py:463-467`](../src/h2p_rsd_junipr/models/ar_junipr.py#L463-L467)). Run that
test on CPU — which is what everything under [`tests/`](../tests/) already does.

## What is actually worth doing on this box

In order, and the list is short by design:

1. **The thread cap**, with a Mac-correct comment — `MKLDNN not found`, OpenMP fork/join,
   default 12, flat from 1 to 4, and it helps the batched sections too. ~12% overall.
2. **Re-cost the cell-2 header** from "about an hour" to the ~4–5 min this box measures.
   This is the change that alters what a reader *does*.

Then, only if wanted for their own sake rather than for speed:

3. **Shared draws** (original step 4) — for exact pairing across §3/§4/§5, ~1 min.
4. **Batched coordinate draws** (original step 2) — the right shape, ~1.5 min.

Original steps 2–4 come to roughly **2.5 min combined** here. They are no longer
performance work on this Mac; they are correctness and tidiness work that happens to be
faster.

## Reproducing this table on another box

No harness was committed — it is a dozen lines and the numbers age. The recipe:

1. Load the checkpoint with `load_for_inference(..., map_location="cpu")`, rebuild with
   `Geometry.from_config` + `build_model`, and build a `MatchedLundDataset` over the first
   few hundred jets of `data/jet_aux_asym_test.root` with `AUX = model.aux_feature_names`.
2. **Per operation:** time `model.sample(xf, nx, 200)` over ~3 jets, `sample_coordinates`
   over ~12 draws of one jet, `length_pmf` over ~40 jets, and — under
   `torch.inference_mode()` — `encode` + `n_head` + `nll_terms` over `collate`d batches of
   256. Warm up once per device before timing; call `torch.mps.synchronize()` around every
   MPS interval.
3. **Per section:** call `run_calibration`, `run_closure(continuous=False)`,
   `run_closure(continuous=True)`, `run_tarp` and `leading_estimators.collect` directly
   with small `n_jets` (20 / 20 / 4 / 4 / 20) and divide.
4. Sweep `torch.set_num_threads` over 12/8/4/2/1 in one process — it takes effect
   in-process on the OpenMP backend — and repeat on `mps`.
5. Report the box: `torch.__config__.parallel_info()`, core split, torch version, and
   **what else was running**. The original plan's table and this one differ mostly because
   that last line differs.
