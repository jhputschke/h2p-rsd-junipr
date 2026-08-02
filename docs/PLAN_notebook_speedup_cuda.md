# PLAN — CUDA for the production-test notebook, and the TF32 trap underneath it

**Status:** proposed, not implemented. Deferred deliberately: it was found while running
[`PLAN_prod_test_v1.md`](PLAN_prod_test_v1.md)'s grid, and changing the device mid-run
would have moved numbers that run was quoting. Companion to
[`PLAN_prod_test_speedup.md`](PLAN_prod_test_speedup.md) (Part A/B, CPU) and
[`PLAN_prod_test_speedup_mac.md`](PLAN_prod_test_speedup_mac.md) (the MPS measurement).

---

## 1. The problem

`notebooks/prod_test_v1.ipynb` §6 fits `(T, tilt)` and the empty-tree `tau` from a per-jet
`length_pmf`. Its cost depends entirely on whether the checkpoint has an explicit `q(N|x)`
head:

| family | `length_pmf` is | measured |
|---|---|---:|
| `use_multiplicity_head=True` (v3/v4 default) | `softmax(n_head(e))`, one batched op | **0.73 ms/jet** |
| `use_multiplicity_head=False` | a 500-draw autoregressive sample per jet | **52–62 ms/jet** |

**71×**, or ~41 min for §6's 40 000 jets (20 000 fit + 20 000 report).

This is not a legacy path. `scripts/run_prod_test_v1.sh` fields `v1_contstop_s0/s1` as
`ar_junipr_v4` with `model.use_multiplicity_head=false`, and those are the arms gate G8
favours in [`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md) §4.8. Running the notebook
on one is what exposed this, and it was worked around by hand-cutting `N_FIT` 20 000 →
5 000 — which is why that artifact records `empty_tree.recalibration.fit_jets = 5000`
while the committed notebook says 20 000. **The notebook and its own artifact currently
disagree, and the fix is a re-run, not an edit.**

The `PLAN_prod_test_speedup.md` fixes do not help here. Three of its four are
family-agnostic; the fourth — "§6 fit set + both length_pmfs, 13.3 min → 7 s" — is labelled
*"batched n_head"* and is exactly the identity this family does not have.

## 2. What was measured (GB10, real checkpoints, `data/jet_aux_asym_test.root`)

| approach | result |
|---|---|
| batch `sample()` **across jets**, CPU | **nothing.** 61.4 ms/jet best (B=32) vs 61.65 today; *worse* at B=128 |
| compact dead rows inside the sampler | ~2× (1568 alive row-steps needed vs 3062 computed per jet) |
| **run on CUDA** | **~25×.** 2.25–3.5 ms/jet; §6 at 40 000 jets ≈ **41 min → ~2–3 min** |

The loop is **compute-bound in the 900-cell `split_head`**, per step per draw — so more
parallelism does not help and the GPU simply does the same FLOPs faster. B=1 already gives
3.52 ms/jet, so cross-jet batching is not needed.

There is no cheaper algorithmic dodge: `length_pmf` needs only the chain *length*, but the
next step's `p_cont` is conditioned on the drawn cell (`inference/sampling.py:46`), so the
900-way multinomial cannot be skipped.

`h2p-rsd-junipr eval` **already** runs on CUDA unconditionally (`cli.py:196` →
`select_device()`), verified across every eval path. Only this notebook and
`scripts/eval_prod_test_v1.sh` force CPU.

## 3. The load-bearing finding: TF32

CUDA here is **not** "the same numbers, faster". `nn.GRU` — the decoder at
`models/ar_junipr.py:139`, in every sampling path for every arm, plus the `gru` encoder —
is run by cuDNN in **TF32 by default**: a 10-bit mantissa, where every published number in
this repo is fp32.

`max|Δp|` between the batched and batch-1 `length_pmf`, 256 test jets:

| arm | cpu | cuda, box default | cuda, TF32 both off | cuda, matmul TF32 on |
|---|---|---|---|---|
| `v1_base_s0` (lundnet) | 0 | 5.1e-07 | 5.1e-07 | 1.5e-03 |
| **`v1_gru_s0` (gru)** | 0 | **3.8e-04** | 6.3e-07 | 2.1e-03 |
| `v1_deepsets_s0` | 0 | 4.8e-07 | 4.8e-07 | 2.2e-03 |

A tolerance chosen from a `lundnet` arm — where cuDNN is not in the path at all — fails on
`v1_gru_s0` at torch's own defaults. Pinning TF32 off costs **zero** (4.71 vs 4.72 ms/jet;
these paths are launch-latency-bound). **Pin it, and record the pin in the artifact** —
it is the one setting that moves numbers by 1e-3 without changing a line of code, and
`torch.set_float32_matmul_precision("high")` is standard advice anyone might add.

## 4. WP-1 — the notebook (seven cells)

**Cell 2 (§0 params).** `DEVICE = "cpu"` **stays the default**; document `"cuda"` as the
opt-in with the measured 25×. If an `"auto"` is offered it must resolve **cuda-or-cpu,
never mps** — deliberately diverging from the sibling notebooks and
`scripts/lund_closure_report.py:530`, which use `select_device()` (cuda > **mps** > cpu).
[`PLAN_prod_test_speedup_mac.md`](PLAN_prod_test_speedup_mac.md) §21/§217 measured **mps
10–30× slower than cpu** on every path this notebook uses and says *"do not add an `auto`
or `mps` branch"*; inheriting that preference order would be a silent 10–30× regression on
Apple silicon. Apply `TORCH_THREADS` only on the cpu backend and print which branch ran.
Retitle the cost table `MEASURED … on cpu`; append only what was actually re-measured.

**Cell 4 (§1 load).** Resolve the device, then on cuda pin
`torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 =
False`, with §3's table as the comment. Warn loudly on mps.

**Cell 6 (`METRICS["run"]`).** Add `device`, `cuda_device`, `torch_version`, and a `tf32`
block (`matmul`, `cudnn`, `float32_matmul_precision`); change `torch_threads` from the
constant to `torch.get_num_threads()`. CUDA and CPU are different RNG streams *and*
different float kernels, so an artifact that does not name its backend is ambiguous.
Additive only — `make_prod_closure_nb.py` and `lund_closure_report.py` read only
`run.checkpoint` / `test_path` / `train_path`.

**Cell 34 (§6).** Promote the hardcoded `POP[:20000]` to a named `N_PMF_TEST` beside
`N_FIT`, so both halves of the 40 000 are visible and adjustable. Print a projected §6 cost
keyed on `hasattr(model, "n_head")` and `device.type` *before* the two `length_pmfs` calls,
and say: if tempted to cut `N_FIT`, move to cuda instead — `tau` is a quantile and
`(T, tilt)` a 2-parameter fit, and 5 000 jets puts visible MC error on both. Record
`n_report_jets` beside `recalibration.fit_jets`.

**Cell 35 (the blocker).** Bit-identity is a **cpu** property; on cuda the batched and
batch-1 calls select different reduction kernels. Keep exact equality on cpu, `_TOL = 1e-5`
on cuda. The failure this check exists to catch — collate padding reaching the encoder — is
O(1e-2), four orders above the 1e-7 kernel floor; **~1e-3 means TF32 got re-enabled**, and
the message should say exactly that. Print `BIT-IDENTICAL` only when the max is exactly 0.

**Cell 46 (§9).** When the v0 artifact's `run.device` differs from this run's, print that
the deltas carry a second source — different kernels, not just a different RNG stream.
Non-asserting.

**Prose.** Cell 0's *"bit-identical, not merely close"* bullet and
`notebooks/README.md:130-131` gain the cpu/cuda qualifier.

## 5. WP-2 — the eval driver and `cli.py`

Add `--device {cpu|auto|cuda}` to `scripts/eval_prod_test_v1.sh`, **defaulting to `cpu`**
so today's behaviour is unchanged. `cpu` keeps `CUDA_VISIBLE_DEVICES=""` (the only lever —
`eval` has no device flag); other values simply do not set it, and drop the CPU-only
`OMP_NUM_THREADS=2`. The default stays cpu because `scripts/prod_test_v1_gates.py` compares
arms **to each other**, so a half-cpu/half-cuda grid is a silent ranking hazard: flipping
the flag means re-running all 11 arms. Add `"device": str(device)` to `cmd_eval`'s metrics
dict for the same provenance reason — `lund_closure_report.py:1321` already sets that
precedent. Correct the stale premise at `eval_prod_test_v1_stream.sh:6`: CPU-only eval was
a choice for concurrency with training, not a property of eval.

## 6. Non-goal — dead-row compaction in `ancestral_sample_cells`

~49% of row-steps compute already-dead chains, so compaction is worth ~2× on CPU.
**Do not do it**, in this order of weight:

1. **It buys least where we are going.** CUDA is launch/sync-bound (~0.19 ms per sequential
   step). Compaction removes no sequential steps; it only narrows them. The CPU cross-jet
   result — nothing at B=32, worse at B=128 — says the same thing.
2. **The cost is a contract, not a test.** `tests/test_parity.py:95-98` and
   `scripts/verify_parity.py:135-139` assert the sampler is bit-identical to the vendored
   reference at a fixed seed. Demonstrated divergence: `[60,6]` vs `[60,7]`. Not recoverable
   by RNG bookkeeping — variable-batch kernels differ anyway.
3. **The blast radius is larger than it looks.** One class serves all four `ar_junipr` names
   (`ar_junipr.py:57`); the sampler is chosen per *instance* at `:460`. `v1_contstop_s0/s1`
   are v4 with the head **off**, so compaction would move numbers G8 currently quotes.

If revisited: opt-in `compact_dead_rows: bool = False` plus a distributional test, beside
the `vonmises_sample` host-sync item at `PLAN_prod_test_speedup.md` §225-244.

## 7. Verification

1. `pytest tests/test_notebooks.py -q` (AST-parses every cell), then full `pytest tests -q`;
   `python scripts/verify_parity.py` must stay green — the positive proof the sampler was
   **not** touched.
2. `pre-commit run --all-files`, then `git diff --stat` empty (nbstripout is hook + filter).
   The notebook diff must touch only cells 0, 2, 4, 6, 34, 35, 46.
3. **Tolerance check, standalone (~1 min).** Re-measure §3's table for `v1_base_s0`,
   `v1_gru_s0`, `v1_deepsets_s0` in four configs. Accept `_TOL = 1e-5` only if it
   reproduces — this is what proves the TF32 pin does work rather than being cargo cult.
4. **Behaviour-preserving, cpu.** Run end to end with `DEVICE="cpu"` on
   `runs/prod_test_v0/20260731-212800-8209a78a33/best.ckpt` and diff the artifact against
   the committed one: every cell-level number **bit-identical**, with exactly the new
   `run.device` / `run.tf32` / `run.torch_version` keys added.
5. **n_head arm on cuda.** `v1_gru_s0` — the arm that would have failed, *not* lundnet.
   Cell 35 prints a max below 1e-5 and above 0, naming cuda. Then re-run once with the TF32
   pin commented out and confirm cell 35 **fails**: a tolerance nobody has seen fail is
   untested.
6. **The goal.** `v1_contstop_s0`, `DEVICE="cuda"`, `N_FIT=20000`. §6 ≈ 2–3 min; artifact
   shows `fit_jets == 20000`; cell 35 takes the TV branch and passes. **Record end-to-end
   wall clock** — see §8.
7. **CUDA determinism.** Run (6) twice, diff `tau`/`T`/`tilt`. `seed_everything` sets
   `cudnn.deterministic`, but nothing here has ever checked it holds for these ops.
8. Rewrite the §0 cost table from (4)–(6)'s measured per-stage prints. No estimated tables.

## 8. Risks

- **CUDA may make other sections slower.** `beam_search_cells`
  (`inference/point_estimate.py:127-135`) does a `.tolist()` per beam per step at batch 1,
  and `vonmises_sample` (`distributions.py:222`) syncs per rejection iteration. §5 runs 2 000
  MAP decodes. If a section regresses, it goes **in the cost table with a pointer** to the
  follow-ups at `PLAN_prod_test_speedup.md` §225-244 — not swept up, and not fixed by a
  per-section device.
- **Artifacts stop being cross-comparable across devices.** Different RNG stream *and*
  different kernels. Mitigated by recording the device, not by pretending otherwise.
- **The mps trap.** `select_device()` prefers mps over cpu; the mac plan measured that as
  10–30× *slower*. Hence the deliberate divergence in WP-1.
- **A latent crash the mac plan flags** (§221-224): a CPU `torch.Generator` against an MPS
  model raises `Expected a 'mps' device type for generator`. `decode_generator`
  (`models/base.py:74-93`) is already keyed per device; anything new must be too.

## 9. Why this is worth doing beyond the wall clock

For the continue/stop family `length_pmf` draws `n_samples=500` (`models/base.py:292-305`),
so `q(0|x)` carries a per-jet MC standard error of ~0.016 at `p ≈ 0.17` and a resolution
floor of 1/500 — while §6 quotes a Brier *reliability* of 5.0e-04 for that arm. Part of
that decomposition is sampler noise. On CPU that cannot be checked; on CUDA it can — re-fit
at `n_samples=2000` on a 2 000-jet subset and see whether `T`, `tilt` and `tau` move.

## References

`PLAN_prod_test_speedup.md` · `PLAN_prod_test_speedup_mac.md` (the MPS measurement) ·
`PLAN_prod_test_v1.md` · `PROD_TEST_v1_RESULTS.md` §4.8 (the G8 arms this affects).
