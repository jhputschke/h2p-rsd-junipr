# PLAN — Eliminate the unphysical MAP = 0 (empty-tree) collapse

Status: **implemented and on by default** — `decode.min_emissions: 1`,
`decode.length_penalty`, `inference/point_estimate.py::beam_search_cells`, covered by
`tests/test_length_floor.py`. Adds a minimum-emission floor (`N_split ≥
min_emissions`) and length-normalized beam search so the MAP point estimate is never
the unphysical empty tree, plus median reporting and a training probe.

## Context — why this change

`notebooks/inference_demo.ipynb` §6a revealed that the MAP point estimate
`ŷ = argmax_y q_φ(y|x)` returns **0 splittings for ~22% of jets** — unphysical, since a
groomed jet that survived the algorithm has ≥1 primary splitting. Question: *can more
training fix it, or can the model itself avoid unphysical solutions?*

**More training only shrinks it; it is not mere under-training.** The continue/stop head
is fully supervised against the true multiplicity (teacher-forced `cont_tgt = (idx < n)`,
`src/h2p_rsd_junipr/models/ar_junipr.py:150-153`), the checkpoint is at the repo's
converged val NLL (20.72 ≈ v2 reference 20.76), and at step 0 the model wants to emit
(`p_cont0 ≈ 1.000`). MAP=0 is a structural property of the **joint argmax of a
high-entropy discrete sequence distribution**: each emission pays the 100-way cell head's
categorical entropy (~1.4 nats) while "stop" costs a roughly fixed amount, so for
high-multiplicity jets the single most-probable explicit tree scores *below* the empty
tree even though the posterior mass sits at ~6 nodes. This is the classic beam-search
length/brevity bias; the repo already notes "the MAP can be unrepresentative in high
dimensions" (`docs/README_PHYSICS.md:331`). It was never seen before because the old
closure suite reported posterior-mode/mean and printed only one jet's MAP — MAP
multiplicity was never aggregated.

**The model/decoder can avoid it outright.** Confirmed scope: **all three tiers**,
`min_emissions = 1` as the **new default but user-configurable**, for **all three
families** (`ar_junipr`, `cinn`, `diffusion`).

## Key facts (verified)

- `beam_search_cells` (`src/h2p_rsd_junipr/inference/point_estimate.py:69-103`) has no
  length normalization and no min-length; the empty STOP at line 90 can win the global
  argmax.
- cINN/diffusion `map_estimate` (`src/h2p_rsd_junipr/models/cinn.py:181-184`,
  `src/h2p_rsd_junipr/models/diffusion.py:157-160`) early-return empty when
  `n_star = argmax(n_head)` is 0; both ignore `**kw`.
- `cfg.decode` (`src/h2p_rsd_junipr/config.py:136-142`) is **dead code** — never read;
  `map_estimate` always uses hardcoded defaults despite `docs/USAGE.md:327` promising
  "full control over … beam width".
- `config_hash` (`src/h2p_rsd_junipr/config.py:240-244`) hashes the whole config; adding
  fields changes it for **new** runs only. Old checkpoints carry their own snapshot
  (rebuilt via `OmegaConf.create`), so decode reads must tolerate missing keys.
- Parity guard: `per_jet_nll` is asserted bit-for-bit (`tests/test_parity.py`,
  `scripts/verify_parity.py`, atol 1e-5) — any training-side knob must default to a
  no-op. `scripts/verify_synthetic.py` checks **posterior** bands (not MAP), so MAP
  decoding changes don't affect it.

## Design decisions

- **`min_emissions` enforced at the function-signature default (=1)**, not only via
  config — because the notebook/serving call `map_estimate(xf, nx)` with no kwargs and
  must still get n≥1. Config (`cfg.decode.min_emissions`) overrides it.
- **`length_penalty` = GNMT-style division** `score / max(len,1)**alpha` applied only at
  final ranking. `alpha = 0.0` (default) is an exact no-op preserving today's behavior;
  division (not `+alpha*len`) is scale-robust across jets.
- **Cell-head label smoothing** (Tier 3 knob) defaults to `0.0`, gated by `if eps > 0`, so
  the default `per_jet_nll` path is bit-identical → parity holds.
- **Backward-compat**: all decode reads go through one helper using `OmegaConf.select`
  (returns `None`, never raises) + default backfill. No call site may read
  `cfg.decode.<newfield>` directly.
- **Sampling-side n=0 left as-is (documented):** posterior *draws* can still be length 0 (a
  legitimate-but-uncertain draw); truncating them would distort SBC/PIT/coverage.
  `min_emissions` constrains the **point estimate** only. A sampling floor is an opt-in
  future extension, off by default.

## Tier 1 — decoding fix (no retrain)

1. `src/h2p_rsd_junipr/config.py`: add `min_emissions:int=1`, `length_penalty:float=0.0`
   to `DecodeConfig`; add a module-level `decode_params(cfg) -> dict` (7 keys, tolerant
   via `OmegaConf.select` + defaults).
2. `configs/decode/default.yaml`: mirror the two new fields.
3. `src/h2p_rsd_junipr/inference/point_estimate.py` `beam_search_cells`: add
   `min_emissions=1, length_penalty=0.0`; gate both finish sites (lines 90, 99-101) on
   `len(cells) >= min_emissions`; sort `finished` by length-normalized score at the end;
   defensive fallback to best active beam if `finished` empty. Return contract unchanged.
4. `src/h2p_rsd_junipr/models/ar_junipr.py`: `map_decode` gains the two params and forwards
   them; `map_estimate` forwards only the 5 beam keys (curated, to avoid `TypeError` on
   sampling keys).
5. `src/h2p_rsd_junipr/models/cinn.py` + `src/h2p_rsd_junipr/models/diffusion.py`: replace
   `if n_star == 0: return empty` with `n_star = max(n_star, kw.get("min_emissions", 1))`
   (keep existing `min(n_star, n_cells)` bound).
6. Plumb `decode_params(cfg)` into the three live call sites:
   `src/h2p_rsd_junipr/serving/api.py` `predict` (beam keys → `map_estimate`;
   `n_posterior_samples` → `sample_batch`); `src/h2p_rsd_junipr/eval/closure.py`
   `run_closure` / `print_point_estimate` (add optional `decode=None` param);
   `src/h2p_rsd_junipr/cli.py` `cmd_eval` (compute from the checkpoint's snapshot cfg, pass
   down).

## Tier 2 — reporting (posterior median, keep mean)

- `src/h2p_rsd_junipr/serving/api.py` `predict`: add `posterior_mult_median` (additive,
  non-breaking).
- `src/h2p_rsd_junipr/eval/closure.py`: accumulate per-jet `np.median(mults)`; add
  `mult_bias_posterior_median` to `run_closure` metrics + a print line; add `median=` to
  the `print_point_estimate` per-jet line.
- Notebook already has the median panel; only its README narrative needs a touch.

## Tier 3 — training probe

- `src/h2p_rsd_junipr/config.py` `ARJuniprConfig`: add `cell_label_smoothing:float=0.0`;
  mirror in `configs/model/ar_junipr_v*.yaml`.
- `src/h2p_rsd_junipr/models/ar_junipr.py` `per_jet_nll`: when `eps>0`, smooth the split
  term `((1-eps)*chosen + eps*mean_logp)`; `eps==0` keeps the exact current ops (parity).
- New `scripts/probe_map_collapse.py` (modeled on `verify_synthetic.py` + `sweep.py`):
  sweeps `{max_epochs} × {cell_label_smoothing} × {min_emissions}` (AR) and `{max_epochs}
  × {min_emissions}` (cinn/diffusion); per variant prints a fixed-width table of
  `map0_frac, nll/jet, bias_map, bias_mean, bias_median`. Baseline row `(epochs=20, ls=0,
  min_emissions=0)` reproduces the ~0.22 collapse; `min_emissions=1` must drive
  `map0_frac → 0` with NLL within band. Shows whether extra epochs / smoothing reduce the
  *underlying* pressure vs. the floor just masking it.

## Tests

- `tests/test_models.py`: assert `mp.multiplicity >= 1` (default) for all families; new
  `test_map_respects_min_emissions` (min_emissions=3 → ≥3; =0 may return 0);
  `test_length_penalty_is_noop_at_zero` (AR).
- `tests/test_config.py`: `cfg.decode.min_emissions == 1`; `decode_params` backfills
  missing keys without raising (old-snapshot guard).
- `tests/test_checkpoint.py`: a snapshot lacking the new decode keys round-trips its own
  `config_hash` (resume guard intact).
- New `tests/test_decode_plumbing.py`: `predict` output has `posterior_mult_median` and
  `map_multiplicity >= 1`.

## Docs

- `docs/USAGE.md`: fix the `model MAP = 0` example output; document `map_estimate(...,
  min_emissions=, length_penalty=)` and that decode is now plumbed from `cfg.decode`; add
  `median` to record/serving examples.
- `docs/README_PHYSICS.md`: expand the MAP-caveat into the empty-tree collapse + the
  floor/median fix; note the optional GNMT length penalty; note cINN/diffusion now report
  a "constrained MAP under a minimum-emission floor".
- `docs/PRODUCTION-PLAN-v4.md`: add the two `DecodeConfig` fields + a note; mention MAP
  **and** posterior mean **and median** in reporting.
- `notebooks/README.md`: note §6a now reads ~0% with `min_emissions=1`.

## Verification

1. `python -m pytest tests/test_models.py tests/test_config.py tests/test_decode_plumbing.py -q`
   — min_emissions/median/plumbing pass.
2. `python scripts/verify_parity.py` + `pytest tests/test_parity.py -q` — bit-for-bit
   `per_jet_nll` (proves smoothing default is a no-op).
3. `python scripts/verify_synthetic.py` — posterior bands still PASS; printed
   `print_point_estimate` now shows `MAP >= 1` and a `median=` field.
4. `python scripts/probe_map_collapse.py model=ar_junipr_v2 encoder=gru --min-emissions 0,1`
   (repeat for cinn, diffusion) — `min_emiss=1` row reports `map0_frac = 0.000`, NLL in band.
5. Re-run `notebooks/inference_demo.ipynb` §6a: the `MAP mode-collapse (n=0) frac` reads
   ~0% (the no-kwarg `map_estimate` call hits the signature-level `min_emissions=1`).
6. `python -m pytest -q` — full suite green.

Run the above in the conda `fno_env_mlx` environment.

## Risks

- **config_hash changes for new runs** — expected; run-dir names change. Resume re-hashes
  the checkpoint's own snapshot (self-consistent), so the guard still matches; covered by
  the new `test_checkpoint` case.
- **Old checkpoints loadable** — only via `decode_params`/`OmegaConf.select`; never
  `cfg.decode.<newfield>` directly.
- **`**kw` key collision** — call sites pass only beam keys to `map_estimate`, only
  sampling keys to `sample`/`sample_batch`.
- **cINN/diffusion clamp inflates reported MAP logprob** — intended (physical 1-emission
  tree over the unphysical empty one); documented as "constrained MAP".
