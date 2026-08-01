# PLAN — A learned per-jet lower bound on multiplicity (quantile floor of P(n|x))

Status: **implemented, opt-in** — `decode.length_floor_quantile` (default `0.0` == off),
applied in `eval/closure.py::print_point_estimate` and `serving/api.py::predict`, covered
by `tests/test_length_floor.py`. Note it does **not** reach `run_closure`'s `map_or_mbr`,
so the closure table is unfloored whatever this is set to
([`PLAN_prod_test_v0.md`](PLAN_prod_test_v0.md) check 11). Replaces reliance on the hard-coded constant
`decode.min_emissions=1` with an opt-in, per-jet floor derived from the model's own
learned length distribution P(n|x). Builds on `docs/PLAN_NsplitMinCut.md` (merged).

## Context — why this change

The merged fix floors the MAP at a **hard global constant** `decode.min_emissions=1`.
Follow-up: set that lower bound in a **more natural, learned, per-jet** way. The model
already learns a per-jet length distribution P(n|x):

- **cINN/diffusion** carry an explicit categorical head `P(n|x)=softmax(n_head(e))`
  (`src/h2p_rsd_junipr/models/cinn.py:105-108`, `src/h2p_rsd_junipr/models/diffusion.py:56-59`),
  trained on the true `n` (the `logp_n` term in `per_jet_nll`).
- **AR** (`src/h2p_rsd_junipr/models/ar_junipr.py`) has an implicit P(n|x); its empirical
  pmf is the multiplicity histogram of the posterior draws callers already take
  (`len(d) for d in model.sample(...)`).

The joint-argmax MAP is length-biased *low* (residual MAP multiplicity bias −0.11 even
after the constant floor). P(n|x) is the model's unbiased length belief, so flooring the
MAP length at a **low quantile of P(n|x)** transfers that belief into the point estimate
and cuts the residual under-count. One knob `alpha` spans it: `alpha->0` = today's
physical floor; `alpha->median` ≈ a length-conditioned MAP at that quantile.

**Confirmed scope (user):** (A) per-jet **quantile floor**; AR P(n|x) from **reused
posterior samples**; **opt-in** (`length_floor_quantile` default 0.0, so the merged
behavior is unchanged).

## Key insight (validated): rides on the existing `min_emissions` plumbing

Effective per-jet floor `= max(min_emissions, floor(Q_alpha(P(n|x))))`, passed as
`min_emissions=` to the **unchanged** `map_estimate`. So **no change** to
`beam_search_cells` (`src/h2p_rsd_junipr/inference/point_estimate.py:69-118`, already gates
STOP on `len(cells) >= min_emissions`) or the cINN/diffusion `n_star` clamp
(`cinn.py:185`, `diffusion.py:161`). New code is only: the config knob, a `length_pmf`
accessor, two small helpers, and call-site wiring. `alpha=0.0` short-circuits before any
new code path -> structural parity preserved.

## Edits (ordered)

1. **Config knob** — `src/h2p_rsd_junipr/config.py`: add `length_floor_quantile: float =
   0.0` to `DecodeConfig` and to `_DECODE_DEFAULTS` (`decode_params` auto-threads and
   backfills it for old snapshots). Mirror in `configs/decode/default.yaml`.
2. **`length_pmf` accessor** — `src/h2p_rsd_junipr/models/base.py`: add `import numpy as
   np` and a concrete `length_pmf(self, xf, nx, mults=None, n_samples=500) -> np.ndarray`
   on `PosteriorModel`. Default (AR / sampler-based): if `mults is None`, draw
   `self.sample`; return `np.bincount(mults)` normalized to sum 1. Override in `cinn.py` +
   `diffusion.py` to return `softmax(n_head(e))` (exact, cheap; add `import numpy as np`
   there too).
3. **Helpers** — new `src/h2p_rsd_junipr/inference/length.py` (kept out of the
   parity-critical `point_estimate.py`; export from `src/h2p_rsd_junipr/inference/__init__.py`):
   - `quantile_floor(pmf, alpha) -> int`: smallest `n` with `cdf(n) >= alpha`
     (`np.searchsorted(np.cumsum(pmf), alpha)`), clamped to `[0, len-1]`.
   - `learned_min_emissions(model, xf, nx, *, quantile, base_floor, mults=None,
     n_samples=500) -> int`: `quantile<=0 -> base_floor` (short-circuit, no pmf); else
     `max(base_floor, quantile_floor(model.length_pmf(...), quantile))`.
4. **Wire into MAP call sites that already draw samples** (reuse `mults`, no double-sample):
   - `src/h2p_rsd_junipr/serving/api.py` `predict`: reorder to sample first; when `alpha>0`,
     set `min_emissions=learned_min_emissions(..., mults=mults)` in the `map_estimate`
     kwargs.
   - `src/h2p_rsd_junipr/eval/closure.py` `print_point_estimate`: already samples before
     `map_estimate`; inject the same `min_emissions` override. (`run_closure` is
     posterior-mode only — untouched; `cli.py` already threads `decode` — no change.)
5. **Notebook §6a** — `notebooks/inference_demo.ipynb`: add `LENGTH_FLOOR_QUANTILE=0.15`
   param; in the §6a loop compute a 4th estimator **"MAP (learned floor)"** from the per-jet
   `mults` already drawn; add it to the panel / bias-RMSE print and a markdown bullet.
   Demonstrates reduced MAP under-count with n=0 still 0%.

## Tests

- New `tests/test_length_floor.py` (reuse `conftest.batch`): `quantile_floor` unit cases;
  `length_pmf` sums to 1 (all 4 selectors); `length_pmf` matches `softmax(n_head)` for
  cINN/diffusion; `alpha=0` no-op (MAP identical); `alpha=0.9` floors `multiplicity >= eff`
  for all three families.
- `tests/test_config.py`: add the new key to the full-set + old-snapshot backfill
  assertions. `tests/test_decode_plumbing.py`: a `predict` with
  `length_floor_quantile=0.9` floors the MAP up, never below.

## Docs

- `docs/USAGE.md`: the knob + a learned-floor inference snippet.
- `docs/README_PHYSICS.md`: the P(n|x) quantile-floor concept; note `alpha->median` ≈
  length-conditioned MAP.
- `docs/PRODUCTION-PLAN-v4.md`: add the `DecodeConfig` field + a note.
- `notebooks/README.md`: the 4th §6a estimator.

## Verification

1. `python -m pytest tests/test_length_floor.py tests/test_config.py tests/test_decode_plumbing.py tests/test_models.py -q`
2. `python scripts/verify_parity.py` + `python -m pytest tests/test_parity.py -q` —
   bit-for-bit (`alpha=0` is a structural no-op; likelihood untouched).
3. `python scripts/verify_synthetic.py` — posterior bands unchanged.
4. CLI A/B: `eval <ckpt> decode.length_floor_quantile=0.0` vs `=0.5` — MAP length floored up.
5. Re-run `notebooks/inference_demo.ipynb` §6a — 4th panel shows reduced under-count bias,
   n=0 at 0%.
6. `python -m pytest -q` — full suite green.

Run in the conda `fno_env_mlx` environment.

## Risks

- **P(n=0) mass** — low quantile may be 0; `max(base_floor, ...)` keeps the n>=1 guarantee
  (the learned floor only ever *raises* the bound).
- **Large alpha** — approaches a length-conditioned MAP at that quantile (intended;
  documented); a floor above `max_emissions` is absorbed by the beam degenerate-fallback
  and the `min(n_star, n_cells)` bound.
- **config_hash** changes for new runs (same as the merged knobs); old checkpoints load via
  the tolerant `decode_params` / `OmegaConf.select` backfill.
- **AR floor is mildly stochastic** (sampled pmf) — callers already seed; cINN/diffusion are
  exact / deterministic.
- **numpy in base.py** — a new import, but numpy is already a hard dep; keeps the pmf in the
  same form callers' `mults` histograms use.
