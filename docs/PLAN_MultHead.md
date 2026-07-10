# PLAN — Multiplicity head: promote length-conditioning to a first-class factorization

**Status:** proposed (not yet implemented)

Elevate length-conditioning on the autoregressive JUNIPR from a decoding trick to
model structure: a dedicated, calibrated multiplicity head realizing

    q(y | x) = q(N | x) · q(y | N, x)

directly on `ARJunipr`. Opt-in, defaulting off, so with/without the head is a
single config flip and the current likelihood/MAP path stays bit-for-bit identical.
Builds on the merged `PLAN_NsplitMinCut.md`, `PLAN_QuantileMinCut.md`, and
`PLAN_MBR_PerturbativeLund.md`.

## Context

The primary posterior model, [ar_junipr.py](../src/h2p_rsd_junipr/models/ar_junipr.py),
encodes the emission multiplicity `N` **implicitly**, as a product of per-step
continue/stop Bernoulli probabilities from `cont_head` (ar_junipr.py:76, trained
at ar_junipr.py:148-155). Length is therefore never a well-calibrated marginal;
it is the tail of an autoregressive product. This is the source of two documented
pathologies: MAP collapse toward `n=0` (`scripts/probe_map_collapse.py`) and a
signed marginal-multiplicity bias in ancestral draws (the "+0.86"-style bias),
which the current stack only patches downstream via `min_emissions` /
`length_floor_quantile` ([inference/length.py](../src/h2p_rsd_junipr/inference/length.py))
— a *decoding* trick.

A dedicated multiplicity head (i) kills the short-sequence mode degeneracy at its
source — the argmax over `N` is now over a low-dimensional, well-calibrated
categorical marginal rather than an implicit product of continue-probabilities —
and (ii) gives a clean handle on exposure bias: ancestral draws inherit a
calibrated multiplicity marginal, so the MBR candidate pool
([inference/mbr.py](../src/h2p_rsd_junipr/inference/mbr.py)) is far less skewed. It
matches recent external validation — the first high-precision generative-unfolding
framework for jet substructure is a staged pipeline whose first stage unfolds
multiplicity, with kinematics generated conditional on it (arXiv:2510.19906).

The cINN and diffusion families **already** implement this exact factorization
(`n_head` → `logp_n`, `length_pmf` override, sample-N-then-cells), so this is a
mirror of a proven in-repo pattern, not new machinery.

## Design (recommended approach)

Add a categorical multiplicity head to `ARJunipr`, gated by a new bool
`use_multiplicity_head` on `ARJuniprConfig` — the same opt-in idiom as
`continuous_coords` (a bool that gates a whole head + likelihood term, config.py:88,
ar_junipr.py:170-177) and `cell_label_smoothing` (a `getattr`-tolerant field,
ar_junipr.py:59).

**When the head is ON** (`use_multiplicity_head=True`):

- **Head:** build `self.n_head = Sequential(Linear(ctx,ctx), ReLU, Linear(ctx, max_emissions+1))`
  — copied verbatim from cINN (cinn.py:106-109). Do **not** build `cont_head`.
- **Likelihood** (`per_jet_nll`): replace the continue/stop term (`cont_ll`) with
  `logp_n`, gathered at the true length exactly as cINN (cinn.py:133-135):
  `-(logp_n + split_ll + coord_ll)`. The cell/coord terms are unchanged — they are
  already teacher-forced over the true length, so truncation at `N` gives a proper
  conditional `q(y|N,x)`. This is a valid normalized density:
  `Σ_y q(y|x) = Σ_N q(N|x)·1 = 1`.
- **Sampling:** draw `N_k ~ q(N|x)` from `softmax(n_head(e))`, then decode each chain
  to exactly `N_k` cells (no stop head). `cont_temperature` still applies to the cell
  logits (unchanged meaning).
- **MAP:** `N* = max(argmax q(N|x), min_emissions)`, then greedy-decode exactly `N*`
  cells. `length_floor_quantile` composes unchanged (it only raises `min_emissions`
  via `learned_min_emissions`, inference/length.py:34).
- **`length_pmf`:** override to the exact `softmax(n_head(e))` — copied from
  cinn.py:178-182. This is the single P(N|x) accessor already consumed by
  `learned_min_emissions` and closure, so everything downstream slots in for free.

**When the head is OFF** (default): the code path, module list, and `state_dict`
are byte-identical to today — `n_head` is never constructed, `cont_head` is, and
`per_jet_nll` runs the current lines unchanged. This preserves parity and lets
strict `load_state_dict` (checkpoint.py:90) keep loading pre-existing AR checkpoints.

**Kinematics conditioning = truncate-only.** `N` sets the step count; the cell/coord
heads are not fed `N` explicitly (they see it only through the AR state + truncation).
This is the smallest coherent diff and the exact structural analog of cINN.

> **Future extension (documented, not built now): feed `N` into the decoder.**
> To make the kinematics *explicitly* conditional on the total `N` (closer to the
> arXiv:2510.19906 staged pipeline), add a small `n_embed = nn.Embedding(max+1, d)`
> (or a scalar "remaining-count countdown" feature) and concatenate it to `dec_in`
> (ar_junipr.py:71,108) and to each head input (ar_junipr.py:76-82,147). This
> changes head input dims (new params, larger diff, more decode branching) but
> remains a valid factorization. Left as an opt-in follow-up behind a second flag
> (e.g. `n_conditioning: "none" | "embed"`) so the truncate-only model stays the
> default comparison point.

## Concrete edits

**Model config — [config.py](../src/h2p_rsd_junipr/config.py)**
- `ARJuniprConfig`: add `use_multiplicity_head: bool = False` and `max_emissions: int = 25`
  (categorical size when the head is on; mirrors `CINNConfig.max_emissions`,
  config.py:101). Read both in the model via `getattr(m, ..., default)` for
  old-checkpoint tolerance (like `cell_label_smoothing`, ar_junipr.py:59).
- Register the head-on variant idiomatically (mirrors the v1/v2 split):
  add `"ar_junipr_v3": ARJuniprConfig` to `MODEL_SCHEMA` (config.py:190), add
  `"ar_junipr_v3"` to the `@register_model(...)` decorator (ar_junipr.py:44), and
  add `configs/model/ar_junipr_v3.yaml` (`continuous_coords: true`,
  `use_multiplicity_head: true`). Users then toggle via `model=ar_junipr_v3` **or**
  `model=ar_junipr_v2 model.use_multiplicity_head=true`.
- `DecodeConfig` + `_DECODE_DEFAULTS` (config.py:139-164, 279-299): add
  `mbr_resample_to_qn: bool = False` (kept in both, per the `decode_params` contract).

**Model — [ar_junipr.py](../src/h2p_rsd_junipr/models/ar_junipr.py)**
- `__init__`: read the flags; gate `n_head` vs `cont_head` construction.
- `per_jet_nll` (ar_junipr.py:137): branch — head-on computes `logp_n` and returns
  `-(logp_n + split_ll + coord_ll)`; head-off is the current code verbatim.
- Add `_step_cells(tok, e, h) -> (split_logits, h)` (a `cont_head`-free sibling of
  `_step_batched`, ar_junipr.py:193) reused by the head-on sampler and greedy MAP.
- `sample` (ar_junipr.py:202): head-on draws `N_k ~ softmax(n_head(e))` then calls the
  new fixed-length sampler; head-off unchanged.
- `map_estimate`/`map_decode` (ar_junipr.py:218-298): head-on computes floored `N*`
  and greedy-decodes exactly `N*` cells; head-off unchanged.
- `describe_sequence` (ar_junipr.py:231): head-on total log-density =
  `logp_n(L) + Σ(split+coord)` (drop the `logp_cont`/`logp_stop` terms, set
  `node.logp_cont=0`) so `.logprob` stays consistent with `log_prob` (keeps MBR
  `.logprob` and NLL parity coherent).
- Override `length_pmf`: head-on → exact `softmax(n_head(e))` (copy cinn.py:178-182);
  head-off → `super().length_pmf(...)` (base sampler histogram, base.py:96).

**Sampler — [sampling.py](../src/h2p_rsd_junipr/inference/sampling.py)**
- Add `ancestral_sample_cells_fixed_length(step_cells, e, h0, start_token, lengths,
  device, cont_temperature=1.0)`: same batched on-device structure as
  `ancestral_sample_cells` (sampling.py:20) but decodes `max(lengths)` steps and keeps
  cell `(k,t)` while `t < lengths[k]` — no continue/stop draw.

**MBR reporting + reweighting scaffold**
- [closure.py](../src/h2p_rsd_junipr/eval/closure.py) `run_closure` (closure.py:77): add
  **per-N stratified** signed multiplicity bias. Bin jets by true `N` (`ny_true`,
  e.g. `[1-3],[4-6],[7-10],11+`) and accumulate `n_mean_bias` / `n_median_bias` /
  `n_mbr_bias` (already computed per jet at closure.py:119-127) per bin; emit a small
  table (`mult_bias_by_N`) beside the existing scalars. This is the headline test for
  whether the marginal bias propagates into MBR, stratified by true `N`.
- [mbr.py](../src/h2p_rsd_junipr/inference/mbr.py) `mbr_select` (mbr.py:315): add
  `resample_to_qn: bool = False`. When set, form importance weights over the support
  draws to match the calibrated marginal —
  `w_k = q(N=|y^(k)| | x) / p_emp(N=|y^(k)|)` from `model.length_pmf(xf, nx,
  mults=[len(d) for d in draws])` and the draw histogram — and use a **weighted** risk
  `risk = (D @ w) / w.sum()` in place of `D.mean(axis=1)` (mbr.py:344). This corrects
  the Monte-Carlo risk's multiplicity marginal at the **decoding layer only** — the
  trained likelihood is untouched, so ratio analyses are unaffected (contrast:
  minimum-risk / sequence-level fine-tuning would distort it). Wire `mbr_resample_to_qn`
  through `mbr_kwargs_from_decode` (mbr.py:50). Keep candidate generation as unbiased
  ancestral sampling (no beam / low-temperature candidates), per the design intent.

## Backward compatibility

- Default `use_multiplicity_head=False` ⇒ AR `state_dict` and `per_jet_nll` are
  byte-identical to today; strict `restore_into` / `load_for_inference`
  (checkpoint.py:90-109) load old AR checkpoints unchanged.
- New config fields are read via `getattr`/`decode_params` backfill, so old snapshots
  whose config predates them still rebuild (the `test_checkpoint.py:60-94` pattern).
- Turning the head on changes `config_hash`, so `Trainer.resume` (trainer.py:167)
  correctly refuses to resume an off-run into an on-run — expected.

## Tests (mirror `tests/test_*.py` conventions, `conftest.py` fixtures)

- **New `tests/test_multiplicity_head.py`:** head-on `log_prob` finite and decomposes
  as `logp_n + split + coord`; `length_pmf` exact, normalized, `== softmax(n_head(e))`;
  sampled length distribution tracks `q(N|x)` (statistical); MAP never returns `n=0`
  with `min_emissions=1`; greedy MAP length `== N*`.
- **Parity (extend `tests/test_parity.py`):** head **off** reproduces today's
  `per_jet_nll` bit-for-bit against `scripts/reference/conditional_rsd_junipr_v2.py`.
- **Checkpoint (extend `tests/test_checkpoint.py`):** head-on model round-trips;
  a saved off-model / old snapshot (no `n_head` key) still loads strictly.
- **MBR (extend `tests/test_mbr.py`):** `mbr_resample_to_qn=False` == current argmin
  (parity); reweighting changes selection under a skewed pmf yet still never selects the
  empty tree when non-empty draws dominate (the existing headline invariant,
  test_mbr.py:156).
- **Config (`tests/test_config.py`, `tests/test_decode_plumbing.py`):** new fields
  present, defaulted, and backfilled by `decode_params`.

## Verification (end-to-end)

1. `pytest tests/ -q` — new + existing suites, especially parity and checkpoint.
2. **Parity guard:** `python scripts/verify_parity.py` (head off ⇒ unchanged NLL).
3. **Train a head-on model on synthetic** and eval closure:
   `h2p-rsd-junipr train model=ar_junipr_v3 trainer.fast_dev_run=true` then
   `h2p-rsd-junipr eval <run> decode.point_estimator=mbr` — confirm the per-N
   stratified multiplicity-bias table prints and that MAP `map0_frac ≈ 0`
   (contrast `scripts/probe_map_collapse.py` on a v2 run).
4. **Exposure-bias measurement:** compare `mult_bias_mbr` (stratified by true N)
   across four cells — {v2 head-off, v3 head-on} × {`mbr_resample_to_qn` off/on} — to
   quantify how much marginal bias propagates into MBR and how much the reweighting
   removes.
