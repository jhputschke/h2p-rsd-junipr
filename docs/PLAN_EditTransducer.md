# PLAN — Edit transducer: hadron→parton as a learned smearing + birth/death process

**Status: proposed.** A **fourth model family** (`edit_v1` / `edit_v2`) beside §5.1 AR
JUNIPR, §5.2 cINN, §5.3 diffusion/CFM. It changes the *factorization* of
`q(y|x)` — not the target, not the objective, not the geometry — by making the
hadron-level tree the **anchor** of the parton-level tree rather than only a
conditioning vector. Nothing in
[ar_junipr.py](../src/h2p_rsd_junipr/models/ar_junipr.py) is touched, so likelihood
parity (`scripts/verify_parity.py`, `tests/test_parity.py`) is unaffected by
construction. Builds on `PLAN_MultHead.md`, `PLAN_MBR_PerturbativeLund.md`,
`PLAN_empty_parton_tree.md`, and `PLAN_UPDATES.md` WP3.

## Context

Every family in the repo generates `y` **from scratch**, conditioned on `x` only
through the encoder — the pooled `e(x)` (v1–v3) or the per-node states under
cross-attention (v4, `PLAN_UPDATES.md` WP3). The decoder must therefore *relearn*
that `y ≈ x` wherever hadronization is weak. The closure suite already says this is
costing us: `dlund_identity` — the "treat `x` as if it were the truth" baseline —
beats `dlund_posterior_mode`
([closure.py:260-262](../src/h2p_rsd_junipr/eval/closure.py)). At high `k_t` the
hadron node sits essentially on the parton node, so *do nothing* is a strong
estimator, and a from-scratch decoder spends capacity rediscovering it.

The physics says the residual is a **kernel**, not a fresh draw. Local
parton–hadron duality (Azimov, Dokshitzer, Khoze & Troyan, *Z. Phys. C* **27**
(1985) 65) is exactly the statement that the hadron configuration tracks the parton
configuration up to a local, power-suppressed smearing; organized as a convolution
this is the shape-function picture (Korchemsky & Sterman, hep-ph/9902341). Three
properties make it *learnable and structured* rather than generic noise:

- the smearing scale runs as `Λ_eff / k_t`, so the width is a predictable function
  of the node's own coordinates;
- for a groomed jet, the nonperturbative correction scales with the catchment
  geometry, hence with `R_g` and `z_g` (Hoang, Mateu, Pathak, Stewart et al.,
  arXiv:1906.11843) — heteroscedasticity with a physical argument;
- prong multiplicity proxies color charge through Casimir scaling of the emission
  density (Dreyer, Salam & Soyez, arXiv:1807.04758).

Births and deaths are equally structured, and the current data shows how large they
are: in `cpp/test_data/jets.root` (PYTHIA 8.3, `z_cut=0.1`, 1 GeV floor) **6.9% of
jets have no hadron-level primary emission and 16.0% have no parton-level one**
(`tests/test_empty_sequences.py`). Hadron-level nodes below the perturbative floor
have no parton image (deletions); parton nodes whose hadron image migrated across
the grooming boundary have no hadron anchor (insertions).

The obstruction is that **node-level parton↔hadron correspondence is not
observable** — the same wall HOMER hits in hadronization fitting (Bierlich et al.,
arXiv:2410.06342; Assi et al., arXiv:2503.05667). So the alignment must be a
**latent variable that is marginalized**, never a supervised per-node target. That
is precisely a pair-HMM / neural transducer: latent monotone alignments summed by
dynamic programming (Graves, arXiv:1211.3711; CTC: Graves et al., ICML 2006;
training exactly through the lattice: Imputer, Chan et al., arXiv:2002.08926). The
edit-based decoding literature (Insertion Transformer, Stern et al.,
arXiv:1902.03249; Levenshtein Transformer, Gu et al., arXiv:1905.11006) resorts to
heuristic surrogates because its lattices are enormous. **Ours are not:** with
`n_x, n_y ≲ 25` the `O(n_x · n_y)` forward recursion is exact, cheap, and fully
differentiable. The lattice size is the reason this family is attractive here and
not in NLP.

Distribution-level analogues exist — MC-derived bin-by-bin hadron→parton corrections
in Lund-plane measurements (cf. ATLAS, arXiv:2004.03540), and staged generative
unfolding that produces multiplicity first and kinematics conditionally
(arXiv:2510.19906). The edit transducer is the **per-jet, probabilistic**
generalization of the former and gets the latter's staging for free (see *Exact
length marginal*).

## Design (recommended approach)

### The generative process

State `(i, j)`: `i` hadron nodes of `x` consumed, `j` parton nodes of `y` emitted.
Alignments are monotone in the angular ordering both sequences already carry.

- at `i < n_x`: a categorical over `{ADVANCE, EMIT}`;
- at `i = n_x`: a categorical over `{STOP, EMIT}` (trailing insertions);
- an `EMIT` produces `y_{j+1}` from a **two-component mixture**

      p_anch(i,j) · f_shift(y | x_i, ·)  +  (1 − p_anch(i,j)) · f_free(y | ·)

  and advances `j`; `ADVANCE` consumes `x_i` with no emission.

Read off the semantics: an `ADVANCE` with no preceding anchored emit at that column
is a **deletion**; an anchored emit is a **kept, smeared** node; a free emit is an
**insertion**. This is the RNN-T lattice, so `Σ_y q(y|x) = 1` holds by construction
(Graves, arXiv:1211.3711) — no bespoke normalization argument, which matters because
`exact_likelihood = True` is a contract flag other code trusts
([base.py:36-41](../src/h2p_rsd_junipr/models/base.py)).

    log q(y|x) = logsumexp over monotone paths,  forward recursion α(i,j), O(n_x·n_y)

evaluated in the log domain. Deletion/insertion counts are *not* free parameters: the
multiplicity is `n_y = n_x − #del + #ins`, so length is anchored at `|x|` and the
open-ended continue/stop mechanism — the documented seat of the `+0.86`-style
marginal bias and of MAP collapse (`scripts/probe_map_collapse.py`) — is **removed
structurally**, not recalibrated downstream.

### Anchors come off `xf` — no data-layer change

`node_features` ([features.py:194](../src/h2p_rsd_junipr/features.py)) stores
`(ln 1/ΔR, ln k_t, ln z, sin ψ, cos ψ)` **unstandardized**, so the anchor coordinates
are `xf[..., :3]` plus `atan2(xf[..., 3], xf[..., 4])`. Aux columns
(`PLAN_Input.md`) append *after* index 4, so widening is harmless.
[dataset.py](../src/h2p_rsd_junipr/data/dataset.py) and `collate` are untouched.

### Heads (all reusing existing distributions)

| piece | parametrization | reuse |
|---|---|---|
| op logits | MLP on `[s_i, c_j, e]` → 2 logits | — |
| `p_anch` | same MLP, third logit | — |
| shift in `(ln 1/ΔR, ln k_t)` | truncated normal centered at the anchor, truncated to the geometry range | `trunc_normal_logpdf` ([distributions.py:42](../src/h2p_rsd_junipr/distributions.py)) |
| shift in `ln z` | normal centered at the anchor | `gauss_logpdf` (:28) |
| shift in `ψ` | von Mises centered at `ψ_i` | `vonmises_logpdf` (:114) |
| free emission | cell categorical + within-cell offsets | the v2 `split_head` / `coord_head` pattern ([ar_junipr.py:209-232](../src/h2p_rsd_junipr/models/ar_junipr.py)) |

`s_i` are the encoder's per-node states via `Encoder.forward_seq`
([encoders/base.py](../src/h2p_rsd_junipr/encoders/base.py)); `returns_sequence=True`
is therefore a hard requirement, satisfied by `gru`, `lundnet`, `deepsets` alike.

Truncating the plane shifts to the geometry range (rather than using an unbounded
normal) keeps the density normalized on exactly the support the geometry defines and
inherits the existing support guard (`tests/test_support_guard.py`).

**Physics-form width (the point of the exercise).** Parametrize

    σ_{ln k_t}(i) = σ_0 + Λ_eff · exp(−ln k_t^{(i)})

with learnable `(σ_0, Λ_eff)` and optional mild `(R_g, N)` dependence, rather than a
free MLP output. This makes the learned kernel *directly confrontable* with the
shape-function expectation (arXiv:1906.11843) instead of being an opaque function,
and it is a strong regularizer in the low-statistics tail. Keep the free-MLP variant
behind a flag as the ablation.

### Two stages

- **`edit_v1` (pair-HMM).** Ops and shifts conditioned on `(i, s_i, e)` only. No
  prefix conditioning at all ⇒ **zero exposure bias anywhere**, at the cost of
  conditionally independent shifts (no recoil correlation among `y` nodes).
- **`edit_v2` (transducer).** Add a prediction network over the emitted prefix. The
  RNN-T trick keeps this cheap: the prefix state `c_j` depends only on `j`, so it is
  `O(n_y)` GRU steps computed once per column, then joined with `s_i` across the
  lattice. Teacher forcing enters the *prefix* only — never the length.

Let held-out NLL adjudicate. Do not build `edit_v2` before the stage-1 diagnostic
below says the anchoring assumption holds.

### Exact length marginal — free, and better than a head

Marginalize the coordinates: emission densities integrate to 1, leaving a purely
**structural** DP over `(i, j)` whose terminal value gives `q(N=n | x)` exactly, in
`O(n_x · n_max)`, **with no extra parameters**. Override `length_pmf`
([base.py:224](../src/h2p_rsd_junipr/models/base.py)) with it. This delivers what
`PLAN_MultHead.md`'s `n_head` delivers — a calibrated, first-class `q(N|x)` — but
exact rather than learned, and explicitly conditioned on `|x|`.

Two consequences worth naming. First, `empty_gate`
([length.py:140](../src/h2p_rsd_junipr/inference/length.py)) now reads an *exact*
`q(N=0|x)`: the empty parton tree (16.0% of jets) is the delete-all path, which this
family represents natively — `PLAN_empty_parton_tree.md`'s decode-layer threshold
becomes a diagnostic rather than a necessity. Second, `learned_min_emissions` and
`length_floor_quantile` compose unchanged, since they only consume `length_pmf`.

### Point estimate, sampling, coordinates

- **`map_estimate`**: exact MAP requires an argmax over a marginal-over-alignments and
  is intractable; use the Viterbi path plus head modes and label it a **surrogate**
  (the same honesty pattern as `Diffusion.exact_likelihood=False`). Note that MAP
  collapse to `n=0` is structurally suppressed here: it needs `ADVANCE` at all `n_x`
  columns *and* `STOP`, where v1/v2 need one stop draw.
- **Default `decode.point_estimator=mbr`** for this family; `map_or_mbr`
  ([base.py:195](../src/h2p_rsd_junipr/models/base.py)) supplies MBR with zero
  per-family code once `sample` and `sample_coordinates` exist.
  [mbr.py](../src/h2p_rsd_junipr/inference/mbr.py) is untouched.
- **`sample`**: ancestral walk over the lattice; coordinates from the drawn mixture
  component; cells via `geometry.to_cell` ([geometry.py:58](../src/h2p_rsd_junipr/geometry.py)).
- **`sample_coordinates(cells)`**: the one genuinely new inference routine.
  Coordinates are *not* conditionally independent of the alignment given the cell
  chain, so this runs a **constrained** forward–backward over paths consistent with
  those cells, samples an alignment, and draws coordinates from the corresponding
  component truncated to the cell. `O(n_x · n_y)`. Required by `describe_cells`
  ([base.py:144](../src/h2p_rsd_junipr/models/base.py)) and hence by MBR.
- **`coordinate_cdfs`**: stage 1 returns `None` (`supports_coordinate_pit=False`).
  The exact prefix-conditional CDF `F(y_j | y_{<j}, x)` is available from the same
  forward recursion as a responsibility-weighted mixture of `trunc_normal_cdf` (:52),
  `gauss_cdf` (:37), `vonmises_cdf` (:151). Land it in `edit_v2` once the DP is
  trusted; the WP2 calibration suite then consumes it with no changes.

### Physics diagnostics that fall out of the DP

Forward–backward responsibilities `γ(i,j)` are a *posterior over alignments* — the
emergent-alignment readout, obtained without ever supervising one:

1. **conditional residual widths** binned in `(k_t, R_g, N)` → the falsifiable check
   motivating the whole family: do they collapse onto `σ = σ_0 + Λ_eff/k_t`?
2. **deletion rate vs `ln k_t`** → should track the sub-floor fragmentation population.
3. **free-emission (insertion) rate vs distance to the grooming boundary** → the
   kernel-boundary object; the quantity that would make a multi-`(z_cut, β)` study
   identifiable, since hadronization does not know about `z_cut` and only the
   acceptance boundary moves.
4. **crossing-pair count** in sampled alignments → the monotonicity audit (risks).

> **Out of scope, deliberately.** A **full-tree** target turns alignment
> marginalization into a tree-edit DP (`O(n_x² n_y²)`-class, different machinery) —
> that is a separate plan, and it is a *decoder* change of the same kind as recursive
> JUNIPR (arXiv:1804.09720). Likewise the bridge/diffusion reading of the same
> physics (I²SB, arXiv:2302.05872; stochastic interpolants, arXiv:2303.08797; jet
> unfolding with bridges, arXiv:2308.12351) shares nothing with the sequence stack
> and belongs in §5.3, where transdimensionality is the open obstacle.

## Concrete edits

**New — [models/edit_dp.py](../src/h2p_rsd_junipr/models/edit_dp.py)**
Pure tensor functions, **no `nn.Module`**, so the numerics are testable in isolation:
`forward_logsumexp`, `forward_backward_responsibilities`, `structural_length_pmf`,
`constrained_forward_backward` (cell-chain conditioned), `viterbi_path`. Batched over
jets with `(nx, ny)` masks; log-domain throughout.

**New — [models/edit.py](../src/h2p_rsd_junipr/models/edit.py)**
`EditTransducer(PosteriorModel)`, `@register_model("edit_v1", "edit_v2", "edit")`.
Implements `log_prob`, `sample`, `map_estimate`, `sample_coordinates`, `length_pmf`;
sets `exact_likelihood=True`, `has_continuous_coords=True`,
`supports_coordinate_pit=False`, `aux_feature_names` from `configured_aux_names`
(the v1–v4 idiom, [ar_junipr.py:93-95](../src/h2p_rsd_junipr/models/ar_junipr.py)).
Raises a clear error when `encoder.returns_sequence is False`, mirroring the v4
cross-attention check (ar_junipr.py:99-106).

**Registry — [models/base.py](../src/h2p_rsd_junipr/models/base.py) and
[models/\_\_init\_\_.py](../src/h2p_rsd_junipr/models/__init__.py)**
Add `edit` to the two side-effect import lines (base.py:243, base.py:265) and export
`EditTransducer`.

**Config — [config.py](../src/h2p_rsd_junipr/config.py)**
`EditTransducerConfig`: `ctx_dim`, `op_head_layers`, `shift_head_layers`,
`free_head_layers`, `sigma_floor`, `kappa_max`, `max_emissions`,
`physics_width: bool = True` (the `σ_0 + Λ_eff/k_t` form; `False` = free MLP
ablation), `prefix_conditioning: bool` (`False` = `edit_v1`, `True` = `edit_v2`).
Register `"edit_v1"` / `"edit_v2"` in `MODEL_SCHEMA` (config.py:290). **No new
`DecodeConfig` fields** — `point_estimator`, `empty_threshold`,
`length_floor_quantile` already exist and all compose.

**Configs — `configs/model/edit_v1.yaml`, `configs/model/edit_v2.yaml`**
Same header-comment convention as `configs/model/ar_junipr_v4.yaml`, stating the
`returns_sequence` requirement and the v1/v2 difference.

**Closure — [eval/closure.py](../src/h2p_rsd_junipr/eval/closure.py)**
Add `frac_anchored`, `delete_rate`, `insert_rate` to the metrics dict (closure.py:263),
family-gated by `getattr(model, "edit_summary", None)` so the other families are
untouched.

**Unchanged, by design:** `dataset.py` / `collate`, `features.py`, `geometry.py`,
`distributions.py`, `mbr.py`, `length.py`, `trainer.py`, `checkpoint.py`, `serving/`,
and **all of `ar_junipr.py`**.

## Why retrofit rather than start fresh

Accounting against `prod_test_v0` as it stands: **reused unchanged** — the data layer
and `collate`, `Geometry`, all six `distributions.py` densities and their CDFs/samplers,
the entire encoder registry (`forward_seq` already exists from WP3), `LundNode` /
`LundPointEstimate`, the whole of `inference/mbr.py` and `inference/length.py`,
`Trainer`, checkpointing, the OmegaConf schema machinery, the closure/calibration
suites, and the serving layer. **New** — two files, of which one is pure numerics.

The decisive argument is not the file count, though. Attributing any improvement to
the *edit factorization* — rather than to incidental differences in geometry,
matching, or optimization — requires both families under one harness with one
geometry and one dataset. A fresh repo would re-implement ~90% of the surface **and
still need `prod_test_v0` alive** to make the comparison. The `PosteriorModel`
contract (base.py:84-143) was designed for exactly this; `edit` is its fourth
consumer, not a special case.

## Backward compatibility

- Purely additive: no existing family's `state_dict`, `log_prob`, or decode path
  changes ⇒ `scripts/verify_parity.py` and `tests/test_parity.py` pass untouched,
  and every existing checkpoint reloads.
- `registered_models()` grows; `tests/test_models.py:11-18` parametrizes over an
  explicit `MODELS` list, so the new family must be **added there** and satisfy the
  generic contract suite from day one — that is the intended gate, not an obstacle.
- New config fields live on a new schema class, so no old snapshot needs backfilling.

## Tests (mirror `tests/test_*.py` conventions, `conftest.py` fixtures)

- **New `tests/test_edit_dp.py` — headline:** for `n_x, n_y ≤ 4`, **enumerate every
  monotone path explicitly** and assert the forward recursion reproduces the
  enumerated `log q(y|x)` to `1e-6`. This is the entire novel numerical risk, pinned
  in one test. Also: `Σ_n structural_length_pmf == 1`; responsibilities sum to 1 per
  column; `viterbi ≤ forward`; masked/batched agrees with per-jet loops.
- **New `tests/test_edit_model.py`:** contract — `log_prob` finite and shaped `(B,)`;
  sampled multiplicities track the exact `length_pmf` (statistical); every drawn
  coordinate maps back to its drawn cell under `geometry.to_cell`;
  `describe_cells(...).logprob == log_prob(...)` on the same chain.
- **Extend `tests/test_models.py`:** add `["model=edit_v1", "encoder=gru"]` to `MODELS`.
- **Extend `tests/test_empty_sequences.py`:** `nx == 0` (pure-insertion limit — must
  reduce exactly to the free head) and `ny == 0` (delete-all path has finite,
  non-degenerate log-probability).
- **Extend `tests/test_mbr.py`:** MBR over edit draws never selects the empty tree
  when non-empty draws dominate (the existing headline invariant).
- **Extend `tests/test_checkpoint.py`:** round-trip an `edit_v1` model.

## Verification (end-to-end)

1. `pytest tests/ -q`, with `tests/test_edit_dp.py` as the gate.
2. `python scripts/verify_parity.py` — must be unchanged; if it moves, something
   outside the two new files was edited.
3. `h2p-rsd-junipr train model=edit_v1 encoder=gru trainer.fast_dev_run=true`, then
   `h2p-rsd-junipr eval <run> decode.point_estimator=mbr`. Compare `dlund_*` and
   `mult_bias_*` against `scripts/baseline_v2_reference.txt` and a v3 run.
4. **The decisive check, before building `edit_v2`.** On the PYTHIA RNTuple path,
   dump residual widths from the responsibilities `γ(i,j)`, binned in
   `(k_t, R_g, N)`, and fit `σ = σ_0 + Λ_eff · exp(−ln k_t)`. If `Λ_eff` lands at
   `O(1 GeV)` with a stable fit, the family's inductive bias is validated and the
   learned kernel is publishable on its own terms. **If the widths are flat in
   `k_t`, the anchoring assumption is wrong** — stop, and do not build stage 2.
5. **The motivating A/B:** `dlund_posterior_mode` / `dlund_mbr` versus
   `dlund_identity`. The claim this family exists to make is that anchoring closes
   that gap; the closure suite already measures it, so the claim is falsifiable on
   day one.

## Risks

1. **Monotone alignments cannot represent reordering.** Two nearby nodes that swap
   order between levels cost a delete+insert pair — representable, statistically
   inefficient. Audit via crossing-pair counts in sampled alignments (diagnostic 4);
   if rare, ignore.
2. **Mixture identifiability.** The anchored and free components can trade off. Mitigate
   with the physics width form, a small `σ` initialization, and `p_anch` initialized
   high; monitor `frac_anchored` in closure.
3. **`nx == 0` jets (6.9%)** have no anchors; the model must reduce exactly to the
   free head there, which is a test, not a hope.
4. **Log-domain numerics** in the lattice — the enumeration test is the guard.
