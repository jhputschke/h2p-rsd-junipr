# PLAN — Post-review updates: exact-likelihood CFM family, calibration suite v2, sequence conditioning, v3 follow-through, systematics chain completion

Status: **WP1–WP4 implemented** (merge order WP2 → WP1 → WP4 → WP3, one commit each);
**WP5 not started**. Five work packages derived from a literature/code review of the
repo against the 2024–2026 unfolding and SBI literature. Each WP is independently
mergeable, opt-in with defaults off (the established `use_multiplicity_head` /
`point_estimator="map"` idiom), and leaves the likelihood/MAP parity paths
byte-identical when off. Builds on the merged `PLAN_NsplitMinCut.md`,
`PLAN_QuantileMinCut.md`, `PLAN_MBR_PerturbativeLund.md`, and the implemented
`PLAN_MultHead.md` (v3).

## Implementation status

| WP | Status | Landed as | Docs |
|---|---|---|---|
| **WP2** calibration suite v2 | ✅ merged | `eval/calibration.py`, `eval/report.py`, `distributions.py` (`vonmises_cdf`, `trunc_normal_cdf`), `experiment.*` switches, `tests/test_calibration_v2.py` | CONFIGURATION §8 |
| **WP1** exact-likelihood CFM | ✅ merged | `models/cfm.py`, `CFMConfig`, `configs/model/cfm.yaml`, `exact_likelihood` flag, `tests/test_cfm.py` | CONFIGURATION §4 |
| **WP4** v3 follow-through | ✅ merged | `data/stats.py` support guard, `presets/ab_v2_v3.yaml`, `scripts/ab_v2_v3.py`, `tests/test_support_guard.py` | CONFIGURATION §7 (v3 semantics + feed-N decision rule) |
| **WP3** sequence conditioning | ✅ merged | `Encoder.forward_seq`, `use_cross_attention`, `model=ar_junipr_v4`, `tests/test_xattn.py` | CONFIGURATION §4 |
| **WP5** systematics chain | ⛔ not started | — | — |

### Deviations from the plan as written, and why

1. **A second contract addition: `PosteriorModel.training_objective`.** The plan allows
   only `exact_likelihood`. But `cfm` has an exact `log_prob` that is *not* its training
   loss (flow matching regresses a vector field; the ODE runs only at evaluation), and
   the trainer called `-log_prob` directly. The alternatives were both worse: train by
   backpropagating through the ODE (slow, unstable, and defeats the point of flow
   matching), or make `log_prob` return the surrogate while training — which is exactly
   the dishonesty WP1 exists to remove. `training_objective` defaults to `-log_prob`, so
   every existing family's loop is bit-identical, and nothing outside `models/` branches
   on the family. Pinned by `test_cfm.py::test_other_families_keep_maximum_likelihood_training`.
2. **`coordinate_cdfs` rather than eval-side per-family PIT code.** The plan's WP2.1
   describes different PIT constructions per family. Implementing that in
   `eval/calibration.py` would have put family branching outside `models/`, against the
   cross-cutting rule. Instead each family returns its own transform (or `None`), and
   `eval` consumes one uniform dict.
3. **`ψ` in `cfm` is box-mapped, not periodic.** The plan says "the periodic ψ via the
   same fixed bijections the AR heads use (tanh-box / angle wrap)". Implemented as wrap
   into `(-π, π]` then tanh-box, which is exactly normalized on the physical support but
   does *not* close the seam at ±π the way the AR von Mises head does. Documented as a
   known limitation; closing it structurally needs Riemannian flow matching, which the
   WP's own non-goals exclude.
4. **TARP ties use the mid-rank convention**, matching the existing SBC statistic.
   Without it the discrete cell chains tie often enough to push every `f` down and fake
   over-dispersion. `tarp_refs` is the size of the reference *pool*; each test jet draws
   one reference from it, which keeps the cost linear in `n_jets`.
5. **The A/B's headline (b) statistic changed.** The plan expects `mbr_resample_to_qn`
   weights ≈ 1 under v3. Measured, the raw spread is **not** ≈ 0 — `w_k` compares an
   exact head against a K-draw histogram, so `w ≠ 1` at `O(1/√K)` even for a perfect
   sampler. `scripts/ab_v2_v3.py` therefore reports the **excess over a finite-K null**,
   which *is* ≈ 0. Quoting the raw spread would have made a genuine no-op look like a
   live correction.

### Results against the stated exit criteria

- **WP1**: coordinate density integrates to `1.004 ± 0.009` over the physical support
  (sign-flipped control: `0.856`, ~16σ away — the test caught a real divergence sign
  error during development); exact 4-VJP divergence matches the full autograd Jacobian
  trace to `1e-5`; forward/reverse round trip agrees to `1e-4` at 128 steps;
  `train model=cfm` runs end to end.
- **WP2**: `vonmises_cdf` matches quadrature of `vonmises_logpdf` to `1e-6` for
  `κ ∈ [0.01, 50]`; the PIT of self-generated data is uniform and a ×0.5-width head is
  flagged U-shaped; TARP reads the diagonal on a self-consistent posterior and drops
  below it for an over-confident one; the all-off metric dict is bit-for-bit the old one.
- **WP3**: OFF-path `state_dict` keys, NLL, samples and MAP identical to v2/v3; padding
  nodes provably receive zero attention weight; teacher-forced and incremental decode
  paths agree step by step. At matched parameter count (+1.1%), `encoder=gru`: on
  **synthetic** data (15 epochs) **val NLL/jet 17.85 (v4) vs 21.68 (v3)** — the stated
  criterion, comfortably met; but on **real PYTHIA** data
  (`cpp/test_data/jets.root`, 12 epochs) **4.64 (v4) vs 4.61 (v3)** — a wash. The
  mechanism explains the split: that sample's mean hadron multiplicity is 1.74 (6.9% of
  jets have no hadron emission at all), so there is no fixed-length bottleneck to remove
  and the capacity `dec_dim` gave up is simply lost. Recorded because it is the more
  useful result: **the exit criterion was met on the generator the plan named, and does
  not transfer to the data.** Adoption stays gated on the WP4 A/B, run on the target
  sample.
- **WP4**: guard unit-tested at both thresholds and gated on the family having a head;
  `scripts/ab_v2_v3.py --fast` runs end to end in CI's fast tier. The fast tier already
  reproduces (a): v2 at `min_emissions=0` collapses to the empty tree for 100% of jets,
  v3 for 0%.

Worked walkthrough of the new calibration and metrics on real PYTHIA data:
[`notebooks/calibration_v2_walkthrough.ipynb`](../notebooks/calibration_v2_walkthrough.ipynb).

> **Line anchors.** File:line references below were taken from the tree at commit
> `20a8686` (2026-07-27). Re-verify before editing; merges shift them.

Review conclusions this plan implements, in one line each:

1. **WP1** — `models/diffusion.py` returns a DSM surrogate from `log_prob`
   (diffusion.py:105–112), breaking the one-contract invariant that `log_prob` is a
   normalized density; add a conditional flow-matching family with an *exact*
   probability-flow-ODE likelihood (Lipman et al., arXiv:2210.02747; Song et al.,
   arXiv:2011.13456; FMPE, Wildberger, Dax et al., NeurIPS 2023, arXiv:2305.17161),
   matching the family the field converged on for substructure-scale generative
   unfolding (Huetsch et al., SciPost Phys. **18** (2025) 070, arXiv:2404.18807;
   Petitjean et al., arXiv:2510.19906).
2. **WP2** — `eval/calibration.py` runs SBC/PIT **on the multiplicity only**
   (calibration.py:32–36) — exactly the marginal v3 optimizes directly, so the
   current suite certifies v3 near-tautologically; add per-coordinate PITs,
   region-stratified coverage, and TARP (Lemos et al., arXiv:2302.03026) using the
   perturbative-Lund EMD already in `inference/mbr.py` as the distance.
3. **WP3** — the `Encoder` ABC contracts to a pooled `(B, ctx_dim)` vector
   (encoders/base.py:27–29) tiled at every decoder step (ar_junipr.py:`_decode_states`,
   `e_seq = e.unsqueeze(1).expand(...)`); add an opt-in sequence-states contract and
   cross-attention in the AR decoder (Transfermer/JetGPT evidence: arXiv:2404.18807
   §2.3; Butter et al., arXiv:2305.10475).
4. **WP4** — v3 changed what the decode knobs and MBR are *for*; re-measure
   `min_emissions` / `length_floor_quantile` / `mbr_resample_to_qn` under v3, guard
   the categorical `N ≤ max_emissions` support tail, and pin the decision rule for
   the deferred feed-N-into-decoder extension of `PLAN_MultHead.md`.
5. **WP5** — the quoted dominant systematic (PYTHIA-vs-HERWIG,
   `eval/systematics.py:18`, `configs/experiment/pythia_vs_herwig.yaml`) has no
   generator-B producer: `herwig_driver` exists only as a comment
   (cpp/apps/pythia_driver.cpp:12); build it, and wire exact hadronization-parameter
   reweighting (PYTHIA automated fragmentation variations, arXiv:2308.13459; exact
   post-hoc string reweighting, Assi et al., SciPost Phys. **19** (2025) 104,
   arXiv:2505.00142) into the RNTuple schema and the systematics evaluator.

Recommended merge order: **WP2 → WP1 → WP4 → WP3 → WP5** (WP2 first because every
other WP's exit criterion consumes it; WP5 is independent and can proceed in
parallel on the C++ side). WP1–WP4 were merged in exactly that order; see
"Implementation status" above.

---

## WP1 — `models/cfm.py`: conditional flow matching with exact ODE likelihood

### Context

The three families are registered in `MODEL_SCHEMAS` (config.py:199–204) and share
the `PosteriorModel` contract (`log_prob` documented as "(B,) log q_phi(y|x)",
models/base.py:32–35). `CINN` honors it exactly (conditional RealNVP,
cinn.py:36–93, exact log-density at cinn.py:69). `Diffusion` does not: its
`per_jet_nll` sums the exact discrete terms with a
"denoising-score-matching surrogate used as a (negative) log-density proxy"
(diffusion.py:105–109) and `log_prob` returns its negation (diffusion.py:111–112);
`map_estimate` uses a posterior-mean `_x0` surrogate (diffusion.py:175). NLL-based
model selection and the likelihood-ratio deliverable are therefore valid only
across `ar_junipr_*` and `cinn`. The module header itself scopes a
"score/bridge model and probability-flow-ODE likelihood" as the real target
(diffusion.py:4–9).

### Design

**New family, not a rewrite of `diffusion`.** Add `models/cfm.py` with
`@register_model("cfm")`, `CFMConfig` in config.py, `configs/model/cfm.yaml`, and
an entry in `models/__init__.py`. `diffusion` stays as-is (it is the registry's
cheap-sampler baseline) but gains an honesty flag (below). Structure mirrors the
proven cINN factorization exactly:

    q(y|x) = q(N|x) · Π_t q(cell_t | x, prefix) · Π_t p_cfm(coords_t | ctx_t)

- `n_head`: copied from cinn.py:106–109 (categorical over `0..max_emissions`,
  NLL-trained, `length_pmf` override per cinn.py:178–182).
- Cell terms: reuse the CINN's categorical cell treatment unchanged.
- Coordinate density `p_cfm`: a conditional vector field
  `v_θ(x_t, t, ctx) : R^4 × [0,1] × R^ctx → R^4` (MLP, `hidden_dim`, Fourier time
  features), trained with the standard CFM regression to the OT-path target
  `u_t(x|x_1) = x_1 − x_0` (Lipman et al., arXiv:2210.02747), conditioning vector
  built the same way the AR `coord_head` input is (ar_junipr.py:93–95: decoder
  state ⊕ context ⊕ cell embedding — here: context ⊕ cell embedding, no AR state).

**Exact likelihood, exact divergence.** `log_prob` integrates the probability-flow
ODE with the instantaneous change of variables
`d log p / dt = −∇·v_θ`. The coordinate dimension is **4**, so compute the
divergence *exactly* with 4 vector-Jacobian products per ODE step — no Hutchinson
estimator, no stochastic likelihood. Fixed-step RK4 (or Heun) with
`n_ode_steps` config; wrap in `torch.enable_grad()` inside the otherwise
`inference_mode` evaluation path. Support handling: coordinates are trained in an
unbounded standardized space; map to the RSD-allowed box `(±half_u, ±half_v)` and
the periodic ψ via the same fixed bijections the AR heads use
(tanh-box / angle wrap), adding their closed-form log-Jacobians — this keeps the
density normalized on the physical support (the caveat the discretized grid could
never satisfy).

```python
# config.py (append after DiffusionConfig, config.py:113–119)
@dataclass
class CFMConfig:
    name: str = "cfm"
    ctx_dim: int = 64
    hidden_dim: int = 64
    n_ode_steps: int = 32          # likelihood + sampling ODE steps (RK4)
    max_emissions: int = 25        # multiplicity-head support, mirrors CINN
    time_features: int = 16        # Fourier features for t
    sigma_min: float = 1e-3        # OT-path terminal width (Lipman Eq. 20)
```

**Contract honesty flag.** Add to `PosteriorModel` (models/base.py:31):

```python
class PosteriorModel(nn.Module, ABC):
    exact_likelihood: bool = True   # False => log_prob is a training surrogate
```

Set `exact_likelihood = False` on `Diffusion` only. `cmd_eval` (cli.py:75) and
`serving/api.py` print a single warning when reporting NLL/log-ratios from a
non-exact family. No behavior change anywhere else; the attribute default keeps
every existing family untouched.

**Sampling / MAP / MBR.** `sample`: draw `N ~ q(N|x)`, cells autoregressively as
in CINN, coordinates by forward ODE integration from the base Gaussian.
`map_estimate`: `N* = max(argmax q(N|x), min_emissions)` + per-cell coordinate
point via a short gradient ascent on the exact `log p_cfm` (or the ODE-pushforward
of the base mode as the cheap default — config `cfm_map: str = "ode_mode"|"ascent"`).
MBR needs nothing: `mbr_select` (mbr.py:345) consumes `sample`/`describe_cells`
family-agnostically (models/base.py:46–54).

### Tests & exit criteria

- `tests/test_cfm.py`: (i) shape/registry smoke via the existing
  `tests/test_models.py` parametrization pattern; (ii) **normalization test** — on
  a frozen context, Monte-Carlo integrate `exp(log_prob)` of the 4-d coordinate
  density over the box against 1 within tolerance (the test the grid head could
  not pass); (iii) **round-trip test** — `log_prob` of a forward-ODE sample is
  finite and matches the change-of-variables accumulation to `1e-4` at
  `n_ode_steps=128`; (iv) divergence check: 4-vjp exact trace equals autograd
  `jacobian` trace on random inputs.
- Integration: `h2p-rsd-junipr train model=cfm` passes the §14-phase-5 gate
  (integration train + closure) on synthetic data; SBC/TARP from WP2 pass.
- Exit: NLL comparable in scale to `cinn` on the synthetic generator; a one-line
  docs/CONFIGURATION.md §4 entry; `Diffusion` docstring updated to state its
  surrogate status explicitly.

### Non-goals

No Schrödinger-bridge variant, no minibatch-OT couplings (arXiv:2302.00482), no
replacement of the DDPM sampler — all deferred until the plain OT-path CFM has
closure numbers.

---

## WP2 — Calibration suite v2: coordinate PITs, region stratification, TARP

### Context

`run_calibration` (eval/calibration.py:19) computes: SBC ranks **of the
multiplicity** (calibration.py:32–34, per Talts et al., arXiv:1804.06788), a
multiplicity PIT (calibration.py:36), and leading-cell 68% HPD coverage
(calibration.py:39–48). For `ar_junipr_v3` the `n_head` is trained by direct NLL
on `N` (ar_junipr.py:165–166), so SBC-on-N is calibrated nearly by construction —
the planned v2-vs-v3 A/B under the current suite is biased toward v3. The
closed-form CDFs needed for coordinate PITs already exist or are one function
away: `std_normal_cdf` (distributions.py:22) gives the truncated-normal CDF from
the same quantities `trunc_normal_logpdf` uses (distributions.py:26–34); the von
Mises CDF needs a small Bessel-series helper next to `vonmises_logpdf`
(distributions.py:67). The tree-level distance for TARP already exists:
`lund_cloud` (mbr.py:71), `lund_emd` (mbr.py:237), `lund_emd_matrix` (mbr.py:286).

### Design

Extend `eval/calibration.py` with three additive functions; `run_calibration`
grows optional kwargs and new metric keys, and returns the current dict unchanged
when the new switches are off (default off ⇒ CI numbers stable).

1. **`coordinate_pits(model, val_ds, geometry, device, n_jets, K)`** — for AR
   v2/v3: teacher-force the true `y`, evaluate the conditional CDF of each
   continuous head at the truth per emission index
   (trunc-normal for `du, dv`; normal for `ln z`; von Mises via the new
   `vonmises_cdf`), aggregate rank histograms per coordinate and per emission
   index; report per-coordinate KS distance to uniform. For `cinn`/`cfm`: PIT via
   the base-space coordinates (RealNVP inverse, cinn.py:69; CFM reverse ODE).
   `diffusion` is excluded (no exact density; see WP1 flag).
2. **Region stratification** — every existing metric (SBC-N, coverage, and the
   new PITs) additionally binned by the *leading-emission Lund cell quadrant*
   (reuse `geometry.leading_emission_cell` as in the closure suite), emitting a
   `metrics["by_region"]` sub-dict. This is the review's "region-stratified
   coverage" deliverable and the direct precondition for the heavy-ion
   localization claim.
3. **`run_tarp(model, val_ds, geometry, device, n_jets, K, n_refs)`** — TARP
   expected-coverage (Lemos et al., arXiv:2302.03026) on *tree-valued* posteriors:
   for each jet, distance = `lund_emd` between clouds (`lund_cloud` with the
   decode-configured `mbr_*` kwargs via `mbr_kwargs_from_decode`, mbr.py:50);
   reference points drawn from the pooled posterior draws of *other* jets in the
   evaluation batch (a support-covering reference distribution; document this
   choice and expose `tarp_reference: str = "pooled"|"prior"` for a
   prior-simulated alternative). Output: the ECP-vs-credibility curve and its
   maximal deviation `tarp_max_dev`.

```python
# config.py — ExperimentConfig (config.py:176–181) grows:
    pit_coords: bool = False        # WP2.1 per-coordinate PITs
    stratify_regions: bool = False  # WP2.2 region-binned metrics
    tarp: bool = False              # WP2.3 TARP curve
    tarp_refs: int = 100
    tarp_reference: str = "pooled"  # pooled | prior
```

`cmd_eval` (cli.py:115–119) threads the new flags; figures land in the run dir
next to the existing closure artifacts.

### Tests & exit criteria

- `tests/test_calibration_v2.py`: `vonmises_cdf` vs numerical integration of
  `vonmises_logpdf` to `1e-6`; PIT of samples drawn *from the model itself* is
  uniform (self-consistency, the SBC null); TARP on a deliberately
  overconfident model (σ scaled ×0.5) shows the documented undercoverage
  signature; all-off path returns today's metric dict bit-for-bit.
- Exit: the v2-vs-v3 A/B (WP4) is gated on `pit_coords` + `tarp`, not SBC-N; the
  mandatory-validation section of docs/CONFIGURATION.md §8 documents why.

### Non-goals

No DRP/expected-coverage variants beyond TARP; no attempt to calibrate
`diffusion` (blocked on WP1's exact-likelihood alternative).

---

## WP3 — Sequence conditioning: `SequenceEncoder` contract + cross-attention decoder

### Context

`Encoder.forward` contracts to `(B, Mx, n_node_feat), (B,) -> (B, out_dim)`
(encoders/base.py:27–29); `GRUEncoder` mean-pools its per-node states away
(encoders/gru.py:40–42) and the AR decoder tiles the single pooled `e` at every
step (`_decode_states`, ar_junipr.py: `e_seq = e.unsqueeze(1).expand(-1, L+1, -1)`).
Every hadron-level node is thus visible to the parton-level decoder only through
one ctx_dim vector — the classic fixed-length bottleneck. LundNet's graph
structure is likewise pooled before the decoder sees it.

### Design

**Encoder side — additive capability, not a new ABC.** Add to `Encoder`
(encoders/base.py:23):

```python
class Encoder(nn.Module, ABC):
    out_dim: int
    returns_sequence: bool = False           # capability flag

    def forward_seq(self, xf, nx):
        """(B, Mx, d_seq), (B, Mx) mask — per-node states before pooling.
        Only valid when returns_sequence=True."""
        raise NotImplementedError
```

Implement `forward_seq` for `gru` (the pre-pool `out` and `mask` it already
computes, encoders/gru.py:40–41), `lundnet` (per-node EdgeConv states before
readout), and `deepsets` (per-node embeddings before the sum). `forward`
(pooled) is untouched — parity path identical.

**Model side — opt-in cross-attention.** New field on `ARJuniprConfig`
(config.py:81–99, same idiom as `use_multiplicity_head`, config.py:93):

```python
    use_cross_attention: bool = False   # decoder attends to hadron-node states
    xattn_heads: int = 4
```

When ON, `ARJunipr.__init__` builds
`self.xattn = nn.MultiheadAttention(self.dec_dim, xattn_heads, batch_first=True)`
plus a `d_seq -> dec_dim` key/value projection; `build_model` raises a config
error if the selected encoder has `returns_sequence=False`. In `_decode_states`,
after the GRU:

```python
out, _ = self.decoder(inp, self._init_hidden(e))
if self.use_cross_attention:
    seq, mask = self.encoder_net.forward_seq(xf, nx)
    attn, _ = self.xattn(out, self.kv_proj(seq), self.kv_proj(seq),
                         key_padding_mask=~mask.bool())
    out = out + attn                      # residual: head input dims unchanged
return out
```

The residual form keeps every head's input width (`dec_dim + ctx_dim [+ emb]`,
ar_junipr.py:89–95) unchanged, so the OFF path's module list and `state_dict` are
byte-identical to today and strict `load_state_dict` (train/checkpoint.py:90 per
`PLAN_MultHead.md`) keeps loading all existing checkpoints. `_decode_states` is
teacher-forced and causal in `y` while attention is over `x` only, so the AR
factorization and the sampling/beam paths (`sample_batch`, `map_decode`) inherit
the change through the shared state computation with no decode-logic edits —
verify the incremental single-step decode path reuses `_decode_states`-equivalent
state updates; if it maintains its own GRU stepping, mirror the residual there.

### Tests & exit criteria

- `tests/test_xattn.py`: OFF-path `state_dict` keys and NLL bit-identical to
  v2/v3 today; ON-path shapes; attention mask correctness (padding nodes receive
  zero weight); config error for `encoder=deepsets` only if `forward_seq` is
  left unimplemented there.
- Exit: on synthetic data, `use_cross_attention=true` at matched parameter count
  (shrink `dec_dim` to compensate) reaches ≤ the baseline NLL, and the WP2
  region-stratified PITs do not degrade. Adoption for physics runs is decided by
  the WP4 A/B, not by this WP.

### Non-goals

No transformer decoder replacement of the GRU, no VQ tokenization (information
loss; see review), no encoder pre-training. Those are follow-ups only if
cross-attention exhausts its headroom.

---

## WP4 — v3 follow-through: A/B protocol, decode-knob re-measurement, N-tail guard

### Context

v3 (`use_multiplicity_head=True`, ar_junipr.py:62, 83–87, 165–166, 353–360) kills
the two length pathologies at source. Consequently: `min_emissions` /
`length_floor_quantile` (DecodeConfig, config.py:150–157; inference/length.py)
were patches for the v2 joint-argmax collapse; `mbr_resample_to_qn` +
`_qn_importance_weights` (config.py:171, mbr.py:316) were the decode-layer fix
for the biased MBR candidate pool. Under v3 all three are expected to be
near-no-ops — that expectation must be *measured*, not assumed. Separately, the
categorical support is `N ≤ max_emissions = 25` (config.py:97): a truth sequence
longer than the support receives zero likelihood mass in a way the v2 stop-head
never did.

### Design

1. **Support-tail guard.** In the datamodule stats pass, compute
   `P_data(N > model.max_emissions)`; **hard error** above `1e-3`, warning above
   `1e-4`, with the offending `z_cut/β/ln kt`-floor context in the message. Add
   the same assert to `tests/test_multiplicity_head.py` against the synthetic
   generator. (Cheap, do first.)
2. **A/B preset.** `presets/ab_v2_v3.yaml` + `scripts/ab_v2_v3.py`: the grid
   {`ar_junipr_v2`, `ar_junipr_v3`} × {`point_estimator=map`, `mbr`} ×
   {`min_emissions=0,1`} × {`length_floor_quantile=0.0,0.5`} ×
   {`mbr_resample_to_qn=false,true`}, each evaluated with the full WP2 suite plus
   the closure metrics. Deliverable: one table answering (a) is the empty-tree
   collapse gone at `min_emissions=0` under v3 (expected yes — the argmax is over
   `q(N|x)`); (b) is `mbr_resample_to_qn` a no-op under v3 (expected weights ≈ 1);
   (c) residual ±multiplicity bias of ancestral draws (the "+0.86" successor
   number) — attribute what remains to coordinate-level exposure bias, which v3
   does not touch.
3. **Decision rule for feed-N-into-decoder** (the documented deferred extension
   in `PLAN_MultHead.md`): trigger **only if** WP2's coordinate PITs or TARP show
   miscalibration of `q(y|N,x)` that is systematically `N`- or region-dependent
   at the level of the quoted generator systematic (eval/systematics.py:18
   spread). Record the rule here so the extension is not re-litigated.
4. **Docs**: docs/CONFIGURATION.md §7/§10 gain a "v3 semantics" paragraph per
   knob: which knobs are live under v3 (`cont_temperature` on cell logits,
   `max_emissions` as support), which are legacy-v2 (`min_emissions` as a floor on
   an already-well-posed argmax, `length_floor_quantile`), pointing at the A/B
   table.

### Tests & exit criteria

- Guard unit test (1); A/B script runs end-to-end on `data=synthetic` in CI's
  fast tier with reduced `closure_jets`.
- Exit: the A/B table exists as a run-dir artifact + a docs table; every knob's
  v3 status is documented; the feed-N decision rule is written down.

---

## WP5 — Systematics chain completion: `herwig_driver`, exact fragmentation reweighting, aggregate cross-check

### Context

`generator_spread` (eval/systematics.py:18) and
`configs/experiment/pythia_vs_herwig.yaml` (generator_b: HERWIG-7.3) presuppose a
generator-B RNTuple that nothing produces: cpp/apps contains only
`pythia_driver.cpp` (whose comment at line 12 promises "A parallel herwig_driver
(cluster model; Bellm et al., EPJC 76 (2016) 196, arXiv:1512.01178)") and
`read_lund_rntuple.cpp`. The prior-dependence program additionally calls for
cheap fragmentation-parameter variations; PYTHIA ships automated per-event
fragmentation-variation weights (Bierlich et al., arXiv:2308.13459;
`VariationFrag`), with exact post-hoc flavor/kinematics string reweighting
established by Assi et al. (SciPost Phys. **19** (2025) 104, arXiv:2505.00142).

### Design

1. **`cpp/apps/herwig_driver.cpp`.** Mirror `pythia_driver.cpp`'s structure:
   read an input card, run HERWIG 7.3 (angular-ordered shower + cluster
   hadronization), hand final-state hadrons *and* pre-hadronization partons to
   the shared `lund_writer` (cpp/src/lund_writer.cpp) so the RNTuple schema is
   byte-compatible (the `tag` field, lund_writer.cpp:25, carries the generator
   label). Practical route: drive HERWIG via its `Herwig read/run` API or, to
   avoid linking ThePEG, accept HepMC3 input (`herwig … > events.hepmc`) and add
   a thin `hepmc_driver.cpp` that clusters from HepMC3 — decide at
   implementation time; the HepMC3 route also future-proofs JEWEL/hybrid-model
   input for the heavy-ion extension. CMake target + a `docker/Dockerfile.cpp`
   layer; a C++ smoke test asserting schema identity against a PYTHIA file.
2. **Per-event fragmentation-variation weights.** Extend `pythia_driver.cpp` to
   enable `VariationFrag`-style automated variations for the configured
   parameter list (aLund/bLund/sigma, StringZ/StringPT) and persist the weight
   vector per jet in the RNTuple (schema: `frag_weights: vector<double>` +
   run-level names). `eval/systematics.py` gains
   `frag_spread(model, val_ds, ...)`: re-evaluate every WP2 metric and the
   posterior summaries under each weight column; the envelope is the
   fragmentation-prior systematic — the concrete realization of the review's
   "cheap PYTHIA reweighting first" step, now exact rather than approximate.
3. **Aggregate cross-check (stretch, non-blocking).** `scripts/aggregate_check.py`:
   an OmniFold-style two-classifier reweighting (Andreassen et al., PRL **124**
   (2020) 182001, arXiv:1911.09107; non-iterative variant AUSSIE, Ore & Plehn,
   arXiv:2602.24282) of the parton-level Lund density from the matched pairs,
   compared against the pushforward of the amortized posterior — the
   distribution-level sanity check experimental referees will expect. Torch-only,
   no new dependency; lives in `scripts/`, not the package.

### Tests & exit criteria

- Schema-identity C++ test (1); Python-side: `frag_weights` round-trips through
  `data/rntuple.py`; `frag_spread` on a two-weight toy reproduces hand-computed
  envelopes.
- Exit: `experiment=pythia_vs_herwig` runs end-to-end from generation to the
  `generator_spread` figure; the fragmentation envelope is a standard entry in
  the systematics record; the aggregate check exists as a runnable script.

### Non-goals

No HOMER-style extraction of the fragmentation function (complementary program,
different estimand); no JEWEL/medium driver yet (blocked on the heavy-ion design
note); no in-repo OmniFold productionization.

---

## Cross-cutting rules (all WPs)

- **Parity.** Every switch defaults off; OFF paths must be byte-identical in
  module lists, `state_dict`, and NLL (extend `tests/test_parity.py` /
  `scripts/verify_parity.py` accordingly). New config fields ride the tolerant
  `decode_params` / `OmegaConf.select` backfill (config.py:351) so old
  checkpoints keep loading; `config_hash` changes for new runs only.
- **One contract.** Nothing outside `models/` may branch on the family; the only
  contract addition is the WP1 `exact_likelihood` class attribute.
- **Docs + citations.** Each merged WP updates docs/CONFIGURATION.md and appends
  its references to CITATION.cff / README §References.

## References (new to this plan)

- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, *Flow Matching for
  Generative Modeling*, ICLR 2023, arXiv:2210.02747.
- Y. Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021,
  arXiv:2011.13456 (probability-flow ODE).
- J. Wildberger, M. Dax et al., *Flow Matching for Scalable Simulation-Based
  Inference*, NeurIPS 2023, arXiv:2305.17161.
- N. Huetsch et al., *The Landscape of Unfolding with Machine Learning*,
  *SciPost Phys.* **18** (2025) 070, arXiv:2404.18807.
- A. Petitjean et al., *Generative Unfolding of Jets and Their Substructure*,
  arXiv:2510.19906.
- P. Lemos, A. Coogan, Y. Hezaveh, L. Perreault-Levasseur, *Sampling-Based
  Accuracy Testing of Posterior Estimators (TARP)*, ICML 2023, arXiv:2302.03026.
- S. Talts et al., *Simulation-Based Calibration*, arXiv:1804.06788.
- A. Butter et al., *Jet Diffusion versus JetGPT*, arXiv:2305.10475.
- C. Bierlich et al., *Reweighting Monte Carlo Predictions and Automated
  Fragmentation Variations in Pythia 8*, arXiv:2308.13459.
- B. Assi et al., *Post-hoc Reweighting of Hadron Production in the Lund String
  Model*, *SciPost Phys.* **19** (2025) 104, arXiv:2505.00142.
- A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman, J. Thaler,
  *OmniFold*, *Phys. Rev. Lett.* **124** (2020) 182001, arXiv:1911.09107.
- A. Ore, T. Plehn, *Unfolding without Iterations, Adversaries, or Surrogates
  (AUSSIE)*, arXiv:2602.24282.
- J. Bellm et al., *Herwig 7.0/7.1*, *Eur. Phys. J. C* **76** (2016) 196,
  arXiv:1512.01178.
