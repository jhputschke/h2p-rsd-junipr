# PLAN — Production test v1: one targeted intervention per localized failure, with pre-registered gates

**Status: implemented and run** — 11 arms, all evaluated; results in
[`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md). Gates on `v1_base`, unanimous over
three seeds: **G1 PASS, G2 PASS, G3 FAIL, G4 FAIL, G5 ATTRIBUTED, G6 PASS, G7 FAIL**; G8
decided **against** the fielded explicit-`q(N|x)` family. Three §12 triggers were tested:
the joint-coordinate-density and §4.4 spline triggers **fired** (G3 fails on the truncated
head); the WP3 secondary-plane trigger **did not** (the aux isolation is null on every
deciding metric). Nothing was changed mid-run in response to any of them.

Companion to [`PLAN_prod_test_v0.md`](PLAN_prod_test_v0.md) and
[`PROD_TEST_v0_RESULTS.md`](PROD_TEST_v0_RESULTS.md); as there, **this plan holds the
design and the rationale, the results document will hold only numbers.** Every pass
criterion below is fixed before the grid runs.

> **Two corrections found during implementation.** Neither changes a gate; both change a
> name or an arm, and are recorded here rather than silently absorbed.
>
> 1. **`decode.cont_temperature` was already taken.** §5.2 proposes it as "a single
>    scalar on the continue logit"; the field of that name has existed since WP2 as the
>    softmax temperature on the **cell** logits. Implementing the plan's meaning under
>    the plan's name would have silently redefined a live knob, so the continue-logit
>    temperature ships as **`decode.continue_temperature`** and `cont_temperature` keeps
>    its meaning. Everything else about §5.2 is unchanged.
> 2. **`v1_base` and `v1_nhead` were the same model.** §8 lists `v1_base` as
>    `ar_junipr_v4` and `v1_nhead` as the "explicit `q(N|x)` factorization" — but
>    `configs/model/ar_junipr_v4.yaml` already sets `use_multiplicity_head: true`, so the
>    two arms differ in nothing. Gate G8's stated rationale (SBC-N must not decide,
>    *because* v3's `n_head` is calibrated on it by construction) only has content when
>    one arm has the head and the other does not, so the arm that was missing is the
>    **implicit continue/stop** one. It is fielded as **`v1_contstop`** and G8 is
>    evaluated as `v1_base` (explicit) vs `v1_contstop` (implicit), on the metrics G8
>    names. This is also the only arm on which `continue_temperature` is not a no-op.

**Framing.** v0 established that the *density* beats the baseline (posterior-series W1
gmean 0.414 vs plain RSD) and localized every failure: one coordinate head with the wrong
support, one structural miscalibration of the multiplicity marginal, one estimator
pathology in the decode layer, and one unattributed region. v1 is the run in which each
localized failure receives exactly one targeted intervention and a pre-registered number
that decides it.

---

## 1. Context — what v0 left open

From `PROD_TEST_v0_RESULTS.md` §9, restated as the questions v1 must answer:

| v0 issue | v1 intervention | decided by |
|---|---|---|
| `ln z` PIT KS 0.066 vs crit 0.016 (§9.1) | bounded-support `ln z` head (WP-A) | gate G3 |
| 0.88% soft-drop violations (§9.4) | same head — zero by construction | gate G2 |
| SBC-on-N χ² 107, coverage 0.538, ⟨N⟩ 1.15 vs 1.40 (§9.2) | gate temperature + explicit-N arm (WP-B) | gate G4 |
| MAP/MBR worse than plain RSD; ψ row 17.5× (§9.5, §6) | estimator repairs in `inference/` (WP-C) | gate G6 |
| `narrow_soft` 0.357 coverage (§9.3) | region × coordinate PIT attribution (WP-D) | gate G5 |
| two-seed band, loose TARP null (§10, §4) | third seed; TARP power (arms; WP-D) | gates G7–G8 |

The aux-isolation and encoder A/B probes (v0 §10–§11) ride the same grid.

## 2. Key facts (verified against v0 and the merged plans)

- **`ln z` is the only unbounded coordinate head.** The coordinate likelihood is
  `TN(du)·TN(dv)·N(ln z)·vM(ψ)`; `du`/`dv` use `trunc_normal_logpdf` and are the two
  *best*-calibrated coordinates (PIT KS 0.011–0.013), while the `ln z` Normal fails
  (0.066 ± 0.008). Under the fielded grooming (`z_cut = 0.1`, `beta = 0`) every retained
  splitting satisfies $z \in (z_{\rm cut}, \tfrac12]$ — Soft Drop lower bound (Larkoski
  et al., arXiv:1402.2657; RSD: Dreyer et al., arXiv:1804.03657) and
  $z = \min(p_{T,1},p_{T,2})/(p_{T,1}{+}p_{T,2}) \le \tfrac12$ by construction — so
  $\ln z \in (\ln z_{\rm cut},\, \ln\tfrac12]$, width $\ln 5 \approx 1.61$. The 0.88%
  violation rate is the lower-tail leak realized at sampling; the PIT last-bin deficit /
  0.7–0.9 excess is the upper-support signature.
- **The teacher-forced length distribution is calibrated; the sampled one is not.**
  `fit_length_recalibration` returns `(T, tilt) = (1.010, +0.011)`; `q(0|x)` emp/pred
  0.983 (v0 §7). Yet ancestral sampling gives ⟨N⟩ = 1.15 against truth 1.40, SBC-on-N
  χ² = 107 vs χ²(9) 95% = 16.90 (Talts et al., arXiv:1804.06788), coverage 0.538. The
  mismatch lives between the teacher-forced and on-policy prefix distributions —
  exposure bias (Ranzato et al., arXiv:1511.06732; Bengio et al., arXiv:1506.03099).
- **v3's `n_head` is trained by direct NLL on `N`** (ar_junipr.py:165–166, per
  `PLAN_UPDATES.md` WP2), so SBC-on-N is calibrated for it *nearly by construction*.
  WP2's exit criterion already rules that the family A/B must be gated on coordinate
  PITs + TARP + coverage, **not** SBC-N. v1 inherits that rule (gate G8).
- **The ψ head is right and the estimator is wrong** (v0 §6): median von Mises
  κ = 0.022 (peak/trough 1.04), sampled posterior |R| = 0.031 vs truth 0.045, but
  MAP/MBR attach the conditional *mode* — at κ ≈ 0 the direction of a near-zero
  resultant (Mardia & Jupp, *Directional Statistics*, Wiley 2000) — giving |R| = 0.69
  and a 17.5× pooled ψ row that inflates both decode gmeans. This is the per-jet
  instance of mode-unrepresentativeness in high-entropy sequence posteriors
  (Stahlberg & Byrne, arXiv:1908.10090; Eikema & Aziz, arXiv:2005.10283).
- **`mbr_select` (mbr.py:345) consumes `sample`/`describe_cells` family-agnostically**,
  and the medoid it returns *is a genuine posterior sample* — its own sampled
  coordinates are available and currently discarded in favor of re-attached modes.
- **TARP is load-bearing and its null is loose.** Max dev 0.037 against a 95% null
  floor of 0.079 at n = 300 (v0 §4). TARP anchors the entire "shape fine, multiplicity
  not" reading and must be given power commensurate with that role (Lemos et al.,
  arXiv:2205.03910).
- **The WP2 hooks already exist**: `pit_coords`, `stratify_regions`, `tarp`,
  `tarp_refs`, `tarp_reference` on `ExperimentConfig` (config.py:176–181);
  `std_normal_cdf` (distributions.py:22) yields the truncated-normal CDF for free.
- **Effective rank 37.2 of 64** on the `Linear(64, 900)` split head (v0 §8): the
  softmax-bottleneck rank bound (Yang et al., arXiv:1711.03953) is *not* binding —
  capacity, if probed, is a decoder-trunk question, and only a probe.
- **Parity rules** (cross-cutting, per `PLAN_UPDATES.md`): every new switch defaults
  off with a byte-identical OFF path (`state_dict`, `log_prob`); config reads via the
  tolerant `decode_params` / `OmegaConf.select` backfill; `config_hash` moves for new
  runs only; old checkpoints keep loading.
- **`tau` bookkeeping** (`tau.fitted_under`, notebook assertion) merged in v0 §7 —
  the pattern every new fitted inference-layer scalar must follow.

## 3. Design decisions

- **One coordinate-head change, justified structurally, not by A/B.** Support
  correctness needs no experiment to be adopted; the single legacy arm exists for
  *attribution* (reproduce the v0 failure under identical data), not adoption.
- **NLL comparability breaks at the head change.** A different coordinate normalization
  shifts NLL/jet; v1 numbers are comparable only within v1. The legacy arm is the
  bridge to v0. State this in the results doc preamble.
- **The tempered sampler is an inference-layer object.** `decode.cont_temperature`
  (Guo et al., arXiv:1706.04599, applied to the continue head at *sampling only*) never
  touches `per_jet_nll`, likelihoods, or ratios. It follows the `tau` pattern:
  fitted on training-val, stored in the artifact with `fitted_under`, applied frozen
  to test, with the closure notebook asserting the pair.
- **Pre-registered quoted configuration.** If the untempered sampler passes G4, it is
  the deliverable and the temperature is reported as a null check. If it fails and the
  tempered sampler passes *without degrading TARP or coordinate PITs*, the tempered
  sampler is the deliverable; both are always reported. No post-hoc choice.
- **Mode attachment is gated on identifiability.** New `decode.kappa_min_mode`
  (default proposed 0.5, i.e. peak/trough $e^{2\kappa} \approx e$) — below it the
  point estimate carries a *sampled* ψ and flags `psi_identified = false` per node.
  MBR carries the medoid's own sampled coordinates verbatim for **all** coordinates
  (the medoid is a sample; re-attaching modes forfeits exactly that property). MAP is
  demoted to a diagnostic: computed, never in headline tables.
- **All decode-layer changes live in `inference/`**, outside parity-critical files,
  and are retroactively runnable on the v0 checkpoint (they require no retraining) —
  do so, for continuity of the record.
- **Whole-file evaluation only** for NLL deltas, with the v0 §5 caution printed
  whenever `|delta|/band < 3`. The 40k-subsample verdict flip is the standing reason.
- **No likelihood reweighting for `narrow_soft`** or anywhere else: observed
  pathologies are attributed first (WP-D), treated structurally second.

## 4. WP-A — bounded-support `ln z` head

1. `config.py` (`ARJuniprConfig`, mirrored in `configs/model/ar_junipr_v*.yaml`):
   `lnz_support: str = "legacy"` (`"legacy" | "physical"`). `"legacy"` is bit-identical
   to today (parity). The v1 preset sets `"physical"`.
2. Model: when `"physical"`, replace the `ln z` Normal with the existing truncated
   normal on $[\ln z_{\rm cut} + \beta\,(\ln(\Delta R/R))\,,\ \ln\tfrac12]$, bounds
   computed from the grooming record persisted in the artifact (at the fielded
   `beta = 0` the bound is constant; implement the general cell-conditional form now so
   `beta != 0` files need no code change). Reuse `trunc_normal_logpdf`; the PIT comes
   from `std_normal_cdf` through the WP2 `pit_coords` path unchanged.
3. Sampling and point estimates inherit the truncation (draws and modes are inside the
   support by construction).
4. If — and only if — G3 fails on the truncated head, the pre-authorized escalation is
   a monotone rational-quadratic spline on the same interval (Durkan et al.,
   arXiv:1906.04032), as a follow-up plan, not a mid-run change. The deeper structural
   note for that plan: with the `LundGenerator` conventions,
   $\ln z = u + v - \ln p_{T,\rm sum}^{(t)}$ holds *exactly*, so the
   coordinate-independence-given-cell assumption is violated by a kinematic identity
   the per-node joint flow (`PLAN_UPDATES.md` WP1) resolves.

## 5. WP-B — multiplicity

1. **Diagnostic first (no training):** `eval/` gains a per-step comparison of the
   continue probability teacher-forced vs on-policy at matched depth, reported per
   depth bin. This attributes the ⟨N⟩ deficit to exposure bias vs prefix support
   drift before any remedy is defaulted.
2. **`decode.cont_temperature: float = 1.0`** (exact no-op at 1.0 — parity): a single
   scalar on the continue logit at sampling, fitted by matching the held-out
   training-val N-marginal mean; artifact records value + `fitted_under`; frozen on
   test. Applies retroactively to the v0 checkpoint.
3. **Explicit-N arm:** the `n_head` variant (v3-style: categorical `q(N|x)` by direct
   NLL, cells conditioned on the realized N) fielded under the *same* truncated-`ln z`
   head and aux configuration as the baseline, so the family difference is the
   multiplicity factorization alone. This is v0 §10's "v3 vs v4", finally measured —
   under the G8 gating rule.

## 6. WP-C — decode-layer estimator repairs (no retraining)

1. `inference/mbr.py`: `mbr_select` returns the medoid with its own sampled
   coordinates; no mode re-attachment. Both backends (`energyflow`, `pot`) unchanged
   in the risk computation; the headline `multiplicity >= 1` test with
   `min_emissions = 0` carries over.
2. `inference/point_estimate.py` + model `map_estimate`s: κ-gate per the design
   decision; `psi_identified` flag threaded into `LundPointEstimate` nodes.
3. Reporting: MAP moves out of headline tables into diagnostics; the decode headline
   is MBR (medoid / geo-median), the population headline stays the decode-free
   posterior series — consistent with the MBR rationale of consensus-over-mode
   (Kumar & Byrne, HLT-NAACL 2004; Eikema & Aziz, arXiv:2005.10283).

## 7. WP-D — assessment-suite upgrades

1. **Support audit as a standard scored metric:** window, soft-drop, and `kt`-floor
   violation rates of the sampled posterior enter `dist_closure_metrics.json` with a
   hard-zero target (v0 found the 0.88% ad hoc in §8; a regression must never ride
   along unmeasured again).
2. **TARP power:** run at the 2 000-jet tier with `tarp_refs >= 200`; recompute the
   null band at the run's own (n, refs); require the floor itself below 0.05 before
   the statistic is quoted. Add region-stratified TARP over the v0 quadrants where
   the per-region n permits a meaningful null.
3. **Region × coordinate PITs:** the `stratify_regions` × `pit_coords` cross — the
   instrument that attributes `narrow_soft` (leak-concentration vs genuine
   under-conditioning) and localizes any residual `ln z` misfit.
4. **`empty_threshold_for_rate` / `cont_temperature` bookkeeping asserts** in the
   closure notebook, extending the v0 §7 `tau.fitted_under` pattern.

## 8. The grid

All arms use `lnz_support = "physical"` unless named otherwise, `n_bins = 30`, the
asymmetric-floor train file (seed 1) and test file (seed 2), 60 epochs — identical to
v0 except where the arm's one variable differs.

| arm | varies | seeds | trainings |
|---|---|---|---|
| `v1_base` | — (v4 + `lundnet` + aux(9) + physical `ln z`) | 0, 1, 2 | 3 |
| `v1_legacy_lnz` | `lnz_support = "legacy"` | 0 | 1 |
| `v1_ctrl` | `aux_features = [ln_pt, abs_eta]` | 0, 1, 2 | 3 |
| `v1_nhead` | explicit `q(N\|x)` factorization | 0, 1 | 2 |
| `v1_gru` | `encoder = gru` | 0 | 1 |
| `v1_deepsets` | `encoder = deepsets` | 0 | 1 |
| `v1_wide` *(optional)* | `dec_dim = 128` | 0 | 1 |

11 (+1 optional) trainings at ~1 h — one overnight grid. The training-free items
(WP-B.2, WP-C, WP-D) additionally run on the v0 checkpoint
`runs/prod_test_v0/20260731-212800-8209a78a33/best.ckpt`.

The aux question v1 answers is **isolation** (`v1_base` vs `v1_ctrl`: what do the
secondary-plane + groomed-mass columns buy over pure jet kinematics), not existence —
v0 §5 settled existence and its strata need no re-measurement.

## 9. Pre-registered gates

Evaluated on the independent seed-2 test file; coverage intervals are 95% Wilson
(Brown, Cai & DasGupta, Statist. Sci. **16** (2001) 101); regions with n < 30 reported
`scored: false` as in v0 §4.

| # | gate | criterion |
|---|---|---|
| G1 | acceptance (carried) | medoid **and** geo-median beat identity on both estimators (cell tier, off-grid tier), agreeing in sign |
| G2 | support | sampled window / soft-drop / floor violation rates ≡ 0 on `v1_base`; any nonzero value is a bug, not a finding |
| G3 | `ln z` PIT | KS ≤ crit (0.016 at the v0 n) on `v1_base`; `v1_legacy_lnz` reproduces the v0-scale failure (attribution) |
| G4 | N-marginal | on the quoted configuration: SBC-on-N χ² < 16.90 **and** leading-cell 68% coverage Wilson-consistent with 0.68 in every scoreable region **and** ⟨N⟩_post/⟨N⟩_truth ∈ [0.95, 1.05] frozen-transfer |
| G5 | `narrow_soft` | coverage passes G4's regional clause, **or** the region × coordinate PITs deliver a documented mechanistic attribution (which passes the gate and opens a follow-up WP) |
| G6 | decode | repaired-MBR W1 gmean < 1.0 vs plain RSD; point-estimate ψ resultant \|R\| within 2× of truth's; no scoreable observable above 3× |
| G7 | TARP | max dev inside the recomputed null band, with the band's floor < 0.05 |
| G8 | family A/B | `v1_base` vs `v1_nhead` decided on coordinate PITs + TARP + coverage + held-out NLL; SBC-N reported but non-deciding (v3's `n_head` is calibrated on it nearly by construction) |

**Validity checks carried unchanged from v0 §1:** seed disjointness with the `full`
fingerprint (sequences + jet four-vector), asymmetry verification of the test file,
and the train/test same-generator noise-floor table as the systematic stand-in.

## 10. Tests

- `tests/test_lnz_support.py`: MC normalization of the four-coordinate density over
  the box ≈ 1 on a frozen context (physical mode); `"legacy"` bit-identical
  `per_jet_nll` (parity guard); sampled violation count ≡ 0 at 10⁵ draws; PIT of
  model-drawn samples uniform (the SBC null, per the WP2 self-consistency pattern);
  general-β bound formula against hand-computed values.
- `tests/test_cont_temperature.py`: `T = 1.0` bit-identical sampling statistics under
  a fixed seed; fitted `T` reproduces the val N-marginal mean within tolerance;
  artifact round-trips value + `fitted_under`; `per_jet_nll` provably untouched.
- `tests/test_mbr.py` (extend): medoid coordinates equal the selected sample's
  coordinates exactly; κ-gate returns sampled ψ + `psi_identified=False` below
  threshold and the mode + `True` above; headline `multiplicity >= 1`
  (`min_emissions = 0`) retained.
- `tests/test_calibration_v2.py` (extend): support-audit metrics present and zero on
  a generator-drawn sample; TARP null band recomputed at configured (n, refs);
  stratified PIT keys present per region.
- Parity: `scripts/verify_parity.py` + `tests/test_parity.py` bit-for-bit with all
  new switches at defaults (`lnz_support="legacy"`, `cont_temperature=1.0`,
  κ-gate at its no-op bound for the pinned reference path).

## 11. Docs

- `docs/CONFIGURATION.md` §4/§7: `lnz_support`, `cont_temperature`,
  `kappa_min_mode`, TARP power fields; a boxed note that NLL is not comparable across
  the head change.
- `docs/README_PHYSICS.md`: the ψ identifiability caveat (mode of a flat circular
  density) and the medoid-carries-its-sample rule; MAP's demotion to diagnostic.
- `PROD_TEST_v1_RESULTS.md` skeleton mirroring v0's structure, gates G1–G8 as its
  section spine.

## 12. Non-goals (deferred, with triggers)

- **Per-node joint coordinate density** (cINN coords / CFM, `PLAN_UPDATES.md` WP1).
  Trigger: G3 fails on the truncated head, or the region × coordinate PITs show the
  independence-given-cell approximation binding.
- **Full-tree LundNet with secondary-plane sequences** (WP3; Dreyer & Qu,
  arXiv:2012.08526). Trigger: `v1_base` vs `v1_ctrl` shows the secondary columns
  carry the aux gain.
- **HERWIG driver / fragmentation-variation weights** (WP5). Not in v1; the noise-floor
  table remains the stand-in, as v0 §10 records.
- **Finer grid / more cells.** Blocked by the v0 §8 capacity argument until the
  decoder-trunk probe (`v1_wide`) says otherwise.
- **Scheduled sampling or any training-side exposure-bias remedy** — improper
  objective (Huszár, arXiv:1511.05101); the likelihood stays faithful.
- **Sampling-side multiplicity floor** — unchanged from `PLAN_NsplitMinCut.md`:
  truncating draws would distort SBC/PIT/coverage.

## 13. Verification

1. `python -m pytest tests/test_lnz_support.py tests/test_cont_temperature.py tests/test_mbr.py tests/test_calibration_v2.py -q`
2. `python scripts/verify_parity.py` + `pytest tests/test_parity.py -q` — bit-for-bit
   with defaults (proves every switch is a no-op off).
3. Launch the §8 grid; `h2p-rsd-junipr eval` each `best.ckpt` with
   `experiment.pit_coords=true experiment.stratify_regions=true experiment.tarp=true`.
4. Retroactive pass of WP-B.2 / WP-C / WP-D on the v0 checkpoint; append to the v0
   results doc as an addendum, not a revision.
5. Fill `PROD_TEST_v1_RESULTS.md` against gates G1–G8.

Run in the conda `fno_env_mlx` environment.

## 14. Risks

- **Quoted-configuration ambiguity** (tempered vs untempered sampler) — closed by the
  pre-registration in §3; both always reported.
- **NLL discontinuity across the head change** — expected and documented; the
  `v1_legacy_lnz` arm is the bridge; never compare v1 NLL to v0 NLL directly.
- **`config_hash` churn** — expected for new runs; resume self-consistency covered by
  the existing checkpoint test.
- **G4's frozen-transfer tolerance** may bind on a genuinely well-fitted temperature
  if the seed-2 file's N-marginal drifts beyond the §1 noise floor — in that case the
  gate fails *informatively* (it is measuring transfer, which is its job).
- **v3-by-construction bias in G8** — mitigated by the deciding-metric rule; SBC-N is
  reported for completeness only.
- **MBR backend scale** — the `energyflow`/`pot` backends agree on the argmin but not
  the numeric scale (merged `PLAN_MBR_PerturbativeLund.md`); one backend per analysis,
  named in the artifact.

## References

Larkoski, Marzani, Soyez & Thaler, arXiv:1402.2657 · Dreyer, Necib, Soyez & Thaler
(RSD), arXiv:1804.03657 · Dreyer, Salam & Soyez, arXiv:1807.04758 · Talts et al. (SBC),
arXiv:1804.06788 · Lemos et al. (TARP), arXiv:2205.03910 · Stahlberg & Byrne,
arXiv:1908.10090 · Eikema & Aziz, arXiv:2005.10283 · Kumar & Byrne, HLT-NAACL 2004 ·
Ranzato et al., arXiv:1511.06732 · Bengio et al., arXiv:1506.03099 · Huszár,
arXiv:1511.05101 · Guo et al., arXiv:1706.04599 · Durkan et al., arXiv:1906.04032 ·
Yang et al., arXiv:1711.03953 · Dreyer & Qu (LundNet), arXiv:2012.08526 · Mardia &
Jupp, *Directional Statistics*, Wiley 2000 · Brown, Cai & DasGupta, Statist. Sci.
**16** (2001) 101.
