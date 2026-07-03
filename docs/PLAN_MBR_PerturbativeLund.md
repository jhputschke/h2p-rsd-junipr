# PLAN — MBR point estimate under a perturbative Lund (EMD) metric

Status: proposed (not yet implemented). Adds a *minimum Bayes risk* point estimator that
selects, from the posterior draws callers already take, the tree of least expected
perturbative-Lund distance to the posterior. Opt-in via `decode.point_estimator="mbr"`;
the default (`"map"`) short-circuits before any new code, so the likelihood and the MAP
path are untouched (parity preserved). The optimal-transport solve has two interchangeable
backends — a self-contained POT augmented-cost form (`pot`) and the reference Energy
Mover's Distance implementation `energyflow.emds` (`energyflow`; Komiske, Metodiev &
Thaler, *Phys. Rev. Lett.* **123** (2019) 041801, arXiv:1902.02346). Reproducing the KMT
collider-event EMD *exactly* (its coordinates, `R`, `beta`, `norm`) is surfaced as
configuration and left to the user, not pinned to the paper. Builds on
`docs/PLAN_NsplitMinCut.md` and `docs/PLAN_QuantileMinCut.md` (both merged).

## Context — why this change

The MinCut plans established that the empty-tree collapse is a property of the **decision
rule**, not the trained density: `ŷ = argmax_y q_φ(y|x)` is the joint mode, which for a
high-entropy discrete sequence posterior is length-biased and collapses to `n=0` for a
large fraction of jets even at converged NLL (`PLAN_NsplitMinCut.md`). `min_emissions`
and `length_floor_quantile` *mask* that bias by clamping length, but they remain
mode-based estimators of a slice — they answer "the single most probable tree (of length
≥ floor)", a quantity whose optimality holds only under exact-whole-tree 0/1 loss, which
no analysis cares about.

MBR replaces the mode with the Bayes-optimal decision under a loss we *do* care about
(Kumar & Byrne, HLT-NAACL 2004; Eikema & Aziz, COLING 2020, arXiv:2005.10283, who show
the mode is essentially arbitrary while the distribution is faithful — exactly our SBC
picture). Two properties make it the principled point estimate here:

- **Alignment-free by construction.** The loss is a distance between two *radiation
  patterns*, not between paired nodes, so it respects the project's hard constraint that
  there is **no per-node x↔y correspondence** — only jet-level pairing is valid.
- **The empty tree is never selected, with no floor.** An empty cloud has large expected
  distance to typical non-empty draws (pure mass-imbalance penalty), so it can never
  minimize the risk. The brevity bias is removed *structurally* rather than clamped —
  `min_emissions` becomes unnecessary for the MBR estimator (it is retained only for the
  MAP estimator).

The perturbative-region restriction — elsewhere a principle applied to *inputs* — here
enters cleanly through the **support of the ground metric**: emissions below a `ln kt`
cut are dropped before the distance is computed, so hadronization-region jitter cannot
dominate the risk. Physics motivation and estimator robustness coincide.

## Key facts (inherited from the merged MinCut work — reconfirm against the current tree)

> Line numbers below are quoted from `PLAN_NsplitMinCut.md` / `PLAN_QuantileMinCut.md`;
> re-verify before editing, since the merges may have shifted them.

- Posterior draws are already taken at every MAP call site
  (`serving/api.py:predict`, `eval/closure.py:print_point_estimate`), so MBR reuses them
  with **no extra sampling** — same pattern as `learned_min_emissions(..., mults=mults)`.
- `map_estimate`/`map_tree` return a structured `LundPointEstimate`
  (`.multiplicity`, `.nodes` each carrying `(ln 1/ΔR, ln kt, ln z, ψ)` in v2, `.logprob`).
  MBR must return the **same type** so it is a drop-in for every consumer.
- Posterior samples are emission sequences (cells in v1; continuous-coordinate nodes in
  v2); `geometry` exposes cell→centre and `lund_distance`/`leading_emission_cell`
  already used in the closure suite. The MBR cloud adapter reuses these.
- `DecodeConfig` is threaded through `decode_params(cfg)` with tolerant
  `OmegaConf.select` backfill; `config_hash` changes for **new** runs only; old
  checkpoints load. New decode fields must follow this path (never read
  `cfg.decode.<newfield>` directly).
- Parity guard `scripts/verify_parity.py` + `tests/test_parity.py` (atol 1e-5) asserts
  `per_jet_nll` bit-for-bit. **MBR touches no training/likelihood code and lives outside
  `point_estimate.py`, so parity is preserved trivially.**
- `scripts/verify_synthetic.py` checks **posterior** bands, not the point estimate, so it
  is unaffected.

## The estimator

Given `K` posterior draws `S = {y⁽¹⁾,…,y⁽ᴷ⁾} ~ qφ(·|x)` (the same draws used for the
credible bands), sampling-based MBR (Eikema & Aziz, EMNLP 2022) uses `S` as both the
candidate set and the Monte-Carlo support for the expectation:

$$
\hat{y}_{\mathrm{MBR}} \;=\; \arg\min_{h\in\mathcal{C}}\; \frac{1}{K}\sum_{k=1}^{K} d\big(h, y^{(k)}\big),
\qquad \mathcal{C}\subseteq \mathcal{S},
$$

where `d` is the perturbative-Lund distance below. With `C = S` this is `O(K²)`
evaluations of `d`; `mbr_n_candidates` optionally shrinks `C` (asymmetric MBR) for speed.
The winning `ĥ` is a genuine drawn tree, so it already has coordinates; we additionally
report `log qφ(ĥ|x)` (one cheap `model.log_prob`) and the achieved risk
`r(ĥ) = mean_k d(ĥ, y⁽ᵏ⁾)` alongside it. The risk is a decision-theoretic score, **not**
a likelihood — reported separately from NLL.

## The metric — perturbative Lund EMD

Each draw is mapped to a weighted point cloud in the Lund plane and compared by the
Energy Mover's Distance (Komiske, Metodiev & Thaler, *Phys. Rev. Lett.* **123** (2019)
041801, arXiv:1902.02346), an optimal-transport distance that is IRC-safe and defined on
sets of unequal cardinality.

For a draw `y` with emissions `eᵢ = (ln 1/ΔRᵢ, ln ktᵢ, ln zᵢ, ψᵢ)`:

- **Cloud.** Keep only emissions in the perturbative region, `ln ktᵢ ≥ ln kt_cut`. Each
  surviving emission contributes a point `pᵢ = (ln 1/ΔRᵢ, ln ktᵢ)` (optionally extended
  with `ln zᵢ` and the periodic `ψᵢ`) and a weight `wᵢ` set by `mbr_weight`: `kt`
  (default, IRC-safe momentum scale), `z` (momentum fraction), or `unit`.
- **Distance.** With ground metric the Euclidean distance (raised to `beta`) in Lund
  coordinates,

$$
d(y, y') \;=\; \min_{f_{ij}\ge 0}\ \sum_{ij} f_{ij}\,\lVert p_i - p'_j\rVert^{\beta}
\;+\; R\,\Big|\, \textstyle\sum_i w_i - \sum_j w'_j \Big|,
$$

the standard EMD with a mass-imbalance penalty of radius `R`. The penalty term is what
makes an empty (or shorter) cloud expensive against typical non-empty draws, so the MBR
winner is never the empty tree — no floor required.

Design choices and why:

- **Alignment-free / IRC-safe.** EMD compares the radiation pattern as a whole; it needs
  no node pairing and inherits IRC safety from the `kt`-weighting and the KMT
  construction — both core project constraints.
- **Perturbative restriction = metric support.** `ln kt_cut` defaults to the geometry's
  high-`kt` / grooming threshold (do **not** hard-code a second physics constant; inherit
  it from the region config) so the loss is computed where the inverse is well-posed.
- **`R` sets the length/kinematics trade-off.** Large `R` penalises multiplicity
  mismatch heavily (MBR tracks the count); small `R` favours kinematic agreement of the
  shared hard emissions. Default `R ≈` Lund-plane diameter; check closure-metric
  stability in `R` (unequal-mass EMD is known to depend on it).
- **`beta`** is the angular weighting exponent; `beta = 1` is the true 1-Wasserstein EMD
  of KMT, `beta = 2` an energy-distance-like variant. Default `1.0`.
- **v1 vs v2.** v1 cells use cell centres (grid-quantised, coarser metric); v2 uses
  continuous coordinates (preferred). The adapter handles both; the cloud is the only
  representation MBR sees.

### Backends — `pot` and `energyflow`

The OT solve is pluggable via `mbr_backend`; both implement the same mathematical object
above, and the choice is about provenance and batching, not semantics:

- **`pot` (default, self-contained).** Build the explicit augmented cost — pad the
  smaller cloud with a sink particle of weight `|Σw − Σw'|` at ground distance `R` from
  all real points — and call `ot.emd2`. No physics package, fewest dependencies; the
  imbalance penalty `R·|Σw − Σw'|` is written out by hand, matching the equation above
  exactly.
- **`energyflow` (literature reference).** Map each cloud to an `(M, 1+gdim)` event array
  with the weight in column 0 and the Lund coordinates after it, and call
  `energyflow.emds(events_C, events_S, R=R, beta=beta, norm=norm, gdim=gdim,
  periodic_phi=mbr_periodic_phi)` for the whole pairwise matrix in one batched,
  multiprocessed call. EnergyFlow implements the unequal-mass term internally (by adding a
  particle carrying the lesser total weight), so we pass clouds **unpadded** on this path.
- **`surrogate` (fast pre-filter).** A fully vectorised binned Lund-image χ² with no OT;
  used to rank candidates cheaply before refining the top few with true EMD (see Cost &
  scaling).

> **Convention note (left to the user).** EnergyFlow normalises ground distances by `R`
> internally, so for unequal total weight its returned value equals the augmented-cost
> `pot` value divided by `R` (and `norm=True` rescales weights to unit sum, which
> *removes* the imbalance term — see Risks). The two backends therefore agree on the
> *argmin* but not on the numeric scale unless conventions are matched. Reproducing the
> KMT collider-event EMD verbatim — hadronic `(pT, y, φ)` coordinates with `periodic_phi`,
> their `R`, `beta = 1`, `norm` — is a configuration the user dials in through the `mbr_*`
> knobs; the defaults here are tuned for the Lund-plane application, not pinned to the
> paper.

## Edits (ordered)

1. **Config knobs** — `src/h2p_rsd_junipr/config.py`: add to `DecodeConfig` and
   `_DECODE_DEFAULTS` (so `decode_params` auto-threads + backfills for old snapshots),
   mirror in `configs/decode/default.yaml`:
   - `point_estimator: str = "map"`  ∈ `{map, mbr}`  — default reproduces today exactly.
   - `mbr_backend: str = "pot"`  ∈ `{pot, energyflow, surrogate}`.
   - `mbr_n_candidates: int = 0`  — `0` ⇒ all draws are candidates.
   - `mbr_lnkt_cut: float | None = None`  — `None` ⇒ inherit the geometry/region cut.
   - `mbr_weight: str = "kt"`  ∈ `{kt, z, unit}`.
   - `mbr_coords: str = "lnDR_lnkt"`  ∈ `{lnDR_lnkt, +lnz, +psi}`  — which columns enter
     the ground metric (`gdim` follows; `+psi` engages periodicity).
   - `mbr_R: float = <Lund diameter>`.
   - `mbr_beta: float = 1.0`  — energyflow angular exponent; `pot` path raises the cost to
     this power too, so the two stay consistent.
   - `mbr_norm: bool = False`  — energyflow weight normalisation; **off by default** so the
     imbalance term (and the empty-tree-never-wins property) is kept.
   - `mbr_periodic_phi: bool = False`, `mbr_phi_col: int = -1`  — only used when
     `mbr_coords="+psi"`; exact phi-column wiring is version-dependent and left to the
     user (see the convention note).
   `mbr_R`, `mbr_beta`, `mbr_norm`, `mbr_periodic_phi` are pass-throughs whose KMT-exact
   values are the user's call.

2. **New module** `src/h2p_rsd_junipr/inference/mbr.py` (kept out of the parity-critical
   `point_estimate.py`, mirroring `inference/length.py`; export from
   `inference/__init__.py`). Pure post-hoc inference, no model/training imports beyond the
   `PosteriorModel` contract:
   - `lund_cloud(draw, geom, *, lnkt_cut, weight, coords) -> (pts: (m,g) np.ndarray, w: (m,) np.ndarray)`
     — cells→centres or v2 nodes→coords, perturbative cut applied; weights *not*
     pre-normalised (let `mbr_norm` decide). `coords` selects `g ∈ {2,3,4}`.
   - `cloud_to_event(pts, w) -> (m, 1+g) np.ndarray` — energyflow event layout: weight in
     column 0, coordinates after. Single shared adapter for the energyflow path.
   - `lund_emd(cloud_a, cloud_b, *, R, beta, norm, periodic_phi, phi_col, backend) -> float`
     — single-pair distance. `pot`: build the augmented cost + sink particle, `ot.emd2`.
     `energyflow`: `energyflow.emd.emd(cloud_to_event(a), cloud_to_event(b), R=R,
     beta=beta, norm=norm, gdim=g, periodic_phi=periodic_phi)`. `surrogate`: binned χ².
   - `lund_emd_matrix(clouds_C, clouds_S, *, ..., backend) -> (|C|, K) np.ndarray` — the
     primitive MBR actually uses. **`energyflow` path is one `energyflow.emd.emds(eventsC,
     eventsS, R=, beta=, norm=, gdim=, periodic_phi=)` call** (batched + multiprocessed);
     `pot` path loops with a cached per-pair ground-distance matrix; `surrogate` is fully
     vectorised.
   - `mbr_select(model, xf, nx, *, draws=None, n_samples, lnkt_cut, weight, coords, R,
     beta, norm, periodic_phi, phi_col, n_candidates, backend, geom) -> LundPointEstimate`
     — if `draws is None`, sample; else **reuse** them. Builds clouds once, fills the
     `|C|×K` matrix via `lund_emd_matrix`, returns the argmin tree as a `LundPointEstimate`
     with `.logprob = model.log_prob(ĥ)` and an added `.risk`.
   - **Lazy, per-backend imports.** `import ot` and `import energyflow` happen *inside* the
     chosen path, each guarded by `try/except ImportError` raising one actionable message
     (`"mbr_backend='energyflow' requires the [mbr] extra: pip install energyflow"`). The
     default `point_estimator="map"` path imports neither.

3. **`PosteriorModel` hook** — `src/h2p_rsd_junipr/models/base.py`: no new abstract
   method needed (MBR consumes `sample`/`sample_batch` + `log_prob`, already in the
   contract). Add only a thin convenience `map_or_mbr(self, xf, nx, **decode)` on the base
   that dispatches on `decode["point_estimator"]` to `map_estimate` (default) or
   `inference.mbr.mbr_select`, so all three families gain MBR with no per-family code.

4. **Wire into the call sites that already draw samples** (reuse, no double-sample):
   - `serving/api.py` `predict`: sample first (already the QuantileMinCut order); when
     `point_estimator=="mbr"`, call `mbr_select(..., draws=draws)` and return its tree as
     the point estimate; add `mbr_risk` and `mbr_backend` to the response (additive,
     non-breaking).
   - `eval/closure.py`: add MBR as an **additional estimator** in the aggregate closure
     (`run_closure`) and the per-jet print, beside MAP / posterior-mode / learned-floor —
     so the leading-emission Lund-distance and multiplicity-bias panels gain an `MBR`
     series. This is the scientific payoff: does MBR beat the mode estimators on
     `dLund-to-truth` and `⟨n − n_true⟩`?
   - `cli.py` `cmd_eval`: thread `decode.point_estimator` (and the `mbr_*` keys) from the
     checkpoint snapshot via `decode_params`.

5. **Notebook §6** — `notebooks/inference_demo.ipynb`: add a 5th estimator,
   **"MBR (perturbative Lund)"**, computed from the per-jet `draws` already taken; add it
   to the multiplicity panel, the leading-emission distance panel, and the bias/RMSE
   print; one markdown bullet explaining it is mode-free and floor-free, and a one-line
   `mbr_backend` toggle so the demo runs without `energyflow` installed (defaults to
   `pot`).

## Cost & scaling

- Clouds are tiny (~`n` perturbative emissions, single digits), so each EMD is a small
  LP; the cost is `#jets × |C| × K` solves.
- **`energyflow` batches the pairwise matrix for free**: `energyflow.emds` computes the
  whole `|C|×K` block in one multiprocessed call, so the per-jet matrix is a single
  invocation. Prefer it when throughput matters and the dependency is acceptable.
- Other levers, in order: (a) `mbr_n_candidates` to shrink `C` while keeping full `S` for
  the expectation; (b) cache ground-distance matrices on the `pot` path; (c)
  `mbr_backend="surrogate"` (binned Lund-image χ²) as a fast first pass, refining the top
  few candidates with true EMD; (d) batch the `pot` solves.

## Tests

- New `tests/test_mbr.py` (reuse `conftest.batch`), parametrised over
  `backend ∈ {pot, energyflow, surrogate}` with `pytest.importorskip("energyflow")` /
  `importorskip("ot")` guarding the optional paths:
  - `lund_cloud` drops sub-`lnkt_cut` emissions; weights are raw (un-normalised); handles
    empty draws; `coords` selects the right `gdim`.
  - `lund_emd`: `d(a,a)=0`, symmetry, `d(∅, non-empty) > 0`; finite on ragged
    cardinalities.
  - **Backend agreement:** `pot` and `energyflow` give the **same argmin** over a fixed
    candidate set; their values agree up to the documented `1/R` convention (assert on the
    ratio / on rank order, *not* bit-for-bit — they need not be numerically identical).
  - `lund_emd_matrix` (`energyflow.emds`) matches looped `lund_emd` (energyflow) on a toy.
  - **Headline:** `mbr_select` never returns `n=0` when any non-empty draw exists — assert
    `multiplicity ≥ 1` **with `min_emissions=0`** (no floor), the property the MinCut
    floors had to enforce by clamping. (Skip under `mbr_norm=True`, which removes the
    imbalance term — see Risks.)
  - `point_estimator="map"` is a structural no-op (MAP identical) — dispatch short-circuit.
  - determinism under the existing seed; all three families return a valid tree.
- `tests/test_config.py`: new keys present in the full set; `decode_params` backfills them
  for an old snapshot without raising.
- `tests/test_decode_plumbing.py`: a `predict` with `point_estimator="mbr"` returns a
  non-empty tree and `mbr_risk`/`mbr_backend` fields; with `"map"` the output is unchanged.
- `tests/test_checkpoint.py`: a snapshot lacking the `mbr_*` keys round-trips its own
  `config_hash` (resume guard intact).

## Docs

- `docs/USAGE.md`: the `point_estimator`/`mbr_*` knobs + an inference snippet for **both**
  backends; note MBR needs the optional `[mbr]` extra and that `pot` and `energyflow` are
  independently importable.
- `docs/README_PHYSICS.md`: the MBR concept (decision rule vs density); the perturbative
  Lund EMD (IRC-safety, `lnkt_cut` as metric support, `R`, `beta`); the **two backends and
  the `1/R` / `norm` convention mapping**; an explicit pointer that KMT-exact reproduction
  is user configuration; and **why MBR needs no `min_emissions` floor** (contrast with the
  MinCut estimators).
- `docs/PRODUCTION-PLAN-v4.md`: the new `DecodeConfig` fields + a one-line note that MAP,
  posterior mean/median, **and MBR (two backends)** are now available point estimators.
- `notebooks/README.md`: the 5th §6 estimator and the `mbr_backend` toggle.

## Verification

Run in the conda `fno_env_mlx` environment.

1. `pip install -e ".[mbr]"` — the extra pulls **both** `pot` (e.g. `pot>=0.9`) and
   `energyflow` (e.g. `energyflow>=1.3`, EMD available from 0.11.0+). Each is lazy-imported,
   so a missing one only errors when its backend is explicitly selected.
2. `python -m pytest tests/test_mbr.py tests/test_config.py tests/test_decode_plumbing.py tests/test_models.py -q`.
3. `python scripts/verify_parity.py` + `python -m pytest tests/test_parity.py -q` —
   bit-for-bit `per_jet_nll` (MBR is off by default and touches no likelihood code).
4. `python scripts/verify_synthetic.py` — posterior bands unchanged.
5. CLI A/B/C: `... eval <ckpt> decode.point_estimator=map` vs
   `decode.point_estimator=mbr decode.mbr_backend=pot` vs `... mbr_backend=energyflow` —
   MBR reported; the `n=0` fraction is 0% **with `decode.min_emissions=0`** (floor-free);
   the two backends agree on the selected tree; compare `dLund-to-truth` and
   `⟨n − n_true⟩` against MAP and the learned-floor MAP.
6. Re-run `notebooks/inference_demo.ipynb` §6 — the 5th panel shows MBR vs MAP vs
   learned-floor MAP vs posterior median; flip `mbr_backend` to confirm both run.
7. `python -m pytest -q` — full suite green.

## Risks

- **`O(K²)` cost at production scale** — clouds are tiny so each solve is cheap; the
  scaling levers are `energyflow.emds` batching, `mbr_n_candidates`, and the `surrogate`
  pre-filter (see Cost & scaling).
- **Backend convention mismatch** — `pot` (hand-rolled augmented cost) and `energyflow`
  implement the same object but differ by EnergyFlow's internal `1/R` ground-distance
  normalisation; they agree on the argmin, not the numeric scale. Pick **one backend per
  analysis** for comparable risk numbers; the value is reported with its `mbr_backend` tag.
- **`mbr_norm=True` re-enables degeneracy** — normalising weights to unit total removes
  the imbalance term, so a shorter/empty cloud is no longer penalised for missing mass and
  the empty-tree-never-wins guarantee is lost. Off by default; documented; the headline
  test is skipped under it.
- **Metric sensitivity (`R`, `beta`, `lnkt_cut`)** — physics knobs. Inherit `lnkt_cut`
  from the region config; document the `R`↔(length vs kinematics) trade-off and check
  closure-metric stability across `R`.
- **New dependencies (POT, energyflow)** — lazy, per-backend imports; optional `[mbr]`
  extra; clear error only when the backend is requested. Default path and parity stay
  dependency-free. `energyflow` pulls in the `Wasserstein`/POT stack — heavier than `pot`
  alone, which is why `pot` is the default.
- **periodic-φ wiring is version-dependent** — `periodic_phi`/`phi_col` semantics have
  shifted across EnergyFlow releases; only engaged when `mbr_coords="+psi"`, and exact
  reproduction is left to the user per the convention note.
- **v1 grid quantisation** — cell-centre clouds coarsen the metric; acceptable, v2 is
  preferred; documented.
- **MBR risk ≠ likelihood** — surface `risk` and `log qφ` as distinct fields; never feed
  `risk` into anything expecting an NLL.
- **Genuinely empty/short posteriors** — if a jet's draws are mostly empty (high, honest
  uncertainty), MBR may pick a short tree. That is *correct* — it reflects the posterior,
  unlike a hard floor that would manufacture emissions. Documented as intended behavior.
- **`config_hash`** changes for new runs (as with the merged knobs); old checkpoints load
  via the tolerant `decode_params` / `OmegaConf.select` backfill.
