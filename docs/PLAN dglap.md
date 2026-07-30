# PLAN — DGLAP baseline: perturbative-prior-anchored discrete heads

**Status:** proposed (not yet implemented)

Anchor the discrete heads of every posterior family at the first-principles
perturbative Lund density: a fixed, precomputed per-cell log-prior added to the
`split_head` / `cell_head` logits (and, optionally, a Poisson anchor on the
multiplicity head), so the network learns a *residual* correction

    logits(cell | ·) = b_cell + NN(·),        b_cell = ln ∫_cell ρ_pert(u, v) du dv

instead of the absolute density. Softmax renormalizes automatically, so the
likelihood stays exactly normalized and the network retains full capacity to
undo the baseline — the fitted object is still the simulator-implied posterior
(no auxiliary loss, no likelihood distortion). Opt-in via
`model.dglap_baseline` (default `"none"`), off path byte-identical to today.
Builds on the merged `PLAN_MultHead.md`; composes with `PLAN_Input.md`.

## Context

The coordinate basis already encodes the leading DGLAP structure: the soft and
collinear singularities `dz/z · dθ/θ` are absorbed into the log-Lund measure, so
the emission density is approximately uniform in `(ln 1/Δ, ln kt)` (Dreyer,
Salam & Soyez, arXiv:1807.04758). What the heads currently relearn from data is
the known *residual* perturbative shape on top of that flat limit: the running
coupling α_s(kt) — the dominant gradient of the Lund density along `ln kt` —
the finite-z parts of the Altarelli–Parisi kernels (Altarelli & Parisi, Nucl.
Phys. B126 (1977) 297), the kinematic edges of the Lund triangle, and the
Sudakov/Poisson structure of the groomed emission count (Frye, Larkoski, Thaler
& Zhou, arXiv:1704.06266). The single-emission density is known analytically to
NLL (Lifson, Salam & Soyez, arXiv:2007.06578).

Building the known part in as a fixed logit offset is the established pattern in
neural phase-space sampling — map the singular/known structure analytically,
learn the smooth remainder (i-flow, arXiv:2001.05486; Bothmann et al.,
arXiv:2001.05478; MadNIS, arXiv:2212.06172). For a *posterior* q(y|x) the payoff
is directional: where x is informative the network overrides the prior; where
hadronization has washed the information out, the posterior relaxes toward the
perturbative baseline instead of an arbitrary network extrapolation. With
`dglap_zero_init` the model *starts* exactly at the baseline. In the heavy-ion
extension, medium modification then appears as localized structure in the
learned correction across the Lund plane — the full-tree z_g-style probe.

## Design (recommended approach)

### 1. Physics module — `src/h2p_rsd_junipr/dglap.py` (new, flat like `geometry.py`)

```python
def alpha_s_1loop(kt, lambda_qcd=0.25, nf=5, kt_freeze=1.0):
    """One-loop running coupling, frozen below kt_freeze (the grooming floor)."""

def lund_density_ll(u, v, *, color_factor, lambda_qcd, nf, kt_freeze):
    """LL primary-plane density rho(u, v) = 2 * C * alpha_s(e**v) / pi
    (Dreyer, Salam & Soyez, arXiv:1807.04758) — u-independent at this order."""

def cell_log_prior(geometry, cfg_model) -> tuple[torch.Tensor, float]:
    """(b_cell (n_cells,) log-normalized per-cell prior, lambda0 total expected
    groomed multiplicity = sum_cell ∫ rho). Modes: "ll" (analytic above) |
    "table" (np.load((n_bins, n_bins)) density grid; shape validated against
    geometry, else raise). Integration: midpoint with 8x8 subdivisions per cell
    (rho is smooth in log coordinates; exactness is not required — any residual
    quadrature error is absorbed by the learned correction).
    Optional static kinematic mask (dglap_pt_ref is not None): subtract
    dglap_mask_penalty from cells whose center violates either linear edge at
    pT = pt_ref — the hard edge v <= ln(pt_ref/2) - u and, for beta = 0, the
    Soft Drop wedge v >= ln(z_cut * pt_ref) - u (both straight lines in (u, v):
    the log-coordinate rationale). FINITE penalty, never -inf: reconstructed
    truth cells may violate the soft-limit edges, and the NLL must stay finite."""

def poisson_log_pmf(lam, n_max) -> torch.Tensor:
    """(n_max+1,) log Poisson(n; lam), renormalized over the truncated support."""
```

The `"table"` mode decouples the anchor from the in-model `"ll"` closed form:
any density on the geometry grid can be supplied. Production of the grid is part
of this plan (§5); an externally produced NLL grid (Lifson–Salam–Soyez,
arXiv:2007.06578) remains loadable through the same interface. Which anchor was
used is a physics statement, not a detail, and it follows the *origin* of the
density, not the backend: analytic grids (`ll`, `lo_ap`, an LSS table) and
parton-level events from an NLL-accurate shower are *perturbative* anchors;
generator events with hadronization/tune content are *simulator-prior* anchors.
The run config records `dglap_table`, and the §5 sidecar records how — and from
what — it was made. The heavy-ion interpretation (correction = medium
modification relative to vacuum pQCD) requires a perturbative anchor.

### 2. Config — shared fields on all three model schemas

Append to `ARJuniprConfig`, `CINNConfig`, `DiffusionConfig` (config.py:80–119):

```python
    dglap_baseline: str = "none"           # none | ll | table  ("none" == parity)
    dglap_color_factor: float = 1.3333333  # C_F (quark); 3.0 for gluon-initiated
    dglap_lambda_qcd: float = 0.25         # GeV; one-loop
    dglap_nf: int = 5
    dglap_kt_freeze: float = 1.0           # freeze alpha_s below (== grooming floor)
    dglap_table: str | None = None         # .npy grid for mode "table"
    dglap_pt_ref: float | None = None      # static Lund-edge mask at this pT; None == off
    dglap_z_cut: float = 0.1               # SD wedge of the mask (mirror the writer default)
    dglap_mask_penalty: float = 20.0       # finite additive logit penalty
    dglap_anchor_n: bool = False           # Poisson(lambda0) anchor on the length head
    dglap_zero_init: bool = True           # zero final head layer -> start AT the baseline
    #                                        (read only when dglap_baseline != "none")
```

All reads in the families use the `getattr(m, "dglap_baseline", "none")` idiom
(the `cell_label_smoothing` precedent, ar_junipr.py:59), so old checkpoint
configs load unchanged.

### 3. Family choke points — one wrapper per head, no call-site drift

**ARJunipr.** `self.split_head(...)` is called at exactly five sites
(ar_junipr.py:180 likelihood, :209 `_step`, :217 `_step_batched`, :226
`_step_cells`, :313 staged-MAP scoring). Add in `__init__`
(after ar_junipr.py:94):

```python
self._dglap_on = getattr(m, "dglap_baseline", "none") != "none"
if self._dglap_on:
    b, lam0 = cell_log_prior(geometry, m)
    self.register_buffer("dglap_bias", b, persistent=False)     # recomputed from
    if getattr(m, "dglap_anchor_n", False):                     # config at init;
        if not self.use_multiplicity_head:                      # never in the
            raise ValueError("dglap_anchor_n requires the v3 "  # state_dict
                             "multiplicity head (PLAN_MultHead)")
        self.register_buffer("dglap_n_bias",
                             poisson_log_pmf(lam0, self.max_emissions),
                             persistent=False)
```

and route **every** head call through:

```python
def _split_logits(self, h):   # replaces raw self.split_head(...) at all 5 sites
    raw = self.split_head(h)
    return raw + self.dglap_bias if self._dglap_on else raw

def _n_logits(self, e):       # replaces raw self.n_head(...) at all its sites
    raw = self.n_head(e)      # (ar_junipr.py:166, 237, 274, 302; test_length_floor
    return raw + self.dglap_n_bias if ... else raw   # reads model.n_head -> update)
```

Because beam search and ancestral sampling receive the model's step callables
(`_step*`), the bias propagates identically to the NLL, MAP, samples, and
`length_pmf` — a single definition, no likelihood/decode inconsistency possible.

**cINN / diffusion.** Same two wrappers around `cell_head` / `n_head`
(cinn.py:106–113 construction; call sites cinn.py:133, 137, 160–162, 186–190;
diffusion.py:57–63, 83–94, 138–139, 158, 165–166). The per-step categorical
structure differs (cINN's cells are conditionally iid given e), but the anchor
is the same additive prior.

### 4. Start-at-baseline init

When `dglap_baseline != "none"` and `dglap_zero_init`, zero the final `Linear`
of `split_head`/`cell_head` (and `n_head` when anchored): the untrained model's
cell marginal is then exactly `softmax(b_cell)` and its length pmf exactly the
truncated Poisson — physical tails from step 0, and the early-training
zero-multiplicity pathology cannot be seeded by random logits. Off path this
code never runs (init parity).

### 5. Table production — `scripts/make_dglap_table.py` (new, standalone)

A standalone tool in the `scripts/` house pattern (argparse; imports the package,
is not imported by it), writing `out.npy` — the `(n_bins, n_bins)` density grid
`cell_log_prior` consumes — plus a sidecar `out.json` recording backend, all
arguments, the geometry, the package version, and (measured backend) the input
path and the datamodule-style content fingerprint. Two backends; the numerical
core of each is a plain importable function so the tests need no ROOT and no CLI:

```python
def measured_table(jets, geometry, level="parton") -> np.ndarray:
    """Average groomed primary Lund density of the pipeline itself: histogram the
    per-jet sequences over Geometry.to_cell, weighted by each jet's `weight`,
    divided by (sum of weights) x (cell_wu * cell_wv). `level="parton"` (default)
    is the y-side — the density the anchored heads actually model; "hadron" is
    available for diagnostics. Consistent BY CONSTRUCTION with the writer's
    grooming (SD boundary + kt floor + matching acceptance), since it is built
    from the persisted sequences. This is a simulator-prior anchor (see §1)."""

def lo_ap_table(geometry, *, pt_ref, flavor="quark",
                lambda_qcd=0.25, nf=5, kt_freeze=1.0) -> np.ndarray:
    """LO Altarelli-Parisi density at fixed reference pT:
    rho(u, v) = alpha_s(kt)/pi * zP(z) evaluated at z = e^(u+v)/pt_ref (the
    soft-limit Lund map kt = z pT Delta), with the exact LO kernel for the
    chosen flavor (Altarelli & Parisi 1977; zP(z) -> C_F resp. C_A as z -> 0,
    so the soft corner reproduces the in-model "ll" mode) and rho = 0 outside
    z in (0, 1/2] — the hard edge enters the density itself, superseding the
    static dglap_pt_ref logit mask when this backend is used. alpha_s is the
    same dglap.alpha_s_1loop (single definition, no drift)."""
```

CLI wiring: `--backend {measured,lo_ap}`, `--out`, geometry overrides
(`--n-bins`, `--u-range`, `--v-range`, defaulting to the package defaults),
`--path/--ntuple/--level` for `measured` (via `data/rntuple.py:load_rntuple`),
`--pt-ref/--flavor/--lambda-qcd/--nf/--kt-freeze` for `lo_ap`; post-processing
`--combine q.npy g.npy --gluon-fraction f` and `--boundary-fill {ll,lo_ap}
--u-min-valid u0` operate on existing tables and write a new one with a sidecar
referencing the parents. The tool refuses
to overwrite an existing table without `--force` (a silently regenerated anchor
under an unchanged `dglap_table` path would defeat the sidecar provenance).

**Shower-NLL tier — PanScales ee processes through the measured backend.**
`measured_table` is agnostic to the event source; fed *parton-level* events
from an NLL-accurate shower it yields a perturbative anchor with zero
resummation code. Availability constrains the scope, not the mechanism: the
public PanScales framework (gitlab.com/panscales/panscales-0.X; van Beekveld
et al., SciPost Phys. Codebases 31 (2024), arXiv:2312.13275; NLL design goal
Dasgupta et al., arXiv:2002.11114) covers e+e- and, for hadron collisions,
colour-singlet production only (formulation arXiv:2205.02237, validation
arXiv:2207.09467) — **pp dijets, i.e. QCD 2->2 with four coloured legs, are
not available**, so the tier runs on ee: Z -> qqbar for a *flavor-pure quark*
table and H -> gg for a *flavor-pure gluon* table (the framework's own
reference processes), with the central-jet mapping E_jet ~ pt_ref and shower
cutoff below the 1 GeV kt floor. Region of validity follows from collinear
universality: in the **collinear core** (large u) the density is governed by
alpha_s(kt), the kernels, and multiple-emission effects independent of the
production environment — there the ee tables are genuinely NLL. In the
**boundary band** (u -> 0, DeltaR ~ R) — ISR entering the cone, colour
connections to the beam, pp non-global/clustering structure — an ee density is
wrong at single-log level; that band is covered at NLL only by the LSS pp
calculation (its soft large-angle + clustering terms, arXiv:2007.06578), so
for boundary cells an author-supplied LSS grid is the sole NLL source. The
script therefore post-processes: `--boundary-fill {ll,lo_ap}` replaces cells
with `u < --u-min-valid` by the analytic density, and the sidecar records the
band, so the anchor's region of validity is machine-readable downstream while
`cell_log_prior` stays sidecar-blind (§1 interface unchanged). Two-table
mixing for the dijet flavor admixture: `--combine q.npy g.npy
--gluon-fraction f` (sidecar records both parents; f is a config-level
approximation — the learned correction absorbs its x-dependence). Event
routing as before: an event-input path in `write_lund_rntuple.cpp` (HepMC or
the PanScales-PYTHIA interface; small writer extension) or a PanScales-side
analysis writing the `.npy` in the package `Geometry`, the C++ route preferred
for grooming consistency by construction. The exactness argument that makes
*ungroomed* predictions usable is unchanged: the pipeline's grooming is a
per-emission filter on an unchanged C/A primary spine (`primaryLund` filters
`LundGenerator` declusterings, it never re-declusters), so the groomed-primary
*average density* on the retained region equals the ungroomed density
restricted to it — exact for `b_cell` and `lambda0`; grooming matters only for
joint/multiplicity correlations, which the anchor deliberately does not claim.
The sidecar's event-source and band records fix the anchor class and validity
region (§1).

What the tiers buy: `measured` on PYTHIA closes the loop end-to-end from an
existing `jets.root` and anchors at the exact simulator prior; `lo_ap` upgrades
the analytic anchor beyond the in-model soft-limit `"ll"` — finite-z falloff
toward the hard edge, flavor choice; `measured` on PanScales ee parton-level
events is the *collinear-region* NLL perturbative tier, boundary-filled and
flavor-mixed as above. Full-plane NLL for pp jets exists only as an
author-supplied LSS grid; only the analytic resummation itself stays out of
repo (Non-goals).

## Family coverage

The anchor lives entirely on the head side, so it is orthogonal to the encoder
choice (`gru` | `lundnet` | `deepsets`) and to the aux features of
`PLAN_Input.md`; it composes with any of them without interaction. What differs
per family is *which likelihood factors* are anchored:

| Likelihood factor            | ar_junipr (v2)      | ar_junipr (v3)     | cinn               | diffusion          |
|------------------------------|---------------------|--------------------|--------------------|--------------------|
| cells (`_split_logits`)      | anchored (per step) | anchored (per step)| anchored           | anchored           |
| length (`_n_logits`)         | — (continue/stop; `dglap_anchor_n` raises) | Poisson anchor | Poisson anchor | Poisson anchor |
| continuous coords            | not anchored (WP1)  | not anchored (WP1) | not anchored (WP1) | not anchorable (DSM) |
| anchored factors exact?      | yes                 | yes                | yes                | discrete yes; coords surrogate |

- **cINN is the structurally natural fit.** Its factorization — categorical
  q(N|x) plus conditionally iid cells given e — is the LL groomed emission
  model itself: independent emissions from the Lund density with Poisson
  multiplicity (Frye, Larkoski, Thaler & Zhou, arXiv:1704.06266). With
  `dglap_anchor_n` + `dglap_zero_init`, the *untrained* anchored cINN **is**
  the LL groomed shower exactly, and everything it learns is the correction to
  LL: inter-emission correlations, non-Poissonian multiplicity, and the
  hadron-level conditioning.
- **AR anchors the conditionals.** The bias enters each step's
  p(cell_t | prefix, x) as the same static prior — correct at LL (independent
  emissions), but the learned correction at step t mixes genuine physics with
  autoregressive bookkeeping; interpret per-step corrections accordingly.
- **Diffusion: discrete heads only.** Its multiplicity/cell factors anchor
  identically to cINN and stay exact, but the coordinate part is trained by
  denoising score matching around a Gaussian reference — no additive baseline
  insertion exists short of changing the reference process (out of scope; cf.
  the planned `exact_likelihood=False` flag). Treat anchored diffusion as a
  sampler baseline, not a likelihood-ratio tool.

## Interactions

- **`PLAN_MultHead.md`:** `dglap_anchor_n` is only meaningful with an explicit
  length head (v3 / cINN / diffusion); on the default continue/stop AR model it
  raises (fail loud) rather than silently doing nothing.
- **`PLAN_Input.md`:** independent and composable — aux features widen the
  conditioning, the baseline anchors the output density. `dglap_pt_ref` is a
  *static* stand-in for the true per-jet kinematic edge; the per-jet mask (edges
  from each jet's own pT) needs `jet_pt` threaded to decode time and is deferred
  (Non-goals).
- **MBR / decode layer:** untouched — MBR operates on draws, and the draws
  simply follow the (anchored) density. `decode_params` gains no fields.
- **WP1 (CFM) in `PLAN_UPDATES.md`:** the continuous-coordinate analog — using
  the baseline as the flow/CFM *base distribution* (finite-z P(z) tilt of the
  `ln z` head) — belongs there, not here.

## Tests & exit criteria

- **Unit (`tests/test_dglap.py`):** `alpha_s_1loop` positive, monotone
  decreasing, frozen below `kt_freeze`; `exp(b_cell)` sums to 1; `b_cell`
  invariant under a global rescale of ρ (normalization); `lambda0 > 0` and
  scales linearly with `color_factor` in `"ll"` mode; table-shape mismatch
  raises; with `dglap_pt_ref` set, penalized cells are exactly those violating
  the two straight edges; `poisson_log_pmf` normalized on the truncated support.
- **Table production (`tests/test_make_table.py`):** `measured_table` on a
  `synthetic_matched_dataset` jets list (no ROOT needed): counts route to the
  correct cells, per-jet weights are respected, densities are non-negative and
  integrate (x cell area) to the mean groomed multiplicity of the input;
  `lo_ap_table`: exactly zero above the z = 1/2 edge, and in the soft corner
  (z -> 0 cells) the ratio to the in-model `"ll"` density -> 1 for the matching
  color factor; round-trip: a written table loads through `cell_log_prior` with
  the identical `b_cell` as passing the array in memory; sidecar written;
  overwrite without `--force` raises.
- **Parity (the merged gate):** `dglap_baseline="none"` → `state_dict`,
  `log_prob`, and `scripts/verify_parity.py` bit-identical; `tests/test_parity.py`
  untouched; old checkpoints load strictly (buffers are `persistent=False` and
  off path never constructed).
- **Anchor behavior:** with baseline on + `zero_init`, the untrained AR model's
  split marginal equals `softmax(b_cell)` to float tolerance and `length_pmf`
  equals the truncated Poisson (v3); one training step decreases NLL. For the
  cINN the check is exact and stronger: the untrained anchored model's
  `log_prob` of any (N, cells) configuration equals the closed-form LL value
  `log Poisson(N; lambda0) + sum_t b_cell[t]` — assert equality, not closeness.
- **Synthetic honesty gate:** the synthetic simulator's density is *not* the LL
  Lund density, i.e. the baseline is deliberately misspecified there — the test
  is capacity-to-undo: trained val NLL with baseline on matches baseline off
  within seed spread on synthetic. A hard improvement claim on synthetic data
  would be testing the wrong thing.
- **Physics gate (PYTHIA path, ≥3 seeds):** (i) *init* NLL with the baseline is
  far below random-init (the anchor is roughly right for QCD); (ii) trained NLL
  is not worse than parity; (iii) region-stratified closure/coverage (WP2.2
  hooks) improves, or at least does not degrade, in the sparsely populated Lund
  corners — that is where the prior should earn its keep; (iv) extrapolation
  probe: train on one pT slice, evaluate on the adjacent slice — the anchored
  model should degrade more gracefully. Run the A/B on both exact-likelihood
  families (`ar_junipr_v3` and `cinn`); the cINN run doubles as the cleanest
  read of the learned correction, since its baseline is exactly LL. Adopt into
  a preset only on (i)–(iii).
- **Anchor-variation systematic (region-stratified):** re-evaluate the learned
  correction under the available anchors (`ll` | `lo_ap` | ee-shower-NLL |
  LSS grid if obtained) — retraining per anchor, or at minimum re-fitting from
  the shared zero-init recipe — and report the spread in the `generator_b`
  style of `eval/systematics.py`, **per Lund region** (WP2.2 stratification),
  with the sidecar's boundary band reported separately: that band is where
  anchor swaps disagree most and where correction structure must not be
  over-interpreted. This caution is doubled in the heavy-ion extension —
  medium-induced radiation is itself soft and wide-angle, i.e. it populates
  the same boundary cells as the anchor deficiency, so a medium-modification
  claim there requires the LSS-grade anchor, not the ee tier. The LSS
  precision itself — 5–7% at high pT, ~20% near the low-kt edge
  (arXiv:2007.06578) — sets the scale below which attribution is not
  meaningful anywhere on the plane.

## Non-goals

- **Per-jet kinematic edges** (mask from each jet's own pT at likelihood *and*
  decode time): requires threading `jet_pt` through the step callables; deferred
  until `PLAN_Input.md` lands and the static-mask study motivates it.
- **Continuous-coordinate baseline** (finite-z Altarelli–Parisi tilt of the
  `ln z` head, ψ structure): WP1 CFM base-distribution work.
- **Flavor mixtures** at the baseline level (q/g-weighted `color_factor`, or a
  per-jet flavor-dependent bias): a single configured color factor only; the
  learned correction absorbs the admixture. Revisit with the flavor-informing
  aux features.
- **No in-repo NLL resummation.** §5 now covers three tiers — measured
  (simulator prior), LO-AP (analytic), and shower-NLL (PanScales parton level
  through the measured backend); what remains excluded is reimplementing the
  Lifson–Salam–Soyez analytic resummation itself (CMW-scheme running, B-terms,
  boundary clustering/non-global logs, NLO matching; arXiv:2007.06578) — an
  author-supplied LSS grid loads through the same `"table"` interface with its
  own sidecar.
- **No auxiliary DGLAP loss term** — ever, per the don't-distort-the-likelihood
  principle: the baseline is a reparameterization of the same exact likelihood,
  not a penalty pulling the fit away from the target.

## References

Dreyer, Salam & Soyez, arXiv:1807.04758 · Lifson, Salam & Soyez,
arXiv:2007.06578 · Altarelli & Parisi, Nucl. Phys. B126 (1977) 297 · Frye,
Larkoski, Thaler & Zhou, arXiv:1704.06266 · Gao, Isaacson & Krause (i-flow),
arXiv:2001.05486 · Bothmann et al., arXiv:2001.05478 · Heimel et al. (MadNIS),
arXiv:2212.06172 · Larkoski, Marzani, Soyez & Thaler, arXiv:1402.2657 ·
Dasgupta et al. (NLL showers), arXiv:2002.11114 · van Beekveld et al.
(PanScales framework), SciPost Phys. Codebases 31 (2024), arXiv:2312.13275
