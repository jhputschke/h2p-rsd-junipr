# h2p-rsd-junipr

**Amortized hadron-to-parton (h2p) hadronization-inversion posterior** over groomed
Lund trees. Given the hadron-level recursive-Soft-Drop primary Lund sequence `x` of
a jet, the model learns an amortized neural posterior `q_φ(y | x) ≈ p(y | x)` over
the matched **pre-hadronization parton-level** groomed tree `y` (Cranmer, Brehmer &
Louppe, *PNAS* **117** (2020) 30055, arXiv:1911.01429; Papamakarios & Murray,
*NeurIPS* 2016, arXiv:1605.06376).

This is the productionized form of the `conditional_rsd_junipr_v2.py` research
script, built to [`PRODUCTION-PLAN-v4.md`](docs/PRODUCTION-PLAN-v4.md): one model
contract, many families behind a registry; config-first (OmegaConf, no Hydra); a
lean custom Trainer with exact checkpoint/resume; a built+tested C++ data-generation
stage (FastJet + fjcontrib LundPlane + PYTHIA 8); and a mandatory validation suite.

> **Physics background:** [`README_PHYSICS.md`](docs/README_PHYSICS.md) explains the QCD
> setting (parton shower → Lund-string hadronization → grooming), why the inverse is
> a posterior rather than a function, and how each physics choice maps onto the code.

## Why "h2p"

The forward physics map is parton → hadron (showering then hadronization). This
repo learns the **inverse**: hadron → parton, as a calibrated posterior, not a
point estimate — in high dimensions the mode can be unrepresentative, so every jet
is reported with both a MAP/beam estimate and a posterior summary. For the point
estimate you can pick the joint-mode **MAP** (`decode.point_estimator=map`, floored
away from the empty tree) or the mode-free, **floor-free MBR** — the drawn tree of
least expected perturbative-Lund EMD to the posterior
(`decode.point_estimator=mbr`, `[mbr]` extra; two backends, `pot` default and
`energyflow`).

## Install

```bash
pip install -e .                 # core: torch, numpy, omegaconf, uproot
pip install -e ".[track,serve,dev]"   # optional extras
pip install -e ".[mbr]"          # MBR point estimator (pot backend); add ".[energyflow]" for the reference EMD
```

After a fresh clone, run once to keep notebook outputs out of git:

```bash
bash setup_nbstripout.sh   # --status to check, --global to apply machine-wide
```

`.gitattributes` (committed) says `*.ipynb` goes through the `nbstripout` filter;
the filter *driver* lives in `.git/config`, which git never clones — so this step is
per-clone, and skipping it means notebooks get committed with their outputs. The
script installs `nbstripout` if missing and registers the driver for **this repo
only**. `pre-commit install` activates the same stripping as a second, independent
net (see `.pre-commit-config.yaml`).

Stripped: outputs, execution counts, and two per-machine metadata keys —
`metadata.kernelspec` (whose kernel you happened to pick) and
`metadata.language_info.version` (your Python patch version). Without those, a
notebook shows as modified for whoever opened it last. The key list is duplicated
in `setup_nbstripout.sh` and `.pre-commit-config.yaml`; change both together or
the two strippers will undo each other.

> **ARM64 (Dell GB10, Apple Silicon):** the `[energyflow]` extra pulls in
> `wasserstein`, a C++ extension with no prebuilt ARM wheel, so pip compiles it
> from source. Its headers declare `enum ... : char` members with value `-1`,
> which is out of range because `char` is *unsigned* by default on ARM — the
> build fails with `enumerator value '-1' is outside the range of underlying
> type 'char'`. `wasserstein` also `#include`s `<omp.h>`, which Apple clang does not
> ship, so the build additionally needs an OpenMP include/lib (conda's `llvm-openmp`,
> already present in most envs; `conda install -c conda-forge llvm-openmp` otherwise).
> Force a signed `char` **and** point the compiler at conda's OpenMP:
>
> ```bash
> CFLAGS="-fsigned-char -I$CONDA_PREFIX/include" \
> CXXFLAGS="-fsigned-char -I$CONDA_PREFIX/include" \
> LDFLAGS="-L$CONDA_PREFIX/lib -lomp -Wl,-rpath,$CONDA_PREFIX/lib" \
>   pip install --no-binary wasserstein -e ".[energyflow]"
> ```
>
> Two *runtime* quirks on this platform are handled automatically by the package, so
> no action is needed: PyTorch and `wasserstein` each link an OpenMP runtime (macOS
> would abort with `OMP: Error #15`), so `inference.mbr` sets `KMP_DUPLICATE_LIB_OK=TRUE`
> before first use; and `wasserstein`'s batched `emds` uses `np.array(..., copy=False)`,
> which raises under NumPy ≥ 2, so the energyflow backend falls back to the (identical)
> per-pair `emd`. The `[mbr]` default (`pot` backend) needs no compilation and none of
> this applies. On x86-64 (`char` is signed there, OpenMP is found) no flags are needed.

## Quickstart

```bash
# train the §5.1 autoregressive JUNIPR (v2, continuous coords) on synthetic data
h2p-rsd-junipr train model=ar_junipr_v2 encoder=gru trainer.max_epochs=20

# v3 = v2 + a first-class multiplicity head q(N|x): q(y|x) = q(N|x)·q(y|N,x)
h2p-rsd-junipr train model=ar_junipr_v3 encoder=gru trainer.max_epochs=20

# v4 = v3 + decoder cross-attention over the per-node hadron states
h2p-rsd-junipr train model=ar_junipr_v4 encoder=lundnet model.dec_dim=52

# swap the model family or encoder — drop-in, no code changes
h2p-rsd-junipr train model=cinn encoder=lundnet encoder.num_layers=3 geometry.n_bins=16

# conditional flow matching with an EXACT probability-flow-ODE likelihood
h2p-rsd-junipr train model=cfm encoder=gru

# closure / calibration / point-estimate on held-out jets
h2p-rsd-junipr eval runs/<id>/best.ckpt

# ...with the full calibration suite (per-coordinate PITs, region strata, TARP)
h2p-rsd-junipr eval runs/<id>/best.ckpt \
    experiment.pit_coords=true experiment.stratify_regions=true experiment.tarp=true

# generate real data with PYTHIA 8 (after building cpp/, below) then train on it
h2p-rsd-junipr generate 100000 jets.root 1
h2p-rsd-junipr train data=rntuple data.path=jets.root model=ar_junipr_v2
```

CLI overrides are OmegaConf-dotted and **type-checked against the schema** — a typo
like `optim.lrr=1e-3` or `geometry.n_bins=ten` fails at load, not at hour three.

> **Full workflow guide:** [`docs/USAGE.md`](docs/USAGE.md) covers training and the
> run-dir layout, `best.ckpt` vs `last.ckpt`, pre-empt/resume (with a worked
> example), the eval/calibration metrics, programmatic inference
> (`log_prob`/`sample`/`map_estimate`), export, serving, and sweeps.
>
> **Config knob reference:** [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) explains
> every parameter field-by-field — geometry, data, encoder, model, optim, trainer, and
> the inference/decode knobs (the MAP floor / mincut, the learned quantile floor, length
> penalty, sampling temperature, and the `point_estimator` / `mbr_*` MBR knobs).

## Verification (this is the acceptance test)

Two checks reproduce the original v2 script on the **same synthetic data**:

```bash
python scripts/verify_parity.py      # bit-for-bit NLL parity vs the v2 script
python scripts/verify_synthetic.py   # full train + closure on the v2 synthetic data
```

- **`verify_parity.py`** copies the monolithic v2 model's weights into the
  refactored `ARJunipr` (split encoder + registry) and asserts `per_jet_nll`
  matches — observed **max |Δ| = 0.0** (`scripts/verify_parity` output). The module
  split does not change the likelihood.
- **`verify_synthetic.py`** trains `ar_junipr_v2` for 20 epochs on the identical
  synthetic dataset (seed 0, 8000 jets) and runs the closure suite. Result
  (`scripts/verify_synthetic_result.txt`): final **val NLL/jet = 20.76** vs the v2
  reference 20.71; posterior mean multiplicity 6.33 vs 6.14; leading-cell 68%
  coverage 0.71 vs 0.68 — PASS. The MAP / plain-RSD / truth trees are identical.

## Data contract

One entry per jet (`write_lund_rntuple` → `Jets` RNTuple): `event`, `jet_index`,
`weight`, jet kinematics, grooming provenance `(z_cut, beta, kt_floor, kt_floor_sec,
generator)`,
and two **node-unaligned** jagged sequences `x_*` (hadron) and `y_*` (parton) in
`(ln 1/ΔR, ln k_t, ln z, ψ)`. There is by design no per-node x↔y correspondence,
so the objective and all closure observables are jet-level.

## Model families (one contract: `log_prob` / `sample` / `map_estimate`)

| family | module | `log_prob` exact? | status |
|---|---|---|---|
| §5.1 autoregressive JUNIPR (v1 cells / v2 +continuous coords / v3 +multiplicity head / v4 +cross-attention) | `models/ar_junipr.py` | ✅ | primary, verified |
| §5.2 conditional normalizing flow (cINN) | `models/cinn.py` | ✅ | functional baseline |
| §5.3 conditional diffusion / bridge | `models/diffusion.py` | ❌ surrogate | cheap-sampler baseline |
| §5.4 conditional flow matching (exact probability-flow-ODE likelihood) | `models/cfm.py` | ✅ | verified density, unvalidated posterior |

Encoders (`gru`, `lundnet`, `deepsets`) are independently pluggable; any encoder
pairs with any decoder family.

> **What the status column means.** Only `ar_junipr` is *verified* in the
> [Verification](#verification-this-is-the-acceptance-test) sense — weight-level parity plus a
> train+closure run — and it is the only family with trained runs on real data under
> [`runs/`](runs/). The other three carry the full shared contract (calibration hooks, the
> decode floors, aux conditioning, the empty-sequence and support guards), each parametrized
> over every family in `tests/`, but none has a closure or calibration verdict on data beyond
> the short demo in [`calibration_v2_walkthrough.ipynb`](notebooks/calibration_v2_walkthrough.ipynb).
> "Baseline" is about that missing validation, not about missing plumbing.

- **`ar_junipr_v3`** promotes the sequence length to a first-class categorical `q(N|x)`
  head — the factorization `q(y|x) = q(N|x)·q(y|N,x)`, opt-in via
  `use_multiplicity_head` and off by default (v2 stays bit-for-bit unchanged);
  see [`docs/PLAN_MultHead.md`](docs/PLAN_MultHead.md).
- **`ar_junipr_v4`** additionally lets the decoder cross-attend to the encoder's
  *per-node* hadron states instead of only the pooled `e(x)`, removing the fixed-length
  bottleneck. Residual, so the off path is byte-identical. Whether it helps depends on how
  long your hadron sequences are — it is a large win on the synthetic generator and a wash
  on the tightly-groomed PYTHIA sample; measure it on your data
  ([`CONFIGURATION.md` §4](docs/CONFIGURATION.md#4-model--the-posterior-family)).
- **`cfm`** is the exact-likelihood member of the continuous-time family. The
  `exact_likelihood` column above is a real class attribute: `diffusion`'s `log_prob` is
  a denoising-score-matching **surrogate**, so its NLL is not comparable with the others
  and `train`/`eval`/`serve` say so out loud. Use `cfm` for NLL model selection and
  likelihood ratios. Its *density* is checked quantitatively — exact divergence against
  autograd, and the coordinate density integrating to **1.004 ± 0.009** over the physical
  support, with a sign-flipped control that fails by many σ ([`tests/test_cfm.py`](tests/test_cfm.py)).
  That is a correctness result, not a calibration one: nothing yet says the *posterior* it
  produces is trustworthy on data.

The post-review work packages behind the last three rows — and the calibration suite that
gates them — are in [`docs/PLAN_UPDATES.md`](docs/PLAN_UPDATES.md): WP1–WP4 are merged,
WP5 (the systematics chain) is not started.

## Is the posterior calibrated?

A conditional generator is not calibrated for free (the original cINN unfolding came out
too narrow), so this gates "trustworthy". Beyond SBC/PIT/coverage on the multiplicity,
three opt-in diagnostics test what SBC-on-N cannot — and must, since `ar_junipr_v3`
optimizes that marginal directly and would pass it near-tautologically:

```bash
h2p-rsd-junipr eval runs/<id>/best.ckpt \
    experiment.pit_coords=true experiment.stratify_regions=true experiment.tarp=true
```

- **per-coordinate PITs** — the kinematics, coordinate by coordinate, via each family's
  exact conditional CDFs, broken down by emission index and region. U-shaped ⇒
  over-confident, dome ⇒ over-dispersed. Each report carries the `space` it was computed
  in, and only `ar_junipr` is `physical`: `cinn` and `cfm` report `latent`, because both
  reach their density through a map that mixes the four coordinates (coupling layers, the
  probability-flow ODE), so a base dimension is not one physical coordinate. The latent
  histograms are still a genuine per-dimension test — under a calibrated flow every base
  marginal is exactly `N(0,1)` — but they do not localize *which* kinematic is off.
- **region stratification** — every metric binned by the leading emission's Lund
  quadrant, so calibration that only holds *on average* over the plane cannot pass.
- **TARP** expected coverage (Lemos et al., arXiv:2302.03026) on tree-valued posteriors
  under the perturbative-Lund EMD — a *joint* test in the physics metric.

Walkthrough on real PYTHIA data:
[`notebooks/calibration_v2_walkthrough.ipynb`](notebooks/calibration_v2_walkthrough.ipynb).
Reference: [`docs/CONFIGURATION.md` §8](docs/CONFIGURATION.md#8-experiment--evaluation-suite).

## C++ data generation

```bash
conda activate js_fno                # dependencies live in this env (see note below)
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
ctest --test-dir cpp/build           # Soft Drop boundary + matching unit tests
./cpp/build/pythia_driver 100000 jets.root 1 cpp/cards/pp_dijet.cmnd  # nEvents out seed card
./cpp/build/read_lund_rntuple jets.root Jets   # inspect: schema, provenance, #jets, first jet
```

> **Dependencies via conda.** ROOT, FastJet, fjcontrib, and PYTHIA 8 are provided by
> the `js_fno` conda environment, so configure with that environment **activated** —
> CMake keys off `$CONDA_PREFIX` to locate them. Note that the conda-forge
> `fastjet-contrib` package ships LundPlane bundled inside `libfastjetcontribfragile`
> (there is no standalone `libLundPlane`); the CMake accepts either layout, so
> source/homebrew installs still work. If you switch environments, delete
> `cpp/build/CMakeCache.txt` before reconfiguring.

Finds ROOT (≥6.36, RNTuple), FastJet, fjcontrib LundPlane, and PYTHIA 8 (optional;
falls back to a toy event source). The `pythia_driver` reads the pre-hadronization
shower partons from the PYTHIA event record with MPI off (Bierlich et al.,
arXiv:2203.11601); a parallel HERWIG driver powers the §8 generator systematic.

**Configuration, not hardcoding.** Generation *and* the FastJet/grooming
parameters are driven by a single PYTHIA command card (`cpp/cards/pp_dijet.cmnd`):
the jet radius/ptMin/acceptance/match-cone and the Soft Drop `z_cut/beta/R0/kt`
floor are registered as custom PYTHIA settings (`cpp/include/run_settings.hpp`), so
one file sets everything and the grooming values are stamped into the RNTuple
provenance. Omit the card to fall back to the built-in defaults.

**Asymmetric k_t floors** (`cpp/cards/pp_dijet_asym_floor.cmnd`). `SoftDrop:ktFloorSec`
sets a separate, looser floor for the **off-spine** branches of the aux traversal,
leaving the spine — and therefore the persisted `x`/`y` sequences, inputs *and* targets
— bit-for-bit unchanged. It exists because the aux scalars are conditioning inputs,
never targets, and a fixed *absolute* floor cuts far deeper off-spine than on it: at the
1 GeV default 80.6% of jets carry no passing secondary splitting at all. Measured on
identical events, `1.0 / 0.2` takes `⟨x_nsec⟩` from 0.251 to 2.213 (zero fraction 80.6%
→ 20.5%) with `⟨n_x⟩` and `⟨n_y⟩` unmoved. Unset (the default) mirrors `ktFloor`, which
is the historical single-floor behaviour. Note it *redefines* `x_mg`/`x_ptg` — see
[`docs/PLAN_Input.md`](docs/PLAN_Input.md) ("Asymmetric k_t floors") for the trade-off,
the generator-systematic caveat, and the production recipe for scanning floors without
re-running PYTHIA per floor.

**Aux conditioning columns.** Besides the two primary sequences, the writer persists
per-jet **groomed all-branch** scalars the primary-only sequence structurally cannot
represent: `x_mg` (the pipeline-groomed jet mass — every primary node is recorded
massless) and `x_nsec` (grooming-passing splittings on *non-primary* branches).
`fullLundAux` computes them by recursing over the whole C/A tree under the same
`passesGroom` predicate the sequences use, so `n_primary` provably equals the primary
sequence length (`cpp/tests/test_lund_io.cpp`). They are opt-in on the Python side via
`encoder.aux_features` (nine registered features; `pt_g`, `|eta|` and the secondary-plane
kinematics are implemented but not yet A/B'd) and **off by default — the A/B of the
original triple did not clear its adoption bar on the reference sample** (−0.029 nat/jet against a 0.029 seed spread, because 82.6 % of
those jets have no secondary-plane activity at all to exploit); see
[`docs/PLAN_Input.md`](docs/PLAN_Input.md) and
[`notebooks/aux_input_ab.ipynb`](notebooks/aux_input_ab.ipynb) for the numbers and for
what to change before re-judging. Files written before these columns existed still
read — the reader guards on the RNTuple descriptor.

## Layout

`src/h2p_rsd_junipr/` — `config` · `geometry` · `features` · `distributions` ·
`data/` · `encoders/` · `models/` · `inference/` · `eval/` · `train/` · `serving/`
· `cli`. `configs/` — YAML group files composed by `config.py`. `cpp/` — the
generation stage. `tests/` — pytest mirror. See
[`docs/PRODUCTION-PLAN-v4.md`](docs/PRODUCTION-PLAN-v4.md) for the full design and
the §14 phased roadmap.
