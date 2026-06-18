# h2p-rsd-junipr

**Amortized hadron-to-parton (h2p) hadronization-inversion posterior** over groomed
Lund trees. Given the hadron-level recursive-Soft-Drop primary Lund sequence `x` of
a jet, the model learns an amortized neural posterior `q_φ(y | x) ≈ p(y | x)` over
the matched **pre-hadronization parton-level** groomed tree `y` (Cranmer, Brehmer &
Louppe, *PNAS* **117** (2020) 30055, arXiv:1911.01429; Papamakarios & Murray,
*NeurIPS* 2016, arXiv:1605.06376).

This is the productionized form of the `conditional_rsd_junipr_v2.py` research
script, built to [`PRODUCTION-PLAN-v4.md`](PRODUCTION-PLAN-v4.md): one model
contract, many families behind a registry; config-first (OmegaConf, no Hydra); a
lean custom Trainer with exact checkpoint/resume; a built+tested C++ data-generation
stage (FastJet + fjcontrib LundPlane + PYTHIA 8); and a mandatory validation suite.

> **Physics background:** [`README_PHYSICS.md`](README_PHYSICS.md) explains the QCD
> setting (parton shower → Lund-string hadronization → grooming), why the inverse is
> a posterior rather than a function, and how each physics choice maps onto the code.

## Why "h2p"

The forward physics map is parton → hadron (showering then hadronization). This
repo learns the **inverse**: hadron → parton, as a calibrated posterior, not a
point estimate — in high dimensions the mode can be unrepresentative, so every jet
is reported with both a MAP/beam estimate and a posterior summary.

## Install

```bash
pip install -e .                 # core: torch, numpy, omegaconf, uproot
pip install -e ".[track,serve,dev]"   # optional extras
```

## Quickstart

```bash
# train the §5.1 autoregressive JUNIPR (v2, continuous coords) on synthetic data
h2p-rsd-junipr train model=ar_junipr_v2 encoder=gru trainer.max_epochs=20

# swap the model family or encoder — drop-in, no code changes
h2p-rsd-junipr train model=cinn encoder=lundnet encoder.num_layers=3 geometry.n_bins=16

# closure / calibration / point-estimate on held-out jets
h2p-rsd-junipr eval runs/<id>/best.ckpt

# generate real data with PYTHIA 8 (after building cpp/, below) then train on it
h2p-rsd-junipr generate 100000 jets.root 1
h2p-rsd-junipr train data=rntuple data.path=jets.root model=ar_junipr_v2
```

CLI overrides are OmegaConf-dotted and **type-checked against the schema** — a typo
like `optim.lrr=1e-3` or `geometry.n_bins=ten` fails at load, not at hour three.

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
`weight`, jet kinematics, grooming provenance `(z_cut, beta, kt_floor, generator)`,
and two **node-unaligned** jagged sequences `x_*` (hadron) and `y_*` (parton) in
`(ln 1/ΔR, ln k_t, ln z, ψ)`. There is by design no per-node x↔y correspondence,
so the objective and all closure observables are jet-level.

## Model families (one contract: `log_prob` / `sample` / `map_estimate`)

| family | module | status |
|---|---|---|
| §5.1 autoregressive JUNIPR (v1 cells / v2 +continuous coords) | `models/ar_junipr.py` | primary, verified |
| §5.2 conditional normalizing flow (cINN) | `models/cinn.py` | functional baseline |
| §5.3 conditional diffusion / bridge | `models/diffusion.py` | functional baseline |

Encoders (`gru`, `lundnet`, `deepsets`) are independently pluggable; any encoder
pairs with any decoder family.

## C++ data generation

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
ctest --test-dir cpp/build           # Soft Drop boundary + matching unit tests
./cpp/build/pythia_driver 100000 jets.root 1
```

Finds ROOT (≥6.36, RNTuple), FastJet, fjcontrib LundPlane, and PYTHIA 8 (optional;
falls back to a toy event source). The `pythia_driver` reads the pre-hadronization
shower partons from the PYTHIA event record with MPI off (Bierlich et al.,
arXiv:2203.11601); a parallel HERWIG driver powers the §8 generator systematic.

## Layout

`src/h2p_rsd_junipr/` — `config` · `geometry` · `features` · `distributions` ·
`data/` · `encoders/` · `models/` · `inference/` · `eval/` · `train/` · `serving/`
· `cli`. `configs/` — YAML group files composed by `config.py`. `cpp/` — the
generation stage. `tests/` — pytest mirror. See `PRODUCTION-PLAN-v4.md` for the
full design and the §14 phased roadmap.
