# Production Plan — RSD-JUNIPR Amortized Hadronization-Inversion Posterior

*From the research scripts (`conditional_rsd_junipr_v2.py`, `rsd_junipr_primary.py`,
`write_lund_rntuple.cpp`) to a maintainable, reproducible, multi-model research
codebase.*

**Plan version 4.** Configuration uses **OmegaConf with structured (dataclass)
schemas but no Hydra**: OmegaConf supplies runtime type validation, struct-mode
typo rejection, deep merge, dotted-CLI parsing, and interpolation, while config-
group composition stays an explicit ~60-line loop and sweeps stay a small script
(§2). This sits between v2 (Hydra + OmegaConf) and v3 (no Hydra, no OmegaConf —
plain dataclasses + pyyaml). Unchanged from v3: the lean custom `Trainer` (not
Lightning, §6) and dependency-free CSV/JSONL + optional TensorBoard logging (§9).
CLI ergonomics and the model-abstraction/registry design are the same throughout.

The deliverable is an amortized neural posterior estimator $q_\phi(y\mid x)\approx
p(y\mid x)$ over groomed Lund trees (Cranmer, Brehmer & Louppe, *PNAS* **117**
(2020) 30055, arXiv:1911.01429; Papamakarios & Murray, *NeurIPS* (2016),
arXiv:1605.06376). The current `ConditionalPrimaryLundJUNIPR` is the §5.1
autoregressive realization (JUNIPR, Andreassen et al., *EPJC* **79** (2019) 102,
arXiv:1804.09720; binary JUNIPR, *PRL* **123** (2019) 182001, arXiv:1906.10137);
the cINN (§5.2; Bellagente et al., *SciPost Phys.* **9** (2020) 074,
arXiv:2006.06685) and diffusion/Schrödinger-bridge (§5.3; arXiv:2404.18807)
families must be **drop-in alternatives**, not rewrites.

The four explicit requirements drive the whole design:

1. **Model flexibility** — §5.1 / §5.2 / §5.3 behind one interface and a registry.
2. **Parameter flexibility** — every knob (`emb_dim`, `enc_dim`, `dec_dim`,
   `ctx_dim`, `N_BINS`, the Lund ranges, encoder type, head depths, optimiser)
   in versioned config, never hard-coded.
3. **Training + checkpoint/resume** — a first-class trainer with exact resume.
4. **Self-contained data generation** — the C++ RNTuple writer promoted to a
   built, tested, containerized stage of the pipeline.

---

## 0. Guiding principles

- **Config-first, code-second.** No physics or architecture constant lives in
  source. The discretisation globals (`LN_INVDELTA_RANGE`, `LN_KT_RANGE`,
  `N_BINS`) and every `__init__` width become config fields with schema
  validation and provenance.
- **One contract, many models.** All posterior estimators expose
  `log_prob`, `sample`, `map_estimate`. The trainer, the validation suite, and
  the serving layer never know which family they hold.
- **Reproducibility is a feature.** Every run pins config + git SHA + data
  fingerprint + RNG state into the checkpoint, so any result is re-creatable.
- **Validation is not optional** (methodology §6): closure, posterior
  calibration (SBC; Talts et al., arXiv:1804.06788), and the PYTHIA-vs-HERWIG
  generator systematic (Bierlich et al., arXiv:2203.11601; Bellm et al.,
  arXiv:1512.01178) ship as runnable commands, not afterthoughts.
- **Keep the science honest.** The discretised likelihood is cell-size
  dependent and the model is generator-conditional; the repo must surface these
  (e.g. report the generator-spread systematic) rather than hide them.

---

## 1. Target repository layout

```
rsd-junipr/
├── pyproject.toml                # packaging, deps, entry points, tool config
├── README.md                     # quickstart + pointers
├── LICENSE
├── CITATION.cff                  # how to cite + the physics references
├── .pre-commit-config.yaml       # black, ruff, isort, mypy, nbstripout
├── .github/workflows/ci.yml      # lint + unit + integration + (opt) C++ build
├── docker/
│   ├── Dockerfile.cpp            # ROOT 6.36 + FastJet + fjcontrib LundPlane + PYTHIA
│   └── Dockerfile.train          # CUDA + PyTorch + the Python package
├── configs/                      # plain YAML group files, composed by config.py via OmegaConf (no Hydra)
│   ├── config.yaml               # group selectors + global fields
│   ├── data/{synthetic,rntuple}.yaml
│   ├── model/{ar_junipr_v2,ar_junipr_v1,cinn,diffusion}.yaml
│   ├── encoder/{gru,lundnet,deepsets}.yaml
│   ├── trainer/{default,fast_dev}.yaml   # ddp.yaml added only if multi-node is needed (§6.3)
│   └── experiment/{closure,pythia_vs_herwig}.yaml
├── cpp/                          # data generation (see §5)
│   ├── CMakeLists.txt
│   ├── src/write_lund_rntuple.cpp        # promoted from the script
│   ├── src/lund_io.{hpp,cpp}             # LundSeq, primaryLund, matching
│   ├── apps/pythia_driver.cpp            # PYTHIA 8.3 event source -> writer
│   └── tests/test_lund_io.cpp            # catch2/gtest unit tests
├── src/rsd_junipr/
│   ├── __init__.py
│   ├── config.py                # dataclass schemas + OmegaConf loader (§2)
│   ├── geometry.py               # to_cell, cell_center, ranges (was globals)
│   ├── features.py               # node_features, node_raw, seq_cells
│   ├── distributions.py          # trunc-normal, von Mises, log I0 (was helpers)
│   ├── data/
│   │   ├── rntuple.py            # uproot reader for jets.root:Jets
│   │   ├── synthetic.py          # matched-pair simulator (was _hadronize etc.)
│   │   ├── dataset.py            # MatchedLundDataset, collate
│   │   └── datamodule.py         # splits, loaders, caching, systematics tag
│   ├── encoders/
│   │   ├── base.py              # Encoder ABC
│   │   ├── gru.py               # current bi-GRU encoder
│   │   ├── lundnet.py           # EdgeConv graph encoder (arXiv:2012.08526)
│   │   └── deepsets.py          # permutation-invariant baseline
│   ├── models/
│   │   ├── base.py              # PosteriorModel ABC + registry
│   │   ├── ar_junipr.py         # §5.1 (cell + continuous heads, v1/v2 via cfg)
│   │   ├── cinn.py              # §5.2 conditional normalizing flow
│   │   └── diffusion.py         # §5.3 conditional diffusion / bridge
│   ├── inference/
│   │   ├── point_estimate.py    # LundNode, LundPointEstimate, beam search
│   │   └── sampling.py          # batched ancestral / flow / SDE sampling
│   ├── eval/
│   │   ├── closure.py           # MAP recovers y; samples bracket it
│   │   ├── calibration.py       # SBC, coverage, PIT (arXiv:1804.06788)
│   │   └── systematics.py       # generator-spread comparison
│   ├── train/
│   │   ├── trainer.py           # Trainer class: epoch loop, grad clip, weighted NLL, optional AMP/compile
│   │   ├── logging.py           # lightweight Logger: CSV/JSONL + optional TensorBoard
│   │   ├── checkpoint.py        # save/resume spec (§6)
│   │   └── callbacks.py         # early stop, EMA, LR monitor, sample-on-eval
│   ├── serving/
│   │   ├── export.py            # TorchScript / ONNX of the encoder + heads
│   │   └── api.py               # FastAPI: x -> {MAP, posterior summary}
│   └── cli.py                    # `rsd-junipr {generate,train,eval,export,serve}`
├── tests/                        # pytest mirror of src/
├── notebooks/                    # validation + physics figures (nbstripped)
└── scripts/                      # thin wrappers, sweep.py, SLURM submitters
```

Migration mapping (so nothing is lost):

| Current symbol | New home |
|---|---|
| `to_cell`, `cell_center`, `LN_*`, `N_BINS` | `geometry.py` (+ config) |
| `node_features`, `node_raw`, `seq_cells` | `features.py` |
| `_gauss_logpdf`, `_trunc_normal_logpdf`, `_vonmises_logpdf`, `_log_bessel_i0` | `distributions.py` |
| `_sample_parton_sequence`, `_hadronize`, `synthetic_matched_dataset` | `data/synthetic.py` |
| `load_rntuple` | `data/rntuple.py` |
| `MatchedLundDataset`, `collate` | `data/dataset.py` |
| `ConditionalPrimaryLundJUNIPR` | split into `encoders/gru.py` + `models/ar_junipr.py` |
| `LundNode`, `LundPointEstimate`, `map_decode` | `inference/point_estimate.py` |
| `sample_batch` | `inference/sampling.py` |
| `leading_emission_cell`, `lund_distance`, `truth_tree_str` | `eval/closure.py` |
| `main()` | dissolved into `cli.py` + `train/trainer.py` + `eval/` |

---

## 2. Configuration system (parameter flexibility)

Use **structured (dataclass) schemas backed by OmegaConf, with plain YAML for the
values and a small in-repo loader** (`config.py`, §2.1) — OmegaConf, no Hydra.
OmegaConf turns the dataclasses below into a validated, struct-mode config object
(runtime type checks, unknown-key rejection, interpolation, deep merge); the
loader only has to select the group files and merge them, since Hydra's
defaults-list composition is the one piece replaced by hand. The dataclasses are
the schema and are otherwise unchanged from v2.

```python
# src/rsd_junipr/config.py
@dataclass
class GeometryConfig:                              # OmegaConf stores sequences as lists, not tuples
    ln_invdelta_range: List[float] = field(default_factory=lambda: [0.0, 6.0])
    ln_kt_range:       List[float] = field(default_factory=lambda: [0.0, 6.0])
    n_bins: int = 10                       # -> n_cells = n_bins**2 (derived in code, §2.1)

@dataclass
class EncoderConfig:
    name: str = "gru"                      # gru | lundnet | deepsets
    emb_dim: int = 32
    hidden_dim: int = 64                   # was enc_dim
    num_layers: int = 1                    # << the "encoder depth" knob
    bidirectional: bool = True
    dropout: float = 0.1                   # wired in (the script defines but never applies it)

@dataclass
class ARJuniprConfig:
    ctx_dim: int = 64
    dec_dim: int = 64
    dec_layers: int = 1                    # decoder depth (needs matching h0 init)
    split_head_layers: int = 2
    coord_head_layers: int = 2
    continuous_coords: bool = True         # True == v2, False == v1
    sigma_floor: float = 1e-2
    kappa_max: float = 50.0

@dataclass
class OptimConfig:
    lr: float = 2e-3
    weight_decay: float = 3e-4
    scheduler: str = "cosine"
    eta_min: float = 3e-4
    grad_clip: float = 1.0

@dataclass
class TrainerConfig:
    max_epochs: int = 20
    batch_size: int = 64
    seed: int = 0
    amp: bool = False                      # off by default — model is overhead-bound (§6.1)
    compile: bool = False                  # torch.compile(mode="reduce-overhead") when True
    fast_dev_run: bool = False             # CI smoke path (~2 steps)
    ema_decay: Optional[float] = None      # OmegaConf wants Optional[...] for nullable fields
@dataclass
class DecodeConfig:
    beam_width: int = 8
    topk_cells: int = 6
    max_emissions: int = 25
    n_posterior_samples: int = 500
    cont_temperature: float = 1.0          # exposure-bias remedy, sampling-time only
    min_emissions: int = 1                 # MAP floor: never the unphysical empty tree
    length_penalty: float = 0.0            # GNMT score/len**alpha at final beam rank; 0 == off
    length_floor_quantile: float = 0.0     # learned per-jet MAP floor: max(min_emissions,
    #                                        Q_alpha(P(n|x))); 0.0 == off (opt-in)
    # --- MBR point estimator (opt-in; point_estimator=map reproduces today exactly) ---
    point_estimator: str = "map"           # map | mbr
    mbr_backend: str = "pot"               # pot (default, self-contained) | energyflow | surrogate
    mbr_n_candidates: int = 0              # 0 == all draws are candidates
    mbr_lnkt_cut: Optional[float] = None   # None inherits the geometry ln_kt floor (metric support)
    mbr_weight: str = "kt"                 # kt | z | unit
    mbr_coords: str = "lnDR_lnkt"          # lnDR_lnkt | +lnz | +psi
    mbr_R: float = 8.485                   # imbalance-penalty radius ~ Lund-plane diameter
    mbr_beta: float = 1.0                  # ground-distance exponent (1.0 == KMT EMD)
    mbr_norm: bool = False                 # energyflow weight normalisation; off keeps the imbalance term
    mbr_periodic_phi: bool = False         # wrap the psi column (mbr_coords=+psi)
    mbr_phi_col: int = -1                  # psi column index; -1 == last coordinate
```

OmegaConf schema conventions (assumes `from dataclasses import field`,
`from typing import List, Optional, Any`): type sequence fields as `List[...]` with
`field(default_factory=...)` since OmegaConf coerces tuples to `ListConfig`; use
`Optional[...]` for nullable fields; and a root `Config` aggregates the sub-configs
with the polymorphic `model` field typed `Any = MISSING`, bound to the chosen
family's schema at load (§2.1).

Notes tied to the model discussion:

- `encoder.num_layers` is the **encoder depth** knob; `ar_junipr.dec_layers` the
  decoder depth. The plan's `models/ar_junipr.py` must build `h0` with shape
  `(dec_layers, B, dec_dim)` so deepening the decoder is config-only.
- `geometry.n_bins` drives `n_cells`, the categorical-head width, **and** the
  within-cell truncation bounds of the v2 continuous head — keep these derived,
  never independently set.
- `decode.cont_temperature` is the documented remedy for the over-counted
  multiplicity (exposure bias; Bengio et al., NeurIPS 2015, arXiv:1506.03099) —
  a *sampling-time* knob that never touches the trained likelihood.
- `decode.min_emissions` (default 1) prevents the degenerate empty-tree MAP — the
  brevity bias of an un-normalized argmax over the high-entropy cell head — and
  `decode.length_penalty` (GNMT `score/len**alpha`) is the length-normalization knob.
  Both are *decode-time* only; cINN/diffusion clamp their categorical multiplicity
  head identically. These params are now read from `cfg.decode` end-to-end (eval CLI,
  serving) via the `decode_params()` accessor, which tolerates pre-floor checkpoint
  snapshots; at `eval` the checkpoint's snapshot decode is the default but an explicit
  CLI `decode.*` override still wins (so an A/B like `decode.length_floor_quantile=0.9`
  takes effect on a trained checkpoint). **Reporting** carries the MAP, the posterior
  mean **and the posterior median** (the recommended multiplicity point estimate).
- `decode.length_floor_quantile` (default 0.0 == off) is the *learned, per-jet*
  generalization of `min_emissions`: instead of a hard global constant the MAP length
  is floored at the `alpha`-quantile of the model's own length belief P(n|x) — the
  effective floor is `max(min_emissions, floor(Q_alpha(P(n|x))))`, passed straight into
  the unchanged `map_estimate`. P(n|x) is read from the cINN/diffusion multiplicity
  head exactly and from reused posterior draws for AR (`models/*.length_pmf`,
  `inference/length.py`). The floor only ever *raises* the bound (n>=1 preserved), and
  `alpha=0` short-circuits to today's behavior (structural parity). `alpha->median`
  approaches a length-conditioned MAP at that quantile.
- `decode.point_estimator` (default `map`) selects the point estimate. Beside the MAP
  and the posterior mean/median, `mbr` returns the **minimum-Bayes-risk** tree — the
  drawn tree of least expected perturbative-Lund EMD to the posterior (`inference/mbr.py`),
  reusing the draws already taken. It is **floor-free** (an empty cloud pays the full
  mass-imbalance penalty, so it never wins on a non-empty-dominated posterior — 0% `n=0`
  with `min_emissions=0`, vs the MAP's collapse). The OT solve has **two interchangeable
  backends** (`decode.mbr_backend`): a self-contained POT augmented-cost form (`pot`,
  default) and the reference `energyflow` EMD (Komiske, Metodiev & Thaler, arXiv:1902.02346);
  they agree on the argmin but differ by EnergyFlow's internal `1/R` scale, so quote one
  backend per analysis. Both are lazy-imported (optional `[mbr]` / `[energyflow]` extras),
  so the default `map` path and `per_jet_nll` parity stay dependency-free. The `.risk` is a
  decision-theoretic score, reported separately from the NLL.

CLI override example (ergonomics identical across versions — here OmegaConf's
dotted parsing, not Hydra, implements them):

```bash
rsd-junipr train model=cinn encoder=lundnet \
  encoder.num_layers=3 geometry.n_bins=16 optim.lr=1e-3 trainer.max_epochs=100
```

### 2.1 Config loader — OmegaConf, no Hydra

Values live in plain-YAML group files; a ~60-line `config.py` uses OmegaConf to
turn the dataclass schema into a validated config, compose the selected groups,
apply CLI overrides, and snapshot the result — no Hydra `@main`, `ConfigStore`, or
defaults-list machinery.

```
configs/
├── config.yaml                 # group selectors + global fields
├── geometry/default.yaml
├── model/{ar_junipr_v2,ar_junipr_v1,cinn,diffusion}.yaml
├── encoder/{gru,lundnet,deepsets}.yaml
├── trainer/{default,fast_dev}.yaml
└── experiment/{closure,pythia_vs_herwig}.yaml
```

```yaml
# configs/config.yaml — selectors pick one file per group; global fields merged last
defaults: {geometry: default, model: ar_junipr_v2, encoder: gru, trainer: default}
# any top-level keys here override the merged groups (e.g. trainer.seed: 0)
```

```python
# src/rsd_junipr/config.py  (sketch)
from omegaconf import OmegaConf, DictConfig, MISSING
GROUPS = ("geometry", "model", "encoder", "trainer", "experiment")
MODEL_SCHEMA   = {"ar_junipr_v2": ARJuniprConfig, "ar_junipr_v1": ARJuniprConfig,
                  "cinn": CINNConfig, "diffusion": DiffusionConfig}
ENCODER_SCHEMA = {"gru": EncoderConfig, "lundnet": LundNetEncoderConfig}

def load_config(argv=None) -> DictConfig:
    base = OmegaConf.load(CONFIGS / "config.yaml")             # selectors + globals
    sel, dotlist = split_args(argv, GROUPS)                    # "model=cinn" -> sel; "encoder.num_layers=3" -> dotlist
    selectors = OmegaConf.merge(base.defaults, sel)            # CLI group picks override the base
    cfg = OmegaConf.structured(Config)                         # typed skeleton, struct mode ON
    cfg.model   = OmegaConf.structured(MODEL_SCHEMA[selectors.model])      # bind chosen family schema
    cfg.encoder = OmegaConf.structured(ENCODER_SCHEMA[selectors.encoder])  # bind chosen encoder schema
    for group, name in selectors.items():                      # compose group YAML, validated by merge
        cfg[group] = OmegaConf.merge(cfg[group], OmegaConf.load(CONFIGS / group / f"{name}.yaml"))
    cfg = OmegaConf.merge(cfg, {k: v for k, v in base.items() if k != "defaults"})  # global fields
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))            # value overrides
    OmegaConf.resolve(cfg)                                     # resolve ${...} interpolations
    return cfg
```

Because `cfg` is structured, every `OmegaConf.merge` and `from_dotlist` is
**type-checked against the schema and rejects unknown keys** — a misspelled
`optim.lrr=1e-3` or a `geometry.n_bins=ten` fails at load, not at hour three of
training. That validation is OmegaConf's job here, not hand-rolled (the §6
checkpoint stores `OmegaConf.to_container(cfg, resolve=True)`; provenance is one
line, `OmegaConf.save(cfg, run_dir / "config.yaml")`).

Notes and trade-offs specific to OmegaConf-without-Hydra:

- **Composition is explicit.** Selecting one file per group and merging is the ~10
  lines above; it replaces Hydra's defaults list. Cross-group references come free
  via interpolation, e.g. `run_name: ${model.name}_${encoder.name}`.
- **Derived values.** `n_cells = n_bins**2` and the truncation bounds stay derived
  in the model builder (keeping them locked together), not in YAML. For config-side
  derivation, register a resolver once —
  `OmegaConf.register_new_resolver("sq", lambda x: x * x)` then
  `n_cells: ${sq:${geometry.n_bins}}`.
- **Polymorphic `model`.** Typed `Any = MISSING` in the root `Config` and bound to
  the chosen family's structured schema at load, so each family is still validated.
- **OmegaConf gotchas.** Sequences are `ListConfig` (tuples become lists — type them
  `List[float]`); nullable fields need `Optional[...]`; required fields use `MISSING`.
- **Sweeps.** Hydra's `--multirun` is the one capability genuinely lost; replace it
  with a ~20-line `scripts/sweep.py` that loops `load_config` over an explicit grid
  (submitting to SLURM if needed), referenced in §8/§14.

---

## 3. Model abstraction & registry (model flexibility)

The single contract every family implements:

```python
# src/rsd_junipr/models/base.py
class PosteriorModel(nn.Module, ABC):
    @abstractmethod
    def log_prob(self, batch: dict) -> Tensor: ...        # (B,) log q_phi(y|x)
    @abstractmethod
    def sample(self, xf, nx, n: int) -> list: ...         # posterior draws
    @abstractmethod
    def map_estimate(self, xf, nx) -> "LundPointEstimate": ...

_REGISTRY: dict[str, type[PosteriorModel]] = {}
def register_model(name): ...                              # decorator
def build_model(cfg) -> PosteriorModel:                    # cfg.model.name -> class
    return _REGISTRY[cfg.model.name](cfg)
```

- `weighted_nll` becomes a trainer-side loss = `-(w * model.log_prob(batch)).sum() / w.sum()`,
  identical to today's objective but model-agnostic — conditional MLE whose
  minimiser is the true posterior (arXiv:1605.06376; arXiv:1911.01429).
- **§5.1** `ar_junipr.py`: today's three heads (`P_cont`, `P_split`, continuous
  coords); v1/v2 selected by `continuous_coords`. `map_estimate` = beam search +
  conditional modes (the existing staged MAP).
- **§5.2** `cinn.py`: a conditional flow on a padded/structured latent plus a
  multiplicity head (arXiv:2006.06685; arXiv:2212.08674; jet-substructure
  generative unfolding arXiv:2510.19906). `log_prob` exact via change-of-vars;
  `map_estimate` via latent-zero decode or short optimisation.
- **§5.3** `diffusion.py`: conditional score/bridge model (arXiv:2404.18807);
  `log_prob` by probability-flow ODE, `sample` by reverse SDE,
  `map_estimate` by a posterior-mean/MAP surrogate.

Encoders are independently pluggable behind `Encoder.forward(xf, nx) -> (B, ctx)`:
`gru` (current), `lundnet` (EdgeConv graph over the hadron-level Lund tree;
Dreyer & Qu, *JHEP* **03** (2021) 052, arXiv:2012.08526), `deepsets` (baseline).
Any encoder pairs with any decoder family.

---

## 4. Data pipeline

The contract is fixed by `write_lund_rntuple.cpp`: one entry per jet, with
`event`, `jet_index`, `weight`, jet kinematics, the grooming provenance
`(z_cut, beta, kt_floor, generator)`, and the two **node-unaligned** jagged
sequences `x_*` (hadron) and `y_*` (parton) in `(ln 1/ΔR, ln k_t, ln z, ψ)`.
There is by design no per-node x↔y correspondence, so the objective stays
jet-level (methodology §3; HOMER, Bierlich et al. arXiv:2410.06342; Assi et al.
arXiv:2503.05667).

Stages:

1. **Generate** (`cpp/`, §5) → `jets.root`.
2. **Read** (`data/rntuple.py`): uproot → per-jet dict, unchanged from
   `load_rntuple`, plus the `generator` tag retained for systematics.
3. **Preprocess & cache**: build `(xf, yc, yraw, nx, ny, w)` once and persist
   sharded tensors (`.pt`/`webdataset`) keyed by a **data fingerprint** (hash of
   file + grooming params) so training never re-parses ROOT and runs are
   reproducible.
4. **Split** by `event` id (not by jet) to prevent leakage between jets of the
   same event; deterministic given `seed`.
5. **`LundDataModule`** yields train/val/test loaders and exposes the grooming
   provenance and generator tag downstream.

Keep the `kt_floor` perturbative cut intact end-to-end — the soft/wide-angle
corner yields a near-meaningless posterior (methodology §6; Lifson, Salam &
Soyez, *JHEP* **10** (2020) 170, arXiv:2007.06578).

---

## 5. C++ data-generation module

Promote the script to a built, tested component.

- **Build**: `cpp/CMakeLists.txt` finding ROOT (≥6.36, RNTuple), FastJet, and
  fjcontrib `LundPlane`. Replaces the hand `g++` line in the file header.
- **Refactor**: move `LundSeq`, `primaryLund`, and `getMatchedHadronPartonJets`
  into `lund_io.{hpp,cpp}`; keep `write_lund_rntuple.cpp` as the writer; add
  `apps/pythia_driver.cpp` implementing the documented event source — the
  **pre-hadronization shower partons** read from the PYTHIA 8.3 event record
  (not a final-state dump), with MPI disabled for a pure hadronization study
  (Bierlich et al., arXiv:2203.11601). This fills the `nEvents`/`getEventHadrons`
  hooks the script leaves as stubs.
- **Provenance**: write the `generator` tag per entry (already supported, e.g.
  `"PYTHIA-8.312:tune-Monash"`); add a one-entry `Meta` RNTuple (the cleaner
  pattern the file comment suggests) recording generator, tune, and pipeline
  params for the systematics step.
- **HERWIG**: a parallel `herwig_driver` (cluster model; Bellm et al.,
  arXiv:1512.01178) emitting the identical schema — this is what powers the
  dominant generator systematic in §8.
- **Tests**: `cpp/tests/` (catch2/gtest) on `primaryLund` (Soft Drop boundary
  $z > z_{\rm cut}(\Delta/R_0)^\beta$ and the $\ln k_t$ floor; Larkoski et al.,
  arXiv:1402.2657; Dreyer et al., arXiv:1804.03657) and on the greedy one-to-one
  matching.
- **Containerize**: `docker/Dockerfile.cpp` pins the full ROOT+FastJet+fjcontrib
  +PYTHIA toolchain so generation is reproducible off-cluster.
- **Wire to CLI**: `rsd-junipr generate ...` shells out to the built binary (or
  documents the `cmake --build` + run), writing `jets.root` consumed by stage 4.

---

## 6. Training engine + checkpoint/resume

The compute profile (§6.1) makes the framework choice easy: the §5.1 model is
~0.11M parameters, its full training state is a few MB, and wall-clock is bound by
the jet count and data-feeding overhead, not by arithmetic. So the **default is a
lean custom `Trainer` class — no PyTorch Lightning.** Lightning's heavy machinery
(DDP, FSDP, sharded mixed precision, gradient accumulation) solves costs this
model does not incur, and a ~150-line loop owns the variable-length masking and
weighted NLL more transparently. Retain today's mechanics: AdamW (Loshchilov &
Hutter, ICLR 2019, arXiv:1711.05101), cosine annealing (arXiv:1608.03983),
gradient-norm clipping (Pascanu et al., ICML 2013, arXiv:1211.5063), weighted NLL.
For a small, overhead-bound model the relevant accelerator is
`torch.compile(mode="reduce-overhead")`, **not** mixed precision (AMP buys close
to nothing when you are not tensor-core- or bandwidth-bound); AMP is wired in but
defaults off.

### 6.1 Compute & memory profile (why no framework by default)

- **Parameters** ≈ 110k (encoder ≈47k, decoder ≈39k, three heads ≈26k at the
  default `emb_dim=32`, `enc_dim=dec_dim=ctx_dim=64`, `n_cells=100`) — about
  1000× smaller than BERT-base.
- **Memory**: ~0.45 MB fp32 weights; <2 MB with AdamW moments + gradients;
  activations a few MB at `batch_size=64` (sequences are short — mean multiplicity
  ~5–6, capped at `max_emissions≈25`). Batch sizes in the thousands fit trivially.
- **Compute**: per-jet forward ≈1 MFLOP; an epoch is (jet count) × a few MFLOPs —
  sub-second of pure arithmetic per epoch even at 1M jets. The cost is therefore
  (dataset size × epochs) plus I/O and tiny-op kernel-launch overhead.
- **Order of magnitude**: 1M jets × 20 epochs ≈ a few hours on CPU / ~15–40 min on
  one GPU; at 10M+ the bottleneck is the uproot read + preprocessing — a
  `DataLoader`/caching problem (§4), not a trainer problem.

### 6.2 The Trainer

A framework-free loop driving any `PosteriorModel` through its `log_prob`:

```python
# src/rsd_junipr/train/trainer.py
class Trainer:
    """~150-line training loop for any PosteriorModel. Owns the epoch loop,
    optional AMP/compile, grad clipping, scheduling, checkpointing, and
    lightweight logging. No Lightning."""

    def __init__(self, model, optimizer, scheduler, loaders, cfg, logger, device, run_dir):
        self.model = model.to(device)
        self.opt, self.sched = optimizer, scheduler
        self.train_loader, self.val_loader = loaders
        self.cfg, self.log, self.device, self.run_dir = cfg, logger, device, run_dir
        self.scaler = torch.amp.GradScaler(device.type, enabled=cfg.trainer.amp)
        self.epoch, self.step, self.best_val = 0, 0, float("inf")
        if cfg.trainer.compile:                      # reduce-overhead helps tiny models most
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def fit(self):
        for self.epoch in range(self.epoch, self.cfg.trainer.max_epochs):
            train_nll = self._train_epoch()
            val_nll = self._validate()
            self.sched.step()
            self.log.log(self.step, {"epoch": self.epoch, "train_nll": train_nll,
                                     "val_nll": val_nll, "lr": self.sched.get_last_lr()[0]})
            self.save("last.ckpt")
            if val_nll < self.best_val:
                self.best_val = val_nll
                self.save("best.ckpt")

    def _move(self, b):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def _train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            batch = self._move(batch)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.cfg.trainer.amp):
                nll = -self.model.log_prob(batch)                       # (B,)
                loss = (batch["w"] * nll).sum() / batch["w"].sum().clamp(min=1e-8)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.opt); self.scaler.update()
            self.step += 1; total += loss.item(); n += 1
            if self.cfg.trainer.fast_dev_run and self.step >= 2:        # CI smoke path
                break
        return total / max(n, 1)

    @torch.inference_mode()
    def _validate(self):
        self.model.eval()
        num = den = 0.0
        for batch in self.val_loader:
            batch = self._move(batch)
            nll = -self.model.log_prob(batch)
            num += (batch["w"] * nll).sum().item(); den += batch["w"].sum().item()
        return num / max(den, 1e-8)

    def save(self, name):
        save_checkpoint(self.run_dir / name, model=self.model, optimizer=self.opt,
                        scheduler=self.sched, scaler=self.scaler, epoch=self.epoch,
                        step=self.step, best_val=self.best_val, cfg=self.cfg)

    @classmethod
    def resume(cls, path, loaders, logger, device, run_dir):
        """Rebuild model/opt/sched from the snapshotted config, restore all state
        (incl. RNG, epoch, step, best_val), then continue. config_hash mismatch
        is a hard error (see §6 checkpoint spec)."""
        state = load_checkpoint(path, map_location=device)
        trainer = build_trainer_from_config(state["config"], loaders, logger, device, run_dir)
        trainer._restore(state)                       # load_state_dicts + set_rng_state + counters
        return trainer
```

The weighted NLL is exactly today's objective, now model-agnostic. `fast_dev_run`
gives the CI smoke path (§11). Everything else (AMP, compile) is config-gated and
off by default.

### 6.3 When to reach for Lightning / a managed trainer

Revisit this when the *model*, not the dataset, becomes the cost:

- **§5.3 diffusion / Schrödinger-bridge** — training and especially sampling cost
  scale with the number of denoising steps (many forward passes per jet) and the
  denoiser is typically larger; managed AMP, `torch.compile`, and sharding then
  earn their keep (profiling context: arXiv:2404.18807).
- **LundNet/EdgeConv encoder** (torch-geometric) is heavier than the GRU, though
  usually still single-GPU.
- **Genuine multi-GPU / multi-node** training (DDP or FSDP): here Lightning or
  bare `torchrun` removes real boilerplate; the custom `Trainer` is single-device
  by design.
- **Large hyperparameter sweeps / many concurrent runs** wanting standardized
  orchestration, callbacks, profilers, and hosted dashboards maintained for you.

Because every family sits behind the same `PosteriorModel` contract and the
`Trainer` only ever calls `model.log_prob`, dropping in a `LightningModule` later
is a trainer-layer swap, not a model rewrite.

**Checkpoint specification** (`train/checkpoint.py`) — exact resume requires more
than `state_dict`:

```python
{
  "format_version": 2,
  "model":     {"name": cfg.model.name, "state_dict": ...},
  "config":    OmegaConf.to_container(cfg, resolve=True),   # full run config snapshot
  "optimizer": opt.state_dict(),
  "scheduler": sched.state_dict(),
  "scaler":    scaler.state_dict(),             # AMP grad scaler
  "ema":       ema.state_dict() | None,
  "epoch": int, "global_step": int,
  "best_val_nll": float,
  "rng": {"torch": ..., "torch_cuda": ..., "numpy": ..., "python": ...},
  "git_sha": ..., "config_hash": ..., "data_fingerprint": ...,
}
```

- `save_checkpoint(path, ...)` writes `last.ckpt` every N steps and `best.ckpt`
  on val-NLL improvement; keep top-k.
- `resume(path)` restores **all** of the above, including RNG and dataloader
  position, so a pre-empted run continues bit-for-bit.
- `config_hash` mismatch on resume is a hard error (no silent architecture
  drift); `git_sha` and `data_fingerprint` are recorded for audit.
- Export-only loading (`load_for_inference`) reads `model` + `config` and ignores
  optimiser state.

`rsd-junipr train ... trainer.resume_from=runs/<id>/last.ckpt` resumes;
omitting it starts fresh.

---

## 7. Inference & posterior products

`inference/sampling.py` keeps the batched on-device `sample_batch` (single host
sync) and generalises it: §5.1 ancestral, §5.2 flow inverse, §5.3 reverse-SDE,
all returning the same posterior-draw structure. `inference/point_estimate.py`
keeps `LundNode`/`LundPointEstimate` and beam search. Per jet the repo reports
**both** the MAP/beam estimate and a posterior summary (mean, 68% credible
region, multiplicity distribution) — in high dimensions the mode can be
unrepresentative (methodology §6).

---

## 8. Validation & calibration suite (mandatory)

Each is a CLI subcommand emitting figures + a JSON metrics record:

- **Closure** (`eval closure`): on held-out generator data, MAP recovers the true
  $y$ and posterior draws bracket it (leading-emission Lund distance, multiplicity
  bias — already prototyped).
- **Posterior calibration** (`eval calibration`): coverage, PIT, and
  simulation-based calibration (Talts et al., arXiv:1804.06788). Conditional-
  generator posteriors are not automatically calibrated — the original cINN
  unfolding came out too narrow (arXiv:2006.06685) — so this gates "trustworthy."
- **Generator systematic** (`eval systematics`): train on PYTHIA, evaluate the
  MAP/posterior spread against a HERWIG-trained copy and alternative string tunes
  (arXiv:2203.11601; arXiv:1512.01178). The inter-model spread **is** the
  dominant systematic and must be quoted.

These are run in CI on the synthetic generator (fast) and on real data before any
physics claim.

---

## 9. Experiment tracking & reproducibility

- **Logging (lightweight by default).** A small `Logger` protocol
  (`log(step, metrics)`, `log_artifact(path)`) with a **dependency-free default**:
  a `CSVLogger`/`JSONLLogger` that appends `metrics.csv`/`metrics.jsonl` to the run
  dir, plus an optional TensorBoard `SummaryWriter` for curves and sample-tree
  images. No account, no service, no network. Hosted backends (Weights & Biases,
  MLflow) are **optional** implementations of the same protocol, enabled via the
  `[track]` extra (§10) only if a team wants shared dashboards — the `Trainer`
  never imports them.
- **Run dir**: `runs/<timestamp>-<cfg-hash>/` holding the resolved config (written
  by the §2.1 loader via `OmegaConf.save(cfg, ...)`), checkpoints,
  `metrics.{csv,jsonl}`, TensorBoard events, and eval artifacts.
- **Determinism**: global seeding of torch/numpy/python, `cudnn.deterministic=True`
  / `benchmark=False`, with residual CUDA-atomic non-determinism documented.
- **Data versioning**: DVC (or a manifest of fingerprints) so `jets.root` ↔ run
  linkage is explicit.

---

## 10. Packaging & environments

- `pyproject.toml` (PEP 621), `src/` layout, console entry point
  `rsd-junipr = rsd_junipr.cli:main`.
- Pinned `requirements.lock`. Core deps stay minimal (torch, numpy, uproot,
  omegaconf — structured-config validation via OmegaConf, no Hydra). Optional
  extras: `[lundnet]` (torch-geometric), `[track]` (tensorboard; wandb/mlflow),
  `[serve]` (fastapi, onnxruntime), `[dev]`. OmegaConf bundles its own YAML
  support, so no separate `pyyaml` pin is needed.
- Two Docker images (C++ generation; CUDA training/serving). PyTorch 2.x,
  Python ≥3.10.

---

## 11. Testing & CI/CD

- **Unit**: cell round-trip (`to_cell`/`cell_center`); densities integrate to 1
  (truncated-normal normaliser, von Mises, the A&S `log I0` branches);
  `log_prob` finite and shaped `(B,)` for every registered model; checkpoint
  save→resume round-trip restores state exactly; collate padding/masking.
- **Integration**: `trainer.fast_dev_run=true` runs the `Trainer` for ~2 steps on
  synthetic data for each `model ∈ {ar_junipr_v1, ar_junipr_v2, cinn}` and asserts
  the loss is finite/decreasing and a posterior draw is valid.
- **C++**: build + run `cpp/tests` (Soft Drop boundary, matching).
- **CI** (`.github/workflows/ci.yml`): ruff + black + mypy, then pytest, then the
  fast integration train; C++ build behind a job using the C++ image.
- **Pre-commit**: formatting, lint, notebook stripping.

---

## 12. Serving / export / deployment

Grounded in the deployment skill's rules:

- **Caveat first**: autoregressive sampling and beam search contain Python
  control flow, so `torch.jit.trace` is wrong (it captures one branch). Export the
  **encoder + per-step heads** via `torch.jit.script` (or ONNX with `dynamic_axes`
  on sequence length) and keep beam search / sampling as a thin Python/LibTorch
  loop around the scripted step. Always `model.eval()` + `torch.no_grad()` and
  verify with `torch.allclose` against the eager model.
- **`serving/api.py`**: FastAPI service taking a hadron-level Lund sequence and
  returning the MAP tree plus a posterior summary; load via `load_for_inference`.
- **Optional**: ONNX Runtime for the encoder, int8 quantisation for CPU serving
  if throughput demands it.

---

## 13. Documentation

- `mkdocs` site: quickstart, the data contract, config reference (auto-rendered
  from the dataclass schemas), the model-family guide (§5.1/5.2/5.3), and the
  mandatory validation playbook (§8).
- `notebooks/`: closure and calibration figures, a per-jet posterior walkthrough,
  the generator-systematic comparison — nbstripped in git, rendered in docs.
- `CITATION.cff` carrying the physics + method references so downstream users
  cite correctly.

---

## 14. Phased roadmap

| Phase | Goal | Key deliverables | Exit criterion |
|---|---|---|---|
| **0. Scaffold** | Package skeleton | `pyproject.toml`, `src/` layout, CI lint, config schemas + OmegaConf loader (§2.1) | `pip install -e .`; `rsd-junipr --help` |
| **1. Refactor** | Script → modules, no behaviour change | geometry/features/distributions/data/encoders/models split; `ar_junipr` v1+v2 behind the registry | bit-comparable NLL to the script on synthetic data |
| **2. Trainer** | Robust training + resume | custom `Trainer` (§6.2), checkpoint spec (§6), CSV/TensorBoard logging | pre-empt/resume reproduces the loss curve |
| **3. Data** | C++ promoted + real data | CMake build, PYTHIA + HERWIG drivers, cached datamodule, C++ tests | `generate`→`train` on real `jets.root` end-to-end |
| **4. Validation** | Trust the output | closure / SBC / systematics commands + figures | calibrated coverage; quoted generator spread |
| **5. New models** | §5.2 / §5.3 | `cinn.py`, `diffusion.py` as registry drop-ins | both pass the integration train + closure |
| **6. Serve** | Deployable | scripted export, FastAPI, docs site | `allclose` parity; live `predict` endpoint |

Phases 1–2 are the highest-leverage and unblock everything; phases 5–6 are
additive once the contract and trainer are stable.

---

## 15. References

**Posterior / method.** Cranmer, Brehmer & Louppe, *PNAS* **117** (2020) 30055,
arXiv:1911.01429 · Papamakarios & Murray, *NeurIPS* (2016), arXiv:1605.06376 ·
Greenberg, Nonnenmacher & Macke, *ICML* (2019), arXiv:1905.07488 · Talts et al.,
*SBC*, arXiv:1804.06788.

**Models.** JUNIPR, Andreassen et al., *EPJC* **79** (2019) 102, arXiv:1804.09720;
binary JUNIPR, *PRL* **123** (2019) 182001, arXiv:1906.10137 · cINN unfolding,
Bellagente et al., *SciPost Phys.* **9** (2020) 074, arXiv:2006.06685; iterative
variant, Backes et al., *SciPost Phys. Core* **7** (2024) 007, arXiv:2212.08674 ·
LundNet, Dreyer & Qu, *JHEP* **03** (2021) 052, arXiv:2012.08526 · unfolding
landscape, arXiv:2404.18807; generative unfolding of jets, arXiv:2510.19906.

**Physics pipeline.** anti-$k_t$, Cacciari, Salam & Soyez, *JHEP* **04** (2008) 063,
arXiv:0802.1189; FastJet, *EPJC* **72** (2012) 1896, arXiv:1111.6097 · C/A,
Dokshitzer et al., *JHEP* **08** (1997) 001, arXiv:hep-ph/9707323 · Soft Drop,
Larkoski et al., *JHEP* **05** (2014) 146, arXiv:1402.2657 · Recursive Soft Drop,
Dreyer et al., *JHEP* **06** (2018) 093, arXiv:1804.03657 · Lund plane, Dreyer,
Salam & Soyez, *JHEP* **12** (2018) 064, arXiv:1807.04758 · Lund density, Lifson,
Salam & Soyez, *JHEP* **10** (2020) 170, arXiv:2007.06578.

**Generators / hadronization.** PYTHIA 8.3, Bierlich et al., *SciPost Phys.
Codebases* (2022), arXiv:2203.11601 · HERWIG 7, Bellm et al., *EPJC* **76** (2016)
196, arXiv:1512.01178 · string fragmentation, Andersson et al., *Phys. Rept.*
**97** (1983) 31 · HOMER, Bierlich et al., arXiv:2410.06342; Assi et al.,
arXiv:2503.05667.

**Training components.** AdamW, Loshchilov & Hutter, ICLR 2019, arXiv:1711.05101 ·
SGDR/cosine, arXiv:1608.03983 · gradient clipping, Pascanu et al., ICML 2013,
arXiv:1211.5063 · exposure bias / scheduled sampling, Bengio et al., NeurIPS 2015,
arXiv:1506.03099.
