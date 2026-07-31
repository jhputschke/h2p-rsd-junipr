"""Configuration system (§2): dataclass schemas backed by OmegaConf, no Hydra.

OmegaConf turns the dataclasses below into a validated, struct-mode config object
(runtime type checks, unknown-key rejection, interpolation, deep merge). The
loader (`load_config`) only selects the group files and merges them; Hydra's
defaults-list composition is the one piece replaced by hand (~60 lines).

Every physics/architecture knob from `conditional_rsd_junipr_v2.py` lives here as
a schema field, never hard-coded in source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf

# Repo-root-relative config directory (src/h2p_rsd_junipr/config.py -> ../../configs)
CONFIGS = Path(__file__).resolve().parents[2] / "configs"

GROUPS = ("geometry", "data", "model", "encoder", "optim", "trainer", "decode", "experiment")


# ---------------------------------------------------------------------------
# Sub-config schemas
# ---------------------------------------------------------------------------
@dataclass
class GeometryConfig:
    # OmegaConf stores sequences as ListConfig, so type ranges as List[float].
    ln_invdelta_range: list[float] = field(default_factory=lambda: [0.0, 6.0])
    ln_kt_range: list[float] = field(default_factory=lambda: [0.0, 6.0])
    n_bins: int = 10  # -> n_cells = n_bins**2 (derived in geometry.py, §2.1)


@dataclass
class DataConfig:
    source: str = "synthetic"          # synthetic | rntuple
    path: str = "jets.root"            # rntuple path (source == rntuple)
    ntuple: str = "Jets"
    n_jets: int = 8000                 # synthetic dataset size
    seed: int = 0                      # synthetic / split seed
    val_fraction: float = 0.1
    min_val: int = 200
    cache_dir: str | None = None    # preprocessed-tensor cache (§4 stage 3)
    max_emissions: int = 20            # synthetic parton sequence cap
    # --- jet-pT window applied to the loaded jets (docs/PLAN_jet_xsection.md §2) ---
    # Both bounds `null` == off, and off is the byte-identical path: no jet is dropped
    # and the data fingerprint is unchanged. Half-open, `pt_min <= pt < pt_max`, so
    # adjacent windows tile a sample without double-counting a jet.
    pt_var: str = "jet_pt"             # jet_pt (ungroomed) | x_ptg (groomed); aliases in
    #                                    datamodule.PT_SELECT_VARS
    pt_min: float | None = None        # GeV, inclusive lower edge; null == unbounded
    pt_max: float | None = None        # GeV, EXCLUSIVE upper edge; null == unbounded


# Aux conditioning (docs/PLAN_Input.md): per-jet GROOMED scalars broadcast onto every
# node of xf, so the encoder sees what the primary-only sequence cannot represent.
# `[]` (the default) is the byte-identical off path — same state_dict, same log_prob.
# Registered names live in features.AUX_FEATURES; read them via
# `features.configured_aux_names`, never `cfg.encoder.aux_features` directly, so old
# checkpoint snapshots without the field keep loading.
#   CLI: encoder.aux_features='[ln_mg_pt,nsec,ln_pt]'
_AUX_DOC = "groomed per-jet conditioning scalars; [] == off (docs/PLAN_Input.md)"


@dataclass
class EncoderConfig:
    name: str = "gru"                  # gru | lundnet | deepsets
    emb_dim: int = 32
    hidden_dim: int = 64               # was enc_dim
    num_layers: int = 1                # the "encoder depth" knob
    bidirectional: bool = True
    dropout: float = 0.1               # wired in (the script defines but never applies it)
    aux_features: list[str] = field(default_factory=list)  # see _AUX_DOC


@dataclass
class LundNetEncoderConfig:
    name: str = "lundnet"
    emb_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    k: int = 4                         # EdgeConv neighbourhood (chain graph -> sequential)
    dropout: float = 0.1
    aux_features: list[str] = field(default_factory=list)  # see _AUX_DOC


@dataclass
class DeepSetsEncoderConfig:
    name: str = "deepsets"
    emb_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    aux_features: list[str] = field(default_factory=list)  # see _AUX_DOC


@dataclass
class ARJuniprConfig:
    name: str = "ar_junipr"
    ctx_dim: int = 64
    dec_dim: int = 64
    dec_layers: int = 1                # decoder depth (h0 built (dec_layers, B, dec_dim))
    split_head_layers: int = 2
    coord_head_layers: int = 2
    continuous_coords: bool = True     # True == v2, False == v1
    sigma_floor: float = 1e-2
    kappa_max: float = 50.0
    cell_label_smoothing: float = 0.0  # split-head label smoothing; 0.0 == off (likelihood
    #                                    parity preserved). Probe knob for the MAP collapse.
    use_multiplicity_head: bool = False  # False == today's implicit continue/stop length model
    #                                      (bit-parity preserved; no n_head built). True promotes
    #                                      length to a first-class categorical head realizing
    #                                      q(y|x) = q(N|x) q(y|N,x) (docs/PLAN_MultHead.md).
    max_emissions: int = 25            # categorical size of the multiplicity head (n = 0..max);
    #                                    only used when use_multiplicity_head=True. Mirrors CINN.
    use_cross_attention: bool = False  # decoder attends to the encoder's per-node hadron
    #                                    states instead of only the pooled e(x) (WP3). Off keeps
    #                                    the module list and state_dict byte-identical; requires
    #                                    an encoder with returns_sequence=True.
    xattn_heads: int = 4               # attention heads; must divide dec_dim


@dataclass
class CINNConfig:
    name: str = "cinn"
    ctx_dim: int = 64
    n_blocks: int = 6
    hidden_dim: int = 64
    max_emissions: int = 25
    sigma_floor: float = 1e-2
    kappa_max: float = 50.0


@dataclass
class DiffusionConfig:
    name: str = "diffusion"
    ctx_dim: int = 64
    hidden_dim: int = 64
    n_steps: int = 50
    max_emissions: int = 25


@dataclass
class CFMConfig:
    """Conditional flow matching with an EXACT probability-flow-ODE likelihood
    (docs/PLAN_UPDATES.md WP1). Unlike `diffusion`, `log_prob` here is a normalized
    density, so its NLL is comparable with `cinn` / `ar_junipr_*`."""

    name: str = "cfm"
    ctx_dim: int = 64
    hidden_dim: int = 64
    n_ode_steps: int = 32          # likelihood + sampling ODE steps
    ode_solver: str = "rk4"        # rk4 (4 field evals/step) | heun (2, ~2x faster)
    max_emissions: int = 25        # multiplicity-head support, mirrors CINN
    time_features: int = 16        # Fourier features for t
    sigma_min: float = 1e-3        # OT-path terminal width (Lipman Eq. 20)
    cfm_map: str = "ode_mode"      # MAP coordinates: ode_mode (push the base mode) | ascent


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
    amp: bool = False                  # off by default — model is overhead-bound (§6.1)
    compile: bool = False              # torch.compile(mode="reduce-overhead") when True
    fast_dev_run: bool = False         # CI smoke path (~2 steps)
    ema_decay: float | None = None  # OmegaConf wants Optional[...] for nullable fields
    num_workers: int = 0
    resume_from: str | None = None
    deterministic: bool = True


@dataclass
class DecodeConfig:
    beam_width: int = 8
    topk_cells: int = 6
    max_emissions: int = 25
    n_posterior_samples: int = 500
    cont_temperature: float = 1.0      # exposure-bias remedy, sampling-time only
    min_emissions: int = 1             # MAP floor: the point estimate never collapses
    #                                    to the unphysical empty tree (>=1 splitting)
    length_penalty: float = 0.0        # GNMT-style score/len**alpha at final beam rank;
    #                                    0.0 == no normalization (default = today's behavior)
    length_floor_quantile: float = 0.0 # per-jet MAP floor from the learned P(n|x): the
    #                                    effective floor is max(min_emissions, Q_alpha(P(n|x))).
    #                                    0.0 == off (short-circuits; merged behavior unchanged)
    # --- MBR point estimator (docs/PLAN_MBR_PerturbativeLund.md). All default so
    #     point_estimator="map" reproduces today exactly and imports no OT backend.
    point_estimator: str = "map"        # map | mbr
    mbr_backend: str = "pot"            # pot (default, self-contained) | energyflow | surrogate
    mbr_n_candidates: int = 0           # 0 => every draw is a candidate (full O(K^2) MBR)
    mbr_lnkt_cut: float | None = None   # None => inherit the geometry ln_kt floor (metric support)
    mbr_weight: str = "kt"             # kt | z | unit — Lund-cloud point weights
    mbr_coords: str = "lnDR_lnkt"      # lnDR_lnkt | +lnz | +psi — ground-metric columns (gdim follows)
    mbr_R: float = 8.485               # imbalance-penalty radius ~ Lund-plane diameter (default geometry)
    mbr_beta: float = 1.0              # ground-distance exponent (1.0 == KMT 1-Wasserstein EMD)
    mbr_norm: bool = False             # energyflow weight normalisation; off keeps the imbalance term
    mbr_periodic_phi: bool = False     # wrap the psi column (only when mbr_coords="+psi")
    mbr_phi_col: int = -1              # psi column index; -1 => last coordinate
    mbr_resample_to_qn: bool = False   # reweight the MBR support to the calibrated q(N|x) marginal
    #                                    (decode-layer exposure-bias fix; off keeps the plain mean risk)


@dataclass
class ExperimentConfig:
    name: str = "default"
    closure_jets: int = 300
    n_closure_samples: int = 200
    generator_b: str | None = None  # second generator for the systematic (§8)
    # --- calibration suite v2 (docs/PLAN_UPDATES.md WP2). All default off, so the
    #     reported metric dict is bit-identical to the pre-WP2 suite until opted in.
    pit_coords: bool = False        # per-coordinate PITs (exact conditional CDFs)
    stratify_regions: bool = False  # every metric also binned by leading-emission quadrant
    tarp: bool = False              # TARP expected-coverage curve on tree-valued posteriors
    tarp_refs: int = 100            # size of the TARP reference pool
    tarp_reference: str = "pooled"  # pooled (posterior draws of other jets) | prior (truth trees)
    closure_continuous: bool = False  # leading-emission distances OFF the cell grid, via
    #                                   sample_coordinates (the cell metric is quantisation-
    #                                   limited); costs closure_jets * n_closure_samples passes


@dataclass
class Config:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: Any = MISSING               # polymorphic; bound to the chosen family at load
    encoder: Any = MISSING             # polymorphic; bound to the chosen encoder at load
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    run_name: str = "${model.name}_${encoder.name}"
    run_root: str = "runs"


# Polymorphic group -> schema binding (§2.1)
MODEL_SCHEMA = {
    "ar_junipr_v2": ARJuniprConfig,
    "ar_junipr_v1": ARJuniprConfig,
    "ar_junipr_v3": ARJuniprConfig,   # v2 backbone + first-class multiplicity head
    "ar_junipr_v4": ARJuniprConfig,   # v3 backbone + decoder cross-attention over x
    "cinn": CINNConfig,
    "diffusion": DiffusionConfig,
    "cfm": CFMConfig,                 # exact probability-flow-ODE likelihood
}
ENCODER_SCHEMA = {
    "gru": EncoderConfig,
    "lundnet": LundNetEncoderConfig,
    "deepsets": DeepSetsEncoderConfig,
}


# ---------------------------------------------------------------------------
# Loader (§2.1) — OmegaConf, no Hydra
# ---------------------------------------------------------------------------
def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("sq"):
        OmegaConf.register_new_resolver("sq", lambda x: x * x)


def _pop_base(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Split out an optional `base=<path>` token — a custom top-level config file
    (own `defaults:` block + globals). Last one wins; everything else passes through."""
    base_path: Path | None = None
    rest: list[str] = []
    for tok in argv:
        if tok.startswith("base="):
            base_path = Path(tok.split("=", 1)[1]).expanduser().resolve()
        else:
            rest.append(tok)
    return base_path, rest


def _group_file(roots: tuple[Path, ...], group: str, name: str) -> Path:
    """First `<root>/<group>/<name>.yaml` that exists, custom roots before `configs/`.

    A missing file is a hard error: silently falling back to the schema defaults turns
    e.g. `model=ar_junipr_v3` into a plain v2 with no warning."""
    for root in roots:
        path = root / group / f"{name}.yaml"
        if path.exists():
            return path
    avail = sorted({p.stem for root in roots for p in (root / group).glob("*.yaml")})
    raise FileNotFoundError(
        f"no {group} config {name!r} in {[str(r / group) for r in roots]}"
        + (f"; available: {avail}" if avail else "")
    )


def _split_args(argv: list[str], groups: tuple[str, ...]):
    """Partition CLI tokens into group selectors (group=name) and dotted value
    overrides (a.b.c=value)."""
    selectors: dict[str, str] = {}
    dotlist: list[str] = []
    for tok in argv:
        if "=" not in tok:
            raise ValueError(f"override must be key=value: {tok!r}")
        key, val = tok.split("=", 1)
        if key in groups:
            selectors[key] = val
        else:
            dotlist.append(tok)
    return selectors, dotlist


def load_config(argv: list[str] | None = None) -> DictConfig:
    """Compose the validated, struct-mode run config: base selectors + globals,
    CLI group picks, per-group YAML, then dotted value overrides. Every merge is
    type-checked against the schema and rejects unknown keys.

    `base=<path>` layers a custom top-level config file over `configs/config.yaml`
    (it need only list what it changes) and adds its own directory as a group-file
    root, searched before `configs/` — so a custom tree can shadow `<group>/<name>.yaml`
    while inheriting every file it does not ship."""
    _register_resolvers()
    argv = list(argv or [])
    base_path, argv = _pop_base(argv)
    base = OmegaConf.load(CONFIGS / "config.yaml")          # selectors + globals
    roots: tuple[Path, ...] = (CONFIGS,)
    if base_path is not None:
        if not base_path.is_file():
            raise FileNotFoundError(f"base config not found: {base_path}")
        base = OmegaConf.merge(base, OmegaConf.load(base_path))
        roots = tuple(dict.fromkeys((base_path.parent, CONFIGS)))
    selectors_cli, dotlist = _split_args(argv, GROUPS)
    selectors = OmegaConf.merge(base.defaults, selectors_cli)   # CLI picks override base

    cfg = OmegaConf.structured(Config)                      # typed skeleton, struct mode ON
    cfg.model = OmegaConf.structured(MODEL_SCHEMA[selectors.model])
    cfg.encoder = OmegaConf.structured(ENCODER_SCHEMA[selectors.encoder])

    for group in GROUPS:
        name = selectors.get(group)
        if name is None:
            continue
        cfg[group] = OmegaConf.merge(cfg[group], OmegaConf.load(_group_file(roots, group, name)))

    # global top-level fields (everything in base except the defaults block)
    globals_ = {k: v for k, v in OmegaConf.to_container(base).items() if k != "defaults"}
    if globals_:
        cfg = OmegaConf.merge(cfg, globals_)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    # record which model/encoder variant was selected (v1 vs v2 etc.)
    cfg.model.name = selectors.model
    OmegaConf.resolve(cfg)
    return cfg


def explicit_group_keys(argv: list[str] | None, group: str) -> set[str]:
    """Which `<group>` fields THIS invocation named explicitly.

    `load_config` cannot answer that. It always returns a fully populated group, seeded
    from `configs/config.yaml`'s defaults, so "the user asked for `beam_width=16`" and
    "nobody said anything and the default is 8" come back identical. `eval` needs the
    distinction because its baseline is the CHECKPOINT's snapshot, not the repo default:
    a group nobody named must stay exactly as the checkpoint left it, or every eval would
    silently re-decode a trained model with `configs/decode/default.yaml`.

    Explicit means, in the same precedence order `load_config` merges them: every field of
    the group file picked by `group=name` (or by `defaults.<group>` inside a `base=` file),
    the keys of an inline `<group>:` block in that base file, and the dotted
    `group.field=value` CLI tokens.

    Only the NAMES are returned. Read the values off the composed config, so
    interpolations (`mbr_lnkt_cut: ${geometry.ln_kt_range[0]}`) are already resolved —
    a group file loaded on its own cannot resolve a cross-group reference."""
    argv = list(argv or [])
    base_path, rest = _pop_base(argv)
    roots: tuple[Path, ...] = (CONFIGS,)
    keys: set[str] = set()
    selector: str | None = None

    if base_path is not None and base_path.is_file():
        base = OmegaConf.load(base_path)
        roots = tuple(dict.fromkeys((base_path.parent, CONFIGS)))
        sel = OmegaConf.select(base, f"defaults.{group}")
        if sel is not None:
            selector = str(sel)
        block = OmegaConf.select(base, group)
        if block is not None:
            keys |= {str(k) for k in block.keys()}

    for tok in rest:
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        if key == group:
            selector = val                       # a later selector wins, as in load_config
        elif key.startswith(f"{group}."):
            keys.add(key[len(group) + 1:].split(".", 1)[0])

    if selector is not None:
        keys |= {str(k) for k in OmegaConf.load(_group_file(roots, group, selector)).keys()}
    return keys


def config_hash(cfg: DictConfig) -> str:
    """Stable hash of the resolved config — used for run-dir naming and the
    checkpoint config_hash guard (§6)."""
    payload = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def save_config(cfg: DictConfig, path: Path) -> None:
    OmegaConf.save(cfg, path)


def to_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


# Decode defaults, kept in sync with DecodeConfig — the single source of truth for
# tolerant reads of cfg.decode (see decode_params).
_DECODE_DEFAULTS: dict = {
    "beam_width": 8,
    "topk_cells": 6,
    "max_emissions": 25,
    "n_posterior_samples": 500,
    "cont_temperature": 1.0,
    "min_emissions": 1,
    "length_penalty": 0.0,
    "length_floor_quantile": 0.0,
    "point_estimator": "map",
    "mbr_backend": "pot",
    "mbr_n_candidates": 0,
    "mbr_lnkt_cut": None,
    "mbr_weight": "kt",
    "mbr_coords": "lnDR_lnkt",
    "mbr_R": 8.485,
    "mbr_beta": 1.0,
    "mbr_norm": False,
    "mbr_periodic_phi": False,
    "mbr_phi_col": -1,
    "mbr_resample_to_qn": False,
}


# Experiment defaults, kept in sync with ExperimentConfig — same contract as
# _DECODE_DEFAULTS, for the calibration-suite switches added by WP2.
_EXPERIMENT_DEFAULTS: dict = {
    "name": "default",
    "closure_jets": 300,
    "n_closure_samples": 200,
    "generator_b": None,
    "pit_coords": False,
    "stratify_regions": False,
    "tarp": False,
    "tarp_refs": 100,
    "tarp_reference": "pooled",
    "closure_continuous": False,
}


def _tolerant_group(cfg, group: str, defaults: dict) -> dict:
    """Resolved group kwargs, tolerant of OLD checkpoint snapshots whose `<group>`
    block predates newer fields. `OmegaConf.select` returns None for an absent node
    even under struct mode, so a missing key falls back to the schema default."""
    out = dict(defaults)
    node = OmegaConf.select(cfg, group)
    if node is not None:
        for k in out:
            v = OmegaConf.select(node, k)
            if v is not None:
                out[k] = v
    return out


def experiment_params(cfg) -> dict:
    """Resolved `experiment` kwargs (see `_tolerant_group`). This is the ONLY
    supported way to read the calibration-suite switches, so a pre-WP2 checkpoint
    snapshot evaluates with them off rather than crashing."""
    return _tolerant_group(cfg, "experiment", _EXPERIMENT_DEFAULTS)


def decode_params(cfg) -> dict:
    """Resolved decode kwargs, tolerant of OLD checkpoint snapshots whose `decode`
    block predates newer fields (e.g. min_emissions/length_penalty).

    `OmegaConf.select` returns None for an absent node even under struct mode, so a
    missing key falls back to the DecodeConfig default rather than raising. This is the
    ONLY supported way to read decode params — never access `cfg.decode.<newfield>`
    directly, or loading a pre-change checkpoint will crash."""
    out = dict(_DECODE_DEFAULTS)
    dec = OmegaConf.select(cfg, "decode")
    if dec is not None:
        for k in out:
            v = OmegaConf.select(dec, k)
            if v is not None:
                out[k] = v
    return out
