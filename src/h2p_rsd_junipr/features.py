"""Per-node feature builders (was `node_features`, `node_raw`, `seq_cells`).

Geometry-free continuous features; cell discretisation lives on `Geometry`.

Also home to the AUX conditioning registry (docs/PLAN_Input.md): per-jet **groomed**
scalars the primary-only hadron sequence `x` structurally cannot represent — the
pipeline-groomed jet mass, the all-branch-minus-primary splitting count, and the jet
scale — written by the C++ stage and broadcast onto every node of `xf`. All three are
groomed, so they keep the NP/UE suppression that motivated Soft Drop; ungroomed
observables are deliberately out of scope.
"""

from __future__ import annotations

import math

import numpy as np

# encoder/decoder INPUT features: (ln 1/DeltaR, ln kt, ln z, sin psi, cos psi)
N_NODE_FEAT = 5

# ---------------------------------------------------------------------------
# Aux conditioning features (docs/PLAN_Input.md)
# ---------------------------------------------------------------------------
# Standardization is FIXED, never data-dependent: a checkpoint must mean the same
# thing on a new sample, and a fitted normalizer would silently re-scale the input
# when the jet spectrum changes.
MG_EPS = 1e-3   # GeV; a single-prong groomed jet has m_g == 0 exactly -> log floor
PT_REF = 100.0  # GeV; ln(pt/PT_REF) centres the scale feature on the card's pTHatMin
ETA_REF = 2.0   # the standard |y| acceptance; a FIXED constant, never read from data,
#                 so |eta|/ETA_REF means the same thing across samples and checkpoints

# Source RNTuple columns each aux feature reads. Used to fail loud, early, with a
# message naming the missing column rather than a NaN surfacing 200 epochs later.
AUX_SOURCES: dict[str, tuple[str, ...]] = {
    "ln_mg_pt": ("x_mg", "jet_pt"),
    "nsec": ("x_nsec",),
    "ln_pt": ("jet_pt",),
    # --- groomed momentum, the in-scope partner of ln_mg_pt ---
    "ln_ptg_pt": ("x_ptg", "jet_pt"),
    # --- jet-level context ---
    "abs_eta": ("jet_eta",),
    # --- secondary-plane KINEMATICS (gate on has_sec; see AUX_FEATURES) ---
    "has_sec": ("x_nsec",),
    "ln_kt_sec": ("x_kt_sec_max", "x_nsec"),
    "ln_kt_sec_sum": ("x_kt_sec_sum", "x_nsec"),
    "sec_depth": ("x_sec_attach", "x_nsec"),
}


def _jet_pt(jet: dict) -> float:
    pt = _finite(jet, "jet_pt")
    if pt <= 0.0:
        raise ValueError(f"aux feature needs jet_pt > 0, got {pt!r}")
    return pt


def _mg(jet: dict) -> float:
    mg = _finite(jet, "x_mg")
    if mg < 0.0:
        raise ValueError(f"aux feature needs x_mg >= 0, got {mg!r}")
    return mg


def _nsec(jet: dict) -> float:
    nsec = _finite(jet, "x_nsec")
    if nsec < 0.0:  # -1 is the reader's "column absent" sentinel
        raise ValueError(
            f"aux feature needs x_nsec >= 0, got {nsec!r} (-1 is the sentinel for a "
            "jets.root written before the aux columns existed)"
        )
    return nsec


def _ptg(jet: dict) -> float:
    ptg = _finite(jet, "x_ptg")
    if ptg <= 0.0:
        raise ValueError(f"aux feature needs x_ptg > 0, got {ptg!r}")
    return ptg


def _sec(jet: dict, name: str) -> float:
    """A secondary-plane quantity, validated and gated on `x_nsec`.

    When `x_nsec == 0` the C++ side writes 0 and the value is UNDEFINED, not measured —
    return 0 so the log1p transform maps it to exactly 0 and `has_sec` carries the
    distinction. When there IS a secondary the value must be a real non-negative number;
    a sentinel or a missing column raises, as everywhere else."""
    if _nsec(jet) == 0:
        return 0.0
    value = _finite(jet, name)
    if value < 0.0:  # -1 is the reader's "column absent" sentinel
        raise ValueError(
            f"aux feature needs {name} >= 0, got {value!r} (-1 is the sentinel for a "
            "jets.root written before the aux columns existed)"
        )
    return value


def _finite(jet: dict, name: str) -> float:
    if name not in jet:
        raise ValueError(
            f"aux feature source column {name!r} missing from the jet record; "
            "re-write jets.root with the current cpp/ writer (docs/PLAN_Input.md)"
        )
    value = float(jet[name])
    if not math.isfinite(value):
        raise ValueError(f"aux feature source column {name!r} is not finite ({value!r})")
    return value


AUX_FEATURES = {
    # ln(m_g / pt): the dimensionless groomed-mass observable, Lund-plane natural and
    # the one quantity every primary node being massless throws away.
    "ln_mg_pt": lambda j: math.log(max(_mg(j), MG_EPS) / _jet_pt(j)),
    # log1p of the secondary-plane splitting count: density on non-primary branches
    # tracks the emitting prong's Casimir (Dreyer, Soyez & Takacs, arXiv:2112.09140).
    "nsec": lambda j: math.log1p(_nsec(j)),
    # the scale anchor; already written per jet, never previously read.
    "ln_pt": lambda j: math.log(_jet_pt(j) / PT_REF),
    # ln(pt_g / pt): how much MOMENTUM grooming removed. The in-scope partner of
    # ln_mg_pt, and deliberately not the mass-drop ratio ln(m_g/m): combined with
    # ln_mg_pt that would be an invertible reparameterization giving the encoder
    # ln(m/pt) -- the UNGROOMED mass, which grooming-first design excludes. With
    # ln_pt this instead yields ln(pt_g), a groomed quantity. Negative, ~0 when
    # grooming removed nothing.
    "ln_ptg_pt": lambda j: math.log(min(_ptg(j) / _jet_pt(j), 1.0)),
    # |eta| / 2 (the standard acceptance). At fixed pt the quark/gluon fraction varies
    # strongly with rapidity (valence PDFs -> forward jets are quark-enriched), and the
    # posterior over y is implicitly a flavour mixture. NOTE this is a PRIOR handle, not
    # a measurement one: it works by telling the model which mixture it is in, so it
    # carries more generator-composition dependence than the groomed observables.
    "abs_eta": lambda j: abs(_finite(j, "jet_eta")) / ETA_REF,
    # --- secondary-plane kinematics -------------------------------------------------
    # These are UNDEFINED when there is no off-spine splitting (82.6% of the reference
    # sample), so they ship with an explicit presence indicator and take a neutral 0 when
    # absent. `has_sec` lets the encoder gate them instead of reading 0 as a measurement;
    # log1p is used precisely so "absent" maps to 0 and any real value is bounded away
    # from it (kt >= kt_floor => log1p(kt) >= log(1 + kt_floor)).
    "has_sec": lambda j: 1.0 if _nsec(j) > 0 else 0.0,
    # hardest off-spine splitting: separates ONE hard secondary prong (a genuinely
    # three-pronged jet) from several soft ones -- same n_sec, different physics.
    "ln_kt_sec": lambda j: math.log1p(_sec(j, "x_kt_sec_max")),
    # total off-spine hardness; differs from the above only when n_sec > 1.
    "ln_kt_sec_sum": lambda j: math.log1p(_sec(j, "x_kt_sec_sum")),
    # which primary node the hardest secondary hangs off (0 == the widest-angle
    # splitting). A secondary off the first emission is a different topology from one
    # deep in the shower.
    "sec_depth": lambda j: math.log1p(_sec(j, "x_sec_attach")),
}


def aux_vector(jet: dict, names) -> np.ndarray:
    """`(n_aux,)` float32 aux features for one jet, in the order given.

    Raises ValueError on a missing, non-finite or sentinel source field: an old
    jets.root without the aux columns must fail loud, not train on NaNs."""
    names = tuple(names)
    out = np.empty(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        if name not in AUX_FEATURES:
            raise KeyError(f"unknown aux feature {name!r}; registered: {sorted(AUX_FEATURES)}")
        value = AUX_FEATURES[name](jet)
        if not math.isfinite(value):
            raise ValueError(f"aux feature {name!r} evaluated to {value!r}")
        out[i] = value
    return out


def aux_source_fields(names) -> tuple[str, ...]:
    """The RNTuple columns the given aux features read, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for name in names:
        for src in AUX_SOURCES.get(name, ()):
            seen.setdefault(src, None)
    return tuple(seen)


def configured_aux_names(enc_cfg) -> tuple[str, ...]:
    """`encoder.aux_features` as a validated tuple — the ONE supported way to read it.

    `getattr`-tolerant (same idiom as `cell_label_smoothing`), so a checkpoint config
    snapshot predating this field rebuilds as the plain no-aux model rather than
    crashing. An empty tuple is the default and the byte-identical off path."""
    names = tuple(getattr(enc_cfg, "aux_features", None) or ())
    unknown = [n for n in names if n not in AUX_FEATURES]
    if unknown:
        raise KeyError(
            f"unknown encoder.aux_features {unknown}; registered: {sorted(AUX_FEATURES)}"
        )
    return names


def node_features(ln_invd, ln_kt, ln_z, psi) -> np.ndarray:
    """Continuous per-node feature matrix (n, 5) for the encoder/decoder inputs."""
    ln_invd = np.asarray(ln_invd, dtype=np.float32)
    ln_kt = np.asarray(ln_kt, dtype=np.float32)
    ln_z = np.asarray(ln_z, dtype=np.float32)
    psi = np.asarray(psi, dtype=np.float32)
    if ln_invd.size == 0:
        return np.zeros((0, N_NODE_FEAT), dtype=np.float32)
    return np.stack(
        [ln_invd, ln_kt, ln_z, np.sin(psi), np.cos(psi)], axis=1
    ).astype(np.float32)


def with_aux(xf: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """Broadcast the per-jet aux vector onto every node of `xf` -> `(n, 5 + n_aux)`.

    Constant per-node columns, not a widened signature: conditioning is threaded
    everywhere as `(xf, nx)`, so this reaches every consumer — encoders, closure,
    calibration, MBR, serving — through the existing plumbing with no interface churn.
    Broadcasting globals per point is standard particle-cloud practice and acts as
    feature-wise conditioning of the node embedding (cf. FiLM, arXiv:1709.07871).

    KNOWN LIMITATION: a jet with `nx == 0` has no rows to carry the aux signal, so it
    silently loses it. `LundDataModule.setup` reports that fraction when aux is on."""
    if aux.size == 0:
        return xf
    tiled = np.broadcast_to(aux.astype(np.float32), (xf.shape[0], aux.shape[0]))
    return np.concatenate([xf, tiled], axis=1).astype(np.float32)


def node_raw(ln_invd, ln_kt, ln_z, psi) -> np.ndarray:
    """Raw continuous TARGETS (n, 4) = (ln 1/DeltaR, ln kt, ln z, psi) for the
    continuous coordinate likelihood. psi is kept as a raw angle (not sin/cos) so
    the von Mises term can be evaluated directly."""
    arrs = [np.asarray(a, dtype=np.float32) for a in (ln_invd, ln_kt, ln_z, psi)]
    if arrs[0].size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(arrs, axis=1).astype(np.float32)
