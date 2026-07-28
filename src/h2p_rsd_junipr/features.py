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

# Source RNTuple columns each aux feature reads. Used to fail loud, early, with a
# message naming the missing column rather than a NaN surfacing 200 epochs later.
AUX_SOURCES: dict[str, tuple[str, ...]] = {
    "ln_mg_pt": ("x_mg", "jet_pt"),
    "nsec": ("x_nsec",),
    "ln_pt": ("jet_pt",),
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
