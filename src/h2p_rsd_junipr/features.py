"""Per-node feature builders (was `node_features`, `node_raw`, `seq_cells`).

Geometry-free continuous features; cell discretisation lives on `Geometry`.
"""

from __future__ import annotations

import numpy as np

# encoder/decoder INPUT features: (ln 1/DeltaR, ln kt, ln z, sin psi, cos psi)
N_NODE_FEAT = 5


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


def node_raw(ln_invd, ln_kt, ln_z, psi) -> np.ndarray:
    """Raw continuous TARGETS (n, 4) = (ln 1/DeltaR, ln kt, ln z, psi) for the
    continuous coordinate likelihood. psi is kept as a raw angle (not sin/cos) so
    the von Mises term can be evaluated directly."""
    arrs = [np.asarray(a, dtype=np.float32) for a in (ln_invd, ln_kt, ln_z, psi)]
    if arrs[0].size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(arrs, axis=1).astype(np.float32)
