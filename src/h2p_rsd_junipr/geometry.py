"""Lund-plane discretisation (was the module-level globals of the v2 script).

`(ln 1/DeltaR, ln kt) -> flat cell id` and its inverse, promoted to a config-built
`Geometry` object. The default geometry (ranges (0,6), n_bins=10) reproduces the
script's `to_cell`/`cell_center`/cell-centre buffers exactly, so the discretised
likelihood is unchanged. `n_cells = n_bins**2` and the within-cell offset bounds
are *derived* here and never set independently (§2 note).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Geometry:
    ln_invdelta_range: tuple[float, float] = (0.0, 6.0)
    ln_kt_range: tuple[float, float] = (0.0, 6.0)
    n_bins: int = 10

    @classmethod
    def from_config(cls, gcfg) -> Geometry:
        u = tuple(float(x) for x in gcfg.ln_invdelta_range)
        v = tuple(float(x) for x in gcfg.ln_kt_range)
        return cls(ln_invdelta_range=(u[0], u[1]), ln_kt_range=(v[0], v[1]), n_bins=int(gcfg.n_bins))

    # ---- derived quantities ------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.n_bins * self.n_bins

    @property
    def start_token(self) -> int:
        return self.n_cells  # extra embedding index used as the start token

    @property
    def cell_wu(self) -> float:
        lo, hi = self.ln_invdelta_range
        return (hi - lo) / self.n_bins

    @property
    def cell_wv(self) -> float:
        lo, hi = self.ln_kt_range
        return (hi - lo) / self.n_bins

    @property
    def half_u(self) -> float:
        return self.cell_wu / 2.0

    @property
    def half_v(self) -> float:
        return self.cell_wv / 2.0

    # ---- forward / inverse map --------------------------------------------
    def to_cell(self, ln_invdelta: float, ln_kt: float) -> int:
        lo_u, hi_u = self.ln_invdelta_range
        lo_v, hi_v = self.ln_kt_range
        x = float(np.clip(ln_invdelta, lo_u, hi_u))
        y = float(np.clip(ln_kt, lo_v, hi_v))
        ix = min(int((x - lo_u) / (hi_u - lo_u) * self.n_bins), self.n_bins - 1)
        iy = min(int((y - lo_v) / (hi_v - lo_v) * self.n_bins), self.n_bins - 1)
        return ix * self.n_bins + iy

    def cell_center(self, cell: int) -> tuple[float, float]:
        ix, iy = divmod(cell, self.n_bins)
        lo_u = self.ln_invdelta_range[0]
        lo_v = self.ln_kt_range[0]
        return lo_u + (ix + 0.5) * self.cell_wu, lo_v + (iy + 0.5) * self.cell_wv

    def seq_cells(self, ln_invd, ln_kt) -> np.ndarray:
        """Discretise a (ln 1/DeltaR, ln kt) sequence to flat Lund-cell ids."""
        return np.array([self.to_cell(a, b) for a, b in zip(ln_invd, ln_kt)], dtype=np.int64)

    def cell_center_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(cell_cx, cell_cy) lookup tensors, one entry per cell — registered as
        buffers by the model so they move with `.to(device)`."""
        ix = torch.arange(self.n_cells) // self.n_bins
        iy = torch.arange(self.n_cells) % self.n_bins
        cx = self.ln_invdelta_range[0] + (ix + 0.5).float() * self.cell_wu
        cy = self.ln_kt_range[0] + (iy + 0.5).float() * self.cell_wv
        return cx, cy


# Default geometry matching the v2 script's module globals (back-compat / tests).
DEFAULT_GEOMETRY = Geometry()
