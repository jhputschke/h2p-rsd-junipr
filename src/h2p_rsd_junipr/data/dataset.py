"""Torch Dataset + collate (was `MatchedLundDataset`, `collate`).

The dataset is geometry-aware (cell targets `yc` come from `Geometry.seq_cells`),
so the discretisation is config-driven rather than a module global.

`aux_features` (docs/PLAN_Input.md, default `()`) appends per-jet groomed scalars as
constant per-node columns of `xf`. Only the CONDITIONING side widens: `yf`/`yraw`
keep widths 5/4 and the decoder is untouched.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..features import (
    N_NODE_FEAT,
    aux_source_fields,
    aux_vector,
    node_features,
    node_raw,
    with_aux,
)
from ..geometry import DEFAULT_GEOMETRY, Geometry


def _check_aux_sources(jets, names) -> None:
    """Fail loud, once, before building tensors: the aux columns are produced by the
    C++ writer and have no honest stand-in anywhere else.

    In particular there is no synthetic proxy. `synthetic_matched_dataset` has no
    secondary Lund planes, so any proxy would be a function of x — redundant by
    construction, and it would fake exactly the information gain aux exists to
    measure. The physics A/B runs on the PYTHIA RNTuple path only."""
    if not names or not jets:
        return
    probe = jets[0]
    missing = [f for f in aux_source_fields(names) if f not in probe]
    if not missing:
        return
    if probe.get("generator") == "synthetic":
        raise ValueError(
            f"encoder.aux_features={list(names)} is not available on data.source=synthetic: "
            "the synthetic generator has no secondary Lund planes, and any proxy would be a "
            "function of x — redundant by construction, and it would fake the information "
            "gain aux exists to measure. Run the aux A/B on the PYTHIA RNTuple path "
            "(data=rntuple)."
        )
    raise ValueError(
        f"encoder.aux_features={list(names)} needs the columns {missing}, absent from this "
        "jets.root. Re-write it with the current cpp/ writer (docs/PLAN_Input.md stage 1)."
    )


class MatchedLundDataset(Dataset):
    def __init__(self, jets, geometry: Geometry = DEFAULT_GEOMETRY, aux_features=()):
        self.geometry = geometry
        self.aux_feature_names = tuple(aux_features or ())
        _check_aux_sources(jets, self.aux_feature_names)
        self.items = []
        for j in jets:
            xf = node_features(*j["x"])  # (nx, 5)  encoder input
            if self.aux_feature_names:
                # constant per-node columns -> (nx, 5 + n_aux); an empty x keeps 0 rows
                xf = with_aux(xf, aux_vector(j, self.aux_feature_names))
            yf = node_features(*j["y"])  # (ny, 5)  (kept for parity; unused by decoder)
            yc = geometry.seq_cells(j["y"][0], j["y"][1])  # (ny,)  discrete cell targets
            yr = node_raw(*j["y"])  # (ny, 4)  continuous coordinate targets
            self.items.append(
                dict(
                    xf=torch.tensor(xf),
                    nx=len(xf),
                    yf=torch.tensor(yf),
                    yc=torch.tensor(yc),
                    yraw=torch.tensor(yr),
                    ny=len(yc),
                    w=torch.tensor(float(j["weight"]), dtype=torch.float32),
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    B = len(batch)
    nx = torch.tensor([b["nx"] for b in batch], dtype=torch.long)
    ny = torch.tensor([b["ny"] for b in batch], dtype=torch.long)
    w = torch.stack([b["w"] for b in batch])
    Mx, My = int(nx.max()), int(ny.max())

    # x width is INFERRED (5, or 5 + n_aux), never the N_NODE_FEAT constant: mixing
    # widths in one batch is a config error, not something to pad over silently.
    Fx = int(batch[0]["xf"].shape[1])
    if any(int(b["xf"].shape[1]) != Fx for b in batch):
        widths = sorted({int(b["xf"].shape[1]) for b in batch})
        raise ValueError(f"collate got mixed x feature widths {widths} in one batch")

    xf = torch.zeros(B, Mx, Fx)
    yf = torch.zeros(B, My, N_NODE_FEAT)  # target side is never widened by aux
    yc = torch.zeros(B, My, dtype=torch.long)
    yraw = torch.zeros(B, My, 4)
    for i, b in enumerate(batch):
        xf[i, : b["nx"]] = b["xf"]
        yf[i, : b["ny"]] = b["yf"]
        yc[i, : b["ny"]] = b["yc"]
        yraw[i, : b["ny"]] = b["yraw"]
    return dict(xf=xf, nx=nx, yf=yf, yc=yc, yraw=yraw, ny=ny, w=w)
