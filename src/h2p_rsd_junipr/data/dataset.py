"""Torch Dataset + collate (was `MatchedLundDataset`, `collate`).

The dataset is geometry-aware (cell targets `yc` come from `Geometry.seq_cells`),
so the discretisation is config-driven rather than a module global.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..features import N_NODE_FEAT, node_features, node_raw
from ..geometry import DEFAULT_GEOMETRY, Geometry


class MatchedLundDataset(Dataset):
    def __init__(self, jets, geometry: Geometry = DEFAULT_GEOMETRY):
        self.geometry = geometry
        self.items = []
        for j in jets:
            xf = node_features(*j["x"])  # (nx, 5)  encoder input
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

    xf = torch.zeros(B, Mx, N_NODE_FEAT)
    yf = torch.zeros(B, My, N_NODE_FEAT)
    yc = torch.zeros(B, My, dtype=torch.long)
    yraw = torch.zeros(B, My, 4)
    for i, b in enumerate(batch):
        xf[i, : b["nx"]] = b["xf"]
        yf[i, : b["ny"]] = b["yf"]
        yc[i, : b["ny"]] = b["yc"]
        yraw[i, : b["ny"]] = b["yraw"]
    return dict(xf=xf, nx=nx, yf=yf, yc=yc, yraw=yraw, ny=ny, w=w)
