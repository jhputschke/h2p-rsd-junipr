"""LundDataModule (§4): source selection, deterministic split, loaders, caching,
and the data fingerprint that ties a run to its data.

Split is by `event` id when available (so jets of the same event never straddle
the train/val boundary; §4 stage 4) and otherwise by jet index — deterministic
given the seed either way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ..geometry import Geometry
from .dataset import MatchedLundDataset, collate
from .rntuple import load_rntuple
from .synthetic import synthetic_matched_dataset


def _fingerprint(jets, cfg_data) -> str:
    """Hash of source + grooming params + a content sample, so jets.root <-> run
    linkage is explicit (§9 data versioning)."""
    h = hashlib.sha1()
    h.update(str(cfg_data.source).encode())
    h.update(str(cfg_data.path).encode())
    h.update(str(cfg_data.seed).encode())
    h.update(str(len(jets)).encode())
    for j in jets[: min(len(jets), 64)]:
        h.update(np.asarray(j["x"][0], dtype=np.float32).tobytes())
        h.update(np.asarray(j["y"][0], dtype=np.float32).tobytes())
    return h.hexdigest()[:12]


class LundDataModule:
    def __init__(self, cfg, geometry: Geometry):
        self.cfg = cfg
        self.geometry = geometry
        self.jets: list | None = None
        self.train_jets: list | None = None
        self.val_jets: list | None = None
        self.fingerprint: str | None = None

    # ---- load + split ------------------------------------------------------
    def setup(self) -> LundDataModule:
        d = self.cfg.data
        jets = None
        if d.source == "rntuple":
            jets = load_rntuple(d.path, d.ntuple)
        if jets is None:
            jets = synthetic_matched_dataset(d.n_jets, seed=d.seed, max_emissions=d.max_emissions)
        self.jets = jets
        self.fingerprint = _fingerprint(jets, d)

        n_val = max(int(d.min_val), len(jets) // int(round(1.0 / d.val_fraction)))
        # Default (no event ids): trailing split, matching the v2 script exactly.
        events = [j.get("event") for j in jets]
        if all(e is not None for e in events) and len(set(events)) > 1:
            self.train_jets, self.val_jets = self._split_by_event(jets, events, n_val, d.seed)
        else:
            self.train_jets, self.val_jets = jets[:-n_val], jets[-n_val:]
        return self

    @staticmethod
    def _split_by_event(jets, events, n_val_target, seed):
        uniq = sorted(set(events))
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        val_events: set = set()
        count = 0
        per_event = {}
        for j, e in zip(jets, events):
            per_event.setdefault(e, []).append(j)
        for e in uniq:
            if count >= n_val_target:
                break
            val_events.add(e)
            count += len(per_event[e])
        train = [j for j, e in zip(jets, events) if e not in val_events]
        val = [j for j, e in zip(jets, events) if e in val_events]
        return train, val

    # ---- datasets / loaders ------------------------------------------------
    def datasets(self):
        train = MatchedLundDataset(self.train_jets, self.geometry)
        val = MatchedLundDataset(self.val_jets, self.geometry)
        return train, val

    def loaders(self):
        train, val = self.datasets()
        bs = int(self.cfg.trainer.batch_size)
        nw = int(self.cfg.trainer.num_workers)
        train_loader = DataLoader(
            train, batch_size=bs, shuffle=True, collate_fn=collate, drop_last=True, num_workers=nw
        )
        val_loader = DataLoader(
            val, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=nw
        )
        return train_loader, val_loader

    # ---- preprocess cache (§4 stage 3) ------------------------------------
    def cache_path(self) -> Path | None:
        if not self.cfg.data.cache_dir:
            return None
        return Path(self.cfg.data.cache_dir) / f"jets_{self.fingerprint}.pt"
