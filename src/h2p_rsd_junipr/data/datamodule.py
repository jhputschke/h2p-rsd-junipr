"""LundDataModule (§4): source selection, deterministic split, loaders, caching,
and the data fingerprint that ties a run to its data.

Split is by `event` id when available (so jets of the same event never straddle
the train/val boundary; §4 stage 4) and otherwise by jet index — deterministic
given the seed either way.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..features import configured_aux_names
from ..geometry import Geometry
from .dataset import MatchedLundDataset, collate
from .rntuple import load_rntuple
from .synthetic import synthetic_matched_dataset

# Which per-jet column `data.pt_var` may cut on, and the spellings accepted for it.
# Deliberately a closed set: a typo must name the alternatives, not silently select on
# a column that does not exist and drop every jet.
PT_SELECT_VARS: dict[str, str] = {
    "jet_pt": "jet_pt", "pt": "jet_pt",            # ungroomed jet pT, as clustered
    "x_ptg": "x_ptg", "ptg": "x_ptg", "pt_g": "x_ptg",  # groomed (SoftDrop-kept) pT
}


def select_pt_range(jets, var="jet_pt", lo=None, hi=None, *, verbose=True):
    """Keep only jets with `lo <= jet[var] < hi` (half-open; either bound may be None).

    Both bounds None is the off path: the same list object is returned, so the
    fingerprint and every downstream tensor are untouched.

    A jet whose `var` is absent or NaN fails both comparisons and is dropped — that is
    the right answer for a jet the file cannot place (synthetic jets carry no `jet_pt`,
    `rntuple.py` fills the column with its NaN sentinel when the file predates it) but a
    silent one, so an all-unset sample raises with the column named instead of handing
    back an empty dataset 200 lines later."""
    if (lo is None and hi is None) or not jets:
        return jets
    try:
        col = PT_SELECT_VARS[str(var)]
    except KeyError:
        raise ValueError(
            f"data.pt_var={var!r} is not a selectable jet-pT column; "
            f"use one of {sorted(set(PT_SELECT_VARS))}"
        ) from None

    lo_f = -math.inf if lo is None else float(lo)
    hi_f = math.inf if hi is None else float(hi)
    if not lo_f < hi_f:
        raise ValueError(
            f"empty jet-pT window: data.pt_min={lo} is not below data.pt_max={hi} "
            f"(the window is half-open, pt_min <= {col} < pt_max)"
        )
    if lo is not None and lo_f < 0.0:   # `lo is None` means -inf, which is not an error
        raise ValueError(f"data.pt_min={lo} is negative; {col} is a pT in GeV")

    vals = np.array([float(j.get(col, math.nan)) for j in jets], dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError(
            f"jet-pT selection on {col!r} requested (data.pt_min={lo}, data.pt_max={hi}) "
            f"but NO jet carries a finite {col}: every jet would be dropped. Synthetic "
            f"jets have no pT at all, and an RNTuple written before the aux columns "
            f"existed stores the NaN sentinel — check data.source / data.path, or clear "
            f"data.pt_min / data.pt_max."
        )
    keep = finite & (vals >= lo_f) & (vals < hi_f)
    out = [j for j, k in zip(jets, keep) if k]
    if not out:
        raise ValueError(
            f"jet-pT selection {col} in [{lo_f:g}, {hi_f:g}) kept 0 of {len(jets)} jets; "
            f"the sample spans {np.nanmin(vals):g}-{np.nanmax(vals):g} GeV. "
            f"Widen data.pt_min / data.pt_max."
        )
    if verbose:
        n_unset = int((~finite).sum())
        kept = vals[keep]
        print(
            f"[data] jet-pT selection {col} in [{lo_f:g}, {hi_f:g}) GeV: kept "
            f"{len(out)}/{len(jets)} jets ({len(out) / len(jets):.1%}), "
            f"kept range {kept.min():g}-{kept.max():g} GeV"
            + (f"; {n_unset} jets dropped for having no finite {col}" if n_unset else "")
        )
    return out


def _fingerprint(jets, cfg_data, aux_features=()) -> str:
    """Hash of source + grooming params + a content sample, so jets.root <-> run
    linkage is explicit (§9 data versioning).

    `aux_features` is part of the hash because it changes the WIDTH of the tensors
    built from these jets — without it the §4 preprocessed-tensor cache could serve
    a stale-width `xf` to a model expecting the aux columns.

    An active jet-pT window is mixed in for the same reason, and ONLY when active so
    that a run without one hashes exactly as it did before the knob existed: two
    different windows can leave the same jet count with the same leading jets, which
    the length + content-sample terms alone would not tell apart."""
    h = hashlib.sha1()
    h.update(str(cfg_data.source).encode())
    h.update(str(cfg_data.path).encode())
    h.update(str(cfg_data.seed).encode())
    h.update(str(len(jets)).encode())
    h.update(",".join(aux_features).encode())
    window = tuple(OmegaConf.select(cfg_data, k) for k in ("pt_var", "pt_min", "pt_max"))
    if window[1] is not None or window[2] is not None:
        h.update(f"pt:{window[0]}:{window[1]}:{window[2]}".encode())
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
        # tolerant read: a checkpoint config predating the field yields ()
        self.aux_features = configured_aux_names(cfg.encoder)

    # ---- load + split ------------------------------------------------------
    def setup(self) -> LundDataModule:
        d = self.cfg.data
        jets = None
        if d.source == "rntuple":
            jets = load_rntuple(d.path, d.ntuple)
        if jets is None:
            jets = synthetic_matched_dataset(d.n_jets, seed=d.seed, max_emissions=d.max_emissions)
        # Before the fingerprint and before the split: the window defines WHICH jets this
        # run is about, so it must be reflected in the run<->data hash (and hence in the
        # preprocessed-tensor cache key), and train/val must be drawn from the same
        # window. `OmegaConf.select` because a pre-window checkpoint snapshot has no such
        # key and `d.pt_min` would raise on it at eval time.
        pt_min, pt_max = (OmegaConf.select(d, k) for k in ("pt_min", "pt_max"))
        jets = select_pt_range(
            jets, OmegaConf.select(d, "pt_var") or "jet_pt", pt_min, pt_max
        )
        self.jets = jets
        self.fingerprint = _fingerprint(jets, d, self.aux_features)
        self._report_aux_coverage(jets)

        n_val = max(int(d.min_val), len(jets) // int(round(1.0 / d.val_fraction)))
        if (pt_min is not None or pt_max is not None) and len(jets) <= n_val:
            # `min_val` is a floor, so a narrow window silently hands every surviving jet
            # to validation and trains on nothing. Say which knob to move.
            raise ValueError(
                f"jet-pT selection left {len(jets)} jets, but the split reserves "
                f"n_val={n_val} of them (data.min_val={d.min_val}, "
                f"data.val_fraction={d.val_fraction}): the train split would be empty. "
                f"Widen data.pt_min / data.pt_max, or lower data.min_val."
            )
        # Default (no event ids): trailing split, matching the v2 script exactly.
        events = [j.get("event") for j in jets]
        if all(e is not None for e in events) and len(set(events)) > 1:
            self.train_jets, self.val_jets = self._split_by_event(jets, events, n_val, d.seed)
        else:
            self.train_jets, self.val_jets = jets[:-n_val], jets[-n_val:]
        return self

    def _report_aux_coverage(self, jets) -> None:
        """Report the fraction of jets whose aux signal is structurally lost.

        Aux rides as constant per-node columns of `xf`, so a jet with an EMPTY groomed
        hadron tree (`nx == 0`, physical and not rare — the RNTuple path keeps a jet
        whenever EITHER level survives grooming) has no rows to carry it. If this
        fraction is material, the fix is signature widening or folding aux into the
        WP3 cross-attention conditioning, not a silent tolerance."""
        if not self.aux_features or not jets:
            return
        n_empty = sum(1 for j in jets if len(j["x"][0]) == 0)
        frac = n_empty / len(jets)
        print(
            f"[data] aux_features={list(self.aux_features)}: {frac:.2%} of jets have an empty "
            f"hadron tree (nx=0) and therefore carry NO aux signal (broadcast has no rows)."
        )

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
        train = MatchedLundDataset(self.train_jets, self.geometry, self.aux_features)
        val = MatchedLundDataset(self.val_jets, self.geometry, self.aux_features)
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
