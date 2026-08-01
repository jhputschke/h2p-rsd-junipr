"""Dataset statistics and the multiplicity-support guard (docs/PLAN_UPDATES.md WP4).

A categorical multiplicity head (`ar_junipr_v3`, `cinn`, `diffusion`, `cfm`) has a
FINITE support `N = 0..model.max_emissions`. The v2 continue/stop head had none: it
could always emit one more, so a long truth sequence was merely improbable. Under a
head, a truth with `N > max_emissions` is clamped into the last bin — it receives
the wrong likelihood, silently, for every such jet, and the resulting model is
mis-normalized in exactly the tail the physics cares about.

Grooming parameters move that tail: loosening `z_cut`, lowering the `k_t` floor, or
raising `beta` all admit more primary emissions. So the check has to run against the
data actually loaded, not once at design time — this module is called from `train`
and `eval` after the datamodule is set up.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

# Above ERROR the head is wrong for enough jets to bias the length marginal; between
# WARN and ERROR it is a tail worth knowing about. Both are fractions of all jets.
SUPPORT_TAIL_ERROR = 1e-3
SUPPORT_TAIL_WARN = 1e-4


def multiplicity_stats(jets) -> dict:
    """Truth-multiplicity summary of a loaded jet list (no torch, no Dataset)."""
    n = np.array([len(j["y"][0]) for j in jets], dtype=int)
    if n.size == 0:
        return {"n_jets": 0, "max": 0, "mean": float("nan"), "counts": np.zeros(1, dtype=int)}
    return {
        "n_jets": int(n.size),
        "max": int(n.max()),
        "mean": float(n.mean()),
        "p99": float(np.percentile(n, 99)),
        "frac_empty": float((n == 0).mean()),
        "counts": np.bincount(n),
    }


def _grooming_context(jets) -> str:
    """The z_cut / beta / k_t-floor the file was written with — the knobs that move
    the multiplicity tail, so the error message says what to change."""
    j = jets[0] if jets else {}
    parts = [f"{k}={j[k]:g}" for k in ("z_cut", "beta", "kt_floor")
             if k in j and j[k] == j[k]]  # skip NaN (synthetic data carries none)
    return ", ".join(parts) if parts else "grooming parameters unrecorded (synthetic data?)"


def model_support(cfg) -> int | None:
    """`model.max_emissions` when the family actually has a categorical multiplicity
    head, else None (the v1/v2 continue/stop model has unbounded support)."""
    max_em = OmegaConf.select(cfg, "model.max_emissions")
    if max_em is None:
        return None
    name = str(OmegaConf.select(cfg, "model.name") or "")
    if name.startswith("ar_junipr") and not bool(
        OmegaConf.select(cfg, "model.use_multiplicity_head") or False
    ):
        return None                       # v1/v2: max_emissions is inert here
    if name.startswith("edit"):
        # Same story, different mechanism. The edit transducer's length model is the
        # open-ended STOP/EMIT lattice, so a long truth is improbable, not mis-normalized:
        # `model.max_emissions` there is the *readout* width of the exact structural
        # q(N|x) and the sampler's cap, and it never touches `log_prob`. Firing the guard
        # would refuse to train on data the likelihood handles correctly.
        return None
    return int(max_em)


def check_multiplicity_support(jets, cfg, *, strict=True, verbose=True) -> dict:
    """Guard `P_data(N > model.max_emissions)`: hard error above 1e-3, warning above
    1e-4, silent below. Returns the statistics either way.

    Cheap (one pass over the loaded lengths) and deliberately run BEFORE training
    starts: the failure it catches is invisible in the loss curve. `strict=False`
    downgrades the error to a warning — used at `eval`, where the model is already
    trained and refusing to report on it helps nobody."""
    stats = multiplicity_stats(jets)
    support = model_support(cfg)
    stats["support"] = support
    if support is None or stats["n_jets"] == 0:
        stats["tail_fraction"] = 0.0
        return stats

    counts = stats["counts"]
    over = int(counts[support + 1:].sum()) if counts.size > support + 1 else 0
    frac = over / stats["n_jets"]
    stats["tail_fraction"] = float(frac)
    msg = (
        f"P_data(N > model.max_emissions={support}) = {frac:.2e} "
        f"({over}/{stats['n_jets']} jets; max N = {stats['max']}). "
        f"Truth sequences past the categorical support are clamped into the last bin, "
        f"so they get the WRONG likelihood and the length marginal is biased. "
        f"Context: {_grooming_context(jets)}. "
        f"Fix by raising model.max_emissions (>= {stats['max']}) or tightening the "
        f"grooming (higher z_cut / higher k_t floor) so the tail is not populated."
    )
    if frac > SUPPORT_TAIL_ERROR:
        if strict:
            raise ValueError("[data] multiplicity support exceeded: " + msg)
        print("[data] WARNING (support exceeded): " + msg)
        return stats
    if frac > SUPPORT_TAIL_WARN:
        print("[data] WARNING: " + msg)
    elif verbose and over:
        print(f"[data] multiplicity support OK: {msg.split('.')[0]}.")
    return stats
