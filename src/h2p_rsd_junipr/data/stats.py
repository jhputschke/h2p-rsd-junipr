"""Dataset statistics, the multiplicity-support guard (docs/PLAN_UPDATES.md WP4), and
the `ln z` grooming-record guard (docs/PLAN_prod_test_v1.md WP-A).

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

import math

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


# ---------------------------------------------------------------------------
# WP-A: the ln z support declared by the model vs the one the file was groomed to
# ---------------------------------------------------------------------------
def check_lnz_support(jets, cfg, *, strict=True, verbose=True) -> dict:
    """Guard `model.lnz_support='physical'` against the data actually loaded.

    `(lnz_zcut, lnz_beta)` are config fields because `build_model` sees only the config
    — but they are properties of the FILE, and a mismatch is silent and total: the head
    normalizes over an interval the truth does not live on, so every `ln z` likelihood,
    PIT and draw is wrong while the loss curve looks ordinary. So the declared pair is
    checked against the jets' own grooming record, and the truth `ln z` values are
    checked against the resulting interval.

    Returns the audit dict; `strict=False` downgrades the error to a warning (`eval`,
    where the model is already trained). A no-op in `legacy` mode and for data carrying
    no grooming record (synthetic), which is reported rather than assumed."""
    out: dict = {"support": str(OmegaConf.select(cfg, "model.lnz_support") or "legacy")}
    if out["support"] != "physical" or not jets:
        return out
    z_cut = float(OmegaConf.select(cfg, "model.lnz_zcut") or 0.1)
    beta = float(OmegaConf.select(cfg, "model.lnz_beta") or 0.0)
    out.update(z_cut=z_cut, beta=beta)

    def _record(key):
        vals = np.array([j[key] for j in jets if key in j and j[key] == j[key]], dtype=float)
        return vals

    file_zcut, file_beta = _record("z_cut"), _record("beta")
    if file_zcut.size == 0 or file_beta.size == 0:
        out["checked"] = False
        if verbose:
            print("[data] NOTE: model.lnz_support='physical' but the loaded jets carry no "
                  "grooming record (synthetic data?), so the declared "
                  f"(z_cut={z_cut:g}, beta={beta:g}) could not be verified against them.")
        return out
    out["checked"] = True
    out["file_z_cut"] = [float(v) for v in np.unique(file_zcut)]
    out["file_beta"] = [float(v) for v in np.unique(file_beta)]

    problems = []
    if not np.allclose(file_zcut, z_cut) or not np.allclose(file_beta, beta):
        problems.append(
            f"model.lnz_zcut/lnz_beta = ({z_cut:g}, {beta:g}) but the file was groomed with "
            f"z_cut in {out['file_z_cut']}, beta in {out['file_beta']}"
        )
    # ...and the interval those numbers imply must actually contain the truth. This
    # catches a convention error (a sign on beta, an R != 1) that matching scalars cannot.
    lnz = np.concatenate([j["y"][2] for j in jets if len(j["y"][2])]) if jets else np.zeros(0)
    if lnz.size:
        u = np.concatenate([j["y"][0] for j in jets if len(j["y"][0])])
        lo = math.log(z_cut) - beta * u
        hi = math.log(0.5)
        n_below = int((lnz < lo - 1e-6).sum())
        n_above = int((lnz > hi + 1e-6).sum())
        out.update(n_emissions=int(lnz.size), n_below_lo=n_below, n_above_hi=n_above,
                   frac_outside=float((n_below + n_above) / lnz.size))
        if n_below or n_above:
            problems.append(
                f"{n_below + n_above}/{lnz.size} truth emissions fall OUTSIDE "
                f"(ln z_cut - beta*ln(1/DeltaR), ln 1/2] = "
                f"({math.log(z_cut):.4f} - {beta:g}*u, {hi:.4f}] "
                f"({n_below} below, {n_above} above)"
            )
    out["ok"] = not problems
    if problems:
        msg = ("[data] ln z support mismatch: " + "; ".join(problems)
               + ". The physical `ln z` head normalizes over that interval, so a mismatch "
                 "silently mis-normalizes every ln z likelihood, PIT and draw. Fix the "
                 "model.lnz_zcut/lnz_beta pair, or train with model.lnz_support=legacy.")
        if strict:
            raise ValueError(msg)
        print("[data] WARNING: " + msg[len("[data] "):])
    elif verbose:
        print(f"[data] ln z support OK: physical, z_cut={z_cut:g}, beta={beta:g}"
              + (f", all {out['n_emissions']} truth emissions inside "
                 f"[{math.log(z_cut):.4f}, {math.log(0.5):.4f}]" if lnz.size else ""))
    return out
