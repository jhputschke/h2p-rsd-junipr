"""WP-G of docs/PLAN_prod_test_edit.md: the anchoring diagnostic, at production scale.

    python scripts/edit_anchoring_diagnostic.py [--run-root runs/prod_test_edit]
                                                [--data data/jet_aux_asym_test.root]
                                                [--n-jets 4000] [--only ARM,...]

Writes `anchoring_diagnostic.json` beside each edit checkpoint. Every number here comes
off `model.alignment_posterior` — a posterior over alignments obtained without ever
supervising one — so this file measures a mechanism, not a fit.

**What gate E7 reads, and from where.** `PLAN_EditTransducer.md`'s verification 4 is a
stage gate: *"if the widths are flat in `k_t`, the anchoring assumption is wrong — stop,
and do not build stage 2."* Its `Lambda_eff = 1.29 GeV, R^2 = 1.000` is a 6-epoch fit on
a 54k-jet test file, not production. §1 below re-runs it at production scale.

E7 is quoted from `e_v1_freewidth`, the arm with `model.physics_width=false`. That is
not a detail. Quoting `Lambda_eff` from the arm that was *told* the functional form
restates the parametrization; the free-MLP arm is the independent measurement, and it is
the rule `PLAN_EditTransducer.md` already set for itself. The physics arms are computed
too — a diagnostic that only runs on one arm cannot be cross-checked — but the file
records `is_readout_arm` so a reader cannot mistake which is which.

The six sections, in the plan's order:

1. responsibility-weighted residual widths binned in `ln k_t` (and `R_g`, `N`), fit to
   `sigma = sigma_0 + Lambda_eff . exp(-ln k_t)`;
2. `frac_anchored` / `delete_rate` / `insert_rate` — the 6-epoch reference is
   `frac_anchored = 0.20`, and mixture identifiability (risk 2) is live: collapse toward
   the free head makes edit an expensive AR model, and that outcome is a null result;
3. deletion rate vs `ln k_t` — should track the sub-floor fragmentation population;
4. free-emission (insertion) rate vs distance to the grooming boundary;
5. crossing-pair count in sampled alignments — the monotonicity audit (risk 1). The RNN-T
   lattice is monotone by construction, so a nonzero count is a BUG in the walk, not a
   finding about the physics, and it is asserted here rather than assumed;
6. the `n_x = 0` rate on the production file, which bounds how much of the sample the
   anchoring mechanism can act on at all.

A flat production fit is an *informative* failure — it says the anchoring premise does
not hold on this selection — and the plan pre-commits to reporting it as such rather than
retuning into a pass. Nothing in this script decides anything; §8's E7 criterion
(`Lambda_eff in [0.2, 5] GeV and R^2 >= 0.9`) was fixed before the grid ran, and
`scripts/prod_test_edit_gates.py` applies it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]

COORDS = ("ln_invDelta", "ln_kt", "ln_z", "psi")
# Bin edges are FIXED here rather than derived per arm: a per-arm quantile binning would
# make the width-vs-k_t curves of two arms functions of two different x-grids, and E7
# compares them.
LNKT_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
RG_EDGES = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)          # ln(1/DeltaR) of the anchor
N_EDGES = (0, 1, 2, 3, 5, 8, 13, 26)                          # hadron multiplicity |x|
SD_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5)               # ln z - (ln z_cut - beta u)
MIN_BIN_WEIGHT = 30.0   # below this a bin is reported `scored: false`, as in the gates


# ---------------------------------------------------------------------------
# weighted statistics
# ---------------------------------------------------------------------------
def _wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _weighted_moments(x, w):
    """`(sum_w, mean, sigma)` of `x` under weights `w`, or NaNs when the weight is 0."""
    sw = float(w.sum())
    if sw <= 0.0:
        return 0.0, float("nan"), float("nan")
    mean = float((w * x).sum() / sw)
    var = float((w * (x - mean) ** 2).sum() / sw)
    return sw, mean, math.sqrt(max(var, 0.0))


def _binned_widths(resid, weight, key, edges):
    """Responsibility-weighted residual width per coordinate, per bin of `key`.

    `resid` is `(m, 4)`, `weight` `(m,)`, `key` `(m,)`. The weight is
    `gamma_emit * r_anch` — the posterior probability that this parton node was an
    ANCHORED emission from this hadron column — so a node the model does not believe is
    anchored contributes almost nothing, which is exactly the intended semantics: the
    width being measured is the smearing kernel's, not the insertion head's."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        row = {"lo": float(lo), "hi": float(hi),
               "center": float(0.5 * (lo + hi)), "sigma": {}, "mean": {}}
        w = weight[m]
        row["sum_w"] = float(w.sum())
        row["n"] = int(m.sum())
        row["scored"] = bool(row["sum_w"] >= MIN_BIN_WEIGHT)
        for k, name in enumerate(COORDS):
            r = resid[m, k]
            if name == "psi":
                r = _wrap_to_pi(r)
            _sw, mean, sig = _weighted_moments(r, w)
            row["mean"][name] = mean
            row["sigma"][name] = sig
        out.append(row)
    return out


def _fit_shape_function(rows, coord):
    """Weighted least squares of `sigma = sigma_0 + Lambda_eff . exp(-ln k_t)`.

    Linear in `(sigma_0, Lambda_eff)` on the basis `[1, exp(-ln k_t)]`, so it is a
    two-parameter WLS and not an optimiser with a starting point to argue about. `R^2` is
    the weighted coefficient of determination on the same weights. Bins below
    `MIN_BIN_WEIGHT` are dropped, and how many survived is reported beside the fit —
    an `R^2 = 1.000` on two points is not a measurement, and E7's `>= 0.9` must not be
    satisfiable that way."""
    use = [r for r in rows if r["scored"] and np.isfinite(r["sigma"][coord])]
    if len(use) < 3:
        return {"n_bins": len(use), "sigma_0": float("nan"),
                "lambda_eff": float("nan"), "r2": float("nan"),
                "note": "fewer than 3 scoreable bins — not fitted"}
    x = np.array([r["center"] for r in use], dtype=float)
    y = np.array([r["sigma"][coord] for r in use], dtype=float)
    w = np.array([r["sum_w"] for r in use], dtype=float)
    A = np.stack([np.ones_like(x), np.exp(-x)], axis=1)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    pred = A @ coef
    ybar = float((w * y).sum() / w.sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    return {
        "n_bins": len(use),
        "sigma_0": float(coef[0]),
        "lambda_eff": float(coef[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        # A fit can have a fine R^2 and still be flat: `Lambda_eff` is what makes the
        # width fall with k_t, and E7's real question is whether it is nonzero and of
        # hadronic size. The relative variation over the fitted range says it directly.
        "sigma_range": [float(y.min()), float(y.max())],
        "falls_with_kt": bool(y[0] > y[-1]),
        "monotone": bool(np.all(np.diff(y) <= 1e-9)),
    }


# ---------------------------------------------------------------------------
# the pass over the data
# ---------------------------------------------------------------------------
def _collect(model, val_ds, geometry, device, n_jets, z_cut, beta, batch_size=64):
    """One `alignment_posterior` pass, accumulated into flat arrays.

    Batched: the posterior is a forward-backward over the whole lattice, so a per-jet
    loop would pay the encoder cost `n_jets` times for nothing."""
    from h2p_rsd_junipr.data.dataset import collate

    resid, weight, kt, rg, nx_of, sd_dist, ins_w = [], [], [], [], [], [], []
    del_num, del_den, del_kt = [], [], []
    n_x0 = n_seen = 0
    frac_a, del_r, ins_r = [], [], []

    for start in range(0, n_jets, batch_size):
        items = [val_ds[i] for i in range(start, min(start + batch_size, n_jets))]
        b = collate(items)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        post = model.alignment_posterior(b)
        summ = model.edit_summary(b)
        frac_a.append(summ["frac_anchored"])
        del_r.append(summ["delete_rate"])
        ins_r.append(summ["insert_rate"])

        gam = post["gamma_emit"].cpu().numpy()          # (B, n_col, Ny)
        ra = post["r_anch"].cpu().numpy()               # (B, n_col, Ny)
        anch = post["anchor"].cpu().numpy()             # (B, n_col, 4)
        y = b["yraw"].cpu().numpy()                     # (B, Ny, 4)
        nxs = b["nx"].cpu().numpy()
        nys = b["ny"].cpu().numpy()
        n_seen += len(items)
        n_x0 += int((nxs == 0).sum())

        w = gam * ra                                    # anchored-emission responsibility
        B, n_col, Ny = w.shape
        for bi in range(B):
            nxi, nyi = int(nxs[bi]), int(nys[bi])
            if nyi:
                # --- §1 residual widths, over every (column, emission) pair -------
                r = y[bi, None, :nyi, :] - anch[bi, :nxi, None, :]      # (nxi, nyi, 4)
                wi = w[bi, :nxi, :nyi]
                if nxi:
                    resid.append(r.reshape(-1, 4))
                    weight.append(wi.reshape(-1))
                    kt.append(np.repeat(anch[bi, :nxi, 1], nyi))
                    rg.append(np.repeat(anch[bi, :nxi, 0], nyi))
                    nx_of.append(np.full(nxi * nyi, nxi, dtype=float))
                # --- §4 insertion share vs distance to the grooming boundary ------
                # 1 - (anchored share) is the free-head responsibility of emission t.
                ins = 1.0 - w[bi, :, :nyi].sum(axis=0)
                ins_w.append(ins)
                lo = math.log(z_cut) - beta * y[bi, :nyi, 0]
                sd_dist.append(y[bi, :nyi, 2] - lo)
            # --- §3 deletion rate vs the column's own ln k_t ----------------------
            if nxi:
                kept = w[bi, :nxi, :nyi].sum(axis=1) if nyi else np.zeros(nxi)
                del_num.append(np.clip(1.0 - kept, 0.0, 1.0))
                del_den.append(np.ones(nxi))
                del_kt.append(anch[bi, :nxi, 1])

    def cat(xs, width=None):
        if not xs:
            return np.zeros((0, width) if width else 0)
        return np.concatenate(xs, axis=0)

    return {
        "resid": cat(resid, 4), "weight": cat(weight), "kt": cat(kt), "rg": cat(rg),
        "nx_of": cat(nx_of), "sd_dist": cat(sd_dist), "ins_w": cat(ins_w),
        "del_num": cat(del_num), "del_den": cat(del_den), "del_kt": cat(del_kt),
        "frac_anchored": np.concatenate(frac_a) if frac_a else np.zeros(0),
        "delete_rate": np.concatenate(del_r) if del_r else np.zeros(0),
        "insert_rate": np.concatenate(ins_r) if ins_r else np.zeros(0),
        "n_jets": n_seen, "n_x_zero": n_x0,
    }


def _rate_by_bin(value, key, edges, weight=None):
    """Weighted mean of `value` per bin of `key` — the shape §3 and §4 report."""
    w = np.ones_like(value) if weight is None else weight
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        sw = float(w[m].sum())
        rows.append({
            "lo": float(lo), "hi": float(hi), "center": float(0.5 * (lo + hi)),
            "n": int(m.sum()), "sum_w": sw,
            "rate": float((w[m] * value[m]).sum() / sw) if sw > 0 else float("nan"),
            "scored": bool(sw >= MIN_BIN_WEIGHT),
        })
    return rows


def _crossing_pairs(model, val_ds, device, n_jets, seed=0):
    """§5, the monotonicity audit (risk 1).

    Draw an alignment per jet from `P(alignment | x, y)` and count pairs `(s, t)` with
    `s < t` but `col[s] > col[t]`. The RNN-T lattice cannot produce one — the walk only
    ever advances — so this is a bug check on the walk, and its expected value is a hard
    zero. It is here because "the alignment is monotone" is a load-bearing assumption of
    the whole family (it is what makes `n_y = n_x - #del + #ins` an accounting identity),
    and a load-bearing assumption that is never measured is a hope."""
    from h2p_rsd_junipr.models import edit_dp

    g = torch.Generator(device="cpu").manual_seed(seed)
    n_cross = n_align = n_scored = 0
    for i in range(n_jets):
        item = val_ds[i]
        cells = item["yc"].tolist()
        if not cells:
            continue
        xf = item["xf"].unsqueeze(0).to(device)
        nx = torch.tensor([int(item["nx"])], device=device)
        nx0, L = int(item["nx"]), len(cells)
        S, e, anchor, ok = model._encode(xf, nx)
        S, anchor, ok = S[:, : nx0 + 1], anchor[:, : nx0 + 1], ok[:, : nx0 + 1]
        log_stay, log_emit = model._op_logprobs(S, e)
        yc = torch.tensor([cells], dtype=torch.long, device=device)
        C = model._prefix_states(yc, e)[:, :L] if model.prefix_conditioning else None
        p = model._emit_params(model._emit_input(S, e, C),
                               anchor[:, :, None, :], ok[:, :, None])
        log_mass, _ = model._log_cell_mass(p, yc[:, None, :])
        n_col = log_stay.shape[1]
        stay = log_stay[:, :, None].expand(1, n_col, L + 1)
        edge = log_emit[:, :, None] + log_mass
        cols = edit_dp.sample_alignment(stay[0], edge[0], nx0, L, generator=g)
        n_align += 1
        n_scored += len(cols)
        n_cross += sum(1 for s in range(len(cols)) for t in range(s + 1, len(cols))
                       if cols[s] > cols[t])
    return {"n_alignments": n_align, "n_emissions_aligned": n_scored,
            "n_crossing_pairs": int(n_cross),
            "monotone": bool(n_cross == 0)}


# ---------------------------------------------------------------------------
# per-arm driver
# ---------------------------------------------------------------------------
def run_arm(stamp_dir: Path, *, data_path=None, n_jets=4000, n_align=300,
            device="cpu", verbose=True) -> dict | None:
    from h2p_rsd_junipr.data.datamodule import LundDataModule
    from h2p_rsd_junipr.eval.support import grooming_from_jets
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    ckpt = stamp_dir / "best.ckpt"
    if not ckpt.is_file():
        return None
    info = load_for_inference(str(ckpt), map_location=device)
    cfg = OmegaConf.create(info["config"])
    family = str(OmegaConf.select(cfg, "model.name") or "")
    if not family.startswith("edit"):
        if verbose:
            print(f"[anchoring] {stamp_dir.parent.name}: model={family!r} is not an edit "
                  f"family — nothing to read (alignment_posterior is where every number "
                  f"here comes from). Skipped.")
        return None
    if data_path:
        cfg.data.path = str(data_path)

    dev = torch.device(device)
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(dev)
    model.load_state_dict(info["model_state"])
    model.eval()
    seed_everything(int(OmegaConf.select(cfg, "trainer.seed") or 0), True)

    dm = LundDataModule(cfg, geom).setup()
    dm.train_jets, dm.val_jets = [], dm.jets          # an explicitly named file is a TEST set
    _, val_ds = dm.datasets()
    n_jets = min(int(n_jets), len(val_ds))
    groom = grooming_from_jets(dm.val_jets)
    z_cut = groom["z_cut"] if groom["z_cut"] == groom["z_cut"] else 0.1
    beta = groom["beta"] if groom["beta"] == groom["beta"] else 0.0

    acc = _collect(model, val_ds, geom, dev, n_jets, z_cut, beta)

    widths_kt = _binned_widths(acc["resid"], acc["weight"], acc["kt"], LNKT_EDGES)
    fits = {c: _fit_shape_function(widths_kt, c) for c in COORDS}
    physics_width = bool(OmegaConf.select(cfg, "model.physics_width"))

    def nanmean(a):
        a = np.asarray(a, dtype=float)
        return float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else float("nan")

    out = {
        "arm": stamp_dir.parent.name,
        "checkpoint": str(ckpt),
        "model": family,
        "encoder": str(OmegaConf.select(cfg, "encoder.name")),
        "lnz_support": str(OmegaConf.select(cfg, "model.lnz_support") or "legacy"),
        "physics_width": physics_width,
        # E7 is READ OFF the free-MLP arm. Quoting Lambda_eff from the arm that was told
        # the functional form restates the parametrization; this flag is what stops a
        # reader (or a table) from doing that by accident.
        "is_readout_arm": (not physics_width),
        "device": str(dev),
        "data": {"path": str(cfg.data.path), "n_jets": int(acc["n_jets"]),
                 "z_cut": float(z_cut), "beta": float(beta)},

        # --- 1. residual widths, and the shape-function fit E7 reads ---------------
        "widths": {
            "by_ln_kt": widths_kt,
            "by_Rg": _binned_widths(acc["resid"], acc["weight"], acc["rg"], RG_EDGES),
            "by_N": _binned_widths(acc["resid"], acc["weight"], acc["nx_of"], N_EDGES),
        },
        "shape_function_fit": fits,
        "fit_form": "sigma = sigma_0 + Lambda_eff * exp(-ln k_t)   (Lambda_eff in GeV)",

        # --- 2. the edit accounting (already in closure; repeated here per arm) -----
        "edit_summary": {
            "frac_anchored": nanmean(acc["frac_anchored"]),
            "delete_rate": nanmean(acc["delete_rate"]),
            "insert_rate": nanmean(acc["insert_rate"]),
            "reference_6_epoch_frac_anchored": 0.20,
        },

        # --- 3. deletion rate vs ln k_t -------------------------------------------
        "delete_rate_by_ln_kt": _rate_by_bin(acc["del_num"], acc["del_kt"], LNKT_EDGES,
                                             weight=acc["del_den"]),

        # --- 4. insertion rate vs distance to the grooming boundary ----------------
        "insert_rate_by_soft_drop_distance": _rate_by_bin(
            acc["ins_w"], acc["sd_dist"], SD_EDGES),
        "soft_drop_distance": "ln z - (ln z_cut - beta * ln(1/DeltaR)), in ln z units",

        # --- 5. the monotonicity audit --------------------------------------------
        "alignment_monotonicity": _crossing_pairs(model, val_ds, dev,
                                                  min(n_align, n_jets)),

        # --- 6. the n_x = 0 rate, which bounds the mechanism's reach ---------------
        "n_x_zero_rate": (float(acc["n_x_zero"]) / acc["n_jets"]) if acc["n_jets"] else
                         float("nan"),
        "n_x_zero_note": ("jets with no hadron nodes reduce EXACTLY to the free head, so "
                          "the anchoring mechanism cannot act on them at all; this rate "
                          "bounds its reach on this file (6.9% in the PYTHIA reference)"),
    }
    # The learned scalars, when the arm has them — reported BESIDE the free-arm fit, never
    # instead of it. `physics_width_params` returns None under the ablation, which is the
    # whole reason the ablation is the readout.
    out["physics_width_params"] = model.physics_width_params()

    path = stamp_dir / "anchoring_diagnostic.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    if verbose:
        f = fits["ln_kt"]
        print(f"[anchoring] {out['arm']:<22} physics_width={physics_width} "
              f"frac_anchored={out['edit_summary']['frac_anchored']:.3f} "
              f"delete={out['edit_summary']['delete_rate']:.3f} "
              f"| ln_kt width fit: Lambda_eff={f['lambda_eff']:.3g} GeV "
              f"sigma_0={f['sigma_0']:.3g} R2={f['r2']:.3f} on {f['n_bins']} bins"
              f"{'' if f['falls_with_kt'] else '  [FLAT/RISING]'}"
              f" | crossings={out['alignment_monotonicity']['n_crossing_pairs']}"
              f" | n_x=0 {out['n_x_zero_rate']:.1%}")
        print(f"[anchoring] wrote {path}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="runs/prod_test_edit")
    ap.add_argument("--data", default="data/jet_aux_asym_test.root",
                    help="the file to read the diagnostic on; the plan's seed-2 test file. "
                         "Empty string keeps the checkpoint's own training file.")
    ap.add_argument("--n-jets", type=int, default=4000)
    ap.add_argument("--n-align", type=int, default=300,
                    help="jets used for the §5 crossing-pair audit (one alignment draw "
                         "each; it is a per-jet python walk, unlike everything else here)")
    ap.add_argument("--device", default="cpu",
                    help="cpu, per WP-F.1's standing whole-grid device rule")
    ap.add_argument("--only", default="", help="comma-separated arm names")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    if not root.is_absolute():
        root = REPO / root
    if not root.is_dir():
        print(f"no such run root: {root}")
        return 1
    only = {s for s in a.only.split(",") if s}
    n = 0
    for arm in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
        if only and arm.name not in only:
            continue
        for stamp in sorted(arm.iterdir()):
            if (stamp / "best.ckpt").is_file():
                if run_arm(stamp, data_path=(a.data or None), n_jets=a.n_jets,
                           n_align=a.n_align, device=a.device) is not None:
                    n += 1
    print(f"[anchoring] {n} edit arm(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
