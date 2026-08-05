"""Why does `dv` fail its PIT and `du` never does? (docs/PLAN_lnz_spline_head.md §7.1)

Once the `ln z` spline landed, `dv` — the within-cell `ln kt` offset — became the binding
coordinate: it fails on every seed at 1.04-1.12x its critical value while `du` sits at
0.61-0.85x and has never failed. Both are two-parameter truncated normals on bounded
intervals of the SAME width, so "the family is too rigid" does not by itself explain why
one fails and the other does not. This script measures which explanation is right, because
they call for different fixes:

  * an EDGE effect (the `kt_floor` cut sits inside the `ln kt` range and not inside the
    angular one) => the fix is a SUPPORT correction, the `lnz_support` move again;
  * a within-cell SHAPE the truncated normal cannot express => the fix is a spline.

Three measurements, in the order they narrow it:

1. **`dv`'s PIT stratified by `ln kt` cell.** If the defect is the `kt_floor` edge it must
   concentrate in the cell touching it.
2. **The truth's within-cell offset shape**, per axis: the marginal log-density gradient
   across one cell, and the offset's skew/kurtosis. This is what the head has to represent.
3. **The tilt budget.** A truncated normal on `[-h, h]` has log-density slope `mu/sigma^2`,
   so across the cell it can tilt by `2h*mu/sigma^2` — and `mu` is CLAMPED to `+-h` by the
   `h*tanh` parameterization. A wide sigma therefore costs tilt authority that cannot be
   bought back. Compare what each head achieves against what the data asks for.

Run (needs trained arms under `runs/lnz_spline/`):

    python scripts/offset_head_diagnostic.py
    python scripts/offset_head_diagnostic.py --arms spline_s0,spline_s2 --n-jets 2000
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from omegaconf import OmegaConf  # noqa: E402

from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate  # noqa: E402
from h2p_rsd_junipr.data.rntuple import load_rntuple  # noqa: E402
from h2p_rsd_junipr.eval.report import save_metrics  # noqa: E402
from h2p_rsd_junipr.geometry import Geometry  # noqa: E402
from h2p_rsd_junipr.models.base import build_model  # noqa: E402
from h2p_rsd_junipr.train.checkpoint import load_for_inference  # noqa: E402


def ks(u) -> float:
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - u, u - (i - 1) / n)))


def critical(n) -> float:
    return 1.36 / math.sqrt(float(n)) if n else float("nan")


def load_arm(arm: str, run_root: Path):
    hits = glob.glob(str(run_root / arm / "*" / "best.ckpt"))
    if not hits:
        return None, None
    info = load_for_inference(sorted(hits)[-1], map_location="cpu")
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom)
    model.load_state_dict(info["model_state"])
    return model.eval(), geom


def truth_shape(jets, geom) -> dict:
    """What the data asks of a within-cell offset head, per axis.

    `slope_per_cell` is the marginal log-density change across one cell — the tilt the
    head must reproduce. `skew`/`excess_kurtosis` are the offset's own shape, pooled over
    populated cells: near-uniform (kurtosis ~ -1.2) means the head mostly needs to be FLAT,
    which is exactly the regime in which a truncated normal's tilt authority collapses."""
    u = np.concatenate([j["y"][0] for j in jets if len(j["y"][0])])
    v = np.concatenate([j["y"][1] for j in jets if len(j["y"][0])])
    cells = np.concatenate([geom.seq_cells(j["y"][0], j["y"][1])
                            for j in jets if len(j["y"][0])])
    cx = np.array([geom.cell_center(c)[0] for c in range(geom.n_cells)])
    cy = np.array([geom.cell_center(c)[1] for c in range(geom.n_cells)])
    out = {}
    for name, x, off, width in (("du", u, u - cx[cells], geom.cell_wu),
                                ("dv", v, v - cy[cells], geom.cell_wv)):
        lo, hi = np.percentile(x, [2, 98])
        counts, _ = np.histogram(x, bins=np.arange(lo, hi + width, width))
        counts = counts[counts > 5].astype(float)
        slope = float(np.abs(np.diff(np.log(counts))).mean())
        sk, ku, w = [], [], []
        # 200 keeps a cell's skew/kurtosis meaningful; with a small sample no cell
        # qualifies, which is why this runs on the WHOLE file and not on the subsample
        # the model is evaluated over (it measures the data, not the model).
        for c in np.unique(cells):
            m = cells == c
            if m.sum() < 200:
                continue
            z = off[m]
            s = z.std()
            if s < 1e-6:
                continue
            z = (z - z.mean()) / s
            sk.append(abs(float((z ** 3).mean())))
            ku.append(float((z ** 4).mean()) - 3.0)
            w.append(int(m.sum()))
        w = np.asarray(w, float)
        if w.sum() > 0:
            w /= w.sum()
        out[name] = {
            "slope_per_cell": slope,
            "skew": float(np.average(sk, weights=w)) if len(sk) else float("nan"),
            "excess_kurtosis": float(np.average(ku, weights=w)) if len(ku) else float("nan"),
            "n_cells": len(sk),
        }
    return out


@torch.inference_mode()
def pit_by_kt_cell(model, geom, batch) -> list[dict]:
    """`dv`'s PIT, stratified by the cell's `ln kt` index — measurement 1."""
    out = model.coordinate_cdfs(batch)
    u, mask = out["u"].numpy(), out["mask"].numpy()
    iv = (batch["yc"].numpy() % geom.n_bins)[mask]
    dv = u[..., 1][mask]
    rows = [{"scope": "all", "n": int(dv.size), "ks": ks(dv),
             "ratio": ks(dv) / critical(dv.size), "mean_pit": float(dv.mean())}]
    for cell in range(geom.n_bins):
        m = iv == cell
        if m.sum() < 100:
            continue
        rows.append({
            "scope": f"ln_kt cell {cell} [{cell * geom.cell_wv:.1f}, "
                     f"{(cell + 1) * geom.cell_wv:.1f})",
            "touches_kt_floor": cell == 0,
            "n": int(m.sum()), "ks": ks(dv[m]),
            "ratio": ks(dv[m]) / critical(int(m.sum())),
            "mean_pit": float(dv[m].mean()),
        })
    return rows


@torch.inference_mode()
def tilt_budget(model, batch, needs: dict) -> dict:
    """What tilt each head can actually produce — measurement 3, and the decisive one.

    A truncated normal on `[-h, h]` has log-density slope `mu/sigma^2`, so across the cell
    it tilts by `2h*mu/sigma^2`. `mu` is clamped to `+-h` by `h*tanh`, so once `sigma`
    grows past ~`sqrt(2)h` the achievable tilt falls below what a modestly sloped density
    needs and NO setting of the two parameters recovers it. A spline has no such coupling:
    it can be flat and tilted at once."""
    L = batch["yc"].shape[1]
    e = model.encode(batch["xf"], batch["nx"])
    out = model._decode_states(batch["yc"], e, model.xattn_kv(batch["xf"], batch["nx"]))
    eh = torch.cat([out[:, :L, :], e.unsqueeze(1).expand(-1, L, -1)], dim=-1)
    p = model._coord_params(torch.cat([eh, model.y_embed(batch["yc"].clamp(min=0))], dim=-1))
    m = torch.arange(L).unsqueeze(0) < batch["ny"].unsqueeze(1)
    res = {}
    for name, mean, sig, h in (("du", p.du_mean, p.du_sig, model.half_u),
                               ("dv", p.dv_mean, p.dv_sig, model.half_v)):
        if mean is None:  # this head is a spline on this arm — the coupling does not apply
            res[name] = {"parameterization": "spline", "note": "flat and tilted at once"}
            continue
        mu, s = mean[m], sig[m]
        tilt = 2.0 * h * mu.abs() / s ** 2
        res[name] = {
            "parameterization": "truncnorm",
            "sigma_over_h": float((s / h).median()),
            "abs_mean_over_h": float((mu.abs() / h).median()),
            "frac_mean_pinned_at_bound": float((mu.abs() > 0.95 * h).float().mean()),
            "achievable_tilt_median": float(tilt.median()),
            "achievable_tilt_p90": float(tilt.quantile(0.9)),
            "tilt_needed": needs[name]["slope_per_cell"],
            "sufficient": bool(float(tilt.median()) >= needs[name]["slope_per_cell"]),
        }
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-root", default="runs/lnz_spline")
    p.add_argument("--arms", default="spline_s0,spline_s1,spline_s2")
    p.add_argument("--test-path", default="data/jet_aux_asym_test.root")
    p.add_argument("--n-jets", type=int, default=2000)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    root = REPO / args.run_root
    all_jets = [j for j in load_rntuple(str(REPO / args.test_path), "Jets") if len(j["x"][0])]
    jets = all_jets[: args.n_jets]      # the model pass; the truth shape uses every jet
    report = {"run": {"run_root": args.run_root, "arms": args.arms,
                      "test_path": args.test_path, "n_jets": len(jets)}, "arms": {}}

    for arm in args.arms.split(","):
        model, geom = load_arm(arm, root)
        if model is None:
            print(f"[diag] {arm}: no checkpoint under {root / arm} — skipped")
            continue
        ds = MatchedLundDataset(jets, geom, aux_features=tuple(model.aux_feature_names))
        batch = collate([ds[i] for i in range(len(ds))])
        needs = truth_shape(all_jets, geom)
        rows = pit_by_kt_cell(model, geom, batch)
        budget = tilt_budget(model, batch, needs)
        report["arms"][arm] = {"pit_by_kt_cell": rows, "tilt_budget": budget}
        report["truth_shape"] = needs

        print(f"\n===== {arm} =====")
        print(f"  1. dv PIT by ln kt cell   (overall {rows[0]['ratio']:.2f}x on n={rows[0]['n']})")
        print(f"     {'cell':<28}{'n':>7}{'KS':>9}{'ratio':>8}{'mean PIT':>10}")
        for r in rows[1:]:
            edge = "  <- touches kt_floor" if r.get("touches_kt_floor") else ""
            print(f"     {r['scope']:<28}{r['n']:>7}{r['ks']:>9.4f}{r['ratio']:>7.2f}x"
                  f"{r['mean_pit']:>10.3f}{edge}")
        print("  2. what the truth asks of the offset head")
        for k, v in needs.items():
            print(f"     {k}: |dln p per cell| = {v['slope_per_cell']:.3f}   "
                  f"|skew| = {v['skew']:.3f}   excess kurtosis = {v['excess_kurtosis']:+.3f}"
                  f"   ({v['n_cells']} cells)")
        print("  3. the tilt budget  (truncated normal: 2h*mu/sigma^2, mu clamped to +-h)")
        for k, v in budget.items():
            if v["parameterization"] == "spline":
                print(f"     {k}: spline — {v['note']}")
                continue
            verdict = "SUFFICIENT" if v["sufficient"] else "*** INSUFFICIENT ***"
            print(f"     {k}: sigma = {v['sigma_over_h']:.1f}h   mean pinned at the bound "
                  f"{v['frac_mean_pinned_at_bound']:.3f}   achievable tilt "
                  f"{v['achievable_tilt_median']:.3f} vs needed {v['tilt_needed']:.3f}"
                  f"   {verdict}")

    if not report["arms"]:
        print("[diag] nothing to measure.")
        return 2
    out = Path(args.out) if args.out else root / "offset_head_diagnostic.json"
    save_metrics(report, out)
    print(f"\n[diag] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
