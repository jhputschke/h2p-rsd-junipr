"""Which leading-emission point estimator actually beats plain RSD, and is the answer
limited by the estimator or by the information in x?

`run_closure` reported `dlund_posterior_mode` >= `dlund_identity` on the walkthrough
`ar_junipr_v3` checkpoint — the model tying, or losing to, using the hadron sequence
unchanged. That looked like a model failure on the most perturbative observable in the
jet. It was not. Two decode artifacts, both measurable here:

  1. WRONG ESTIMATOR. The modal leading cell minimises expected 0-1 loss, but the
     score is `lund_distance`. The loss-matched choice over the same support is the
     MEDOID (`eval.closure.medoid_cell`) — MBR applied to a one-node cloud.
  2. QUANTISATION. At the default geometry a cell is ~0.6 wide and the distances are
     ~0.6, so the cell-level metric is mostly measuring the grid. Off the grid
     (`sample_coordinates` + `eval.closure.geometric_median`) it resolves.

Measured, 2000 val jets / K=200 / ar_junipr_v3, ratio to identity(x):

    cell level    mode 1.030   medoid 0.944
    continuous    mode 1.074   geo-median 0.905   (95% CI [0.882, 0.928])

So the model beats plain RSD on leading-emission position by ~10%, decisively. What it
does NOT do is win everywhere: stratified by truth leading ln kt, it wins the hard and
middle thirds and LOSES the soft third — the reverse of the usual expectation. A
calibrated, fully-conditional posterior yields a Bayes estimator that dominates any
function of x, including the identity, so losing there says the model is
under-conditioning in the soft/wide-angle corner. That corner is also where
`run_calibration`'s region-stratified coverage is worst, which is the same finding
arriving twice.

`run_closure` now reports the medoid unconditionally and the continuous row under
`experiment.closure_continuous=true`. This script is the study behind that change: it
adds the oracle, the per-jet win rates, the kt stratification and a paired bootstrap,
none of which belong in the per-eval metric dict.

Run:
  python scripts/leading_estimators.py runs/<...>/best.ckpt
  python scripts/leading_estimators.py runs/<...>/best.ckpt --jets 2000 --draws 200
  python scripts/leading_estimators.py runs/<...>/best.ckpt --out runs/<...>   # + json/md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from h2p_rsd_junipr.data.datamodule import LundDataModule
from h2p_rsd_junipr.eval.closure import (
    geometric_median,
    leading_emission_cell,
    lund_distance,
    medoid_cell,
)
from h2p_rsd_junipr.features import node_raw
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model
from h2p_rsd_junipr.train.checkpoint import load_for_inference
from h2p_rsd_junipr.train.trainer import seed_everything


def _leading(arr):
    a = np.asarray(arr, dtype=float)
    return None if (a.ndim != 2 or not a.shape[0]) else a[int(np.argmax(a[:, 1])), :2]


def collect(model, val_ds, jets, geom, device, n_jets, K, n_cont, draws_by_jet=None):
    """One pass. Every estimator for a jet shares that jet's draws, so all the
    comparisons below are exactly paired — but the RNG stream differs from a run
    without the continuous branch, so absolute numbers are not comparable across
    scripts. Ratios are.

    `draws_by_jet` extends that sharing ACROSS callers: a notebook that already drew
    K posterior samples per jet for another section passes them here rather than
    paying for a second pass (docs/PLAN_prod_test_speedup.md §4). None re-samples,
    which is what the CLI below does."""
    cell_rows, cont_rows = [], []
    with torch.inference_mode():
        for i in range(min(n_jets, len(val_ds))):
            item = val_ds[i]
            xf = item["xf"].unsqueeze(0).to(device)
            nx = torch.tensor([item["nx"]], device=device)
            ly = leading_emission_cell(item["yc"].tolist(), geom)
            x_cells = geom.seq_cells(jets[i]["x"][0], jets[i]["x"][1]).tolist()

            draws = model.sample_batch(xf, nx, K) if draws_by_jet is None else draws_by_jet[i]
            lead = [c for c in (leading_emission_cell(d, geom) for d in draws) if c is not None]
            if ly is None or not lead:
                continue                       # the selection run_closure applies
            vals, counts = np.unique(np.asarray(lead), return_counts=True)
            mode_cell = int(vals[counts.argmax()])
            med_cell = medoid_cell(lead, geom)
            yraw = np.asarray(item["yraw"].numpy(), dtype=float)
            cell_rows.append((
                lund_distance(leading_emission_cell(x_cells, geom), ly, geom),
                lund_distance(mode_cell, ly, geom),
                lund_distance(med_cell, ly, geom),
                min(lund_distance(int(c), ly, geom) for c in lead),   # oracle
                float(yraw[int(np.argmax(yraw[:, 1])), 1]),           # truth leading ln kt
                float(mode_cell == ly), float(med_cell == ly),
            ))

            if len(cont_rows) >= n_cont:
                continue
            pts = []
            # one batched call per jet, not one per draw: the per-draw hook re-encodes
            # the same jet K times (docs/PLAN_prod_test_speedup.md §2)
            for c in model.sample_coordinates_many(xf, nx, [list(d) for d in draws if len(d)]):
                if c is None:
                    return np.array(cell_rows), np.zeros((0, 5))   # no coordinate density
                p = _leading(c.detach().cpu().double().numpy().reshape(-1, 4))
                if p is not None:
                    pts.append(p)
            y_lead, x_lead = _leading(yraw), _leading(node_raw(*jets[i]["x"]))
            if len(pts) >= 2 and y_lead is not None and x_lead is not None:
                P = np.asarray(pts)
                cont_rows.append((
                    float(np.linalg.norm(x_lead - y_lead)),
                    float(np.linalg.norm(np.asarray(geom.cell_center(mode_cell)) - y_lead)),
                    float(np.linalg.norm(geometric_median(P) - y_lead)),
                    float(np.linalg.norm(P.mean(0) - y_lead)),
                    float(np.min(np.linalg.norm(P - y_lead, axis=1))),
                ))
    return np.array(cell_rows), np.array(cont_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--jets", type=int, default=2000, help="cell-level jets")
    ap.add_argument("--draws", type=int, default=200, help="posterior draws per jet")
    ap.add_argument("--cont-jets", type=int, default=1200,
                    help="jets for the continuous pass (K forward passes each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write leading_estimators.{json,md} here")
    a = ap.parse_args()

    seed_everything(a.seed)
    device = torch.device("cpu")     # one jet at a time; a GPU never amortises dispatch
    info = load_for_inference(a.ckpt, map_location=device)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(device)
    model.load_state_dict(info["model_state"])
    model.eval()
    dm = LundDataModule(cfg, geom).setup()
    _, val_ds = dm.datasets()

    print(f"checkpoint {a.ckpt}\n{info['model_name']}, geometry n_bins={geom.n_bins}, "
          f"K={a.draws} draws")
    R, C = collect(model, val_ds, dm.val_jets, geom, device, a.jets, a.draws, a.cont_jets)

    out: dict = {"checkpoint": str(a.ckpt), "model": info["model_name"],
                 "n_draws": a.draws, "seed": a.seed}
    lines: list[str] = []

    def emit(s=""):
        print(s)
        lines.append(s)

    base = float(np.nanmean(R[:, 0]))
    emit(f"\n=== cell level, {len(R)} jets with a truth leading emission ===")
    emit(f"{'estimator':<26}{'mean d':>9}{'median d':>10}{'ratio':>8}{'exact':>8}")
    cell_names = ["identity(x)", "mode (was reported)", "medoid (loss-matched)", "oracle"]
    exact = [None, float(R[:, 5].mean()), float(R[:, 6].mean()), None]
    for j, nm in enumerate(cell_names):
        m = float(np.nanmean(R[:, j]))
        emit(f"{nm:<26}{m:>9.4f}{float(np.nanmedian(R[:, j])):>10.4f}{m / base:>8.3f}"
             f"{('--' if exact[j] is None else f'{exact[j]:.1%}'):>8}")
        out[f"cell_{nm.split()[0]}"] = {"mean": m, "ratio": m / base, "exact": exact[j]}
    ok = np.isfinite(R[:, 0])
    emit(f"\nbeats identity per jet:  medoid {np.mean(R[ok, 2] < R[ok, 0]):.1%}"
         f"   mode {np.mean(R[ok, 1] < R[ok, 0]):.1%}   [{int(ok.sum())} jets with non-empty x]")
    emit("the oracle is min over draws, so it measures whether the SUPPORT covers the "
         "truth\n(it nearly always does) -- it is not an achievable point-estimate ceiling.")

    emit("\n--- stratified by truth leading ln kt (thirds) ---")
    qs = np.quantile(R[:, 4], [0.0, 1 / 3, 2 / 3, 1.0])
    emit(f"{'ln kt bin':<20}{'jets':>6}{'identity':>10}{'mode':>10}{'medoid':>10}{'medoid/id':>11}")
    strat = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (R[:, 4] >= lo) & (R[:, 4] <= hi)
        idm, mdm = float(np.nanmean(R[m, 0])), float(np.nanmean(R[m, 2]))
        emit(f"[{lo:5.2f}, {hi:5.2f}]{'':<7}{int(m.sum()):>6}{idm:>10.4f}"
             f"{float(np.nanmean(R[m, 1])):>10.4f}{mdm:>10.4f}{mdm / idm:>11.3f}")
        strat.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                      "identity": idm, "medoid": mdm, "ratio": mdm / idm})
    out["by_leading_lnkt"] = strat
    emit("a ratio > 1 in the SOFT bin is the signature to chase: a calibrated, fully "
         "conditional\nposterior beats any function of x, so losing to identity there "
         "means the model is\nunder-conditioning in the soft/wide-angle corner.")

    if len(C):
        cbase = float(C[:, 0].mean())
        emit(f"\n=== continuous, no cell quantisation, {len(C)} jets ===")
        cont_names = ["identity(x)", "mode cell centre", "geo-median", "mean", "oracle"]
        for j, nm in enumerate(cont_names):
            m = float(C[:, j].mean())
            emit(f"{nm:<26}{m:>9.4f}{m / cbase:>8.3f}")
            out[f"cont_{nm.split()[0]}"] = {"mean": m, "ratio": m / cbase}
        rng = np.random.default_rng(a.seed)
        b = np.array([(lambda s: C[s, 2].mean() / C[s, 0].mean())(
            rng.integers(0, len(C), len(C))) for _ in range(4000)])
        lo, hi = (float(v) for v in np.quantile(b, [0.025, 0.975]))
        emit(f"\ngeo-median beats identity on {np.mean(C[:, 2] < C[:, 0]):.1%} of jets")
        emit(f"paired bootstrap of geo-median/identity: {cbase and C[:, 2].mean() / cbase:.3f}"
             f"  95% CI [{lo:.3f}, {hi:.3f}]  P(ratio<1) = {np.mean(b < 1):.3f}")
        out["bootstrap"] = {"ratio": float(C[:, 2].mean() / cbase), "ci95": [lo, hi],
                            "p_ratio_lt_1": float(np.mean(b < 1)), "n_boot": 4000}
    else:
        emit("\n(no continuous rows: this family has no coordinate density)")

    if a.out:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "leading_estimators.json").write_text(json.dumps(out, indent=2))
        (d / "leading_estimators.md").write_text(
            "# Leading-emission point estimators\n\n```\n" + "\n".join(lines) + "\n```\n")
        print(f"\n[leading_estimators] wrote {d/'leading_estimators.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
