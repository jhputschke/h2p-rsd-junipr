"""FastAPI service (§12): x -> {MAP tree, posterior summary}. Loaded via
`load_for_inference`. FastAPI is an optional `[serve]` dependency, imported lazily
so the core package never requires it.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import OmegaConf, decode_params
from ..features import node_features
from ..geometry import Geometry
from ..inference.length import learned_min_emissions
from ..models.base import build_model
from ..train.checkpoint import load_for_inference


def load_service_model(ckpt_path: str, device: torch.device):
    info = load_for_inference(ckpt_path, map_location=device)
    cfg = OmegaConf.create(info["config"])
    geometry = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geometry).to(device)
    model.load_state_dict(info["model_state"])
    model.eval()
    return model, geometry


def predict(model, geometry, device, x_seq, decode: dict | None = None) -> dict:
    """x_seq: dict with lists lnInvDelta, lnkt, lnz, psi -> point estimate + posterior
    summary. `decode` is a decode_params(cfg) dict; when None the model defaults apply
    (the MAP floor min_emissions=1 still holds via the method signature).

    The point estimate is selected by `decode.point_estimator` (default "map"): "map"
    is the beam-search MAP (with the learned floor below); "mbr" is the
    minimum-Bayes-risk tree of least expected perturbative-Lund EMD to the posterior
    (`inference.mbr`), which needs no floor. Either way the posterior draws are taken
    once and reused (no double-sample), and the response adds `mbr_risk`/`mbr_backend`
    only under MBR (additive, non-breaking).

    When `decode.length_floor_quantile > 0` (MAP only) the MAP multiplicity is floored
    per jet at the learned quantile of P(n|x). alpha=0 (the default) short-circuits and
    the MAP is identical to today's hard-floored beam."""
    dec = dict(decode or {})
    xf = torch.tensor(
        node_features(x_seq["lnInvDelta"], x_seq["lnkt"], x_seq["lnz"], x_seq["psi"])
    ).unsqueeze(0).to(device)
    nx = torch.tensor([xf.shape[1]], device=device)
    draws = model.sample_batch(xf, nx, int(dec.get("n_posterior_samples", 200)))
    mults = np.array([len(d) for d in draws])
    is_mbr = str(dec.get("point_estimator", "map")) == "mbr"
    alpha = float(dec.get("length_floor_quantile", 0.0))
    if not is_mbr and alpha > 0.0:  # learned per-jet floor reuses the draws (no double-sample)
        dec["min_emissions"] = learned_min_emissions(
            model, xf, nx, quantile=alpha,
            base_floor=int(dec.get("min_emissions", 1)), mults=mults,
        )
    y_hat = model.map_or_mbr(xf, nx, draws=draws, **dec)  # MAP or MBR, reusing draws
    out = {
        "map_multiplicity": y_hat.multiplicity,
        "map_logprob": y_hat.logprob,
        "map_nodes": [
            {"cell": n.cell, "ln_invDelta": n.ln_invDelta, "ln_kt": n.ln_kt,
             "ln_z": n.ln_z, "psi": n.psi} for n in y_hat.nodes
        ],
        "posterior_mult_mean": float(mults.mean()),
        "posterior_mult_median": float(np.median(mults)),
        "posterior_mult_std": float(mults.std()),
        "posterior_mult_68CR": [float(np.percentile(mults, 16)), float(np.percentile(mults, 84))],
    }
    if is_mbr:  # decision-theoretic score (NOT a likelihood) + its backend tag
        out["mbr_risk"] = float(y_hat.risk) if y_hat.risk is not None else float("nan")
        out["mbr_backend"] = str(dec.get("mbr_backend", "pot"))
    return out


def create_app(ckpt_path: str):
    from fastapi import FastAPI  # optional [serve] dependency
    from pydantic import BaseModel

    device = torch.device("cpu")
    model, geometry = load_service_model(ckpt_path, device)
    decode = decode_params(OmegaConf.create(load_for_inference(ckpt_path, map_location=device)["config"]))
    app = FastAPI(title="h2p-rsd-junipr")

    class LundSeq(BaseModel):
        lnInvDelta: list[float]
        lnkt: list[float]
        lnz: list[float]
        psi: list[float]

    @app.post("/predict")
    def _predict(seq: LundSeq):
        return predict(model, geometry, device, seq.model_dump(), decode=decode)

    return app
