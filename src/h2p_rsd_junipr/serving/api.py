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
    """x_seq: dict with lists lnInvDelta, lnkt, lnz, psi -> MAP + posterior summary.
    `decode` is a decode_params(cfg) dict; when None the model defaults apply (the
    MAP floor min_emissions=1 still holds via the method signature)."""
    dec = decode or {}
    xf = torch.tensor(
        node_features(x_seq["lnInvDelta"], x_seq["lnkt"], x_seq["lnz"], x_seq["psi"])
    ).unsqueeze(0).to(device)
    nx = torch.tensor([xf.shape[1]], device=device)
    y_hat = model.map_estimate(xf, nx, **dec)  # map_estimate keeps only the beam keys
    draws = model.sample_batch(xf, nx, int(dec.get("n_posterior_samples", 200)))
    mults = np.array([len(d) for d in draws])
    return {
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
