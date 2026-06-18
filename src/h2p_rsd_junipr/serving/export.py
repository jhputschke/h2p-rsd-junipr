"""Export (§12). Caveat first: autoregressive sampling and beam search contain
Python control flow, so `torch.jit.trace` is wrong (it captures one branch).
Export the *encoder* (and per-step heads) via `torch.jit.script`/ONNX and keep
beam search / sampling as a thin Python loop around the scripted step. Always
`model.eval()` + `no_grad()` and verify with `torch.allclose`.
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_encoder_torchscript(model, out_path: Path, example=None, verify: bool = True):
    """Export the encoder submodule via TorchScript and verify parity against
    eager. The encoder has no data-dependent Python control flow (just GRU +
    masked pooling + a linear), so tracing is correct here — unlike the
    autoregressive sampling / beam search, which must stay a Python loop (§12).
    `torch.jit.trace` is used because the package's `from __future__ import
    annotations` stringizes hints that `torch.jit.script` cannot resolve."""
    if example is None:
        raise ValueError("an (xf, nx) example is required to trace the encoder")
    model.eval()
    enc = model.encoder_net
    xf, nx = example
    with torch.no_grad():
        traced = torch.jit.trace(enc, (xf, nx), check_trace=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    if verify:
        with torch.no_grad():
            a = enc(xf, nx)
            b = traced(xf, nx)
        if not torch.allclose(a, b, atol=1e-5):
            raise RuntimeError("traced encoder does not match eager (allclose failed)")
    return str(out_path)


def export_encoder_onnx(model, out_path: Path, example):
    """ONNX export of the encoder with dynamic sequence length."""
    model.eval()
    enc = model.encoder_net
    xf, nx = example
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        enc, (xf, nx), str(out_path),
        input_names=["xf", "nx"], output_names=["ctx"],
        dynamic_axes={"xf": {0: "batch", 1: "seq"}, "nx": {0: "batch"}},
        opset_version=17,
    )
    return str(out_path)
