"""Batched ancestral sampling (was `sample_batch`).

`ancestral_sample_cells` is the family-agnostic batched on-device sampler (single
host sync at the end), parameterised by a batched step
`step(tok, e, h) -> (p_cont (K,), split_logits (K, n_cells), h)`. §5.1 ancestral
sampling uses it directly; §5.2 flow inverse / §5.3 reverse-SDE provide their own
`sample` returning the same posterior-draw structure.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

BatchedStep = Callable[[torch.Tensor, torch.Tensor, object], tuple]


@torch.inference_mode()
def ancestral_sample_cells(
    step: BatchedStep,
    e: torch.Tensor,
    h0,
    start_token: int,
    n_samples: int,
    device: torch.device,
    max_emissions: int = 25,
    cont_temperature: float = 1.0,
) -> list[list[int]]:
    """Draw `n_samples` posterior CELL chains y ~ q_phi(.|x) for ONE jet in
    parallel. `cont_temperature` is the sampling-time exposure-bias remedy
    (softmax temperature on the cell logits; never touches the trained likelihood)."""
    K = n_samples
    tok = torch.full((K, 1), start_token, dtype=torch.long, device=device)
    alive = torch.ones(K, dtype=torch.bool, device=device)
    cells = torch.zeros(K, max_emissions, dtype=torch.long, device=device)
    emitted = torch.zeros(K, max_emissions, dtype=torch.bool, device=device)
    h = h0
    for t in range(max_emissions):
        p_cont, split_logits, h = step(tok, e, h)  # (K,), (K, n_cells)
        cont = (torch.rand(K, device=device) < p_cont) & alive
        probs = F.softmax(split_logits / cont_temperature, dim=-1)
        draw = torch.multinomial(probs, 1).squeeze(-1)
        cells[:, t] = draw
        emitted[:, t] = cont
        alive = cont
        tok = draw.unsqueeze(1)
        if not bool(alive.any()):
            break
    cells_np = cells.cpu().numpy()
    emitted_np = emitted.cpu().numpy()
    return [cells_np[k, emitted_np[k]].tolist() for k in range(K)]
