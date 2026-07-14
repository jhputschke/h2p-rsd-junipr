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


CellStep = Callable[[torch.Tensor, torch.Tensor, object], tuple]


@torch.inference_mode()
def ancestral_sample_cells_fixed_length(
    step_cells: CellStep,
    e: torch.Tensor,
    h0,
    start_token: int,
    lengths,
    device: torch.device,
    cont_temperature: float = 1.0,
) -> list[list[int]]:
    """Draw one CELL chain per prescribed length, for ONE jet in parallel.

    Companion to `ancestral_sample_cells` for the first-class factorization
    q(y|x) = q(N|x) q(y|N,x): the caller has already drawn per-chain lengths
    `lengths[k] = N_k ~ q(N|x)`, so there is NO continue/stop draw — chain `k`
    emits cells while `t < N_k` and is inert afterwards. `step_cells(tok, e, h)
    -> (split_logits (K, n_cells), h)` is the cont_head-free decoder step.
    `cont_temperature` keeps the same softmax-temperature meaning on the cell logits."""
    lengths_t = torch.as_tensor(list(lengths), dtype=torch.long, device=device)
    K = int(lengths_t.shape[0])
    L = int(lengths_t.max().item()) if K > 0 else 0
    if K == 0 or L == 0:
        return [[] for _ in range(K)]
    tok = torch.full((K, 1), start_token, dtype=torch.long, device=device)
    cells = torch.zeros(K, L, dtype=torch.long, device=device)
    emitted = torch.arange(L, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)  # (K, L)
    h = h0
    for t in range(L):
        split_logits, h = step_cells(tok, e, h)  # (K, n_cells)
        probs = F.softmax(split_logits / cont_temperature, dim=-1)
        draw = torch.multinomial(probs, 1).squeeze(-1)
        cells[:, t] = draw
        tok = draw.unsqueeze(1)
    cells_np = cells.cpu().numpy()
    emitted_np = emitted.cpu().numpy()
    return [cells_np[k, emitted_np[k]].tolist() for k in range(K)]
