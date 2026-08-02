"""Latent-alignment dynamic programming for the edit transducer (docs/PLAN_EditTransducer.md).

Pure tensor functions -- **no `nn.Module`, no model state** -- so the numerics can be
pinned in isolation against explicit path enumeration (`tests/test_edit_dp.py`). The
model file owns every parameter; this file owns only the recursion.

The lattice is the RNN-T one (Graves, arXiv:1211.3711). A state `(i, j)` means `i`
hadron nodes of `x` consumed and `j` parton nodes of `y` emitted; the two outgoing
edges are

    STAY   (i, j) -> (i+1, j)     weight `log_stay[i, j]`   (ADVANCE, or STOP at i = nx)
    EMIT   (i, j) -> (i, j+1)     weight `log_emit_edge[i, j]`

and a complete path runs from `(0, 0)` to `(nx, ny)` and then takes the STOP branch,
which is the *same* categorical slot as ADVANCE evaluated at the terminal column `i = nx`
-- that identification is what makes `sum_y q(y|x) = 1` hold with no bespoke
normalization argument. `log_emit_edge` carries the op log-probability AND the emission
log-density together, so the same recursion serves the likelihood (density edges), the
exact length marginal (structural edges only) and the constrained cell-conditioned
posterior (cell-mass edges); the caller decides which by what it puts in the edges.

Everything is log-domain and batched over jets with per-jet `(nx, ny)`. Padding beyond a
jet's own terminal is never *read* -- the recursion is strictly forward, so lattice
entries at `i > nx` or `j > ny` cannot reach `(nx, ny)` -- so no masking is needed for
`forward_*`; `backward_beta` does gate explicitly, because it starts at the terminal.

`NEG` is a finite stand-in for `-inf`: `exp(NEG)` underflows to 0 exactly, while
`logaddexp` and `+` stay off the `inf - inf` / `0 * inf` branches that would put NaN into
a gradient.
"""

from __future__ import annotations

import torch

# Finite -inf. Every recursion clamps back to it, so a chain of impossible transitions
# can never accumulate its way to -inf (and thence to NaN in the backward pass).
NEG = -1e9


def _zeros(like: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, device=like.device, dtype=like.dtype)


def _full_neg(like: torch.Tensor, n: int) -> torch.Tensor:
    return torch.full((n,), NEG, device=like.device, dtype=like.dtype)


def _check(log_stay: torch.Tensor, log_emit_edge: torch.Tensor) -> tuple[int, int, int]:
    B, n_col, J = log_stay.shape
    if log_emit_edge.shape != (B, n_col, J - 1):
        raise ValueError(
            f"log_emit_edge must be (B, n_col, J-1) = {(B, n_col, J - 1)}, got {tuple(log_emit_edge.shape)}"
        )
    return B, n_col, J


def forward_alpha(log_stay: torch.Tensor, log_emit_edge: torch.Tensor) -> torch.Tensor:
    """`alpha[b, i, j]` = log-sum over monotone paths from `(0,0)` to `(i,j)`.

    `log_stay` is `(B, n_col, J)` and `log_emit_edge` is `(B, n_col, J-1)`; the result is
    `(B, n_col, J)`. `O(n_col*J)` sequential steps, each vectorised over the batch."""
    B, n_col, J = _check(log_stay, log_emit_edge)
    a: list[list[torch.Tensor]] = [[None] * J for _ in range(n_col)]  # type: ignore[list-item]
    a[0][0] = _zeros(log_stay, B)
    for i in range(n_col):
        for j in range(J):
            if i == 0 and j == 0:
                continue
            if i == 0:
                v = a[i][j - 1] + log_emit_edge[:, i, j - 1]
            elif j == 0:
                v = a[i - 1][j] + log_stay[:, i - 1, j]
            else:
                v = torch.logaddexp(
                    a[i - 1][j] + log_stay[:, i - 1, j],
                    a[i][j - 1] + log_emit_edge[:, i, j - 1],
                )
            a[i][j] = v.clamp(min=NEG)
    return torch.stack([torch.stack(row, dim=1) for row in a], dim=1)  # (B, n_col, J)


def forward_logsumexp(
    log_stay: torch.Tensor, log_emit_edge: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor
) -> torch.Tensor:
    """`(B,)` log q(y|x) = logsumexp over every monotone alignment, plus the terminal STOP.

    This is the family's entire likelihood: the alignment is a latent variable and is
    *marginalised*, never supervised (node-level parton<->hadron correspondence is not
    observable -- cf. HOMER, arXiv:2410.06342)."""
    alpha = forward_alpha(log_stay, log_emit_edge)
    b = torch.arange(alpha.shape[0], device=alpha.device)
    return alpha[b, nx, ny] + log_stay[b, nx, ny]


def backward_beta(
    log_stay: torch.Tensor, log_emit_edge: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor
) -> torch.Tensor:
    """`beta[b, i, j]` = log-prob of completing `(i,j)` -> terminal, STOP included.

    Unlike `forward_alpha` this one *must* mask: the recursion starts at the per-jet
    terminal `(nx, ny)`, so a padded lattice entry would otherwise inherit a path that
    does not exist for that jet. Entries with `i > nx` or `j > ny` come back at `NEG`,
    which is what makes the responsibilities below vanish there without extra masking."""
    B, n_col, J = _check(log_stay, log_emit_edge)
    beta: list[list[torch.Tensor]] = [[None] * J for _ in range(n_col)]  # type: ignore[list-item]
    neg = _full_neg(log_stay, B)
    for i in reversed(range(n_col)):
        for j in reversed(range(J)):
            terms = []
            if i + 1 < n_col:
                terms.append(
                    torch.where(nx > i, log_stay[:, i, j] + beta[i + 1][j], neg)
                )
            if j + 1 < J:
                terms.append(
                    torch.where(ny > j, log_emit_edge[:, i, j] + beta[i][j + 1], neg)
                )
            if not terms:
                v = neg
            elif len(terms) == 1:
                v = terms[0]
            else:
                v = torch.logaddexp(terms[0], terms[1])
            # the terminal itself: nothing left to do but STOP
            v = torch.where((nx == i) & (ny == j), log_stay[:, i, j], v)
            beta[i][j] = v.clamp(min=NEG)
    return torch.stack([torch.stack(row, dim=1) for row in beta], dim=1)


def forward_backward_responsibilities(
    log_stay: torch.Tensor, log_emit_edge: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor
) -> dict:
    """Posterior over alignments given `(x, y)` -- the emergent-alignment readout.

    Returns `{"gamma_emit": (B, n_col, J-1), "gamma_stay": (B, n_col-1, J), "log_z": (B,)}`, where
    `gamma_emit[b, i, t]` is the posterior probability that `y_t` was emitted from lattice
    column `i` and `gamma_stay[b, i, j]` that the path advanced past hadron node `i` with
    `j` nodes already emitted. `gamma_emit` sums to exactly 1 over `i` for every
    `t < ny` -- each parton node is emitted once, from somewhere.

    Nothing here is supervised: these are read *off* the fitted likelihood, which is the
    only honest way to get a parton<->hadron alignment out of data that never records
    one."""
    alpha = forward_alpha(log_stay, log_emit_edge)
    beta = backward_beta(log_stay, log_emit_edge, nx, ny)
    b = torch.arange(alpha.shape[0], device=alpha.device)
    log_z = alpha[b, nx, ny] + log_stay[b, nx, ny]
    z = log_z[:, None, None]
    J = log_stay.shape[2]
    gamma_emit = torch.exp(
        alpha[:, :, : J - 1] + log_emit_edge + beta[:, :, 1:] - z
    )
    gamma_stay = torch.exp(
        alpha[:, :-1, :] + log_stay[:, :-1, :] + beta[:, 1:, :] - z
    )
    return {"gamma_emit": gamma_emit, "gamma_stay": gamma_stay, "log_z": log_z}


def structural_length_pmf(
    log_stay: torch.Tensor, log_emit: torch.Tensor, nx: torch.Tensor, max_n: int
) -> torch.Tensor:
    """Exact `q(N = n | x)` for `n = 0..max_n`, `(B, max_n+1)`, with NO extra parameters.

    Marginalising the coordinates out of `forward_logsumexp` is free: every emission
    density integrates to 1, so what is left is the *structural* DP over `(i, j)` with the
    op log-probabilities alone on the edges. The terminal value at `(nx, n)` is therefore
    the exact length marginal -- what `PLAN_MultHead.md`'s learned `n_head` approximates,
    but conditioned on `|x|` and with nothing fitted.

    This is exact only while the op probabilities are free of the emitted prefix, which is
    why the op head never sees it in either stage (see `models/edit.py`). `log_stay` /
    `log_emit` are `(B, n_col)` for that reason. The returned rows are renormalised: mass on
    `n > max_n` exists and is simply not represented."""
    B, n_col = log_stay.shape
    J = int(max_n) + 1
    stay = log_stay[:, :, None].expand(B, n_col, J)
    edge = log_emit[:, :, None].expand(B, n_col, J - 1)
    alpha = forward_alpha(stay, edge)
    b = torch.arange(B, device=alpha.device)
    logp = alpha[b, nx, :] + log_stay[b, nx][:, None]  # (B, J)
    p = torch.exp(logp - logp.max(dim=-1, keepdim=True).values)
    return p / p.sum(dim=-1, keepdim=True).clamp(min=1e-30)


# ---------------------------------------------------------------------------
# per-jet decode helpers (inference only: one jet, python floats, no autograd)
# ---------------------------------------------------------------------------
def viterbi_path(
    log_stay: torch.Tensor,
    log_emit_edge: torch.Tensor,
    nx: int,
    min_n: int = 0,
    one_per_column: bool = False,
) -> tuple[float, list[int]]:
    """Best monotone path, maximising over terminal lengths `n` in `[min_n, J-1]`.

    `log_stay` is `(n_col, J)` and `log_emit_edge` is `(n_col, J-1)` for ONE jet. Returns
    `(score, columns)` with `columns[t]` the lattice column that emitted `y_t`, so
    `len(columns)` is the decoded multiplicity.

    This is the MAP **surrogate**: the exact MAP is an argmax of a
    marginal-over-alignments and is intractable, so the estimate is the best single
    alignment rather than the best `y`. Its score is a lower bound on `log q(y|x)` for the
    same `y` (one path <= the sum over paths), which is the property the DP test pins.

    `one_per_column=True` restricts the search to alignments in which each HADRON column
    emits at most once (the terminal column, which is where trailing insertions come from,
    stays unlimited). That is the edit-distance semantics the family is built on — a
    hadron node is kept-and-smeared or deleted — and it is the same assumption the
    deletion-rate accounting makes. Without it the unrestricted argmax is degenerate
    whenever the emission parameters do not depend on `j`: the stay edges cost the same on
    every complete path, so the best path simply repeats the single best-scoring column
    `n` times. That degeneracy is a property of the decision rule, not of the fit, which
    is why it is fixed here in the decode rather than papered over in the model."""
    n_col = int(nx) + 1
    J = log_stay.shape[1]
    stay = log_stay[:n_col].tolist()
    edge = log_emit_edge[:n_col].tolist() if J > 1 else [[] for _ in range(n_col)]
    # slot 0 == "this column has not emitted yet"; slot 1 == "it has". Unused (and
    # unreachable) when one_per_column is False, so that path is the plain lattice.
    delta = [[[NEG, NEG] for _ in range(J)] for _ in range(n_col)]
    back: list[list[list[tuple[str, int, int, int] | None]]] = [
        [[None, None] for _ in range(J)] for _ in range(n_col)
    ]
    delta[0][0][0] = 0.0
    for j in range(J):
        for i in range(1, n_col):  # STAY edges within column j (i ascending); they reset the slot
            for f in (0, 1):
                cand = delta[i - 1][j][f] + stay[i - 1][j]
                if cand > delta[i][j][0]:
                    delta[i][j][0], back[i][j][0] = cand, ("stay", i - 1, j, f)
        if j + 1 < J:  # ...then seed column j+1 through the EMIT edges
            for i in range(n_col):
                dst = 1 if (one_per_column and i < n_col - 1) else 0
                cand = delta[i][j][0] + edge[i][j]
                if cand > delta[i][j + 1][dst]:
                    delta[i][j + 1][dst], back[i][j + 1][dst] = cand, ("emit", i, j, 0)

    lo = max(0, min(int(min_n), J - 1))
    slot = [0 if delta[n_col - 1][n][0] >= delta[n_col - 1][n][1] else 1 for n in range(J)]
    scores = [delta[n_col - 1][n][slot[n]] + stay[n_col - 1][n] for n in range(J)]
    n_star = max(range(lo, J), key=lambda n: scores[n])

    cols: list[int] = []
    i, j, f = n_col - 1, n_star, slot[n_star]
    while back[i][j][f] is not None:
        op, pi, pj, pf = back[i][j][f]  # type: ignore[misc]
        if op == "emit":
            cols.append(pi)
        i, j, f = pi, pj, pf
    cols.reverse()
    return float(scores[n_star]), cols


def sample_alignment_batch(
    log_stay: torch.Tensor,
    log_emit_edge: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`sample_alignment` for a whole BLOCK of draws at once -> `columns (B, J-1)`.

    Same walk, same conditional, `B` of them advancing in LOCKSTEP — the pattern
    `EditTransducer.sample` already uses for the ancestral walk. `log_stay` `(B, n_col, J)`,
    `log_emit_edge` `(B, n_col, J-1)`, `nx`/`ny` `(B,)`. Row `b`'s alignment is
    `columns[b, :ny[b]]`; entries past `ny[b]` are 0 and mean nothing.

    Why this exists. The per-draw `sample_alignment` is a python `while` of `nx + ny`
    steps, and `sample_coordinates` runs one per draw — so a K-draw posterior pays K walks
    plus K encoder passes plus K head evaluations. MEASURED, that call is ~6 ms and is
    almost entirely dispatch: its cost is FLAT in `L` (9.1 ms at L = 1, 13.3 ms at L = 100)
    and flat in `n_x`, i.e. it is ~100 tiny op launches on `L`-sized tensors rather than
    arithmetic. So the fix is not to make any one step cheaper — hoisting the encoder out
    of the loop is only 1.2x — but to stop paying the fixed cost K times, which is what
    `EditTransducer.sample_coordinates_many` does on top of this.

    The loop runs `max_b(nx_b + ny_b)` steps because each step advances exactly one of
    `i`, `j`; rows that reach their terminal early are held there by `done`. That makes
    the step count data-dependent but identical for every row, which is the property that
    lets the walk vectorise at all.

    **This reorders RNG consumption** relative to `sample_alignment`: one `rand(B)` per
    step instead of one `rand(())` per step per draw. The draws are from the same
    conditional and agree in distribution, but they are NOT the same draws — the same
    trade `ARJunipr.sample_coordinates_many` already makes, and the reason
    `tests/test_edit_batched_coords.py` checks agreement in DISTRIBUTION rather than
    bit-identity."""
    B, n_col, J = _check(log_stay, log_emit_edge)
    dev = log_stay.device
    L = J - 1
    cols = torch.zeros(B, max(L, 1), dtype=torch.long, device=dev)
    if L == 0 or B == 0:
        return cols[:, :L]
    nx = nx.to(device=dev, dtype=torch.long).expand(B)
    ny = ny.to(device=dev, dtype=torch.long).expand(B)
    beta = backward_beta(log_stay, log_emit_edge, nx, ny)
    b_idx = torch.arange(B, device=dev)
    i = torch.zeros(B, dtype=torch.long, device=dev)
    j = torch.zeros(B, dtype=torch.long, device=dev)
    for _ in range(int((nx + ny).max())):
        can_stay = i < nx
        can_emit = j < ny
        # every index is clamped BEFORE use: a finished row keeps walking through the
        # loop with `can_stay = can_emit = False`, and an unclamped `i + 1` at the
        # terminal column would index past the lattice rather than be masked away.
        ic, jc = i.clamp(max=n_col - 1), j.clamp(max=L - 1)
        w_stay = log_stay[b_idx, ic, jc] + beta[b_idx, (i + 1).clamp(max=n_col - 1), jc]
        w_emit = log_emit_edge[b_idx, ic, jc] + beta[b_idx, ic, (j + 1).clamp(max=J - 1)]
        u = torch.rand(B, device=dev, generator=generator)
        take = torch.where(can_stay & can_emit, u < torch.sigmoid(w_emit - w_stay),
                           can_emit)
        cols[b_idx, jc] = torch.where(take, i, cols[b_idx, jc])
        j = j + take.long()
        i = i + (can_stay & ~take).long()
    return cols[:, :L]


def sample_alignment(
    log_stay: torch.Tensor,
    log_emit_edge: torch.Tensor,
    nx: int,
    ny: int,
    *,
    generator: torch.Generator | None = None,
) -> list[int]:
    """One draw from `P(alignment | x, y)` -- the constrained forward-backward walk.

    `log_stay` `(n_col, J)`, `log_emit_edge` `(n_col, J-1)`, one jet. Returns `columns[t]` for
    `t = 0..ny-1`. Coordinates are NOT conditionally independent of the alignment given a
    cell chain, so `sample_coordinates` has to draw an alignment here first and only then
    the coordinates of the component it implies."""
    dev = log_stay.device
    nx_t = torch.tensor([int(nx)], device=dev)
    ny_t = torch.tensor([int(ny)], device=dev)
    beta = backward_beta(log_stay[None], log_emit_edge[None], nx_t, ny_t)[0]
    cols: list[int] = []
    i, j = 0, 0
    while not (i == int(nx) and j == int(ny)):
        can_stay = i < int(nx)
        can_emit = j < int(ny)
        if can_stay and can_emit:
            w_stay = log_stay[i, j] + beta[i + 1, j]
            w_emit = log_emit_edge[i, j] + beta[i, j + 1]
            p_emit = torch.sigmoid(w_emit - w_stay)
            u = torch.rand((), device=dev, generator=generator)
            take_emit = bool(u < p_emit)
        else:
            take_emit = can_emit
        if take_emit:
            cols.append(i)
            j += 1
        else:
            i += 1
    return cols
