"""Exact top-k enumeration of the SKELETON posterior, with dominance certificates
(docs/PLAN_ModeMassAudit.md WP-1/WP-2).

The *skeleton* is `S = (N, c_0 ... c_{N-1})`: the multiplicity plus the ORDERED cell
sequence, order being the primary declustering order. Two facts make the search here
exact rather than a beam-search approximation.

**Fact 1 — the skeleton marginal is exact, no integration required.** The per-node
coordinate factor is a proper density *given the cell* (it integrates to 1 over its
support — that is what `tests/test_lnz_support.py`'s MC-normalization test certifies),
so marginalising the continuous coordinates collapses analytically and

    q(S | x) = [prod_t P_cont(cont | h_t, e) P_split(c_t | h_t, e)] P_cont(stop | h_N, e)

is computable from the discrete heads alone. The empty skeleton (N = 0) is a
first-class row with mass `P_cont(stop | h_0, e)` — the same `q(0|x)` the empty-tree
analysis thresholds.

**Fact 2 — prefix mass equals subtree mass, so best-first is exact.** The total mass of
every skeleton extending a prefix `c_{0:t}` equals the prefix's own accumulated product,
because the remaining factors are normalized and sum to 1 over all futures. A
uniform-cost search on the prefix tree with that mass as priority therefore pops
completed skeletons in exact descending mass order (Dijkstra on the prefix tree; the
monotone-score best-first framing is Meister, Vieira & Cotterell, *TACL* **8** (2020)
795, arXiv:2007.03909, and the exact-enumeration-of-NMT-modes precedent is Stahlberg &
Byrne, arXiv:1908.10090).

**What "certified" means here, precisely.** The search prunes (a child below the
relative floor, the tail beyond `topk_children`, a prefix at `max_emissions`, a frontier
eviction) and every pruned branch's exact mass is accumulated at prune time in closed
form, so the bookkeeping identity

    sum_i M_i  +  frontier mass  +  pruned mass  =  1

holds at every termination point (`total_log_mass`, asserted by T2). A skeleton that was
never enumerated therefore has mass at most `pruned`, which gives the two certificates
this module reports:

* `certified` — the whole returned list is the true top-k: `pruned < M_k` (or the search
  exhausted the space) *and* `k` completions were found inside the budget.
* `certified_top1` — `M_1` is the true mode: `pruned < M_1`. Note that `M_1 > 1/2` is
  self-certifying whatever the pruning did, since the total mass is 1 by construction —
  which is exactly why the pre-registered dominance thresholds are quoted at 0.5.

An uncertified `M_1` is a LOWER bound on the true top-1 mass (the search always keeps
the highest-mass child of every expansion), so `frac(M_1 >= m)` computed from these
numbers is a lower bound on the true fraction. Never the other way round.

**`M_1` IS RESOLUTION-RELATIVE, AND EXACTNESS IS NOT INVARIANCE.** The two facts above
buy an exact probability of a well-defined event; they do not buy a grid-free notion of
dominance, and the difference is not a technicality. A skeleton bundles two kinds of
degree of freedom:

* genuinely discrete and grid-free — the multiplicity `N`, and the ORDER of the
  splittings;
* a discretized continuum — the cell labels, whose probability is `~ density x area`.

Refining `n_bins` therefore drives every `N >= 1` skeleton's mass toward zero (measured
on the fielded checkpoint: a 9x coarser cell raised the best one-splitting skeleton from
0.015 to 0.098, very nearly linear in area) while leaving `q(N=0|x)` — the one skeleton
that references no cell — untouched. Both the LEVEL of `frac(M_1 >= m)` and the IDENTITY
of the argmax are consequences of the grid, and `n_bins` is not a physics choice.
Conditioning on `N` does not repair this: at fixed `N` the area factors are shared, so
ratios survive but absolute masses still scale.

So `M_1` is quotable as a same-geometry, same-checkpoint comparison (which is what the
plan's §7.5 cross-family delta needs) and is NOT a grid-free statement that "a dominant
parton skeleton exists". `node_hpd_area` below is the companion that is: the smallest
Lund-plane AREA holding a given fraction of a node's positional posterior, in physical
units, with a limit under refinement.

Nothing here writes to the estimator stack: the audit reads the posterior, and no
decode-layer behaviour (MBR, MAP, floors) changes because of it.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

# The three search strategies. Every family maps onto one of them; a family that maps
# onto none raises from `PosteriorModel.skeleton_search_spec` by name, loudly.
KINDS = ("ar", "nhead", "factorized")

_NEG_INF = float("-inf")


def _logaddexp(a: float, b: float) -> float:
    """`log(e^a + e^b)` for plain floats, -inf-safe (math.log1p under the hood)."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _logsumexp(values) -> float:
    out = _NEG_INF
    for v in values:
        out = _logaddexp(out, v)
    return out


@dataclass
class SkeletonSearchSpec:
    """What `enumerate_skeletons` needs from a family, and nothing else.

    `kind="ar"` — `step(tok, e, h) -> (p_cont, logp_split, h')`, the SAME incremental
    decode API `sample_batch` and `beam_search_cells` consume, so the search needs no
    new model code.

    `kind="nhead"` — an explicit `q(N|x)` (`log_qn`) plus a cont_head-free
    `step_cells(tok, e, h) -> (split_logits, h')`; the cells are conditioned on the
    realized N, so a fixed-length search runs per N and the merged heap is seeded with
    one root per N at its own `log q(N|x)`.

    `kind="factorized"` — `q(N|x)` and cell log-probabilities that do not depend on the
    prefix at all (cINN/diffusion). Runs on the `nhead` machinery with a constant step,
    which is the same exactness argument and needs no separate code path.
    """

    kind: str
    e: torch.Tensor | None = None
    h0: object = None
    start_token: int = 0
    step: Callable | None = None          # kind == "ar"
    step_cells: Callable | None = None    # kind == "nhead"
    log_qn: torch.Tensor | None = None    # kind in ("nhead", "factorized")
    log_cells: torch.Tensor | None = None  # kind == "factorized"
    max_emissions: int = 25
    family: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"skeleton search kind must be one of {KINDS}, got {self.kind!r}")


@dataclass
class SkeletonEnumeration:
    """The top-k skeletons of one jet, in exact descending mass order, plus the
    accounting that turns every summary number into a bound."""

    skeletons: list[tuple[list[int], float]] = field(default_factory=list)  # (cells, log_mass)
    frontier_log_mass: float = _NEG_INF   # logsumexp over the unexpanded heap
    pruned_log_mass: float = _NEG_INF     # exact, accumulated at prune time
    certified: bool = False               # the whole list is the true top-k
    certified_top1: bool = False          # M_1 is the true mode
    n_expansions: int = 0
    exhausted: bool = False               # the space ran out before the budget did
    kind: str = ""

    # -- derived, all in linear mass -----------------------------------------
    @property
    def masses(self) -> list[float]:
        return [math.exp(lm) for _c, lm in self.skeletons]

    @property
    def m1(self) -> float:
        return math.exp(self.skeletons[0][1]) if self.skeletons else 0.0

    @property
    def m2(self) -> float:
        return math.exp(self.skeletons[1][1]) if len(self.skeletons) > 1 else 0.0

    @property
    def top1_cells(self) -> list[int]:
        return list(self.skeletons[0][0]) if self.skeletons else []

    @property
    def coverage(self) -> float:
        """`C_k = sum_{i<=k} M_i` — the enumerated mass."""
        return math.exp(_logsumexp([lm for _c, lm in self.skeletons]))

    @property
    def frontier(self) -> float:
        return math.exp(self.frontier_log_mass)

    @property
    def pruned(self) -> float:
        return math.exp(self.pruned_log_mass)

    @property
    def total_log_mass(self) -> float:
        """`log(C_k + frontier + pruned)` — 0 to float tolerance, at every termination
        point. This is the validity check, not a physics gate (T2)."""
        return _logaddexp(_logaddexp(_logsumexp([lm for _c, lm in self.skeletons]),
                                     self.frontier_log_mass), self.pruned_log_mass)

    @property
    def remainder_bound(self) -> float:
        """Certified upper bound on the un-enumerated mass, `1 - C_k <= this`."""
        return math.exp(_logaddexp(self.frontier_log_mass, self.pruned_log_mass))

    def rank_of(self, cells) -> int:
        """1-based rank of `cells` among the enumerated skeletons, or 0 if absent.

        0 means "not in the top-k that was found", NOT "impossible": read it beside
        `certified` and `remainder_bound`."""
        target = [int(c) for c in cells]
        for i, (c, _lm) in enumerate(self.skeletons):
            if list(c) == target:
                return i + 1
        return 0

    def entropy_lower_bound(self) -> float:
        """`-sum_i M_i log M_i` over the enumerated set — a certified LOWER bound on the
        skeleton-marginal entropy `H(S|x)` when the enumeration is a proper top-k
        prefix of the distribution (the tail can only add entropy)."""
        return float(-sum(m * lm for m, (_c, lm) in zip(self.masses, self.skeletons)))


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------
def _child_masses(log_m: float, logp: torch.Tensor, topk_children: int, thresh_log: float):
    """`(kept [(log_mass, cell)], dropped_log_mass)` for one expansion, exactly.

    The dropped tail is summed in CLOSED FORM from the same log-softmax rather than
    child by child, which is what keeps `pruned_log_mass` exact however aggressive the
    caps are (`m * p_cont * sum_{c dropped} softmax[c]`)."""
    n = int(logp.numel())
    k = min(int(topk_children), n) if topk_children else n
    top = torch.topk(logp, k=k)
    kept, kept_lp = [], []
    for lp, cell in zip(top.values.tolist(), top.indices.tolist()):
        lm = log_m + lp
        if lm >= thresh_log:
            kept.append((lm, int(cell)))
            kept_lp.append(lp)
        # else: falls through to the dropped remainder below
    # dropped = everything the softmax holds minus what was kept, in log space. The
    # subtraction is done on the PROBABILITY simplex (total = 1) so it stays exact.
    kept_log = _logsumexp(kept_lp) if kept_lp else _NEG_INF
    if kept_log >= 0.0:                       # numerically all of it was kept
        return kept, _NEG_INF
    if kept_log == _NEG_INF:                  # nothing kept: the whole child mass goes
        return kept, log_m
    dropped = -math.expm1(kept_log)
    return kept, (log_m + math.log(dropped)) if dropped > 0.0 else _NEG_INF


@torch.inference_mode()
def enumerate_skeletons(model, xf, nx, *, k: int = 64, budget: int = 20_000,
                        prune_rel: float = 1e-6, topk_children: int = 0,
                        max_frontier: int = 20_000, max_emissions: int | None = None,
                        eps_n: float = 1e-4, spec=None) -> SkeletonEnumeration:
    """Top-`k` skeletons of ONE jet in exact descending mass order, with certificates.

    `prune_rel` — a child whose exact mass is below `prune_rel x m_ref` is pruned, with
    `m_ref` the best completion found so far and 1 before there are any (an absolute
    floor early on). See `_threshold` for why "best so far" and "k-th so far" coincide
    here.
    `topk_children` caps the cell children per expansion (0 == every cell); the tail is
    accounted in closed form, so a cap costs certification on a flat head rather than
    correctness — and on a 900-cell geometry that trade is a bad one (capping at 64 took
    the certified fraction from 97% to 20% for 28% of the runtime), so the default is
    uncapped and `prune_rel` does the pruning instead.
    `max_frontier` bounds memory by evicting the LOWEST-mass heap entries into
    `pruned_log_mass` — accounted the same way.
    `eps_n` drops `N` values with `q(N|x) < eps_n` in the `nhead`/`factorized` families
    (their mass likewise goes to `pruned_log_mass`).

    The enumeration is UNCONSTRAINED: `min_emissions` shapes the point estimate, not the
    posterior, and a floor here would distort the mass accounting the same way a
    sampling floor distorts SBC/PIT. The empty skeleton is a first-class row.
    """
    spec = model.skeleton_search_spec(xf, nx) if spec is None else spec
    max_em = int(spec.max_emissions if max_emissions is None else max_emissions)
    if spec.kind == "ar":
        return _search_ar(spec, k=k, budget=budget, prune_rel=prune_rel,
                          topk_children=topk_children, max_frontier=max_frontier,
                          max_emissions=max_em)
    return _search_fixed_length(spec, k=k, budget=budget, prune_rel=prune_rel,
                                topk_children=topk_children, max_frontier=max_frontier,
                                max_emissions=max_em, eps_n=eps_n)


def _threshold(found: list[float], k: int, prune_rel: float) -> float:
    """`log(prune_rel * m_ref)`, with `m_ref` the k-th best completion so far, the best
    one before there are k, and 1 before there are any.

    Because the search stops the moment the k-th completion is popped, the first branch
    is unreachable from `enumerate_skeletons` and the live behaviour is
    `prune_rel * M_1` (`prune_rel * 1`, an absolute floor, until the first completion).
    It is written in full anyway so the rule does not change meaning if a caller ever
    runs the search past k."""
    if prune_rel <= 0.0:
        return _NEG_INF
    ref = found[k - 1] if len(found) >= k else (found[0] if found else 0.0)
    return ref + math.log(prune_rel)


def _evict(heap: list, max_frontier: int, pruned_log: float) -> tuple[list, float]:
    """Keep the `max_frontier` highest-mass entries; the rest is pruned, exactly."""
    if max_frontier <= 0 or len(heap) <= max_frontier:
        return heap, pruned_log
    heap.sort()                                   # priority = -log_mass, so ascending = best
    dropped = heap[max_frontier:]
    for entry in dropped:
        pruned_log = _logaddexp(pruned_log, -entry[0])
    heap = heap[:max_frontier]
    heapq.heapify(heap)
    return heap, pruned_log


def _search_ar(spec, *, k, budget, prune_rel, topk_children, max_frontier, max_emissions):
    """Continue/stop (ar_junipr_v1/v2/v4-without-n_head) — the plan's WP-1 verbatim."""
    e = spec.e
    dev = e.device
    counter = 0
    # (-log_mass, tie, done, cells, tok, h)
    heap = [(-0.0, counter, False, (), spec.start_token, spec.h0)]
    found: list[tuple[list[int], float]] = []
    found_lm: list[float] = []
    pruned_log = _NEG_INF
    n_exp = 0

    while heap and len(found) < k and n_exp < budget:
        neg, _tie, done, cells, tok, h = heapq.heappop(heap)
        log_m = -neg
        if done:
            found.append((list(cells), log_m))
            found_lm.append(log_m)
            continue
        n_exp += 1
        tok_t = torch.full((1, 1), int(tok), dtype=torch.long, device=dev)
        p_cont, logp_split, h_next = spec.step(tok_t, e, h)
        p_cont = min(max(float(p_cont), 0.0), 1.0)
        # STOP completes the skeleton here; its mass is exact and final.
        log_stop = log_m + (math.log1p(-p_cont) if p_cont < 1.0 else _NEG_INF)
        thresh = _threshold(found_lm, k, prune_rel)
        if log_stop > _NEG_INF:
            if log_stop >= thresh:
                counter += 1
                heapq.heappush(heap, (-log_stop, counter, True, cells, tok, None))
            else:
                pruned_log = _logaddexp(pruned_log, log_stop)
        log_cont = log_m + (math.log(p_cont) if p_cont > 0.0 else _NEG_INF)
        if log_cont == _NEG_INF:
            continue
        if len(cells) >= max_emissions:
            # The decode contract's depth cap. Every skeleton beyond it is unreachable,
            # so its whole mass is pruned — recorded, never silently dropped.
            pruned_log = _logaddexp(pruned_log, log_cont)
            continue
        kept, dropped_log = _child_masses(log_cont, logp_split, topk_children, thresh)
        pruned_log = _logaddexp(pruned_log, dropped_log)
        for lm, cell in kept:
            counter += 1
            heapq.heappush(heap, (-lm, counter, False, (*cells, cell), cell, h_next))
        heap, pruned_log = _evict(heap, max_frontier, pruned_log)

    return _finish(heap, found, pruned_log, n_exp, k, budget, spec.kind)


def _search_fixed_length(spec, *, k, budget, prune_rel, topk_children, max_frontier,
                         max_emissions, eps_n):
    """Explicit `q(N|x)` families: one fixed-length search per N, merged on ONE heap.

    Seeding the heap with a root per N at priority `q(N|x)` is what makes the merge
    exact — a fixed-length subtree's mass is its prefix mass for exactly the reason the
    variable-length one's is, so the pops are still globally descending.
    """
    log_qn = spec.log_qn
    dev = log_qn.device
    constant = spec.kind == "factorized"
    counter = 0
    heap: list = []
    pruned_log = _NEG_INF
    log_eps = math.log(eps_n) if eps_n > 0.0 else _NEG_INF
    for n, lq in enumerate(log_qn.tolist()):
        if n > max_emissions:
            pruned_log = _logaddexp(pruned_log, lq)
            continue
        if lq < log_eps:            # below the pre-registered eps_N; mass accounted
            pruned_log = _logaddexp(pruned_log, lq)
            continue
        counter += 1
        heapq.heappush(heap, (-lq, counter, n == 0, (), spec.start_token, spec.h0, n))

    found: list[tuple[list[int], float]] = []
    found_lm: list[float] = []
    n_exp = 0
    while heap and len(found) < k and n_exp < budget:
        neg, _tie, done, cells, tok, h, n_target = heapq.heappop(heap)
        log_m = -neg
        if done:
            found.append((list(cells), log_m))
            found_lm.append(log_m)
            continue
        n_exp += 1
        if constant:
            logp_cells, h_next = spec.log_cells, h
        else:
            tok_t = torch.full((1, 1), int(tok), dtype=torch.long, device=dev)
            logits, h_next = spec.step_cells(tok_t, spec.e, h)
            logp_cells = torch.log_softmax(logits.reshape(-1), dim=-1)
        thresh = _threshold(found_lm, k, prune_rel)
        kept, dropped_log = _child_masses(log_m, logp_cells, topk_children, thresh)
        pruned_log = _logaddexp(pruned_log, dropped_log)
        for lm, cell in kept:
            counter += 1
            heapq.heappush(heap, (-lm, counter, len(cells) + 1 == n_target,
                                  (*cells, cell), cell, h_next, n_target))
        heap, pruned_log = _evict(heap, max_frontier, pruned_log)

    return _finish(heap, found, pruned_log, n_exp, k, budget, spec.kind)


def _finish(heap, found, pruned_log, n_exp, k, budget, kind) -> SkeletonEnumeration:
    frontier_log = _logsumexp([-entry[0] for entry in heap])
    exhausted = not heap
    m1_log = found[0][1] if found else _NEG_INF
    mk_log = found[-1][1] if found else _NEG_INF
    # A completion still sitting on the heap is not "missing" — it is accounted in
    # `frontier` and would be popped next — but it does mean the list is short of k, so
    # the top-k claim needs the frontier to be exhausted (or k reached).
    complete = len(found) >= k or exhausted
    return SkeletonEnumeration(
        skeletons=found,
        frontier_log_mass=frontier_log,
        pruned_log_mass=pruned_log,
        certified=bool(found and complete and pruned_log < mk_log),
        certified_top1=bool(found and pruned_log < m1_log),
        n_expansions=int(n_exp),
        exhausted=bool(exhausted),
        kind=str(kind),
    )


# ---------------------------------------------------------------------------
# Teacher-forced scoring of ONE given skeleton
# ---------------------------------------------------------------------------
@torch.inference_mode()
def skeleton_log_prob(model, cells, xf, nx, *, spec=None) -> float:
    """`log q(S | x)` for one given cell chain — the exact skeleton marginal.

    Accumulated along the chain through the family's own incremental step, so it is the
    same arithmetic the search uses (T3 pins the two together). Two callers: the TRUTH
    skeleton's mass and rank, and the reused MBR posterior draws that feed the entropy
    estimate.
    """
    spec = model.skeleton_search_spec(xf, nx) if spec is None else spec
    cells = [int(c) for c in cells]
    if spec.kind == "ar":
        dev = spec.e.device
        h, tok, total = spec.h0, spec.start_token, 0.0
        for c in cells:
            tok_t = torch.full((1, 1), int(tok), dtype=torch.long, device=dev)
            p_cont, logp_split, h = spec.step(tok_t, spec.e, h)
            p_cont = min(max(float(p_cont), 0.0), 1.0)
            total += (math.log(p_cont) if p_cont > 0.0 else _NEG_INF)
            total += float(logp_split[c])
            tok = c
        tok_t = torch.full((1, 1), int(tok), dtype=torch.long, device=dev)
        p_cont, _lp, _h = spec.step(tok_t, spec.e, h)
        p_cont = min(max(float(p_cont), 0.0), 1.0)
        return float(total + (math.log1p(-p_cont) if p_cont < 1.0 else _NEG_INF))

    n = len(cells)
    qn = spec.log_qn
    # Past the categorical's support the LENGTH term is clamped to the last bin, which is
    # what `nll_terms` / `describe_sequence` do (`ny.clamp(max=max_emissions)`). Scoring
    # it as -inf here instead would make the audit disagree with the model's own
    # likelihood on exactly the jets the support guard already flags.
    total = float(qn[min(n, int(qn.numel()) - 1)]) if int(qn.numel()) else _NEG_INF
    if spec.kind == "factorized":
        return float(total + sum(float(spec.log_cells[c]) for c in cells))
    dev = qn.device
    h, tok = spec.h0, spec.start_token
    for c in cells:
        tok_t = torch.full((1, 1), int(tok), dtype=torch.long, device=dev)
        logits, h = spec.step_cells(tok_t, spec.e, h)
        total += float(torch.log_softmax(logits.reshape(-1), dim=-1)[c])
        tok = c
    return float(total)


# ---------------------------------------------------------------------------
# The grid-free companion to M_1
# ---------------------------------------------------------------------------
@torch.inference_mode()
def node_hpd_area(model, xf, nx, *, alphas=(0.5, 0.9), sub=7, cells_logp=None,
                  spec=None) -> dict | None:
    """The smallest Lund-plane AREA holding a fraction `alpha` of the FIRST splitting's
    positional posterior — the dominance statement `M_1` cannot make.

    `M_1` answers "how much mass sits on one cell", which is `~ density x cell area` and
    so is a readout of `n_bins`. This answers "how large a region does the posterior
    actually occupy", in `ln(1/DeltaR) x ln k_t` units, which has a limit as the grid
    refines and is comparable across geometries and families.

    Computed exactly rather than sampled. The full density over the first node's position
    is the mixture `sum_c P_split(c|h_0,e) * TN(du|c) TN(dv|c)`, and each component is
    supported on ITS OWN CELL and nowhere else — the coordinate head's truncated normals
    are bounded by construction. So the mixture is block-wise and a `sub x sub` grid
    inside every cell evaluates it everywhere; the alpha-highest-density region is then
    read off by sorting pixels. `mass_quadrature` is returned as the check that it is
    (1.0 to the quadrature's own accuracy, ~2e-3 at sub=7).

    Returned beside it, and the reason this is not merely a rescaling of `M_1`:

    * `sigma_u`, `sigma_v` — the coordinate head's own widths at the modal cell, i.e. the
      resolution the MODEL claims for itself.
    * `sigma_box_area` = `(2 sigma_u)(2 sigma_v)`, and `area_over_sigma_box`. Above 1 the
      positional spread is genuinely wider than the model's own per-node width; near 1 the
      structure is as determined as this model can express.
    * `truncation_saturated` — `sigma > half-cell` on BOTH axes, i.e. the head wants to be
      wider than a cell and the truncation forbids it. When that is true the within-cell
      density is nearly uniform and the model is carrying its coordinate uncertainty in
      the CELL distribution instead — which is exactly the regime where a small `M_1` says
      "the grid is finer than the model's resolution" rather than "the posterior is
      fragmented". It is a property of the (geometry, checkpoint) pair, and it is the
      first thing to read before quoting any mode mass.

    Returns None for a family with no continuous coordinate head (there is no positional
    density to take a region of) or a non-`ar` search spec, rather than inventing one.
    """
    geom = model.geometry
    if not bool(getattr(model, "has_continuous_coords", False)):
        return None
    params_fn = getattr(model, "_coord_params_padded", None)
    if params_fn is None:
        return None
    spec = model.skeleton_search_spec(xf, nx) if spec is None else spec
    if spec.kind != "ar":
        # `nhead`/`factorized` reach the first node's cell head differently; the area is
        # well defined there too, but the caller must supply `cells_logp` for it.
        if cells_logp is None:
            return None

    dev = xf.device
    n_cells, hu, hv = geom.n_cells, geom.half_u, geom.half_v
    if cells_logp is None:
        tok = torch.full((1, 1), int(spec.start_token), dtype=torch.long, device=dev)
        _p_cont, cells_logp, _h = spec.step(tok, spec.e, spec.h0)
    p0 = torch.softmax(cells_logp.reshape(-1), dim=-1)

    # Every one-node chain's coordinate head in ONE pass: the decoder state after START is
    # shared across cells, so only the cell embedding differs between rows.
    yc = torch.arange(n_cells, dtype=torch.long, device=dev).unsqueeze(1)
    du_m, dv_m, du_s, dv_s = (t.squeeze(1) for t in params_fn(xf, nx, yc)[:4])

    from ..distributions import trunc_normal_logpdf

    step_u, step_v = 2 * hu / sub, 2 * hv / sub
    du = torch.tensor([-hu + (k + 0.5) * step_u for k in range(sub)], device=dev)
    dv = torch.tensor([-hv + (k + 0.5) * step_v for k in range(sub)], device=dev)
    lu = trunc_normal_logpdf(du.view(1, sub), du_m.view(-1, 1), du_s.view(-1, 1), -hu, hu)
    lv = trunc_normal_logpdf(dv.view(1, sub), dv_m.view(-1, 1), dv_s.view(-1, 1), -hv, hv)
    comp = (lu.unsqueeze(2) + lv.unsqueeze(1)).exp()          # (n_cells, sub, sub), each ->1
    pix = float(step_u * step_v)
    dens = (p0.view(-1, 1, 1) * comp).reshape(-1)
    c_star = int(p0.argmax())

    def _area(d) -> dict:
        srt, _ = torch.sort(d.reshape(-1), descending=True)
        cum = torch.cumsum(srt, dim=0) * pix
        tot = float(cum[-1])
        out = {}
        for a in alphas:
            k = int(torch.searchsorted(cum, torch.tensor(a * tot, device=cum.device))) + 1
            out[f"{a:g}"] = float(min(k, srt.numel()) * pix)
        return out

    full, one = _area(dens), _area(comp[c_star])
    s_u, s_v = float(du_s[c_star]), float(dv_s[c_star])
    box = (2.0 * s_u) * (2.0 * s_v)
    return {
        "alphas": [float(a) for a in alphas],
        "area": full,
        # The same region for ONE component. NOTE: when the head is truncation-saturated
        # this measures the CELL, not the physics — read `truncation_saturated` first.
        "area_one_component": one,
        "sqrt_area": {a: math.sqrt(v) for a, v in full.items()},
        "n_cells_equivalent": {a: v / (4.0 * hu * hv) for a, v in full.items()},
        "sigma_u": s_u, "sigma_v": s_v, "sigma_box_area": box,
        "area_over_sigma_box": {a: (v / box if box > 0 else float("nan"))
                                for a, v in full.items()},
        "truncation_saturated": bool(s_u > hu and s_v > hv),
        "modal_cell": c_star, "modal_cell_mass": float(p0[c_star]),
        "cell_area": float(4.0 * hu * hv),
        "mass_quadrature": float(dens.sum() * pix),
    }


@torch.inference_mode()
def skeleton_log_probs(model, chains, xf, nx, *, spec=None) -> list[float]:
    """`skeleton_log_prob` for MANY chains of ONE jet, sharing the spec (one encode).

    Deliberately a loop over the incremental path rather than a batched teacher-forced
    pass: the batched form is family-specific, and the audit's cost is dominated by the
    search, not by this. It is the same trade the plan's §10 compute estimate assumes.
    """
    spec = model.skeleton_search_spec(xf, nx) if spec is None else spec
    return [skeleton_log_prob(model, c, xf, nx, spec=spec) for c in chains]


def entropy_from_draws(log_masses) -> dict:
    """`H_hat = -mean_k log q(S^(k)|x)` over posterior DRAWS, and `e^H_hat`.

    An unbiased estimator of the skeleton-marginal entropy `H(S|x)` (the draws come from
    `q(S|x)` itself), whose exponential is the effective number of skeletons — the
    typical-set reading of Cover & Thomas, *Elements of Information Theory*, ch. 3. The
    enumerated-set `-sum_i M_i log M_i` is reported beside it as the certified lower
    bound; the two answer the same question from opposite sides.
    """
    lm = [float(v) for v in log_masses if math.isfinite(float(v))]
    if not lm:
        return {"H_hat": float("nan"), "eff_skeletons": float("nan"), "n_draws": 0}
    h = -sum(lm) / len(lm)
    return {"H_hat": float(h), "eff_skeletons": float(math.exp(h)), "n_draws": int(len(lm))}
