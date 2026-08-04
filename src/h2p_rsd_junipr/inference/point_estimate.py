"""Point-estimate structures + beam search (was `LundNode`, `LundPointEstimate`,
`map_decode`).

`LundNode`/`LundPointEstimate` are family-agnostic containers; `beam_search_cells`
is the generic MAP cell-structure search, parameterised by a single-jet step fn
`step(tok, e, h) -> (p_cont, logp_split, h)` so any autoregressive family reuses
it. The continuous coordinates are attached afterwards by the model's own head
(staged MAP).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class LundNode:
    """One primary splitting of the MAP groomed parton-shower configuration, with
    CONTINUOUS coordinates predicted by the model. The primary branch is a
    CATERPILLAR: at step `depth` the leading prong splits, emitting a softer prong
    with this node's kinematics. `parent` is the previous node."""

    depth: int
    parent: int
    cell: int
    ln_invDelta: float
    ln_kt: float
    ln_z: float
    psi: float
    kt: float
    delta_R: float
    z: float
    logp_split: float
    logp_coord: float
    logp_cont: float
    # --- psi identifiability (docs/PLAN_prod_test_v1.md WP-C.2) ----------------
    # The von Mises concentration this node's psi came from. It is a diagnostic in its
    # own right, not decoration: at kappa ~ 0 the density is flat and its MODE is the
    # direction of a near-zero resultant (Mardia & Jupp, *Directional Statistics*,
    # Wiley 2000), i.e. arbitrary. v0 measured median kappa = 0.022 — peak/trough
    # 1.04 — while MAP/MBR reported a psi resultant |R| = 0.69 against a truth of
    # 0.045. A psi panel is unreadable without this column beside it.
    kappa: float | None = None
    # Is the reported `psi` an IDENTIFIED mode?
    #   True  — it is the conditional mode and kappa >= decode.kappa_min_mode.
    #   False — the mode was rejected as unidentified and a DRAW was substituted.
    #   None  — the node carries a genuine posterior sample (the MBR medoid), where
    #           "is the mode identified" is not a question that applies.
    psi_identified: bool | None = None


@dataclass
class LundPointEstimate:
    """The single most likely groomed parton-shower configuration
    y_hat = argmax_y q_phi(y | x), as a primary Lund tree of continuous-coordinate
    nodes, plus the FULL joint log-density log q_phi(y_hat | x)."""

    nodes: list
    logprob: float
    multiplicity: int
    # Decision-theoretic MBR score (mean perturbative-Lund EMD to the posterior),
    # set only by `inference.mbr.mbr_select`; None for MAP/mode estimates. This is
    # NOT a likelihood — never feed it to anything expecting an NLL.
    risk: float | None = None
    # Where the continuous coordinates came from: "mode" (the staged MAP's head modes)
    # or "sample" (a genuine draw — what the MBR medoid carries, since the medoid IS a
    # posterior sample and re-attaching modes forfeits exactly that property).
    coords_source: str = "mode"
    # --- posterior-cluster scalars (docs/PLAN_PosteriorClusters.md WP3) ---------------
    # Set only by `inference.mbr.mbr_cluster_set` / `PosteriorModel.predict_set`, and None
    # on every other path — including the MBR medoid, which is a centrality estimate with
    # no cluster attached. They live here so `eval/closure.py`, `serving/` and the
    # notebooks carry the scalars without a signature change.
    #
    # `cluster_mass` is a PROBABILITY: the posterior mass of the selected explanation. It
    # is NOT calibrated until docs/PLAN_PosteriorClusters.md WP5 says it is (the joint tree
    # posterior is over-confident by v1 TARP), and under `decode.cluster_split=false` it is
    # biased HIGH — the same draws define the cluster and are counted into it.
    cluster_mass: float | None = None
    # `cluster_entropy` is an AMBIGUITY over discrete alternatives, H(m) = -sum m log m in
    # nats. It is not a width and must never be quoted as one: a bimodal posterior
    # summarised as mean +/- sd points at a configuration neither mode supports. The width
    # of the selected explanation is the cluster RADIUS (`PosteriorSetEstimate.radii[0]`).
    cluster_entropy: float | None = None

    @property
    def n_psi_unidentified(self) -> int:
        """Nodes whose reported psi is a mode the head does not actually identify."""
        return sum(1 for n in self.nodes if n.psi_identified is False)

    def pretty(self) -> str:
        head = (
            f"{'MBR' if self.risk is not None else 'MAP'} groomed shower: "
            f"{self.multiplicity} primary splittings, "
            f"log q(y_hat|x) = {self.logprob:.3f}   (coordinates: {self.coords_source})"
        )
        rows = [
            f"  [{n.depth}] kt={n.kt:6.2f} GeV  DeltaR={n.delta_R:5.3f}  z={n.z:5.3f}  "
            f"psi={n.psi:+5.2f}{'' if n.psi_identified is not False else '*'}  "
            f"(ln1/DR={n.ln_invDelta:4.2f}, lnkt={n.ln_kt:4.2f}, "
            f"lnz={n.ln_z:5.2f})  logP={n.logp_split + n.logp_coord:+.2f}"
            for n in self.nodes
        ]
        if self.n_psi_unidentified:
            rows.append(f"  * psi drawn, not moded: kappa below the identifiability bound "
                        f"({self.n_psi_unidentified} of {self.multiplicity} nodes)")
        return "\n".join([head, *rows]) if rows else head + "\n  (empty: MAP is immediate stop)"


StepFn = Callable[[torch.Tensor, torch.Tensor, object], tuple]


@torch.inference_mode()
def beam_search_cells(
    step: StepFn,
    e: torch.Tensor,
    h0,
    start_token: int,
    device: torch.device,
    beam_width: int = 8,
    topk_cells: int = 6,
    max_emissions: int = 25,
    min_emissions: int = 1,
    length_penalty: float = 0.0,
) -> list[int]:
    """MAP cell structure: argmax over (continue/stop, cell) by beam search.
    `step(tok, e, h) -> (p_cont: float, logp_split: (n_cells,), h)`.

    `min_emissions` is a hard floor on the returned length: a STOP shorter than the
    floor is never recorded, so the MAP never collapses to the unphysical empty tree
    (a groomed jet has >=1 primary splitting). `length_penalty` (alpha) ranks finished
    hypotheses by GNMT-style `score / len**alpha` to counter the brevity bias of an
    un-normalized argmax over a high-entropy categorical head; alpha=0 (and
    min_emissions=0) reproduces the raw-score behavior exactly. Pruning within a step
    stays on raw score, where all candidates share a length."""
    start = torch.full((1, 1), start_token, dtype=torch.long, device=device)
    active = [(0.0, [], h0, start)]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(max_emissions):
        cand = []
        for score, cells, h, tok in active:
            p_cont, logp_split, h_next = step(tok, e, h)
            p_cont = min(max(p_cont, 1e-8), 1 - 1e-8)
            if len(cells) >= min_emissions:                         # STOP (floor-gated)
                finished.append((score + math.log(1 - p_cont), cells))
            top = torch.topk(logp_split, k=min(topk_cells, logp_split.numel()))
            for lp, cell in zip(top.values.tolist(), top.indices.tolist()):
                nt = torch.tensor([[cell]], dtype=torch.long, device=device)
                cand.append((score + math.log(p_cont) + lp, cells + [cell], h_next, nt))
        if not cand:
            break
        cand.sort(key=lambda b: b[0], reverse=True)
        active = cand[:beam_width]
    for score, cells, h, tok in active:                            # terminal flush
        if len(cells) >= min_emissions:
            p_cont, _, _ = step(tok, e, h)
            finished.append((score + math.log(min(max(1 - p_cont, 1e-8), 1.0)), cells))
    if not finished:  # degenerate (e.g. max_emissions < min_emissions): best active beam
        active.sort(key=lambda b: b[0], reverse=True)
        return active[0][1]
    finished.sort(key=lambda b: b[0] / max(len(b[1]), 1) ** length_penalty, reverse=True)
    return finished[0][1]
