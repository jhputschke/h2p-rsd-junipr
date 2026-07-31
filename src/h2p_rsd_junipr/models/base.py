"""Model abstraction & registry (§3): one contract, many posterior families.

The trainer, validation suite, and serving layer only ever touch `log_prob`,
`sample`, `map_estimate` — they never know which family they hold.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from ..geometry import Geometry
from ..inference.point_estimate import LundNode, LundPointEstimate

_REGISTRY: dict[str, type[PosteriorModel]] = {}


def register_model(*names: str):
    def deco(cls):
        for n in names:
            _REGISTRY[n] = cls
        return cls

    return deco


class PosteriorModel(nn.Module, ABC):
    # --- contract flags (WP1/WP2 of docs/PLAN_UPDATES.md) ---------------------
    # Is `log_prob` a NORMALIZED log-density, or a training surrogate? Every family
    # that honors the contract leaves this True; `Diffusion` sets it False (its
    # log_prob is a denoising-score-matching proxy). Consumers that report NLLs or
    # likelihood RATIOS (cli `eval`, serving) warn when it is False rather than
    # branching on the family name.
    exact_likelihood: bool = True
    # Does the family expose closed-form / invertible per-coordinate CDFs, i.e. can
    # `coordinate_cdfs` return a probability-integral transform? (WP2 coordinate PITs.)
    supports_coordinate_pit: bool = False
    # Aux conditioning columns this model's encoder was BUILT for, in order
    # (docs/PLAN_Input.md). Every family sets it from `encoder.aux_features`; the ()
    # default is the off path, and it is what serving reads to decide whether a
    # request must carry the aux sources.
    aux_feature_names: tuple[str, ...] = ()
    # Post-hoc scalar temperature on the multiplicity head's logits, fitted on held-out
    # jets by `inference.length.fit_length_temperature` and set from
    # `decode.length_temperature`. It is a DECODE-layer recalibration: it moves
    # `length_pmf` and the N drawn by `sample`, and deliberately never touches
    # `log_prob` or the `logprob` a point estimate reports — those are the trained
    # likelihood. 1.0 is off and bit-identical (division by 1.0 is exact). A no-op for
    # families with no explicit `n_head` (ar_junipr_v1/v2), where the length belief IS
    # the sampler histogram and tempering one without the other would decouple them.
    length_temperature: float = 1.0
    # The companion tilt: a term LINEAR IN n added to the same logits. A temperature is
    # symmetric about the mode and so cannot produce the monotone ramp the head actually
    # shows (empirical/predicted = 1.90, 0.96, 0.93, 0.80, 0.68 at n = 0..4); the tilt is
    # what moves mass between short and long trees. 0.0 is off.
    length_tilt: float = 0.0

    def recalibrated_n_logits(self, z):
        """`z / T + tilt * n` — post-hoc affine recalibration of the multiplicity logits.

        The single place the two knobs are applied, so `length_pmf` and the `N` drawn by
        `sample` can never disagree. Returns `z` untouched when both are off, which is
        what makes the default path bit-identical rather than merely close."""
        if self.length_temperature == 1.0 and self.length_tilt == 0.0:
            return z
        out = z / self.length_temperature
        if self.length_tilt:
            n = torch.arange(z.shape[-1], device=z.device, dtype=z.dtype)
            out = out + self.length_tilt * n
        return out
    # Does the family have a continuous coordinate density, i.e. does
    # `sample_coordinates` return coordinates rather than None? False means its nodes
    # only ever carry the two Lund-cell coordinates, and ln z / psi are UNSET — a
    # consumer that plots or scores them is reading a filler constant as a prediction.
    # Distinct from `supports_coordinate_pit`: `diffusion` has coordinates but no
    # closed-form CDF, so it is True here and False there.
    has_continuous_coords: bool = False

    @abstractmethod
    def log_prob(self, batch: dict) -> torch.Tensor:
        """(B,) log q_phi(y | x)."""
        ...

    def training_objective(self, batch: dict) -> torch.Tensor:
        """(B,) per-jet quantity the trainer MINIMIZES.

        Defaults to `-log_prob` — i.e. maximum likelihood, which is what every
        likelihood-trained family wants, so the default keeps the loop bit-identical
        for all of them. The hook exists because a family can have an exact
        `log_prob` that is *not* its training objective: `cfm` regresses a conditional
        vector field (Lipman et al., arXiv:2210.02747) and only integrates the
        probability-flow ODE at evaluation time. Overriding this — instead of letting
        `log_prob` return the cheap surrogate — is what keeps the one-contract
        invariant that `log_prob` is always a normalized density."""
        return -self.log_prob(batch)

    def coordinate_cdfs(self, batch: dict) -> dict | None:
        """Per-emission probability-integral transforms of the TRUE coordinates,
        teacher-forced on `batch` — the input to the WP2 per-coordinate PITs.

        Returns `{"names": [str, ...], "u": (B, L, D) in [0,1], "mask": (B, L) bool,
        "space": "physical"|"latent"}` or **None** when the family cannot provide one
        (no exact coordinate density, or no continuous coordinates at all). Families
        with an explicit coordinate head report the transform in PHYSICAL coordinates
        (one PIT per Lund coordinate); families whose coordinates go through a
        normalizing map report it in the LATENT base space (one PIT per base
        dimension, uniform under a calibrated model either way).

        This is the only place the per-family difference lives: `eval/calibration.py`
        consumes the dict and never asks which family produced it."""
        return None

    @abstractmethod
    def sample(self, xf: torch.Tensor, nx: torch.Tensor, n: int) -> list:
        """`n` posterior draws (cell chains) for one jet."""
        ...

    @abstractmethod
    def map_estimate(self, xf: torch.Tensor, nx: torch.Tensor) -> LundPointEstimate:
        ...

    def sample_coordinates(self, xf, nx, cells) -> torch.Tensor | None:
        """`(L, 4)` continuous coordinates drawn from `q(coords | cells, x)` for one
        jet, in `features.node_raw` column order `(ln 1/DeltaR, ln kt, ln z, psi)` —
        or **None** when the family has no continuous coordinate density.

        This is the coordinate half of a posterior draw. `sample` deliberately returns
        cell chains only, so without this hook the sole family-agnostic way to place a
        drawn tree in the Lund plane is at cell centres — which leaves ln z and psi
        with nothing to hold but a filler constant. Every consumer that wants a draw's
        coordinates (`describe_cells`, and through it the MBR winner and the notebooks'
        posterior-predictive series) goes through here, so the placeholder path is
        entered only by families that genuinely have no coordinates to give.

        Returning None and setting `has_continuous_coords = False` is a legitimate
        implementation — `ar_junipr_v1` is exactly that model — but the two must agree.
        """
        return None

    def describe_cells(self, xf, nx, cells) -> LundPointEstimate:
        """One posterior draw (a cell chain) -> LundPointEstimate: the model's joint
        log-density of that chain, with each node's coordinates drawn from
        `sample_coordinates` when the family has them and placed at the Lund-cell
        centre when it does not.

        The MBR winner (`inference.mbr.mbr_select`) is a genuine drawn tree, so a draw
        from `q(coords | cells, x)` is the coordinate-space completion of it. AR still
        overrides this with its staged decode (head MODES rather than draws), which is
        what a point estimate wants.

        Without coordinates the ln z / psi entries are PLACEHOLDERS, not predictions:
        `ln z = 0` means `z = 1`, the softer prong taking the whole jet, which is below
        no grooming boundary because it is not physical at all. They are here so the
        node type stays one shape across families; `has_continuous_coords` is the flag
        that says whether they mean anything, and the log-density below is only exact
        in the same case."""
        geom = self.geometry
        cells = [int(c) for c in cells]
        L = len(cells)
        dev = xf.device
        drawn = self.sample_coordinates(xf, nx, cells) if L else None
        nodes, rows = [], []
        for t, c in enumerate(cells):
            if drawn is None:
                u, v = geom.cell_center(c)
                lz, ps, zed = 0.0, 0.0, 1.0     # placeholders -- see the docstring
            else:
                u, v, lz, ps = (float(drawn[t, j]) for j in range(4))
                zed = math.exp(lz)
            rows.append([u, v, lz, ps])
            nodes.append(
                LundNode(
                    depth=t, parent=t - 1, cell=c,
                    ln_invDelta=u, ln_kt=v, ln_z=lz, psi=ps,
                    kt=math.exp(v), delta_R=math.exp(-u), z=zed,
                    logp_split=0.0, logp_coord=0.0, logp_cont=0.0,
                )
            )
        if L > 0:
            yc = torch.tensor([cells], dtype=torch.long, device=dev)
            yraw = torch.tensor([rows], dtype=torch.float32, device=dev)
        else:
            yc = torch.zeros(1, 0, dtype=torch.long, device=dev)
            yraw = torch.zeros(1, 0, 4, dtype=torch.float32, device=dev)
        batch = {"xf": xf, "nx": nx, "yc": yc,
                 "ny": torch.tensor([L], device=dev), "yraw": yraw}
        with torch.inference_mode():
            logprob = float(self.log_prob(batch)[0])
        return LundPointEstimate(nodes=nodes, logprob=logprob, multiplicity=L)

    def map_or_mbr(self, xf, nx, *, draws=None, **decode) -> LundPointEstimate:
        """Point estimate dispatched by ``decode['point_estimator']``: ``"map"``
        (default) -> ``map_estimate``; ``"mbr"`` -> minimum-Bayes-risk selection over
        posterior draws (`inference.mbr.mbr_select`, reusing ``draws`` when given).
        A thin convenience so all three families gain MBR with no per-family code;
        the ``"map"`` branch imports no OT backend, preserving parity.

        ``decode['empty_threshold']`` (default 0.0 == off) adds an emptiness decision
        *before* either shape decode: when ``q(N=0|x) >= tau`` the answer is the empty
        tree. The parton target genuinely is empty for ~17% of jets, and neither
        estimator can say so — the MAP because ``argmax_n q(n|x)`` lands at 0 essentially
        never however much mass sits there, MBR because the perturbative-Lund EMD's
        imbalance term makes an empty cloud near-maximal risk
        (docs/PLAN_empty_parton_tree.md). Living here rather than per family gives every
        family the stage at once and keeps ``map_estimate`` a pure shape decode."""
        tau = float(decode.get("empty_threshold", 0.0))
        if tau > 0.0:  # local import: the default decode enters no new code path
            from ..inference.length import empty_gate

            pmf = self.length_pmf(xf, nx, mults=[len(d) for d in draws] if draws else None)
            if empty_gate(pmf, tau):
                return self.describe_cells(xf, nx, [])
        if str(decode.get("point_estimator", "map")) == "mbr":
            from ..inference.mbr import mbr_kwargs_from_decode, mbr_select

            return mbr_select(self, xf, nx, draws=draws, geom=self.geometry,
                              **mbr_kwargs_from_decode(decode))
        return self.map_estimate(xf, nx, **decode)

    def length_pmf(self, xf, nx, mults=None, n_samples: int = 500) -> np.ndarray:
        """The model's per-jet length belief P(n|x) as a normalized pmf over n=0,1,...

        Default (sampler-based: AR and any family without an explicit length head):
        the empirical multiplicity histogram of posterior draws. If `mults` (the
        per-draw multiplicities the caller already computed) is given it is reused —
        no second sample. cINN/diffusion override this with their exact softmax head.
        """
        if mults is None:
            mults = [len(d) for d in self.sample(xf, nx, n_samples)]
        counts = np.bincount(np.asarray(mults, dtype=int))
        total = counts.sum()
        if total == 0:
            return np.array([1.0])  # degenerate (no draws): all mass at n=0
        return counts / total


def build_model(cfg, geometry: Geometry) -> PosteriorModel:
    # import for side-effect registration
    from . import ar_junipr, cfm, cinn, diffusion  # noqa: F401

    name = cfg.model.name
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    model = _REGISTRY[name](cfg, geometry)
    # Read from the config snapshot so a checkpoint carries its own recalibration;
    # `cmd_eval` re-applies it afterwards so a lifted `decode.length_temperature` wins.
    from ..config import decode_params

    dec = decode_params(cfg)
    t, tilt = float(dec["length_temperature"]), float(dec["length_tilt"])
    if (t != 1.0 or tilt != 0.0) and not hasattr(model, "n_head"):
        print(f"[model] WARNING: decode.length_temperature={t:g}/length_tilt={tilt:g} "
              f"is a NO-OP for "
              f"{name!r} — it has no multiplicity head, so its length belief is the "
              f"sampler histogram and tempering it would decouple the two.")
    model.length_temperature, model.length_tilt = t, tilt
    return model


def registered_models() -> list[str]:
    from . import ar_junipr, cfm, cinn, diffusion  # noqa: F401

    return sorted(_REGISTRY)
