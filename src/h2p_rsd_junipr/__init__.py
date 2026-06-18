"""h2p-rsd-junipr: amortized hadron-to-parton hadronization-inversion posterior
q_phi(y | x) over groomed Lund trees (RSD-JUNIPR).

One contract (`PosteriorModel`: log_prob / sample / map_estimate), many families
(§5.1 autoregressive JUNIPR, §5.2 cINN, §5.3 diffusion) behind a registry, driven
by a config-first design (OmegaConf, no Hydra) and a lean custom Trainer.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import load_config
from .geometry import Geometry

__all__ = ["load_config", "Geometry", "__version__"]
