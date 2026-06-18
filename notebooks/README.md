# notebooks/

Validation + physics figures, nbstripped in git (see `.pre-commit-config.yaml`)
and rendered in the docs:

- **inference_demo.ipynb** — standalone end-to-end evaluation: load a `best.ckpt`,
  take a `jets.root` test file **or** generate synthetic matched test data, and show
  posterior + MAP point-estimate performance vs truth (single-jet posterior & Lund
  plane, multiplicity / leading-emission recovery, coordinate marginals, SBC / PIT /
  coverage). Runs with no arguments. Combines the three notebooks below into one.
- **closure.ipynb** — leading-emission Lund distance, multiplicity bias, MAP vs
  plain-RSD vs truth trees (§8 closure).
- **calibration.ipynb** — SBC rank histogram, PIT, coverage (Talts et al.,
  arXiv:1804.06788).
- **posterior_walkthrough.ipynb** — a per-jet posterior: MAP tree + 68% credible
  region + multiplicity distribution.
- **generator_systematic.ipynb** — PYTHIA-trained vs HERWIG-trained MAP/posterior
  spread (§8); the inter-model spread is the dominant systematic.

Generate the underlying metrics with `h2p-rsd-junipr eval <ckpt>` (writes a JSON
metrics record + the CSV/JSONL training curves in the run dir).
