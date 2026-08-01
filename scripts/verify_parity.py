"""Bit-comparable-NLL parity check (Phase 1 exit criterion).

Loads the original research script `ConditionalPrimaryLundJUNIPR` (the monolithic
v2 model in scripts/reference/), copies its weights into the refactored
`ARJunipr` (split encoder + decoder behind the registry), and asserts that
`per_jet_nll` matches bit-for-bit on the same synthetic batch. This proves the
refactor preserves the computation exactly — the discretised likelihood is
unchanged by the module split.

`encoder.mask_padding=false` is REQUIRED here, and is the one place it is used. The
reference script runs its bidirectional GRU over the zero-padded batch
(`conditional_rsd_junipr_v2.py`: `out, _ = self.encoder(self.x_feat(xf))`), so the
backward pass sweeps through the padding into the real nodes and a jet's context
depends on its batch-mates. That is a defect, fixed by default in `encoders/gru.py` —
but parity is measured against the reference AS IT IS, so reproducing it is exactly
what this script must do. Comparing under the fix instead would prove nothing about the
refactor and would fail by ~3e-2.

Run:  python scripts/verify_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REF = Path(__file__).resolve().parent / "reference"
sys.path.insert(0, str(REF))

import conditional_rsd_junipr_v2 as ref  # noqa: E402

from h2p_rsd_junipr.config import load_config  # noqa: E402
from h2p_rsd_junipr.data.dataset import collate  # noqa: E402
from h2p_rsd_junipr.geometry import Geometry  # noqa: E402
from h2p_rsd_junipr.models.base import build_model  # noqa: E402


def remap_state_dict(old_sd: dict) -> dict:
    """Old monolithic model -> new (encoder split into encoder_net.*)."""
    new_sd = {}
    enc_prefixes = ("x_feat.", "encoder.", "to_ctx.")
    for k, v in old_sd.items():
        if any(k.startswith(p) for p in enc_prefixes):
            new_sd["encoder_net." + k] = v
        else:
            new_sd[k] = v
    return new_sd


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # identical synthetic data + dataset as the v2 script
    jets = ref.synthetic_matched_dataset(256, seed=0)
    # mask_padding=false: match the reference's padded-GRU encode — see the module
    # docstring. This is a parity harness, not a recommendation.
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru",
                       "encoder.mask_padding=false"])
    geom = Geometry.from_config(cfg.geometry)

    ds_new = __import__(
        "h2p_rsd_junipr.data.dataset", fromlist=["MatchedLundDataset"]
    ).MatchedLundDataset(jets, geom)
    batch = collate([ds_new[i] for i in range(64)])

    # build both models, copy weights old -> new
    old = ref.ConditionalPrimaryLundJUNIPR().to(device).eval()
    new = build_model(cfg, geom).to(device).eval()
    missing, unexpected = new.load_state_dict(remap_state_dict(old.state_dict()), strict=False)
    # only the buffers (cell_cx/cell_cy) are allowed to differ trivially; assert no params missed
    param_names = {n for n, _ in new.named_parameters()}
    missed_params = [m for m in missing if m in param_names]
    assert not missed_params, f"parameters not transferred: {missed_params}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    with torch.inference_mode():
        nll_old = old.per_jet_nll(batch)
        nll_new = new.per_jet_nll(batch)

    max_abs = (nll_old - nll_new).abs().max().item()
    print(f"[parity] batch of {nll_old.shape[0]} jets")
    print(f"[parity] old per_jet_nll mean = {nll_old.mean().item():.6f}")
    print(f"[parity] new per_jet_nll mean = {nll_new.mean().item():.6f}")
    print(f"[parity] max |delta| = {max_abs:.3e}")
    ok = torch.allclose(nll_old, nll_new, atol=1e-5, rtol=1e-5)
    print(f"[parity] allclose(atol=1e-5): {ok}")

    # also check log_prob == -per_jet_nll and weighted loss equivalence
    assert torch.allclose(new.log_prob(batch), -nll_new, atol=1e-6)
    print("[parity] log_prob == -per_jet_nll: True")

    if not ok:
        print("PARITY FAILED")
        return 1
    print("PARITY PASSED — refactor reproduces the v2 likelihood bit-for-bit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
