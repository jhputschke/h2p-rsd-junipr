"""Bit-comparable-NLL parity against the original v2 research script (Phase 1
exit criterion). Skips cleanly if the reference script is absent."""

import sys
from pathlib import Path

import pytest
import torch

REF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "reference"
REF_FILE = REF_DIR / "conditional_rsd_junipr_v2.py"


@pytest.mark.skipif(not REF_FILE.exists(), reason="reference v2 script not vendored")
def test_refactor_matches_reference_nll():
    sys.path.insert(0, str(REF_DIR))
    import conditional_rsd_junipr_v2 as ref

    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model

    torch.manual_seed(0)
    jets = ref.synthetic_matched_dataset(128, seed=0)
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru"])
    geom = Geometry.from_config(cfg.geometry)
    b = collate([MatchedLundDataset(jets, geom)[i] for i in range(32)])

    old = ref.ConditionalPrimaryLundJUNIPR().eval()
    new = build_model(cfg, geom).eval()

    enc_prefixes = ("x_feat.", "encoder.", "to_ctx.")
    remap = {
        ("encoder_net." + k if any(k.startswith(p) for p in enc_prefixes) else k): v
        for k, v in old.state_dict().items()
    }
    missing, unexpected = new.load_state_dict(remap, strict=False)
    param_names = {n for n, _ in new.named_parameters()}
    assert not [m for m in missing if m in param_names]
    assert not unexpected

    with torch.inference_mode():
        assert torch.allclose(old.per_jet_nll(b), new.per_jet_nll(b), atol=1e-5)
