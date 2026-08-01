"""Bit-comparable-NLL parity against the original v2 research script (Phase 1
exit criterion). Skips cleanly if the reference script is absent.

`encoder.mask_padding=false` is required and deliberate: the reference runs its
bidirectional GRU over the zero-padded batch, so a jet's context depends on its
batch-mates. `encoders/gru.py` fixes that by default; parity is measured against the
reference AS IT IS, so this harness must reproduce the defect. See
`tests/test_encoder_padding.py` for the property the fix actually buys, and
`scripts/verify_parity.py` for the same pinning."""

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
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru",
                       "encoder.mask_padding=false"])   # match the reference; see docstring
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


@pytest.mark.skipif(not REF_FILE.exists(), reason="reference v2 script not vendored")
def test_decode_switches_at_their_no_op_settings_reproduce_the_reference():
    """`per_jet_nll` cannot see the decode switches at all — `continue_temperature` is
    sampling-only and the psi identifiability gate touches only the reported mode — so a
    likelihood-only parity check would pass with either silently live. This pins the
    other half: the pinned reference path of docs/PLAN_prod_test_v1.md §10."""
    sys.path.insert(0, str(REF_DIR))
    import conditional_rsd_junipr_v2 as ref

    from h2p_rsd_junipr.config import load_config
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model

    torch.manual_seed(0)
    jets = ref.synthetic_matched_dataset(64, seed=0)
    cfg = load_config(["model=ar_junipr_v2", "encoder=gru", "encoder.mask_padding=false",
                       "model.lnz_support=legacy", "decode.continue_temperature=1.0",
                       "decode.kappa_min_mode=0.0"])
    geom = Geometry.from_config(cfg.geometry)
    ds = MatchedLundDataset(jets, geom)

    old = ref.ConditionalPrimaryLundJUNIPR().eval()
    new = build_model(cfg, geom).eval()
    enc_prefixes = ("x_feat.", "encoder.", "to_ctx.")
    new.load_state_dict(
        {("encoder_net." + k if any(k.startswith(p) for p in enc_prefixes) else k): v
         for k, v in old.state_dict().items()}, strict=False)

    assert new.kappa_min_mode == 0.0 and new.continue_temperature == 1.0
    for i in range(3):
        xf, nx = ds[i]["xf"].unsqueeze(0), torch.tensor([ds[i]["nx"]])
        a = old.map_tree(xf, nx, beam_width=8, topk_cells=6, max_emissions=25)
        b = new.map_estimate(xf, nx, beam_width=8, topk_cells=6, max_emissions=25,
                             min_emissions=0, length_penalty=0.0)
        assert [n.cell for n in a.nodes] == [n.cell for n in b.nodes]
        assert a.logprob == pytest.approx(b.logprob, abs=1e-5)
        for p, q in zip(a.nodes, b.nodes):
            assert p.psi == pytest.approx(q.psi, abs=1e-6)
            assert p.ln_z == pytest.approx(q.ln_z, abs=1e-6)
        torch.manual_seed(123)
        da = old.sample_batch(xf, nx, 64, max_emissions=25)
        torch.manual_seed(123)
        assert new.sample(xf, nx, 64, max_emissions=25) == da
