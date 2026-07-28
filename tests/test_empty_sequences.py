"""Jets with an EMPTY groomed sequence, at hadron level and at parton level.

Both are physical and both are common in real data: in `cpp/test_data/jets.root`
(PYTHIA 8.3, z_cut=0.1, k_t floor 1 GeV) **6.9% of jets have no hadron-level primary
emission** and 16.0% have no parton-level one. The synthetic generator produces neither,
which is exactly why this went unnoticed — batched training never sees it either, because
`collate` pads to the batch maximum, so `Mx >= 1` as long as one jet in the batch is
non-empty.

Every PER-JET inference path does see it, though: there `Mx = nx`, and `nn.GRU` raises
`Expected sequence length to be larger than 0` on a zero-length sequence. That made the
whole eval suite unusable on real data. These tests pin the fix at both levels — the
encoders, and the models that consume them.
"""

from __future__ import annotations

import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.encoders.base import build_encoder
from h2p_rsd_junipr.features import N_NODE_FEAT
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

ENCODERS = ["gru", "lundnet", "deepsets"]
FAMILIES = [
    ["model=ar_junipr_v2"],
    ["model=ar_junipr_v3"],
    ["model=ar_junipr_v4"],
    ["model=cinn"],
    ["model=cfm", "model.n_ode_steps=8"],
    ["model=diffusion"],
]


def _empty_x():
    return torch.zeros(1, 0, N_NODE_FEAT), torch.tensor([0])


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ENCODERS)
def test_encoder_handles_an_empty_hadron_sequence(name):
    enc = build_encoder(load_config([f"encoder={name}"]).encoder, 64, N_NODE_FEAT).eval()
    xf, nx = _empty_x()
    e = enc(xf, nx)
    assert e.shape == (1, 64) and torch.isfinite(e).all()
    seq, mask = enc.forward_seq(xf, nx)
    assert seq.shape == (1, 0, enc.seq_dim) and mask.shape == (1, 0)


@pytest.mark.parametrize("name", ENCODERS)
def test_empty_jet_in_a_padded_batch_matches_the_single_jet_call(name):
    """The batched and per-jet routes must agree for an empty jet — otherwise training
    (batched, padded) and evaluation (per-jet) would silently model different things."""
    enc = build_encoder(load_config([f"encoder={name}"]).encoder, 64, N_NODE_FEAT).eval()
    torch.manual_seed(0)
    padded = torch.randn(2, 4, N_NODE_FEAT)
    padded[0] = 0.0                       # jet 0 is empty, padded to the batch maximum
    with torch.inference_mode():
        batched = enc(padded, torch.tensor([0, 4]))
        alone = enc(*_empty_x())
    assert torch.allclose(batched[0], alone[0], atol=1e-6)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=lambda s: s[0].split("=")[1])
@pytest.mark.parametrize("enc", ["gru", "deepsets"])
def test_every_family_infers_on_an_empty_hadron_jet(sel, enc):
    geom = Geometry()
    model = build_model(load_config([*sel, f"encoder={enc}"]), geom).eval()
    xf, nx = _empty_x()
    with torch.inference_mode():
        draws = model.sample(xf, nx, 4)
        assert len(draws) == 4
        pe = model.map_estimate(xf, nx)
        assert pe.multiplicity == len(pe.nodes) >= 1     # default min_emissions=1
        batch = {"xf": xf, "nx": nx, "yc": torch.tensor([[3, 7]]),
                 "ny": torch.tensor([2]), "yraw": torch.zeros(1, 2, 4)}
        assert torch.isfinite(model.log_prob(batch)).all()


@pytest.mark.parametrize("sel", FAMILIES, ids=lambda s: s[0].split("=")[1])
def test_every_family_scores_an_empty_PARTON_tree(sel):
    """`ny = 0` is the other empty case — 16% of the real sample. It must produce a
    finite log-density (the model is entitled to say the empty tree is likely), not a
    shape error from the L == 0 branches."""
    geom = Geometry()
    model = build_model(load_config([*sel, "encoder=gru"]), geom).eval()
    torch.manual_seed(0)
    batch = {"xf": torch.randn(1, 3, N_NODE_FEAT), "nx": torch.tensor([3]),
             "yc": torch.zeros(1, 0, dtype=torch.long), "ny": torch.tensor([0]),
             "yraw": torch.zeros(1, 0, 4)}
    with torch.inference_mode():
        assert torch.isfinite(model.log_prob(batch)).all()


def test_cross_attention_row_with_no_hadron_nodes_is_finite():
    """A fully key-masked row: softmax over nothing. Some torch versions return NaN,
    which would poison the whole batch's gradients — the residual must be dropped."""
    geom = Geometry()
    model = build_model(load_config(["model=ar_junipr_v4", "encoder=deepsets"]), geom).eval()
    torch.manual_seed(0)
    xf = torch.randn(2, 4, N_NODE_FEAT)
    xf[0] = 0.0
    batch = {"xf": xf, "nx": torch.tensor([0, 4]),
             "yc": torch.tensor([[3, 7], [1, 2]]), "ny": torch.tensor([2, 2]),
             "yraw": torch.zeros(2, 2, 4)}
    with torch.inference_mode():
        lp = model.log_prob(batch)
    assert torch.isfinite(lp).all()
    # ...and the empty jet gets the plain pooled-context decoder, i.e. no residual
    with torch.inference_mode():
        e = model.encode(batch["xf"], batch["nx"])
        kv = model.xattn_kv(batch["xf"], batch["nx"])
        with_attn = model._decode_states(batch["yc"], e, kv)
        without = model._decode_states(batch["yc"], e, None)
    assert torch.allclose(with_attn[0], without[0], atol=1e-6)      # empty jet: unchanged
    assert not torch.allclose(with_attn[1], without[1], atol=1e-4)  # real jet: attended


def test_full_pipeline_on_a_dataset_containing_empty_jets():
    """The end-to-end shape of the bug: a dataset with empty jets, a padded batch, and
    the per-jet loop the eval suite runs."""
    geom = Geometry()
    import numpy as np

    jets = []
    for n_x, n_y in ((0, 2), (3, 0), (0, 0), (2, 3)):
        jets.append({
            "weight": 1.0, "event": None,
            "x": tuple(np.zeros(n_x, dtype=np.float32) for _ in range(4)),
            "y": tuple(np.zeros(n_y, dtype=np.float32) for _ in range(4)),
        })
    ds = MatchedLundDataset(jets, geom)
    model = build_model(load_config(["model=ar_junipr_v3", "encoder=gru"]), geom).eval()
    with torch.inference_mode():
        assert torch.isfinite(model.log_prob(collate([ds[i] for i in range(4)]))).all()
        for i in range(4):                                  # the per-jet eval loop
            item = ds[i]
            xf = item["xf"].unsqueeze(0)
            nx = torch.tensor([item["nx"]])
            assert len(model.sample_batch(xf, nx, 3)) == 3
            assert model.map_estimate(xf, nx).multiplicity >= 1
