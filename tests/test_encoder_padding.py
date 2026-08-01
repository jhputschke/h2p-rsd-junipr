"""A jet must encode the same alone as it does in a batch.

`collate` pads `xf` to the batch maximum, and two encoders read that padding:

* `gru` runs a BIDIRECTIONAL GRU over the padded tensor, so the backward pass starts in
  the padding and sweeps through it into the real nodes;
* `lundnet`'s chain EdgeConv takes each node's neighbour as `i+1` and self-loops at
  `h[:, -1]` — the last row of the PADDED tensor, not each jet's last real node, so a
  jet's final node read its neighbour out of the padding.

Both make `e(x)` a function of the batch composition. That is wrong on its own terms, and
it also splits training (always batched) from single-jet inference (`sample`,
`map_estimate`, `length_pmf`), where `Mx == nx` and no padding exists — the model is then
asked to decode from a context distribution it never saw. Measured on the production-test
checkpoint, mean `q(0|x)` came out 0.053 single-jet and 0.155 batched: the same jets.

`mask_padding=False` reproduces the defect and exists only for `scripts/verify_parity.py`,
which measures against the original script — the reference has it too.
"""

from __future__ import annotations

import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models.base import build_model

ENCODERS = ["gru", "lundnet", "deepsets"]
N_FEAT = 5


def _model(encoder, mask_padding=True, seed=0):
    cfg = load_config([f"encoder={encoder}",
                       f"encoder.mask_padding={str(mask_padding).lower()}"])
    geom = Geometry.from_config(cfg.geometry)
    torch.manual_seed(seed)
    return build_model(cfg, geom).eval(), geom


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("nx", [1, 2, 3])
def test_encode_is_invariant_to_padding(encoder, nx):
    """The headline property. `nx = 1` is the case that matters most: it is the modal
    multiplicity on real jets, and for it the single node IS the last node, so the whole
    context was corrupted."""
    m, _ = _model(encoder)
    torch.manual_seed(7)
    x = torch.randn(1, nx, N_FEAT)
    n = torch.tensor([nx])
    with torch.inference_mode():
        tight = m.encode(x, n)
        for pad_to in (nx + 1, nx + 4, nx + 9):
            padded = torch.zeros(1, pad_to, N_FEAT)
            padded[0, :nx] = x[0]
            assert torch.allclose(m.encode(padded, n), tight, atol=1e-6), (
                f"{encoder}: encoding changed when padded from {nx} to {pad_to}"
            )


@pytest.mark.parametrize("encoder", ENCODERS)
def test_a_jet_encodes_the_same_whatever_it_is_batched_with(encoder):
    """The operational form: the same jet, alone and beside a much longer one."""
    m, geom = _model(encoder)
    jets = synthetic_matched_dataset(32, seed=0)
    ds = MatchedLundDataset(jets, geom)
    lens = [ds[i]["nx"] for i in range(len(ds))]
    short = min(range(len(ds)), key=lambda i: lens[i])
    long_ = max(range(len(ds)), key=lambda i: lens[i])
    if lens[short] == lens[long_]:
        pytest.skip("synthetic sample has no length spread")

    with torch.inference_mode():
        alone = m.encode(*(lambda b: (b["xf"], b["nx"]))(collate([ds[short]])))
        together = m.encode(*(lambda b: (b["xf"], b["nx"]))(collate([ds[short], ds[long_]])))
    assert torch.allclose(alone[0], together[0], atol=1e-6), (
        f"{encoder}: a jet's context depends on its batch-mates"
    )


@pytest.mark.parametrize("encoder", ENCODERS)
def test_per_jet_nll_is_invariant_to_batching(encoder):
    """What the invariance is FOR: a held-out NLL must not depend on how the eval loop
    happened to chunk the file."""
    m, geom = _model(encoder, seed=3)
    ds = MatchedLundDataset(synthetic_matched_dataset(24, seed=1), geom)
    with torch.inference_mode():
        one_by_one = torch.cat([m.per_jet_nll(collate([ds[i]])) for i in range(len(ds))])
        in_one_batch = m.per_jet_nll(collate([ds[i] for i in range(len(ds))]))
    assert torch.allclose(one_by_one, in_one_batch, atol=1e-5), (
        f"{encoder}: per-jet NLL depends on the batch it was computed in "
        f"(max delta {float((one_by_one - in_one_batch).abs().max()):.3e})"
    )


@pytest.mark.parametrize("encoder", ["gru", "lundnet"])
def test_mask_padding_false_reproduces_the_defect(encoder):
    """The legacy path must still be padding-SENSITIVE, or `verify_parity.py` would be
    comparing against something the reference does not do."""
    m, _ = _model(encoder, mask_padding=False)
    torch.manual_seed(7)
    x = torch.randn(1, 1, N_FEAT)
    n = torch.tensor([1])
    padded = torch.zeros(1, 5, N_FEAT)
    padded[0, :1] = x[0]
    with torch.inference_mode():
        assert not torch.allclose(m.encode(padded, n), m.encode(x, n), atol=1e-6), (
            f"{encoder}: mask_padding=False no longer reproduces the reference's "
            f"behaviour, so the parity harness is measuring the wrong thing"
        )


def test_deepsets_was_already_correct():
    """A permutation-invariant masked sum never read the padding; pinned so a future
    refactor cannot quietly regress it into the same trap."""
    m, _ = _model("deepsets", mask_padding=False)
    torch.manual_seed(7)
    x = torch.randn(1, 2, N_FEAT)
    n = torch.tensor([2])
    padded = torch.zeros(1, 6, N_FEAT)
    padded[0, :2] = x[0]
    with torch.inference_mode():
        assert torch.allclose(m.encode(padded, n), m.encode(x, n), atol=1e-6)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_a_pre_field_snapshot_backfills_to_the_legacy_path(encoder):
    """A checkpoint written before `mask_padding` existed was TRAINED with the defect.
    It must evaluate that way — masking it would put it in a regime it never saw, which
    is a different silent error, not a fix."""
    from omegaconf import OmegaConf

    cfg = load_config([f"encoder={encoder}"])
    snap = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    del snap.encoder.mask_padding                      # a pre-field snapshot
    geom = Geometry.from_config(snap.geometry)
    torch.manual_seed(0)
    legacy = build_model(snap, geom).eval()
    torch.manual_seed(0)
    explicit, _ = _model(encoder, mask_padding=False)

    torch.manual_seed(7)
    x = torch.randn(1, 1, N_FEAT)
    padded = torch.zeros(1, 4, N_FEAT)
    padded[0, :1] = x[0]
    n = torch.tensor([1])
    with torch.inference_mode():
        assert torch.allclose(legacy.encode(padded, n), explicit.encode(padded, n), atol=1e-6)
