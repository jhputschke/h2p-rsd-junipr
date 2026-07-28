"""Sequence conditioning: `Encoder.forward_seq` + decoder cross-attention
(docs/PLAN_UPDATES.md WP3).

The pooled `Encoder` contract squeezes every hadron-level node into one `ctx_dim`
vector, tiled at every decoder step — the classic fixed-length bottleneck. The
opt-in fix lets the decoder attend to the per-node states instead.

The properties that make it safe to merge with the switch defaulted off:
  * OFF path is byte-identical — same `state_dict` keys, same NLL, same samples;
  * ON path masks padding correctly (a padded node can never receive weight);
  * the residual form leaves every head's input width unchanged, so old checkpoints
    still load strictly;
  * an encoder without per-node states is a config error, not a silent fallback.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.encoders.base import Encoder, build_encoder, register_encoder
from h2p_rsd_junipr.features import N_NODE_FEAT
from h2p_rsd_junipr.models.base import build_model

ENCODERS = ["gru", "lundnet", "deepsets"]


def _pair(sel_a, sel_b, geom):
    """Two models built from the same seed, so any difference is the config's."""
    torch.manual_seed(0)
    a = build_model(load_config(sel_a), geom).eval()
    torch.manual_seed(0)
    b = build_model(load_config(sel_b), geom).eval()
    return a, b


# ---------------------------------------------------------------------------
# encoder side
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ENCODERS)
def test_forward_seq_shapes_and_mask(name, batch):
    b, _ = batch
    cfg = load_config([f"encoder={name}"])
    enc = build_encoder(cfg.encoder, 64, N_NODE_FEAT).eval()
    assert enc.returns_sequence is True and enc.seq_dim > 0
    seq, mask = enc.forward_seq(b["xf"], b["nx"])
    assert seq.shape == (b["xf"].shape[0], b["xf"].shape[1], enc.seq_dim)
    assert mask.dtype == torch.bool and mask.shape == b["xf"].shape[:2]
    # the mask IS the padding structure, not an approximation of it
    assert torch.equal(mask.sum(1), b["nx"])
    for i, n in enumerate(b["nx"].tolist()):
        assert bool(mask[i, :n].all()) and not bool(mask[i, n:].any())


@pytest.mark.parametrize("name", ENCODERS)
def test_forward_is_unchanged_by_the_refactor(name, batch):
    """`forward` and `forward_seq` share one `_states` call, so the pooled output must
    still be exactly the masked mean of the per-node states (+ the multiplicity feature)."""
    b, _ = batch
    cfg = load_config([f"encoder={name}"])
    enc = build_encoder(cfg.encoder, 64, N_NODE_FEAT).eval()
    with torch.inference_mode():
        seq, mask = enc.forward_seq(b["xf"], b["nx"])
        m = mask.float().unsqueeze(-1)
        pooled = (seq * m).sum(1) / m.sum(1).clamp(min=1.0)
        nx_feat = torch.log1p(b["nx"].float()).unsqueeze(-1)
        proj = enc.to_ctx if hasattr(enc, "to_ctx") else enc.rho
        assert torch.allclose(enc(b["xf"], b["nx"]),
                              proj(torch.cat([pooled, nx_feat], -1)), atol=1e-6)


def test_base_encoder_refuses_forward_seq_without_states():
    class _Pooled(Encoder):
        def __init__(self):
            super().__init__()
            self.out_dim = 4

        def forward(self, xf, nx):
            return torch.zeros(xf.shape[0], 4)

    enc = _Pooled()
    assert enc.returns_sequence is False
    with pytest.raises(NotImplementedError, match="returns_sequence=False"):
        enc.forward_seq(torch.zeros(1, 2, N_NODE_FEAT), torch.tensor([2]))


# ---------------------------------------------------------------------------
# OFF-path parity — the merge condition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base", ["ar_junipr_v2", "ar_junipr_v3"])
def test_off_path_state_dict_and_nll_are_identical(base, batch):
    b, geom = batch
    a, c = _pair([f"model={base}"], [f"model={base}", "model.use_cross_attention=false"], geom)
    assert list(a.state_dict()) == list(c.state_dict())
    assert not hasattr(a, "xattn") and not hasattr(a, "kv_proj")
    with torch.inference_mode():
        assert torch.equal(a.log_prob(b), c.log_prob(b))
    torch.manual_seed(1)
    da = a.sample(b["xf"][:1], b["nx"][:1], 8)
    torch.manual_seed(1)
    dc = c.sample(b["xf"][:1], b["nx"][:1], 8)
    assert da == dc
    assert a.map_decode(b["xf"][:1], b["nx"][:1]) == c.map_decode(b["xf"][:1], b["nx"][:1])


def test_off_path_checkpoint_loads_into_an_off_model(tmp_path, batch):
    """A pre-WP3 checkpoint has no xattn keys; strict loading must still succeed."""
    b, geom = batch
    old = build_model(load_config(["model=ar_junipr_v3"]), geom)
    new = build_model(load_config(["model=ar_junipr_v3"]), geom)
    new.load_state_dict(old.state_dict())        # strict, no missing/unexpected keys
    old.eval(), new.eval()
    with torch.inference_mode():
        assert torch.allclose(old.log_prob(b), new.log_prob(b), atol=1e-6)


def test_on_path_adds_exactly_the_attention_parameters(batch):
    b, geom = batch
    off, on = _pair(["model=ar_junipr_v3"], ["model=ar_junipr_v4"], geom)
    extra = set(on.state_dict()) - set(off.state_dict())
    assert all(k.startswith(("xattn.", "kv_proj.")) for k in extra)
    assert set(off.state_dict()) - set(on.state_dict()) == set()   # nothing removed
    # head input widths are unchanged: the residual form is what guarantees it
    assert on.split_head[0].in_features == off.split_head[0].in_features
    assert on.coord_head[0].in_features == off.coord_head[0].in_features


# ---------------------------------------------------------------------------
# ON path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enc", ENCODERS)
def test_on_path_runs_end_to_end(enc, batch):
    b, geom = batch
    m = build_model(load_config(["model=ar_junipr_v4", f"encoder={enc}"]), geom).eval()
    with torch.inference_mode():
        lp = m.log_prob(b)
        assert lp.shape == (b["xf"].shape[0],) and torch.isfinite(lp).all()
        xf, nx = b["xf"][:1], b["nx"][:1]
        draws = m.sample(xf, nx, 6)
        assert len(draws) == 6 and all(0 <= c < geom.n_cells for d in draws for c in d)
        pe = m.map_estimate(xf, nx)
        assert pe.multiplicity == len(pe.nodes) >= 1
        assert m.coordinate_cdfs(b) is not None      # PIT still available under xattn


def test_padding_nodes_receive_zero_attention_weight(batch):
    """The correctness condition for the mask: perturbing a PADDED hadron node must not
    change any output. If the key-padding mask were inverted or dropped, it would."""
    b, geom = batch
    m = build_model(load_config(["model=ar_junipr_v4", "encoder=deepsets"]), geom).eval()
    xf = b["xf"].clone()
    nx = b["nx"]
    with torch.inference_mode():
        base = m.log_prob({**b, "xf": xf})
        poisoned = xf.clone()
        for i, n in enumerate(nx.tolist()):
            if n < poisoned.shape[1]:
                poisoned[i, n:] = 1e3            # garbage beyond the true length
        assert torch.allclose(base, m.log_prob({**b, "xf": poisoned}), atol=1e-5)


def test_attention_actually_uses_the_hadron_states(batch):
    """The complement: perturbing a VALID node must change the likelihood — otherwise
    the attention is wired but inert and the whole WP is a no-op."""
    b, geom = batch
    m = build_model(load_config(["model=ar_junipr_v4", "encoder=deepsets"]), geom).eval()
    with torch.inference_mode():
        base = m.log_prob(b)
        bumped = b["xf"].clone()
        bumped[:, 0, :] += 2.0                   # a real node, inside every jet's length
        assert not torch.allclose(base, m.log_prob({**b, "xf": bumped}), atol=1e-4)


def test_teacher_forced_and_incremental_paths_agree(batch):
    """`_decode_states` (teacher-forced, whole sequence) and `_step_cells` (incremental,
    its own GRU stepping) must apply the SAME cross-attention residual — the failure
    mode the plan flagged: mirror it in the incremental path or sampling silently
    diverges from the likelihood."""
    b, geom = batch
    m = build_model(load_config(["model=ar_junipr_v4", "encoder=gru"]), geom).eval()
    xf, nx = b["xf"][:1], b["nx"][:1]
    cells = [7, 42, 13]
    with torch.inference_mode():
        e = m.encode(xf, nx)
        kv = m.xattn_kv(xf, nx)
        yc = torch.tensor([cells])
        full = m._decode_states(yc, e, kv)                     # (1, L+1, dec)
        h = m._init_hidden(e)
        tok = torch.full((1, 1), m.start_token, dtype=torch.long)
        for t, c in enumerate([*cells]):
            hv, h = m._step_core(tok, e, h, kv)
            assert torch.allclose(hv[:, : m.dec_dim], full[:, t, :], atol=1e-5), t
            tok = torch.tensor([[c]], dtype=torch.long)


def test_sample_shares_one_kv_across_draws(batch):
    """`xattn_kv` is computed once per jet and broadcast over the K draws — cheap, and
    the reason a K=500 posterior does not re-encode the hadron sequence 500 times."""
    b, geom = batch
    m = build_model(load_config(["model=ar_junipr_v4", "encoder=gru"]), geom).eval()
    calls = {"n": 0}
    orig = m.encoder_net.forward_seq

    def counting(xf, nx):
        calls["n"] += 1
        return orig(xf, nx)

    m.encoder_net.forward_seq = counting
    with torch.inference_mode():
        m.sample(b["xf"][:1], b["nx"][:1], 32)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# config errors
# ---------------------------------------------------------------------------
def test_encoder_without_sequence_support_is_a_config_error(batch):
    """A silent fallback to the pooled context would be the worst outcome: the run
    would look fine and quietly not do what was asked."""
    _, geom = batch

    @register_encoder("_pooled_only_test")
    class _PooledOnly(Encoder):
        returns_sequence = False

        def __init__(self, cfg, ctx_dim, n_node_feat):
            super().__init__()
            self.out_dim = int(ctx_dim)
            self.lin = nn.Linear(n_node_feat, self.out_dim)

        def forward(self, xf, nx):
            return self.lin(xf).mean(1)

    cfg = load_config(["model=ar_junipr_v4", "encoder=gru"])
    cfg.encoder.name = "_pooled_only_test"
    with pytest.raises(ValueError, match="returns_sequence=False"):
        build_model(cfg, geom)


def test_head_count_must_divide_dec_dim(batch):
    _, geom = batch
    with pytest.raises(ValueError, match="must divide"):
        build_model(load_config(["model=ar_junipr_v4", "model.xattn_heads=7"]), geom)


def test_v4_selector_matches_the_explicit_flags(batch):
    b, geom = batch
    a, c = _pair(
        ["model=ar_junipr_v4"],
        ["model=ar_junipr_v3", "model.use_cross_attention=true", "model.xattn_heads=4"],
        geom,
    )
    assert list(a.state_dict()) == list(c.state_dict())
    with torch.inference_mode():
        assert torch.allclose(a.log_prob(b), c.log_prob(b), atol=1e-6)
