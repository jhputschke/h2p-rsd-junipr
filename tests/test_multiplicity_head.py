"""First-class multiplicity head on AR-JUNIPR (docs/PLAN_MultHead.md): the
factorization q(y|x) = q(N|x) q(y|N,x). Covers the categorical head, exact
length_pmf, N ~ q(N|x) sampling, fixed-length MAP, describe/log_prob consistency,
the head-off parity/back-compat guarantees, and the MBR q(N|x) reweighting.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import decode_params, load_config
from h2p_rsd_junipr.inference import mbr
from h2p_rsd_junipr.inference.point_estimate import LundPointEstimate
from h2p_rsd_junipr.models.base import build_model

POT_OK = importlib.util.find_spec("ot") is not None


class _FixedHead(torch.nn.Module):
    """A multiplicity head returning fixed logits regardless of the jet — pins
    q(N|x) to a known shape (same device pattern as tests/test_decode_plumbing.py)."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, e):
        return self.logits.expand(e.shape[0], -1)


def _v3(batch, extra=None):
    b, geom = batch
    cfg = load_config(["model=ar_junipr_v3", "encoder=gru"] + (extra or []))
    return build_model(cfg, geom).eval(), b, geom


def _pin(model, pmf: torch.Tensor):
    model.n_head = _FixedHead(torch.log(pmf.clamp(min=1e-12)))


# --- config / construction ------------------------------------------------------
def test_v3_config_and_head_gating(batch):
    model, _, _ = _v3(batch)
    assert model.use_multiplicity_head is True
    assert model.max_emissions == 25
    assert hasattr(model, "n_head") and not hasattr(model, "cont_head")


def test_v2_unchanged_head_off(batch):
    """The default AR model is byte-identical in structure: cont_head, no n_head."""
    b, geom = batch
    model = build_model(load_config(["model=ar_junipr_v2", "encoder=gru"]), geom).eval()
    assert model.use_multiplicity_head is False
    assert hasattr(model, "cont_head") and not hasattr(model, "n_head")
    assert torch.isfinite(model.log_prob(b)).all()


# --- likelihood / length_pmf ----------------------------------------------------
def test_log_prob_finite_and_shaped(batch):
    model, b, _ = _v3(batch)
    lp = model.log_prob(b)
    assert lp.shape == (b["xf"].shape[0],)
    assert torch.isfinite(lp).all()


def test_length_pmf_is_exact_softmax_head(batch):
    model, b, _ = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    pmf = model.length_pmf(xf, nx)
    assert pmf.shape == (model.max_emissions + 1,)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)
    with torch.inference_mode():
        ref = torch.softmax(model.n_head(model.encode(xf, nx)), dim=-1).squeeze(0).numpy()
    assert np.allclose(pmf, ref, atol=1e-6)


def test_length_pmf_batches_consistently(batch):
    """One padded batch through `n_head(encode(...))` must reproduce the per-jet
    `length_pmf` row for row, to float32 tolerance.

    This is the identity `notebooks/prod_test_v1.ipynb` §6 relies on to replace 40,000
    batch-1 calls (13 min, docs/PLAN_prod_test_speedup.md §3) with a chunked pass. No RNG
    is involved, so what this pins is the COMPOSITION: the batched helper has to apply the
    head, `recalibrated_n_logits` and the softmax in the same order `length_pmf` does.

    How much of that composition is actually covered, measured rather than assumed: the
    fixture leaves `length_temperature=1.0` / `length_tilt=0.0`, where
    `recalibrated_n_logits` is the IDENTITY — deleting the call outright still passes here
    (7.4e-9, unchanged). So this covers the head and the softmax, not the recalibration.
    The notebook runs with those knobs live — cell 35 saves and restores them around its
    own check — and with them on, dropping the call moves the pmf by 1.5e-01. If that step
    becomes load-bearing, parameterise the knobs here; today it is untested.

    NOT a padding check, despite what this claimed until now. `b["xf"][i:i + 1]` slices the
    batch dimension only, so both sides hand the encoder the same padded width and the
    comparison is structurally blind to a leak: measured with `encoder.mask_padding=False`,
    i.e. the defect fully switched on, it moves by 4e-9 and the old `array_equal` would
    have sailed straight through. That invariant is guarded properly — against the trimmed
    `Mx == nx` single-jet path, and from both directions — in `tests/test_encoder_padding.py`.

    Tolerance rather than equality, because batched float32 GEMM is not batch-size
    invariant: a bare `nn.Linear` on (8, 9, 32) already differs from the same rows fed as
    (1, 9, 32) by ~3e-7 on Apple Silicon (bit-identical on Linux/x86), which reaches the
    pmf as ~2 ulp. `atol` matches `test_encoder_padding.py`, and still leaves ~200x
    headroom to the 2e-4 a real leak produces on the comparison that can actually see one."""
    model, b, _ = _v3(batch)
    B = b["xf"].shape[0]
    per_jet = [model.length_pmf(b["xf"][i:i + 1], b["nx"][i:i + 1]) for i in range(B)]
    with torch.inference_mode():
        e = model.encode(b["xf"], b["nx"])
        batched = torch.softmax(model.recalibrated_n_logits(model.n_head(e)), dim=-1).numpy()
    assert batched.shape == (B, model.max_emissions + 1)
    for i, p in enumerate(per_jet):
        np.testing.assert_allclose(
            batched[i], p, rtol=0, atol=1e-6,
            err_msg=f"jet {i}: batched pmf differs from batch-1",
        )


def test_describe_matches_log_prob_of_modes(batch):
    """Factorization consistency: describe_sequence's total (log q(N|x) + Σ cell + Σ
    coord at the head-mode coords) equals -per_jet_nll evaluated at those same coords."""
    model, b, _ = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    pe = model.map_estimate(xf, nx, min_emissions=2)
    cells = [n.cell for n in pe.nodes]
    yraw = torch.tensor(
        [[[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes]], dtype=torch.float32
    )
    b1 = {"xf": xf, "nx": nx, "yc": torch.tensor([cells]),
          "ny": torch.tensor([len(cells)]), "yraw": yraw}
    with torch.inference_mode():
        assert pe.logprob == pytest.approx(float(model.log_prob(b1)[0]), abs=1e-3)


# --- sampling: N ~ q(N|x) --------------------------------------------------------
def test_sample_lengths_follow_qn(batch):
    """With q(N|x) pinned to a two-point mass at {2, 6}, drawn lengths take only those
    values in ~equal proportion (the length now comes from the head, not a stop draw)."""
    model, b, _ = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    pmf = torch.zeros(model.max_emissions + 1)
    pmf[2], pmf[6] = 0.5, 0.5
    _pin(model, pmf)
    torch.manual_seed(0)
    lengths = np.array([len(d) for d in model.sample(xf, nx, 400)])
    assert set(np.unique(lengths)).issubset({2, 6})
    assert 0.35 < (lengths == 2).mean() < 0.65        # ~50/50 within sampling noise


# --- MAP: floored argmax of q(N|x) ----------------------------------------------
def test_map_length_is_floored_argmax(batch):
    model, b, _ = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    pmf = torch.zeros(model.max_emissions + 1)
    pmf[4] = 1.0
    _pin(model, pmf)
    assert model.map_estimate(xf, nx, min_emissions=1).multiplicity == 4   # argmax
    assert model.map_estimate(xf, nx, min_emissions=7).multiplicity == 7   # floor raises it
    mp = model.map_estimate(xf, nx, min_emissions=1)
    assert mp.multiplicity == len(mp.nodes)                                # a valid tree


def test_map_never_empty_default_floor(batch):
    model, b, _ = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    pmf = torch.zeros(model.max_emissions + 1)
    pmf[0] = 1.0                                        # head prefers the empty tree...
    _pin(model, pmf)
    assert model.map_estimate(xf, nx).multiplicity >= 1  # ...but default min_emissions=1 floors it


# --- checkpoint round-trip ------------------------------------------------------
def test_v3_checkpoint_roundtrip(tmp_path, batch):
    from h2p_rsd_junipr.train.checkpoint import load_for_inference, save_checkpoint
    from h2p_rsd_junipr.train.trainer import build_components

    b, geom = batch
    cfg = load_config(["model=ar_junipr_v3", "encoder=gru"])
    model, opt, sched = build_components(cfg, geom, torch.device("cpu"))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "v3.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=1, step=1, best_val=0.0, cfg=cfg)
    info = load_for_inference(path)
    assert info["model_name"] == "ar_junipr_v3"
    m2 = build_model(cfg, geom)
    m2.load_state_dict(info["model_state"])            # strict: n_head keys present both sides
    model.eval()
    m2.eval()
    with torch.inference_mode():
        assert torch.allclose(model.log_prob(b), m2.log_prob(b), atol=1e-6)


# --- MBR q(N|x) reweighting -----------------------------------------------------
@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_mbr_resample_to_qn_off_is_plain_mean(batch):
    """resample_to_qn=False reproduces the plain-mean-risk argmin exactly."""
    model, b, geom = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    draws = [[12, 34, 56], [12, 34], [5, 34, 56], [12, 30, 56]] * 3
    a = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend="pot", resample_to_qn=False)
    c = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend="pot", resample_to_qn=False)
    assert a.multiplicity == c.multiplicity and a.risk == pytest.approx(c.risk)


@pytest.mark.skipif(not POT_OK, reason="POT not installed")
def test_mbr_resample_to_qn_reweights_and_stays_nonempty(batch):
    """A q(N|x) pinned toward the longer draws shifts the weighted risk toward longer
    candidates, yet the estimate never collapses to the empty tree."""
    model, b, geom = _v3(batch)
    xf, nx = b["xf"][:1], b["nx"][:1]
    pmf = torch.zeros(model.max_emissions + 1)
    pmf[3] = 1.0                                        # all q(N|x) mass on N=3
    _pin(model, pmf)
    # a mix of 2- and 3-emission draws (+ empties): reweighting up-weights the 3s
    draws = [[12, 34], [5, 30]] * 4 + [[12, 34, 56], [5, 34, 56]] * 4 + [[], []]
    pe = mbr.mbr_select(model, xf, nx, draws=draws, geom=geom, backend="pot", resample_to_qn=True)
    assert isinstance(pe, LundPointEstimate)
    assert pe.multiplicity >= 1 and pe.multiplicity == len(pe.nodes)
    assert np.isfinite(pe.risk)


# --- support guard (docs/PLAN_UPDATES.md WP4) -----------------------------------
def test_head_support_covers_the_synthetic_generator(small_jets):
    """The head's categorical support must cover the data it will be trained on: a
    truth longer than `max_emissions` is clamped into the last bin and gets the WRONG
    likelihood. Asserted here, next to the head, as well as in the datamodule guard."""
    from h2p_rsd_junipr.data.stats import check_multiplicity_support

    cfg = load_config(["model=ar_junipr_v3", "encoder=gru"])
    stats = check_multiplicity_support(small_jets, cfg)
    assert stats["max"] <= int(cfg.model.max_emissions)
    assert stats["tail_fraction"] == 0.0


def test_decode_carries_mbr_resample_flag():
    cfg = load_config(["model=ar_junipr_v3", "encoder=gru"])
    dec = decode_params(cfg)
    assert dec["mbr_resample_to_qn"] is False          # default off (parity)
    assert mbr.mbr_kwargs_from_decode({**dec, "mbr_resample_to_qn": True})["resample_to_qn"] is True
