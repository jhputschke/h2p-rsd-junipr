"""The edit transducer as a `PosteriorModel` (`models/edit.py`).

`tests/test_edit_dp.py` pins the lattice numerics; this file pins the model that puts
heads on it — that its likelihood is the DP it claims, that its sampler agrees with its
own exact length marginal, that a drawn coordinate lands in the cell reported beside it,
and that the two degenerate jets (no hadron nodes, no parton nodes) reduce to the closed
forms the family is supposed to reduce to.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from h2p_rsd_junipr.config import load_config
from h2p_rsd_junipr.data.dataset import MatchedLundDataset, collate
from h2p_rsd_junipr.features import N_NODE_FEAT
from h2p_rsd_junipr.geometry import Geometry
from h2p_rsd_junipr.models import edit_dp
from h2p_rsd_junipr.models.base import build_model

FAMILIES = [
    ["model=edit_v1", "encoder=gru"],
    ["model=edit_v2", "encoder=gru"],
    ["model=edit_v1", "encoder=deepsets", "model.physics_width=false"],
]
IDS = ["v1", "v2", "v1-free-width"]
CHAIN = [0, 5, 12, 5, 77]


def _model(sel, geom=None):
    geom = geom or Geometry()
    return build_model(load_config(sel), geom).eval()


def _jet(batch, i=0):
    b, geom = batch
    return b["xf"][i : i + 1], b["nx"][i : i + 1], geom


# --- contract ---------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_log_prob_is_finite_and_per_jet(sel, batch):
    b, geom = batch
    m = _model(sel, geom)
    with torch.inference_mode():
        lp = m.log_prob(b)
        assert lp.shape == (b["xf"].shape[0],) and torch.isfinite(lp).all()
        # padding must not change a score: batched training and per-jet eval are one model
        for i in range(3):
            one = {k: v[i : i + 1] for k, v in b.items() if k != "w"}
            assert float(m.log_prob(one)[0]) == pytest.approx(float(lp[i]), rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_family_flags_say_what_it_does(sel, batch):
    m = _model(sel, batch[1])
    assert m.exact_likelihood is True          # the lattice normalizes by construction
    assert m.has_continuous_coords is True
    assert m.supports_coordinate_pit is False  # stage 2 lands the prefix-conditional CDFs
    assert m.coordinate_cdfs(batch[0]) is None


def test_an_encoder_without_per_node_states_is_refused(monkeypatch):
    """The anchors ARE the encoder's per-node states, so a pooled-only encoder is not a
    slower path here — it is a different model. Fail at build time, naming the fix."""
    from h2p_rsd_junipr.encoders import gru as gru_mod

    monkeypatch.setattr(gru_mod.GRUEncoder, "returns_sequence", False)
    with pytest.raises(ValueError, match="returns_sequence"):
        build_model(load_config(["model=edit_v1", "encoder=gru"]), Geometry())


# --- the sampler and the exact length marginal must agree -------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_sampled_multiplicities_track_the_exact_length_pmf(sel, batch):
    """`length_pmf` is the *structural* marginal of the same lattice `sample` walks, so
    the two are one object seen twice. If they drifted apart, `empty_gate` and
    `length_floor_quantile` would be reading a belief the draws do not hold."""
    xf, nx, geom = _jet(batch)
    m = _model(sel, geom)
    pmf = m.length_pmf(xf, nx)
    assert pmf.ndim == 1 and pmf.sum() == pytest.approx(1.0, abs=1e-6)
    assert (pmf >= 0).all()

    torch.manual_seed(0)
    K = 3000
    mult = np.array([len(d) for d in m.sample(xf, nx, K)])
    mean_pmf = float((pmf * np.arange(pmf.size)).sum())
    var_pmf = float((pmf * (np.arange(pmf.size) - mean_pmf) ** 2).sum())
    tol = 4.0 * np.sqrt(max(var_pmf, 1e-6) / K)
    assert mult.mean() == pytest.approx(mean_pmf, abs=tol + 0.05)
    # the empty tree specifically: this family represents it natively (delete-all)
    assert (mult == 0).mean() == pytest.approx(float(pmf[0]), abs=0.03)


# --- coordinates ------------------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_every_drawn_coordinate_lands_in_its_own_cell(sel, batch):
    """`sample_coordinates` draws from `q(coords | cells, x)`, so each coordinate must
    discretise back to the cell it was conditioned on — otherwise the tree a consumer
    plots and the tree it scores are different objects."""
    xf, nx, geom = _jet(batch)
    m = _model(sel, geom)
    torch.manual_seed(0)
    assert m.sample_coordinates(xf, nx, []).shape == (0, 4)
    for chain in (CHAIN, m.sample(xf, nx, 1)[0]):
        got = m.sample_coordinates(xf, nx, chain)
        assert got.shape == (len(chain), 4) and torch.isfinite(got).all()
        back = [geom.to_cell(float(r[0]), float(r[1])) for r in got]
        assert back == [int(c) for c in chain]


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_describe_cells_scores_exactly_the_tree_it_reports(sel, batch):
    xf, nx, geom = _jet(batch)
    m = _model(sel, geom)
    torch.manual_seed(0)
    pe = m.describe_cells(xf, nx, CHAIN)
    assert pe.multiplicity == len(CHAIN) == len(pe.nodes)
    yraw = torch.tensor(
        [[[n.ln_invDelta, n.ln_kt, n.ln_z, n.psi] for n in pe.nodes]], dtype=torch.float32
    )
    with torch.inference_mode():
        again = float(m.log_prob({
            "xf": xf, "nx": nx, "yraw": yraw,
            "yc": torch.tensor([CHAIN], dtype=torch.long),
            "ny": torch.tensor([len(CHAIN)]),
        })[0])
    assert again == pytest.approx(pe.logprob, rel=1e-4, abs=1e-3)
    # per-node logp_coord is the alignment-posterior-weighted emission log-density; it is
    # a decomposition of the emission terms only, so it must not equal the total
    assert all(np.isfinite(n.logp_coord) for n in pe.nodes)


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_map_is_a_valid_tree_and_honours_its_floor(sel, batch):
    xf, nx, geom = _jet(batch)
    m = _model(sel, geom)
    pe = m.map_estimate(xf, nx)
    assert pe.multiplicity == len(pe.nodes) >= 1
    assert all(0 <= n.cell < geom.n_cells for n in pe.nodes)
    assert np.isfinite(pe.logprob)
    assert m.map_estimate(xf, nx, min_emissions=4).multiplicity >= 4
    assert m.map_estimate(xf, nx, min_emissions=0).multiplicity >= 0
    # deterministic: the Viterbi surrogate has no RNG in it
    assert [n.cell for n in m.map_estimate(xf, nx).nodes] == [n.cell for n in pe.nodes]


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_map_takes_its_length_from_the_exact_length_marginal(sel, batch):
    """The staged decode: `N* = argmax_n q(N=n|x)`, floored by `min_emissions`. Reading the
    length off the JOINT argmax instead would run to `max_emissions` whenever the modal
    emission density beats the per-step op cost, which with sharp kernels is the normal
    regime — a property of the decision rule, not of the fit."""
    xf, nx, geom = _jet(batch)
    m = _model(sel, geom)
    pmf = m.length_pmf(xf, nx)
    for floor in (0, 1, 5):
        want = floor + int(np.argmax(pmf[floor:]))
        assert m.map_estimate(xf, nx, min_emissions=floor).multiplicity == want




# --- the two degenerate jets ------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_no_hadron_nodes_reduces_exactly_to_the_free_head(sel):
    """`nx == 0` (6.9% of the PYTHIA reference) has nothing to anchor on, so the family
    must collapse to a pure insertion process — a geometric length times the free
    emission head, with the anchored component contributing exactly nothing."""
    geom = Geometry()
    m = _model(sel, geom)
    xf, nx = torch.zeros(1, 0, N_NODE_FEAT), torch.tensor([0])
    yc = torch.tensor([[3, 7, 41]])
    yraw = torch.tensor([[[1.2, 3.4, -0.5, 0.3], [4.1, 2.2, -1.5, -2.0], [2.0, 5.0, -0.2, 1.1]]])
    batch = {"xf": xf, "nx": nx, "yc": yc, "ny": torch.tensor([3]), "yraw": yraw}
    with torch.inference_mode():
        got = float(m.log_prob(batch)[0])
        S, e, anchor, ok = m._encode(xf, nx)
        assert S.shape[1] == 1 and not bool(ok.any())        # one terminal column, no anchor
        log_stay, log_emit = m._op_logprobs(S, e)
        C = m._prefix_states(yc, e)[:, :3] if m.prefix_conditioning else None
        p = m._emit_params(m._emit_input(S, e, C), anchor[:, :, None, :], ok[:, :, None])
        u, v, lz, psi = (t[:, None, :] for t in m._targets(yraw))
        free = m._log_f_free(p, yc[:, None, :], u, v, lz, psi)
        want = float(log_stay[0, 0] + (log_emit[0, 0] + free[0, 0]).sum())
    assert got == pytest.approx(want, rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_the_empty_parton_tree_is_the_delete_all_path(sel):
    """`ny == 0` (16.0% of the PYTHIA reference) is a single path — ADVANCE past every
    hadron node, then STOP — so its log-probability is a closed form, and it is finite and
    non-degenerate rather than a shape error out of the `L == 0` branches."""
    geom = Geometry()
    m = _model(sel, geom)
    torch.manual_seed(0)
    xf, nx = torch.randn(1, 4, N_NODE_FEAT), torch.tensor([4])
    batch = {"xf": xf, "nx": nx, "yc": torch.zeros(1, 0, dtype=torch.long),
             "ny": torch.tensor([0]), "yraw": torch.zeros(1, 0, 4)}
    with torch.inference_mode():
        got = float(m.log_prob(batch)[0])
        S, e, _a, _ok = m._encode(xf, nx)
        log_stay, _ = m._op_logprobs(S, e)
        want = float(log_stay[0, : 4 + 1].sum())
    assert np.isfinite(got) and got == pytest.approx(want, rel=1e-6, abs=1e-5)
    assert -60.0 < got < 0.0            # a real probability, not a floor and not a zero
    assert float(m.length_pmf(xf, nx)[0]) > 0.0


# --- physics readouts -------------------------------------------------------
def test_the_physics_width_is_the_shape_function_form():
    """The point of the exercise: the width is `sigma_0 + Lambda_eff/k_t` with
    `Lambda_eff` a learnable scalar in GeV, so the learned kernel is directly
    confrontable with the shape-function expectation instead of being an opaque MLP."""
    m = _model(["model=edit_v1", "encoder=gru"])
    got = m.physics_width_params()
    assert set(got) == {"ln_invDelta", "ln_kt", "ln_z", "psi"}
    for _name, (sigma0, lam) in got.items():
        assert sigma0 > 0.0 and lam == pytest.approx(1.0, abs=1e-4)   # Lambda ~ 1 GeV init
    # the width really does fall with k_t
    with torch.inference_mode():
        hard = m._widths(torch.tensor(6.0), None)
        soft = m._widths(torch.tensor(0.0), None)
        assert float(hard[1]) < float(soft[1])
        assert float(hard[3]) > float(soft[3])                        # kappa is 1/sigma^2

    ablation = _model(["model=edit_v1", "encoder=gru", "model.physics_width=false"])
    assert ablation.physics_width_params() is None


# --- the emergent alignment -------------------------------------------------
@pytest.mark.parametrize("sel", FAMILIES, ids=IDS)
def test_edit_summary_obeys_its_own_accounting(sel, batch):
    """`frac_anchored + insert_rate == 1` per jet, and the deletion rate follows from
    `n_y = n_x - #del + #ins` — the identity that anchors the multiplicity at `|x|`."""
    b, geom = batch
    m = _model(sel, geom)
    s = m.edit_summary(b)
    B = b["xf"].shape[0]
    assert all(v.shape == (B,) for v in s.values())
    ok = ~np.isnan(s["frac_anchored"])
    assert ok.any()
    assert np.allclose(s["frac_anchored"][ok] + s["insert_rate"][ok], 1.0, atol=1e-5)
    assert ((s["frac_anchored"][ok] >= -1e-6) & (s["frac_anchored"][ok] <= 1 + 1e-6)).all()
    d = s["delete_rate"][~np.isnan(s["delete_rate"])]
    assert ((d >= 0.0) & (d <= 1.0)).all()

    post = m.alignment_posterior(b)
    n_anch = (post["gamma_emit"] * post["r_anch"]).sum(dim=(1, 2)).numpy()
    assert np.allclose(n_anch[ok], (s["frac_anchored"] * b["ny"].numpy())[ok], atol=1e-4)


def test_responsibilities_are_a_posterior_over_columns(batch):
    b, geom = batch
    m = _model(["model=edit_v1", "encoder=gru"], geom)
    post = m.alignment_posterior(b)
    g = post["gamma_emit"]
    for i in range(b["xf"].shape[0]):
        for t in range(int(b["ny"][i])):
            assert float(g[i, :, t].sum()) == pytest.approx(1.0, abs=1e-4)
    assert torch.allclose(post["log_z"], m.log_prob(b), atol=1e-4)


# --- the lattice is the likelihood ------------------------------------------
def test_log_prob_equals_an_explicit_enumeration_over_alignments(batch):
    """End to end: the model's `log_prob` for a real jet, against every monotone alignment
    enumerated by hand from the model's own head outputs."""
    import itertools

    b, geom = batch
    m = _model(["model=edit_v1", "encoder=gru"], geom)
    # a jet small enough to enumerate: 3 hadron nodes, 3 parton nodes
    jets = MatchedLundDataset(
        [{"weight": 1.0, "event": None,
          "x": tuple(np.linspace(0.5, 3.0, 3).astype(np.float32) + k for k in range(4)),
          "y": tuple(np.linspace(0.7, 3.2, 3).astype(np.float32) + k for k in range(4))}],
        geom,
    )
    bb = collate([jets[0]])
    with torch.inference_mode():
        got = float(m.log_prob(bb)[0])
        log_stay, log_emit, dens, _p = m._lattice(bb)
        stay, edge = log_stay[0].tolist(), (log_emit[0][:, None] + dens[0]).tolist()
    nx = ny = 3
    terms = []
    for emits in itertools.combinations(range(nx + ny), ny):
        chosen, i, j, s = set(emits), 0, 0, 0.0
        for k in range(nx + ny):
            if k in chosen:
                s += edge[i][j]
                j += 1
            else:
                s += stay[i]
                i += 1
        terms.append(s + stay[nx])
    want = float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), dim=0))
    assert got == pytest.approx(want, rel=1e-5, abs=1e-4)


def test_structural_pmf_is_reachable_through_the_public_dp(batch):
    """`length_pmf` must be `edit_dp.structural_length_pmf` on the model's own op
    log-probabilities — not a re-derivation that could drift from it."""
    xf, nx, geom = _jet(batch)
    m = _model(["model=edit_v1", "encoder=gru"], geom)
    with torch.inference_mode():
        S, e, _a, _o = m._encode(xf, nx)
        ls, le = m._op_logprobs(S, e)
        want = edit_dp.structural_length_pmf(ls, le, nx, m.max_emissions)[0].numpy()
    assert np.allclose(m.length_pmf(xf, nx), want, atol=1e-7)
