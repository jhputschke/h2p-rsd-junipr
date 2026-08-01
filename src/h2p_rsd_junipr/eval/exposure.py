"""WP-B of docs/PLAN_prod_test_v1.md: what the multiplicity marginal is actually doing.

v0 reported the length model as broken on two counts — `mean_mult_posterior` 1.15
against a truth of 1.40, and SBC-on-N chi^2 107 against a chi^2(9) 95% point of 16.90.
The plan's WP-B.1 makes the diagnostic run **before** any remedy is defaulted, because
both numbers turn on a comparison the metric never states:

* **Which population.** `mean_mult_posterior` was conditioned on the truth having a
  leading emission and `mean_mult_true` was not, so the two rows were means over
  different jets. Worse, selecting jets by `N_true >= 1` and comparing them to
  `E_q[N|x]` is regression to the mean: the deficit it produces is negative *by
  construction*, even for a perfectly calibrated posterior. `length_marginal` reports
  both populations side by side and names which one gate G4 is about.

* **Which null.** The SBC rank of a DISCRETE quantity cannot be uniform on [0, 1]:
  with `N` effectively taking a handful of values, the mid-rank statistic lands on a
  handful of atoms and a 10-bin chi^2 is large however well calibrated the model is.
  chi^2(9) is the null for a *continuous* rank. `sbc_n_selfconsistency_null` replaces
  it with the only defensible reference — the statistic's own distribution when the
  truth is drawn from `q(N|x)` itself, at this jet sample and this discreteness.
  (Talts et al., arXiv:1804.06788 §5 raise the discrete-rank tie problem; a simulated
  null is the standard answer when the ranks cannot be randomised away.)

* **Which stage.** `continue_prob_by_depth` is the exposure-bias probe proper
  (Ranzato et al., arXiv:1511.06732; Bengio et al., arXiv:1506.03099): the continue
  probability at matched depth, teacher-forced on the truth prefix versus on-policy on
  the sampler's own prefix. A gap that grows with depth is exposure bias; a flat offset
  is a miscalibrated head. It applies only to the implicit continue/stop family — a
  model with an explicit `q(N|x)` head takes no per-step continue decision, and saying
  so is more useful than reporting a number computed from a head that is not there.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import collate


def _batched(val_ds, device, n_jets, chunk=200):
    """Yield collated chunks of `val_ds[0..n_jets)` on `device`."""
    for start in range(0, n_jets, chunk):
        items = [val_ds[i] for i in range(start, min(start + chunk, n_jets))]
        batch = collate(items)
        yield {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def exact_length_pmfs(model, val_ds, device, n_jets=2000, chunk=200):
    """`(P (n_jets, max+1), N_true (n_jets,))` from the model's exact `q(N|x)` head,
    or `None` for a family whose length belief is only the sampler histogram.

    Batched on purpose: the per-jet `length_pmf` is a forward pass per jet, and every
    consumer here wants the same thing for thousands of jets."""
    if not hasattr(model, "n_head"):
        return None
    import torch.nn.functional as F

    n_jets = min(int(n_jets), len(val_ds))
    P, Nt = [], []
    with torch.inference_mode():
        for b in _batched(val_ds, device, n_jets, chunk):
            e = model.encode(b["xf"], b["nx"])
            logits = model.recalibrated_n_logits(model.n_head(e))
            P.append(F.softmax(logits, dim=-1).cpu().numpy())
            Nt.append(b["ny"].cpu().numpy())
    return np.concatenate(P), np.concatenate(Nt)


# ---------------------------------------------------------------------------
# The N marginal, on both populations
# ---------------------------------------------------------------------------
def length_marginal(model, val_ds, device, n_jets=2000, K=200, draws_by_jet=None,
                    verbose=True) -> dict:
    """`<N>` truth vs posterior, on the FULL population and on the truth-nonempty one.

    Gate G4's `<N>_post/<N>_truth` clause is about the **full** population: it asks
    whether the model reproduces the multiplicity marginal of the jets it was given.
    The truth-nonempty row is reported beside it and explicitly flagged, because
    selecting on the truth makes it a biased comparison, not a second measurement of
    the same thing.

    The posterior mean is the exact `E_q[N|x]` where the family has a head for it (no
    Monte-Carlo noise, and it is what `sample` draws from), and the mean over `K`
    sampler draws otherwise. When both exist the sampled row is reported too — a
    disagreement between them would mean the sampler and the head have come apart,
    which is a bug rather than a finding."""
    n_jets = min(int(n_jets), len(val_ds))
    out: dict = {"n_jets": int(n_jets), "K": int(K)}
    exact = exact_length_pmfs(model, val_ds, device, n_jets=n_jets)
    n_true = np.array([len(val_ds[i]["yc"]) for i in range(n_jets)], dtype=float)

    if exact is not None:
        P, n_true = exact
        n_true = n_true.astype(float)
        support = np.arange(P.shape[1], dtype=float)
        n_post = P @ support
        out["posterior_source"] = "exact q(N|x)"
    else:
        n_post = np.array(
            [np.mean([len(d) for d in (
                draws_by_jet[i] if draws_by_jet is not None
                else model.sample_batch(val_ds[i]["xf"].unsqueeze(0).to(device),
                                        torch.tensor([val_ds[i]["nx"]], device=device), K)
            )]) for i in range(n_jets)], dtype=float,
        )
        out["posterior_source"] = f"sampler histogram ({K} draws/jet)"

    keep = n_true >= 1
    out["full"] = {
        "n_jets": int(n_true.size),
        "mean_n_truth": float(n_true.mean()),
        "mean_n_posterior": float(n_post.mean()),
        "ratio": float(n_post.mean() / n_true.mean()) if n_true.mean() else float("nan"),
        "signed_bias": float((n_post - n_true).mean()),
    }
    out["truth_nonempty"] = {
        "n_jets": int(keep.sum()),
        "mean_n_truth": float(n_true[keep].mean()) if keep.any() else float("nan"),
        "mean_n_posterior": float(n_post[keep].mean()) if keep.any() else float("nan"),
        "ratio": (float(n_post[keep].mean() / n_true[keep].mean())
                  if keep.any() and n_true[keep].mean() else float("nan")),
        "signed_bias": float((n_post[keep] - n_true[keep]).mean()) if keep.any() else float("nan"),
        # Not a second measurement of the same quantity: see the docstring.
        "selection_biased": True,
    }
    out["gate_population"] = "full"
    if verbose:
        f, s = out["full"], out["truth_nonempty"]
        print(f"\nN marginal ({out['posterior_source']}, {n_jets} jets):")
        print(f"    {'population':>16} {'jets':>6} {'<N> truth':>10} {'<N> post':>9}"
              f" {'ratio':>7} {'bias':>8}")
        print(f"    {'full (G4)':>16} {f['n_jets']:>6} {f['mean_n_truth']:>10.4f}"
              f" {f['mean_n_posterior']:>9.4f} {f['ratio']:>7.4f} {f['signed_bias']:>+8.4f}")
        print(f"    {'truth N >= 1':>16} {s['n_jets']:>6} {s['mean_n_truth']:>10.4f}"
              f" {s['mean_n_posterior']:>9.4f} {s['ratio']:>7.4f} {s['signed_bias']:>+8.4f}"
              f"   <- SELECTED ON TRUTH: negative bias by construction")
    return out


# ---------------------------------------------------------------------------
# SBC-on-N against its own null
# ---------------------------------------------------------------------------
def _sbc_midranks(P, n):
    """Exact mid-rank of `n` under the pmf rows of `P` — the `K -> inf` limit of
    `#{draws < n} + 0.5 #{draws == n}` over draws from `q(N|x)`, with no sampling
    noise of its own."""
    idx = np.arange(P.shape[0])
    n = np.clip(np.asarray(n, dtype=int), 0, P.shape[1] - 1)
    cdf = np.cumsum(P, axis=1)
    below = np.where(n > 0, cdf[idx, np.maximum(n - 1, 0)], 0.0)
    return below + 0.5 * P[idx, n]


def _chi2_uniform(values, n_bins):
    hist, _ = np.histogram(np.asarray(values, dtype=float), bins=n_bins, range=(0.0, 1.0))
    expected = len(values) / n_bins if len(values) else 1.0
    return float(np.sum((hist - expected) ** 2 / max(expected, 1e-8)))


def sbc_n_selfconsistency_null(model, val_ds, device, n_jets=2000, n_bins=10,
                               n_reps=200, seed=0, verbose=True) -> dict | None:
    """SBC-on-N against a **simulated** null instead of chi^2(n_bins - 1).

    `N` is discrete and, here, effectively supported on a handful of values, so its
    mid-rank statistic lands on a handful of atoms and the 10-bin chi^2 is large for
    ANY model — the chi^2(9) reference belongs to a continuous rank. The null this
    computes is the distribution of the same statistic when the truth is drawn from
    `q(N|x)` itself: same jets, same conditioning, same discreteness, calibrated by
    construction. The observed chi^2 is then quoted as a percentile of that.

    Returns None for a family with no exact `q(N|x)` (the null would need the sampler
    for both halves, which conflates its Monte-Carlo noise with the statistic)."""
    exact = exact_length_pmfs(model, val_ds, device, n_jets=n_jets)
    if exact is None:
        return None
    P, n_true = exact
    obs = _chi2_uniform(_sbc_midranks(P, n_true), n_bins)
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(P, axis=1)
    null = np.empty(int(n_reps), dtype=float)
    for r in range(int(n_reps)):  # inverse-CDF draw of one N per jet from its own pmf
        n_sim = (cdf < rng.random((P.shape[0], 1))).sum(axis=1)
        null[r] = _chi2_uniform(_sbc_midranks(P, n_sim), n_bins)
    pct = float(np.mean(null < obs) * 100.0)
    out = {
        "sbc_n_chi2": obs,
        "sbc_n_chi2_bins": int(n_bins),
        "sbc_n_null_reps": int(n_reps),
        "sbc_n_null_mean": float(null.mean()),
        "sbc_n_null_p95": float(np.percentile(null, 95)),
        "sbc_n_null_p99": float(np.percentile(null, 99)),
        "sbc_n_percentile_in_null": pct,
        "sbc_n_exceeds_null95": bool(obs > np.percentile(null, 95)),
        # The reference the pre-WP-B suite quoted, kept so the two can be compared.
        "sbc_n_chi2_crit95_continuous": float(_chi2_crit95(n_bins - 1)),
        "n_jets": int(P.shape[0]),
        "n_distinct_true": int(np.unique(n_true).size),
    }
    if verbose:
        print(f"\nSBC-on-N against its OWN null ({out['n_jets']} jets, {n_bins} bins, "
              f"{n_reps} reps):")
        print(f"  observed chi^2 = {obs:.1f}"
              f"   simulated null: mean {out['sbc_n_null_mean']:.1f},"
              f" 95% {out['sbc_n_null_p95']:.1f}"
              f"  -> observed sits at the {pct:.0f}th percentile"
              f" ({'ABOVE' if out['sbc_n_exceeds_null95'] else 'below'} the 95% point)")
        print(f"  the continuous-rank reference chi^2({n_bins - 1}) 95% ="
              f" {out['sbc_n_chi2_crit95_continuous']:.2f} is NOT the null here:"
              f" N takes {out['n_distinct_true']} distinct values, so the mid-rank"
              f" statistic cannot be uniform on [0, 1] for any model")
    return out


def _chi2_crit95(dof: int) -> float:
    from .calibration import chi2_crit95

    return chi2_crit95(dof)


# ---------------------------------------------------------------------------
# Exposure bias proper: teacher-forced vs on-policy continue probability
# ---------------------------------------------------------------------------
def continue_prob_by_depth(model, val_ds, device, n_jets=300, K=64, max_depth=8,
                           verbose=True) -> dict | None:
    """Mean continue probability at each depth, teacher-forced vs on-policy.

    Teacher-forced: `p(continue | truth prefix of length t, x)` for every jet whose
    truth reaches depth `t`. On-policy: the same head read at depth `t` along the
    SAMPLER's own prefixes, over chains still alive there. Same head, same depth, two
    prefix distributions — so the difference is the prefix, which is what exposure bias
    means (Ranzato et al., arXiv:1511.06732).

    A gap growing with depth is exposure bias: the sampler has walked into prefixes the
    teacher-forced training never showed it. A constant offset is a miscalibrated head
    and would be fixed by the temperature, not by the prefix distribution.

    Returns None for a family with an explicit `q(N|x)` head — it takes no per-step
    continue decision, so there is no such probability to read."""
    if not hasattr(model, "cont_head"):
        return None
    n_jets = min(int(n_jets), len(val_ds))
    D = int(max_depth)
    tf_sum, tf_n = np.zeros(D + 1), np.zeros(D + 1)
    op_sum, op_n = np.zeros(D + 1), np.zeros(D + 1)

    with torch.inference_mode():
        # --- teacher-forced: one batched pass over the truth prefixes ---------
        for b in _batched(val_ds, device, n_jets):
            e = model.encode(b["xf"], b["nx"])
            out = model._decode_states(b["yc"], e, model.xattn_kv(b["xf"], b["nx"]))
            Lp1 = out.shape[1]
            eh = torch.cat([out, e.unsqueeze(1).expand(-1, Lp1, -1)], dim=-1)
            p = torch.sigmoid(model.cont_head(eh).squeeze(-1)).cpu().numpy()  # (B, L+1)
            ny = b["ny"].cpu().numpy()
            for t in range(min(D + 1, Lp1)):
                sel = ny >= t              # the truth prefix reaches depth t
                if sel.any():
                    tf_sum[t] += float(p[sel, t].sum())
                    tf_n[t] += int(sel.sum())

        # --- on-policy: the sampler's own prefixes, same head -----------------
        for i in range(n_jets):
            item = val_ds[i]
            xf = item["xf"].unsqueeze(0).to(device)
            nx = torch.tensor([item["nx"]], device=device)
            e = model.encode(xf, nx).expand(K, -1).contiguous()
            h = model._init_hidden(e)
            kv = model.xattn_kv(xf, nx)
            tok = torch.full((K, 1), model.start_token, dtype=torch.long, device=device)
            alive = torch.ones(K, dtype=torch.bool, device=device)
            for t in range(D + 1):
                p_cont, split_logits, h = model._step_batched(tok, e, h, kv)
                a = alive.cpu().numpy()
                if a.any():
                    op_sum[t] += float(p_cont.cpu().numpy()[a].sum())
                    op_n[t] += int(a.sum())
                cont = (torch.rand(K, device=device) < p_cont) & alive
                draw = torch.multinomial(torch.softmax(split_logits, dim=-1), 1).squeeze(-1)
                alive = cont
                tok = draw.unsqueeze(1)
                if not bool(alive.any()):
                    break

    with np.errstate(invalid="ignore", divide="ignore"):
        tf = np.where(tf_n > 0, tf_sum / np.maximum(tf_n, 1), np.nan)
        op = np.where(op_n > 0, op_sum / np.maximum(op_n, 1), np.nan)
    rows = [
        {"depth": int(t), "p_cont_teacher_forced": float(tf[t]), "n_teacher_forced": int(tf_n[t]),
         "p_cont_on_policy": float(op[t]), "n_on_policy": int(op_n[t]),
         "gap": float(op[t] - tf[t])}
        for t in range(D + 1) if tf_n[t] > 0 or op_n[t] > 0
    ]
    gaps = np.array([r["gap"] for r in rows], dtype=float)
    finite = np.isfinite(gaps)
    out = {
        "by_depth": rows, "n_jets": int(n_jets), "K": int(K),
        "mean_gap": float(np.nanmean(gaps)) if finite.any() else float("nan"),
        # A slope in depth is the exposure-bias signature; a flat offset is not.
        "gap_slope": (float(np.polyfit(np.arange(len(gaps))[finite], gaps[finite], 1)[0])
                      if int(finite.sum()) >= 2 else float("nan")),
    }
    if verbose:
        print(f"\ncontinue probability by depth ({n_jets} jets, {K} draws/jet):")
        print(f"    {'depth':>5} {'teacher-forced':>15} {'on-policy':>10} {'gap':>8}"
              f" {'n_tf':>7} {'n_op':>7}")
        for r in rows:
            print(f"    {r['depth']:>5} {r['p_cont_teacher_forced']:>15.4f}"
                  f" {r['p_cont_on_policy']:>10.4f} {r['gap']:>+8.4f}"
                  f" {r['n_teacher_forced']:>7} {r['n_on_policy']:>7}")
        print(f"    mean gap {out['mean_gap']:+.4f}, slope in depth {out['gap_slope']:+.4f}"
              f"   (slope != 0 => exposure bias; flat offset => head calibration)")
    return out


def run_exposure(model, val_ds, device, n_jets=2000, K=200, depth_jets=300,
                 draws_by_jet=None, verbose=True) -> dict:
    """The whole WP-B diagnostic block, as one dict for `eval_metrics.json`."""
    out = {"length_marginal": length_marginal(model, val_ds, device, n_jets=n_jets, K=K,
                                              draws_by_jet=draws_by_jet, verbose=verbose)}
    null = sbc_n_selfconsistency_null(model, val_ds, device, n_jets=n_jets, verbose=verbose)
    if null is not None:
        out["sbc_n_null"] = null
    depth = continue_prob_by_depth(model, val_ds, device, n_jets=depth_jets, verbose=verbose)
    if depth is not None:
        out["continue_by_depth"] = depth
    elif verbose:
        print("\ncontinue probability by depth: this family has an explicit q(N|x) head "
              "and takes no per-step continue decision — no such probability exists.")
    return out
