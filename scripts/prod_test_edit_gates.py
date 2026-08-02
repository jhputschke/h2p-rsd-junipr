"""Evaluate the pre-registered gates E1-E9 of docs/PLAN_prod_test_edit.md.

    python scripts/prod_test_edit_gates.py [--run-root runs/prod_test_edit]
                                           [--reference-root runs/prod_test_v1]
                                           [--out docs/PROD_TEST_edit_TABLES.md]

The arithmetic of the plan's §8, and nothing else. The criteria were fixed before the
grid ran; this file only applies them. The per-gate machinery is **imported verbatim**
from `scripts/prod_test_v1_gates.py` — `gate_g1/g2/g4/g5/g6/g7` are the same functions
reading the same keys, so an E-gate and its G-ancestor cannot drift apart, and a fix to
one is a fix to both. What is new here is the cross-FAMILY A/B (`v1_contstop` vs `e_v1`),
the parameter-count column, and E9.

Three rules it enforces that a reader would otherwise have to remember:

* **NLL is not comparable across the `ln z` head change.** Any table that would put a
  `physical` NLL beside a `legacy` one prints the value with a `!` and a footnote instead
  of ranking them. E6 is quotable only when both sides are `physical` (plan §3).
* **A gate whose input is missing is `n/a`, never `pass`.** An arm not evaluated with the
  switch a gate needs cannot satisfy it by silence.
* **A gate whose INSTRUMENT the family does not implement is `n/a` with a named reason.**
  That is E9: the edit family has no `coordinate_cdfs`, so G3, `pit_ks_max` and the
  region x coordinate cross are unavailable, and the one v1 gate still open — the `ln z`
  shape *inside* its support — cannot be read here at all. This is a **known
  incompleteness of the comparison**, declared up front rather than discovered in the
  results, and it must never render as a pass.

The reference arms live under a DIFFERENT run root and are not retrained: WP-F.1
re-evaluates `runs/prod_test_v1/v1_contstop_s0` and `_s1` on this run's code path,
`EDGE_TOL` convention and device, so every number in the head-to-head comes from one
evaluation pass. The published two-seed bands are printed beside the recomputed ones —
if they disagree, the re-evaluation is the number and the disagreement is the finding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from prod_test_v1_gates import (  # noqa: E402  — the point is that these are NOT re-written
    _arm_family,
    _fmt,
    _get,
    _verdict,
    gate_g1,
    gate_g2,
    gate_g4,
    gate_g5,
    gate_g6,
    gate_g7,
    load_arms,
)

# The v1 winner, as docs/PROD_TEST_v1_RESULTS.md §4.8/§4.9 published it. Printed BESIDE
# the re-evaluated numbers, never in place of them (see the module docstring).
PUBLISHED = {
    "best val NLL/jet": (3.7927, 3.7799, 3.8054),
    "TARP max dev": (0.0212, 0.0200, 0.0225),
    "`coverage_68`": (0.5307, 0.5304, 0.5310),
    "medoid/identity": (0.9307, 0.9286, 0.9327),
}
TARP_NULL_PUBLISHED = 0.0275
Q0_AUC_PUBLISHED = 0.827          # v1_contstop_s0's deep pass, 97 018 jets

# Gate E7's pre-registered criterion, from plan §8. Fixed before the grid ran.
E7_LAMBDA_RANGE = (0.2, 5.0)
E7_R2_MIN = 0.9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _best_val_nll(entry) -> float | None:
    f = entry["dir"] / "metrics.csv"
    if not f.is_file():
        return None
    rows = [r for r in f.read_text().splitlines()[1:] if r]
    return min(float(r.split(",")[3]) for r in rows) if rows else None


_PARAM_CACHE: dict[Path, int | None] = {}


def _param_count(entry) -> int | None:
    """Trainable parameters of the arm, rebuilt from its own config snapshot.

    Plan §13's first risk: `ctx_dim = 64` (edit) and `dec_dim = 64` (AR) are NOT the same
    budget, and edit's free-cell head is a `Linear(ctx, 900)` evaluated across the whole
    lattice. A family claim made across a >2x budget gap is stated as confounded, so the
    number has to be in the table rather than in a reader's head.

    `model.parameters()`, not the state dict: the state dict also carries the `cell_cx` /
    `cell_cy` geometry BUFFERS, which are `2 * n_cells` numbers that no optimizer ever
    touches — 1 800 of them at `n_bins = 30`, i.e. the difference between two arms could
    be entirely buffer. Rebuilt from the snapshot config with no weights loaded, so it
    costs a module construction and nothing else."""
    ck = entry["dir"] / "best.ckpt"
    if ck in _PARAM_CACHE:
        return _PARAM_CACHE[ck]
    n = None
    if ck.is_file():
        try:
            from omegaconf import OmegaConf

            from h2p_rsd_junipr.geometry import Geometry
            from h2p_rsd_junipr.models.base import build_model
            from h2p_rsd_junipr.train.checkpoint import load_for_inference

            info = load_for_inference(str(ck), map_location="cpu")
            cfg = OmegaConf.create(info["config"])
            model = build_model(cfg, Geometry.from_config(cfg.geometry))
            n = int(sum(p.numel() for p in model.parameters()))
        except Exception as exc:                               # pragma: no cover
            print(f"[gates] parameter count unavailable for {entry['dir']}: {exc}")
    _PARAM_CACHE[ck] = n
    return n


def _lnz_support(entry) -> str:
    cfgf = entry["dir"] / "config.yaml"
    if not cfgf.is_file():
        return "?"
    from omegaconf import OmegaConf

    return str(OmegaConf.select(OmegaConf.load(cfgf), "model.lnz_support") or "legacy")


def _family_vals(arms, fam, getter):
    out = []
    for arm, e in arms.items():
        if _arm_family(arm) != fam:
            continue
        v = getter(arm, e)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return sorted(out)


def _band(vals):
    if not vals:
        return None
    return (sum(vals) / len(vals), min(vals), max(vals), len(vals))


def _fmt_band(b, spec=".4f"):
    if b is None:
        return "n/a"
    m, lo, hi, n = b
    return f"{format(m, spec)} [{format(lo, spec)}, {format(hi, spec)}] (n={n})"


def _clears(a, b):
    """`True` when band `a` and band `b` do not overlap — the plan's A/B clause.

    E4 and E5 both say it in the same words: *the band must CLEAR the reference's to
    claim an improvement; overlapping bands are a tie and are reported as one.* A
    difference of means is not a finding while the bands overlap, and a two-seed
    reference band is narrow because it is 2 draws, not because it is stable (plan §13)."""
    if a is None or b is None:
        return None
    return bool(a[2] < b[1] or b[2] < a[1])


# Which way is better, per quantity. Separation is only half of the A/B clause — the
# plan's words are "must CLEAR the reference's band **to claim an improvement**", and a
# separated band on the wrong side is a clear DEGRADATION. Reporting "clears" without the
# direction is the one way this table could read as a win while saying the opposite.
_BETTER = {
    "best val NLL/jet": "lower",
    "TARP max dev": "lower",
    "`coverage_68`": "0.68",
    "medoid/identity": "lower",
    "geo-median/identity": "lower",
    "`<N>` ratio": "1.0",
}


def _improves(label, ref_band, arm_band):
    """`True` when `arm_band`'s mean is better than `ref_band`'s on `label`'s own scale."""
    d = _BETTER.get(label)
    if d is None or ref_band is None or arm_band is None:
        return None
    if d == "lower":
        return bool(arm_band[0] < ref_band[0])
    target = 0.68 if d == "0.68" else 1.0
    return bool(abs(arm_band[0] - target) < abs(ref_band[0] - target))


def _ab_verdict(label, ref_band, arm_band, name="`e_v1`"):
    """The A/B cell: separation AND direction, never separation alone."""
    clears = _clears(ref_band, arm_band)
    if clears is None:
        return "**no band** on one side"
    if not clears:
        return "no — bands overlap, a tie"
    better = _improves(label, ref_band, arm_band)
    if better is None:
        return "**yes** — bands separate"
    return (f"**yes — {name} is better**" if better
            else f"**yes, but the WRONG WAY — {name} is worse**")


QUANTITIES = [
    ("best val NLL/jet", lambda a, e: _best_val_nll(e), "lower"),
    ("TARP max dev", lambda a, e: _get(e["metrics"], "calibration.tarp.tarp_max_dev"), "lower"),
    ("`coverage_68`", lambda a, e: _get(e["metrics"], "calibration.coverage_68"), "0.68"),
    ("medoid/identity",
     lambda a, e: ((_get(e["metrics"], "closure.dlund_posterior_medoid") or float("nan"))
                   / (_get(e["metrics"], "closure.dlund_identity") or float("nan"))), "lower"),
    ("geo-median/identity",
     lambda a, e: ((_get(e["metrics"], "closure.dlund_posterior_geomedian_cont") or float("nan"))
                   / (_get(e["metrics"], "closure.dlund_identity_cont") or float("nan"))), "lower"),
    ("`<N>` ratio", lambda a, e: _get(e["metrics"], "closure.mean_mult_ratio"), "[0.95, 1.05]"),
    ("parameters", lambda a, e: _param_count(e), "reported"),
]


def _deep_pass(entry) -> dict:
    """The notebook's `prod_test_v1_metrics.json` for this arm, or `{}`.

    The deep pass (97 018 jets) and the eval suite (2 000 / 300) are different tiers and
    write different files; the two E3 clauses below live only in the deep one, so they are
    read from there or reported `n/a` — never silently dropped."""
    if entry is None:
        return {}
    for name in ("prod_test_v1/prod_test_v1_metrics.json",
                 "prod_test_edit/prod_test_edit_metrics.json"):
        f = entry["dir"] / name
        if f.is_file():
            try:
                return json.loads(f.read_text())
            except Exception:                                  # pragma: no cover
                return {}
    return {}


def _q0_auc(entry):
    """`(auc, source)` for `q(N = 0 | x)`, from whichever deep-pass artifact exists."""
    d = _deep_pass(entry)
    v = _get(d, "empty_tree.auc_q0")
    if v is not None:
        return float(v), "prod_test_v*_metrics.json (notebook §6)"
    if entry is not None:
        f = entry["dir"] / "lund_closure_report" / "dist_closure_metrics.json"
        if f.is_file():
            try:
                v = _get(json.loads(f.read_text()), "EMPTY_TREE.q0.auc")
            except Exception:                                  # pragma: no cover
                v = None
            if v is not None:
                return float(v), "lund_closure_report/dist_closure_metrics.json"
    return None, None


# ---------------------------------------------------------------------------
# the gates that are new in this run
# ---------------------------------------------------------------------------
def gate_e3(m, entry=None) -> dict:
    """Multiplicity. G4's clauses, plus the two this family adds.

    `<N>` is read on the FULL population, never the `N >= 1` selection (v1 §1.1): that
    selection compares truth-selected jets to a posterior mean, which is regression to the
    mean and negative by construction. SBC-on-N is a percentile of its OWN MC null, never
    a chi2(9) (v1 §1.2). The two new clauses are the exact `q(N=0|x)` AUC — this family
    reads it off a structural DP rather than a fitted head — and the agreement of that
    exact `length_pmf` with the sampled multiplicity histogram."""
    base = gate_g4(m)
    parts = [base["detail"]]
    oks = [base["ok"]] if base["ok"] is not None else []

    auc, src = _q0_auc(entry)
    if auc is None:
        parts.append(f"`q(0|x)` AUC: **n/a** — a DEEP-PASS number, written by "
                     f"`notebooks/prod_test_v1.ipynb` §6 (`empty_tree.auc_q0`) or by "
                     f"`scripts/lund_closure_report.py` (`EMPTY_TREE.q0.auc`), never by "
                     f"the eval suite. Reference {Q0_AUC_PUBLISHED:.3f}")
    else:
        parts.append(f"`q(0|x)` AUC {auc:.3f} vs {Q0_AUC_PUBLISHED:.3f} (reference), read "
                     f"from `{src}` — and here it ranks an EXACT `q(N = 0|x)` off the "
                     f"structural DP rather than a fitted head")

    cost = (_deep_pass(entry) or {}).get("length_pmf_cost")
    if cost is None:
        parts.append("exact `length_pmf` vs the sampled multiplicity histogram: **n/a** in "
                     "this artifact — pinned at unit scale by "
                     "`tests/test_edit_model.py::test_sampled_multiplicities_track_the_"
                     "exact_length_pmf`, and priced at production scale by the deep "
                     "notebook pass")
    else:
        parts.append(f"exact `length_pmf` costs {cost.get('ms_per_jet', float('nan')):.2f} "
                     f"ms/jet on {cost.get('device')} over {cost.get('n_jets')} jets, "
                     f"against the sampling reference's "
                     f"{cost.get('sampler_reference_ms_per_jet', float('nan')):g} "
                     f"(results §4.9)")
    return {"ok": (all(oks) if oks else None), "detail": "; ".join(parts)}


def gate_e4(arms, ref_arms) -> dict:
    """TARP — the deciding gate.

    Two clauses, and they are separate. (a) `e_v1`'s max dev must sit inside the null
    recomputed at THIS run's own `(n, refs, alpha grid)`, with the band's floor < 0.05, on
    every seed — that is `gate_g7`, unchanged. (b) the three-seed band must CLEAR
    `v1_contstop`'s to claim an improvement; overlapping bands are a tie."""
    seeds = {a: e["metrics"] for a, e in arms.items() if _arm_family(a) == "e_v1"}
    if not seeds:
        return {"ok": None, "detail": "no `e_v1` arm present"}
    per = {a: gate_g7(m) for a, m in seeds.items()}
    verdicts = [r["ok"] for r in per.values()]
    unanimous = len(set(str(v) for v in verdicts)) == 1
    own = (all(v is True for v in verdicts) if all(v is not None for v in verdicts)
           else None)

    getter = lambda a, e: _get(e["metrics"], "calibration.tarp.tarp_max_dev")  # noqa: E731
    a_band = _band(_family_vals(arms, "e_v1", getter))
    b_band = _band(_family_vals(ref_arms, "v1_contstop", getter))
    clears = _clears(a_band, b_band)
    detail = ["`e_v1` per seed: " + ", ".join(f"{a} {_fmt(_get(m, 'calibration.tarp.tarp_max_dev'))}"
                                               f" -> {_verdict(per[a]['ok'])}"
                                               for a, m in sorted(seeds.items())),
              "unanimous" if unanimous else
              "**SEED-DEPENDENT — the band straddles the criterion**",
              f"A/B: `e_v1` {_fmt_band(a_band)} vs `v1_contstop` {_fmt_band(b_band)} "
              f"(published [{PUBLISHED['TARP max dev'][1]:.4f}, "
              f"{PUBLISHED['TARP max dev'][2]:.4f}]) -> "
              + _ab_verdict("TARP max dev", b_band, a_band)]
    return {"ok": own, "detail": "; ".join(detail),
            "a_band": a_band, "b_band": b_band, "clears": clears}


def gate_e5(arms, ref_arms) -> dict:
    """Coverage. Every scoreable region Wilson-consistent with 0.68 to pass, and the A/B
    delta against the reference must clear the pooled seed spread."""
    seeds = {a: e["metrics"] for a, e in arms.items() if _arm_family(a) == "e_v1"}
    if not seeds:
        return {"ok": None, "detail": "no `e_v1` arm present"}
    parts, oks = [], []
    for a, m in sorted(seeds.items()):
        by_region = _get(m, "calibration.by_region", {}) or {}
        scoreable = {k: v for k, v in by_region.items() if v.get("scored")}
        bad = [k for k, v in scoreable.items() if not v.get("coverage_68_consistent")]
        if scoreable:
            oks.append(not bad)
            parts.append(f"{a}: {len(scoreable) - len(bad)}/{len(scoreable)} scoreable "
                         f"regions Wilson-consistent" + (f" (fails: {', '.join(bad)})"
                                                         if bad else ""))
    getter = lambda a, e: _get(e["metrics"], "calibration.coverage_68")   # noqa: E731
    a_band = _band(_family_vals(arms, "e_v1", getter))
    b_band = _band(_family_vals(ref_arms, "v1_contstop", getter))
    clears = _clears(a_band, b_band)
    parts.append(f"A/B: `e_v1` {_fmt_band(a_band)} vs `v1_contstop` {_fmt_band(b_band)} "
                 f"(published [{PUBLISHED['`coverage_68`'][1]:.4f}, "
                 f"{PUBLISHED['`coverage_68`'][2]:.4f}]) -> "
                 + _ab_verdict("`coverage_68`", b_band, a_band))
    return {"ok": (all(oks) if oks else None), "detail": "; ".join(parts),
            "clears": clears}


def gate_e6(arms, ref_arms, e2_ok) -> dict:
    """Held-out NLL, and the conditions under which it may be quoted at all.

    Quotable only if (a) E2 passes — both arms on the physical `ln z` head, so the number
    is not measuring the head — and (b) both `lnz_support` declarations match. Both
    families are `exact_likelihood = True` and both normalize over the same space, which
    is what makes the comparison a comparison rather than a coincidence; the `ln z`
    normalization is the one thing that would silently break it, and
    `tests/test_nll_comparability.py` is where that is asserted."""
    a_band = _band(_family_vals(arms, "e_v1", lambda a, e: _best_val_nll(e)))
    b_band = _band(_family_vals(ref_arms, "v1_contstop", lambda a, e: _best_val_nll(e)))
    sup_a = {_lnz_support(e) for a, e in arms.items() if _arm_family(a) == "e_v1"}
    sup_b = {_lnz_support(e) for a, e in ref_arms.items() if _arm_family(a) == "v1_contstop"}
    same_head = bool(sup_a) and sup_a == sup_b
    clears = _clears(a_band, b_band)
    detail = [f"`e_v1` {_fmt_band(a_band)} vs `v1_contstop` {_fmt_band(b_band)} "
              f"(published [{PUBLISHED['best val NLL/jet'][1]:.4f}, "
              f"{PUBLISHED['best val NLL/jet'][2]:.4f}])",
              f"`ln z` heads: `e_v1` {sorted(sup_a) or ['?']}, "
              f"`v1_contstop` {sorted(sup_b) or ['?']}"]
    if not same_head:
        detail.append("**!  NOT QUOTABLE — the two sides normalize `ln z` differently, so "
                      "the delta is a constant offset of the head, not a fit difference**")
        return {"ok": None, "detail": "; ".join(detail)}
    if e2_ok is not True:
        detail.append("**NOT QUOTABLE — E2 did not pass, so at least one side's sampler "
                      "leaves the support the density is normalized on**")
        return {"ok": None, "detail": "; ".join(detail)}
    detail.append(_ab_verdict("best val NLL/jet", b_band, a_band))
    ok = None
    if clears and a_band and b_band:
        # a separated band on the WRONG side is a failure of the gate, not a pass of it
        ok = bool(a_band[0] < b_band[0])
    return {"ok": ok, "detail": "; ".join(detail), "clears": clears}


def gate_e7(arms) -> dict:
    """Anchoring — the stage gate that decides whether any `edit_v2` number means anything.

    Read from `e_v1_freewidth` (`model.physics_width=false`), the arm that was NOT told
    the functional form: quoting `Lambda_eff` from the arm that was restates the
    parametrization. `scripts/edit_anchoring_diagnostic.py` writes the fit; this only
    applies §8's pre-registered criterion to it."""
    rows = []
    readout = None
    for arm, e in sorted(arms.items()):
        f = e["dir"] / "anchoring_diagnostic.json"
        if not f.is_file():
            continue
        d = json.loads(f.read_text())
        rows.append((arm, d))
        if d.get("is_readout_arm") and _arm_family(arm) == "e_v1_freewidth":
            readout = (arm, d)
    if not rows:
        return {"ok": None, "rows": [],
                "detail": "no `anchoring_diagnostic.json` — run "
                          "`python scripts/edit_anchoring_diagnostic.py` (WP-G)"}
    if readout is None:
        return {"ok": None, "rows": rows,
                "detail": "the diagnostic exists but not on `e_v1_freewidth` — E7 is READ "
                          "OFF the free-MLP arm by design, so a physics-width arm cannot "
                          "satisfy it (it would restate the parametrization)"}
    arm, d = readout
    fit = _get(d, "shape_function_fit.ln_kt", {}) or {}
    lam, r2 = fit.get("lambda_eff"), fit.get("r2")
    lo, hi = E7_LAMBDA_RANGE
    ok = None
    if lam is not None and r2 is not None and math.isfinite(lam) and math.isfinite(r2):
        ok = bool(lo <= lam <= hi and r2 >= E7_R2_MIN and fit.get("falls_with_kt"))
    frac = _get(d, "edit_summary.frac_anchored")
    detail = (f"read off `{arm}` (physics_width=false): Lambda_eff = {_fmt(lam, '.3g')} GeV "
              f"(criterion [{lo}, {hi}]), R2 = {_fmt(r2, '.3f')} (criterion >= {E7_R2_MIN}) "
              f"on {fit.get('n_bins', '?')} scoreable `ln k_t` bins, widths "
              f"{'fall' if fit.get('falls_with_kt') else '**do NOT fall**'} with k_t; "
              f"frac_anchored = {_fmt(frac, '.3f')} (6-epoch reference 0.20); "
              f"crossing pairs = {_get(d, 'alignment_monotonicity.n_crossing_pairs')}; "
              f"n_x = 0 rate {_fmt(_get(d, 'n_x_zero_rate'), '.1%')}")
    if ok is False:
        detail += ("  — **the anchoring premise does NOT hold on this selection. Every "
                   "`edit_v2` number is null context, not a family result** (plan §8, E7).")
    return {"ok": ok, "rows": rows, "detail": detail}


def gate_e8(arms, e7_ok) -> dict:
    """`edit_v1` vs `edit_v2`, at matched seeds, encoder, batch size and epochs.

    Decided on held-out NLL + TARP + coverage, with parameter counts printed. Coordinate
    PITs cannot decide — neither stage has them (E9). Conditional on E7: a flat width fit
    means there is nothing for a richer emission model to sharpen, so an `e_v2` win would
    be a win at something other than what the family claims."""
    out = {"ok": None, "rows": []}
    for label, getter, _d in QUANTITIES:
        a = _band(_family_vals(arms, "e_v1", getter))
        b = _band(_family_vals(arms, "e_v2", getter))
        if a is None or b is None:
            continue
        out["rows"].append((label, a, b, _clears(a, b)))
    if not out["rows"]:
        return {**out, "detail": "no `e_v2` arm present"}
    dec = {label: c for label, _a, _b, c in out["rows"]
           if label in ("best val NLL/jet", "TARP max dev", "`coverage_68`")}
    detail = "; ".join(f"{k}: {'clears' if v else 'tie (bands overlap)' if v is False else 'n/a'}"
                       for k, v in dec.items())
    if e7_ok is False:
        detail += ("  — **E7 FAILED, so every `e_v2` number above is reported as null "
                   "context, not as a family result**")
    return {**out, "detail": detail}


def gate_e9() -> dict:
    """Pre-registered `n/a` by construction — a declared blind spot, never a pass."""
    return {
        "ok": None,
        "detail": "**n/a by construction, pre-registered.** `EditTransducer."
                  "supports_coordinate_pit = False` and `coordinate_cdfs` returns None, so "
                  "G3, `pit_ks_max` and the region x coordinate cross are unavailable on "
                  "this family. The one v1 gate still open — the `ln z` shape *inside* its "
                  "support, 1.05-2.07x crit — **cannot be read here**, and the head-to-head "
                  "is INCOMPLETE on that axis. Any recommendation to field the edit family "
                  "must state that the comparison never saw that failure. Closing it is "
                  "plan §11's first deferred item, triggered by E4 or E6 favouring edit.",
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build(run_root: Path, ref_root: Path) -> str:
    if not run_root.is_dir():
        return (f"no such run root: {run_root} — has the §7 grid run? "
                f"(`bash scripts/run_prod_test_edit.sh`)\n")
    arms = load_arms(run_root)
    ref_arms = load_arms(ref_root) if ref_root.is_dir() else {}
    ref_arms = {a: e for a, e in ref_arms.items() if _arm_family(a) == "v1_contstop"}
    if not arms:
        return f"no eval_metrics.json under {run_root}\n"
    L: list[str] = []
    P = L.append

    P("<!-- generated by scripts/prod_test_edit_gates.py — do not edit by hand -->")
    P("")
    P("## Arms\n")
    P("| arm | model | encoder | `lnz_support` | `physics_width` | aux | parameters | "
      "best val NLL/jet | eval jets |")
    P("|---|---|---|---|---|---:|---:|---:|---:|")
    from omegaconf import OmegaConf

    for label, table in (("edit", arms), ("reference", ref_arms)):
        for arm, e in table.items():
            m = e["metrics"]
            cfgf = e["dir"] / "config.yaml"
            sup, n_aux, pw = "?", None, "-"
            if cfgf.is_file():
                cfg = OmegaConf.load(cfgf)
                sup = str(OmegaConf.select(cfg, "model.lnz_support") or "legacy")
                aux = OmegaConf.select(cfg, "encoder.aux_features")
                n_aux = len(aux) if aux is not None else None
                pwv = OmegaConf.select(cfg, "model.physics_width")
                pw = "-" if pwv is None else ("yes" if pwv else "no")
            nll = _best_val_nll(e)
            mark = "" if sup == "physical" else " !"
            npar = _param_count(e)
            name = f"`{arm}`" + ("" if label == "edit" else " *(reference)*")
            P(f"| {name} | {m.get('model')} | {m.get('encoder')} | `{sup}` | {pw} | "
              f"{'-' if n_aux is None else n_aux} | "
              f"{'n/a' if npar is None else f'{npar:,}'} | "
              f"{_fmt(nll, '.4f')}{mark} | {_get(m, 'data.n_eval_jets')} |")
    P("")
    P("`!` marks an NLL that is **not comparable** to the rows without it: a different "
      "`ln z` normalization shifts NLL/jet by a constant unrelated to fit quality. NLL "
      "*is* comparable between the edit and AR families — both are `exact_likelihood = "
      "True`, normalized densities over the same space — **as long as their `ln z` heads "
      "match** (gate E6; `tests/test_nll_comparability.py`).")
    P("")
    P("**Parameter counts are a confound, reported rather than assumed away.** "
      "`ctx_dim = 64` (edit) and `dec_dim = 64` (`ar_junipr_v4`) are not the same budget, "
      "and edit's free-cell head is a `Linear(ctx, n_cells)` evaluated across the whole "
      "lattice. If two arms differ by more than ~2x, the family claim between them is "
      "**stated as confounded** (plan §13, carrying v1 §3.1's caveat).")
    P("")

    # --- the device audit: a mixed grid is a silent ranking hazard ------------------
    devs = sorted({str(e["metrics"].get("device")) for e in list(arms.values())
                   + list(ref_arms.values())})
    unrecorded = "None" in devs
    named = [d for d in devs if d != "None"]
    P(f"**Device:** {', '.join(f'`{d}`' for d in named) or '—'}"
      + (". One device across the whole comparison, as WP-F.1 pins it."
         if (len(named) == 1 and not unrecorded) else
         ". **MIXED — cpu and cuda are a different RNG stream *and* different float "
         "kernels, so this grid is a silent ranking hazard. Re-run the eval with one "
         "`--device` before reading anything below.**" if len(named) > 1 else ".")
      + ("  Some artifacts record **no `device` key at all** — they predate the flag, "
         "and WP-F.1's re-evaluation is what backfills them; until then the comparison "
         "cannot claim one device." if unrecorded else ""))
    P("")

    base_arm = next((a for a in arms if _arm_family(a) == "e_v1"), None)
    if base_arm is None:
        P("no `e_v1` arm found; per-gate tables skipped.")
        return "\n".join(L) + "\n"
    base = arms[base_arm]["metrics"]
    base_entry = arms[base_arm]

    g2 = gate_g2(base)
    e7 = gate_e7(arms)
    results = {
        "E1 acceptance": gate_g1(base),
        "E2 support": g2,
        "E3 multiplicity": gate_e3(base, base_entry),
        "E4 **TARP (deciding)**": gate_e4(arms, ref_arms),
        "E5 coverage": gate_e5(arms, ref_arms),
        "E6 held-out NLL": gate_e6(arms, ref_arms, g2["ok"]),
        "E7 anchoring (stage gate)": e7,
        "E8 `edit_v1` vs `edit_v2`": gate_e8(arms, e7["ok"]),
        "E9 coordinate PIT": gate_e9(),
    }
    # Reported beside the gates, not as one: G5/G6 are v1 machinery the plan carries but
    # does not re-register as E-gates.
    carried = {"G5 `narrow_soft` (carried)": gate_g5(base),
               "G6 decode (carried)": gate_g6(base)}

    P(f"## Gates (on `{base_arm}`)\n")
    P("| gate | verdict | numbers |")
    P("|---|---|---|")
    for name, r in results.items():
        P(f"| {name} | {_verdict(r['ok'])} | {r['detail']} |")
    for name, r in carried.items():
        P(f"| {name} | {_verdict(r['ok'])} | {r['detail']} |")
    P("")
    P("E1 is a precondition, not a metric: **nothing below it means anything if it "
      "fails.** E9 is `n/a` by construction and is a declared blind spot — see its row.")
    P("")

    # --- the same gates on every seed ----------------------------------------------
    seeds = {a: e["metrics"] for a, e in arms.items() if _arm_family(a) == "e_v1"}
    if len(seeds) > 1:
        per = {"E1": gate_g1, "E2": gate_g2, "E3": gate_e3, "E4(own)": gate_g7,
               "G5": gate_g5, "G6": gate_g6}
        P("### The same gates on every `e_v1` seed\n")
        P("The plan requires **unanimity across `e_v1`'s three seeds**. Where the band "
          "straddles a criterion, \"passes\" and \"fails\" are both true of this "
          "architecture and neither is true of it alone — the per-seed column IS the "
          "finding. `E4(own)` is E4's first clause only (the arm against its own "
          "recomputed null); the A/B clause is a band comparison and has no per-seed form.")
        P("")
        P("| gate | " + " | ".join(f"`{a}`" for a in sorted(seeds)) + " | band |")
        P("|---|" + "---|" * (len(seeds) + 1))
        for gname, fn in per.items():
            verdicts = {a: fn(seeds[a])["ok"] for a in sorted(seeds)}
            vals = set(str(v) for v in verdicts.values())
            band = ("unanimous" if len(vals) == 1
                    else "**SEED-DEPENDENT — the band straddles the criterion**")
            P(f"| {gname} | " + " | ".join(_verdict(verdicts[a]) for a in sorted(seeds))
              + f" | {band} |")
        P("")
        P("| quantity | " + " | ".join(f"`{a}`" for a in sorted(seeds)) + " | criterion |")
        P("|---|" + "---|" * (len(seeds) + 1))
        for label, path, crit in (
            ("TARP max dev", "calibration.tarp.tarp_max_dev", "<= its recomputed null"),
            ("TARP null 95%", "calibration.tarp.null_band.p95", "floor < 0.05 to be quotable"),
            ("`coverage_68`", "calibration.coverage_68", "Wilson-consistent with 0.68"),
            ("`<N>` ratio", "closure.mean_mult_ratio", "[0.95, 1.05] on the FULL population"),
        ):
            P(f"| {label} | "
              + " | ".join(_fmt(_get(seeds[a], path), ".4f") for a in sorted(seeds))
              + f" | {crit} |")
        P("")

    # --- the cross-family A/B, band against band -----------------------------------
    P("## The head-to-head — `v1_contstop` (implicit continue/stop) vs `e_v1` (edit)\n")
    P("This is the run's question. Band against band on every deciding metric, with the "
      "published two-seed numbers printed beside the re-evaluated ones. **A delta that "
      "does not clear the spread is not a measurement of the factorization; it is a "
      "measurement of the seed.** `v1_contstop`'s band is narrow because it is 2 draws, "
      "not because it is stable (plan §13), so a three-seed edit band clearing it is not "
      "decisive on its own — E4 and E5 say so in their own words.")
    P("")
    P("| quantity | `v1_contstop` (re-evaluated) | published | `e_v1` | delta | "
      "clears the spread? |")
    P("|---|---|---|---|---:|---|")
    for label, getter, _d in QUANTITIES:
        a = _band(_family_vals(ref_arms, "v1_contstop", getter))
        b = _band(_family_vals(arms, "e_v1", getter))
        if a is None and b is None:
            continue
        pub = PUBLISHED.get(label)
        pub_s = "—" if pub is None else f"{pub[0]:.4f} [{pub[1]:.4f}, {pub[2]:.4f}]"
        spec = ",.0f" if label == "parameters" else ".4f"
        delta = (f"{b[0] - a[0]:+{spec}}" if (a and b) else "n/a")
        verdict = _ab_verdict(label, a, b)
        if label == "parameters":
            ratio = (max(a[0], b[0]) / min(a[0], b[0])) if (a and b and min(a[0], b[0])) else float("nan")
            verdict = (f"ratio {ratio:.2f}x — "
                       + ("**> 2x: the family claim is STATED AS CONFOUNDED**"
                          if ratio > 2.0 else "within ~2x"))
        P(f"| {label} | {_fmt_band(a, spec)} | {pub_s} | {_fmt_band(b, spec)} | {delta} | "
          f"{verdict} |")
    P("")
    P(f"TARP is quoted against a null **recomputed at this run's own (n, refs, alpha "
      f"grid)**, not an analytic floor; v1's recomputed 95% point was "
      f"{TARP_NULL_PUBLISHED:.4f}. `pit_ks_max` and the `ln z` PIT KS have **no row here "
      f"and cannot have one** — see E9.")
    P("")

    # --- E2's attribution arm -------------------------------------------------------
    legacy = next((arms[a] for a in arms if _arm_family(a) == "e_v1_legacy_lnz"), None)
    if legacy is not None:
        gl = gate_g2(legacy["metrics"])
        P("## E2 attribution — the `legacy` arm must still fail\n")
        P(f"- `{base_arm}` (physical): {g2['detail']}")
        P(f"- `e_v1_legacy_lnz` (legacy): {gl['detail']} -> "
          + ("reproduces the v0 support failure, as required" if gl["ok"] is False
             else "**PASSES — the arm does not reproduce the v0 failure, so it attributes "
                  "nothing**"))
        P("")
        P("The reference is v0 / `v1_legacy_lnz`: ~0.81% of sampled emissions below the "
          "soft-drop boundary and ~3.98% above `z = 1/2`. This arm exists for "
          "**attribution**, not as an experiment — support correctness needs no A/B to be "
          "adopted (plan §3).")
        P("")

    # --- E7's diagnostic, arm by arm -------------------------------------------------
    if e7["rows"]:
        P("## E7 — the anchoring diagnostic at production scale\n")
        P("| arm | `physics_width` | readout? | Lambda_eff (GeV) | sigma_0 | R2 | bins | "
          "falls with k_t | frac_anchored | delete | insert | crossings | `n_x = 0` |")
        P("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for arm, d in e7["rows"]:
            f = _get(d, "shape_function_fit.ln_kt", {}) or {}
            s = _get(d, "edit_summary", {}) or {}
            P(f"| `{arm}` | {'yes' if d.get('physics_width') else 'no'} | "
              f"{'**yes**' if d.get('is_readout_arm') else 'no'} | "
              f"{_fmt(f.get('lambda_eff'), '.3g')} | {_fmt(f.get('sigma_0'), '.3g')} | "
              f"{_fmt(f.get('r2'), '.3f')} | {f.get('n_bins', '?')} | "
              f"{'yes' if f.get('falls_with_kt') else '**no**'} | "
              f"{_fmt(s.get('frac_anchored'), '.3f')} | "
              f"{_fmt(s.get('delete_rate'), '.3f')} | "
              f"{_fmt(s.get('insert_rate'), '.3f')} | "
              f"{_get(d, 'alignment_monotonicity.n_crossing_pairs')} | "
              f"{_fmt(_get(d, 'n_x_zero_rate'), '.1%')} |")
        P("")
        P("**Lambda_eff is quoted from the `readout?` = yes row and nowhere else.** The "
          "physics-width arms were *told* `sigma = sigma_0 + Lambda_eff/k_t`; reading the "
          "scale back off them restates the parametrization instead of measuring it. "
          "`crossings` is the monotonicity audit — the RNN-T lattice cannot produce a "
          "crossing pair, so a nonzero value is a bug in the walk, not a finding. "
          "`n_x = 0` jets reduce exactly to the free head, so that rate bounds how much "
          "of the sample the anchoring mechanism can act on at all.")
        P("")

    # --- E8 ---------------------------------------------------------------------------
    e8 = results["E8 `edit_v1` vs `edit_v2`"]
    if e8["rows"]:
        P("## E8 — `edit_v1` (pair-HMM) vs `edit_v2` (transducer)\n")
        P("| quantity | `e_v1` | `e_v2` | delta | clears the spread? |")
        P("|---|---|---|---:|---|")
        for label, a, b, _clears_flag in e8["rows"]:
            spec = ",.0f" if label == "parameters" else ".4f"
            P(f"| {label} | {_fmt_band(a, spec)} | {_fmt_band(b, spec)} | "
              f"{b[0] - a[0]:+{spec}} | " + _ab_verdict(label, a, b, "`e_v2`") + " |")
        P("")
        P("Matched seeds, encoder, batch size and epochs, with parameter counts above. "
          "**Coordinate PITs cannot decide this — neither stage has them** (E9). "
          + ("**E7 FAILED: every row here is null context, not a family result.**"
             if e7["ok"] is False else
             "Conditional on E7, which passed." if e7["ok"] is True else
             "E7 has no verdict yet, so these rows are not yet quotable as a family result."))
        P("")

    # --- the one-seed probes ------------------------------------------------------------
    probes = [f for f in ("e_v1_gru", "e_v1_freewidth")
              if any(_arm_family(a) == f for a in arms)]
    if probes:
        P("## One-seed probes — `e_v1_gru`, `e_v1_freewidth`\n")
        P("**One training each.** They carry no band of their own, so the only honest "
          "yardstick is the `e_v1` band at the same configuration, and a difference inside "
          "that band is not a difference. `e_v1_gru` licenses exactly one conclusion — "
          "*worth a proper multi-seed A/B* — which is v1 §3.2's discipline carried "
          "unchanged. `e_v1_freewidth` is not an ablation for its own sake: it is E7's "
          "readout arm.")
        P("")
        P("| quantity | `e_v1` (3 seeds) | " + " | ".join(f"`{f}` (1 seed)" for f in probes)
          + " | any outside the band? |")
        P("|---|---|" + "---|" * (len(probes) + 1))
        for label, getter, _d in QUANTITIES:
            b = _family_vals(arms, "e_v1", getter)
            if not b:
                continue
            spec = ",.0f" if label == "parameters" else ".4f"
            cells, outside = [], []
            for f in probes:
                v = _family_vals(arms, f, getter)
                cells.append(format(v[0], spec) if v else "n/a")
                if v and not (min(b) <= v[0] <= max(b)):
                    outside.append(f)
            P(f"| {label} | {_fmt_band(_band(b), spec)} | " + " | ".join(cells) + " | "
              + (", ".join(f"`{f}`" for f in outside) if outside else "no") + " |")
        P("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="runs/prod_test_edit")
    ap.add_argument("--reference-root", default="runs/prod_test_v1",
                    help="where the re-evaluated `v1_contstop` arms live (WP-F.1). Only "
                         "`v1_contstop*` is read from it.")
    ap.add_argument("--out", default=None, help="write the tables here as well as stdout")
    a = ap.parse_args(argv)

    def _abs(p):
        q = Path(p)
        return q if q.is_absolute() else REPO / q

    text = build(_abs(a.run_root), _abs(a.reference_root))
    print(text)
    if a.out:
        out = _abs(a.out)
        out.write_text(text)
        print(f"[gates] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
