# PLAN — the RQ-spline `ln z` head

Status: **proposed — pre-authorized escalation, fired in v1, not yet implemented.**
Recorded by `PLAN_NCeilingProbe.md` WP-B so the trigger is not re-derived. One of the two
remaining model-side levers (`docs/SUMMARY_Model_Status.md` §3); the other is the N channel,
which `PLAN_NCeilingProbe.md` WP-A settled.

---

## 1. The trigger, already fired

`PROD_TEST_v1_RESULTS.md` §4.2 and §6 record it:

| arm | `ln z` PIT KS | × critical (0.0255) |
|---|---:|---:|
| v0 checkpoint / `v1_legacy_lnz_s0` | 0.0734 | 2.88× |
| `v1_base_s0` | 0.0529 | 2.07× |
| `v1_base_s1` | 0.0270 | 1.05× |
| `v1_base_s2` | 0.0471 | 1.84× |

**G3 fails on every seed**, and the region × coordinate cross localises the residual to
`ln_z × wide_soft` at **2.16×** its own critical value on **2 671 emissions** — the quadrant
holding **94%** of them — with every other scored cell below 1.02×. So this is a mismatch in
the bulk, not in a corner.

`lnz_support="physical"` (v1 WP-A) already closed the *support* half completely: 0.83%
below soft drop and 3.94% above `z = ½` both become 0.0000%. What is left is a **shape**
mismatch *inside* the interval, and — the sentence the escalation rests on — **a truncation
cannot fix a shape mismatch inside the interval.** The truncated normal has two free
numbers per node; the residual is the third and fourth moments of a distribution that is
not Gaussian on `(ln z_cut, ln ½]`.

The escalation was pre-authorized in `PLAN_prod_test_v1.md` §4.4 for exactly this outcome,
recorded as fired in `PROD_TEST_v1_RESULTS.md` §6, and deliberately **not** acted on
mid-run. This document is where it gets built.

---

## 2. The change

A **monotone rational-quadratic spline** (Durkan, Bekasov, Murray & Papamakarios,
*Neural Spline Flows*, arXiv:1906.04032) on the same soft-drop interval the truncated
normal already uses.

- **Config:** `model.lnz_head: "truncnorm" | "spline"`, default `"truncnorm"`. Same
  `getattr`-tolerant read as `lnz_support` / `cell_label_smoothing`, so a checkpoint config
  snapshot predating the field rebuilds as the truncated-normal model rather than crashing.
  `model.lnz_spline_bins: int = 8` alongside it, read only in `"spline"` mode.
- **Off path bit-identical.** The spline lives beside `_coord_logprob`'s
  `trunc_normal_logpdf` branch, `_sample_lnz`'s `trunc_normal_sample` branch and
  `coord_pit`'s `trunc_normal_cdf` branch — three call sites, one flag, and the
  `"truncnorm"` route must be untouched byte-for-byte (the `lnz_support` precedent, and
  what `scripts/verify_parity.py` checks).
- **The interval is unchanged.** `lnz_bounds(cx)` stays exactly as it is — cell-conditional,
  evaluated at the loosest `u` in the cell — so the factorization is not disturbed and the
  support guarantee v1 bought is preserved. The spline maps `(lo, hi] → (0, 1]` monotonically;
  the head emits `3K − 1` numbers per node (widths, heights, derivatives) instead of
  `(mean, sigma)`.
- **Why the spline first and not the joint density:** it is strictly cheaper, it changes
  one factor rather than the factorization, and it leaves every consumer of
  `has_continuous_coords` alone. If it closes G3 the structural escalation is not needed.

## 3. The measurement, pre-registered

- 3-seed training at the **v1 budget**, `v1_base` config with `model.lnz_head=spline`
  as the only difference — same data, same epochs, same encoder, seeds 0/1/2.
- **G3 re-test** on the recorded reference: the per-seed KS against its own 0.0255 critical
  value, and the `ln_z × wide_soft` cell against its 2.16×. The gate closes iff KS < crit on
  **every** seed *and* the bulk cell falls below 1.0×. A two-of-three pass is a partial
  result and is reported as one.
- **Guards that must not move:** the support audit stays at 0.0000% violations (a spline
  that leaks outside `(ln z_cut, ln ½]` is a regression, not a trade); held-out NLL must not
  worsen beyond seed spread; TARP and `pit_ks_max` reported beside G3 so a coordinate fix
  that narrows the joint is visible.
- Report into `PROD_TEST_v1_RESULTS.md` §4.2's table as a fourth block of rows, so the
  before/after is read on one page.

## 4. If it does not close G3

The **per-node joint coordinate density** (cINN-coords / CFM-coords, `PLAN_UPDATES.md` WP1)
is the follow-up, and it is structurally motivated rather than a second guess: under the
`LundGenerator` conventions

    ln z = u + v − ln p_T,sum

holds **exactly**, so independence-given-cell is violated by a *kinematic identity*, and no
per-coordinate head — spline or otherwise — can express that coupling. The spline is
therefore the test of "is the marginal shape wrong?", and its failure is the evidence that
the *factorization* is what is wrong. That trigger has also already fired
(`PROD_TEST_v1_RESULTS.md` §6.1); this ordering just spends the cheap one first.

## 5. Verification

- `pytest tests/` green; ruff clean on touched files; `scripts/verify_parity.py` unchanged.
- A new test asserting the `"truncnorm"` path is bit-identical with the flag absent and with
  it set explicitly (the `lnz_support` legacy-parity test is the template).
- A spline-mode round-trip test: `cdf(sample(...))` uniform on a synthetic head, and the
  log-density integrating to 1 over `(lo, hi]` by quadrature.
