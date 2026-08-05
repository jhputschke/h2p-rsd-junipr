# PLAN — the empty parton tree is a valid answer, and no point estimator can give it

Status: **both work items are implemented.** `decode.empty_threshold`
(default `0.0` = off), `inference.length.empty_gate` / `empty_threshold_for_rate`, the stage
in `models.base.map_or_mbr`, `run_closure`'s `p_empty_*` / `recall_empty` /
`precision_empty` keys, and §5a's gated row all landed; `tests/test_empty_tree_gate.py`
pins them. Measured on the walkthrough `ar_junipr_v3`, fitting τ on one half of the val
split and scoring on the other: `q(0\|x)` AUC **0.760**, under-confident **1.90×**,
τ = 0.166 (83.7th pct), held-out predicted rate **0.172** against truth 0.159, recall
**0.36**, precision **0.33** — close to this plan's projections (τ = 0.1675, 81.8th pct,
F1 ≈ 0.40). The second work item landed as `decode.length_temperature` +
`decode.length_tilt` (`recalibrate_pmf`, `fit_length_recalibration`,
`PosteriorModel.recalibrated_n_logits`), pinned by `tests/test_length_temperature.py` —
**but the scalar temperature this plan proposed provably cannot do the job, and the fix
needed one more parameter.** See the box in that section for the measurement.

At parton level **17.2%** of jets in
`cpp/test_data/jets.root` have zero primary splittings surviving grooming, so the empty
tree is the correct target for roughly one jet in six. Every point estimator in the tree
returns it for ~0% of jets, for three unrelated reasons, none of which is a coding error.
This plan proposes a **two-stage decode**: a per-jet emptiness decision taken on the
model's own `q(N=0|x)`, run *before* the shape decode, opt-in and default-off. A second,
separable work item recalibrates the length head, which is under-confident about emptiness
by ~2x. Builds on `docs/PLAN_NsplitMinCut.md`, `docs/PLAN_QuantileMinCut.md` and
`docs/PLAN_MBR_PerturbativeLund.md` (all merged); the diagnosis comes from
`notebooks/lund_distribution_closure_v2.ipynb` §5a.

## Context — why this change

`PLAN_NsplitMinCut.md` opens: *"the MAP point estimate returns 0 splittings for ~22% of
jets — unphysical, since a groomed jet that survived the algorithm has ≥1 primary
splitting."* That premise is correct about **x** and false about **y**.

The hadron-level sequence is what the algorithm is run on and what we condition on, and it
is legitimate to require `len(x) > 0` — it is a cut any analysis can make. The
**parton-level target** carries no such guarantee: a parton jet can perfectly well have no
splitting passing `z > z_cut (ΔR/R0)^β` with `kt ≥ kt_floor`
(`cpp/include/lund_io.hpp`, `passesGroom`), and hadronisation then manufactures the
structure seen at hadron level. On `cpp/test_data/jets.root` that is **8 631 of 50 290**
jets with `len(x) > 0`, i.e. **17.2%**.

So `n = 0` at the output covers two populations that the current mechanisms cannot separate:

| | what it is | correct response |
|---|---|---|
| **spurious** | the brevity/entropy collapse `PLAN_NsplitMinCut.md` diagnosed — high-multiplicity jets whose joint argmax scores below the empty tree | suppress it (what the floor does) |
| **genuine** | the parton jet really has no primary splitting | **emit it** |

`min_emissions` and `length_floor_quantile` act on the **output length**, so they suppress
both. The MBR estimator removes the brevity bias structurally rather than by clamping —
`PLAN_MBR_PerturbativeLund.md` lists *"the empty tree is never selected, with no floor"* as
an intended property — but the mechanism that achieves it (the mass-imbalance penalty)
suppresses the genuine population just as thoroughly. **Neither mechanism is wrong; both
were designed against a premise that holds for `x` and not for `y`.**

This went unseen because the whole evaluation surface was blind to it in two ways at once.
`notebooks/inference_demo.ipynb` and `notebooks/lund_distribution_closure.ipynb` select
`len(x[0]) and len(y[0])`, a **truth-level cut with no data analogue**, which deletes
exactly this population; and the aggregate multiplicity metrics cannot see it either — on
2 000 jets the MAP's mean multiplicity is **1.41 against truth's 1.42**, essentially
perfect, while getting **0%** of the empty jets right. It compensates by putting 1 where
the answer is 0. `eval/closure.py`'s `mult_bias_*` will never report this.

## Key facts (measured; reconfirm against the current tree)

All on `runs/calibration_v2_walkthrough/ar_junipr_v3/best.ckpt` + `cpp/test_data/jets.root`,
`len(x) > 0` population, `K = 120` draws, decode defaults unless stated.

**F1 — the target rate.** `P(n_y = 0) = 17.2%` over the deployable population (54 007 jets
in file; 3 717 with `len(x) = 0`; 8 631 with `len(x) > 0, len(y) = 0`). `LundDataModule`
applies **no** such filter — `MatchedLundDataset` builds `n_y = 0` items and `log_prob`
scores them (mean NLL 1.80), so the model is trained on them and only the evaluation looked
away.

**F2 — MAP: the argmax, not the floor.** `ar_junipr_v3` sets `use_multiplicity_head=True`,
so `map_estimate` routes through `_map_decode_fixed_length`
(`src/h2p_rsd_junipr/models/ar_junipr.py:404-412`): `n̂ = argmax_n q(n|x)`, then clamped by
`min_emissions`. Measured `P(n̂ = 0) = 0.0%`, **and re-decoding with `min_emissions = 0`
changes nothing** — the clamp never binds because the argmax never lands on 0. With
`q(0|x) ≈ 0.16` against `q(1|x) ≈ 0.30`, predicting 1 is the *correct* 0-1-loss decision.
The MAP is optimal for a loss nobody wants, not broken.

**F3 — MBR: on the ballot, out-priced.** On 120 truth-empty jets, candidates = first 16 of
120 draws:

```
posterior draws that are empty         14.4%
empty tree IS in the candidate set     83.3% of jets
mean risk of the EMPTY candidate       28.73
mean risk of the winner                16.77   (mean n̂ 1.15)
empty wins on                           0.0% of jets
```

`lund_emd(..., norm=False)` (`src/h2p_rsd_junipr/inference/mbr.py:237`) keeps the
imbalance term, so an empty cloud is *entirely* unmatched weight charged at
`mbr_R = 8.485`: risk ≈ `(1 − p₀) · W̄ · R` with `W̄ = Σ exp(v) ≈ 3.4`. The empty tree wins
only as `p₀ → 1`, not `p₀ > ½`.

**F4 — backend-dependent, uniquely.** Identical draws, 400 jets: `pot` and `energyflow`
both give `P(n̂=0) ≈ 0.2%` and 0% recall on truth-empty jets; `surrogate` gives **57.2%**
and **82.4%**, because a normalised binned-image χ² (`_lund_image`) does not carry an
imbalance term. No other observable in the closure suite moves with the backend. Any
statement about emptiness must name its backend.

**F5 — the belief is informative but under-confident 2x.** `q(0|x)` separates the two
classes at **AUC 0.77** (mean 0.16 on truth-empty jets vs 0.08 on the rest), but its
*scale* is wrong — `E[q(0|x)] = 9.2%` against a true rate of `18.2%`, and the reliability
curve is under-confident in every bin:

| `q(0|x)` bin | mean `q(0|x)` | actual rate |
|---|---|---|
| [0.00, 0.05) | 0.017 | 0.042 |
| [0.05, 0.10) | 0.074 | 0.156 |
| [0.10, 0.15) | 0.124 | 0.291 |
| [0.15, 0.20) | 0.173 | 0.331 |
| [0.20, 0.30) | 0.239 | 0.410 |
| [0.30, 1.00) | 0.348 | 0.451 |

Consequences: a **faithful sampler reproduces 9.2%, not 17.2%** (the notebook's posterior
series measures 9.7%, i.e. it is faithful), and **any absolute threshold is mis-set** —
`q(0|x)` never exceeds 0.41, so `τ = 0.5` never fires. The *ranking*, however, is sound,
which is what the proposed rule exploits.

## Why the existing knobs cannot fix it

- **`min_emissions` / `length_floor_quantile`** — clamp the output length. They cannot
  distinguish spurious from genuine `n = 0` because the output is all they see. F2 shows
  the clamp is not even the binding constraint for the multiplicity-head families.
- **`mbr_R` ↓ or `mbr_norm = True`** — would make the empty cloud cheaper, but the
  imbalance term is what makes MBR multiplicity-aware *everywhere else*, and normalising a
  zero-weight cloud is 0/0. Rejected: it fixes 17% of jets by degrading the estimator on
  the other 83%.
- **More training** — `PLAN_NsplitMinCut.md` already established that more training only
  shrinks the *spurious* collapse. F5 shows the belief here is informative already; the
  loss is at the decode, not the fit.

## Proposed fix — a two-stage decode

"Is there anything?" is a **discrete decision** that neither an argmax over `n` nor an
argmin of a continuous transport risk can express. Give it its own rule, taken on the
model's own belief, and run it *before* the shape decode:

```
q(0|x) ≥ τ  ->  return the empty tree
otherwise   ->  today's decode, unchanged (argmax / MBR / floors)
```

Default `τ = 0` disables the stage entirely, so every existing number stays bit-identical.

**Setting τ.** Two regimes, both supported:

- **Rate-matched (recommended for distribution-level work).** `τ` = the `(1 − r)`-quantile
  of `q(0|x)` over a held-out sample, for target rate `r`. Because it thresholds the
  *ranking*, it is immune to the F5 scale error. Measured, `τ` fitted on one half of 6 000
  jets and scored on the other:

  ```
  τ = 0.1675  (81.8th percentile)
  predicted empty rate  17.43%   true rate 17.10%
  precision 40.2%   recall 40.9%   F1 40.5%
  ```

  against 0% recall from every estimator today.
- **Cost-based.** `τ = c / (1 + c)` for a cost ratio `c` of fabricating structure vs
  missing it — the standard asymmetric-loss threshold. **Only meaningful after the
  recalibration below**; on the current head it fires far too rarely (F5).

This is deliberately the same shape as `length_floor_quantile`: a quantile of the model's
own `P(n|x)`, opt-in, default-off — a ceiling where that was a floor.

### Edits (ordered)

1. **`src/h2p_rsd_junipr/config.py`** — `DecodeConfig` (near `length_floor_quantile`,
   line ~198) gains:
   ```python
   empty_threshold: float = 0.0   # 0.0 == off. Decide the EMPTY tree when q(N=0|x) >= this,
   #                                before any shape decode. The parton target really is
   #                                empty for ~17% of jets (docs/PLAN_empty_parton_tree.md);
   #                                min_emissions cannot express that, and MBR's imbalance
   #                                penalty prices it out. Set via empty_threshold_for_rate.
   ```
   plus the matching `_DECODE_DEFAULTS` entry (line ~435). `decode_params` needs no change
   — it already tolerates snapshots predating a field.

2. **`src/h2p_rsd_junipr/inference/length.py`** — beside `quantile_floor` (line 21):
   ```python
   def empty_gate(pmf, tau: float) -> bool:
       """True when the model's own P(N=0|x) clears `tau`. tau <= 0 is always False,
       so the default decode is untouched."""

   def empty_threshold_for_rate(pmfs, rate: float) -> float:
       """The tau reproducing a target empty rate. Thresholds the RANKING of q(0|x),
       so a miscalibrated scale (see the plan's F5) does not move it. Fit on held-out
       jets and freeze — it is a quantile, hence sample-dependent."""
   ```

3. **`src/h2p_rsd_junipr/models/base.py`** — `map_or_mbr` (line 167), first thing:
   ```python
   tau = float(decode.get("empty_threshold", 0.0))
   if tau > 0.0:
       pmf = self.length_pmf(xf, nx, mults=[len(d) for d in draws] if draws else None)
       if empty_gate(pmf, tau):
           return self.describe_cells(xf, nx, [])
   ```
   `describe_cells` already handles `L = 0` (line 116; `mbr_select` uses exactly that for
   its degenerate branch), and reusing `mults` means no extra sampling. Placing it in
   `map_or_mbr` — not in each family — gives all four families the stage at once and keeps
   `map_estimate` a pure shape decode.

4. **`src/h2p_rsd_junipr/eval/closure.py`** — `run_closure` gains `p_empty_true`,
   `p_empty_pred` and the recall on truth-empty jets. The existing `mult_bias_*` keys are
   provably blind to this failure (mean multiplicity 1.41 vs 1.42 at 0% recall), so the
   metrics file should carry it explicitly.

5. **`notebooks/lund_distribution_closure_v2.ipynb`** — §5a gains the gated row, so the
   before/after is visible next to the estimators it fixes.

### Cost & scaling

Negligible. One `length_pmf` call per jet, which is `softmax(n_head(e))` for the
multiplicity-head families and reuses the caller's `mults` histogram otherwise — no extra
sampling and no OT solve. `empty_threshold_for_rate` is one pass of `np.quantile` over a
held-out set, done once and frozen.

### Tests

- `empty_threshold = 0.0` reproduces today's point estimate **exactly** for all four
  families (the parity test that matters; mirror `tests/test_multiplicity_head.py`'s
  strict-load idiom).
- `empty_gate(pmf, tau)` boundary: `tau <= 0` is always False; `pmf[0] == tau` fires.
- A gated `map_or_mbr` returns `multiplicity == 0` and a `LundPointEstimate` with
  `nodes == []` and a finite `logprob`.
- `empty_threshold_for_rate(pmfs, r)` reproduces rate `r` on the sample it was fitted on,
  to within one jet.
- Round-trip through `decode_params` on a checkpoint snapshot predating the field.

### Verification

1. Run `notebooks/lund_distribution_closure_v2.ipynb` §5a with the gate off, then at the
   rate-matched `τ`; `P(n̂=0)` should move from ~0% to the target rate with ~40% precision.
2. Confirm every other panel is unchanged with the gate off (bit-identical
   `dist_closure_metrics.json`).
3. Confirm the gate is backend-independent — unlike F4, it does not touch the MBR risk.
4. Re-run on `runs/aux_input_ab/aux_s0` and a cINN checkpoint: the stage is family-agnostic
   because it consumes only `length_pmf`.

### Risks

- **τ is a quantile, so it is sample-dependent.** Fit on held-out simulation and freeze;
  re-fit per pT window, since the empty rate varies with jet pT. A τ carried across a
  selection change silently mis-sets the rate.
- **Precision is ~40%, not ~100%.** The gate makes the population right and the per-jet
  decision much better than 0%, but it is not a solved classification problem. Report both.
- **It changes what a "point estimate" means** for the ~17%: downstream consumers that
  assume `multiplicity ≥ 1` (plotting code, `leading_emission_cell`, anything indexing
  `nodes[0]`) must tolerate the empty tree. `leading_emission_cell` already returns `None`;
  audit the rest.

## Second work item — recalibrate the length head (separable)

> **Implemented, and this section's premise was wrong.** A single scalar temperature
> cannot do this, for a structural reason: a temperature is *symmetric about the mode*,
> so it can only pull a non-modal class toward uniform or toward zero. `q(0|x) = 0.085`
> sits **above** uniform (`1/26 = 0.038`) and **below** the mode, so flattening pushes it
> *down*. Sweeping `T` over `[0.1, 20]` tops out at `mean q(0|x) = 0.125` against a truth
> of 0.161 — the NLL-fitted `T = 1.10` reaches only 0.091 and buys 0.0008 nat.
>
> The measured miscalibration is not a scaling but a **monotone ramp across n**
> (empirical/predicted = 1.90, 0.96, 0.93, 0.80, 0.68 at `n = 0..4`), and the minimal
> family that can produce a ramp adds a term **linear in n**. `decode.length_tilt` is that
> term; `recalibrate_pmf` / `fit_length_recalibration` apply `softmax(z/T + tilt·n)` and
> nest the scalar case at `tilt = 0` (`with_tilt=False` still fits it, as the baseline).
>
> Held-out, fitting on one half of the val split and scoring on the other:
>
> | | `T=1` | scalar | **+tilt** | truth |
> |---|---|---|---|---|
> | NLL of `N` | 1.2133 | 1.2125 | **1.1810** | |
> | mean `q(0\|x)` | 0.0846 | 0.0910 | **0.1428** | 0.1610 |
> | max `q(0\|x)` | 0.4261 | 0.4166 | **0.5089** | (≥ 0.5 needed) |
> | AUC | 0.760 | 0.760 | 0.758 | |
> | emp/pred at `n=0` | 1.90 | 1.77 | **1.13** | |
>
> Fitted `T = 1.21`, `tilt = −0.31`. Against the three claims below: the **cost-based
> threshold is now usable** (the belief reaches 0.5093); the **posterior series is mostly
> fixed** (0.085 → 0.143 against 0.161, not all the way); the **gate barely moves**
> (recall 0.360 → 0.368, precision 0.333 → 0.340), because a monotone-in-n transform
> largely preserves the cross-jet *ranking* of `q(0|x)` — which is also why the gate
> worked without recalibration in the first place.

F5's under-confidence is uniform and monotone, so a **single scalar temperature on the
`n_head` logits**, fitted post-hoc on held-out jets, should absorb most of it. No
retraining. This is what would:

- make the **cost-based** threshold usable (currently unusable — the belief never reaches
  0.5);
- fix the **posterior series** in the closure notebooks (9.7% → ~17%) without any gate,
  which matters because for distribution-level physics the right object is a draw, not a
  point estimate;
- improve the gate's precision above 40%, since a better-calibrated score usually ranks
  better too.

Worth doing first if only one of the two is done: it is smaller, it needs no config
surface, and it improves the sampler as well as the decode. Note the existing calibration
suite (`eval/calibration.py`, SBC/PIT on multiplicity) did **not** flag this — SBC ranks
are computed against the sampler's own draws, so a uniformly under-confident `q(N|x)` can
still pass. A dedicated `P(N=0)` reliability check is the missing diagnostic.

## Cross-reference — is the cluster mass vector a viable input to the reject rule?

Recorded here as the exit criterion of
[`PLAN_PosteriorClusters.md`](PLAN_PosteriorClusters.md) requires, now that its WP1–WP3
have landed (`inference/clusters.py`, `decode.cluster_posterior`).

**Structurally, yes — and for free.** `inference.mbr._empty_value` returns exactly `0` for
two empty clouds, so at any `K` the `N = 0` draws form a **zero-diameter clique** at a large
constant distance from every non-empty draw. Any density method finds it by construction:
the empty stratum comes back as its own cluster whose radius is `0` and whose mass **is**
`q(0|x)` — the very quantity the gate thresholds, and the one already measured as
well-calibrated (AUC ≈ 0.820) while every point estimator mishandled it. `eval/clusters.py`
reports `G3_empty_mass_vs_q0` for exactly this; a nonzero value there is a metric-convention
bug rather than a finding about the posterior, and on the arms measured so far it is `0.000`.

**So the substitution is testable, but it is not yet an improvement, for two reasons.**

1. **It changes nothing about the number.** The cluster mass of the `N = 0` stratum is the
   empirical `P(N = 0)` over the same draws `length_pmf` already histograms — the identity
   `G3` checks. Routing the gate through the cluster layer would therefore threshold the
   same value at `K²` EMD solves per jet instead of none. The cluster layer earns its cost
   on the `N ≥ 1` strata, where there is a partition to find; at `N = 0` it recovers a
   number that was already available.
2. **What the gate actually wants is a *calibrated* `q(0|x)`, and the cluster layer does
   not supply one.** Chow's rule (Chow, *IEEE Trans. Inf. Theory* **16** (1970) 41) is a
   statement about a posterior probability; `top_mass` is not calibrated until
   `PLAN_PosteriorClusters.md` WP5 says it is, and with `decode.cluster_split=false` it is
   biased **high**. The recalibration work item above (§ "Second work item") is the thing
   that moves this number, and it is independent of clustering.

**Where the cluster layer *does* help the empty-tree decision** is the part the gate cannot
express at all: `entropy` and `top_mass` distinguish "confidently empty" from "split between
empty and one soft emission", which is a different jet and currently gets the same answer.
That is a **reject-with-a-reason** rule rather than a better threshold, and it is the form
worth prototyping if this is revisited.

**Verdict: viable, currently redundant.** Keep `empty_threshold` reading `length_pmf`. If
the gate is reworked into the general `argmin` over an explicit loss on `n` (below), the
cluster mass vector is the natural place to get `q(m|x)` for the `m ≥ 1` terms, and the
substitution becomes worth its cost.

### ...but the composition runs the OTHER way, and there it is not redundant

The paragraphs above answer "can the clusters **replace** this gate?" (no — same number).
Measurement then showed the reverse question is the load-bearing one: **this gate fixes a
real defect in the cluster layer's point summary**, and that composition is now implemented
(`inference.mbr.mbr_cluster_set(empty_threshold=...)`, reached through
`decode.empty_threshold` — the same knob, meaning the same thing, with no new config field).

**The defect.** `PosteriorSetEstimate.members[0]` is the mass argmax over clusters. The
`N = 0` stratum is **atomic** — `_empty_value` returns exactly `0` for two empty clouds, so
every empty draw collapses into one zero-radius cluster carrying the whole of `q(0|x)` —
while the non-empty draws live on a continuum and are **fragmented** into several clusters
by the density method. The argmax therefore compares one lump against the largest of a split
field, and the empty explanation wins far more often than its own mass warrants.

Measured on 600 held-out jets of `data/jet_aux_asym_test.root` at `K = 200`
(`v1_contstop_s0`, `notebooks/per_jets_estimation_cluster.ipynb` §5b):

| series | `P(n=0)` | vs truth | mean mult | vs truth |
|---|---|---:|---|---:|
| truth | 0.167 | — | 1.442 | — |
| MBR medoid | 0.048 | −0.119 | 1.377 | −0.065 |
| `members[0]` (mass argmax) | **0.298** | **+0.131** (~9σ) | 1.105 | −0.337 |
| posterior draw | 0.178 | +0.011 | 1.453 | +0.011 |

That +0.131 is a partition-granularity artifact, not physics, and about two thirds of
`members[0]`'s multiplicity deficit is *this one decision* rather than a shape error.

**Why this gate is the right fix and not a patch.** Gate G3 of
[`PLAN_PosteriorClusters.md`](PLAN_PosteriorClusters.md) measures
`|mass(N=0 cluster) − q(0|x)|` and finds it ~0: the two are the **same number**. So the mass
argmax and this gate differ only in what that number is compared against — a fragmented
competitor set whose size depends on `cluster_min_mass` and the clustering method, or a τ
fitted by rate-matching and frozen. Same information, calibrated decision rule, no new
statistic and no new fit.

**What moves and what does not.** Only the *recommendation* (`PosteriorSetEstimate.point`,
via `point_index`) moves; `members`, `masses`, `radii` and the conformal prefix are
untouched, and `members[0]` keeps meaning "the top-mass exemplar" so the two rules stay
comparable on one object. Unlike `map_or_mbr` the gate does **not** short-circuit the decode
here: the set still carries the empty explanation, because a rejected alternative is still a
reported alternative. And it never fabricates — a `q(0|x) ≥ τ` with no empty draw in the
pool is a disagreement between the length head and the sampler, not a tree, so the
recommendation stays inside `H = {pool}`.

**Read the result with this gate's own accuracy in mind.** τ is fitted by *rate-matching*,
so it fixes the empty **rate** essentially by construction — agreement there is not a
result. Whether it fixes the right **jets** is the measurement, and by the numbers at the
top of this document the gate is a weak classifier (AUC 0.760, recall 0.36, precision 0.33).
Expect the multiplicity marginals to improve a lot and the per-splitting residual much less.
`notebooks/per_jets_estimation_cluster.ipynb` §6b reports both, and conditions the shape
comparison on jets where *every* compared series is non-empty so the emptiness decision is
removed rather than averaged over.

**A reporting wart the gate had, now fixed.** `LundPointEstimate.pretty()` used
`risk is not None` as the MAP-vs-MBR discriminator, and this gate returns *before*
`mbr_select` runs — so an MBR decode that answered the empty tree carried no risk and
**printed itself as a MAP**. The tree was right and the label was wrong, which is the
failure mode this repo tracks most carefully. Provenance is now explicit on the object, the
same pattern as `coords_source`:

```python
estimator: str = "map"    # map | mbr | empty_gate | cluster
```

set at each decision point, so a gated answer reads `EMPTY-GATED groomed shower: ... (q(N=0|x)
cleared tau BEFORE any shape decode — no estimator ran)`. The empty-tree footer was
similarly hard-coded to the MAP's reason ("MAP is immediate stop") for *every* estimator and
is now per-estimator — MBR reaching empty means the posterior is empty-dominated, which is
the estimator working rather than collapsing. `inference.clusters.assert_ancestral_draws`
also rejects a gated tree now: it carries neither `.risk` nor `.cluster_mass`, so a check
written against those two alone would wave it into a pushforward, and it is every bit as
much a *decision* as a medoid.

**This raises the priority of the length-head recalibration below.** The composition is
only as good as τ's ranking of `q(0|x)`, so anything that sharpens that head now improves
the cluster layer's point summary as well as the point estimator's.

## Open questions

- Should the gate be **`P(N=0)` specific, or a general `argmin` over an explicit loss on
  `n`**? The general form `n̂ = argmin_n Σ_m q(m|x) L(n, m)` subsumes it (`L` asymmetric at
  `n=0` recovers the threshold) and would also fix the mode bias at `n ≥ 1`. Larger change;
  the gate is the minimal fix for the observed failure.
- **Does the empty rate depend on pT?** It should — softer jets groom away more. If so, τ
  must be fitted per window, and `PLAN_ProductionAssessment.md` §7's pT windows are the
  natural place to measure it.
- **Is `x_nsec > 0` informative here?** A jet with off-spine activity but no primary parton
  splitting is a distinctive configuration; the aux columns may sharpen `q(0|x)` more
  cheaply than recalibration.
