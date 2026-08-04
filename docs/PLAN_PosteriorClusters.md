# PLAN — Posterior clusters: set-valued prediction, two per-jet confidence scalars, and a gated bounded-loss MBR

Status: **implemented** — WP1, WP2, WP3, WP4a, WP5 and WP6 have landed; **WP4b was never
opened**, which §8.1 already reclassified as optional and §16 allows the plan to exit
without. Six work packages (WP1–WP6), each independently mergeable, opt-in with defaults
off per the established `point_estimator="map"` / `use_multiplicity_head` idiom. Every OFF
path is bit-identical to today. Builds on the merged `PLAN_MBR_PerturbativeLund.md`,
`PLAN_prod_test_v1.md` (WP-C.1, WP-C.2), `PLAN_empty_parton_tree.md`, and
`PLAN_UPDATES.md` WP2 (the calibration suite).

> **Implementation note (2026-08-04).** What shipped, and the four places it departs from
> the text above:
>
> - **`inference/clusters.py`** — `PosteriorClusterSet`, `PosteriorSetEstimate`,
>   `cluster_posterior` (`hdbscan` / `dbscan` / `pam`), `assert_cluster_metric_ok` (the §4
>   guards, all raising), `fit_set_threshold` / `set_size_for`, `assign_truth` /
>   `support_radii`, `random_partition_null`, and `assert_ancestral_draws` (§10.6's
>   pushforward hygiene, as a callable guard — `PLAN_UPDATES.md` WP5's aggregate
>   cross-check does not exist yet, so there is no call site to assert *at*).
> - **`models/base.py: predict_set`** — the sibling of `map_or_mbr`, delegating to
>   `inference/mbr.py: mbr_cluster_set`. `mbr_select` is refactored onto a shared
>   `posterior_distances` so the point estimate and the set read the *same* `D`.
> - **WP4a** — `_reduce_risk`, `bandwidth_quantile`, and `mbr_select(diagnostic_losses=...)`
>   returning `(estimate, side_channel)`. `eval/stability.py` holds the §8.5 columns;
>   `tests/test_stability.py::test_loss_spread_not_in_systematics` enforces §8.6's boundary
>   by parsing `eval/systematics.py`.
> - **WP5/WP6** — `eval/clusters.py` runs G2, G2′, G3, G5, G6, G7, G8 and G8′ in one pass,
>   behind `experiment.cluster_diagnostics`; `eval/report.py: plot_clusters` draws the
>   reliability and coverage figures. Notebooks:
>   `notebooks/per_jets_estimation_cluster.ipynb` (the per-jet study with the quantities
>   re-assigned for the set) and `notebooks/inference_demo_cluster.ipynb` (§10.5's
>   single-jet MDS panel), both generated from `scripts/make_*_cluster_nb.py`.
>
> **Departures, each with its reason:**
>
> 1. **§10.1's diagnostics live in `eval/clusters.py`, not inside `run_closure`.** Every
>    number needs the `K×K` matrix, and `run_closure`'s `map_or_mbr` builds and discards one
>    per jet; a second pass inside it would double the suite's dominant cost (§14's own
>    budget note). They land at `metrics["clusters"]`, the shape `support_audit` /
>    `exposure` / `mode_audit` already use.
> 2. **`pam`'s `k` selection takes a silhouette FLOOR of 0.50, not 0.**  A non-positive
>    threshold is not a threshold: *k*-medoids cuts an isotropic Gaussian blob into three
>    pieces at silhouette 0.32, and reporting that as three posterior explanations is the
>    failure the control arm exists to catch. 0.50 is Kaufman & Rousseeuw's own boundary
>    ("a reasonable structure has been found"), and it is what makes the kill criterion
>    *reachable* by this method.
> 3. **`assign_truth` compares against a per-cluster support radius** (the 95th percentile
>    of member-to-exemplar distance) rather than a quantile over the `radii` vector. With
>    two or three clusters per jet, a quantile over three numbers is the maximum; the
>    per-cluster reach is the quantity "is the truth inside this cluster's support" is
>    actually asking about.
> 4. **`cluster_min_cluster_size`'s auto value is a fraction of the pool actually
>    clustered**, not of `K`. Under `cluster_split` pool A is half the draws, so deriving it
>    from `K` would double the effective threshold and change the partition's granularity —
>    making gate G9 a measurement of granularity *plus* selection bias rather than of
>    selection bias alone.
>
> The `[mbr]` extra gains `scikit-learn >= 1.3`; `cluster_method="pam"` needs none of it, so
> the CI fast tier and every guard test run on a host without it.

> **Line anchors.** File:line references were taken from the tree at commit
> `34e98b8` (2026-08-02). Re-verify before editing; merges shift them.

> **Revision note (v2).** WP order inverted relative to the first draft. The bounded
> loss was WP1; it is now WP4 and is *gated by* diagnostics that only the cluster
> layer can produce. Building the fix before measuring the disease was the error.
> §4 records the metric audit that motivated the guards.
>
> **Revision note (v6).** Added **§8.6**: the linear-vs-bounded spread is *not* a
> systematic and must not enter the uncertainty budget — they are different functionals
> of one posterior, not two approximations to one quantity, and the posterior width is
> already reported by `radii[0]` and the masses. Kept instead as a **stability** check,
> with `argmin_moved` singled out as a clustering-free multimodality flag that works on
> **real data** where G2′ cannot. Placement moved to **`eval/stability.py`**; the module
> boundary is the guard.
>
> **Revision note (v5).** WP4 split at the seam §8.3 identifies. **WP4a** (the ~15-line
> reduction, as an `eval/` diagnostic with no config surface and no `.risk` change) is
> now **unconditional and lands with WP1** — G2, G8 and G8′ become measurements rather
> than predictions, for less code than the argument for deferring them. **WP4b** (the
> supported estimator) stays gated, and §8.3 records why it is not 15 lines: `.risk`
> changes meaning across 14 consumers, and the per-jet ε that makes 4a simple is not
> aggregable across jets. Tier 3 retired into Tier 1.
>
> **Revision note (v4).** Two corrections to WP4, both narrowing it. **(a)** The
> clusters are *bit-identical* under all three losses — `cluster_posterior` and
> `_reduce_risk` both consume `D` and never see each other's output — so `mbr_loss`
> changes only which draw is selected, never the partition, masses, radii or scalars
> (§8.1). **(b)** The bounded loss reintroduces the empty-tree degeneracy that MBR
> removes structurally, because `_empty_value` puts all empty draws at mutual distance
> exactly 0 (§8.4). New blocking gate **G8′**. WP4's remaining justification over
> `predict_set().members[0]` is peak-density-vs-mass, stated in §8.1.
>
> **Revision note (v3).** Added **G2′** (§10.1b), the truth-based counterpart to G2:
> whether the *set* recovers what the point estimate misses, measured against a
> mass-matched random-partition null rather than against the MBR point estimate. G2
> alone asks only whether the medoid is centrally placed and can pass while the set is
> worthless, or fail while it is valuable. The two are now joint gates and may
> legitimately disagree.

---

## 1. The question this plan answers

`mbr_select` (mbr.py:541) returns the **Fréchet median restricted to the sample** —
the draw of least *mean* perturbative-Lund EMD to the posterior. That is a global
*centrality* criterion, the correct Bayes estimator under a loss linear in the
distance, and the right default.

It is not the right criterion when the posterior is **multimodal**. The medoid of a
two-lobed posterior can land in the sparse valley between the lobes, minimizing mean
distance while representing neither explanation. This is not hypothetical here: the
sample space is transdimensional,

$$
\mathcal{Y}=\bigsqcup_{N=0}^{N_{\max}}\mathcal{C}^{\,N},
$$

and the strata are metrically separated by the EMD's imbalance term. A jet whose
posterior is split between "one hard emission" and "two softer emissions consistent
with the same observed $x$" is the hadronization ambiguity expressed as **discrete
alternative explanations**, and a mean-distance criterion smears exactly that.

**The central structural fact this plan exploits:** the cluster layer consumes only
`D`, the pairwise distance matrix `mbr_select` already builds (`lund_emd_matrix`,
mbr.py:482), and `D` is identical under every loss. Changing the *reduction* over `D`
(the bounded loss) and reading *more* off `D` (clusters, masses, scalars) are
orthogonal. Therefore:

- the cluster products run at **stock MBR settings**, with the point estimate
  bit-identical because nothing touches `risk = D.mean(axis=1)` (mbr.py:583);
- the diagnostic that says whether a bounded loss is *needed* — how often the linear
  medoid lands in the dominant cluster — is itself a cluster-layer output.

So the cluster layer ships first and the bounded loss waits on its verdict.

Deliverables, in dependency order:

1. a **cluster layer** over the existing `D`, giving a set-valued prediction with
   per-cluster posterior mass (WP1–WP2);
2. **two per-jet scalars** — top-cluster mass and mass-vector entropy — calibrated
   against truth, so they can be quoted as a confidence and used as the reject
   statistic in the empty-tree decision layer (WP3, WP5);
3. a **bounded-loss reduction**, in two pieces: **WP4a**, a ~15-line diagnostic that
   lands with WP1 and turns G2/G8/G8′ into measurements; and **WP4b**, the supported
   estimator, gated on what 4a measures and narrowed by §8.1 — it moves only the single
   point estimate, never the clusters, and competes against `predict_set().members[0]`
   rather than against the medoid.

**All candidates remain genuine posterior draws.** The hypothesis space stays
$\mathcal{H}=\{\text{pool}\}$ throughout; nothing here constructs a tree the model
did not generate. That closure property is why WP-C.1 (winner keeps its own sampled
coordinates, mbr.py:565–571) was worth landing, and it is not traded away.

---

## 2. Prior findings this plan builds on

- **`mbr_select` already computes the object the cluster layer needs.** The risk is
  one reduction over `D` (mbr.py:583). **Caveat:** with `mbr_n_candidates > 0`
  (mbr.py:559–562) `D` is rectangular; clustering needs the square $K\times K$.
  WP1 guards this.
- **The winner is a genuine posterior sample** (WP-C.1, `coords_source="sample"`,
  point_estimate.py:70). Set-valued output inherits this for free.
- **The empty-cloud distance convention is already pinned.** `_empty_value`
  (mbr.py:136) returns $|W_a-W_b|\cdot\texttt{ground\_scale}$, and $0$ for two empty
  clouds. Consequence, exploited rather than fought: **all empty draws sit at mutual
  distance exactly 0**, forming a zero-diameter cluster any density method finds by
  construction, at large constant distance from every non-empty draw. The $N=0$
  stratum appears as its own cluster whose mass *is* $q(0\mid x)$ — the quantity v1
  measured as well-calibrated (AUC ≈ 0.820, $q(0\mid x)\approx$ true rate) while every
  point estimator mishandled it.
- **`empty_threshold`** (config.py:298–307) currently decides emptiness by quantile
  rate-matching *before* any shape decode (`map_or_mbr`, base.py:277–283). The cluster
  mass vector supplies the calibrated posterior probability Chow's rule actually wants
  (Chow, *IEEE Trans. Inf. Theory* **16** (1970) 41). WP5 makes that substitution
  testable; it does not make it yet.
- **The joint tree posterior is over-confident** (v1 TARP), so a cluster mass read off
  $q_\phi$ is *not* a calibrated probability until WP5 says it is. This is the binding
  constraint on the whole deliverable and is gated accordingly (G6).
- **Parity rules** (cross-cutting): every new switch defaults off with a bit-identical
  OFF path; config reads via the tolerant `decode_params` / `OmegaConf.select` backfill
  (config.py:619–630); `config_hash` moves for new runs only; old checkpoints load.
- **Fitted inference-layer scalars follow the `tau.fitted_under` pattern** (v0 §7):
  value plus fitting provenance in the artifact, frozen on test.

---

## 3. Scope and non-goals

**In scope:** a decode-layer change only. No retraining, no change to `log_prob`,
`per_jet_nll`, or any trained head. Every WP runs on existing checkpoints.

**Non-goals, explicitly:**

- **No enlargement of $\mathcal{H}$.** Consensus/lattice MBR and continuous
  optimization at fixed cell structure would break the in-support guarantee. Out.
- **No new metric.** The perturbative-Lund EMD stays as-is, so the clustering and the
  estimator see the same geometry by construction. §4 constrains which *existing*
  settings of that metric are admissible; it does not add one.
- **No mixture-model fitting.** Distance-matrix methods only; nothing requiring a
  vector-space embedding of $\mathcal{Y}$.
- **No change to the population-level headline.** The decode-free posterior series
  stays the population headline (v1 WP-C.3). Cluster products are per-jet.
- **No pushforward built from point estimates.** Stated because it is the tempting
  misuse: a selected draw is *not* a random draw, so `PLAN_UPDATES.md` WP5's aggregate
  cross-check must consume ancestral draws, never cluster exemplars (§9.6).

---

## 4. Metric audit — measured, not assumed

Clustering imposes requirements on `D` that the point estimator does not. Measured on
40 synthetic Lund clouds (0–5 points each) by loading `mbr.py` standalone; 64 000
triples tested per configuration.

| property | configuration | result |
|---|---|---|
| symmetry | `pot`, β = 1 | max \|D − Dᵀ\| = 1.4 × 10⁻¹⁴ |
| zero diagonal | `pot`, β = 1 | max \|diag\| = 4.9 × 10⁻¹⁶ |
| triangle inequality | `pot`, **β = 1** | **0 violations / 64 000** |
| triangle inequality | `pot`, β = 2 | **300 violations / 64 000** |
| triangle inequality | `surrogate` (`_chi2`) | 2 violations / 64 000 |
| triangle inequality | √(2·`_chi2`) | 0 violations / 64 000 |
| mass sensitivity | same shape, 10× total $k_t$ | `pot` 152.7 · **`surrogate` 0.0** |

Four consequences, each landing as a guard rather than a docs note:

1. **`mbr_beta == 1.0` is required when `cluster_posterior=true`.** β ≠ 1 breaks the
   triangle inequality (KMT's condition for the EMD to be a metric — Komiske, Metodiev
   & Thaler, *Phys. Rev. Lett.* **123** (2019) 041801, arXiv:1902.02346 — is β = 1 with
   $R$ at least half the maximum ground distance), and HDBSCAN's mutual-reachability
   construction assumes a metric. Raise, do not warn.
2. **`mbr_R` is comfortable at the default across all three coordinate modes.** With
   `ln_invdelta_range = ln_kt_range = [0, 6]` (config.py:33–34) the Lund-plane diagonal
   is $6\sqrt2 = 8.485$, which is exactly the `mbr_R` default (config.py:316). Adding
   `+lnz` or `+psi` raises the ground diameter to ≈ 9.9 and ≈ 11.7, still leaving
   $R \ge R_{\max}/2$ with margin. Assert the inequality rather than hard-coding 8.485,
   so a non-default `geometry` range cannot silently break it.
3. **`mbr_backend="surrogate"` is excluded from any quoted mass vector** — not
   principally for the 2/64 000 triangle violations (which √(2·`_chi2`) would fix) but
   because `_lund_image` normalizes (mbr.py:401), making the surrogate *exactly* blind
   to total $k_t$ and multiplicity. It therefore collapses the $N$-stratum separation
   that makes the clusters physical. It remains admissible as a **screening pass for
   G2 only**, where medoid-in-top-cluster is robust to the collapse.
   *(Note it still reproduces the empty-cluster structure — empty↔non-empty = 0.5
   constant, empty↔empty = 0 — so G3 is checkable on the cheap backend.)*
4. **Symmetrize defensively anyway.** `D = 0.5·(D + Dᵀ)` before handing to any
   clustering routine, and zero the diagonal. The measured asymmetry is at solver
   round-off, but `sklearn` raises on exact-symmetry checks, and the `energyflow`
   batched path (`_matrix_ef`, mbr.py:438) has not been audited the same way.

---

## 5. WP1 — cluster layer (`inference/clusters.py`)

New module, distance-matrix-only (no vector-space assumption on $\mathcal{Y}$):

```python
@dataclass
class PosteriorClusterSet:
    labels: np.ndarray          # (K,) int, -1 = noise
    exemplars: list[int]        # draw index per cluster, medoid within cluster
    masses: np.ndarray          # (n_clusters,) posterior mass, sums to <= 1
    radii: np.ndarray           # (n_clusters,) mean within-cluster EMD to exemplar
    top_mass: float             # masses.max()
    entropy: float              # -sum m log m, natural log
    method: str
    eps: float | None           # bandwidth, when the method takes one (dbscan)
    min_cluster_size: int       # hdbscan's control instead
    n_draws: int
    backend: str                # provenance: mass vectors from "surrogate" are invalid

def cluster_posterior(D, *, method="hdbscan", min_mass=0.05, eps=None,
                      weights=None, backend="pot") -> PosteriorClusterSet: ...
```

Methods, all consuming a precomputed **square, symmetric** distance matrix:

- **`hdbscan`** (default) — density-based, no fixed $k$, native noise label (Campello,
  Moulavi & Sander, PAKDD 2013), `metric="precomputed"`. **Takes `min_cluster_size`,
  not a bandwidth** — so no ε pre-registration is required on the default path (see
  §8).
- **`dbscan`** — the ε-explicit fallback (Ester et al., KDD 1996), sharing the
  bandwidth rule with WP4's bounded loss.
- **`pam`** — $k$-medoids with $k$ by silhouette (Kaufman & Rousseeuw, *Finding Groups
  in Data*, Wiley 1990); the deterministic control arm for G2's method-dependence.

Clusters below `min_mass` merge into a residual bucket, so the mass vector stays short
and interpretable. `weights` accepts `_qn_importance_weights` (mbr.py:512), so
`resample_to_qn` composes here as it does in the risk.

**Guards** (all raise, per §4): `mbr_n_candidates != 0` (rectangular `D`);
`mbr_beta != 1.0`; `mbr_R < R_max/2` for the active `mbr_coords`. `backend="surrogate"`
raises unless `screening_only=True` is passed explicitly.

**Dependency:** `sklearn.cluster.HDBSCAN` needs `scikit-learn >= 1.3`, added under the
existing **`[mbr]` extra**, not core — the `point_estimator="map"` path must import
nothing new.

---

## 6. WP2 — set-valued prediction

`models/base.py` gains a sibling to `map_or_mbr` (base.py:261):

```python
def predict_set(self, xf, nx, *, draws=None, coords_by_draw=None, **decode
                ) -> PosteriorSetEstimate:
    """One `LundPointEstimate` per posterior cluster, each a genuine draw, with the
    cluster's posterior mass and radius. `map_or_mbr` remains the point-estimate
    entry point and is untouched."""
```

```python
@dataclass
class PosteriorSetEstimate:
    members: list[LundPointEstimate]   # exemplars, mass-descending
    masses: np.ndarray
    radii: np.ndarray
    top_mass: float
    entropy: float
    clusters: PosteriorClusterSet      # the full labelling, for diagnostics
```

Each member is built through the existing `describe_cells(xf, nx, winner, win_coords)`
path (mbr.py:592), so every exemplar carries its own sampled coordinates and
`coords_source="sample"` exactly as the WP-C.1 medoid does.

---

## 7. WP3 — the two scalars

They are **not** to be folded into a single ±:

- **`top_mass`** — a *probability*: the posterior mass of the selected explanation.
- **`entropy`** $H(m)=-\sum_j m_j\log m_j$ — a per-jet **ambiguity** measure over
  discrete alternatives.
- **`radii[0]`** — the *continuous* resolution within the selected explanation, and the
  only one of the three legitimately reportable as a ±.

A bimodal posterior summarized as mean ± sd points at a configuration neither mode
supports. `LundPointEstimate` gains two optional fields (`cluster_mass`,
`cluster_entropy`, both `None` off-path) so existing single-estimate consumers —
`eval/closure.py`, `serving/`, the notebooks — carry the scalars without a signature
change.

---

## 8. WP4 — bounded-loss reduction, split into **4a (diagnostic, unconditional)** and **4b (product, gated)**

The reduction itself is ~15 lines (§8.2). What is *not* 15 lines is making it a
supported estimator: `.risk` changes meaning and has 14 consumers (§8.3). The split
follows that seam.

| | WP4a — diagnostic | WP4b — product |
|---|---|---|
| lands | with WP1, unconditional | only if 4a's measurements say so |
| reaches | `eval/` columns only | `serving`, `predict`, `export`, headline tables |
| `.risk` | untouched — stays the linear value | needs `risk_kind` provenance |
| config | none — an eval flag | `decode.mbr_loss` knob, documented, supported |
| cost | ~15 lines + one eval column | `risk_kind` across 14 sites, G8′ guard, ε policy, `empty_threshold` interaction, docs, 9-family parity |

**Why 4a is unconditional.** G2, G8 and G8′ were written as predictions to be reasoned
out from `radii`. For 15 lines they become *measurements*: how often the bounded argmin
actually differs from the linear one, whether it moves toward `members[0]` or away, and
whether the empty clique dominates. That is strictly better evidence than §8.0's
argument, and it costs less than the argument did. **Tier 3 therefore collapses into
Tier 1** — no separate arm, just extra columns on `cl_base`.

### 8.1 The losses do not change the clusters — and what WP4b therefore buys

`cluster_posterior` consumes `D`; `_reduce_risk` consumes `D` and emits a $K$-vector of
risks whose argmin is the point estimate. **The partition never sees the risk vector.**
Consequently `labels`, `exemplars`, `masses`, `radii`, `top_mass` and `entropy` are
**bit-identical** across `linear`, `bounded` and `kernel`. The only coupling anywhere is
a shared *parameter* rule (`mbr_loss_quantile` / `cluster_eps_quantile`), and under the
`hdbscan` default even that vanishes, since HDBSCAN takes `min_cluster_size`. Assert
this rather than assume it (§15).

So WP4b's entire effect is on the **single** point estimate, and the relevant comparison
is not against the linear medoid but against `predict_set().members[0]`:

$$
\text{bounded loss}\;\to\;\arg\max\ \text{peak density},
\qquad
\texttt{members[0]}\;\to\;\arg\max\ \textstyle\int \text{density}.
$$

These are different estimators, not two routes to one. A tight minority cluster can have
higher peak density than a broad majority cluster, so they disagree **exactly when the
clusters have unequal `radii`** and agree when they do not. For reporting "the most
probable explanation", mass is the coherent choice — mass *is* the posterior probability
of the region, and it comes with a calibratable number attached (G6) and no free
bandwidth. Peak density answers a different question, and it is not the one the
set-valued deliverable asks.

**This substantially weakens the case for WP4b.** It was justified as the fix for
medoid-in-the-valley; `members[0]` already fixes that, with a calibratable probability
attached and no free bandwidth. The residual case is only that peak density is sometimes
the estimator one wants — a question about the `radii` spread. WP4b is therefore
**optional** and does not block the plan's exit criteria.

**It does not weaken the case for WP4a at all** — the reverse. Everything in this
subsection is an argument from `radii`, and WP4a measures it directly for less code than
the argument took to write (§8.5).

### 8.2 Design and implementation — WP4a

The general Bayes estimator is $\hat y=\arg\min_{y'\in\mathcal H}\mathbb E_{y\sim
q_\phi}[\Delta(y',y)]$; the character of the answer is fixed by $\Delta$ (Goel & Byrne,
*Computer Speech & Language* **14** (2000) 115; Berger, *Statistical Decision Theory and
Bayesian Analysis*, Springer 1985, §2.4). With a loss bounded at scale $\epsilon$,

$$
\Delta_\epsilon(y',y)=\mathbb 1\!\left[d(y',y)>\epsilon\right]
\;\Longrightarrow\;
\text{risk}(y')=1-\tfrac1K\sum_k\mathbb 1\!\left[d(y',y^{(k)})\le\epsilon\right],
$$

so the argmin **maximizes the number of neighbours within $\epsilon$** — a Parzen
window (Silverman, *Density Estimation*, Chapman & Hall 1986, §3) evaluated on the
pool. The estimate is a KDE mode restricted to valid draws.

### Implementation

Replace the single line `risk = D.mean(axis=1)` (mbr.py:583) with a dispatch:

```python
def _reduce_risk(D, w, *, loss, eps):
    """Row-wise Bayes risk under the configured loss. `linear` is the merged
    behaviour and must stay bit-identical."""
    if loss == "linear":
        return (D * w[None, :]).sum(1) / w.sum()
    if loss == "bounded":
        return 1.0 - ((D <= eps) * w[None, :]).sum(1) / w.sum()
    if loss == "kernel":
        return -(np.exp(-0.5 * (D / eps) ** 2) * w[None, :]).sum(1) / w.sum()
    raise ValueError(f"unknown mbr_loss={loss!r}")
```

`w` is uniform unless `resample_to_qn` (mbr.py:579–582), so the existing $q(N\mid x)$
correction composes with all three losses unchanged. **Cost: zero additional EMD calls.**

**ε is pre-registered, not tuned:** $\epsilon = Q_\gamma(\{D_{ij}: i\neq j,\ D_{ij}>0\})$
with $\gamma = 0.10$ fixed before any test run, recorded with `fitted_under`. Tuning ε
against closure metrics is forbidden — it is the free parameter the construction turns
on, and a closure-tuned bandwidth makes G7 circular. The quantile form also makes ε
invariant to the `mbr_norm` / `energyflow` $1/R$ convention (README), which is *why* it
is a quantile rather than an absolute.

**Parity:** the default path must reproduce today's `risk` elementwise and select the
same `win_idx`; unit test asserts `max|Δrisk| == 0.0` on the synthetic fixture, matching
the `verify_parity.py` standard.

**WP4a surface.** No `DecodeConfig` field. `mbr_select` gains
`diagnostic_losses: tuple = ()`; when non-empty it returns, alongside the unchanged
`LundPointEstimate`, a side-channel dict `{loss: win_idx}`. `eval/closure.py` consumes it
behind `experiment.cluster_diagnostics` and reports the columns in §8.5. Nothing else in
the tree sees it, so there is no `.risk` question, no serving surface and no config-hash
churn.

**ε policy for 4a: per-jet.** $\epsilon = Q_\gamma$ of that jet's own off-diagonal
positive distances. Within a jet the neighbour counts are compared at a common $\epsilon$,
which is all the argmin needs — this is a variable-bandwidth (nearest-neighbour) KDE in
the sense of Loftsgaarden & Quesenberry, *Ann. Math. Statist.* **36** (1965) 1049, not an
ad hoc choice. It also removes the `fitted_under` freeze machinery entirely, since a
per-jet statistic is not a fitted scalar. **This is exactly what 4b cannot inherit** —
see §8.3.

### 8.3 Why WP4b is not 15 lines: `.risk` changes meaning

`LundPointEstimate.risk` is documented as "the achieved mean distance" (mbr.py:549).
Under `bounded` it becomes $1-(\text{neighbour fraction})$ — dimensionless, in $[0,1]$,
not an EMD. Fourteen call sites assume otherwise:

- `serving/api.py:93` exports it over the wire as `mbr_risk`;
- `scripts/lund_closure_report.py:702`, `scripts/make_per_jets*.py` and five notebooks
  **aggregate it across jets**;
- `point_estimate.py:81` uses `risk is not None` as the MAP-vs-MBR discriminator in
  `pretty()`.

None of these break loudly. They keep running and emit numbers on a different scale —
the failure mode the repo tracks most carefully, and precisely what `coords_source`
(point_estimate.py:70) exists to prevent for coordinate provenance. WP4b therefore
requires the same pattern:

```python
risk_kind: str = "emd_mean"   # emd_mean | neighbour_deficit | kernel_score
#                               .risk is only comparable within one risk_kind.
```

**And the per-jet ε that makes 4a simple makes 4b harder.** Under a per-jet bandwidth,
`.risk` is comparable *within* a jet but **not across jets** — and the closure scripts
aggregate across jets. So 4b must either freeze one global $\epsilon$ (reintroducing the
`fitted_under` machinery and the pre-registration discipline) or mark `.risk` as
non-aggregable through `risk_kind` and fix the five notebooks. The ε policy is forced by
a downstream consumer, not by the estimator — which is the clearest single statement of
why the seam falls where it does.

**One further interaction to pin.** `map_or_mbr`'s `empty_threshold` gate
(base.py:277–283) decides emptiness *before* the shape decode. Under `bounded`, emptiness
can also arrive *through* the decode (§8.4). Two paths to the same decision with different
calibration; 4b must specify which wins and record it.

---
### 8.4 The empty-clique hazard — blocking for WP4b

`_empty_value` (mbr.py:136) puts all empty draws at mutual distance **exactly 0**. At the
measured ~17 % empty rate and $K = 500$ that is ~85 draws forming a **zero-diameter
clique**. An empty candidate's neighbour count is therefore ~85 *for any* $\epsilon$,
while a non-empty candidate must find 85+ draws within $\epsilon$ to beat it.

So `mbr_loss="bounded"` can **collapse to the empty tree** at small $\epsilon$ —
reproducing exactly the MAP degeneracy the README credits MBR with removing
*structurally* rather than clamping. The linear loss is immune: the empty cloud pays the
full imbalance penalty inside the mean.

Note the $\epsilon$ rule does not protect against this. $\gamma = 0.10$ is a quantile
over the **positive** off-diagonal distances, so the empty–empty pairs are excluded from
setting $\epsilon$ yet still counted in the neighbour tally — the clique is invisible to
the bandwidth rule and decisive in the reduction.

**Measurable before any expensive run** (reuses `D`): per jet, the empty clique size
against the neighbour count of the best non-empty candidate at the frozen $\epsilon$.
This is gate **G8′**, blocking. If the crossover sits anywhere near $\gamma = 0.10$, WP4
requires either an $\epsilon$ floor above the clique scale or explicit exclusion of the
$N = 0$ stratum from the bounded candidate set — at which point it is a *patched*
estimator competing against an unpatched one, and §8.1 already says it is competing for
the wrong prize.

**G8′ is now measured by WP4a, not predicted.** The clique-size-vs-neighbour-count
comparison is one of 4a's columns (§8.5), computed on `cl_base`'s existing `D`.

### 8.5 What WP4a measures — the columns that decide WP4b

Computed in **`eval/stability.py`** (not `systematics.py` — see §8.6) and surfaced by
`eval/closure.py`, all from `cl_base`'s single `D`:

| column | question it settles |
|---|---|
| `argmin_moved` | does `bounded` select a different draw than `linear`? A near-zero rate closes WP4b outright. **Also the one column that survives WP4b's closure** — see §8.6 |
| `bounded_is_members0` | when it moves, does it move **toward** the top-mass exemplar or away? §8.1 predicts "away, when `radii` are unequal" — this tests it |
| `bounded_is_empty` | the G8′ statistic: does the $N=0$ clique win? |
| `empty_clique_size` / `best_nonempty_count` | the G8′ margin, so a near-miss is visible rather than a pass/fail |
| `eps_per_jet` | the realized bandwidth distribution — informs whether a single frozen ε is even viable for 4b |
| `d_bounded` vs `d_mbr` vs `d_top` | closure distance to truth for all three, so G8 is answerable without shipping anything |

Each is a scalar per jet, reusing `D`; the marginal cost over `cl_base` is a few
microseconds and one array comparison.

### 8.6 The loss spread is **not** a systematic — placement is `eval/stability.py`

Recorded as a decision so it is not re-litigated, because the tempting misuse is
specific: folding `d(\hat y_{\rm linear}, \hat y_{\rm bounded})` into the uncertainty
budget alongside `generator_spread`.

**Why it is not a systematic.** `generator_spread` and `frag_weights` vary something
*unknown about nature* — which fragmentation model is right — so their spread is a real
uncertainty on a fixed target. Loss choice varies something *the analyst decides*, and
`linear` and `bounded` are not two approximations to one quantity: they are **different
functionals of the same posterior**, the Fréchet median and a density mode. Quoting their
spread as an uncertainty is quoting the mean-minus-median difference as a systematic on
the mean. Both are exactly right; they answer different questions.

It would also **double-count**. The uncertainty on the parton-level tree is the posterior
width, which `radii[0]` and the cluster masses already report (§7). A loss-choice spread
added in quadrature inflates an interval that already contains the effect.

**The near-precedent that does not carry.** Unfolding analyses do quote a
regularization-strength systematic — varying the IBU iteration count (D'Agostini,
*NIM A* **362** (1995) 487). That varies a regularization *strength* on one estimand.
Here the estimand itself changes. The analogy fails at the point that matters.

**What it is instead: a stability check**, reported *beside* the answer, never folded
into it — the same epistemic role as MAP-vs-MBR disagreement in v1, a diagnostic of
posterior entropy rather than an error bar. Two virtues make it worth keeping:

- **`argmin_moved` is a 1-bit multimodality flag that needs no clustering.** No
  `sklearn`, valid at small $K$, and — the reason it earns its place — **it works on real
  data**, where there is no truth and G2′ is therefore unavailable. It is a coarse proxy
  for `entropy`, degenerate with it wherever clusters exist, but available where they
  do not.
- It is free: WP4a already computes it.

**Placement, enforced by import path.** These columns live in **`eval/stability.py`**,
not `eval/systematics.py`. The module boundary is the cheapest available guard against a
later quadrature sum; a comment in a docstring is not.

**The configuration to avoid**, stated explicitly: `bounded` as the *reported* point
estimate with no cluster layer. That is where both hazards bite and neither safeguard is
present — §8.3's `.risk` becomes a neighbour deficit that five notebooks will silently
aggregate as an EMD, and §8.4's empty clique can capture the argmin with nothing
checking it. For a single tree, `linear` is the safe default and `members[0]` the
principled one.


## 9. WP5 — calibrating the scalars against truth

Three failure modes, three treatments. Run on the v1 held-out set.

1. **Selection bias.** $R_j$ was defined using the same draws whose membership is
   counted, so `top_mass` is biased upward (post-selection inference: Berk, Brown, Buja,
   Zhang & Zhao, *Ann. Statist.* **41** (2013) 802; Fithian, Sun & Taylor,
   arXiv:1410.2597). Fix: **sample splitting** — cluster and pick exemplars on pool A,
   assign a fresh pool B to the A-exemplars by nearest EMD, estimate masses from B.
   `decode.cluster_split: bool = False` (off = today's single-pool estimate, reported as
   biased high). Cost: the B-assignment is $|C|\times K$, not $K^2$.
2. **$q_\phi \neq p$.** Reliability diagram of realized "truth in top cluster" frequency
   against claimed `top_mass`, with ECE and the Brier decomposition into reliability /
   resolution / uncertainty (Murphy, *J. Appl. Meteor.* **12** (1973) 595; Gneiting &
   Raftery, *JASA* **102** (2007) 359) — the latter separating "the numbers are
   miscalibrated" from "the numbers carry no information", which are different failures
   with different fixes. If miscalibrated, one temperature on the mass vector fit on
   validation and frozen (Guo, Pleiss, Sun & Weinberger, ICML 2017, arXiv:1706.04599),
   with the same discipline as `fit_length_recalibration`'s `(T, tilt)`.
3. **Generator dependence.** Recompute the mass vector under the Herwig arm and the
   `frag_weights` variations; quote the spread through
   `eval/systematics.py:generator_spread`. A cluster whose mass moves 0.62 → 0.31
   between generators is not a 62 % statement about nature, and the plan must be able to
   say so. **The linear-vs-bounded loss spread does not belong in this list** — §8.6
   records why, and `eval/stability.py` is where those columns live instead.

**Conformal fallback.** For a guarantee *independent* of $q_\phi$'s calibration,
calibrate a threshold on the accumulated mass over the closure split and emit the
smallest cluster set exceeding it; under exchangeability this gives finite-sample
marginal coverage $\ge 1-\alpha$ however wrong $q_\phi$ is (Vovk, Gammerman & Shafer,
*Algorithmic Learning in a Random World*, Springer 2005; Angelopoulos & Bates,
arXiv:2107.07511). `fit_set_threshold(scores, alpha)` in `inference/clusters.py`, frozen
on test with `fitted_under`. The guarantee is **marginal over jets**, not conditional on
$x$ — the same coverage notion TARP tests, and it must be documented that way rather
than as a per-jet guarantee.

Monte Carlo error on $w_j$ is $\sqrt{w_j(1-w_j)/K}\approx 0.022$ at $K=500$, $w=0.6$ —
negligible against (1)–(3); reported, not gating.

---

## 10. WP6 — eval and notebook integration

1. `eval/closure.py:run_closure` (closure.py:132) gains, behind
   `experiment.cluster_diagnostics`: per-jet `n_clusters`, `top_mass`, `entropy`,
   `truth_in_top` (nearest-exemplar assignment of the truth), and the
   **medoid-in-dominant-cluster** indicator — the G2 statistic.
1b. **The oracle-set diagnostic — G2′.** G2 never consults the truth; it asks only
   whether the medoid is centrally placed. The complementary truth-based question is
   whether the *set* recovers what the point estimate misses. Per jet, over exemplars
   $\{e_j\}$ with masses $\{m_j\}$:

   $$
   d_{\rm top}=d(e_{\arg\max m},\,y_{\rm true}),\quad
   d_{\rm best}=\min_j d(e_j,\,y_{\rm true}),\quad
   d_{\rm mbr}=d(\hat y_{\rm MBR},\,y_{\rm true}).
   $$

   If $p(y\mid x)$ is genuinely multimodal and roughly calibrated, the truth is a draw
   from it and landed in one lobe; on the subset where that lobe is the minority one,
   $d_{\rm best}\ll d_{\rm top}\approx d_{\rm mbr}$, with the gap of order the
   inter-cluster separation. Population-level, $\langle d_{\rm best}\rangle$ should
   undercut $\langle d_{\rm mbr}\rangle$ by roughly $m_{\rm minority}$ times that
   separation.

   $d_{\rm best}$ is an **oracle** quantity — it uses the truth to select the member.
   Legitimate as a diagnostic, dishonest as a reported result: the *set* is the
   deliverable, and $d_{\rm best}$ measures only whether the set is worth reporting.
   It never enters a headline table.

   **Three controls, all mandatory:**

   - **Min-of-$n$ null (the decisive one).** Taking a minimum over $n$ exemplars
     improves the distance even for a *random* partition, purely as an order
     statistic. So $d_{\rm best}<d_{\rm mbr}$ is not evidence of anything. Control:
     partition the pool at random into the same number of groups with the same masses,
     take within-group medoids as exemplars, recompute $d_{\rm best}^{\rm rand}$. **The
     signal is $d_{\rm best}^{\rm real}$ vs $d_{\rm best}^{\rm rand}$, not
     $d_{\rm best}$ vs $d_{\rm mbr}$.** Reuses `D`; no new EMD calls. Averaged over
     ≥ 20 random partitions per jet for a stable null.
   - **Separation-over-width precondition.** $d(e_j,y_{\rm true})$ carries the
     within-cluster scatter `radii[j]` even in the correct lobe, so the effect is
     detectable only when inter-exemplar distance $\gg$ radius. This is a silhouette
     condition, computable from `D` **before any truth is consulted** — report it as a
     precondition, and if it fails, the bimodality is unresolvable at this metric and
     budget whether or not it is real.
   - **Unassigned rate.** Nearest-exemplar assignment is itself a decision, wrong when
     the truth sits between clusters or outside the pool's support. Do not force-assign:
     mark the truth unassigned when it is farther from every exemplar than the 95th
     percentile of within-cluster radius, and report the rate. It is a per-jet
     out-of-support indicator no calibration statistic sees, and it belongs beside the
     existing `support_audit` numbers.

1c. **G2′ and G6 are one measurement pass.** Binning jets by $m_{\rm top}$ and asking
   how often the truth landed in the top cluster *is* the reliability diagram of §9.2.
   G2′ asks whether the truth is much closer to one exemplar than the others; G6 asks
   how often that one is the highest-mass exemplar. Same nearest-exemplar assignment,
   same loop — implement once.

1d. **Scope discipline.** "The jet population is bimodal" and "$p(y\mid x)$ is bimodal
   for this jet" are different claims and only the second is in scope. A bimodal
   population marginal (quark- vs gluon-initiated, say) yields unimodal conditionals
   wherever $x$ separates them; a unimodal marginal can have bimodal conditionals
   wherever the forward map folds. The target is the second: $x$ under-determining
   which shower history occurred, with the candidate explanations being the $N$-strata
   alternatives the imbalance term keeps apart. Report G2′ stratified by $N_{\rm top}$
   vs $N_{\rm second}$ so a split *between* strata is distinguishable from a split
   *within* one.
2. `eval/report.py`: reliability diagram, ECE, Brier decomposition for `top_mass`;
   conformal coverage vs nominal with 95 % Wilson bands (Brown, Cai & DasGupta,
   *Statist. Sci.* **16** (2001) 101), matching the v1 convention.
3. **Stratification by `ln_pt`.** Entropy binned in $\ln p_T$ must be flat. A cluster
   split that tracks jet scale indicates incomplete conditioning rather than physical
   ambiguity; `ln_pt` is already a registered aux feature (features.py `AUX_FEATURES`),
   so this costs nothing. Flat ⇒ the splits are the discrete emission-count explanations
   the plan is after.
4. **Region stratification** over the v0 Lund quadrants, per the standing rule that
   calibration holding only on average does not pass.
5. `notebooks/inference_demo.ipynb` §6: a single-jet panel showing the pool projected by
   classical MDS on `D` (display only — the clustering never sees the embedding),
   exemplars marked, masses annotated, truth overlaid.
6. **Pushforward hygiene, asserted in code.** `PLAN_UPDATES.md` WP5's aggregate
   cross-check must consume ancestral draws. A selected exemplar is systematically
   closer to the bulk than a typical draw, so an exemplar-built pushforward is
   under-dispersed exactly in the soft/wide-angle corner it most needs to get right.
   Assert on the provenance flag, not a comment.

---

## 11. Running at stock MBR settings

WP1–WP3 and WP6's diagnostics are runnable on the existing v1 checkpoints **today**,
with `cluster_posterior=true` as the only intended change. Four settings to check first:

- **`mbr_n_candidates`** — `DecodeConfig` defaults to 0 (square `D`), which is correct.
  But `notebooks/inference_demo.ipynb` sets `MBR_N_CANDIDATES = 24`, which is
  rectangular; anything driven from the notebook must reset it. Guarded by a raise
  rather than a silent override, since overriding would change the point estimate the
  caller asked for.
- **`mbr_beta`** — must be 1.0 (§4.1). The default is 1.0.
- **`mbr_backend`** — `pot` or `energyflow` for any quoted mass vector; `surrogate`
  screening-only (§4.3).
- **`n_posterior_samples = 500`** — sufficient to identify a dominant cluster and
  compute G2; thin for resolving *how many* clusters there are, since `min_mass = 0.05`
  is then only 25 draws. **G2 is answerable at the default; the mass vector's tail is
  not.** This is what the `cl_K250` / `cl_K1000` arms measure.

The point estimate under all of the above is bit-identical to today, because
`mbr_loss` stays `"linear"` and line 583 is untouched.

---

## 12. Config schema diff

`DecodeConfig` (config.py:258), all defaults preserving today's behaviour; mirrored in
`_DECODE_DEFAULTS` (config.py:619) and `configs/`:

```python
# --- cluster layer (WP1-WP3). Independent of the risk reduction below. -------
cluster_posterior: bool = False     # build the K x K cluster labelling. Requires
#                                     mbr_n_candidates == 0 and mbr_beta == 1.0
#                                     (raises otherwise; see PLAN §4).
cluster_method: str = "hdbscan"     # hdbscan | dbscan | pam
cluster_min_cluster_size: int = 0   # hdbscan; 0 => max(5, ceil(0.05 * K))
cluster_eps_quantile: float = 0.10  # dbscan ONLY: eps = Q_gamma of the off-diagonal
#                                     positive distances. Backend- and R-invariant by
#                                     construction. Unused under hdbscan.
cluster_min_mass: float = 0.05      # clusters below this merge into a residual bucket
cluster_split: bool = False         # sample-split the mass estimate (WP5.1). Off keeps
#                                     the single-pool estimate, which is biased HIGH.
set_alpha: float = 0.32             # conformal miscoverage for predict_set (1-sigma)

# --- risk reduction. WP4a adds NO config field: it is an eval-only side channel
#     (`mbr_select(diagnostic_losses=...)`, §8.2). The two fields below belong to
#     WP4b and are NOT added until G8/G8′ and the §8.5 columns justify them. ------
# mbr_loss: str = "linear"          # WP4b ONLY. linear | bounded | kernel. Requires
#                                     `risk_kind` provenance first (§8.3) — .risk is
#                                     not an EMD under bounded/kernel.
# mbr_loss_quantile: float = 0.10   # WP4b ONLY, and only if a FROZEN global epsilon is
#                                     chosen over WP4a's per-jet one (§8.3).
```

`ExperimentConfig`: `cluster_diagnostics: bool = False`.

---

## 13. Pre-registered gates

Evaluated on the independent seed-2 test file; coverage intervals 95 % Wilson; regions
with $n<30$ reported `scored: false`, per the v0/v1 convention.

| # | gate | criterion | gates |
|---|---|---|---|
| G1 | **parity** | `cluster_posterior=false`, `mbr_loss="linear"`: `risk` elementwise identical to merged (`max|Δ| == 0.0`), same `win_idx`; `state_dict` and `log_prob` untouched | all |
| G2 | **necessity** | fraction of jets whose linear medoid lies in the dominant cluster. **≥ 0.90 ⇒ WP4 is closed as unnecessary.** Reported for `hdbscan` and `pam`; the two must agree in verdict or the verdict is "method-dependent, unresolved" | **WP4** |
| G2′ | **set value (truth-based)** | $\langle d_{\rm best}^{\rm real}\rangle$ beats the mass-matched random-partition null $\langle d_{\rm best}^{\rm rand}\rangle$ by more than the seed spread, **and** the silhouette precondition holds on the scored subset. Stratified by whether the top two clusters differ in $N$. Oracle quantity — diagnostic only, never a headline | **WP4**, and whether `predict_set` is worth shipping |
| G3 | **empty stratum** | the $N=0$ draws form exactly one zero-radius cluster; its mass agrees with `length_pmf`'s $q(0\mid x)$ within MC error on ≥ 95 % of jets. Disagreement is a metric-convention bug, not a finding | WP1 |
| G4 | **metric admissibility** | asserted, not measured: `mbr_beta == 1.0`; $R \ge R_{\max}/2$ for the active `mbr_coords`; `D` symmetric to < 10⁻¹⁰ after symmetrization; `surrogate` refused outside screening | WP1 |
| G5 | **budget stability** | `top_mass` and `entropy` at $K=250$ vs $K=1000$ agree within their binomial error on ≥ 90 % of jets, **or** the plan quotes only the $K=1000$ tier | WP3, WP5 |
| G6 | **reliability** | ECE of `top_mass` ≤ 0.05 after the WP5.2 temperature, reliability-curve slope Wilson-consistent with 1. Reported **both** pre- and post-recalibration | WP3, WP5 |
| G7 | **conformal coverage** | empirical set coverage ≥ $1-\alpha$ within its 95 % Wilson band at the frozen threshold | WP5 |
| G8′ | **empty-clique dominance** *(blocking for WP4)* | per jet, the $N=0$ clique size vs the neighbour count of the best non-empty candidate at the frozen $\epsilon$. If the empty candidate wins on > 1 % of jets, `bounded` does **not** ship without an $\epsilon$ floor or $N=0$ exclusion — and §8.1 applies first | **WP4** |
| G8 | **no regression** | *(only if WP4 proceeds)* the bounded-loss estimate's closure W1 gmean is not worse than **both** the linear medoid's and `predict_set().members[0]`'s beyond the seed spread. Per §8.1 the second comparison is the meaningful one; if it fails, `linear` stays default and `bounded` ships diagnostic-only | WP4 |
| G9 | **selection bias** | `cluster_split=true` vs `false` top-mass difference reported; if > 0.05 on average, split becomes the default for any quoted number | WP5 |

**Kill criterion, stated up front:** G2 ≥ 0.90 **and** G2′ null ⇒ the posterior is
effectively unimodal in this metric and at this budget: the medoid is already central and
the set recovers nothing beyond an order statistic. WP4 closes, `predict_set` ships as a
diagnostic rather than a product, and the plan reduces to reporting `radii[0]` as a per-jet
resolution alongside the existing MBR point estimate. The remainder is deferred with a
trigger (revisit after the multi-seed implicit continue/stop candidate lands).

**The two necessity gates are independent and can disagree**, which is informative rather
than awkward. G2 is truth-free and therefore transfers to real data; G2′ uses the truth and
is available only in closure. G2 ≥ 0.90 with G2′ *positive* means the lobes exist but the
medoid sits inside the dominant one — the set is worth shipping, the bounded loss is not.
G2 < 0.90 with G2′ *null* means the clustering is finding structure the metric cannot
resolve against truth: check the silhouette precondition before believing either.

---

## 14. The run grid

All training-free; existing v1 checkpoints, no new trainings.

**Tier 1 — answers G2 at stock settings, cheap:**

| arm | varies | purpose |
|---|---|---|
| `cl_screen` | `mbr_backend=surrogate`, `K=500` | G2 first pass at ~0 cost; verdict only. **Not valid for G2′** — the surrogate is mass-blind (§4.3), so truth-to-exemplar distances collapse the N-strata the diagnostic tests |
| `cl_base` | stock (`pot`, `linear`, `K=500`) + `diagnostic_losses=("bounded","kernel")` | G1–G4, **G2′** (with its ≥20-partition null), **G8, G8′ via the §8.5 columns**, G9 |

**Tier 2 — mass vectors and calibration, gated on Tier 1:**

| arm | varies | purpose |
|---|---|---|
| `cl_K1000` | `n_posterior_samples=1000` | G5, and the quoted mass vectors |
| `cl_K250` | `n_posterior_samples=250` | G5's lower leg |
| `cl_pam` | `cluster_method=pam` | G2 method-dependence control |
| `cl_split` | `cluster_split=true` | G9 |

**Tier 3 — retired.** WP4a folds the bounded and kernel reductions into `cl_base` as
side-channel columns (§8.5), so no separate arm is needed: per §8.1 the partition, masses
and scalars are bit-identical, and only the single point estimate moves. G8, G8′ and the
§8.1 peak-density-vs-mass question are all answered from `cl_base`'s one `D`. WP4b, if it
proceeds, is a productionization PR against measurements already in hand — not a run.

**Budget note.** Density estimation needs resolution in $\mathcal{H}$ itself, so $K$ is
what must grow — the sample size to resolve modes scales far worse than the sample size
to estimate a mean. At $K=1000$ the $K^2$ block is $10^6$ pairs/jet; at the measured
`energyflow` batched rate (2.3 µs/pair, GB10) that is ≈ 2.3 s/jet, so Tier 2 runs at
reduced `closure_jets`. `cl_screen` exists precisely so the expensive tier is only
entered if Tier 1 says there is something to resolve.

---

## 15. Tests

- `tests/test_clusters.py`: synthetic two-lobe distance matrix recovers two clusters
  with the right masses; an all-zero matrix returns one cluster; an all-empty-draw
  matrix returns one zero-radius cluster (G3 in miniature); each §4 guard raises.
- `tests/test_clusters.py::test_random_partition_null`: on a **single-lobe** synthetic
  matrix, `d_best_real` and `d_best_rand` must agree within MC error — the negative
  control proving G2′'s null is doing its job. Paired with a two-lobe positive control
  where the real partition beats the null. Without both, G2′ is an untested order
  statistic.
- `tests/test_mbr.py`: the §4 metric audit as a regression test — symmetry, zero
  diagonal, and triangle inequality on a fixed 40-cloud fixture at β = 1, plus the
  β = 2 **negative** control asserting violations are detected. G1 parity assertion;
  `_reduce_risk` unit-tested against a hand-computed 4×4 matrix for all three losses;
  the rectangular-`D` guard raises.
- `tests/test_clusters.py::test_losses_do_not_move_clusters`: run `cluster_posterior`
  on the same `D` under all three `mbr_loss` settings and assert `labels`, `exemplars`,
  `masses` and `radii` are **bit-identical**. §8.1 is the plan's load-bearing
  orthogonality claim; it should be a test, not a paragraph.
- `tests/test_stability.py::test_loss_spread_not_in_systematics`: assert
  `eval/systematics.py` neither imports from nor emits any `argmin_moved` /
  `d_bounded` key. §8.6 as an executable boundary rather than a convention.
- `tests/test_mbr.py::test_diagnostic_losses_side_channel`: `diagnostic_losses=()` is
  bit-identical to merged (`state_dict`, `risk`, `win_idx`); non-empty returns the side
  channel **without** mutating the returned `LundPointEstimate` — the WP4a containment
  guarantee, asserted rather than described.
- `tests/test_mbr.py::test_empty_clique_dominance`: a fixture with 17 % empty draws;
  assert the bounded loss selects the empty tree at small ε and does not at the ε floor.
  The G8′ hazard as an executable regression.
- Parametrized over every model family as the existing `tests/` are — the cluster layer
  touches only `D`, so family-agnosticism is a cheap invariant to assert.
- CI fast tier: `cluster_posterior=true` on `data=synthetic` with `K=64`.

---

## 16. Exit criteria

Complete when: G1, G3 and G4 pass; **G2 and G2′ are both *reported*** (either outcome is
a result, and the pair jointly decides whether WP4 and `predict_set` proceed — see the
kill criterion), with G2′ accompanied by its random-partition null, its silhouette
precondition and its unassigned rate; G5–G7 exist as run-dir artifacts; `CONFIGURATION.md` §7
documents every new knob including the `mbr_n_candidates` / `mbr_beta` / `surrogate`
guards and the ε pre-registration; **WP4a ships with WP1 and its §8.5 columns are
reported from `eval/stability.py`**, with §8.6's exclusion asserted by test;
**WP4b is explicitly non-blocking** — §8.1 reclassifies it as optional, so
the plan can exit complete with no `mbr_loss` knob ever entering `DecodeConfig`,
provided the §8.5 columns exist; and `PLAN_empty_parton_tree.md` gains a
cross-reference recording whether the cluster mass vector is a viable input to the Chow
reject rule.

---

## 17. References

Angelopoulos & Bates, arXiv:2107.07511 ·
Berger, *Statistical Decision Theory and Bayesian Analysis*, Springer 1985 ·
Berk, Brown, Buja, Zhang & Zhao, *Ann. Statist.* **41** (2013) 802 ·
Brown, Cai & DasGupta, *Statist. Sci.* **16** (2001) 101 ·
Campello, Moulavi & Sander, PAKDD 2013 ·
Chow, *IEEE Trans. Inf. Theory* **16** (1970) 41 ·
Eikema & Aziz, EMNLP 2022, arXiv:2108.04718 ·
Eikema & Aziz, arXiv:2005.10283 ·
Ester, Kriegel, Sander & Xu, KDD 1996 ·
Fithian, Sun & Taylor, arXiv:1410.2597 ·
Gneiting & Raftery, *JASA* **102** (2007) 359 ·
Goel & Byrne, *Computer Speech & Language* **14** (2000) 115 ·
Guo, Pleiss, Sun & Weinberger, ICML 2017, arXiv:1706.04599 ·
Kaufman & Rousseeuw, *Finding Groups in Data*, Wiley 1990 ·
Komiske, Metodiev & Thaler, *Phys. Rev. Lett.* **123** (2019) 041801, arXiv:1902.02346 ·
Kumar & Byrne, HLT-NAACL 2004 ·
Lemos, Coogan, Hezaveh & Perreault-Levasseur, ICML 2023, arXiv:2302.03026 ·
Lifson, Salam & Soyez, *JHEP* **10** (2020) 170, arXiv:2007.06578 ·
Mardia & Jupp, *Directional Statistics*, Wiley 2000 ·
Murphy, *J. Appl. Meteor.* **12** (1973) 595 ·
Silverman, *Density Estimation for Statistics and Data Analysis*, Chapman & Hall 1986 ·
Stahlberg & Byrne, arXiv:1908.10090 ·
Talts, Betancourt, Simpson, Vehtari & Gelman, arXiv:1804.06788 ·
Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, Springer 2005
