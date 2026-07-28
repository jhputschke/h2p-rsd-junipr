# PLAN — Aux conditioning: groomed all-branch observables as encoder input

**Status: implemented, opt-in, NOT adopted as a default.** Stage 1 (C++) and stage 2
(Python) are merged; `encoder.aux_features` defaults to `[]` and the off path is
byte-identical. The A/B was run and **exit criterion (i) fails on this sample**: the
held-out NLL gain (−0.029 nats) is exactly the size of the seed spread and one of three
seeds goes the wrong way. Criterion (iv) is additionally blocked on `PLAN_UPDATES.md`
WP5. See "Results against the stated exit criteria" below — including why this sample
has almost no secondary-plane activity to exploit, and what to change before re-running.

Enrich the conditioning side of q(y | x) with per-jet **groomed** scalars that the
primary-only hadron sequence x cannot represent — the pipeline-groomed jet mass
`x_mg`, the all-branch minus primary passing-splitting count `x_nsec`, and the jet
scale `jet_pt` — produced by the C++ writer and consumed by the encoders as
broadcast per-node features. Opt-in via `encoder.aux_features` (default `[]`), so
the off path is byte-identical to today: same `state_dict`, same `log_prob`, same
RNTuple compatibility. Builds on the merged `PLAN_MBR_PerturbativeLund.md` and
`PLAN_MultHead.md`; complements (does not touch) `docs/PLAN_UPDATES.md` WP3.

## Context

The encoder input is the **primary** Lund sequence only: `primaryLund`
(cpp/src/lund_io.cpp:11–29) iterates `LundGenerator` declusterings and filters on
the Soft Drop boundary + ln kt floor; each surviving emission enters
`node_features` (features.py:14–24) as an effectively structureless
(ln 1/ΔR, ln kt, ln z, ψ) node. Everything inside the softer prongs — the
secondary Lund planes — is discarded at write time. Two conditioning-relevant
quantities therefore never reach e(x):

- **Groomed jet mass.** Each primary node is recorded massless; the secondary
  subjet masses, and hence m_g of the groomed jet, are not functions of x.
- **Secondary splitting activity.** The count of grooming-passing splittings on
  non-primary branches. Secondary-plane density reflects the Casimir of the
  emitting prong, carrying quark/gluon information beyond the primary plane
  (Dreyer, Soyez & Takacs, arXiv:2112.09140; LundNet, Dreyer & Qu,
  arXiv:2012.08526), and the posterior over y is implicitly a flavor mixture.

Both are **groomed** quantities: they retain the NP/UE suppression that motivated
Soft Drop (Larkoski et al., arXiv:1402.2657) and remain usable in the heavy-ion
environment — unlike ungroomed constituent multiplicity or ungroomed mass, which
are deliberately excluded here (see Non-goals). `jet_pt` is already written
(lund_writer.cpp:20) but never read into the jet dicts; it rides along for free
as the scale anchor.

Positive conditional information is expected iff I(y; aux | x) > 0; the cheap
in-repo estimator is the held-out conditional NLL delta, which is exactly what
the exit criteria gate on.

## Implementation status

Both stages are merged. `encoder.aux_features` defaults to `[]`; with it off the
module list, `state_dict`, `log_prob` and `scripts/verify_parity.py` output are
bit-identical (`PARITY PASSED`, max |delta| = 0.000e+00).

| Piece | Landed as |
|---|---|
| Stage 1 — groom predicate | `passesGroom` in `cpp/include/lund_io.hpp`; `primaryLund` rewritten to call it |
| Stage 1 — all-branch traversal | `JetAux` / `fullLundAux` (`cpp/src/lund_io.cpp`) |
| Stage 1 — schema | `x_mg`, `x_nsec`, `x_ptg`, `x_kt_sec_max`, `x_kt_sec_sum`, `x_sec_attach` in `LundWriter`; both drivers fill them; `read_lund_rntuple` guards on the descriptor |
| Stage 2 — registry | `AUX_FEATURES`, `aux_vector`, `with_aux`, `aux_source_fields`, `configured_aux_names` (`features.py`) |
| Stage 2 — data | `rntuple.py` sentinel reads; `MatchedLundDataset(..., aux_features)`; width-inferring `collate` |
| Stage 2 — config | `aux_features` on all three encoder schemas + YAMLs; in `_fingerprint`; `nx == 0` coverage report |
| Stage 2 — models / serving | four `build_encoder` sites; `PosteriorModel.aux_feature_names`; `predict` requires `x_seq["aux"]`; `cmd_export` trace width |
| Tests | `cpp/tests/test_lund_io.cpp` (fixtures, a 200-jet ensemble, and an independent `LundGenerator`-based cross-check), `tests/test_aux_features.py` (55 cases) |
| Data | `cpp/test_data/jets_aux.root` — the **same card and the same 25 000 events** as `jets.root` (identical 54 007 jets), plus the six aux columns |
| A/B | [`notebooks/aux_input_ab.ipynb`](../notebooks/aux_input_ab.ipynb) |

### Registry extension beyond the plan's three features

The plan scopes `[ln_mg_pt, nsec, ln_pt]`. Six more are implemented and tested, all
still opt-in and **none yet A/B'd** — the A/B result below covers the original triple
only.

- **`ln_ptg_pt` = ln(pt_g/pt)**, from a new `x_ptg` column. The in-scope way to express
  "how much did grooming remove". Deliberately **not** the mass drop `ln(m_g/m)`: since
  the encoder already sees `ln(m_g/pt)`, that ratio would be an invertible
  reparameterization handing it `ln(m/pt)` — the ungroomed mass this design excludes.
  `ln(pt_g/pt)` with `ln_pt` yields `ln(pt_g)`, a groomed quantity. Pinned by
  `test_ln_ptg_pt_does_not_reconstruct_the_ungroomed_mass`, which varies `jet_m` by 8×
  and asserts no aux feature moves.
  Measured UE response (medians, 3 000 events, MPI off → on): `pt` **+0.4 %** vs `m`
  **+9.7 %** as a normalizer; at ratio level `pt_g/pt` **−1.6 %**, `m_g/pt` **+6.1 %**,
  `m_g/m` **−5.3 %**. The new feature is the most UE-robust of the three.
  *Note* `pt_g/pt` ≈ 0.40 on this sample, not ≈ 0.95: RSD with no iteration limit plus
  the `k_t` floor discards collinear-but-hard prongs that textbook mMDT would keep. Same
  predicate as `m_g`, by design — but it means the quantity is governed by drops near the
  1 GeV floor (the NP boundary), so it is more NP- than UE-sensitive, and the MPI test
  above does not probe that.
- **`abs_eta` = |eta|/2** (`ETA_REF` is a fixed constant, never read from the data). At
  fixed `pt` the q/g fraction varies strongly with rapidity, and the posterior is
  implicitly a flavour mixture. Honest caveat: this is a **prior** handle, not a
  measurement one — it carries more generator-composition dependence than the groomed
  observables.
- **`has_sec`, `ln_kt_sec`, `ln_kt_sec_sum`, `sec_depth`** — secondary-plane
  *kinematics* rather than just the count, from new `x_kt_sec_max` / `x_kt_sec_sum` /
  `x_sec_attach` columns. One hard off-spine splitting and several soft ones give the
  same `n_sec` but different physics. These are **undefined** when `n_sec == 0`, so they
  ship with an explicit presence indicator and a neutral 0, and `log1p` is chosen so the
  neutral point is exactly 0 with any real value bounded away from it. Their
  absent-column sentinel is `-1`, not `0`, because `0` is legitimate. `kt_sec_max` is
  cross-checked against the independent `LundGenerator`-based reference alongside
  `n_all`.
  **They inherit the sparsity that made `nsec` unmeasurable** (17.4 % of jets have any
  secondary at all) and are worth evaluating only together with a working point that
  raises `⟨n_sec⟩` — see the grooming scan below.

### Deviations from the plan as written, and why

1. **`LundWriter::fill` takes the whole `JetAux`, not two extra scalars.** The
   `n_all - n_primary` subtraction and its unsigned-underflow guard then live in
   exactly one place instead of at every call site.
2. **`configured_aux_names` replaces the plan's inline
   `getattr(cfg.encoder, "aux_features", None) or []`.** That idiom would have been
   copied at four model sites plus the datamodule plus serving — six tolerant reads,
   none of which validates the names. One helper validates once, keeps the
   old-checkpoint tolerance in a single place (the `decode_params` / `experiment_params`
   idiom), and turns a typo into a `KeyError` at build time rather than a silent
   `ValueError` per jet. `with_aux` / `aux_source_fields` likewise deduplicate the
   broadcast and the error text between the dataset and the serving path.
3. **The synthetic refusal keys on the missing COLUMNS, not on `data.source`.** The
   plan says "`aux_features != []` with `data.source=synthetic` raises in
   `MatchedLundDataset`" — but the dataset is handed a jet list and never sees
   `cfg.data`. Keying on absent aux sources is strictly stronger: it also catches a
   *real* `jets.root` written before the schema change, and it lets the plumbing tests
   do what the plan asks (inject aux fields on fixture dicts) without a
   source-name escape hatch. The synthetic case still gets its own message, naming
   why no proxy is offered.
4. **`_fingerprint` takes the aux tuple as an argument** rather than reaching into
   `cfg.encoder`; it is a `cfg.data` helper, and coupling the two config groups inside
   it would be worse than one extra parameter.
5. **The C++ test gained a 200-jet random ensemble.** Four hand-built fixtures are
   shallow by construction, and a spine/predicate drift between `fullLundAux` and
   `primaryLund` would only show up on a deep, wide C/A tree. The ensemble replays
   `n_primary == primaryLund(...).size()` on 31-particle jets and additionally pins
   `m_g <= m_ungroomed`.
6. **`cmd_export` is in scope after all.** The plan does not mention it, but it traced
   the encoder with a hardcoded `torch.zeros(1, 3, 5)`, which fails for an aux
   checkpoint; the width now follows `model.aux_feature_names`.

### Results against the stated exit criteria

**C++.** All fixture checks pass, plus a 200-jet random-ensemble replay of
`n_primary == primaryLund(...).size()` on 31-particle jets and `m_g <= m_ungroomed`.
On the real sample: `m_g <= m_jet` for all 54 007 jets, `⟨m_g/m⟩ = 0.25`, and
`n_x == 0 ⟹ n_sec == 0` exactly, as the recursion structurally requires.

**Parity.** `PARITY PASSED`, max |delta| = 0.000e+00. Off-path `state_dict` keys,
tensors and `log_prob` identical; on-path costs 96 of 117 190 parameters and changes
no other shape. 268 tests pass.

**Physics A/B — the adoption gate FAILS on criterion (i), so aux is NOT adopted.**
`ar_junipr_v3 + gru`, 15 epochs, 3 seeds, `cpp/test_data/jets_aux.root`. Covers the
plan's original triple only; the six later registry entries are not yet A/B'd:

| arm | held-out NLL/jet | Δ vs baseline |
|---|---|---|
| baseline | 4.6136 ± 0.0205 | — |
| `+ [ln_mg_pt, nsec, ln_pt]` | 4.5848 ± 0.0202 | **−0.0288** |
| `ln_pt` only | 4.5948 ± 0.0408 | −0.019 |
| `ln_mg_pt` only | 4.6052 ± 0.0239 | −0.008 |
| `nsec` only | 4.5909 ± 0.0286 | −0.023 |

- **(i) FAIL.** The delta (−0.0288) exactly equals the combined seed spread (0.0288),
  and the paired per-seed deltas are `[+0.007, −0.061, −0.033]` — one of three seeds
  goes the wrong way. Not distinguishable from seed noise.
- **(ii) PASS.** Posterior-median multiplicity bias −0.234 → −0.264; MBR
  perturbative-Lund risk 23.69 → 22.69 (improves).
- **(iii) PASS.** Coordinate-PIT max KS 0.0583 → 0.0613, both under the 0.0648
  critical value; 68% coverage 0.69 → 0.67; SBC mean rank 0.469 → 0.496.
  Aux-stratified: at `n_sec >= 2` (217 jets) the aux arm's `du`/`dv` KS run
  0.021/0.034 → 0.053/0.074 — inside the critical value at that sample size, but the
  first number to watch on a sample with real `n_sec` headroom.
- **(iv) BLOCKED, not skipped.** There is no generator-B producer
  (`PLAN_UPDATES.md` WP5 not started), so `eval/systematics.py` has nothing to
  compare against and the prior-systematic side of the trade cannot be measured.

**Why the gate fails, mechanistically — the useful part of the result.** This
grooming working point leaves almost none of the information the plan is about in the
sample: **82.6% of jets have `n_sec == 0` exactly** (mean 0.22, max 7), and a further
6.9% have `n_x == 0` and structurally cannot carry aux at all. For most jets the
headline new observable is a constant. Two diagnostics confirm the aggregate number is
mostly noise rather than signal: the `n_x == 0` jets — which *cannot* receive aux, so
their two arms are the same function of the same input — "gain" −0.024 nats, the same
size as the overall −0.029; and no single-feature arm's delta exceeds its own spread.
The one place the design's own prediction does show up is the `n_sec = 2–3` stratum,
which gains **−0.100 nats/jet**, ~3× the sample average.

The scope of the finding is therefore: *the machinery is correct and the information
is real where it exists, but this sample has almost none of it.* This mirrors
`PLAN_UPDATES.md` WP3, which met its criterion on the synthetic generator and washed
out on this same file, for the same underlying reason: this sample is short on the
structure the feature exploits.

#### Is the 82.6% zero fraction a bug — is the traversal only walking the spine?

The obvious suspicion, since the persisted sequences *are* primary-only. Two checks say
no; `x_nsec` comes from `fullLundAux`, a different code path from `primaryLund`.

1. **An independent implementation agrees exactly.** `cpp/tests/test_lund_io.cpp`
   carries a second traversal built on fjcontrib's own `LundGenerator` (different
   declustering machinery, different harder/softer determination, `Delta`/`z`/`k_t` from
   `LundDeclustering`'s cached values): walk the primary plane, and at each passing
   splitting recurse into `d.softer()`. Over 200 jets × 3 `k_t` floors it matches
   `n_all` **exactly** — 5676 passing splittings, **3141 of them off-spine**.
2. **`n_sec` responds strongly to the cuts.** A spine-only traversal would report
   `n_sec == 0` in every row below. 3 000 events per setting:

| z_cut | k_t floor | pTHat | R | ⟨n_x⟩ | ⟨n_sec⟩ | P(n_sec ≥ 1) |
|---|---|---|---|---|---|---|
| 0.10 | 1.0 | >100 | 0.4 | 1.74 | **0.23** | 17% |
| 0.10 | **0.2** | >100 | 0.4 | 3.44 | **2.04** | 75% |
| **0.02** | **0.2** | >100 | 0.4 | 5.37 | **2.53** | 81% |
| 0.10 | 1.0 | **>500** | 0.4 | 1.93 | **0.67** | 33% |
| 0.10 | 1.0 | >100 | **0.8** | 2.10 | **0.62** | 36% |

So the zero fraction is a statement about the working point, not the code. The reason it
is so high at the default is that secondary activity requires **two** things to compound:
a passing primary splitting to open the plane at all (⟨n_x⟩ = 1.74, so only ~1.7 planes
per jet), *and* a further `k_t ≥ 1 GeV`, `z > 0.1` splitting inside a softer prong that
carries only `z · p_T` ≈ 10–30 GeV and sits inside an angular region smaller than its
parent's `ΔR`. A 1 GeV floor is a large fraction of such a subjet's available `k_t`,
which is why dropping the floor to 0.2 GeV multiplies `⟨n_sec⟩` by ~9.

**Concretely, before re-judging the physics:** `SoftDrop:ktFloor = 0.2` is the single
highest-leverage change (0.23 → 2.04); `pTHatMin = 500` or `R = 0.8` each give ~3×
for free. Note the first two rows also change what the *targets* are — a lower floor
admits more primary emissions too (⟨n_x⟩ 1.74 → 3.44), so that is a different
learning problem, not just a re-scoring of this one, and `max_emissions` support should
be re-checked via `check_multiplicity_support`.

Worked A/B: [`notebooks/aux_input_ab.ipynb`](../notebooks/aux_input_ab.ipynb).

## Design (recommended approach)

### Stage 1 — C++: compute and persist the aux columns

**1. Single groom predicate.** Factor the per-splitting cut out of `primaryLund`
(lund_io.cpp:19–21) into `lund_io.hpp` so the primary path and the new traversal
cannot drift:

```cpp
// lund_io.hpp
inline bool passesGroom(double Delta, double kt, double z, const GroomParams& g) {
  if (Delta <= 0.0 || kt <= 0.0 || z <= 0.0) return false;
  if (z <= g.z_cut * std::pow(Delta / g.R0, g.beta)) return false;  // SD boundary
  return kt >= g.kt_floor;                                          // perturbative floor
}
```

`primaryLund` is rewritten to call it (behavioural no-op; the existing
`test_lund_io` boundary tests are the guard).

**2. All-branch traversal.** New in `lund_io.hpp` / `lund_io.cpp`:

```cpp
struct JetAux {
  float mg = 0.f;               // pipeline-groomed jet mass (RSD-drop semantics)
  std::uint32_t n_primary = 0;  // passing primary splittings (consistency check)
  std::uint32_t n_all = 0;      // passing splittings over ALL branches
};
JetAux fullLundAux(const fastjet::PseudoJet& jet, const GroomParams& g);
```

Implementation: recluster the constituents with
`JetDefinition(cambridge_algorithm, JetDefinition::max_allowable_R)` (the same
C/A recluster `LundGenerator` performs internally), then recurse over
`has_parents(p1, p2)` with p1 the harder prong, using the Lund conventions of
Dreyer, Salam & Soyez (arXiv:1807.04758): Δ = ΔR(p1,p2), z = pT2/(pT1+pT2),
kt = pT2·Δ. At each splitting: if `passesGroom`, count it (into `n_primary` when
on the hardest-branch spine, always into `n_all`) and recurse into **both**
prongs; otherwise drop the softer prong (recursive-SD semantics, Dreyer et al.,
arXiv:1804.03657, extended by the pipeline's kt floor) and recurse into the
harder prong only. The recursion returns the kept 4-momentum; `mg` is its mass.
Document explicitly in the header: `mg` is the **pipeline-groomed** mass (the
same predicate as the persisted sequences, including the kt floor), not the
textbook z_cut-only Soft Drop mass — one grooming definition per file.

**3. Schema.** `LundWriter` (lund_writer.hpp:36–44, lund_writer.cpp:14–41) gains
two fields, filled from `fullLundAux(hadronJet, g)` in
`processEvent` (write_lund_rntuple.cpp:44–57):

```cpp
fXmg_   = model->MakeField<float>("x_mg");           // hadron-level, conditioning side only
fXnsec_ = model->MakeField<std::uint32_t>("x_nsec"); // n_all - n_primary
```

The parton side y is **not** extended — the target definition (groomed parton
primary sequence) is untouched. `read_lund_rntuple.cpp` prints the two new
fields when present (guard with the RNTuple descriptor so old files still read).
The toy source (write_lund_rntuple.cpp:60–86) exercises the path unchanged.

### Stage 2 — Python: aux registry, broadcast node features, config

**Key decision — broadcast, not signature widening.** Conditioning is threaded
everywhere as `(xf, nx)`: the `Encoder` contract (encoders/base.py:26–29), all
`PosteriorModel` methods, closure (closure.py:225–238), calibration
(calibration.py:24–29), MBR (mbr.py:345), serving (api.py:46–49). Appending the
aux scalars as **constant per-node columns of xf** reaches every consumer through
that existing plumbing with zero interface churn: the encoders already
parameterize their input width (`n_node_feat` constructor argument, so the
encoder modules themselves need **no diff**), and `describe_cells`,
`sample_batch`, MBR, closure and calibration all just carry the wider xf.
Broadcasting globals per point is standard in particle-cloud practice and acts as
feature-wise conditioning of the per-node embedding (cf. FiLM, Perez et al.,
arXiv:1709.07871).

*Known limitation:* jets with `nx == 0` (nothing survives grooming at hadron
level; possible on the RNTuple path since `processEvent` only skips when both
levels are empty) lose the aux signal — an empty xf has no rows to carry it.
`LundDataModule.setup` logs the affected fraction when aux is on. If that
fraction proves material, escalate to the signature-widening alternative
(`forward(xf, nx, aux=None)` + threading through every call site above) or fold
aux into the WP3 cross-attention conditioning; not built now.

**1. `features.py` — registry with fixed standardization** (no data-dependent
normalization: checkpoint portability):

```python
MG_EPS = 1e-3  # GeV; single-prong groomed jets have m_g == 0

AUX_FEATURES = {
    "ln_mg_pt": lambda j: math.log(max(float(j["x_mg"]), MG_EPS) / float(j["jet_pt"])),
    "nsec":     lambda j: math.log1p(float(j["x_nsec"])),
    "ln_pt":    lambda j: math.log(float(j["jet_pt"]) / 100.0),
}

def aux_vector(jet: dict, names) -> np.ndarray:
    """(n_aux,) float32; raises ValueError on missing/non-finite source fields
    (an old jets.root without the columns must fail loud, not train on NaNs)."""
```

**2. `data/rntuple.py`** — extend the jet dict (rntuple.py:35–47) via the
existing tolerant `scalar()` reads: `jet_pt` (default NaN), `x_mg` (default NaN),
`x_nsec` (default −1). Defaults are sentinels that `aux_vector` rejects.

**3. `data/dataset.py`** — `MatchedLundDataset(jets, geometry, aux_features=())`;
after `xf = node_features(*j["x"])` (dataset.py:20), when aux is on:

```python
aux = aux_vector(j, self.aux_features)                       # (n_aux,)
xf = np.concatenate([xf, np.broadcast_to(aux, (len(xf), len(aux)))], axis=1)
```

`collate` (dataset.py:47–56) infers the x width from the batch
(`Fx = batch[0]["xf"].shape[1]`, asserted uniform) instead of the `N_NODE_FEAT`
constant; `yf`/`yraw` stay at width 5/4 — the decoder side is untouched.

**4. `config.py`** — `aux_features: list[str] = field(default_factory=list)` on
all three encoder schemas (config.py:54–78); mirror into
`configs/encoder/{gru,lundnet,deepsets}.yaml`. CLI:
`encoder.aux_features='[ln_mg_pt,nsec,ln_pt]'`. Add the aux tuple to
`_fingerprint` (datamodule.py:24–34) so the §4 preprocessed-tensor cache cannot
serve a stale width.

**5. Model families** — the single per-family diff, at the three
`build_encoder(cfg.encoder, self.ctx_dim, N_NODE_FEAT)` sites
(ar_junipr.py:71, cinn.py:104, diffusion.py:55):

```python
n_in = N_NODE_FEAT + len(getattr(cfg.encoder, "aux_features", None) or [])  # getattr-
self.encoder_net = build_encoder(cfg.encoder, self.ctx_dim, n_in)           # tolerant:
self.aux_feature_names = tuple(getattr(cfg.encoder, "aux_features", None) or [])
```

(the `getattr` idiom matches `cell_label_smoothing`, ar_junipr.py:59, so old
checkpoint configs load as today). `datamodule.datasets()` (datamodule.py:90–93)
passes `aux_features` from the same config node.

**6. Serving** — `predict` (api.py:46–49) requires `x_seq["aux"]` (a name→value
dict) when `model.aux_feature_names` is non-empty, assembles it in configured
order through `aux_vector`, and broadcasts onto xf; missing keys raise. Response
metadata echoes the active aux names (additive, non-breaking).

**7. Synthetic path — fail loud, no proxies.** `synthetic_matched_dataset` has no
secondary planes; any proxy would be a function of x, i.e. redundant by
construction, and would fake exactly the information gain this plan exists to
measure. `aux_features != []` with `data.source=synthetic` raises in
`MatchedLundDataset`. Plumbing tests inject aux fields on fixture dicts instead;
the physics A/B runs on the PYTHIA RNTuple path only.

## Tests & exit criteria

- **C++ (`cpp/tests/test_lund_io.cpp`, target at CMakeLists.txt:86):**
  `fullLundAux(...).n_primary == primaryLund(...).lnkt.size()` on the two-prong
  fixtures across (z_cut, kt_floor) settings — the predicate-consistency
  guarantee; on a three-prong jet with a hard secondary splitting,
  `n_all > n_primary` and `mg > 0`; raising `z_cut` weakly decreases `n_all` and
  `mg`; fully-groomed jet gives `mg == 0`, `n_all == 0`.
- **Parity (the merged gate):** with `aux_features=[]` the `state_dict` keys and
  shapes, `log_prob`, and `scripts/verify_parity.py` output are bit-identical to
  main; `tests/test_parity.py` passes untouched. Old checkpoints load strictly
  (checkpoint config lacks the field → `getattr` default → same widths).
- **`tests/test_aux_features.py`:** registry guards (MG_EPS floor; ValueError on
  NaN / missing / `x_nsec == -1` sentinels); dataset broadcast width incl. the
  empty-x `(0, 5+n_aux)` case; collate width inference on mixed lengths; each
  family builds with the widened first Linear and trains one step; synthetic +
  aux raises; serving round-trip with and without aux.
- **Physics A/B (adoption gate, PYTHIA path, ≥3 seeds):** `gru` vs
  `gru + [ln_mg_pt, nsec, ln_pt]`. Adopt as a default preset only if (i) the
  held-out conditional NLL improves beyond the seed spread, (ii) closure
  multiplicity bias and the MBR perturbative-Lund risk do not degrade, (iii)
  SBC/PIT — additionally **stratified in aux bins** (WP2 hooks) — shows no new
  undercoverage, and (iv) the `experiment.generator_b` / fragmentation-reweight
  spread (eval/systematics.py) is re-measured and the change documented; a
  sharper posterior bought with a materially larger prior systematic is reported,
  not silently adopted.

## Non-goals

- **Secondary-plane sequences.** Persisting full trees (`LundWithSecondary` or a
  serialized recursive structure) and encoding them belongs to the full-tree
  LundNet / WP3 cross-attention work; this plan ships scalars only.
- **Ungroomed observables** (constituent multiplicity, ungroomed mass, girth):
  deliberately excluded — IRC-unsafe / UE- and heavy-ion-background-sensitive
  conditioning contradicts the grooming-first design; revisit, if ever, as a
  separate plan with its own systematics budget.
- **No decoder or target changes**; no learned normalization of aux (fixed affine
  transforms only); no second grooming definition for `mg`.

## References

Dreyer, Salam & Soyez, arXiv:1807.04758 · Dreyer et al. (RSD), arXiv:1804.03657 ·
Larkoski, Marzani, Soyez & Thaler, arXiv:1402.2657 · Dreyer, Soyez & Takacs,
arXiv:2112.09140 · Dreyer & Qu (LundNet), arXiv:2012.08526 · Perez et al. (FiLM),
arXiv:1709.07871
