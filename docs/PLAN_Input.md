# PLAN — Aux conditioning: groomed all-branch observables as encoder input

**Status:** proposed (not yet implemented)

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
