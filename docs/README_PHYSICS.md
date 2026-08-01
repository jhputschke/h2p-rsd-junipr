# The Physics of `h2p-rsd-junipr`

*Amortized per-jet posterior inference of the groomed parton-shower configuration
across hadronization.*

This document is the physics companion to [`README.md`](../README.md). It explains
**what** the code infers and **why** the inference is well-posed, working from the
methodological note `amortized_posterior_hadronization.md` and the background
surveys on parton showers and on hadronization/grooming. Wherever a physics idea
maps onto a concrete piece of the repository, the file or command is named in
`monospace`, so this doubles as a guided tour of the implementation.

---

## 1. The problem in one sentence

Given the hadron-level groomed Lund tree of a jet, infer — **jet by jet, as a
calibrated posterior** — the most likely groomed *parton-shower* configuration that
fragmented into it, with all probabilities defined by a fixed generator (PYTHIA 8.3;
Bierlich et al., *SciPost Phys. Codebases* (2022), arXiv:2203.11601).

Formally, with `x` the hadron-level tree and `y` the matched parton-level tree, the
target is the **per-jet hadronization-inversion posterior**

$$
p(y\mid x)\;\propto\;\underbrace{p(x\mid y)}_{\text{hadronization}}\;\underbrace{p(y)}_{\text{shower prior}},
$$

and the package learns a single amortized surrogate $q_\phi(y\mid x)\approx p(y\mid x)$
once, then evaluates it per jet. This is **amortized simulation-based inference /
neural posterior estimation** (Cranmer, Brehmer & Louppe, *PNAS* **117** (2020)
30055, arXiv:1911.01429; Papamakarios & Murray, *NeurIPS* (2016), arXiv:1605.06376;
Greenberg, Nonnenmacher & Macke, *ICML* (2019), arXiv:1905.07488). In the code the
amortized model is any `PosteriorModel` (`src/h2p_rsd_junipr/models/`); its
`log_prob` *is* $\log q_\phi(y\mid x)$.

---

## 2. QCD background: from the hard scale to the detector

A hadronic final state is built by a tower of approximations matched across scales
(`parton_showers_survey.md`):

1. **Hard scattering** — fixed-order perturbation theory at scale $Q$.
2. **Parton shower** — the Markovian, exclusive, probabilistic solution of the
   collinear (DGLAP) evolution, dressed by the **Sudakov form factor** (the
   no-emission survival probability) and constrained by **colour coherence**: soft
   gluons at large angle cannot resolve a jet's internal colour and add coherently,
   confining radiation to angular-ordered cones (Marchesini & Webber 1984; Catani,
   Marchesini & Webber, *Nucl. Phys. B* **349** (1991) 635). PYTHIA implements a
   $p_T$-ordered dipole shower; HERWIG an angular-ordered one — a difference that
   resurfaces as the dominant *systematic* in §8.
3. **Hadronization** — below $Q_0\sim1$ GeV confinement turns coloured partons into
   colour-singlet hadrons. This is **non-perturbative and model-dependent**.
4. **Measurement** — jets are built from the resulting stable hadrons.

The boundary between steps 2 and 3 is exactly where this repository operates: `y`
lives at the *end of the shower*, `x` after *hadronization*.

### 2.1 The Lund string model (the forward simulator)

PYTHIA hadronizes via the **Lund string** model (Andersson, Gustafson, Ingelman &
Sjöstrand, *Phys. Rept.* **97** (1983) 31): the colour field between a separating
$q\bar q$ pair is a relativistic flux tube of tension $\kappa\approx1$ GeV/fm that
breaks by Schwinger-like $q\bar q$ tunnelling, with longitudinal momentum sharing
drawn from the symmetric Lund fragmentation function

$$
f(z)\;\propto\;\frac{(1-z)^a}{z}\,\exp\!\Big(-\frac{b\,m_\perp^2}{z}\Big),\qquad m_\perp^2=m_h^2+p_\perp^2 .
$$

This is a **stochastic, many-to-one** operation (§3 below), which is precisely why
the inverse is a posterior, not a function. HERWIG instead uses the **cluster**
model (Webber, *Nucl. Phys. B* **238** (1984) 492) — same role, different physics,
hence a handle on model dependence.

### 2.2 Hadronization is a power correction

Because the relevant scale is $\Lambda_{\rm QCD}$, hadronization shifts perturbative
predictions by terms suppressed by the hard scale,

$$
\langle V\rangle=\langle V\rangle_{\rm pert}+c_V\,\frac{\mathcal A}{Q}+\mathcal O\!\Big(\frac1{Q^2}\Big),
$$

(the dispersive picture; Dokshitzer & Webber, arXiv:hep-ph/9504219). For jets the
shift is localised at the jet boundary — the mean jet-$p_T$ shift scales as
$\sim-\Lambda_{\rm QCD}/R$ (Dasgupta, Magnea & Salam, *JHEP* **02** (2008) 055,
arXiv:0712.3014). Crucially, **jet mass and substructure are *more* NP-sensitive than
the jet energy**, because they weight soft, wide-angle radiation — exactly the region
hadronization and the underlying event dominate. That sensitivity is the problem
grooming was invented to solve.

### 2.3 Local parton–hadron duality: the residual is a *kernel*

Power counting says hadronization is small; **local parton–hadron duality** says more
than that — it says it is *local*. The hadron configuration tracks the parton
configuration up to a short-distance smearing, so hadron-level distributions follow
parton-level ones bin by bin up to power-suppressed corrections (Azimov, Dokshitzer,
Khoze & Troyan, *Z. Phys. C* **27** (1985) 65). Organised as a convolution, that
smearing is the **shape function** (Korchemsky & Sterman, arXiv:hep-ph/9902341).

Three properties make the smearing *structured* rather than generic noise, and they
are what §8's edit transducer is built on:

- the scale runs as $\Lambda_{\rm eff}/k_t$ — a shift of order $\Lambda_{\rm eff}$ GeV
  in $k_t$ is a width $\Lambda_{\rm eff}/k_t$ in $\ln k_t$, so the width is a
  *predictable function of a node's own coordinates*, tight on hard/early emissions and
  loose near the floor;
- for a groomed jet the NP correction scales with the catchment geometry, hence with
  $R_g$ and $z_g$ (Hoang, Mateu, Pathak, Stewart et al., arXiv:1906.11843) —
  heteroscedasticity with a physical argument behind it;
- prong multiplicity proxies colour charge through Casimir scaling of the emission
  density (Dreyer, Salam & Soyez, arXiv:1807.04758).

Births and deaths are equally structured: hadron-level nodes below the perturbative
floor have no parton image, and parton nodes whose hadron image migrated across the
grooming boundary have no hadron anchor. In `cpp/test_data/jets.root` (PYTHIA 8.3,
$z_{\rm cut}=0.1$, 1 GeV floor) **6.9% of jets have no hadron-level primary emission and
16.0% no parton-level one** (`tests/test_empty_sequences.py`) — this is a large effect,
not a tail.

Every family in §8 *except* the edit transducer ignores all of this: it conditions on
`x` only through an encoder, so the decoder must relearn from data that $y\approx x$
wherever hadronization is weak. The edit transducer instead writes the smearing down as
$\sigma=\sigma_0+\Lambda_{\rm eff}\,e^{-\ln k_t}$ with $\Lambda_{\rm eff}$ **learnable**,
which turns a modelling assumption into a *measurement*: see §8 and §9.

---

## 3. Why the map is a posterior, not a function

Lund-string fragmentation is stochastic and many-to-one: one groomed parton tree `y`
hadronizes into a *distribution* of hadron trees `x`, and many `y` can yield the same
`x`. There is **no deterministic inverse** and no unique "the parton configuration."
The well-defined target is the posterior $p(y\mid x)$, and "most likely
configuration" means the **MAP estimate** $\hat y_{\rm MAP}(x)=\arg\max_y q_\phi(y\mid x)$.

Two consequences are built into the code:

- **Report the width, not just the mode.** Every jet gets *both* a MAP/beam estimate
  and a posterior summary (mean, **median**, 68% credible region, multiplicity
  distribution) — `eval.closure.print_point_estimate`. In high dimensions the mode can
  be unrepresentative (`amortized_posterior_hadronization.md` §6), so a regressor that
  learns only the conditional mean is the wrong tool: it blurs precisely where the
  inverse is ambiguous. This package models a density, never a point regression.
- **The MAP is floored away from the empty tree.** The joint mode $\hat y_{\rm MAP}$ of
  a discrete autoregressive posterior is **length-biased**: every emission pays the
  Lund-cell head's categorical entropy while "stop" costs a roughly fixed amount, so
  for high-multiplicity jets the single most-probable explicit tree scores *below* the
  empty tree — and the un-floored argmax collapses to **0 splittings**, which is
  unphysical (a groomed jet has $\ge 1$ primary splitting). The decoder enforces
  `decode.min_emissions` (default 1) — a *constrained MAP under a minimum-emission
  floor* — and an optional GNMT length penalty `decode.length_penalty`
  ($\text{score}/\text{len}^{\alpha}$) to counter the brevity bias; cINN/diffusion clamp
  their multiplicity head the same way. A **learned, per-jet** generalization of that
  floor is `decode.length_floor_quantile` ($\alpha$, default 0 = off): the model already
  learns a per-jet length belief $P(n\mid x)$ (an explicit categorical head for
  cINN/diffusion; the empirical multiplicity of posterior draws for the autoregressive
  model), which — unlike the joint argmax — is *unbiased* in length. Flooring the MAP at
  a low quantile of it, $\hat n=\max(\texttt{min\_emissions},\,\lfloor
  Q_\alpha(P(n\mid x))\rfloor)$, transfers that belief into the point estimate and cuts
  the residual under-count; the floor only ever *raises* the bound (so $n\ge 1$ is
  preserved), $\alpha\to0$ reproduces the hard floor exactly, and $\alpha\to$ median
  approaches a **length-conditioned MAP** at that quantile. For a *count*, still prefer
  the **posterior median**: the MAP is the wrong summary for multiplicity. Quantified in
  `notebooks/inference_demo.ipynb` §6a and `scripts/probe_map_collapse.py`.
- **Length as a first-class factor — the multiplicity head.** The learned floor above is
  a *decoding* trick built on $P(n\mid x)$; `model=ar_junipr_v3` (opt-in
  `use_multiplicity_head`, off by default) promotes it to **model structure**, factorizing
  $$
  q_\phi(y\mid x) = q_\phi(N\mid x)\,\; q_\phi(y\mid N,x)
  $$
  with a dedicated categorical multiplicity head $q_\phi(N\mid x)$ (the same head cINN and
  diffusion carry) in place of the autoregressive continue/stop product. The kinematics
  $q_\phi(y\mid N,x)$ are the unchanged JUNIPR cell/coordinate heads, run for exactly $N$
  steps. Two consequences: **(i)** the length is now a calibrated, low-dimensional marginal,
  so the argmax over $N$ is over a well-behaved categorical rather than an implicit product of
  continue-probabilities — the short-sequence MAP degeneracy is killed *at its source*, not
  clamped, and $P(n\mid x)$ is read **exactly** from the head; **(ii)** ancestral draws then
  inherit a calibrated multiplicity marginal, giving a clean handle on the sampler's
  exposure bias (below). This mirrors the first high-precision generative-unfolding framework
  for jet substructure, a staged pipeline whose first stage unfolds the multiplicity with the
  kinematics generated conditional on it (arXiv:2510.19906). Because it is a
  bool switch, `ar_junipr_v2` stays bit-for-bit identical when off. See
  [`PLAN_MultHead.md`](PLAN_MultHead.md).
- **Length anchored at $|x|$ — the edit transducer.** The multiplicity head above makes
  the length *calibrated*; `model=edit_v1` makes it **structural**. If every parton node
  is a kept (smeared) hadron node, an insertion, or a deleted hadron node, then
  $$
  n_y = n_x - \#\text{del} + \#\text{ins},
  $$
  so the multiplicity is pinned to $|x|$ and the open-ended continue/stop mechanism —
  the seat of the brevity bias every knob above exists to patch — is *removed*, not
  recalibrated. Two things fall out for free. Marginalising the coordinates out of the
  same dynamic program leaves a purely structural recursion whose terminal value is
  $q_\phi(N\mid x)$ **exactly, with no extra parameters** — what `ar_junipr_v3` learns
  with a head, here derived and explicitly conditioned on $|x|$. And the **empty parton
  tree** (16.0% of jets, §2.3) is simply the delete-all path, so it is represented
  natively rather than reached by a decode-layer threshold: on the PYTHIA test file the
  predicted empty rate is 0.173 against a truth of 0.160 with `decode.empty_threshold`
  **off**. MAP collapse to $n=0$ is structurally suppressed too — it now requires
  ADVANCE at all $n_x$ columns *and* a STOP, where the autoregressive families need one
  stop draw. See §8 and [`PLAN_EditTransducer.md`](PLAN_EditTransducer.md).
- **MBR — a floor-free alternative decision rule.** The floors above *mask* the
  length bias of the joint mode; **minimum Bayes risk** removes it at the source by
  changing the decision rule, not the density (`decode.point_estimator="mbr"`,
  `inference/mbr.py`). From the same posterior draws the credible bands already use,
  MBR returns the drawn tree of least *expected* distance to the posterior (Kumar &
  Byrne, HLT-NAACL 2004; Eikema & Aziz, arXiv:2005.10283, who show the mode is
  essentially arbitrary while the distribution is faithful — exactly the SBC picture):
  $$
  \hat y_{\rm MBR}=\arg\min_{h\in\mathcal C}\ \frac1K\sum_{k=1}^{K} d\big(h,y^{(k)}\big),\qquad \mathcal C\subseteq\{y^{(k)}\}\sim q_\phi(\cdot\mid x).
  $$
  Two properties make it the principled point estimate here. **(i) Alignment-free**:
  the loss $d$ is a distance between two *radiation patterns*, not between paired
  nodes, so it respects the hard **no per-node $x\leftrightarrow y$ correspondence**
  constraint (§5) — only jet-level pairing is used. **(ii) The empty tree is never
  selected, with no floor**: an empty cloud has large expected distance to typical
  non-empty draws (it pays the full mass-imbalance penalty), so it can never minimise
  the risk. On a trained checkpoint, floor-free (`min_emissions=0`) MAP collapses to
  $n=0$ for a large fraction of jets while MBR stays at **0%** — the brevity bias is
  removed *structurally* rather than clamped, so `min_emissions` is unnecessary for
  this estimator. (If a jet's draws are *genuinely* mostly empty — honest high
  uncertainty — MBR picks a short tree; that is correct, unlike a floor that would
  manufacture emissions.) The winner is a genuine drawn tree, reported as the same
  `LundPointEstimate` with an added decision-theoretic `.risk` (**not** a likelihood).

  The metric $d$ is the **perturbative-Lund Energy Mover's Distance** (Komiske,
  Metodiev & Thaler, *PRL* **123** (2019) 041801, arXiv:1902.02346): each draw becomes
  a weighted point cloud in the Lund plane — one point per emission *above a $\ln k_t$
  cut* (weight $k_t$, IRC-safe) — and clouds are compared by optimal transport,
  $d(y,y')=\min_{f\ge0}\sum_{ij}f_{ij}\lVert p_i-p'_j\rVert^{\beta}+R\,|\sum_i w_i-\sum_j w'_j|$.
  The **perturbative restriction enters as the support of the ground metric**
  (`decode.mbr_lnkt_cut`, inheriting the geometry/grooming $\ln k_t$ floor of §4 — no
  second physics constant), so hadronization-region jitter cannot dominate the risk;
  the imbalance radius `R` sets the length↔kinematics trade-off (large $R$ tracks the
  count) and defaults to $\approx$ the Lund-plane diameter; $\beta=1$ is the
  1-Wasserstein EMD. The OT solve has **two interchangeable backends**
  (`decode.mbr_backend`): a self-contained POT augmented-cost form (`pot`, default,
  no physics package) and the reference `energyflow` implementation. They implement
  the same object and agree on the **argmin**; because EnergyFlow normalises ground
  distances by $R$, its numeric value equals the `pot` value $/R$ (for $\beta=1$) — so
  pick **one backend per analysis** for comparable `risk` numbers, and `mbr_norm=True`
  (unit-sum weights) *removes* the imbalance term and its empty-tree guarantee (off by
  default). Reproducing the **KMT collider-event EMD verbatim** (hadronic
  $(p_T,y,\phi)$ coordinates, their $R,\beta,\mathrm{norm},\texttt{periodic\_phi}$) is
  a configuration the user dials in through the `mbr_*` knobs, not pinned to the paper.
  Quantified in `notebooks/inference_demo.ipynb` §6. One caveat inherited from the sampler:
  MBR candidates are ancestral draws, so the candidate pool carries the posterior's
  marginal-multiplicity bias. The un-normalized EMD partially self-corrects (mass imbalance is
  penalized), but the residual is empirical — the closure suite reports the signed
  multiplicity bias of the MBR estimate **stratified by true $N$** to expose it. If it
  survives, `decode.mbr_resample_to_qn` reweights the candidate pool to the calibrated
  $q_\phi(N\mid x)$ marginal — a **decoding-layer** correction that leaves the likelihood (and
  thus any likelihood-ratio analysis) intact, unlike minimum-risk / sequence fine-tuning.
  It is most effective with a calibrated head (`ar_junipr_v3`, cINN, diffusion).
- **A direct conditional MLE.** A simpler MAP-on-a-tractable-likelihood precedent is
  the Ginkgo / Quantum-Trellis line (arXiv:2105.10512, arXiv:2112.12795); here the
  likelihood is the learned $q_\phi$.

---

## 4. Grooming makes the inverse tractable

Grooming reclusters a jet and discards soft, wide-angle constituents — the radiation
most contaminated by hadronization, underlying event and pileup — while keeping the
hard, collinear core (`hadronization_grooming_survey.md` §4). The analytically
cleanest groomer is **Soft Drop** (Larkoski, Marzani, Soyez & Thaler, *JHEP* **05**
(2014) 146, arXiv:1402.2657): recluster with Cambridge/Aachen, walk back through the
tree, and at each step test

$$
\frac{\min(p_{T,1},p_{T,2})}{p_{T,1}+p_{T,2}}\;>\;z_{\rm cut}\Big(\frac{\Delta R_{12}}{R_0}\Big)^{\beta}.
$$

If the softer branch fails it is dropped and the recursion continues into the harder
branch (Recursive Soft Drop; Dreyer, Necib, Soyez & Thaler, arXiv:1804.03657).
$z_{\rm cut}$ sets the soft threshold and $\beta$ the angular reach: **$\beta=0$**
(mMDT) removes soft radiation at all angles and is the most NP-robust working point,
for which the leading hadronization correction to the groomed observable is
parametrically suppressed (Dasgupta et al., arXiv:1307.0007; Frye et al.,
arXiv:1603.09338; Hoang et al., arXiv:1906.11843).

**In the code.** This is exactly `primaryLund` in `cpp/src/lund_io.cpp`:

```cpp
if (z <= g.z_cut * std::pow(Delta / g.R0, g.beta)) continue;  // Soft Drop boundary
if (kt < g.kt_floor)                               continue;  // perturbative floor
```

The defaults (`GroomParams`: `z_cut=0.1, beta=0, R0=1, kt_floor=1` GeV) are the
CMS/ATLAS standard mMDT point. The $\ln k_t$ **floor** keeps only the
hadronization-robust, perturbative band of the Lund plane — without it the
soft/wide-angle corner yields a broad, near-meaningless posterior
(`amortized_posterior_hadronization.md` §6; the primary-Lund-plane density that the
posterior should respect there is itself calculable: Lifson, Salam & Soyez, *JHEP*
**10** (2020) 170, arXiv:2007.06578).

**The floor is not a Soft Drop parameter.** Soft Drop is $z_{\rm cut}$, $\beta$, $R_0$;
the $k_t$ floor is a second condition **AND-ed** with it in one per-splitting predicate
(`passesGroom`), evaluated *during* the declustering rather than as a pass over its
output. Two consequences worth internalizing, because they differ:

- On the persisted sequences the floor only decides which emissions are *recorded* —
  the declustering chain always follows the harder prong, so it never truncates the
  walk. Raising the floor after the fact is therefore an exact set operation.
- On the all-branch traversal (`fullLundAux`, which produces `x_mg` / `x_ptg`) a
  failing splitting **discards the softer prong**, so the floor changes the surviving
  momentum. Those are *pipeline*-groomed, not textbook $z_{\rm cut}$-only Soft Drop
  quantities, and they move a lot: measured on 3220 jets from identical events, floor
  $1.0 \to 0.2$ GeV shifts median $p_{T,g}/p_T$ from 0.394 to 0.686.

**Asymmetric floors.** Because the aux scalars are *conditioning* inputs rather than
targets, `SoftDrop:ktFloorSec` may groom off-spine branches below the spine floor —
recovering the secondary-plane information the primary sequence structurally cannot
carry, with the inputs and targets left bit-for-bit unchanged. See
[`PLAN_Input.md`](PLAN_Input.md) ("Asymmetric k_t floors") and
[`cpp/cards/pp_dijet_asym_floor.cmnd`](../cpp/cards/pp_dijet_asym_floor.cmnd).

---

## 5. The Lund-plane representation used here

Each groomed tree is its **primary Lund sequence**: an ordered list of declusterings,
each a point in the Lund plane (Dreyer, Salam & Soyez, *JHEP* **12** (2018) 064,
arXiv:1807.04758) with coordinates

$$
\big(\ln \tfrac1{\Delta R},\ \ln k_t,\ \ln z,\ \psi\big),
$$

— inverse angle (collinear ⇒ large), transverse momentum (the perturbative ordering
variable; high $k_t$ ⇒ hard/early), momentum fraction, and azimuth of the softer
prong about the harder. Angular ordering from colour coherence (§2) makes the
sequence naturally ordered in $\ln 1/\Delta R$.

**In the code.** `src/h2p_rsd_junipr/geometry.py` discretises $(\ln 1/\Delta R,\ \ln
k_t)$ into an $N_{\rm bins}\times N_{\rm bins}$ grid of Lund **cells** (`to_cell` /
`cell_center`), and `features.py` carries the continuous tuple; $\psi$ is periodic so
the model treats it with a von Mises density (`distributions.vonmises_logpdf`), never
a Gaussian that would be wrong at the $\pm\pi$ wrap. The grid ranges, $N_{\rm bins}$,
and the floor are all config fields (`configs/geometry/default.yaml`), so the
discretisation is a knob, not a hard-coded constant.

The **data contract** (one entry per jet, written by `write_lund_rntuple`): `event`,
`weight`, jet kinematics, the grooming provenance $(z_{\rm cut},\beta,k_t^{\rm
floor},\text{generator})$, and the two **node-unaligned** jagged sequences `x_*` and
`y_*`. There is by design **no per-node $x\leftrightarrow y$ correspondence**.

---

## 6. Matched $(x,y)$ pairs from the generator

For each event (implemented in `cpp/apps/pythia_driver.cpp`):

1. **Jet finding** — cluster final-state hadrons with anti-$k_t$ (Cacciari, Salam &
   Soyez, arXiv:0802.1189) at radius $R$ via FastJet (arXiv:1111.6097).
2. **Hadron tree `x`** — recluster jet constituents with C/A and apply the Soft-Drop
   boundary + $k_t$ floor (`primaryLund`).
3. **Parton tree `y`** — take the **pre-hadronization shower partons** (the string
   endpoints, read from the PYTHIA event record as status codes 71–79, *not* a
   final-state dump), match them to the same jet, and apply the *identical* anti-$k_t$
   + C/A + Soft-Drop pipeline.
4. **Same-jet pairing** — `getMatchedHadronPartonJets` pairs the two levels
   geometrically (greedy, hardest-first, one-to-one within a cone). The pairing is
   exceptionally well conditioned because the shower is *shared upstream*: the only
   stochastic stage separating `x` and `y` is hadronization, an
   $\mathcal O(\Lambda_{\rm QCD}/Q)$ power correction on the jet axis.

```cpp
pythia.readString("PartonLevel:MPI = off");  // pure hadronization study
...
if (prt.isParton() && as >= 71 && as <= 79)  // pre-hadronization (string endpoints)
```

**MPI is disabled** so the underlying event does not smear `x` relative to `y` — a
pure hadronization study (`amortized_posterior_hadronization.md` §3); fold MPI into
the forward model if you want it. Because node-level correspondence is unavailable —
the same obstacle the HOMER hadronization-fitting programme confronts (Bierlich et
al., arXiv:2410.06342; Assi et al., arXiv:2503.05667) — the training objective is a
**jet-level** conditional likelihood, never a per-node regression. The synthetic
stand-in (`data/synthetic.py`) mimics this forward map (kt-dependent smearing + soft
migration) so the pipeline is testable without a generator.

**Unobservable is not the same as absent.** The correspondence exists in the physics
(§2.3) — it is simply not *recorded*, and no amount of generator bookkeeping would make
it a target one could regress against without committing to a matching convention that
is itself unmeasurable. The edit transducer (§8) therefore treats the alignment as a
**latent variable and marginalises it**: every way of reading `y` as a smeared, edited
copy of `x` is summed over, exactly, by dynamic programming, and the likelihood is still
the jet-level $\log q_\phi(y\mid x)$ above. Nothing is supervised per node, so the
constraint in §5 is respected rather than circumvented. What comes back afterwards is a
*posterior over alignments* — the forward–backward responsibilities $\gamma(i,j)$,
reported by `eval` as `frac_anchored` / `insert_rate` / `delete_rate` — an emergent
matching, obtained without ever having been given one.

---

## 7. The amortized objective

Train $q_\phi$ by conditional maximum likelihood on the matched pairs,

$$
\phi^\star=\arg\min_\phi\;\mathbb E_{(x,y)\sim\text{gen}}\big[-\log q_\phi(y\mid x)\big],
$$

whose minimiser at infinite data and capacity is the true posterior $p(y\mid x)$
(Papamakarios & Murray, arXiv:1605.06376; Cranmer et al., arXiv:1911.01429). This is
per-event probabilistic unfolding to parton level — the role conditional generative
networks were shown to play by the cINN of Bellagente et al. (*SciPost Phys.* **9**
(2020) 074, arXiv:2006.06685).

**In the code.** The per-jet weighted NLL in `train/trainer.py` is exactly this
objective, model-agnostic:

```python
nll  = -self.model.log_prob(batch)                 # (B,)  -log q_phi(y|x)
loss = (batch["w"] * nll).sum() / batch["w"].sum()
```

---

## 8. Architectures (one contract, five families)

All families expose `log_prob` / `sample` / `map_estimate` (`models/base.py`). The
first four differ in *how* they model $q_\phi(y\mid x)$; the fifth differs in how it
**factorizes** it:

- **§5.1 Conditional autoregressive RSD-JUNIPR** (`models/ar_junipr.py`, the
  recommended and verified model). An encoder $e(x)$ over the hadron tree (bi-GRU,
  or a LundNet/EdgeConv graph net; Dreyer & Qu, arXiv:2012.08526) feeds a three-head
  autoregressive decoder over `y`, extending the unsupervised JUNIPR density
  (Andreassen et al., *EPJC* **79** (2019) 102, arXiv:1804.09720; binary JUNIPR,
  *PRL* **123** (2019) 182001):
  $$
  q_\phi(y\mid x)=\Big[\textstyle\prod_t P_{\rm cont}(h_t,e)\,P_{\rm split}(c_t\mid h_t,e)\Big]\,P_{\rm cont}^{\rm stop},
  $$
  with **v2** adding a continuous within-cell coordinate head (truncated normals on
  $\ln 1/\Delta R,\ln k_t$, a normal on $\ln z$, von Mises on $\psi$). **MAP** is beam
  search over the decoded tree (floored at `decode.min_emissions`, with an optional
  `decode.length_penalty`); the **posterior** is ancestral sampling. Variable
  multiplicity is handled natively; only jet-level pairing is needed; the likelihood
  is explicit. **v3** (`use_multiplicity_head`) replaces the continue/stop product
  $\prod_t P_{\rm cont}\,P_{\rm cont}^{\rm stop}$ with an explicit categorical
  $q_\phi(N\mid x)$, giving the first-class factorization $q_\phi(y\mid x)=q_\phi(N\mid
  x)\,q_\phi(y\mid N,x)$ (§3, "Length as a first-class factor") while keeping the same
  cell/coordinate heads.
- **§5.2 Conditional normalizing flow / cINN** (`models/cinn.py`) — exact density,
  trivial sampling; variable multiplicity via a multiplicity head + structured latent
  (Bellagente et al., arXiv:2006.06685; Backes et al., arXiv:2212.08674).
- **§5.3 Conditional diffusion / Schrödinger bridge** (`models/diffusion.py`) — the
  most expressive option (arXiv:2404.18807). Its `log_prob` is a denoising-score-matching
  **surrogate**, not a density (`exact_likelihood=False`), so its NLLs are comparable
  only within the family.
- **§5.4 Conditional flow matching** (`models/cfm.py`) — the same cINN factorization with
  the coordinate density given by a regressed vector field (Lipman et al.,
  arXiv:2210.02747; FMPE, arXiv:2305.17161), and an **exact** probability-flow-ODE
  likelihood (arXiv:2011.13456) with the 4-dimensional divergence computed exactly. The
  exact-likelihood member of the continuous-time family.
- **§5.5 Edit transducer** (`models/edit.py`, `models/edit_dp.py`) — the only family that
  does not generate `y` from scratch. Everything above conditions on `x` through the
  encoder alone, so the decoder must *relearn* that $y\approx x$ wherever hadronization
  is weak; here the hadron tree is the **anchor** of the parton tree, and the physics of
  §2.3 goes into the factorization rather than being left for the network to discover:
  $$
  \text{state }(i,j):\quad
  i<n_x:\ \{\text{ADVANCE},\text{EMIT}\}\qquad
  i=n_x:\ \{\text{STOP},\text{EMIT}\}
  $$
  $$
  \text{EMIT}:\quad y_{j+1}\sim p_{\rm anch}\,f_{\rm shift}(\cdot\mid x_i)\;+\;(1-p_{\rm anch})\,f_{\rm free}(\cdot).
  $$
  An ADVANCE with no anchored emit at that column is a **deletion**, an anchored emit a
  **kept, smeared** node, a free emit an **insertion**. The alignment is latent and
  summed by an $\mathcal O(n_x n_y)$ forward recursion (§6) — this is the RNN-T lattice
  (Graves, arXiv:1211.3711; CTC, Graves et al., *ICML* 2006), so $\sum_y q_\phi(y\mid
  x)=1$ holds *by construction* and the exact-likelihood claim is structural rather than
  asserted. The edit-based decoding literature (Insertion Transformer, arXiv:1902.03249;
  Levenshtein Transformer, arXiv:1905.11006) resorts to heuristic surrogates because its
  lattices are enormous; with $n_x,n_y\lesssim25$ ours is exact, cheap and fully
  differentiable — the lattice size is why this family is attractive *here* and not in
  NLP.

  The physics enters twice. The smearing width is the **shape-function form**
  $\sigma=\sigma_0+\Lambda_{\rm eff}e^{-\ln k_t}$ with $(\sigma_0,\Lambda_{\rm eff})$
  learnable, so the learned kernel is directly confrontable with the expectation of §2.3
  instead of being an opaque MLP output — and $\Lambda_{\rm eff}$ comes out **in GeV**
  (`model.physics_width=false` swaps in the free-MLP ablation). And the length is
  anchored at $|x|$ with an exact $q_\phi(N\mid x)$ (§3). Stage 2 (`edit_v2`) adds a
  prediction network over the emitted prefix for recoil correlation among the `y` nodes;
  the point estimator of choice is MBR, since the joint mode of a variable-dimension
  density is not a useful summary here. See
  [`PLAN_EditTransducer.md`](PLAN_EditTransducer.md).

  Distribution-level analogues exist — MC-derived bin-by-bin hadron→parton corrections in
  Lund-plane measurements (cf. ATLAS, arXiv:2004.03540) and staged generative unfolding
  that produces multiplicity first (arXiv:2510.19906). This is the **per-jet,
  probabilistic** generalization of the former, and it gets the latter's staging for free.

In this repo §5.1 is production-grade; §5.2–§5.5 share the contract and pass the smoke
train. §5.5 is the only one that changes the *factorization* rather than the density
class, which is why it is the one with a falsifiable physics claim attached (§9).

---

## 9. Validation — mandatory, not optional

The output is only trustworthy after these checks (`eval/`,
`amortized_posterior_hadronization.md` §6):

- **Closure** (`eval.closure`) — on held-out generator data, $\hat y_{\rm MAP}$
  recovers the true `y` and posterior draws bracket it, measured with
  *node-alignment-free* observables (leading-emission Lund distance, multiplicity
  bias), since no per-node correspondence exists.
- **Posterior calibration** (`eval.calibration`) — per-event posteriors from
  conditional generators are **not automatically calibrated**; the original cINN
  unfolding came out "too narrow" (arXiv:2006.06685). Coverage / PIT / simulation-
  based calibration (Talts et al., arXiv:1804.06788) gate "trustworthy."
- **Generator dependence — the dominant systematic** (`eval.systematics`). Trained on
  PYTHIA, the model returns the most likely *PYTHIA* configuration and transports
  PYTHIA's prior $p(y)$ and forward model $p(x\mid y)$. Quantify by retraining on a
  **cluster-model** generator (HERWIG 7; Bellm et al., *EPJC* **76** (2016) 196,
  arXiv:1512.01178) and on alternative string tunes; the **inter-model spread is the
  systematic** and must be quoted (a parallel `herwig_driver` emitting the identical
  schema is the intended source; `configs/experiment/pythia_vs_herwig.yaml`).
- **Clustering/grooming migration** — even absent hadronization, clustering hadrons
  vs. partons can select slightly different groomed trees; closure exposes this small
  residual on top of hadronization.
- **Stay perturbative** — keep the $\ln k_t$ floor.
- **Is the smearing actually a $\Lambda_{\rm eff}/k_t$ kernel?** (edit transducer only.)
  Unlike everything above, this one can falsify a *physics* premise rather than a fit.
  Bin the residuals $y_t-x_i$ weighted by the alignment posterior $\gamma(i,j)$ in $\ln
  k_t$ (`model.alignment_posterior`) and fit $\sigma=\sigma_0+\Lambda_{\rm eff}e^{-\ln
  k_t}$. **If the widths come out flat in $k_t$, local parton–hadron duality is not what
  is organising this data** and the family's inductive bias is wrong. Run it with
  `model.physics_width=false`, whose widths are a free MLP output never told the
  functional form — otherwise you are reading back the parametrization you imposed. On
  `cpp/test_data/jets.root` (6-epoch fit) the ablation gives $\Lambda_{\rm eff}=1.29$ GeV
  at $R^2=1.000$ for $\ln k_t$: $\mathcal O(1$ GeV$)$, i.e. the shape-function scale, so
  the premise holds on that sample. It is **sample-dependent** — re-run per selection.

---

## 10. The physics knobs are config, not constants

Every physics choice is a versioned config field (`configs/`), never hard-coded:

| Physics | Config | Default |
|---|---|---|
| Lund-plane ranges, grid $N_{\rm bins}$ | `geometry.*` | $(0,6)^2$, 10 |
| Soft Drop $z_{\rm cut}$, $\beta$, $R_0$, $k_t$ floor | `GroomParams` (C++) / RNTuple provenance | 0.1, 0, 1, 1 GeV |
| generator / tune (systematic) | `experiment.generator_b` | — |
| encoder over $x$ (gru / lundnet / deepsets) | `encoder.*` | gru |
| posterior family (§5.1–§5.5) | `model=…` | ar_junipr_v2 |
| point estimator (MAP vs MBR) + EMD metric | `decode.point_estimator`, `decode.mbr_*` | map, pot |
| smearing kernel: shape-function form vs free MLP (§5.5) | `model.physics_width` | true |

```bash
# train on PYTHIA, then quote the HERWIG spread as the systematic
h2p-rsd-junipr generate 1000000 jets_pythia.root 1
h2p-rsd-junipr train data=rntuple data.path=jets_pythia.root model=ar_junipr_v2
h2p-rsd-junipr eval runs/<id>/best.ckpt experiment=pythia_vs_herwig
```

---

## 11. Caveats worth keeping in view

- The learned posterior is **generator-conditional**: it is the most likely
  configuration *under the chosen generator and tune*, not a generator-independent
  truth. The PYTHIA-vs-HERWIG spread is the honest uncertainty.
- The discretised likelihood is **cell-size dependent**; $N_{\rm bins}$ and the
  within-cell continuous head are locked together in the model builder so they cannot
  drift apart.
- The MAP can be unrepresentative in high dimensions — always read it alongside the
  posterior summary.
- **The edit transducer's alignments are monotone**, so two nearby nodes that swap order
  between levels cost a delete+insert pair: representable, but statistically
  inefficient. Angular ordering (§5) is what makes this a mild assumption rather than a
  wrong one; audit it with the crossing-pair count in sampled alignments.
- **Anchored vs free emissions are only weakly identifiable.** The two mixture
  components can trade off — a smeared copy and a fresh draw at the same place explain
  the data equally well — which is one reason the width is pinned to a physics form
  rather than left free. Watch `frac_anchored` in the closure output; it is not yet
  converged in the short fits quoted above.
- Both stages report `supports_coordinate_pit = False`: the exact prefix-conditional
  coordinate CDF is available from the same recursion but has not been landed, so the
  per-coordinate PIT panel is silent for this family.

---

## 12. Consolidated references

**Generator, jets, grooming, Lund plane.** PYTHIA 8.3, Bierlich et al., *SciPost
Phys. Codebases* (2022), arXiv:2203.11601 · HERWIG 7, Bellm et al., *EPJC* **76**
(2016) 196, arXiv:1512.01178 · anti-$k_t$, Cacciari, Salam & Soyez, *JHEP* **04**
(2008) 063, arXiv:0802.1189; FastJet, *EPJC* **72** (2012) 1896, arXiv:1111.6097 ·
C/A, Dokshitzer et al., *JHEP* **08** (1997) 001, arXiv:hep-ph/9707323 · Soft Drop,
Larkoski et al., *JHEP* **05** (2014) 146, arXiv:1402.2657; mMDT, Dasgupta et al.,
*JHEP* **09** (2013) 029, arXiv:1307.0007; Recursive Soft Drop, Dreyer et al., *JHEP*
**06** (2018) 093, arXiv:1804.03657 · Lund plane, Dreyer, Salam & Soyez, *JHEP* **12**
(2018) 064, arXiv:1807.04758; primary-Lund density, Lifson, Salam & Soyez, *JHEP*
**10** (2020) 170, arXiv:2007.06578 · jet NP/UE balance, Dasgupta, Magnea & Salam,
*JHEP* **02** (2008) 055, arXiv:0712.3014.

**Showers, coherence, hadronization.** Coherent branching, Catani, Marchesini &
Webber, *Nucl. Phys. B* **349** (1991) 635 · Lund string, Andersson, Gustafson,
Ingelman & Sjöstrand, *Phys. Rept.* **97** (1983) 31 · cluster model, Webber, *Nucl.
Phys. B* **238** (1984) 492 · **local parton–hadron duality**, Azimov, Dokshitzer, Khoze
& Troyan, *Z. Phys. C* **27** (1985) 65; **shape function**, Korchemsky & Sterman, *Nucl.
Phys. B* **555** (1999) 335, arXiv:hep-ph/9902341 · groomed-mass NP corrections, Frye et
al., arXiv:1603.09338; Hoang et al., *JHEP* **12** (2019) 002, arXiv:1906.11843 · power
corrections, Dokshitzer & Webber, arXiv:hep-ph/9504219.

**Tree/Lund ML models.** JUNIPR, Andreassen et al., *EPJC* **79** (2019) 102,
arXiv:1804.09720; binary JUNIPR, *PRL* **123** (2019) 182001, arXiv:1906.10137 ·
LundNet, Dreyer & Qu, *JHEP* **03** (2021) 052, arXiv:2012.08526 · tractable-likelihood
shower inference, arXiv:2105.10512, arXiv:2112.12795.

**Latent-alignment / edit models (§5.5).** RNN transducer, Graves, arXiv:1211.3711 ·
CTC, Graves, Fernández, Gomez & Schmidhuber, *ICML* (2006) · training through the
lattice, Imputer, Chan et al., arXiv:2002.08926 · Insertion Transformer, Stern et al.,
arXiv:1902.03249; Levenshtein Transformer, Gu et al., arXiv:1905.11006 · conditional
flow matching, Lipman et al., *ICLR* (2023), arXiv:2210.02747; FMPE, Wildberger, Dax et
al., *NeurIPS* (2023), arXiv:2305.17161; probability-flow ODE, Song et al., *ICLR*
(2021), arXiv:2011.13456 · distribution-level hadron→parton corrections in the Lund
plane, ATLAS, arXiv:2004.03540.

**Amortized inference, posterior estimation, unfolding.** Cranmer, Brehmer & Louppe,
*PNAS* **117** (2020) 30055, arXiv:1911.01429 · Papamakarios & Murray, *NeurIPS*
(2016), arXiv:1605.06376 · Greenberg, Nonnenmacher & Macke, *ICML* (2019),
arXiv:1905.07488 · cINN unfolding, Bellagente et al., *SciPost Phys.* **9** (2020)
074, arXiv:2006.06685; iterative variant, Backes et al., *SciPost Phys. Core* **7**
(2024) 007, arXiv:2212.08674 · OmniFold, Andreassen et al., *PRL* **124** (2020)
182001, arXiv:1911.09107 · unfolding landscape, arXiv:2404.18807; generative unfolding
of jets, arXiv:2510.19906 · SBC, Talts et al., arXiv:1804.06788.

**Hadronization fitting / ML hadronization (context).** HOMER, Bierlich et al.,
arXiv:2410.06342; Assi et al., arXiv:2503.05667 · MLHAD, Ilten et al., *SciPost Phys.*
**14** (2023) 027, arXiv:2203.04983; Bierlich et al., *SciPost Phys.* **17** (2024)
045, arXiv:2311.09296 · HadML, Ghosh et al., *PRD* **106** (2022) 096020,
arXiv:2203.12660 · review, Badger et al., *SciPost Phys.* **14** (2023) 079,
arXiv:2203.07460.

---

*Sourced from `md/amortized_posterior_hadronization.md` (primary), with background
from `md/hadronization_grooming_survey.md` and `md/parton_showers_survey.md`. arXiv
identifiers were carried over from those notes; confirm author lists on INSPIRE
before use in print, and consult the latest literature for the rapidly developing
generative-unfolding and ML-hadronization areas.*
