# The Physics of `h2p-rsd-junipr`

*Amortized per-jet posterior inference of the groomed parton-shower configuration
across hadronization.*

This document is the physics companion to [`README.md`](README.md). It explains
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

---

## 3. Why the map is a posterior, not a function

Lund-string fragmentation is stochastic and many-to-one: one groomed parton tree `y`
hadronizes into a *distribution* of hadron trees `x`, and many `y` can yield the same
`x`. There is **no deterministic inverse** and no unique "the parton configuration."
The well-defined target is the posterior $p(y\mid x)$, and "most likely
configuration" means the **MAP estimate** $\hat y_{\rm MAP}(x)=\arg\max_y q_\phi(y\mid x)$.

Two consequences are built into the code:

- **Report the width, not just the mode.** Every jet gets *both* a MAP/beam estimate
  and a posterior summary (mean, 68% credible region, multiplicity distribution) —
  `eval.closure.print_point_estimate`. In high dimensions the mode can be
  unrepresentative (`amortized_posterior_hadronization.md` §6), so a regressor that
  learns only the conditional mean is the wrong tool: it blurs precisely where the
  inverse is ambiguous. This package models a density, never a point regression.
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

## 8. Architectures (one contract, three families)

All families expose `log_prob` / `sample` / `map_estimate` (`models/base.py`), in
decreasing affinity to the Lund-tree representation:

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
  search over the decoded tree; the **posterior** is ancestral sampling. Variable
  multiplicity is handled natively; only jet-level pairing is needed; the likelihood
  is explicit.
- **§5.2 Conditional normalizing flow / cINN** (`models/cinn.py`) — exact density,
  trivial sampling; variable multiplicity via a multiplicity head + structured latent
  (Bellagente et al., arXiv:2006.06685; Backes et al., arXiv:2212.08674).
- **§5.3 Conditional diffusion / Schrödinger bridge** (`models/diffusion.py`) — the
  most expressive option (arXiv:2404.18807).

In this repo §5.1 is production-grade; §5.2/§5.3 are functional baseline drop-ins
that share the contract and pass the smoke train (the §14 phased roadmap).

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

---

## 10. The physics knobs are config, not constants

Every physics choice is a versioned config field (`configs/`), never hard-coded:

| Physics | Config | Default |
|---|---|---|
| Lund-plane ranges, grid $N_{\rm bins}$ | `geometry.*` | $(0,6)^2$, 10 |
| Soft Drop $z_{\rm cut}$, $\beta$, $R_0$, $k_t$ floor | `GroomParams` (C++) / RNTuple provenance | 0.1, 0, 1, 1 GeV |
| generator / tune (systematic) | `experiment.generator_b` | — |
| encoder over $x$ (gru / lundnet / deepsets) | `encoder.*` | gru |
| posterior family (§5.1/5.2/5.3) | `model=…` | ar_junipr_v2 |

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
Phys. B* **238** (1984) 492 · groomed-mass NP corrections, Frye et al., arXiv:1603.09338;
Hoang et al., *JHEP* **12** (2019) 002, arXiv:1906.11843 · power corrections,
Dokshitzer & Webber, arXiv:hep-ph/9504219.

**Tree/Lund ML models.** JUNIPR, Andreassen et al., *EPJC* **79** (2019) 102,
arXiv:1804.09720; binary JUNIPR, *PRL* **123** (2019) 182001, arXiv:1906.10137 ·
LundNet, Dreyer & Qu, *JHEP* **03** (2021) 052, arXiv:2012.08526 · tractable-likelihood
shower inference, arXiv:2105.10512, arXiv:2112.12795.

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
