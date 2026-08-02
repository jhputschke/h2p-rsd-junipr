# Production test edit — results

**Status: complete.** All 9 arms of the [`PLAN_prod_test_edit.md`](PLAN_prod_test_edit.md) §7
grid trained and evaluated, the reference re-evaluated on the same code path and device, and
gates E1–E9 applied. The plan holds the design and the rationale, and every pass criterion in
it was fixed before the grid started — read the plan for *why* each gate exists. Companions:
[`PROD_TEST_v0_RESULTS.md`](PROD_TEST_v0_RESULTS.md),
[`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md).

**Verdict in one line:** the edit factorization **loses**, unanimously across three seeds and
on all three deciding metrics, with bands that do not overlap the reference's — *and* the
run's second product is a positive one: the anchoring premise the family rests on is
**confirmed** at production scale (E7), so the failure is not that the physics is wrong but
that only ~19% of parton nodes are anchored, which leaves an expensive AR model doing the
rest (§6).

Regenerate every table below from the artifacts:

```bash
bash scripts/run_prod_test_edit.sh                                    # the §7 grid, 9 trainings
bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_edit --device cpu
bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_v1  --device cpu \
     --only v1_contstop_s0,v1_contstop_s1                             # WP-F.1: re-evaluate the reference
python scripts/edit_anchoring_diagnostic.py --run-root runs/prod_test_edit --n-jets 4000
python scripts/prod_test_edit_gates.py --run-root runs/prod_test_edit \
    --reference-root runs/prod_test_v1 --out docs/PROD_TEST_edit_TABLES.md
```

[`PROD_TEST_edit_TABLES.md`](PROD_TEST_edit_TABLES.md) is that generated output, committed
beside this document so the prose here can be checked against machine-produced numbers.

| | |
|---|---|
| grid | `scripts/run_prod_test_edit.sh` — 9 trainings, `presets/prod_test_edit.yaml` + one override each |
| base arm | `edit_v1` + `lundnet`, aux(9), `n_bins = 30`, 60 epochs, **`lnz_support: physical`** |
| reference | `runs/prod_test_v1/v1_contstop_s0` and `_s1` — **re-evaluated, not retrained** (WP-F.1) |
| train | `data/jet_aux_asym.root` — 495 071 jets, seed 1 |
| test | `data/jet_aux_asym_test.root` — 97 018 jets, seed 2 |
| grooming | `z_cut = 0.1`, `beta = 0`, `kt_floor = 1.0`, `kt_floor_sec = 0.2` |
| device | `cpu` for every evaluation on **both** roots, recorded in each artifact (WP-F.1) |
| environment | conda `js_fno`, torch 2.11+cu130, one NVIDIA GB10 |

> ⚠️ **NLL comparability.** WP-E put the same truncated `ln z` head on the edit family that
> `v1_contstop` was trained with, which is what makes E6 a measurement of the factorization
> rather than of the head. Both families are `exact_likelihood = True`, both are normalized
> densities on the same `(u, v, ln z, ψ)` space, both declare `lnz_support: physical`, and
> both were scored on the **same val split** (`_split_by_event` keys off `data.seed`, which
> is 0 in both presets). The `e_v1_legacy_lnz` arm is an *attribution* arm and its NLL
> belongs in no column with the others.

> ⚠️ **E9 is a declared blind spot.** The edit family has no `coordinate_cdfs`, so G3,
> `pit_ks_max` and the region × coordinate cross **cannot be read here at all**. The one v1
> gate still open — the `ln z` shape *inside* its support — is invisible to this comparison.
> It does not change the verdict (the family loses on the axes that *are* readable), but any
> future claim about this family must state it. See §4.9.

---

## 0. Deviations from the plan, and when they were made

Two, both after the grid ran, both recorded here rather than in a footnote.

**0.1 `scripts/refresh_support_audit.py` was NOT run (plan §12.5).** Three reasons, in order
of weight:

1. **It was redundant.** The refresh exists to bring artifacts written under an *older*
   `EDGE_TOL` up to the current convention, so a gate table cannot mix two. Every one of the
   11 artifacts in this comparison — 9 edit arms and 2 re-evaluated reference arms — was
   written by *this* run's eval, and `eval/support.py` last changed on 2026-08-01, before all
   of them. They already come from one code path at one convention, which is the only
   property the refresh establishes.
2. **On the edit root it costs ~16 hours.** `EditTransducer` does not override
   `sample_coordinates_many`, so the base class loops `sample_coordinates` once per draw:
   2 000 jets × 200 draws = **400 000 sequential lattice builds per arm**, ~1 h 50 m each,
   nine arms. See §6.4 — this is a fired trigger, not an accepted cost.
3. **On the v1 root `--force` would collaterally rewrite all 11 arms**, including the nine
   that [`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md) is written against. Altering a
   completed run's artifacts for no benefit to this one is a worse outcome than skipping.

The skip was checked rather than assumed. The refresh had completed **one** arm
(`e_v1_freewidth_s0`) before it was stopped, so that arm has been audited under *both*
procedures, and they agree on everything scored: all four rates `0.0`, `passes: true`. The
only difference is the emission count — 559 341 (eval, shared draws) vs 558 938 (refresh,
re-sampled) — 0.07% MC noise from a different draw set.

**0.2 One arm carries a stamp the others do not.** As a consequence of 0.1,
`e_v1_freewidth_s0`'s `eval_metrics.json` carries `audit_refreshed_at_edge_tol: 1e-06` and
the other eight do not. Substantively inert per the check above, but it is a real
inconsistency in the files and is named here rather than left to be discovered.

**0.3 The deep notebook pass reports no aux ablation for this grid.** `notebooks/prod_test_v1.ipynb`
§2c compares aux-on against aux-off arms under `ABLATION_ROOT`. Left pointing at
`runs/prod_test_v0/ablation` it would have entered *this* run's edit checkpoint into a grid
whose other three arms are v0 `ar_junipr` checkpoints, and the resulting "aux ON − aux OFF"
delta would have been a **family** difference wearing an ablation's label — the within-arm
"seed band" it is judged against would have been |edit − AR| ≈ 1 nat and would have swallowed
the effect entirely. The aux question belongs to v0 §5 / v1 §3.1 and is not this run's: plan
§7 holds aux fixed at nine columns across every edit arm. `ABLATION_ROOT` therefore names a
directory that does not exist, `arm_checkpoints` degrades to the headline arm alone, and §2c
says so rather than inventing a number.

**0.4 Four notebook cells assumed the AR family and were made family-tolerant.** WP-F.3 as
planned covered §0, §6 and §8; running it surfaced three more places where the notebook was
written against `ar_junipr_*` rather than against the model base class. None changes an AR
number:

| cell | assumption | resolution |
|---|---|---|
| §2b | `model.nll_terms()` — the length × cell × coordinate split | the **total** now comes from `log_prob` on both paths (identical to the three-term sum for AR, and asserted to be); the decomposition prints `n/a` where the factorization has no such factors |
| §3 | `model.split_head` | falls back to `free_cell_head`, the edit family's cell categorical — the same object in the same role, so the rank argument transfers |
| §6 check | non-`n_head` ⇒ sampler, checked by a Monte-Carlo TV bound | a third branch: the edit `length_pmf` is an exact parameter-free DP, so it is asserted **bit-identical** across two evaluations, not merely within MC noise |
| §0 | `ABLATION_ROOT` | see 0.3 |

The §2b fix is the one that mattered: without it the deep pass aborted, and with it the
notebook reports the one NLL number that is defined for both families and is exactly what
gate E6 compares.

## 1. WP-E — what the `ln z` support port changed, and what it did not

The one code change this run made, and it is a support correction rather than an experiment
(plan §3). `EditTransducerConfig` gained `lnz_support` / `lnz_zcut` / `lnz_beta` with the
same names and semantics as `ARJuniprConfig`, so `data.stats.check_lnz_support` covers the
family unchanged.

**The guard fired on every arm, at production scale:**

```
[data] ln z support OK: physical, z_cut=0.1, beta=0,
       all 700330 truth emissions inside [-2.3026, -0.6931]
```

**One tightening over the AR implementation.** The bound is read at the node's **own** `u`,
`lo(u) = ln z_cut − β·u`, not at the loosest `u` in its cell. This factorization supports
that — the emission density is `f(u, v)·f(ln z | u)·f(ψ)` — where the AR coordinate head, a
product of factors independent *given the cell*, cannot. So it is the exact Soft Drop
boundary, the same expression the guard checks the truth against, with no `|β|·half_u` slack.
At the fielded `β = 0` the two conventions coincide.

**`_log_cell_mass` is unchanged, and that is a fact rather than an oversight.** A cell is a
box in `(u, v)` only, and `∫ f(ln z | u) d ln z = 1` for every `u`, so the constrained
forward–backward behind `sample_coordinates` draws bit-identical alignments under both
supports. `tests/test_edit_lnz_support.py` asserts this rather than trusting it, on the cell
masses, the anchored shares and the drawn alignment.

**Parity held.** `scripts/verify_parity.py` reproduces the reference v2 likelihood and decode
bit-for-bit, and neither it nor `models/ar_junipr.py` was touched — asserted, not assumed.
No parameter and no buffer was added, so `legacy` state dicts stay byte-identical.

## 2. Is the test valid?

| check | result |
|---|---|
| `check_lnz_support` against the file's own grooming record, before every arm trained | **pass**, all 700 330 truth emissions inside the declared interval |
| val split identical across the two grids | **yes** — `_split_by_event` keys off `data.seed`, 0 in both presets, so "best val NLL/jet" is comparable across roots |
| one device across the whole comparison | **yes** — `cpu`, recorded in all 11 artifacts (WP-F.1 backfilled the key the pre-flag v1 files lacked) |
| the reference reproduces its published band | **yes, to every printed digit** — see §3.2 |
| alignment monotonicity (risk 1) | **0 crossing pairs** on every arm, 300 sampled alignments each |

## 3. Grid arms

All 9 arms trained 60 epochs at batch 256, `n_bins = 30`.

| arm | varies | seeds | best val NLL/jet | parameters |
|---|---|---:|---:|---:|
| `e_v1` | — | 0, 1, 2 | 4.853 / 4.894 / 4.853 | 163 996 |
| `e_v2` | `prefix_conditioning = true` | 0, 1, 2 | 4.441 / 4.429 / 4.443 | 228 092 |
| `e_v1_legacy_lnz` | `lnz_support = legacy` | 0 | 4.997 **!** | 163 996 |
| `e_v1_freewidth` | `physics_width = false` | 0 | 4.820 | 164 256 |
| `e_v1_gru` | `encoder = gru` | 0 | 4.846 | 446 300 |
| **`v1_contstop`** *(reference)* | — | 0, 1 | 3.780 / 3.805 | 259 053 |

**!** not comparable — a different `ln z` normalization shifts NLL/jet by a constant unrelated
to fit quality.

### 3.1 Parameter counts do NOT confound this result

`e_v1` at 163 996 against `v1_contstop` at 259 053 is a ratio of **1.58×**, inside the ~2×
threshold plan §13 set for declaring a family claim confounded — and it runs the *wrong way*
for an excuse: the edit arm loses while being the smaller model, so a capacity argument would
have to explain why 1.58× fewer parameters costs 1.07 nat, 5.6× the TARP deviation and a
failed coverage clause simultaneously. `e_v2` at 228 092 is 1.14× below the reference and
still loses. The confound is named and it does not rescue the family.

### 3.2 The reference re-evaluation reproduced its published band exactly

WP-F.1's whole purpose, and it is worth stating that it worked:

| | re-evaluated (this run, cpu) | published (v1 results §4.8/§4.9) |
|---|---|---|
| TARP max dev | [0.0200, 0.0225] | [0.0200, 0.0225] |
| recomputed null 95% | 0.0275 | 0.0275 |
| `coverage_68` | [0.5304, 0.5310] | [0.5304, 0.5310] |
| medoid/identity | [0.9286, 0.9327] | [0.9286, 0.9327] |

Nothing in the eval path drifted between v1 and this run, so the head-to-head is genuinely
one evaluation pass rather than a comparison across two.

### 3.3 `edit_v2` memory at `n_bins = 30` — plan §13's risk does not bite

Measured on the production file at batch 256, one GB10: **peak 191 MiB CUDA for `edit_v2`,
122 MiB for `edit_v1`.** No batch-size drop was needed, so E8 is matched at the reference's
256 and no optimization difference contaminates it. Throughput ~22 ms/step (`e_v2`) and
~28 ms/step (`e_v1`), i.e. ~0.6–0.8 min/epoch.

## 4. Gates E1–E9

### 4.1 E1 — acceptance: **PASS**, unanimous

medoid/identity **0.957**, geo-median/identity **0.948** — both below 1 and agreeing in sign,
on all three seeds. The precondition holds, so everything below means something.

### 4.2 E2 — support: **PASS**, unanimous, and the attribution arm attributes

| arm | out of window | soft drop | `z > ½` | `k_t` floor | verdict |
|---|---:|---:|---:|---:|---|
| every `physical` arm (8) | 0.00000% | 0.00000% | 0.00000% | 0.00000% | **PASS** |
| `e_v1_legacy_lnz_s0` | 0.00000% | **1.40311%** | **3.90537%** | 0.00000% | FAIL, as required |

The plan pre-registered that the legacy arm should reproduce "the ~0.81% / ~3.98% failure".
It came in at 1.40% / 3.91% — the `z > ½` wall essentially dead on, the soft-drop wall
somewhat worse than v0's. So the leak is real, it is the unbounded `ln z` head, and WP-E
removes it **completely** rather than reducing it. This is what makes E6 quotable.

### 4.3 E3 — multiplicity: **FAIL**, unanimous — but read which clause failed

| clause | result |
|---|---|
| ⟨N⟩_post/⟨N⟩_truth on the **full** population | **0.9964 / 0.9802 / 0.9764** — all inside [0.95, 1.05], **passes** |
| leading-cell 68% coverage, every scoreable region | 0/2 Wilson-consistent (`wide_soft`, `narrow_soft`) — **fails** |
| SBC-on-N against its own MC null | not run in this artifact (`exposure_diagnostic` produced no simulated null for this family) |
| `q(0\|x)` AUC vs 0.827 | see §4.11 |

The **marginal** multiplicity is right — arguably the family's best single number, and exactly
what the structural `q(N|x)` was supposed to buy. What fails is regional coverage, i.e. the
joint. That is the same shape of defect v1 diagnosed, not a new one, and the edit
factorization does not fix it.

### 4.4 E4 — TARP, the deciding gate: **FAIL**, unanimous

| seed | max dev | recomputed null 95% | verdict |
|---|---:|---:|---|
| `e_v1_s0` | 0.1215 | 0.0275 | FAIL |
| `e_v1_s1` | 0.1210 | 0.0275 | FAIL |
| `e_v1_s2` | 0.1130 | 0.0275 | FAIL |

Band **0.1185 [0.1130, 0.1215]** against `v1_contstop`'s **0.0212 [0.0200, 0.0225]**. The
bands do not overlap and the separation is **the wrong way**: the edit posterior is 5.6×
further from calibrated than the arm it was supposed to beat, against a null whose floor
(0.0275 < 0.05) makes the statistic quotable. Both clauses of E4 fail, on every seed.

This is the gate the run was designed around, and it is unambiguous.

### 4.5 E5 — coverage: **FAIL**, unanimous

`coverage_68` band **0.5179 [0.5078, 0.5233]** against **0.5307 [0.5304, 0.5310]**. Zero of
two scoreable regions Wilson-consistent with 0.68, on every seed. The bands clear each other,
again the wrong way. v1's coverage clause failed at 0.518–0.540 against 0.68; the edit family
lands in the same place, so the third factorization does not move the defect v1 attributed.

### 4.6 E6 — held-out NLL: **FAIL**

**4.8667 [4.8525, 4.8944]** against **3.7927 [3.7799, 3.8054]** — **+1.074 nat**, bands far
apart. Quotable on both preconditions: E2 passes on both sides, and both arms declare
`lnz_support: physical` (the gate script refuses to rank a mismatched pair). `e_v2` narrows it
to +0.645 nat and still loses.

For scale, the `ln z` head alone is worth 0.144 nat on this family (`e_v1_legacy_lnz` 4.997 vs
`e_v1_s0` 4.853), closely matching the ~0.15 nat v1 measured for the same intervention. The
family gap is **seven times** the head effect.

### 4.7 E7 — anchoring, the stage gate: **PASS** — and this is the run's positive finding

Read off `e_v1_freewidth_s0` (`physics_width = false`), the arm never told the functional
form, on 4 000 production jets:

| | measured | criterion |
|---|---:|---|
| **Λ_eff** (`ln k_t`) | **0.631 GeV** | [0.2, 5] GeV |
| **R²** | **0.949** | ≥ 0.9 |
| scoreable `ln k_t` bins | 6 | ≥ 3 to fit at all |
| widths fall with `k_t` | **yes**, monotonically | required |
| `frac_anchored` | 0.224 | reported |

The residual width in `ln k_t` falls from **0.661 to 0.177** across the fitted range — a
factor of 3.7 — and the fit is good on the other coordinates too: `ln z` Λ_eff = 0.317 GeV
(R² 0.866), `ψ` 2.144 (R² 0.953), `ln 1/ΔR` 0.268 (R² 0.728). Every one of the nine arms
independently shows falling widths with Λ_eff in 0.32–0.63 GeV and R² 0.80–0.99.

**So the anchoring premise holds on production data.** The 6-epoch, 54k-jet `Λ_eff = 1.29 GeV,
R² = 1.000` that [`PLAN_EditTransducer.md`](PLAN_EditTransducer.md) shipped on was optimistic
in magnitude but right in kind: there really is a `Λ_eff/k_t` smearing kernel of hadronic
scale, measured independently of the parametrization, at production scale. The plan
pre-committed to reporting a *flat* fit as an informative failure; it is only fair to record
the converse with the same weight.

### 4.8 E8 — `edit_v1` vs `edit_v2`: `e_v2` wins, conditional on E7 (which passed)

| quantity | `e_v1` | `e_v2` | clears? |
|---|---|---|---|
| best val NLL/jet | 4.8667 [4.8525, 4.8944] | **4.4376 [4.4287, 4.4426]** | **yes — `e_v2`** |
| TARP max dev | 0.1185 [0.1130, 0.1215] | **0.0717 [0.0660, 0.0775]** | **yes — `e_v2`** |
| `coverage_68` | 0.5179 | 0.5107 | tie |
| medoid/identity | 0.9501 | 0.9384 | tie |
| `<N>` ratio | 0.9843 | 0.9849 | tie |
| parameters | 163 996 | 228 092 | +39% |

Matched seeds, encoder, batch size and epochs. Recoil correlation among the `y` nodes is worth
0.43 nat and cuts the TARP deviation by 40% — a real effect, at +39% parameters. It does not
change the verdict: `e_v2`'s TARP band [0.0660, 0.0775] is still **3.4×** the reference's, and
its NLL still **+0.645 nat**. E9 applies here too — coordinate PITs cannot adjudicate this
comparison because neither stage has them.

### 4.9 E9 — coordinate PIT: **`n/a` by construction, pre-registered**

`EditTransducer.supports_coordinate_pit = False` and `coordinate_cdfs` returns `None`, so G3,
`pit_ks_max` and the region × coordinate cross are unavailable. The one v1 gate still open —
the `ln z` shape *inside* its support, 1.05–2.07× crit — **was never readable here**, and the
head-to-head is incomplete on that axis. Recorded as a blind spot, never as a pass.

Plan §11 makes closing it (the exact prefix-conditional CDF as a responsibility-weighted
mixture) conditional on "E4 or E6 favouring edit, making it a fielding candidate". **Neither
does**, so the trigger has *not* fired and the work stays deferred. The blind spot cost this
run nothing, because the family lost on the axes that were readable.

### 4.10 The one-seed probes

One training each; they carry no band and license only "worth a proper multi-seed A/B"
(v1 §3.2's discipline).

| quantity | `e_v1` band (3 seeds) | `e_v1_gru` | `e_v1_freewidth` |
|---|---|---:|---:|
| best val NLL/jet | 4.8667 [4.8525, 4.8944] | 4.8462 *(outside, better)* | 4.8199 *(outside, better)* |
| TARP max dev | 0.1185 [0.1130, 0.1215] | 0.1240 *(outside, worse)* | 0.1155 |
| `coverage_68` | 0.5179 [0.5078, 0.5233] | 0.5221 | 0.5060 *(outside)* |

`e_v1_freewidth` beating every physics-width seed on NLL (4.8199 vs [4.8525, 4.8944]) is worth
naming: **imposing the shape-function form costs likelihood** even though E7 confirms the form
is the right one. The free MLP can fit the same falling kernel *and* whatever else is in the
residual. One seed, so it licenses an A/B and nothing more.

### 4.11 `e_v1_s0` assessed in depth

`notebooks/prod_test_v1.ipynb` at `e_v1_s0`, `N_FIT = 20 000`, the full 97 018-jet test file
and 137 353 truth emissions — matching what v1 results §4.9 did for `v1_contstop_s0`.
Artifact: `runs/prod_test_edit/e_v1_s0/<stamp>/prod_test_v1/prod_test_v1_metrics.json`.

| | `e_v1_s0` deep pass |
|---|---|
| held-out NLL/jet, whole test file | **4.8813** |
| acceptance | **PASS** — medoid/identity 0.9114, off-grid geo-median/identity 0.8807, both beat identity |
| `coverage_68` | 0.5177, Wilson [0.4937, 0.5416] on 1 667 jets |
| TARP max dev (300-jet tier) | 0.1133 |
| support, all four walls, posterior | **0.00000%** — and 0 draws even *on* a bound |
| support, truth control | 0.00000%, 2 emissions on a boundary of 137 353 |
| cell occupancy | posterior reaches 341 cells, misses 15 of the 275 the truth occupies |
| `free_cell_head` effective rank | 35.7 of 64 |

**The `length_pmf` cost, which is what the plan asked this pass to measure.** The exact
structural DP runs at **10.12 ms/jet** on cpu over 40 000 jets (405 s), against results
§4.9's **52.19 ms/jet** for `v1_contstop`'s 500-draw sampler — a **5.2× reduction**, and it
is what let `N_FIT` stay at 20 000 rather than being cut to 5 000. The two are not the same
object: the reference is a 500-draw *histogram* of `q(N|x)` and this is `q(N|x)` itself, so
the comparison is of what each costs to obtain. The §6 self-consistency check confirms the
DP is deterministic — two independent evaluations agree bit-for-bit, which a sampler-based
belief cannot do and which is why that check needed its own branch (§0.4).

**E3's open clause, now closed: `q(0|x)` AUC = 0.770**, against the `v1_contstop` reference
of 0.827 (0.824 in its own artifact). So the edit family's *exact* `q(N = 0|x)` — read off a
structural DP with no fitted head, on the delete-all path the family represents natively —
**ranks empty-vs-nonempty jets worse than the AR family's fitted continue/stop head does.**
That is a striking result and it belongs beside E6: the one place the structural marginal
should have had an unambiguous advantage, it does not. The gated empty rate confirms it —
0.2277 predicted against 0.1605 true, a 1.42× over-prediction at recall 0.425, with `tau`
frozen from the training-file val split.

The `<N>` marginal remains the family's best number (§4.3), so the defect is not that it
gets the *average* multiplicity wrong — it is that `q(N|x)` does not *discriminate* per jet
as well as the head it was supposed to replace.

**§2c reports no aux ablation** and **§2b reports no length/split/coord decomposition**, both
by construction — see §0.3 and §0.4.

## 5. The head-to-head, in one place

| quantity | `v1_contstop` (re-evaluated) | `e_v1` | delta | separated? |
|---|---|---|---:|---|
| best val NLL/jet | 3.7927 [3.7799, 3.8054] | 4.8667 [4.8525, 4.8944] | **+1.0740** | yes — **worse** |
| TARP max dev | 0.0212 [0.0200, 0.0225] | 0.1185 [0.1130, 0.1215] | **+0.0972** | yes — **worse** |
| `coverage_68` | 0.5307 [0.5304, 0.5310] | 0.5179 [0.5078, 0.5233] | −0.0128 | yes — **worse** |
| medoid/identity | 0.9307 [0.9286, 0.9327] | 0.9501 [0.9371, 0.9574] | +0.0194 | yes — **worse** |
| geo-median/identity | 0.9393 [0.9392, 0.9393] | 0.9430 [0.9395, 0.9477] | +0.0037 | yes — **worse** |
| `<N>` ratio | 1.0084 [1.0008, 1.0161] | 0.9843 [0.9764, 0.9964] | −0.0241 | both inside [0.95, 1.05] |
| parameters | 259 053 | 163 996 | −95 057 | 1.58×, not confounding |

Every deciding row separates, and every one of them separates the wrong way. The `⟨N⟩` row is
the exception worth stating precisely: the reference is nominally closer to 1, but **both
arms pass E3's criterion**, so that row ranks without deciding.

The plan warned that `v1_contstop`'s band is narrow because it is 2 draws rather than because
it is stable, and that a three-seed edit band clearing it would not be decisive on its own.
That caution is not load-bearing here: the gaps are 5–50× the width of either band.

## 6. What this run found

### 6.1 The question it was built to answer

*Does a third factorization of the length/shape coupling — one designed against v1's defect
before that defect was measured — beat the arm v1 picked?* **No.** Not on TARP, not on
coverage, not on held-out NLL, not on any seed, and not by a margin that any band or capacity
argument reaches.

**And it loses on its own strongest ground.** The edit family's `q(N|x)` is exact,
parameter-free and explicitly conditioned on `|x|`, where `v1_contstop`'s is a fitted
per-step continue/stop product — plan §1 put that contrast at the head of the table. Yet the
exact marginal **discriminates worse**: `q(0|x)` AUC **0.770** against the fitted head's
**0.827** (§4.11). Exactness bought calibration of the *average* (⟨N⟩ ratio 0.9843, the
family's best number) and cost discrimination *per jet*. Being the right functional form is
not the same as being the more informative one.

### 6.2 …but the family's physics claim survives

E7 is not a consolation prize. The edit transducer's falsifiable claim was that hadronization
smears parton nodes with a width running as `Λ_eff/k_t`, and that claim is **confirmed on
production data, measured off the arm that was never told the form**: Λ_eff = 0.631 GeV,
R² = 0.949, widths falling monotonically by a factor of 3.7 across six `k_t` bins, reproduced
with consistent sign and scale by all nine arms. That is a physics result independent of
whether the posterior is any good.

### 6.3 The mechanism that explains the loss

The WP-G diagnostic says where the family spends itself, and the two rates are the answer:

| | `e_v1` (3 seeds) | `e_v2` (3 seeds) |
|---|---:|---:|
| `frac_anchored` — share of parton nodes that are smeared copies | **0.186–0.195** | **0.159–0.170** |
| `delete_rate` — share of hadron nodes that anchor nothing | 0.843–0.849 | 0.860–0.868 |

**Only about a fifth of parton emissions are anchored.** The other four fifths come from the
free insertion head — which is an ordinary cell-categorical-plus-offsets model, i.e. exactly
what plan §13's risk 2 called "an expensive AR model". The `n_y = n_x − #del + #ins` identity
is structurally sound, and the alignment is provably monotone (0 crossing pairs), but the
anchoring term the family exists for touches 19% of the emissions. The premise is right and
its reach is small.

Two secondary rates confirm the mechanism is doing something physical rather than degenerate:

- **Deletion falls with `k_t`**: 0.939 in `ln k_t ∈ [0, 0.5)` down to 0.320 in `[2.5, 3.0)`.
  Soft hadron nodes are deleted, hard ones are kept and smeared — the sub-floor fragmentation
  population the plan predicted (§6.3).
- **Insertion falls with distance from the grooming boundary**: 0.931 within 0.1 of the
  soft-drop wall down to ~0.71 far from it. Emissions hugging the boundary are the ones the
  free head supplies.

And `frac_anchored` did **not** recover with training: it was 0.20 at 6 epochs on the small
file and is 0.19 at 60 epochs on production. `p_anch` initialization and the identifiability
of the two-component mixture are the place to look, not the epoch count.

### 6.4 Triggers that have fired

- **A batched `sample_coordinates_many` for the edit family.** The base class loops the
  per-draw hook, so the support audit costs 400 000 sequential lattice builds per arm
  (~1 h 50 m) where the AR family — which overrides it — takes minutes. This blocked plan
  §12.5 (§0.1) and dominated every evaluation in this run. It is the same fix
  `PLAN_prod_test_speedup.md` §2 made for AR ("67 of the 109 min"), and the base-class
  docstring explicitly invites the override. **It was deliberately not made mid-run**: an
  override "is free to reorder RNG consumption — the draws are still draws from the same
  conditional, but they are NOT the same draws", so landing it would have made post-fix
  audits non-comparable with the ones already written. Follow-up plan.
- **`physics_width = true` costs likelihood.** `e_v1_freewidth` beats all three physics-width
  seeds on NLL while E7 confirms the imposed form is correct. One seed; a proper multi-seed
  A/B would say whether the regularization is worth its cost.
- **`frac_anchored` ≈ 0.19 is the ceiling on this family's mechanism.** Anything that raises
  it — `p_anch` initialization, a stronger prior, an identifiability penalty — is where a
  follow-up would have to start, and until it moves, the family is an AR model with an
  expensive lattice attached.
- **`notebooks/prod_test_v1.ipynb` reaches through the model base class in five places.**
  It was written when only the AR family existed, and pointing it at a fourth family
  surfaced every one: `nll_terms` and `per_jet_nll` (both `ar_junipr_*` methods, not base-class
  contract), `split_head`, `n_head`, and a `None`-formatting crash in the §9 summary once a
  metric legitimately did not exist. Each was found by *running* rather than by reading, at
  one aborted deep pass apiece. All are fixed and none changes an AR number (§0.4), but the
  notebook should either declare an AR dependency in §0 or use the base-class contract
  throughout — `tests/test_notebooks.py` parses the cells and cannot catch this class of
  defect. A cheap guard would be the audit that finally found them all at once: enumerate
  every `model.<attr>` the notebook touches and assert it against each registered family.

### 6.5 Triggers that did NOT fire

- **`coordinate_cdfs` for the edit family** (plan §11) stays deferred: its trigger was "E4 or
  E6 favours edit", and neither does.
- **The monotone rational-quadratic spline** and the **per-node joint coordinate density**
  remain v1's fired triggers, belonging to the AR family's follow-up. Nothing here moves them.

## 7. Verdict

**Do not field the edit transducer.** `v1_contstop` remains the recommendation v1 made, and
this run — which was built to overturn it — strengthens rather than weakens it: a third
factorization, designed specifically against v1's diagnosed defect, loses on every metric that
diagnosed it, unanimously, at a smaller parameter budget.

Three qualifications, all of which cut toward the same conclusion rather than against it:

1. The comparison never saw the `ln z` shape failure v1 left open (E9). Since the edit family
   lost on the readable axes, closing that blind spot would only have added another chance to
   lose.
2. `edit_v2` is better than `edit_v1` on NLL and TARP, and if this family is revisited that is
   the stage to revisit. It is still 3.4× the reference's TARP deviation.
3. `v1_contstop` itself is a two-seed arm that v1 fielded as a comparison rather than a
   validated deliverable, and this run does not change that. It re-evaluated it and reproduced
   its band exactly; it did not add seeds to it.

**The run's durable products** are the E7 measurement — a `Λ_eff = 0.631 GeV` shape-function
scale, extracted from a latent alignment nobody supervised, on production data, off the arm
that was not told the answer — and the E2 demonstration that WP-E removes the `ln z` support
failure completely on a fourth model family.

**The most transferable negative result** is §4.11's: an exact, parameter-free `q(N|x)` was
*worse* at ranking empty jets than a fitted continue/stop head (AUC 0.770 vs 0.827), while
being better at the marginal ⟨N⟩. Structural exactness is not a substitute for a
discriminative fit, and this run is the counterexample. Anyone reaching for "make the length
model exact" as a remedy for a calibration defect should read that number first — it is the
cleanest thing this grid established, and it did not require the family to win anything.

## 8. What is not measured

- **The `ln z` shape inside its support** (E9) — structurally unreadable on this family.
- **Split variance.** `data.seed` stays 0; nothing here bounds what a different train/val
  split would do.
- **HERWIG driver / fragmentation-variation weights.** Still absent, so the train/test deltas
  remain the noise-floor stand-in and the largest unquantified systematic (v0 §10, v1 §8).
- **SBC-on-N against its own MC null** — no simulated null was produced for this family in
  these artifacts, so E3's first clause is unread.
- **Whether a higher `frac_anchored` would change the verdict.** Everything in §6.3 says the
  mechanism's reach is the binding constraint; nothing here tests raising it.
</content>
