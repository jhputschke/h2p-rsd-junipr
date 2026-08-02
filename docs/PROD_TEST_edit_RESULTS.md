# Production test edit — results

**Status: complete.** All 9 arms of the [`PLAN_prod_test_edit.md`](PLAN_prod_test_edit.md) §7
grid trained and evaluated, the reference re-evaluated on the same code path and device, and
gates E1–E9 applied. The plan holds the design and the rationale, and every pass criterion in
it was fixed before the grid started — read the plan for *why* each gate exists. Companions:
[`PROD_TEST_v0_RESULTS.md`](PROD_TEST_v0_RESULTS.md),
[`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md).

**Verdict in one line:** on the pre-registered gates — per-jet posterior calibration — the
edit factorization **loses**, unanimously across three seeds and on all three deciding
metrics, with bands that do not overlap the reference's; on **MBR-decoded observable
spectra**, an instrument the gates do not cover, `edit_v2` is competitive with the reference
and better on two of three metrics (§4.13). Two positive products beside the negative one:
the anchoring premise the family rests on is **confirmed** at production scale (E7,
Λ_eff = 0.631 GeV), and the decode-side multiplicity collapse the family was built to remove
**structurally** turns out to be *worse* here than in the arm that still has the mechanism
(§4.13) — which relocates that defect away from the continue/stop head. See §7.

Regenerate every table below from the artifacts:

```bash
bash scripts/run_prod_test_edit.sh                                    # the §7 grid, 9 trainings
bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_edit --device cpu
bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_v1  --device cpu \
     --only v1_contstop_s0,v1_contstop_s1                             # WP-F.1: re-evaluate the reference
python scripts/edit_anchoring_diagnostic.py --run-root runs/prod_test_edit --n-jets 4000
python scripts/prod_test_edit_gates.py --run-root runs/prod_test_edit \
    --reference-root runs/prod_test_v1 --out docs/PROD_TEST_edit_TABLES.md

# the deep pass (§4.11-4.12): notebooks/prod_test_v1.ipynb, CKPT_ROOT set per arm
jupyter nbconvert --to notebook --execute notebooks/prod_test_v1.ipynb  # CKPT_ROOT = .../e_v2_s0

# distribution closure (§4.13) — NOT a plan §12 step; v0 and v1 ran it the same way, after
# the grid. Needs --prod-metrics: the script's own search is runs/prod_test_v*, which does
# not match runs/prod_test_edit.
python scripts/lund_closure_report.py --device cpu \
    --prod-metrics runs/prod_test_edit/e_v2_s0/*/prod_test_v1/prod_test_v1_metrics.json
```

[`notebooks/lund_distribution_closure_prod_test_edit.ipynb`](../notebooks/lund_distribution_closure_prod_test_edit.ipynb)
is the notebook form of that last step, **generated** by `scripts/make_prod_closure_nb.py`
and **pinned to `e_v2_s0`** — the arm E8 selected. Both of its artifact globs are pinned
inside `runs/prod_test_edit/`, because the generator's default fallback is
`runs/prod_test_v*`, which does not match this run root and would have repointed the
notebook at an `ar_junipr` checkpoint rather than merely a different arm.

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

Five, all after the grid ran, all recorded here rather than in a footnote.

**0.0 The deep pass was repeated on `e_v2_s0`.** Plan §12.8 names `e_v1_s0`, and that is
what §4.11 reports. But the plan was written before the grid, when E7 — the stage gate
`edit_v2` is conditional on — might still have failed, in which case every `e_v2` number
would have been null context. **E7 passed and E8 then selected `edit_v2`** (it clears
`edit_v1` on both of E8's deciding metrics), so the arm the plan named is the one the run
rejected. §4.12 repeats the deep pass on `e_v2_s0` and `notebooks/prod_test_v1.ipynb`'s
`CKPT_ROOT` now points there. §4.11 is kept rather than replaced: the two together are what
show the `q(0|x)` deficit is a family property and not a stage-1 artifact.

**0.1 `scripts/refresh_support_audit.py` was skipped, then RUN — plan §12.5 is honoured.**
It was skipped when the grid first finished, on three stated reasons; one of them was
wrong by ~8x, the fix that removed it landed, and the step was then run properly. Both
halves are recorded, because a plan step that gets honoured only after its stated obstacle
turns out to be overstated is a fact about this document, not just about the code.

The three reasons as given, and how each held up:

1. **Redundancy — held.** The refresh exists to bring artifacts written under an *older*
   `EDGE_TOL` up to the current convention. Every artifact in this comparison was written
   by this run's eval on one code path, so there were no two conventions to reconcile.
2. **"~16 hours on the edit root" — WRONG, by about 8x.** That figure came from a single
   arm's file mtime during a run that was killed partway, under no `OMP_NUM_THREADS` cap
   and with contention I did not record — not from a controlled measurement. The
   controlled numbers, same function and same inputs with only the sampler swapped:

   | `run_support_audit`, 2 000 jets | per arm | 9 arms |
   |---|---:|---:|
   | looped (base class) | 13.8 min | ~2 h 04 m |
   | batched (§6.4's fix) | ~3.3 min | **29.6 min, MEASURED** |

   So the pre-fix cost was ~2 h, not ~16 h. It was still the single most expensive step in
   the run, and the fix still bought ~4.2x — but the magnitude that justified skipping was
   not real.
3. **Collateral rewrite of the v1 root — held, and still governs.** `--force` on
   `runs/prod_test_v1` would rewrite all 11 arms, including the nine
   [`PROD_TEST_v1_RESULTS.md`](PROD_TEST_v1_RESULTS.md) is written against. **The v1 root
   was therefore left untouched and remains so**, which is also correct on its own terms:
   only `edit.py` / `edit_dp.py` changed, so no AR artifact could have moved.

**What the refresh changed: nothing that decides anything.** Re-run under the batched
sampler, TARP and `coverage_68` came back **bit-identical** on every arm — they consume
cell chains from `sample_batch` and never touch `sample_coordinates_many`. Only
coordinate-level quantities moved, all within MC noise: `<N>` ratio 0.9843 -> 0.9848, the
`e_v1_legacy_lnz` leak rates 1.403% / 3.905% -> 1.352% / 3.887%. Every `physical` arm is
still a hard zero on all four walls and the legacy arm still fails, so **no gate verdict
moved**.

**0.2 The stamp inconsistency is resolved.** While the skip stood, one arm
(`e_v1_freewidth_s0`) carried `audit_refreshed_at_edge_tol: 1e-06` from the partial run and
the other eight did not. The completed refresh stamps all nine, so the edit root is now
uniform. The v1 root is deliberately unstamped — see 0.1 reason 3.

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

**0.5 A distribution closure was run, which plan §12 does not ask for.** v0 and v1 both ran
`scripts/lund_closure_report.py` as a follow-up to their grids rather than inside them, and
the reference arm `v1_contstop_s0` therefore *has* such an artifact while this run's arms
initially did not — an asymmetry in the head-to-head, even though it was not a gap against
the plan. §4.13 closes it for `e_v1_s0` and `e_v2_s0` at the reference's own settings. It
turned out to be the one instrument that **disagrees** with the gates, so running it changed
what §7 says.

**A note on how two of the corrections above were found.** §4.11 and the first version of
§4.13's prose both generalized from `e_v1_s0` — the arm the *pre-grid* plan named — to "the
edit family", and both were wrong: the AUC deficit does generalize (§4.12) but the closure
result does not (`e_v1` loses to identity on all three MBR metrics; `e_v2` beats it on all
three). The plan naming one arm for the deep pass is a reasonable thing to do before a grid
runs; treating that arm as the family afterwards is not, and this document had to be
corrected twice for it.

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

### 4.12 `e_v2_s0` assessed in depth — the arm E8 actually selected

`e_v1_s0` is the stage E8 rejected. The plan named it for the deep pass (§12.8) because it
was written before E7 had passed, and `e_v2` was to be quoted only conditionally. E7 passed,
so the deep pass was repeated on `e_v2_s0` and this is the row that represents the family.

| | `e_v1_s0` | **`e_v2_s0`** | `v1_contstop_s0` |
|---|---:|---:|---:|
| held-out NLL/jet, whole test file | 4.8813 | **4.4697** | 3.7927 |
| `q(0\|x)` AUC | 0.7700 | 0.7677 | 0.824 |
| `length_pmf` ms/jet | 10.125 | 10.135 | 52.19 |
| medoid/identity | 0.9114 | 0.8972 | — |
| geo-median/identity | 0.8807 | 0.8882 | — |
| `coverage_68` | 0.5177 | 0.4985 | — |
| truth cells missed by the posterior | 15 | 14 | — |
| `free_cell_head` effective rank (of 64) | 35.7 | 41.0 | — |
| acceptance | PASS | PASS | PASS |
| support, all four walls | 0.00000% | 0.00000% | 0.00000% |

Three things this pair settles that one arm could not:

- **The `q(0|x)` deficit is a family property, not a stage-1 artifact** — 0.770 and 0.768 are
  the same number. Whatever it is, `edit_v2`'s prediction network does not fix it. (But see
  §4.13: a second instrument puts the same quantity at 0.807–0.810 against 0.818, so the size
  of the deficit is instrument-dependent and the finding does not survive as stated.)
- **`length_pmf` costs the same in both stages, to three digits** (10.125 vs 10.135 ms/jet).
  That is the structural claim made visible rather than argued: `length_pmf` reads only the
  op head, which is prefix-free in *both* stages — the very condition that keeps it exact.
  The prediction network costs emissions, not length.
- **`e_v2` is WORSE on `coverage_68` at this tier** (0.4985 vs 0.5177), where the gate tier
  called it a tie. Consistent with E8, which found coverage the one metric whose bands
  overlapped: the intra-family win is on NLL and TARP, and it does not extend to coverage.

### 4.13 Distribution closure — the instrument that disagrees

Not a plan §12 step: v0 and v1 both ran the Lund distribution closure as a follow-up to
their grids rather than inside them, and this run did the same. `scripts/lund_closure_report.py`
against each arm's own deep-pass artifact, 2 000 jets at K = 120, `mbr_backend=pot`, all
defaults — the same settings the reference arm was run at, so the three columns are
like-for-like.

`gmean_ratio` is posterior/identity: **below 1 means the model beats the decode-free RSD
baseline**, which is the comparison v0 §7 established as the one that matters.

| | `e_v1_s0` | **`e_v2_s0`** | `v1_contstop_s0` |
|---|---:|---:|---:|
| **MBR** W1 | 1.204 (6/14) | **0.667** (8/14) | 0.656 (9/14) |
| **MBR** KS | 1.263 (4/13) | **0.763** (7/13) | 0.860 (7/13) |
| **MBR** χ² | 1.122 (5/10) | **0.438** (7/10) | 0.917 (6/10) |
| MAP W1 | 2.823 (2/12) | 2.090 (4/12) | 1.414 (4/13) |
| MAP KS | 2.807 (0/11) | 2.183 (3/11) | 1.743 (2/12) |
| MAP χ² | 6.954 (0/9) | 4.280 (0/9) | 3.633 (0/9) |
| `q(0\|x)` AUC | 0.8103 | 0.8070 | 0.8181 |

**On MBR-decoded observable spectra `e_v2` is competitive with the reference and beats it on
two of three metrics** — KS 0.763 vs 0.860 and χ² 0.438 vs 0.917, with W1 a tie at 0.667 vs
0.656. All three are below 1, so it beats identity. `e_v1` does not: all three of its ratios
exceed 1. On MAP both edit stages remain clearly worse, so the effect is MBR-specific.

This is a genuine disagreement between instruments, and §7 reports it as one rather than
picking the reading that agrees with the gates.

**The caveat that belongs beside the win**, because it is the same defect the gates found:

| mean multiplicity | truth | identity | MAP | **MBR** | posterior |
|---|---:|---:|---:|---:|---:|
| `e_v1_s0` | 1.435 | 1.864 | 1.036 | **1.042** | 1.444 |
| `e_v2_s0` | 1.435 | 1.864 | 1.025 | **1.066** | 1.416 |
| `v1_contstop_s0` | 1.435 | 1.864 | 1.075 | **1.370** | 1.455 |

Every **posterior** mean is close to truth (1.416–1.455 vs 1.435) — the marginal is right, as
E3 found. But both edit stages' **MBR decode collapses to ~1.05** where the reference holds
1.370. So `e_v2` achieves its closure win with trees about 1.07 nodes long against a truth of
1.435: the observable *shapes* it does produce are good, and it produces too few of them.

**The irony is worth stating.** Plan §1 justified this family on precisely this point — the
open-ended continue/stop mechanism is "the seat of the marginal multiplicity bias and of MAP
collapse", removed *structurally* by anchoring `n_y` at `|x|`. The mechanism was removed and
the decode collapse is **worse** than in the arm that still has it (1.04–1.07 vs 1.370). The
collapse therefore does not live in the continue/stop head; removing it was treating a
symptom's suspected cause and the symptom got worse.

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
For the three deciding rows that caution is not load-bearing: the gaps are 5–50× the width
of either band. **For the decode-tier rows it is load-bearing, and it was measured.**

> ⚠️ **The decode-tier band separations are NOT stable under resampling.** Regenerating
> these artifacts under the batched sampler (§0.1) — a change that draws different samples
> from the same conditional and nothing else — left NLL, TARP and `coverage_68` untouched
> (the first is deterministic, the other two bit-identical, since they consume cell chains
> and never reach the coordinate sampler) and **flipped two band-separation verdicts**:
>
> | row | before | after |
> |---|---|---|
> | `<N>` ratio, this table | separated, edit worse | **tie — bands overlap** |
> | geo-median/identity, E8 | tie | **separated, `e_v2` better** |
>
> Both are decode-tier ratios from `run_closure` on 300 jets, where three seeds give bands
> ~0.02 wide and the deltas are ~0.01–0.02. A band that clears by less than its own width
> clears by luck. **No E-gate verdict moved**, because E4/E5/E6 are decided on TARP,
> coverage and NLL — but the `medoid/identity`, `geo-median/identity` and `<N>` rows of this
> table and of §4.8 should be read as descriptive, not as measurements, and
> `scripts/prod_test_edit_gates.py` currently prints `**yes**` for a separation of any
> width. Closing that would mean a resampling band, not a seed band — see §8.

## 6. What this run found

### 6.1 The question it was built to answer

*Does a third factorization of the length/shape coupling — one designed against v1's defect
before that defect was measured — beat the arm v1 picked?* **On the metrics that decided v1,
no.** Not on TARP, not on coverage, not on held-out NLL, not on any seed, and not by a margin
that any band or capacity argument reaches — for either stage.

The qualification, added after the fact and kept because it is true: those three metrics are
all **per-jet posterior calibration**, which is what v1's defect was and therefore what this
run was pre-registered to measure. On **marginal observable spectra under MBR decode** —
§4.13, not a plan §12 step — `edit_v2` is competitive with the reference and better on two of
three. The gates decide the question as posed; they do not decide every question.

**And on its own strongest ground it does not win either.** The edit family's `q(N|x)` is
exact, parameter-free and explicitly conditioned on `|x|`, where `v1_contstop`'s is a fitted
per-step continue/stop product — plan §1 put that contrast at the head of the table. The
exact marginal does not convert that into a better belief about `N`:

| `q(0\|x)` AUC | `e_v1_s0` | `e_v2_s0` | `v1_contstop_s0` |
|---|---:|---:|---:|
| deep pass, 97 018 jets | 0.7700 | 0.7677 | 0.824 |
| closure report, 2 000 jets, `len(x) > 0` | 0.8103 | 0.8070 | 0.8181 |

Read both rows before quoting either. The deficit is the **same in both stages** (0.770,
0.768), so it is a family property rather than a stage-1 artifact — but it is **0.05 on one
instrument and 0.008 on the other**, and the two differ in tier and in selection. That is
not a finding; it is a quantity whose value depends on how it is measured, and it is
recorded here so nobody quotes the larger number alone.

What the exact marginal *did* buy is the average: ⟨N⟩ ratio 0.9843 and posterior means of
1.416–1.444 against a truth of 1.435 (§4.13), the family's best numbers. What it did not buy
is the decode — see §4.13's multiplicity table, where both stages collapse to ~1.05 against
the reference's 1.370.

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

- **A batched `sample_coordinates_many` for the edit family — FIRED AND FIXED.** The base
  class loops the per-draw hook, so the support audit paid one full call per draw. Deferred
  mid-run on purpose (an override "is free to reorder RNG consumption… they are NOT the
  same draws", so landing it would have made post-fix audits non-comparable with the ones
  already written), then landed once the grid's artifacts were complete and regenerated
  together. What it took, and what it bought:

  | | |
  |---|---|
  | new code | `edit_dp.sample_alignment_batch` — B lattice walks in lockstep |
  | | `EditTransducer.sample_coordinates_many`, sharing `_coord_lattice` with the single-draw path |
  | speedup, K = 200 | **222x** (`edit_v1`), **39.5x** (`edit_v2`) |
  | `run_support_audit`, per arm | 13.8 min -> **3.3 min** |
  | `refresh_support_audit`, 9 arms | ~2 h 04 m -> **29.6 min** |
  | full eval, 9 arms @ concurrency 5 | ~55 min -> **26.9 min** |
  | single-draw path | **bit-identical** to before (max\|Δ\| = 0 over 480 values) |

  The diagnosis is worth keeping separately from the fix. The per-draw call is ~6 ms and
  its cost is **flat in `L` and flat in `n_x`** — 9.1 ms at `L = 1`, 13.3 ms at `L = 100` —
  so it is ~100 tiny op launches, not arithmetic, and the `O(n_x n_y)` recursion is
  irrelevant at these sizes. That rules out the obvious fix: hoisting the encoder out of
  the K-loop is worth **1.2x**, measured, and it was the *RNG-preserving* option. The only
  thing that helps is paying the fixed cost once per jet instead of once per draw.

  This is the fix [`PLAN_prod_test_speedup.md`](PLAN_prod_test_speedup.md) §2 already made
  for the AR family, plus the one piece AR never needed: its coordinates are conditionally
  independent given the cell chain, so its batched path is a padded teacher-forced replay,
  while here the alignment is **latent** and has to be sampled per draw before any
  coordinate can be. Hence `sample_alignment_batch`, and hence tests that check
  DISTRIBUTIONAL agreement (two-sample KS per coordinate, plus the alignment-column
  histogram) rather than bit-identity — `tests/test_edit_batched_coords.py`.
- **`physics_width = true` costs likelihood.** `e_v1_freewidth` beats all three physics-width
  seeds on NLL while E7 confirms the imposed form is correct. One seed; a proper multi-seed
  A/B would say whether the regularization is worth its cost.
- **`frac_anchored` ≈ 0.19 is the ceiling on this family's mechanism.** Anything that raises
  it — `p_anch` initialization, a stronger prior, an identifiability penalty — is where a
  follow-up would have to start, and until it moves, the family is an AR model with an
  expensive lattice attached.
- **The decode-side multiplicity collapse is NOT in the continue/stop head.** This family
  removed that head structurally and collapsed *harder* — MBR mean 1.04–1.07 against the
  reference's 1.370 and a truth of 1.435 (§4.13) — while its posterior mean stayed correct.
  So the collapse lives in the decision rule (MBR/MAP over a high-entropy sequence posterior)
  rather than in the length parametrization, which is where v0 and v1 both looked. That is a
  redirection for any follow-up on MAP collapse, and it is the most reusable thing this grid
  produced.
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

**Do not field the edit transducer — but the two instruments disagree, and the disagreement
is the more useful output.**

On what the pre-registered gates measure — **per-jet posterior calibration** — the family
loses without ambiguity. E4 (TARP), E5 (coverage) and E6 (NLL) fail unanimously across three
seeds for `edit_v1`, and `edit_v2` fails all three as well (TARP 0.0717 vs 0.0212, NLL 4.4697
vs 3.7927, coverage 0.5107 vs 0.5307). Bands do not overlap anywhere, and the edit arms are
the *smaller* models, so capacity does not rescue them. `v1_contstop` remains the
recommendation v1 made.

On **MBR-decoded observable spectra**, which the gates do not measure, `edit_v2` is
competitive with the reference and better on two of three metrics (§4.13: KS 0.763 vs 0.860,
χ² 0.438 vs 0.917, W1 0.667 vs 0.656). That does not overturn the verdict — a model whose
posterior is miscalibrated per jet is not fielded because its marginals look right — but it
does mean "the edit factorization is worse at everything" would be false, and this document
does not say it.

The reconciliation is in §4.13's multiplicity table: `edit_v2` gets good observable *shapes*
out of MBR trees that are ~1.07 nodes long against a truth of 1.435. Right shapes, too few
nodes. A closure metric pooled over emissions can be satisfied that way; a per-jet posterior
cannot.

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

**The candidate transferable negative result** is §4.11's: on `e_v1_s0`, an exact,
parameter-free `q(N|x)` was *worse* at ranking empty jets than a fitted continue/stop head
(AUC 0.770 vs 0.827), while being better at the marginal ⟨N⟩. If §4.12 reproduces it on
`e_v2_s0` then structural exactness is not a substitute for a discriminative fit, and anyone
reaching for "make the length model exact" as a remedy for a calibration defect should read
that number first. On one arm of one stage it is a lead, not a law — stated here as such.

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
- **Distribution closure on more than one seed.** §4.13 is `e_v1_s0` and `e_v2_s0` against
  `v1_contstop_s0` — one seed each, so it carries no band. `e_v2`'s χ² win (0.438 vs 0.917)
  is large enough that a seed band is unlikely to erase it, but that is an expectation and
  not a measurement.
- **Why the two `q(0|x)` AUC instruments disagree** (0.77 on the deep pass, 0.81 on the
  closure, for the same arms). Tier and selection both differ; which of the two accounts for
  it was not isolated.
- **Whether `edit_v2`'s closure win survives a decode that does not collapse.** Its MBR trees
  are ~1.07 nodes long against a truth of 1.435 (§4.13); nothing here separates "good
  observable shapes" from "good shapes because too few nodes were emitted to get them wrong".
- **A resampling band on the decode-tier ratios.** §5 shows two band-separation verdicts
  flipping when the same model is re-evaluated with different draws, so the seed band alone
  understates the uncertainty on `medoid/identity`, `geo-median/identity` and `<N>` at 300
  jets. The right instrument is the seed band *convolved with* a resampling band — several
  evaluations of one checkpoint — and this run has exactly two evaluations, which is enough
  to show the problem and not enough to quantify it.
</content>
