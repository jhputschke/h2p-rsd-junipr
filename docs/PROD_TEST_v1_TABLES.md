## Arms

| arm | model | encoder | `lnz_support` | `q(N\|x)` head | aux | best val NLL/jet | eval jets |
|---|---|---|---|---|---:|---:|---:|
| `v1_base_s0` | ar_junipr_v4 | lundnet | `physical` | yes | 9 | 3.9235 | 97018 |
| `v1_base_s1` | ar_junipr_v4 | lundnet | `physical` | yes | 9 | 3.9036 | 97018 |
| `v1_base_s2` | ar_junipr_v4 | lundnet | `physical` | yes | 9 | 3.9237 | 97018 |
| `v1_contstop_s0` | ar_junipr_v4 | lundnet | `physical` | no | 9 | 3.7799 | 97018 |
| `v1_contstop_s1` | ar_junipr_v4 | lundnet | `physical` | no | 9 | 3.8054 | 97018 |
| `v1_ctrl_s0` | ar_junipr_v4 | lundnet | `physical` | yes | 2 | 3.9386 | 97018 |
| `v1_ctrl_s1` | ar_junipr_v4 | lundnet | `physical` | yes | 2 | 3.9133 | 97018 |
| `v1_ctrl_s2` | ar_junipr_v4 | lundnet | `physical` | yes | 2 | 3.9124 | 97018 |
| `v1_deepsets_s0` | ar_junipr_v4 | deepsets | `physical` | yes | 9 | 3.8855 | 97018 |
| `v1_gru_s0` | ar_junipr_v4 | gru | `physical` | yes | 9 | 3.8988 | 97018 |
| `v1_legacy_lnz_s0` | ar_junipr_v4 | lundnet | `legacy` | yes | 9 | 4.0703 ! | 97018 |

`!` marks an NLL that is **not comparable** to the rows without it: a different `ln z` normalization shifts NLL/jet by a constant unrelated to fit quality. NLL *is* comparable between the explicit-`q(N|x)` and continue/stop arms — both are normalized densities over the same space — as long as their `ln z` heads match.

## Gates (on `v1_base_s0`)

| gate | verdict | numbers |
|---|---|---|
| G1 acceptance | **PASS** | cell medoid/identity = 0.924, off-grid geo-median/identity = 0.932 |
| G2 support | **PASS** | out_of_window 0.0000%, soft_drop 0.0000%, z_above_half 0.0000%, kt_floor 0.0000% |
| G3 `ln z` PIT | **FAIL** | KS 0.0529 vs crit 0.0255 on 2834 emissions |
| G4 N marginal | **FAIL** | SBC-on-N chi2 223.3 at the 95th percentile of its own null (95% point 223.3) -> FAIL; <N>_post/<N>_truth = 1.0075 (full population, 300 jets) -> pass; leading-cell 68% coverage Wilson-consistent in 0/2 scoreable regions (fails: wide_soft, narrow_soft) |
| G5 `narrow_soft` | **ATTRIBUTED** | coverage 0.479 [0.38, 0.58] on n = 96, scored = True; worst coordinate there: ln_z KS 0.090 on n = 111 = 0.69x its critical value; NO coordinate exceeds its critical value here, so the coverage deficit is NOT a coordinate miscalibration in this quadrant — it is a structural/multiplicity effect, and that is the documented attribution |
| G6 decode | **PASS (psi clause underpowered)** | MBR/identity = 0.965 -> pass; psi |R| point 0.0401 (uniform floor 0.0449, Rayleigh p 0.53) vs truth 0.0781 (floor 0.0429, p 0.07) -- at least one row is consistent with UNIFORM, so the 0.51x ratio is not a measurement. The gate's substance (no manufactured anisotropy) is met: the point estimate is at or below its own uniform floor; psi mode unidentified for 0.0% of nodes (kappa_min_mode = 0.5), coordinates carried as 'sample' |
| G7 TARP | **FAIL** | max dev 0.041 vs null 95% 0.027 at n = 2000; floor < 0.05 => quotable |
| G8 family A/B | n/a | pit_ks_max: v1_base 0.0529 vs v1_contstop 0.0482 -> v1_contstop; tarp_max_dev: v1_base 0.0415 vs v1_contstop 0.0200 -> v1_contstop; coverage_68: v1_base 0.5400 vs v1_contstop 0.5310 -> v1_base; SBC-N percentile v1_base 95 vs v1_contstop n/a (NON-DECIDING) |

### The same gates on every seed

| gate | `v1_base_s0` | `v1_base_s1` | `v1_base_s2` | band |
|---|---|---|---|---|
| G1 | **PASS** | **PASS** | **PASS** | unanimous |
| G2 | **PASS** | **PASS** | **PASS** | unanimous |
| G3 | **FAIL** | **FAIL** | **FAIL** | unanimous |
| G4 | **FAIL** | **FAIL** | **FAIL** | unanimous |
| G5 | **ATTRIBUTED** | **ATTRIBUTED** | **ATTRIBUTED** | unanimous |
| G6 | **PASS (psi clause underpowered)** | **PASS (psi clause underpowered)** | **PASS (psi clause underpowered)** | unanimous |
| G7 | **FAIL** | **FAIL** | **FAIL** | unanimous |

| quantity | `v1_base_s0` | `v1_base_s1` | `v1_base_s2` | criterion |
|---|---|---|---|---|
| `ln z` PIT KS | 0.0529 | 0.0270 | 0.0471 | <= 1.36/sqrt(n) |
| TARP max dev | 0.0415 | 0.0350 | 0.0335 | <= its recomputed null |
| TARP null 95% | 0.0275 | 0.0275 | 0.0275 | — |
| `coverage_68` | 0.5400 | 0.5185 | 0.5179 | Wilson-consistent with 0.68 |
| `<N>` ratio | 1.0075 | 0.9939 | 0.9866 | [0.95, 1.05] |

G8 has no verdict column by design: it is a comparison whose deciding metrics are listed, not a threshold. SBC-N is reported and does not decide (the explicit-`q(N|x)` arm is calibrated on it nearly by construction).

## G3 attribution — the `legacy` arm must still fail

- `v1_base` (physical): KS 0.0529 vs crit 0.0255 on 2834 emissions
- `v1_legacy_lnz`: KS 0.0734 vs crit 0.0255 on 2834 emissions -> still fails, as required
- support audit on the legacy arm: out_of_window 0.0000%, soft_drop 0.8123%, z_above_half 3.9800%, kt_floor 0.0000%

## Aux isolation — `v1_base` (9 columns) vs `v1_ctrl` (`ln_pt`, `abs_eta`)

| quantity | `v1_base` mean [band] | `v1_ctrl` mean [band] | delta | clears the spread? |
|---|---|---|---:|---|
| best val NLL/jet | 3.9169 [3.9036, 3.9237] (n=3) | 3.9215 [3.9124, 3.9386] (n=3) | -0.0045 | no — inside the seed spread |
| `ln z` PIT KS | 0.0423 [0.0270, 0.0529] (n=3) | 0.0466 [0.0382, 0.0592] (n=3) | -0.0043 | no — inside the seed spread |
| `pit_ks_max` | 0.0441 [0.0324, 0.0529] (n=3) | 0.0466 [0.0382, 0.0592] (n=3) | -0.0025 | no — inside the seed spread |
| TARP max dev | 0.0367 [0.0335, 0.0415] (n=3) | 0.0360 [0.0340, 0.0395] (n=3) | +0.0007 | no — inside the seed spread |
| `coverage_68` | 0.5255 [0.5179, 0.5400] (n=3) | 0.5366 [0.5280, 0.5507] (n=3) | -0.0111 | no — inside the seed spread |
| medoid/identity | 0.9312 [0.9240, 0.9373] (n=3) | 0.9386 [0.9309, 0.9504] (n=3) | -0.0073 | no — inside the seed spread |

A delta that does not clear the seed spread is not a measurement of the aux columns; it is a measurement of the seed. Plan §12's WP3 trigger — *the secondary columns carry the aux gain* — requires a delta that clears it.

## G8 family A/B — explicit `q(N|x)` vs implicit continue/stop

| quantity | `v1_base` mean [band] | `v1_contstop` mean [band] | delta | clears the spread? |
|---|---|---|---:|---|
| best val NLL/jet | 3.9169 [3.9036, 3.9237] (n=3) | 3.7927 [3.7799, 3.8054] (n=2) | +0.1242 | **yes** |
| `ln z` PIT KS | 0.0423 [0.0270, 0.0529] (n=3) | 0.0398 [0.0315, 0.0482] (n=2) | +0.0025 | no — inside the seed spread |
| `pit_ks_max` | 0.0441 [0.0324, 0.0529] (n=3) | 0.0398 [0.0315, 0.0482] (n=2) | +0.0043 | no — inside the seed spread |
| TARP max dev | 0.0367 [0.0335, 0.0415] (n=3) | 0.0212 [0.0200, 0.0225] (n=2) | +0.0154 | **yes** |
| `coverage_68` | 0.5255 [0.5179, 0.5400] (n=3) | 0.5307 [0.5304, 0.5310] (n=2) | -0.0053 | no — inside the seed spread |
| medoid/identity | 0.9312 [0.9240, 0.9373] (n=3) | 0.9307 [0.9286, 0.9327] (n=2) | +0.0006 | no — inside the seed spread |

**SBC-on-N is reported below and does not decide**, per gate G8: the explicit head is trained by direct NLL on `N`, so it is calibrated on that statistic nearly by construction and an A/B judged on it is biased toward it. The deciding metrics are the coordinate PITs, TARP, coverage and held-out NLL — and unlike the `ln z` head change, NLL **is** comparable here: both factorizations are normalized densities over the same space.

## Encoder probe — `lundnet` vs `gru` vs `deepsets`

**One seed each.** The plan budgets a single training per encoder, so these rows carry no band of their own and the only honest yardstick is the `v1_base` band at the same configuration. A difference inside that band is not a difference between encoders — v0 §10 left this probe open precisely because the earlier attempt could not clear it.

| quantity | `v1_base` (lundnet, 3 seeds) | `v1_gru` (1 seed) | `v1_deepsets` (1 seed) | any outside the band? |
|---|---|---|---|---|
| best val NLL/jet | 3.9169 [3.9036, 3.9237] | 3.8988 | 3.8855 | `v1_gru`, `v1_deepsets` |
| `ln z` PIT KS | 0.0423 [0.0270, 0.0529] | 0.0537 | 0.0513 | `v1_gru` |
| `pit_ks_max` | 0.0441 [0.0324, 0.0529] | 0.0537 | 0.0513 | `v1_gru` |
| TARP max dev | 0.0367 [0.0335, 0.0415] | 0.0275 | 0.0360 | `v1_gru` |
| `coverage_68` | 0.5255 [0.5179, 0.5400] | 0.5519 | 0.5328 | `v1_gru` |
| medoid/identity | 0.9312 [0.9240, 0.9373] | 0.9254 | 0.9253 | no |

## Seed bands

| arm | seeds | `dlund_medoid/identity` | `pit_ks_max` | `coverage_68` |
|---|---:|---|---|---|
| `v1_base` | 3 | 0.923978–0.937290 | 0.032354–0.052859 | 0.517900–0.539976 |
| `v1_contstop` | 2 | 0.928644–0.932725 | 0.031456–0.048200 | 0.530430–0.531026 |
| `v1_ctrl` | 3 | 0.930868–0.950432 | 0.038165–0.059240 | 0.528043–0.550716 |

