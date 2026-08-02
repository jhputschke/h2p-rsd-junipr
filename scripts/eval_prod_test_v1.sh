#!/usr/bin/env bash
# Evaluate every arm of the v1 grid on the independent test file (plan §13.3), with the
# WP-D switches the gates need.
#
#   bash scripts/eval_prod_test_v1.sh [--concurrency N] [--run-root DIR] [--only ARM,...]
#                                     [--device cpu|auto|cuda]
#
# --device DEFAULTS TO cpu, and changing it is a whole-grid decision, not a per-arm one.
# `scripts/prod_test_v1_gates.py` compares the arms TO EACH OTHER, and cpu and cuda are a
# different RNG stream *and* different float kernels — so a half-cpu/half-cuda grid is a
# silent ranking hazard. Flipping this flag means re-running all 11 arms.
#   cpu   `CUDA_VISIBLE_DEVICES=""` — the only lever there is, since `eval` has no device
#         flag of its own (cli.py calls `select_device()` unconditionally) — plus the
#         CPU-only `OMP_NUM_THREADS=2` that keeps the concurrent arms from thrashing.
#   auto  neither is set, so `select_device()` picks: cuda > mps > cpu.
#   cuda  the same, and fails loudly if torch cannot see a GPU rather than falling back.
# The device that actually ran is recorded in each arm's `eval_metrics.json` under
# `device`, so a mixed grid is at least detectable after the fact.
#
# TWO passes per arm, because one cannot be afforded:
#
#   A. calibration tier — 2000 jets, MAP decode. Gates G2, G3, G4, G5, G7. The 2000-jet
#      tier is not a preference: TARP's recomputed null band has a 95% point of 0.073 at
#      n = 300 and 0.028 at n = 2000, and G7 requires that floor below 0.05 before the
#      statistic may be quoted at all.
#   B. decode tier — 300 jets, MBR decode, coordinates off the cell grid. Gates G1, G6.
#      MBR is O(n_candidates x K) optimal-transport solves PER JET; at the calibration
#      tier with a full candidate set that is 8e7 solves per arm, so the decode questions
#      are asked on the tier they can be asked on, exactly as v0's notebook tiered them.
#
# Each pass writes `eval_metrics.json` beside the checkpoint, so they are captured under
# distinct names and merged: the merged file takes calibration/support/exposure from A and
# closure from B, and records the tier each came from under `tiers`.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_ROOT="runs/prod_test_v1"
TEST_FILE="data/jet_aux_asym_test.root"
CONCURRENCY=6
ONLY=""
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --run-root)    RUN_ROOT="$2"; shift 2 ;;
    --test-file)   TEST_FILE="$2"; shift 2 ;;
    --only)        ONLY="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    -h|--help)     sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$DEVICE" in
  cpu)  EVAL_ENV=(CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2) ;;
  auto) EVAL_ENV=() ;;
  cuda)
    EVAL_ENV=()
    # `eval` has no device flag, so "cuda" cannot be *requested* — only left available.
    # Refuse up front rather than silently producing a cpu grid labelled cuda.
    python - <<'PY' || exit 2
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("--device cuda: torch.cuda.is_available() is False — nothing would use a GPU.")
PY
    ;;
  *) echo "--device must be cpu, auto or cuda (got: $DEVICE)" >&2; exit 2 ;;
esac

COMMON="data.path=$TEST_FILE experiment.pit_coords=true experiment.stratify_regions=true"

# Pass A — the calibration tier.
PASS_A="$COMMON experiment.closure_jets=2000 experiment.n_closure_samples=200 \
  experiment.tarp=true experiment.tarp_refs=200 experiment.tarp_null_reps=4000 \
  experiment.tarp_stratify=true experiment.support_audit=true \
  experiment.exposure_diagnostic=true"

# Pass B — the decode tier. `min_emissions=0` keeps MBR floor-free, as v0 quoted it.
PASS_B="$COMMON experiment.closure_jets=300 experiment.n_closure_samples=200 \
  experiment.closure_continuous=true experiment.support_audit=true \
  decode.point_estimator=mbr decode.mbr_backend=pot decode.mbr_n_candidates=64 \
  decode.min_emissions=0"

mkdir -p "$RUN_ROOT/logs"
mapfile -t CKPTS < <(ls -d "$RUN_ROOT"/*/*/best.ckpt 2>/dev/null)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "no best.ckpt under $RUN_ROOT — has the grid finished?" >&2
  exit 1
fi
echo "[eval] ${#CKPTS[@]} checkpoints, concurrency $CONCURRENCY, test file $TEST_FILE, device $DEVICE"

merge_py() {
python - "$1" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
a = json.loads((d / "eval_metrics_calib.json").read_text())
b_path = d / "eval_metrics_decode.json"
out = dict(a)
out["tiers"] = {"calibration": {"n_jets": a["experiment"]["closure_jets"],
                                "decode": a["decode"]["point_estimator"]}}
if b_path.is_file():
    b = json.loads(b_path.read_text())
    # closure (and the psi block inside it) comes from the DECODE tier; everything
    # calibration-shaped stays on the tier it was measured at, and both are named.
    out["closure"] = b["closure"]
    out["decode"] = b["decode"]
    out["decode_inert"] = b["decode_inert"]
    out["support_audit_decode_tier"] = b.get("support_audit")
    out["tiers"]["decode"] = {"n_jets": b["experiment"]["closure_jets"],
                              "decode": b["decode"]["point_estimator"],
                              "mbr_n_candidates": b["decode"].get("mbr_n_candidates")}
(d / "eval_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"[eval] merged -> {d / 'eval_metrics.json'}")

# Regenerate the figures FROM THE MERGED RECORD. Each pass writes `calibration_*.png`
# beside the checkpoint, so pass B (300 jets) was overwriting pass A's (2000) and the
# artifact ended up with figures and JSON describing different tiers -- the region panel
# read `narrow_soft n=18, not scored` next to a JSON saying `n=111, scored`. Redrawing
# from `out` makes the two consistent by construction rather than by discipline.
import sys
sys.path.insert(0, "src")
from h2p_rsd_junipr.eval.report import plot_calibration
figs = plot_calibration(out["calibration"], d)
print(f"[eval] redrew {len(figs)} figure(s) from the merged record "
      f"(calibration tier: {out['tiers']['calibration']['n_jets']} jets)")
PY
}

one_arm() {
  local ckpt="$1"
  local dir arm
  dir="$(dirname "$ckpt")"
  arm="$(basename "$(dirname "$dir")")"
  local log="$RUN_ROOT/logs/eval_$arm.log"
  {
    echo "===== PASS A (calibration tier) ====="
    # shellcheck disable=SC2086
    env "${EVAL_ENV[@]}" h2p-rsd-junipr eval "$ckpt" $PASS_A \
      && mv "$dir/eval_metrics.json" "$dir/eval_metrics_calib.json"
    echo "===== PASS B (decode tier) ====="
    # shellcheck disable=SC2086
    env "${EVAL_ENV[@]}" h2p-rsd-junipr eval "$ckpt" $PASS_B \
      && mv "$dir/eval_metrics.json" "$dir/eval_metrics_decode.json"
  } > "$log" 2>&1
  merge_py "$dir" >> "$log" 2>&1
  echo "[eval] $arm done -> $log"
}

pids=()
for ckpt in "${CKPTS[@]}"; do
  arm="$(basename "$(dirname "$(dirname "$ckpt")")")"
  if [[ -n "$ONLY" && ",$ONLY," != *",$arm,"* ]]; then continue; fi
  while [[ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]]; do sleep 15; done
  echo "[eval] launching $arm"
  one_arm "$ckpt" &
  pids+=($!)
  sleep 2
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[eval] all done (fail=$fail)"
exit "$fail"
