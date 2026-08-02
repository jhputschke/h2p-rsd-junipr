#!/usr/bin/env bash
# The §7 grid of docs/PLAN_prod_test_edit.md, as one command.
#
#   bash scripts/run_prod_test_edit.sh [--concurrency N] [--smoke] [--only ARM[,ARM...]]
#
# Every arm is `presets/prod_test_edit.yaml` plus the ONE override that names it, so the
# arms differ by exactly the variable in the plan's table and nothing else — the same
# shape scripts/run_prod_test_v1.sh uses. Seeds are `trainer.seed` at a fixed
# `data.seed = 0`, so the band measures initialisation and batch ordering, NOT split
# variance (plan §7).
#
# The REFERENCE is not retrained. `runs/prod_test_v1/v1_contstop_s0` and `_s1` already
# exist; WP-F.1 re-EVALUATES them on this run's code path and device so the head-to-head
# is one evaluation pass:
#
#   bash scripts/eval_prod_test_v1.sh --run-root runs/prod_test_v1 --device cpu \
#        --only v1_contstop_s0,v1_contstop_s1
#
# --smoke runs every arm end to end on tiny data (~2 min) and is where `edit_v2`'s peak
# memory at n_bins = 30 gets measured (plan §9): the free-cell head is evaluated at
# (B, n_x+1, n_y, 900), which `edit_v1` collapses to T = 1 and is ~n_y times cheaper.
#
# Logs land in $RUN_ROOT/logs/<arm>.log and the run directories under $RUN_ROOT/<arm>/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_ROOT="runs/prod_test_edit"
CONCURRENCY=4
SMOKE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --smoke)       SMOKE=1; shift ;;
    --only)        ONLY="$2"; shift 2 ;;
    -h|--help)     sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# arm | overrides (space separated).  The plan's §7 table, one line each.
#
# `e_v1`      the deliverable candidate: edit_v1 + lundnet + aux(9) + physical ln z.
# `e_v2`      stage 2. TRAINED unconditionally but QUOTED conditionally: gate E7 is a
#             stage gate, and a flat production width fit demotes every e_v2 number to
#             null context (plan §3, §8).
# `e_v1_legacy_lnz`  attribution, not an experiment — it must reproduce the v0 support
#             failure under identical data, exactly as `v1_legacy_lnz` did.
# `e_v1_freewidth`   the READOUT arm for E7: Lambda_eff quoted from the arm that was NOT
#             told the functional form.
# `e_v1_gru`  one training, licensing only "worth a proper multi-seed A/B" — v1 §3.2's
#             discipline. It is here because v1 found `gru` moved TARP and coverage in the
#             same direction `v1_contstop` did.
ARMS=(
  "e_v1_s0|trainer.seed=0"
  "e_v1_s1|trainer.seed=1"
  "e_v1_s2|trainer.seed=2"
  "e_v2_s0|trainer.seed=0 model=edit_v2 model.prefix_conditioning=true"
  "e_v2_s1|trainer.seed=1 model=edit_v2 model.prefix_conditioning=true"
  "e_v2_s2|trainer.seed=2 model=edit_v2 model.prefix_conditioning=true"
  "e_v1_legacy_lnz_s0|trainer.seed=0 model.lnz_support=legacy"
  "e_v1_freewidth_s0|trainer.seed=0 model.physics_width=false"
  "e_v1_gru_s0|trainer.seed=0 encoder=gru"
)

EXTRA=""
if [[ "$SMOKE" == "1" ]]; then
  # A real end-to-end pass of every arm in ~2 min: same code path, tiny data.
  EXTRA="trainer=fast_dev data.n_jets=512 experiment.closure_jets=16 experiment.n_closure_samples=8"
  RUN_ROOT="runs/prod_test_edit_smoke"
fi

mkdir -p "$RUN_ROOT/logs"
echo "[grid] $RUN_ROOT, concurrency $CONCURRENCY, ${#ARMS[@]} arms$([[ $SMOKE == 1 ]] && echo ' (SMOKE)')"

pids=()
names=()
launch() {
  local arm="$1" over="$2"
  local log="$RUN_ROOT/logs/$arm.log"
  echo "[grid] launching $arm  ($over)"
  # /usr/bin/time -v gives peak RSS per arm, which is the edit_v2 memory number the plan
  # asks the smoke pass for. Absent (macOS), fall back to running the command bare.
  local timer=()
  if [[ "$SMOKE" == "1" ]] && /usr/bin/time -v true >/dev/null 2>&1; then
    timer=(/usr/bin/time -v)
  fi
  # shellcheck disable=SC2086
  "${timer[@]}" h2p-rsd-junipr train base=presets/prod_test_edit.yaml \
      run_root="$RUN_ROOT/$arm" $over $EXTRA > "$log" 2>&1 &
  pids+=($!)
  names+=("$arm")
}

running() {
  local n=0 p
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done
  echo "$n"
}

for entry in "${ARMS[@]}"; do
  arm="${entry%%|*}"; over="${entry#*|}"
  if [[ -n "$ONLY" && ",$ONLY," != *",$arm,"* ]]; then continue; fi
  while [[ "$(running)" -ge "$CONCURRENCY" ]]; do sleep 20; done
  launch "$arm" "$over"
  sleep 2                       # stagger the RNTuple reads
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[grid] FAILED: ${names[$i]}  (see $RUN_ROOT/logs/${names[$i]}.log)"
    fail=1
  fi
done

echo "[grid] done. best val NLL per arm:"
for entry in "${ARMS[@]}"; do
  arm="${entry%%|*}"
  log="$RUN_ROOT/logs/$arm.log"
  [[ -f "$log" ]] && printf '  %-22s %s\n' "$arm" "$(grep -o 'best val NLL/jet = [0-9.]*' "$log" | tail -1)"
done

if [[ "$SMOKE" == "1" ]]; then
  echo "[grid] peak RSS per arm (the edit_v2 memory measurement, plan §9):"
  for entry in "${ARMS[@]}"; do
    arm="${entry%%|*}"
    log="$RUN_ROOT/logs/$arm.log"
    [[ -f "$log" ]] && printf '  %-22s %s\n' "$arm" \
      "$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "$log" | tail -1)"
  done
fi
exit "$fail"
