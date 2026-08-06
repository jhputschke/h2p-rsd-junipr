#!/usr/bin/env bash
# The §8 grid of docs/PLAN_prod_test_v1.md, as one command.
#
#   bash scripts/run_prod_test_v1.sh [--concurrency N] [--smoke] [--only ARM[,ARM...]]
#
# Every arm is `presets/prod_test_v1.yaml` plus the ONE override that names it, so the
# arms differ by exactly the variable in the plan's table and nothing else. Seeds are
# `trainer.seed`, which also drives the data split.
#
# Concurrency: the v0 grid ran four of these trainings at once on one GB10 and finished
# in ~58 min wall clock; the model is 265k parameters, so the bottleneck is the input
# pipeline rather than the GPU. Default 4 reproduces that. Raise it if the box is idle.
#
# Logs land in $RUN_ROOT/logs/<arm>.log and the run directories under $RUN_ROOT/<arm>/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_ROOT="runs/prod_test_v1"
CONCURRENCY=4
SMOKE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --smoke)       SMOKE=1; shift ;;
    --only)        ONLY="$2"; shift 2 ;;
    -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# arm | overrides (space separated).  The plan's table, one line each.
#
# `v1_contstop` is the plan's `v1_nhead` arm under the name of what it actually varies.
# The plan's table lists `v1_base` as v4 and `v1_nhead` as "explicit q(N|x)" — but
# ar_junipr_v4 ALREADY sets use_multiplicity_head=true, so those two arms would be the
# same model. Gate G8's rationale ("SBC-N is calibrated for v3's n_head nearly by
# construction, so it must not decide") only means anything if one arm HAS the head and
# the other does not, so the missing arm is the implicit continue/stop one. G8 is
# evaluated as `v1_base` (explicit) vs `v1_contstop` (implicit), on the metrics the gate
# names. See docs/PROD_TEST_v1_RESULTS.md §0.
ARMS=(
  "v1_base_s0|trainer.seed=0"
  "v1_base_s1|trainer.seed=1"
  "v1_base_s2|trainer.seed=2"
  "v1_legacy_lnz_s0|trainer.seed=0 model.lnz_support=legacy"
  "v1_ctrl_s0|trainer.seed=0 encoder.aux_features=[ln_pt,abs_eta]"
  "v1_ctrl_s1|trainer.seed=1 encoder.aux_features=[ln_pt,abs_eta]"
  "v1_ctrl_s2|trainer.seed=2 encoder.aux_features=[ln_pt,abs_eta]"
  "v1_contstop_s0|trainer.seed=0 model.use_multiplicity_head=false"
  "v1_contstop_s1|trainer.seed=1 model.use_multiplicity_head=false"
  # The seed-2 control the v1 grid never needed and docs/PLAN_lnz_spline_head.md §10 does:
  # its `contstop_spline_s2` arm has to be paired with the SAME configuration and seed
  # under the truncated-normal head, and this is that arm.
  "v1_contstop_s2|trainer.seed=2 model.use_multiplicity_head=false"
  "v1_gru_s0|trainer.seed=0 encoder=gru"
  "v1_deepsets_s0|trainer.seed=0 encoder=deepsets"
)

EXTRA=""
if [[ "$SMOKE" == "1" ]]; then
  # A real end-to-end pass of every arm in ~2 min: same code path, tiny data.
  EXTRA="trainer=fast_dev data.n_jets=512 experiment.closure_jets=16 experiment.n_closure_samples=8"
  RUN_ROOT="runs/prod_test_v1_smoke"
fi

mkdir -p "$RUN_ROOT/logs"
echo "[grid] $RUN_ROOT, concurrency $CONCURRENCY, ${#ARMS[@]} arms$([[ $SMOKE == 1 ]] && echo ' (SMOKE)')"

pids=()
names=()
launch() {
  local arm="$1" over="$2"
  local log="$RUN_ROOT/logs/$arm.log"
  echo "[grid] launching $arm  ($over)"
  # shellcheck disable=SC2086
  h2p-rsd-junipr train base=presets/prod_test_v1.yaml \
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
  [[ -f "$log" ]] && printf '  %-20s %s\n' "$arm" "$(grep -o 'best val NLL/jet = [0-9.]*' "$log" | tail -1)"
done
exit "$fail"
