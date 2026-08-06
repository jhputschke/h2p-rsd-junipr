#!/usr/bin/env bash
# The measurement of docs/PLAN_lnz_spline_head.md §3, as one command.
#
#   bash scripts/run_lnz_spline.sh [--concurrency N] [--smoke] [--only ARM[,ARM...]]
#
# Every arm is `presets/prod_test_v1.yaml` — the SAME preset the v1 grid was trained from,
# so the data file, geometry, encoder, aux columns, epochs and batch size are held fixed —
# plus `model.lnz_head=spline` and the one override that names the arm. That is what makes
# the control free: `runs/prod_test_v1/v1_base_s{0,1,2}` are the same configuration with
# the same seeds and the truncated-normal head, already trained and already evaluated, and
# the spline change is bit-identical off its own switch (tests/test_lnz_spline.py).
#
# Arms:
#   spline_s0/s1/s2       the pre-registered 3-seed re-test of gate G3 against the recorded
#                         1.05-2.07x. Compare seed to seed with v1_base_s0/s1/s2.
#   contstop_spline_s0    NOT part of the pre-registered gate: a transfer check. `v1_base`
#                         is the explicit-q(N|x) family, which fails G7/TARP and is not what
#                         is fielded; the continue/stop family is. The ln z head is shared,
#                         so this asks whether a coordinate fix measured on one carries to
#                         the other. Its control is `runs/prod_test_v1/v1_contstop_s0`.
#
# Then evaluate with the SAME two-tier command the v1 campaign used, so the PIT numbers are
# produced by the same code path as the numbers they are compared against:
#
#   bash scripts/eval_prod_test_v1.sh --run-root runs/lnz_spline --device cpu
#   python scripts/lnz_spline_gates.py
#
# Timing reference: the v1 grid ran four of these trainings at once on one GB10 in ~58 min.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_ROOT="runs/lnz_spline"
CONCURRENCY=4
SMOKE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --smoke)       SMOKE=1; shift ;;
    --only)        ONLY="$2"; shift 2 ;;
    -h|--help)     sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# §7.1 / §7.2 arms are launched from the SAME script and preset so the whole comparison
# stays one grid. `EXTRA` carries only what every arm shares; each arm's own line carries
# what makes it that arm.
ARMS=(
  "spline_s0|trainer.seed=0"
  "spline_s1|trainer.seed=1"
  "spline_s2|trainer.seed=2"
  "contstop_spline_s0|trainer.seed=0 model.use_multiplicity_head=false"
  # --- §10 (docs/PLAN_next_steps.md B1): two MORE continue/stop spline seeds, so the
  #     FIELDED family has three paired seeds instead of one. §6.4 measured the spline's
  #     TARP gain on the explicit-q(N|x) family (2 of 3 crossing below the null band) and
  #     the single continue/stop arm moved the OTHER way, 0.0200 -> 0.0255. Their control
  #     for seed 2 does not exist either -- `scripts/run_prod_test_v1.sh` grows
  #     `v1_contstop_s2` in the same commit. Verdict rule: §10.3, fixed first.
  "contstop_spline_s1|trainer.seed=1 model.use_multiplicity_head=false"
  "contstop_spline_s2|trainer.seed=2 model.use_multiplicity_head=false"
  # --- §7.1: the dv spline, on top of the ln z spline, same three seeds -------------
  "dvspline_s0|trainer.seed=0 model.dv_head=spline"
  "dvspline_s1|trainer.seed=1 model.dv_head=spline"
  "dvspline_s2|trainer.seed=2 model.dv_head=spline"
  # --- §7.2a: three MORE ln z-spline seeds. 3 seeds cannot tell "one marginal seed"
  #     from "a 1-in-3 failure rate"; 6 can. Same config as spline_s0/1/2 exactly.
  "spline_s3|trainer.seed=3"
  "spline_s4|trainer.seed=4"
  "spline_s5|trainer.seed=5"
  # --- §7.2b: is seed 2's 4% miss expressiveness or variance? K was chosen, not fitted.
  #     ONE arm, on the seed that missed, and K is NOT tuned across seeds afterwards.
  "spline_k16_s2|trainer.seed=2 model.lnz_spline_bins=16"
  # --- §8.5(1): the CONDITIONING experiment. The dv spline failed because the residual
  #     is a per-cell LOCATION bias identical under both density families — a limit on
  #     what the head can predict, not express. The cell reaches it only as a learned
  #     embedding of a categorical id; this adds the cell's continuous centre. Control is
  #     spline_s0/1/2, so the row prices the conditioning change alone.
  "cellctr_s0|trainer.seed=0 model.coord_cell_center=true"
  "cellctr_s1|trainer.seed=1 model.coord_cell_center=true"
  "cellctr_s2|trainer.seed=2 model.coord_cell_center=true"
  # --- §8.5(2): does the dv spline pay ONCE the conditioning is fixed? One seed, because
  #     it only becomes a question if cellctr_* moves the per-cell bias at all.
  "cellctr_dvspline_s0|trainer.seed=0 model.coord_cell_center=true model.dv_head=spline"
)

EXTRA="model.lnz_head=spline"
if [[ "$SMOKE" == "1" ]]; then
  # A real end-to-end pass of every arm in ~2 min: same code path, tiny data.
  EXTRA="$EXTRA trainer=fast_dev data.n_jets=512 experiment.closure_jets=16 \
         experiment.n_closure_samples=8"
  RUN_ROOT="runs/lnz_spline_smoke"
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
