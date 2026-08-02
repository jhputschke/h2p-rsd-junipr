#!/usr/bin/env bash
# Evaluate v1 grid arms AS THEY FINISH, in parallel with the arms still training.
#
#   bash scripts/eval_prod_test_v1_stream.sh [--concurrency N] [--expect 11]
#
# Training is GPU-bound and evaluation is run on the CPU here (`eval_prod_test_v1.sh`
# defaults to `--device cpu`, i.e. `CUDA_VISIBLE_DEVICES=""`), so the two barely contend —
# waiting for the whole grid before starting any eval leaves ~20 cores idle for hours.
# That is a CHOICE made for this concurrency, not a property of `eval`: `h2p-rsd-junipr
# eval` calls `select_device()` and runs on cuda wherever one is visible. Streaming
# alongside training is exactly the case where cpu is the right choice, so this script
# does not expose `--device` — pass it to `eval_prod_test_v1.sh` directly, and only for a
# whole grid at once (see its header). This polls for arms that have a `best.ckpt` and no
# merged `eval_metrics.json` yet, and hands each to `eval_prod_test_v1.sh --only`.
#
# An arm is "finished" when its log carries the trainer's own completion line, not when
# `best.ckpt` merely exists — the checkpoint is rewritten at every improvement, so
# evaluating on its presence would evaluate a mid-training model.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_ROOT="runs/prod_test_v1"
CONCURRENCY=4
EXPECT=11
POLL=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --expect)      EXPECT="$2"; shift 2 ;;
    --run-root)    RUN_ROOT="$2"; shift 2 ;;
    --poll)        POLL="$2"; shift 2 ;;
    -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

declare -A started=()
done_n=0
while [[ "$done_n" -lt "$EXPECT" ]]; do
  for log in "$RUN_ROOT"/logs/*.log; do
    [[ -f "$log" ]] || continue
    arm="$(basename "$log" .log)"
    [[ "$arm" == eval_* ]] && continue
    [[ -n "${started[$arm]:-}" ]] && continue
    grep -q "best val NLL/jet" "$log" || continue        # the trainer's completion line
    ckpt="$(ls -d "$RUN_ROOT/$arm"/*/best.ckpt 2>/dev/null | head -1)"
    [[ -n "$ckpt" ]] || continue
    while [[ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]]; do sleep 10; done
    echo "[stream] evaluating $arm"
    started[$arm]=1
    bash scripts/eval_prod_test_v1.sh --run-root "$RUN_ROOT" --only "$arm" --concurrency 1 &
    sleep 2
  done
  done_n="${#started[@]}"
  [[ "$done_n" -lt "$EXPECT" ]] && sleep "$POLL"
done
echo "[stream] all $EXPECT arms handed off; waiting for the last evals"
wait
echo "[stream] done"
