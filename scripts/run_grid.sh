#!/usr/bin/env bash
# Runs the LeJEPA, tune and final stages for one config, each split across
# SHARDS processes that share the GPU. Stages are resumable, so rerunning
# this after an interruption picks up where it stopped. The tune stage
# repeats in rounds while the learning-rate ladder still has rungs to try.
set -euo pipefail
config=$1
shards=${SHARDS:-3}
name=$(basename "$config" .yaml)
mkdir -p logs

run_stage() {
  for ((i = 0; i < shards; i++)); do
    uv run python -m bias.grid "$config" --stage "$1" --shard "$i/$shards" \
      >> "logs/${name}_${1}_${i}.log" 2>&1 &
  done
  wait
}

run_stage pretrain
echo "$(date -u +%FT%TZ) $name pretrain done"
round=1
# A cap, so a job that keeps failing cannot loop forever.
while (( round <= 6 )) &&
  (( $(uv run python -m bias.grid "$config" --stage tune --pending) > 0 )); do
  run_stage tune
  echo "$(date -u +%FT%TZ) $name tune round $round done"
  round=$((round + 1))
done
echo "$(date -u +%FT%TZ) $name tune done"
run_stage final
echo "$(date -u +%FT%TZ) $name final done"
if grep -q '^budget_check:' "$config"; then
  run_stage budget
  echo "$(date -u +%FT%TZ) $name budget done"
fi
if grep -q '^leakage:' "$config"; then
  run_stage leakage
  echo "$(date -u +%FT%TZ) $name leakage done"
fi
