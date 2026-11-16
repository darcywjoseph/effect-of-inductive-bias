#!/usr/bin/env bash
# Runs the LeJEPA, tune and final stages for one config, each split across
# SHARDS processes that share the GPU. Stages are resumable, so rerunning
# this after an interruption picks up where it stopped.
set -euo pipefail
config=$1
shards=${SHARDS:-3}
mkdir -p logs
for stage in pretrain tune final; do
  for ((i = 0; i < shards; i++)); do
    uv run python -m bias.grid "$config" --stage "$stage" --shard "$i/$shards" \
      >> "logs/$(basename "$config" .yaml)_${stage}_${i}.log" 2>&1 &
  done
  wait
  echo "$(date -u +%FT%TZ) $stage done"
done
