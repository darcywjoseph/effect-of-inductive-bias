#!/usr/bin/env bash
# Runs one config's grid after another, so a single pod works through the
# study unattended. Each grid resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
for config in "$@"; do
  case $config in
    *audio*) uv run python -m bias.data.download --only audio ;;
    *hsi*) uv run python -m bias.data.download --only hsi ;;
  esac
  scripts/run_grid.sh "$config" >> logs/grid.log 2>&1
done
