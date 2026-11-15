#!/bin/zsh
# Pilot: waits for the GPU to be free, then benchmarks and runs
# CNN and ViT, no augmentation, at 1%, 10%, 100%, seed 0.
set -e
cd "$(dirname "$0")/.."
# Busy = any A-JEPA python process that is not stopped (state T).
busy() {
  ps -Ao stat=,args= | grep "A-JEPA-eval-on-AnyGoal/.venv/bin/python" \
    | grep -v grep | grep -qv "^T"
}
while busy; do sleep 60; done
echo "gpu free at $(date)"
uv run python scripts/bench_throughput.py
for size in 1% 10% 100%; do
  for model in cnn vit; do
    uv run python -m bias.train configs/images.yaml --model $model \
      --size $size --seed 0 --lr 1e-3 --wd 0.05 --stage pilot --log results/pilot.csv
    if busy; then echo "WARNING: other GPU work started during $model $size"; fi
  done
done
echo "pilot done at $(date)"
