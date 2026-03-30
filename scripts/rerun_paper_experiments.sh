#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/paper_env.sh"

cd "$ROOT_DIR"

paper_sync

CONFIGS=(
  "experiments/configs/simba_drilldown.yml"
  "experiments/configs/benchmark_mdi_vs_baselines.yml"
  "experiments/configs/sdss_vs_baselines.yml"
)

for config in "${CONFIGS[@]}"; do
  echo "=== Running ${config} ==="
  paper_qdp run-experiment -c "$config"
  paper_qdp analyze-simple -c "$config"
  echo
done

echo "=== Regenerating paper outputs from the latest results ==="
paper_python extract_table_data.py --use-latest-results
paper_python generate_publication_figures.py --use-latest-results
