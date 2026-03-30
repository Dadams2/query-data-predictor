#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/paper_env.sh"

cd "$ROOT_DIR"

paper_sync

echo "=== sanity tests ==="
paper_pytest tests/test_config_manager.py tests/test_metrics.py tests/test_analysis_simple.py -q

echo
echo "=== Verifying paper tables ==="
paper_python extract_table_data.py --check

echo
echo "=== Regenerating paper figures ==="
paper_python generate_publication_figures.py

echo
echo "=== Checking expected figure outputs ==="
for output in \
  "artifacts/paper/figures/workload_comparison.pdf" \
  "artifacts/paper/figures/workload_comparison_simba.pdf" \
  "artifacts/paper/figures/workload_comparison_adversarial.pdf" \
  "artifacts/paper/figures/workload_comparison_sdss.pdf" \
  "artifacts/paper/figures/workload_comparison_legend.pdf" \
  "artifacts/paper/figures/gap_analysis.pdf" \
  "artifacts/paper/figures/overlap_analysis.pdf" \
  "artifacts/paper/figures/decay_sensitivity.pdf"
do
  if [[ ! -f "$output" ]]; then
    echo "Missing expected figure output: $output" >&2
    exit 1
  fi
done

echo
echo "Done"
