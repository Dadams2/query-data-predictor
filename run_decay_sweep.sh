#!/usr/bin/env bash
# Run all decay sweep experiments and analyze results.
# Usage: bash run_decay_sweep.sh
#
# Estimated time: ~4-5 hours for all 8 decay rates.
# Each config runs MDI only on 3 sessions × 5 gaps.

set -e
cd "$(dirname "$0")"

source scripts/paper_env.sh

paper_sync

RATES=("0.00" "0.01" "0.05" "0.10" "0.15" "0.20" "0.30" "0.50")
TOTAL=${#RATES[@]}
COMPLETED=0

echo "=== Decay Rate Sensitivity Sweep ==="
echo "Running $TOTAL experiments (MDI only, 3 sessions × 5 gaps each)"
echo "Started at: $(date)"
echo ""

for rate in "${RATES[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    config="experiments/configs/decay_sweep/decay_${rate}.yml"
    echo "[$COMPLETED/$TOTAL] Running decay_rate=${rate} — $(date)"

    paper_qdp run-experiment -c "$config"

    # Find the most recent results directory
    results_dir=$(ls -td results/decay_sweep/decay_${rate}/decay_sweep_${rate}_* 2>/dev/null | head -1)
    if [ -n "$results_dir" ]; then
        echo "  Analyzing ${results_dir}..."
        paper_qdp analyze-results -c "$config" -r "$results_dir"
    else
        echo "  WARNING: No results found for decay_rate=${rate}"
    fi
    echo ""
done

echo "=== All experiments complete at $(date) ==="
echo ""
echo "Results in results/decay_sweep/"
echo "Run: paper_python generate_publication_figures.py  to regenerate figures"
