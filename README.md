# Query Data Predictor

This repository contains the code, datasets, experiment configurations, and
artifact-generation workflow for the vision paper:

`Beyond Active Learning: Implicit Recommendation and Diagnostic Benchmarks for Exploratory Data Analysis [Vision]`

The submission artifact is organized around the final outputs that need to be
reproduced from this repository:

- Table 1: SIMBA drill-down results
- Table 2: adversarial benchmark results
- Table 3: sampled SkyServer SQL-log results
- Figure 1: workload comparison

The remaining figures produced by the repo are treated as supplementary:
`gap_analysis.pdf`, `overlap_analysis.pdf`, and `decay_sensitivity.pdf`.

The frozen paper snapshots used for default verification live under
`previous-results/` with standard names, while fresh reruns continue to write
timestamped outputs under `results/`.

## Quick Start


```bash
git clone git@github.com:Dadams2/query-data-predictor.git
git checkout VLDB_2026
```

The repository is set up to use `uv`. All submission-facing scripts keep caches
inside the repository so they also work in restricted environments.

```bash
bash scripts/verify_artifact.sh
```

That command will:

- install/sync dependencies with `uv`
- run a small smoke suite for the artifact workflow
- recompute the paper tables from the included result snapshots
- regenerate the paper figures

## Reproduction Modes

### 1. Verify from included results

This is the recommended review path. It uses the bundled result snapshots
listed in [paper_artifact_manifest.yaml](/Users/DAADAMS/Other/query-data-predictor/paper_artifact_manifest.yaml).

```bash
bash scripts/reproduce_tables.sh
bash scripts/reproduce_figures.sh
```

Outputs:

- `artifacts/paper/tables/paper_tables.md`
- `artifacts/paper/tables/paper_table_rows.tex`
- `artifacts/paper/tables/paper_tables.json`
- `artifacts/paper/figures/workload_comparison.pdf`
- `artifacts/paper/figures/workload_comparison_simba.pdf`
- `artifacts/paper/figures/workload_comparison_adversarial.pdf`
- `artifacts/paper/figures/workload_comparison_sdss.pdf`
- `artifacts/paper/figures/workload_comparison_legend.pdf`
- `artifacts/paper/figures/gap_analysis.pdf`
- `artifacts/paper/figures/overlap_analysis.pdf`
- `artifacts/paper/figures/decay_sensitivity.pdf`

### 2. Rerun the paper experiments

If you want to regenerate the underlying result directories rather than verify
against the included snapshots:

```bash
bash scripts/rerun_paper_experiments.sh
```

This runs and analyzes the three paper benchmarks using:

- [experiments/configs/simba_drilldown.yml](/Users/DAADAMS/Other/query-data-predictor/experiments/configs/simba_drilldown.yml)
- [experiments/configs/benchmark_mdi_vs_baselines.yml](/Users/DAADAMS/Other/query-data-predictor/experiments/configs/benchmark_mdi_vs_baselines.yml)
- [experiments/configs/sdss_vs_baselines.yml](/Users/DAADAMS/Other/query-data-predictor/experiments/configs/sdss_vs_baselines.yml)

After rerunning, the figure/table scripts will prefer the manifest's frozen
paper snapshots when present. The rerun script explicitly switches the table and
figure generators to `--use-latest-results`, so the full reproduction pipeline
still uses the fresh timestamped outputs in `results/`. Those latest reruns are
not required to numerically match the frozen paper snapshot exactly, so this
mode regenerates outputs without checking them against the paper expectations.

## Paper Output Map

The canonical mapping from experiments to paper outputs lives in
[paper_artifact_manifest.yaml](/Users/DAADAMS/Other/query-data-predictor/paper_artifact_manifest.yaml).

In short:

- `simba_drilldown` feeds the SIMBA table
- `adversarial_benchmark` feeds the adversarial table and gap analysis
- `sdss_logs` feeds the SkyServer table
- [generate_publication_figures.py](/Users/DAADAMS/Other/query-data-predictor/generate_publication_figures.py) builds the final figure files under `artifacts/paper/figures/`
- [extract_table_data.py](/Users/DAADAMS/Other/query-data-predictor/extract_table_data.py) extracts the exact paper table values
- `previous-results/paper/` stores the frozen benchmark snapshots with stable names
- `previous-results/decay-sweep/` stores the frozen decay-sweep snapshots with stable names
- If `vision-paper/figures/` exists locally, the figure generator also mirrors the figure assets there for LaTeX compilation

## Data Included in the Repository

The paper currently uses three data sources already materialized in the repo:

- `data/datasets/simba_drilldown/`
- `data/datasets/benchmark_mdi/`
- `data/datasets/` for the sampled SkyServer SQL sessions

The SDSS corpus metadata currently includes 463 sessions in
`data/datasets/metadata.csv`. The paper benchmark uses a fixed 10-session subset
specified in the SDSS config above.

## Notes for Reviewers

- The main artifact path is `verify from included results`, not `rerun everything`.
- `uv` is the supported package manager for this repo.
- The scripts set local cache directories for `uv`, Matplotlib, and fontconfig.
- If `.venv/` already exists, the scripts reuse it instead of forcing a new `uv sync`.
- This repository does not need the LaTeX paper source to reproduce the final
  tables and figure files.

## Related Files

- [paper_artifact_manifest.yaml](/Users/DAADAMS/Other/query-data-predictor/paper_artifact_manifest.yaml)
- [generate_publication_figures.py](/Users/DAADAMS/Other/query-data-predictor/generate_publication_figures.py)
- [extract_table_data.py](/Users/DAADAMS/Other/query-data-predictor/extract_table_data.py)
- [scripts/reproduce_tables.sh](/Users/DAADAMS/Other/query-data-predictor/scripts/reproduce_tables.sh)
- [scripts/reproduce_figures.sh](/Users/DAADAMS/Other/query-data-predictor/scripts/reproduce_figures.sh)
- [scripts/rerun_paper_experiments.sh](/Users/DAADAMS/Other/query-data-predictor/scripts/rerun_paper_experiments.sh)
- [scripts/verify_artifact.sh](/Users/DAADAMS/Other/query-data-predictor/scripts/verify_artifact.sh)
