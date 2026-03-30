# Query Data Predictor

This repository contains the code, datasets, experiment configurations, and
artifact-generation workflow for the results reproduced here.

## Main Outputs

The main outputs are:

- `workload_comparison.pdf`
- `paper_table_rows.tex`

Supplementary outputs are:

- `gap_analysis.pdf`
- `overlap_analysis.pdf`
- `decay_sensitivity.pdf`

All generated table files are written to `artifacts/paper/tables/`.
All generated figure files are written to `artifacts/paper/figures/`.

## Setup

This repository uses `uv`. The helper scripts also reuse `.venv/` if it
already exists.

If you do not have `uv` installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You can either choose to use your own shell environment or use the provided helper scripts. The helper scripts set up a consistent environment and also verify that the artifact workflow is working correctly.:

```bash
git clone git@github.com:Dadams2/query-data-predictor.git
cd query-data-predictor
bash scripts/verify_artifact.sh
```

To set up the environment manually, you can run:

```bash
uv sync --no-dev
``` 

Activate the virtual environment with `source .venv/bin/activate`.
You should be able to run:

```bash
query-data-predictor --help
```
to see the available commands.


## Reproducing Experimental Results

The main reproduction path is to rerun the experiments from scratch and then
regenerate the tables and figures from those fresh results:

```bash
bash scripts/rerun_paper_experiments.sh
```

This script runs:

- `experiments/configs/simba_drilldown.yml`
- `experiments/configs/benchmark_mdi_vs_baselines.yml`
- `experiments/configs/sdss_vs_baselines.yml`

and then rebuilds:

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

This takes many hours. The SDSS and adversarial reruns are not quick, and the
decay sweep is separate and even more long-running.

## Compare Against Existing Snapshot Data

If you only want to compare your environment against some exmaple frozen snapshot data,
use:

```bash
bash scripts/reproduce_tables.sh
bash scripts/reproduce_figures.sh
```

This mode uses the standardized snapshots stored under:

- `previous-results/paper/`
- `previous-results/decay-sweep/`

## Generating Your Own Adversarial Benchmark

To create a fresh adversarial benchmark dataset instead of using the bundled
one:

```bash
source scripts/paper_env.sh
paper_sync
paper_python tools/generate_benchmark.py \
  --output data/datasets/benchmark_custom \
  --config-output experiments/configs/benchmark_custom.yml \
  --num-sessions 3 \
  --queries-per-session 30 \
  --total-rows 5000 \
  --seed 42
```

This will:

- create a new synthetic dataset in `data/datasets/benchmark_custom/`
- write a matching experiment config to `experiments/configs/benchmark_custom.yml`

Then run the experiment and analysis:

```bash
source scripts/paper_env.sh
paper_qdp run-experiment -c experiments/configs/benchmark_custom.yml
paper_qdp analyze-simple -c experiments/configs/benchmark_custom.yml
```

If you want to avoid mixing outputs with the default benchmark runs, edit the
generated config first and change:

- `output.output_directory`
- `experiment.name`

You can also do a quick self-check of the generator itself:

```bash
source scripts/paper_env.sh
paper_sync
paper_python tools/generate_benchmark.py --verify --seed 42
```

## Other Long-Running Experiments

The temporal decay sweep is available separately:

```bash
bash run_decay_sweep.sh
```

That sweep is also multi-hour and regenerates the supplementary
`decay_sensitivity.pdf` figure.

## Data Used

The repository includes these benchmark datasets:

- `data/datasets/simba_drilldown/` generated with SIMBA
- `data/datasets/benchmark_mdi/` generated with the MDI generator
- `data/datasets/` generated from skyserver SQL sessions

The SDSS corpus metadata includes 463 sessions in `data/datasets/metadata.csv`.
The reproduced SDSS benchmark uses the fixed session subset listed in
`experiments/configs/sdss_vs_baselines.yml`.


## Useful Files

- `generate_publication_figures.py`
- `extract_table_data.py`
- `tools/generate_benchmark.py`
- `scripts/reproduce_tables.sh`
- `scripts/reproduce_figures.sh`
- `scripts/rerun_paper_experiments.sh`
- `scripts/verify_artifact.sh`
