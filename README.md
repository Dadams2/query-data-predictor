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

## Local Postgres With Docker

Local Postgres can be run with Docker Compose:

```bash
cp .env.example .env
docker compose up -d
```

The Compose service starts one Postgres container and can host multiple
databases. The generated `benchmark_mdi` database is initialized directly
from `docker/postgres/init/10-benchmark_mdi.sql`. To restore the other logical
backups on first startup, put them in
`docker/postgres/backups/` before running Compose:

- `sdss.sql` or `sdss.dump` restores into database `sdss`
- `simba_circactivity.sql` or `simba_circactivity.dump` restores into database `simba_circactivity`
- `simba_sdss.sql` or `simba_sdss.dump` restores into database `simba_sdss`

Backup files are ignored by git. To re-run restores from scratch:

```bash
docker compose down -v
docker compose up -d
```

To migrate an existing local Postgres database into the Docker setup:

```bash
scripts/dump_postgres_db.sh sdss
scripts/dump_postgres_db.sh simba_sdss
docker compose up -d
```

The dump script reads `.env` if present and writes custom-format logical
backups under `docker/postgres/backups/`.

Each experiment selects its workload and database in YAML:

```yaml
experiment:
  dataset: sdss

query_runner:
  dbname: sdss
```

Credentials and connection location come from `PG_DATA_USER`,
`PG_SESSION_PASSWORD`, `PG_HOST`, and `PG_PORT`. Workload queries are read from
`queries/<dataset>/queries.csv`.

## Query Result Cache

PostgreSQL is the source of truth. On the first run, each query result is
written to `data/<dataset>/<database>/<sha256>.parquet` with a matching `.sql`
file. Later runs read the Parquet file first, so a fully populated experiment
can run while PostgreSQL is unavailable.

Inspect a cache entry with pandas:

```bash
uv run python -c "import pandas as pd; print(pd.read_parquet('data/sdss/sdss/FILE.parquet').head())"
```

The matching `FILE.sql` identifies the query. Invalidate a dataset cache with:

```bash
rm -rf data/sdss
```

Failed queries are not cached. They are recorded in `query_errors.json` under
the timestamped experiment result directory, and pairs involving them are
skipped.


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

To recreate the database-backed adversarial benchmark:

```bash
source scripts/paper_env.sh
paper_sync
paper_python tools/generate_benchmark.py \
  --dataset benchmark_custom \
  --sql-output docker/postgres/init/10-benchmark_mdi.sql \
  --config-output experiments/configs/benchmark_custom.yml \
  --num-sessions 3 \
  --queries-per-session 30 \
  --total-rows 5000 \
  --seed 42
```

This writes:

- `queries/benchmark_custom/queries.csv`
- a PostgreSQL initialization script containing the fact and query-membership tables
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

## Destination-Recovery Benchmark (EDBT 2027 vision paper)

`src/query_data_predictor/destination_benchmark.py` tests whether a scorer can
anticipate where an exploration trajectory is heading, using planted
destinations in a generated table. It needs no database. The submitted
artifact, with the full results and reproduction instructions, is the
`edbt2027-submission` tag (branch `EDBT_2027`). Why its MDI settings differ
from the original model is recorded in `docs/mdi-design-decisions.md`. To
rerun the reported seeds:

```bash
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m query_data_predictor.destination_benchmark \
  --output artifacts/destination-rerun --seeds 100 101 102 103 104 105 106 107 108 109
```

`tools/sdss_characterise.py` reproduces the SkyServer log characterisation from
`data/skyserver_sessions.csv.gz`, a query-only extract of the 462 sessions.
Download it first:

```bash
bash scripts/download_skyserver_extract.sh
.venv/bin/python tools/sdss_characterise.py --output artifacts/sdss-characterisation
```

## Data Used

The repository tracks workload queries rather than query results:

- `queries/sdss/queries.csv`
- `queries/simba_simple/queries.csv`
- `queries/simba_complex/queries.csv`
- `queries/simba_drilldown/queries.csv`
- `queries/simba_sdss/queries.csv`
- `queries/benchmark_mdi/queries.csv`

The SDSS workload includes 463 sessions. Query result data is generated into
the ignored `data/` cache as experiments run.
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
