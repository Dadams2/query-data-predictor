from pathlib import Path

import pandas as pd
import yaml


def test_all_experiment_configs_reference_database_workloads():
    for path in Path("experiments/configs").rglob("*.yml"):
        config = yaml.safe_load(path.read_text())
        dataset = config.get("experiment", {}).get("dataset")
        database = config.get("query_runner", {}).get("dbname")
        assert dataset, f"{path} has no experiment.dataset"
        assert database, f"{path} has no query_runner.dbname"
        assert Path("queries", dataset, "queries.csv").exists(), (
            f"{path} references missing queries/{dataset}/queries.csv"
        )


def test_sdss_workload_contains_all_sessions():
    queries = pd.read_csv("queries/sdss/queries.csv", usecols=["session_id"])
    assert queries["session_id"].nunique() == 463
