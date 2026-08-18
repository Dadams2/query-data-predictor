from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from query_data_predictor.experiment_runner import ExperimentRunner
from query_data_predictor.query_runner import QueryRunner


class _StatefulRecommender:
    def __init__(self):
        self.clear_history_calls = 0

    def clear_history(self):
        self.clear_history_calls += 1


def test_session_predict_with_gap_resets_stateful_recommenders(tmp_path):
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = {'experiment': {}}
    runner.output_dir = Path(tmp_path)
    runner.recommenders = {
        'stateful': _StatefulRecommender(),
        'stateless': object(),
    }
    runner.get_results = MagicMock(return_value={'ok': True})

    current_results = pd.DataFrame({'a': [1, 2]})
    future_results = pd.DataFrame({'a': [2]})

    query_result_sequence = MagicMock()
    query_result_sequence.iter_query_result_pairs_with_text.return_value = [
        (1, 2, current_results, future_results, 'select 1', 'select 2')
    ]
    runner.query_result_sequence = query_result_sequence

    runner.session_predict_with_gap('session-1', 10)

    assert runner.recommenders['stateful'].clear_history_calls == 1
    assert runner.get_results.call_count == 2


def test_query_runner_params_config_overrides_env(monkeypatch):
    monkeypatch.setenv("PG_DATA", "env_db")
    monkeypatch.setenv("PG_DATA_USER", "env_user")
    monkeypatch.setenv("PG_SESSION_PASSWORD", "env_password")
    monkeypatch.setenv("PG_HOST", "env_host")
    monkeypatch.setenv("PG_PORT", "15432")

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = {
        "query_runner": {
            "dbname": "config_db",
            "user": "config_user",
            "password": "config_password",
            "host": "config_host",
            "port": "25432",
            "timeout": 30,
        }
    }

    assert runner._query_runner_params() == {
        "dbname": "config_db",
        "user": "config_user",
        "password": "config_password",
        "host": "config_host",
        "port": "25432",
    }


def test_query_runner_keeps_password_param():
    runner = QueryRunner(
        dbname="db",
        user="user",
        password="password",
        host="host",
        port="5433",
    )

    assert runner.db_params == {
        "dbname": "db",
        "user": "user",
        "password": "password",
        "host": "host",
        "port": "5433",
    }


def test_writes_workload_query_errors(tmp_path):
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.output_dir = tmp_path
    runner.query_result_sequence = MagicMock()
    runner.query_result_sequence.errors = {
        ("session", 1): {
            "session_id": "session",
            "query_position": 1,
            "query": "SELECT broken",
            "error_message": "database error",
        }
    }

    runner._write_query_errors()

    contents = (tmp_path / "query_errors.json").read_text()
    assert "SELECT broken" in contents
    assert "database error" in contents
