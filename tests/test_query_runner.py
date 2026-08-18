import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from query_data_predictor.query_runner import QueryRunner


def _database_result(runner, rows=((1, "one"),)):
    cursor = MagicMock()
    cursor.description = [("id",), ("name",)]
    cursor.fetchall.return_value = list(rows)
    runner.connect = MagicMock(side_effect=lambda: setattr(runner, "cursor", cursor))
    return cursor


def test_cache_miss_hit_and_invalidation(tmp_path):
    query = "SELECT id, name FROM things"
    runner = QueryRunner("test", cache_dir=tmp_path)
    cursor = _database_result(runner)

    expected = pd.DataFrame({"id": [1], "name": ["one"]})
    pd.testing.assert_frame_equal(runner.execute_query(query), expected)
    assert runner.database_queries == 1
    assert cursor.execute.call_count == 1

    offline = QueryRunner("test", cache_dir=tmp_path)
    offline.connect = MagicMock(side_effect=ConnectionError("database offline"))
    pd.testing.assert_frame_equal(offline.execute_query(query), expected)
    assert offline.cache_hits == 1
    offline.connect.assert_not_called()

    parquet_path, sql_path = offline._cache_paths(query)
    assert parquet_path.exists()
    assert sql_path.read_text() == query

    parquet_path.unlink()
    with pytest.raises(ConnectionError, match="database offline"):
        offline.execute_query(query)


def test_query_failures_are_not_cached(tmp_path):
    runner = QueryRunner("test", cache_dir=tmp_path)
    cursor = _database_result(runner)
    cursor.execute.side_effect = RuntimeError("bad query")

    with pytest.raises(RuntimeError, match="bad query"):
        runner.execute_query("SELECT broken")

    assert not list(tmp_path.rglob("*.parquet"))


def test_password_is_kept_out_of_cache_settings():
    runner = QueryRunner("db", user="user", password="secret", cache_dir="data/example")
    assert runner.db_params["password"] == "secret"
    assert runner.cache_dir.name == "db"


@pytest.mark.skipif(
    os.getenv("QDP_RUN_DB_TESTS") != "1",
    reason="set QDP_RUN_DB_TESTS=1 to run PostgreSQL integration tests",
)
def test_postgres_result_is_cached_offline(tmp_path):
    from dotenv import load_dotenv

    load_dotenv(".env")
    runner = QueryRunner(
        dbname="benchmark_mdi",
        user=os.getenv("PG_DATA_USER"),
        password=os.getenv("PG_SESSION_PASSWORD"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        cache_dir=tmp_path,
    )
    query = "SELECT region FROM orders ORDER BY row_id LIMIT 1"
    expected = runner.execute_query(query)
    runner.disconnect()

    offline = QueryRunner("benchmark_mdi", cache_dir=tmp_path)
    offline.connect = MagicMock(side_effect=ConnectionError("offline"))
    pd.testing.assert_frame_equal(offline.execute_query(query), expected)
    offline.connect.assert_not_called()
