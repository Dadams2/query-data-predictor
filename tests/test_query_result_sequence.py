from unittest.mock import MagicMock

import pandas as pd
import pytest

from query_data_predictor.query_result_sequence import QueryResultSequence


@pytest.fixture
def queries_path(tmp_path):
    path = tmp_path / "queries.csv"
    pd.DataFrame(
        {
            "session_id": ["session-1", "session-1", "session-1", "session-2"],
            "query_position": [3, 1, 5, 0],
            "query": ["SELECT 3", "SELECT 1", "SELECT 5", "SELECT 0"],
        }
    ).to_csv(path, index=False)
    return path


def test_orders_queries_and_iterates_gaps(queries_path):
    runner = MagicMock()
    runner.execute_query.side_effect = lambda query: pd.DataFrame({"value": [query]})
    sequence = QueryResultSequence(queries_path, runner)

    assert sequence.get_sessions() == ["session-1", "session-2"]
    assert sequence.get_ordered_query_ids("session-1") == [1, 3, 5]

    pairs = list(sequence.iter_query_result_pairs_with_text("session-1", gap=2))
    assert len(pairs) == 1
    assert pairs[0][0:2] == (1, 5)
    assert pairs[0][4:6] == ("SELECT 1", "SELECT 5")


def test_records_failed_query_once_and_skips_affected_pairs(queries_path):
    runner = MagicMock()

    def execute(query):
        if query == "SELECT 3":
            raise RuntimeError("database error")
        return pd.DataFrame({"value": [query]})

    runner.execute_query.side_effect = execute
    sequence = QueryResultSequence(queries_path, runner)

    assert list(sequence.iter_query_result_pairs("session-1")) == []
    assert list(sequence.iter_query_result_pairs("session-1")) == []
    assert len(sequence.errors) == 1
    assert next(iter(sequence.errors.values()))["error_message"] == "database error"
    assert runner.execute_query.call_args_list.count((("SELECT 3",),)) == 1


def test_rejects_invalid_query_file(tmp_path):
    path = tmp_path / "queries.csv"
    pd.DataFrame({"session_id": [1], "query": ["SELECT 1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="query_position"):
        QueryResultSequence(path, MagicMock())
