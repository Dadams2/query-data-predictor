from unittest.mock import MagicMock

import pandas as pd

from query_data_predictor.recommender.query_expansion_recommender import QueryExpansionRecommender
from query_data_predictor.recommender.temporal_interest_recommender import (
    ExploratoryInterestRecommender,
    TemporalInterestRecommender,
)


CONFIG = {
    "recommendation": {"mode": "top_k", "top_k": 2},
    "temporal_interest": {"decay": 0.1, "recurrence_weight": 0.5, "history_window": 3},
    "query_expansion": {"allow_predicate_removal": True},
}


def test_temporal_interest_ranks_recurrent_tuple_and_resets():
    recommender = TemporalInterestRecommender(CONFIG)
    recommender.recommend_tuples(pd.DataFrame({"kind": ["a", "b"]}))

    result = recommender.recommend_tuples(pd.DataFrame({"kind": ["b", "c"]}), top_k=3)

    assert result.iloc[0]["kind"] == "b"
    assert set(result["kind"]) == {"a", "b", "c"}
    recommender.clear_history()
    assert recommender.last_candidates.empty
    assert recommender._history == []


def test_exploratory_interest_adds_database_candidates():
    recommender = ExploratoryInterestRecommender(CONFIG, query_runner=MagicMock())
    current = pd.DataFrame({"kind": ["a"]})
    recommender.expander.recommend_tuples = MagicMock(
        return_value=pd.DataFrame({"kind": ["a", "outside"]})
    )

    recommender.recommend_tuples(current, top_k=2, current_query="select kind from items")

    assert set(recommender.last_candidates["kind"]) == {"a", "outside"}


def test_query_expansion_can_mask_an_unhandled_predicate():
    recommender = QueryExpansionRecommender(CONFIG, query_runner=MagicMock())
    query = "SELECT * FROM stars WHERE magnitude < 20 LIMIT 100"

    assert recommender._relax_condition(query, "magnitude < 20") == (
        "SELECT * FROM stars WHERE TRUE LIMIT 100"
    )


def test_query_expansion_ranking_is_deterministic():
    recommender = QueryExpansionRecommender(CONFIG, query_runner=MagicMock())
    rows = pd.DataFrame({"a": [1, 2], "b": [None, 3]})

    first = recommender._rank_expansion_results(rows, {})
    second = recommender._rank_expansion_results(rows, {})

    pd.testing.assert_frame_equal(first, second)
