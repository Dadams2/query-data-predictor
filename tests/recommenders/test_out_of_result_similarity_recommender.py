from unittest.mock import Mock

import pandas as pd

from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.recommender.out_of_result_similarity_recommender import (
    OutOfResultSimilarityRecommender,
)


COLUMNS = list(OutOfResultSimilarityRecommender.columns)


def row(**changes):
    values = {
        "region": "EUROPE",
        "category": "FINANCE",
        "supplier_type": "DOMESTIC",
        "price_range": "LOW",
        "quantity_bin": "SMALL",
        "priority": "STANDARD",
        "ship_mode": "RAIL",
        "year": "2024",
        "quarter": "Q2",
    }
    values.update(changes)
    return values


def test_ranks_unseen_rows_by_centroid_and_breaks_ties_by_row_id():
    current = pd.DataFrame([row(), row(), row(year="2023")])
    database = pd.DataFrame([
        {"row_id": 9, **row()},
        {"row_id": 2, **row(quarter="Q3")},
        {"row_id": 3, **row(ship_mode="AIR")},
        {"row_id": 1, **row(region="ASIA", category="TECHNOLOGY")},
    ])
    runner = Mock(spec=QueryRunner)
    runner.execute_query.return_value = database

    recommender = OutOfResultSimilarityRecommender(
        {"recommendation": {"top_k": 2}}, runner
    )
    result = recommender.recommend_tuples(current)

    assert result.to_dict("records") == [row(quarter="Q3"), row(ship_mode="AIR")]
    assert list(result.columns) == COLUMNS
    assert len(recommender.last_candidates) == 3
    assert row() not in result.to_dict("records")


def test_top_k_override_empty_input_and_database_failure():
    runner = Mock(spec=QueryRunner)
    runner.execute_query.return_value = pd.DataFrame([
        {"row_id": 1, **row(region="ASIA")},
        {"row_id": 2, **row(region="AFRICA")},
    ])
    recommender = OutOfResultSimilarityRecommender({}, runner)

    assert len(recommender.recommend_tuples(pd.DataFrame([row()]), top_k=1)) == 1
    assert recommender.recommend_tuples(pd.DataFrame(columns=COLUMNS)).empty

    runner.execute_query.side_effect = RuntimeError("database unavailable")
    assert recommender.recommend_tuples(pd.DataFrame([row()])).empty
