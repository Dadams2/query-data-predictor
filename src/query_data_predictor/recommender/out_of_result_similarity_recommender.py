"""Small categorical similarity search for the MDI benchmark."""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_recommender import BaseRecommender
from ..query_runner import QueryRunner


logger = logging.getLogger(__name__)


class OutOfResultSimilarityRecommender(BaseRecommender):
    """Rank unseen MDI orders by agreement with the current-result centroid."""

    columns = (
        "region", "category", "supplier_type", "price_range", "quantity_bin",
        "priority", "ship_mode", "year", "quarter",
    )

    def __init__(self, config: Dict[str, Any], query_runner: Optional[QueryRunner] = None):
        super().__init__(config)
        if query_runner is None:
            raise ValueError("QueryRunner is required for OutOfResultSimilarityRecommender")
        self.query_runner = query_runner
        self.last_candidates = pd.DataFrame()

    def recommend_tuples(
        self, current_results: pd.DataFrame, top_k: Optional[int] = None, **kwargs
    ) -> pd.DataFrame:
        self._validate_input(current_results)
        self.last_candidates = pd.DataFrame(columns=current_results.columns)
        if current_results.empty:
            return self.last_candidates.copy()

        missing = set(self.columns) - set(current_results.columns)
        if missing:
            raise ValueError(f"MDI result is missing columns: {sorted(missing)}")

        try:
            # ponytail: scan all 5k PoC rows; add a database/vector index if this grows.
            quoted_columns = ", ".join(f'"{column}"' for column in self.columns)
            candidates = self.query_runner.execute_query(
                f"SELECT row_id, {quoted_columns} FROM orders"
            )
        except Exception as exc:
            logger.error("Could not load MDI similarity candidates: %s", exc)
            return self.last_candidates.copy()

        seen = pd.MultiIndex.from_frame(current_results[list(self.columns)].drop_duplicates())
        candidate_keys = pd.MultiIndex.from_frame(candidates[list(self.columns)])
        unseen = candidates.loc[~candidate_keys.isin(seen)].copy()
        if unseen.empty:
            return self.last_candidates.copy()

        centroid = current_results[list(self.columns)].mode(dropna=False).iloc[0]
        unseen["_similarity_score"] = unseen[list(self.columns)].eq(centroid).mean(axis=1)
        ranked = unseen.sort_values(
            ["_similarity_score", "row_id"], ascending=[False, True], kind="stable"
        )
        self.last_candidates = ranked[list(current_results.columns)].reset_index(drop=True)
        return self._limit_output(self.last_candidates, top_k=top_k)
