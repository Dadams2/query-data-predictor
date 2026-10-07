"""History-aware recommenders for revealed analytical interest."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_recommender import BaseRecommender
from .query_expansion_recommender import QueryExpansionRecommender
from ..query_runner import QueryRunner


class TemporalInterestRecommender(BaseRecommender):
    """Rank current and previously observed tuples using a decayed interest profile."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        temporal = config.get("temporal_interest", {})
        self.decay = float(temporal.get("decay", 0.1))
        self.recurrence_weight = float(temporal.get("recurrence_weight", 0.5))
        self.history_window = int(temporal.get("history_window", 10))
        self._history: list[pd.DataFrame] = []
        self.last_candidates = pd.DataFrame()

    def recommend_tuples(
        self, current_results: pd.DataFrame, top_k: Optional[int] = None, **kwargs
    ) -> pd.DataFrame:
        self._validate_input(current_results)
        if current_results.empty:
            self.last_candidates = current_results.copy()
            return current_results.copy()

        self._add_history(current_results)
        return self._rank_candidates(self._history_candidates(current_results.columns), top_k)

    def _add_history(self, results: pd.DataFrame) -> None:
        self._history.append(results.copy())
        if len(self._history) > self.history_window:
            self._history.pop(0)

    def _compatible_history(self, columns) -> list[pd.DataFrame]:
        names = list(columns)
        wanted = set(names)
        return [frame.reindex(columns=names) for frame in self._history if set(frame.columns) == wanted]

    def _history_candidates(self, columns) -> pd.DataFrame:
        frames = self._compatible_history(columns)
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)

    def _rank_candidates(self, candidates: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
        candidates = candidates.drop_duplicates(ignore_index=True)
        self.last_candidates = candidates.copy()
        if candidates.empty:
            return candidates

        history = self._compatible_history(candidates.columns)
        weights = np.exp(-self.decay * np.arange(len(history) - 1, -1, -1))
        row_hashes = pd.util.hash_pandas_object(candidates, index=False).to_numpy()

        recurrence = np.zeros(len(candidates), dtype=float)
        weighted_rows = []
        for weight, frame in zip(weights, history):
            hashes = set(pd.util.hash_pandas_object(frame, index=False).to_numpy())
            recurrence += weight * np.fromiter((value in hashes for value in row_hashes), dtype=float)
            weighted = frame.copy()
            weighted["__weight"] = weight / max(len(frame), 1)
            weighted_rows.append(weighted)
        recurrence /= weights.sum() or 1.0

        affinity = self._attribute_affinity(candidates, pd.concat(weighted_rows, ignore_index=True))
        score = self.recurrence_weight * recurrence + (1.0 - self.recurrence_weight) * affinity
        ranked = candidates.assign(__score=score, __order=np.arange(len(candidates)))
        ranked = ranked.sort_values(["__score", "__order"], ascending=[False, True], kind="stable")
        return self._limit_output(ranked.drop(columns=["__score", "__order"]), top_k=top_k).reset_index(drop=True)

    @staticmethod
    def _attribute_affinity(candidates: pd.DataFrame, observations: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(candidates), dtype=float)
        used = 0
        weights = observations["__weight"].to_numpy(dtype=float)
        total_weight = weights.sum() or 1.0

        for column in candidates.columns:
            observed = observations[column]
            if pd.api.types.is_numeric_dtype(observed):
                values = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(values)
                candidate_values = pd.to_numeric(candidates[column], errors="coerce").to_numpy(dtype=float)
                if not valid.any():
                    continue
                valid_weights = weights[valid]
                mean = np.average(values[valid], weights=valid_weights)
                variance = np.average((values[valid] - mean) ** 2, weights=valid_weights)
                if variance <= 1e-12:
                    contribution = np.isclose(candidate_values, mean, equal_nan=False).astype(float)
                else:
                    contribution = np.exp(-0.5 * (candidate_values - mean) ** 2 / variance)
                contribution[~np.isfinite(candidate_values)] = 0.0
            else:
                profile = (
                    observations.assign(__value=observed.astype("string").fillna("<NA>"))
                    .groupby("__value", dropna=False)["__weight"]
                    .sum()
                    .div(total_weight)
                )
                contribution = (
                    candidates[column]
                    .astype("string")
                    .fillna("<NA>")
                    .map(profile)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
            scores += contribution
            used += 1

        return scores / max(used, 1)

    def clear_history(self) -> None:
        self._history.clear()
        self.last_candidates = pd.DataFrame()


class ExploratoryInterestRecommender(TemporalInterestRecommender):
    """Rank a bounded query-expanded candidate set with the temporal profile."""

    def __init__(self, config: Dict[str, Any], query_runner: Optional[QueryRunner] = None):
        if query_runner is None:
            raise ValueError("QueryRunner is required for ExploratoryInterestRecommender")
        super().__init__(config)
        self.max_candidates = int(config.get("exploratory_interest", {}).get("max_candidates", 1000))
        self.expander = QueryExpansionRecommender(config, query_runner=query_runner)

    def recommend_tuples(
        self, current_results: pd.DataFrame, top_k: Optional[int] = None, **kwargs
    ) -> pd.DataFrame:
        self._validate_input(current_results)
        if current_results.empty:
            self.last_candidates = current_results.copy()
            return current_results.copy()

        expanded = self.expander.recommend_tuples(
            current_results,
            top_k=len(current_results) + self.max_candidates,
            **kwargs,
        )
        self._add_history(current_results)
        history = self._history_candidates(current_results.columns)
        if set(expanded.columns) == set(current_results.columns):
            expanded = expanded.reindex(columns=current_results.columns).tail(self.max_candidates)
            candidates = pd.concat([history, expanded], ignore_index=True)
        else:
            candidates = history
        return self._rank_candidates(candidates, top_k)

    def clear_history(self) -> None:
        super().clear_history()
        self.expander._last_session_context = None

