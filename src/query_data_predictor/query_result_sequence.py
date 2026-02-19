"""
A module for iterating over pairs of query results in a session, for prediction experiments.
"""

import pandas as pd
from query_data_predictor.dataloader import DataLoader
from typing import Iterator, Tuple, Any

class QueryResultSequence:
    def __init__(self, dataloader: DataLoader):
        """
        Initialize with a DataLoader instance that provides access to session metadata and results.
        Args:
            dataloader: A DataLoader instance that has access to session metadata and results
        """
        self.dataloader = dataloader
        self.id_cache = {}

    def _iter_id_pairs(self, session_id: str, gap: int = 1) -> Iterator[Tuple[Any, Any]]:
        """
        Internal helper that yields (current_query_id, future_query_id) pairs
        for the given session and gap.
        """
        query_ids = self.get_ordered_query_ids(session_id)
        for i in range(len(query_ids) - gap):
            yield query_ids[i], query_ids[i + gap]

    def get_ordered_query_ids(self, session_id: str) -> list:
        """
        Returns a list of query IDs in order for the session.
        Assumes session data is a DataFrame with a 'query_position' column.
        """
        if session_id in self.id_cache:
            return self.id_cache[session_id]
        
        session_data = self.dataloader.get_results_for_session(session_id)
        
        # Handle empty DataFrame
        if session_data.empty:
            self.id_cache[session_id] = []
            return []
        
        # Extract query positions and sort them
        query_ids = sorted(session_data['query_position'].tolist())
        self.id_cache[session_id] = query_ids
        return query_ids

    def get_query_results(self, session_id: str, query_id: str) -> pd.DataFrame:
        """
        Returns the results DataFrame for a given query in a session.
        """
        return self.dataloader.get_results_for_query(session_id, query_id)

    def iter_query_result_pairs(self, session_id: str, gap: int = 1) -> Iterator[Tuple[str, str, pd.DataFrame, pd.DataFrame]]:
        """
        Yields (current_query_id, future_query_id, current_results, future_results)
        for all valid pairs in the session with the specified gap.
        """
        for curr_id, fut_id in self._iter_id_pairs(session_id, gap):
            curr_res = self.get_query_results(session_id, curr_id)
            fut_res = self.get_query_results(session_id, fut_id)
            if not curr_res.empty and not fut_res.empty:
                yield curr_id, fut_id, curr_res, fut_res

    def iter_query_result_pairs_with_text(self, session_id: str, gap: int = 1) -> Iterator[Tuple[str, str, pd.DataFrame, pd.DataFrame, str, str]]:
        """
        Yields (current_query_id, future_query_id, current_results, future_results, current_query_text, future_query_text)
        for all valid pairs in the session with the specified gap.
        """
        for curr_id, fut_id in self._iter_id_pairs(session_id, gap):
            try:
                curr_res, curr_text = self.dataloader.get_results_for_query_with_text(session_id, curr_id)
                fut_res, fut_text = self.dataloader.get_results_for_query_with_text(session_id, fut_id)
                if not curr_res.empty and not fut_res.empty:
                    yield curr_id, fut_id, curr_res, fut_res, curr_text, fut_text
            except ValueError as e:
                # Skip pairs where query text is not available
                print(f"Skipping query pair {curr_id}->{fut_id} due to error: {e}")
                continue

    def get_query_pair_with_gap(self, session_id: str, current_query_id: str, gap: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (current_results, future_results) for the specified gap.
        Raises IndexError if the future query does not exist.
        """
        query_ids = self.get_ordered_query_ids(session_id)
        if current_query_id not in query_ids:
            raise ValueError(f"Query ID {current_query_id} not found in session {session_id}.")
        idx = query_ids.index(current_query_id)
        future_idx = idx + gap
        if future_idx >= len(query_ids):
            raise IndexError(f"No future query with gap {gap} from query {current_query_id} in session {session_id}.")
        future_query_id = query_ids[future_idx]
        current_results = self.get_query_results(session_id, current_query_id)
        future_results = self.get_query_results(session_id, future_query_id)
        return current_results, future_results
