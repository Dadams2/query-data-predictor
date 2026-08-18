"""Iterate over query-result pairs from a CSV workload."""

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"session_id", "query_position", "query"}


class QueryResultSequence:
    def __init__(self, queries_path, query_runner):
        self.queries_path = Path(queries_path)
        if not self.queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {self.queries_path}")

        self.queries = pd.read_csv(self.queries_path)
        missing = REQUIRED_COLUMNS - set(self.queries.columns)
        if missing:
            raise ValueError(f"Queries file is missing columns: {', '.join(sorted(missing))}")
        if self.queries.duplicated(["session_id", "query_position"]).any():
            raise ValueError("Queries file contains duplicate session/query positions")
        if self.queries["session_id"].isna().any():
            raise ValueError("Queries file contains empty session IDs")
        positions = pd.to_numeric(self.queries["query_position"], errors="coerce")
        if positions.isna().any() or not positions.mod(1).eq(0).all():
            raise ValueError("Query positions must be integers")
        self.queries["query_position"] = positions.astype(int)
        if not self.queries["query"].map(lambda query: isinstance(query, str) and bool(query.strip())).all():
            raise ValueError("Queries file contains empty queries")

        self.query_runner = query_runner
        self.errors = {}

    def _session_rows(self, session_id):
        rows = self.queries[self.queries["session_id"].astype(str) == str(session_id)]
        if rows.empty:
            raise ValueError(f"Session ID {session_id} not found in queries file")
        return rows.sort_values("query_position")

    def get_sessions(self):
        return self.queries["session_id"].drop_duplicates().tolist()

    def get_ordered_query_ids(self, session_id):
        return self._session_rows(session_id)["query_position"].tolist()

    def get_query_text(self, session_id, query_id):
        rows = self._session_rows(session_id)
        query = rows[rows["query_position"] == query_id]
        if query.empty:
            raise ValueError(f"Query ID {query_id} not found in session {session_id}")
        return query.iloc[0]["query"]

    def get_query_results(self, session_id, query_id):
        return self.query_runner.execute_query(self.get_query_text(session_id, query_id))

    def get_query_results_with_text(self, session_id, query_id):
        query = self.get_query_text(session_id, query_id)
        return self.query_runner.execute_query(query), query

    def _result_or_none(self, session_id, query_id):
        key = (str(session_id), int(query_id))
        if key in self.errors:
            return None
        try:
            return self.get_query_results(session_id, query_id)
        except Exception as exc:
            query = self.get_query_text(session_id, query_id)
            self.errors[key] = {
                "session_id": session_id,
                "query_position": query_id,
                "query": query,
                "error_message": str(exc),
            }
            logger.error("Workload query failed for session %s query %s: %s", session_id, query_id, exc)
            return None

    def _iter_id_pairs(self, session_id, gap=1):
        query_ids = self.get_ordered_query_ids(session_id)
        for index in range(len(query_ids) - gap):
            yield query_ids[index], query_ids[index + gap]

    def iter_query_result_pairs(self, session_id, gap=1):
        for current_id, future_id in self._iter_id_pairs(session_id, gap):
            current = self._result_or_none(session_id, current_id)
            future = self._result_or_none(session_id, future_id)
            if current is not None and future is not None and not current.empty and not future.empty:
                yield current_id, future_id, current, future

    def iter_query_result_pairs_with_text(self, session_id, gap=1):
        for current_id, future_id, current, future in self.iter_query_result_pairs(session_id, gap):
            yield (
                current_id,
                future_id,
                current,
                future,
                self.get_query_text(session_id, current_id),
                self.get_query_text(session_id, future_id),
            )

    def get_query_pair_with_gap(self, session_id, current_query_id, gap=1):
        query_ids = self.get_ordered_query_ids(session_id)
        if current_query_id not in query_ids:
            raise ValueError(f"Query ID {current_query_id} not found in session {session_id}.")
        future_index = query_ids.index(current_query_id) + gap
        if future_index >= len(query_ids):
            raise IndexError(
                f"No future query with gap {gap} from query {current_query_id} in session {session_id}."
            )
        return (
            self.get_query_results(session_id, current_query_id),
            self.get_query_results(session_id, query_ids[future_index]),
        )
