import hashlib
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
import psycopg2


logger = logging.getLogger(__name__)


class QueryRunner:
    def __init__(
        self,
        dbname,
        user=None,
        host="localhost",
        port="5432",
        cache_dir=None,
        statement_timeout_seconds=120,
        **kwargs,
    ):
        self.db_params = {
            key: value
            for key, value in {
                "dbname": dbname,
                "user": user,
                "host": host,
                "port": port,
                **kwargs,
            }.items()
            if value is not None
        }
        self.cache_dir = Path(cache_dir) / str(dbname) if cache_dir else None
        self.statement_timeout_seconds = statement_timeout_seconds
        self.conn = None
        self.cursor = None
        self.cache_hits = 0
        self.database_queries = 0

    def connect(self):
        """Connect lazily to PostgreSQL for cache misses."""
        if self.conn is not None and not self.conn.closed:
            return

        logger.info(
            "Connecting to database: %s@%s:%s",
            self.db_params["dbname"],
            self.db_params.get("host", "localhost"),
            self.db_params.get("port", "5432"),
        )
        self.conn = psycopg2.connect(**self.db_params)
        self.conn.set_session(readonly=True, autocommit=True)
        self.cursor = self.conn.cursor()
        if self.statement_timeout_seconds is not None:
            self.cursor.execute(
                "SET statement_timeout = %s",
                (int(self.statement_timeout_seconds * 1000),),
            )

    def disconnect(self):
        """Disconnect from PostgreSQL."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        finally:
            self.cursor = None
            self.conn = None

    def _cache_paths(self, query):
        digest = hashlib.sha256(
            f"{self.db_params['dbname']}\0{query}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.parquet", self.cache_dir / f"{digest}.sql"

    def _read_cache(self, query):
        if self.cache_dir is None:
            return None

        result_path, _ = self._cache_paths(query)
        if not result_path.exists():
            return None

        try:
            result = pd.read_parquet(result_path)
        except Exception as exc:
            logger.warning("Ignoring unreadable query cache %s: %s", result_path, exc)
            return None

        self.cache_hits += 1
        logger.debug("Query cache hit: %s", result_path)
        return result

    @staticmethod
    def _atomic_write(path, writer):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            writer(temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_cache(self, query, result):
        if self.cache_dir is None:
            return

        result_path, query_path = self._cache_paths(query)
        try:
            self._atomic_write(result_path, lambda path: result.to_parquet(path, index=False))
            self._atomic_write(query_path, lambda path: path.write_text(query, encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not cache query result %s: %s", result_path, exc)

    def execute_query(self, query):
        """Return a cached result or execute a read-only query and cache it."""
        cached = self._read_cache(query)
        if cached is not None:
            return cached

        self.connect()
        logger.debug("Executing query: %s", query[:100])
        self.cursor.execute(query)
        if self.cursor.description is None:
            raise ValueError("Query did not return a result set")

        result = pd.DataFrame(
            self.cursor.fetchall(),
            columns=[description[0] for description in self.cursor.description],
        )
        self.database_queries += 1
        self._write_cache(query, result)
        return result

    def execute_queries(self, queries):
        """Execute multiple queries and return their DataFrames."""
        return [self.execute_query(query) for query in queries]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
