"""
Kernel Density Recommender - Out-of-results recommendations via density estimation.

Models each SQL query as a region in data space, estimates an interest density
function using KDE over the region sequence, then samples and scores candidate
tuples from the full database.
"""

import re
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass

from .base_recommender import BaseRecommender
from .region_extractor import RegionExtractor, QueryRegion
from .interest_density import InterestDensity, KernelConfig
from ..query_runner import QueryRunner

logger = logging.getLogger(__name__)


@dataclass
class CandidateBudget:
    """Budget manager for candidate sampling queries."""
    max_queries: int = 5
    max_candidates: int = 1000
    max_time: float = 30.0
    executed_queries: int = 0
    total_candidates: int = 0
    start_time: Optional[float] = None

    def start_session(self):
        self.executed_queries = 0
        self.total_candidates = 0
        self.start_time = time.time()

    def can_execute(self) -> bool:
        if self.start_time is None:
            self.start_session()
        if self.executed_queries >= self.max_queries:
            return False
        if time.time() - self.start_time > self.max_time:
            return False
        if self.total_candidates >= self.max_candidates:
            return False
        return True

    def record(self, count: int):
        self.executed_queries += 1
        self.total_candidates += count


class KernelDensityRecommender(BaseRecommender):
    """Recommender using kernel density estimation over query regions.

    Extracts geometric regions from SQL queries, maintains a density model
    of the analyst's interest, and recommends tuples from anywhere in the
    database that fall in high-density areas.
    """

    def __init__(self, config: Dict[str, Any], query_runner: Optional[QueryRunner] = None):
        super().__init__(config)

        if query_runner is None:
            raise ValueError("QueryRunner is required for KernelDensityRecommender")

        self.query_runner = query_runner

        # Configuration
        kd_config = self.config.get('kernel_density', {})

        kernel_config = KernelConfig(
            kernel_type=kd_config.get('kernel_type', 'gaussian'),
            bandwidth_scale=kd_config.get('bandwidth_scale', 1.0),
            temporal_decay=kd_config.get('temporal_decay', 0.1),
            normalize_attributes=kd_config.get('normalize_attributes', True),
        )

        self.extractor = RegionExtractor()
        self.density = InterestDensity(kernel_config)

        # Candidate sampling config
        sampling_config = kd_config.get('candidate_sampling', {})
        self.budget = CandidateBudget(
            max_queries=sampling_config.get('max_queries', 5),
            max_candidates=sampling_config.get('max_candidates', 1000),
            max_time=sampling_config.get('max_time', 30.0),
        )
        self.expansion_factor = sampling_config.get('expansion_factor', 0.3)
        self.near_boundary_ratio = sampling_config.get('near_boundary_ratio', 0.8)
        self.random_sample_limit = sampling_config.get('random_sample_limit', 200)

        # State
        self._seen_hashes: Set[int] = set()
        self._stats_initialized = False

    def recommend_tuples(self, current_results: pd.DataFrame, top_k: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Recommend tuples from the database based on interest density.

        Args:
            current_results: DataFrame with the current query's results
            top_k: Number of tuples to return
            **kwargs: Must include 'current_query_text' for SQL parsing

        Returns:
            DataFrame with recommended tuples ranked by interest density
        """
        self._validate_input(current_results)

        if current_results.empty:
            return pd.DataFrame()

        current_query_text = kwargs.get('current_query_text', None)

        # 1. Extract region from SQL or fall back to result stats
        if current_query_text:
            region = self.extractor.extract(current_query_text)
            # If no bounds were extracted, fall back to result stats
            if not region.bounds:
                region = self.extractor.extract_from_results(current_results, current_query_text)
        else:
            region = self.extractor.extract_from_results(current_results)

        # 2. Add region to density model
        self.density.add_region(region)

        # 3. Initialize attribute stats from first result set
        if not self._stats_initialized and not current_results.empty:
            self._initialize_attribute_stats(current_results)

        # 4. Record current result tuples as "seen"
        for _, row in current_results.iterrows():
            self._seen_hashes.add(self._tuple_hash(row))

        # 5. Sample candidates from database
        table_name = region.table_name
        if not table_name:
            logger.warning("Could not determine table name, returning empty results")
            return pd.DataFrame()

        # Parse the query structure to preserve SELECT/GROUP BY expressions
        query_skeleton = self._parse_query_skeleton(current_query_text) if current_query_text else None

        try:
            candidates = self._sample_candidates(table_name, query_skeleton, list(current_results.columns))
        except Exception as e:
            logger.error(f"Error sampling candidates: {e}", exc_info=True)
            return pd.DataFrame()

        if candidates.empty:
            return pd.DataFrame()

        # 6. Score and rank candidates
        scored = self._score_and_rank_candidates(candidates)

        # 7. Return top-k
        return self._limit_output(scored, top_k=top_k)

    def _parse_query_skeleton(self, sql: str) -> Optional[Dict[str, str]]:
        """Parse a SQL query into its structural components (SELECT, FROM, GROUP BY).

        Returns a dict with 'select', 'from_table', and 'group_by' parts so we can
        reconstruct queries with the same column schema but different WHERE clauses.
        """
        if not sql:
            return None

        # Extract SELECT clause (everything between SELECT and FROM)
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return None
        select_clause = select_match.group(1).strip()

        # Extract FROM table
        from_match = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
        if not from_match:
            return None
        from_table = from_match.group(1)

        # Extract GROUP BY clause (if present)
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql, re.IGNORECASE | re.DOTALL)
        group_by = group_match.group(1).strip().rstrip(';') if group_match else None

        # Check if this is an aggregation query
        is_aggregation = group_by is not None or re.search(
            r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(', sql, re.IGNORECASE
        ) is not None

        return {
            'select': select_clause,
            'from_table': from_table,
            'group_by': group_by,
            'is_aggregation': is_aggregation,
        }

    def _initialize_attribute_stats(self, df: pd.DataFrame):
        """Compute running mean/std from the first observed result set."""
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                mean = float(df[col].mean())
                std = float(df[col].std())
                if std < 1e-20:
                    std = 1.0
                stats[col] = (mean, std)

        self.density.update_attribute_stats(stats)
        self._stats_initialized = True

    def _get_real_attr_name(self, col: str, region: QueryRegion) -> Optional[str]:
        """Map a result column name back to the real database attribute.

        For bin_* columns, look up the original attribute in region.bin_columns.
        For regular columns, return as-is if it's not a derived alias.
        """
        # Check if it's a bin column with a known mapping
        for attr, divisor in region.bin_columns.items():
            # e.g. attr='redshift' -> col='bin_redshift'
            if col.lower() == f'bin_{attr.lower()}':
                return attr

        # Skip known derived columns
        if col.startswith('bin_') or col.endswith('_bin'):
            return None
        if col.lower() in ('cnt', 'count'):
            return None
        if col.lower().startswith('average_'):
            return None

        return col

    def _sample_candidates(self, table_name: str, query_skeleton: Optional[Dict[str, str]],
                           result_columns: List[str]) -> pd.DataFrame:
        """Sample candidate tuples from the database.

        If the original query uses aggregation (GROUP BY), we reuse its
        SELECT/GROUP BY structure so candidates have matching columns.
        Otherwise we query raw columns.

        80% budget: near-boundary queries expanding recent regions
        20% budget: random sample from full table
        """
        self.budget.start_session()
        all_candidates = []

        # Near-boundary sampling (80% of budget)
        near_budget = int(self.budget.max_queries * self.near_boundary_ratio)
        recent_regions = self.density.regions[-3:] if len(self.density.regions) >= 3 else self.density.regions

        for region in recent_regions:
            if not self.budget.can_execute() or self.budget.executed_queries >= near_budget:
                break

            query = self._generate_near_boundary_query(region, table_name, query_skeleton, result_columns)
            if query:
                try:
                    result = self.query_runner.execute_query(query)
                    self.budget.record(len(result))
                    if not result.empty:
                        all_candidates.append(result)
                except Exception as e:
                    logger.warning(f"Near-boundary query failed: {e}")
                    self.budget.record(0)

        # Random sampling (remaining budget)
        if self.budget.can_execute():
            query = self._generate_random_sample_query(table_name, query_skeleton, result_columns,
                                                       self.random_sample_limit)
            try:
                result = self.query_runner.execute_query(query)
                self.budget.record(len(result))
                if not result.empty:
                    all_candidates.append(result)
            except Exception as e:
                logger.warning(f"Random sample query failed: {e}")
                self.budget.record(0)

        if not all_candidates:
            return pd.DataFrame()

        combined = pd.concat(all_candidates, ignore_index=True)
        combined = combined.drop_duplicates()
        return combined

    def _build_where_conditions(self, region: QueryRegion) -> List[str]:
        """Build WHERE conditions from region bounds, mapping bin columns to real attributes."""
        conditions = []
        for attr, bound in region.bounds.items():
            if not bound.is_numeric or not bound.intervals:
                continue

            # Map result column names to real DB attributes
            real_attr = self._get_real_attr_name(attr, region)
            if real_attr is None:
                continue

            lo = bound.overall_min
            hi = bound.overall_max
            if lo is None or hi is None:
                continue

            extent = hi - lo
            expansion = extent * self.expansion_factor
            new_lo = lo - expansion
            new_hi = hi + expansion

            conditions.append(f"{real_attr} BETWEEN {new_lo} AND {new_hi}")

        return conditions

    def _generate_near_boundary_query(self, region: QueryRegion, table_name: str,
                                      query_skeleton: Optional[Dict[str, str]],
                                      result_columns: List[str]) -> Optional[str]:
        """Generate a query that expands a region's bounds by expansion_factor.

        Preserves the original query's SELECT/GROUP BY structure so that
        candidate results have the same columns as the evaluation expects.
        """
        conditions = self._build_where_conditions(region)
        if not conditions:
            return None

        where = ' AND '.join(conditions)

        if query_skeleton and query_skeleton.get('is_aggregation'):
            # Aggregation query: preserve SELECT and GROUP BY, add WHERE
            select = query_skeleton['select']
            group_by = query_skeleton['group_by']
            query = f"SELECT {select} FROM {table_name} WHERE {where}"
            if group_by:
                query += f" GROUP BY {group_by}"
            return query

        # Non-aggregation: select the real DB columns
        db_columns = self._filter_db_columns(result_columns, region)
        if not db_columns:
            db_columns = ['*']
        cols_str = ', '.join(db_columns[:10])
        return f"SELECT {cols_str} FROM {table_name} WHERE {where} ORDER BY RANDOM() LIMIT {self.random_sample_limit}"

    def _generate_random_sample_query(self, table_name: str, query_skeleton: Optional[Dict[str, str]],
                                      result_columns: List[str], limit: int) -> str:
        """Generate a random sample query from the full table.

        Preserves the original query's SELECT/GROUP BY structure.
        """
        if query_skeleton and query_skeleton.get('is_aggregation'):
            # For aggregation queries, run the full query without WHERE
            # (or with a random subsample via tablesample if available)
            select = query_skeleton['select']
            group_by = query_skeleton['group_by']
            query = f"SELECT {select} FROM {table_name}"
            if group_by:
                query += f" GROUP BY {group_by}"
            return query

        db_columns = self._filter_db_columns(result_columns, QueryRegion())
        if not db_columns:
            db_columns = ['*']
        cols_str = ', '.join(db_columns[:10])
        return f"SELECT {cols_str} FROM {table_name} ORDER BY RANDOM() LIMIT {limit}"

    def _filter_db_columns(self, columns: List[str], region: QueryRegion) -> List[str]:
        """Filter out columns that are SQL aliases and don't exist in the database."""
        filtered = []
        for col in columns:
            if col.startswith('bin_'):
                continue
            if col.endswith('_bin'):
                continue
            if col.lower() in ('cnt', 'count'):
                continue
            if col.lower().startswith('average_'):
                continue
            filtered.append(col)
        return filtered

    def _score_and_rank_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score candidate tuples and exclude already-seen tuples."""
        # Convert rows to dicts for scoring
        points = []
        valid_indices = []
        for idx, row in candidates.iterrows():
            h = self._tuple_hash(row)
            if h in self._seen_hashes:
                continue
            point = {}
            for col in candidates.columns:
                val = row[col]
                if pd.api.types.is_numeric_dtype(type(val)) and pd.notna(val):
                    point[col] = float(val)
            points.append(point)
            valid_indices.append(idx)

        if not points:
            return pd.DataFrame(columns=candidates.columns)

        scores = self.density.score_points_batch(points)

        result = candidates.loc[valid_indices].copy()
        result['_density_score'] = scores
        result = result.sort_values('_density_score', ascending=False)
        result = result.drop(columns=['_density_score'])

        return result.reset_index(drop=True)

    def _tuple_hash(self, row) -> int:
        """Compute a hash for deduplication."""
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append(None)
            else:
                vals.append(v)
        return hash(tuple(vals))

    def name(self) -> str:
        return "KernelDensityRecommender"
