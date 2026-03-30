"""
Region Extractor - Converts SQL queries into geometric regions for density estimation.

Parses SQL queries (particularly SIMBA-style) into QueryRegion objects that represent
axis-aligned bounding boxes in the data space. These regions are used as implicit
feedback for the kernel density interest model.
"""

import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttributeBound:
    """Stores per-attribute intervals and/or categorical values.

    For numeric attributes: stores a list of (lo, hi) intervals.
    For categorical attributes: stores a set of allowed values.
    """
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    categorical_values: Set[str] = field(default_factory=set)

    @property
    def centroid(self) -> Optional[float]:
        """Return the centroid of all intervals."""
        if not self.intervals:
            return None
        all_lo = [lo for lo, _ in self.intervals]
        all_hi = [hi for _, hi in self.intervals]
        return (min(all_lo) + max(all_hi)) / 2.0

    @property
    def extent(self) -> Optional[float]:
        """Return the total extent (max - min) across all intervals."""
        if not self.intervals:
            return None
        all_lo = [lo for lo, _ in self.intervals]
        all_hi = [hi for _, hi in self.intervals]
        ext = max(all_hi) - min(all_lo)
        return ext if ext > 0 else 1e-10

    @property
    def overall_min(self) -> Optional[float]:
        """Return the minimum value across all intervals."""
        if not self.intervals:
            return None
        return min(lo for lo, _ in self.intervals)

    @property
    def overall_max(self) -> Optional[float]:
        """Return the maximum value across all intervals."""
        if not self.intervals:
            return None
        return max(hi for _, hi in self.intervals)

    @property
    def is_numeric(self) -> bool:
        return len(self.intervals) > 0

    @property
    def is_categorical(self) -> bool:
        return len(self.categorical_values) > 0


@dataclass
class QueryRegion:
    """Represents a query as a geometric region in data space.

    Attributes:
        table_name: Source table name
        bounds: Per-attribute bounds extracted from WHERE clause
        select_columns: Columns in SELECT clause
        bin_columns: FLOOR(attr/divisor) patterns mapping attr -> divisor
        raw_sql: Original SQL string
    """
    table_name: Optional[str] = None
    bounds: Dict[str, AttributeBound] = field(default_factory=dict)
    select_columns: List[str] = field(default_factory=list)
    bin_columns: Dict[str, float] = field(default_factory=dict)
    raw_sql: Optional[str] = None


class RegionExtractor:
    """Extracts geometric regions from SQL queries.

    Handles SIMBA-style queries with OR-connected ranges, FLOOR binning patterns,
    categorical equality predicates, and standard comparison operators.
    """

    def extract(self, sql: str) -> QueryRegion:
        """Parse a SQL query into a QueryRegion.

        Args:
            sql: SQL query string

        Returns:
            QueryRegion representing the query's geometric region
        """
        region = QueryRegion(raw_sql=sql)

        region.table_name = self._parse_table_name(sql)
        region.select_columns = self._parse_select_columns(sql)
        region.bin_columns = self._parse_bin_columns(sql)
        region.bounds = self._parse_where_bounds(sql)

        return region

    def extract_from_results(self, results: pd.DataFrame, sql: str = None) -> QueryRegion:
        """Fallback extraction from DataFrame statistics when SQL parsing fails.

        Args:
            results: DataFrame of query results
            sql: Optional SQL string for additional parsing

        Returns:
            QueryRegion with bounds inferred from data min/max
        """
        region = QueryRegion(raw_sql=sql)

        if sql:
            region.table_name = self._parse_table_name(sql)
            region.select_columns = self._parse_select_columns(sql)
            region.bin_columns = self._parse_bin_columns(sql)

        if results.empty:
            return region

        region.select_columns = region.select_columns or list(results.columns)

        for col in results.columns:
            if pd.api.types.is_numeric_dtype(results[col]) and results[col].notna().any():
                lo = float(results[col].min())
                hi = float(results[col].max())
                if lo == hi:
                    hi = lo + 1e-10
                bound = AttributeBound(intervals=[(lo, hi)])
                region.bounds[col] = bound
            elif results[col].dtype == object and results[col].notna().any():
                unique_vals = set(results[col].dropna().unique())
                if len(unique_vals) <= 20:  # Only treat as categorical if not too many values
                    bound = AttributeBound(categorical_values=unique_vals)
                    region.bounds[col] = bound

        return region

    def _parse_table_name(self, sql: str) -> Optional[str]:
        """Extract table name from FROM clause."""
        match = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _parse_select_columns(self, sql: str) -> List[str]:
        """Extract column names from SELECT clause."""
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return []

        select_clause = select_match.group(1).strip()
        if select_clause == '*':
            return []

        columns = []
        # Split by comma, handling FLOOR(...) AS ... patterns
        parts = re.split(r',\s*(?![^()]*\))', select_clause)
        for part in parts:
            part = part.strip()
            # Handle AS aliases
            as_match = re.search(r'\bAS\s+(\w+)', part, re.IGNORECASE)
            if as_match:
                columns.append(as_match.group(1))
            else:
                # Handle table.column or just column
                col_match = re.search(r'(?:\w+\.)?(\w+)\s*$', part)
                if col_match:
                    columns.append(col_match.group(1))

        return columns

    def _parse_bin_columns(self, sql: str) -> Dict[str, float]:
        """Extract FLOOR(attr/divisor) AS bin_attr patterns."""
        bins = {}
        pattern = r'FLOOR\((\w+)/([\d.]+)\)\s+AS\s+(\w+)'
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            attr = match.group(1)
            divisor = float(match.group(2))
            bins[attr] = divisor
        return bins

    def _parse_where_bounds(self, sql: str) -> Dict[str, AttributeBound]:
        """Parse WHERE clause into attribute bounds.

        Handles:
        - SIMBA range: (attr >= lo and attr < hi) connected by OR
        - Categorical: (attr = 'value')
        - Simple comparisons: attr > val, attr < val, attr BETWEEN lo AND hi
        """
        bounds: Dict[str, AttributeBound] = {}

        # Extract WHERE clause
        where_match = re.search(
            r'\bWHERE\s+(.*?)(?:\bORDER\b|\bGROUP\b|\bLIMIT\b|\bHAVING\b|$)',
            sql, re.IGNORECASE | re.DOTALL
        )
        if not where_match:
            return bounds

        where_clause = where_match.group(1).strip()

        # Parse range predicates (SIMBA style: attr >= lo and attr < hi)
        self._parse_range_predicates(where_clause, bounds)

        # Parse categorical predicates (attr = 'value')
        self._parse_categorical_predicates(where_clause, bounds)

        # Parse BETWEEN predicates
        self._parse_between_predicates(where_clause, bounds)

        # Parse simple comparison predicates (attr > val, attr < val, etc.)
        self._parse_simple_comparisons(where_clause, bounds)

        return bounds

    def _parse_range_predicates(self, where_clause: str, bounds: Dict[str, AttributeBound]):
        """Parse SIMBA-style range predicates: (attr >= lo and attr < hi).

        These appear as OR-connected groups like:
        (attr >= 20130915.0 and attr < 20130930.0) or (attr >= 20131001.0 and attr < 20131015.0)
        """
        pattern = r'(\w+)\s*>=\s*([\d.eE+-]+)\s+and\s+\1\s*<\s*([\d.eE+-]+)'
        for match in re.finditer(pattern, where_clause, re.IGNORECASE):
            attr = match.group(1)
            lo = float(match.group(2))
            hi = float(match.group(3))

            if attr not in bounds:
                bounds[attr] = AttributeBound()
            bounds[attr].intervals.append((lo, hi))

    def _parse_categorical_predicates(self, where_clause: str, bounds: Dict[str, AttributeBound]):
        """Parse categorical equality predicates: (attr = 'value')."""
        pattern = r"(\w+)\s*=\s*'([^']+)'"
        for match in re.finditer(pattern, where_clause, re.IGNORECASE):
            attr = match.group(1)
            value = match.group(2)

            if attr not in bounds:
                bounds[attr] = AttributeBound()
            bounds[attr].categorical_values.add(value)

    def _parse_between_predicates(self, where_clause: str, bounds: Dict[str, AttributeBound]):
        """Parse BETWEEN predicates: attr BETWEEN lo AND hi."""
        pattern = r'(\w+)\s+BETWEEN\s+([\d.eE+-]+)\s+AND\s+([\d.eE+-]+)'
        for match in re.finditer(pattern, where_clause, re.IGNORECASE):
            attr = match.group(1)
            lo = float(match.group(2))
            hi = float(match.group(3))

            if attr not in bounds:
                bounds[attr] = AttributeBound()
            # Only add if not already covered by range predicates
            if not bounds[attr].intervals or not any(
                abs(l - lo) < 1e-10 and abs(h - hi) < 1e-10
                for l, h in bounds[attr].intervals
            ):
                bounds[attr].intervals.append((lo, hi))

    def _parse_simple_comparisons(self, where_clause: str, bounds: Dict[str, AttributeBound]):
        """Parse simple comparison operators: attr > val, attr < val, attr >= val, attr <= val.

        Skips attributes already handled by range or BETWEEN predicates.
        """
        # Match attr > val or attr >= val (but not part of range predicate pair)
        gt_pattern = r'(\w+)\s*>\s*([\d.eE+-]+)'
        lt_pattern = r'(\w+)\s*<\s*([\d.eE+-]+)'

        for match in re.finditer(gt_pattern, where_clause):
            attr = match.group(1)
            val = float(match.group(2))
            # Skip if already handled by range predicates
            if attr in bounds and bounds[attr].intervals:
                continue
            if attr not in bounds:
                bounds[attr] = AttributeBound()
            # Use val as lower bound, with a large upper bound
            bounds[attr].intervals.append((val, val + 1e10))

        for match in re.finditer(lt_pattern, where_clause):
            attr = match.group(1)
            val = float(match.group(2))
            # Skip if already handled by range predicates
            if attr in bounds and bounds[attr].intervals:
                continue
            if attr not in bounds:
                bounds[attr] = AttributeBound()
            bounds[attr].intervals.append((-1e10, val))
