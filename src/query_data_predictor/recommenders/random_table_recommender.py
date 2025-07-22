"""
Random Table Recommender - Baseline for out-of-results recommendations.

This recommender selects random tuples from the same table as the current query results.
It serves as a simple baseline for comparison with more sophisticated out-of-results methods.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base_recommender import BaseRecommender
from ..query_runner import QueryRunner

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[str]
    estimated_rows: int = 0


class RandomTableRecommender(BaseRecommender):
    """
    Baseline recommender that selects random tuples from the same table as current results.
    
    This recommender infers the source table from current query results and generates
    a simple random sampling query to find additional tuples from the same table.
    It serves as a baseline for comparison with more sophisticated query expansion methods.
    """
    
    def __init__(self, config: Dict[str, Any], query_runner: Optional[QueryRunner] = None):
        """
        Initialize the random table recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
            query_runner: Database query runner instance (required for database access)
        """
        super().__init__(config)
        
        if query_runner is None:
            raise ValueError("QueryRunner is required for RandomTableRecommender")
        
        self.query_runner = query_runner
        
        # Configuration
        random_config = self.config.get('random_table', {})
        self.max_sample_size = random_config.get('max_sample_size', 1000)
        self.min_sample_size = random_config.get('min_sample_size', 10)
        self.exclude_current_results = random_config.get('exclude_current_results', True)
        
        # SDSS-specific table knowledge (can be extended for other domains)
        self.known_tables = {
            'SpecObj': ['specObjID', 'z', 'zConf', 'SpecClass', 'objID', 'primTarget', 'ra', 'dec'],
            'PhotoObj': ['objID', 'ra', 'dec', 'type', 'modelmag_g', 'modelmag_r', 'modelmag_i', 'run', 'rerun'],
            'SpecPhotoAll': ['specObjID', 'objID', 'z', 'zConf', 'SpecClass', 'primTarget', 'ra', 'dec', 'modelMag_r']
        }
        
        # Cache for table information
        self._table_info_cache = {}
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples by randomly sampling from the same table as current results.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments including session context
            
        Returns:
            DataFrame with recommended tuples from random table sampling
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        logger.info(f"Starting random table recommendation for {len(current_results)} current results")
        
        try:
            # 1. Infer the source table from current results
            table_info = self._infer_source_table(current_results, **kwargs)
            
            if not table_info:
                logger.warning("Could not infer source table, returning empty results")
                return pd.DataFrame()
            
            # 2. Generate random sampling query
            random_query = self._generate_random_sampling_query(table_info, current_results)
            
            if not random_query:
                logger.warning("Could not generate random sampling query")
                return pd.DataFrame()
            
            # 3. Execute the random sampling query
            random_results = self._execute_random_query(random_query)
            
            # 4. Post-process and filter results
            final_results = self._post_process_results(random_results, current_results)
            
            # Apply output limiting
            return self._limit_output(final_results)
            
        except Exception as e:
            logger.error(f"Error in random table recommendation: {e}", exc_info=True)
            # Fallback: return empty DataFrame rather than crash
            return pd.DataFrame()
    
    def _infer_source_table(self, current_results: pd.DataFrame, **kwargs) -> Optional[TableInfo]:
        """
        Infer the source table from current results and context.
        
        Args:
            current_results: Current query results
            **kwargs: Additional context including potential source query
            
        Returns:
            TableInfo object or None if table cannot be inferred
        """
        columns = list(current_results.columns)
        
        # Method 1: Try to extract table from source query if available
        if 'current_query' in kwargs:
            table_from_query = self._extract_table_from_query(kwargs['current_query'])
            if table_from_query:
                return TableInfo(
                    name=table_from_query,
                    columns=self.known_tables.get(table_from_query, columns)
                )
        
        # Method 2: Match columns against known tables
        best_match = None
        best_score = 0
        
        for table_name, table_columns in self.known_tables.items():
            # Calculate column overlap score
            overlap = len(set(columns) & set(table_columns))
            score = overlap / max(len(table_columns), len(columns)) if max(len(table_columns), len(columns)) > 0 else 0
            
            if score > best_score and score > 0.3:  # At least 30% overlap
                best_score = score
                best_match = table_name
        
        if best_match:
            logger.debug(f"Inferred source table: {best_match} (score: {best_score:.2f})")
            return TableInfo(
                name=best_match,
                columns=self.known_tables[best_match]
            )
        
        # Method 3: If we can't identify a known table, try to guess from data
        # This is a fallback for unknown schemas
        if len(columns) > 0:
            # Use the first columns that might indicate a table
            guessed_table = self._guess_table_name(columns)
            if guessed_table:
                return TableInfo(
                    name=guessed_table,
                    columns=columns
                )
        
        logger.warning(f"Could not infer source table from columns: {columns}")
        return None
    
    def _extract_table_from_query(self, query: str) -> Optional[str]:
        """Extract table name from SQL query using simple regex."""
        try:
            # Look for FROM clause
            from_match = re.search(r'\bFROM\s+(\w+)', query, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1)
                # Validate against known tables
                if table_name in self.known_tables:
                    return table_name
                # Return even if not in known tables, might be valid
                return table_name
            
            # Look for JOIN clauses
            join_match = re.search(r'\bJOIN\s+(\w+)', query, re.IGNORECASE)
            if join_match:
                table_name = join_match.group(1)
                if table_name in self.known_tables:
                    return table_name
                return table_name
        
        except Exception as e:
            logger.debug(f"Error extracting table from query: {e}")
        
        return None
    
    def _guess_table_name(self, columns: List[str]) -> Optional[str]:
        """Guess table name from column patterns."""
        # Simple heuristics based on common column naming patterns
        column_str = ' '.join(columns).lower()
        
        if 'specobj' in column_str or 'spec_obj' in column_str:
            return 'SpecObj'
        elif 'photoobj' in column_str or 'photo_obj' in column_str:
            return 'PhotoObj'
        elif any(col.lower().startswith('spec') for col in columns):
            return 'SpecObj'  # Default to SpecObj for spectroscopic data
        elif any(col.lower().startswith('photo') for col in columns):
            return 'PhotoObj'  # Default to PhotoObj for photometric data
        
        # If we have ra/dec coordinates, could be any astronomical table
        if 'ra' in columns and 'dec' in columns:
            # Default to PhotoObj as it's the most general
            return 'PhotoObj'
        
        return None
    
    def _generate_random_sampling_query(self, table_info: TableInfo, current_results: pd.DataFrame) -> Optional[str]:
        """
        Generate a random sampling query for the identified table.
        
        Args:
            table_info: Information about the source table
            current_results: Current query results for reference
            
        Returns:
            SQL query string for random sampling
        """
        try:
            # Determine columns to select
            # Use the same columns as current results if they exist in the table
            current_columns = list(current_results.columns)
            available_columns = table_info.columns
            
            # Select columns that exist in both current results and the table
            select_columns = []
            for col in current_columns:
                if col in available_columns:
                    select_columns.append(col)
            
            # If no matching columns, use the first few columns from the table
            if not select_columns:
                select_columns = available_columns[:min(6, len(available_columns))]
            
            if not select_columns:
                logger.warning("No columns available for random sampling query")
                return None
            
            # Determine sample size
            target_sample_size = self._determine_sample_size(len(current_results))
            
            # Build the query
            columns_str = ', '.join(select_columns)
            
            # Generate exclusion clause if needed
            exclusion_clause = ""
            if self.exclude_current_results:
                exclusion_clause = self._build_exclusion_clause(table_info, current_results)
            
            # Use TABLESAMPLE if available (PostgreSQL), otherwise ORDER BY RANDOM()
            # TABLESAMPLE is more efficient for large tables
            query = f"""
            SELECT {columns_str}
            FROM {table_info.name}
            {exclusion_clause}
            ORDER BY RANDOM()
            LIMIT {target_sample_size}
            """
            
            return query.strip()
            
        except Exception as e:
            logger.error(f"Error generating random sampling query: {e}")
            return None
    
    def _determine_sample_size(self, current_result_size: int) -> int:
        """Determine appropriate sample size based on current results and configuration."""
        # Base sample size on current result size
        base_size = max(current_result_size, self.min_sample_size)
        
        # Apply maximum limit
        sample_size = min(base_size * 2, self.max_sample_size)  # Get up to 2x current results
        
        # Ensure we stay within bounds
        sample_size = max(self.min_sample_size, min(sample_size, self.max_sample_size))
        
        logger.debug(f"Determined sample size: {sample_size} (current results: {current_result_size})")
        return sample_size
    
    def _build_exclusion_clause(self, table_info: TableInfo, current_results: pd.DataFrame) -> str:
        """
        Build a WHERE clause to exclude current results from random sampling.
        
        Args:
            table_info: Information about the source table
            current_results: Current results to exclude
            
        Returns:
            SQL WHERE clause string
        """
        if current_results.empty:
            return ""
        
        # Find a suitable key column for exclusion
        key_columns = ['objID', 'specObjID', 'id']  # Common key columns in SDSS
        exclusion_column = None
        
        for col in key_columns:
            if col in current_results.columns and col in table_info.columns:
                exclusion_column = col
                break
        
        if not exclusion_column:
            # If no key column found, skip exclusion to avoid complex queries
            logger.debug("No suitable key column found for exclusion, skipping exclusion clause")
            return ""
        
        # Get unique values from current results
        unique_values = current_results[exclusion_column].dropna().unique()
        
        # Limit the number of values to avoid overly long queries
        if len(unique_values) > 100:
            logger.debug(f"Too many values for exclusion ({len(unique_values)}), skipping exclusion clause")
            return ""
        
        if len(unique_values) == 0:
            return ""
        
        # Build exclusion clause
        if isinstance(unique_values[0], str):
            values_str = "', '".join(str(v) for v in unique_values)
            exclusion_clause = f"WHERE {exclusion_column} NOT IN ('{values_str}')"
        else:
            values_str = ", ".join(str(v) for v in unique_values)
            exclusion_clause = f"WHERE {exclusion_column} NOT IN ({values_str})"
        
        logger.debug(f"Built exclusion clause for {len(unique_values)} values")
        return exclusion_clause
    
    def _execute_random_query(self, query: str) -> pd.DataFrame:
        """Execute the random sampling query."""
        try:
            logger.debug(f"Executing random sampling query: {query[:200]}{'...' if len(query) > 200 else ''}")
            result_df = self.query_runner.execute_query(query)
            logger.debug(f"Random sampling query returned {len(result_df)} rows")
            return result_df
        except Exception as e:
            logger.error(f"Failed to execute random sampling query: {e}")
            return pd.DataFrame()
    
    def _post_process_results(self, random_results: pd.DataFrame, current_results: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process random results to ensure compatibility and quality.
        
        Args:
            random_results: Results from random sampling query
            current_results: Original current results for reference
            
        Returns:
            Processed random results
        """
        if random_results.empty:
            return random_results
        
        # Ensure column compatibility
        if not current_results.empty:
            # Reorder columns to match current results if possible
            common_columns = [col for col in current_results.columns if col in random_results.columns]
            if common_columns:
                # Keep common columns in the same order, add extra columns at the end
                extra_columns = [col for col in random_results.columns if col not in common_columns]
                new_column_order = common_columns + extra_columns
                random_results = random_results[new_column_order]
        
        # Remove any duplicates
        random_results = random_results.drop_duplicates()
        
        # Add metadata to indicate this is from random sampling
        random_results = random_results.copy()
        # We could add metadata columns, but for compatibility with existing systems, we'll skip this
        
        logger.debug(f"Post-processed {len(random_results)} random results")
        return random_results
    
    def name(self) -> str:
        """Return the name of the recommender."""
        return "RandomTableRecommender"
