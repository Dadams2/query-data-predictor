"""
Query Expansion Recommender - Out of Results Recommender Implementation.

This recommender analyzes current query results and generates new database queries
to find additional interesting tuples beyond the current result set.
"""

import pandas as pd
import numpy as np
import re
import sqlparse
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict
import logging
import time
from dataclasses import dataclass

from .base_recommender import BaseRecommender
from ..query_runner import QueryRunner

logger = logging.getLogger(__name__)


@dataclass
class QueryCandidate:
    """Represents a candidate query for expansion."""
    query: str
    confidence: float
    expansion_type: str
    source_pattern: str
    estimated_cost: int = 1


class QueryBudgetManager:
    """Manages computational budget for query execution."""
    
    def __init__(self, max_queries: int = 5, max_execution_time: float = 30.0, max_results_per_query: int = 1000):
        self.max_queries = max_queries
        self.max_execution_time = max_execution_time
        self.max_results_per_query = max_results_per_query
        self.executed_queries = 0
        self.start_time = None
        self.total_results = 0
    
    def start_session(self):
        """Start a new query session."""
        self.executed_queries = 0
        self.total_results = 0
        self.start_time = time.time()
    
    def can_execute_query(self) -> bool:
        """Check if we can execute another query within budget."""
        if self.start_time is None:
            self.start_session()
        
        if self.executed_queries >= self.max_queries:
            logger.debug(f"Query limit reached: {self.executed_queries}/{self.max_queries}")
            return False
        
        elapsed = time.time() - self.start_time
        if elapsed > self.max_execution_time:
            logger.debug(f"Time limit reached: {elapsed:.2f}s/{self.max_execution_time}s")
            return False
        
        if self.total_results >= self.max_results_per_query * self.max_queries:
            logger.debug(f"Result limit reached: {self.total_results}")
            return False
        
        return True
    
    def record_execution(self, result_count: int):
        """Record the execution of a query."""
        self.executed_queries += 1
        self.total_results += result_count


class QueryExpansionRecommender(BaseRecommender):
    """
    Recommender that generates new database queries to find additional interesting tuples.
    
    This recommender analyzes current query results and database schema to generate
    expansion queries that find related, similar, or scientifically interesting data
    beyond the current result set.
    """
    
    def __init__(self, config: Dict[str, Any], query_runner: Optional[QueryRunner] = None):
        """
        Initialize the query expansion recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
            query_runner: Database query runner instance (required for out-of-results queries)
        """
        super().__init__(config)
        
        if query_runner is None:
            raise ValueError("QueryRunner is required for QueryExpansionRecommender")
        
        self.query_runner = query_runner
        
        # Configuration
        expansion_config = self.config.get('query_expansion', {})
        self.enable_range_expansion = expansion_config.get('enable_range_expansion', True)
        self.enable_constraint_relaxation = expansion_config.get('enable_constraint_relaxation', True)
        self.enable_join_exploration = expansion_config.get('enable_join_exploration', True)
        self.enable_column_expansion = expansion_config.get('enable_column_expansion', True)
        self.enable_similarity_search = expansion_config.get('enable_similarity_search', True)
        
        # Budget management
        budget_config = expansion_config.get('budget', {})
        self.budget_manager = QueryBudgetManager(
            max_queries=budget_config.get('max_queries', 5),
            max_execution_time=budget_config.get('max_execution_time', 30.0),
            max_results_per_query=budget_config.get('max_results_per_query', 1000)
        )
        
        # Query generation parameters
        self.range_expansion_factor = expansion_config.get('range_expansion_factor', 0.2)  # 20% expansion
        self.min_confidence_threshold = expansion_config.get('min_confidence_threshold', 0.3)
        self.max_query_complexity = expansion_config.get('max_query_complexity', 5)
        
        # SDSS-specific knowledge
        self.sdss_tables = {
            'SpecObj': ['specObjID', 'z', 'zConf', 'SpecClass', 'objID', 'primTarget'],
            'PhotoObj': ['objID', 'ra', 'dec', 'type', 'modelmag_g', 'modelmag_r', 'modelmag_i'],
            'SpecPhotoAll': ['specObjID', 'objID', 'z', 'zConf', 'SpecClass', 'primTarget', 'ra', 'dec', 'modelMag_r']
        }
        
        self.common_joins = [
            ('SpecObj', 'PhotoObj', 'objID'),
            ('SpecObj', 'SpecPhotoAll', 'specObjID'),
            ('PhotoObj', 'SpecPhotoAll', 'objID')
        ]
        
        # Cache for query patterns
        self._query_templates = {}
        self._last_session_context = None
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples by executing expansion queries against the database.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments including session context
            
        Returns:
            DataFrame with recommended tuples from database queries
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        logger.info(f"Starting query expansion recommendation for {len(current_results)} current results")
        
        # Start budget management
        self.budget_manager.start_session()
        
        try:
            # 1. Analyze current results to understand patterns
            analysis = self._analyze_current_results(current_results, **kwargs)
            
            # 2. Generate expansion query candidates
            query_candidates = self._generate_expansion_queries(current_results, analysis, **kwargs)
            
            # 3. Rank and filter candidates
            ranked_candidates = self._rank_query_candidates(query_candidates, analysis)
            
            # 4. Execute promising queries within budget
            expansion_results = self._execute_expansion_queries(ranked_candidates)
            
            # 5. Combine, deduplicate, and rank final results
            final_recommendations = self._combine_and_rank_results(
                current_results, expansion_results, analysis
            )
            
            # Apply output limiting
            return self._limit_output(final_recommendations)
            
        except Exception as e:
            logger.error(f"Error in query expansion recommendation: {e}", exc_info=True)
            # Fallback: return empty DataFrame rather than crash
            return pd.DataFrame()
    
    def _analyze_current_results(self, current_results: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze current results to understand patterns and context.
        
        Args:
            current_results: Current query results
            **kwargs: Additional context
            
        Returns:
            Analysis dictionary with patterns and statistics
        """
        analysis = {
            'row_count': len(current_results),
            'column_count': len(current_results.columns),
            'columns': list(current_results.columns),
            'numeric_columns': [],
            'categorical_columns': [],
            'ranges': {},
            'distributions': {},
            'dominant_values': {},
            'inferred_table': None,
            'inferred_query_type': None
        }
        
        # Analyze column types and statistics
        for col in current_results.columns:
            if pd.api.types.is_numeric_dtype(current_results[col]):
                analysis['numeric_columns'].append(col)
                # Calculate ranges for numeric columns
                if not current_results[col].empty and current_results[col].notna().any():
                    analysis['ranges'][col] = {
                        'min': current_results[col].min(),
                        'max': current_results[col].max(),
                        'mean': current_results[col].mean(),
                        'std': current_results[col].std()
                    }
            else:
                analysis['categorical_columns'].append(col)
                # Get value distributions for categorical columns
                value_counts = current_results[col].value_counts()
                analysis['distributions'][col] = value_counts.head(10).to_dict()
                if not value_counts.empty:
                    analysis['dominant_values'][col] = value_counts.index[0]
        
        # Infer likely table based on column patterns
        analysis['inferred_table'] = self._infer_table_from_columns(analysis['columns'])
        
        # Try to extract query context if available
        if 'current_query' in kwargs:
            analysis['source_query'] = kwargs['current_query']
            analysis['inferred_query_type'] = self._infer_query_type(kwargs['current_query'])
        
        logger.debug(f"Analysis complete: {analysis['inferred_table']} table with {len(analysis['numeric_columns'])} numeric columns")
        
        return analysis
    
    def _generate_expansion_queries(self, current_results: pd.DataFrame, 
                                   analysis: Dict[str, Any], **kwargs) -> List[QueryCandidate]:
        """
        Generate candidate expansion queries based on current results analysis.
        
        Args:
            current_results: Current query results
            analysis: Analysis results from _analyze_current_results
            **kwargs: Additional context
            
        Returns:
            List of QueryCandidate objects
        """
        candidates = []
        
        # 1. Range expansion queries
        if self.enable_range_expansion:
            candidates.extend(self._generate_range_expansion_queries(current_results, analysis))
        
        # 2. Constraint relaxation queries
        if self.enable_constraint_relaxation:
            candidates.extend(self._generate_constraint_relaxation_queries(analysis, **kwargs))
        
        # 3. Join exploration queries
        if self.enable_join_exploration:
            candidates.extend(self._generate_join_exploration_queries(current_results, analysis))
        
        # 4. Column expansion queries
        if self.enable_column_expansion:
            candidates.extend(self._generate_column_expansion_queries(current_results, analysis))
        
        # 5. Similarity search queries
        if self.enable_similarity_search:
            candidates.extend(self._generate_similarity_search_queries(current_results, analysis))
        
        logger.debug(f"Generated {len(candidates)} query candidates")
        return candidates
    
    def _generate_range_expansion_queries(self, current_results: pd.DataFrame, 
                                        analysis: Dict[str, Any]) -> List[QueryCandidate]:
        """Generate queries that expand numeric ranges."""
        candidates = []
        
        if not analysis.get('inferred_table'):
            return candidates
        
        table = analysis['inferred_table']
        
        for col in analysis['numeric_columns']:
            if col in analysis['ranges']:
                range_info = analysis['ranges'][col]
                
                # Expand range by configured factor
                range_span = range_info['max'] - range_info['min']
                expansion = range_span * self.range_expansion_factor
                
                new_min = range_info['min'] - expansion
                new_max = range_info['max'] + expansion
                
                # Generate expansion query
                if col in ['z', 'zConf']:  # Special handling for redshift
                    query = f"""
                    SELECT {', '.join(analysis['columns'][:6])}  -- Limit columns for performance
                    FROM {table}
                    WHERE {col} BETWEEN {new_min:.6f} AND {new_max:.6f}
                    AND {col} NOT BETWEEN {range_info['min']:.6f} AND {range_info['max']:.6f}
                    LIMIT {self.budget_manager.max_results_per_query}
                    """
                elif col in ['ra', 'dec']:  # Coordinate expansion
                    query = f"""
                    SELECT {', '.join(analysis['columns'][:6])}
                    FROM {table}
                    WHERE {col} BETWEEN {new_min:.6f} AND {new_max:.6f}
                    AND {col} NOT BETWEEN {range_info['min']:.6f} AND {range_info['max']:.6f}
                    LIMIT {self.budget_manager.max_results_per_query}
                    """
                else:  # Generic numeric expansion
                    query = f"""
                    SELECT {', '.join(analysis['columns'][:6])}
                    FROM {table}
                    WHERE {col} BETWEEN {new_min} AND {new_max}
                    AND {col} NOT BETWEEN {range_info['min']} AND {range_info['max']}
                    LIMIT {self.budget_manager.max_results_per_query}
                    """
                
                candidates.append(QueryCandidate(
                    query=query.strip(),
                    confidence=0.8,
                    expansion_type="range_expansion",
                    source_pattern=f"{col}_range_expansion",
                    estimated_cost=2
                ))
        
        return candidates
    
    def _generate_constraint_relaxation_queries(self, analysis: Dict[str, Any], **kwargs) -> List[QueryCandidate]:
        """Generate queries by relaxing constraints from original query."""
        candidates = []
        
        if 'source_query' not in analysis:
            return candidates
        
        source_query = analysis['source_query']
        
        try:
            # Parse the source query
            parsed = sqlparse.parse(source_query)[0]
            
            # Extract WHERE conditions
            where_conditions = self._extract_where_conditions(source_query)
            
            if not where_conditions:
                return candidates
            
            # Try relaxing different constraints
            for condition in where_conditions[:3]:  # Limit to first 3 conditions
                relaxed_query = self._relax_condition(source_query, condition)
                if relaxed_query and relaxed_query != source_query:
                    candidates.append(QueryCandidate(
                        query=relaxed_query,
                        confidence=0.6,
                        expansion_type="constraint_relaxation",
                        source_pattern=f"relax_{condition[:20]}",
                        estimated_cost=3
                    ))
        
        except Exception as e:
            logger.debug(f"Could not parse source query for constraint relaxation: {e}")
        
        return candidates
    
    def _generate_join_exploration_queries(self, current_results: pd.DataFrame, 
                                         analysis: Dict[str, Any]) -> List[QueryCandidate]:
        """Generate queries that join with related tables."""
        candidates = []
        
        inferred_table = analysis.get('inferred_table')
        if not inferred_table:
            return candidates
        
        # Find possible joins
        for table1, table2, join_col in self.common_joins:
            if inferred_table == table1:
                # Join with table2
                target_table = table2
                join_column = join_col
            elif inferred_table == table2:
                # Join with table1
                target_table = table1
                join_column = join_col
            else:
                continue
            
            # Check if we have the join column
            if join_column not in analysis['columns']:
                continue
            
            # Get some values from the join column for the query
            if join_column in current_results.columns:
                sample_values = current_results[join_column].dropna().head(10).tolist()
                if not sample_values:
                    continue
                
                # Create IN clause with sample values
                if isinstance(sample_values[0], str):
                    values_str = "', '".join(str(v) for v in sample_values)
                    in_clause = f"'{values_str}'"
                else:
                    values_str = ", ".join(str(v) for v in sample_values)
                    in_clause = values_str
                
                # Generate join query
                target_columns = self.sdss_tables.get(target_table, ['*'])[:6]  # Limit columns
                query = f"""
                SELECT {', '.join(target_columns)}
                FROM {target_table}
                WHERE {join_column} IN ({in_clause})
                LIMIT {self.budget_manager.max_results_per_query}
                """
                
                candidates.append(QueryCandidate(
                    query=query,
                    confidence=0.7,
                    expansion_type="join_exploration",
                    source_pattern=f"join_{target_table}",
                    estimated_cost=3
                ))
        
        return candidates
    
    def _generate_column_expansion_queries(self, current_results: pd.DataFrame, 
                                         analysis: Dict[str, Any]) -> List[QueryCandidate]:
        """Generate queries that add more columns from the same table."""
        candidates = []
        
        inferred_table = analysis.get('inferred_table')
        if not inferred_table or inferred_table not in self.sdss_tables:
            return candidates
        
        # Get all available columns for this table
        all_columns = self.sdss_tables[inferred_table]
        current_columns = set(analysis['columns'])
        additional_columns = [col for col in all_columns if col not in current_columns]
        
        if not additional_columns:
            return candidates
        
        # Use dominant values or ranges to recreate the query with more columns
        conditions = []
        
        # Add conditions based on current data patterns
        for col in analysis['categorical_columns']:
            if col in analysis['dominant_values']:
                dominant_val = analysis['dominant_values'][col]
                if isinstance(dominant_val, str):
                    conditions.append(f"{col} = '{dominant_val}'")
                else:
                    conditions.append(f"{col} = {dominant_val}")
        
        for col in analysis['numeric_columns']:
            if col in analysis['ranges']:
                range_info = analysis['ranges'][col]
                conditions.append(f"{col} BETWEEN {range_info['min']} AND {range_info['max']}")
        
        if conditions:
            # Combine current and additional columns
            expanded_columns = list(current_columns) + additional_columns[:3]  # Add up to 3 new columns
            
            query = f"""
            SELECT {', '.join(expanded_columns)}
            FROM {inferred_table}
            WHERE {' AND '.join(conditions[:3])}  -- Limit conditions for performance
            LIMIT {self.budget_manager.max_results_per_query}
            """
            
            candidates.append(QueryCandidate(
                query=query,
                confidence=0.5,
                expansion_type="column_expansion",
                source_pattern="add_columns",
                estimated_cost=2
            ))
        
        return candidates
    
    def _generate_similarity_search_queries(self, current_results: pd.DataFrame, 
                                          analysis: Dict[str, Any]) -> List[QueryCandidate]:
        """Generate queries to find similar objects based on key characteristics."""
        candidates = []
        
        inferred_table = analysis.get('inferred_table')
        if not inferred_table:
            return candidates
        
        # For astronomical data, create similarity queries based on key parameters
        if 'z' in analysis['numeric_columns'] and 'z' in analysis['ranges']:
            # Find objects with similar redshifts
            z_range = analysis['ranges']['z']
            z_tolerance = 0.05  # 5% tolerance
            
            query = f"""
            SELECT {', '.join(analysis['columns'][:6])}
            FROM {inferred_table}
            WHERE z BETWEEN {z_range['mean'] - z_tolerance} AND {z_range['mean'] + z_tolerance}
            AND zConf > 0.95
            LIMIT {self.budget_manager.max_results_per_query}
            """
            
            candidates.append(QueryCandidate(
                query=query,
                confidence=0.6,
                expansion_type="similarity_search",
                source_pattern="similar_redshift",
                estimated_cost=2
            ))
        
        if 'ra' in analysis['numeric_columns'] and 'dec' in analysis['numeric_columns']:
            # Find nearby objects
            if 'ra' in analysis['ranges'] and 'dec' in analysis['ranges']:
                ra_center = analysis['ranges']['ra']['mean']
                dec_center = analysis['ranges']['dec']['mean']
                radius = 0.1  # 0.1 degree radius
                
                query = f"""
                SELECT {', '.join(analysis['columns'][:6])}
                FROM {inferred_table}
                WHERE ra BETWEEN {ra_center - radius} AND {ra_center + radius}
                AND dec BETWEEN {dec_center - radius} AND {dec_center + radius}
                LIMIT {self.budget_manager.max_results_per_query}
                """
                
                candidates.append(QueryCandidate(
                    query=query,
                    confidence=0.6,
                    expansion_type="similarity_search",
                    source_pattern="spatial_proximity",
                    estimated_cost=2
                ))
        
        return candidates
    
    def _rank_query_candidates(self, candidates: List[QueryCandidate], 
                             analysis: Dict[str, Any]) -> List[QueryCandidate]:
        """Rank query candidates by confidence and estimated value."""
        if not candidates:
            return []
        
        # Sort by confidence (descending) and cost (ascending)
        ranked = sorted(candidates, key=lambda c: (-c.confidence, c.estimated_cost))
        
        # Filter by minimum confidence threshold
        filtered = [c for c in ranked if c.confidence >= self.min_confidence_threshold]
        
        logger.debug(f"Ranked {len(filtered)} candidates above confidence threshold {self.min_confidence_threshold}")
        
        return filtered
    
    def _execute_expansion_queries(self, candidates: List[QueryCandidate]) -> List[pd.DataFrame]:
        """Execute expansion queries within budget constraints."""
        results = []
        
        for candidate in candidates:
            if not self.budget_manager.can_execute_query():
                logger.debug("Query budget exhausted, stopping execution")
                break
            
            try:
                logger.debug(f"Executing {candidate.expansion_type} query: {candidate.source_pattern}")
                
                # Execute the query
                result_df = self.query_runner.execute_query(candidate.query)
                
                # Record execution
                self.budget_manager.record_execution(len(result_df))
                
                if not result_df.empty:
                    # Add metadata about the expansion
                    result_df = result_df.copy()
                    result_df['_expansion_type'] = candidate.expansion_type
                    result_df['_expansion_confidence'] = candidate.confidence
                    results.append(result_df)
                    
                    logger.debug(f"Query returned {len(result_df)} results")
                else:
                    logger.debug("Query returned no results")
                    
            except Exception as e:
                logger.warning(f"Failed to execute expansion query ({candidate.expansion_type}): {e}")
                # Record the attempt even if it failed
                self.budget_manager.record_execution(0)
                continue
        
        logger.info(f"Executed {self.budget_manager.executed_queries} expansion queries, "
                   f"retrieved {self.budget_manager.total_results} total results")
        
        return results
    
    def _combine_and_rank_results(self, current_results: pd.DataFrame, 
                                expansion_results: List[pd.DataFrame], 
                                analysis: Dict[str, Any]) -> pd.DataFrame:
        """Combine and rank all results."""
        if not expansion_results:
            return current_results.copy()
        
        # Combine all expansion results
        combined_expansions = []
        for df in expansion_results:
            # Remove metadata columns before combining
            df_clean = df.drop(columns=[col for col in df.columns if col.startswith('_')], errors='ignore')
            combined_expansions.append(df_clean)
        
        if not combined_expansions:
            return current_results.copy()
        
        # Concatenate all expansion results
        all_expansion_results = pd.concat(combined_expansions, ignore_index=True)
        
        # Remove duplicates (both within expansions and with current results)
        # Use a subset of columns for duplicate detection to be more lenient
        key_columns = []
        for col in ['objID', 'specObjID', 'ra', 'dec']:
            if col in all_expansion_results.columns:
                key_columns.append(col)
                break  # Use first available key column
        
        if key_columns:
            all_expansion_results = all_expansion_results.drop_duplicates(subset=key_columns)
        else:
            # Fallback: drop exact duplicates
            all_expansion_results = all_expansion_results.drop_duplicates()
        
        # Remove results that are already in current_results
        if key_columns:
            # Anti-join: keep expansion results not in current results
            merged = all_expansion_results.merge(
                current_results[key_columns], on=key_columns, how='left', indicator=True
            )
            new_results = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        else:
            new_results = all_expansion_results
        
        # Rank expansion results by interestingness
        if not new_results.empty:
            new_results = self._rank_expansion_results(new_results, analysis)
        
        logger.info(f"Combined results: {len(current_results)} current + {len(new_results)} new = "
                   f"{len(current_results) + len(new_results)} total")
        
        # Combine current and new results, prioritizing current results
        if not new_results.empty:
            final_results = pd.concat([current_results, new_results], ignore_index=True)
        else:
            final_results = current_results.copy()
        
        return final_results
    
    def _rank_expansion_results(self, expansion_results: pd.DataFrame, 
                              analysis: Dict[str, Any]) -> pd.DataFrame:
        """Rank expansion results by interestingness."""
        if expansion_results.empty:
            return expansion_results
        
        # Simple ranking based on completeness and diversity
        scores = np.ones(len(expansion_results))
        
        # Boost score for complete records (fewer null values)
        null_counts = expansion_results.isnull().sum(axis=1)
        completeness_scores = 1.0 - (null_counts / len(expansion_results.columns))
        scores = scores * completeness_scores
        
        # Add some randomness to ensure diversity
        random_factor = np.random.random(len(expansion_results)) * 0.1
        scores = scores + random_factor
        
        # Sort by score
        expansion_results_copy = expansion_results.copy()
        expansion_results_copy['_rank_score'] = scores
        ranked = expansion_results_copy.sort_values('_rank_score', ascending=False)
        
        return ranked.drop(columns=['_rank_score'])
    
    # Helper methods
    
    def _infer_table_from_columns(self, columns: List[str]) -> Optional[str]:
        """Infer the most likely table based on column names."""
        best_match = None
        best_score = 0
        
        for table, table_columns in self.sdss_tables.items():
            # Calculate overlap score
            overlap = len(set(columns) & set(table_columns))
            score = overlap / len(table_columns) if table_columns else 0
            
            if score > best_score:
                best_score = score
                best_match = table
        
        return best_match if best_score > 0.3 else None
    
    def _infer_query_type(self, query: str) -> str:
        """Infer the type of query from the SQL string."""
        query_lower = query.lower()
        
        if 'qso' in query_lower or 'quasar' in query_lower:
            return 'quasar_search'
        elif 'spec' in query_lower:
            return 'spectroscopic_search'
        elif 'photo' in query_lower:
            return 'photometric_search'
        else:
            return 'general_search'
    
    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions from a SQL query."""
        # Simple regex-based extraction (could be improved with proper SQL parsing)
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return []
        
        where_clause = where_match.group(1).strip()
        
        # Split on AND/OR (simple approach)
        conditions = re.split(r'\s+(?:AND|OR)\s+', where_clause, flags=re.IGNORECASE)
        
        return [cond.strip() for cond in conditions if cond.strip()]
    
    def _relax_condition(self, query: str, condition: str) -> Optional[str]:
        """Relax a specific condition in the query."""
        # For range conditions, expand the range
        range_match = re.search(r'(\w+)\s+between\s+([\d.]+)\s+and\s+([\d.]+)', condition, re.IGNORECASE)
        if range_match:
            col, min_val, max_val = range_match.groups()
            min_val, max_val = float(min_val), float(max_val)
            
            # Expand range by 20%
            range_span = max_val - min_val
            expansion = range_span * 0.2
            new_min = min_val - expansion
            new_max = max_val + expansion
            
            new_condition = f"{col} BETWEEN {new_min} AND {new_max}"
            relaxed_query = query.replace(condition, new_condition)
            
            # Add LIMIT if not present
            if 'limit' not in relaxed_query.lower():
                relaxed_query += f" LIMIT {self.budget_manager.max_results_per_query}"
            
            return relaxed_query
        
        # For equality conditions, remove them entirely (risky, so we return None)
        return None
    
    def name(self) -> str:
        """Return the name of the recommender."""
        return "QueryExpansionRecommender"
