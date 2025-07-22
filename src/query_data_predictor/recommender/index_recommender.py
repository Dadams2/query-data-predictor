"""
Index-based recommender with pre-built search indices for fast lookups.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class IndexRecommender(BaseRecommender):
    """
    High-performance recommender using pre-built indices for fast lookups.
    Uses inverted indices, hash tables, and tree structures for near constant-time lookups.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the index recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Configuration
        index_config = self.config.get('indexing', {})
        self.index_types = index_config.get('index_types', ['value', 'range', 'pattern'])
        self.max_index_size = index_config.get('max_index_size', 10000)
        self.enable_compound_indices = index_config.get('enable_compound_indices', True)
        self.max_compound_size = index_config.get('max_compound_size', 3)
        self.scoring_method = index_config.get('scoring_method', 'weighted')  # 'simple', 'weighted', 'compound'
        
        # Index structures
        self.value_indices = {}  # col -> {value -> [row_indices]}
        self.range_indices = {}  # col -> sorted list of (value, row_index) pairs
        self.pattern_indices = {}  # pattern -> [row_indices]
        self.compound_indices = {}  # (col1, col2, ...) -> {(val1, val2, ...) -> [row_indices]}
        self.frequency_indices = {}  # col -> {value -> frequency}
        
        # Cache for computed scores
        self._score_cache = {}
        self._last_processed_hash = None
        self._indices_built = False
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using pre-built indices for fast lookups.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # Generate hash for caching
        df_hash = pd.util.hash_pandas_object(current_results).sum()
        
        # Check cache
        if df_hash == self._last_processed_hash and df_hash in self._score_cache:
            logger.debug("Using cached index results")
            scores = self._score_cache[df_hash]
        else:
            # Build indices if not already built
            if not self._indices_built:
                self._build_indices(current_results)
            
            # Compute scores using indices
            scores = self._compute_index_scores(current_results)
            
            # Cache results
            self._score_cache[df_hash] = scores
            self._last_processed_hash = df_hash
        
        # Rank by scores
        current_results_copy = current_results.copy()
        current_results_copy['index_score'] = scores
        
        # Sort by score (descending)
        sorted_df = current_results_copy.sort_values('index_score', ascending=False)
        
        # Apply output limiting
        result_df = self._limit_output(sorted_df.drop(columns=['index_score']))
        
        return result_df
    
    def _build_indices(self, df: pd.DataFrame) -> None:
        """
        Build all types of indices for fast lookups.
        
        Args:
            df: Input DataFrame to index
        """
        if df.empty:
            return
        
        logger.debug(f"Building indices for DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        # Clear existing indices
        self._clear_indices()
        
        # Build value indices
        if 'value' in self.index_types:
            self._build_value_indices(df)
        
        # Build range indices
        if 'range' in self.index_types:
            self._build_range_indices(df)
        
        # Build pattern indices
        if 'pattern' in self.index_types:
            self._build_pattern_indices(df)
        
        # Build compound indices
        if self.enable_compound_indices:
            self._build_compound_indices(df)
        
        # Build frequency indices
        self._build_frequency_indices(df)
        
        self._indices_built = True
        logger.debug("Index building completed")
    
    def _build_value_indices(self, df: pd.DataFrame) -> None:
        """Build inverted indices for exact value lookups."""
        for col in df.columns:
            self.value_indices[col] = defaultdict(list)
            
            for idx, value in enumerate(df[col]):
                # Limit index size to prevent memory explosion
                if len(self.value_indices[col]) < self.max_index_size:
                    self.value_indices[col][value].append(idx)
    
    def _build_range_indices(self, df: pd.DataFrame) -> None:
        """Build range indices for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Create sorted list of (value, index) pairs
            value_index_pairs = [(value, idx) for idx, value in enumerate(df[col]) if pd.notna(value)]
            value_index_pairs.sort(key=lambda x: x[0])
            self.range_indices[col] = value_index_pairs
    
    def _build_pattern_indices(self, df: pd.DataFrame) -> None:
        """Build pattern indices for common patterns."""
        # Create patterns based on column combinations
        columns = list(df.columns)
        
        # Single column patterns (top N values)
        for col in columns:
            value_counts = df[col].value_counts()
            top_values = value_counts.head(10).index  # Top 10 values
            
            for value in top_values:
                pattern_key = f"{col}={value}"
                matching_indices = df[df[col] == value].index.tolist()
                if len(matching_indices) <= self.max_index_size:
                    self.pattern_indices[pattern_key] = matching_indices
    
    def _build_compound_indices(self, df: pd.DataFrame) -> None:
        """Build compound indices for multi-column combinations."""
        from itertools import combinations
        
        columns = list(df.columns)
        
        # Build indices for column combinations
        for size in range(2, min(self.max_compound_size + 1, len(columns) + 1)):
            for col_combo in combinations(columns, size):
                index_key = col_combo
                self.compound_indices[index_key] = defaultdict(list)
                
                for idx, row in df.iterrows():
                    values = tuple(row[col] for col in col_combo)
                    # Limit compound index size
                    if len(self.compound_indices[index_key]) < self.max_index_size:
                        self.compound_indices[index_key][values].append(idx)
    
    def _build_frequency_indices(self, df: pd.DataFrame) -> None:
        """Build frequency indices for scoring."""
        for col in df.columns:
            self.frequency_indices[col] = df[col].value_counts().to_dict()
    
    def _compute_index_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute scores using index lookups.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of scores for each row
        """
        scores = np.zeros(len(df))
        
        if self.scoring_method == 'simple':
            scores = self._compute_simple_index_scores(df)
        elif self.scoring_method == 'weighted':
            scores = self._compute_weighted_index_scores(df)
        elif self.scoring_method == 'compound':
            scores = self._compute_compound_index_scores(df)
        else:
            # Default to weighted
            scores = self._compute_weighted_index_scores(df)
        
        return scores
    
    def _compute_simple_index_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute simple scores using value frequency lookups."""
        scores = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            for col in df.columns:
                value = row[col]
                if col in self.frequency_indices:
                    score += self.frequency_indices[col].get(value, 0)
            scores[i] = score
        
        return scores
    
    def _compute_weighted_index_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute weighted scores considering column importance."""
        scores = np.zeros(len(df))
        
        # Compute column weights based on entropy
        column_weights = self._compute_column_weights()
        
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            for col in df.columns:
                value = row[col]
                if col in self.frequency_indices:
                    frequency = self.frequency_indices[col].get(value, 0)
                    weight = column_weights.get(col, 1.0)
                    score += frequency * weight
            scores[i] = score
        
        return scores
    
    def _compute_compound_index_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute scores using compound indices."""
        scores = np.zeros(len(df))
        
        # Start with simple scores
        scores = self._compute_weighted_index_scores(df)
        
        # Add compound pattern scores
        for i, (_, row) in enumerate(df.iterrows()):
            compound_score = 0.0
            pattern_count = 0
            
            # Check compound indices
            for index_key, compound_index in self.compound_indices.items():
                if len(index_key) <= len(df.columns):
                    values = tuple(row[col] for col in index_key)
                    if values in compound_index:
                        # Score based on frequency of this compound pattern
                        pattern_frequency = len(compound_index[values])
                        # Weight by pattern complexity (more columns = higher weight)
                        complexity_weight = len(index_key)
                        compound_score += pattern_frequency * complexity_weight
                        pattern_count += 1
            
            # Average compound score
            if pattern_count > 0:
                compound_score = compound_score / pattern_count
            
            # Combine simple and compound scores
            scores[i] = 0.7 * scores[i] + 0.3 * compound_score
        
        return scores
    
    def _compute_column_weights(self) -> Dict[str, float]:
        """Compute weights for columns based on their information content."""
        column_weights = {}
        
        for col, freq_dict in self.frequency_indices.items():
            if freq_dict:
                # Compute normalized entropy
                total_count = sum(freq_dict.values())
                entropy = 0.0
                
                for count in freq_dict.values():
                    if count > 0:
                        prob = count / total_count
                        entropy -= prob * np.log2(prob)
                
                # Higher entropy = lower weight (more uniform = less informative)
                max_entropy = np.log2(len(freq_dict)) if len(freq_dict) > 1 else 1
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Invert entropy to get weight (lower entropy = higher weight)
                column_weights[col] = 1.0 - normalized_entropy + 0.1  # Add small base weight
            else:
                column_weights[col] = 0.1
        
        # Normalize weights
        total_weight = sum(column_weights.values())
        if total_weight > 0:
            column_weights = {k: v/total_weight for k, v in column_weights.items()}
        
        return column_weights
    
    def _range_lookup(self, col: str, value: float, tolerance: float = 0.1) -> List[int]:
        """
        Perform range lookup using range indices.
        
        Args:
            col: Column name
            value: Target value
            tolerance: Range tolerance (as fraction)
            
        Returns:
            List of matching row indices
        """
        if col not in self.range_indices:
            return []
        
        range_data = self.range_indices[col]
        if not range_data:
            return []
        
        # Binary search for efficiency
        target_min = value * (1 - tolerance)
        target_max = value * (1 + tolerance)
        
        matching_indices = []
        for val, idx in range_data:
            if target_min <= val <= target_max:
                matching_indices.append(idx)
            elif val > target_max:
                break  # Since data is sorted, no more matches
        
        return matching_indices
    
    def _pattern_lookup(self, pattern: str) -> List[int]:
        """
        Perform pattern lookup using pattern indices.
        
        Args:
            pattern: Pattern string
            
        Returns:
            List of matching row indices
        """
        return self.pattern_indices.get(pattern, [])
    
    def _clear_indices(self) -> None:
        """Clear all indices."""
        self.value_indices.clear()
        self.range_indices.clear()
        self.pattern_indices.clear()
        self.compound_indices.clear()
        self.frequency_indices.clear()
        self._score_cache.clear()
        self._indices_built = False
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about built indices."""
        stats = {
            'indices_built': self._indices_built,
            'value_indices_count': len(self.value_indices),
            'range_indices_count': len(self.range_indices),
            'pattern_indices_count': len(self.pattern_indices),
            'compound_indices_count': len(self.compound_indices),
            'frequency_indices_count': len(self.frequency_indices),
            'cache_size': len(self._score_cache)
        }
        
        # Add size statistics
        total_value_entries = sum(len(idx) for idx in self.value_indices.values())
        total_pattern_entries = sum(len(indices) for indices in self.pattern_indices.values())
        total_compound_entries = sum(len(idx) for idx in self.compound_indices.values())
        
        stats.update({
            'total_value_entries': total_value_entries,
            'total_pattern_entries': total_pattern_entries,
            'total_compound_entries': total_compound_entries
        })
        
        return stats
    
    def rebuild_indices(self, df: pd.DataFrame) -> None:
        """Force rebuild of all indices."""
        self._clear_indices()
        self._build_indices(df)
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "IndexRecommender"
