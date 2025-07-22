"""
Cached frequency-based recommender with pre-computed frequency tables and fast lookups.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict, Counter
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class FrequencyRecommender(BaseRecommender):
    """
    High-performance recommender using cached frequency computations and fast lookups.
    Pre-computes frequency tables and uses efficient data structures for recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the frequency recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Frequency caches
        self._column_frequencies = {}
        self._tuple_frequencies = {}
        self._pattern_frequencies = {}
        self._cached_scores = {}
        self._last_processed_hash = None
        
        # Configuration
        freq_config = self.config.get('frequency', {})
        self.scoring_method = freq_config.get('method', 'weighted')  # 'simple', 'weighted', 'pattern'
        self.pattern_length = freq_config.get('pattern_length', 2)  # For pattern-based scoring
        self.min_frequency = freq_config.get('min_frequency', 1)
        self.cache_enabled = freq_config.get('cache_enabled', True)
        self.max_unique_values = freq_config.get('max_unique_values', 50)
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using cached frequency computations.
        
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
        if self.cache_enabled and df_hash == self._last_processed_hash and df_hash in self._cached_scores:
            logger.debug("Using cached frequency scores")
            frequency_scores = self._cached_scores[df_hash]
        else:
            # Build frequency tables
            self._build_frequency_tables(current_results)
            
            # Compute frequency scores
            frequency_scores = self._compute_frequency_scores(current_results)
            
            # Cache results
            if self.cache_enabled:
                self._cached_scores[df_hash] = frequency_scores
                self._last_processed_hash = df_hash
        
        # Rank by frequency scores
        current_results_copy = current_results.copy()
        current_results_copy['frequency_score'] = frequency_scores
        
        # Sort by frequency score (descending)
        sorted_df = current_results_copy.sort_values('frequency_score', ascending=False)
        
        # Apply output limiting
        result_df = self._limit_output(sorted_df.drop(columns=['frequency_score']))
        
        return result_df
    
    def _build_frequency_tables(self, df: pd.DataFrame) -> None:
        """
        Build frequency tables for fast lookups.
        
        Args:
            df: Input DataFrame
        """
        # Clear existing caches
        self._column_frequencies.clear()
        self._tuple_frequencies.clear()
        self._pattern_frequencies.clear()
        
        # Limit unique values to prevent memory explosion
        df_limited = self._limit_unique_values(df)
        
        # Build column-wise frequency tables
        for col in df_limited.columns:
            value_counts = df_limited[col].value_counts()
            self._column_frequencies[col] = value_counts.to_dict()
        
        # Build tuple frequency table
        tuple_strings = df_limited.apply(lambda row: '|'.join(row.astype(str)), axis=1)
        tuple_counts = tuple_strings.value_counts()
        self._tuple_frequencies = tuple_counts.to_dict()
        
        # Build pattern frequency table if using pattern-based scoring
        if self.scoring_method == 'pattern':
            self._build_pattern_frequencies(df_limited)
    
    def _limit_unique_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limit unique values per column to prevent memory explosion.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with limited unique values
        """
        df_limited = df.copy()
        
        for col in df_limited.columns:
            unique_count = df_limited[col].nunique()
            if unique_count > self.max_unique_values:
                # Keep only most frequent values
                value_counts = df_limited[col].value_counts()
                top_values = value_counts.head(self.max_unique_values - 1).index
                df_limited[col] = df_limited[col].apply(
                    lambda x: x if x in top_values else 'OTHER'
                )
        
        return df_limited
    
    def _build_pattern_frequencies(self, df: pd.DataFrame) -> None:
        """
        Build frequency table for column patterns.
        
        Args:
            df: Input DataFrame
        """
        columns = list(df.columns)
        
        # Generate column combinations
        from itertools import combinations
        
        for length in range(1, min(self.pattern_length + 1, len(columns) + 1)):
            for col_combo in combinations(columns, length):
                pattern_key = '|'.join(col_combo)
                
                # Count patterns for this combination
                pattern_counts = defaultdict(int)
                for _, row in df.iterrows():
                    pattern_value = '|'.join(str(row[col]) for col in col_combo)
                    pattern_counts[pattern_value] += 1
                
                self._pattern_frequencies[pattern_key] = dict(pattern_counts)
    
    def _compute_frequency_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute frequency scores for each row using fast lookups.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of frequency scores
        """
        scores = np.zeros(len(df))
        df_limited = self._limit_unique_values(df)
        
        if self.scoring_method == 'simple':
            scores = self._compute_simple_frequency_scores(df_limited)
        elif self.scoring_method == 'weighted':
            scores = self._compute_weighted_frequency_scores(df_limited)
        elif self.scoring_method == 'pattern':
            scores = self._compute_pattern_frequency_scores(df_limited)
        else:
            # Default to weighted
            scores = self._compute_weighted_frequency_scores(df_limited)
        
        # Normalize scores
        if len(scores) > 0:
            max_score = np.max(scores)
            if max_score > 0:
                scores = scores / max_score
        
        # Add small random noise to break ties
        scores += np.random.random(len(scores)) * 1e-10
        
        return scores
    
    def _compute_simple_frequency_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute simple frequency scores based on individual value frequencies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of frequency scores
        """
        scores = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            for col in df.columns:
                value = str(row[col])
                if col in self._column_frequencies:
                    score += self._column_frequencies[col].get(value, 0)
            scores[i] = score
        
        return scores
    
    def _compute_weighted_frequency_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute weighted frequency scores considering column importance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of frequency scores
        """
        scores = np.zeros(len(df))
        
        # Compute column weights based on entropy (less entropy = more weight)
        column_weights = {}
        for col in df.columns:
            if col in self._column_frequencies:
                frequencies = list(self._column_frequencies[col].values())
                total = sum(frequencies)
                if total > 0:
                    # Compute normalized entropy
                    entropy = -sum((f/total) * np.log2(f/total + 1e-10) for f in frequencies if f > 0)
                    # Invert entropy (lower entropy = higher weight)
                    max_entropy = np.log2(len(frequencies)) if len(frequencies) > 1 else 1
                    column_weights[col] = max_entropy - entropy + 1
                else:
                    column_weights[col] = 1.0
            else:
                column_weights[col] = 1.0
        
        # Normalize weights
        total_weight = sum(column_weights.values())
        if total_weight > 0:
            column_weights = {k: v/total_weight for k, v in column_weights.items()}
        
        # Compute weighted scores
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            for col in df.columns:
                value = str(row[col])
                if col in self._column_frequencies:
                    freq = self._column_frequencies[col].get(value, 0)
                    weight = column_weights.get(col, 1.0)
                    score += freq * weight
            scores[i] = score
        
        return scores
    
    def _compute_pattern_frequency_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute pattern-based frequency scores using column combinations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of frequency scores
        """
        scores = np.zeros(len(df))
        columns = list(df.columns)
        
        from itertools import combinations
        
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            pattern_count = 0
            
            # Check patterns of different lengths
            for length in range(1, min(self.pattern_length + 1, len(columns) + 1)):
                for col_combo in combinations(columns, length):
                    pattern_key = '|'.join(col_combo)
                    pattern_value = '|'.join(str(row[col]) for col in col_combo)
                    
                    if pattern_key in self._pattern_frequencies:
                        freq = self._pattern_frequencies[pattern_key].get(pattern_value, 0)
                        # Weight by pattern length (longer patterns more important)
                        weight = length
                        score += freq * weight
                        pattern_count += 1
            
            # Average score across patterns
            if pattern_count > 0:
                score = score / pattern_count
            
            scores[i] = score
        
        return scores
    
    def clear_cache(self) -> None:
        """Clear all cached frequency tables."""
        self._column_frequencies.clear()
        self._tuple_frequencies.clear()
        self._pattern_frequencies.clear()
        self._cached_scores.clear()
        self._last_processed_hash = None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        return {
            'column_frequencies_size': len(self._column_frequencies),
            'tuple_frequencies_size': len(self._tuple_frequencies),
            'pattern_frequencies_size': len(self._pattern_frequencies),
            'cached_scores_size': len(self._cached_scores),
            'cache_enabled': self.cache_enabled
        }
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "FrequencyRecommender"
