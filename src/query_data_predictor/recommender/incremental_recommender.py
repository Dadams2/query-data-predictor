"""
Incremental learning recommender that learns patterns incrementally and updates model efficiently.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class IncrementalRecommender(BaseRecommender):
    """
    High-performance recommender using incremental learning.
    Updates recommendation model efficiently with new data without full recomputation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the incremental recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Configuration
        inc_config = self.config.get('incremental', {})
        self.learning_rate = inc_config.get('learning_rate', 0.1)
        self.decay_factor = inc_config.get('decay_factor', 0.95)  # For forgetting old patterns
        self.window_size = inc_config.get('window_size', 1000)  # Size of sliding window
        self.min_pattern_frequency = inc_config.get('min_pattern_frequency', 2)
        self.update_threshold = inc_config.get('update_threshold', 0.1)  # When to trigger updates
        self.enable_online_learning = inc_config.get('enable_online_learning', True)
        
        # Model state
        self.pattern_weights = defaultdict(float)  # pattern -> weight
        self.pattern_frequencies = defaultdict(int)  # pattern -> frequency
        self.feature_weights = defaultdict(float)  # feature -> weight
        self.data_window = deque(maxlen=self.window_size)  # Sliding window of recent data
        self.total_updates = 0
        self.last_model_hash = None
        
        # Performance tracking
        self.prediction_history = deque(maxlen=100)  # Track recent prediction quality
        self.adaptation_rate = 0.1  # How fast to adapt to new patterns
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using incremental learning model.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # Update model with new data
        if self.enable_online_learning:
            self._update_model_incremental(current_results)
        
        # Compute scores using current model
        scores = self._compute_incremental_scores(current_results)
        
        # Update data window
        self._update_data_window(current_results)
        
        # Rank by scores
        current_results_copy = current_results.copy()
        current_results_copy['incremental_score'] = scores
        
        # Sort by score (descending)
        sorted_df = current_results_copy.sort_values('incremental_score', ascending=False)
        
        # Apply output limiting
        result_df = self._limit_output(sorted_df.drop(columns=['incremental_score']))
        
        return result_df
    
    def _update_model_incremental(self, df: pd.DataFrame) -> None:
        """
        Update the model incrementally with new data.
        
        Args:
            df: New data to learn from
        """
        # Extract patterns from new data
        new_patterns = self._extract_patterns(df)
        
        # Update pattern frequencies and weights
        for pattern, frequency in new_patterns.items():
            # Incremental frequency update
            old_freq = self.pattern_frequencies[pattern]
            new_freq = old_freq + frequency
            self.pattern_frequencies[pattern] = new_freq
            
            # Update pattern weight using exponential moving average
            old_weight = self.pattern_weights[pattern]
            importance_score = self._compute_pattern_importance(pattern, frequency, len(df))
            new_weight = old_weight * (1 - self.learning_rate) + importance_score * self.learning_rate
            self.pattern_weights[pattern] = new_weight
        
        # Update feature weights
        self._update_feature_weights(df)
        
        # Apply decay to old patterns to encourage forgetting
        self._apply_decay()
        
        # Prune low-frequency patterns
        self._prune_patterns()
        
        self.total_updates += 1
        
        # Adaptive learning rate
        self._adapt_learning_rate()
    
    def _extract_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Extract patterns from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of pattern -> frequency
        """
        patterns = defaultdict(int)
        
        # Single column patterns
        for col in df.columns:
            for value in df[col]:
                pattern = f"{col}={value}"
                patterns[pattern] += 1
        
        # Two-column patterns (most common combinations)
        columns = list(df.columns)
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                for _, row in df.iterrows():
                    pattern = f"{col1}={row[col1]}&{col2}={row[col2]}"
                    patterns[pattern] += 1
        
        # Row-level patterns (complete rows)
        for _, row in df.iterrows():
            row_pattern = "&".join(f"{col}={row[col]}" for col in df.columns)
            patterns[row_pattern] += 1
        
        return dict(patterns)
    
    def _compute_pattern_importance(self, pattern: str, frequency: int, data_size: int) -> float:
        """
        Compute importance score for a pattern.
        
        Args:
            pattern: Pattern string
            frequency: Frequency of pattern in current data
            data_size: Size of current data
            
        Returns:
            Importance score
        """
        # Base importance from frequency
        frequency_score = frequency / data_size
        
        # Pattern complexity bonus (more complex patterns are more informative)
        complexity = pattern.count('&') + 1
        complexity_bonus = np.log(complexity + 1)
        
        # Recency bonus (newer patterns get slight boost)
        recency_bonus = 1.0
        
        # Historical frequency consideration
        historical_freq = self.pattern_frequencies.get(pattern, 0)
        if historical_freq > 0:
            # Balance new evidence with historical evidence
            historical_weight = historical_freq / (historical_freq + frequency)
            frequency_score = (1 - historical_weight) * frequency_score + historical_weight * (historical_freq / self.window_size)
        
        return frequency_score * complexity_bonus * recency_bonus
    
    def _update_feature_weights(self, df: pd.DataFrame) -> None:
        """
        Update feature weights incrementally.
        
        Args:
            df: Input DataFrame
        """
        # Compute feature importance based on variance and frequency
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric features, use variance
                feature_variance = df[col].var()
                importance = np.log(feature_variance + 1)
            else:
                # For categorical features, use entropy
                value_counts = df[col].value_counts()
                total = len(df)
                entropy = -sum((count/total) * np.log2(count/total) for count in value_counts.values() if count > 0)
                importance = entropy
            
            # Update weight using exponential moving average
            old_weight = self.feature_weights[col]
            new_weight = old_weight * (1 - self.learning_rate) + importance * self.learning_rate
            self.feature_weights[col] = new_weight
    
    def _apply_decay(self) -> None:
        """Apply decay factor to all pattern weights to encourage forgetting."""
        for pattern in self.pattern_weights:
            self.pattern_weights[pattern] *= self.decay_factor
    
    def _prune_patterns(self) -> None:
        """Remove patterns with very low frequency or weight."""
        # Remove patterns with frequency below threshold
        patterns_to_remove = []
        for pattern, freq in self.pattern_frequencies.items():
            if freq < self.min_pattern_frequency or self.pattern_weights[pattern] < 0.001:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            del self.pattern_frequencies[pattern]
            del self.pattern_weights[pattern]
    
    def _adapt_learning_rate(self) -> None:
        """Adapt learning rate based on prediction quality."""
        if len(self.prediction_history) > 10:
            recent_quality = np.mean(list(self.prediction_history)[-10:])
            
            # If recent predictions are poor, increase learning rate
            # If recent predictions are good, decrease learning rate
            if recent_quality < 0.5:
                self.learning_rate = min(0.3, self.learning_rate * 1.1)
            else:
                self.learning_rate = max(0.01, self.learning_rate * 0.95)
    
    def _compute_incremental_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute scores using the incremental model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of scores for each row
        """
        scores = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            
            # Score based on matching patterns
            row_patterns = self._get_row_patterns(row, df.columns)
            
            for pattern in row_patterns:
                if pattern in self.pattern_weights:
                    score += self.pattern_weights[pattern]
            
            # Score based on feature weights
            feature_score = 0.0
            for col in df.columns:
                feature_weight = self.feature_weights.get(col, 0.0)
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric features, use normalized value
                    col_values = df[col].values
                    if len(col_values) > 1 and not np.isnan(row[col]):
                        col_mean = np.mean(col_values)
                        col_std = np.std(col_values)
                        normalized_value = (row[col] - col_mean) / (col_std + 1e-10)
                        feature_score += abs(normalized_value) * feature_weight
                else:
                    # For categorical features, use feature weight directly
                    feature_score += feature_weight
            
            # Combine pattern and feature scores
            total_score = 0.7 * score + 0.3 * feature_score
            scores[i] = total_score
        
        # Normalize scores
        if len(scores) > 0:
            max_score = np.max(scores)
            if max_score > 0:
                scores = scores / max_score
        
        return scores
    
    def _get_row_patterns(self, row: pd.Series, columns: List[str]) -> List[str]:
        """
        Get all patterns that match a given row.
        
        Args:
            row: Row data
            columns: Column names
            
        Returns:
            List of matching patterns
        """
        patterns = []
        
        # Single column patterns
        for col in columns:
            pattern = f"{col}={row[col]}"
            patterns.append(pattern)
        
        # Two-column patterns
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                pattern = f"{col1}={row[col1]}&{col2}={row[col2]}"
                patterns.append(pattern)
        
        # Full row pattern
        row_pattern = "&".join(f"{col}={row[col]}" for col in columns)
        patterns.append(row_pattern)
        
        return patterns
    
    def _update_data_window(self, df: pd.DataFrame) -> None:
        """
        Update the sliding window of recent data.
        
        Args:
            df: New data to add to window
        """
        # Add new data to window
        for _, row in df.iterrows():
            self.data_window.append(row.to_dict())
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for inspection."""
        return {
            'total_patterns': len(self.pattern_weights),
            'total_features': len(self.feature_weights),
            'total_updates': self.total_updates,
            'current_learning_rate': self.learning_rate,
            'window_size': len(self.data_window),
            'top_patterns': dict(sorted(self.pattern_weights.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]),
            'top_features': dict(sorted(self.feature_weights.items(), 
                                      key=lambda x: x[1], reverse=True)[:10])
        }
    
    def reset_model(self) -> None:
        """Reset the incremental model to initial state."""
        self.pattern_weights.clear()
        self.pattern_frequencies.clear()
        self.feature_weights.clear()
        self.data_window.clear()
        self.prediction_history.clear()
        self.total_updates = 0
        self.learning_rate = self.config.get('incremental', {}).get('learning_rate', 0.1)
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate."""
        self.learning_rate = max(0.001, min(1.0, learning_rate))
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about learned patterns."""
        if not self.pattern_weights:
            return {"message": "No patterns learned yet"}
        
        # Most important patterns
        top_patterns = sorted(self.pattern_weights.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Pattern types analysis
        single_col_patterns = [p for p in self.pattern_weights.keys() if '&' not in p]
        multi_col_patterns = [p for p in self.pattern_weights.keys() if '&' in p]
        
        return {
            'total_patterns': len(self.pattern_weights),
            'single_column_patterns': len(single_col_patterns),
            'multi_column_patterns': len(multi_col_patterns),
            'top_patterns': top_patterns[:10],
            'average_pattern_weight': np.mean(list(self.pattern_weights.values())),
            'pattern_weight_std': np.std(list(self.pattern_weights.values()))
        }
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "IncrementalRecommender"
