"""
Sampling-based recommender that uses statistical sampling to reduce computation on large datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class SamplingRecommender(BaseRecommender):
    """
    High-performance recommender using statistical sampling to reduce computation.
    Provides tunable accuracy/speed tradeoff through intelligent sampling strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sampling recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Configuration
        sampling_config = self.config.get('sampling', {})
        self.sampling_method = sampling_config.get('method', 'stratified')  # 'random', 'stratified', 'systematic'
        self.sample_size = sampling_config.get('sample_size', 1000)  # Max sample size
        self.sample_ratio = sampling_config.get('sample_ratio', 0.1)  # Sample 10% of data
        self.min_sample_size = sampling_config.get('min_sample_size', 100)
        self.scoring_method = sampling_config.get('scoring_method', 'variance')  # 'variance', 'entropy', 'frequency'
        self.use_confidence_intervals = sampling_config.get('use_confidence_intervals', True)
        self.confidence_level = sampling_config.get('confidence_level', 0.95)
        
        self._sample_cache = {}
        self._score_cache = {}
        self._last_processed_hash = None
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using sampling-based approximation.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # For small datasets, don't use sampling
        if len(current_results) <= self.min_sample_size:
            return self._compute_full_scores(current_results)
        
        # Generate hash for caching
        df_hash = pd.util.hash_pandas_object(current_results).sum()
        
        # Check cache
        if df_hash == self._last_processed_hash and df_hash in self._score_cache:
            logger.debug("Using cached sampling results")
            scores = self._score_cache[df_hash]
        else:
            # Determine sample size
            effective_sample_size = self._determine_sample_size(len(current_results))
            
            # Create sample
            sample_df, sample_indices = self._create_sample(current_results, effective_sample_size)
            
            # Compute scores on sample
            sample_scores = self._compute_sample_scores(sample_df)
            
            # Extrapolate scores to full dataset
            scores = self._extrapolate_scores(current_results, sample_df, sample_scores, sample_indices)
            
            # Cache results
            self._score_cache[df_hash] = scores
            self._last_processed_hash = df_hash
        
        # Rank by scores
        current_results_copy = current_results.copy()
        current_results_copy['sampling_score'] = scores
        
        # Sort by score (descending)
        sorted_df = current_results_copy.sort_values('sampling_score', ascending=False)
        
        # Apply output limiting
        result_df = self._limit_output(sorted_df.drop(columns=['sampling_score']))
        
        return result_df
    
    def _determine_sample_size(self, total_size: int) -> int:
        """
        Determine optimal sample size based on total dataset size.
        
        Args:
            total_size: Total number of records
            
        Returns:
            Optimal sample size
        """
        # Use smaller of ratio-based or fixed sample size
        ratio_based = max(self.min_sample_size, int(total_size * self.sample_ratio))
        fixed_based = self.sample_size
        
        return min(ratio_based, fixed_based, total_size)
    
    def _create_sample(self, df: pd.DataFrame, sample_size: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create a representative sample from the dataset.
        
        Args:
            df: Input DataFrame
            sample_size: Size of sample to create
            
        Returns:
            Tuple of (sample DataFrame, indices of sampled rows)
        """
        if self.sampling_method == 'random':
            return self._random_sampling(df, sample_size)
        elif self.sampling_method == 'stratified':
            return self._stratified_sampling(df, sample_size)
        elif self.sampling_method == 'systematic':
            return self._systematic_sampling(df, sample_size)
        else:
            # Default to random
            return self._random_sampling(df, sample_size)
    
    def _random_sampling(self, df: pd.DataFrame, sample_size: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Random sampling."""
        indices = np.random.choice(len(df), size=sample_size, replace=False)
        sample_df = df.iloc[indices].copy()
        return sample_df, indices
    
    def _stratified_sampling(self, df: pd.DataFrame, sample_size: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Stratified sampling based on data distribution."""
        # Use first column for stratification (or most diverse column)
        if df.empty:
            return df, np.array([])
        
        # Find column with good distribution
        strata_col = None
        max_unique = 0
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count > max_unique and unique_count < len(df) * 0.8:  # Not too unique, not too uniform
                max_unique = unique_count
                strata_col = col
        
        if strata_col is None:
            # Fall back to random sampling
            return self._random_sampling(df, sample_size)
        
        # Perform stratified sampling
        stratified_indices = []
        strata_groups = df.groupby(strata_col)
        
        for stratum, group in strata_groups:
            stratum_size = len(group)
            # Sample proportionally
            stratum_sample_size = max(1, int(sample_size * stratum_size / len(df)))
            stratum_sample_size = min(stratum_sample_size, stratum_size)
            
            if stratum_sample_size > 0:
                stratum_indices = np.random.choice(
                    group.index, 
                    size=stratum_sample_size, 
                    replace=False
                )
                stratified_indices.extend(stratum_indices)
        
        # Adjust if we have too many samples
        if len(stratified_indices) > sample_size:
            stratified_indices = np.random.choice(
                stratified_indices, 
                size=sample_size, 
                replace=False
            )
        
        indices = np.array(stratified_indices)
        sample_df = df.loc[indices].copy()
        
        return sample_df, indices
    
    def _systematic_sampling(self, df: pd.DataFrame, sample_size: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Systematic sampling with regular intervals."""
        interval = len(df) // sample_size
        start = np.random.randint(0, interval)
        indices = np.arange(start, len(df), interval)[:sample_size]
        sample_df = df.iloc[indices].copy()
        return sample_df, indices
    
    def _compute_sample_scores(self, sample_df: pd.DataFrame) -> np.ndarray:
        """
        Compute scores on the sample dataset.
        
        Args:
            sample_df: Sample DataFrame
            
        Returns:
            Array of scores for sample
        """
        if self.scoring_method == 'variance':
            return self._compute_variance_scores(sample_df)
        elif self.scoring_method == 'entropy':
            return self._compute_entropy_scores(sample_df)
        elif self.scoring_method == 'frequency':
            return self._compute_frequency_scores(sample_df)
        else:
            # Default to variance
            return self._compute_variance_scores(sample_df)
    
    def _compute_variance_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute variance-based scores."""
        scores = np.zeros(len(df))
        
        # Process only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return scores
        
        numeric_data = df[numeric_cols].values
        
        # Compute variance for each row compared to column means
        col_means = np.mean(numeric_data, axis=0)
        col_stds = np.std(numeric_data, axis=0) + 1e-10  # Avoid division by zero
        
        # Normalize data
        normalized_data = (numeric_data - col_means) / col_stds
        
        # Compute row-wise variance
        scores = np.var(normalized_data, axis=1)
        
        return scores
    
    def _compute_entropy_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute entropy-based scores."""
        scores = np.zeros(len(df))
        
        # Discretize data for entropy calculation
        df_discrete = df.copy()
        for col in df_discrete.columns:
            if pd.api.types.is_numeric_dtype(df_discrete[col]):
                # Discretize numeric columns
                try:
                    df_discrete[col] = pd.cut(df_discrete[col], bins=5, labels=False, duplicates='drop')
                except ValueError:
                    # Handle edge cases
                    df_discrete[col] = 0
        
        # Compute entropy for each row
        for i, (_, row) in enumerate(df_discrete.iterrows()):
            row_entropy = 0.0
            for col in df_discrete.columns:
                # Simple approximation: use value frequency as probability
                value = row[col]
                col_values = df_discrete[col].values
                prob = np.sum(col_values == value) / len(col_values)
                if prob > 0:
                    row_entropy -= prob * np.log2(prob)
            scores[i] = row_entropy
        
        return scores
    
    def _compute_frequency_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute frequency-based scores."""
        scores = np.zeros(len(df))
        
        # Compute value frequencies
        for i, (_, row) in enumerate(df.iterrows()):
            score = 0.0
            for col in df.columns:
                value = row[col]
                # Count frequency of this value in the column
                freq = (df[col] == value).sum()
                score += freq
            scores[i] = score
        
        return scores
    
    def _extrapolate_scores(self, full_df: pd.DataFrame, sample_df: pd.DataFrame, 
                          sample_scores: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
        """
        Extrapolate sample scores to full dataset.
        
        Args:
            full_df: Full DataFrame
            sample_df: Sample DataFrame
            sample_scores: Scores computed on sample
            sample_indices: Indices of sampled rows
            
        Returns:
            Array of scores for full dataset
        """
        full_scores = np.zeros(len(full_df))
        
        # Simple approach: use nearest neighbor for non-sampled points
        sampled_mask = np.zeros(len(full_df), dtype=bool)
        sampled_mask[sample_indices] = True
        
        # Set scores for sampled points
        full_scores[sample_indices] = sample_scores
        
        # For non-sampled points, use interpolation
        non_sampled_indices = np.where(~sampled_mask)[0]
        
        if len(non_sampled_indices) > 0 and len(sample_scores) > 0:
            # Use simple average of sample scores for non-sampled points
            # More sophisticated methods could use k-NN or other interpolation
            mean_score = np.mean(sample_scores)
            std_score = np.std(sample_scores)
            
            # Add some randomness based on sample distribution
            for idx in non_sampled_indices:
                # Use position-based interpolation
                position_ratio = idx / len(full_df)
                # Linear interpolation between min and max sample scores
                min_score = np.min(sample_scores)
                max_score = np.max(sample_scores)
                interpolated_score = min_score + position_ratio * (max_score - min_score)
                
                # Add noise
                noise = np.random.normal(0, std_score * 0.1)
                full_scores[idx] = interpolated_score + noise
        
        return full_scores
    
    def _compute_full_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute full scores for small datasets without sampling."""
        scores = self._compute_variance_scores(df)
        
        df_copy = df.copy()
        df_copy['sampling_score'] = scores
        sorted_df = df_copy.sort_values('sampling_score', ascending=False)
        
        return self._limit_output(sorted_df.drop(columns=['sampling_score']))
    
    def get_sample_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling behavior."""
        return {
            'sampling_method': self.sampling_method,
            'sample_size': self.sample_size,
            'sample_ratio': self.sample_ratio,
            'min_sample_size': self.min_sample_size,
            'scoring_method': self.scoring_method,
            'cache_size': len(self._score_cache)
        }
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "SamplingRecommender"
