"""
Fast similarity-based recommender using vectorized operations and optimized similarity metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class SimilarityRecommender(BaseRecommender):
    """
    High-performance recommender using optimized similarity computations.
    Uses vectorized operations and efficient similarity metrics for fast recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the similarity recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = None
        self._similarity_cache = {}
        self._encoded_cache = {}
        self._last_processed_hash = None
        
        # Get configuration
        sim_config = self.config.get('similarity', {})
        self.similarity_metric = sim_config.get('metric', 'cosine')  # 'cosine', 'euclidean', 'manhattan'
        self.use_pca = sim_config.get('use_pca', True)
        self.pca_components = sim_config.get('pca_components', 0.95)  # Keep 95% variance
        self.max_features = sim_config.get('max_features', 100)  # Limit feature explosion
        self.cache_enabled = sim_config.get('cache_enabled', True)
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using fast similarity computations.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty or len(current_results) < 2:
            return pd.DataFrame()
        
        # Generate hash for caching
        df_hash = pd.util.hash_pandas_object(current_results).sum()
        
        # Check cache
        if self.cache_enabled and df_hash == self._last_processed_hash and df_hash in self._similarity_cache:
            logger.debug("Using cached similarity results")
            similarity_scores = self._similarity_cache[df_hash]
            encoded_df = self._encoded_cache[df_hash]
        else:
            # Encode and vectorize data efficiently
            encoded_df = self._encode_dataframe_fast(current_results)
            
            if encoded_df.empty:
                return pd.DataFrame()
            
            # Compute similarity scores
            similarity_scores = self._compute_similarity_scores_fast(encoded_df)
            
            # Cache results
            if self.cache_enabled:
                self._similarity_cache[df_hash] = similarity_scores
                self._encoded_cache[df_hash] = encoded_df
                self._last_processed_hash = df_hash
        
        # Rank by similarity scores
        current_results_copy = current_results.copy()
        current_results_copy['similarity_score'] = similarity_scores
        
        # Sort by similarity score (descending)
        sorted_df = current_results_copy.sort_values('similarity_score', ascending=False)
        
        # Apply output limiting
        result_df = self._limit_output(sorted_df.drop(columns=['similarity_score']))
        
        return result_df
    
    def _encode_dataframe_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Efficiently encode DataFrame for similarity computation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Encoded DataFrame suitable for similarity computation
        """
        if df.empty:
            return pd.DataFrame()
        
        # Copy for encoding
        encoded_df = df.copy()
        
        # Limit categorical features to prevent feature explosion
        for col in encoded_df.columns:
            if encoded_df[col].dtype == 'object':
                unique_count = encoded_df[col].nunique()
                if unique_count > self.max_features:
                    # Keep only top frequent values
                    value_counts = encoded_df[col].value_counts()
                    top_values = value_counts.head(self.max_features - 1).index
                    encoded_df[col] = encoded_df[col].apply(
                        lambda x: x if x in top_values else 'OTHER'
                    )
        
        # Encode categorical variables efficiently
        categorical_cols = encoded_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded_df[col] = self.label_encoders[col].fit_transform(encoded_df[col].astype(str))
            else:
                # Handle new categories
                try:
                    encoded_df[col] = self.label_encoders[col].transform(encoded_df[col].astype(str))
                except ValueError:
                    # New categories found, refit
                    encoded_df[col] = self.label_encoders[col].fit_transform(encoded_df[col].astype(str))
        
        # Handle missing values efficiently
        encoded_df = encoded_df.fillna(encoded_df.mean(numeric_only=True))
        encoded_df = encoded_df.fillna(0)  # For any remaining NaN
        
        # Ensure all columns are numeric
        for col in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(encoded_df[col]):
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce').fillna(0)
        
        return encoded_df
    
    def _compute_similarity_scores_fast(self, encoded_df: pd.DataFrame) -> np.ndarray:
        """
        Compute similarity scores efficiently using vectorized operations.
        
        Args:
            encoded_df: Encoded DataFrame
            
        Returns:
            Array of similarity scores for each row
        """
        if encoded_df.empty or len(encoded_df) < 2:
            return np.ones(len(encoded_df))
        
        # Convert to numpy array for speed
        data_matrix = encoded_df.values.astype(np.float32)  # Use float32 for memory efficiency
        
        # Apply dimensionality reduction if enabled
        if self.use_pca and data_matrix.shape[1] > 3:
            if self.pca is None:
                n_components = self.pca_components
                if isinstance(n_components, float):
                    # Keep percentage of variance
                    self.pca = PCA(n_components=n_components, random_state=42)
                else:
                    # Keep fixed number of components
                    n_components = min(n_components, data_matrix.shape[1])
                    self.pca = PCA(n_components=n_components, random_state=42)
                
                data_matrix = self.pca.fit_transform(data_matrix)
            else:
                data_matrix = self.pca.transform(data_matrix)
        
        # Standardize data for better similarity computation
        if data_matrix.shape[0] > 1:
            data_matrix = self.scaler.fit_transform(data_matrix)
        
        # Compute similarity matrix efficiently
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(data_matrix)
        elif self.similarity_metric == 'euclidean':
            # Convert distance to similarity (larger values = more similar)
            distance_matrix = euclidean_distances(data_matrix)
            # Normalize distances and convert to similarity
            max_distance = np.max(distance_matrix)
            if max_distance > 0:
                similarity_matrix = 1 - (distance_matrix / max_distance)
            else:
                similarity_matrix = np.ones_like(distance_matrix)
        elif self.similarity_metric == 'manhattan':
            # Manhattan distance using efficient broadcasting
            distance_matrix = np.sum(np.abs(data_matrix[:, np.newaxis] - data_matrix), axis=2)
            max_distance = np.max(distance_matrix)
            if max_distance > 0:
                similarity_matrix = 1 - (distance_matrix / max_distance)
            else:
                similarity_matrix = np.ones_like(distance_matrix)
        else:
            # Default to cosine
            similarity_matrix = cosine_similarity(data_matrix)
        
        # Compute aggregate similarity scores for each row
        # Use mean similarity to all other rows (excluding self)
        np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
        
        # Compute row-wise mean similarity
        similarity_scores = np.mean(similarity_matrix, axis=1)
        
        # Add small random noise to break ties
        similarity_scores += np.random.random(len(similarity_scores)) * 1e-10
        
        return similarity_scores
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "SimilarityRecommender"
