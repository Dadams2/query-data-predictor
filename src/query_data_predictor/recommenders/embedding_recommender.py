"""
Neural embedding recommender using pre-trained embeddings or fast neural networks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class EmbeddingRecommender(BaseRecommender):
    """
    High-performance recommender using neural embeddings for fast inference.
    Embeds data into vector space and uses fast nearest neighbor search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Configuration
        embed_config = self.config.get('embedding', {})
        self.embedding_dim = embed_config.get('embedding_dim', 64)
        self.use_pretrained = embed_config.get('use_pretrained', False)
        self.embedding_method = embed_config.get('method', 'pca')  # 'pca', 'random', 'learned'
        self.nn_algorithm = embed_config.get('nn_algorithm', 'auto')  # 'auto', 'ball_tree', 'kd_tree', 'brute'
        self.n_neighbors = embed_config.get('n_neighbors', 10)
        self.distance_metric = embed_config.get('distance_metric', 'cosine')
        self.enable_caching = embed_config.get('enable_caching', True)
        
        # Model components
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.embedding_model = None
        self.nn_model = None
        self.pca_model = None
        
        # Caching
        self._embedding_cache = {}
        self._similarity_cache = {}
        self._last_processed_hash = None
        self._is_fitted = False
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using neural embeddings and fast nearest neighbor search.
        
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
        if (self.enable_caching and df_hash == self._last_processed_hash and 
            df_hash in self._similarity_cache):
            logger.debug("Using cached embedding results")
            similarity_scores = self._similarity_cache[df_hash]
        else:
            # Encode and embed data
            encoded_df = self._encode_dataframe(current_results)
            
            if encoded_df.empty:
                return pd.DataFrame()
            
            # Create or update embeddings
            embeddings = self._create_embeddings(encoded_df)
            
            # Compute similarity scores
            similarity_scores = self._compute_embedding_similarity(embeddings)
            
            # Cache results
            if self.enable_caching:
                self._similarity_cache[df_hash] = similarity_scores
                self._last_processed_hash = df_hash
        
        # Rank by similarity scores
        current_results_copy = current_results.copy()
        current_results_copy['embedding_score'] = similarity_scores
        
        # Sort by score (descending)
        sorted_df = current_results_copy.sort_values('embedding_score', ascending=False)
        
        # Apply output limiting
        result_df = self._limit_output(sorted_df.drop(columns=['embedding_score']))
        
        return result_df
    
    def _encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode DataFrame for embedding computation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Encoded DataFrame
        """
        if df.empty:
            return pd.DataFrame()
        
        encoded_df = df.copy()
        
        # Encode categorical variables
        for col in encoded_df.columns:
            if encoded_df[col].dtype == 'object':
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
        
        # Handle missing values
        encoded_df = encoded_df.fillna(0)
        
        # Ensure all columns are numeric
        for col in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(encoded_df[col]):
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce').fillna(0)
        
        return encoded_df
    
    def _create_embeddings(self, encoded_df: pd.DataFrame) -> np.ndarray:
        """
        Create embeddings for the encoded data.
        
        Args:
            encoded_df: Encoded DataFrame
            
        Returns:
            Array of embeddings
        """
        if encoded_df.empty:
            return np.array([])
        
        # Convert to numpy array
        data_matrix = encoded_df.values.astype(np.float32)
        
        # Standardize data
        if not self._is_fitted:
            data_matrix = self.scaler.fit_transform(data_matrix)
            self._is_fitted = True
        else:
            data_matrix = self.scaler.transform(data_matrix)
        
        # Create embeddings based on method
        if self.embedding_method == 'pca':
            embeddings = self._create_pca_embeddings(data_matrix)
        elif self.embedding_method == 'random':
            embeddings = self._create_random_embeddings(data_matrix)
        elif self.embedding_method == 'learned':
            embeddings = self._create_learned_embeddings(data_matrix)
        else:
            # Default to PCA
            embeddings = self._create_pca_embeddings(data_matrix)
        
        return embeddings
    
    def _create_pca_embeddings(self, data_matrix: np.ndarray) -> np.ndarray:
        """Create embeddings using PCA dimensionality reduction."""
        if data_matrix.shape[1] <= self.embedding_dim:
            # If data already has fewer dimensions, pad with zeros
            padded = np.zeros((data_matrix.shape[0], self.embedding_dim))
            padded[:, :data_matrix.shape[1]] = data_matrix
            return padded
        
        if self.pca_model is None:
            self.pca_model = PCA(n_components=self.embedding_dim, random_state=42)
            embeddings = self.pca_model.fit_transform(data_matrix)
        else:
            embeddings = self.pca_model.transform(data_matrix)
        
        return embeddings.astype(np.float32)
    
    def _create_random_embeddings(self, data_matrix: np.ndarray) -> np.ndarray:
        """Create embeddings using random projection."""
        # Simple random projection
        if not hasattr(self, '_random_projection_matrix'):
            np.random.seed(42)
            self._random_projection_matrix = np.random.randn(
                data_matrix.shape[1], self.embedding_dim
            ).astype(np.float32)
            # Normalize columns
            self._random_projection_matrix /= np.linalg.norm(
                self._random_projection_matrix, axis=0, keepdims=True
            )
        
        embeddings = data_matrix @ self._random_projection_matrix
        return embeddings.astype(np.float32)
    
    def _create_learned_embeddings(self, data_matrix: np.ndarray) -> np.ndarray:
        """Create embeddings using simple neural network (autoencoder-style)."""
        # Simple autoencoder approach using matrix factorization
        if not hasattr(self, '_embedding_weights'):
            # Initialize embedding weights
            np.random.seed(42)
            input_dim = data_matrix.shape[1]
            
            # Simple approach: use SVD for matrix factorization
            try:
                U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
                # Take top embedding_dim components
                k = min(self.embedding_dim, len(s))
                self._embedding_weights = Vt[:k].T
            except np.linalg.LinAlgError:
                # Fall back to random projection
                return self._create_random_embeddings(data_matrix)
        
        # Project data to embedding space
        embeddings = data_matrix @ self._embedding_weights
        
        # Pad or truncate to desired dimension
        if embeddings.shape[1] < self.embedding_dim:
            padded = np.zeros((embeddings.shape[0], self.embedding_dim))
            padded[:, :embeddings.shape[1]] = embeddings
            embeddings = padded
        elif embeddings.shape[1] > self.embedding_dim:
            embeddings = embeddings[:, :self.embedding_dim]
        
        return embeddings.astype(np.float32)
    
    def _compute_embedding_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity scores using embeddings.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(embeddings) == 0:
            return np.array([])
        
        if len(embeddings) == 1:
            return np.array([1.0])
        
        # Use nearest neighbors for efficient similarity computation
        if self.nn_model is None or len(embeddings) != self.nn_model.n_samples_fit_:
            n_neighbors = min(self.n_neighbors, len(embeddings))
            self.nn_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm=self.nn_algorithm,
                metric=self.distance_metric
            )
            self.nn_model.fit(embeddings)
        
        # Compute similarities
        if self.distance_metric in ['cosine', 'euclidean']:
            # For these metrics, closer distance = higher similarity
            distances, indices = self.nn_model.kneighbors(embeddings)
            
            # Convert distances to similarities
            if self.distance_metric == 'cosine':
                # Cosine distance -> cosine similarity
                similarities = 1 - distances
            else:
                # Euclidean distance -> similarity (inverse)
                max_distance = np.max(distances) + 1e-10
                similarities = 1 - (distances / max_distance)
            
            # Compute mean similarity for each point (excluding self)
            similarity_scores = np.mean(similarities[:, 1:], axis=1)  # Skip first (self)
        else:
            # For other metrics, compute pairwise similarities directly
            from sklearn.metrics.pairwise import pairwise_distances
            
            distance_matrix = pairwise_distances(embeddings, metric=self.distance_metric)
            np.fill_diagonal(distance_matrix, np.inf)  # Exclude self
            
            # Convert to similarity (inverse distance)
            max_distance = np.max(distance_matrix[distance_matrix != np.inf])
            similarity_matrix = 1 - (distance_matrix / (max_distance + 1e-10))
            similarity_matrix[distance_matrix == np.inf] = 0
            
            # Compute mean similarity for each point
            similarity_scores = np.mean(similarity_matrix, axis=1)
        
        return similarity_scores
    
    def _get_embedding_for_query(self, query_embedding: np.ndarray, 
                                all_embeddings: np.ndarray) -> np.ndarray:
        """
        Get similarity scores for a query embedding against all embeddings.
        
        Args:
            query_embedding: Single query embedding
            all_embeddings: All embeddings to compare against
            
        Returns:
            Array of similarity scores
        """
        if len(all_embeddings) == 0:
            return np.array([])
        
        # Compute similarities between query and all embeddings
        if self.distance_metric == 'cosine':
            # Cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            all_norms = np.linalg.norm(all_embeddings, axis=1)
            
            if query_norm > 0 and np.all(all_norms > 0):
                similarities = np.dot(all_embeddings, query_embedding) / (all_norms * query_norm)
            else:
                similarities = np.zeros(len(all_embeddings))
        else:
            # Euclidean or other distance metrics
            from sklearn.metrics.pairwise import pairwise_distances
            
            distances = pairwise_distances(
                all_embeddings, 
                query_embedding.reshape(1, -1), 
                metric=self.distance_metric
            ).flatten()
            
            # Convert to similarity
            max_distance = np.max(distances) + 1e-10
            similarities = 1 - (distances / max_distance)
        
        return similarities
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        info = {
            'embedding_dim': self.embedding_dim,
            'embedding_method': self.embedding_method,
            'distance_metric': self.distance_metric,
            'n_neighbors': self.n_neighbors,
            'is_fitted': self._is_fitted,
            'has_pca_model': self.pca_model is not None,
            'has_nn_model': self.nn_model is not None,
            'cache_size': len(self._similarity_cache)
        }
        
        if self.pca_model is not None:
            info['pca_explained_variance_ratio'] = self.pca_model.explained_variance_ratio_.sum()
        
        return info
    
    def clear_cache(self) -> None:
        """Clear embedding and similarity caches."""
        self._embedding_cache.clear()
        self._similarity_cache.clear()
        self._last_processed_hash = None
    
    def reset_model(self) -> None:
        """Reset the embedding model."""
        self.label_encoders.clear()
        self.scaler = StandardScaler()
        self.embedding_model = None
        self.nn_model = None
        self.pca_model = None
        self.clear_cache()
        self._is_fitted = False
        
        # Clear learned weights
        if hasattr(self, '_random_projection_matrix'):
            delattr(self, '_random_projection_matrix')
        if hasattr(self, '_embedding_weights'):
            delattr(self, '_embedding_weights')
    
    def precompute_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Precompute embeddings for a dataset (useful for batch processing).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of embeddings
        """
        encoded_df = self._encode_dataframe(df)
        embeddings = self._create_embeddings(encoded_df)
        return embeddings
    
    def recommend_from_embeddings(self, embeddings: np.ndarray, 
                                 original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recommend tuples using precomputed embeddings.
        
        Args:
            embeddings: Precomputed embeddings
            original_df: Original DataFrame
            
        Returns:
            DataFrame with recommendations
        """
        if len(embeddings) == 0 or original_df.empty:
            return pd.DataFrame()
        
        # Compute similarity scores
        similarity_scores = self._compute_embedding_similarity(embeddings)
        
        # Rank by scores
        df_copy = original_df.copy()
        df_copy['embedding_score'] = similarity_scores
        
        # Sort and limit
        sorted_df = df_copy.sort_values('embedding_score', ascending=False)
        result_df = self._limit_output(sorted_df.drop(columns=['embedding_score']))
        
        return result_df
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "EmbeddingRecommender"
