"""
Tests for the EmbeddingRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.embedding_recommender import EmbeddingRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestEmbeddingRecommender:
    """Test suite for EmbeddingRecommender class."""
    
    def test_init(self, sample_config):
        """Test EmbeddingRecommender initialization."""
        recommender = EmbeddingRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "EmbeddingRecommender"
        assert recommender.embedding_dim == 64  # default
        assert recommender.embedding_method == 'pca'  # default
        assert hasattr(recommender, 'label_encoders')
        assert hasattr(recommender, 'scaler')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom embedding configuration."""
        config = {
            'embedding': {
                'embedding_dim': 32,
                'use_pretrained': True,
                'method': 'learned',
                'nn_algorithm': 'ball_tree',
                'n_neighbors': 5,
                'distance_metric': 'euclidean',
                'enable_caching': False
            },
            'recommendation': {'top_k': 5}
        }
        recommender = EmbeddingRecommender(config)
        assert recommender.embedding_dim == 32
        assert recommender.use_pretrained is True
        assert recommender.embedding_method == 'learned'
        assert recommender.nn_algorithm == 'ball_tree'
        assert recommender.n_neighbors == 5
        assert recommender.distance_metric == 'euclidean'
        assert recommender.enable_caching is False
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = EmbeddingRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = EmbeddingRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = EmbeddingRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_recommend_tuples_mixed_data(self, sample_config, mixed_dataframe):
        """Test recommend_tuples with mixed data types."""
        recommender = EmbeddingRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should handle mixed data types
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(mixed_dataframe)
        assert list(result.columns) == list(mixed_dataframe.columns)
    
    def test_encode_dataframe(self, sample_config, mixed_dataframe):
        """Test DataFrame encoding for embeddings."""
        recommender = EmbeddingRecommender(sample_config)
        encoded_df = recommender._encode_dataframe(mixed_dataframe)
        
        # Should return encoded DataFrame
        assert isinstance(encoded_df, pd.DataFrame)
        assert len(encoded_df) == len(mixed_dataframe)
        # All columns should be numeric after encoding
        for dtype in encoded_df.dtypes:
            assert np.issubdtype(dtype, np.number)
    
    def test_encode_dataframe_empty(self, sample_config, empty_dataframe):
        """Test encoding with empty DataFrame."""
        recommender = EmbeddingRecommender(sample_config)
        encoded_df = recommender._encode_dataframe(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(encoded_df, pd.DataFrame)
        assert encoded_df.empty
    
    def test_create_embeddings_pca(self, simple_dataframe):
        """Test PCA embedding creation."""
        config = {
            'embedding': {'method': 'pca', 'embedding_dim': 2},
            'recommendation': {'top_k': 5}
        }
        recommender = EmbeddingRecommender(config)
        
        # Encode data first
        encoded_df = recommender._encode_dataframe(simple_dataframe)
        embeddings = recommender._create_embeddings(encoded_df)
        
        # Should create valid embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(simple_dataframe)
        assert embeddings.shape[1] <= recommender.embedding_dim
    
    def test_create_embeddings_random(self, simple_dataframe):
        """Test random embedding creation."""
        config = {
            'embedding': {'method': 'random', 'embedding_dim': 3},
            'recommendation': {'top_k': 5}
        }
        recommender = EmbeddingRecommender(config)
        
        # Encode data first
        encoded_df = recommender._encode_dataframe(simple_dataframe)
        embeddings = recommender._create_embeddings(encoded_df)
        
        # Should create valid embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(simple_dataframe)
        assert embeddings.shape[1] == recommender.embedding_dim
    
    def test_create_embeddings_learned(self, simple_dataframe):
        """Test learned embedding creation."""
        config = {
            'embedding': {'method': 'learned', 'embedding_dim': 4},
            'recommendation': {'top_k': 5}
        }
        recommender = EmbeddingRecommender(config)
        
        # Encode data first
        encoded_df = recommender._encode_dataframe(simple_dataframe)
        embeddings = recommender._create_embeddings(encoded_df)
        
        # Should create valid embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(simple_dataframe)
    
    def test_compute_embedding_similarity(self, sample_config, simple_dataframe):
        """Test embedding similarity computation."""
        recommender = EmbeddingRecommender(sample_config)
        
        # Create simple embeddings
        embeddings = np.random.random((len(simple_dataframe), 4))
        similarity_scores = recommender._compute_embedding_similarity(embeddings)
        
        # Should return similarity scores
        assert isinstance(similarity_scores, np.ndarray)
        assert len(similarity_scores) == len(simple_dataframe)
    
    def test_different_embedding_methods(self, simple_dataframe):
        """Test all embedding methods."""
        methods = ['pca', 'random', 'learned']
        
        for method in methods:
            config = {
                'embedding': {'method': method, 'embedding_dim': 3},
                'recommendation': {'top_k': 3}
            }
            recommender = EmbeddingRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_different_distance_metrics(self, simple_dataframe):
        """Test different distance metrics."""
        metrics = ['cosine', 'euclidean', 'manhattan']
        
        for metric in metrics:
            config = {
                'embedding': {'distance_metric': metric},
                'recommendation': {'top_k': 3}
            }
            recommender = EmbeddingRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all metrics
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_caching_functionality(self, sample_config, simple_dataframe):
        """Test embedding computation caching."""
        recommender = EmbeddingRecommender(sample_config)
        
        # First call
        result1 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Second call with same data should use cache
        result2 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_clear_cache(self, sample_config, simple_dataframe):
        """Test clearing embedding caches."""
        recommender = EmbeddingRecommender(sample_config)
        
        # Build caches
        recommender.recommend_tuples(simple_dataframe)
        
        # Clear caches
        recommender.clear_cache()
        assert len(recommender._embedding_cache) == 0
        assert len(recommender._similarity_cache) == 0
    
    def test_reset_model(self, sample_config, simple_dataframe):
        """Test resetting the embedding model."""
        recommender = EmbeddingRecommender(sample_config)
        
        # Build model state
        recommender.recommend_tuples(simple_dataframe)
        
        # Reset model
        recommender.reset_model()
        assert not recommender._is_fitted
        assert recommender.embedding_model is None
        assert recommender.nn_model is None
    
    def test_get_embedding_info(self, sample_config, simple_dataframe):
        """Test getting embedding information."""
        recommender = EmbeddingRecommender(sample_config)
        
        # Process data to generate embeddings
        recommender.recommend_tuples(simple_dataframe)
        
        # Get embedding info
        info = recommender.get_embedding_info()
        
        # Should return info dictionary
        assert isinstance(info, dict)
    
    def test_precompute_embeddings(self, sample_config, simple_dataframe):
        """Test precomputing embeddings."""
        recommender = EmbeddingRecommender(sample_config)
        
        embeddings = recommender.precompute_embeddings(simple_dataframe)
        
        # Should return precomputed embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(simple_dataframe)
    
    def test_recommend_from_embeddings(self, sample_config, simple_dataframe):
        """Test recommending from precomputed embeddings."""
        recommender = EmbeddingRecommender(sample_config)
        
        # Precompute embeddings
        embeddings = recommender.precompute_embeddings(simple_dataframe)
        
        # Recommend from embeddings
        result = recommender.recommend_from_embeddings(embeddings, simple_dataframe)
        
        # Should return recommendations
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(simple_dataframe)
    
    def test_nearest_neighbors_algorithms(self, simple_dataframe):
        """Test different nearest neighbor algorithms."""
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        
        for algorithm in algorithms:
            config = {
                'embedding': {'nn_algorithm': algorithm},
                'recommendation': {'top_k': 3}
            }
            recommender = EmbeddingRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all algorithms
            assert isinstance(result, pd.DataFrame)
    
    def test_embedding_dimensions(self, simple_dataframe):
        """Test different embedding dimensions."""
        dimensions = [2, 8, 16, 32]
        
        for dim in dimensions:
            config = {
                'embedding': {'embedding_dim': dim, 'method': 'random'},
                'recommendation': {'top_k': 3}
            }
            recommender = EmbeddingRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all dimensions
            assert isinstance(result, pd.DataFrame)
    
    def test_large_dataset_performance(self, sample_config, large_dataframe):
        """Test performance with larger dataset."""
        recommender = EmbeddingRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle large dataset efficiently
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
    
    def test_categorical_encoding(self, sample_config):
        """Test encoding of categorical variables."""
        cat_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            'num': [1, 2, 3, 4, 5]
        })
        
        recommender = EmbeddingRecommender(sample_config)
        encoded_df = recommender._encode_dataframe(cat_df)
        
        # Should encode categorical variables properly
        assert isinstance(encoded_df, pd.DataFrame)
        assert len(encoded_df) == len(cat_df)
        # All columns should be numeric
        for dtype in encoded_df.dtypes:
            assert np.issubdtype(dtype, np.number)
    
    def test_missing_values_handling(self, sample_config):
        """Test handling of missing values."""
        df_with_na = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['a', np.nan, 'c', 'd'],
            'C': [1.1, 2.2, 3.3, np.nan]
        })
        
        recommender = EmbeddingRecommender(sample_config)
        result = recommender.recommend_tuples(df_with_na)
        
        # Should handle missing values without errors
        assert isinstance(result, pd.DataFrame)
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        recommender = EmbeddingRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = EmbeddingRecommender(sample_config)
        assert recommender.name() == "EmbeddingRecommender"
    
    def test_config_without_embedding_section(self, simple_dataframe):
        """Test with config missing embedding section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = EmbeddingRecommender(config)
        
        # Should use default embedding configuration
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.embedding_method == 'pca'  # default value
    
    def test_invalid_embedding_method(self, simple_dataframe):
        """Test handling of invalid embedding method."""
        config = {
            'embedding': {'method': 'invalid_method'},
            'recommendation': {'top_k': 3}
        }
        recommender = EmbeddingRecommender(config)
        
        # Should handle invalid method gracefully
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
