"""
Tests for the SimilarityRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.similarity_recommender import SimilarityRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestSimilarityRecommender:
    """Test suite for SimilarityRecommender class."""
    
    def test_init(self, sample_config):
        """Test SimilarityRecommender initialization."""
        recommender = SimilarityRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "SimilarityRecommender"
        assert recommender.similarity_metric == 'cosine'  # default
        assert hasattr(recommender, 'scaler')
        assert hasattr(recommender, 'label_encoders')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom similarity configuration."""
        config = {
            'similarity': {
                'metric': 'euclidean',
                'use_pca': False,
                'pca_components': 0.8,
                'max_features': 50,
                'cache_enabled': False
            },
            'recommendation': {'top_k': 5}
        }
        recommender = SimilarityRecommender(config)
        assert recommender.similarity_metric == 'euclidean'
        assert recommender.use_pca is False
        assert recommender.pca_components == 0.8
        assert recommender.max_features == 50
        assert recommender.cache_enabled is False
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = SimilarityRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = SimilarityRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = SimilarityRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return empty DataFrame (need at least 2 rows for similarity)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_mixed_data(self, sample_config, mixed_dataframe):
        """Test recommend_tuples with mixed data types."""
        recommender = SimilarityRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should handle mixed data types
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(mixed_dataframe)
        assert list(result.columns) == list(mixed_dataframe.columns)
    
    def test_recommend_tuples_large_dataset(self, sample_config, large_dataframe):
        """Test recommend_tuples with larger dataset."""
        recommender = SimilarityRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work efficiently with larger datasets
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
        assert list(result.columns) == list(large_dataframe.columns)
    
    def test_encode_dataframe_fast(self, sample_config, mixed_dataframe):
        """Test efficient DataFrame encoding."""
        recommender = SimilarityRecommender(sample_config)
        encoded_df = recommender._encode_dataframe_fast(mixed_dataframe)
        
        # Should return encoded DataFrame
        assert isinstance(encoded_df, pd.DataFrame)
        assert len(encoded_df) == len(mixed_dataframe)
        # All columns should be numeric after encoding
        for dtype in encoded_df.dtypes:
            assert np.issubdtype(dtype, np.number)
    
    def test_encode_dataframe_fast_empty(self, sample_config, empty_dataframe):
        """Test encoding with empty DataFrame."""
        recommender = SimilarityRecommender(sample_config)
        encoded_df = recommender._encode_dataframe_fast(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(encoded_df, pd.DataFrame)
        assert encoded_df.empty
    
    def test_compute_similarity_scores_fast(self, sample_config):
        """Test fast similarity score computation."""
        recommender = SimilarityRecommender(sample_config)
        
        # Create simple numeric DataFrame
        test_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [2, 4, 6, 8]
        })
        
        scores = recommender._compute_similarity_scores_fast(test_df)
        
        # Should return similarity scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(test_df)
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
    
    def test_compute_similarity_scores_fast_empty(self, sample_config):
        """Test similarity computation with empty DataFrame."""
        recommender = SimilarityRecommender(sample_config)
        empty_df = pd.DataFrame()
        
        scores = recommender._compute_similarity_scores_fast(empty_df)
        
        # Should return empty array
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0
    
    def test_compute_similarity_scores_fast_single_row(self, sample_config):
        """Test similarity computation with single row."""
        recommender = SimilarityRecommender(sample_config)
        single_df = pd.DataFrame({'A': [1], 'B': [2]})
        
        scores = recommender._compute_similarity_scores_fast(single_df)
        
        # Should return empty array (need at least 2 rows)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0
    
    def test_different_similarity_metrics(self, simple_dataframe):
        """Test different similarity metrics."""
        metrics = ['cosine', 'euclidean', 'manhattan']
        
        for metric in metrics:
            config = {
                'similarity': {'metric': metric},
                'recommendation': {'top_k': 3}
            }
            recommender = SimilarityRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all metrics
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_pca_configuration(self, simple_dataframe):
        """Test PCA configuration options."""
        # Test with PCA enabled
        config_pca = {
            'similarity': {'use_pca': True, 'pca_components': 0.9},
            'recommendation': {'top_k': 3}
        }
        recommender = SimilarityRecommender(config_pca)
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        
        # Test with PCA disabled
        config_no_pca = {
            'similarity': {'use_pca': False},
            'recommendation': {'top_k': 3}
        }
        recommender = SimilarityRecommender(config_no_pca)
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_caching_functionality(self, sample_config, simple_dataframe):
        """Test similarity computation caching."""
        recommender = SimilarityRecommender(sample_config)
        
        # First call
        result1 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Second call with same data should use cache
        result2 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_categorical_feature_limitation(self, sample_config):
        """Test that categorical features are limited to prevent explosion."""
        # Create DataFrame with many unique categorical values
        large_cat_df = pd.DataFrame({
            'cat_col': [f'cat_{i}' for i in range(200)],  # Many unique values
            'num_col': range(200)
        })
        
        recommender = SimilarityRecommender(sample_config)
        encoded_df = recommender._encode_dataframe_fast(large_cat_df)
        
        # Should handle large categorical data gracefully
        assert isinstance(encoded_df, pd.DataFrame)
        assert len(encoded_df) == len(large_cat_df)
    
    def test_missing_values_handling(self, sample_config):
        """Test handling of missing values."""
        df_with_na = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['a', np.nan, 'c', 'd'],
            'C': [1.1, 2.2, 3.3, np.nan]
        })
        
        recommender = SimilarityRecommender(sample_config)
        result = recommender.recommend_tuples(df_with_na)
        
        # Should handle missing values without errors
        assert isinstance(result, pd.DataFrame)
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        recommender = SimilarityRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
    
    def test_invalid_similarity_metric(self, simple_dataframe):
        """Test handling of invalid similarity metric."""
        config = {
            'similarity': {'metric': 'invalid_metric'},
            'recommendation': {'top_k': 3}
        }
        recommender = SimilarityRecommender(config)
        
        # Should handle invalid metric gracefully (fall back to default)
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = SimilarityRecommender(sample_config)
        assert recommender.name() == "SimilarityRecommender"
    
    def test_config_without_similarity_section(self, simple_dataframe):
        """Test with config missing similarity section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = SimilarityRecommender(config)
        
        # Should use default similarity configuration
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.similarity_metric == 'cosine'  # default value
