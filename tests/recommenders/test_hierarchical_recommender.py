"""
Tests for the HierarchicalRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.hierarchical_recommender import HierarchicalRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestHierarchicalRecommender:
    """Test suite for HierarchicalRecommender class."""
    
    def test_init(self, sample_config):
        """Test HierarchicalRecommender initialization."""
        recommender = HierarchicalRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "HierarchicalRecommender"
        assert recommender.coarse_method == 'clustering'  # default
        assert recommender.fine_method == 'similarity'  # default
        assert recommender.coarse_ratio == 0.3  # default
        assert hasattr(recommender, 'scaler')
        assert hasattr(recommender, 'kmeans')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom hierarchical configuration."""
        config = {
            'hierarchical': {
                'coarse_method': 'sampling',
                'fine_method': 'variance',
                'coarse_ratio': 0.5,
                'min_coarse_candidates': 20,
                'max_coarse_candidates': 200,
                'n_clusters': 5
            },
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        assert recommender.coarse_method == 'sampling'
        assert recommender.fine_method == 'variance'
        assert recommender.coarse_ratio == 0.5
        assert recommender.min_coarse_candidates == 20
        assert recommender.max_coarse_candidates == 200
        assert recommender.n_clusters == 5
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_small_dataset(self, sample_config, simple_dataframe):
        """Test with small dataset (below coarse filtering threshold)."""
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should handle small datasets gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(simple_dataframe)
    
    def test_recommend_tuples_large_dataset(self, sample_config, large_dataframe):
        """Test with large dataset requiring hierarchical processing."""
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should use hierarchical approach
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
    
    def test_coarse_filtering_clustering(self, large_dataframe):
        """Test coarse filtering with clustering method."""
        config = {
            'hierarchical': {'coarse_method': 'clustering', 'n_clusters': 3},
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with clustering-based coarse filtering
        assert isinstance(result, pd.DataFrame)
    
    def test_coarse_filtering_sampling(self, large_dataframe):
        """Test coarse filtering with sampling method."""
        config = {
            'hierarchical': {'coarse_method': 'sampling'},
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with sampling-based coarse filtering
        assert isinstance(result, pd.DataFrame)
    
    def test_coarse_filtering_frequency(self, large_dataframe):
        """Test coarse filtering with frequency method."""
        config = {
            'hierarchical': {'coarse_method': 'frequency'},
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with frequency-based coarse filtering
        assert isinstance(result, pd.DataFrame)
    
    def test_fine_filtering_similarity(self, sample_config, large_dataframe):
        """Test fine filtering with similarity method."""
        recommender = HierarchicalRecommender(sample_config)
        recommender.fine_method = 'similarity'
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with similarity-based fine filtering
        assert isinstance(result, pd.DataFrame)
    
    def test_fine_filtering_variance(self, large_dataframe):
        """Test fine filtering with variance method."""
        config = {
            'hierarchical': {'fine_method': 'variance'},
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with variance-based fine filtering
        assert isinstance(result, pd.DataFrame)
    
    def test_fine_filtering_entropy(self, large_dataframe):
        """Test fine filtering with entropy method."""
        config = {
            'hierarchical': {'fine_method': 'entropy'},
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with entropy-based fine filtering
        assert isinstance(result, pd.DataFrame)
    
    def test_different_coarse_methods(self, large_dataframe):
        """Test all coarse filtering methods."""
        methods = ['clustering', 'sampling', 'frequency']
        
        for method in methods:
            config = {
                'hierarchical': {'coarse_method': method},
                'recommendation': {'top_k': 3}
            }
            recommender = HierarchicalRecommender(config)
            result = recommender.recommend_tuples(large_dataframe)
            
            # Should work with all coarse methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_different_fine_methods(self, large_dataframe):
        """Test all fine filtering methods."""
        methods = ['similarity', 'variance', 'entropy']
        
        for method in methods:
            config = {
                'hierarchical': {'fine_method': method},
                'recommendation': {'top_k': 3}
            }
            recommender = HierarchicalRecommender(config)
            result = recommender.recommend_tuples(large_dataframe)
            
            # Should work with all fine methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_coarse_ratio_settings(self, large_dataframe):
        """Test different coarse ratio settings."""
        ratios = [0.1, 0.3, 0.5, 0.8]
        
        for ratio in ratios:
            config = {
                'hierarchical': {'coarse_ratio': ratio},
                'recommendation': {'top_k': 5}
            }
            recommender = HierarchicalRecommender(config)
            result = recommender.recommend_tuples(large_dataframe)
            
            # Should work with all ratios
            assert isinstance(result, pd.DataFrame)
    
    def test_candidate_limits(self, large_dataframe):
        """Test minimum and maximum candidate limits."""
        config = {
            'hierarchical': {
                'min_coarse_candidates': 10,
                'max_coarse_candidates': 50
            },
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should respect candidate limits
        assert isinstance(result, pd.DataFrame)
    
    def test_clustering_configuration(self, large_dataframe):
        """Test clustering configuration options."""
        config = {
            'hierarchical': {
                'coarse_method': 'clustering',
                'n_clusters': 5
            },
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should use specified number of clusters
        assert isinstance(result, pd.DataFrame)
        assert recommender.n_clusters == 5
    
    def test_caching_functionality(self, sample_config, large_dataframe):
        """Test hierarchical computation caching."""
        recommender = HierarchicalRecommender(sample_config)
        
        # First call
        result1 = recommender.recommend_tuples(large_dataframe.copy())
        
        # Second call with same data should use cache
        result2 = recommender.recommend_tuples(large_dataframe.copy())
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_mixed_data_handling(self, sample_config, mixed_dataframe):
        """Test handling of mixed data types."""
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should handle mixed data types
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(mixed_dataframe.columns)
    
    def test_single_row_handling(self, sample_config, single_row_dataframe):
        """Test handling of single row input."""
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should handle single row gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_very_small_coarse_ratio(self, large_dataframe):
        """Test with very small coarse ratio."""
        config = {
            'hierarchical': {'coarse_ratio': 0.01},  # Very small ratio
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle very small ratios (minimum candidates should apply)
        assert isinstance(result, pd.DataFrame)
    
    def test_very_large_coarse_ratio(self, large_dataframe):
        """Test with very large coarse ratio."""
        config = {
            'hierarchical': {'coarse_ratio': 0.99},  # Very large ratio
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle very large ratios (maximum candidates should apply)
        assert isinstance(result, pd.DataFrame)
    
    def test_performance_with_large_clusters(self, large_dataframe):
        """Test performance with many clusters."""
        config = {
            'hierarchical': {
                'coarse_method': 'clustering',
                'n_clusters': 20  # Many clusters
            },
            'recommendation': {'top_k': 5}
        }
        recommender = HierarchicalRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle many clusters efficiently
        assert isinstance(result, pd.DataFrame)
    
    def test_numeric_data_scaling(self, sample_config):
        """Test proper scaling of numeric data."""
        numeric_df = pd.DataFrame({
            'col1': [1, 100, 1000, 10000],
            'col2': [0.1, 1.0, 10.0, 100.0],
            'col3': [-5, 0, 5, 10]
        })
        
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(numeric_df)
        
        # Should handle different scales properly
        assert isinstance(result, pd.DataFrame)
    
    def test_categorical_data_encoding(self, sample_config):
        """Test encoding of categorical data."""
        cat_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'] * 10,
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'] * 10,
            'num': range(50)
        })
        
        recommender = HierarchicalRecommender(sample_config)
        result = recommender.recommend_tuples(cat_df)
        
        # Should handle categorical data properly
        assert isinstance(result, pd.DataFrame)
    
    def test_all_recommendation_modes(self, all_recommendation_modes, large_dataframe):
        """Test all recommendation output modes."""
        recommender = HierarchicalRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(large_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = HierarchicalRecommender(sample_config)
        assert recommender.name() == "HierarchicalRecommender"
    
    def test_config_without_hierarchical_section(self, large_dataframe):
        """Test with config missing hierarchical section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = HierarchicalRecommender(config)
        
        # Should use default hierarchical configuration
        result = recommender.recommend_tuples(large_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.coarse_method == 'clustering'  # default value
    
    def test_invalid_coarse_method(self, large_dataframe):
        """Test handling of invalid coarse method."""
        config = {
            'hierarchical': {'coarse_method': 'invalid_method'},
            'recommendation': {'top_k': 3}
        }
        recommender = HierarchicalRecommender(config)
        
        # Should handle invalid method gracefully
        result = recommender.recommend_tuples(large_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_invalid_fine_method(self, large_dataframe):
        """Test handling of invalid fine method."""
        config = {
            'hierarchical': {'fine_method': 'invalid_method'},
            'recommendation': {'top_k': 3}
        }
        recommender = HierarchicalRecommender(config)
        
        # Should handle invalid method gracefully
        result = recommender.recommend_tuples(large_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_edge_case_n_clusters(self, sample_config, simple_dataframe):
        """Test edge cases for number of clusters."""
        recommender = HierarchicalRecommender(sample_config)
        recommender.n_clusters = 1  # Single cluster
        
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should handle single cluster case
        assert isinstance(result, pd.DataFrame)
    
    def test_cache_invalidation(self, sample_config, large_dataframe):
        """Test cache invalidation with different data."""
        recommender = HierarchicalRecommender(sample_config)
        
        # Process first dataset
        result1 = recommender.recommend_tuples(large_dataframe)
        
        # Process different dataset
        different_df = large_dataframe.copy()
        different_df['new_col'] = range(len(different_df))
        result2 = recommender.recommend_tuples(different_df)
        
        # Should handle different data properly
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
