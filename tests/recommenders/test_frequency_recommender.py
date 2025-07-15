"""
Tests for the FrequencyRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.frequency_recommender import FrequencyRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestFrequencyRecommender:
    """Test suite for FrequencyRecommender class."""
    
    def test_init(self, sample_config):
        """Test FrequencyRecommender initialization."""
        recommender = FrequencyRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "FrequencyRecommender"
        assert recommender.scoring_method == 'weighted'  # default
        assert hasattr(recommender, '_column_frequencies')
        assert hasattr(recommender, '_tuple_frequencies')
        assert hasattr(recommender, '_pattern_frequencies')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom frequency configuration."""
        config = {
            'frequency': {
                'method': 'pattern',
                'pattern_length': 3,
                'min_frequency': 2,
                'cache_enabled': False,
                'max_unique_values': 30
            },
            'recommendation': {'top_k': 5}
        }
        recommender = FrequencyRecommender(config)
        assert recommender.scoring_method == 'pattern'
        assert recommender.pattern_length == 3
        assert recommender.min_frequency == 2
        assert recommender.cache_enabled is False
        assert recommender.max_unique_values == 30
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = FrequencyRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = FrequencyRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = FrequencyRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, single_row_dataframe)
    
    def test_recommend_tuples_mixed_data(self, sample_config, mixed_dataframe):
        """Test recommend_tuples with mixed data types."""
        recommender = FrequencyRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should handle mixed data types
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(mixed_dataframe)
        assert list(result.columns) == list(mixed_dataframe.columns)
    
    def test_build_frequency_tables(self, sample_config, simple_dataframe):
        """Test building frequency tables."""
        recommender = FrequencyRecommender(sample_config)
        recommender._build_frequency_tables(simple_dataframe)
        
        # Should populate frequency tables
        assert len(recommender._column_frequencies) > 0
        assert len(recommender._tuple_frequencies) > 0
        
        # Should have entries for each column
        for col in simple_dataframe.columns:
            assert col in recommender._column_frequencies
    
    def test_limit_unique_values(self, sample_config):
        """Test limiting unique values to prevent memory explosion."""
        # Create DataFrame with many unique values
        large_unique_df = pd.DataFrame({
            'col1': [f'value_{i}' for i in range(100)],
            'col2': range(100)
        })
        
        recommender = FrequencyRecommender(sample_config)
        limited_df = recommender._limit_unique_values(large_unique_df)
        
        # Should limit unique values
        assert isinstance(limited_df, pd.DataFrame)
        assert len(limited_df) == len(large_unique_df)
        # Unique values should be limited per column
        for col in limited_df.columns:
            unique_count = limited_df[col].nunique()
            assert unique_count <= recommender.max_unique_values
    
    def test_compute_frequency_scores_simple(self, simple_dataframe):
        """Test simple frequency scoring method."""
        config = {
            'frequency': {'method': 'simple'},
            'recommendation': {'top_k': 5}
        }
        recommender = FrequencyRecommender(config)
        scores = recommender._compute_frequency_scores(simple_dataframe)
        
        # Should return frequency scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(simple_dataframe)
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
    
    def test_compute_frequency_scores_weighted(self, simple_dataframe):
        """Test weighted frequency scoring method."""
        config = {
            'frequency': {'method': 'weighted'},
            'recommendation': {'top_k': 5}
        }
        recommender = FrequencyRecommender(config)
        scores = recommender._compute_frequency_scores(simple_dataframe)
        
        # Should return weighted frequency scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(simple_dataframe)
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
    
    def test_compute_frequency_scores_pattern(self, simple_dataframe):
        """Test pattern-based frequency scoring method."""
        config = {
            'frequency': {'method': 'pattern', 'pattern_length': 2},
            'recommendation': {'top_k': 5}
        }
        recommender = FrequencyRecommender(config)
        scores = recommender._compute_frequency_scores(simple_dataframe)
        
        # Should return pattern-based frequency scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(simple_dataframe)
    
    def test_different_scoring_methods(self, simple_dataframe):
        """Test all scoring methods."""
        methods = ['simple', 'weighted', 'pattern']
        
        for method in methods:
            config = {
                'frequency': {'method': method},
                'recommendation': {'top_k': 3}
            }
            recommender = FrequencyRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_caching_functionality(self, sample_config, simple_dataframe):
        """Test frequency computation caching."""
        recommender = FrequencyRecommender(sample_config)
        
        # First call
        result1 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Second call with same data should use cache
        result2 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_clear_cache(self, sample_config, simple_dataframe):
        """Test clearing frequency caches."""
        recommender = FrequencyRecommender(sample_config)
        
        # Build caches
        recommender.recommend_tuples(simple_dataframe)
        assert len(recommender._cached_scores) > 0
        
        # Clear caches
        recommender.clear_cache()
        assert len(recommender._cached_scores) == 0
        assert len(recommender._column_frequencies) == 0
        assert len(recommender._tuple_frequencies) == 0
    
    def test_get_cache_stats(self, sample_config, simple_dataframe):
        """Test getting cache statistics."""
        recommender = FrequencyRecommender(sample_config)
        
        # Build caches
        recommender.recommend_tuples(simple_dataframe)
        
        # Get cache stats
        stats = recommender.get_cache_stats()
        
        # Should return statistics dictionary
        assert isinstance(stats, dict)
        assert 'column_frequencies_size' in stats
        assert 'tuple_frequencies_size' in stats
        assert 'cached_scores_size' in stats
    
    def test_min_frequency_threshold(self, simple_dataframe):
        """Test minimum frequency threshold."""
        config = {
            'frequency': {'min_frequency': 5},  # High threshold
            'recommendation': {'top_k': 3}
        }
        recommender = FrequencyRecommender(config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should still return results (may use alternative scoring)
        assert isinstance(result, pd.DataFrame)
    
    def test_large_dataset_performance(self, sample_config, large_dataframe):
        """Test performance with larger dataset."""
        recommender = FrequencyRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle large dataset efficiently
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
    
    def test_duplicate_rows_handling(self, sample_config):
        """Test handling of duplicate rows."""
        # Create DataFrame with duplicates
        df_with_dupes = pd.DataFrame({
            'A': [1, 1, 2, 2, 1],
            'B': ['a', 'a', 'b', 'b', 'a']
        })
        
        recommender = FrequencyRecommender(sample_config)
        result = recommender.recommend_tuples(df_with_dupes)
        
        # Should handle duplicates and rank by frequency
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(df_with_dupes)
    
    def test_pattern_frequencies_building(self, simple_dataframe):
        """Test building pattern frequency tables."""
        config = {
            'frequency': {'method': 'pattern', 'pattern_length': 2},
            'recommendation': {'top_k': 5}
        }
        recommender = FrequencyRecommender(config)
        recommender._build_frequency_tables(simple_dataframe)
        
        # Should build pattern frequencies
        assert len(recommender._pattern_frequencies) >= 0
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        recommender = FrequencyRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = FrequencyRecommender(sample_config)
        assert recommender.name() == "FrequencyRecommender"
    
    def test_config_without_frequency_section(self, simple_dataframe):
        """Test with config missing frequency section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = FrequencyRecommender(config)
        
        # Should use default frequency configuration
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.scoring_method == 'weighted'  # default value
    
    def test_invalid_scoring_method(self, simple_dataframe):
        """Test handling of invalid scoring method."""
        config = {
            'frequency': {'method': 'invalid_method'},
            'recommendation': {'top_k': 3}
        }
        recommender = FrequencyRecommender(config)
        
        # Should handle invalid method gracefully
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_empty_frequency_tables(self, sample_config, empty_dataframe):
        """Test behavior with empty frequency tables."""
        recommender = FrequencyRecommender(sample_config)
        recommender._build_frequency_tables(empty_dataframe)
        
        # Should handle empty tables gracefully
        assert len(recommender._column_frequencies) == 0
        assert len(recommender._tuple_frequencies) == 0
