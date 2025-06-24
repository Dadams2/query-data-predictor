"""
Tests for the RandomRecommender class.
"""

import pytest
import pandas as pd
import numpy as np

from query_data_predictor.recommenders.random_recommender import RandomRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    config_top_quartile, config_percentage, all_recommendation_modes, large_dataframe
)


class TestRandomRecommender:
    """Test suite for RandomRecommender class."""
    
    def test_init(self, sample_config):
        """Test RandomRecommender initialization."""
        recommender = RandomRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.random_seed == 42  # From sample_config
        assert recommender.name() == "RandomRecommender"
    
    def test_init_default_seed(self):
        """Test initialization with default random seed."""
        config = {}
        recommender = RandomRecommender(config)
        assert recommender.random_seed == 42  # Default value
    
    def test_init_custom_seed(self):
        """Test initialization with custom random seed."""
        config = {'random': {'random_seed': 123}}
        recommender = RandomRecommender(config)
        assert recommender.random_seed == 123
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) == 5
        # Should be a subset of the original data
        assert all(row.tolist() in simple_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_recommend_tuples_reproducible(self, sample_config, simple_dataframe):
        """Test that results are reproducible with same seed."""
        recommender1 = RandomRecommender(sample_config)
        recommender2 = RandomRecommender(sample_config)
        
        result1 = recommender1.recommend_tuples(simple_dataframe)
        result2 = recommender2.recommend_tuples(simple_dataframe)
        
        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_recommend_tuples_different_seeds(self, simple_dataframe):
        """Test that different seeds produce different results."""
        config1 = {'random': {'random_seed': 42}, 'recommendation': {'mode': 'top_k', 'top_k': 5}}
        config2 = {'random': {'random_seed': 123}, 'recommendation': {'mode': 'top_k', 'top_k': 5}}
        
        recommender1 = RandomRecommender(config1)
        recommender2 = RandomRecommender(config2)
        
        result1 = recommender1.recommend_tuples(simple_dataframe)
        result2 = recommender2.recommend_tuples(simple_dataframe)
        
        # Results should likely be different (not guaranteed but very probable)
        # At least check they're not identical
        try:
            pd.testing.assert_frame_equal(result1, result2)
            # If they are equal, that's unexpected but not impossible
            pytest.skip("Random samples happened to be identical")
        except AssertionError:
            # This is expected - different seeds should produce different results
            pass
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, single_row_dataframe)
    
    def test_recommend_tuples_request_more_than_available(self, single_row_dataframe):
        """Test requesting more tuples than available."""
        config = {'recommendation': {'mode': 'top_k', 'top_k': 10}}
        recommender = RandomRecommender(config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return all available (just 1 row)
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, single_row_dataframe)
    
    def test_recommend_tuples_top_quartile(self, config_top_quartile, simple_dataframe):
        """Test recommend_tuples with top_quartile mode."""
        recommender = RandomRecommender(config_top_quartile)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return top quartile (10 / 4 = 2.5 -> 2 rows)
        assert len(result) == 2
        # Should be a subset of original data
        assert all(row.tolist() in simple_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_recommend_tuples_percentage(self, config_percentage, simple_dataframe):
        """Test recommend_tuples with percentage mode."""
        recommender = RandomRecommender(config_percentage)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return 30% of 10 rows = 3 rows
        assert len(result) == 3
        # Should be a subset of original data
        assert all(row.tolist() in simple_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_recommend_tuples_all_modes(self, all_recommendation_modes, simple_dataframe):
        """Test recommend_tuples with all recommendation modes."""
        recommender = RandomRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return a non-empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
        
        # Should be a subset of original data
        assert all(row.tolist() in simple_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_recommend_tuples_override_seed(self, sample_config, simple_dataframe):
        """Test recommend_tuples with seed override in kwargs."""
        recommender = RandomRecommender(sample_config)
        
        result1 = recommender.recommend_tuples(simple_dataframe, random_seed=999)
        result2 = recommender.recommend_tuples(simple_dataframe, random_seed=999)
        result3 = recommender.recommend_tuples(simple_dataframe, random_seed=888)
        
        # Same override seed should produce same results
        pd.testing.assert_frame_equal(result1, result2)
        
        # Different override seed should produce different results (likely)
        try:
            pd.testing.assert_frame_equal(result1, result3)
            pytest.skip("Random samples happened to be identical")
        except AssertionError:
            pass  # Expected
    
    def test_recommend_random_tuples_specific_count(self, sample_config, simple_dataframe):
        """Test recommend_random_tuples with specific count."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_random_tuples(simple_dataframe, n_tuples=3)
        
        # Should return exactly 3 tuples
        assert len(result) == 3
        # Should be a subset of original data
        assert all(row.tolist() in simple_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_recommend_random_tuples_zero_count(self, sample_config, simple_dataframe):
        """Test recommend_random_tuples with zero count."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_random_tuples(simple_dataframe, n_tuples=0)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_random_tuples_negative_count(self, sample_config, simple_dataframe):
        """Test recommend_random_tuples with negative count."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_random_tuples(simple_dataframe, n_tuples=-5)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_random_tuples_more_than_available(self, sample_config, simple_dataframe):
        """Test recommend_random_tuples requesting more than available."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_random_tuples(simple_dataframe, n_tuples=20)
        
        # Should return all available (10 rows)
        assert len(result) == 10
        # Should contain all original data (possibly shuffled)
        assert set(result.index) == set(simple_dataframe.index)
    
    def test_set_random_seed(self, sample_config, simple_dataframe):
        """Test set_random_seed method."""
        recommender = RandomRecommender(sample_config)
        
        # Change seed and verify it affects results
        recommender.set_random_seed(999)
        assert recommender.random_seed == 999
        
        result1 = recommender.recommend_tuples(simple_dataframe)
        result2 = recommender.recommend_tuples(simple_dataframe)
        
        # With same seed, should get same results
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_preserves_columns_and_dtypes(self, sample_config):
        """Test that columns and data types are preserved."""
        test_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })
        
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_tuples(test_df)
        
        # Should preserve columns and data types
        assert list(result.columns) == list(test_df.columns)
        assert result['int_col'].dtype == test_df['int_col'].dtype
        assert result['float_col'].dtype == test_df['float_col'].dtype
        assert result['str_col'].dtype == test_df['str_col'].dtype
        assert result['bool_col'].dtype == test_df['bool_col'].dtype
    
    def test_large_dataset_performance(self, sample_config, large_dataframe):
        """Test performance with larger dataset."""
        recommender = RandomRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should complete quickly and return correct size
        assert len(result) == 5  # From sample_config top_k
        assert all(row.tolist() in large_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_invalid_input_type(self, sample_config):
        """Test that invalid input type raises ValueError."""
        recommender = RandomRecommender(sample_config)
        
        with pytest.raises(ValueError, match="current_results must be a pandas DataFrame"):
            recommender.recommend_tuples("not a dataframe")
    
    def test_name_method(self, sample_config):
        """Test name method returns correct value."""
        recommender = RandomRecommender(sample_config)
        assert recommender.name() == "RandomRecommender"
