"""
Tests for the DummyRecommender class.
"""

import pytest
import pandas as pd

from query_data_predictor.recommender.dummy_recommender import DummyRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    config_top_quartile, config_percentage, all_recommendation_modes, large_dataframe
)


class TestDummyRecommender:
    """Test suite for DummyRecommender class."""
    
    def test_init(self, sample_config):
        """Test DummyRecommender initialization."""
        recommender = DummyRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "DummyRecommender"
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) == 5
        # Should be the first 5 rows of input
        pd.testing.assert_frame_equal(result, simple_dataframe.head(5))
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, single_row_dataframe)
    
    def test_recommend_tuples_top_quartile(self, config_top_quartile, simple_dataframe):
        """Test recommend_tuples with top_quartile mode."""
        recommender = DummyRecommender(config_top_quartile)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return top quartile (10 / 4 = 2.5 -> 2 rows, but max(1, 2) = 2)
        assert len(result) == 2
        pd.testing.assert_frame_equal(result, simple_dataframe.head(2))
    
    def test_recommend_tuples_percentage(self, config_percentage, simple_dataframe):
        """Test recommend_tuples with percentage mode."""
        recommender = DummyRecommender(config_percentage)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return 30% of 10 rows = 3 rows
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, simple_dataframe.head(3))
    
    def test_recommend_tuples_all_modes(self, all_recommendation_modes, simple_dataframe):
        """Test recommend_tuples with all recommendation modes."""
        recommender = DummyRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return a non-empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
        
        # Should be a subset of the original data (first n rows)
        expected_size = len(result)
        pd.testing.assert_frame_equal(result, simple_dataframe.head(expected_size))
    
    def test_recommend_tuples_larger_dataset(self, sample_config, large_dataframe):
        """Test recommend_tuples with larger dataset."""
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should be limited by top_k (5)
        assert len(result) == 5
        # Should be the first 5 rows
        pd.testing.assert_frame_equal(result, large_dataframe.head(5))
    
    def test_recommend_tuples_with_kwargs(self, sample_config, simple_dataframe):
        """Test recommend_tuples ignores additional keyword arguments."""
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(
            simple_dataframe, 
            extra_param="ignored",
            another_param=123
        )
        
        # Should work normally and ignore extra kwargs
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        pd.testing.assert_frame_equal(result, simple_dataframe.head(5))
    
    def test_recommend_tuples_preserves_columns(self, sample_config):
        """Test that recommend_tuples preserves all columns."""
        # Create a DataFrame with many columns
        test_df = pd.DataFrame({
            f'col_{i}': range(i, i+10) for i in range(5)
        })
        
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(test_df)
        
        # Should preserve all columns
        assert list(result.columns) == list(test_df.columns)
        assert len(result) == 5
    
    def test_recommend_tuples_preserves_index(self, sample_config):
        """Test that recommend_tuples preserves the original index."""
        # Create DataFrame with custom index
        test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': list('abcdefghij')
        }, index=[f'row_{i}' for i in range(10)])
        
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(test_df)
        
        # Should preserve the index
        expected_index = [f'row_{i}' for i in range(5)]
        assert list(result.index) == expected_index
    
    def test_recommend_tuples_data_types_preserved(self, sample_config):
        """Test that data types are preserved."""
        test_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })
        
        recommender = DummyRecommender(sample_config)
        result = recommender.recommend_tuples(test_df)
        
        # Should preserve data types
        assert result['int_col'].dtype == test_df['int_col'].dtype
        assert result['float_col'].dtype == test_df['float_col'].dtype
        assert result['str_col'].dtype == test_df['str_col'].dtype
        assert result['bool_col'].dtype == test_df['bool_col'].dtype
    
    def test_invalid_input_type(self, sample_config):
        """Test that invalid input type raises ValueError."""
        recommender = DummyRecommender(sample_config)
        
        with pytest.raises(ValueError, match="current_results must be a pandas DataFrame"):
            recommender.recommend_tuples("not a dataframe")
    
    def test_name_method(self, sample_config):
        """Test name method returns correct value."""
        recommender = DummyRecommender(sample_config)
        assert recommender.name() == "DummyRecommender"
    
    def test_config_without_recommendation_section(self, simple_dataframe):
        """Test behavior when config doesn't have recommendation section."""
        config = {}  # Empty config
        recommender = DummyRecommender(config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should use default top_k = 10
        assert len(result) == 10  # All rows since default is 10 and we have 10 rows
        pd.testing.assert_frame_equal(result, simple_dataframe)
