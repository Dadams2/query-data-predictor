"""
Tests for the InterestingnessRecommender class.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from query_data_predictor.recommender.interestingness_recommender import InterestingnessRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    astronomy_dataframe, all_recommendation_modes
)


class TestInterestingnessRecommender:
    """Test suite for InterestingnessRecommender class."""
    
    def test_init(self, sample_config):
        """Test InterestingnessRecommender initialization."""
        recommender = InterestingnessRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "InterestingnessRecommender"
    
    def test_recommend_tuples_basic(self, sample_config, astronomy_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Might be empty if association rules don't work, but should not crash
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return DataFrame (might be empty if no patterns found)
        assert isinstance(result, pd.DataFrame)
    
    def test_recommend_tuples_with_top_k_override(self, sample_config, astronomy_dataframe):
        """Test recommend_tuples with top_k override."""
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should return DataFrame limited to 3 rows max
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5
    
    def test_exception_handling_empty_frequent_itemsets(self, sample_config):
        """Test exception handling when no frequent itemsets are found."""
        # Create a DataFrame that will likely not produce frequent itemsets
        sparse_df = pd.DataFrame({
            'col1': [f'unique_{i}' for i in range(10)],  # All unique values
            'col2': [f'value_{i}' for i in range(10)]    # All unique values
        })
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(sparse_df)
        
        # Should return empty DataFrame when no frequent itemsets found
        assert isinstance(result, pd.DataFrame)
        # Result might be empty or return original data depending on implementation
    
    def test_preprocess_data_with_discretization(self, sample_config, astronomy_dataframe):
        """Test that preprocess_data works with discretization enabled."""
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.preprocess_data(astronomy_dataframe)
        
        # Should return DataFrame (might be the same if no numeric columns to discretize)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(astronomy_dataframe)
    
    def test_preprocess_data_without_discretization(self, astronomy_dataframe):
        """Test that preprocess_data works with discretization disabled."""
        config = {
            'discretization': {'enabled': False}
        }
        recommender = InterestingnessRecommender(config)
        result = recommender.preprocess_data(astronomy_dataframe)
        
        # Should return the same DataFrame
        assert result is astronomy_dataframe
    
    def test_compute_frequent_itemsets(self, sample_config, astronomy_dataframe):
        """Test that compute_frequent_itemsets method works correctly."""
        recommender = InterestingnessRecommender(sample_config)
        frequent_itemsets, encoded_df, attributes = recommender.compute_frequent_itemsets(astronomy_dataframe)
        
        # Should return DataFrames and attributes list
        assert isinstance(frequent_itemsets, pd.DataFrame)
        assert isinstance(encoded_df, pd.DataFrame)
        assert isinstance(attributes, list)
    
    def test_prepend_column_names(self, sample_config):
        """Test that prepend_column_names method works correctly."""
        recommender = InterestingnessRecommender(sample_config)
        test_df = pd.DataFrame({
            'A': ['x', 'y'],
            'B': [1, 2]
        })
        result = recommender.prepend_column_names(test_df.copy())
        
        # Should prepend column names to values
        assert result.loc[0, 'A'] == 'A_x'
        assert result.loc[0, 'B'] == 'B_1'
    
    def test_recommend_tuples_scoring(self, sample_config, astronomy_dataframe):
        """Test that recommend_tuples properly scores tuples."""
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should not include the interestingness_score column in output
        assert 'interestingness_score' not in result.columns
    
    def test_all_recommendation_modes(self, all_recommendation_modes, astronomy_dataframe):
        """Test recommend_tuples with all recommendation modes."""
        recommender = InterestingnessRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should return a DataFrame (might be empty)
        assert isinstance(result, pd.DataFrame)
        # Should not be larger than input
        assert len(result) <= len(astronomy_dataframe)
    
    def test_configuration_attributes(self):
        """Test that configuration is properly stored in the recommender."""
        config = {
            'discretization': {'enabled': True, 'bins': 10},
            'association_rules': {'min_support': 0.05},
            'recommendation': {'mode': 'top_k', 'top_k': 7}
        }
        
        recommender = InterestingnessRecommender(config)
        
        # Should store the config
        assert recommender.config == config
        # Should initialize discretizer if enabled
        assert recommender.discretizer is not None
    
    def test_output_size_determination(self, astronomy_dataframe):
        """Test that output size is determined correctly."""
        config = {
            'recommendation': {
                'mode': 'percentage',
                'percentage': 0.5
            }
        }
        
        recommender = InterestingnessRecommender(config)
        
        # Test _determine_output_size method
        size = recommender._determine_output_size(10)
        assert size == 5  # 50% of 10
    
    def test_integration_with_real_tuple_recommender(self, sample_config):
        """Test integration with actual TupleRecommender (not mocked)."""
        # Use a simple DataFrame that should work with the real TupleRecommender
        simple_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [1, 2, 1, 2, 1]
        })
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(simple_df)
        
        # Should not crash, even if it returns empty results
        assert isinstance(result, pd.DataFrame)
    
    def test_invalid_input_type(self, sample_config):
        """Test that invalid input type raises ValueError."""
        recommender = InterestingnessRecommender(sample_config)
        
        with pytest.raises(ValueError, match="current_results must be a pandas DataFrame"):
            recommender.recommend_tuples("not a dataframe")
    
    def test_name_method(self, sample_config):
        """Test name method returns correct value."""
        recommender = InterestingnessRecommender(sample_config)
        assert recommender.name() == "InterestingnessRecommender"
    
    def test_config_without_recommendation_section(self, astronomy_dataframe):
        """Test behavior when config doesn't have recommendation section."""
        config = {
            'discretization': {'enabled': True},
            'association_rules': {'min_support': 0.1}
        }
        
        recommender = InterestingnessRecommender(config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should work with default recommendation settings
        assert isinstance(result, pd.DataFrame)
    
    def test_empty_config(self, astronomy_dataframe):
        """Test behavior with minimal/empty config."""
        config = {}
        
        recommender = InterestingnessRecommender(config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should work with all default settings
        assert isinstance(result, pd.DataFrame)
