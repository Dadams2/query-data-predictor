"""
Tests for the BaseRecommender class.
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from query_data_predictor.recommenders.base_recommender import BaseRecommender, RecommendationMode
from .test_fixtures import (
    sample_config, simple_dataframe, config_top_quartile, 
    config_percentage, all_recommendation_modes
)


class ConcreteRecommender(BaseRecommender):
    """Concrete implementation of BaseRecommender for testing."""
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Simple implementation that returns the input DataFrame limited by config."""
        return self._limit_output(current_results)


class TestBaseRecommender:
    """Test suite for BaseRecommender class."""
    
    def test_init(self, sample_config):
        """Test BaseRecommender initialization."""
        recommender = ConcreteRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "ConcreteRecommender"
    
    def test_determine_output_size_top_k(self, sample_config):
        """Test _determine_output_size with top_k mode."""
        recommender = ConcreteRecommender(sample_config)
        size = recommender._determine_output_size(100)
        assert size == 5  # From config top_k
    
    def test_determine_output_size_top_quartile(self, config_top_quartile):
        """Test _determine_output_size with top_quartile mode."""
        recommender = ConcreteRecommender(config_top_quartile)
        size = recommender._determine_output_size(100)
        assert size == 25  # 100 // 4
        
        # Test with small dataset
        size = recommender._determine_output_size(3)
        assert size == 1  # max(1, 3 // 4)
    
    def test_determine_output_size_percentage(self, config_percentage):
        """Test _determine_output_size with percentage mode."""
        recommender = ConcreteRecommender(config_percentage)
        size = recommender._determine_output_size(100)
        assert size == 30  # 100 * 0.3
        
        # Test with small dataset
        size = recommender._determine_output_size(2)
        assert size == 1  # max(1, int(2 * 0.3))
    
    def test_determine_output_size_invalid_mode(self):
        """Test _determine_output_size with invalid mode."""
        config = {
            'recommendation': {
                'mode': 'invalid_mode',
                'top_k': 7
            }
        }
        recommender = ConcreteRecommender(config)
        size = recommender._determine_output_size(100)
        assert size == 7  # Should fall back to top_k
    
    def test_limit_output_empty_dataframe(self, sample_config):
        """Test _limit_output with empty DataFrame."""
        recommender = ConcreteRecommender(sample_config)
        empty_df = pd.DataFrame()
        result = recommender._limit_output(empty_df)
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_limit_output_smaller_than_limit(self, sample_config, simple_dataframe):
        """Test _limit_output when DataFrame is smaller than limit."""
        # Config has top_k = 5, but simple_dataframe has 10 rows
        # So first create a smaller dataframe
        small_df = simple_dataframe.head(3)
        recommender = ConcreteRecommender(sample_config)
        result = recommender._limit_output(small_df)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, small_df)
    
    def test_limit_output_larger_than_limit(self, sample_config, simple_dataframe):
        """Test _limit_output when DataFrame is larger than limit."""
        recommender = ConcreteRecommender(sample_config)
        result = recommender._limit_output(simple_dataframe)
        assert len(result) == 5  # Limited by top_k
        pd.testing.assert_frame_equal(result, simple_dataframe.head(5))
    
    def test_validate_input_valid(self, sample_config, simple_dataframe):
        """Test _validate_input with valid DataFrame."""
        recommender = ConcreteRecommender(sample_config)
        # Should not raise any exception
        recommender._validate_input(simple_dataframe)
    
    def test_validate_input_invalid_type(self, sample_config):
        """Test _validate_input with invalid input type."""
        recommender = ConcreteRecommender(sample_config)
        with pytest.raises(ValueError, match="current_results must be a pandas DataFrame"):
            recommender._validate_input("not a dataframe")
    
    def test_validate_input_none(self, sample_config):
        """Test _validate_input with None input."""
        recommender = ConcreteRecommender(sample_config)
        with pytest.raises(ValueError, match="current_results must be a pandas DataFrame"):
            recommender._validate_input(None)
    
    def test_name_method(self, sample_config):
        """Test name method returns correct class name."""
        recommender = ConcreteRecommender(sample_config)
        assert recommender.name() == "ConcreteRecommender"
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test that all recommendation modes work correctly."""
        recommender = ConcreteRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Result should not be empty (simple_dataframe has 10 rows)
        assert not result.empty
        # Result should not be larger than input
        assert len(result) <= len(simple_dataframe)
    
    def test_custom_config_key(self, simple_dataframe):
        """Test using custom config key for output size determination."""
        config = {
            'custom_key': {
                'mode': 'top_k',
                'top_k': 3
            }
        }
        recommender = ConcreteRecommender(config)
        size = recommender._determine_output_size(10, 'custom_key')
        assert size == 3
        
        result = recommender._limit_output(simple_dataframe, 'custom_key')
        assert len(result) == 3
    
    def test_recommendation_mode_enum(self):
        """Test RecommendationMode enum values."""
        assert RecommendationMode.TOP_K.value == "top_k"
        assert RecommendationMode.TOP_QUARTILE.value == "top_quartile"
        assert RecommendationMode.PERCENTAGE.value == "percentage"
