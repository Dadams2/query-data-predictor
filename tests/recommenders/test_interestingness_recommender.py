"""
Tests for the InterestingnessRecommender class.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.interestingness_recommender import InterestingnessRecommender
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
        assert hasattr(recommender, 'tuple_recommender')
    
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
        result = recommender.recommend_tuples(astronomy_dataframe, top_k=3)
        
        # Should return DataFrame limited to 3 rows max
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3
    
    @patch('query_data_predictor.recommenders.interestingness_recommender.TupleRecommender')
    def test_tuple_recommender_exception_handling(self, mock_tuple_recommender_class, sample_config, astronomy_dataframe):
        """Test exception handling when tuple recommender fails."""
        # Create mock that raises exception
        mock_instance = Mock()
        mock_instance.recommend_tuples.side_effect = Exception("Tuple recommender failed")
        mock_tuple_recommender_class.return_value = mock_instance
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should return empty DataFrame when tuple recommender fails
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('query_data_predictor.recommenders.interestingness_recommender.TupleRecommender')
    def test_preprocess_data_delegation(self, mock_tuple_recommender_class, sample_config, astronomy_dataframe):
        """Test that preprocess_data is properly delegated."""
        mock_instance = Mock()
        mock_instance.preprocess_data.return_value = astronomy_dataframe
        mock_tuple_recommender_class.return_value = mock_instance
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.preprocess_data(astronomy_dataframe)
        
        # Should call the underlying tuple recommender's method
        mock_instance.preprocess_data.assert_called_once_with(astronomy_dataframe)
        assert result is astronomy_dataframe
    
    @patch('query_data_predictor.recommenders.interestingness_recommender.TupleRecommender')
    def test_mine_association_rules_delegation(self, mock_tuple_recommender_class, sample_config, astronomy_dataframe):
        """Test that mine_association_rules is properly delegated."""
        mock_instance = Mock()
        mock_rules = pd.DataFrame({'rule': ['test_rule']})
        mock_instance.mine_association_rules.return_value = mock_rules
        mock_tuple_recommender_class.return_value = mock_instance
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.mine_association_rules(astronomy_dataframe)
        
        # Should call the underlying tuple recommender's method
        mock_instance.mine_association_rules.assert_called_once_with(astronomy_dataframe)
        assert result is mock_rules
    
    @patch('query_data_predictor.recommenders.interestingness_recommender.TupleRecommender')
    def test_generate_summaries_delegation(self, mock_tuple_recommender_class, sample_config, astronomy_dataframe):
        """Test that generate_summaries is properly delegated."""
        mock_instance = Mock()
        mock_summaries = [{'summary': 'test'}]
        mock_instance.generate_summaries.return_value = mock_summaries
        mock_tuple_recommender_class.return_value = mock_instance
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.generate_summaries(astronomy_dataframe)
        
        # Should call the underlying tuple recommender's method
        mock_instance.generate_summaries.assert_called_once_with(astronomy_dataframe)
        assert result is mock_summaries
    
    @patch('query_data_predictor.recommenders.interestingness_recommender.TupleRecommender')
    def test_recommend_tuples_delegation(self, mock_tuple_recommender_class, sample_config, astronomy_dataframe):
        """Test that recommend_tuples properly delegates to TupleRecommender."""
        mock_instance = Mock()
        mock_result = pd.DataFrame({'col': [1, 2, 3]})
        mock_instance.recommend_tuples.return_value = mock_result
        mock_tuple_recommender_class.return_value = mock_instance
        
        recommender = InterestingnessRecommender(sample_config)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should call the underlying tuple recommender's method
        mock_instance.recommend_tuples.assert_called_once()
        # Should apply additional limiting
        assert isinstance(result, pd.DataFrame)
    
    def test_all_recommendation_modes(self, all_recommendation_modes, astronomy_dataframe):
        """Test recommend_tuples with all recommendation modes."""
        recommender = InterestingnessRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(astronomy_dataframe)
        
        # Should return a DataFrame (might be empty)
        assert isinstance(result, pd.DataFrame)
        # Should not be larger than input
        assert len(result) <= len(astronomy_dataframe)
    
    def test_configuration_passed_to_tuple_recommender(self):
        """Test that configuration is passed to the underlying TupleRecommender."""
        config = {
            'discretization': {'enabled': True, 'bins': 10},
            'association_rules': {'min_support': 0.05},
            'recommendation': {'mode': 'top_k', 'top_k': 7}
        }
        
        with patch('query_data_predictor.recommenders.interestingness_recommender.TupleRecommender') as mock_class:
            recommender = InterestingnessRecommender(config)
            
            # Should initialize TupleRecommender with the same config
            mock_class.assert_called_once_with(config)
    
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
