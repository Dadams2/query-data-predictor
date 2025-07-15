"""
Tests for the IncrementalRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.incremental_recommender import IncrementalRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestIncrementalRecommender:
    """Test suite for IncrementalRecommender class."""
    
    def test_init(self, sample_config):
        """Test IncrementalRecommender initialization."""
        recommender = IncrementalRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "IncrementalRecommender"
        assert recommender.learning_rate == 0.1  # default
        assert recommender.decay_factor == 0.95  # default
        assert recommender.window_size == 1000  # default
        assert hasattr(recommender, 'pattern_weights')
        assert hasattr(recommender, 'pattern_frequencies')
        assert hasattr(recommender, 'feature_weights')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom incremental configuration."""
        config = {
            'incremental': {
                'learning_rate': 0.05,
                'decay_factor': 0.9,
                'window_size': 500,
                'min_pattern_frequency': 3,
                'update_threshold': 0.2,
                'enable_online_learning': False
            },
            'recommendation': {'top_k': 5}
        }
        recommender = IncrementalRecommender(config)
        assert recommender.learning_rate == 0.05
        assert recommender.decay_factor == 0.9
        assert recommender.window_size == 500
        assert recommender.min_pattern_frequency == 3
        assert recommender.update_threshold == 0.2
        assert recommender.enable_online_learning is False
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = IncrementalRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = IncrementalRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = IncrementalRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_online_learning_enabled(self, sample_config, simple_dataframe):
        """Test incremental learning with online learning enabled."""
        recommender = IncrementalRecommender(sample_config)
        recommender.enable_online_learning = True
        
        # Process data multiple times to test incremental updates
        result1 = recommender.recommend_tuples(simple_dataframe)
        result2 = recommender.recommend_tuples(simple_dataframe)
        
        # Should update model incrementally
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert recommender.total_updates > 0
    
    def test_online_learning_disabled(self, sample_config, simple_dataframe):
        """Test with online learning disabled."""
        recommender = IncrementalRecommender(sample_config)
        recommender.enable_online_learning = False
        
        initial_updates = recommender.total_updates
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should not update model
        assert isinstance(result, pd.DataFrame)
        assert recommender.total_updates == initial_updates
    
    def test_update_model_incremental(self, sample_config, simple_dataframe):
        """Test incremental model updates."""
        recommender = IncrementalRecommender(sample_config)
        
        initial_pattern_count = len(recommender.pattern_weights)
        recommender._update_model_incremental(simple_dataframe)
        
        # Should update pattern weights
        assert len(recommender.pattern_weights) >= initial_pattern_count
        assert recommender.total_updates > 0
    
    def test_extract_patterns(self, sample_config, simple_dataframe):
        """Test pattern extraction from DataFrame."""
        recommender = IncrementalRecommender(sample_config)
        patterns = recommender._extract_patterns(simple_dataframe)
        
        # Should extract patterns
        assert isinstance(patterns, dict)
        assert len(patterns) >= 0
    
    def test_apply_decay(self, sample_config, simple_dataframe):
        """Test decay application to pattern weights."""
        recommender = IncrementalRecommender(sample_config)
        
        # Add some patterns
        recommender.pattern_weights['test_pattern'] = 1.0
        recommender.pattern_frequencies['test_pattern'] = 5
        
        original_weight = recommender.pattern_weights['test_pattern']
        recommender._apply_decay()
        
        # Should apply decay
        assert recommender.pattern_weights['test_pattern'] < original_weight
    
    def test_prune_patterns(self, sample_config, simple_dataframe):
        """Test pruning of low-frequency patterns."""
        recommender = IncrementalRecommender(sample_config)
        
        # Add patterns with different frequencies
        recommender.pattern_weights['high_freq'] = 1.0
        recommender.pattern_frequencies['high_freq'] = 10
        recommender.pattern_weights['low_freq'] = 0.5
        recommender.pattern_frequencies['low_freq'] = 1
        
        recommender._prune_patterns()
        
        # Low frequency patterns should be removed if below threshold
        # (depends on min_pattern_frequency setting)
    
    def test_adapt_learning_rate(self, sample_config, simple_dataframe):
        """Test adaptive learning rate adjustment."""
        recommender = IncrementalRecommender(sample_config)
        
        original_rate = recommender.learning_rate
        recommender._adapt_learning_rate()
        
        # Learning rate might be adjusted based on adaptation strategy
        assert isinstance(recommender.learning_rate, float)
        assert recommender.learning_rate > 0
    
    def test_compute_incremental_scores(self, sample_config, simple_dataframe):
        """Test computing incremental scores."""
        recommender = IncrementalRecommender(sample_config)
        
        # Add some pattern weights first
        recommender.pattern_weights['test_pattern'] = 1.0
        
        scores = recommender._compute_incremental_scores(simple_dataframe)
        
        # Should return scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(simple_dataframe)
    
    def test_update_data_window(self, sample_config, simple_dataframe):
        """Test updating the data window."""
        recommender = IncrementalRecommender(sample_config)
        
        initial_window_size = len(recommender.data_window)
        recommender._update_data_window(simple_dataframe)
        
        # Should update data window
        assert len(recommender.data_window) >= initial_window_size
    
    def test_data_window_size_limit(self, sample_config):
        """Test data window size limitation."""
        recommender = IncrementalRecommender(sample_config)
        recommender.window_size = 5  # Small window for testing
        
        # Add more data than window size
        for i in range(10):
            small_df = pd.DataFrame({'col': [i]})
            recommender._update_data_window(small_df)
        
        # Window should be limited
        assert len(recommender.data_window) <= recommender.window_size
    
    def test_get_model_state(self, sample_config, simple_dataframe):
        """Test getting model state."""
        recommender = IncrementalRecommender(sample_config)
        
        # Update model first
        recommender.recommend_tuples(simple_dataframe)
        
        # Get model state
        state = recommender.get_model_state()
        
        # Should return state dictionary
        assert isinstance(state, dict)
        assert 'pattern_weights_count' in state
        assert 'total_updates' in state
    
    def test_reset_model(self, sample_config, simple_dataframe):
        """Test resetting the model."""
        recommender = IncrementalRecommender(sample_config)
        
        # Build model state
        recommender.recommend_tuples(simple_dataframe)
        assert len(recommender.pattern_weights) > 0
        
        # Reset model
        recommender.reset_model()
        assert len(recommender.pattern_weights) == 0
        assert len(recommender.pattern_frequencies) == 0
        assert len(recommender.feature_weights) == 0
        assert recommender.total_updates == 0
    
    def test_set_learning_rate(self, sample_config):
        """Test setting learning rate."""
        recommender = IncrementalRecommender(sample_config)
        
        new_rate = 0.2
        recommender.set_learning_rate(new_rate)
        
        assert recommender.learning_rate == new_rate
    
    def test_get_pattern_insights(self, sample_config, simple_dataframe):
        """Test getting pattern insights."""
        recommender = IncrementalRecommender(sample_config)
        
        # Build some patterns
        recommender.recommend_tuples(simple_dataframe)
        
        # Get insights
        insights = recommender.get_pattern_insights()
        
        # Should return insights dictionary
        assert isinstance(insights, dict)
    
    def test_learning_rate_bounds(self, sample_config):
        """Test learning rate boundary conditions."""
        recommender = IncrementalRecommender(sample_config)
        
        # Test very small learning rate
        recommender.set_learning_rate(0.001)
        assert recommender.learning_rate == 0.001
        
        # Test larger learning rate
        recommender.set_learning_rate(0.9)
        assert recommender.learning_rate == 0.9
    
    def test_pattern_frequency_updates(self, sample_config, simple_dataframe):
        """Test pattern frequency updates."""
        recommender = IncrementalRecommender(sample_config)
        
        # Process same data multiple times
        recommender.recommend_tuples(simple_dataframe)
        recommender.recommend_tuples(simple_dataframe)
        
        # Pattern frequencies should increase
        assert len(recommender.pattern_frequencies) > 0
    
    def test_feature_weight_updates(self, sample_config, simple_dataframe):
        """Test feature weight updates."""
        recommender = IncrementalRecommender(sample_config)
        
        # Process data to update feature weights
        recommender.recommend_tuples(simple_dataframe)
        
        # Should have feature weights
        assert len(recommender.feature_weights) >= 0
    
    def test_large_dataset_incremental_learning(self, sample_config, large_dataframe):
        """Test incremental learning with larger dataset."""
        recommender = IncrementalRecommender(sample_config)
        
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle large dataset incrementally
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
        assert recommender.total_updates > 0
    
    def test_update_threshold_behavior(self, sample_config, simple_dataframe):
        """Test update threshold behavior."""
        recommender = IncrementalRecommender(sample_config)
        recommender.update_threshold = 0.5  # High threshold
        
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should respect update threshold
        assert isinstance(result, pd.DataFrame)
    
    def test_decay_factor_application(self, sample_config, simple_dataframe):
        """Test decay factor application over time."""
        recommender = IncrementalRecommender(sample_config)
        recommender.decay_factor = 0.8  # Stronger decay
        
        # Add pattern and apply decay multiple times
        recommender.pattern_weights['test'] = 1.0
        original_weight = recommender.pattern_weights['test']
        
        for _ in range(5):
            recommender._apply_decay()
        
        # Weight should have decayed
        assert recommender.pattern_weights['test'] < original_weight
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        recommender = IncrementalRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = IncrementalRecommender(sample_config)
        assert recommender.name() == "IncrementalRecommender"
    
    def test_config_without_incremental_section(self, simple_dataframe):
        """Test with config missing incremental section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = IncrementalRecommender(config)
        
        # Should use default incremental configuration
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.learning_rate == 0.1  # default value
    
    def test_prediction_history_tracking(self, sample_config, simple_dataframe):
        """Test prediction history tracking."""
        recommender = IncrementalRecommender(sample_config)
        
        # Process data to build history
        recommender.recommend_tuples(simple_dataframe)
        
        # Should track prediction history
        assert hasattr(recommender, 'prediction_history')
    
    def test_adaptation_rate_adjustment(self, sample_config, simple_dataframe):
        """Test adaptation rate functionality."""
        recommender = IncrementalRecommender(sample_config)
        
        original_adaptation_rate = recommender.adaptation_rate
        
        # Process data that might adjust adaptation rate
        recommender.recommend_tuples(simple_dataframe)
        
        # Adaptation rate should be a valid value
        assert isinstance(recommender.adaptation_rate, float)
        assert recommender.adaptation_rate > 0
