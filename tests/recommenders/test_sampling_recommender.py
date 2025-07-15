"""
Tests for the SamplingRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.sampling_recommender import SamplingRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestSamplingRecommender:
    """Test suite for SamplingRecommender class."""
    
    def test_init(self, sample_config):
        """Test SamplingRecommender initialization."""
        recommender = SamplingRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "SamplingRecommender"
        assert recommender.sampling_method == 'stratified'  # default
        assert recommender.sample_size == 1000  # default
        assert recommender.sample_ratio == 0.1  # default
    
    def test_init_with_custom_config(self):
        """Test initialization with custom sampling configuration."""
        config = {
            'sampling': {
                'method': 'random',
                'sample_size': 500,
                'sample_ratio': 0.2,
                'min_sample_size': 50,
                'scoring_method': 'entropy',
                'use_confidence_intervals': False,
                'confidence_level': 0.99
            },
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        assert recommender.sampling_method == 'random'
        assert recommender.sample_size == 500
        assert recommender.sample_ratio == 0.2
        assert recommender.min_sample_size == 50
        assert recommender.scoring_method == 'entropy'
        assert recommender.use_confidence_intervals is False
        assert recommender.confidence_level == 0.99
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = SamplingRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = SamplingRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_small_dataset(self, sample_config, simple_dataframe):
        """Test recommend_tuples with small dataset (no sampling needed)."""
        recommender = SamplingRecommender(sample_config)
        # simple_dataframe has 10 rows, which is below min_sample_size default (100)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should process without sampling
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(simple_dataframe)
    
    def test_recommend_tuples_large_dataset(self, sample_config, large_dataframe):
        """Test recommend_tuples with large dataset requiring sampling."""
        recommender = SamplingRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should use sampling for efficiency
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
    
    def test_determine_sample_size(self, sample_config):
        """Test determining optimal sample size."""
        recommender = SamplingRecommender(sample_config)
        
        # Test with various dataset sizes
        assert recommender._determine_sample_size(50) == 100  # min_sample_size
        assert recommender._determine_sample_size(1000) == 100  # 10% of 1000
        assert recommender._determine_sample_size(20000) == 1000  # sample_size limit
    
    def test_create_sample_random(self, large_dataframe):
        """Test random sampling method."""
        config = {
            'sampling': {'method': 'random'},
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        
        sample_size = 50
        sample_df, indices = recommender._create_sample(large_dataframe, sample_size)
        
        # Should create valid sample
        assert isinstance(sample_df, pd.DataFrame)
        assert isinstance(indices, np.ndarray)
        assert len(sample_df) == sample_size
        assert len(indices) == sample_size
        assert all(0 <= idx < len(large_dataframe) for idx in indices)
    
    def test_create_sample_stratified(self, mixed_dataframe):
        """Test stratified sampling method."""
        config = {
            'sampling': {'method': 'stratified'},
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        
        sample_size = 6
        sample_df, indices = recommender._create_sample(mixed_dataframe, sample_size)
        
        # Should create valid stratified sample
        assert isinstance(sample_df, pd.DataFrame)
        assert isinstance(indices, np.ndarray)
        assert len(sample_df) == sample_size
        assert len(indices) == sample_size
    
    def test_create_sample_systematic(self, large_dataframe):
        """Test systematic sampling method."""
        config = {
            'sampling': {'method': 'systematic'},
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        
        sample_size = 20
        sample_df, indices = recommender._create_sample(large_dataframe, sample_size)
        
        # Should create valid systematic sample
        assert isinstance(sample_df, pd.DataFrame)
        assert isinstance(indices, np.ndarray)
        assert len(sample_df) == sample_size
        assert len(indices) == sample_size
    
    def test_compute_sample_scores_variance(self, mixed_dataframe):
        """Test variance-based scoring method."""
        config = {
            'sampling': {'scoring_method': 'variance'},
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        scores = recommender._compute_sample_scores(mixed_dataframe)
        
        # Should return variance scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(mixed_dataframe)
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
    
    def test_compute_sample_scores_entropy(self, mixed_dataframe):
        """Test entropy-based scoring method."""
        config = {
            'sampling': {'scoring_method': 'entropy'},
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        scores = recommender._compute_sample_scores(mixed_dataframe)
        
        # Should return entropy scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(mixed_dataframe)
    
    def test_compute_sample_scores_frequency(self, mixed_dataframe):
        """Test frequency-based scoring method."""
        config = {
            'sampling': {'scoring_method': 'frequency'},
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        scores = recommender._compute_sample_scores(mixed_dataframe)
        
        # Should return frequency scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(mixed_dataframe)
    
    def test_different_sampling_methods(self, large_dataframe):
        """Test all sampling methods."""
        methods = ['random', 'stratified', 'systematic']
        
        for method in methods:
            config = {
                'sampling': {'method': method},
                'recommendation': {'top_k': 3}
            }
            recommender = SamplingRecommender(config)
            result = recommender.recommend_tuples(large_dataframe)
            
            # Should work with all methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_different_scoring_methods(self, large_dataframe):
        """Test all scoring methods."""
        methods = ['variance', 'entropy', 'frequency']
        
        for method in methods:
            config = {
                'sampling': {'scoring_method': method},
                'recommendation': {'top_k': 3}
            }
            recommender = SamplingRecommender(config)
            result = recommender.recommend_tuples(large_dataframe)
            
            # Should work with all scoring methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_caching_functionality(self, sample_config, large_dataframe):
        """Test sampling computation caching."""
        recommender = SamplingRecommender(sample_config)
        
        # First call
        result1 = recommender.recommend_tuples(large_dataframe.copy())
        
        # Second call with same data should use cache
        result2 = recommender.recommend_tuples(large_dataframe.copy())
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_extrapolate_scores(self, sample_config, large_dataframe):
        """Test score extrapolation from sample to full dataset."""
        recommender = SamplingRecommender(sample_config)
        
        # Create a sample
        sample_size = 20
        sample_df, sample_indices = recommender._create_sample(large_dataframe, sample_size)
        sample_scores = np.random.random(sample_size)
        
        # Extrapolate scores
        full_scores = recommender._extrapolate_scores(
            large_dataframe, sample_df, sample_scores, sample_indices
        )
        
        # Should return scores for full dataset
        assert isinstance(full_scores, np.ndarray)
        assert len(full_scores) == len(large_dataframe)
    
    def test_get_sample_statistics(self, sample_config, large_dataframe):
        """Test getting sample statistics."""
        recommender = SamplingRecommender(sample_config)
        
        # Process data to generate statistics
        recommender.recommend_tuples(large_dataframe)
        
        # Get statistics
        stats = recommender.get_sample_statistics()
        
        # Should return statistics dictionary
        assert isinstance(stats, dict)
    
    def test_confidence_intervals(self, large_dataframe):
        """Test confidence interval computation."""
        config = {
            'sampling': {
                'use_confidence_intervals': True,
                'confidence_level': 0.95
            },
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with confidence intervals enabled
        assert isinstance(result, pd.DataFrame)
    
    def test_sample_ratio_vs_sample_size(self, large_dataframe):
        """Test interaction between sample_ratio and sample_size."""
        config = {
            'sampling': {
                'sample_ratio': 0.05,  # 5%
                'sample_size': 1000    # Fixed size
            },
            'recommendation': {'top_k': 5}
        }
        recommender = SamplingRecommender(config)
        
        # For large_dataframe (100 rows), 5% = 5 rows, but min is sample_size constraint
        sample_size = recommender._determine_sample_size(len(large_dataframe))
        assert sample_size == 100  # Should use min_sample_size (100) as it's larger than 5% of 100
    
    def test_empty_sample_handling(self, sample_config):
        """Test handling when sample would be empty."""
        # Create very small DataFrame
        tiny_df = pd.DataFrame({'A': [1], 'B': [2]})
        
        recommender = SamplingRecommender(sample_config)
        result = recommender.recommend_tuples(tiny_df)
        
        # Should handle gracefully (no sampling for tiny datasets)
        assert isinstance(result, pd.DataFrame)
    
    def test_all_recommendation_modes(self, all_recommendation_modes, large_dataframe):
        """Test all recommendation output modes."""
        recommender = SamplingRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(large_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = SamplingRecommender(sample_config)
        assert recommender.name() == "SamplingRecommender"
    
    def test_config_without_sampling_section(self, large_dataframe):
        """Test with config missing sampling section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = SamplingRecommender(config)
        
        # Should use default sampling configuration
        result = recommender.recommend_tuples(large_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.sampling_method == 'stratified'  # default value
    
    def test_invalid_sampling_method(self, large_dataframe):
        """Test handling of invalid sampling method."""
        config = {
            'sampling': {'method': 'invalid_method'},
            'recommendation': {'top_k': 3}
        }
        recommender = SamplingRecommender(config)
        
        # Should handle invalid method gracefully (fall back to default)
        result = recommender.recommend_tuples(large_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_invalid_scoring_method(self, large_dataframe):
        """Test handling of invalid scoring method."""
        config = {
            'sampling': {'scoring_method': 'invalid_method'},
            'recommendation': {'top_k': 3}
        }
        recommender = SamplingRecommender(config)
        
        # Should handle invalid method gracefully
        result = recommender.recommend_tuples(large_dataframe)
        assert isinstance(result, pd.DataFrame)
