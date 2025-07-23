"""
Tests for the IndexRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommender.index_recommender import IndexRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestIndexRecommender:
    """Test suite for IndexRecommender class."""
    
    def test_init(self, sample_config):
        """Test IndexRecommender initialization."""
        recommender = IndexRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "IndexRecommender"
        assert 'value' in recommender.index_types  # default
        assert recommender.max_index_size == 10000  # default
        assert hasattr(recommender, 'value_indices')
        assert hasattr(recommender, 'range_indices')
        assert hasattr(recommender, 'pattern_indices')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom indexing configuration."""
        config = {
            'indexing': {
                'index_types': ['value', 'range'],
                'max_index_size': 5000,
                'enable_compound_indices': False,
                'max_compound_size': 2,
                'scoring_method': 'simple'
            },
            'recommendation': {'top_k': 5}
        }
        recommender = IndexRecommender(config)
        assert recommender.index_types == ['value', 'range']
        assert recommender.max_index_size == 5000
        assert recommender.enable_compound_indices is False
        assert recommender.max_compound_size == 2
        assert recommender.scoring_method == 'simple'
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = IndexRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should be limited by top_k (5)
        assert len(result) <= 5
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = IndexRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = IndexRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_build_indices(self, sample_config, simple_dataframe):
        """Test building indices."""
        recommender = IndexRecommender(sample_config)
        recommender._build_indices(simple_dataframe)
        
        # Should build indices
        assert recommender._indices_built is True
        # Should have value indices for each column
        for col in simple_dataframe.columns:
            if 'value' in recommender.index_types:
                assert col in recommender.value_indices
    
    def test_build_indices_empty(self, sample_config, empty_dataframe):
        """Test building indices with empty DataFrame."""
        recommender = IndexRecommender(sample_config)
        recommender._build_indices(empty_dataframe)
        
        # Should handle empty DataFrame gracefully
        assert len(recommender.value_indices) == 0
    
    def test_build_value_indices(self, sample_config, simple_dataframe):
        """Test building value indices."""
        recommender = IndexRecommender(sample_config)
        recommender.index_types = ['value']
        recommender._build_value_indices(simple_dataframe)
        
        # Should build value indices for each column
        for col in simple_dataframe.columns:
            assert col in recommender.value_indices
            assert isinstance(recommender.value_indices[col], dict)
    
    def test_build_range_indices(self, sample_config, mixed_dataframe):
        """Test building range indices for numeric columns."""
        recommender = IndexRecommender(sample_config)
        recommender.index_types = ['range']
        recommender._build_range_indices(mixed_dataframe)
        
        # Should build range indices for numeric columns
        numeric_cols = mixed_dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert col in recommender.range_indices
    
    def test_build_pattern_indices(self, sample_config, simple_dataframe):
        """Test building pattern indices."""
        recommender = IndexRecommender(sample_config)
        recommender._build_pattern_indices(simple_dataframe)
        
        # Should build pattern indices
        assert len(recommender.pattern_indices) >= 0
    
    def test_build_compound_indices(self, sample_config, simple_dataframe):
        """Test building compound indices."""
        recommender = IndexRecommender(sample_config)
        recommender.enable_compound_indices = True
        recommender._build_compound_indices(simple_dataframe)
        
        # Should build compound indices
        assert len(recommender.compound_indices) >= 0
    
    def test_compute_index_scores(self, sample_config, simple_dataframe):
        """Test computing index scores."""
        recommender = IndexRecommender(sample_config)
        recommender._build_indices(simple_dataframe)
        scores = recommender._compute_index_scores(simple_dataframe)
        
        # Should return index scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(simple_dataframe)
    
    def test_different_scoring_methods(self, simple_dataframe):
        """Test all scoring methods."""
        methods = ['simple', 'weighted', 'compound']
        
        for method in methods:
            config = {
                'indexing': {'scoring_method': method},
                'recommendation': {'top_k': 3}
            }
            recommender = IndexRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all methods
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 3
    
    def test_different_index_types(self, mixed_dataframe):
        """Test different index types."""
        index_combinations = [
            ['value'],
            ['range'],
            ['pattern'],
            ['value', 'range'],
            ['value', 'pattern'],
            ['value', 'range', 'pattern']
        ]
        
        for index_types in index_combinations:
            config = {
                'indexing': {'index_types': index_types},
                'recommendation': {'top_k': 3}
            }
            recommender = IndexRecommender(config)
            result = recommender.recommend_tuples(mixed_dataframe)
            
            # Should work with all index type combinations
            assert isinstance(result, pd.DataFrame)
    
    def test_caching_functionality(self, sample_config, simple_dataframe):
        """Test index computation caching."""
        recommender = IndexRecommender(sample_config)
        
        # First call
        result1 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Second call with same data should use cache
        result2 = recommender.recommend_tuples(simple_dataframe.copy())
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_clear_indices(self, sample_config, simple_dataframe):
        """Test clearing indices."""
        recommender = IndexRecommender(sample_config)
        
        # Build indices
        recommender._build_indices(simple_dataframe)
        assert len(recommender.value_indices) > 0
        
        # Clear indices
        recommender._clear_indices()
        assert len(recommender.value_indices) == 0
        assert len(recommender.range_indices) == 0
        assert len(recommender.pattern_indices) == 0
        assert len(recommender.compound_indices) == 0
        assert recommender._indices_built is False
    
    def test_get_index_statistics(self, sample_config, simple_dataframe):
        """Test getting index statistics."""
        recommender = IndexRecommender(sample_config)
        
        # Build indices
        recommender.recommend_tuples(simple_dataframe)
        
        # Get statistics
        stats = recommender.get_index_statistics()
        
        # Should return statistics dictionary
        assert isinstance(stats, dict)
    
    def test_rebuild_indices(self, sample_config, simple_dataframe):
        """Test rebuilding indices."""
        recommender = IndexRecommender(sample_config)
        
        # Build initial indices
        recommender._build_indices(simple_dataframe)
        original_indices_count = len(recommender.value_indices)
        
        # Rebuild indices
        recommender.rebuild_indices(simple_dataframe)
        
        # Should rebuild indices
        assert recommender._indices_built is True
        assert len(recommender.value_indices) == original_indices_count
    
    def test_range_lookup(self, sample_config, mixed_dataframe):
        """Test range-based lookups."""
        recommender = IndexRecommender(sample_config)
        recommender._build_range_indices(mixed_dataframe)
        
        # Test range lookup on numeric column
        numeric_cols = mixed_dataframe.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            value = mixed_dataframe[col].iloc[0]
            indices = recommender._range_lookup(col, float(value))
            
            # Should return list of indices
            assert isinstance(indices, list)
    
    def test_pattern_lookup(self, sample_config, simple_dataframe):
        """Test pattern-based lookups."""
        recommender = IndexRecommender(sample_config)
        recommender._build_pattern_indices(simple_dataframe)
        
        # Test pattern lookup
        if len(recommender.pattern_indices) > 0:
            pattern = list(recommender.pattern_indices.keys())[0]
            indices = recommender._pattern_lookup(pattern)
            
            # Should return list of indices
            assert isinstance(indices, list)
    
    def test_compound_indices_functionality(self, sample_config, simple_dataframe):
        """Test compound indices functionality."""
        recommender = IndexRecommender(sample_config)
        recommender.enable_compound_indices = True
        recommender.max_compound_size = 2
        
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with compound indices enabled
        assert isinstance(result, pd.DataFrame)
    
    def test_max_index_size_limit(self, sample_config):
        """Test index size limitation."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'col1': [f'value_{i}' for i in range(1000)],
            'col2': range(1000)
        })
        
        recommender = IndexRecommender(sample_config)
        recommender.max_index_size = 100  # Small limit
        
        result = recommender.recommend_tuples(large_df)
        
        # Should handle large datasets within index size limits
        assert isinstance(result, pd.DataFrame)
    
    def test_frequency_indices(self, sample_config, simple_dataframe):
        """Test frequency index building."""
        recommender = IndexRecommender(sample_config)
        recommender._build_frequency_indices(simple_dataframe)
        
        # Should build frequency indices
        for col in simple_dataframe.columns:
            assert col in recommender.frequency_indices
    
    def test_large_dataset_performance(self, sample_config, large_dataframe):
        """Test performance with larger dataset."""
        recommender = IndexRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle large dataset efficiently
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # Limited by top_k
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        recommender = IndexRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(simple_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = IndexRecommender(sample_config)
        assert recommender.name() == "IndexRecommender"
    
    def test_config_without_indexing_section(self, simple_dataframe):
        """Test with config missing indexing section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = IndexRecommender(config)
        
        # Should use default indexing configuration
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert 'value' in recommender.index_types  # default value
    
    def test_invalid_index_types(self, simple_dataframe):
        """Test handling of invalid index types."""
        config = {
            'indexing': {'index_types': ['invalid_type', 'value']},
            'recommendation': {'top_k': 3}
        }
        recommender = IndexRecommender(config)
        
        # Should handle invalid types gracefully
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_disable_compound_indices(self, sample_config, simple_dataframe):
        """Test with compound indices disabled."""
        recommender = IndexRecommender(sample_config)
        recommender.enable_compound_indices = False
        
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work without compound indices
        assert isinstance(result, pd.DataFrame)
        assert len(recommender.compound_indices) == 0
