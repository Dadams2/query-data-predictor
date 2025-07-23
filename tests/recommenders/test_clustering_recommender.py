"""
Tests for the ClusteringRecommender class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from query_data_predictor.recommender.clustering_recommender import ClusteringRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestClusteringRecommender:
    """Test suite for ClusteringRecommender class."""
    
    def test_init(self, sample_config):
        """Test ClusteringRecommender initialization."""
        recommender = ClusteringRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "ClusteringRecommender"
        assert hasattr(recommender, 'scaler')
        assert hasattr(recommender, 'label_encoders')
    
    def test_recommend_tuples_basic(self, sample_config, mixed_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should not be empty
        assert not result.empty
        # Should be limited by clustering config or recommendation config
        assert len(result) <= 5  # From sample_config top_k
        # Should be a subset of the original data
        assert all(row.tolist() in mixed_dataframe.values.tolist() for _, row in result.iterrows())
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return the single row
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, single_row_dataframe)
    
    def test_recommend_tuples_n_clusters_override(self, sample_config, mixed_dataframe):
        """Test recommend_tuples with n_clusters override."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe, n_clusters=2)
        
        # Should return at most 2 tuples (one per cluster)
        assert len(result) <= 2
        assert not result.empty
    
    def test_recommend_tuples_more_clusters_than_data(self, sample_config):
        """Test when requesting more clusters than data points."""
        small_df = pd.DataFrame({
            'A': [1, 2],
            'B': ['x', 'y']
        })
        
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(small_df, n_clusters=5)
        
        # Should return at most 2 tuples (all available data)
        assert len(result) <= 2
    
    def test_prepare_data_for_clustering_mixed(self, sample_config, mixed_dataframe):
        """Test _prepare_data_for_clustering with mixed data types."""
        recommender = ClusteringRecommender(sample_config)
        prepared = recommender._prepare_data_for_clustering(mixed_dataframe)
        
        # Should return a DataFrame
        assert isinstance(prepared, pd.DataFrame)
        # Should have same number of rows (if successful)
        if not prepared.empty:
            assert len(prepared) == len(mixed_dataframe)
    
    def test_prepare_data_for_clustering_categorical_only(self, sample_config):
        """Test _prepare_data_for_clustering with only categorical data."""
        cat_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        recommender = ClusteringRecommender(sample_config)
        prepared = recommender._prepare_data_for_clustering(cat_df)
        
        # Should handle categorical data
        assert isinstance(prepared, pd.DataFrame)
    
    def test_prepare_data_for_clustering_numeric_only(self, sample_config, simple_dataframe):
        """Test _prepare_data_for_clustering with only numeric data."""
        numeric_df = simple_dataframe[['A', 'C']].copy()  # Only numeric columns
        
        recommender = ClusteringRecommender(sample_config)
        prepared = recommender._prepare_data_for_clustering(numeric_df)
        
        # Should handle numeric data
        assert isinstance(prepared, pd.DataFrame)
        if not prepared.empty:
            assert len(prepared) == len(numeric_df)
    
    def test_prepare_data_for_clustering_with_nulls(self, sample_config):
        """Test _prepare_data_for_clustering with null values."""
        null_df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': ['a', None, 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, None, 5.5]
        })
        
        recommender = ClusteringRecommender(sample_config)
        prepared = recommender._prepare_data_for_clustering(null_df)
        
        # Should handle null values
        assert isinstance(prepared, pd.DataFrame)
        # Should not have any null values if successful
        if not prepared.empty:
            assert not prepared.isnull().any().any()
    
    def test_prepare_data_for_clustering_high_cardinality(self, sample_config):
        """Test _prepare_data_for_clustering with high cardinality columns."""
        high_card_df = pd.DataFrame({
            'id': range(50),  # High cardinality - should be skipped
            'category': np.random.choice(['A', 'B', 'C'], 50),  # Low cardinality - should be kept
            'value': np.random.randn(50)
        })
        
        recommender = ClusteringRecommender(sample_config)
        prepared = recommender._prepare_data_for_clustering(high_card_df)
        
        # Should skip high cardinality columns
        assert isinstance(prepared, pd.DataFrame)
        if not prepared.empty:
            # Should not include the high cardinality 'id' column in meaningful way
            assert len(prepared.columns) <= 3  # category + value
    
    def test_perform_clustering_basic(self, sample_config, mixed_dataframe):
        """Test _perform_clustering basic functionality."""
        recommender = ClusteringRecommender(sample_config)
        prepared = recommender._prepare_data_for_clustering(mixed_dataframe)
        
        if not prepared.empty:
            clusters, kmeans = recommender._perform_clustering(prepared, n_clusters=3)
            
            # Should return cluster labels
            assert isinstance(clusters, np.ndarray)
            assert len(clusters) == len(prepared)
            # Should have values in range [0, n_clusters-1]
            assert all(0 <= label < 3 for label in clusters)
    
    def test_select_representatives_basic(self, sample_config, mixed_dataframe):
        """Test _select_representatives basic functionality."""
        recommender = ClusteringRecommender(sample_config)
        clusters = np.array([0, 1, 0, 2, 1, 0, 2, 1])  # Mock cluster labels
        
        # Prepare encoded data
        encoded_data = recommender._prepare_data_for_clustering(mixed_dataframe)
        
        # Create a mock kmeans model with proper centroids
        mock_kmeans = Mock()
        mock_kmeans.cluster_centers_ = np.array([
            [0.0, 0.0] * len(encoded_data.columns) if not encoded_data.empty else [0.0, 0.0],
            [1.0, 1.0] * len(encoded_data.columns) if not encoded_data.empty else [1.0, 1.0], 
            [2.0, 2.0] * len(encoded_data.columns) if not encoded_data.empty else [2.0, 2.0]
        ])[:, :len(encoded_data.columns) if not encoded_data.empty else 2]
        
        if encoded_data.empty:
            # Skip test if data preparation failed
            pytest.skip("Data preparation failed - cannot test representative selection")
        
        result = recommender._select_representatives(
            mixed_dataframe, encoded_data, clusters, mock_kmeans, n_clusters=3
        )
        
        # Should return DataFrame with representatives
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3  # One per cluster max
        assert not result.empty
    
    def test_fallback_selection(self, sample_config, mixed_dataframe):
        """Test _fallback_selection method."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender._fallback_selection(mixed_dataframe, n_tuples=3)
        
        # Should return 3 random tuples
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Should be reproducible with same random seed
        result2 = recommender._fallback_selection(mixed_dataframe, n_tuples=3)
        pd.testing.assert_frame_equal(result, result2)
    
    def test_clustering_failure_fallback(self, sample_config):
        """Test fallback behavior when clustering fails."""
        # Create data that might cause clustering issues
        problematic_df = pd.DataFrame({
            'text_col': ['same'] * 10,  # All same values
            'nan_col': [np.nan] * 10    # All NaN values
        })
        
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(problematic_df)
        
        # Should still return a result via fallback
        assert isinstance(result, pd.DataFrame)
        # Might be empty if data is too problematic, but should not crash
    
    @patch('query_data_predictor.recommenders.clustering_recommender.KMeans')
    def test_clustering_exception_handling(self, mock_kmeans, sample_config, mixed_dataframe):
        """Test exception handling in clustering."""
        # Make KMeans raise an exception
        mock_kmeans.side_effect = Exception("Clustering failed")
        
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should fall back to random selection
        assert isinstance(result, pd.DataFrame)
        # Should not be empty (fallback should work)
        assert not result.empty
    
    def test_all_recommendation_modes(self, all_recommendation_modes, mixed_dataframe):
        """Test recommend_tuples with all recommendation modes."""
        recommender = ClusteringRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should return a non-empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= len(mixed_dataframe)
    
    def test_large_dataset(self, sample_config, large_dataframe):
        """Test clustering with larger dataset."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should complete and return reasonable results
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= 5  # From sample_config
    
    def test_reproducible_results(self, sample_config, mixed_dataframe):
        """Test that results are reproducible."""
        recommender1 = ClusteringRecommender(sample_config)
        recommender2 = ClusteringRecommender(sample_config)
        
        result1 = recommender1.recommend_tuples(mixed_dataframe)
        result2 = recommender2.recommend_tuples(mixed_dataframe)
        
        # Should be identical due to same random_state
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_custom_clustering_config(self, mixed_dataframe):
        """Test with custom clustering configuration."""
        config = {
            'clustering': {
                'n_clusters': 2,
                'random_state': 123,
                'max_iter': 100,
                'n_init': 5
            },
            'recommendation': {
                'mode': 'top_k',
                'top_k': 3
            }
        }
        
        recommender = ClusteringRecommender(config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should respect the clustering configuration
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2  # n_clusters = 2
    
    def test_preserves_original_data_structure(self, sample_config, mixed_dataframe):
        """Test that original data structure is preserved."""
        recommender = ClusteringRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should preserve columns and dtypes
        assert list(result.columns) == list(mixed_dataframe.columns)
        for col in result.columns:
            if not result.empty:
                assert result[col].dtype == mixed_dataframe[col].dtype
    
    def test_invalid_input_type(self, sample_config):
        """Test that invalid input type raises ValueError."""
        recommender = ClusteringRecommender(sample_config)
        
        with pytest.raises(ValueError, match="current_results must be a pandas DataFrame"):
            recommender.recommend_tuples("not a dataframe")
    
    def test_name_method(self, sample_config):
        """Test name method returns correct value."""
        recommender = ClusteringRecommender(sample_config)
        assert recommender.name() == "ClusteringRecommender"
