"""
Tests for the AssociationRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from query_data_predictor.recommenders.association_recommender import AssociationRecommender
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestAssociationRecommender:
    """Test suite for AssociationRecommender class."""
    
    def test_init(self, sample_config):
        """Test AssociationRecommender initialization."""
        recommender = AssociationRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "AssociationRecommender"
        assert recommender.min_support == 0.1  # from sample_config
        assert recommender.min_confidence == 0.7  # from sample_config
        assert recommender.metric == 'confidence'
        assert recommender.max_unique_values == 15  # default
    
    def test_init_with_custom_config(self):
        """Test initialization with custom association configuration."""
        config = {
            'association_rules': {
                'min_support': 0.05,
                'min_threshold': 0.6,
                'metric': 'lift',
                'max_unique_values': 20
            },
            'recommendation': {'top_k': 5}
        }
        recommender = AssociationRecommender(config)
        assert recommender.min_support == 0.05
        assert recommender.min_confidence == 0.6
        assert recommender.metric == 'lift'
        assert recommender.max_unique_values == 20
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should have same columns as input
        assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(single_row_dataframe)
        
        # Should return empty DataFrame (need multiple rows for association rules)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_insufficient_data(self, sample_config):
        """Test with insufficient data for association rules."""
        # Create DataFrame with only 1 row
        small_df = pd.DataFrame({'A': [1], 'B': ['a']})
        
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(small_df)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_limit_unique_values(self, sample_config):
        """Test limiting unique values to prevent combinatorial explosion."""
        # Create DataFrame with many unique values
        large_unique_df = pd.DataFrame({
            'col1': [f'value_{i}' for i in range(100)],
            'col2': range(100)
        })
        
        recommender = AssociationRecommender(sample_config)
        limited_df = recommender._limit_unique_values(large_unique_df)
        
        # Should limit unique values per column
        assert isinstance(limited_df, pd.DataFrame)
        for col in limited_df.columns:
            unique_count = limited_df[col].nunique()
            assert unique_count <= recommender.max_unique_values
    
    def test_mine_rules_from_history(self, sample_config):
        """Test mining association rules from query history."""
        # Create DataFrame with repeated patterns for association rules
        pattern_df = pd.DataFrame({
            'A': ['a1', 'a1', 'a2', 'a1', 'a2', 'a1'],
            'B': ['b1', 'b2', 'b1', 'b2', 'b1', 'b2'],
            'C': ['c1', 'c1', 'c2', 'c1', 'c2', 'c1']
        })
        
        recommender = AssociationRecommender(sample_config)
        rules = recommender._mine_rules_from_history(pattern_df)
        
        # Should return DataFrame (might be empty if no strong rules found)
        assert isinstance(rules, pd.DataFrame)
    
    def test_df_to_query_sequence(self, sample_config, simple_dataframe):
        """Test converting DataFrame to query sequence."""
        recommender = AssociationRecommender(sample_config)
        sequence = recommender._df_to_query_sequence(simple_dataframe)
        
        # Should return list of sets
        assert isinstance(sequence, list)
        assert len(sequence) == len(simple_dataframe)
        for item_set in sequence:
            assert isinstance(item_set, list)
    
    def test_query_sequence_to_df(self, sample_config, simple_dataframe):
        """Test converting query sequence back to DataFrame."""
        recommender = AssociationRecommender(sample_config)
        
        # Convert to sequence and back
        sequence = recommender._df_to_query_sequence(simple_dataframe)
        reconstructed = recommender._query_sequence_to_df(sequence, simple_dataframe.columns)
        
        # Should reconstruct DataFrame structure
        assert isinstance(reconstructed, pd.DataFrame)
        assert list(reconstructed.columns) == list(simple_dataframe.columns)
    
    def test_different_metrics(self, simple_dataframe):
        """Test different association rule metrics."""
        metrics = ['confidence', 'lift', 'leverage', 'conviction']
        
        for metric in metrics:
            config = {
                'association_rules': {
                    'min_support': 0.1,
                    'metric': metric,
                    'min_threshold': 0.5
                },
                'recommendation': {'top_k': 3}
            }
            recommender = AssociationRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all metrics (might return empty if no rules found)
            assert isinstance(result, pd.DataFrame)
    
    def test_different_support_thresholds(self, simple_dataframe):
        """Test different minimum support thresholds."""
        support_values = [0.05, 0.1, 0.2, 0.5]
        
        for support in support_values:
            config = {
                'association_rules': {
                    'min_support': support,
                    'metric': 'confidence',
                    'min_threshold': 0.5
                },
                'recommendation': {'top_k': 3}
            }
            recommender = AssociationRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all support thresholds
            assert isinstance(result, pd.DataFrame)
    
    def test_different_confidence_thresholds(self, simple_dataframe):
        """Test different confidence thresholds."""
        confidence_values = [0.3, 0.5, 0.7, 0.9]
        
        for confidence in confidence_values:
            config = {
                'association_rules': {
                    'min_support': 0.1,
                    'metric': 'confidence',
                    'min_threshold': confidence
                },
                'recommendation': {'top_k': 3}
            }
            recommender = AssociationRecommender(config)
            result = recommender.recommend_tuples(simple_dataframe)
            
            # Should work with all confidence thresholds
            assert isinstance(result, pd.DataFrame)
    
    def test_with_repeated_patterns(self, sample_config):
        """Test with data containing repeated patterns."""
        # Create data with clear patterns for association rules
        repeated_df = pd.DataFrame({
            'item1': ['A', 'A', 'B', 'A', 'B', 'A'] * 5,
            'item2': ['X', 'Y', 'X', 'Y', 'X', 'Y'] * 5,
            'item3': ['1', '1', '2', '1', '2', '1'] * 5
        })
        
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(repeated_df)
        
        # Should find patterns in repeated data
        assert isinstance(result, pd.DataFrame)
    
    def test_categorical_data_handling(self, sample_config):
        """Test handling of categorical data."""
        cat_df = pd.DataFrame({
            'category1': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2', 'cat1'],
            'category2': ['type_a', 'type_b', 'type_a', 'type_c', 'type_b', 'type_a'],
            'category3': ['group_x', 'group_y', 'group_x', 'group_z', 'group_y', 'group_x']
        })
        
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(cat_df)
        
        # Should handle categorical data properly
        assert isinstance(result, pd.DataFrame)
    
    def test_mixed_data_types(self, sample_config, mixed_dataframe):
        """Test handling of mixed data types."""
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(mixed_dataframe)
        
        # Should handle mixed data types
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(mixed_dataframe.columns)
    
    def test_large_dataset_performance(self, sample_config, large_dataframe):
        """Test performance with larger dataset."""
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(large_dataframe)
        
        # Should handle large dataset efficiently
        assert isinstance(result, pd.DataFrame)
    
    def test_no_frequent_itemsets(self, sample_config):
        """Test behavior when no frequent itemsets are found."""
        # Create sparse data unlikely to have frequent patterns
        sparse_df = pd.DataFrame({
            'col1': [f'unique_{i}' for i in range(20)],
            'col2': [f'value_{i}' for i in range(20)],
            'col3': [f'item_{i}' for i in range(20)]
        })
        
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(sparse_df)
        
        # Should return empty DataFrame when no patterns found
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_high_support_threshold(self, sample_config, simple_dataframe):
        """Test with very high support threshold."""
        recommender = AssociationRecommender(sample_config)
        recommender.min_support = 0.9  # Very high threshold
        
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should return empty or minimal results with high threshold
        assert isinstance(result, pd.DataFrame)
    
    def test_low_support_threshold(self, sample_config, simple_dataframe):
        """Test with very low support threshold."""
        recommender = AssociationRecommender(sample_config)
        recommender.min_support = 0.01  # Very low threshold
        
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with low threshold (may find more rules)
        assert isinstance(result, pd.DataFrame)
    
    def test_max_unique_values_setting(self, sample_config):
        """Test different max_unique_values settings."""
        values = [5, 10, 20, 50]
        
        for max_val in values:
            recommender = AssociationRecommender(sample_config)
            recommender.max_unique_values = max_val
            
            # Create test data
            test_df = pd.DataFrame({
                'col1': [f'val_{i}' for i in range(30)],
                'col2': range(30)
            })
            
            limited = recommender._limit_unique_values(test_df)
            
            # Should respect max_unique_values limit
            for col in limited.columns:
                assert limited[col].nunique() <= max_val
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        recommender = AssociationRecommender(all_recommendation_modes)
        result = recommender.recommend_tuples(simple_dataframe)
        
        # Should work with all recommendation modes
        assert isinstance(result, pd.DataFrame)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = AssociationRecommender(sample_config)
        assert recommender.name() == "AssociationRecommender"
    
    def test_config_without_association_section(self, simple_dataframe):
        """Test with config missing association_rules section."""
        config = {'recommendation': {'top_k': 3}}
        recommender = AssociationRecommender(config)
        
        # Should use default association configuration
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert recommender.min_support == 0.1  # default value
        assert recommender.min_confidence == 0.5  # default value
    
    def test_invalid_metric(self, simple_dataframe):
        """Test handling of invalid metric."""
        config = {
            'association_rules': {'metric': 'invalid_metric'},
            'recommendation': {'top_k': 3}
        }
        recommender = AssociationRecommender(config)
        
        # Should handle invalid metric gracefully
        result = recommender.recommend_tuples(simple_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_transaction_encoding_error_handling(self, sample_config):
        """Test handling of transaction encoding errors."""
        # Create DataFrame that might cause encoding issues
        problematic_df = pd.DataFrame({
            'col1': [None, 'a', 'b'],
            'col2': ['x', None, 'y'],
            'col3': ['1', '2', None]
        })
        
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(problematic_df)
        
        # Should handle encoding issues gracefully
        assert isinstance(result, pd.DataFrame)
    
    def test_fpgrowth_empty_transactions(self, sample_config):
        """Test FP-Growth with empty transactions."""
        empty_transactions_df = pd.DataFrame({
            'col1': ['', '', ''],
            'col2': ['', '', '']
        })
        
        recommender = AssociationRecommender(sample_config)
        result = recommender.recommend_tuples(empty_transactions_df)
        
        # Should handle empty transactions gracefully
        assert isinstance(result, pd.DataFrame)
        assert result.empty  # Should return empty result
