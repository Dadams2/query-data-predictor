"""
Tests for the Multi-Dimensional Interestingness Recommender.
"""

import pandas as pd
import pytest
import numpy as np

from query_data_predictor.recommender import MultiDimensionalInterestingnessRecommender


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        'multidimensional_interestingness': {
            'alpha': 0.4,
            'beta': 0.4,
            'gamma': 0.2,
            'association_weights': {
                'confidence': 0.3,
                'support': 0.2,
                'lift': 0.3,
                'j_measure': 0.2
            },
            'diversity_weights': {
                'shannon': 0.25,
                'simpson': 0.20,
                'gini': 0.20,
                'berger': 0.15,
                'mcintosh': 0.20
            },
            'rule_decay_rate': 0.1,
            'summary_decay_rate': 0.1
        },
        'association_rules': {
            'enabled': True,
            'min_support': 0.1,
            'min_threshold': 0.5,
            'metric': 'confidence'
        },
        'summaries': {
            'enabled': True,
            'desired_size': 5
        },
        'discretization': {
            'enabled': True,
            'method': 'equal_width',
            'bins': 3,
            'save_params': False
        },
        'recommendation': {
            'mode': 'top_k',
            'top_k': 5
        }
    }


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'col1': np.random.randint(1, 10, 20),
        'col2': np.random.choice(['A', 'B', 'C'], 20),
        'col3': np.random.randn(20) * 10,
    })


def test_initialization(basic_config):
    """Test recommender initialization."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    assert recommender.alpha == 0.4
    assert recommender.beta == 0.4
    assert recommender.gamma == 0.2
    assert recommender.lambda_1 == 0.3
    assert recommender.mu_1 == 0.25


def test_recommend_tuples_basic(basic_config, sample_data):
    """Test basic recommendation functionality."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    result = recommender.recommend_tuples(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 5  # top_k = 5
    assert len(result) > 0


def test_recommend_tuples_with_top_k(basic_config, sample_data):
    """Test recommendation with explicit top_k parameter."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    result = recommender.recommend_tuples(sample_data, top_k=3)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 3


def test_empty_dataframe(basic_config):
    """Test handling of empty DataFrame."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    empty_df = pd.DataFrame()
    result = recommender.recommend_tuples(empty_df)
    
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_single_row(basic_config):
    """Test handling of single row DataFrame."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    single_row = pd.DataFrame({'col1': [1], 'col2': ['A']})
    result = recommender.recommend_tuples(single_row)
    
    assert isinstance(result, pd.DataFrame)
    # Should return empty as we need at least 2 rows
    assert result.empty


def test_historical_memory(basic_config, sample_data):
    """Test that historical memory is maintained."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    # First query
    result1 = recommender.recommend_tuples(sample_data)
    stats1 = recommender.get_history_stats()
    
    assert stats1['total_tuples'] == 20
    assert stats1['session_counter'] == 1
    
    # Second query
    result2 = recommender.recommend_tuples(sample_data)
    stats2 = recommender.get_history_stats()
    
    assert stats2['total_tuples'] == 40  # 20 + 20
    assert stats2['session_counter'] == 2


def test_novelty_detection(basic_config):
    """Test novelty detection with repeated vs new values."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    # First query with repeated values
    df1 = pd.DataFrame({
        'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'col2': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
    })
    recommender.recommend_tuples(df1)
    
    # Second query with mix of repeated and novel values
    df2 = pd.DataFrame({
        'col1': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4],  # More novel values
        'col2': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'D']
    })
    result = recommender.recommend_tuples(df2)
    
    # Novel values should be ranked higher
    # Check that at least some results were returned
    assert len(result) > 0


def test_temporal_decay(basic_config, sample_data):
    """Test temporal decay - recent patterns should have more weight."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    # Query 1
    recommender.recommend_tuples(sample_data)
    
    # Query 2 (age = 1 for previous patterns)
    recommender.recommend_tuples(sample_data)
    
    # Query 3 (age = 2 for first patterns, age = 1 for second)
    result = recommender.recommend_tuples(sample_data)
    
    stats = recommender.get_history_stats()
    assert stats['session_counter'] == 3
    # Rules and summaries should have been accumulated
    assert stats['total_rules'] > 0 or stats['total_summaries'] > 0


def test_clear_history(basic_config, sample_data):
    """Test clearing historical memory."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    # Build up some history
    recommender.recommend_tuples(sample_data)
    recommender.recommend_tuples(sample_data)
    
    stats_before = recommender.get_history_stats()
    assert stats_before['total_tuples'] > 0
    
    # Clear history
    recommender.clear_history()
    
    stats_after = recommender.get_history_stats()
    assert stats_after['total_tuples'] == 0
    assert stats_after['session_counter'] == 0


def test_association_component_updates_scores_with_boolean_mask(basic_config):
    """Test score updates use aligned boolean indexing safely."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)

    encoded_df = pd.DataFrame(
        {
            'col1_A': [True, False, True],
            'col2_B': [True, True, False],
        }
    )
    association_rules = pd.DataFrame(
        [
            {
                'antecedents': frozenset({'col1_A'}),
                'consequents': frozenset({'col2_B'}),
                'confidence': 1.0,
                'support': 0.5,
                'lift': 1.2,
                'leverage': 0.1,
            }
        ]
    )

    scores = recommender._compute_association_component(
        encoded_df=encoded_df,
        association_rules=association_rules,
        current_timestamp=1,
    )

    assert list(scores.index) == list(encoded_df.index)
    assert scores.iloc[0] > 0
    assert scores.iloc[1] == 0
    assert scores.iloc[2] == 0


def test_config_weights_sum():
    """Test that configuration weight sums are reasonable."""
    config = {
        'multidimensional_interestingness': {
            'alpha': 0.4,
            'beta': 0.4,
            'gamma': 0.2,
            'association_weights': {
                'confidence': 0.25,
                'support': 0.25,
                'lift': 0.25,
                'j_measure': 0.25
            },
            'diversity_weights': {
                'shannon': 0.2,
                'simpson': 0.2,
                'gini': 0.2,
                'berger': 0.2,
                'mcintosh': 0.2
            }
        },
        'association_rules': {'min_support': 0.1, 'min_threshold': 0.5},
        'summaries': {'desired_size': 5},
        'discretization': {'enabled': False},
        'recommendation': {'mode': 'top_k', 'top_k': 5}
    }
    
    recommender = MultiDimensionalInterestingnessRecommender(config)
    
    # Check main weights
    main_sum = recommender.alpha + recommender.beta + recommender.gamma
    assert 0.99 <= main_sum <= 1.01  # Allow small floating point errors
    
    # Check association weights
    assoc_sum = (recommender.lambda_1 + recommender.lambda_2 + 
                 recommender.lambda_3 + recommender.lambda_4)
    assert 0.99 <= assoc_sum <= 1.01
    
    # Check diversity weights
    div_sum = (recommender.mu_1 + recommender.mu_2 + recommender.mu_3 + 
               recommender.mu_4 + recommender.mu_5)
    assert 0.99 <= div_sum <= 1.01


def test_name(basic_config):
    """Test recommender name method."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    assert recommender.name() == "MultiDimensionalInterestingnessRecommender"


def test_invalid_input(basic_config):
    """Test error handling with invalid input."""
    recommender = MultiDimensionalInterestingnessRecommender(basic_config)
    
    with pytest.raises(ValueError):
        recommender.recommend_tuples("not a dataframe")


def test_discretization_disabled():
    """Test recommender with discretization disabled."""
    config = {
        'multidimensional_interestingness': {
            'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2,
            'association_weights': {'confidence': 0.3, 'support': 0.2, 'lift': 0.3, 'j_measure': 0.2},
            'diversity_weights': {'shannon': 0.2, 'simpson': 0.2, 'gini': 0.2, 'berger': 0.2, 'mcintosh': 0.2}
        },
        'association_rules': {'min_support': 0.1, 'min_threshold': 0.5},
        'summaries': {'desired_size': 5},
        'discretization': {'enabled': False},
        'recommendation': {'mode': 'top_k', 'top_k': 5}
    }
    
    recommender = MultiDimensionalInterestingnessRecommender(config)
    
    df = pd.DataFrame({
        'col1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'col2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X']
    })
    
    result = recommender.recommend_tuples(df)
    assert isinstance(result, pd.DataFrame)
