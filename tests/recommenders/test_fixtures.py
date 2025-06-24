"""
Test fixtures and utilities for recommender tests.
"""

import pandas as pd
import numpy as np
import pytest
from typing import Dict, Any


@pytest.fixture
def sample_config():
    """Basic configuration for recommenders."""
    return {
        'recommendation': {
            'mode': 'top_k',
            'top_k': 5,
            'percentage': 0.2
        },
        'clustering': {
            'n_clusters': 3,
            'random_state': 42,
            'max_iter': 300,
            'n_init': 10
        },
        'random': {
            'random_seed': 42
        },
        'discretization': {
            'enabled': True,
            'method': 'equal_width',
            'bins': 5,
            'save_params': False
        },
        'association_rules': {
            'min_support': 0.1,
            'metric': 'confidence',
            'min_threshold': 0.7
        },
        'summaries': {
            'desired_size': 5,
            'weights': None
        }
    }


@pytest.fixture
def simple_dataframe():
    """Simple DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1]
    })


@pytest.fixture 
def mixed_dataframe():
    """DataFrame with mixed data types for testing."""
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8],
        'categorical_col': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2', 'cat1', 'cat3', 'cat2'],
        'float_col': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        'string_col': ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta']
    })


@pytest.fixture
def large_dataframe():
    """Larger DataFrame for testing clustering and performance."""
    np.random.seed(42)
    n_rows = 100
    return pd.DataFrame({
        'id': range(n_rows),
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'value': np.random.uniform(0, 100, n_rows)
    })


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_row_dataframe():
    """Single row DataFrame for edge case testing."""
    return pd.DataFrame({
        'col1': [1],
        'col2': ['test'],
        'col3': [3.14]
    })


@pytest.fixture
def config_top_quartile():
    """Configuration for top quartile mode."""
    return {
        'recommendation': {
            'mode': 'top_quartile'
        }
    }


@pytest.fixture
def config_percentage():
    """Configuration for percentage mode."""
    return {
        'recommendation': {
            'mode': 'percentage',
            'percentage': 0.3
        }
    }


# Test data inspired by the existing COMPLEX_TRANSACTIONS
@pytest.fixture
def astronomy_dataframe():
    """DataFrame with astronomy-like data for testing."""
    return pd.DataFrame({
        'specobjid': ['obj_001', 'obj_002', 'obj_003', 'obj_004', 'obj_005', 'obj_006'],
        'specclass': ['class_1', 'class_2', 'class_1', 'class_3', 'class_2', 'class_1'],
        'z_bin': ['bin_1', 'bin_2', 'bin_1', 'bin_3', 'bin_2', 'bin_1'],
        'zconf_bin': ['conf_1', 'conf_2', 'conf_1', 'conf_3', 'conf_2', 'conf_1'],
        'primtarget': [4, 20, 4, 4, 20, 4]
    })


@pytest.fixture(params=[
    'top_k',
    'top_quartile', 
    'percentage'
])
def all_recommendation_modes(request):
    """Parametrized fixture for all recommendation modes."""
    if request.param == 'top_k':
        return {
            'recommendation': {
                'mode': 'top_k',
                'top_k': 3
            }
        }
    elif request.param == 'top_quartile':
        return {
            'recommendation': {
                'mode': 'top_quartile'
            }
        }
    elif request.param == 'percentage':
        return {
            'recommendation': {
                'mode': 'percentage',
                'percentage': 0.25
            }
        }
