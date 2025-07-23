from fixtures import sample_config
from query_data_predictor.recommender.tuple_recommender import TupleRecommender
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Fixture for sample DataFrame data."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8],  # Integer column - won't be discretized
        'B': [10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8],  # Float column - will be discretized
        'C': ['a', 'b', 'a', 'b', 'c', 'a', 'b', 'c']  # String column - won't be discretized
    })


@pytest.fixture
def enhanced_config():
    """Enhanced configuration for testing all features."""
    return {
        "discretization": {
            "enabled": True,
            "method": "equal_width",
            "bins": 3,
            "save_params": False
        },
        "association_rules": {
            "enabled": True,
            "min_support": 0.1,
            "metric": "confidence", 
            "min_threshold": 0.3
        },
        "summaries": {
            "enabled": True,
            "desired_size": 3,
            "weights": None
        },
        "interestingness": {
            "enabled": True,
            "measures": ["variance", "simpson", "shannon"]
        },
        "recommendation": {
            "enabled": True,
            "method": "hybrid",
            "top_k": 5,
            "score_threshold": 0.0
        }
    }

class TestTupleRecommender:
    """Test suite for the tuple recommender."""
    
    @pytest.fixture
    def recommender(self, sample_config):
        """Fixture for TupleRecommender instance with basic config."""
        return TupleRecommender(sample_config)
    
    @pytest.fixture  
    def enhanced_recommender(self, enhanced_config):
        """Fixture for TupleRecommender instance with enhanced config."""
        return TupleRecommender(enhanced_config)
    
    def test_initialization(self, sample_config):
        """Test recommender initialization."""
        recommender = TupleRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.discretizer is not None  # Should be initialized
        assert recommender.association_evaluator is None  # Not initialized yet
        assert recommender.summary_evaluator is None  # Not initialized yet
    
    def test_preprocess_data(self, recommender, sample_data):
        """Test data preprocessing with discretization."""
        processed = recommender.preprocess_data(sample_data.copy())        

        # Check that data has been processed
        assert processed is not None
        assert len(processed) == len(sample_data)
        
        # Check that float columns have been discretized (original columns dropped, _bin columns added)
        original_float_cols = sample_data.select_dtypes(include=[np.float64]).columns
        for col in original_float_cols:
            assert col not in processed.columns, f"Original float column {col} should be removed"
            assert f"{col}_bin" in processed.columns, f"Discretized column {col}_bin should be present"
        
        # Check that non-float columns remain unchanged
        non_float_cols = sample_data.select_dtypes(exclude=[np.float64]).columns
        for col in non_float_cols:
            assert col in processed.columns, f"Non-float column {col} should remain"
            pd.testing.assert_series_equal(processed[col], sample_data[col], f"Non-float column {col} values should be unchanged")
    
    def test_preprocess_data_disabled(self, sample_data):
        """Test preprocessing when discretization is disabled."""
        config = {
            "discretization": {"enabled": False},
            "recommendation": {"top_k": 5}
        }
        recommender = TupleRecommender(config)
        processed = recommender.preprocess_data(sample_data.copy())
        
        # Should return the same data when discretization is disabled
        pd.testing.assert_frame_equal(processed, sample_data)
    
    def test_prepend_column_names(self, recommender):
        """Test prepending column names to values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        result = recommender.prepend_column_names(df.copy())
        
        # Check that values are prepended with column names
        assert result['col1'][0] == 'col1_1'
        assert result['col1'][1] == 'col1_2'
        assert result['col2'][0] == 'col2_a'
        assert result['col2'][1] == 'col2_b'
    
    def test_prepend_column_names_empty_df(self, recommender):
        """Test prepending column names with empty DataFrame."""
        df = pd.DataFrame()
        result = recommender.prepend_column_names(df.copy())
        
        # Result should be empty DataFrame
        assert result.empty
    
    def test_prepare_data_for_fpgrowth(self, recommender, sample_data):
        """Test data preparation for FP-Growth algorithm."""
        processed_data = recommender.preprocess_data(sample_data)
        encoded_df, attributes = recommender._prepare_data_for_fpgrowth(processed_data)
        
        # Check that encoding was successful
        assert not encoded_df.empty
        assert len(attributes) == len(sample_data.columns)
        assert attributes == list(sample_data.columns)
        
        # Check that the encoded dataframe has boolean values only
        assert encoded_df.dtypes.apply(lambda x: x == bool).all()
    
    def test_compute_frequent_itemsets(self, enhanced_recommender, sample_data):
        """Test frequent itemset computation."""
        processed_data = enhanced_recommender.preprocess_data(sample_data)
        frequent_itemsets = enhanced_recommender.compute_frequent_itemsets(processed_data)
        
        # Check that itemsets were computed (may be empty with small sample)
        assert isinstance(frequent_itemsets, pd.DataFrame)
    
    def test_recommend_tuples_basic(self, recommender, sample_data):
        """Test basic tuple recommendation."""
        recommendations = recommender.recommend_tuples(sample_data, top_k=3)
        
        # Basic validation of the recommendations
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 3  # Should not exceed top_k
        assert len(recommendations) <= len(sample_data)  # Should not exceed input size
    
    def test_recommend_tuples_association_rules(self, enhanced_config, sample_data):
        """Test tuple recommendation using association rules only."""
        enhanced_config['recommendation']['method'] = 'association_rules'
        recommender = TupleRecommender(enhanced_config)
        
        recommendations = recommender.recommend_tuples(sample_data, top_k=3)
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 3
    
    def test_recommend_tuples_summaries(self, enhanced_config, sample_data):
        """Test tuple recommendation using summaries only."""
        enhanced_config['recommendation']['method'] = 'summaries'
        recommender = TupleRecommender(enhanced_config)
        
        recommendations = recommender.recommend_tuples(sample_data, top_k=3)
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 3
    
    def test_recommend_tuples_hybrid(self, enhanced_recommender, sample_data):
        """Test hybrid tuple recommendation."""
        recommendations = enhanced_recommender.recommend_tuples(sample_data, top_k=5)
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 5
    
    def test_recommend_tuples_empty_data(self, recommender):
        """Test recommendation with empty DataFrame."""
        empty_df = pd.DataFrame()
        recommendations = recommender.recommend_tuples(empty_df)
        
        assert isinstance(recommendations, pd.DataFrame)
        assert recommendations.empty
    
    def test_recommend_tuples_insufficient_data(self, recommender):
        """Test recommendation with insufficient data (less than 2 rows)."""
        small_df = pd.DataFrame({'A': [1], 'B': [2]})
        recommendations = recommender.recommend_tuples(small_df)
        
        assert isinstance(recommendations, pd.DataFrame)
        assert recommendations.empty
    
    def test_compute_tuple_interestingness_scores(self, enhanced_recommender, sample_data):
        """Test computation of interestingness scores."""
        processed_data = enhanced_recommender.preprocess_data(sample_data)
        frequent_itemsets = enhanced_recommender.compute_frequent_itemsets(processed_data)
        
        if not frequent_itemsets.empty:
            scores = enhanced_recommender._compute_tuple_interestingness_scores(
                processed_data, frequent_itemsets
            )
            
            assert isinstance(scores, pd.Series)
            assert len(scores) == len(processed_data)
            assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_score_threshold_filtering(self, enhanced_config, sample_data):
        """Test that score threshold filtering works."""
        enhanced_config['recommendation']['score_threshold'] = 0.5
        recommender = TupleRecommender(enhanced_config)
        
        recommendations = recommender.recommend_tuples(sample_data, top_k=10)
        
        # Should return fewer recommendations due to threshold filtering
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= len(sample_data)
    
    def test_config_parameter_usage(self, sample_data):
        """Test that configuration parameters are properly used."""
        config = {
            "discretization": {"enabled": True, "bins": 2},
            "association_rules": {"min_support": 0.2, "min_threshold": 0.6},
            "summaries": {"desired_size": 2},
            "recommendation": {"method": "hybrid", "top_k": 2}
        }
        
        recommender = TupleRecommender(config)
        recommendations = recommender.recommend_tuples(sample_data)
        
        # Should respect the top_k parameter
        assert len(recommendations) <= 2
    
    def test_two_tuple_edge_case(self):
        """Test edge case with only 2 tuples that previously caused infinite loops."""
        config = {
            "discretization": {"enabled": False},
            "association_rules": {"enabled": True, "min_support": 0.5, "min_threshold": 0.5},
            "summaries": {"enabled": True, "desired_size": 5},  # Higher than number of tuples
            "recommendation": {"method": "hybrid", "top_k": 5}
        }
        
        # Test with identical tuples
        identical_data = pd.DataFrame({
            'col1': ['A', 'A'],
            'col2': ['X', 'X'],
            'col3': [1, 1]
        })
        
        recommender = TupleRecommender(config)
        result = recommender.recommend_tuples(identical_data, top_k=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2
        
        # Test with partially similar tuples
        similar_data = pd.DataFrame({
            'col1': ['A', 'B'],
            'col2': ['X', 'X'],
            'col3': [1, 1]
        })
        
        result2 = recommender.recommend_tuples(similar_data, top_k=2)
        
        assert isinstance(result2, pd.DataFrame)
        assert len(result2) <= 2
    
    def test_fallback_scoring(self, enhanced_config):
        """Test that fallback scoring works when pattern mining fails."""
        # Create data that will likely cause pattern mining to fail
        small_data = pd.DataFrame({
            'A': [1, 2],
            'B': [3.0, 4.0],  # Float column for discretization
            'C': ['x', 'y']
        })
        
        recommender = TupleRecommender(enhanced_config)
        
        # Force the use of fallback scoring by causing failures
        recommender.config['association_rules']['min_support'] = 0.99  # Too high
        recommender.config['summaries']['desired_size'] = 0  # Invalid
        
        result = recommender.recommend_tuples(small_data, top_k=2)
        
        # Should still return results using fallback scoring
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2