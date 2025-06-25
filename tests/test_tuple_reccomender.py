from fixtures import sample_config
from query_data_predictor.tuple_recommender import TupleRecommender
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """Fixture for sample DataFrame data."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.1, 20.2, 30.3, 40.4, 50],
        'C': ['a', 'b', 'a', 'b', 'c']
    })

class TestTupleRecommender:
    """Test suite for the tuple recommender."""
    
    @pytest.fixture
    def recommender(self, sample_config):
        """Fixture for TupleRecommender instance."""
        return TupleRecommender(sample_config)
    
    def test_preprocess_data(self, recommender, sample_data):
        """Test data preprocessing."""
        processed = recommender.preprocess_data(sample_data.copy())        

        # Check that numeric columns have been discretized
        assert processed is not None
        assert not processed.equals(sample_data)  # Should be different after discretization
        
    def test_recommend_tuples(self, recommender, sample_data):
        """Test tuple recommendation."""
        recommendations = recommender.recommend_tuples(sample_data, top_k=3)
        
        # Basic validation of the recommendations
        assert recommendations is not None
        # The number of recommendations may vary based on the algorithm, so we don't check the exact count


    def test_prepend_column_names(self, recommender):
        """Test prepending column names to values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        result = recommender.prepend_column_names(df.copy())
        # Check that values are prepended with column names
        assert result['col1'][0] == 'col1_1'
        assert result['col2'][0] == 'col2_a'

    def test_prepend_column_names_empty_df(self, recommender):
        """Test prepending column names with empty DataFrame."""
        df = pd.DataFrame()
        result = recommender.prepend_column_names(df.copy())
        
        # Result should be empty DataFrame
        assert result.empty