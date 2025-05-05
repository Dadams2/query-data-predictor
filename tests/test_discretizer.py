import os
import pytest
import polars as pl
import polars.testing as pl_testing
import numpy as np
import tempfile
import pickle
from query_data_predictor.discretizer import Discretizer

class TestDiscretizer:
    @pytest.fixture
    def sample_df(self):
        """Fixture to create a sample DataFrame with float columns."""
        return pl.DataFrame({
            'float_col1': [1.1, 2.2, 3.3, 4.4, 5.5],
            'float_col2': [10.1, 20.2, 30.3, 40.4, 50.5],
            'int_col': [1, 2, 3, 4, 5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
    
    @pytest.fixture
    def temp_file_path(self):
        """Fixture to create a temporary file path for save/load testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        temp_file.close()
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    def test_init_default(self):
        """Test default initialization."""
        discretizer = Discretizer()
        assert discretizer.method == 'equal_width'
        assert discretizer.bins == 5
        assert discretizer.save_path is None
        assert discretizer.discretization_params == {}
    
    def test_init_custom(self):
        """Test custom initialization."""
        discretizer = Discretizer(method='equal_freq', bins=10, save_path='test.pkl')
        assert discretizer.method == 'equal_freq'
        assert discretizer.bins == 10
        assert discretizer.save_path == 'test.pkl'
        assert discretizer.discretization_params == {}
    
    def test_save_load_params(self, temp_file_path, sample_df):
        """Test saving and loading parameters."""
        # Create discretizer with a save path
        discretizer1 = Discretizer(save_path=temp_file_path)
        
        # Add some discretization parameters
        discretizer1.discretize_dataframe(sample_df.clone())
        
        # Create a new discretizer and load the saved parameters
        discretizer2 = Discretizer(load_path=temp_file_path)
        
        # Check that parameters were loaded correctly
        assert discretizer1.discretization_params.keys() == discretizer2.discretization_params.keys()
        for key in discretizer1.discretization_params:
            np.testing.assert_array_almost_equal(
                discretizer1.discretization_params[key], 
                discretizer2.discretization_params[key]
            )
    
    def test_load_params_nonexistent(self):
        """Test loading parameters from a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Discretizer(load_path='nonexistent_file.pkl')
    
    def test_discretize_equal_width(self, sample_df):
        """Test discretization with equal_width method."""
        discretizer = Discretizer(method='equal_width', bins=3)
        result = discretizer.discretize_dataframe(sample_df.clone())
        
        # Check that original float columns are dropped
        assert 'float_col1' not in result.columns
        assert 'float_col2' not in result.columns
        
        # Check that binned columns are added
        assert 'float_col1_bin' in result.columns
        assert 'float_col2_bin' in result.columns
        
        # Check that discretization parameters are stored
        assert 'float_col1' in discretizer.discretization_params
        assert 'float_col2' in discretizer.discretization_params
        
        # Check that bin values are within expected range (0 to bins)
        assert result['float_col1_bin'].min() >= 0
        assert result['float_col1_bin'].max() <= 4
    
    def test_discretize_equal_freq(self, sample_df):
        """Test discretization with equal_freq method."""
        discretizer = Discretizer(method='equal_freq', bins=3)
        result = discretizer.discretize_dataframe(sample_df.clone())
        
        # Check that binned columns are added
        assert 'float_col1_bin' in result.columns
        assert 'float_col2_bin' in result.columns
        
        # Check that bin values are within expected range
        assert result['float_col1_bin'].min() >= 0
        assert result['float_col1_bin'].max() < 5
    
    def test_discretize_kmeans(self, sample_df):
        """Test discretization with kmeans method."""
        discretizer = Discretizer(method='kmeans', bins=2)
        result = discretizer.discretize_dataframe(sample_df.clone())
        
        # Check that binned columns are added
        assert 'float_col1_bin' in result.columns
        assert 'float_col2_bin' in result.columns
        
        # Check that bin values match kmeans output (0 or 1 for 2 clusters)
        assert set(result['float_col1_bin'].unique()).issubset({0, 1})
    
    def test_discretize_invalid_method(self, sample_df):
        """Test discretization with invalid method raises error."""
        discretizer = Discretizer(method='invalid_method')
        with pytest.raises(ValueError, match="Invalid method"):
            discretizer.discretize_dataframe(sample_df.clone())
    
    def test_discretize_no_float_columns(self):
        """Test discretization with DataFrame containing no float columns."""
        df = pl.DataFrame({'int_col': [1, 2, 3], 'str_col': ['a', 'b', 'c']})
        discretizer = Discretizer()
        result = discretizer.discretize_dataframe(df.clone())
        
        # Result should be the same as input
        pl_testing.assert_frame_equal(result, df)
        assert discretizer.discretization_params == {}
    
    def test_discretize_with_existing_params(self, sample_df):
        """Test discretization with existing parameters."""
        # First discretize to get parameters
        discretizer = Discretizer(method='equal_width', bins=3)
        discretizer.discretize_dataframe(sample_df.clone())
        
        # Now discretize a similar DataFrame
        new_df = pl.DataFrame({
            'float_col1': [1.5, 2.5, 3.5, 4.5, 5.5],
            'float_col2': [15.1, 25.2, 35.3, 45.4, 55.5]
        })
        
        result = discretizer.discretize_dataframe(new_df.clone())
        
        # Check that discretization used existing parameters
        assert 'float_col1_bin' in result.columns
        assert 'float_col2_bin' in result.columns
    
    def test_prepend_column_names(self):
        """Test prepending column names to values."""
        df = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        discretizer = Discretizer()
        result = discretizer.prepend_column_names(df.clone())
        print(result) 
        # Check that values are prepended with column names
        assert result['col1'][0] == 'col1_1'
        assert result['col2'][0] == 'col2_a'
    
    def test_prepend_column_names_empty_df(self):
        """Test prepending column names with empty DataFrame."""
        df = pl.DataFrame()
        discretizer = Discretizer()
        result = discretizer.prepend_column_names(df.clone())
        
        # Result should be empty DataFrame
        assert result.is_empty()