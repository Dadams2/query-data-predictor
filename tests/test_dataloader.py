import os
import pytest
import pandas as pd
import numpy as np
import pickle
import pathlib
import tempfile
from unittest.mock import patch, MagicMock

REAL_METADATA = pathlib.Path(__file__).parent.parent / "data" / "datasets" 

from query_data_predictor.dataloader import DataLoader

    
@pytest.fixture
def sample_dataset_dir():
    """Create a temporary directory with sample data for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a metadata.csv file
        metadata_path = os.path.join(temp_dir, "metadata.csv")
        metadata_df = pd.DataFrame({
            "session_id": [1001, 1002, 1003],  # Use integers, not strings
            "path": ["session_1001.pkl", "session_1002.pkl", "session_1003.pkl"]
        })
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create sample pickle files
        # Dict-based sample data
        dict_data = {
            1: np.array([1, 2, 3]),
            2: np.array([4, 5, 6])
        }
        with open(os.path.join(temp_dir, "session_1001.pkl"), "wb") as f:
            pickle.dump(dict_data, f)
        
        # Polars DataFrame sample data
        data = pd.DataFrame({
            "query_position": [1, 2, 3],
            "result": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        })
        with open(os.path.join(temp_dir, "session_1002.pkl"), "wb") as f:
            pickle.dump(data, f)
            
        # Create a file that doesn't exist in metadata for error testing
        with open(os.path.join(temp_dir, "session_1003.pkl"), "wb") as f:
            pickle.dump({}, f)
            
        yield temp_dir

def test_init(sample_dataset_dir):
    """Test initialization of the DataLoader."""
    # Test successful initialization
    loader = DataLoader(sample_dataset_dir)
    assert loader.dataset_dir == pathlib.Path(sample_dataset_dir)
    assert isinstance(loader.metadata, pd.DataFrame)
    assert len(loader.metadata) == 3
    assert loader.memory_cache == {}
    
    # Test initialization with non-existent directory
    with pytest.raises(FileNotFoundError):
        DataLoader("/nonexistent/directory")

def test_get_results_for_session(sample_dataset_dir):
    """Test retrieving results for a session."""
    loader = DataLoader(sample_dataset_dir)
    
    # Test retrieving dict-based data
    dict_data = loader.get_results_for_session(1001)
    assert isinstance(dict_data, dict)
    assert 1 in dict_data
    assert 2 in dict_data
    assert np.array_equal(dict_data[1], np.array([1, 2, 3]))
    
    # Test retrieving DataFrame-based data
    df_data = loader.get_results_for_session(1002)
    assert hasattr(df_data, "filter")  # Check if it's a DataFrame
    assert df_data.shape[0] == 3
    
    # Test caching
    assert 1001 in loader.memory_cache
    assert 1002 in loader.memory_cache
    
    # Test error handling for non-existent session
    with pytest.raises(ValueError):
        loader.get_results_for_session(9999)

def test_get_results_for_query(sample_dataset_dir):
    """Test retrieving results for a specific query."""
    loader = DataLoader(sample_dataset_dir)
    
    # Test with dictionary data
    result1 = loader.get_results_for_query(1001, 1)
    assert np.array_equal(result1, np.array([1, 2, 3]))
    
    # Test with DataFrame data
    result2 = loader.get_results_for_query(1002, 1)
    assert len(result2) == 1
    
    # Test error handling for non-existent query in dictionary
    with pytest.raises(ValueError):
        loader.get_results_for_query(1001, 100000)
        
    # Test error handling for non-existent query in DataFrame
    with pytest.raises(ValueError):
        loader.get_results_for_query(1002, -20)
        
    # Test caching behavior - modifying the cache for a new query
    loader.memory_cache[1003] = {20: np.array([10, 20, 30])}
    result3 = loader.get_results_for_query(1003, 20)
    assert np.array_equal(result3, np.array([10, 20, 30]))

def test_metadata_column_handling(sample_dataset_dir):
    """Test handling of different metadata column names."""
    loader = DataLoader(sample_dataset_dir)
    
    # Rename the path column to filepath
    loader.metadata.rename(columns={"path": "filepath"}, inplace=True)
    
    # The method should still work with the new column name
    dict_data = loader.get_results_for_session(1001)
    assert isinstance(dict_data, dict)
    assert 1 in dict_data
    
    # Test with no valid column name
    loader.metadata.rename(columns={"filepath": "invalid_column"}, inplace=True)
    with pytest.raises(ValueError, match="Metadata does not contain a filepath or path column"):
        loader.get_results_for_session(1001)

def test_file_not_found(sample_dataset_dir):
    """Test error handling when a data file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a metadata.csv file with a non-existent data file
        metadata_path = os.path.join(temp_dir, "metadata.csv")
        metadata_df = pd.DataFrame({
            "session_id": [1001],
            "path": ["nonexistent.pkl"]
        })
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create a new DataLoader instance with the temporary directory
        loader = DataLoader(temp_dir)
        
        # Test that the appropriate error is raised
        with pytest.raises(FileNotFoundError):
            loader.get_results_for_session(1001)

def test_sample_data():
    """Test the sample_data method."""
    loader = DataLoader(REAL_METADATA)
    data = loader.get_results_for_query(11305, 24)
    data2 = loader.get_results_for_query(11306, 1)
    print(data)

    