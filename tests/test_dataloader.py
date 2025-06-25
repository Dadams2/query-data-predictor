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

# The sample_dataset_dir fixture is now in conftest.py

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
    
    # Test retrieving session DataFrame
    session_data = loader.get_results_for_session(1001)
    assert isinstance(session_data, pd.DataFrame)
    assert len(session_data) == 4  # Session 1001 has 4 queries
    assert "session_id" in session_data.columns
    assert "query_position" in session_data.columns
    assert "results_filepath" in session_data.columns
    
    # Test retrieving another session
    session_data2 = loader.get_results_for_session(1002)
    assert isinstance(session_data2, pd.DataFrame)
    assert len(session_data2) == 3  # Session 1002 has 3 queries
    
    # Test caching
    assert 1001 in loader.memory_cache
    assert 1002 in loader.memory_cache
    
    # Test error handling for non-existent session
    with pytest.raises(ValueError):
        loader.get_results_for_session(9999)

def test_get_results_for_query(sample_dataset_dir):
    """Test retrieving results for a specific query."""
    loader = DataLoader(sample_dataset_dir)
    
    # Test retrieving query results (should return actual result DataFrames)
    result1 = loader.get_results_for_query(1001, 0)
    assert isinstance(result1, pd.DataFrame)
    assert len(result1) == 3  # result1_df has 3 rows
    assert "ra" in result1.columns
    assert "dec" in result1.columns
    assert "objid" in result1.columns
    
    # Test retrieving another query result
    result2 = loader.get_results_for_query(1001, 1)
    assert isinstance(result2, pd.DataFrame)
    assert len(result2) == 2  # result2_df has 2 rows
    
    # Test with different session and query with gaps
    result3 = loader.get_results_for_query(1002, 0)
    assert isinstance(result3, pd.DataFrame)
    assert len(result3) == 3  # result5_df has 3 rows
    
    # Test with query position 3 in session 1001
    result4 = loader.get_results_for_query(1001, 3)
    assert isinstance(result4, pd.DataFrame)
    assert len(result4) == 3  # result3_df has 3 rows
    
    # Test error handling for non-existent query
    with pytest.raises(ValueError):
        loader.get_results_for_query(1001, 100000)
        
    # Test error handling for non-existent session
    with pytest.raises(ValueError):
        loader.get_results_for_query(9999, 0)

def test_metadata_column_handling(sample_dataset_dir):
    """Test handling of different metadata column names."""
    loader = DataLoader(sample_dataset_dir)
    
    # Rename the path column to filepath
    loader.metadata.rename(columns={"path": "filepath"}, inplace=True)
    
    # The method should still work with the new column name
    session_data = loader.get_results_for_session(1001)
    assert isinstance(session_data, pd.DataFrame)
    assert len(session_data) == 4
    
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
            "path": [os.path.join(temp_dir, "nonexistent.pkl")]
        })
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create a new DataLoader instance with the temporary directory
        loader = DataLoader(temp_dir)
        
        # Test that the appropriate error is raised
        with pytest.raises(FileNotFoundError):
            loader.get_results_for_session(1001)

def test_query_result_file_structure(sample_dataset_dir):
    """Test that query result files are properly structured and accessible."""
    loader = DataLoader(sample_dataset_dir)
    
    # Test that we can retrieve specific query results
    result = loader.get_results_for_query(1001, 0)
    assert isinstance(result, pd.DataFrame)
    assert "ra" in result.columns
    assert "dec" in result.columns
    assert "modelmag_g" in result.columns
    
    # Verify the data values match what we put in
    assert result.iloc[0]["ra"] == 359.78
    assert result.iloc[0]["objid"] == 758874373140775052
    
    # Test another query in the same session
    result2 = loader.get_results_for_query(1001, 1)
    assert isinstance(result2, pd.DataFrame)
    assert len(result2) == 2
    assert result2.iloc[0]["ra"] == 360.12

def test_session_structure(sample_dataset_dir):
    """Test that session files contain the expected structure."""
    loader = DataLoader(sample_dataset_dir)
    
    session_data = loader.get_results_for_session(1001)
    
    # Check that all expected columns are present
    expected_columns = ["session_id", "query_position", "current_query", 
                       "results_filepath", "query_type", "query_length", 
                       "token_count", "has_join", "has_where", 
                       "result_column_count", "result_row_count"]
    
    for col in expected_columns:
        assert col in session_data.columns, f"Missing column: {col}"
    
    # Check data types and values
    assert session_data["session_id"].iloc[0] == 1001
    assert session_data["query_position"].iloc[0] == 0
    assert session_data["query_type"].iloc[0] == "SELECT"
    assert session_data["has_where"].iloc[0] == True
    
    # Check that query positions have gaps as expected
    query_positions = sorted(session_data["query_position"].tolist())
    assert query_positions == [0, 1, 3, 5]  # Has gaps

def test_sample_data():
    """Test the sample_data method."""
    loader = DataLoader(REAL_METADATA)
    data = loader.get_results_for_query(11305, 24)
    data2 = loader.get_results_for_query(11306, 1)
    print(data)

    