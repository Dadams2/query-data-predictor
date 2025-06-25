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
            "session_id": [1001, 1002, 1003],
            "path": [
                os.path.join(temp_dir, "query_prediction_session_1001.pkl"),
                os.path.join(temp_dir, "query_prediction_session_1002.pkl"), 
                os.path.join(temp_dir, "query_prediction_session_1003.pkl")
            ]
        })
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create query_results subdirectory for result files
        results_dir = os.path.join(temp_dir, "query_results")
        os.makedirs(results_dir)
        
        # Create sample result DataFrames
        result1_df = pd.DataFrame({
            "ra": [359.78, 357.68, 358.24],
            "dec": [36.23, 33.33, 35.23],
            "type": [3, 3, 3],
            "modelmag_g": [12.08, 11.76, 11.72],
            "objid": [758874373140775052, 758874297454690378, 758874298527646095]
        })
        result1_path = os.path.join(results_dir, "results_session_1001_query_0.pkl")
        with open(result1_path, "wb") as f:
            pickle.dump(result1_df, f)
            
        result2_df = pd.DataFrame({
            "ra": [360.12, 358.45],
            "dec": [37.45, 34.67], 
            "type": [3, 3],
            "modelmag_g": [13.21, 12.34],
            "objid": [758874373140775999, 758874297454691000]
        })
        result2_path = os.path.join(results_dir, "results_session_1001_query_1.pkl")
        with open(result2_path, "wb") as f:
            pickle.dump(result2_df, f)
            
        result3_df = pd.DataFrame({
            "ra": [361.45, 359.23, 357.89],
            "dec": [38.12, 35.78, 33.45],
            "type": [3, 3, 3], 
            "modelmag_g": [14.56, 13.78, 12.99],
            "objid": [758874373140776111, 758874297454691222, 758874298527647333]
        })
        result3_path = os.path.join(results_dir, "results_session_1002_query_0.pkl")
        with open(result3_path, "wb") as f:
            pickle.dump(result3_df, f)
        
        # Create session DataFrames that match the real structure
        session1_df = pd.DataFrame({
            "session_id": [1001, 1001],
            "query_position": [0, 1],
            "current_query": ["SELECT ra,dec,type FROM PhotoObj WHERE ra > 359", "SELECT ra,dec,type FROM PhotoObj WHERE dec > 35"],
            "results_filepath": [result1_path, result2_path],
            "query_type": ["SELECT", "SELECT"],
            "query_length": [45, 46],
            "token_count": [8, 8],
            "has_join": [False, False],
            "has_where": [True, True],
            "result_column_count": [5, 5],
            "result_row_count": [3, 2]
        })
        with open(os.path.join(temp_dir, "query_prediction_session_1001.pkl"), "wb") as f:
            pickle.dump(session1_df, f)
        
        session2_df = pd.DataFrame({
            "session_id": [1002],
            "query_position": [0],
            "current_query": ["SELECT ra,dec,type FROM PhotoObj WHERE ra < 362"],
            "results_filepath": [result3_path],
            "query_type": ["SELECT"],
            "query_length": [45],
            "token_count": [8],
            "has_join": [False],
            "has_where": [True],
            "result_column_count": [5],
            "result_row_count": [3]
        })
        with open(os.path.join(temp_dir, "query_prediction_session_1002.pkl"), "wb") as f:
            pickle.dump(session2_df, f)
            
        # Create an empty session for error testing
        session3_df = pd.DataFrame()
        with open(os.path.join(temp_dir, "query_prediction_session_1003.pkl"), "wb") as f:
            pickle.dump(session3_df, f)
            
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
    
    # Test retrieving session DataFrame
    session_data = loader.get_results_for_session(1001)
    assert isinstance(session_data, pd.DataFrame)
    assert len(session_data) == 2  # Session 1001 has 2 queries
    assert "session_id" in session_data.columns
    assert "query_position" in session_data.columns
    assert "results_filepath" in session_data.columns
    
    # Test retrieving another session
    session_data2 = loader.get_results_for_session(1002)
    assert isinstance(session_data2, pd.DataFrame)
    assert len(session_data2) == 1  # Session 1002 has 1 query
    
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
    
    # Test with different session
    result3 = loader.get_results_for_query(1002, 0)
    assert isinstance(result3, pd.DataFrame)
    assert len(result3) == 3  # result3_df has 3 rows
    
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
    assert len(session_data) == 2
    
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

def test_sample_data():
    """Test the sample_data method."""
    loader = DataLoader(REAL_METADATA)
    data = loader.get_results_for_query(11305, 24)
    data2 = loader.get_results_for_query(11306, 1)
    print(data)

    