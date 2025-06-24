import os
import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
from unittest.mock import MagicMock, patch

from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.dataloader import DataLoader


@pytest.fixture
def sequence_test_dataset_dir():
    """Create a temporary directory with sample data specifically for sequence testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a metadata.csv file
        metadata_path = os.path.join(temp_dir, "metadata.csv")
        metadata_df = pd.DataFrame({
            "session_id": [1001, 1002, 1003],
            "path": ["session_1001.pkl", "session_1002.pkl", "session_1003.pkl"]
        })
        metadata_df.to_csv(metadata_path, index=False)
        
        # Session 1001: DataFrame-based data with sequential query IDs
        df_data_1 = pd.DataFrame({
            "query_position": [1, 2, 3, 5],  # Sequential with gap
            "result": [
                [10, 20, 30], 
                [40, 50, 60], 
                [70, 80, 90], 
                [100, 110, 120]
            ],
            "other_col": ["w", "x", "y", "z"]
        })
        with open(os.path.join(temp_dir, "session_1001.pkl"), "wb") as f:
            pickle.dump(df_data_1, f)
        
        # Session 1002: DataFrame-based data 
        df_data_2 = pd.DataFrame({
            "query_position": [0, 1, 2, 4],
            "result": [
                [1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9], 
                [10, 11, 12]
            ],
            "other_col": ["a", "b", "c", "d"]
        })
        with open(os.path.join(temp_dir, "session_1002.pkl"), "wb") as f:
            pickle.dump(df_data_2, f)
            
        # Session 1003: Single query (edge case)
        single_query_data = pd.DataFrame({
            "query_position": [0],
            "result": [[1, 2, 3]],
            "other_col": ["single"]
        })
        with open(os.path.join(temp_dir, "session_1003.pkl"), "wb") as f:
            pickle.dump(single_query_data, f)
            
        yield temp_dir


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader for testing without file I/O."""
    mock_loader = MagicMock(spec=DataLoader)
    
    # Mock session data - DataFrame format only
    mock_session_data = {
        "session_1": pd.DataFrame({
            "query_position": [1, 2, 3],
            "result": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "other_col": ["a", "b", "c"]
        }),
        "session_2": pd.DataFrame({
            "query_position": [0, 2, 4],
            "result": [[10, 11], [12, 13], [14, 15]],
            "other_col": ["x", "y", "z"]
        }),
        "empty_session": pd.DataFrame(columns=["query_position", "result", "other_col"])
    }
    
    def mock_get_results_for_session(session_id):
        if session_id in mock_session_data:
            return mock_session_data[session_id]
        raise ValueError(f"Session {session_id} not found")
    
    def mock_get_results_for_query(session_id, query_id):
        session_data = mock_get_results_for_session(session_id)
        filtered_data = session_data[session_data['query_position'] == query_id]
        if len(filtered_data) == 0:
            raise ValueError(f"Query {query_id} not found in session {session_id}")
        return filtered_data
    
    mock_loader.get_results_for_session.side_effect = mock_get_results_for_session
    mock_loader.get_results_for_query.side_effect = mock_get_results_for_query
    
    return mock_loader


class TestQueryResultSequence:
    
    def test_init(self, mock_dataloader):
        """Test initialization of QueryResultSequence."""
        sequence = QueryResultSequence(mock_dataloader)
        assert sequence.dataloader is mock_dataloader
        assert sequence.id_cache == {}

    def test_get_ordered_query_ids_dataframe_data(self, sequence_test_dataset_dir):
        """Test getting ordered query IDs from DataFrame-based session data."""
        loader = DataLoader(sequence_test_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Test with session 1001 (DataFrame data)
        query_ids = sequence.get_ordered_query_ids(1001)
        assert isinstance(query_ids, list)
        assert set(query_ids) == {1, 2, 3, 5}  # Query positions from the DataFrame
        assert query_ids == sorted(query_ids)  # Should be sorted
        
        # Test caching
        assert 1001 in sequence.id_cache
        cached_ids = sequence.get_ordered_query_ids(1001)
        assert query_ids == cached_ids

    def test_get_ordered_query_ids_empty_session(self, mock_dataloader):
        """Test getting query IDs from an empty session."""
        sequence = QueryResultSequence(mock_dataloader)
        
        query_ids = sequence.get_ordered_query_ids("empty_session")
        assert query_ids == []

    def test_get_query_results(self, mock_dataloader):
        """Test getting query results."""
        sequence = QueryResultSequence(mock_dataloader)
        
        result = sequence.get_query_results("session_1", 1)
        assert isinstance(result, pd.DataFrame)
        mock_dataloader.get_results_for_query.assert_called_with("session_1", 1)

    def test_iter_query_result_pairs_basic(self, mock_dataloader):
        """Test iterating over query result pairs with default gap."""
        sequence = QueryResultSequence(mock_dataloader)
        
        # Mock the ordered query IDs
        sequence.id_cache["session_1"] = [1, 2, 3]
        
        # Mock query results to return non-empty DataFrames
        def mock_get_query_results(session_id, query_id):
            return pd.DataFrame({"data": [query_id * 10, query_id * 20]})
        
        sequence.get_query_results = mock_get_query_results
        
        pairs = list(sequence.iter_query_result_pairs("session_1"))
        
        assert len(pairs) == 2  # 3 queries - 1 gap = 2 pairs
        
        # Check first pair
        curr_id, fut_id, curr_res, fut_res = pairs[0]
        assert curr_id == 1
        assert fut_id == 2
        assert not curr_res.empty
        assert not fut_res.empty
        
        # Check second pair
        curr_id, fut_id, curr_res, fut_res = pairs[1]
        assert curr_id == 2
        assert fut_id == 3

    def test_iter_query_result_pairs_with_gap(self, mock_dataloader):
        """Test iterating over query result pairs with custom gap."""
        sequence = QueryResultSequence(mock_dataloader)
        sequence.id_cache["session_1"] = [1, 2, 3, 4, 5]
        
        def mock_get_query_results(session_id, query_id):
            return pd.DataFrame({"data": [query_id]})
        
        sequence.get_query_results = mock_get_query_results
        
        # Test with gap of 2
        pairs = list(sequence.iter_query_result_pairs("session_1", gap=2))
        
        assert len(pairs) == 3  # 5 queries - 2 gap = 3 pairs
        
        # Check pairs
        curr_id, fut_id, curr_res, fut_res = pairs[0]
        assert curr_id == 1
        assert fut_id == 3
        
        curr_id, fut_id, curr_res, fut_res = pairs[1]
        assert curr_id == 2
        assert fut_id == 4
        
        curr_id, fut_id, curr_res, fut_res = pairs[2]
        assert curr_id == 3
        assert fut_id == 5

    def test_iter_query_result_pairs_empty_results_filtered(self, mock_dataloader):
        """Test that pairs with empty results are filtered out."""
        sequence = QueryResultSequence(mock_dataloader)
        sequence.id_cache["session_1"] = [1, 2, 3]
        
        def mock_get_query_results(session_id, query_id):
            if query_id == 2:
                return pd.DataFrame()  # Empty DataFrame
            return pd.DataFrame({"data": [query_id]})
        
        sequence.get_query_results = mock_get_query_results
        
        pairs = list(sequence.iter_query_result_pairs("session_1"))
        
        # Should skip the pair (1,2) because query 2 returns empty results
        assert len(pairs) == 0  # No valid pairs

    def test_get_query_pair_with_gap_success(self, mock_dataloader):
        """Test getting a specific query pair with gap."""
        sequence = QueryResultSequence(mock_dataloader)
        sequence.id_cache["session_1"] = [1, 2, 3, 4]
        
        def mock_get_query_results(session_id, query_id):
            return pd.DataFrame({"data": [query_id * 10]})
        
        sequence.get_query_results = mock_get_query_results
        
        current_results, future_results = sequence.get_query_pair_with_gap("session_1", 2, gap=1)
        
        assert not current_results.empty
        assert not future_results.empty
        assert current_results.iloc[0]["data"] == 20  # Query 2 * 10
        assert future_results.iloc[0]["data"] == 30   # Query 3 * 10

    def test_get_query_pair_with_gap_query_not_found(self, mock_dataloader):
        """Test error when specified query is not found."""
        sequence = QueryResultSequence(mock_dataloader)
        sequence.id_cache["session_1"] = [1, 2, 3]
        
        with pytest.raises(ValueError, match="Query ID 5 not found in session session_1"):
            sequence.get_query_pair_with_gap("session_1", 5, gap=1)

    def test_get_query_pair_with_gap_future_query_not_found(self, mock_dataloader):
        """Test error when future query with gap doesn't exist."""
        sequence = QueryResultSequence(mock_dataloader)
        sequence.id_cache["session_1"] = [1, 2, 3]
        
        with pytest.raises(IndexError, match="No future query with gap 2 from query 2 in session session_1"):
            sequence.get_query_pair_with_gap("session_1", 2, gap=2)

    def test_integration_with_real_dataloader(self, sequence_test_dataset_dir):
        """Test integration with actual DataLoader."""
        loader = DataLoader(sequence_test_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Test with session 1001 (DataFrame data)
        query_ids = sequence.get_ordered_query_ids(1001)
        assert len(query_ids) > 0
        
        # Test getting query results
        first_query_id = query_ids[0]
        results = sequence.get_query_results(1001, first_query_id)
        assert results is not None
        assert isinstance(results, pd.DataFrame)
        
        # Test iteration (if there are multiple queries)
        if len(query_ids) > 1:
            pairs = list(sequence.iter_query_result_pairs(1001))
            assert isinstance(pairs, list)

    def test_single_query_session(self, sequence_test_dataset_dir):
        """Test handling of sessions with only one query."""
        loader = DataLoader(sequence_test_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Session 1003 has only one query
        query_ids = sequence.get_ordered_query_ids(1003)
        assert len(query_ids) == 1
        
        # Should return empty list for pairs since we need at least 2 queries
        pairs = list(sequence.iter_query_result_pairs(1003))
        assert pairs == []
        
        # Should raise IndexError for gap query
        with pytest.raises(IndexError):
            sequence.get_query_pair_with_gap(1003, 0, gap=1)

    def test_dataframe_based_session_multiple(self, sequence_test_dataset_dir):
        """Test with DataFrame-based session data for multiple sessions."""
        loader = DataLoader(sequence_test_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Test with session 1001 (DataFrame data)
        query_ids_1001 = sequence.get_ordered_query_ids(1001)
        assert len(query_ids_1001) > 0
        expected_positions_1001 = [1, 2, 3, 5]  # From the test data
        assert set(query_ids_1001) == set(expected_positions_1001)
        
        # Test with session 1002 (DataFrame data)
        query_ids_1002 = sequence.get_ordered_query_ids(1002)
        assert len(query_ids_1002) > 0
        expected_positions_1002 = [0, 1, 2, 4]  # From the test data
        assert set(query_ids_1002) == set(expected_positions_1002)

    def test_caching_behavior(self, mock_dataloader):
        """Test that query ID caching works correctly."""
        sequence = QueryResultSequence(mock_dataloader)
        
        # First call should populate cache
        query_ids_1 = sequence.get_ordered_query_ids("session_1")
        assert "session_1" in sequence.id_cache
        
        # Second call should use cache
        query_ids_2 = sequence.get_ordered_query_ids("session_1")
        assert query_ids_1 == query_ids_2
        
        # DataLoader should only be called once
        mock_dataloader.get_results_for_session.assert_called_once_with("session_1")
