import os
import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
from unittest.mock import MagicMock, patch

from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.dataloader import DataLoader

# The sample_dataset_dir fixture is now shared in conftest.py


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader for testing without file I/O."""
    mock_loader = MagicMock(spec=DataLoader)
    
    # Mock session data - DataFrame format matching real structure
    mock_session_data = {
        "session_1": pd.DataFrame({
            "session_id": ["session_1", "session_1", "session_1"],
            "query_position": [1, 2, 3],
            "current_query": ["SELECT * FROM table1", "SELECT * FROM table2", "SELECT * FROM table3"],
            "results_filepath": ["/path/to/results1.pkl", "/path/to/results2.pkl", "/path/to/results3.pkl"],
            "query_type": ["SELECT", "SELECT", "SELECT"],
            "result_column_count": [5, 4, 6],
            "result_row_count": [10, 8, 12]
        }),
        "session_2": pd.DataFrame({
            "session_id": ["session_2", "session_2", "session_2"],
            "query_position": [0, 2, 4],
            "current_query": ["SELECT * FROM tableA", "SELECT * FROM tableB", "SELECT * FROM tableC"],
            "results_filepath": ["/path/to/resultsA.pkl", "/path/to/resultsB.pkl", "/path/to/resultsC.pkl"],
            "query_type": ["SELECT", "SELECT", "SELECT"],
            "result_column_count": [3, 2, 4],
            "result_row_count": [5, 3, 7]
        }),
        "empty_session": pd.DataFrame(columns=["session_id", "query_position", "current_query", "results_filepath"])
    }
    
    # Mock query results - return sample DataFrames
    mock_query_results = {
        ("session_1", 1): pd.DataFrame({"ra": [1.1, 1.2], "dec": [2.1, 2.2], "objid": [101, 102]}),
        ("session_1", 2): pd.DataFrame({"ra": [2.1, 2.2, 2.3], "dec": [3.1, 3.2, 3.3], "objid": [201, 202, 203]}),
        ("session_1", 3): pd.DataFrame({"ra": [3.1], "dec": [4.1], "objid": [301]}),
        ("session_2", 0): pd.DataFrame({"ra": [0.1, 0.2], "dec": [1.1, 1.2], "objid": [1, 2]}),
        ("session_2", 2): pd.DataFrame({"ra": [2.5], "dec": [3.5], "objid": [25]}),
        ("session_2", 4): pd.DataFrame({"ra": [4.1, 4.2, 4.3], "dec": [5.1, 5.2, 5.3], "objid": [401, 402, 403]}),
    }
    
    def mock_get_results_for_session(session_id):
        if session_id in mock_session_data:
            return mock_session_data[session_id]
        raise ValueError(f"Session {session_id} not found")
    
    def mock_get_results_for_query(session_id, query_id):
        key = (session_id, query_id)
        if key in mock_query_results:
            return mock_query_results[key]
        raise ValueError(f"Query {query_id} not found in session {session_id}")
    
    mock_loader.get_results_for_session.side_effect = mock_get_results_for_session
    mock_loader.get_results_for_query.side_effect = mock_get_results_for_query
    
    return mock_loader


class TestQueryResultSequence:
    
    def test_init(self, mock_dataloader):
        """Test initialization of QueryResultSequence."""
        sequence = QueryResultSequence(mock_dataloader)
        assert sequence.dataloader is mock_dataloader
        assert sequence.id_cache == {}

    def test_get_ordered_query_ids_dataframe_data(self, sample_dataset_dir):
        """Test getting ordered query IDs from DataFrame-based session data."""
        loader = DataLoader(sample_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Test with session 1001 (DataFrame data with gaps)
        query_ids = sequence.get_ordered_query_ids(1001)
        assert isinstance(query_ids, list)
        assert set(query_ids) == {0, 1, 3, 5}  # Query positions from the DataFrame
        assert query_ids == sorted(query_ids)  # Should be sorted
        
        # Test caching
        assert 1001 in sequence.id_cache
        cached_ids = sequence.get_ordered_query_ids(1001)
        assert query_ids == cached_ids
        
        # Test with session 1002 (also has gaps)
        query_ids_2 = sequence.get_ordered_query_ids(1002)
        assert set(query_ids_2) == {0, 2, 4}
        assert query_ids_2 == sorted(query_ids_2)

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

    def test_integration_with_real_dataloader(self, sample_dataset_dir):
        """Test integration with actual DataLoader."""
        loader = DataLoader(sample_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Test with session 1001 (DataFrame data with gaps)
        query_ids = sequence.get_ordered_query_ids(1001)
        assert len(query_ids) > 0
        assert set(query_ids) == {0, 1, 3, 5}
        
        # Test getting query results
        first_query_id = query_ids[0]  # Should be 0
        results = sequence.get_query_results(1001, first_query_id)
        assert results is not None
        assert isinstance(results, pd.DataFrame)
        assert "ra" in results.columns
        assert "dec" in results.columns
        
        # Test iteration (if there are multiple queries)
        if len(query_ids) > 1:
            pairs = list(sequence.iter_query_result_pairs(1001))
            assert isinstance(pairs, list)
            # Should have 3 pairs: (0,1), (1,3), (3,5)
            assert len(pairs) == 3
            
            # Check first pair
            curr_id, fut_id, curr_res, fut_res = pairs[0]
            assert curr_id == 0
            assert fut_id == 1
            assert isinstance(curr_res, pd.DataFrame)
            assert isinstance(fut_res, pd.DataFrame)

    def test_single_query_session(self, sample_dataset_dir):
        """Test handling of sessions with only one query."""
        loader = DataLoader(sample_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Session 1003 has only one query
        query_ids = sequence.get_ordered_query_ids(1003)
        assert len(query_ids) == 1
        assert query_ids == [0]
        
        # Should return empty list for pairs since we need at least 2 queries
        pairs = list(sequence.iter_query_result_pairs(1003))
        assert pairs == []
        
        # Should raise IndexError for gap query
        with pytest.raises(IndexError):
            sequence.get_query_pair_with_gap(1003, 0, gap=1)

    def test_dataframe_based_session_multiple(self, sample_dataset_dir):
        """Test with DataFrame-based session data for multiple sessions."""
        loader = DataLoader(sample_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Test with session 1001 (DataFrame data with gaps)
        query_ids_1001 = sequence.get_ordered_query_ids(1001)
        assert len(query_ids_1001) > 0
        expected_positions_1001 = [0, 1, 3, 5]  # From the shared test data
        assert set(query_ids_1001) == set(expected_positions_1001)
        
        # Test with session 1002 (DataFrame data with gaps)
        query_ids_1002 = sequence.get_ordered_query_ids(1002)
        assert len(query_ids_1002) > 0
        expected_positions_1002 = [0, 2, 4]  # From the shared test data
        assert set(query_ids_1002) == set(expected_positions_1002)
        
        # Test that we can get actual results for these queries
        result_1001_0 = sequence.get_query_results(1001, 0)
        assert isinstance(result_1001_0, pd.DataFrame)
        assert "ra" in result_1001_0.columns
        
        result_1002_2 = sequence.get_query_results(1002, 2)
        assert isinstance(result_1002_2, pd.DataFrame)
        assert "ra" in result_1002_2.columns

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

    def test_real_data_gap_behavior(self, sample_dataset_dir):
        """Test gap behavior with realistic data that has gaps in query positions."""
        loader = DataLoader(sample_dataset_dir)
        sequence = QueryResultSequence(loader)
        
        # Session 1001 has query positions: [0, 1, 3, 5]
        query_ids = sequence.get_ordered_query_ids(1001)
        assert query_ids == [0, 1, 3, 5]
        
        # Test gap=1 pairs: (0,1), (1,3), (3,5)
        pairs_gap1 = list(sequence.iter_query_result_pairs(1001, gap=1))
        assert len(pairs_gap1) == 3
        
        curr_ids = [pair[0] for pair in pairs_gap1]
        fut_ids = [pair[1] for pair in pairs_gap1]
        assert curr_ids == [0, 1, 3]
        assert fut_ids == [1, 3, 5]
        
        # Test gap=2 pairs: (0,3), (1,5)
        pairs_gap2 = list(sequence.iter_query_result_pairs(1001, gap=2))
        assert len(pairs_gap2) == 2
        
        curr_ids_2 = [pair[0] for pair in pairs_gap2]
        fut_ids_2 = [pair[1] for pair in pairs_gap2]
        assert curr_ids_2 == [0, 1]
        assert fut_ids_2 == [3, 5]
        
        # Test gap=3 pairs: (0,5)
        pairs_gap3 = list(sequence.iter_query_result_pairs(1001, gap=3))
        assert len(pairs_gap3) == 1
        assert pairs_gap3[0][0] == 0
        assert pairs_gap3[0][1] == 5
        
        # Test that all results are proper DataFrames
        for curr_id, fut_id, curr_res, fut_res in pairs_gap1:
            assert isinstance(curr_res, pd.DataFrame)
            assert isinstance(fut_res, pd.DataFrame)
            assert not curr_res.empty
            assert not fut_res.empty
            assert "ra" in curr_res.columns
            assert "ra" in fut_res.columns
