"""
Tests for the RandomTableRecommender class.
"""
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock

from query_data_predictor.recommender.random_table_recommender import RandomTableRecommender, TableInfo
from query_data_predictor.query_runner import QueryRunner


class TestRandomTableRecommender:
    """Test suite for RandomTableRecommender class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'random_table': {
                'max_sample_size': 100,
                'min_sample_size': 5,
                'exclude_current_results': True
            },
            'recommendation': {
                'top_k': 10
            }
        }
    
    @pytest.fixture
    def mock_query_runner(self):
        """Mock QueryRunner for testing."""
        mock_runner = Mock(spec=QueryRunner)
        return mock_runner
    
    @pytest.fixture
    def sdss_spec_dataframe(self):
        """Sample SDSS spectroscopic data."""
        return pd.DataFrame({
            'specObjID': [1001, 1002, 1003],
            'objID': [2001, 2002, 2003],
            'z': [0.1, 0.2, 0.3],
            'zConf': [0.95, 0.98, 0.92],
            'SpecClass': ['GALAXY', 'QSO', 'STAR'],
            'ra': [180.5, 181.0, 181.5],
            'dec': [45.2, 45.5, 45.8]
        })
    
    @pytest.fixture
    def sdss_photo_dataframe(self):
        """Sample SDSS photometric data."""
        return pd.DataFrame({
            'objID': [3001, 3002, 3003],
            'ra': [182.1, 182.5, 183.0],
            'dec': [46.0, 46.3, 46.6],
            'type': [3, 6, 3],
            'modelmag_g': [18.5, 19.2, 20.1],
            'modelmag_r': [17.8, 18.5, 19.4]
        })
    
    def test_init_with_query_runner(self, sample_config, mock_query_runner):
        """Test initialization with QueryRunner."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        assert recommender.query_runner == mock_query_runner
        assert recommender.max_sample_size == 100
        assert recommender.min_sample_size == 5
        assert recommender.exclude_current_results == True
    
    def test_init_without_query_runner(self, sample_config):
        """Test initialization without QueryRunner raises error."""
        with pytest.raises(ValueError, match="QueryRunner is required"):
            RandomTableRecommender(sample_config)
    
    def test_init_with_default_config(self, mock_query_runner):
        """Test initialization with minimal config uses defaults."""
        config = {'recommendation': {'top_k': 5}}
        recommender = RandomTableRecommender(config, mock_query_runner)
        assert recommender.max_sample_size == 1000  # default
        assert recommender.min_sample_size == 10    # default
        assert recommender.exclude_current_results == True  # default
    
    def test_recommend_tuples_empty_input(self, sample_config, mock_query_runner):
        """Test recommend_tuples with empty input."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        result = recommender.recommend_tuples(pd.DataFrame())
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_infer_source_table_from_spec_columns(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test table inference from spectroscopic columns."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        table_info = recommender._infer_source_table(sdss_spec_dataframe)
        
        assert table_info is not None
        assert table_info.name == 'SpecObj'
        assert 'specObjID' in table_info.columns
    
    def test_infer_source_table_from_photo_columns(self, sample_config, mock_query_runner, sdss_photo_dataframe):
        """Test table inference from photometric columns."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        table_info = recommender._infer_source_table(sdss_photo_dataframe)
        
        assert table_info is not None
        assert table_info.name == 'PhotoObj'
        assert 'objID' in table_info.columns
    
    def test_infer_source_table_from_query_context(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test table inference from query context."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # Test with query context
        kwargs = {'current_query': 'SELECT * FROM SpecObj WHERE z > 0.1'}
        table_info = recommender._infer_source_table(sdss_spec_dataframe, **kwargs)
        
        assert table_info is not None
        assert table_info.name == 'SpecObj'
    
    def test_extract_table_from_query(self, sample_config, mock_query_runner):
        """Test extraction of table name from SQL queries."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # Test FROM clause
        query1 = "SELECT * FROM SpecObj WHERE z > 0.1"
        assert recommender._extract_table_from_query(query1) == "SpecObj"
        
        # Test JOIN clause
        query2 = "SELECT s.* FROM PhotoObj p JOIN SpecObj s ON p.objID = s.objID"
        table = recommender._extract_table_from_query(query2)
        assert table in ["PhotoObj", "SpecObj"]  # Either is valid
        
        # Test case insensitive
        query3 = "select * from photoobj where ra > 180"
        assert recommender._extract_table_from_query(query3) == "photoobj"
    
    def test_generate_random_sampling_query(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test random sampling query generation."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        table_info = TableInfo(name="SpecObj", columns=['specObjID', 'objID', 'z', 'zConf'])
        query = recommender._generate_random_sampling_query(table_info, sdss_spec_dataframe)
        
        assert query is not None
        assert "SELECT" in query.upper()
        assert "FROM SpecObj" in query
        assert "ORDER BY RANDOM()" in query
        assert "LIMIT" in query
    
    def test_build_exclusion_clause(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test building exclusion clause for current results."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        table_info = TableInfo(name="SpecObj", columns=['specObjID', 'objID', 'z'])
        exclusion_clause = recommender._build_exclusion_clause(table_info, sdss_spec_dataframe)
        
        # Should build exclusion clause with objID
        assert "WHERE" in exclusion_clause
        assert "NOT IN" in exclusion_clause
        assert "objID" in exclusion_clause
    
    def test_determine_sample_size(self, sample_config, mock_query_runner):
        """Test sample size determination logic."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # Small result set
        size1 = recommender._determine_sample_size(3)
        assert size1 >= recommender.min_sample_size
        assert size1 <= recommender.max_sample_size
        
        # Large result set
        size2 = recommender._determine_sample_size(600)
        assert size2 <= recommender.max_sample_size
        
        # Very large result set should be capped
        size3 = recommender._determine_sample_size(10000)
        assert size3 == recommender.max_sample_size
    
    def test_recommend_tuples_full_workflow(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test full recommendation workflow."""
        # Setup mock query runner to return some random results
        mock_random_results = pd.DataFrame({
            'specObjID': [9001, 9002, 9003],
            'objID': [8001, 8002, 8003],
            'z': [0.4, 0.5, 0.6],
            'zConf': [0.93, 0.96, 0.99],
            'SpecClass': ['QSO', 'GALAXY', 'QSO'],
            'ra': [190.1, 190.5, 191.0],
            'dec': [50.2, 50.5, 50.8]
        })
        
        mock_query_runner.execute_query.return_value = mock_random_results
        
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        result = recommender.recommend_tuples(sdss_spec_dataframe)
        
        # Should return DataFrame with results
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= sample_config['recommendation']['top_k']
        
        # Should have called query runner
        mock_query_runner.execute_query.assert_called_once()
        
        # Check that the query looks reasonable
        called_query = mock_query_runner.execute_query.call_args[0][0]
        assert "SELECT" in called_query.upper()
        assert "FROM" in called_query.upper()
        assert "ORDER BY RANDOM()" in called_query.upper()
    
    def test_recommend_tuples_with_query_context(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test recommendation with query context."""
        mock_random_results = sdss_spec_dataframe.copy()
        mock_random_results['specObjID'] = [9001, 9002, 9003]  # Different IDs
        mock_query_runner.execute_query.return_value = mock_random_results
        
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # Include query context
        kwargs = {'current_query': 'SELECT * FROM SpecObj WHERE z BETWEEN 0.1 AND 0.3'}
        result = recommender.recommend_tuples(sdss_spec_dataframe, **kwargs)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_query_runner.execute_query.assert_called_once()
    
    def test_recommend_tuples_error_handling(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test error handling in recommendation process."""
        # Mock query runner to raise an exception
        mock_query_runner.execute_query.side_effect = Exception("Database error")
        
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        result = recommender.recommend_tuples(sdss_spec_dataframe)
        
        # Should handle error gracefully and return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_post_process_results(self, sample_config, mock_query_runner, sdss_spec_dataframe):
        """Test post-processing of random results."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # Create random results with different column order
        random_results = pd.DataFrame({
            'z': [0.4, 0.5],
            'specObjID': [9001, 9002],
            'objID': [8001, 8002],
            'extra_col': ['A', 'B']
        })
        
        processed = recommender._post_process_results(random_results, sdss_spec_dataframe)
        
        # Should reorder columns to match current results where possible
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 2
        # specObjID should come before z (matching current_results order)
        current_columns = list(sdss_spec_dataframe.columns)
        processed_columns = list(processed.columns)
        
        # Find common columns and check their relative order
        common_cols = [col for col in current_columns if col in processed_columns]
        if len(common_cols) > 1:
            for i in range(len(common_cols) - 1):
                curr_idx = processed_columns.index(common_cols[i])
                next_idx = processed_columns.index(common_cols[i + 1])
                # Order should be preserved for common columns
                assert curr_idx < next_idx
    
    def test_name_method(self, sample_config, mock_query_runner):
        """Test the name method."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        assert recommender.name() == "RandomTableRecommender"
    
    def test_unknown_table_handling(self, sample_config, mock_query_runner):
        """Test handling of unknown table schemas."""
        # DataFrame with columns not in known_tables
        unknown_df = pd.DataFrame({
            'custom_id': [1, 2, 3],
            'custom_field': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        table_info = recommender._infer_source_table(unknown_df)
        
        # Should handle gracefully (might return None or make a guess)
        if table_info:
            assert isinstance(table_info.name, str)
            assert len(table_info.columns) > 0
    
    def test_exclusion_with_no_key_columns(self, sample_config, mock_query_runner):
        """Test exclusion clause when no suitable key columns are available."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # DataFrame without common key columns
        df_no_keys = pd.DataFrame({
            'custom_col1': [1, 2, 3],
            'custom_col2': ['A', 'B', 'C']
        })
        
        table_info = TableInfo(name="CustomTable", columns=['custom_col1', 'custom_col2'])
        exclusion_clause = recommender._build_exclusion_clause(table_info, df_no_keys)
        
        # Should return empty exclusion clause
        assert exclusion_clause == ""
    
    def test_large_exclusion_set_handling(self, sample_config, mock_query_runner):
        """Test handling when exclusion set is too large."""
        recommender = RandomTableRecommender(sample_config, mock_query_runner)
        
        # Create DataFrame with many unique objIDs
        large_df = pd.DataFrame({
            'objID': list(range(1000, 1150)),  # 150 unique IDs > 100 limit
            'value': ['A'] * 150
        })
        
        table_info = TableInfo(name="TestTable", columns=['objID', 'value'])
        exclusion_clause = recommender._build_exclusion_clause(table_info, large_df)
        
        # Should skip exclusion for large sets
        assert exclusion_clause == ""
