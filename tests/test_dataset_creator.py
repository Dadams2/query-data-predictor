import os
import pytest
import pandas as pd
from dotenv import load_dotenv
from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.dataset_creator import DatasetCreator
from pathlib import Path
from query_data_predictor.sdss_json_importer import JsonDataImporter

# Load environment variables from .env file
load_dotenv()

TEST_QUERY = "SELECT bestobjID,z FROM SpecObj WHERE (SpecClass=3 or SpecClass=4) and z between 1.95 and 2 and zConf>0.99"
SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sdss_joined_sample.json"
SAMPLE_CSV_PATH = Path(__file__).parent.parent / "data" / "SQL_workload1.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets"

# Get database connection parameters from environment variables
DB_NAME = os.getenv("PG_DATA")
DB_USER = os.getenv("PG_DATA_USER")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")

class TestDatasetCreator:
    """Test class for DatasetCreator."""
    @pytest.fixture
    def query_runner(self):
        """Create and return a QueryRunner instance with connection params from env vars."""
        runner = QueryRunner(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
        return runner

    @pytest.fixture
    def data_loader(self):
        """Fixture to create and cleanup a DataLoader instance."""
        loader = JsonDataImporter(SAMPLE_DATA_PATH)
        yield loader
        loader.close()

    @pytest.fixture
    def dataset_creator(self, query_runner, data_loader):
        """Fixture to create and cleanup a DatasetCreator instance."""
        creator = DatasetCreator(data_loader=data_loader, query_runner=query_runner, output_dir=OUTPUT_DIR)
        yield creator
        creator.close()
    
    def test_extract_query_features(self, dataset_creator):
        """Test extraction of features from SQL query."""
        query = TEST_QUERY
        features = dataset_creator._extract_query_features(query)
        
        # Basic assertions about extracted features
        assert isinstance(features, dict)
        assert features['query_type'] == 'SELECT'
        assert features['query_length'] > 0
        assert features['token_count'] > 0
        assert features['has_where'] == True
        
    def test_extract_result_features(self, dataset_creator, query_runner):
        """Test extraction of features from query results."""
        # Execute a query to get results
        query_runner.connect()
        try:
            results = query_runner.execute_query(TEST_QUERY)
            # the query runner returns a DataFrame-like object but we can treat it as a DataFrame for feature extraction
            # as usually we will be working with loaded piccle files
            columns = results.columns.to_list()
            
            features = dataset_creator._extract_result_features(columns, results)
            
            # Basic assertions
            assert isinstance(features, dict)
            assert features['result_column_count'] == len(columns)
            assert features['result_row_count'] == len(results)
            
            # Check column type features
            for i in range(len(columns)):
                assert f'col_{i}_type' in features
        finally:
            query_runner.disconnect()
    
    def test_get_result_signature(self, dataset_creator, query_runner):
        """Test generation of result signature."""
        query_runner.connect()
        try:
            results = query_runner.execute_query(TEST_QUERY)
            columns = results.columns
            
            signature = dataset_creator._get_result_signature(columns, results)
            
            # Basic assertions
            assert isinstance(signature, str)
            assert len(signature) > 0
            
            # Signature should contain column names and types
            for col in columns:
                assert col in signature
        finally:
            query_runner.disconnect()
    
    def test_empty_result_signature(self, dataset_creator):
        """Test signature generation for empty results."""
        # Create empty DataFrame with some columns
        empty_df = pd.DataFrame({"col1": [], "col2": []})
        
        signature = dataset_creator._get_result_signature(empty_df.columns, empty_df)
        assert signature == "empty_result"
    
    def test_build_dataset(self, dataset_creator, tmp_path, monkeypatch):
        """Test dataset building with mock data."""
        # Set temporary output directory
        dataset_creator.output_dir = Path(tmp_path)
        
        # Mock get_sessions to return a single session
        monkeypatch.setattr(dataset_creator.data_loader, "get_sessions", lambda: [1])
        
        # Mock get_queries_for_session to return two test queries
        test_queries = [TEST_QUERY, TEST_QUERY]  # Using same query twice for simplicity
        monkeypatch.setattr(
            dataset_creator.data_loader, 
            "get_queries_for_session", 
            lambda _: test_queries
        )
        
        # Build dataset
        dataset_info = dataset_creator.build_dataset(session_id=1)
        
        # Check if dataset was created
        assert len(dataset_info) == 1
        assert os.path.exists(dataset_info["filepath"][0])
    
    def test_extract_result_features_with_duplicate_columns(self, dataset_creator):
        """Test that _extract_result_features handles duplicate column names correctly."""
        # Create a DataFrame with duplicate column names
        data = {
            'id': [1, 2, 3, 4, 5],
            'z': [15.34, 18.99, 15.89, 16.80, 15.51],
            'z.1': [0.07, 2.03, 0.10, 0.11, 0.06],  # pandas will rename duplicate columns with .1, .2, etc.
        }
        df = pd.DataFrame(data)
        
        # Manually recreate a DataFrame with duplicate column names
        # This is tricky in pandas, so we'll use a workaround by renaming after creation
        columns = ['id', 'z', 'z']
        df.columns = columns
        
        # Call the method
        features = dataset_creator._extract_result_features(columns, df)
        
        # Verify correct features were extracted
        assert features['result_column_count'] == 3
        assert features['result_row_count'] == 5
        
        # Check if duplicate columns were handled correctly
        assert 'col_1_0_type' in features  # First 'z' column
        assert 'col_1_1_type' in features  # Second 'z' column
        
        # Check numeric statistics for both z columns
        assert 'col_1_0_min' in features
        assert 'col_1_0_max' in features
        assert 'col_1_1_min' in features
        assert 'col_1_1_max' in features
        
        # Verify the actual values
        assert features['col_1_0_min'] == 15.34
        assert features['col_1_1_min'] == 0.06