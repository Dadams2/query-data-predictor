import os
import pytest
import pandas as pd
from dotenv import load_dotenv
from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.dataset_creator import DatasetCreator
from pathlib import Path
from query_data_predictor.sdss_json_importer import JsonDataImporter
import pickle

# Load environment variables from .env file
load_dotenv()

TEST_QUERY = "SELECT bestobjID,z FROM SpecObj WHERE (SpecClass=3 or SpecClass=4) and z between 1.95 and 2 and zConf>0.99"
SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sdss_joined_sample.json"
SAMPLE_CSV_PATH = Path(__file__).parent.parent / "data" / "SQL_workload1.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "query_results"

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
        creator = DatasetCreator(
            data_loader=data_loader, 
            query_runner=query_runner, 
            output_dir=OUTPUT_DIR,
            results_dir=RESULTS_DIR
        )
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
            # Convert polars DataFrame to pandas if needed
            if not isinstance(results, pd.DataFrame):
                results = results.to_pandas()
            columns = list(results.columns)
            
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
            # Convert polars DataFrame to pandas if needed
            if not isinstance(results, pd.DataFrame):
                results = results.to_pandas()
            columns = list(results.columns)
            
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
        # Set temporary output and results directories
        temp_output_dir = tmp_path / "datasets"
        temp_results_dir = tmp_path / "results"
        os.makedirs(temp_output_dir, exist_ok=True)
        os.makedirs(temp_results_dir, exist_ok=True)
        
        dataset_creator.output_dir = str(temp_output_dir)
        dataset_creator.results_dir = str(temp_results_dir)
        
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
        
        # Load the created dataset and check for results_filepath
        with open(dataset_info["filepath"][0], 'rb') as f:
            dataset = pickle.load(f)
        
        # Check if dataset contains the results_filepath field and the files exist
        assert 'results_filepath' in dataset.columns
        for filepath in dataset['results_filepath']:
            assert os.path.exists(filepath)
            
            # Verify that the result file contains a pandas DataFrame
            with open(filepath, 'rb') as f:
                result_data = pickle.load(f)
                assert isinstance(result_data, pd.DataFrame)