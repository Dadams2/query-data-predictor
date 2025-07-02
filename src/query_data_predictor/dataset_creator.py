import pandas as pd
import base64
from typing import List, Dict, Optional, Any
import sqlparse
import pickle
from pathlib import Path
import logging

from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.importer import DataImporter
from query_data_predictor.sdss_json_importer import JsonDataImporter
from query_data_predictor.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class DatasetCreator:
    """
    Create a dataset for training a classification model to predict the tuples
    returned by a user's next query based on the current query and its results.
    """
    
    def __init__(
        self, 
        data_loader: DataImporter = None, 
        query_runner: QueryRunner = None,
        json_path: str = None, 
        output_dir: str = "datasets", 
        dataset_prefix: str = "query_prediction",
        results_dir: str = "query_results"  # New parameter for results directory
    ):
        """
        Initialize the dataset creator with either an existing DataLoader or path to JSON file.
        
        Args:
            data_loader: Optional existing DataLoader instance
            json_path: Optional path to JSON file with query data
            output_dir: Directory to save the dataset files
            dataset_prefix: Prefix for dataset filenames
            results_dir: Directory to save query results files
        """
        if data_loader:
            self.data_loader = data_loader
        elif json_path:
            self.data_loader = JsonDataImporter(json_path)
        else:
            raise ValueError("Either data_loader or json_path must be provided")
        
        self.output_dir = Path(output_dir)
        self.dataset_prefix = dataset_prefix
        self.results_dir = Path(results_dir)  # Store the results directory path
        
        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)  # Create results directory
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        if query_runner is None:
            # Initialize QueryRunner with default parameters
            raise ValueError("QueryRunner must be provided")
        self.query_runner = query_runner 
        self.data_loader.load_data()
        self.query_runner.connect()
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """
        Extract features from a SQL query using the FeatureExtractor.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary of features
        """
        return self.feature_extractor.extract_query_features(query)
    
    def _extract_result_features(self, columns: List[str], results: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features from query results using the FeatureExtractor.
        
        Args:
            columns: List of column names
            results: Pandas DataFrame with results
            
        Returns:
            Dictionary of features
        """
        return self.feature_extractor.extract_result_features(columns, results)
    
    def _get_result_signature(self, columns: List[str], results: pd.DataFrame) -> str:
        """
        Create a signature that represents the structure of the query results.
        
        Args:
            columns: List of column names
            results: Pandas DataFrame with results
            
        Returns:
            String signature of the results
        """
        if results.empty:
            return "empty_result"
        
        # Convert first few rows to strings to represent the data pattern
        max_rows = min(5, len(results))
        sample_results = results.head(max_rows)
        
        # Create a signature based on column names and data types
        col_types = []
        for col in columns:
            # Get types of values in this column
            col_dtype = str(results[col].dtype)
            col_types.append(f"{col}:{col_dtype}")
        
        return "|".join(col_types)
    
    def build_dataset(self, session_id: Optional[int] = None):
        """
        Build datasets for training a query result prediction model.

        Args:
            session_id: Optional. If provided, build dataset only for this session.
                        Otherwise, build datasets for all sessions.

        Returns:
            Pandas DataFrame containing metadata with columns 'session_id' and 'filepath'.
        """
        metadata_rows = []

        try:
            # Get all sessions or filter for a specific one
            sessions = self.data_loader.get_sessions()
            if session_id is not None:
                sessions = [session_id] if session_id in sessions else None
            if sessions is None:
                logger.warning(f"Session {session_id} not found")
                return pd.DataFrame(columns=['session_id', 'filepath'])

            for current_session_id in sessions:
                queries = self.data_loader.get_queries_for_session(current_session_id)
                logger.info(f"Processing session {current_session_id} with {len(queries)} queries")
                dataset_rows = []

                # Process each query and its next query
                for i in range(len(queries)):
                    current_query = queries[i]

                    # Skip if current or next query is empty or not a SELECT
                    if not current_query:
                        continue
                    
                    # Use sqlparse to reliably check query type
                    try:
                        parsed_query = sqlparse.parse(current_query)[0]
                        query_type = parsed_query.get_type()
                        if query_type != 'SELECT':
                            continue
                    except (IndexError, AttributeError):
                        # If parsing fails, fall back to simple check
                        if not current_query.upper().strip().startswith('SELECT'):
                            continue

                    # Execute current query using QueryRunner
                    try:
                        curr_results = self.query_runner.execute_query(current_query)
                        # Convert polars DataFrame to pandas if needed
                        if not isinstance(curr_results, pd.DataFrame):
                            curr_results = curr_results.to_pandas()
                        curr_columns = list(curr_results.columns)
                        
                        # Save the query results to a file
                        results_filename = f"results_session_{current_session_id}_query_{i}.pkl"
                        results_filepath = self.results_dir / results_filename
                        
                        # Process the DataFrame to handle binary/image data before pickling
                        pickle_safe_results = self._process_for_pickling(curr_results)
                        
                        # Save processed results to file
                        with open(results_filepath, 'wb') as f:
                            pickle.dump(pickle_safe_results, f)
                            
                    except Exception as e:
                        # Log the error details
                        logger.error(f"Error executing current query: {e}")
                        continue
                    
                    try:
                        # Extract features and target
                        query_features = self._extract_query_features(current_query)
                        result_features = self._extract_result_features(curr_columns, curr_results)
                    except Exception as e:
                        from IPython import embed
                        embed()
                        # Log the error details
                        logger.error(f"Error extracting features: {e}")
                        continue

                    # Combine all features
                    all_features = {
                        'session_id': current_session_id,
                        'query_position': i,
                        'current_query': current_query,
                        'results_filepath': results_filepath  # Add the path to the results file
                    }
                    all_features.update(query_features)
                    all_features.update(result_features)

                    dataset_rows.append(all_features)

                if dataset_rows:
                    # Create dataset DataFrame with Pandas instead of Polars
                    dataset = pd.DataFrame(dataset_rows)

                    # Save dataset for this session
                    output_path = self.output_dir / f"{self.dataset_prefix}_session_{current_session_id}.pkl"
                    with open(output_path, 'wb') as f:
                        pickle.dump(dataset, f)

                    # Add metadata row for the session
                    metadata_rows.append({
                        'session_id': current_session_id,
                        'filepath': output_path
                    })

                    logger.info(f"Dataset for session {current_session_id} saved to {output_path} with {len(dataset_rows)} samples")

        finally:
            # Disconnect QueryRunner
            self.query_runner.disconnect()

        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)

        # Save metadata to CSV
        metadata_csv_path = self.output_dir / 'metadata.csv'
        metadata_df.to_csv(metadata_csv_path, index=False)
        logger.info(f"Metadata saved to {metadata_csv_path}")

        return metadata_df
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'query_runner'):
            self.query_runner.disconnect()
        
        if hasattr(self, 'data_loader') and self.data_loader:
            self.data_loader.close()
    
    def _is_binary_data(self, series):
        """Check if a pandas Series likely contains binary data"""
        # Check a sample of the data to determine if it's binary
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
            
        # Check the first non-null value
        sample = non_null.iloc[0]
        
        # Check if it's bytes, memoryview, or a list of bytes
        if isinstance(sample, (bytes, memoryview)):
            return True
        elif isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], bytes):
            return True
        return False
    
    def _process_for_pickling(self, df):
        """Process a DataFrame to make it picklable by converting binary data to base64"""
        result_df = df.copy()
        
        for col in df.columns:
            if self._is_binary_data(df[col]):
                # Convert binary data to base64 encoded strings
                result_df[col] = df[col].apply(
                    lambda x: base64.b64encode(x).decode('utf-8') if isinstance(x, (bytes, memoryview))
                    else [base64.b64encode(b).decode('utf-8') for b in x] if isinstance(x, list) 
                    else x
                )
                
        return result_df