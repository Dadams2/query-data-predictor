import os
import pandas as pd
from typing import List, Dict, Optional, Any
import sqlparse
import pickle

from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.importer import DataImporter
from query_data_predictor.sdss_json_importer import JsonDataImporter

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
        
        self.output_dir = output_dir
        self.dataset_prefix = dataset_prefix
        self.results_dir = results_dir  # Store the results directory path
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)  # Create results directory
        
        if query_runner is None:
            # Initialize QueryRunner with default parameters
            raise ValueError("QueryRunner must be provided")
        self.query_runner = query_runner 
        self.data_loader.load_data()
        self.query_runner.connect()
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """
        Extract features from a SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary of features
        """
        # Parse the query
        parsed = sqlparse.parse(query)[0]
        
        # Get query type (SELECT, INSERT, etc.)
        query_type = parsed.get_type()
        
        # Basic features
        features = {
            'query_type': query_type,
            'query_length': len(query),
            'token_count': len(parsed.tokens),
        }
        
        if query_type == 'SELECT':
            features['has_join'] = 'JOIN' in query.upper()
            features['has_where'] = 'WHERE' in query.upper()
            features['has_group_by'] = 'GROUP BY' in query.upper()
            features['has_order_by'] = 'ORDER BY' in query.upper()
            agg_funcs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
            for func in agg_funcs:
                features[f'has_{func.lower()}'] = func in query.upper()
        
        return features
    
    def _extract_result_features(self, columns: List[str], results: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features from query results.
        
        Args:
            columns: List of column names
            results: Pandas DataFrame with results
            
        Returns:
            Dictionary of features
        """
        features = {
            'result_column_count': len(columns),
            'result_row_count': len(results) if not results.empty else 0,
        }
        
        # If we have results, add more features
        if not results.empty:
            # Get column types
            for i, col in enumerate(columns):
                # Check column for type
                col_type = results[col].dtype
                features[f'col_{i}_type'] = str(col_type)
            
            # Basic statistics for numeric columns
            for i, col in enumerate(columns):
                try:
                    if pd.api.types.is_numeric_dtype(results[col]):
                        features[f'col_{i}_min'] = results[col].min()
                        features[f'col_{i}_max'] = results[col].max()
                        features[f'col_{i}_mean'] = results[col].mean()
                        features[f'col_{i}_std'] = results[col].std()
                except:
                    pass
        
        return features
    
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
                print(f"Session {session_id} not found")
                return pd.DataFrame(columns=['session_id', 'filepath'])

            for current_session_id in sessions:
                queries = self.data_loader.get_queries_for_session(current_session_id)
                print(f"Processing session {current_session_id} with {len(queries)} queries")
                dataset_rows = []

                # Process each query and its next query
                for i in range(len(queries)):
                    current_query = queries[i]

                    # Skip if current or next query is empty or not a SELECT
                    if not current_query:
                        continue
                    if not current_query.upper().startswith('SELECT'):
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
                        results_filepath = os.path.join(self.results_dir, results_filename)
                        
                        # Save results to file
                        with open(results_filepath, 'wb') as f:
                            pickle.dump(curr_results, f)
                            
                    except Exception as e:
                        # pass over errors for now
                        print(f"Error executing current query: {e}")
                        continue

                    # Extract features and target
                    query_features = self._extract_query_features(current_query)
                    result_features = self._extract_result_features(curr_columns, curr_results)

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
                    output_path = os.path.join(self.output_dir, f"{self.dataset_prefix}_session_{current_session_id}.pkl")
                    with open(output_path, 'wb') as f:
                        pickle.dump(dataset, f)

                    # Add metadata row for the session
                    metadata_rows.append({
                        'session_id': current_session_id,
                        'filepath': output_path
                    })

                    print(f"Dataset for session {current_session_id} saved to {output_path} with {len(dataset_rows)} samples")

        finally:
            # Disconnect QueryRunner
            self.query_runner.disconnect()

        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)

        # Save metadata to CSV
        metadata_csv_path = os.path.join(self.output_dir, 'metadata.csv')
        metadata_df.to_csv(metadata_csv_path, index=False)
        print(f"Metadata saved to {metadata_csv_path}")

        return metadata_df
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'query_runner'):
            self.query_runner.disconnect()
        
        if hasattr(self, 'data_loader') and self.data_loader:
            self.data_loader.close()