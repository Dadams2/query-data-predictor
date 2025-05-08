import os
import numpy as np
import polars as pl
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sqlparse
import pickle

from query_data_predictor.sdss_json_importer import DataLoader
from query_data_predictor.query_runner import QueryRunner

class DatasetCreator:
    """
    Create a dataset for training a classification model to predict the tuples
    returned by a user's next query based on the current query and its results.
    """
    
    def __init__(
        self, 
        data_loader: DataLoader = None, 
        query_runner: QueryRunner = None,
        json_path: str = None, 
        output_dir: str = "datasets", 
        dataset_prefix: str = "query_prediction"
    ):
        """
        Initialize the dataset creator with either an existing DataLoader or path to JSON file.
        
        Args:
            data_loader: Optional existing DataLoader instance
            json_path: Optional path to JSON file with query data
            output_dir: Directory to save the dataset files
            dataset_prefix: Prefix for dataset filenames
        """
        if data_loader:
            self.data_loader = data_loader
        elif json_path:
            self.data_loader = DataLoader(json_path)
        else:
            raise ValueError("Either data_loader or json_path must be provided")
        
        self.output_dir = output_dir
        self.dataset_prefix = dataset_prefix
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if query_runner is None:
            # Initialize QueryRunner with default parameters
            raise ValueError("QueryRunner must be provided")
        self.query_runner = query_runner 
    
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
    
    def _extract_result_features(self, columns: List[str], results: pl.DataFrame) -> Dict[str, Any]:
        """
        Extract features from query results.
        
        Args:
            columns: List of column names
            results: Polars DataFrame with results
            
        Returns:
            Dictionary of features
        """
        features = {
            'result_column_count': len(columns),
            'result_row_count': results.height if not results.is_empty() else 0,
        }
        
        # If we have results, add more features
        if not results.is_empty():
            # Get column types
            for i, col in enumerate(columns):
                # Check column for type
                col_type = results.schema[col]
                features[f'col_{i}_type'] = str(col_type)
            
            # Basic statistics for numeric columns
            for i, col in enumerate(columns):
                try:
                    if pl.datatypes.is_numeric(results.schema[col]):
                        features[f'col_{i}_min'] = results[col].min().item()
                        features[f'col_{i}_max'] = results[col].max().item()
                        features[f'col_{i}_mean'] = results[col].mean().item()
                        features[f'col_{i}_std'] = results[col].std().item()
                except:
                    pass
        
        return features
    
    def _get_result_signature(self, columns: List[str], results: pl.DataFrame) -> str:
        """
        Create a signature that represents the structure of the query results.
        
        Args:
            columns: List of column names
            results: Polars DataFrame with results
            
        Returns:
            String signature of the results
        """
        if results.is_empty():
            return "empty_result"
        
        # Convert first few rows to strings to represent the data pattern
        max_rows = min(5, results.height)
        sample_results = results.head(max_rows)
        
        # Create a signature based on column names and data types
        col_types = []
        for col in columns:
            # Get types of values in this column
            col_dtype = str(results.schema[col])
            col_types.append(f"{col}:{col_dtype}")
        
        return "|".join(col_types)
    
    def build_dataset(self, session_id: Optional[int] = None):
        """
        Build datasets for training a query result prediction model.
        
        Args:
            session_id: Optional. If provided, build dataset only for this session.
                        Otherwise, build datasets for all sessions.
        
        Returns:
            Dictionary mapping session_ids to dataset files
        """
        datasets_info = {}
        
        # Connect QueryRunner
        self.query_runner.connect()
        
        try:
            # Get all sessions or filter for a specific one
            sessions = self.data_loader.get_sessions()
            if session_id is not None:
                sessions = [session_id] if session_id in sessions else None
            if sessions is None:
                print(f"Session {session_id} not found")
                return datasets_info

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
                        curr_columns = curr_results.columns
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
                    }
                    all_features.update(query_features)
                    all_features.update(result_features)
                    
                    dataset_rows.append(all_features)
                
                if dataset_rows:
                    # Create dataset DataFrame with Polars
                    dataset = pl.DataFrame(dataset_rows)
                    
                    # Save dataset for this session
                    output_path = os.path.join(self.output_dir, f"{self.dataset_prefix}_session_{current_session_id}.pkl")
                    with open(output_path, 'wb') as f:
                        pickle.dump(dataset, f)
                    
                    datasets_info[current_session_id] = {
                        'file_path': output_path,
                        'samples': len(dataset_rows)
                    }
                    
                    print(f"Dataset for session {current_session_id} saved to {output_path} with {len(dataset_rows)} samples")
        
        finally:
            # Disconnect QueryRunner
            self.query_runner.disconnect()
        
        return datasets_info
    
    # def prepare_train_test_split(self, dataset: pl.DataFrame, test_size: float = 0.2, random_state: int = 42):
    #     """
    #     Prepare training and test datasets for classification.
        
    #     Args:
    #         dataset: Dataset DataFrame (Polars)
    #         test_size: Proportion of data to use for testing
    #         random_state: Random seed for reproducibility
            
    #     Returns:
    #         X_train, X_test, y_train, y_test
    #     """
    #     # Text features from the query
    #     queries = dataset.get_column('current_query').to_list()
    #     vectorizer = TfidfVectorizer(max_features=100)
    #     query_vectors = vectorizer.fit_transform(queries)
        
    #     # Create feature matrix
    #     feature_cols = [col for col in dataset.columns if col not in 
    #                    ['session_id', 'current_query', 'next_query', 'next_result_signature']]
        
    #     # Convert Polars DataFrame to pandas for sklearn compatibility
    #     X_features_pd = dataset.select(feature_cols).to_pandas()
        
    #     # Convert categorical features to numeric
    #     for col in X_features_pd.select_dtypes(include=['object']):
    #         X_features_pd[col] = X_features_pd[col].astype('category').cat.codes
        
    #     # Combine text features and tabular features
    #     X = np.hstack([query_vectors.toarray(), X_features_pd.values])
        
    #     # Target is the next result signature
    #     y = dataset.get_column('next_result_signature').to_list()
        
    #     # Split into train and test sets
    #     return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'query_runner'):
            self.query_runner.disconnect()
        
        if hasattr(self, 'data_loader') and self.data_loader:
            self.data_loader.close()