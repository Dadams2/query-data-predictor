"""
Feature extraction utilities for SQL queries and results.
"""

import pandas as pd
import sqlparse
from typing import List, Dict, Any


class FeatureExtractor:
    """
    Extracts features from SQL queries and their results for machine learning.
    """
    
    def extract_query_features(self, query: str) -> Dict[str, Any]:
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
    
    def extract_result_features(self, columns: List[str], results: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features from query results.
        
        Args:
            columns: List of column names
            results: Pandas DataFrame with results
            
        Returns:
            Dictionary of features
        """
        # Validate inputs
        if results is None or not isinstance(results, pd.DataFrame):
            raise ValueError("Invalid results: Expected a Pandas DataFrame.")
        if columns is None or not isinstance(columns, list):
            raise ValueError("Invalid columns: Expected a list of column names.")
        
        features = {
            'result_column_count': len(columns),
            'result_row_count': len(results) if not results.empty else 0,
        }
        
        # If we have results, add more features
        if not results.empty:
            # Get column types
            for i, col in enumerate(columns):
                if col in results.columns:
                    col_data = results[col]
                    if isinstance(col_data, pd.DataFrame):
                        for j in range(col_data.shape[1]):
                            series = col_data.iloc[:, j]
                            col_type = series.dtype
                            features[f'col_{i}_{j}_type'] = str(col_type)
                    else:
                        col_type = col_data.dtype
                        features[f'col_{i}_type'] = str(col_type)
            
            # Basic statistics for numeric columns
            for i, col in enumerate(columns):
                try:
                    if col in results.columns:
                        col_data = results[col]
                        if isinstance(col_data, pd.DataFrame):
                            for j in range(col_data.shape[1]):
                                series = col_data.iloc[:, j]
                                if pd.api.types.is_numeric_dtype(series):
                                    features[f'col_{i}_{j}_min'] = series.min()
                                    features[f'col_{i}_{j}_max'] = series.max()
                                    features[f'col_{i}_{j}_mean'] = series.mean()
                                    features[f'col_{i}_{j}_std'] = series.std()
                        elif pd.api.types.is_numeric_dtype(col_data):
                            features[f'col_{i}_min'] = col_data.min()
                            features[f'col_{i}_max'] = col_data.max()
                            features[f'col_{i}_mean'] = col_data.mean()
                            features[f'col_{i}_std'] = col_data.std()
                except Exception as e:
                    # Log the specific error but continue processing
                    print(f"Error processing statistics for column {col}: {e}")
        
        return features
