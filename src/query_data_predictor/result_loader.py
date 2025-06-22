"""
A module for loading and processing query result data from the sessions 
and preparing them for the experimentation framework.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from query_data_predictor.dataloader import DataLoader


class ResultLoader:
    """
    Class to load and process the actual result data for queries
    to be used in the experimentation framework.
    """
    
    def __init__(self, dataloader: DataLoader):
        """
        Initialize with a DataLoader instance that provides access to session metadata.
        
        Args:
            dataloader: A DataLoader instance that has access to session metadata
        """
        self.dataloader = dataloader
        
    def get_result_data(self, session_id: str, query_id: str) -> pd.DataFrame:
        """
        Get the actual result data for a specific query in a session.
        
        Args:
            session_id: The ID of the session
            query_id: The ID of the query
            
        Returns:
            The query results as a pandas DataFrame
        """
        # Get session data which includes results_filepath
        return self.dataloader.get_results_for_query(session_id, query_id)
    
    def get_consecutive_query_results(self, session_id: str, query_id: str, gap: int = 1) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get the results for the current query and a future query in the session with a specified gap.
        
        Args:
            session_id: The ID of the session
            query_id: The ID of the current query
            gap: Number of queries to skip (default: 1, which means adjacent query)
            
        Returns:
            A tuple of (current_results, future_results) where future_results may be None
            if there is no future query at the specified gap
        """
        # Get the current query results
        current_results = self.get_result_data(session_id, query_id)
        
        # Try to get the future query's results
        try:
            future_query_id = int(query_id) + gap  # Apply the gap to get the future query ID
            future_results = self.get_result_data(session_id, str(future_query_id))
            return current_results, future_results
        except (ValueError, KeyError):
            # No future query or future query has no results
            return current_results, None
    
        
    def get_query_pairs_with_gap(self, session_id: str, gap: int = 1) -> List[Tuple[str, str]]:
        """
        Get all valid pairs of query IDs with a specified gap between them for a session.
        
        Args:
            session_id: The ID of the session
            gap: Number of queries to skip (default: 1, which means adjacent queries)
            
        Returns:
            List of tuples (start_query_id, target_query_id)
        """
        # Get all results for the session
        all_results = self.dataloader.get_results_for_session(session_id)
        
        if not all_results:
            return []
            
        # Get sorted query IDs as integers
        query_ids = sorted([int(qid) for qid in all_results.keys()])
        
        pairs = []
        
        # Generate all possible pairs with the specified gap
        for i in range(len(query_ids) - gap):
            start_id = query_ids[i]
            target_id = query_ids[i + gap]
            
            start_query_id = str(start_id)
            target_query_id = str(target_id)
            
            # Get the result data for both queries
            start_results = all_results[start_query_id]
            target_results = all_results[target_query_id]
            
            # Only include pairs where both queries have non-empty results
            if not start_results.empty and not target_results.empty:
                pairs.append((start_query_id, target_query_id))
                
        return pairs
    
    def get_results_with_gap(self, session_id: str, gap: int = 1) -> List[Tuple[str, str, pd.DataFrame, pd.DataFrame]]:
        """
        Get all valid pairs of queries with their results for a specified gap.
        
        Args:
            session_id: The ID of the session
            gap: Number of queries to skip (default: 1)
            
        Returns:
            List of tuples (start_query_id, target_query_id, start_results, target_results)
        """
        # Get all pairs of query IDs
        pairs = self.get_query_pairs_with_gap(session_id, gap)
        
        if not pairs:
            return []
            
        # Get results for each pair
        all_results = self.get_all_session_results(session_id)
        
        result_pairs = []
        for start_id, target_id in pairs:
            start_results = all_results[start_id]
            target_results = all_results[target_id]
            
            result_pairs.append((start_id, target_id, start_results, target_results))
        
        return result_pairs
        
    def get_all_session_gaps(self, session_id: str, max_gap: int = 3) -> Dict[int, List[Tuple[str, str, pd.DataFrame, pd.DataFrame]]]:
        """
        Get all valid query pairs with different gaps for a session, up to a maximum gap.
        
        Args:
            session_id: The ID of the session
            max_gap: Maximum number of queries to skip (default: 3)
            
        Returns:
            Dictionary mapping gap size to list of query pairs with their results
        """
        results = {}
        
        # Get pairs for each gap size from 1 to max_gap
        for gap in range(1, max_gap + 1):
            pairs = self.get_results_with_gap(session_id, gap)
            if pairs:  # Only include non-empty results
                results[gap] = pairs
                
        return results
    
    def get_result(self, session_id: str, query_position: int) -> Optional[pd.DataFrame]:
        """
        Get the result for a specific query position in a session.
        
        Args:
            session_id: The ID of the session
            query_position: The position of the query in the session
            
        Returns:
            The query results as a DataFrame, or None if not found
        """
        try:
            return self.get_result_data(session_id, query_position)
        except Exception:
            return None
