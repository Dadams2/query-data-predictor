"""
Dummy recommender that simply returns all or a subset of the current query results.
"""

import pandas as pd
from typing import Dict, Any, Optional

from .base_recommender import BaseRecommender


class DummyRecommender(BaseRecommender):
    """
    A dummy recommender that returns all or a configured subset of the current query results.
    This serves as a baseline recommender for comparison with more sophisticated methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dummy recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples by returning the current query results (possibly limited).
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments (ignored)
            
        Returns:
            DataFrame with recommended tuples (same as current results, possibly limited)
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # Apply output limiting based on configuration
        return self._limit_output(current_results.copy())
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "DummyRecommender"
