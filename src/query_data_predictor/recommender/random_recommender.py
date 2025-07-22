"""
Random recommender that returns a random selection of tuples.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base_recommender import BaseRecommender


class RandomRecommender(BaseRecommender):
    """
    A random recommender that returns a random selection of tuples from the current query results.
    This serves as a baseline for comparison with more sophisticated methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the random recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        
        # Set random seed for reproducibility if specified
        random_config = self.config.get('random', {})
        self.random_seed = random_config.get('random_seed', 42)
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples by randomly selecting from the current query results.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
                - random_seed: Override the random seed
            
        Returns:
            DataFrame with randomly selected tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # Use provided random seed or default
        random_seed = kwargs.get('random_seed', self.random_seed)
        
        # Determine how many tuples to select
        target_size = self._determine_output_size(len(current_results))
        
        # Ensure we don't try to select more tuples than available
        n_select = min(target_size, len(current_results))
        
        if n_select >= len(current_results):
            # If we need all or more tuples than available, return all (possibly shuffled)
            result = current_results.copy()
            if random_seed is not None:
                result = result.sample(frac=1, random_state=random_seed)
            return result
        
        # Randomly sample the required number of tuples
        return current_results.sample(n=n_select, random_state=random_seed)
    
    def recommend_random_tuples(self, current_results: pd.DataFrame, n_tuples: int, **kwargs) -> pd.DataFrame:
        """
        Recommend a specific number of random tuples.
        
        Args:
            current_results: DataFrame with the current query's results
            n_tuples: Exact number of tuples to recommend
            **kwargs: Additional keyword arguments
                - random_seed: Override the random seed
            
        Returns:
            DataFrame with randomly selected tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty or n_tuples <= 0:
            return pd.DataFrame()
        
        # Use provided random seed or default
        random_seed = kwargs.get('random_seed', self.random_seed)
        
        # Ensure we don't try to select more tuples than available
        n_select = min(n_tuples, len(current_results))
        
        if n_select >= len(current_results):
            # If we need all or more tuples than available, return all (possibly shuffled)
            result = current_results.copy()
            if random_seed is not None:
                result = result.sample(frac=1, random_state=random_seed)
            return result
        
        # Randomly sample the required number of tuples
        return current_results.sample(n=n_select, random_state=random_seed)
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducible results.
        
        Args:
            seed: Random seed value
        """
        self.random_seed = seed
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "RandomRecommender"
