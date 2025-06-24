"""
Base recommender class for query result prediction.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from enum import Enum


class RecommendationMode(Enum):
    """Enumeration for different recommendation modes."""
    TOP_K = "top_k"
    TOP_QUARTILE = "top_quartile"
    PERCENTAGE = "percentage"


class BaseRecommender(ABC):
    """
    Abstract base class for all recommenders.
    
    All recommenders should inherit from this class and implement the recommend_tuples method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the recommender with configuration parameters.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        self.config = config
    
    @abstractmethod
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples for the next query based on the current query's results.
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        pass
    
    def _determine_output_size(self, total_tuples: int, config_key: str = 'recommendation') -> int:
        """
        Determine the number of tuples to return based on configuration.
        
        Args:
            total_tuples: Total number of available tuples
            config_key: Configuration key to look for settings (default: 'recommendation')
            
        Returns:
            Number of tuples to return
        """
        rec_config = self.config.get(config_key, {})
        mode = rec_config.get('mode', RecommendationMode.TOP_K.value)
        
        if mode == RecommendationMode.TOP_K.value:
            return rec_config.get('top_k', 10)
        elif mode == RecommendationMode.TOP_QUARTILE.value:
            return max(1, total_tuples // 4)
        elif mode == RecommendationMode.PERCENTAGE.value:
            percentage = rec_config.get('percentage', 0.1)
            return max(1, int(total_tuples * percentage))
        else:
            # Default to top_k
            return rec_config.get('top_k', 10)
    
    def _limit_output(self, df: pd.DataFrame, config_key: str = 'recommendation') -> pd.DataFrame:
        """
        Limit the output DataFrame to the configured number of tuples.
        
        Args:
            df: Input DataFrame to limit
            config_key: Configuration key to look for settings (default: 'recommendation')
            
        Returns:
            Limited DataFrame
        """
        if df.empty:
            return df
            
        target_size = self._determine_output_size(len(df), config_key)
        
        if len(df) <= target_size:
            return df
        
        return df.head(target_size)
    
    def _validate_input(self, current_results: pd.DataFrame) -> None:
        """
        Validate input DataFrame.
        
        Args:
            current_results: DataFrame to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(current_results, pd.DataFrame):
            raise ValueError("current_results must be a pandas DataFrame")
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return self.__class__.__name__
