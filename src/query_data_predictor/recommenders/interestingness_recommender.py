"""
Interestingness recommender that wraps the existing TupleRecommender functionality.
"""

import pandas as pd
from typing import Dict, Any, Optional

from .base_recommender import BaseRecommender
from ..tuple_recommender import TupleRecommender


class InterestingnessRecommender(BaseRecommender):
    """
    An interestingness recommender that uses association rules and summaries
    to recommend tuples based on interestingness measures.
    
    This is a wrapper around the existing TupleRecommender class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the interestingness recommender.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        super().__init__(config)
        self.tuple_recommender = TupleRecommender(config)
    
    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using interestingness measures (association rules and summaries).
        
        Args:
            current_results: DataFrame with the current query's results
            **kwargs: Additional keyword arguments
                - top_k: Override the number of tuples to recommend
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty:
            return pd.DataFrame()
        
        # Get top_k from kwargs or configuration
        top_k = kwargs.get('top_k')
        if top_k is None:
            top_k = self._determine_output_size(len(current_results))
        
        # Use the existing tuple recommender
        try:
            recommendations = self.tuple_recommender.recommend_tuples(current_results, top_k=top_k)
            
            # Apply additional output limiting based on our configuration
            return self._limit_output(recommendations)
            
        except Exception as e:
            # If the tuple recommender fails, return empty DataFrame
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data using the underlying tuple recommender.
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        return self.tuple_recommender.preprocess_data(df)
    
    def mine_association_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mine association rules using the underlying tuple recommender.
        
        Args:
            df: Input DataFrame (should be preprocessed/discretized)
            
        Returns:
            DataFrame with association rules
        """
        return self.tuple_recommender.mine_association_rules(df)
    
    def generate_summaries(self, df: pd.DataFrame) -> list:
        """
        Generate summaries using the underlying tuple recommender.
        
        Args:
            df: Input DataFrame (should be preprocessed/discretized)
            
        Returns:
            List of summary dictionaries
        """
        return self.tuple_recommender.generate_summaries(df)
    
    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "InterestingnessRecommender"
