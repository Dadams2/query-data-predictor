"""
Tuple recommender for predicting results of subsequent queries in a session.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Union, Optional
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from query_data_predictor.discretizer import Discretizer
from query_data_predictor.association_interestingness import AssociationEvaluator
from query_data_predictor.summary_interestingness import SummaryEvaluator


class TupleRecommender:
    """
    Recommender for predicting tuples that will appear in subsequent queries.
    Uses interestingness-based scoring combining association rules and summaries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tuple recommender with configuration parameters.
        
        Args:
            config: Configuration dictionary with settings for the recommendation process
        """
        self.config = config
        self.discretizer = None
        self.association_evaluator = None
        self.summary_evaluator = None
        
        # Initialize discretizer based on configuration
        if config.get('discretization', {}).get('enabled', True):
            disc_config = config.get('discretization', {})
            self.discretizer = Discretizer(
                method=disc_config.get('method', 'equal_width'),
                bins=disc_config.get('bins', 5),
                save_path=disc_config.get('params_path') if disc_config.get('save_params', True) else None
            )
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by applying discretization if enabled.
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        if self.discretizer and self.config.get('discretization', {}).get('enabled', True):
            return self.discretizer.discretize_dataframe(df)
        return df
    
    def compute_frequent_itemsets(self, df: pd.DataFrame, min_support: Optional[float] = None) -> pd.DataFrame:
        """
        Compute frequent itemsets from the DataFrame using FP-Growth.
        
        Args:
            df: Input DataFrame (should be preprocessed/discretized)
            min_support: Minimum support threshold (uses config if not provided)
            
        Returns:
            DataFrame with frequent itemsets
        """
        # Get configuration for association rules if min_support not provided
        if min_support is None:
            assoc_config = self.config.get('association_rules', {})
            min_support = assoc_config.get('min_support', 0.1)
        
        # Convert DataFrame to encoded format for fpgrowth
        encoded_df, attributes = self._prepare_data_for_fpgrowth(df)
        
        if encoded_df.empty:
            return pd.DataFrame()
        
        try:
            # Print number of transactions and attributes for debugging
            print(f"Number of transactions: {len(encoded_df)}, Number of attributes: {len(attributes)}")

            # Mine frequent itemsets using FP-Growth
            frequent_itemsets = fpgrowth(
                encoded_df, 
                min_support=min_support, 
                use_colnames=True,
                verbose=0  # Suppress verbose output for performance
            )
            return frequent_itemsets
        except Exception as e:
            # Return empty DataFrame if FP-Growth fails
            print(f"Warning: FP-Growth failed with error: {e}")
            return pd.DataFrame()
    
    def recommend_tuples(self, current_results: pd.DataFrame, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Recommend tuples for the next query based on interestingness scores.
        
        This method implements the core recommendation logic by:
        1. Preprocessing the input data (discretization if enabled)
        2. Converting to transaction format suitable for pattern mining
        3. Mining frequent itemsets using FP-Growth
        4. Computing interestingness scores using association rules and/or summaries
        5. Ranking all tuples by their interestingness scores
        6. Returning the top-k most interesting tuples
        
        Args:
            current_results: DataFrame with the current query's results
            top_k: Number of top tuples to recommend (overrides config value if provided)
            
        Returns:
            DataFrame with recommended tuples, ranked by interestingness score
        """
        # Use configuration top_k if not provided
        if top_k is None:
            top_k = self.config.get('recommendation', {}).get('top_k', 10)
        
        # Preprocess data (discretization if enabled)
        processed_df = self.preprocess_data(current_results)
        
        if processed_df.empty or len(processed_df) < 2:
            return pd.DataFrame()
        
        # Convert to transaction format for pattern mining
        encoded_df, attributes = self._prepare_data_for_fpgrowth(processed_df)
        
        if encoded_df.empty:
            return pd.DataFrame()
        
        # Compute frequent itemsets
        frequent_itemsets = self.compute_frequent_itemsets(processed_df)
        
        if frequent_itemsets.empty:
            # No patterns found, return top tuples by frequency
            return processed_df.head(top_k)
        
        # Calculate interestingness scores for all tuples
        scores = self._compute_tuple_interestingness_scores(processed_df, frequent_itemsets)
        
        # Add scores to the dataframe
        scored_df = processed_df.copy()
        scored_df['interestingness_score'] = scores
        
        # Sort by interestingness score and return top-k
        scored_df = scored_df.sort_values('interestingness_score', ascending=False)
        
        # Apply score threshold if configured
        score_threshold = self.config.get('recommendation', {}).get('score_threshold', 0.0)
        if score_threshold > 0:
            scored_df = scored_df[scored_df['interestingness_score'] >= score_threshold]
        
        # Return top-k tuples (excluding the score column)
        result_df = scored_df.head(top_k).drop(columns=['interestingness_score'])
        
        return result_df
    
    def _compute_tuple_interestingness_scores(self, df: pd.DataFrame, frequent_itemsets: pd.DataFrame) -> pd.Series:
        """
        Compute interestingness scores for all tuples using association rules and/or summaries.
        
        This method combines scores from different interestingness measures based on configuration:
        - Association rules: Use rule-based interestingness (confidence, lift, etc.)
        - Summaries: Use summary-based interestingness (variance, shannon entropy, etc.)
        - Hybrid: Combine both approaches with configurable weights
        
        Args:
            df: Preprocessed DataFrame with tuples
            frequent_itemsets: DataFrame with frequent itemsets from FP-Growth
            
        Returns:
            Series with interestingness scores for each tuple
        """
        # Get configuration for scoring
        method = self.config.get('recommendation', {}).get('method', 'hybrid')
        alpha = 0.5  # Weight for association rules
        beta = 0.5   # Weight for summaries
        
        # Initialize scores
        rule_scores = pd.Series([0.0] * len(df), index=df.index)
        summary_scores = pd.Series([0.0] * len(df), index=df.index)
        
        # Compute association rule scores if enabled
        if method in ['association_rules', 'hybrid'] and self.config.get('association_rules', {}).get('enabled', True):
            try:
                rule_scores = self._compute_association_rule_scores(df, frequent_itemsets)
            except Exception as e:
                print(f"Warning: Association rule scoring failed: {e}")
        
        # Compute summary scores if enabled  
        if method in ['summaries', 'hybrid'] and self.config.get('summaries', {}).get('enabled', True):
            try:
                summary_scores = self._compute_summary_scores(df, frequent_itemsets)
            except Exception as e:
                print(f"Warning: Summary scoring failed: {e}")
        
        # Combine scores based on method
        if method == 'association_rules':
            return rule_scores
        elif method == 'summaries':
            return summary_scores
        else:  # hybrid
            return alpha * rule_scores + beta * summary_scores
    
    def _compute_association_rule_scores(self, df: pd.DataFrame, frequent_itemsets: pd.DataFrame) -> pd.Series:
        """
        Compute interestingness scores using association rules.
        
        Args:
            df: Preprocessed DataFrame  
            frequent_itemsets: DataFrame with frequent itemsets
            
        Returns:
            Series with association rule-based scores
        """
        # Create association evaluator
        self.association_evaluator = AssociationEvaluator(frequent_itemsets)
        
        # Mine association rules with config parameters
        assoc_config = self.config.get('association_rules', {})
        rules_df = self.association_evaluator.mine_association_rules(
            metric=assoc_config.get('metric', 'confidence'),
            min_confidence=assoc_config.get('min_threshold', 0.7)
        )
        
        if rules_df.empty:
            return pd.Series([0.0] * len(df), index=df.index)
        
        # Get measure columns for scoring
        interestingness_config = self.config.get('interestingness', {})
        measure_columns = ['confidence', 'lift', 'leverage']  # Default measures
        
        # Evaluate tuples using association rules
        scores = self.association_evaluator.evaulate_df(
            df, 
            rules_df, 
            measure_columns=measure_columns,
            parallel=len(df) > 1000
        )
        
        return scores
    
    def _compute_summary_scores(self, df: pd.DataFrame, frequent_itemsets: pd.DataFrame) -> pd.Series:
        """
        Compute interestingness scores using summaries.
        
        Args:
            df: Preprocessed DataFrame
            frequent_itemsets: DataFrame with frequent itemsets
            
        Returns:
            Series with summary-based scores
        """
        # Create summary evaluator
        summary_config = self.config.get('summaries', {})
        desired_size = summary_config.get('desired_size', 5)
        
        self.summary_evaluator = SummaryEvaluator(frequent_itemsets, desired_size)
        
        # Generate summaries
        weights = summary_config.get('weights')
        summaries = self.summary_evaluator.summarise_df(
            df,
            desired_size=desired_size,
            weights=weights
        )
        
        if not summaries:
            return pd.Series([0.0] * len(df), index=df.index)
        
        # Evaluate tuples using summaries
        scores = self.summary_evaluator.evaulate_df(df, summaries)
        
        return scores
    
    def _prepare_data_for_fpgrowth(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert a DataFrame to a format suitable for fpgrowth algorithm using TransactionEncoder.
        
        This method prepares the data for frequent pattern mining by:
        1. Prepending column names to values to create meaningful itemsets
        2. Converting to transaction format 
        3. One-hot encoding using TransactionEncoder for efficient fpgrowth processing
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (encoded DataFrame suitable for fpgrowth, list of original attributes)
        """
        if df.empty:
            return pd.DataFrame(), []
        
        # Prepend column names to create meaningful item identifiers
        df_with_names = self.prepend_column_names(df.copy())
        
        # Convert DataFrame to transactions
        transactions = df_with_names.to_dict(orient="records")
        attributes = list(df.columns)  # Original attribute names
        transaction_items = [list(t.values()) for t in transactions]
        
        # Use TransactionEncoder for efficient one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transaction_items).transform(transaction_items)
        
        # Create encoded DataFrame
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        return encoded_df, attributes

    def prepend_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepend the column name to all the values in the DataFrame.
        
        This creates meaningful itemsets where each item is formatted as "column_value",
        which helps with interpretability and prevents conflicts between values
        from different columns.
        
        Args:
            df: The DataFrame containing the data.
        
        Returns:
            DataFrame with column names prepended to the values.
        """
        for column in df.columns:
            df[column] = df[column].apply(lambda x: f"{column}_{x}")
        return df