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
        
        # Initialize components based on configuration
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

    def mine_association_rules(self, df: pd.DataFrame, frequent_itemsets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Mine association rules from the DataFrame.
        
        Args:
            df: Input DataFrame (should be preprocessed/discretized)
            frequent_itemsets: Pre-computed frequent itemsets (optional)
            
        Returns:
            DataFrame with association rules
        """
        # Get configuration for association rules
        assoc_config = self.config.get('association_rules', {})
        
        # Use provided frequent itemsets or compute them
        if frequent_itemsets is None:
            frequent_itemsets = self.compute_frequent_itemsets(df)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Create association evaluator
        self.association_evaluator = AssociationEvaluator(frequent_itemsets)
        
        # Mine rules with parameters from config
        rules_df = self.association_evaluator.mine_association_rules(
            metric=assoc_config.get('metric', 'confidence'),
            min_confidence=assoc_config.get('min_threshold', 0.7)
        )
        
        return rules_df
    
    def generate_summaries(self, df: pd.DataFrame, frequent_itemsets: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Generate summaries from the DataFrame.
        
        Args:
            df: Input DataFrame (should be preprocessed/discretized)
            frequent_itemsets: Pre-computed frequent itemsets (optional)
            
        Returns:
            List of summary dictionaries
        """
        # Get configuration for summaries
        summary_config = self.config.get('summaries', {})
        desired_size = summary_config.get('desired_size', 5)
        
        # Use provided frequent itemsets or compute them with lower support for summaries
        if frequent_itemsets is None:
            frequent_itemsets = self.compute_frequent_itemsets(df, min_support=0.05)
        
        if frequent_itemsets.empty:
            return []
        
        # Create summary evaluator
        self.summary_evaluator = SummaryEvaluator(frequent_itemsets, desired_size)
        
        # Generate summaries
        weights = summary_config.get('weights')
        summaries = self.summary_evaluator.summarise_df(
            df, 
            desired_size=desired_size,
            weights=weights
        )
        
        return summaries
    
    def recommend_tuples(self, current_results: pd.DataFrame, top_k: Optional[int] = None, frequent_itemsets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Recommend tuples for the next query based on the current query's results.
        
        Args:
            current_results: DataFrame with the current query's results
            top_k: Number of top tuples to recommend (overrides config value if provided)
            frequent_itemsets: Pre-computed frequent itemsets (optional, for performance optimization)
            
        Returns:
            DataFrame with recommended tuples
        """
        # Use configuration top_k if not provided
        if top_k is None:
            top_k = self.config.get('recommendation', {}).get('top_k', 10)
        
        # Get recommendation method from config
        method = self.config.get('recommendation', {}).get('method', 'association_rules')
        
        # Preprocess data
        processed_df = self.preprocess_data(current_results)
        
        # If frequent_itemsets provided, use them; otherwise compute in each method
        # Depending on the method, generate recommendations
        if method == 'association_rules':
            return self._recommend_with_association_rules(processed_df, top_k, frequent_itemsets)
        elif method == 'summaries':
            return self._recommend_with_summaries(processed_df, top_k, frequent_itemsets)
        elif method == 'hybrid':
            # TODO: Fix this to make it look more like the contribution in paper
            if frequent_itemsets is not None:
                # Use provided frequent itemsets for both methods
                rule_recs = self._recommend_with_association_rules(processed_df, top_k // 2, frequent_itemsets)
                summary_recs = self._recommend_with_summaries(processed_df, top_k // 2, frequent_itemsets)
                
                # Combine recommendations
                if rule_recs.empty:
                    return summary_recs
                if summary_recs.empty:
                    return rule_recs
                    
                # Combine and deduplicate
                combined = pd.concat([rule_recs, summary_recs]).drop_duplicates()
                
                # Limit to top_k
                if len(combined) > top_k:
                    combined = combined.head(top_k)
                    
                return combined
            else:
                # Use the existing hybrid method that computes frequent itemsets once
                return self._recommend_hybrid(processed_df, top_k)
        else:
            raise ValueError(f"Unknown recommendation method: {method}")
    
    def _recommend_with_association_rules(self, df: pd.DataFrame, top_k: int, frequent_itemsets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Recommend tuples using association rules.
        
        Args:
            df: Preprocessed DataFrame
            top_k: Number of tuples to recommend
            frequent_itemsets: Pre-computed frequent itemsets (optional)
            
        Returns:
            DataFrame with recommended tuples
        """
        # Mine association rules
        rules_df = self.mine_association_rules(df, frequent_itemsets)
        
        if rules_df.empty:
            # No rules found, return empty DataFrame
            return pd.DataFrame()
        
        # Get threshold from config
        score_threshold = self.config.get('recommendation', {}).get('score_threshold', 0.5)
        
        # Filter rules by threshold
        rules_df = rules_df[rules_df['confidence'] >= score_threshold]
        
        if rules_df.empty:
            return pd.DataFrame()
        
        # Sort rules by confidence
        rules_df = rules_df.sort_values('confidence', ascending=False)
        
        # Generate recommended tuples from rules
        recommendations = self._generate_tuples_from_rules(rules_df, top_k)
        
        return recommendations
    
    def _recommend_with_summaries(self, df: pd.DataFrame, top_k: int, frequent_itemsets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Recommend tuples using summaries.
        
        Args:
            df: Preprocessed DataFrame
            top_k: Number of tuples to recommend
            frequent_itemsets: Pre-computed frequent itemsets (optional)
            
        Returns:
            DataFrame with recommended tuples
        """
        # Generate summaries
        summaries = self.generate_summaries(df, frequent_itemsets)
        
        if not summaries:
            # No summaries found, return empty DataFrame
            return pd.DataFrame()
        
        # Convert summaries to tuples
        recommendations = self._generate_tuples_from_summaries(summaries, top_k)
        
        return recommendations
    
    def _recommend_hybrid(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Recommend tuples using a hybrid of association rules and summaries.
        
        Args:
            df: Preprocessed DataFrame
            top_k: Number of tuples to recommend
            
        Returns:
            DataFrame with recommended tuples
        """
        # Compute frequent itemsets once for both methods
        frequent_itemsets = self.compute_frequent_itemsets(df)
        
        # Get recommendations from both methods using shared frequent itemsets
        rule_recs = self._recommend_with_association_rules(df, top_k // 2, frequent_itemsets)
        summary_recs = self._recommend_with_summaries(df, top_k // 2, frequent_itemsets)
        
        # Combine recommendations
        if rule_recs.empty:
            return summary_recs
        if summary_recs.empty:
            return rule_recs
            
        # Combine and deduplicate
        combined = pd.concat([rule_recs, summary_recs]).drop_duplicates()
        
        # Limit to top_k
        if len(combined) > top_k:
            combined = combined.head(top_k)
            
        return combined
    
    def _prepare_data_for_fpgrowth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a DataFrame to a format suitable for fpgrowth algorithm using TransactionEncoder.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Encoded DataFrame suitable for fpgrowth
        """
        if df.empty:
            return pd.DataFrame()
        
        # Filter out columns with too many unique values to prevent memory explosion
        # filtered_df = df.copy()
        # for col in df.columns:
        #     if df[col].nunique() > 100:  # Arbitrary threshold to avoid explosion of columns
        #         filtered_df = filtered_df.drop(columns=[col])
        
        # if filtered_df.empty:
        #     return pd.DataFrame()
        
        df = self.prepend_column_names(df)
        
        # Convert DataFrame to transactions using the original format (column_value)
        # This creates meaningful item identifiers that can be sorted
        transactions = df.to_dict(orient="records")
        attributes = list(df.columns)
        transaction_items = [list(t.values()) for t in transactions]
        # Use TransactionEncoder for efficient one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transaction_items).transform(transaction_items)
        # Create encoded DataFrame
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
        return encoded_df, attributes
    
    def _generate_tuples_from_rules(self, rules_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Generate tuples from association rules.
        
        Args:
            rules_df: DataFrame with association rules
            top_k: Maximum number of tuples to generate
            
        Returns:
            DataFrame with generated tuples
        """
        # Extract consequents from rules
        all_items = []
        for _, rule in rules_df.iterrows():
            consequents = list(rule['consequents'])
            all_items.extend(consequents)
            
            # Stop if we have enough items
            if len(all_items) >= top_k:
                break
        
        # Convert items to a DataFrame
        if not all_items:
            return pd.DataFrame()
            
        # Parse items (format: "column_value") into columns and values
        tuple_data = {}
        
        for item in all_items[:top_k]:
            # Split by first underscore
            parts = item.split('_', 1)
            if len(parts) == 2:
                col, val = parts
                
                # Initialize column in data if not present
                if col not in tuple_data:
                    tuple_data[col] = []
                    
                # Add value to column
                tuple_data[col].append(val)
        
        # Create DataFrame from tuple data
        # Ensure all columns have the same length by padding with None
        max_len = max([len(vals) for vals in tuple_data.values()]) if tuple_data else 0
        
        for col, vals in tuple_data.items():
            if len(vals) < max_len:
                tuple_data[col].extend([None] * (max_len - len(vals)))
        
        return pd.DataFrame(tuple_data) if tuple_data else pd.DataFrame()
    
    def _generate_tuples_from_summaries(self, summaries: List[Dict[str, Any]], top_k: int) -> pd.DataFrame:
        """
        Generate tuples from summaries.
        
        Args:
            summaries: List of summary dictionaries
            top_k: Maximum number of tuples to generate
            
        Returns:
            DataFrame with generated tuples
        """
        # Extract attributes from summaries
        tuple_data = {}
        
        for summary in summaries:
            for attr, val in summary.items():
                # Skip count attribute
                if attr == 'count':
                    continue
                    
                # Skip wildcard values
                if val == '*':
                    continue
                
                # Initialize column in data if not present
                if attr not in tuple_data:
                    tuple_data[attr] = []
                
                # Parse the value (format: "attribute_value")
                parts = val.split('_', 1)
                if len(parts) == 2:
                    # Add value to column
                    tuple_data[attr].append(parts[1])
        
        # Create DataFrame from tuple data
        # Ensure all columns have the same length by padding with None
        max_len = max([len(vals) for vals in tuple_data.values()]) if tuple_data else 0
        
        for col, vals in tuple_data.items():
            if len(vals) < max_len:
                tuple_data[col].extend([None] * (max_len - len(vals)))
        
        df = pd.DataFrame(tuple_data) if tuple_data else pd.DataFrame()
        
        # Limit to top_k rows
        if len(df) > top_k:
            df = df.head(top_k)
            
        return df


    def prepend_column_names(self, df):
        """
        Prepend the column name to all the values in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: DataFrame with column names prepended to the values.
        """
        for column in df.columns:
            df[column] = df[column].apply(lambda x: f"{column}_{x}")
        return df