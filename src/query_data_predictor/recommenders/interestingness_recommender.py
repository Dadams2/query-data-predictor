import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Union, Optional
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import logging

from ..discretizer import Discretizer
from ..association_interestingness import AssociationEvaluator
from ..summary_interestingness import SummaryEvaluator
from .base_recommender import BaseRecommender


logger = logging.getLogger(__name__)

class InterestingnessRecommender(BaseRecommender):
    """
    Recommender for predicting tuples that will appear in subsequent queries.
    Uses interestingness-based scoring combining association rules and summaries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discretizer = None
        self.association_evaluator = None
        self.summary_evaluator = None
        
        # Performance caching
        self._frequent_itemsets_cache = None
        self._encoded_df_cache = None
        self._attributes_cache = None
        self._last_processed_df_hash = None
        
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
        Compute frequent itemsets from the DataFrame using FP-Growth with caching.
        
        Args:
            df: Input DataFrame (should be preprocessed/discretized)
            min_support: Minimum support threshold (uses config if not provided)
            
        Returns:
            Tuple of (frequent_itemsets, encoded_df, attributes)
        """
        # Generate hash for caching
        df_hash = pd.util.hash_pandas_object(df).sum()
        
        # Check if we can use cached results
        if (self._frequent_itemsets_cache is not None and 
            self._last_processed_df_hash == df_hash):
            logger.debug("Using cached frequent itemsets")
            return self._frequent_itemsets_cache, self._encoded_df_cache, self._attributes_cache
        
        # Get configuration for association rules if min_support not provided
        if min_support is None:
            assoc_config = self.config.get('association_rules', {})
            min_support = assoc_config.get('min_support', 0.1)
        
        # Convert DataFrame to encoded format for fpgrowth
        encoded_df, attributes = self._prepare_data_for_fpgrowth(df)
        
        if encoded_df.empty:
            return pd.DataFrame(), pd.DataFrame(), []
        
        try:
            # Log number of transactions and attributes for debugging
            logger.debug(f"Number of transactions: {len(encoded_df)}, Number of attributes: {len(attributes)}")

            # Mine frequent itemsets using FP-Growth
            frequent_itemsets = fpgrowth(
                encoded_df, 
                min_support=min_support, 
                use_colnames=True,
                verbose=0  # Suppress verbose output for performance
            )
            
            # Cache results
            self._frequent_itemsets_cache = frequent_itemsets
            self._encoded_df_cache = encoded_df
            self._attributes_cache = attributes
            self._last_processed_df_hash = df_hash
            
            return frequent_itemsets, encoded_df, attributes
        except Exception as e:
            # Return empty DataFrame if FP-Growth fails
            logger.warning(f"FP-Growth failed with error: {e}")
            return pd.DataFrame(), pd.DataFrame(), []



    def _prepare_data_for_fpgrowth(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert a DataFrame to a format suitable for fpgrowth algorithm using TransactionEncoder.
        Optimized version with better memory usage and performance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (encoded DataFrame suitable for fpgrowth, list of original attributes)
        """
        if df.empty:
            return pd.DataFrame(), []
        
        # Adaptive limits based on data size
        max_unique_values = min(15, max(5, len(df) // 50))  # Adaptive limit
        
        # Limit unique values per column to prevent combinatorial explosion
        df_limited = df.copy()
        
        for col in df_limited.columns:
            unique_values = df_limited[col].nunique()
            if unique_values > max_unique_values:
                # Keep only the most frequent values, replace others with 'OTHER'
                value_counts = df_limited[col].value_counts()
                top_values = value_counts.head(max_unique_values - 1).index
                df_limited[col] = df_limited[col].apply(
                    lambda x: x if x in top_values else 'OTHER'
                )
                logger.debug(f"Column {col}: limited from {unique_values} to {df_limited[col].nunique()} unique values")
        
        # Prepend column names to create meaningful item identifiers
        df_with_names = self.prepend_column_names(df_limited)
        
        # More efficient transaction conversion
        transactions = []
        for _, row in df_with_names.iterrows():
            transactions.append([str(val) for val in row.values if pd.notna(val)])
        
        attributes = list(df.columns)
        
        # Use TransactionEncoder for efficient one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        
        # Create encoded DataFrame with boolean types for FP-Growth compatibility
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_, dtype=bool)  # Use bool for FP-Growth
        
        logger.debug(f"Encoded DataFrame: {len(encoded_df)} rows, {len(encoded_df.columns)} columns")
        
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

    def recommend_tuples(self, current_results: pd.DataFrame) -> pd.DataFrame:

        # discretize the data if enabled
        processed_df = self.preprocess_data(current_results)


        if processed_df.empty or len(processed_df) < 2:
            return current_results
        
        # Compute frequent itemsets 
        # TODO: figure out if there should be extra handling for small results sets
        frequent_itemsets, encoded_df, attributes = self.compute_frequent_itemsets(processed_df)
        if frequent_itemsets.empty:
            logger.warning("No frequent itemsets found. Returning empty DataFrame.")
            return pd.DataFrame()
        
        scores = self._compute_tuple_interestingness_scores(encoded_df, frequent_itemsets)
        
        # sort dataframe by interestingness score as index
        
        # Sort by interestingness score and return top-k
        processed_df['interestingness_score'] = scores
        
        # Sort by interestingness score and return top-k
        scored_df = processed_df.sort_values('interestingness_score', ascending=False)

        score_threshold = self.config.get('recommendation', {}).get('score_threshold', 0.0)
        if score_threshold > 0:
            scored_df = scored_df[scored_df['interestingness_score'] >= score_threshold]
        
        output_size = self._determine_output_size(len(scored_df), 'recommendation')

        # Return top-k tuples (excluding the score column)
        result_df = scored_df.head(output_size).drop(columns=['interestingness_score'])
        
        return result_df

    def _compute_tuple_interestingness_scores(self, encoded_df: pd.DataFrame, frequent_itemsets: pd.DataFrame) -> pd.Series:
        """
        Compute interestingness scores for all tuples using association rules and/or summaries.
        
        This method combines scores from different interestingness measures based on configuration:
        - Association rules: Use rule-based interestingness (confidence, lift, etc.)
        - Summaries: Use summary-based interestingness (variance, shannon entropy, etc.)
        - Hybrid: Combine both approaches with configurable weights
        
        For large datasets, applies performance optimizations:
        - Sampling for datasets with > 1000 rows
        - Early termination for very large datasets
        - Parallel processing when available
        
        Args:
            df: Preprocessed DataFrame with tuples
            frequent_itemsets: DataFrame with frequent itemsets from FP-Growth
            
        Returns:
            Series with interestingness scores for each tuple
        """
        # Handle edge case with very small datasets
        if len(encoded_df) < 2:
            # For single tuple, return neutral score
            return pd.Series([1.0] * len(encoded_df), index=encoded_df.index)
        
        
        # Get configuration for scoring
        method = self.config.get('recommendation', {}).get('method', 'hybrid')
        alpha = 0.5  # Weight for association rules
        beta = 0.5   # Weight for summaries
        #TODO make these configurable
        
        # Initialize scores with default values
        rule_scores = pd.Series([0.0] * len(encoded_df), index=encoded_df.index)
        summary_scores = pd.Series([0.0] * len(encoded_df), index=encoded_df.index)
        
        
        # Compute association rule scores if enabled
        if method in ['association_rules', 'hybrid'] and self.config.get('association_rules', {}).get('enabled', True):
            try:
                # Use AssociationEvaluator to compute scores based on frequent itemsets
                self.association_evaluator = AssociationEvaluator(frequent_itemsets)
                
                # Adaptive thresholds based on data size
                min_confidence = self.config.get('association_rules', {}).get('min_threshold', 0.5)
                max_rules = 500 if len(encoded_df) > 1000 else 1000
                if len(encoded_df) > 1000:
                    min_confidence = max(min_confidence, 0.7)  # Higher threshold for large datasets
                
                association_rules = self.association_evaluator.mine_association_rules(
                    metric=self.config.get('association_rules', {}).get('metric', 'confidence'),
                    min_confidence=min_confidence,
                    max_rules=max_rules
                )
                
                # Limit rules for performance (use the already computed max_rules or a fallback)
                fallback_max_rules = 500 if len(encoded_df) > 1000 else 1000
                if len(association_rules) > fallback_max_rules:
                    # Sort by metric and take top rules
                    metric_col = self.config.get('association_rules', {}).get('metric', 'confidence')
                    if metric_col in association_rules.columns:
                        association_rules = association_rules.nlargest(fallback_max_rules, metric_col)
                    else:
                        association_rules = association_rules.head(fallback_max_rules)
                    logger.info(f"Limited association rules to {fallback_max_rules} for performance")
                
                # Use parallel processing for large datasets
                use_parallel = len(encoded_df) > 300 and len(association_rules) > 50
                rule_scores = self.association_evaluator.evaluate_df(
                    encoded_df, 
                    association_rules, 
                    measure_columns=['confidence', 'lift', 'leverage'],
                    parallel=use_parallel
                )
                
                if rule_scores.empty:
                    logger.warning("Association rule scoring returned empty scores. Using fallback.")
                    rule_scores = pd.Series([0.0] * len(encoded_df), index=encoded_df.index)

            except Exception as e:
                logger.error(f"Association rule scoring failed: {e}")
                rule_scores = pd.Series([0.0] * len(encoded_df), index=encoded_df.index)
        
        # Compute summary scores if enabled  
        if method in ['summaries', 'hybrid'] and self.config.get('summaries', {}).get('enabled', True):
            try:
                # Adaptive summary size based on data size
                desired_size = self.config.get('summaries', {}).get('desired_size', 10)
                if len(encoded_df) > 1000:
                    desired_size = min(desired_size, len(encoded_df) // 100)  # Limit summary size for large datasets
                
                self.summary_evaluator = SummaryEvaluator(frequent_itemsets, desired_size)
                summaries_df = self.summary_evaluator.summarise_df(
                    transactions_df=encoded_df,
                    weights=self.config.get('summaries', {}).get('weights', None),
                    desired_size=desired_size
                )
                summary_scores = self.summary_evaluator.evaluate_df(
                    encoded_df, 
                    summaries_df
                )
            except Exception as e:
                logger.warning(f"Summary scoring failed: {e}")
                summary_scores = pd.Series([0.0] * len(encoded_df), index=encoded_df.index)
        
        # Combine scores based on method
        if method == 'association_rules':
            return rule_scores
        elif method == 'summaries':
            return summary_scores
        else:  # hybrid
            return alpha * rule_scores + beta * summary_scores


    def name(self) -> str:
        """
        Return the name of the recommender.
        
        Returns:
            String name of the recommender
        """
        return "InterestingnessRecommender"
