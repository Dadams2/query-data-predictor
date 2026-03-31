"""
Multi-dimensional interestingness recommender with temporal decay and novelty scoring.

This recommender implements the interestingness contribution scheme described in the paper,
combining association rules, diversity-based summarization, and novelty detection with
temporal decay to adapt to evolving analytical interests.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict, Counter
import logging
from datetime import datetime
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from .base_recommender import BaseRecommender
from ..discretizer import Discretizer
from ..association_interestingness import AssociationEvaluator
from ..summary_interestingness import SummaryEvaluator
from ..interestingness import InterestingnessMeasures

logger = logging.getLogger(__name__)


class MultiDimensionalInterestingnessRecommender(BaseRecommender):
    """
    Advanced recommender using multi-dimensional interestingness with temporal adaptation.
    
    Implements the full interestingness contribution scheme:
    I(t) = α * Σ(w_r * I_assoc(r)/|X_r ∪ Y_r| * f_r(t)) 
         + β * Σ(w_s * I_div(s)/|s| * g_s(t)) 
         + γ * Novelty(t)
    
    Features:
    - Temporal decay weights for rules and summaries
    - Multi-measure association rule scoring (confidence, support, lift, J-measure)
    - Multi-measure diversity scoring (Shannon, Simpson, Gini, Berger, McIntosh)
    - Novelty detection based on inverse frequency
    - Historical memory of all tuples and rules
    - Pattern length normalization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-dimensional interestingness recommender.
        
        Args:
            config: Configuration dictionary with settings
        """
        super().__init__(config)
        
        # Historical memory
        self._tuple_history = []  # List of (timestamp, tuple_hash, tuple_data)
        self._rule_history = []   # List of (timestamp, rule, measures)
        self._summary_history = []  # List of (timestamp, summary, measures)
        self._attribute_value_frequencies = defaultdict(lambda: defaultdict(int))
        self._session_counter = 0
        
        # Discretizer
        self.discretizer = None
        if config.get('discretization', {}).get('enabled', True):
            disc_config = config.get('discretization', {})
            self.discretizer = Discretizer(
                method=disc_config.get('method', 'equal_width'),
                bins=disc_config.get('bins', 5),
                save_path=disc_config.get('params_path') if disc_config.get('save_params', True) else None
            )
        
        # Get multi-dimensional configuration
        md_config = self.config.get('multidimensional_interestingness', {})
        
        # Main weights (α, β, γ)
        self.alpha = md_config.get('alpha', 0.4)  # Association rules weight
        self.beta = md_config.get('beta', 0.4)    # Summaries weight
        self.gamma = md_config.get('gamma', 0.2)  # Novelty weight
        
        # Association rule measure weights (λ_1, λ_2, λ_3, λ_4)
        assoc_weights = md_config.get('association_weights', {})
        self.lambda_1 = assoc_weights.get('confidence', 0.3)
        self.lambda_2 = assoc_weights.get('support', 0.2)
        self.lambda_3 = assoc_weights.get('lift', 0.3)
        self.lambda_4 = assoc_weights.get('j_measure', 0.2)  # J-measure approximation
        
        # Diversity measure weights (μ_1, μ_2, μ_3, μ_4, μ_5)
        div_weights = md_config.get('diversity_weights', {})
        self.mu_1 = div_weights.get('shannon', 0.25)
        self.mu_2 = div_weights.get('simpson', 0.20)
        self.mu_3 = div_weights.get('gini', 0.20)
        self.mu_4 = div_weights.get('berger', 0.15)
        self.mu_5 = div_weights.get('mcintosh', 0.20)
        
        # Temporal decay parameters
        self.lambda_r = md_config.get('rule_decay_rate', 0.1)      # Rule decay rate
        self.lambda_s = md_config.get('summary_decay_rate', 0.1)   # Summary decay rate
        
        # Other parameters
        self.min_support = config.get('association_rules', {}).get('min_support', 0.1)
        self.min_confidence = config.get('association_rules', {}).get('min_threshold', 0.5)
        self.desired_summary_size = config.get('summaries', {}).get('desired_size', 10)
        
        # Performance caching
        self._frequent_itemsets_cache = None
        self._encoded_df_cache = None
        self._attributes_cache = None
        self._last_processed_df_hash = None
        
        logger.info(f"Initialized MultiDimensionalInterestingnessRecommender with α={self.alpha}, β={self.beta}, γ={self.gamma}")
    
    def recommend_tuples(self, current_results: pd.DataFrame, top_k: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Recommend tuples using multi-dimensional interestingness scoring.
        
        Args:
            current_results: DataFrame with current query results
            top_k: Number of tuples to return (overrides config if provided)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with recommended tuples
        """
        self._validate_input(current_results)
        
        if current_results.empty or len(current_results) < 2:
            return pd.DataFrame()
        
        # Increment session counter for temporal tracking
        self._session_counter += 1
        current_timestamp = self._session_counter
        
        # Update tuple history and attribute-value frequencies
        self._update_tuple_history(current_results, current_timestamp)
        
        # Preprocess (discretize) data
        processed_df = self._preprocess_data(current_results)
        
        # Mine frequent itemsets (with caching)
        frequent_itemsets, encoded_df, attributes = self._compute_frequent_itemsets(processed_df)
        
        if frequent_itemsets.empty:
            logger.warning("No frequent itemsets found. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Mine association rules and update history
        association_rules = self._mine_association_rules(frequent_itemsets, current_timestamp)
        
        # Generate summaries and update history
        summaries = self._generate_summaries(encoded_df, attributes, frequent_itemsets, current_timestamp)
        
        # Compute multi-dimensional interestingness scores
        scores = self._compute_multidimensional_scores(
            current_results,
            processed_df,
            encoded_df,
            association_rules,
            summaries,
            current_timestamp
        )
        
        # Apply threshold if configured
        score_threshold = self.config.get('recommendation', {}).get('score_threshold', 0.0)
        if score_threshold > 0:
            scores = scores[scores >= score_threshold]
        
        if scores.empty:
            return pd.DataFrame()
        
        # Sort and select top tuples
        sorted_indices = scores.sort_values(ascending=False).index
        
        # Determine output size
        if top_k is not None:
            output_size = top_k
        else:
            output_size = self._determine_output_size(len(scores), 'recommendation')
        
        top_indices = sorted_indices[:output_size]
        
        # Return original (non-discretized) tuples
        result_df = current_results.loc[top_indices].copy()
        
        return result_df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data by applying discretization if enabled."""
        if self.discretizer and self.config.get('discretization', {}).get('enabled', True):
            df_copy = df.copy(deep=False)
            return self.discretizer.discretize_dataframe(df_copy)
        return df
    
    def _compute_frequent_itemsets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Compute frequent itemsets with caching.
        
        Returns:
            Tuple of (frequent_itemsets, encoded_df, attributes)
        """
        # Generate hash for caching
        df_hash = pd.util.hash_pandas_object(df).sum()
        
        # Check cache
        if (self._frequent_itemsets_cache is not None and 
            self._last_processed_df_hash == df_hash):
            logger.debug("Using cached frequent itemsets")
            return self._frequent_itemsets_cache, self._encoded_df_cache, self._attributes_cache
        
        # Convert DataFrame to encoded format
        encoded_df, attributes = self._prepare_data_for_fpgrowth(df)
        
        if encoded_df.empty:
            return pd.DataFrame(), pd.DataFrame(), []
        
        try:
            # Mine frequent itemsets using FP-Growth
            frequent_itemsets = fpgrowth(
                encoded_df,
                min_support=self.min_support,
                use_colnames=True,
                verbose=0
            )
            
            # Cache results
            self._frequent_itemsets_cache = frequent_itemsets
            self._encoded_df_cache = encoded_df
            self._attributes_cache = attributes
            self._last_processed_df_hash = df_hash
            
            return frequent_itemsets, encoded_df, attributes
        except Exception as e:
            logger.warning(f"FP-Growth failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), []
    
    def _prepare_data_for_fpgrowth(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert DataFrame to format suitable for fpgrowth algorithm.
        
        Returns:
            Tuple of (encoded DataFrame, list of attributes)
        """
        if df.empty:
            return pd.DataFrame(), []
        
        # Adaptive limits based on data size
        max_unique_values = min(15, max(5, len(df) // 50))
        
        # Limit unique values per column
        df_limited = df.copy()
        
        for col in df_limited.columns:
            unique_values = df_limited[col].nunique()
            if unique_values > max_unique_values:
                value_counts = df_limited[col].value_counts()
                top_values = value_counts.head(max_unique_values - 1).index
                df_limited[col] = df_limited[col].apply(
                    lambda x: x if x in top_values else 'OTHER'
                )
        
        # Prepend column names to create meaningful item identifiers
        df_with_names = self._prepend_column_names(df_limited)
        
        # Convert to transactions
        transactions = []
        for _, row in df_with_names.iterrows():
            transactions.append([str(val) for val in row.values if pd.notna(val)])
        
        attributes = list(df.columns)
        
        # Use TransactionEncoder for efficient one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_, dtype=bool, index=df_with_names.index)
        
        return encoded_df, attributes
    
    def _prepend_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend column name to all values in the DataFrame."""
        df_copy = df.copy()
        for column in df_copy.columns:
            df_copy[column] = df_copy[column].apply(lambda x: f"{column}_{x}")
        return df_copy
    
    def _mine_association_rules(self, frequent_itemsets: pd.DataFrame, timestamp: int) -> pd.DataFrame:
        """
        Mine association rules and update history.
        
        Args:
            frequent_itemsets: Frequent itemsets DataFrame
            timestamp: Current timestamp for temporal tracking
            
        Returns:
            Association rules DataFrame
        """
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        try:
            evaluator = AssociationEvaluator(frequent_itemsets)
            
            # Mine rules with various metrics
            rules = evaluator.mine_association_rules(
                metric='confidence',
                min_confidence=self.min_confidence,
                max_rules=500
            )
            
            if not rules.empty:
                # Store rules in history with timestamp
                for idx, rule in rules.iterrows():
                    rule_data = {
                        'antecedents': rule['antecedents'],
                        'consequents': rule['consequents'],
                        'confidence': rule.get('confidence', 0),
                        'support': rule.get('support', 0),
                        'lift': rule.get('lift', 0),
                        'leverage': rule.get('leverage', 0),
                    }
                    self._rule_history.append((timestamp, rule_data))
                
                logger.debug(f"Mined {len(rules)} association rules")
            
            return rules
        except Exception as e:
            logger.warning(f"Association rule mining failed: {e}")
            return pd.DataFrame()
    
    def _generate_summaries(self, encoded_df: pd.DataFrame, attributes: List[str], 
                          frequent_itemsets: pd.DataFrame, timestamp: int) -> List[Dict]:
        """
        Generate summaries and update history.
        
        Args:
            encoded_df: Encoded transactions DataFrame
            attributes: List of attribute names
            frequent_itemsets: Frequent itemsets DataFrame
            timestamp: Current timestamp
            
        Returns:
            List of summary dictionaries
        """
        if encoded_df.empty or frequent_itemsets.empty:
            return []
        
        try:
            evaluator = SummaryEvaluator(frequent_itemsets, self.desired_summary_size)
            
            summaries = evaluator.summarise_df(
                transactions_df=encoded_df,
                desired_size=self.desired_summary_size
            )
            
            if summaries:
                # Store summaries in history with timestamp
                for summary in summaries:
                    self._summary_history.append((timestamp, summary))
                
                logger.debug(f"Generated {len(summaries)} summaries")
            
            return summaries
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return []
    
    def _compute_multidimensional_scores(self, original_df: pd.DataFrame, processed_df: pd.DataFrame,
                                        encoded_df: pd.DataFrame, association_rules: pd.DataFrame,
                                        summaries: List[Dict], current_timestamp: int) -> pd.Series:
        """
        Compute multi-dimensional interestingness scores using the full equation.
        
        I(t) = α * Σ(w_r * I_assoc(r)/|X_r ∪ Y_r| * f_r(t)) 
             + β * Σ(w_s * I_div(s)/|s| * g_s(t)) 
             + γ * Novelty(t)
        
        Args:
            original_df: Original (non-discretized) DataFrame
            processed_df: Processed (discretized) DataFrame
            encoded_df: Encoded transactions DataFrame
            association_rules: Mined association rules
            summaries: Generated summaries
            current_timestamp: Current timestamp
            
        Returns:
            Series of interestingness scores
        """
        n_tuples = len(encoded_df)
        scores = pd.Series(0.0, index=encoded_df.index)
        
        # Component 1: Association rule contribution with temporal decay
        if self.alpha > 0 and not association_rules.empty:
            rule_scores = self._compute_association_component(
                encoded_df, association_rules, current_timestamp
            )
            scores += self.alpha * rule_scores
            logger.debug(f"Association component: mean={rule_scores.mean():.4f}, max={rule_scores.max():.4f}")
        
        # Component 2: Summary diversity contribution with temporal decay
        if self.beta > 0 and summaries:
            summary_scores = self._compute_summary_component(
                encoded_df, summaries, current_timestamp
            )
            scores += self.beta * summary_scores
            logger.debug(f"Summary component: mean={summary_scores.mean():.4f}, max={summary_scores.max():.4f}")
        
        # Component 3: Novelty contribution
        if self.gamma > 0:
            novelty_scores = self._compute_novelty_component(original_df)
            scores += self.gamma * novelty_scores
            logger.debug(f"Novelty component: mean={novelty_scores.mean():.4f}, max={novelty_scores.max():.4f}")
        
        return scores
    
    def _compute_association_component(self, encoded_df: pd.DataFrame, 
                                      association_rules: pd.DataFrame,
                                      current_timestamp: int) -> pd.Series:
        """
        Compute association rule component: Σ(w_r * I_assoc(r)/|X_r ∪ Y_r| * f_r(t))
        
        Where:
        - w_r = e^(-λ_r * age(r)) is the temporal decay weight
        - I_assoc(r) = λ_1*confidence + λ_2*support + λ_3*lift + λ_4*J-measure
        - |X_r ∪ Y_r| is the pattern length normalization
        - f_r(t) is the indicator function for tuple membership
        """
        scores = pd.Series(0.0, index=encoded_df.index)
        
        # Consider both current rules and historical rules with decay
        all_rules = []
        
        # Add current rules with age 0
        for idx, rule in association_rules.iterrows():
            all_rules.append((0, rule.to_dict()))
        
        # Add historical rules with their ages
        for timestamp, rule_data in self._rule_history:
            age = current_timestamp - timestamp
            if age > 0:  # Don't duplicate current rules
                all_rules.append((age, rule_data))
        
        # Process each rule
        for age, rule in all_rules:
            # Compute temporal decay weight
            w_r = np.exp(-self.lambda_r * age)
            
            # Extract rule components
            antecedents = rule.get('antecedents', frozenset())
            consequents = rule.get('consequents', frozenset())
            
            # Compute I_assoc(r) - multi-measure association interestingness
            confidence = rule.get('confidence', 0)
            support = rule.get('support', 0)
            lift = rule.get('lift', 0)
            leverage = rule.get('leverage', 0)
            
            # J-measure approximation using leverage (simplified)
            # J(X;Y) ≈ leverage in information-theoretic terms
            j_measure = abs(leverage) if leverage != 0 else 0
            
            I_assoc = (self.lambda_1 * confidence + 
                      self.lambda_2 * support + 
                      self.lambda_3 * lift + 
                      self.lambda_4 * j_measure)
            
            # Pattern length normalization: |X_r ∪ Y_r|
            pattern_length = len(antecedents) + len(consequents)
            if pattern_length == 0:
                continue
            
            normalized_score = I_assoc / pattern_length
            
            # Compute weighted score
            weighted_score = w_r * normalized_score
            
            # Apply to tuples that match the rule (indicator function f_r(t))
            matching_mask = self._check_rule_match(encoded_df, antecedents, consequents)
            if matching_mask.any():
                scores.loc[matching_mask] = scores.loc[matching_mask] + weighted_score
        
        return scores
    
    def _compute_summary_component(self, encoded_df: pd.DataFrame,
                                  summaries: List[Dict],
                                  current_timestamp: int) -> pd.Series:
        """
        Compute summary diversity component: Σ(w_s * I_div(s)/|s| * g_s(t))
        
        Where:
        - w_s = e^(-λ_s * age(s)) is the temporal decay weight
        - I_div(s) = μ_1*Shannon + μ_2*(1-Simpson) + μ_3*Gini + μ_4*(1-Berger) + μ_5*McIntosh
        - |s| is the summary size normalization
        - g_s(t) is the indicator function for tuple membership in summary
        """
        scores = pd.Series(0.0, index=encoded_df.index)
        
        # Collect all summaries (current + historical) with ages
        all_summaries = []
        
        # Add current summaries with age 0
        for summary in summaries:
            all_summaries.append((0, summary))
        
        # Add historical summaries with their ages
        for timestamp, summary in self._summary_history:
            age = current_timestamp - timestamp
            if age > 0:  # Don't duplicate current summaries
                all_summaries.append((age, summary))
        
        # Compute diversity measures for all summaries
        if all_summaries:
            # Extract counts for diversity calculation
            summary_counts = [s[1].get('Count', 1) for s in all_summaries]
            
            # Compute diversity measures
            measures = InterestingnessMeasures(summary_counts)
            diversity_measures = measures.calculate_all()
            
            # Compute I_div using the weighted combination
            shannon = diversity_measures.get('shannon', 0)
            simpson = diversity_measures.get('simpson', 0)
            gini = diversity_measures.get('gini', 0)
            berger = diversity_measures.get('berger', 0)
            mcintosh = diversity_measures.get('mcintosh', 0)
            
            I_div = (self.mu_1 * shannon +
                    self.mu_2 * (1 - simpson) +
                    self.mu_3 * gini +
                    self.mu_4 * (1 - berger) +
                    self.mu_5 * mcintosh)
            
            # Process each summary
            for idx, (age, summary) in enumerate(all_summaries):
                # Compute temporal decay weight
                w_s = np.exp(-self.lambda_s * age)
                
                # Summary size normalization: |s|
                summary_size = len([v for k, v in summary.items() if k != 'Count' and v != '*'])
                if summary_size == 0:
                    summary_size = 1  # Avoid division by zero
                
                normalized_score = I_div / summary_size
                
                # Compute weighted score
                weighted_score = w_s * normalized_score
                
                # Apply to tuples that match the summary (indicator function g_s(t))
                matching_mask = self._check_summary_match(encoded_df, summary)
                if matching_mask.any():
                    scores.loc[matching_mask] = scores.loc[matching_mask] + weighted_score
        
        return scores
    
    def _compute_novelty_component(self, original_df: pd.DataFrame) -> pd.Series:
        """
        Compute novelty component: Novelty(t) = Σ(1 / log(1 + freq(a, value(t, a))))
        
        Novelty measures the inverse frequency of attribute-value combinations,
        promoting discovery of rare but potentially significant data patterns.
        """
        scores = pd.Series(0.0, index=original_df.index)
        
        for idx, row in original_df.iterrows():
            novelty_score = 0.0
            
            for attr in original_df.columns:
                value = row[attr]
                
                # Get frequency of this attribute-value combination
                freq = self._attribute_value_frequencies[attr][value]
                
                # Compute inverse frequency (with log smoothing)
                if freq > 0:
                    novelty_score += 1.0 / np.log(1 + freq)
                else:
                    # New unseen combination - maximum novelty
                    novelty_score += 1.0
            
            scores[idx] = novelty_score
        
        # Normalize by number of attributes
        if len(original_df.columns) > 0:
            scores = scores / len(original_df.columns)
        
        return scores
    
    def _check_rule_match(self, encoded_df: pd.DataFrame, 
                         antecedents: frozenset, consequents: frozenset) -> pd.Series:
        """
        Check which tuples match the rule (both antecedents and consequents).
        
        Returns:
            Boolean Series indicating matching tuples
        """
        mask = pd.Series(True, index=encoded_df.index)
        
        # Check antecedents
        for item in antecedents:
            if item in encoded_df.columns:
                mask &= encoded_df[item]
            else:
                return pd.Series(False, index=encoded_df.index)
        
        # Check consequents
        for item in consequents:
            if item in encoded_df.columns:
                mask &= encoded_df[item]
            else:
                return pd.Series(False, index=encoded_df.index)
        
        return mask
    
    def _check_summary_match(self, encoded_df: pd.DataFrame, summary: Dict) -> pd.Series:
        """
        Check which tuples match the summary pattern.
        
        Returns:
            Boolean Series indicating matching tuples
        """
        mask = pd.Series(True, index=encoded_df.index)
        
        for key, value in summary.items():
            if key == 'Count':
                continue
            
            if value == '*':
                # Wildcard matches anything
                continue
            
            # Check if the item exists in encoded_df
            if value in encoded_df.columns:
                mask &= encoded_df[value]
            else:
                # If the specific value doesn't exist, no match
                return pd.Series(False, index=encoded_df.index)
        
        return mask
    
    def _update_tuple_history(self, df: pd.DataFrame, timestamp: int):
        """
        Update historical memory with new tuples and their attribute-value frequencies.
        
        Args:
            df: DataFrame with current tuples
            timestamp: Current timestamp
        """
        for idx, row in df.iterrows():
            # Generate hash for tuple
            tuple_hash = pd.util.hash_pandas_object(row).values[0]
            
            # Store tuple in history
            self._tuple_history.append((timestamp, tuple_hash, row.to_dict()))
            
            # Update attribute-value frequencies
            for attr in df.columns:
                value = row[attr]
                self._attribute_value_frequencies[attr][value] += 1
        
        logger.debug(f"Updated tuple history: {len(self._tuple_history)} total tuples")
    
    def get_history_stats(self) -> Dict[str, int]:
        """
        Get statistics about the historical memory.
        
        Returns:
            Dictionary with counts of stored items
        """
        return {
            'total_tuples': len(self._tuple_history),
            'total_rules': len(self._rule_history),
            'total_summaries': len(self._summary_history),
            'session_counter': self._session_counter,
        }
    
    def clear_history(self):
        """Clear all historical memory (for testing or reset)."""
        self._tuple_history.clear()
        self._rule_history.clear()
        self._summary_history.clear()
        self._attribute_value_frequencies.clear()
        self._session_counter = 0
        logger.info("Cleared all historical memory")
    
    def name(self) -> str:
        """Return the name of the recommender."""
        return "MultiDimensionalInterestingnessRecommender"
