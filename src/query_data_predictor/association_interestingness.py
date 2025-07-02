from query_data_predictor.association_rules import association_rules
from joblib import Parallel, delayed
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AssociationEvaluator:

    def __init__(self, frequent_itemsets_df):
        # consider merging this in future
        self.frequent_itemsets = frequent_itemsets_df

    
    def mine_association_rules(self, **kwargs):
        """
        Mine association rules from the frequent itemsets DataFrame.
        
        Returns
        -------
        """
        # Placeholder for actual implementation
        # This should return a DataFrame with columns: 'antecedents', 'consequents', 'support', 'confidence'

        # mine arules passing kwargs
        metric = kwargs.get('metric', 'confidence')
        min_confidence = kwargs.get('min_confidence', 0.5)
        max_rules = kwargs.get('max_rules', 1000)  # Limit rules for performance

        rules = association_rules(self.frequent_itemsets,
                                metric=metric,
                                min_threshold=min_confidence,
                                max_rules=max_rules)
        
        return rules
    

    def evaluate_df(self, transactions_df, rules_df, measure_columns=None, parallel=True, n_jobs=-1):
        """
        Computes the interestingness contribution for each transaction based on matching rules.
        
        This method is optimized for performance by building an index that maps items to rules,
        avoiding the need to scan all rules for each transaction.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            DataFrame containing transactions.
        rules_df : pd.DataFrame
            DataFrame containing association rules.
        measure_columns : list, optional
            List of columns to use for scoring (default is ['confidence', 'lift', 'leverage']).
        parallel : bool, optional
            Whether to use parallel processing for large datasets (default is True).
        n_jobs : int, optional
            Number of jobs for parallel processing. -1 means all available cores (default is -1).

        Returns
        -------
        pd.Series
            Series containing interestingness scores for each transaction.
        """
        if measure_columns is None:
            measure_columns = ['confidence', 'lift', 'leverage']
        
        if rules_df.empty:
            return pd.Series([0] * len(transactions_df), index=transactions_df.index)
        
        # Filter valid measures to avoid recomputation
        valid_measures = [m for m in measure_columns if m in rules_df.columns]
        
        # Build an index mapping items to rules for faster lookups
        item_to_rules = {}
        rule_data = []
        
        for i, row in rules_df.iterrows():
            antecedents = set(row['antecedents'])
            consequents = set(row['consequents'])
            all_items = antecedents.union(consequents)
            
            # Store rule data for faster access
            rule_info = {
                'antecedents': antecedents,
                'consequents': consequents,
                'all_items': all_items,
                'scores': {measure: row[measure] for measure in valid_measures if pd.notna(row[measure]) and row[measure] != float('inf')}
            }
            rule_data.append(rule_info)
            
            # Map each item to this rule
            for item in all_items:
                if item not in item_to_rules:
                    item_to_rules[item] = []
                item_to_rules[item].append(len(rule_data) - 1)
        
        def calculate_row_score_optimized(transaction_row):
            total_score = 0
            # Get active items (items that are True in this transaction)
            active_items = [item for item, value in transaction_row.items() if value]
            
            # Get candidate rules (rules that involve at least one active item)
            candidate_rule_indices = set()
            for item in active_items:
                if item in item_to_rules:
                    candidate_rule_indices.update(item_to_rules[item])
            
            # Check only candidate rules
            for rule_idx in candidate_rule_indices:
                rule = rule_data[rule_idx]
                # Check if all antecedents and consequents are satisfied
                if (all(transaction_row.get(item, False) for item in rule['antecedents']) and 
                    all(transaction_row.get(item, False) for item in rule['consequents'])):
                    # Add all valid measure scores
                    total_score += sum(rule['scores'].values())
            
            return total_score
        
        # Use parallel processing for larger datasets if enabled
        if parallel and len(transactions_df) > 250:
            try:
                scores = Parallel(n_jobs=n_jobs)(
                    delayed(calculate_row_score_optimized)(row) for _, row in transactions_df.iterrows()
                )
                return pd.Series(scores, index=transactions_df.index)
            except ImportError:
                logger.warning("joblib not installed. Falling back to sequential processing.")
                
        # Standard apply for smaller datasets or when parallel is disabled
        return transactions_df.apply(calculate_row_score_optimized, axis=1)