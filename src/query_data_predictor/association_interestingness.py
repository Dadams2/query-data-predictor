from mlxtend.frequent_patterns import association_rules
from joblib import Parallel, delayed
import pandas as pd

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

        rules = association_rules(self.frequent_itemsets,
                                metric=metric,
                                min_threshold=min_confidence)
        
        return rules
    

    def evaulate_df(self, transactions_df, rules_df, measure_columns=None, parallel=True, n_jobs=-1):
        """
        Computes the interestingness contribution for each transaction based on matching rules.

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
        
        # Pre-compute rule sizes and filter valid measures to avoid recomputation
        rule_sizes = rules_df.apply(lambda row: len(row['antecedents']) + len(row['consequents']), axis=1)
        valid_measures = [m for m in measure_columns if m in rules_df.columns]
        
        # Create lookup dictionaries for fast access
        antecedents_list = rules_df['antecedents'].tolist()
        consequents_list = rules_df['consequents'].tolist()
        measure_values = {measure: rules_df[measure].tolist() for measure in valid_measures}
        
        def calculate_row_score(transaction_row):
            total_score = 0

            for i in range(len(rules_df)):
                # Fast set-based matching
                if (all(transaction_row[item] for item in antecedents_list[i]) and 
                    all(transaction_row[item] for item in consequents_list[i])):
                    # rule_size = rule_sizes.iloc[i]
                    for measure in valid_measures:
                        measure_score = measure_values[measure][i]
                        if pd.notna(measure_score) and measure_score != float('inf'):
                            total_score += measure_score 
            return total_score
        
        # Use parallel processing for larger datasets if enabled
        if parallel and len(transactions_df) > 1000:
            try:
                scores = Parallel(n_jobs=n_jobs)(
                    delayed(calculate_row_score)(row) for _, row in transactions_df.iterrows()
                )
                return pd.Series(scores, index=transactions_df.index)
            except ImportError:
                print("joblib not installed. Falling back to sequential processing.")
                
        # Standard apply for smaller datasets or when parallel is disabled
        return transactions_df.apply(calculate_row_score, axis=1)