import pandas as pd
from query_data_predictor.interestingness import InterestingnessMeasures
from query_data_predictor.summariser import get_candidate_summaries, bus_summarization_with_candidates, append_count_to_summaries 

class SummaryEvaluator:

    def __init__(self, frequent_itemsets_df, desired_size=4):
        # consider merging this in future
        self.frequent_itemsets = frequent_itemsets_df
        self.desired_size = desired_size

    
    def summarise_df(self, transactions_df, desired_size=None, weights=None):
        """
        Summarize the DataFrame using the BUS algorithm.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            DataFrame with transactions.
        weights : Dict[str, float]
            Dictionary mapping attribute to weight.
        desired_size : int
            Desired size of the summary.

        Returns
        -------
        List[Dict[str, str]]
            List of candidate summaries.
        """
        attributes = transactions_df.columns
        transactions = transactions_df.to_dict(orient='records')
        
        summaries = get_candidate_summaries(self.frequent_itemsets, attributes)
        
        if weights is None:
            weights = {attr: 1 for attr in attributes}
        if desired_size is None:
            desired_size = self.desired_size

        final_summaries = bus_summarization_with_candidates(transactions, weights, desired_size, summaries)
        append_count_to_summaries(final_summaries, transactions, attributes)
        return final_summaries

    def evaluate_df(self, transactions_df, summaries_df):
        """
        Evaluate the summaries using the BUS algorithm.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            DataFrame with transactions.
        summaries_df : pd.DataFrame
            DataFrame with summaries.

        Returns
        -------
        List[Dict[str, str]]
            List of candidate summaries.
        """
        summary_counts = [summary["Count"] for summary in summaries_df]
        im = InterestingnessMeasures(summary_counts)
        summary_scores = im.calculate_all()

        def calculate_row_score(transaction_row):
            total_score = 0
            for summary in summaries_df:
                if self.matches_summary(transaction_row, summary):
                    summary_size = len([v for v in summary.values() if v != '*'])
                    for measure, value in summary_scores.items():
                        if pd.notna(value):
                            normalized_summary_score = value / summary_size
                            total_score += normalized_summary_score
            return total_score
        
        return transactions_df.apply(calculate_row_score, axis=1)

    def matches_summary(self, row, summary):
        """
        Checks if a transaction row matches a summary pattern, handling wildcards (*).
        
        Parameters
        ----------
        row : pd.Series
            A single row from the transactions DataFrame.
        summary : dict
            A summary pattern with possible wildcards (*).
        
        Returns
        -------
        bool
            True if the row matches the summary pattern.
        """
        for key, value in summary.items():
            if key != 'Count' and value != '*' and row[key] != value:
                return False
        return True

