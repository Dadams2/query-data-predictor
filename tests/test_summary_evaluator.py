from fixtures import *
from query_data_predictor.summary_interestingness import SummaryEvaluator
from mlxtend.frequent_patterns import fpgrowth


def test_summary_evaluator(simple_transactions):
    """
    Test the AssociationEvaluator class.
    """
    # Generate frequent itemsets
    frequent_itemsets = fpgrowth(simple_transactions, min_support=0.6, use_colnames=True)

    # Initialize AssociationEvaluator
    evaluator = SummaryEvaluator(frequent_itemsets)

    summaries = evaluator.summarise_df(simple_transactions, desired_size=4)
    
    # Assert the summaries are not empty
    assert summaries is not None
    assert len(summaries) == 1

    scores = evaluator.evaulate_df(simple_transactions, summaries)

    assert scores[0] == 5

def test_summary_evaluator(complex_transactions):
    """
    Test the AssociationEvaluator class.
    """
    # Generate frequent itemsets
    frequent_itemsets = fpgrowth(complex_transactions, min_support=0.6, use_colnames=True)

    # Initialize AssociationEvaluator
    evaluator = SummaryEvaluator(frequent_itemsets)

    summaries = evaluator.summarise_df(complex_transactions, desired_size=4)
    
    # Assert the summaries are not empty
    scores = evaluator.evaulate_df(complex_transactions, summaries)
    assert scores[0] == 5