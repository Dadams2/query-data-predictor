from query_data_predictor.summariser import get_candidate_summaries, bus_summarization_with_candidates, summary_loss, transactions_covered, append_count_to_summaries
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pytest
import pandas as pd
from fixtures import *


def test_summary_loss():
    summary = {"bread": "bread", "milk": "*", "diaper": "*", "beer": "*", "coke": "*"}
    weights = {"bread": 1, "milk": 1, "diaper": 1, "beer": 1, "coke": 1}
    loss = summary_loss(summary, weights)
    assert loss == 4.0

def test_get_candidate_summaries(simple_transactions):
    pass

def test_simple_final_summaries(simple_transactions):
    attributes = simple_transactions.columns
    transactions = simple_transactions.to_dict(orient='records')
    frequent_itemsets = fpgrowth(simple_transactions, min_support=0.5, use_colnames=True)

    summaries = get_candidate_summaries(frequent_itemsets, attributes)
    assert len(summaries) == len(frequent_itemsets)

    weights = {attr: 1 for attr in attributes}
    final_summaries = bus_summarization_with_candidates(transactions, weights, DESIRED_SIZE, summaries)
    append_count_to_summaries(final_summaries, SIMPLE_TRANSACTIONS, attributes)
    assert final_summaries[0]['Count'] == 5

def test_complex_final_summaries(complex_transactions):
    attributes = complex_transactions.columns
    transactions = complex_transactions.to_dict(orient='records')
    frequent_itemsets = fpgrowth(complex_transactions, min_support=0.5, use_colnames=True)

    summaries = get_candidate_summaries(frequent_itemsets, attributes)
    assert len(summaries) == len(frequent_itemsets)

    weights = {attr: 1 for attr in attributes}
    final_summaries = bus_summarization_with_candidates(transactions, weights, DESIRED_SIZE, summaries)
    append_count_to_summaries(final_summaries, COMPLEX_TRANSACTIONS, attributes)
    print(final_summaries)
    assert final_summaries[0]['Count'] == 19 