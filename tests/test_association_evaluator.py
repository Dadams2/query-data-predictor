import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from query_data_predictor.association_interestingness import AssociationEvaluator
import pytest
from fixtures import *


def test_association_evaluator(simple_transactions):
    """
    Test the AssociationEvaluator class.
    """
    # Generate frequent itemsets
    frequent_itemsets = fpgrowth(simple_transactions, min_support=0.6, use_colnames=True)

    # Initialize AssociationEvaluator
    evaluator = AssociationEvaluator(frequent_itemsets)

    # Mine association rules
    rules_df = evaluator.mine_association_rules(metric="confidence", min_confidence=0.7)

    # Validate the mined rules
    assert not rules_df.empty
    assert "antecedents" in rules_df.columns
    assert "consequents" in rules_df.columns
    assert "confidence" in rules_df.columns

    # Evaluate interestingness scores
    scores = evaluator.evaulate_df(simple_transactions, rules_df, measure_columns=['confidence'], parallel=False)

    assert scores[0] == pytest.approx(1.50, rel=1e-2)
    assert scores[1] == pytest.approx(3.25, rel=1e-2)
    assert scores[2] == pytest.approx(3.25, rel=1e-2)
    assert scores[3] == pytest.approx(6.25, rel=1e-2)
    assert len(scores) == len(simple_transactions)


def test_association_evaluator(complex_transactions):
    """
    Test the AssociationEvaluator class.
    """
    # Generate frequent itemsets
    frequent_itemsets = fpgrowth(complex_transactions, min_support=0.6, use_colnames=True)

    # Initialize AssociationEvaluator
    evaluator = AssociationEvaluator(frequent_itemsets)

    # Mine association rules
    rules_df = evaluator.mine_association_rules(metric="confidence", min_confidence=0.7)
    scores = evaluator.evaulate_df(complex_transactions, rules_df, measure_columns=['confidence'], parallel=False)

    assert len(scores) == len(complex_transactions)
    print(scores)
    assert scores[0] == pytest.approx(602.0, rel=1e-2)
    assert scores[1] == pytest.approx(602.0, rel=1e-2)


