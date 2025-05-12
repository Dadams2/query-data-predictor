from query_data_predictor.summariser import get_candidate_summaries, bus_summarization_with_candidates, summary_loss, transactions_covered, append_count_to_summaries
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pytest
import pandas as pd

#TODO : improve the test cases to cover all the edge cases
COMPLEX_TRANSACTIONS = [
    {"specobjid": "specobjid_177834390330540032", "specclass": "specclass_3", "objid": "objid_587729233054400633", "primtarget": "primtarget_4", "z_bin": "z_bin_3", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_189656694588964864", "specclass": "specclass_3", "objid": "objid_587731185119723742", "primtarget": "primtarget_4", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_1"},
    {"specobjid": "specobjid_173613531499855872", "specclass": "specclass_3", "objid": "objid_588007003651440738", "primtarget": "primtarget_4", "z_bin": "z_bin_3", "zconf_bin": "zconf_bin_3"},
    {"specobjid": "specobjid_112531068556410880", "specclass": "specclass_3", "objid": "objid_588015507668205622", "primtarget": "primtarget_4", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_193317404057534464", "specclass": "specclass_3", "objid": "objid_588015509807628362", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_3"},
    {"specobjid": "specobjid_109716225382154240", "specclass": "specclass_3", "objid": "objid_588015509807628362", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_147434863362834432", "specclass": "specclass_3", "objid": "objid_587726031717859456", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_3"},
    {"specobjid": "specobjid_152782748036104192", "specclass": "specclass_3", "objid": "objid_587728669876355252", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_1"},
    {"specobjid": "specobjid_124916208276340736", "specclass": "specclass_3", "objid": "objid_587725470667964632", "primtarget": "primtarget_20", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_1"},
    {"specobjid": "specobjid_84945766837125120",  "specclass": "specclass_3", "objid": "objid_588848900462936177", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_95080395132895232",  "specclass": "specclass_3", "objid": "objid_587725041708630034", "primtarget": "primtarget_4", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_84947059693584384",  "specclass": "specclass_3", "objid": "objid_588848900462936177", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_3"},
    {"specobjid": "specobjid_140679300448780288", "specclass": "specclass_3", "objid": "objid_587725818030391412", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_3"},
    {"specobjid": "specobjid_172204786647564288", "specclass": "specclass_3", "objid": "objid_587725817494962248", "primtarget": "primtarget_4", "z_bin": "z_bin_3", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_136456828245508096", "specclass": "specclass_3", "objid": "objid_587725816403722357", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_1"},
    {"specobjid": "specobjid_99864586903093248",  "specclass": "specclass_3", "objid": "objid_587725590384017626", "primtarget": "primtarget_4", "z_bin": "z_bin_2", "zconf_bin": "zconf_bin_1"},
    {"specobjid": "specobjid_140960802755575808", "specclass": "specclass_3", "objid": "objid_587726033308942463", "primtarget": "primtarget_4", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_1"},
    {"specobjid": "specobjid_101272086374252544", "specclass": "specclass_3", "objid": "objid_587725504486703225", "primtarget": "primtarget_4", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_2"},
    {"specobjid": "specobjid_133642478615003136", "specclass": "specclass_3", "objid": "objid_587725075529400435", "primtarget": "primtarget_4", "z_bin": "z_bin_1", "zconf_bin": "zconf_bin_2"}
]


SIMPLE_TRANSACTIONS = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "egg"},
    {"milk", "diaper", "beer", "coke"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "coke"}
]

DESIRED_SIZE = 4

@pytest.fixture
def simple_transactions():
    """Fixture to create and return a TransactionEncoder instance."""
    te = TransactionEncoder()
    te_ary = te.fit(SIMPLE_TRANSACTIONS).transform(SIMPLE_TRANSACTIONS)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

@pytest.fixture
def complex_transactions():
    """Fixture to create and return a TransactionEncoder instance."""
    te = TransactionEncoder()
    te_ary = te.fit(COMPLEX_TRANSACTIONS).transform(COMPLEX_TRANSACTIONS)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

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