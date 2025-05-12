import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from query_data_predictor.association_interestingness import AssociationEvaluator
import pytest

# todo integrate this with the summarisers
SAMPLE_TRANSACTIONS = [
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
    te_ary = te.fit(SAMPLE_TRANSACTIONS).transform(SAMPLE_TRANSACTIONS)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

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


