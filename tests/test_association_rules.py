import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from query_data_predictor.association_rules import association_rules

# this file exists to test the association rules implementation against the mlxtend library
# should it change


dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

def test_association_rules():
    """
    Test the association rules implementation.
    """
    # Convert the dataset into a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
    ### alternatively:
    #frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
    #frequent_itemsets = fpmax(df, min_support=0.6, use_colnames=True)

    print(frequent_itemsets)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7) 

    assert rules["consequents"].iloc[0] == frozenset({'Eggs'})
    assert rules["antecedents"].iloc[0] == frozenset({'Kidney Beans'})
    assert rules["support"].iloc[0] == 0.8
    assert rules["confidence"].iloc[0] == 0.80
    assert rules["lift"].iloc[0] == 1.0
    assert rules["leverage"].iloc[0] == 0.0
    assert rules["conviction"].iloc[0] == 1.0
    assert rules["zhangs_metric"].iloc[0] == 0.0
    assert rules["antecedent support"].iloc[0] == 1.0
    assert rules["consequent support"].iloc[0] == 0.8   