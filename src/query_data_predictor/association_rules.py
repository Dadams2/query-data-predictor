
# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Function for generating association rules
#
# Author: Joshua Goerner <https://github.com/JoshuaGoerner>
#         Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations

import numpy as np
import pandas as pd


def association_rules(df, metric="confidence", min_threshold=0.8, support_only=False,
                     max_antecedent_len=None, min_antecedent_len=1, batch_size=200, max_rules=None):
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']

    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      **Automatically set to 'support' if `support_only=True`.**
      Otherwise, supported metrics are 'support', 'confidence', 'lift',
      'leverage', 'conviction' and 'zhangs_metric'
      These metrics are computed as follows:

      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]\n
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n
      - leverage(A->C) = support(A->C) - support(A)*support(C),
        range: [-1, 1]\n
      - conviction = [1 - support(C)] / [1 - confidence(A->C)],
        range: [0, inf]\n
      - zhangs_metric(A->C) =
        leverage(A->C) / max(support(A->C)*(1-support(A)), support(A)*(support(C)-support(A->C)))
        range: [-1,1]\n

    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric,
      via the `metric` parameter,
      to decide whether a candidate rule is of interest.

    support_only : bool (default: False)
      Only computes the rule support and fills the other
      metric columns with NaNs. This is useful if:

      a) the input DataFrame is incomplete, e.g., does
      not contain support values for all rule antecedents
      and consequents

      b) you simply want to speed up the computation because
      you don't need the other metrics.
      
    max_antecedent_len : int or None (default: None)
      Maximum length of antecedent to consider. If None, all lengths are considered.
      Reducing this value can significantly improve performance.
      
    min_antecedent_len : int (default: 1)
      Minimum length of antecedent to consider.
      
    batch_size : int (default: 100)
      Number of itemsets to process in each batch to reduce memory usage.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

    """
    if not df.shape[0]:
        raise ValueError(
            "The input DataFrame `df` containing " "the frequent itemsets is empty."
        )

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError(
            "Dataframe needs to contain the\
                         columns 'support' and 'itemsets'"
        )

    def conviction_helper(sAC, sA, sC):
        confidence = sAC / sA
        conviction = np.empty(confidence.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            confidence = confidence[np.newaxis]
            sAC = sAC[np.newaxis]
            sA = sA[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[confidence < 1.0] = (1.0 - sC[confidence < 1.0]) / (
            1.0 - confidence[confidence < 1.0]
        )

        return conviction

    def zhangs_metric_helper(sAC, sA, sC):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](sAC, sA, sC)

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignoring the divide by 0 warning since it is addressed in the below np.where
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)

        return zhangs_metric

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, __: sA,
        "consequent support": lambda _, __, sC: sC,
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC / sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC) / sC,
        "leverage": lambda sAC, sA, sC: metric_dict["support"](sAC, sA, sC) - sA * sC,
        "conviction": lambda sAC, sA, sC: conviction_helper(sAC, sA, sC),
        "zhangs_metric": lambda sAC, sA, sC: zhangs_metric_helper(sAC, sA, sC),
    }

    columns_ordered = [
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
    ]

    # check for metric compliance
    if support_only:
        metric = "support"
    else:
        if metric not in metric_dict.keys():
            raise ValueError(
                "Metric must be 'confidence' or 'lift', got '{}'".format(metric)
            )

    # get dict of {frequent itemset} -> support
    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # Process in batches to reduce memory usage
    keys_list = list(frequent_items_dict.keys())
    
    for i in range(0, len(keys_list), batch_size):
        batch_keys = keys_list[i:i+batch_size]
        
        # iterate over all frequent itemsets in this batch
        for k in batch_keys:
            k_len = len(k)
            if k_len <= 1:  # Skip itemsets of size 1 as they can't be split
                continue
                
            sAC = frequent_items_dict[k]
            
            # Determine range of antecedent lengths to consider
            max_len = k_len - 1 if max_antecedent_len is None else min(max_antecedent_len, k_len - 1)
            min_len = max(min_antecedent_len, 1)
            
            if max_len < min_len:
                continue  # Skip if constraints can't be satisfied
                
            # Generate combinations of appropriate lengths
            for idx in range(min_len, max_len + 1):
                # Pre-compute if we're going to do support_only to avoid unnecessary combinations
                if not support_only:
                    # Check if any antecedent of this length could pass the threshold
                    # This is a heuristic and may not apply to all metrics
                    if metric == "confidence" and sAC / idx < min_threshold:
                        continue  # Skip this length as it can't meet threshold
                
                for c in combinations(k, r=idx):
                    antecedent = frozenset(c)
                    consequent = k.difference(antecedent)

                    if support_only:
                        # support doesn't need these,
                        # hence, placeholders should suffice
                        sA = None
                        sC = None
                    else:
                        try:
                            sA = frequent_items_dict[antecedent]
                            sC = frequent_items_dict[consequent]
                            
                            # Early pruning based on metric
                            if metric == "confidence" and sAC / sA < min_threshold:
                                continue
                            elif metric == "lift" and (sAC / sA) / sC < min_threshold:
                                continue
                        except KeyError as e:
                            s = (
                                str(e) + "You are likely getting this error"
                                " because the DataFrame is missing "
                                " antecedent and/or consequent "
                                " information."
                                " You can try using the "
                                " `support_only=True` option"
                            )
                            raise KeyError(s)
                            
                    score = metric_dict[metric](sAC, sA, sC)
                    if score >= min_threshold:
                        rule_antecedents.append(antecedent)
                        rule_consequents.append(consequent)
                        rule_supports.append([sAC, sA, sC])
                        
                        # Early termination if max_rules is reached
                        if max_rules and len(rule_antecedents) >= max_rules:
                            break
        
        # Early termination if max_rules is reached
        if max_rules and len(rule_antecedents) >= max_rules:
            break

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + columns_ordered)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        if support_only:
            sAC = rule_supports[0]
            for m in columns_ordered:
                df_res[m] = np.nan
            df_res["support"] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            for m in columns_ordered:
                df_res[m] = metric_dict[m](sAC, sA, sC)

        return df_res
