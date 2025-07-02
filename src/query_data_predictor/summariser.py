from typing import Any, Dict, List, Optional
import pandas as pd

def convert_itemset_to_summary(itemset: tuple, attributes: List[str]) -> Dict[str, str]:
    """
    Convert a frequent itemset (e.g. ("primtarget_4", "specclass_3"))
    into a full candidate summary dictionary. For each attribute, if an item
    from the itemset matches that attribute (assumed format "attribute_value"),
    that value is assigned; otherwise a wildcard '*' is used.

    Parameters
    ----------
    itemset : tuple
        Frequent itemset.
    attributes : List[str]
        List of attribute names.

    Returns
    -------
    Dict[str, str]
        Candidate summary as a dictionary.
    """
    summary = {attr: "*" for attr in attributes}
    for item in itemset:
        parts = item.split("_", 1)
        if len(parts) == 2:
            attr_name = parts[0]
            if attr_name in attributes:
                summary[attr_name] = item
    return summary

def get_candidate_summaries(frequent_itemsets: pd.DataFrame,
                            attributes: List[str]) -> List[Dict[str, str]]:
    """
    Convert all frequent itemsets to candidate summaries.

    Parameters
    ----------
    frequent_itemsets : pd.DataFrame
        DataFrame with frequent itemsets.
    attributes : List[str]
        List of attribute names.

    Returns
    -------
    List[Dict[str, str]]
        List of candidate summary dictionaries.
    """
    candidates = [convert_itemset_to_summary(itemset, attributes)
                  for itemset in frequent_itemsets["itemsets"]]
    return candidates

# =============================================================================
# BUS Summarization Functions
# =============================================================================
def summary_loss(summary: Dict[str, str], weights: Dict[str, float]) -> float:
    """
    Compute the information loss for a summary.
    Loss is the sum of weights for attributes that are wildcards.

    Parameters
    ----------
    summary : Dict[str, str]
        A candidate summary.
    weights : Dict[str, float]
        Dictionary mapping attribute to weight.

    Returns
    -------
    float
        Information loss value.
    """
    return sum(weights[attr] for attr, val in summary.items() if val == "*")

def covers(candidate: Dict[str, str], transaction: Dict[str, Any],
           attributes: List[str]) -> bool:
    """
    Check whether a candidate summary covers a fully specified transaction.

    Parameters
    ----------
    candidate : Dict[str, str]
        A candidate summary with potential wildcards.
    transaction : Dict[str, Any]
        A transaction (fully specified).
    attributes : List[str]
        List of attribute names.

    Returns
    -------
    bool
        True if candidate covers transaction; False otherwise.
    """
    for attr in attributes:
        if candidate[attr] != "*" and candidate[attr] != transaction[attr]:
            return False
    return True

def select_best(candidates: List[Dict[str, str]],
                summaries: List[Dict[str, Any]],
                weights: Dict[str, float]) -> Dict[str, str]:
    """
    Select the candidate summary that provides the best trade-off between
    compaction gain and extra loss.

    Parameters
    ----------
    candidates : List[Dict[str, str]]
        List of candidate summaries.
    summaries : List[Dict[str, Any]]
        Current list of summaries.
    weights : Dict[str, float]
        Dictionary mapping attribute to weight.

    Returns
    -------
    Dict[str, str]
        The best candidate summary.
    """
    attributes = list(weights.keys())
    
    # Precompute summary losses for each summary (avoid recalculating)
    summary_losses = [summary_loss(s, weights) for s in summaries]
    
    best_candidate = None
    best_loss_val = float('inf')  # Initialize to infinity for comparison
    best_gain = 0
    
    for cand in candidates:
        # Calculate candidate loss once outside the inner loop
        cand_loss = summary_loss(cand, weights)
        
        covered_indices = []
        for i, s in enumerate(summaries):
            if covers(cand, s, attributes):
                covered_indices.append(i)
        
        gain = len(covered_indices) - 1
        if gain <= 1:
            continue
            
        # Calculate total extra loss using precomputed summary losses
        total_extra_loss = sum(cand_loss - summary_losses[i] for i in covered_indices)
        
        if (best_candidate is None or total_extra_loss < best_loss_val or
                (total_extra_loss == best_loss_val and gain > best_gain)):
            best_candidate = cand
            best_loss_val = total_extra_loss
            best_gain = gain
            
    return best_candidate

def bus_summarization_with_candidates(transactions: List[Dict[str, Any]],
                                      weights: Dict[str, float],
                                      desired_size: int,
                                      candidate_summaries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Run the BUS summarization algorithm using provided candidate summaries.
    Optimized version with early termination and better performance.

    Parameters
    ----------
    transactions : List[Dict[str, Any]]
        List of transactions.
    weights : Dict[str, float]
        Dictionary mapping attribute to weight.
    desired_size : int
        Desired number of summaries.
    candidate_summaries : List[Dict[str, str]]
        Candidate summaries derived from frequent itemsets.

    Returns
    -------
    List[Dict[str, Any]]
        Final set of summaries.
    """
    attributes = list(weights.keys())
    summaries = [t.copy() for t in transactions]  # Start with each transaction
    
    # Early termination if already at desired size
    if len(summaries) <= desired_size:
        return summaries
    
    # Limit candidate summaries for performance
    max_candidates = min(len(candidate_summaries), 500)  # Limit candidates
    candidate_summaries = candidate_summaries[:max_candidates]
    
    # Precompute candidate losses to avoid recalculation
    candidate_losses = {i: summary_loss(cand, weights) for i, cand in enumerate(candidate_summaries)}
    
    # Safety measures to prevent infinite loops
    max_iterations = min(len(transactions), 100)  # Limit iterations
    iteration_count = 0
    previous_summary_count = len(summaries)
    stagnation_counter = 0
    max_stagnation = 3  # Allow 3 iterations without progress

    while len(summaries) > desired_size and iteration_count < max_iterations:
        candidate_idx = select_best_optimized(candidate_summaries, summaries, weights, candidate_losses, attributes)
        if candidate_idx is None:
            break
            
        candidate = candidate_summaries[candidate_idx]
        covered = [s for s in summaries if covers(candidate, s, attributes)]
        if len(covered) <= 1:
            break
        
        # Remove covered summaries and add candidate summary.
        summaries = [s for s in summaries if s not in covered]
        summaries.append(candidate)
        
        # Check for stagnation (no progress in reducing summaries)
        if len(summaries) >= previous_summary_count:
            stagnation_counter += 1
            if stagnation_counter >= max_stagnation:
                break
        else:
            stagnation_counter = 0
            
        previous_summary_count = len(summaries)
        iteration_count += 1
        
    return summaries


def select_best_optimized(candidates: List[Dict[str, str]],
                         summaries: List[Dict[str, Any]],
                         weights: Dict[str, float],
                         candidate_losses: Dict[int, float],
                         attributes: List[str]) -> Optional[int]:
    """
    Optimized version of select_best that uses precomputed losses and better indexing.
    
    Returns the index of the best candidate instead of the candidate itself.
    """
    # Precompute summary losses for each summary (avoid recalculating)
    summary_losses = [summary_loss(s, weights) for s in summaries]
    
    best_candidate_idx = None
    best_loss_val = float('inf')
    best_gain = 0
    
    for i, cand in enumerate(candidates):
        # Use precomputed candidate loss
        cand_loss = candidate_losses[i]
        
        # Find covered summaries more efficiently
        covered_indices = []
        for j, s in enumerate(summaries):
            if covers(cand, s, attributes):
                covered_indices.append(j)
        
        gain = len(covered_indices) - 1
        if gain <= 1:
            continue
            
        # Calculate total extra loss using precomputed summary losses
        total_extra_loss = sum(cand_loss - summary_losses[j] for j in covered_indices)
        
        if (best_candidate_idx is None or total_extra_loss < best_loss_val or
                (total_extra_loss == best_loss_val and gain > best_gain)):
            best_candidate_idx = i
            best_loss_val = total_extra_loss
            best_gain = gain
            
    return best_candidate_idx

# =============================================================================
# Utility Functions for Summary Counts
# =============================================================================
def transactions_covered(summary: Dict[str, Any],
                           transactions: List[Dict[str, Any]],
                           attributes: List[str]) -> List[Dict[str, Any]]:
    """
    Return the list of transactions that are covered by a given summary.

    Parameters
    ----------
    summary : Dict[str, Any]
        A summary (with possible wildcards).
    transactions : List[Dict[str, Any]]
        List of all transactions.
    attributes : List[str]
        List of attribute names.

    Returns
    -------
    List[Dict[str, Any]]
        Transactions covered by the summary.
    """
    return [t for t in transactions if covers(summary, t, attributes)]

def append_count_to_summaries(summaries: List[Dict[str, Any]],
                              transactions: List[Dict[str, Any]],
                              attributes: List[str]) -> None:
    """
    Append a 'Count' attribute to each summary that is the number of transactions it covers.

    Parameters
    ----------
    summaries : List[Dict[str, Any]]
        List of summaries.
    transactions : List[Dict[str, Any]]
        List of transactions.
    attributes : List[str]
        List of attribute names.
    """
    for summary in summaries:
        covered = transactions_covered(summary, transactions, attributes)
        summary["Count"] = len(covered)

