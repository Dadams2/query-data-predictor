"""
Metrics module for evaluating query result predictions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Any, Union, Optional


class EvaluationMetrics:
    """
    Class for computing evaluation metrics between predicted and actual query results.
    """
    
    def __init__(self, jaccard_threshold: float = 0.5, column_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the metrics class.
        
        Args:
            jaccard_threshold: Threshold for Jaccard similarity to consider tuples as matches
            column_weights: Optional dictionary mapping column names to weights for similarity calculation
        """
        self.jaccard_threshold = jaccard_threshold
        self.column_weights = column_weights
    
    def accuracy(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
        """
        Calculate the accuracy of predictions.
        Accuracy is defined as the ratio of correctly predicted tuples to the total number of tuples in the actual results.
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            Accuracy score between 0 and 1
        """
        if actual.empty:
            if predicted.empty:
                return 1.0  # Both empty means perfect prediction
            return 0.0
        
        if predicted.empty:
            return 0.0
        
        # Convert DataFrames to sets of tuple representations
        actual_tuples = self._dataframe_to_tuple_set(actual)
        pred_tuples = self._dataframe_to_tuple_set(predicted)
        
        # Count exact matches
        matches = actual_tuples.intersection(pred_tuples)
        
        # Calculate accuracy
        return len(matches) / len(actual_tuples)
    
    def overlap_accuracy(self, previous: pd.DataFrame, actual: pd.DataFrame, predicted: pd.DataFrame) -> float:
        """
        Calculate the overlap accuracy metric.
        For each tuple in the predicted results, check if there's a matching tuple in the overlap
        between the previous and actual results.
        
        Args:
            previous: DataFrame with previous results
            actual: DataFrame with actual results
            predicted: DataFrame with predicted results
            
        Returns:
            Overlap accuracy score between 0 and 1
        """
        if actual.empty or previous.empty:
            return 0.0
        
        # Calculate the overlap between previous and actual
        overlap = pd.merge(previous, actual, how='inner')

        # Calculate the accuracy of predicted compared to the overlap
        correct_predictions = pd.merge(predicted, overlap, how='inner')
        accuracy = len(correct_predictions) / len(overlap) if len(overlap) > 0 else 0.0

        return accuracy
    
    def jaccard_similarity(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
        """
        Calculate Jaccard similarity between predicted and actual results.
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        if actual.empty and predicted.empty:
            return 1.0
        
        if actual.empty or predicted.empty:
            return 0.0
        
        # Convert DataFrames to sets of tuple representations
        actual_tuples = self._dataframe_to_tuple_set(actual)
        pred_tuples = self._dataframe_to_tuple_set(predicted)
        
        # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection = len(actual_tuples.intersection(pred_tuples))
        union = len(actual_tuples.union(pred_tuples))
        
        return intersection / union if union > 0 else 0.0
    
    def precision(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
        """
        Calculate precision of predictions.
        Precision is defined as the ratio of correctly predicted tuples to the total number of tuples in the predicted results.
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            Precision score between 0 and 1
        """
        if predicted.empty:
            if actual.empty:
                return 1.0  # Both empty means perfect prediction
            return 0.0
        
        # Convert DataFrames to sets of tuple representations
        actual_tuples = self._dataframe_to_tuple_set(actual)
        pred_tuples = self._dataframe_to_tuple_set(predicted)
        
        # Count exact matches
        matches = actual_tuples.intersection(pred_tuples)
        
        # Calculate precision
        return len(matches) / len(pred_tuples)
    
    def recall(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
        """
        Calculate recall of predictions.
        Recall is defined as the ratio of correctly predicted tuples to the total number of tuples in the actual results.
        (Same as accuracy in this context, but included for consistency with standard metrics terminology)
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            Recall score between 0 and 1
        """
        return self.accuracy(predicted, actual)
    
    def f1_score(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
        """
        Calculate F1 score of predictions.
        F1 score is the harmonic mean of precision and recall.
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            F1 score between 0 and 1
        """
        precision_val = self.precision(predicted, actual)
        recall_val = self.recall(predicted, actual)
        
        if precision_val == 0 and recall_val == 0:
            return 0.0
        
        return 2 * (precision_val * recall_val) / (precision_val + recall_val)
    
    def jaccard_precision_recall(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate precision and recall using Jaccard similarity with a threshold.
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            Dictionary with 'precision', 'recall', and 'f1' scores
        """
        if actual.empty:
            if predicted.empty:
                return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if predicted.empty:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Convert to list of dictionaries for comparison
        actual_records = actual.to_dict('records')
        pred_records = predicted.to_dict('records')
        
        # Count matches using Jaccard similarity
        true_positives = 0
        
        # For each actual tuple, check if there's a similar predicted tuple
        for act_tuple in actual_records:
            for pred_tuple in pred_records:
                if self._tuple_similarity(act_tuple, pred_tuple) >= self.jaccard_threshold:
                    true_positives += 1
                    break
        
        # Calculate metrics
        precision = true_positives / len(pred_records) if pred_records else 0.0
        recall = true_positives / len(actual_records) if actual_records else 0.0
        
        f1 = 0.0
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def standard_metrics(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            predicted: DataFrame with predicted results
            actual: DataFrame with actual results
            
        Returns:
            Dictionary mapping metric names to their values
        """
        metrics = {
            'accuracy': self.accuracy(predicted, actual),
            'jaccard_similarity': self.jaccard_similarity(predicted, actual),
            'precision': self.precision(predicted, actual),
            'recall': self.recall(predicted, actual),
            'f1_score': self.f1_score(predicted, actual)
        }
        
        # Add Jaccard-based metrics
        jaccard_metrics = self.jaccard_precision_recall(predicted, actual)
        metrics.update({
            'jaccard_precision': jaccard_metrics['precision'],
            'jaccard_recall': jaccard_metrics['recall'],
            'jaccard_f1': jaccard_metrics['f1']
        })
        
        return metrics
    
    def _dataframe_to_tuple_set(self, df: pd.DataFrame) -> Set[Tuple]:
        """
        Convert a DataFrame to a set of tuples for comparison.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            Set of tuples representing the DataFrame rows
        """
        # Convert each row to a tuple of values
        return {tuple(row) for row in df.itertuples(index=False, name=None)}
    
    def _tuple_similarity(self, tuple1: Dict[str, Any], tuple2: Dict[str, Any]) -> float:
        """
        Calculate the similarity between two tuples using weighted Jaccard similarity.
        
        Args:
            tuple1: First tuple as a dictionary
            tuple2: Second tuple as a dictionary
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get all keys from both tuples
        all_keys = set(tuple1.keys()).union(set(tuple2.keys()))
        
        if not all_keys:
            return 1.0  # Both empty
        
        # Calculate weighted matches
        matches = 0.0
        total_weight = 0.0
        
        for key in all_keys:
            # Get weight for this column
            weight = 1.0
            if self.column_weights and key in self.column_weights:
                weight = self.column_weights[key]
            
            total_weight += weight
            
            # Check if values match
            if key in tuple1 and key in tuple2:
                value1 = tuple1[key]
                value2 = tuple2[key]
                
                if value1 == value2:
                    matches += weight
                elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # For numeric values, calculate similarity based on relative difference
                    max_val = max(abs(value1), abs(value2))
                    if max_val > 0:
                        similarity = 1.0 - min(abs(value1 - value2) / max_val, 1.0)
                        matches += weight * similarity
        
        # Calculate similarity
        return matches / total_weight if total_weight > 0 else 0.0
