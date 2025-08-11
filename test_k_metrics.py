#!/usr/bin/env python3
"""
Test script for the new @k metrics functionality.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_data_predictor.metrics import EvaluationMetrics


def create_test_data():
    """Create test prediction and actual data."""
    
    # Create actual results (ground truth)
    np.random.seed(42)
    actual_data = {
        'id': range(1, 21),  # 20 actual results
        'value': np.random.randint(1, 100, 20),
        'category': np.random.choice(['A', 'B', 'C'], 20)
    }
    actual_df = pd.DataFrame(actual_data)
    
    # Create predicted results (some overlap with actual)
    # First 10 predictions match actual, next 5 are partially matching, rest are new
    predicted_data = {
        'id': list(range(1, 11)) + list(range(16, 21)) + list(range(21, 26)),  # 15 predictions
        'value': (list(actual_df['value'][:10]) + 
                 list(actual_df['value'][15:20]) + 
                 list(np.random.randint(100, 200, 5))),
        'category': (list(actual_df['category'][:10]) + 
                    list(actual_df['category'][15:20]) + 
                    list(np.random.choice(['D', 'E'], 5)))
    }
    predicted_df = pd.DataFrame(predicted_data)
    
    return predicted_df, actual_df


def test_k_metrics():
    """Test the new @k metrics functionality."""
    
    print("Testing new @k metrics functionality")
    print("=" * 50)
    
    # Create test data
    predicted_df, actual_df = create_test_data()
    
    print(f"Predicted results: {len(predicted_df)} rows")
    print(f"Actual results: {len(actual_df)} rows")
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Test standard metrics
    print("\n1. Standard Metrics:")
    print("-" * 30)
    standard_metrics = evaluator.standard_metrics(predicted_df, actual_df)
    for metric, value in standard_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test @k metrics for different k values
    print("\n2. @k Metrics:")
    print("-" * 30)
    k_values = [5, 10, 15]
    
    for k in k_values:
        print(f"\nFor k={k}:")
        precision_k = evaluator.precision_at_k(predicted_df, actual_df, k)
        recall_k = evaluator.recall_at_k(predicted_df, actual_df, k)
        f1_k = evaluator.f1_at_k(predicted_df, actual_df, k)
        
        print(f"  Precision@{k}: {precision_k:.4f}")
        print(f"  Recall@{k}: {recall_k:.4f}")
        print(f"  F1@{k}: {f1_k:.4f}")
    
    # Test batch @k metrics calculation
    print("\n3. Batch @k Metrics:")
    print("-" * 30)
    batch_metrics = evaluator.precision_recall_at_k_range(predicted_df, actual_df, k_values)
    
    for metric_type, k_dict in batch_metrics.items():
        print(f"{metric_type}:")
        for k, value in k_dict.items():
            print(f"  @{k}: {value:.4f}")
    
    # Test extended metrics with k values
    print("\n4. Extended Metrics with Auto-k:")
    print("-" * 30)
    extended_metrics = evaluator.extended_metrics_with_k(predicted_df, actual_df)
    
    # Separate @k metrics from standard metrics
    standard_keys = [k for k in extended_metrics.keys() if '_at_k_' not in k]
    k_keys = [k for k in extended_metrics.keys() if '_at_k_' in k]
    
    print("Standard metrics:")
    for key in standard_keys:
        print(f"  {key}: {extended_metrics[key]:.4f}")
    
    print("\n@k metrics:")
    for key in sorted(k_keys):
        print(f"  {key}: {extended_metrics[key]:.4f}")
    
    print("\n" + "=" * 50)
    print("@k metrics testing completed successfully!")


if __name__ == "__main__":
    test_k_metrics()
