#!/usr/bin/env python3
"""
Test script to verify that the sensitivity and specificity analysis works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "experiments"))

from experiment_analyzer import ExperimentAnalyzer

def create_test_data():
    """Create synthetic test data with sensitivity and specificity metrics."""
    
    np.random.seed(42)
    n_experiments = 100
    
    # Create synthetic experimental results
    data = []
    recommenders = ['RandomRecommender', 'AssociationRecommender', 'QueryExpansionRecommender']
    gaps = [1, 2, 3, 4, 5]
    
    for i in range(n_experiments):
        recommender = np.random.choice(recommenders)
        gap = np.random.choice(gaps)
        
        # Generate correlated metrics
        base_performance = np.random.uniform(0.3, 0.9)
        noise = np.random.normal(0, 0.1)
        
        # Simulate realistic relationships between metrics
        precision = np.clip(base_performance + noise, 0, 1)
        recall = np.clip(base_performance + noise + 0.1, 0, 1)
        
        # Predicted and actual counts for calculating specificity
        predicted_count = np.random.randint(10, 200)
        actual_count = np.random.randint(10, 200)
        intersection_count = np.random.randint(0, min(predicted_count, actual_count))
        union_count = predicted_count + actual_count - intersection_count
        
        # Calculate derived metrics
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        roc_auc = np.random.uniform(0.4, 0.8)
        
        data.append({
            'meta_session_id': f'session_{i // 20}',
            'meta_recommender_name': recommender,
            'meta_gap': gap,
            'meta_timestamp': pd.Timestamp.now(),
            'meta_status': 'completed',
            'meta_execution_time_seconds': np.random.uniform(0.1, 10.0),
            'eval_overlap_accuracy': base_performance,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1_score': f1_score,
            'eval_roc_auc': roc_auc,
            'eval_predicted_count': predicted_count,
            'eval_actual_count': actual_count,
            'eval_intersection_count': intersection_count,
            'eval_union_count': union_count,
            'rec_predicted_count': predicted_count
        })
    
    return pd.DataFrame(data)

def test_analyzer():
    """Test the experiment analyzer with synthetic data."""
    
    print("🧪 Testing Sensitivity and Specificity Analysis...")
    
    # Create test data
    test_df = create_test_data()
    print(f"✓ Created {len(test_df)} synthetic experimental results")
    
    # Create a temporary analyzer instance
    class TestAnalyzer(ExperimentAnalyzer):
        def __init__(self):
            # Skip the normal initialization
            self.results_df = None
            self.metadata_summary = {}
            self.output_dir = Path("test_output")
            self.output_dir.mkdir(exist_ok=True)
    
    analyzer = TestAnalyzer()
    
    # Manually set the test data
    analyzer.results_df = analyzer._enrich_results_data(test_df)
    
    print("✓ Data enrichment completed")
    print(f"  - Sensitivity column added: {'eval_sensitivity' in analyzer.results_df.columns}")
    print(f"  - Specificity column added: {'eval_specificity' in analyzer.results_df.columns}")
    
    # Test individual visualizations
    test_methods = [
        ('sensitivity_gap_analysis', 'Sensitivity Gap Analysis'),
        ('specificity_gap_analysis', 'Specificity Gap Analysis'), 
        ('sensitivity_specificity_comparison', 'Sensitivity & Specificity Comparison'),
        ('roc_curve_analysis', 'ROC Curve Analysis'),
        ('accuracy_comparison', 'Updated Accuracy Comparison')
    ]
    
    successful_plots = 0
    for method_name, description in test_methods:
        try:
            method = getattr(analyzer, f'_create_{method_name}')
            fig = method(figsize=(10, 8))
            
            if fig is not None:
                # Save the plot
                output_path = analyzer.output_dir / f"test_{method_name}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✓ {description}: Saved to {output_path}")
                successful_plots += 1
            else:
                print(f"⚠️  {description}: No plot generated (likely missing data)")
                
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
    
    # Test statistical summary
    try:
        stats = analyzer.create_statistical_summary()
        print(f"✓ Statistical summary generated with {len(stats)} sections")
        
        if 'performance_statistics' in stats:
            perf_stats = stats['performance_statistics']
            sens_stats = [k for k in perf_stats.keys() if 'sensitivity' in k]
            spec_stats = [k for k in perf_stats.keys() if 'specificity' in k]
            print(f"  - Sensitivity statistics: {len(sens_stats)} metrics")
            print(f"  - Specificity statistics: {len(spec_stats)} metrics")
            
    except Exception as e:
        print(f"❌ Statistical summary: Error - {e}")
    
    print(f"\n🎉 Test completed! {successful_plots}/{len(test_methods)} visualizations successful")
    print(f"📁 Test outputs saved to: {analyzer.output_dir}")
    
    # Show sample data
    print("\n📊 Sample of enriched data:")
    cols_to_show = ['meta_recommender_name', 'eval_sensitivity', 'eval_specificity', 'eval_precision', 'eval_recall']
    available_cols = [col for col in cols_to_show if col in analyzer.results_df.columns]
    print(analyzer.results_df[available_cols].head())

if __name__ == "__main__":
    test_analyzer()
