"""
Extract metrics from experimental results to populate LaTeX tables.

This script analyzes experiment result files and generates formatted
output that can be directly copied into the LaTeX tables.

Usage:
    python extract_table_data.py --results-dir results/
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def compute_metrics_for_result(result: Dict) -> Dict[str, float]:
    """Compute metrics for a single experiment result."""
    try:
        recommended = set([tuple(sorted(r.items())) for r in result['recommended_results']])
        future = set([tuple(sorted(r.items())) for r in result['future_results']])
        
        if len(recommended) == 0 or len(future) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'overlap': 0.0}
        
        overlap = len(recommended & future)
        precision = overlap / len(recommended)
        recall = overlap / len(future)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        overlap_ratio = overlap / len(future)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'overlap': overlap_ratio
        }
    except Exception as e:
        print(f"Warning: Failed to compute metrics: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'overlap': 0.0}


def extract_table2_data(results_dir: Path) -> str:
    """
    Extract data for Table 2: Overall performance on SDSS (gap=1, k=10).
    """
    print("\n" + "="*70)
    print("TABLE 2: SDSS Benchmark Results (gap=1, k=10)")
    print("="*70)
    
    # Load all results for gap=1
    results_by_recommender = defaultdict(list)
    
    for json_file in results_dir.glob("*__gap-1.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for result in data:
                if result.get('error_message'):
                    continue
                    
                recommender = result['recommender_name']
                metrics = compute_metrics_for_result(result)
                results_by_recommender[recommender].append(metrics)
                
        except Exception as e:
            print(f"Warning: Failed to process {json_file}: {e}")
    
    # Compute averages
    latex_rows = []
    name_map = {
        'multidimensional_interestingness': 'MDI (Ours)',
        'interestingness': 'Interestingness',
        'clustering': 'Clustering',
        'sampling': 'Sampling',
        'random': 'Random'
    }
    
    order = ['multidimensional_interestingness', 'interestingness', 
             'clustering', 'sampling', 'random']
    
    for recommender in order:
        if recommender not in results_by_recommender:
            continue
            
        metrics_list = results_by_recommender[recommender]
        if not metrics_list:
            continue
            
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
            'overlap': np.mean([m['overlap'] for m in metrics_list])
        }
        
        name = name_map.get(recommender, recommender)
        
        # Format as LaTeX table row
        row = f"{name} & {avg_metrics['precision']:.3f} & {avg_metrics['recall']:.3f} & " \
              f"{avg_metrics['f1']:.3f} & {avg_metrics['overlap']:.3f} \\\\"
        
        latex_rows.append(row)
        
        # Print summary
        print(f"\n{name}:")
        print(f"  Precision: {avg_metrics['precision']:.3f}")
        print(f"  Recall:    {avg_metrics['recall']:.3f}")
        print(f"  F1:        {avg_metrics['f1']:.3f}")
        print(f"  Overlap:   {avg_metrics['overlap']:.3f}")
        print(f"  N samples: {len(metrics_list)}")
    
    print("\n" + "-"*70)
    print("LaTeX table rows (copy into Table 2):")
    print("-"*70)
    for row in latex_rows:
        print(row)
    
    return "\n".join(latex_rows)


def extract_table3_data(results_dir: Path) -> str:
    """
    Extract data for Table 3: Ablation study.
    Requires experiments with different configurations.
    """
    print("\n" + "="*70)
    print("TABLE 3: Ablation Study (gap=1, k=10)")
    print("="*70)
    print("\nNote: This requires running experiments with different MDI configurations.")
    print("See publication_guide.md for configuration examples.")
    
    # Check for ablation experiment results
    ablation_dirs = [
        ('full', 'MDI (full)'),
        ('no_novelty', 'No novelty (γ=0)'),
        ('no_diversity', 'No diversity (β=0)'),
        ('no_rules', 'No rules (α=0)'),
        ('rules_only', 'Rules only (β=γ=0)')
    ]
    
    latex_rows = []
    
    for config_name, display_name in ablation_dirs:
        config_dir = results_dir / f"ablation_{config_name}"
        
        if not config_dir.exists():
            print(f"\nWarning: {config_dir} not found - skipping {display_name}")
            continue
        
        # Load results
        metrics_list = []
        times = []
        
        for json_file in config_dir.glob("*__gap-1.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for result in data:
                    if result['recommender_name'] != 'multidimensional_interestingness':
                        continue
                    if result.get('error_message'):
                        continue
                    
                    metrics = compute_metrics_for_result(result)
                    metrics_list.append(metrics)
                    
                    if 'execution_time' in result:
                        times.append(result['execution_time'] * 1000)  # Convert to ms
                        
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
        
        if not metrics_list:
            continue
        
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list])
        }
        avg_time = np.mean(times) if times else 0.0
        
        row = f"{display_name} & {avg_metrics['precision']:.3f} & " \
              f"{avg_metrics['recall']:.3f} & {avg_metrics['f1']:.3f} & {avg_time:.1f} \\\\"
        
        latex_rows.append(row)
        
        print(f"\n{display_name}:")
        print(f"  Precision: {avg_metrics['precision']:.3f}")
        print(f"  Recall:    {avg_metrics['recall']:.3f}")
        print(f"  F1:        {avg_metrics['f1']:.3f}")
        print(f"  Time:      {avg_time:.1f} ms")
    
    if latex_rows:
        print("\n" + "-"*70)
        print("LaTeX table rows (copy into Table 3):")
        print("-"*70)
        for row in latex_rows:
            print(row)
    
    return "\n".join(latex_rows)


def extract_dataset_stats(data_dir: Path) -> str:
    """
    Extract data for Table 1: Dataset characteristics.
    """
    print("\n" + "="*70)
    print("TABLE 1: Dataset Characteristics")
    print("="*70)
    
    # Load metadata
    metadata_file = data_dir / "metadata.csv"
    
    if not metadata_file.exists():
        print(f"\nWarning: {metadata_file} not found")
        return ""
    
    metadata = pd.read_csv(metadata_file)
    
    # Load sample sessions to compute statistics
    query_counts = []
    result_sizes = []
    
    print(f"\nAnalyzing {len(metadata)} sessions...")
    
    for idx, row in metadata.iterrows():
        session_file = Path(row['path'])
        
        if not session_file.exists():
            continue
        
        try:
            import pickle
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            # Count queries
            if 'queries' in session_data:
                query_counts.append(len(session_data['queries']))
            elif 'query_results' in session_data:
                query_counts.append(len(session_data['query_results']))
            
            # Get result sizes
            if 'query_results' in session_data:
                for qr in session_data['query_results']:
                    if isinstance(qr, pd.DataFrame):
                        result_sizes.append(len(qr))
                    elif isinstance(qr, list):
                        result_sizes.append(len(qr))
                        
        except Exception as e:
            continue
    
    # Compute statistics
    total_sessions = len(metadata)
    avg_queries = np.mean(query_counts) if query_counts else 0
    avg_result_size = np.mean(result_sizes) if result_sizes else 0
    
    print(f"\nSDSS Dataset:")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Avg queries/session: {avg_queries:.1f}")
    print(f"  Avg result size: {avg_result_size:.1f}")
    
    latex_row = f"SDSS & {total_sessions} & {avg_queries:.1f} & {avg_result_size:.1f} \\\\"
    
    print("\n" + "-"*70)
    print("LaTeX table row (copy into Table 1):")
    print("-"*70)
    print(latex_row)
    
    return latex_row


def main():
    parser = argparse.ArgumentParser(
        description='Extract metrics from experiment results for LaTeX tables'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results/memory_reccomender'),
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/datasets'),
        help='Directory containing dataset metadata'
    )
    parser.add_argument(
        '--tables',
        nargs='+',
        choices=['1', '2', '3', 'all'],
        default=['all'],
        help='Which tables to extract data for'
    )
    
    args = parser.parse_args()
    
    tables_to_extract = args.tables
    if 'all' in tables_to_extract:
        tables_to_extract = ['1', '2', '3']
    
    if '1' in tables_to_extract:
        extract_dataset_stats(args.data_dir)
    
    if '2' in tables_to_extract:
        extract_table2_data(args.results_dir)
    
    if '3' in tables_to_extract:
        extract_table3_data(args.results_dir.parent)  # Look in parent for ablation dirs
    
    print("\n" + "="*70)
    print("Extraction complete!")
    print("="*70)


if __name__ == '__main__':
    main()
