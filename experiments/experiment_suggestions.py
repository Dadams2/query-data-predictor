"""
Configuration sensitivity experiments for recommender systems.

This module provides experiments to test how different configuration parameters
affect recommender performance, helping identify optimal settings and robust configurations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import itertools
from pathlib import Path
import json
import logging

from experiments.recommender_experiments import RecommenderExperimentRunner
from experiments.experiment_analyzer import ExperimentAnalyzer

logger = logging.getLogger(__name__)


class ConfigurationSensitivityExperiment:
    """
    Experiment runner for testing configuration parameter sensitivity.
    """
    
    def __init__(self, base_output_dir: str = "config_sensitivity_experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
    def run_association_rules_sensitivity(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Test sensitivity to association rules parameters.
        
        Tests:
        - min_support: [0.01, 0.05, 0.1, 0.2]
        - min_threshold: [0.1, 0.3, 0.5, 0.7]
        - bins: [3, 5, 7, 10]
        """
        
        # Parameter grid
        param_grid = {
            'min_support': [0.01, 0.05, 0.1, 0.2],
            'min_threshold': [0.1, 0.3, 0.5, 0.7], 
            'bins': [3, 5, 7, 10]
        }
        
        logger.info("Running association rules sensitivity analysis")
        logger.info(f"Parameter grid: {param_grid}")
        
        results = []
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            param_grid['min_support'],
            param_grid['min_threshold'],
            param_grid['bins']
        ))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for combo_idx, (min_support, min_threshold, bins) in enumerate(param_combinations):
            logger.info(f"Combination {combo_idx + 1}/{len(param_combinations)}: "
                       f"support={min_support}, threshold={min_threshold}, bins={bins}")
            
            # Create configuration
            config = {
                "discretization": {
                    "enabled": True,
                    "method": "equal_width",
                    "bins": bins,
                    "save_params": False
                },
                "association_rules": {
                    "enabled": True,
                    "min_support": min_support,
                    "metric": "confidence",
                    "min_threshold": min_threshold
                },
                "summaries": {
                    "enabled": False
                },
                "recommendation": {
                    "enabled": True,
                    "method": "association_rules",
                    "top_k": 10,
                    "score_threshold": 0.0
                }
            }
            
            # Run experiment with this configuration
            combo_results = self._run_single_config_experiment(
                config, session_ids, f"assoc_rules_{combo_idx}"
            )
            
            # Add parameter info to results
            for result in combo_results:
                result.update({
                    'param_min_support': min_support,
                    'param_min_threshold': min_threshold,
                    'param_bins': bins,
                    'config_type': 'association_rules'
                })
            
            results.extend(combo_results)
        
        return {
            'experiment_type': 'association_rules_sensitivity',
            'parameter_grid': param_grid,
            'total_combinations': len(param_combinations),
            'results': results
        }
    
    def run_clustering_sensitivity(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Test sensitivity to clustering parameters.
        
        Tests:
        - n_clusters: [2, 3, 5, 7, 10, 15]
        - random_state: [42, 123, 456] (for reproducibility testing)
        """
        
        param_grid = {
            'n_clusters': [2, 3, 5, 7, 10, 15],
            'random_state': [42, 123, 456]
        }
        
        logger.info("Running clustering sensitivity analysis")
        
        results = []
        param_combinations = list(itertools.product(
            param_grid['n_clusters'],
            param_grid['random_state']
        ))
        
        for combo_idx, (n_clusters, random_state) in enumerate(param_combinations):
            logger.info(f"Combination {combo_idx + 1}/{len(param_combinations)}: "
                       f"clusters={n_clusters}, seed={random_state}")
            
            config = {
                "clustering": {
                    "n_clusters": n_clusters,
                    "random_state": random_state
                },
                "recommendation": {
                    "top_k": 10
                }
            }
            
            combo_results = self._run_single_config_experiment(
                config, session_ids, f"clustering_{combo_idx}"
            )
            
            for result in combo_results:
                result.update({
                    'param_n_clusters': n_clusters,
                    'param_random_state': random_state,
                    'config_type': 'clustering'
                })
            
            results.extend(combo_results)
        
        return {
            'experiment_type': 'clustering_sensitivity',
            'parameter_grid': param_grid,
            'results': results
        }
    
    def run_hybrid_method_comparison(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare different combinations of methods in hybrid approach.
        
        Tests all combinations of:
        - Association rules: enabled/disabled
        - Summaries: enabled/disabled  
        - Interestingness: enabled/disabled
        """
        
        logger.info("Running hybrid method combination analysis")
        
        # All possible combinations of enabled/disabled for 3 methods
        method_combinations = [
            (True, True, True),   # All enabled
            (True, True, False),  # Rules + Summaries
            (True, False, True),  # Rules + Interestingness
            (False, True, True),  # Summaries + Interestingness
            (True, False, False), # Rules only
            (False, True, False), # Summaries only
            (False, False, True), # Interestingness only
        ]
        
        results = []
        
        for combo_idx, (rules_enabled, summaries_enabled, interest_enabled) in enumerate(method_combinations):
            combo_name = f"{'R' if rules_enabled else ''}{'S' if summaries_enabled else ''}{'I' if interest_enabled else ''}"
            logger.info(f"Combination {combo_idx + 1}/{len(method_combinations)}: {combo_name}")
            
            config = {
                "discretization": {
                    "enabled": True,
                    "method": "equal_width", 
                    "bins": 5,
                    "save_params": False
                },
                "association_rules": {
                    "enabled": rules_enabled,
                    "min_support": 0.05,
                    "metric": "confidence",
                    "min_threshold": 0.3
                },
                "summaries": {
                    "enabled": summaries_enabled,
                    "desired_size": 10,
                    "weights": None
                },
                "interestingness": {
                    "enabled": interest_enabled,
                    "measures": ["variance", "simpson", "shannon"]
                },
                "recommendation": {
                    "enabled": True,
                    "method": "hybrid",
                    "top_k": 10,
                    "score_threshold": 0.0
                }
            }
            
            combo_results = self._run_single_config_experiment(
                config, session_ids, f"hybrid_{combo_name}"
            )
            
            for result in combo_results:
                result.update({
                    'param_rules_enabled': rules_enabled,
                    'param_summaries_enabled': summaries_enabled,
                    'param_interestingness_enabled': interest_enabled,
                    'config_type': 'hybrid',
                    'combo_name': combo_name
                })
            
            results.extend(combo_results)
        
        return {
            'experiment_type': 'hybrid_method_comparison',
            'method_combinations': method_combinations,
            'results': results
        }
    
    def _run_single_config_experiment(self, config: Dict[str, Any], 
                                    session_ids: List[str], 
                                    experiment_name: str) -> List[Dict[str, Any]]:
        """Run experiment with a single configuration."""
        
        # Create a temporary config file
        config_file = self.base_output_dir / f"temp_config_{experiment_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # For simplicity, we'll simulate the experiment results
        # In practice, you'd integrate with your actual recommender system
        
        # This is a placeholder - you'd replace this with actual experiment execution
        results = []
        for session_id in session_ids:
            # Simulate results based on configuration
            result = {
                'session_id': session_id,
                'config_name': experiment_name,
                'execution_time': np.random.uniform(0.1, 2.0),
                'overlap_accuracy': np.random.uniform(0.1, 0.9),
                'precision': np.random.uniform(0.1, 0.8),
                'recall': np.random.uniform(0.1, 0.7),
                'recommendations_count': np.random.randint(5, 15)
            }
            results.append(result)
        
        # Clean up temp file
        config_file.unlink()
        
        return results


def run_scalability_experiments():
    """
    Experiments to test how recommenders scale with data size.
    """
    
    # Test with increasing data sizes
    data_size_experiments = [
        {'name': 'small', 'max_sessions': 2, 'max_pairs_per_session': 3},
        {'name': 'medium', 'max_sessions': 5, 'max_pairs_per_session': 5}, 
        {'name': 'large', 'max_sessions': 10, 'max_pairs_per_session': 10},
        {'name': 'xlarge', 'max_sessions': 20, 'max_pairs_per_session': 15}
    ]
    
    return data_size_experiments


def run_temporal_experiments():
    """
    Experiments to understand temporal patterns in recommendation performance.
    """
    
    temporal_experiments = [
        {
            'name': 'query_position_effect',
            'description': 'How does position in query sequence affect recommendations?',
            'analysis': 'Group by current_query_position and analyze performance'
        },
        {
            'name': 'session_length_effect', 
            'description': 'Do longer sessions have different recommendation patterns?',
            'analysis': 'Group sessions by total length and compare performance'
        },
        {
            'name': 'gap_distance_decay',
            'description': 'How does recommendation quality decay with gap distance?',
            'analysis': 'Test gaps 1-20 and model decay function'
        }
    ]
    
    return temporal_experiments


def run_robustness_experiments():
    """
    Experiments to test robustness of recommenders.
    """
    
    robustness_experiments = [
        {
            'name': 'small_result_sets',
            'description': 'Performance with very small input result sets',
            'filter': 'current_result_size < 10'
        },
        {
            'name': 'large_result_sets',
            'description': 'Performance with very large input result sets', 
            'filter': 'current_result_size > 1000'
        },
        {
            'name': 'empty_future_results',
            'description': 'Handling of cases where future query returns no results',
            'filter': 'future_result_size == 0'
        },
        {
            'name': 'highly_similar_queries',
            'description': 'Performance when consecutive queries are very similar',
            'analysis': 'Find query pairs with >90% result overlap'
        },
        {
            'name': 'completely_different_queries',
            'description': 'Performance when consecutive queries have no overlap',
            'analysis': 'Find query pairs with 0% result overlap'
        }
    ]
    
    return robustness_experiments


def run_statistical_experiments():
    """
    Experiments for statistical analysis and significance testing.
    """
    
    statistical_experiments = [
        {
            'name': 'power_analysis',
            'description': 'Determine minimum sample size for detecting differences',
            'method': 'Calculate required sample size for effect sizes of 0.1, 0.3, 0.5'
        },
        {
            'name': 'bootstrap_confidence_intervals',
            'description': 'Bootstrap confidence intervals for performance metrics',
            'method': 'Bootstrap sampling with 1000 iterations'
        },
        {
            'name': 'cross_validation',
            'description': 'K-fold cross validation on sessions',
            'method': 'Split sessions into K folds and validate'
        },
        {
            'name': 'significance_testing',
            'description': 'Pairwise statistical significance tests',
            'method': 'Bonferroni-corrected t-tests between all recommender pairs'
        }
    ]
    
    return statistical_experiments


if __name__ == "__main__":
    # Example usage
    
    # Configuration sensitivity experiments
    config_exp = ConfigurationSensitivityExperiment()
    
    sample_sessions = ['11305', '11306', '11307']  # Use your actual session IDs
    
    # Run association rules sensitivity
    assoc_results = config_exp.run_association_rules_sensitivity(sample_sessions)
    print(f"Association rules experiment: {len(assoc_results['results'])} results")
    
    # Run clustering sensitivity  
    cluster_results = config_exp.run_clustering_sensitivity(sample_sessions)
    print(f"Clustering experiment: {len(cluster_results['results'])} results")
    
    # Run hybrid method comparison
    hybrid_results = config_exp.run_hybrid_method_comparison(sample_sessions)
    print(f"Hybrid method experiment: {len(hybrid_results['results'])} results")
    
    # Print other experiment suggestions
    print("\n=== Other Experiment Suggestions ===")
    
    print("\nScalability Experiments:")
    for exp in run_scalability_experiments():
        print(f"  - {exp['name']}: {exp['max_sessions']} sessions, {exp['max_pairs_per_session']} pairs each")
    
    print("\nTemporal Experiments:")
    for exp in run_temporal_experiments():
        print(f"  - {exp['name']}: {exp['description']}")
    
    print("\nRobustness Experiments:")
    for exp in run_robustness_experiments():
        print(f"  - {exp['name']}: {exp['description']}")
    
    print("\nStatistical Experiments:")
    for exp in run_statistical_experiments():
        print(f"  - {exp['name']}: {exp['description']}")
