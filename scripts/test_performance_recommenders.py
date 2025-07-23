#!/usr/bin/env python3
"""
Performance test script for all high-performance recommenders.
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all recommenders
from query_data_predictor.recommender import (
    SimilarityRecommender,
    FrequencyRecommender, 
    SamplingRecommender,
    HierarchicalRecommender,
    IndexRecommender,
    IncrementalRecommender,
    EmbeddingRecommender,
    ClusteringRecommender,
    EnhancedClusteringRecommender,
    DummyRecommender,
    RandomRecommender
)


class PerformanceTest:
    """Performance testing suite for all recommenders."""
    
    def __init__(self, config_path: str = None):
        """Initialize performance test."""
        self.config_path = config_path or "configs/performance_optimized.yaml"
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'similarity_recommender': {'similarity': {'metric': 'cosine', 'cache_enabled': True}},
            'frequency_recommender': {'frequency': {'method': 'weighted', 'cache_enabled': True}},
            'sampling_recommender': {'sampling': {'method': 'stratified', 'sample_size': 500}},
            'hierarchical_recommender': {'hierarchical': {'coarse_method': 'clustering', 'fine_method': 'similarity'}},
            'index_recommender': {'indexing': {'index_types': ['value', 'pattern']}},
            'incremental_recommender': {'incremental': {'learning_rate': 0.15, 'enable_online_learning': True}},
            'embedding_recommender': {'embedding': {'embedding_dim': 32, 'method': 'pca'}},
            'recommendation': {'mode': 'top_k', 'top_k': 20}
        }
    
    def create_test_data(self, n_rows: int = 1000, n_cols: int = 10) -> pd.DataFrame:
        """Create synthetic test data."""
        np.random.seed(42)
        
        data = {}
        for i in range(n_cols):
            if i % 3 == 0:
                # Categorical column
                categories = [f'cat_{j}' for j in range(10)]
                data[f'col_{i}'] = np.random.choice(categories, size=n_rows)
            elif i % 3 == 1:
                # Numeric column (normal distribution)
                data[f'col_{i}'] = np.random.normal(100, 15, size=n_rows)
            else:
                # Numeric column (uniform distribution)
                data[f'col_{i}'] = np.random.uniform(0, 100, size=n_rows)
        
        return pd.DataFrame(data)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_recommender(self, recommender_class, config: Dict[str, Any], 
                        test_data: pd.DataFrame, test_name: str) -> Dict[str, Any]:
        """Test a single recommender."""
        logger.info(f"Testing {test_name}...")
        
        try:
            # Initialize recommender
            recommender = recommender_class(config)
            
            # Measure memory before
            memory_before = self.get_memory_usage()
            
            # Measure execution time
            start_time = time.time()
            
            # Run recommendation
            recommendations = recommender.recommend_tuples(test_data)
            
            execution_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            memory_delta = memory_after - memory_before
            
            # Validate results
            success = isinstance(recommendations, pd.DataFrame)
            recommendations_count = len(recommendations) if success else 0
            
            result = {
                'recommender': test_name,
                'success': success,
                'execution_time': execution_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta,
                'data_size': len(test_data),
                'recommendations_count': recommendations_count,
                'error': None
            }
            
            logger.info(f"  ✓ {test_name}: {execution_time:.4f}s, {memory_delta:.2f}MB, {recommendations_count} recommendations")
            
        except Exception as e:
            logger.error(f"  ✗ {test_name} failed: {str(e)}")
            result = {
                'recommender': test_name,
                'success': False,
                'execution_time': 0,
                'memory_before': 0,
                'memory_after': 0,
                'memory_delta': 0,
                'data_size': len(test_data),
                'recommendations_count': 0,
                'error': str(e)
            }
        
        return result
    
    def run_performance_tests(self, data_sizes: List[int] = None) -> pd.DataFrame:
        """Run performance tests on all recommenders."""
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 2000]
        
        # Define recommenders to test
        recommenders = [
            (SimilarityRecommender, self.config.get('similarity_recommender', {}), 'SimilarityRecommender'),
            (FrequencyRecommender, self.config.get('frequency_recommender', {}), 'FrequencyRecommender'),
            (SamplingRecommender, self.config.get('sampling_recommender', {}), 'SamplingRecommender'),
            (HierarchicalRecommender, self.config.get('hierarchical_recommender', {}), 'HierarchicalRecommender'),
            (IndexRecommender, self.config.get('index_recommender', {}), 'IndexRecommender'),
            (IncrementalRecommender, self.config.get('incremental_recommender', {}), 'IncrementalRecommender'),
            (EmbeddingRecommender, self.config.get('embedding_recommender', {}), 'EmbeddingRecommender'),
            # Include existing recommenders for comparison
            (ClusteringRecommender, {'clustering': {'n_clusters': 5}}, 'ClusteringRecommender'),
            (EnhancedClusteringRecommender, self.config.get('enhanced_clustering_recommender', {'clustering': {'n_clusters': 5}}), 'EnhancedClusteringRecommender'),
            (DummyRecommender, {}, 'DummyRecommender'),
            (RandomRecommender, {}, 'RandomRecommender')
        ]
        
        all_results = []
        
        for data_size in data_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing with data size: {data_size}")
            logger.info(f"{'='*60}")
            
            # Create test data
            test_data = self.create_test_data(n_rows=data_size, n_cols=8)
            
            # Test each recommender
            for recommender_class, config, name in recommenders:
                # Add general config
                full_config = config.copy()
                full_config.update(self.config.get('recommendation', {}))
                
                result = self.test_recommender(recommender_class, full_config, test_data, name)
                result['data_size'] = data_size
                all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> None:
        """Analyze and display performance results."""
        print(f"\n{'='*80}")
        print("PERFORMANCE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Overall success rate
        success_rate = results_df['success'].mean()
        print(f"\nOverall Success Rate: {success_rate:.1%}")
        
        # Filter successful results
        successful = results_df[results_df['success'] == True]
        
        if successful.empty:
            print("No successful tests to analyze!")
            return
        
        # Performance by recommender
        print(f"\nPerformance by Recommender (Average across all data sizes):")
        print(f"{'Recommender':<25} {'Success':<8} {'Time (s)':<10} {'Memory (MB)':<12} {'Recommendations':<15}")
        print("-" * 75)
        
        for recommender in successful['recommender'].unique():
            data = successful[successful['recommender'] == recommender]
            avg_time = data['execution_time'].mean()
            avg_memory = data['memory_delta'].mean()
            avg_recs = data['recommendations_count'].mean()
            success_count = len(data)
            total_count = len(results_df[results_df['recommender'] == recommender])
            success_pct = success_count / total_count if total_count > 0 else 0
            
            print(f"{recommender:<25} {success_pct:<8.1%} {avg_time:<10.4f} {avg_memory:<12.2f} {avg_recs:<15.1f}")
        
        # Scalability analysis
        print(f"\nScalability Analysis:")
        for recommender in successful['recommender'].unique():
            data = successful[successful['recommender'] == recommender].sort_values('data_size')
            if len(data) > 1:
                # Calculate time per row
                time_per_row = data['execution_time'] / data['data_size']
                memory_per_row = data['memory_delta'] / data['data_size']
                
                print(f"\n{recommender}:")
                print(f"  Time per row: {time_per_row.mean():.6f}s (std: {time_per_row.std():.6f})")
                print(f"  Memory per row: {memory_per_row.mean():.4f}MB (std: {memory_per_row.std():.4f})")
        
        # Best performers
        print(f"\nBest Performers:")
        fastest = successful.groupby('recommender')['execution_time'].mean().nsmallest(3)
        most_efficient = successful.groupby('recommender')['memory_delta'].mean().nsmallest(3)
        
        print(f"  Fastest (avg execution time):")
        for i, (recommender, time_val) in enumerate(fastest.items(), 1):
            print(f"    {i}. {recommender}: {time_val:.4f}s")
        
        print(f"  Most Memory Efficient (avg memory delta):")
        for i, (recommender, memory) in enumerate(most_efficient.items(), 1):
            print(f"    {i}. {recommender}: {memory:.2f}MB")
        
        # Failed tests
        failed = results_df[results_df['success'] == False]
        if not failed.empty:
            print(f"\nFailed Tests:")
            for _, row in failed.iterrows():
                print(f"  {row['recommender']} (data_size={row['data_size']}): {row['error']}")
    
    def save_results(self, results_df: pd.DataFrame, filename: str = None) -> None:
        """Save results to CSV file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"performance_test_results_{timestamp}.csv"
        
        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to: {filename}")
    
    def run_full_test_suite(self) -> pd.DataFrame:
        """Run the complete performance test suite."""
        logger.info("Starting comprehensive performance test suite...")
        
        # Test with different data sizes
        data_sizes = [100, 500, 1000, 2000, 5000]
        
        # Run tests
        results_df = self.run_performance_tests(data_sizes)
        
        # Analyze results
        self.analyze_results(results_df)
        
        # Save results
        self.save_results(results_df)
        
        return results_df


def main():
    """Main function to run performance tests."""
    # Create performance test instance
    tester = PerformanceTest()
    
    # Run full test suite
    results_df = tester.run_full_test_suite()
    
    print(f"\n{'='*80}")
    print("PERFORMANCE TEST COMPLETED")
    print(f"{'='*80}")
    print(f"Total tests run: {len(results_df)}")
    print(f"Successful tests: {results_df['success'].sum()}")
    print(f"Success rate: {results_df['success'].mean():.1%}")


if __name__ == "__main__":
    main()
