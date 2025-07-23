"""
Performance test for the TupleRecommender using real data from the first session.
"""

import pytest
import pandas as pd
import time
import psutil
import os
from pathlib import Path
from query_data_predictor.recommender.tuple_recommender import TupleRecommender
from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence


class TestTupleRecommenderPerformance:
    """Performance test suite for the tuple recommender."""
    
    @pytest.fixture
    def dataset_dir(self):
        """Get the path to the datasets directory."""
        # Get the project root directory
        test_dir = Path(__file__).parent
        project_root = test_dir.parent
        dataset_dir = project_root / "data" / "datasets"
        
        if not dataset_dir.exists():
            pytest.skip(f"Dataset directory not found: {dataset_dir}")
        
        return str(dataset_dir)
    
    @pytest.fixture
    def dataloader(self, dataset_dir):
        """Create a DataLoader instance."""
        return DataLoader(dataset_dir)
    
    @pytest.fixture
    def query_result_sequence(self, dataloader):
        """Create a QueryResultSequence instance."""
        return QueryResultSequence(dataloader)
    
    @pytest.fixture
    def performance_config(self):
        """Configuration optimized for performance testing."""
        return {
            "discretization": {
                "enabled": True,
                "method": "equal_width",
                "bins": 5,
                "save_params": False
            },
            "association_rules": {
                "enabled": True,
                "min_support": 0.05,  # Lower support for more patterns
                "metric": "confidence",
                "min_threshold": 0.3  # Lower threshold for more rules
            },
            "summaries": {
                "enabled": True,
                "desired_size": 10,
                "weights": None
            },
            "interestingness": {
                "enabled": True,
                "measures": ["variance", "simpson", "shannon"]
            },
            "recommendation": {
                "enabled": True,
                "method": "hybrid",  # Test most expensive method
                "top_k": 20,
                "score_threshold": 0.0  # No threshold filtering for performance test
            }
        }
    
    @pytest.fixture
    def recommender(self, performance_config):
        """Create a TupleRecommender instance."""
        return TupleRecommender(performance_config)
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_performance_with_first_session(self, dataloader, query_result_sequence, recommender):
        """Test performance using the first available session."""
        # Get available sessions
        sessions = dataloader.get_sessions()
        
        if not sessions:
            pytest.skip("No sessions found in dataset")
        
        # Use the first session
        first_session = sessions[0]
        print(f"\nTesting with session: {first_session}")
        
        # Get query pairs from the session
        query_pairs = list(query_result_sequence.iter_query_result_pairs(first_session, gap=1))
        
        if not query_pairs:
            pytest.skip(f"No valid query pairs found in session {first_session}")
        
        print(f"Found {len(query_pairs)} query pairs in session")
        
        # Performance metrics storage
        performance_results = {
            'method': [],
            'execution_time': [],
            'memory_before': [],
            'memory_after': [],
            'memory_delta': [],
            'data_size': [],
            'recommendations_count': []
        }
        
        # Test with the first few query pairs to avoid excessive runtime
        max_pairs = min(5, len(query_pairs))
        
        for i, (curr_id, fut_id, curr_results, fut_results) in enumerate(query_pairs[:max_pairs]):
            print(f"\nTesting query pair {i+1}/{max_pairs}: {curr_id} -> {fut_id}")
            print(f"Current results shape: {curr_results.shape}")
            
            # Test each recommendation method
            for method in ['association_rules', 'summaries', 'hybrid']:
                print(f"  Testing method: {method}")
                
                # Update recommender config for this method
                recommender.config['recommendation']['method'] = method
                
                # Measure memory before
                memory_before = self.get_memory_usage()
                
                # Measure execution time and memory usage
                start_time = time.time()
                
                try:
                    recommendations = recommender.recommend_tuples(curr_results, top_k=10)
                    execution_time = time.time() - start_time
                    
                    # Measure memory after
                    memory_after = self.get_memory_usage()
                    memory_delta = memory_after - memory_before
                    
                    # Store results
                    performance_results['method'].append(method)
                    performance_results['execution_time'].append(execution_time)
                    performance_results['memory_before'].append(memory_before)
                    performance_results['memory_after'].append(memory_after)
                    performance_results['memory_delta'].append(memory_delta)
                    performance_results['data_size'].append(len(curr_results))
                    performance_results['recommendations_count'].append(len(recommendations))
                    
                    print(f"    Time: {execution_time:.4f}s")
                    print(f"    Memory delta: {memory_delta:.2f}MB")
                    print(f"    Recommendations: {len(recommendations)}")
                    
                    # Basic validation
                    assert isinstance(recommendations, pd.DataFrame)
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    # Still record the timing and memory for failed attempts
                    execution_time = time.time() - start_time
                    memory_after = self.get_memory_usage()
                    memory_delta = memory_after - memory_before
                    
                    performance_results['method'].append(f"{method}_failed")
                    performance_results['execution_time'].append(execution_time)
                    performance_results['memory_before'].append(memory_before)
                    performance_results['memory_after'].append(memory_after)
                    performance_results['memory_delta'].append(memory_delta)
                    performance_results['data_size'].append(len(curr_results))
                    performance_results['recommendations_count'].append(0)
        
        # Create performance summary
        perf_df = pd.DataFrame(performance_results)
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Group by method and show statistics
        for method in perf_df['method'].unique():
            method_data = perf_df[perf_df['method'] == method]
            
            print(f"\n{method.upper()}:")
            print(f"  Average execution time: {method_data['execution_time'].mean():.4f}s")
            print(f"  Max execution time: {method_data['execution_time'].max():.4f}s")
            print(f"  Average memory delta: {method_data['memory_delta'].mean():.2f}MB")
            print(f"  Max memory delta: {method_data['memory_delta'].max():.2f}MB")
            print(f"  Average recommendations: {method_data['recommendations_count'].mean():.1f}")
            print(f"  Success rate: {(method_data['recommendations_count'] > 0).mean()*100:.1f}%")
        
        # Save performance results to file for further analysis
        output_file = Path(__file__).parent / "performance_results.csv"
        perf_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Performance assertions
        avg_time_by_method = perf_df.groupby('method')['execution_time'].mean()
        
        # Assert that no method takes more than 30 seconds on average
        for method, avg_time in avg_time_by_method.items():
            if not method.endswith('_failed'):
                assert avg_time < 30.0, f"Method {method} too slow: {avg_time:.2f}s average"
        
        # Assert that memory usage doesn't explode (no more than 500MB delta on average)
        avg_memory_by_method = perf_df.groupby('method')['memory_delta'].mean()
        for method, avg_memory in avg_memory_by_method.items():
            if not method.endswith('_failed'):
                assert avg_memory < 500.0, f"Method {method} uses too much memory: {avg_memory:.2f}MB average"

    def test_scalability_analysis(self, dataloader, query_result_sequence, recommender):
        """Test how performance scales with data size."""
        sessions = dataloader.get_sessions()
        
        if not sessions:
            pytest.skip("No sessions found in dataset")
        
        # Get different sized datasets
        scalability_results = {
            'data_size': [],
            'execution_time': [],
            'memory_delta': [],
            'recommendations_count': []
        }
        
        for session in sessions[:3]:  # Test first 3 sessions
            try:
                query_pairs = list(query_result_sequence.iter_query_result_pairs(session, gap=1))
                if not query_pairs:
                    continue
                    
                curr_id, fut_id, curr_results, fut_results = query_pairs[0]
                data_size = len(curr_results)
                
                if data_size < 10:  # Skip very small datasets
                    continue
                
                print(f"\nTesting scalability with {data_size} rows from session {session}")
                
                # Measure performance
                memory_before = self.get_memory_usage()
                start_time = time.time()
                
                recommendations = recommender.recommend_tuples(curr_results, top_k=10)
                
                execution_time = time.time() - start_time
                memory_after = self.get_memory_usage()
                memory_delta = memory_after - memory_before
                
                # Store results
                scalability_results['data_size'].append(data_size)
                scalability_results['execution_time'].append(execution_time)
                scalability_results['memory_delta'].append(memory_delta)
                scalability_results['recommendations_count'].append(len(recommendations))
                
                print(f"  Data size: {data_size}, Time: {execution_time:.4f}s, Memory: {memory_delta:.2f}MB")
                
            except Exception as e:
                print(f"  Error with session {session}: {e}")
                continue
        
        if len(scalability_results['data_size']) < 2:
            pytest.skip("Not enough data points for scalability analysis")
        
        # Analyze scalability
        scalability_df = pd.DataFrame(scalability_results)
        scalability_df = scalability_df.sort_values('data_size')
        
        print("\nScalability Analysis:")
        print(scalability_df.to_string(index=False))
        
        # Basic scalability checks
        time_per_row = scalability_df['execution_time'] / scalability_df['data_size']
        memory_per_row = scalability_df['memory_delta'] / scalability_df['data_size']
        
        print(f"\nTime per row (avg): {time_per_row.mean():.6f}s")
        print(f"Memory per row (avg): {memory_per_row.mean():.4f}MB")
        
        # Assert reasonable scalability
        assert time_per_row.max() < 0.1, f"Time per row too high: {time_per_row.max():.6f}s"
        assert memory_per_row.max() < 1.0, f"Memory per row too high: {memory_per_row.max():.4f}MB"
    
    def test_config_impact_on_performance(self, dataloader, query_result_sequence, performance_config):
        """Test how different configuration parameters impact performance."""
        sessions = dataloader.get_sessions()
        
        if not sessions:
            pytest.skip("No sessions found in dataset")
        
        # Get test data
        query_pairs = list(query_result_sequence.iter_query_result_pairs(sessions[0], gap=1))
        if not query_pairs:
            pytest.skip("No query pairs found")
        
        curr_id, fut_id, curr_results, fut_results = query_pairs[0]
        
        print(f"\nTesting configuration impact with {len(curr_results)} rows")
        
        # Test different configurations
        configs_to_test = [
            ("Low Support", {"association_rules": {"min_support": 0.01}}),
            ("High Support", {"association_rules": {"min_support": 0.2}}),
            ("Many Bins", {"discretization": {"bins": 10}}),
            ("Few Bins", {"discretization": {"bins": 3}}),
            ("Large Summary", {"summaries": {"desired_size": 20}}),
            ("Small Summary", {"summaries": {"desired_size": 3}})
        ]
        
        config_results = []
        
        for config_name, config_override in configs_to_test:
            # Create modified config
            test_config = performance_config.copy()
            
            # Deep merge the override
            for section, params in config_override.items():
                if section in test_config:
                    test_config[section].update(params)
                else:
                    test_config[section] = params
            
            # Test performance
            recommender = TupleRecommender(test_config)
            
            memory_before = self.get_memory_usage()
            start_time = time.time()
            
            try:
                recommendations = recommender.recommend_tuples(curr_results, top_k=10)
                execution_time = time.time() - start_time
                memory_after = self.get_memory_usage()
                
                config_results.append({
                    'config': config_name,
                    'execution_time': execution_time,
                    'memory_delta': memory_after - memory_before,
                    'recommendations': len(recommendations),
                    'success': True
                })
                
                print(f"  {config_name}: {execution_time:.4f}s, {memory_after - memory_before:.2f}MB")
                
            except Exception as e:
                print(f"  {config_name}: Failed - {e}")
                config_results.append({
                    'config': config_name,
                    'execution_time': time.time() - start_time,
                    'memory_delta': 0,
                    'recommendations': 0,
                    'success': False
                })
        
        # Analyze configuration impact
        config_df = pd.DataFrame(config_results)
        successful_configs = config_df[config_df['success']]
        
        if len(successful_configs) > 0:
            print(f"\nConfiguration Impact Summary:")
            print(f"  Fastest config: {successful_configs.loc[successful_configs['execution_time'].idxmin(), 'config']}")
            print(f"  Slowest config: {successful_configs.loc[successful_configs['execution_time'].idxmax(), 'config']}")
            print(f"  Most memory efficient: {successful_configs.loc[successful_configs['memory_delta'].idxmin(), 'config']}")
            
            # Assert all successful configs are reasonably fast
            assert successful_configs['execution_time'].max() < 60.0, "Some configurations are too slow"

    @pytest.fixture
    def performance_config(self):
        """Get the performance config fixture (need to re-define to use in other methods)."""
        return {
            "discretization": {
                "enabled": True,
                "method": "equal_width",
                "bins": 5,
                "save_params": False
            },
            "association_rules": {
                "enabled": True,
                "min_support": 0.05,
                "metric": "confidence",
                "min_threshold": 0.3
            },
            "summaries": {
                "enabled": True,
                "desired_size": 10,
                "weights": None
            },
            "interestingness": {
                "enabled": True,
                "measures": ["variance", "simpson", "shannon"]
            },
            "recommendation": {
                "enabled": True,
                "method": "hybrid",
                "top_k": 20,
                "score_threshold": 0.0
            }
        }


def main():
    """Main function to run all performance tests for profiling with scalene."""
    print("Running TupleRecommender Performance Tests")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestTupleRecommenderPerformance()
    
    # Setup fixtures manually (since we're not using pytest runner)
    try:
        # Get dataset directory manually
        test_dir = Path(__file__).parent
        project_root = test_dir.parent
        dataset_dir = project_root / "data" / "datasets"
        
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return 1
        
        dataset_dir = str(dataset_dir)
        print(f"Using dataset directory: {dataset_dir}")
        
        # Create dataloader
        dataloader = DataLoader(dataset_dir)
        
        # Create query sequence
        query_result_sequence = QueryResultSequence(dataloader)
        
        # Create performance config manually
        performance_config = {
            "discretization": {
                "enabled": True,
                "method": "equal_width",
                "bins": 5,
                "save_params": False
            },
            "association_rules": {
                "enabled": True,
                "min_support": 0.05,  # Lower support for more patterns
                "metric": "confidence",
                "min_threshold": 0.3  # Lower threshold for more rules
            },
            "summaries": {
                "enabled": True,
                "desired_size": 10
            },
            "recommendation": {
                "enabled": True,
                "method": "hybrid",  # Test most expensive method
                "top_k": 20,
                "score_threshold": 0.3
            }
        }
        
        # Create recommender
        recommender = TupleRecommender(performance_config)
        
        print("Fixtures created successfully")
        
        # Run test 1: Performance with first session
        print("\n" + "="*60)
        print("TEST 1: Performance with first session")
        print("="*60)
        try:
            test_instance.test_performance_with_first_session(dataloader, query_result_sequence, recommender)
            print("✓ Test 1 completed successfully")
        except Exception as e:
            print(f"✗ Test 1 failed: {e}")
        
        # # Run test 2: FP-Growth optimization
        # print("\n" + "="*60)
        # print("TEST 2: FP-Growth optimization")
        # print("="*60)
        # try:
        #     test_instance.test_fp_growth_optimization(dataloader, query_sequence, recommender)
        #     print("✓ Test 2 completed successfully")
        # except Exception as e:
        #     print(f"✗ Test 2 failed: {e}")
        
        # # Run test 3: Memory efficiency with large data
        # print("\n" + "="*60)
        # print("TEST 3: Memory efficiency with large data")
        # print("="*60)
        # try:
        #     test_instance.test_memory_efficiency_with_large_data(dataloader, query_sequence, recommender)
        #     print("✓ Test 3 completed successfully")
        # except Exception as e:
        #     print(f"✗ Test 3 failed: {e}")
        
        # print("\n" + "="*60)
        # print("ALL TESTS COMPLETED")
        # print("="*60)
        
    except Exception as e:
        print(f"Failed to setup fixtures: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
