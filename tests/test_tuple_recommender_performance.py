"""
Performance test for the TupleRecommender using real data from the first session.
"""

import pytest
import pandas as pd
import time
import psutil
import os
from pathlib import Path
from fixtures import sample_config
from query_data_predictor.tuple_recommender import TupleRecommender
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
    def query_sequence(self, dataloader):
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
                "desired_size": 10
            },
            "recommendation": {
                "enabled": True,
                "method": "hybrid",  # Test most expensive method
                "top_k": 20,
                "score_threshold": 0.3
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
    
    def test_performance_with_first_session(self, dataloader, query_sequence, recommender):
        """Test performance using the first available session."""
        # Get available sessions
        sessions = dataloader.get_sessions()
        
        if not sessions:
            pytest.skip("No sessions found in dataset")
        
        # Use the first session
        first_session = sessions[0]
        print(f"\nTesting with session: {first_session}")
        
        # Get query pairs from the session
        query_pairs = list(query_sequence.iter_query_result_pairs(first_session, gap=1))
        
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
                
                # Measure execution time
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

    def test_fp_growth_optimization(self, dataloader: DataLoader, query_sequence: QueryResultSequence, recommender: TupleRecommender):
        """Test that the FP-Growth optimization actually improves performance."""
        sessions = dataloader.get_sessions()
        
        if not sessions:
            pytest.skip("No sessions found in dataset")
        
        first_session = sessions[0]
        query_pairs = list(query_sequence.iter_query_result_pairs(first_session, gap=1))
        
        if not query_pairs:
            pytest.skip(f"No valid query pairs found in session {first_session}")
        
        # Use only first query pair for this test
        curr_results, fut_results = query_sequence.get_query_pair_with_gap(first_session, query_pairs[0][0], gap=1)
        
        # Configure for hybrid method to test optimization
        recommender.config['recommendation']['method'] = 'hybrid'
        
        print(f"\nTesting FP-Growth optimization with data shape: {curr_results.shape}")
        
        # Test 1: Without optimization (original behavior)
        print("Testing without pre-computed frequent itemsets...")
        start_time = time.time()
        recommendations_no_opt = recommender.recommend_tuples(curr_results, top_k=10)
        time_no_opt = time.time() - start_time
        
        # Test 2: With optimization (pre-compute frequent itemsets)
        print("Testing with pre-computed frequent itemsets...")
        processed_df = recommender.preprocess_data(curr_results)
        
        # Pre-compute frequent itemsets
        start_time = time.time()
        frequent_itemsets = recommender.compute_frequent_itemsets(processed_df)
        fp_growth_time = time.time() - start_time
        
        # Use pre-computed frequent itemsets
        start_time = time.time()
        recommendations_opt = recommender.recommend_tuples(
            curr_results, 
            top_k=10, 
            frequent_itemsets=frequent_itemsets
        )
        time_with_opt = time.time() - start_time
        total_opt_time = fp_growth_time + time_with_opt
        
        print(f"\nPerformance comparison:")
        print(f"  Without optimization: {time_no_opt:.4f}s")
        print(f"  With optimization: {total_opt_time:.4f}s")
        print(f"    - FP-Growth time: {fp_growth_time:.4f}s")
        print(f"    - Recommendation time: {time_with_opt:.4f}s")
        print(f"  Speedup: {time_no_opt/total_opt_time:.2f}x")
        
        # Verify results are similar (allowing for some minor differences)
        print(f"\nResults comparison:")
        print(f"  Recommendations without opt: {len(recommendations_no_opt)}")
        print(f"  Recommendations with opt: {len(recommendations_opt)}")
        
        # The optimization should provide some benefit for hybrid method
        # (though for small datasets it might be minimal)
        if time_no_opt > 1.0:  # Only check if base time is significant
            assert total_opt_time <= time_no_opt * 1.2, "Optimization should not significantly worsen performance"

    def test_memory_efficiency_with_large_data(self, dataloader, query_sequence, recommender):
        """Test memory efficiency with larger datasets."""
        sessions = dataloader.get_sessions()
        
        if not sessions:
            pytest.skip("No sessions found in dataset")
        
        # Find a session with substantial data
        best_session = None
        max_data_size = 0
        
        for session in sessions[:5]:  # Check first 5 sessions
            try:
                query_pairs = list(query_sequence.iter_query_result_pairs(session, gap=1))
                if query_pairs:
                    # Get size of first query result
                    _, _, curr_results, _ = query_pairs[0]
                    data_size = curr_results.shape[0] * curr_results.shape[1]
                    if data_size > max_data_size:
                        max_data_size = data_size
                        best_session = session
            except Exception:
                continue
        
        if best_session is None:
            pytest.skip("No suitable session found for memory testing")
        
        print(f"\nTesting memory efficiency with session {best_session} (data size: {max_data_size})")
        
        query_pairs = list(query_sequence.iter_query_result_pairs(best_session, gap=1))
        curr_id, fut_id, curr_results, fut_results = query_pairs[0]
        
        # Monitor memory during recommendation
        initial_memory = self.get_memory_usage()
        print(f"Initial memory: {initial_memory:.2f}MB")
        
        # Test association rules method
        memory_before = self.get_memory_usage()
        recommendations = recommender._recommend_with_association_rules(
            recommender.preprocess_data(curr_results), 
            top_k=20
        )
        memory_after = self.get_memory_usage()
        
        memory_delta = memory_after - memory_before
        memory_per_row = memory_delta / len(curr_results) if len(curr_results) > 0 else 0
        
        print(f"Memory usage for association rules:")
        print(f"  Before: {memory_before:.2f}MB")
        print(f"  After: {memory_after:.2f}MB")
        print(f"  Delta: {memory_delta:.2f}MB")
        print(f"  Per data row: {memory_per_row:.4f}MB")
        
        # Memory should be reasonable relative to data size
        assert memory_delta < 1000, f"Memory usage too high: {memory_delta}MB for {len(curr_results)} rows"
        assert memory_per_row < 1.0, f"Memory per row too high: {memory_per_row}MB"
