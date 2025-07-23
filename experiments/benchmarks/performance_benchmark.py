"""
Comprehensive benchmark comparing different recommender approaches using real data.

This benchmark evaluates:
1. TupleRecommender with different methods (association_rules, summaries, hybrid)
2. ClusteringRecommender 
3. InterestingnessRecommender
4. RandomRecommender (baseline)
5. DummyRecommender (baseline)

Metrics evaluated:
- Execution time
- Memory usage
- Recommendation quality (hit rate, coverage, diversity)
- Scalability with data size
- Configuration parameter sensitivity
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union
import logging
from datetime import datetime
import warnings
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import the recommender systems
from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.recommender.tuple_recommender import TupleRecommender
from query_data_predictor.recommender.clustering_recommender import ClusteringRecommender
from query_data_predictor.recommender.interestingness_recommender import InterestingnessRecommender
from query_data_predictor.recommender.random_recommender import RandomRecommender
from query_data_predictor.recommender.dummy_recommender import DummyRecommender
from query_data_predictor.metrics import EvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MemoryProfiler:
    """Simple memory profiler to track memory usage."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.time()
    yield lambda: time.time() - start
    

class ComprehensiveBenchmark:
    """
    Comprehensive benchmark suite for recommender systems.
    """
    
    def __init__(self, dataset_dir: str = "data/datasets", output_dir: str = "experiments/benchmark_results"):
        """
        Initialize the benchmark suite.
        
        Args:
            dataset_dir: Directory containing the dataset files
            output_dir: Base directory for saving all benchmark outputs
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataloader = DataLoader(str(self.dataset_dir))
        self.query_result_sequence = QueryResultSequence(self.dataloader)
        self.evaluator = EvaluationMetrics()
        
        # Configure output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Results storage
        self.benchmark_results = []
        self.memory_profiler = MemoryProfiler()
        
        # Get available sessions
        self.sessions = self.dataloader.get_sessions()[:10]  # Limit to first 10 sessions
        logger.info(f"Found {len(self.sessions)} sessions for benchmarking")
        
        # Initialize configurations for different approaches
        self.configs = self._setup_configurations()
        
    def _setup_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Set up different configurations for testing."""
        return {
            'tuple_association_rules': {
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
                    "enabled": False  # Only association rules
                },
                "recommendation": {
                    "enabled": True,
                    "method": "association_rules",
                    "top_k": 10,
                    "score_threshold": 0.0
                }
            },
            'tuple_summaries': {
                "discretization": {
                    "enabled": True,
                    "method": "equal_width",
                    "bins": 5,
                    "save_params": False
                },
                "association_rules": {
                    "enabled": False  # Only summaries
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
                    "method": "summaries",
                    "top_k": 10,
                    "score_threshold": 0.0
                }
            },
            'tuple_hybrid': {
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
                    "top_k": 10,
                    "score_threshold": 0.0
                }
            },
            'clustering': {
                "clustering": {
                    "n_clusters": 5,
                    "random_state": 42
                },
                "recommendation": {
                    "top_k": 10
                }
            },
            'interestingness_wrapper': {
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
                    "desired_size": 10
                },
                "recommendation": {
                    "top_k": 10
                }
            },
            'random_baseline': {
                "random": {
                    "random_seed": 42
                },
                "recommendation": {
                    "top_k": 10
                }
            },
            'dummy_baseline': {
                "recommendation": {
                    "top_k": 10
                }
            }
        }
    
    def _create_recommenders(self) -> Dict[str, Any]:
        """Create recommender instances with their respective configurations."""
        recommenders = {}
        
        # TupleRecommender variants
        recommenders['tuple_association_rules'] = TupleRecommender(self.configs['tuple_association_rules'])
        recommenders['tuple_summaries'] = TupleRecommender(self.configs['tuple_summaries'])
        recommenders['tuple_hybrid'] = TupleRecommender(self.configs['tuple_hybrid'])
        
        # Other recommenders
        recommenders['clustering'] = ClusteringRecommender(self.configs['clustering'])
        recommenders['interestingness_wrapper'] = InterestingnessRecommender(self.configs['interestingness_wrapper'])
        recommenders['random_baseline'] = RandomRecommender(self.configs['random_baseline'])
        recommenders['dummy_baseline'] = DummyRecommender(self.configs['dummy_baseline'])
        
        return recommenders
    
    def benchmark_session(self, session_id: str, max_pairs: int = 5) -> List[Dict[str, Any]]:
        """
        Benchmark all recommenders on a single session.
        
        Args:
            session_id: Session to benchmark
            max_pairs: Maximum number of query pairs to test
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Benchmarking session {session_id}")
        
        # Get query pairs from the session
        query_pairs = list(self.query_result_sequence.iter_query_result_pairs(session_id, gap=1))
        
        if not query_pairs:
            logger.warning(f"No query pairs found in session {session_id}")
            return []
        
        # Limit query pairs to avoid excessive runtime
        query_pairs = query_pairs[:max_pairs]
        logger.info(f"Testing {len(query_pairs)} query pairs")
        
        # Create recommenders for this session
        recommenders = self._create_recommenders()
        
        session_results = []
        
        for pair_idx, (curr_id, fut_id, curr_results, fut_results) in enumerate(query_pairs):
            logger.info(f"  Pair {pair_idx + 1}/{len(query_pairs)}: {curr_id} -> {fut_id}")
            logger.info(f"  Current results: {curr_results.shape}, Future results: {fut_results.shape}")
            
            if curr_results.empty or len(curr_results) < 2:
                logger.warning(f"  Skipping pair with insufficient current results")
                continue
            
            for recommender_name, recommender in recommenders.items():
                result = self._benchmark_single_recommendation(
                    recommender_name, recommender, curr_results, fut_results,
                    session_id, curr_id, fut_id
                )
                if result:
                    session_results.append(result)
        
        return session_results
    
    def _benchmark_single_recommendation(
        self, 
        recommender_name: str, 
        recommender: Any, 
        curr_results: pd.DataFrame, 
        fut_results: pd.DataFrame,
        session_id: str,
        curr_id: str,
        fut_id: str
    ) -> Dict[str, Any]:
        """
        Benchmark a single recommendation operation.
        
        Returns:
            Dictionary with benchmark results or None if failed
        """
        logger.info(f"    Testing {recommender_name}")
        
        # Record initial memory
        memory_before = self.memory_profiler.get_memory_usage()
        
        try:
            with timer() as get_time:
                # Make recommendation
                recommendations = recommender.recommend_tuples(curr_results)
            
            execution_time = get_time()
            memory_after = self.memory_profiler.get_memory_usage()
            memory_delta = memory_after - memory_before
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                recommendations, fut_results, curr_results
            )
            
            result = {
                'session_id': session_id,
                'current_query_id': curr_id,
                'future_query_id': fut_id,
                'recommender': recommender_name,
                'execution_time': execution_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta,
                'current_data_size': len(curr_results),
                'future_data_size': len(fut_results),
                'recommendations_count': len(recommendations),
                **quality_metrics,
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"      Time: {execution_time:.4f}s, Memory: {memory_delta:.2f}MB, "
                       f"Recommendations: {len(recommendations)}")
            
            return result
            
        except Exception as e:
            logger.error(f"      Error: {str(e)}")
            
            return {
                'session_id': session_id,
                'current_query_id': curr_id,
                'future_query_id': fut_id,
                'recommender': recommender_name,
                'execution_time': 0.0,
                'memory_before': memory_before,
                'memory_after': self.memory_profiler.get_memory_usage(),
                'memory_delta': 0.0,
                'current_data_size': len(curr_results),
                'future_data_size': len(fut_results),
                'recommendations_count': 0,
                'hit_rate': 0.0,
                'coverage': 0.0,
                'diversity': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_quality_metrics(
        self, 
        recommendations: pd.DataFrame, 
        future_results: pd.DataFrame,
        current_results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate recommendation quality metrics.
        
        Args:
            recommendations: Recommended tuples
            future_results: Actual future query results
            current_results: Current query results
            
        Returns:
            Dictionary with quality metrics
        """
        if recommendations.empty or future_results.empty:
            return {
                'hit_rate': 0.0,
                'coverage': 0.0,
                'diversity': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        try:
            # Hit rate: percentage of recommendations that appear in future results
            # Convert to string representation for comparison
            rec_strings = set(recommendations.apply(lambda x: str(tuple(x)), axis=1))
            fut_strings = set(future_results.apply(lambda x: str(tuple(x)), axis=1))
            
            hits = len(rec_strings.intersection(fut_strings))
            hit_rate = hits / len(rec_strings) if len(rec_strings) > 0 else 0.0
            
            # Coverage: percentage of future results covered by recommendations
            coverage = hits / len(fut_strings) if len(fut_strings) > 0 else 0.0
            
            # Diversity: how diverse are the recommendations (unique values per column)
            diversity = 0.0
            if not recommendations.empty and len(recommendations.columns) > 0:
                unique_ratios = []
                for col in recommendations.columns:
                    unique_count = recommendations[col].nunique()
                    total_count = len(recommendations)
                    unique_ratios.append(unique_count / total_count if total_count > 0 else 0)
                diversity = np.mean(unique_ratios)
            
            # Precision and Recall
            precision = hit_rate  # Same as hit rate in this context
            recall = coverage     # Same as coverage in this context
            
            return {
                'hit_rate': hit_rate,
                'coverage': coverage,
                'diversity': diversity,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {
                'hit_rate': 0.0,
                'coverage': 0.0,
                'diversity': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
    
    def run_comprehensive_benchmark(self, max_sessions: int = 5) -> pd.DataFrame:
        """
        Run comprehensive benchmark across multiple sessions.
        
        Args:
            max_sessions: Maximum number of sessions to test
            
        Returns:
            DataFrame with all benchmark results
        """
        logger.info("Starting comprehensive benchmark")
        logger.info(f"Testing {min(max_sessions, len(self.sessions))} sessions")
        
        all_results = []
        
        for session_idx, session_id in enumerate(self.sessions[:max_sessions]):
            logger.info(f"\nSession {session_idx + 1}/{min(max_sessions, len(self.sessions))}: {session_id}")
            
            try:
                session_results = self.benchmark_session(session_id, max_pairs=3)
                all_results.extend(session_results)
                
            except Exception as e:
                logger.error(f"Error benchmarking session {session_id}: {e}")
                continue
        
        if not all_results:
            logger.error("No benchmark results obtained")
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        logger.info(f"\\nBenchmark completed. Total results: {len(results_df)}")
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze benchmark results and generate insights.
        
        Args:
            results_df: DataFrame with benchmark results
            
        Returns:
            Dictionary with analysis results
        """
        if results_df.empty:
            return {}
        
        logger.info("Analyzing benchmark results")
        
        # Filter successful results for most analyses
        successful_results = results_df[results_df['success'] == True]
        
        analysis = {
            'total_tests': len(results_df),
            'successful_tests': len(successful_results),
            'success_rate': len(successful_results) / len(results_df) if len(results_df) > 0 else 0,
        }
        
        if successful_results.empty:
            logger.warning("No successful results to analyze")
            return analysis
        
        # Performance analysis
        performance_by_method = successful_results.groupby('recommender').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'memory_delta': ['mean', 'std', 'min', 'max'],
            'current_data_size': 'mean',
            'recommendations_count': 'mean'
        }).round(4)
        
        # Quality analysis
        quality_by_method = successful_results.groupby('recommender').agg({
            'hit_rate': ['mean', 'std'],
            'coverage': ['mean', 'std'],
            'diversity': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).round(4)
        
        # Scalability analysis
        scalability_correlation = {}
        for method in successful_results['recommender'].unique():
            method_data = successful_results[successful_results['recommender'] == method]
            if len(method_data) > 2:
                time_size_corr = method_data['execution_time'].corr(method_data['current_data_size'])
                memory_size_corr = method_data['memory_delta'].corr(method_data['current_data_size'])
                scalability_correlation[method] = {
                    'time_size_correlation': time_size_corr,
                    'memory_size_correlation': memory_size_corr
                }
        
        analysis.update({
            'performance_by_method': performance_by_method,
            'quality_by_method': quality_by_method,
            'scalability_correlation': scalability_correlation,
            'overall_stats': {
                'avg_execution_time': successful_results['execution_time'].mean(),
                'avg_memory_delta': successful_results['memory_delta'].mean(),
                'avg_hit_rate': successful_results['hit_rate'].mean(),
                'avg_coverage': successful_results['coverage'].mean(),
                'avg_diversity': successful_results['diversity'].mean()
            }
        })
        
        return analysis
    
    def generate_visualizations(self, results_df: pd.DataFrame):
        """
        Generate visualizations for benchmark results.
        
        Args:
            results_df: DataFrame with benchmark results
        """
        successful_results = results_df[results_df['success'] == True]
        if successful_results.empty:
            logger.warning("No successful results to visualize")
            return
        
        logger.info(f"Generating visualizations in {self.output_dir}")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time comparison
        successful_results.boxplot(column='execution_time', by='recommender', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time by Recommender')
        axes[0, 0].set_xlabel('Recommender')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        successful_results.boxplot(column='memory_delta', by='recommender', ax=axes[0, 1])
        axes[0, 1].set_title('Memory Usage by Recommender')
        axes[0, 1].set_xlabel('Recommender')
        axes[0, 1].set_ylabel('Memory Delta (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Hit rate comparison
        successful_results.boxplot(column='hit_rate', by='recommender', ax=axes[1, 0])
        axes[1, 0].set_title('Hit Rate by Recommender')
        axes[1, 0].set_xlabel('Recommender')
        axes[1, 0].set_ylabel('Hit Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Coverage comparison
        successful_results.boxplot(column='coverage', by='recommender', ax=axes[1, 1])
        axes[1, 1].set_title('Coverage by Recommender')
        axes[1, 1].set_xlabel('Recommender')
        axes[1, 1].set_ylabel('Coverage')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scalability analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for method in successful_results['recommender'].unique():
            method_data = successful_results[successful_results['recommender'] == method]
            if len(method_data) > 1:
                axes[0].scatter(method_data['current_data_size'], method_data['execution_time'], 
                              label=method, alpha=0.7)
                axes[1].scatter(method_data['current_data_size'], method_data['memory_delta'], 
                              label=method, alpha=0.7)
        
        axes[0].set_xlabel('Data Size')
        axes[0].set_ylabel('Execution Time (seconds)')
        axes[0].set_title('Execution Time vs Data Size')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_xlabel('Data Size')
        axes[1].set_ylabel('Memory Delta (MB)')
        axes[1].set_title('Memory Usage vs Data Size')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Success rate analysis
        success_rates = results_df.groupby('recommender')['success'].mean()
        
        plt.figure(figsize=(12, 6))
        success_rates.plot(kind='bar')
        plt.title('Success Rate by Recommender')
        plt.xlabel('Recommender')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def save_results(self, results_df: pd.DataFrame, analysis: Dict[str, Any]):
        """
        Save benchmark results and analysis to files.
        
        Args:
            results_df: DataFrame with benchmark results
            analysis: Analysis results dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save analysis
        analysis_file = self.output_dir / f"benchmark_analysis_{timestamp}.json"
        # Convert numpy types to JSON serializable types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                # Convert DataFrame to dict, handling multi-level columns
                df_dict = {}
                for col in obj.columns:
                    if isinstance(col, tuple):
                        # Convert tuple column names to strings
                        col_key = "_".join(str(x) for x in col)
                    else:
                        col_key = str(col)
                    df_dict[col_key] = obj[col].to_dict()
                return df_dict
            return obj
        
        # Deep convert analysis dictionary
        def deep_convert(d):
            if isinstance(d, dict):
                converted = {}
                for k, v in d.items():
                    # Ensure all dictionary keys are strings
                    if isinstance(k, tuple):
                        key = "_".join(str(x) for x in k)
                    else:
                        key = str(k)
                    converted[key] = deep_convert(v)
                return converted
            elif isinstance(d, list):
                return [deep_convert(v) for v in d]
            else:
                return convert_numpy(d)
        
        analysis_serializable = deep_convert(analysis)
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_serializable, f, indent=2, default=str)
        
        # Save summary report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        self._generate_text_report(results_df, analysis, report_file)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"  Raw results: {results_file}")
        logger.info(f"  Analysis: {analysis_file}")
        logger.info(f"  Report: {report_file}")
    
    def _generate_text_report(self, results_df: pd.DataFrame, analysis: Dict[str, Any], output_file: Path):
        """Generate a human-readable text report."""
        with open(output_file, 'w') as f:
            f.write("COMPREHENSIVE RECOMMENDER BENCHMARK REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total tests: {analysis.get('total_tests', 0)}\\n")
            f.write(f"Successful tests: {analysis.get('successful_tests', 0)}\\n")
            f.write(f"Success rate: {analysis.get('success_rate', 0):.2%}\\n\\n")
            
            if 'overall_stats' in analysis:
                f.write("OVERALL PERFORMANCE STATISTICS\\n")
                f.write("-" * 30 + "\\n")
                stats = analysis['overall_stats']
                f.write(f"Average execution time: {stats.get('avg_execution_time', 0):.4f}s\\n")
                f.write(f"Average memory delta: {stats.get('avg_memory_delta', 0):.2f}MB\\n")
                f.write(f"Average hit rate: {stats.get('avg_hit_rate', 0):.4f}\\n")
                f.write(f"Average coverage: {stats.get('avg_coverage', 0):.4f}\\n")
                f.write(f"Average diversity: {stats.get('avg_diversity', 0):.4f}\\n\\n")
            
            # Method comparison
            if not results_df.empty:
                successful = results_df[results_df['success'] == True]
                if not successful.empty:
                    f.write("PERFORMANCE BY METHOD\\n")
                    f.write("-" * 20 + "\\n")
                    
                    for method in successful['recommender'].unique():
                        method_data = successful[successful['recommender'] == method]
                        f.write(f"\\n{method.upper()}:\\n")
                        f.write(f"  Tests: {len(method_data)}\\n")
                        f.write(f"  Avg execution time: {method_data['execution_time'].mean():.4f}s\\n")
                        f.write(f"  Avg memory delta: {method_data['memory_delta'].mean():.2f}MB\\n")
                        f.write(f"  Avg hit rate: {method_data['hit_rate'].mean():.4f}\\n")
                        f.write(f"  Avg coverage: {method_data['coverage'].mean():.4f}\\n")
                        f.write(f"  Avg diversity: {method_data['diversity'].mean():.4f}\\n")
            
            # Failure analysis
            failures = results_df[results_df['success'] == False]
            if not failures.empty:
                f.write("\\n\\nFAILURE ANALYSIS\\n")
                f.write("-" * 15 + "\\n")
                failure_counts = failures['recommender'].value_counts()
                for method, count in failure_counts.items():
                    f.write(f"{method}: {count} failures\\n")
        
        logger.info(f"Text report saved to {output_file}")
    
    @property
    def output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dirna
    
    def get_output_info(self) -> Dict[str, str]:
        """Get information about output directory and files."""
        return {
            'output_directory': str(self.output_dir),
            'timestamp': self.output_dir.name.split('_')[-1] if '_' in self.output_dir.name else '',
            'exists': self.output_dir.exists(),
            'files_count': len(list(self.output_dir.glob('*'))) if self.output_dir.exists() else 0
        }


def main():
    """Main function to run the comprehensive benchmark."""
    print("Starting Comprehensive Recommender Benchmark")
    print("=" * 50)
    
    # Initialize benchmark with output directory
    try:
        benchmark = ComprehensiveBenchmark(output_dir="experiments/benchmark_results")
        print(f"Initialized benchmark with {len(benchmark.sessions)} sessions")
        print(f"Output directory: {benchmark.output_dir}")
    except Exception as e:
        print(f"Failed to initialize benchmark: {e}")
        return 1
    
    # Run benchmark
    try:
        print("\\nRunning comprehensive benchmark...")
        results_df = benchmark.run_comprehensive_benchmark(max_sessions=3)
        
        if results_df.empty:
            print("No results obtained from benchmark")
            return 1
            
        print(f"Benchmark completed with {len(results_df)} total results")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    # Analyze results
    try:
        print("\\nAnalyzing results...")
        analysis = benchmark.analyze_results(results_df)
        print(f"Analysis completed")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        analysis = {}
    
    # Generate visualizations
    try:
        print("\\nGenerating visualizations...")
        benchmark.generate_visualizations(results_df)
        print("Visualizations generated")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    # Save results
    try:
        print("\\nSaving results...")
        benchmark.save_results(results_df, analysis)
        print("Results saved")
        
    except Exception as e:
        print(f"Failed to save results: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    if not results_df.empty:
        successful_results = results_df[results_df['success'] == True]
        
        print(f"Total tests: {len(results_df)}")
        print(f"Successful tests: {len(successful_results)}")
        print(f"Success rate: {len(successful_results) / len(results_df):.2%}")
        
        if not successful_results.empty:
            print(f"\\nAverage performance:")
            print(f"  Execution time: {successful_results['execution_time'].mean():.4f}s")
            print(f"  Memory usage: {successful_results['memory_delta'].mean():.2f}MB")
            print(f"  Hit rate: {successful_results['hit_rate'].mean():.4f}")
            print(f"  Coverage: {successful_results['coverage'].mean():.4f}")
            
            print(f"\\nBest performing methods:")
            best_time = successful_results.loc[successful_results['execution_time'].idxmin(), 'recommender']
            best_memory = successful_results.loc[successful_results['memory_delta'].idxmin(), 'recommender']
            best_hit_rate = successful_results.loc[successful_results['hit_rate'].idxmax(), 'recommender']
            
            print(f"  Fastest: {best_time}")
            print(f"  Most memory efficient: {best_memory}")
            print(f"  Highest hit rate: {best_hit_rate}")
    
    print("\\nBenchmark completed successfully!")
    print(f"All results saved to: {benchmark.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
