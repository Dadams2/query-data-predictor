"""
Comprehensive experiment runner for evaluating recommender systems.
"""

import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
import warnings
import signal
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.metrics import EvaluationMetrics
from query_data_predictor.recommender import (
    DummyRecommender,
    RandomRecommender,
    ClusteringRecommender,
    InterestingnessRecommender
)

from query_data_predictor.logging_config import setup_logging

# Configure logging and visualization
setup_logging()
logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeouts."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class RecommenderExperimentRunner:
    """
    Experiment runner for evaluating different recommender systems.
    """
    
    def __init__(self, config_path: str = "config.yaml", dataset_dir: str = "data/datasets"):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to configuration file
            dataset_dir: Directory containing the dataset files
        """
        self.config = self._load_config(config_path)
        self.dataset_dir = Path(dataset_dir)
        self.dataloader = DataLoader(str(self.dataset_dir))
        self.query_result_sequence = QueryResultSequence(self.dataloader)
        self.evaluator = EvaluationMetrics()
        
        # Results storage
        self.results = []
        
        # Initialize recommenders with top_quartile mode
        self.recommender_config = self._create_recommender_config()
        self.recommenders = self._initialize_recommenders()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'recommendation': {
                    'mode': 'top_quartile'
                },
                'clustering': {
                    'n_clusters': 3,
                    'random_state': 42
                },
                'random': {
                    'random_seed': 42
                },
                'discretization': {
                    'enabled': True,
                    'method': 'equal_width',
                    'bins': 5
                },
                'association_rules': {
                    'min_support': 0.1,
                    'metric': 'confidence',
                    'min_threshold': 0.7
                }
            }
    
    def _create_recommender_config(self) -> Dict[str, Any]:
        """Create standardized configuration for all recommenders."""
        base_config = self.config.copy()
        base_config['recommendation'] = {
            'mode': 'top_quartile'  # Use top_quartile as requested
        }
        return base_config
    
    def _initialize_recommenders(self) -> Dict[str, Any]:
        """Initialize all recommender instances."""
        return {
            'dummy': DummyRecommender(self.recommender_config),
            'random': RandomRecommender(self.recommender_config),
            'clustering': ClusteringRecommender(self.recommender_config),
            'interestingness': InterestingnessRecommender(self.recommender_config)
        }
    
    def run_experiment(self, session_id: str, max_gap: int = 5) -> pd.DataFrame:
        """
        Run comprehensive experiments on a single session.
        
        Args:
            session_id: ID of the session to experiment on
            max_gap: Maximum gap between queries to test
            
        Returns:
            DataFrame with all experiment results
        """
        logger.info(f"Starting experiment on session {session_id}")
        
        # Get all query IDs for this session
        query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)
        if len(query_ids) < 2:
            logger.warning(f"Session {session_id} has fewer than 2 queries, skipping")
            return pd.DataFrame()
        
        logger.info(f"Session {session_id} has {len(query_ids)} queries")
        
        # Run experiments for different gaps
        for gap in range(1, min(max_gap + 1, len(query_ids))):
            logger.info(f"Testing gap {gap}")
            self._run_gap_experiment(session_id, gap)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        logger.info(f"Completed experiment with {len(results_df)} result records")
        
        return results_df
    
    def _run_gap_experiment(self, session_id: str, gap: int):
        """Run experiments for a specific gap between queries."""
        try:
            # Iterate through all valid query pairs with this gap
            for current_id, future_id, current_results, future_results in \
                self.query_result_sequence.iter_query_result_pairs(session_id, gap):
                
                # Skip if current results are empty
                if current_results.empty:
                    continue
                
                # Test each recommender
                for recommender_name, recommender in self.recommenders.items():
                    self._evaluate_recommender(
                        session_id=session_id,
                        current_query_id=current_id,
                        future_query_id=future_id,
                        current_results=current_results,
                        future_results=future_results,
                        recommender_name=recommender_name,
                        recommender=recommender,
                        gap=gap
                    )
                    
        except Exception as e:
            logger.error(f"Error in gap {gap} experiment for session {session_id}: {str(e)}")
    
    def _evaluate_recommender(self, session_id: str, current_query_id: str, 
                            future_query_id: str, current_results: pd.DataFrame,
                            future_results: pd.DataFrame, recommender_name: str,
                            recommender: Any, gap: int):
        """Evaluate a single recommender on a query pair with timeout protection."""
        start_time = time.time()
        
        try:
            # Set timeout based on dataset size (30 seconds for small, 120 for large)
            timeout_seconds = 30 if len(current_results) < 100 else 120
            
            with timeout(timeout_seconds):
                # Get recommendations
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    recommendations = recommender.recommend_tuples(current_results)
            
            # Calculate overlap accuracy
            overlap_accuracy = self.evaluator.overlap_accuracy(
                previous=current_results,
                actual=future_results,
                predicted=recommendations
            )
            
            execution_time = time.time() - start_time
            
            # Record the result
            result_record = {
                'timestamp': datetime.now(),
                'session_id': session_id,
                'current_query_id': current_query_id,
                'future_query_id': future_query_id,
                'gap': gap,
                'recommender': recommender_name,
                'current_result_size': len(current_results),
                'future_result_size': len(future_results),
                'recommendations_size': len(recommendations),
                'overlap_accuracy': overlap_accuracy,
                'current_query_position': current_query_id,
                'future_query_position': future_query_id,
                'execution_time': execution_time
            }
            
            self.results.append(result_record)
            
            logger.debug(f"Evaluated {recommender_name} for gap {gap}: "
                        f"current_size={len(current_results)}, "
                        f"recommendations={len(recommendations)}, "
                        f"overlap_accuracy={overlap_accuracy:.4f}, "
                        f"time={execution_time:.2f}s")
                        
        except TimeoutError as e:
            execution_time = time.time() - start_time
            logger.error(f"Timeout for {recommender_name} on gap {gap}: {str(e)}")
            # Record the timeout
            result_record = {
                'timestamp': datetime.now(),
                'session_id': session_id,
                'current_query_id': current_query_id,
                'future_query_id': future_query_id,
                'gap': gap,
                'recommender': recommender_name,
                'current_result_size': len(current_results),
                'future_result_size': len(future_results),
                'recommendations_size': 0,
                'overlap_accuracy': 0.0,
                'current_query_position': current_query_id,
                'future_query_position': future_query_id,
                'execution_time': execution_time,
                'error': f'TIMEOUT after {timeout_seconds}s'
            }
            self.results.append(result_record)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error evaluating {recommender_name} for gap {gap}: {str(e)}")
            # Record the failure
            result_record = {
                'timestamp': datetime.now(),
                'session_id': session_id,
                'current_query_id': current_query_id,
                'future_query_id': future_query_id,
                'gap': gap,
                'recommender': recommender_name,
                'current_result_size': len(current_results),
                'future_result_size': len(future_results),
                'recommendations_size': 0,
                'overlap_accuracy': 0.0,
                'current_query_position': current_query_id,
                'future_query_position': future_query_id,
                'execution_time': execution_time,
                'error': str(e)
            }
            self.results.append(result_record)
    
    def save_results(self, output_path: str = "experiment_results.csv"):
        """Save results to CSV file."""
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.warning("No results to save")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experimental results and return summary statistics."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Filter out error cases for analysis
        valid_df = df[df['overlap_accuracy'].notna() & (df['overlap_accuracy'] >= 0)]
        
        analysis = {
            'total_experiments': len(df),
            'valid_experiments': len(valid_df),
            'error_rate': (len(df) - len(valid_df)) / len(df) if len(df) > 0 else 0,
        }
        
        if len(valid_df) > 0:
            # Average accuracy per recommender
            avg_accuracy = valid_df.groupby('recommender')['overlap_accuracy'].agg(['mean', 'std', 'count'])
            analysis['average_accuracy_by_recommender'] = avg_accuracy.to_dict()
            
            # Accuracy across session (by query position)
            accuracy_by_position = valid_df.groupby(['recommender', 'current_query_position'])['overlap_accuracy'].mean()
            analysis['accuracy_by_position'] = accuracy_by_position.to_dict()
            
            # Accuracy versus result set size
            valid_df['result_size_bin'] = pd.cut(valid_df['current_result_size'], 
                                               bins=[0, 10, 25, 50, 100, float('inf')], 
                                               labels=['1-10', '11-25', '26-50', '51-100', '100+'])
            accuracy_by_size = valid_df.groupby(['recommender', 'result_size_bin'])['overlap_accuracy'].mean()
            analysis['accuracy_by_result_size'] = accuracy_by_size.to_dict()
            
            # Accuracy versus gap
            accuracy_by_gap = valid_df.groupby(['recommender', 'gap'])['overlap_accuracy'].mean()
            analysis['accuracy_by_gap'] = accuracy_by_gap.to_dict()
        
        return analysis
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        analysis = self.analyze_results()
        
        if not analysis:
            return "No results to analyze."
        
        report = []
        report.append("=" * 60)
        report.append("RECOMMENDER SYSTEM EXPERIMENT RESULTS")
        report.append("=" * 60)
        report.append(f"Total experiments: {analysis['total_experiments']}")
        report.append(f"Valid experiments: {analysis['valid_experiments']}")
        report.append(f"Error rate: {analysis['error_rate']:.2%}")
        report.append("")
        
        if 'average_accuracy_by_recommender' in analysis:
            report.append("AVERAGE OVERLAP ACCURACY BY RECOMMENDER:")
            report.append("-" * 40)
            for recommender, stats in analysis['average_accuracy_by_recommender']['mean'].items():
                std = analysis['average_accuracy_by_recommender']['std'].get(recommender, 0)
                count = analysis['average_accuracy_by_recommender']['count'].get(recommender, 0)
                report.append(f"{recommender:15}: {stats:.4f} ± {std:.4f} (n={count})")
            report.append("")
        
        return "\n".join(report)

    def create_visualizations(self, output_dir: str = "experiment_results", 
                            save_pdf: bool = True) -> None:
        """
        Create comprehensive visualizations of the experiment results.
        
        Args:
            output_dir: Directory to save visualization files
            save_pdf: Whether to save all plots in a single PDF file
        """
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        # Filter out error cases for visualization
        valid_df = df[df['overlap_accuracy'].notna() & (df['overlap_accuracy'] >= 0)]
        
        if len(valid_df) == 0:
            logger.warning("No valid results to visualize")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for visualization
        valid_df = self._prepare_visualization_data(valid_df)
        
        if save_pdf:
            pdf_path = output_path / f"recommender_experiment_plots_{timestamp}.pdf"
            with PdfPages(pdf_path) as pdf:
                self._create_all_plots(valid_df, pdf, output_path, timestamp)
            logger.info(f"All plots saved to PDF: {pdf_path}")
        else:
            self._create_all_plots(valid_df, None, output_path, timestamp)
    
    def _prepare_visualization_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for visualization by adding derived columns."""
        df = df.copy()
        
        # Create result size bins
        df['result_size_bin'] = pd.cut(
            df['current_result_size'], 
            bins=[0, 10, 25, 50, 100, float('inf')], 
            labels=['1-10', '11-25', '26-50', '51-100', '100+']
        )
        
        # Create execution time bins
        df['exec_time_bin'] = pd.cut(
            df['execution_time'], 
            bins=[0, 1, 5, 10, 30, float('inf')], 
            labels=['<1s', '1-5s', '5-10s', '10-30s', '>30s']
        )
        
        # Create gap categories
        df['gap_category'] = df['gap'].apply(
            lambda x: 'Short (1-2)' if x <= 2 else 'Medium (3-4)' if x <= 4 else 'Long (5+)'
        )
        
        return df
    
    def _create_all_plots(self, df: pd.DataFrame, pdf: PdfPages = None, 
                         output_path: Path = None, timestamp: str = None):
        """Create all visualization plots."""
        
        # 1. Overall accuracy comparison
        self._plot_accuracy_comparison(df, pdf, output_path, timestamp)
        
        # 2. Accuracy by gap
        self._plot_accuracy_by_gap(df, pdf, output_path, timestamp)
        
        # 3. Accuracy by result size
        self._plot_accuracy_by_result_size(df, pdf, output_path, timestamp)
        
        # 4. Execution time analysis
        self._plot_execution_time_analysis(df, pdf, output_path, timestamp)
        
        # 5. Performance heatmap
        self._plot_performance_heatmap(df, pdf, output_path, timestamp)
        
        # 6. Distribution plots
        self._plot_accuracy_distributions(df, pdf, output_path, timestamp)
        
        # 7. Correlation analysis
        self._plot_correlation_analysis(df, pdf, output_path, timestamp)
        
        # 8. Time series analysis (if applicable)
        self._plot_time_series_analysis(df, pdf, output_path, timestamp)
        
        # 9. Error analysis
        self._plot_error_analysis(df, pdf, output_path, timestamp)
    
    def _plot_accuracy_comparison(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                output_path: Path = None, timestamp: str = None):
        """Create overall accuracy comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(data=df, x='recommender', y='overlap_accuracy', ax=ax1)
        ax1.set_title('Overlap Accuracy Distribution by Recommender')
        ax1.set_xlabel('Recommender')
        ax1.set_ylabel('Overlap Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Bar plot with error bars
        accuracy_stats = df.groupby('recommender')['overlap_accuracy'].agg(['mean', 'std'])
        sns.barplot(data=df, x='recommender', y='overlap_accuracy', 
                   estimator=np.mean, ci=95, ax=ax2)
        ax2.set_title('Mean Overlap Accuracy by Recommender (with 95% CI)')
        ax2.set_xlabel('Recommender')
        ax2.set_ylabel('Mean Overlap Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'accuracy_comparison_{timestamp}')
    
    def _plot_accuracy_by_gap(self, df: pd.DataFrame, pdf: PdfPages = None, 
                            output_path: Path = None, timestamp: str = None):
        """Create accuracy by gap analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Line plot
        gap_accuracy = df.groupby(['recommender', 'gap'])['overlap_accuracy'].mean().reset_index()
        sns.lineplot(data=gap_accuracy, x='gap', y='overlap_accuracy', 
                    hue='recommender', marker='o', ax=ax1)
        ax1.set_title('Accuracy vs Query Gap')
        ax1.set_xlabel('Gap Between Queries')
        ax1.set_ylabel('Mean Overlap Accuracy')
        ax1.legend(title='Recommender')
        
        # Heatmap
        pivot_gap = gap_accuracy.pivot(index='recommender', columns='gap', values='overlap_accuracy')
        sns.heatmap(pivot_gap, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Accuracy Heatmap by Gap')
        ax2.set_xlabel('Gap Between Queries')
        ax2.set_ylabel('Recommender')
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'accuracy_by_gap_{timestamp}')
    
    def _plot_accuracy_by_result_size(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                    output_path: Path = None, timestamp: str = None):
        """Create accuracy by result size analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Violin plot
        sns.violinplot(data=df, x='result_size_bin', y='overlap_accuracy', 
                      hue='recommender', ax=ax1)
        ax1.set_title('Accuracy Distribution by Result Set Size')
        ax1.set_xlabel('Result Set Size')
        ax1.set_ylabel('Overlap Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Scatter plot with trend lines
        sns.scatterplot(data=df, x='current_result_size', y='overlap_accuracy', 
                       hue='recommender', alpha=0.6, ax=ax2)
        
        # Add trend lines
        for recommender in df['recommender'].unique():
            subset = df[df['recommender'] == recommender]
            if len(subset) > 1:
                sns.regplot(data=subset, x='current_result_size', y='overlap_accuracy', 
                           scatter=False, ax=ax2, label=f'{recommender} trend')
        
        ax2.set_title('Accuracy vs Result Set Size (with trends)')
        ax2.set_xlabel('Current Result Set Size')
        ax2.set_ylabel('Overlap Accuracy')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'accuracy_by_result_size_{timestamp}')
    
    def _plot_execution_time_analysis(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                    output_path: Path = None, timestamp: str = None):
        """Create execution time analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time by recommender
        sns.boxplot(data=df, x='recommender', y='execution_time', ax=ax1)
        ax1.set_title('Execution Time by Recommender')
        ax1.set_xlabel('Recommender')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', rotation=45)
        
        # Execution time vs result size
        sns.scatterplot(data=df, x='current_result_size', y='execution_time', 
                       hue='recommender', alpha=0.6, ax=ax2)
        ax2.set_title('Execution Time vs Result Set Size')
        ax2.set_xlabel('Current Result Set Size')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Execution time distribution
        sns.histplot(data=df, x='execution_time', hue='recommender', 
                    multiple='stack', bins=30, ax=ax3)
        ax3.set_title('Execution Time Distribution')
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Count')
        ax3.set_xscale('log')
        
        # Performance efficiency (accuracy per second)
        df_temp = df.copy()
        df_temp['efficiency'] = df_temp['overlap_accuracy'] / (df_temp['execution_time'] + 0.001)
        sns.barplot(data=df_temp, x='recommender', y='efficiency', ax=ax4)
        ax4.set_title('Performance Efficiency (Accuracy per Second)')
        ax4.set_xlabel('Recommender')
        ax4.set_ylabel('Efficiency (Accuracy/Second)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'execution_time_analysis_{timestamp}')
    
    def _plot_performance_heatmap(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                output_path: Path = None, timestamp: str = None):
        """Create performance heatmap across multiple dimensions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy heatmap by recommender and gap
        pivot_accuracy = df.groupby(['recommender', 'gap'])['overlap_accuracy'].mean().unstack()
        sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1)
        ax1.set_title('Mean Accuracy: Recommender vs Gap')
        
        # Execution time heatmap
        pivot_time = df.groupby(['recommender', 'gap'])['execution_time'].mean().unstack()
        sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Mean Execution Time: Recommender vs Gap')
        
        # Accuracy by recommender and result size bin
        pivot_size = df.groupby(['recommender', 'result_size_bin'])['overlap_accuracy'].mean().unstack()
        sns.heatmap(pivot_size, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3)
        ax3.set_title('Mean Accuracy: Recommender vs Result Size')
        
        # Count heatmap
        pivot_count = df.groupby(['recommender', 'gap']).size().unstack()
        sns.heatmap(pivot_count, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Number of Experiments: Recommender vs Gap')
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'performance_heatmap_{timestamp}')
    
    def _plot_accuracy_distributions(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                   output_path: Path = None, timestamp: str = None):
        """Create accuracy distribution plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall distribution
        sns.histplot(data=df, x='overlap_accuracy', hue='recommender', 
                    multiple='dodge', bins=20, ax=ax1)
        ax1.set_title('Accuracy Distribution by Recommender')
        ax1.set_xlabel('Overlap Accuracy')
        ax1.set_ylabel('Count')
        
        # KDE plot
        sns.kdeplot(data=df, x='overlap_accuracy', hue='recommender', ax=ax2)
        ax2.set_title('Accuracy Density by Recommender')
        ax2.set_xlabel('Overlap Accuracy')
        ax2.set_ylabel('Density')
        
        # Cumulative distribution
        for recommender in df['recommender'].unique():
            subset = df[df['recommender'] == recommender]['overlap_accuracy']
            ax3.hist(subset, bins=20, alpha=0.5, cumulative=True, 
                    density=True, label=recommender)
        ax3.set_title('Cumulative Accuracy Distribution')
        ax3.set_xlabel('Overlap Accuracy')
        ax3.set_ylabel('Cumulative Probability')
        ax3.legend()
        
        # Q-Q plots
        recommenders = df['recommender'].unique()
        if len(recommenders) >= 2:
            from scipy import stats
            rec1_data = df[df['recommender'] == recommenders[0]]['overlap_accuracy']
            rec2_data = df[df['recommender'] == recommenders[1]]['overlap_accuracy']
            stats.probplot(rec1_data, dist="norm", plot=ax4)
            ax4.set_title(f'Q-Q Plot: {recommenders[0]} vs Normal Distribution')
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'accuracy_distributions_{timestamp}')
    
    def _plot_correlation_analysis(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                 output_path: Path = None, timestamp: str = None):
        """Create correlation analysis plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation matrix of numerical variables
        numerical_cols = ['overlap_accuracy', 'current_result_size', 'future_result_size', 
                         'recommendations_size', 'execution_time', 'gap']
        corr_matrix = df[numerical_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0, ax=ax1)
        ax1.set_title('Correlation Matrix of Numerical Variables')
        
        # Pairwise relationships
        # Select a subset for better visualization
        plot_vars = ['overlap_accuracy', 'current_result_size', 'execution_time']
        sns.scatterplot(data=df, x='current_result_size', y='overlap_accuracy', 
                       hue='recommender', size='execution_time', alpha=0.6, ax=ax2)
        ax2.set_title('Accuracy vs Result Size (bubble size = execution time)')
        ax2.set_xlabel('Current Result Set Size')
        ax2.set_ylabel('Overlap Accuracy')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'correlation_analysis_{timestamp}')
    
    def _plot_time_series_analysis(self, df: pd.DataFrame, pdf: PdfPages = None, 
                                 output_path: Path = None, timestamp: str = None):
        """Create time series analysis if timestamp data is available."""
        if 'timestamp' not in df.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Accuracy over time
        sns.lineplot(data=df_sorted, x='timestamp', y='overlap_accuracy', 
                    hue='recommender', ax=ax1)
        ax1.set_title('Accuracy Over Time')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Overlap Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Execution time over time
        sns.lineplot(data=df_sorted, x='timestamp', y='execution_time', 
                    hue='recommender', ax=ax2)
        ax2.set_title('Execution Time Over Time')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'time_series_analysis_{timestamp}')
    
    def _plot_error_analysis(self, df: pd.DataFrame, pdf: PdfPages = None, 
                           output_path: Path = None, timestamp: str = None):
        """Create error analysis plots."""
        # Include error cases for this analysis
        all_df = pd.DataFrame(self.results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error rate by recommender
        error_summary = all_df.groupby('recommender').agg({
            'error': lambda x: x.notna().sum(),
            'overlap_accuracy': 'count'
        }).rename(columns={'error': 'error_count', 'overlap_accuracy': 'total_count'})
        error_summary['error_rate'] = error_summary['error_count'] / error_summary['total_count']
        
        sns.barplot(data=error_summary.reset_index(), x='recommender', y='error_rate', ax=ax1)
        ax1.set_title('Error Rate by Recommender')
        ax1.set_xlabel('Recommender')
        ax1.set_ylabel('Error Rate')
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rate by result size
        all_df['has_error'] = all_df['error'].notna()
        all_df['result_size_bin'] = pd.cut(
            all_df['current_result_size'], 
            bins=[0, 10, 25, 50, 100, float('inf')], 
            labels=['1-10', '11-25', '26-50', '51-100', '100+']
        )
        
        success_by_size = all_df.groupby(['recommender', 'result_size_bin'])['has_error'].apply(
            lambda x: 1 - x.mean()
        ).reset_index()
        success_by_size.columns = ['recommender', 'result_size_bin', 'success_rate']
        
        pivot_success = success_by_size.pivot(index='recommender', 
                                            columns='result_size_bin', 
                                            values='success_rate')
        sns.heatmap(pivot_success, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
        ax2.set_title('Success Rate by Recommender and Result Size')
        
        plt.tight_layout()
        self._save_plot(fig, pdf, output_path, f'error_analysis_{timestamp}')
    
    def _save_plot(self, fig, pdf: PdfPages = None, output_path: Path = None, 
                  filename: str = None):
        """Save plot to PDF and/or individual file."""
        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        
        if output_path is not None and filename is not None:
            filepath = output_path / f"{filename}.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.debug(f"Plot saved to {filepath}")
        
        plt.close(fig)
    
    def create_summary_dashboard(self, output_path: str = "dashboard.html"):
        """Create an interactive HTML dashboard with key metrics."""
        if not self.results:
            logger.warning("No results to create dashboard")
            return
        
        df = pd.DataFrame(self.results)
        valid_df = df[df['overlap_accuracy'].notna() & (df['overlap_accuracy'] >= 0)]
        
        # Generate HTML content
        html_content = self._generate_dashboard_html(valid_df)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_path}")
    
    def _generate_dashboard_html(self, df: pd.DataFrame) -> str:
        """Generate HTML content for the dashboard."""
        # Calculate summary statistics
        stats = df.groupby('recommender')['overlap_accuracy'].agg(['mean', 'std', 'count'])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recommender System Experiment Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; }}
                .recommender {{ background: #e9ecef; padding: 10px; margin: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Recommender System Experiment Results</h1>
            <div class="metric">
                <h2>Summary</h2>
                <p>Total Experiments: {len(df)}</p>
                <p>Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metric">
                <h2>Recommender Performance</h2>
                <table>
                    <tr><th>Recommender</th><th>Mean Accuracy</th><th>Std Dev</th><th>Experiments</th></tr>
        """
        
        for recommender, row in stats.iterrows():
            html += f"""
                    <tr>
                        <td>{recommender}</td>
                        <td>{row['mean']:.4f}</td>
                        <td>{row['std']:.4f}</td>
                        <td>{row['count']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """Main function to run the experiments."""
    # Setup
    experiment_runner = RecommenderExperimentRunner()
    
    # Get first available session
    sessions = experiment_runner.dataloader.get_sessions()
    if not sessions:
        logger.error("No sessions found in dataset")
        return
    
    session_id = sessions[0]  # Use first session (should be 11305)
    logger.info(f"Using session {session_id}")
    
    # Run experiments
    results_df = experiment_runner.run_experiment(session_id, max_gap=5)
    
    if results_df.empty:
        logger.error("No results generated")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"recommender_experiment_results_{timestamp}.csv"
    experiment_runner.save_results(output_file)
    
    # Generate and log summary
    summary = experiment_runner.generate_summary_report()
    logger.info("Experiment Summary Report:")
    logger.info(f"\n{summary}")
    
    # Save detailed analysis
    analysis = experiment_runner.analyze_results()
    analysis_file = f"recommender_experiment_analysis_{timestamp}.pkl"
    with open(analysis_file, 'wb') as f:
        pickle.dump(analysis, f)
    logger.info(f"Detailed analysis saved to {analysis_file}")
    
    # Create comprehensive visualizations
    output_dir = f"experiment_results_{timestamp}"
    experiment_runner.create_visualizations(output_dir=output_dir, save_pdf=True)
    
    # Create interactive dashboard
    dashboard_path = f"{output_dir}/dashboard.html"
    experiment_runner.create_summary_dashboard(dashboard_path)
    
    logger.info(f"Visualizations and dashboard created in {output_dir}")
    logger.info("Experiment completed successfully!")
    
    # Create visualizations
    experiment_runner.create_visualizations(output_dir="experiment_results", save_pdf=True)
    
    # Create summary dashboard
    dashboard_file = f"recommender_experiment_dashboard_{timestamp}.html"
    experiment_runner.create_summary_dashboard(output_path=dashboard_file)


if __name__ == "__main__":
    main()
