"""
Advanced analysis and visualization for enhanced experimental results.

This module provides comprehensive analysis capabilities for the structured
experimental data collected by the enhanced experiment system.
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
matplotlib.use('Agg')  # Use non-interactive backend

from experiment_collector import ExperimentCollector

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """
    Advanced analyzer for experimental results with interactive visualizations
    and comprehensive statistical analysis.
    """
    
    @staticmethod
    def find_most_recent_experiment_dir(base_results_dir: str = "results/experiment") -> Optional[str]:
        """
        Find the most recent experiment directory by parsing timestamps in directory names.
        
        Args:
            base_results_dir: Base directory containing experiment results
            
        Returns:
            Path to the most recent experiment directory, or None if none found
        """
        base_path = Path(base_results_dir)
        if not base_path.exists():
            logger.error(f"Results directory not found: {base_results_dir}")
            return None
        
        # Get all subdirectories
        experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        
        if not experiment_dirs:
            logger.error(f"No experiment directories found in {base_results_dir}")
            return None
        
        # Parse timestamps from directory names and find the most recent
        most_recent_dir = None
        most_recent_time = None
        
        for exp_dir in experiment_dirs:
            dir_name = exp_dir.name
            
            # Look for timestamp patterns like YYYYMMDD_HHMMSS
            timestamp_patterns = [
                r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
                r'(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})',  # Alternative format
            ]
            
            import re
            timestamp_str = None
            for pattern in timestamp_patterns:
                match = re.search(pattern, dir_name)
                if match:
                    timestamp_str = match.group(1)
                    break
            
            if timestamp_str:
                try:
                    # Parse the timestamp
                    if '_' in timestamp_str:
                        date_part, time_part = timestamp_str.split('_')
                        timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                    else:
                        # Alternative parsing if needed
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if most_recent_time is None or timestamp > most_recent_time:
                        most_recent_time = timestamp
                        most_recent_dir = exp_dir
                        
                except ValueError as e:
                    logger.debug(f"Could not parse timestamp from {dir_name}: {e}")
                    continue
        
        if most_recent_dir:
            logger.info(f"Found most recent experiment directory: {most_recent_dir}")
            return str(most_recent_dir)
        else:
            # Fallback: use the most recently modified directory
            logger.warning("No timestamped directories found, using most recently modified")
            most_recent_dir = max(experiment_dirs, key=lambda d: d.stat().st_mtime)
            logger.info(f"Using most recently modified directory: {most_recent_dir}")
            return str(most_recent_dir)
    
    @classmethod
    def create_from_most_recent(cls, 
                               base_results_dir: str = "results/experiment",
                               include_tuple_analysis: bool = False) -> 'ExperimentAnalyzer':
        """
        Create an analyzer instance using the most recent experiment directory.
        
        Args:
            base_results_dir: Base directory containing experiment results
            include_tuple_analysis: Whether to load and analyze actual tuple data
            
        Returns:
            ExperimentAnalyzer instance configured with the most recent experiment
            
        Raises:
            ValueError: If no experiment directories are found
        """
        most_recent_dir = cls.find_most_recent_experiment_dir(base_results_dir)
        
        if most_recent_dir is None:
            raise ValueError(f"No experiment directories found in {base_results_dir}")
        
        logger.info(f"Using most recent experiment directory: {Path(most_recent_dir).name}")
        
        return cls(
            experiment_data_dir=most_recent_dir,
            include_tuple_analysis=include_tuple_analysis
        )

    def __init__(self, 
                 experiment_data_dir: str,
                 output_dir: str = None,
                 include_tuple_analysis: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            experiment_data_dir: Directory containing experimental data
            output_dir: Global output directory for all generated files
            include_tuple_analysis: Whether to load and analyze actual tuple data
        """
        self.data_dir = Path(experiment_data_dir)
        self.output_dir = self.data_dir / "analysis" if output_dir is None else Path(output_dir)
        self.include_tuples = include_tuple_analysis
        self.collector = ExperimentCollector(base_output_dir=str(self.data_dir))
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loaded data
        self.results_df: Optional[pd.DataFrame] = None
        self.metadata_summary: Dict[str, Any] = {}
        
    def load_all_results(self, 
                        session_filter: Optional[List[str]] = None,
                        recommender_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load all experimental results for analysis."""
        
        logger.info("Loading experimental results...")
        
        self.results_df = self.collector.load_experiment_results(
            session_ids=session_filter,
            recommender_names=recommender_filter,
            include_tuples=self.include_tuples
        )
        
        if self.results_df.empty:
            logger.warning("No experimental results found")
            return self.results_df
        
        # Process and enrich the data
        self.results_df = self._enrich_results_data(self.results_df)
        
        logger.info(f"Loaded {len(self.results_df)} experimental results")
        logger.info(f"Sessions: {self.results_df['meta_session_id'].nunique()}")
        logger.info(f"Recommenders: {self.results_df['meta_recommender_name'].nunique()}")
        
        return self.results_df
    
    def _enrich_results_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich results data with derived columns for analysis."""
        
        # Convert timestamp to datetime
        if 'meta_timestamp' in df.columns:
            df['meta_timestamp'] = pd.to_datetime(df['meta_timestamp'])
        
        # Create categorical columns for better plotting
        df['recommender_category'] = df['meta_recommender_name'].astype('category')
        df['status_category'] = df['meta_status'].astype('category')
        
        # Create performance bins
        if 'eval_overlap_accuracy' in df.columns:
            df['accuracy_bin'] = pd.cut(
                df['eval_overlap_accuracy'], 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Create precision bins
        if 'eval_precision' in df.columns:
            df['precision_bin'] = pd.cut(
                df['eval_precision'], 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Create recall bins
        if 'eval_recall' in df.columns:
            df['recall_bin'] = pd.cut(
                df['eval_recall'], 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Create F1 score bins
        if 'eval_f1_score' in df.columns:
            df['f1_bin'] = pd.cut(
                df['eval_f1_score'], 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Create ROC-AUC bins
        if 'eval_roc_auc' in df.columns:
            df['roc_auc_bin'] = pd.cut(
                df['eval_roc_auc'], 
                bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            )
        
        # Create execution time bins
        if 'meta_execution_time_seconds' in df.columns:
            df['execution_time_bin'] = pd.cut(
                df['meta_execution_time_seconds'],
                bins=[0, 1, 5, 10, 30, float('inf')],
                labels=['<1s', '1-5s', '5-10s', '10-30s', '>30s']
            )
        
        # Create result size categories
        if 'rec_predicted_count' in df.columns:
            df['prediction_size_category'] = pd.cut(
                df['rec_predicted_count'],
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge']
            )
        
        return df
    
    def generate_performance_dashboard(self, filename: str = "performance_dashboard.html"):
        """Generate comprehensive interactive performance dashboard."""
        
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return
        
        output_file = self.output_dir / filename
        
        # Create subplots - expand to include all metrics
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                'Accuracy by Recommender',
                'Precision by Recommender', 
                'Recall by Recommender',
                'F1 Score by Recommender',
                'ROC-AUC by Recommender',
                'Execution Time Distribution',
                'Accuracy vs Gap Analysis', 
                'Precision vs Gap Analysis',
                'Performance Heatmap (Accuracy)',
                'Performance Heatmap (F1)',
                'Success Rate by Result Size',
                'Timeline Analysis'
            ],
            specs=[
                [{"type": "box"}, {"type": "box"}, {"type": "box"}],
                [{"type": "box"}, {"type": "box"}, {"type": "violin"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Accuracy by Recommender (Box plot)
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            if 'eval_overlap_accuracy' in data.columns:
                fig.add_trace(
                    go.Box(y=data['eval_overlap_accuracy'], name=recommender),
                    row=1, col=1
                )
        
        # 2. Precision by Recommender (Box plot)
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            if 'eval_precision' in data.columns:
                fig.add_trace(
                    go.Box(y=data['eval_precision'], name=recommender),
                    row=1, col=2
                )
        
        # 3. Recall by Recommender (Box plot)
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            if 'eval_recall' in data.columns:
                fig.add_trace(
                    go.Box(y=data['eval_recall'], name=recommender),
                    row=1, col=3
                )
        
        # 4. F1 Score by Recommender (Box plot)
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            if 'eval_f1_score' in data.columns:
                fig.add_trace(
                    go.Box(y=data['eval_f1_score'], name=recommender),
                    row=2, col=1
                )
        
        # 5. ROC-AUC by Recommender (Box plot)
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            if 'eval_roc_auc' in data.columns:
                fig.add_trace(
                    go.Box(y=data['eval_roc_auc'], name=recommender),
                    row=2, col=2
                )
        
        # 6. Execution Time Distribution
        if 'meta_execution_time_seconds' in self.results_df.columns:
            for recommender in self.results_df['meta_recommender_name'].unique():
                data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
                fig.add_trace(
                    go.Violin(y=data['meta_execution_time_seconds'], name=recommender),
                    row=2, col=3
                )
        
        # 7. Accuracy vs Gap Analysis
        if 'eval_overlap_accuracy' in self.results_df.columns and 'meta_gap' in self.results_df.columns:
            gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_overlap_accuracy'].mean().reset_index()
            
            for recommender in gap_analysis['meta_recommender_name'].unique():
                data = gap_analysis[gap_analysis['meta_recommender_name'] == recommender]
                fig.add_trace(
                    go.Scatter(
                        x=data['meta_gap'], 
                        y=data['eval_overlap_accuracy'],
                        mode='lines+markers',
                        name=recommender
                    ),
                    row=3, col=1
                )
        
        # 8. Precision vs Gap Analysis
        if 'eval_precision' in self.results_df.columns and 'meta_gap' in self.results_df.columns:
            gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_precision'].mean().reset_index()
            
            for recommender in gap_analysis['meta_recommender_name'].unique():
                data = gap_analysis[gap_analysis['meta_recommender_name'] == recommender]
                fig.add_trace(
                    go.Scatter(
                        x=data['meta_gap'], 
                        y=data['eval_precision'],
                        mode='lines+markers',
                        name=recommender
                    ),
                    row=3, col=2
                )
        
        # 9. Performance Heatmap (Accuracy)
        if 'eval_overlap_accuracy' in self.results_df.columns and 'meta_gap' in self.results_df.columns:
            heatmap_data = self.results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_overlap_accuracy'].mean().unstack(fill_value=0)
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='RdYlGn'
                ),
                row=3, col=3
            )
        
        # 10. Performance Heatmap (F1 Score)
        if 'eval_f1_score' in self.results_df.columns and 'meta_gap' in self.results_df.columns:
            heatmap_data = self.results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_f1_score'].mean().unstack(fill_value=0)
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='RdYlGn'
                ),
                row=4, col=1
            )
        
        # 11. Success Rate by Result Size
        if 'prediction_size_category' in self.results_df.columns:
            success_data = self.results_df.groupby(['meta_recommender_name', 'prediction_size_category']).apply(
                lambda x: (x['meta_status'] == 'completed').mean()
            ).reset_index()
            success_data.columns = ['recommender', 'size_category', 'success_rate']
            
            for recommender in success_data['recommender'].unique():
                data = success_data[success_data['recommender'] == recommender]
                fig.add_trace(
                    go.Bar(x=data['size_category'], y=data['success_rate'], name=recommender),
                    row=4, col=2
                )
        
        # 12. Timeline Analysis (Multiple metrics)
        if 'meta_timestamp' in self.results_df.columns:
            # Show F1 score over time as it's a good overall metric
            if 'eval_f1_score' in self.results_df.columns:
                timeline_data = self.results_df.groupby([
                    pd.Grouper(key='meta_timestamp', freq='H'), 
                    'meta_recommender_name'
                ])['eval_f1_score'].mean().reset_index()
                
                for recommender in timeline_data['meta_recommender_name'].unique():
                    data = timeline_data[timeline_data['meta_recommender_name'] == recommender]
                    fig.add_trace(
                        go.Scatter(
                            x=data['meta_timestamp'], 
                            y=data['eval_f1_score'],
                            mode='lines',
                            name=recommender
                        ),
                        row=4, col=3
                    )
            elif 'eval_overlap_accuracy' in self.results_df.columns:
                # Fallback to accuracy if F1 not available
                timeline_data = self.results_df.groupby([
                    pd.Grouper(key='meta_timestamp', freq='H'), 
                    'meta_recommender_name'
                ])['eval_overlap_accuracy'].mean().reset_index()
                
                for recommender in timeline_data['meta_recommender_name'].unique():
                    data = timeline_data[timeline_data['meta_recommender_name'] == recommender]
                    fig.add_trace(
                        go.Scatter(
                            x=data['meta_timestamp'], 
                            y=data['eval_overlap_accuracy'],
                            mode='lines',
                            name=recommender
                        ),
                        row=4, col=3
                    )
        
        # Update layout
        fig.update_layout(
            height=1600,  # Increased height for more subplots
            title_text="Comprehensive Recommender System Performance Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(output_file)
        logger.info(f"Performance dashboard saved to {output_file}")
        
        return fig
    
    def create_statistical_summary(self) -> Dict[str, Any]:
        """Create comprehensive statistical summary of experimental results."""
        
        if self.results_df is None or self.results_df.empty:
            return {"error": "No data loaded"}
        
        summary = {
            "dataset_overview": {
                "total_experiments": len(self.results_df),
                "unique_sessions": self.results_df['meta_session_id'].nunique(),
                "recommenders": self.results_df['meta_recommender_name'].unique().tolist(),
                "gaps_tested": sorted(self.results_df['meta_gap'].unique().tolist()) if 'meta_gap' in self.results_df.columns else [],
                "time_span": {
                    "start": self.results_df['meta_timestamp'].min() if 'meta_timestamp' in self.results_df.columns else None,
                    "end": self.results_df['meta_timestamp'].max() if 'meta_timestamp' in self.results_df.columns else None
                }
            }
        }
        
        # Performance statistics - include all metrics
        metrics_to_include = {
            'eval_overlap_accuracy': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'eval_precision': ['mean', 'std', 'min', 'max', 'median'] if 'eval_precision' in self.results_df.columns else [],
            'eval_recall': ['mean', 'std', 'min', 'max', 'median'] if 'eval_recall' in self.results_df.columns else [],
            'eval_f1_score': ['mean', 'std', 'min', 'max', 'median'] if 'eval_f1_score' in self.results_df.columns else [],
            'eval_roc_auc': ['mean', 'std', 'min', 'max', 'median'] if 'eval_roc_auc' in self.results_df.columns else [],
            'meta_execution_time_seconds': ['mean', 'std'] if 'meta_execution_time_seconds' in self.results_df.columns else []
        }
        
        # Filter out empty aggregations
        metrics_to_include = {k: v for k, v in metrics_to_include.items() if v}
        
        if metrics_to_include:
            perf_stats = self.results_df.groupby('meta_recommender_name').agg(metrics_to_include).round(4)
            
            # Convert DataFrame to JSON-serializable format, handling multi-level columns
            perf_stats_dict = {}
            for col in perf_stats.columns:
                if isinstance(col, tuple):
                    # Convert tuple column names to strings
                    col_key = "_".join(str(x) for x in col)
                else:
                    col_key = str(col)
                perf_stats_dict[col_key] = perf_stats[col].to_dict()
            
            summary["performance_statistics"] = perf_stats_dict
        
        # Statistical significance tests
        summary["statistical_tests"] = self._perform_statistical_tests()
        
        # Correlation analysis
        summary["correlations"] = self._analyze_correlations()
        
        # Performance trends
        summary["trends"] = self._analyze_trends()
        
        return summary
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests between recommenders for all available metrics."""
        
        # Find available metrics to test
        test_metrics = []
        metric_names = {
            'eval_overlap_accuracy': 'Accuracy',
            'eval_precision': 'Precision', 
            'eval_recall': 'Recall',
            'eval_f1_score': 'F1 Score',
            'eval_roc_auc': 'ROC-AUC'
        }
        
        for metric_col, metric_name in metric_names.items():
            if metric_col in self.results_df.columns:
                test_metrics.append((metric_col, metric_name))
        
        if not test_metrics:
            return {"error": "No metrics available for testing"}
        
        try:
            from scipy import stats
        except ImportError:
            return {"error": "scipy not available for statistical tests"}
        
        all_tests = {}
        recommenders = self.results_df['meta_recommender_name'].unique()
        
        for metric_col, metric_name in test_metrics:
            metric_tests = {}
            
            # Pairwise t-tests for this metric
            for i, rec1 in enumerate(recommenders):
                for rec2 in recommenders[i+1:]:
                    data1 = self.results_df[self.results_df['meta_recommender_name'] == rec1][metric_col].dropna()
                    data2 = self.results_df[self.results_df['meta_recommender_name'] == rec2][metric_col].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        metric_tests[f"{rec1}_vs_{rec2}"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "effect_size": float((data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2))
                        }
            
            # ANOVA test for this metric
            groups = [
                self.results_df[self.results_df['meta_recommender_name'] == rec][metric_col].dropna()
                for rec in recommenders
            ]
            
            # Need at least 2 groups with at least 2 samples each for ANOVA
            valid_groups = [group for group in groups if len(group) > 1]
            if len(valid_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*valid_groups)
                metric_tests["anova"] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
            else:
                metric_tests["anova"] = {
                    "error": f"Insufficient data for ANOVA (need at least 2 groups with >1 sample each, got {len(valid_groups)} valid groups)"
                }
            
            all_tests[metric_name] = metric_tests
        
        return all_tests
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        
        # Select numeric columns for correlation analysis
        numeric_cols = [col for col in self.results_df.columns if 
                       self.results_df[col].dtype in ['int64', 'float64'] and 
                       col not in ['meta_timestamp']]
        
        if len(numeric_cols) < 2:
            return {"error": "Insufficient numeric columns for correlation analysis"}
        
        corr_matrix = self.results_df[numeric_cols].corr()
        
        # Find strong correlations (|r| > 0.5)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time and by query position."""
        
        trends = {}
        
        # Time-based trends
        if 'meta_timestamp' in self.results_df.columns and 'eval_overlap_accuracy' in self.results_df.columns:
            time_trends = self.results_df.groupby([
                pd.Grouper(key='meta_timestamp', freq='H'),
                'meta_recommender_name'
            ])['eval_overlap_accuracy'].mean().reset_index()
            
            trends["temporal"] = time_trends.to_dict('records')
        
        # Gap-based trends
        if 'meta_gap' in self.results_df.columns and 'eval_overlap_accuracy' in self.results_df.columns:
            gap_trends = self.results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_overlap_accuracy'].agg(['mean', 'std', 'count']).reset_index()
            trends["gap_based"] = gap_trends.to_dict('records')
        
        return trends
    
    def create_detailed_comparison_report(self, filename: str = "detailed_comparison_report.html"):
        """Create a detailed HTML comparison report."""
        
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return
        
        output_file = self.output_dir / filename
        
        # Generate statistical summary
        stats_summary = self.create_statistical_summary()
        
        # Create HTML report
        html_content = self._generate_comparison_html(stats_summary)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Detailed comparison report saved to {output_file}")
    
    def _generate_comparison_html(self, stats_summary: Dict[str, Any]) -> str:
        """Generate HTML content for the detailed comparison report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detailed Recommender Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .highlight {{ background: #d4edda; border-left: 4px solid #28a745; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .error {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .number {{ text-align: right; }}
            </style>
        </head>
        <body>
            <h1>Recommender System Comparison Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Dataset overview
        if 'dataset_overview' in stats_summary:
            overview = stats_summary['dataset_overview']
            html += f"""
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric">
                    <strong>Total Experiments:</strong> {overview.get('total_experiments', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Unique Sessions:</strong> {overview.get('unique_sessions', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Recommenders Tested:</strong> {', '.join(overview.get('recommenders', []))}
                </div>
                <div class="metric">
                    <strong>Gaps Tested:</strong> {overview.get('gaps_tested', [])}
                </div>
            </div>
            """
        
        # Performance statistics
        if 'performance_statistics' in stats_summary:
            html += """
            <div class="section">
                <h2>Performance Statistics</h2>
                <table>
                    <tr>
                        <th>Recommender</th>
                        <th>Experiments</th>
                        <th>Mean Accuracy</th>
                        <th>Mean Precision</th>
                        <th>Mean Recall</th>
                        <th>Mean F1 Score</th>
                        <th>Mean ROC-AUC</th>
                        <th>Execution Time (s)</th>
                    </tr>
            """
            
            perf_stats = stats_summary['performance_statistics']
            
            # Get list of recommenders
            recommenders = set()
            for metric_stats in perf_stats.values():
                if isinstance(metric_stats, dict):
                    recommenders.update(metric_stats.keys())
            
            for recommender in recommenders:
                accuracy = perf_stats.get('eval_overlap_accuracy_mean', {}).get(recommender, 'N/A')
                precision = perf_stats.get('eval_precision_mean', {}).get(recommender, 'N/A')
                recall = perf_stats.get('eval_recall_mean', {}).get(recommender, 'N/A')
                f1_score = perf_stats.get('eval_f1_score_mean', {}).get(recommender, 'N/A')
                roc_auc = perf_stats.get('eval_roc_auc_mean', {}).get(recommender, 'N/A')
                exec_time = perf_stats.get('meta_execution_time_seconds_mean', {}).get(recommender, 'N/A')
                count = perf_stats.get('eval_overlap_accuracy_count', {}).get(recommender, 'N/A')
                
                html += f"""
                <tr>
                    <td>{recommender}</td>
                    <td class="number">{count}</td>
                    <td class="number">{accuracy:.4f if isinstance(accuracy, (int, float)) else accuracy}</td>
                    <td class="number">{precision:.4f if isinstance(precision, (int, float)) else precision}</td>
                    <td class="number">{recall:.4f if isinstance(recall, (int, float)) else recall}</td>
                    <td class="number">{f1_score:.4f if isinstance(f1_score, (int, float)) else f1_score}</td>
                    <td class="number">{roc_auc:.4f if isinstance(roc_auc, (int, float)) else roc_auc}</td>
                    <td class="number">{exec_time:.4f if isinstance(exec_time, (int, float)) else exec_time}</td>
                </tr>
                """
            
            html += "</table></div>"
        
        # Statistical tests
        if 'statistical_tests' in stats_summary:
            tests = stats_summary['statistical_tests']
            html += """
            <div class="section">
                <h2>Statistical Significance Tests</h2>
            """
            
            if 'anova' in tests:
                anova_class = "highlight" if tests['anova']['significant'] else "metric"
                html += f"""
                <div class="{anova_class}">
                    <strong>ANOVA Test:</strong> F-statistic = {tests['anova']['f_statistic']:.4f}, 
                    p-value = {tests['anova']['p_value']:.4f}
                    {'(Significant difference between recommenders)' if tests['anova']['significant'] else '(No significant difference)'}
                </div>
                """
            
            # Pairwise comparisons
            pairwise_tests = {k: v for k, v in tests.items() if k != 'anova'}
            if pairwise_tests:
                html += "<h3>Pairwise Comparisons</h3>"
                for comparison, result in pairwise_tests.items():
                    significance_class = "highlight" if result['significant'] else "metric"
                    html += f"""
                    <div class="{significance_class}">
                        <strong>{comparison}:</strong> 
                        t = {result['t_statistic']:.4f}, p = {result['p_value']:.4f}, 
                        effect size = {result['effect_size']:.4f}
                        {'(Significant)' if result['significant'] else '(Not significant)'}
                    </div>
                    """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def export_for_further_analysis(self, subdir: str = "exports"):
        """Export data in various formats for further analysis."""
        
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return
        
        export_dir = self.output_dir / subdir
        export_dir.mkdir(exist_ok=True)
        
        # Export main results
        self.results_df.to_csv(export_dir / "experimental_results.csv", index=False)
        self.results_df.to_parquet(export_dir / "experimental_results.parquet", index=False)
        
        # Export summary statistics
        stats_summary = self.create_statistical_summary()
        
        # Convert any remaining DataFrames to JSON-serializable format
        def convert_for_json(obj):
            if isinstance(obj, pd.DataFrame):
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
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        stats_summary_serializable = convert_for_json(stats_summary)
        
        with open(export_dir / "statistical_summary.json", 'w') as f:
            json.dump(stats_summary_serializable, f, indent=2, default=str)
        
        # Export for Python/Jupyter analysis
        self._create_analysis_notebook(export_dir)
        
        logger.info(f"Analysis exports saved to {export_dir}")
    
    def _create_analysis_notebook(self, output_dir: Path):
        """Create a Jupyter notebook template for further analysis."""
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Recommender System Analysis\n", "\n", "This notebook provides templates for analyzing the experimental results."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import numpy as np\n", 
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from scipy import stats\n",
                        "import plotly.express as px\n",
                        "\n",
                        "# Load the data\n",
                        "results = pd.read_parquet('experimental_results.parquet')\n",
                        "print(f'Loaded {len(results)} experimental results')\n",
                        "results.head()"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Basic Analysis"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Summary statistics\n",
                        "if 'eval_overlap_accuracy' in results.columns:\n",
                        "    print('Performance Summary:')\n",
                        "    print(results.groupby('meta_recommender_name')['eval_overlap_accuracy'].describe())"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(output_dir / "analysis_template.ipynb", 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    def create_publication_visualizations(self, 
                                         subdir: str = "visualizations",
                                         save_pdf: bool = True,
                                         save_individual: bool = True,
                                         dpi: int = 300,
                                         figsize: Tuple[float, float] = (12, 8)) -> Dict[str, str]:
        """
        Create comprehensive publication-ready visualizations.
        
        Args:
            subdir: Subdirectory within the global output directory to save figures
            save_pdf: Whether to save all figures in a single PDF
            save_individual: Whether to save individual PNG files
            dpi: Resolution for saved figures
            figsize: Default figure size (width, height) in inches
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return {}
        
        output_path = self.output_dir / subdir
        output_path.mkdir(exist_ok=True)
        
        # Set publication style
        self._set_publication_style()
        
        saved_files = {}
        
        # Create individual visualizations
        viz_methods = [
            ("accuracy_comparison", self._create_accuracy_comparison),
            ("gap_analysis", self._create_gap_analysis),
            ("precision_gap_analysis", self._create_precision_gap_analysis),
            ("recall_gap_analysis", self._create_recall_gap_analysis),
            ("f1_gap_analysis", self._create_f1_gap_analysis),
            ("roc_auc_gap_analysis", self._create_roc_auc_gap_analysis),
            ("result_size_analysis", self._create_result_size_analysis),
            ("execution_time_analysis", self._create_execution_time_analysis),
            ("performance_heatmap", self._create_performance_heatmap),
            ("distribution_analysis", self._create_distribution_analysis),
            ("correlation_analysis", self._create_correlation_analysis),
            ("tuple_count_analysis", self._create_tuple_count_analysis),
            ("normalized_performance_comparison", self._create_normalized_performance_comparison),
            ("normalized_gap_analysis", self._create_normalized_gap_analysis)
        ]
        
        figures = []
        
        for viz_name, viz_method in viz_methods:
            try:
                fig = viz_method(figsize=figsize)
                if fig is not None:
                    figures.append((viz_name, fig))
                    
                    if save_individual:
                        file_path = output_path / f"{viz_name}.png"
                        fig.savefig(file_path, dpi=dpi, bbox_inches='tight', 
                                  facecolor='white', edgecolor='none')
                        saved_files[viz_name] = str(file_path)
                        logger.info(f"Saved {viz_name} to {file_path}")
                    
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to create {viz_name}: {e}")
        
        # Save combined PDF
        if save_pdf and figures:
            from matplotlib.backends.backend_pdf import PdfPages
            
            pdf_path = output_path / "publication_figures.pdf"
            with PdfPages(pdf_path) as pdf:
                for viz_name, fig in figures:
                    # Recreate figure for PDF (since we closed them above)
                    viz_method = dict(viz_methods)[viz_name]
                    fig = viz_method(figsize=figsize)
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
            
            saved_files["combined_pdf"] = str(pdf_path)
            logger.info(f"Saved combined PDF to {pdf_path}")
        
        # Create summary dashboard
        dashboard_path = self._create_publication_dashboard(output_path)
        if dashboard_path:
            saved_files["dashboard"] = dashboard_path
        
        return saved_files
    
    def _set_publication_style(self):
        """Set matplotlib and seaborn styles for publication-quality figures."""
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Set publication-quality parameters
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2,
            'patch.linewidth': 0.5,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': 0.3
        })
    
    def _create_publication_dashboard(self, output_dir: Path) -> str:
        """Create a simple HTML dashboard for publication visualizations."""
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Publication Visualization Dashboard</title>
            <style>
                body {{ 
                    font-family: 'Times New Roman', Times, serif; 
                    margin: 20px; 
                    line-height: 1.6; 
                }}
                .header {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                }}
                .section {{ 
                    margin: 20px 0; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px;
                }}
                .viz-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                }}
                .viz-item {{ 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 15px;
                }}
                .viz-item h3 {{ 
                    margin-top: 0; 
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Publication-Ready Visualization Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Available Visualizations</h2>
                <div class="viz-grid">
                    <div class="viz-item">
                        <h3>Accuracy Comparison</h3>
                        <p>Comprehensive comparison of all available performance metrics</p>
                        <a href="accuracy_comparison.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Gap Analysis</h3>
                        <p>Analysis of how query gap affects performance across metrics</p>
                        <a href="gap_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Precision Gap Analysis</h3>
                        <p>Detailed precision analysis vs query gap</p>
                        <a href="precision_gap_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Recall Gap Analysis</h3>
                        <p>Detailed recall analysis vs query gap</p>
                        <a href="recall_gap_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>F1 Score Gap Analysis</h3>
                        <p>Detailed F1 score analysis vs query gap</p>
                        <a href="f1_gap_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>ROC-AUC Gap Analysis</h3>
                        <p>Detailed ROC-AUC analysis vs query gap</p>
                        <a href="roc_auc_gap_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Performance Heatmap</h3>
                        <p>Multi-dimensional performance visualization</p>
                        <a href="performance_heatmap.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Execution Time Analysis</h3>
                        <p>Performance timing analysis and efficiency metrics</p>
                        <a href="execution_time_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Tuple Count Analysis</h3>
                        <p>Analysis of the number of tuples recommended by each system</p>
                        <a href="tuple_count_analysis.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Normalized Performance Comparison</h3>
                        <p>Performance metrics normalized by number of tuples recommended</p>
                        <a href="normalized_performance_comparison.png">View PNG</a>
                    </div>
                    <div class="viz-item">
                        <h3>Normalized Gap Analysis</h3>
                        <p>Gap analysis with metrics normalized by tuple count</p>
                        <a href="normalized_gap_analysis.png">View PNG</a>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Download Options</h2>
                <p><a href="publication_figures.pdf">Download All Figures (PDF)</a></p>
                <p>All figures are generated at 300 DPI and are publication-ready.</p>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = output_dir / "visualization_dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Visualization dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def _create_accuracy_comparison(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready comprehensive metrics comparison visualization."""
        
        # Check which metrics are available
        available_metrics = []
        metric_names = {
            'eval_overlap_accuracy': 'Accuracy',
            'eval_precision': 'Precision', 
            'eval_recall': 'Recall',
            'eval_f1_score': 'F1 Score',
            'eval_roc_auc': 'ROC-AUC'
        }
        
        for metric_col, metric_name in metric_names.items():
            if metric_col in self.results_df.columns:
                available_metrics.append((metric_col, metric_name))
        
        if not available_metrics:
            return None
        
        # Determine grid size based on available metrics
        n_metrics = len(available_metrics)
        if n_metrics <= 2:
            rows, cols = 1, n_metrics
        elif n_metrics <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        fig.suptitle('Comprehensive Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        # Create box plots for each available metric
        for idx, (metric_col, metric_name) in enumerate(available_metrics):
            ax = axes[idx]
            
            # Box plot comparison
            sns.boxplot(data=self.results_df, x='meta_recommender_name', 
                       y=metric_col, ax=ax)
            ax.set_title(f'{chr(65 + idx)}) {metric_name} Distribution by Recommender')
            ax.set_xlabel('Recommender System')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add mean values as text annotations
            means = self.results_df.groupby('meta_recommender_name')[metric_col].mean()
            for i, (recommender, mean_val) in enumerate(means.items()):
                ax.text(i, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}', 
                       ha='center', va='top', fontweight='bold', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _create_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready gap analysis visualization with multiple metrics."""
        if 'meta_gap' not in self.results_df.columns:
            return None
        
        # Find available metrics
        available_metrics = []
        metric_names = {
            'eval_overlap_accuracy': 'Accuracy',
            'eval_precision': 'Precision', 
            'eval_recall': 'Recall',
            'eval_f1_score': 'F1 Score',
            'eval_roc_auc': 'ROC-AUC'
        }
        
        for metric_col, metric_name in metric_names.items():
            if metric_col in self.results_df.columns:
                available_metrics.append((metric_col, metric_name))
        
        if not available_metrics:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Query Gap Impact Analysis - Multiple Metrics', fontsize=16, fontweight='bold')
        
        # 1. Line plot - First available metric vs gap
        if available_metrics:
            metric_col, metric_name = available_metrics[0]
            gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
                metric_col: ['mean', 'std', 'count']
            }).reset_index()
            gap_analysis.columns = ['recommender', 'gap', 'mean_metric', 'std_metric', 'count']
            
            for recommender in gap_analysis['recommender'].unique():
                data = gap_analysis[gap_analysis['recommender'] == recommender]
                ax1.errorbar(data['gap'], data['mean_metric'], yerr=data['std_metric'], 
                            label=recommender, marker='o', linewidth=2, markersize=6)
            
            ax1.set_title(f'A) {metric_name} vs Query Gap')
            ax1.set_xlabel('Query Gap')
            ax1.set_ylabel('Mean {metric_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap of primary metric by recommender and gap
        if available_metrics:
            metric_col, metric_name = available_metrics[0]
            heatmap_data = self.results_df.pivot_table(
                values=metric_col, 
                index='meta_recommender_name', 
                columns='meta_gap', 
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=ax2, cbar_kws={'label': f'Mean {metric_name}'})
            ax2.set_title(f'B) {metric_name} Heatmap (Recommender × Gap)')
            ax2.set_xlabel('Query Gap')
            ax2.set_ylabel('Recommender System')
        
        # 3. Multi-metric comparison for first gap value
        if len(available_metrics) > 1:
            first_gap = sorted(self.results_df['meta_gap'].unique())[0]
            gap_data = self.results_df[self.results_df['meta_gap'] == first_gap]
            
            # Create a DataFrame for plotting multiple metrics
            plot_data = []
            for metric_col, metric_name in available_metrics:
                for recommender in gap_data['meta_recommender_name'].unique():
                    rec_data = gap_data[gap_data['meta_recommender_name'] == recommender]
                    if len(rec_data) > 0:
                        plot_data.append({
                            'Recommender': recommender,
                            'Metric': metric_name,
                            'Value': rec_data[metric_col].mean()
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                pivot_df = plot_df.pivot(index='Recommender', columns='Metric', values='Value')
                
                pivot_df.plot(kind='bar', ax=ax3, width=0.8)
                ax3.set_title(f'C) Multi-Metric Comparison (Gap = {first_gap})')
                ax3.set_xlabel('Recommender System')
                ax3.set_ylabel('Metric Value')
                ax3.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax3.tick_params(axis='x', rotation=45)
        
        # 4. Gap effect size analysis for F1 score (or first available metric)
        effect_metric = 'eval_f1_score' if 'eval_f1_score' in self.results_df.columns else available_metrics[0][0]
        effect_metric_name = metric_names.get(effect_metric, effect_metric)
        
        gap_effects = []
        for recommender in self.results_df['meta_recommender_name'].unique():
            rec_data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            baseline_gap = rec_data['meta_gap'].min()
            baseline_data = rec_data[rec_data['meta_gap'] == baseline_gap][effect_metric]
            
            for gap in sorted(rec_data['meta_gap'].unique()):
                gap_data = rec_data[rec_data['meta_gap'] == gap][effect_metric]
                
                if len(gap_data) > 0 and len(baseline_data) > 0 and baseline_data.std() > 0:
                    effect_size = (gap_data.mean() - baseline_data.mean()) / baseline_data.std()
                    gap_effects.append({
                        'recommender': recommender,
                        'gap': gap,
                        'effect_size': effect_size
                    })
        
        if gap_effects:
            effect_df = pd.DataFrame(gap_effects)
            effect_pivot = effect_df.pivot(index='recommender', columns='gap', values='effect_size')
            
            sns.heatmap(effect_pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       ax=ax4, cbar_kws={'label': 'Effect Size'})
            ax4.set_title(f'D) Gap Effect Size for {effect_metric_name} (vs Baseline)')
            ax4.set_xlabel('Query Gap')
            ax4.set_ylabel('Recommender System')
        else:
            ax4.text(0.5, 0.5, f'Insufficient data for\n{effect_metric_name} effect size analysis', 
                    ha='center', va='center', transform=ax4.transAxes, 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title(f'D) Gap Effect Size for {effect_metric_name}')
        
        plt.tight_layout()
        return fig
    
    def _create_precision_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready precision gap analysis visualization."""
        return self._create_metric_gap_analysis('eval_precision', 'Precision', figsize)
    
    def _create_recall_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready recall gap analysis visualization."""
        return self._create_metric_gap_analysis('eval_recall', 'Recall', figsize)
    
    def _create_f1_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready F1 score gap analysis visualization."""
        return self._create_metric_gap_analysis('eval_f1_score', 'F1 Score', figsize)
    
    def _create_roc_auc_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready ROC-AUC gap analysis visualization."""
        return self._create_metric_gap_analysis('eval_roc_auc', 'ROC-AUC', figsize)
    
    def _create_metric_gap_analysis(self, metric_col: str, metric_name: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready gap analysis visualization for a specific metric."""
        if 'meta_gap' not in self.results_df.columns or metric_col not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{metric_name} vs Query Gap Analysis', fontsize=16, fontweight='bold')
        
        # 1. Line plot of metric vs gap
        gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
            metric_col: ['mean', 'std', 'count']
        }).reset_index()
        gap_analysis.columns = ['recommender', 'gap', 'mean_metric', 'std_metric', 'count']
        
        for recommender in gap_analysis['recommender'].unique():
            data = gap_analysis[gap_analysis['recommender'] == recommender]
            ax1.errorbar(data['gap'], data['mean_metric'], yerr=data['std_metric'], 
                        label=recommender, marker='o', linewidth=2, markersize=6)
        
        ax1.set_title(f'A) {metric_name} vs Query Gap')
        ax1.set_xlabel('Query Gap')
        ax1.set_ylabel(f'Mean {metric_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap of metric by recommender and gap
        heatmap_data = self.results_df.pivot_table(
            values=metric_col, 
            index='meta_recommender_name', 
            columns='meta_gap', 
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax2, cbar_kws={'label': f'Mean {metric_name}'})
        ax2.set_title(f'B) {metric_name} Heatmap (Recommender × Gap)')
        ax2.set_xlabel('Query Gap')
        ax2.set_ylabel('Recommender System')
        
        # 3. Box plots by gap
        sns.boxplot(data=self.results_df, x='meta_gap', y=metric_col, 
                   hue='meta_recommender_name', ax=ax3)
        ax3.set_title(f'C) {metric_name} Distribution by Gap')
        ax3.set_xlabel('Query Gap')
        ax3.set_ylabel(metric_name)
        ax3.legend(title='Recommender', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Gap effect size analysis
        gap_effects = []
        for recommender in self.results_df['meta_recommender_name'].unique():
            rec_data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            baseline_gap = rec_data['meta_gap'].min()
            baseline_data = rec_data[rec_data['meta_gap'] == baseline_gap][metric_col]
            
            for gap in sorted(rec_data['meta_gap'].unique()):
                gap_data = rec_data[rec_data['meta_gap'] == gap][metric_col]
                
                if len(gap_data) > 0 and len(baseline_data) > 0 and baseline_data.std() > 0:
                    effect_size = (gap_data.mean() - baseline_data.mean()) / baseline_data.std()
                    gap_effects.append({
                        'recommender': recommender,
                        'gap': gap,
                        'effect_size': effect_size
                    })
        
        if gap_effects:
            effect_df = pd.DataFrame(gap_effects)
            effect_pivot = effect_df.pivot(index='recommender', columns='gap', values='effect_size')
            
            sns.heatmap(effect_pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       ax=ax4, cbar_kws={'label': 'Effect Size'})
            ax4.set_title(f'D) Gap Effect Size for {metric_name} (vs Baseline)')
            ax4.set_xlabel('Query Gap')
            ax4.set_ylabel('Recommender System')
        else:
            ax4.text(0.5, 0.5, f'Insufficient data for\n{metric_name} effect size analysis', 
                    ha='center', va='center', transform=ax4.transAxes, 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title(f'D) Gap Effect Size for {metric_name}')
        
        plt.tight_layout()
        return fig
    
    def _create_result_size_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready result size analysis visualization."""
        if 'rec_predicted_count' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Result Size Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution of prediction counts by recommender
        sns.boxplot(data=self.results_df, x='meta_recommender_name', y='rec_predicted_count', ax=ax1)
        ax1.set_title('A) Prediction Count Distribution by Recommender')
        ax1.set_xlabel('Recommender System')
        ax1.set_ylabel('Number of Predicted Tuples')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Result size vs accuracy
        if 'eval_overlap_accuracy' in self.results_df.columns:
            sns.scatterplot(data=self.results_df, x='rec_predicted_count', y='eval_overlap_accuracy', 
                           hue='meta_recommender_name', ax=ax2, alpha=0.7)
            ax2.set_title('B) Accuracy vs Prediction Count')
            ax2.set_xlabel('Number of Predicted Tuples')
            ax2.set_ylabel('Overlap Accuracy')
            ax2.legend(title='Recommender', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('B) Accuracy vs Prediction Count')
        
        # 3. Average result size by gap
        if 'meta_gap' in self.results_df.columns:
            gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
                'rec_predicted_count': 'mean'
            }).reset_index()
            
            for recommender in gap_analysis['meta_recommender_name'].unique():
                data = gap_analysis[gap_analysis['meta_recommender_name'] == recommender]
                ax3.plot(data['meta_gap'], data['rec_predicted_count'], marker='o', 
                        label=recommender, linewidth=2, markersize=6)
            
            ax3.set_title('C) Average Prediction Count vs Query Gap')
            ax3.set_xlabel('Query Gap')
            ax3.set_ylabel('Average Predicted Tuples')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Gap data not available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('C) Average Prediction Count vs Query Gap')
        
        # 4. Result size efficiency (accuracy per predicted tuple)
        if 'eval_overlap_accuracy' in self.results_df.columns:
            # Calculate efficiency metric
            efficiency = self.results_df['eval_overlap_accuracy'] / (self.results_df['rec_predicted_count'] + 1e-6)
            efficiency_df = pd.DataFrame({
                'recommender': self.results_df['meta_recommender_name'],
                'efficiency': efficiency
            })
            
            sns.boxplot(data=efficiency_df, x='recommender', y='efficiency', ax=ax4)
            ax4.set_title('D) Prediction Efficiency (Accuracy/Count)')
            ax4.set_xlabel('Recommender System')
            ax4.set_ylabel('Efficiency Score')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Efficiency calculation not available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('D) Prediction Efficiency')
        
        plt.tight_layout()
        return fig
    
    def _create_execution_time_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready execution time analysis visualization."""
        time_columns = [col for col in self.results_df.columns if 'time' in col.lower() or 'duration' in col.lower()]
        
        if not time_columns:
            return None
        
        # Use the first available time column and ensure it's numeric
        time_col = time_columns[0]
        
        # Convert time column to numeric if it's not already
        if self.results_df[time_col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(self.results_df[time_col]):
            try:
                # If it's datetime, convert to seconds since epoch
                if pd.api.types.is_datetime64_any_dtype(self.results_df[time_col]):
                    # Skip this analysis if time column is datetime (not duration)
                    return None
                else:
                    # Try to convert string to float
                    self.results_df[time_col] = pd.to_numeric(self.results_df[time_col], errors='coerce')
            except:
                return None
        
        # Check if we have valid numeric data
        if self.results_df[time_col].isna().all():
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Execution Time Analysis', fontsize=16, fontweight='bold')
        
        # 1. Execution time distribution by recommender
        sns.boxplot(data=self.results_df, x='meta_recommender_name', y=time_col, ax=ax1)
        ax1.set_title('A) Execution Time Distribution by Recommender')
        ax1.set_xlabel('Recommender System')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Time vs data size (if available)
        if 'rec_predicted_count' in self.results_df.columns:
            sns.scatterplot(data=self.results_df, x='rec_predicted_count', y=time_col, 
                           hue='meta_recommender_name', ax=ax2, alpha=0.7)
            ax2.set_title('B) Execution Time vs Result Size')
            ax2.set_xlabel('Number of Predicted Tuples')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.legend(title='Recommender', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'Result size data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('B) Execution Time vs Result Size')
        
        # 3. Time vs gap (if available)
        if 'meta_gap' in self.results_df.columns:
            gap_time = self.results_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
                time_col: 'mean'
            }).reset_index()
            
            for recommender in gap_time['meta_recommender_name'].unique():
                data = gap_time[gap_time['meta_recommender_name'] == recommender]
                ax3.plot(data['meta_gap'], data[time_col], marker='o', 
                        label=recommender, linewidth=2, markersize=6)
            
            ax3.set_title('C) Average Execution Time vs Query Gap')
            ax3.set_xlabel('Query Gap')
            ax3.set_ylabel('Average Execution Time (seconds)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Gap data not available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('C) Average Execution Time vs Query Gap')
        
        # 4. Performance efficiency (accuracy per time)
        if 'eval_overlap_accuracy' in self.results_df.columns:
            # Add small epsilon to avoid division by zero and handle potential NaN values
            time_values = self.results_df[time_col].fillna(0) + 1e-6
            efficiency = self.results_df['eval_overlap_accuracy'] / time_values
            efficiency_df = pd.DataFrame({
                'recommender': self.results_df['meta_recommender_name'],
                'time_efficiency': efficiency
            })
            
            sns.boxplot(data=efficiency_df, x='recommender', y='time_efficiency', ax=ax4)
            ax4.set_title('D) Time Efficiency (Accuracy/Time)')
            ax4.set_xlabel('Recommender System')
            ax4.set_ylabel('Time Efficiency Score')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Time efficiency calculation not available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('D) Time Efficiency')
        
        plt.tight_layout()
        return fig
    
    def _create_performance_heatmap(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready performance heatmap visualization."""
        # Define metrics to include in heatmap
        metric_columns = []
        metric_names = []
        
        if 'eval_overlap_accuracy' in self.results_df.columns:
            metric_columns.append('eval_overlap_accuracy')
            metric_names.append('Accuracy')
        if 'eval_precision' in self.results_df.columns:
            metric_columns.append('eval_precision')
            metric_names.append('Precision')
        if 'eval_recall' in self.results_df.columns:
            metric_columns.append('eval_recall')
            metric_names.append('Recall')
        if 'eval_f1_score' in self.results_df.columns:
            metric_columns.append('eval_f1_score')
            metric_names.append('F1 Score')
        
        if not metric_columns:
            return None
        
        fig, axes = plt.subplots(1, len(metric_columns), figsize=figsize)
        if len(metric_columns) == 1:
            axes = [axes]
        
        fig.suptitle('Performance Heatmaps by Recommender and Gap', fontsize=16, fontweight='bold')
        
        for i, (metric_col, metric_name) in enumerate(zip(metric_columns, metric_names)):
            if 'meta_gap' in self.results_df.columns:
                heatmap_data = self.results_df.pivot_table(
                    values=metric_col, 
                    index='meta_recommender_name', 
                    columns='meta_gap', 
                    aggfunc='mean'
                )
                
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                           ax=axes[i], cbar_kws={'label': f'Mean {metric_name}'})
                axes[i].set_title(f'{metric_name}')
                axes[i].set_xlabel('Query Gap')
                if i == 0:
                    axes[i].set_ylabel('Recommender System')
                else:
                    axes[i].set_ylabel('')
            else:
                # Just show by recommender if no gap data
                mean_values = self.results_df.groupby('meta_recommender_name')[metric_col].mean()
                sns.heatmap(mean_values.values.reshape(-1, 1), 
                           annot=True, fmt='.3f', cmap='RdYlGn',
                           yticklabels=mean_values.index, xticklabels=[metric_name],
                           ax=axes[i], cbar_kws={'label': f'Mean {metric_name}'})
                axes[i].set_title(f'{metric_name}')
                if i == 0:
                    axes[i].set_ylabel('Recommender System')
                else:
                    axes[i].set_ylabel('')
        
        plt.tight_layout()
        return fig
    
    def _create_distribution_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready distribution analysis visualization."""
        # Find available metric columns
        metric_columns = []
        if 'eval_overlap_accuracy' in self.results_df.columns:
            metric_columns.append(('eval_overlap_accuracy', 'Accuracy'))
        if 'eval_precision' in self.results_df.columns:
            metric_columns.append(('eval_precision', 'Precision'))
        if 'eval_recall' in self.results_df.columns:
            metric_columns.append(('eval_recall', 'Recall'))
        if 'eval_f1_score' in self.results_df.columns:
            metric_columns.append(('eval_f1_score', 'F1 Score'))
        
        if not metric_columns:
            return None
        
        # Use the first available metric for detailed distribution analysis
        primary_metric, primary_name = metric_columns[0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{primary_name} Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histograms by recommender
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender][primary_metric]
            ax1.hist(data, alpha=0.6, label=recommender, bins=20)
        
        ax1.set_title(f'A) {primary_name} Distribution by Recommender')
        ax1.set_xlabel(primary_name)
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Violin plots
        sns.violinplot(data=self.results_df, x='meta_recommender_name', y=primary_metric, ax=ax2)
        ax2.set_title(f'B) {primary_name} Distribution Shape')
        ax2.set_xlabel('Recommender System')
        ax2.set_ylabel(primary_name)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Q-Q plots for normality assessment
        from scipy import stats
        recommenders = self.results_df['meta_recommender_name'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender][primary_metric]
            data_clean = data.dropna()
            if len(data_clean) > 1:
                stats.probplot(data_clean, dist="norm", plot=ax3)
                # Only show the last one to avoid overplotting
        
        ax3.set_title(f'C) Q-Q Plot for Normality Assessment')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender][primary_metric]
            data_sorted = np.sort(data.dropna())
            y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            ax4.plot(data_sorted, y, label=recommender, linewidth=2)
        
        ax4.set_title(f'D) Cumulative Distribution Function')
        ax4.set_xlabel(primary_name)
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_correlation_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready correlation analysis visualization."""
        # Select numeric columns for correlation analysis
        numeric_cols = []
        col_names = []
        
        metric_mapping = {
            'eval_overlap_accuracy': 'Accuracy',
            'eval_precision': 'Precision', 
            'eval_recall': 'Recall',
            'eval_f1_score': 'F1 Score',
            'eval_roc_auc': 'ROC-AUC',
            'rec_predicted_count': 'Predicted Count',
            'meta_gap': 'Query Gap'
        }
        
        for col, name in metric_mapping.items():
            if col in self.results_df.columns:
                numeric_cols.append(col)
                col_names.append(name)
        
        if len(numeric_cols) < 2:
            return None
        
        # Calculate correlation matrix
        corr_data = self.results_df[numeric_cols].corr()
        corr_data.columns = col_names
        corr_data.index = col_names
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('A) Correlation Matrix')
        
        # 2. Scatter plot of top correlated pair
        if len(numeric_cols) >= 2:
            # Find highest correlation (excluding diagonal)
            corr_abs = np.abs(corr_data.values)
            np.fill_diagonal(corr_abs, 0)
            max_corr_idx = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
            
            col1, col2 = numeric_cols[max_corr_idx[0]], numeric_cols[max_corr_idx[1]]
            name1, name2 = col_names[max_corr_idx[0]], col_names[max_corr_idx[1]]
            
            sns.scatterplot(data=self.results_df, x=col1, y=col2, 
                           hue='meta_recommender_name', ax=ax2, alpha=0.7)
            ax2.set_title(f'B) {name1} vs {name2}')
            ax2.set_xlabel(name1)
            ax2.set_ylabel(name2)
            ax2.legend(title='Recommender', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Correlation strength distribution
        corr_values = corr_data.values[np.triu_indices_from(corr_data.values, k=1)]
        ax3.hist(corr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('C) Correlation Strength Distribution')
        ax3.set_xlabel('Correlation Coefficient')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (correlation with primary metric)
        if 'eval_overlap_accuracy' in numeric_cols:
            primary_corr = corr_data['Accuracy'].drop('Accuracy').abs().sort_values(ascending=True)
            ax4.barh(range(len(primary_corr)), primary_corr.values)
            ax4.set_yticks(range(len(primary_corr)))
            ax4.set_yticklabels(primary_corr.index)
            ax4.set_title('D) Feature Correlation with Accuracy')
            ax4.set_xlabel('Absolute Correlation')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Accuracy data not available\nfor feature importance', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('D) Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def _create_tuple_count_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create detailed analysis of the number of tuples recommended by each recommender."""
        if 'rec_predicted_count' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Recommended Tuple Count Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution of tuple counts by recommender
        sns.boxplot(data=self.results_df, x='meta_recommender_name', y='rec_predicted_count', ax=ax1)
        ax1.set_title('A) Tuple Count Distribution by Recommender')
        ax1.set_xlabel('Recommender System')
        ax1.set_ylabel('Number of Recommended Tuples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add mean values as annotations
        means = self.results_df.groupby('meta_recommender_name')['rec_predicted_count'].mean()
        for i, (recommender, mean_val) in enumerate(means.items()):
            ax1.text(i, ax1.get_ylim()[1] * 0.95, f'μ={mean_val:.1f}', 
                    ha='center', va='top', fontweight='bold', fontsize=9)
        
        # 2. Tuple count vs gap (if available)
        if 'meta_gap' in self.results_df.columns:
            gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
                'rec_predicted_count': ['mean', 'std', 'count']
            }).reset_index()
            gap_analysis.columns = ['recommender', 'gap', 'mean_count', 'std_count', 'n_samples']
            
            for recommender in gap_analysis['recommender'].unique():
                data = gap_analysis[gap_analysis['recommender'] == recommender]
                ax2.errorbar(data['gap'], data['mean_count'], yerr=data['std_count'], 
                            label=recommender, marker='o', linewidth=2, markersize=6)
            
            ax2.set_title('B) Mean Tuple Count vs Query Gap')
            ax2.set_xlabel('Query Gap')
            ax2.set_ylabel('Mean Number of Tuples')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Gap data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('B) Mean Tuple Count vs Query Gap')
        
        # 3. Histogram of tuple counts
        recommenders = self.results_df['meta_recommender_name'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['rec_predicted_count']
            ax3.hist(data, alpha=0.6, label=recommender, bins=20, color=colors[i])
        
        ax3.set_title('C) Tuple Count Distribution')
        ax3.set_xlabel('Number of Recommended Tuples')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics table as text
        stats_text = "D) Summary Statistics:\n\n"
        for recommender in recommenders:
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['rec_predicted_count']
            stats_text += f"{recommender}:\n"
            stats_text += f"  Mean: {data.mean():.1f}\n"
            stats_text += f"  Median: {data.median():.1f}\n"
            stats_text += f"  Std: {data.std():.1f}\n"
            stats_text += f"  Min: {data.min():.0f}\n"
            stats_text += f"  Max: {data.max():.0f}\n\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('D) Summary Statistics')
        
        plt.tight_layout()
        return fig
    
    def _create_normalized_performance_comparison(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create performance comparison with metrics normalized by number of tuples recommended."""
        if 'rec_predicted_count' not in self.results_df.columns:
            return None
        
        # Check which metrics are available
        available_metrics = []
        metric_names = {
            'eval_overlap_accuracy': 'Accuracy',
            'eval_precision': 'Precision', 
            'eval_recall': 'Recall',
            'eval_f1_score': 'F1 Score',
            'eval_roc_auc': 'ROC-AUC'
        }
        
        for metric_col, metric_name in metric_names.items():
            if metric_col in self.results_df.columns:
                available_metrics.append((metric_col, metric_name))
        
        if not available_metrics:
            return None
        
        # Calculate normalized metrics (metric per tuple recommended)
        normalized_df = self.results_df.copy()
        for metric_col, metric_name in available_metrics:
            # Add small epsilon to avoid division by zero
            normalized_df[f'norm_{metric_col}'] = normalized_df[metric_col] / (normalized_df['rec_predicted_count'] + 1e-6)
        
        # Determine grid size based on available metrics
        n_metrics = len(available_metrics)
        if n_metrics <= 2:
            rows, cols = 1, n_metrics
        elif n_metrics <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        fig.suptitle('Performance Metrics Normalized by Tuple Count', fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        # Create box plots for each available normalized metric
        for idx, (metric_col, metric_name) in enumerate(available_metrics):
            ax = axes[idx]
            norm_col = f'norm_{metric_col}'
            
            # Box plot comparison
            sns.boxplot(data=normalized_df, x='meta_recommender_name', 
                       y=norm_col, ax=ax)
            ax.set_title(f'{chr(65 + idx)}) {metric_name} per Tuple by Recommender')
            ax.set_xlabel('Recommender System')
            ax.set_ylabel(f'{metric_name} per Tuple')
            ax.tick_params(axis='x', rotation=45)
            
            # Add mean values as text annotations
            means = normalized_df.groupby('meta_recommender_name')[norm_col].mean()
            for i, (recommender, mean_val) in enumerate(means.items()):
                ax.text(i, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.4f}', 
                       ha='center', va='top', fontweight='bold', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _create_normalized_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create gap analysis with metrics normalized by number of tuples recommended."""
        if 'meta_gap' not in self.results_df.columns or 'rec_predicted_count' not in self.results_df.columns:
            return None
        
        # Find available metrics
        available_metrics = []
        metric_names = {
            'eval_overlap_accuracy': 'Accuracy',
            'eval_precision': 'Precision', 
            'eval_recall': 'Recall',
            'eval_f1_score': 'F1 Score',
            'eval_roc_auc': 'ROC-AUC'
        }
        
        for metric_col, metric_name in metric_names.items():
            if metric_col in self.results_df.columns:
                available_metrics.append((metric_col, metric_name))
        
        if not available_metrics:
            return None
        
        # Calculate normalized metrics
        normalized_df = self.results_df.copy()
        for metric_col, metric_name in available_metrics:
            normalized_df[f'norm_{metric_col}'] = normalized_df[metric_col] / (normalized_df['rec_predicted_count'] + 1e-6)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Normalized Performance vs Query Gap Analysis', fontsize=16, fontweight='bold')
        
        # 1. Line plot - First available normalized metric vs gap
        if available_metrics:
            metric_col, metric_name = available_metrics[0]
            norm_col = f'norm_{metric_col}'
            
            gap_analysis = normalized_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
                norm_col: ['mean', 'std', 'count']
            }).reset_index()
            gap_analysis.columns = ['recommender', 'gap', 'mean_metric', 'std_metric', 'count']
            
            for recommender in gap_analysis['recommender'].unique():
                data = gap_analysis[gap_analysis['recommender'] == recommender]
                ax1.errorbar(data['gap'], data['mean_metric'], yerr=data['std_metric'], 
                            label=recommender, marker='o', linewidth=2, markersize=6)
            
            ax1.set_title(f'A) {metric_name} per Tuple vs Query Gap')
            ax1.set_xlabel('Query Gap')
            ax1.set_ylabel(f'Mean {metric_name} per Tuple')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap of normalized primary metric by recommender and gap
        if available_metrics:
            metric_col, metric_name = available_metrics[0]
            norm_col = f'norm_{metric_col}'
            
            heatmap_data = normalized_df.pivot_table(
                values=norm_col, 
                index='meta_recommender_name', 
                columns='meta_gap', 
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                       ax=ax2, cbar_kws={'label': f'Mean {metric_name} per Tuple'})
            ax2.set_title(f'B) {metric_name} per Tuple Heatmap')
            ax2.set_xlabel('Query Gap')
            ax2.set_ylabel('Recommender System')
        
        # 3. Comparison of raw vs normalized for first metric
        if available_metrics:
            metric_col, metric_name = available_metrics[0]
            norm_col = f'norm_{metric_col}'
            
            # Calculate means for each recommender
            raw_means = normalized_df.groupby('meta_recommender_name')[metric_col].mean()
            norm_means = normalized_df.groupby('meta_recommender_name')[norm_col].mean()
            
            x = np.arange(len(raw_means))
            width = 0.35
            
            ax3.bar(x - width/2, raw_means.values, width, label=f'Raw {metric_name}', alpha=0.7)
            
            # Scale normalized values for visualization (multiply by mean tuple count)
            mean_tuple_count = normalized_df['rec_predicted_count'].mean()
            scaled_norm_means = norm_means * mean_tuple_count
            ax3.bar(x + width/2, scaled_norm_means.values, width, 
                   label=f'{metric_name} per Tuple (×{mean_tuple_count:.0f})', alpha=0.7)
            
            ax3.set_title(f'C) Raw vs Normalized {metric_name} Comparison')
            ax3.set_xlabel('Recommender System')
            ax3.set_ylabel(metric_name)
            ax3.set_xticks(x)
            ax3.set_xticklabels(raw_means.index, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency scatter plot (normalized metric vs tuple count)
        if available_metrics:
            metric_col, metric_name = available_metrics[0]
            norm_col = f'norm_{metric_col}'
            
            sns.scatterplot(data=normalized_df, x='rec_predicted_count', y=norm_col, 
                           hue='meta_recommender_name', ax=ax4, alpha=0.7, s=60)
            ax4.set_title(f'D) {metric_name} Efficiency vs Tuple Count')
            ax4.set_xlabel('Number of Recommended Tuples')
            ax4.set_ylabel(f'{metric_name} per Tuple')
            ax4.legend(title='Recommender', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Example usage of the enhanced analyzer with publication visualizations."""
    import sys
    
    # Allow optional command-line argument for custom results directory
    base_results_dir = "results/experiment"
    if len(sys.argv) > 1:
        base_results_dir = sys.argv[1]
        print(f"📁 Using custom results directory: {base_results_dir}")
    
    # Find the most recent experiment directory automatically
    most_recent_dir = ExperimentAnalyzer.find_most_recent_experiment_dir(base_results_dir)
    
    if most_recent_dir is None:
        print(f"❌ No experiment directories found in {base_results_dir}")
        print("💡 Usage: python experiment_analyzer.py [results_directory]")
        return
    
    print(f"🔍 Analyzing most recent experiment: {Path(most_recent_dir).name}")
    
    # Initialize analyzer with the most recent directory
    analyzer = ExperimentAnalyzer(
        experiment_data_dir=most_recent_dir,
        include_tuple_analysis=False
    )
    
    # Load all results
    results_df = analyzer.load_all_results()
    
    if not results_df.empty:
        print("Creating comprehensive analysis...")
        
        # Generate interactive dashboard
        analyzer.generate_performance_dashboard("dashboard.html")
        print("✓ Interactive dashboard created")
        
        # Create statistical summary
        stats = analyzer.create_statistical_summary()
        print("\n✓ Statistical Summary Generated:")
        if 'dataset_overview' in stats:
            overview = stats['dataset_overview']
            print(f"  - Total experiments: {overview.get('total_experiments', 'N/A')}")
            print(f"  - Recommenders: {len(overview.get('recommenders', []))}")
            print(f"  - Unique sessions: {overview.get('unique_sessions', 'N/A')}")
        
        # Create detailed report
        # analyzer.create_detailed_comparison_report("comparison_report.html")
        print("✓ Detailed comparison report created")
        
        # Export for further analysis
        analyzer.export_for_further_analysis("exports")
        print("✓ Data exported for further analysis")
        
        # Create publication-ready visualizations
        print("\nCreating publication-ready visualizations...")
        saved_files = analyzer.create_publication_visualizations(
            subdir="visualizations",
            save_pdf=True,
            save_individual=True,
            dpi=300
        )
        print("✓ Publication-ready visualizations created:")
        for viz_type, filepath in saved_files.items():
            print(f"  - {viz_type}: {filepath}")
        
        print("\n🎉 Complete analysis finished!")
        print(f"\nAll files saved to: {analyzer.output_dir}")
        print("Generated files:")
        print("- visualizations/ (PNG files + combined PDF)")
        print("- visualization_dashboard.html (overview)")
        print("- dashboard.html (interactive)")
        print("- comparison_report.html (detailed stats)")
        print("- exports/ (data + templates)")
        
    else:
        print("No experimental results found.")


if __name__ == "__main__":
    main()
