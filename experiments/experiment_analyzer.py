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
matplotlib.use('Agg')  # Use non-interactive backend

from experiment_collector import ExperimentCollector

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """
    Advanced analyzer for experimental results with interactive visualizations
    and comprehensive statistical analysis.
    """
    
    def __init__(self, 
                 experiment_data_dir: str,
                 output_dir: str = "results/experiment/experiment_results_analysis/analysis",
                 include_tuple_analysis: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            experiment_data_dir: Directory containing experimental data
            output_dir: Global output directory for all generated files
            include_tuple_analysis: Whether to load and analyze actual tuple data
        """
        self.data_dir = Path(experiment_data_dir)
        self.output_dir = Path(output_dir)
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
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Accuracy by Recommender',
                'Execution Time Distribution',
                'Accuracy vs Gap Analysis', 
                'Performance Heatmap',
                'Success Rate by Result Size',
                'Timeline Analysis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "violin"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}]
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
        
        # 2. Execution Time Distribution
        if 'meta_execution_time_seconds' in self.results_df.columns:
            for recommender in self.results_df['meta_recommender_name'].unique():
                data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
                fig.add_trace(
                    go.Violin(y=data['meta_execution_time_seconds'], name=recommender),
                    row=1, col=2
                )
        
        # 3. Accuracy vs Gap Analysis
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
                    row=2, col=1
                )
        
        # 4. Performance Heatmap
        if 'eval_overlap_accuracy' in self.results_df.columns:
            heatmap_data = self.results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_overlap_accuracy'].mean().unstack(fill_value=0)
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='RdYlGn'
                ),
                row=2, col=2
            )
        
        # 5. Success Rate by Result Size
        if 'prediction_size_category' in self.results_df.columns:
            success_data = self.results_df.groupby(['meta_recommender_name', 'prediction_size_category']).apply(
                lambda x: (x['meta_status'] == 'completed').mean()
            ).reset_index()
            success_data.columns = ['recommender', 'size_category', 'success_rate']
            
            for recommender in success_data['recommender'].unique():
                data = success_data[success_data['recommender'] == recommender]
                fig.add_trace(
                    go.Bar(x=data['size_category'], y=data['success_rate'], name=recommender),
                    row=3, col=1
                )
        
        # 6. Timeline Analysis
        if 'meta_timestamp' in self.results_df.columns and 'eval_overlap_accuracy' in self.results_df.columns:
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
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Recommender System Performance Dashboard",
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
        
        # Performance statistics
        if 'eval_overlap_accuracy' in self.results_df.columns:
            perf_stats = self.results_df.groupby('meta_recommender_name').agg({
                'eval_overlap_accuracy': ['count', 'mean', 'std', 'min', 'max', 'median'],
                'eval_precision': ['mean', 'std'] if 'eval_precision' in self.results_df.columns else [],
                'eval_recall': ['mean', 'std'] if 'eval_recall' in self.results_df.columns else [],
                'eval_f1_score': ['mean', 'std'] if 'eval_f1_score' in self.results_df.columns else [],
                'meta_execution_time_seconds': ['mean', 'std'] if 'meta_execution_time_seconds' in self.results_df.columns else []
            }).round(4)
            
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
        """Perform statistical significance tests between recommenders."""
        
        if 'eval_overlap_accuracy' not in self.results_df.columns:
            return {"error": "No accuracy data available"}
        
        from scipy import stats
        
        tests = {}
        recommenders = self.results_df['meta_recommender_name'].unique()
        
        # Pairwise t-tests
        for i, rec1 in enumerate(recommenders):
            for rec2 in recommenders[i+1:]:
                data1 = self.results_df[self.results_df['meta_recommender_name'] == rec1]['eval_overlap_accuracy'].dropna()
                data2 = self.results_df[self.results_df['meta_recommender_name'] == rec2]['eval_overlap_accuracy'].dropna()
                
                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    tests[f"{rec1}_vs_{rec2}"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "effect_size": float((data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2))
                    }
        
        # ANOVA test
        groups = [
            self.results_df[self.results_df['meta_recommender_name'] == rec]['eval_overlap_accuracy'].dropna()
            for rec in recommenders
        ]
        
        if all(len(group) > 1 for group in groups):
            f_stat, p_value = stats.f_oneway(*groups)
            tests["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
        
        return tests
    
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
                        <th>Std Accuracy</th>
                        <th>Min Accuracy</th>
                        <th>Max Accuracy</th>
                        <th>Median Accuracy</th>
                    </tr>
            """
            
            perf_stats = stats_summary['performance_statistics']
            if 'eval_overlap_accuracy' in perf_stats:
                for recommender, stats in perf_stats['eval_overlap_accuracy'].items():
                    html += f"""
                    <tr>
                        <td>{recommender}</td>
                        <td class="number">{stats.get('count', 'N/A')}</td>
                        <td class="number">{stats.get('mean', 'N/A'):.4f}</td>
                        <td class="number">{stats.get('std', 'N/A'):.4f}</td>
                        <td class="number">{stats.get('min', 'N/A'):.4f}</td>
                        <td class="number">{stats.get('max', 'N/A'):.4f}</td>
                        <td class="number">{stats.get('median', 'N/A'):.4f}</td>
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
            ("result_size_analysis", self._create_result_size_analysis),
            ("execution_time_analysis", self._create_execution_time_analysis),
            ("performance_heatmap", self._create_performance_heatmap),
            ("distribution_analysis", self._create_distribution_analysis),
            ("correlation_analysis", self._create_correlation_analysis),
            ("error_analysis", self._create_error_analysis),
            ("temporal_analysis", self._create_temporal_analysis),
            ("statistical_summary_plot", self._create_statistical_summary_plot)
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
    
    def _create_accuracy_comparison(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready accuracy comparison visualization."""
        if 'eval_overlap_accuracy' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Recommender System Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Box plot comparison
        sns.boxplot(data=self.results_df, x='meta_recommender_name', 
                   y='eval_overlap_accuracy', ax=ax1)
        ax1.set_title('A) Accuracy Distribution by Recommender')
        ax1.set_xlabel('Recommender System')
        ax1.set_ylabel('Overlap Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add statistical annotations
        self._add_statistical_annotations(ax1, 'meta_recommender_name', 'eval_overlap_accuracy')
        
        # 2. Violin plot with quartiles
        sns.violinplot(data=self.results_df, x='meta_recommender_name', 
                      y='eval_overlap_accuracy', ax=ax2, inner='quartile')
        ax2.set_title('B) Accuracy Distribution with Quartiles')
        ax2.set_xlabel('Recommender System')
        ax2.set_ylabel('Overlap Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Mean accuracy with confidence intervals
        accuracy_stats = self.results_df.groupby('meta_recommender_name')['eval_overlap_accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Calculate 95% confidence intervals
        from scipy import stats
        accuracy_stats['ci'] = accuracy_stats.apply(
            lambda row: stats.t.interval(0.95, row['count']-1, 
                                       loc=row['mean'], 
                                       scale=row['std']/np.sqrt(row['count']))[1] - row['mean']
            if row['count'] > 1 else 0, axis=1
        )
        
        bars = ax3.bar(accuracy_stats['meta_recommender_name'], accuracy_stats['mean'],
                       yerr=accuracy_stats['ci'], capsize=5, alpha=0.8)
        ax3.set_title('C) Mean Accuracy with 95% Confidence Intervals')
        ax3.set_xlabel('Recommender System')
        ax3.set_ylabel('Mean Overlap Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean_val, ci_val in zip(bars, accuracy_stats['mean'], accuracy_stats['ci']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + ci_val + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Cumulative distribution
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['eval_overlap_accuracy']
            sorted_data = np.sort(data.dropna())
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax4.plot(sorted_data, cumulative, label=recommender, linewidth=2)
        
        ax4.set_title('D) Cumulative Distribution Function')
        ax4.set_xlabel('Overlap Accuracy')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_gap_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready gap analysis visualization."""
        if 'meta_gap' not in self.results_df.columns or 'eval_overlap_accuracy' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Query Gap Impact Analysis', fontsize=16, fontweight='bold')
        
        # 1. Line plot of accuracy vs gap
        gap_analysis = self.results_df.groupby(['meta_recommender_name', 'meta_gap']).agg({
            'eval_overlap_accuracy': ['mean', 'std', 'count']
        }).reset_index()
        gap_analysis.columns = ['recommender', 'gap', 'mean_acc', 'std_acc', 'count']
        
        for recommender in gap_analysis['recommender'].unique():
            data = gap_analysis[gap_analysis['recommender'] == recommender]
            ax1.errorbar(data['gap'], data['mean_acc'], yerr=data['std_acc'], 
                        label=recommender, marker='o', linewidth=2, markersize=6)
        
        ax1.set_title('A) Accuracy vs Query Gap')
        ax1.set_xlabel('Query Gap')
        ax1.set_ylabel('Mean Overlap Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap of accuracy by recommender and gap
        heatmap_data = self.results_df.pivot_table(
            values='eval_overlap_accuracy', 
            index='meta_recommender_name', 
            columns='meta_gap', 
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax2, cbar_kws={'label': 'Mean Accuracy'})
        ax2.set_title('B) Accuracy Heatmap (Recommender × Gap)')
        ax2.set_xlabel('Query Gap')
        ax2.set_ylabel('Recommender System')
        
        # 3. Box plots by gap
        sns.boxplot(data=self.results_df, x='meta_gap', y='eval_overlap_accuracy', 
                   hue='meta_recommender_name', ax=ax3)
        ax3.set_title('C) Accuracy Distribution by Gap')
        ax3.set_xlabel('Query Gap')
        ax3.set_ylabel('Overlap Accuracy')
        ax3.legend(title='Recommender', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Gap effect size analysis
        gap_effects = []
        for recommender in self.results_df['meta_recommender_name'].unique():
            rec_data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
            for gap in sorted(rec_data['meta_gap'].unique()):
                gap_data = rec_data[rec_data['meta_gap'] == gap]['eval_overlap_accuracy']
                baseline_data = rec_data[rec_data['meta_gap'] == rec_data['meta_gap'].min()]['eval_overlap_accuracy']
                
                if len(gap_data) > 0 and len(baseline_data) > 0:
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
            ax4.set_title('D) Gap Effect Size (vs Baseline)')
            ax4.set_xlabel('Query Gap')
            ax4.set_ylabel('Recommender System')
        
        plt.tight_layout()
        return fig
    
    def _create_result_size_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready result size impact analysis."""
        if 'rec_predicted_count' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Result Set Size Impact Analysis', fontsize=16, fontweight='bold')
        
        # Create size bins if not already present
        if 'prediction_size_category' not in self.results_df.columns:
            self.results_df['prediction_size_category'] = pd.cut(
                self.results_df['rec_predicted_count'],
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge']
            )
        
        # 1. Accuracy by result size category
        if 'eval_overlap_accuracy' in self.results_df.columns:
            sns.boxplot(data=self.results_df, x='prediction_size_category', 
                       y='eval_overlap_accuracy', ax=ax1)
            ax1.set_title('A) Accuracy by Result Size Category')
            ax1.set_xlabel('Predicted Result Size Category')
            ax1.set_ylabel('Overlap Accuracy')
        
        # 2. Scatter plot of accuracy vs result size
        if 'eval_overlap_accuracy' in self.results_df.columns:
            for recommender in self.results_df['meta_recommender_name'].unique():
                data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
                ax2.scatter(data['rec_predicted_count'], data['eval_overlap_accuracy'], 
                           label=recommender, alpha=0.6, s=30)
            
            ax2.set_title('B) Accuracy vs Predicted Result Count')
            ax2.set_xlabel('Predicted Result Count')
            ax2.set_ylabel('Overlap Accuracy')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Result size distribution by recommender
        sns.violinplot(data=self.results_df, x='meta_recommender_name', 
                      y='rec_predicted_count', ax=ax3)
        ax3.set_title('C) Predicted Result Size Distribution')
        ax3.set_xlabel('Recommender System')
        ax3.set_ylabel('Predicted Result Count')
        ax3.set_yscale('log')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Success rate by size category
        if 'meta_status' in self.results_df.columns:
            success_data = self.results_df.groupby(['meta_recommender_name', 'prediction_size_category']).apply(
                lambda x: (x['meta_status'] == 'completed').mean()
            ).reset_index()
            success_data.columns = ['recommender', 'size_category', 'success_rate']
            
            success_pivot = success_data.pivot(index='recommender', columns='size_category', values='success_rate')
            sns.heatmap(success_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=ax4, cbar_kws={'label': 'Success Rate'})
            ax4.set_title('D) Success Rate by Size Category')
            ax4.set_xlabel('Result Size Category')
            ax4.set_ylabel('Recommender System')
        
        plt.tight_layout()
        return fig
    
    def _create_execution_time_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready execution time analysis."""
        if 'meta_execution_time_seconds' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Execution Time Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Execution time distribution by recommender
        sns.boxplot(data=self.results_df, x='meta_recommender_name', 
                   y='meta_execution_time_seconds', ax=ax1)
        ax1.set_title('A) Execution Time Distribution')
        ax1.set_xlabel('Recommender System')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Time vs result size
        if 'rec_predicted_count' in self.results_df.columns:
            for recommender in self.results_df['meta_recommender_name'].unique():
                data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
                ax2.scatter(data['rec_predicted_count'], data['meta_execution_time_seconds'], 
                           label=recommender, alpha=0.6, s=30)
            
            ax2.set_title('B) Execution Time vs Result Size')
            ax2.set_xlabel('Predicted Result Count')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency analysis (accuracy per second)
        if 'eval_overlap_accuracy' in self.results_df.columns:
            self.results_df['efficiency'] = self.results_df['eval_overlap_accuracy'] / self.results_df['meta_execution_time_seconds']
            
            sns.boxplot(data=self.results_df, x='meta_recommender_name', 
                       y='efficiency', ax=ax3)
            ax3.set_title('C) Efficiency (Accuracy per Second)')
            ax3.set_xlabel('Recommender System')
            ax3.set_ylabel('Accuracy / Execution Time')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Time distribution histogram
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results_df['meta_recommender_name'].unique())))
        for i, recommender in enumerate(self.results_df['meta_recommender_name'].unique()):
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['meta_execution_time_seconds']
            ax4.hist(data, bins=20, alpha=0.7, label=recommender, color=colors[i], density=True)
        
        ax4.set_title('D) Execution Time Distribution')
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('Density')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_performance_heatmap(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create comprehensive performance heatmap."""
        if 'eval_overlap_accuracy' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Performance Heatmap Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy heatmap by recommender and gap
        if 'meta_gap' in self.results_df.columns:
            heatmap_data = self.results_df.pivot_table(
                values='eval_overlap_accuracy', 
                index='meta_recommender_name', 
                columns='meta_gap', 
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=ax1, cbar_kws={'label': 'Mean Accuracy'})
            ax1.set_title('A) Mean Accuracy by Gap')
            ax1.set_xlabel('Query Gap')
            ax1.set_ylabel('Recommender')
        
        # 2. Count heatmap
        if 'meta_gap' in self.results_df.columns:
            count_data = self.results_df.pivot_table(
                values='eval_overlap_accuracy', 
                index='meta_recommender_name', 
                columns='meta_gap', 
                aggfunc='count'
            )
            
            sns.heatmap(count_data, annot=True, fmt='d', cmap='Blues', 
                       ax=ax2, cbar_kws={'label': 'Number of Experiments'})
            ax2.set_title('B) Experiment Count by Gap')
            ax2.set_xlabel('Query Gap')
            ax2.set_ylabel('Recommender')
        
        # 3. Standard deviation heatmap
        if 'meta_gap' in self.results_df.columns:
            std_data = self.results_df.pivot_table(
                values='eval_overlap_accuracy', 
                index='meta_recommender_name', 
                columns='meta_gap', 
                aggfunc='std'
            )
            
            sns.heatmap(std_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax3, cbar_kws={'label': 'Accuracy Std Dev'})
            ax3.set_title('C) Accuracy Variability by Gap')
            ax3.set_xlabel('Query Gap')
            ax3.set_ylabel('Recommender')
        
        # 4. Performance by result size category
        if 'prediction_size_category' in self.results_df.columns:
            size_perf = self.results_df.pivot_table(
                values='eval_overlap_accuracy',
                index='meta_recommender_name',
                columns='prediction_size_category',
                aggfunc='mean'
            )
            
            sns.heatmap(size_perf, annot=True, fmt='.3f', cmap='RdYlGn',
                       ax=ax4, cbar_kws={'label': 'Mean Accuracy'})
            ax4.set_title('D) Accuracy by Result Size')
            ax4.set_xlabel('Result Size Category')
            ax4.set_ylabel('Recommender')
        
        plt.tight_layout()
        return fig
    
    def _create_distribution_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create publication-ready distribution analysis."""
        if 'eval_overlap_accuracy' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Statistical Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram with KDE
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['eval_overlap_accuracy']
            sns.histplot(data, kde=True, alpha=0.6, label=recommender, ax=ax1)
        
        ax1.set_title('A) Accuracy Distribution with KDE')
        ax1.set_xlabel('Overlap Accuracy')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # 2. Q-Q plots for normality assessment
        from scipy.stats import probplot
        recommenders = self.results_df['meta_recommender_name'].unique()
        n_recs = len(recommenders)
        
        if n_recs > 0:
            for i, recommender in enumerate(recommenders[:4]):  # Limit to 4 for space
                data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['eval_overlap_accuracy'].dropna()
                if len(data) > 0:
                    probplot(data, dist="norm", plot=ax2)
                    ax2.get_lines()[-2].set_alpha(0.7)
                    ax2.get_lines()[-1].set_alpha(0.7)
        
        ax2.set_title('B) Q-Q Plot (Normality Check)')
        
        # 3. Box-Cox transformation analysis
        from scipy.stats import boxcox
        
        try:
            # Only transform positive values
            positive_data = self.results_df[self.results_df['eval_overlap_accuracy'] > 0]['eval_overlap_accuracy']
            if len(positive_data) > 0:
                transformed_data, lambda_val = boxcox(positive_data)
                ax3.hist(transformed_data, bins=30, alpha=0.7, density=True)
                ax3.set_title(f'C) Box-Cox Transformed Data (λ={lambda_val:.3f})')
                ax3.set_xlabel('Transformed Accuracy')
                ax3.set_ylabel('Density')
        except Exception:
            ax3.text(0.5, 0.5, 'Box-Cox transformation\nnot applicable', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('C) Box-Cox Transformation')
        
        # 4. Cumulative distribution comparison
        for recommender in self.results_df['meta_recommender_name'].unique():
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['eval_overlap_accuracy'].dropna()
            if len(data) > 0:
                sorted_data = np.sort(data)
                cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax4.plot(sorted_data, cumulative, label=recommender, linewidth=2)
        
        ax4.set_title('D) Empirical Cumulative Distribution')
        ax4.set_xlabel('Overlap Accuracy')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_correlation_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create correlation analysis visualization."""
        # Select numeric columns
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-meaningful columns
        exclude_cols = ['meta_timestamp']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 2:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Correlation and Relationship Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation matrix
        corr_matrix = self.results_df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('A) Correlation Matrix')
        
        # 2. Scatter plot of key relationships
        if 'eval_overlap_accuracy' in numeric_cols and 'meta_execution_time_seconds' in numeric_cols:
            for recommender in self.results_df['meta_recommender_name'].unique():
                data = self.results_df[self.results_df['meta_recommender_name'] == recommender]
                ax2.scatter(data['meta_execution_time_seconds'], data['eval_overlap_accuracy'], 
                           label=recommender, alpha=0.6, s=30)
            
            ax2.set_title('B) Accuracy vs Execution Time')
            ax2.set_xlabel('Execution Time (seconds)')
            ax2.set_ylabel('Overlap Accuracy')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Partial correlation plot
        if len(numeric_cols) >= 3:
            # Create pairplot for subset of variables
            key_vars = [col for col in ['eval_overlap_accuracy', 'meta_execution_time_seconds', 
                                      'rec_predicted_count', 'meta_gap'] if col in numeric_cols][:4]
            
            if len(key_vars) >= 2:
                subset_data = self.results_df[key_vars + ['meta_recommender_name']].dropna()
                if len(subset_data) > 0:
                    # Create scatter plot matrix
                    from pandas.plotting import scatter_matrix
                    scatter_matrix(subset_data[key_vars], ax=ax3, alpha=0.6, figsize=(6, 6))
                    ax3.set_title('C) Scatter Plot Matrix')
        
        # 4. Feature importance (if accuracy is available)
        if 'eval_overlap_accuracy' in numeric_cols:
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.preprocessing import LabelEncoder
                
                # Prepare data for feature importance
                feature_cols = [col for col in numeric_cols if col != 'eval_overlap_accuracy']
                if len(feature_cols) > 0:
                    X = self.results_df[feature_cols].fillna(0)
                    y = self.results_df['eval_overlap_accuracy'].fillna(0)
                    
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': rf.feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    ax4.barh(importance_df['feature'], importance_df['importance'])
                    ax4.set_title('D) Feature Importance for Accuracy')
                    ax4.set_xlabel('Importance')
            except ImportError:
                ax4.text(0.5, 0.5, 'sklearn not available\nfor feature importance', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('D) Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def _create_error_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create error and failure analysis visualization."""
        if 'meta_status' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Error and Failure Analysis', fontsize=16, fontweight='bold')
        
        # 1. Success rate by recommender
        success_rates = self.results_df.groupby('meta_recommender_name').apply(
            lambda x: (x['meta_status'] == 'completed').mean()
        ).sort_values(ascending=False)
        
        bars = ax1.bar(success_rates.index, success_rates.values)
        ax1.set_title('A) Success Rate by Recommender')
        ax1.set_xlabel('Recommender System')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 2. Error patterns by gap (if available)
        if 'meta_gap' in self.results_df.columns:
            error_by_gap = self.results_df.groupby(['meta_gap', 'meta_recommender_name']).apply(
                lambda x: (x['meta_status'] != 'completed').mean()
            ).reset_index()
            error_by_gap.columns = ['gap', 'recommender', 'error_rate']
            
            error_pivot = error_by_gap.pivot(index='recommender', columns='gap', values='error_rate')
            sns.heatmap(error_pivot, annot=True, fmt='.2f', cmap='Reds', 
                       ax=ax2, cbar_kws={'label': 'Error Rate'})
            ax2.set_title('B) Error Rate by Gap')
            ax2.set_xlabel('Query Gap')
            ax2.set_ylabel('Recommender')
        
        # 3. Status distribution
        status_counts = self.results_df['meta_status'].value_counts()
        wedges, texts, autotexts = ax3.pie(status_counts.values, labels=status_counts.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('C) Overall Status Distribution')
        
        # 4. Time to failure analysis (if execution time available)
        if 'meta_execution_time_seconds' in self.results_df.columns:
            failed_data = self.results_df[self.results_df['meta_status'] != 'completed']
            success_data = self.results_df[self.results_df['meta_status'] == 'completed']
            
            if len(failed_data) > 0 and len(success_data) > 0:
                ax4.hist(success_data['meta_execution_time_seconds'], bins=20, alpha=0.7, 
                        label='Successful', density=True)
                ax4.hist(failed_data['meta_execution_time_seconds'], bins=20, alpha=0.7, 
                        label='Failed', density=True)
                ax4.set_title('D) Execution Time: Success vs Failure')
                ax4.set_xlabel('Execution Time (seconds)')
                ax4.set_ylabel('Density')
                ax4.set_xscale('log')
                ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_temporal_analysis(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create temporal analysis of experimental results."""
        if 'meta_timestamp' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Temporal Analysis of Experiments', fontsize=16, fontweight='bold')
        
        # Ensure timestamp is datetime
        self.results_df['meta_timestamp'] = pd.to_datetime(self.results_df['meta_timestamp'])
        
        # 1. Experiment count over time
        hourly_counts = self.results_df.groupby([
            pd.Grouper(key='meta_timestamp', freq='H'),
            'meta_recommender_name'
        ]).size().reset_index()
        hourly_counts.columns = ['timestamp', 'recommender', 'count']
        
        for recommender in hourly_counts['recommender'].unique():
            data = hourly_counts[hourly_counts['recommender'] == recommender]
            ax1.plot(data['timestamp'], data['count'], label=recommender, marker='o')
        
        ax1.set_title('A) Experiment Frequency Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Experiments per Hour')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance over time
        if 'eval_overlap_accuracy' in self.results_df.columns:
            hourly_perf = self.results_df.groupby([
                pd.Grouper(key='meta_timestamp', freq='H'),
                'meta_recommender_name'
            ])['eval_overlap_accuracy'].mean().reset_index()
            
            for recommender in hourly_perf['meta_recommender_name'].unique():
                data = hourly_perf[hourly_perf['meta_recommender_name'] == recommender]
                ax2.plot(data['meta_timestamp'], data['eval_overlap_accuracy'], 
                        label=recommender, marker='o')
            
            ax2.set_title('B) Performance Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Mean Accuracy')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Execution time trends
        if 'meta_execution_time_seconds' in self.results_df.columns:
            hourly_time = self.results_df.groupby([
                pd.Grouper(key='meta_timestamp', freq='H'),
                'meta_recommender_name'
            ])['meta_execution_time_seconds'].mean().reset_index()
            
            for recommender in hourly_time['meta_recommender_name'].unique():
                data = hourly_time[hourly_time['meta_recommender_name'] == recommender]
                ax3.plot(data['meta_timestamp'], data['meta_execution_time_seconds'], 
                        label=recommender, marker='o')
            
            ax3.set_title('C) Execution Time Trends')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Mean Execution Time (s)')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Daily experiment summary
        daily_summary = self.results_df.groupby(
            self.results_df['meta_timestamp'].dt.date
        ).agg({
            'meta_recommender_name': 'count',
            'eval_overlap_accuracy': 'mean' if 'eval_overlap_accuracy' in self.results_df.columns else lambda x: 0,
            'meta_execution_time_seconds': 'mean' if 'meta_execution_time_seconds' in self.results_df.columns else lambda x: 0
        }).reset_index()
        
        daily_summary.columns = ['date', 'total_experiments', 'mean_accuracy', 'mean_time']
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(daily_summary['date'], daily_summary['total_experiments'], 
                        'b-o', label='Total Experiments')
        
        if 'eval_overlap_accuracy' in self.results_df.columns:
            line2 = ax4_twin.plot(daily_summary['date'], daily_summary['mean_accuracy'], 
                                'r-s', label='Mean Accuracy')
        
        ax4.set_title('D) Daily Experiment Summary')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Total Experiments', color='b')
        ax4_twin.set_ylabel('Mean Accuracy', color='r')
        ax4.tick_params(axis='x', rotation=45)
        
        # Combine legends
        lines = line1
        if 'eval_overlap_accuracy' in self.results_df.columns:
            lines += line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def _create_statistical_summary_plot(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create statistical summary visualization."""
        if 'eval_overlap_accuracy' not in self.results_df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Statistical Summary and Tests', fontsize=16, fontweight='bold')
        
        # 1. Summary statistics table as plot
        summary_stats = self.results_df.groupby('meta_recommender_name')['eval_overlap_accuracy'].describe()
        
        # Create table plot
        ax1.axis('tight')
        ax1.axis('off')
        table_data = summary_stats.round(4).reset_index()
        table = ax1.table(cellText=table_data.values, colLabels=table_data.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax1.set_title('A) Summary Statistics')
        
        # 2. Effect sizes between recommenders
        recommenders = self.results_df['meta_recommender_name'].unique()
        effect_sizes = []
        
        for i, rec1 in enumerate(recommenders):
            for rec2 in recommenders[i+1:]:
                data1 = self.results_df[self.results_df['meta_recommender_name'] == rec1]['eval_overlap_accuracy'].dropna()
                data2 = self.results_df[self.results_df['meta_recommender_name'] == rec2]['eval_overlap_accuracy'].dropna()
                
                if len(data1) > 1 and len(data2) > 1:
                    # Cohen's d
                    pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) / 
                                       (len(data1) + len(data2) - 2))
                    effect_size = (data1.mean() - data2.mean()) / pooled_std
                    effect_sizes.append({
                        'comparison': f'{rec1}\nvs\n{rec2}',
                        'effect_size': effect_size
                    })
        
        if effect_sizes:
            effect_df = pd.DataFrame(effect_sizes)
            bars = ax2.barh(effect_df['comparison'], effect_df['effect_size'])
            ax2.set_title('B) Effect Sizes (Cohen\'s d)')
            ax2.set_xlabel('Effect Size')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Small')
            ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
            ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
            ax2.legend()
        
        # 3. Confidence intervals
        confidence_intervals = []
        for recommender in recommenders:
            data = self.results_df[self.results_df['meta_recommender_name'] == recommender]['eval_overlap_accuracy'].dropna()
            if len(data) > 1:
                from scipy import stats
                mean = data.mean()
                sem = data.sem()
                ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
                confidence_intervals.append({
                    'recommender': recommender,
                    'mean': mean,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'error': ci[1] - mean
                })
        
        if confidence_intervals:
            ci_df = pd.DataFrame(confidence_intervals)
            bars = ax3.bar(ci_df['recommender'], ci_df['mean'], 
                          yerr=ci_df['error'], capsize=5, alpha=0.8)
            ax3.set_title('C) 95% Confidence Intervals')
            ax3.set_xlabel('Recommender System')
            ax3.set_ylabel('Mean Accuracy')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean_val in zip(bars, ci_df['mean']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Power analysis simulation
        if len(recommenders) >= 2:
            # Simulate power analysis for detecting differences
            from scipy.stats import ttest_ind
            
            power_results = []
            effect_sizes_test = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
            
            for effect_size in effect_sizes_test:
                # Use first two recommenders as reference
                ref_data = self.results_df[self.results_df['meta_recommender_name'] == recommenders[0]]['eval_overlap_accuracy'].dropna()
                
                if len(ref_data) > 10:
                    power_count = 0
                    n_simulations = 100
                    
                    for _ in range(n_simulations):
                        # Simulate data with known effect size
                        group1 = np.random.normal(ref_data.mean(), ref_data.std(), len(ref_data))
                        group2 = np.random.normal(ref_data.mean() + effect_size * ref_data.std(), 
                                                ref_data.std(), len(ref_data))
                        
                        _, p_value = ttest_ind(group1, group2)
                        if p_value < 0.05:
                            power_count += 1
                    
                    power = power_count / n_simulations
                    power_results.append({'effect_size': effect_size, 'power': power})
            
            if power_results:
                power_df = pd.DataFrame(power_results)
                ax4.plot(power_df['effect_size'], power_df['power'], 'bo-', linewidth=2)
                ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power')
                ax4.set_title('D) Statistical Power Analysis')
                ax4.set_xlabel('Effect Size')
                ax4.set_ylabel('Statistical Power')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def _add_statistical_annotations(self, ax, x_col: str, y_col: str):
        """Add statistical significance annotations to plots."""
        try:
            from scipy.stats import ttest_ind
            
            groups = self.results_df[x_col].unique()
            if len(groups) < 2:
                return
            
            # Perform pairwise t-tests
            max_y = self.results_df[y_col].max()
            y_offset = max_y * 0.05
            
            for i, group1 in enumerate(groups[:-1]):
                for j, group2 in enumerate(groups[i+1:], i+1):
                    data1 = self.results_df[self.results_df[x_col] == group1][y_col].dropna()
                    data2 = self.results_df[self.results_df[x_col] == group2][y_col].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        _, p_value = ttest_ind(data1, data2)
                        
                        if p_value < 0.001:
                            sig_text = '***'
                        elif p_value < 0.01:
                            sig_text = '**'
                        elif p_value < 0.05:
                            sig_text = '*'
                        else:
                            sig_text = 'ns'
                        
                        # Add significance annotation
                        y_pos = max_y + y_offset * (j - i)
                        ax.annotate(sig_text, xy=((i + j) / 2, y_pos), 
                                  ha='center', va='bottom', fontsize=10)
                        
                        # Add line
                        ax.plot([i, j], [y_pos, y_pos], 'k-', alpha=0.5)
                        
        except ImportError:
            pass  # scipy not available
    
    def _create_publication_dashboard(self, output_dir: Path) -> str:
        """Create an HTML dashboard summarizing all visualizations."""
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Publication-Ready Visualization Dashboard</title>
            <style>
                body {{ 
                    font-family: 'Times New Roman', Times, serif; 
                    margin: 20px; 
                    line-height: 1.6; 
                    background-color: #f8f9fa;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    text-align: center; 
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .section {{ 
                    background: white;
                    margin: 20px 0; 
                    padding: 25px; 
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .viz-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0; 
                }}
                .viz-item {{ 
                    background: white;
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }}
                .viz-item h3 {{ 
                    margin-top: 0; 
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .metric {{ 
                    background: #e9ecef; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                    border-left: 4px solid #667eea;
                }}
                .highlight {{ 
                    background: #d4edda; 
                    border-left: 4px solid #28a745; 
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 15px 0; 
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #667eea; 
                    color: white;
                }}
                .number {{ 
                    text-align: right; 
                }}
                .download-links {{ 
                    background: #fff3cd; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .download-links a {{ 
                    color: #856404; 
                    text-decoration: none; 
                    margin-right: 15px;
                    font-weight: bold;
                }}
                .footer {{ 
                    text-align: center; 
                    margin-top: 40px; 
                    padding: 20px; 
                    background: #343a40; 
                    color: white;
                    border-radius: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Publication-Ready Visualization Dashboard</h1>
                <p>Comprehensive Analysis of Recommender System Experiments</p>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add dataset overview
        if self.results_df is not None and not self.results_df.empty:
            dashboard_html += f"""
            <div class="section">
                <h2>📊 Dataset Overview</h2>
                <div class="metric">
                    <strong>Total Experiments:</strong> {len(self.results_df):,}
                </div>
                <div class="metric">
                    <strong>Unique Sessions:</strong> {self.results_df['meta_session_id'].nunique():,}
                </div>
                <div class="metric">
                    <strong>Recommenders Tested:</strong> {', '.join(self.results_df['meta_recommender_name'].unique())}
                </div>
                <div class="metric">
                    <strong>Date Range:</strong> {self.results_df['meta_timestamp'].min()} to {self.results_df['meta_timestamp'].max()}
                </div>
            </div>
            """
        
        # Add visualization descriptions
        visualizations = [
            {
                "name": "Accuracy Comparison",
                "file": "accuracy_comparison.png",
                "description": "Comprehensive comparison of recommender accuracy using box plots, violin plots, confidence intervals, and cumulative distributions. Shows statistical significance between systems."
            },
            {
                "name": "Gap Analysis", 
                "file": "gap_analysis.png",
                "description": "Analysis of how query gap affects recommender performance. Includes trend lines, heatmaps, and effect size calculations."
            },
            {
                "name": "Result Size Analysis",
                "file": "result_size_analysis.png", 
                "description": "Impact of predicted result set size on accuracy and success rates. Shows scaling behavior and size category effects."
            },
            {
                "name": "Execution Time Analysis",
                "file": "execution_time_analysis.png",
                "description": "Performance timing analysis including efficiency metrics (accuracy per second) and time distribution patterns."
            },
            {
                "name": "Performance Heatmap",
                "file": "performance_heatmap.png",
                "description": "Multi-dimensional performance visualization showing accuracy, variability, and experiment counts across different conditions."
            },
            {
                "name": "Distribution Analysis", 
                "file": "distribution_analysis.png",
                "description": "Statistical distribution analysis including normality checks, transformations, and empirical cumulative distributions."
            },
            {
                "name": "Correlation Analysis",
                "file": "correlation_analysis.png", 
                "description": "Correlation matrices and relationship analysis between experimental variables and performance metrics."
            },
            {
                "name": "Error Analysis",
                "file": "error_analysis.png",
                "description": "Failure pattern analysis including success rates, error conditions, and time-to-failure distributions."
            },
            {
                "name": "Temporal Analysis",
                "file": "temporal_analysis.png",
                "description": "Time-based analysis showing experiment frequency, performance trends, and temporal patterns."
            },
            {
                "name": "Statistical Summary",
                "file": "statistical_summary_plot.png",
                "description": "Comprehensive statistical analysis including effect sizes, confidence intervals, and power analysis."
            }
        ]
        
        dashboard_html += """
            <div class="section">
                <h2>📈 Available Visualizations</h2>
                <div class="viz-grid">
        """
        
        for viz in visualizations:
            dashboard_html += f"""
                <div class="viz-item">
                    <h3>{viz['name']}</h3>
                    <p>{viz['description']}</p>
                    <div class="download-links">
                        <a href="{viz['file']}" target="_blank">📄 View PNG</a>
                    </div>
                </div>
            """
        
        dashboard_html += """
                </div>
            </div>
        """
        
        # Add download section
        dashboard_html += """
            <div class="section">
                <h2>📁 Download Options</h2>
                <div class="download-links">
                    <a href="publication_figures.pdf" target="_blank">📑 Download All Figures (PDF)</a>
                    <a href="../analysis_exports/experimental_results.csv" target="_blank">📊 Download Raw Data (CSV)</a>
                    <a href="../analysis_exports/statistical_summary.json" target="_blank">📋 Download Statistics (JSON)</a>
                </div>
            </div>
        """
        
        # Add usage guide
        dashboard_html += """
            <div class="section">
                <h2>📖 Usage Guide</h2>
                <div class="metric">
                    <h4>For Publications:</h4>
                    <p>All figures are generated at 300 DPI and are publication-ready. Use the individual PNG files for journal submissions or the combined PDF for reports.</p>
                </div>
                <div class="metric">
                    <h4>For Presentations:</h4>
                    <p>PNG files can be directly inserted into PowerPoint or similar presentation software. All text is sized appropriately for projection.</p>
                </div>
                <div class="metric">
                    <h4>For Further Analysis:</h4>
                    <p>Download the raw data files and use the provided R script or Jupyter notebook templates for custom analysis.</p>
                </div>
            </div>
        """
        
        dashboard_html += """
            <div class="footer">
                <p>Generated by ExperimentAnalyzer with Publication-Ready Visualizations</p>
                <p>All figures follow publication standards with appropriate fonts, sizes, and statistical annotations</p>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = output_dir / "visualization_dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Visualization dashboard created: {dashboard_path}")
        return str(dashboard_path)


def main():
    """Example usage of the enhanced analyzer with publication visualizations."""
    
    # Initialize analyzer with global output directory
    analyzer = ExperimentAnalyzer(
        experiment_data_dir="experiment_results",
        output_dir="results/experiment/experiment_results_analysis/analysis",
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
        analyzer.create_detailed_comparison_report("comparison_report.html")
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
