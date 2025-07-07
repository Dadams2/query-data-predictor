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

from experiments.experiment_collector import ExperimentCollector

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """
    Advanced analyzer for experimental results with interactive visualizations
    and comprehensive statistical analysis.
    """
    
    def __init__(self, 
                 experiment_data_dir: str,
                 include_tuple_analysis: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            experiment_data_dir: Directory containing experimental data
            include_tuple_analysis: Whether to load and analyze actual tuple data
        """
        self.data_dir = Path(experiment_data_dir)
        self.include_tuples = include_tuple_analysis
        self.collector = ExperimentCollector(base_output_dir=str(self.data_dir))
        
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
    
    def generate_performance_dashboard(self, output_file: str = "performance_dashboard.html"):
        """Generate comprehensive interactive performance dashboard."""
        
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return
        
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
            
            summary["performance_statistics"] = perf_stats.to_dict()
        
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
    
    def create_detailed_comparison_report(self, output_file: str = "detailed_comparison_report.html"):
        """Create a detailed HTML comparison report."""
        
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return
        
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
    
    def export_for_further_analysis(self, output_dir: str = "analysis_exports"):
        """Export data in various formats for further analysis."""
        
        if self.results_df is None or self.results_df.empty:
            logger.error("No data loaded. Call load_all_results() first.")
            return
        
        export_dir = Path(output_dir)
        export_dir.mkdir(exist_ok=True)
        
        # Export main results
        self.results_df.to_csv(export_dir / "experimental_results.csv", index=False)
        self.results_df.to_parquet(export_dir / "experimental_results.parquet", index=False)
        
        # Export summary statistics
        stats_summary = self.create_statistical_summary()
        with open(export_dir / "statistical_summary.json", 'w') as f:
            json.dump(stats_summary, f, indent=2, default=str)
        
        # Export for R analysis
        self._export_for_r(export_dir)
        
        # Export for Python/Jupyter analysis
        self._create_analysis_notebook(export_dir)
        
        logger.info(f"Analysis exports saved to {export_dir}")
    
    def _export_for_r(self, output_dir: Path):
        """Export data formatted for R analysis."""
        
        # Create R-friendly column names
        r_df = self.results_df.copy()
        r_df.columns = [col.replace('_', '.') for col in r_df.columns]
        
        # Save as CSV with R-friendly format
        r_df.to_csv(output_dir / "results_for_r.csv", index=False)
        
        # Create R analysis script
        r_script = '''
# Load the experimental results
results <- read.csv("results_for_r.csv")

# Basic summary
summary(results)

# ANOVA analysis
if("eval.overlap.accuracy" %in% names(results)) {
  model <- aov(eval.overlap.accuracy ~ meta.recommender.name, data=results)
  summary(model)
  
  # Post-hoc tests
  TukeyHSD(model)
}

# Visualizations
library(ggplot2)

# Accuracy by recommender
if("eval.overlap.accuracy" %in% names(results)) {
  ggplot(results, aes(x=meta.recommender.name, y=eval.overlap.accuracy)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title="Accuracy by Recommender", x="Recommender", y="Overlap Accuracy")
}
'''
        
        with open(output_dir / "analysis_script.R", 'w') as f:
            f.write(r_script)
    
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


def main():
    """Example usage of the enhanced analyzer."""
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(
        experiment_data_dir="enhanced_experiment_results",
        include_tuple_analysis=False
    )
    
    # Load all results
    results_df = analyzer.load_all_results()
    
    if not results_df.empty:
        # Generate dashboard
        analyzer.generate_performance_dashboard("dashboard.html")
        
        # Create statistical summary
        stats = analyzer.create_statistical_summary()
        print("Statistical Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Create detailed report
        analyzer.create_detailed_comparison_report("comparison_report.html")
        
        # Export for further analysis
        analyzer.export_for_further_analysis("analysis_exports")
        
        print("Analysis complete!")
    else:
        print("No experimental results found.")


if __name__ == "__main__":
    main()
