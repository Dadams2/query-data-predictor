"""
Analysis module for query prediction experiment results.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """
    Analyzer for query prediction experiment results.
    Reads experiment output files and generates plots and visualizations.
    """
    
    def __init__(self, results_dir: Path, config: Dict[str, Any] = None):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir: Path to the directory containing experiment results
            config: Configuration dictionary for analysis parameters
        """
        self.results_dir = Path(results_dir)
        self.config = config or {}
        self.results_data = {}
        
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {self.results_dir}")
        
        logger.info(f"Initialized ResultsAnalyzer for directory: {self.results_dir}")
    
    def analyze(self):
        """
        Main analysis method that orchestrates all analysis tasks.
        """
        logger.info("Starting comprehensive results analysis...")
        
        # Load all result files
        self._load_results()
        
        # Log summary information
        self._log_experiment_summary()
        
        # Generate all visualizations per session
        plots_created = self._generate_all_visualizations()
        
        logger.info("Analysis completed")
        return {
            'visualizations': {
                'plots_created': [str(p) for p in plots_created],
                'output_directory': str(self._get_analysis_base_dir())
            },
            'summary': self.get_results_summary()
        }

    def analyze_simple(self):
        """
        Run a simplified analysis that evaluates predictions under three matching
        scenarios: 'raw', 'close', and 'similarity'. For each session and scenario
        create subfolders with plots and CSV summaries.
        """
        logger.info("Starting simple analysis for scenarios: raw, close, similarity")

        # Load results if not already loaded
        if not self.results_data:
            self._load_results()

        if not self.results_data:
            raise ValueError("No results loaded for analysis")

        # Prepare output mapping
        output_dirs = {}

        # For each session, compute metrics per scenario
        scenarios = ['raw', 'close', 'similarity']
        eval_cfg = self.config.get('evaluation', {}) if isinstance(self.config, dict) else {}
        base_jaccard = eval_cfg.get('jaccard_threshold', 0.5)

        for session_id, session_data in self.results_data.items():
            logger.info(f"Simple analysis for session {session_id}")
            session_base = self._get_analysis_base_dir() / f"session_{session_id}"
            session_base.mkdir(parents=True, exist_ok=True)

            # Build flat list of records to iterate
            flat_records = []
            for gap, gap_data in session_data.items():
                if not isinstance(gap_data, list):
                    continue
                for rec in gap_data:
                    flat_records.append((str(gap), rec))

            if not flat_records:
                logger.warning(f"No records for session {session_id}")
                continue

            # Process all records once for all scenarios to minimize memory.
            # This avoids re-reading heavy result data 3x (once per scenario)
            # and allows freeing each record's data after computing all metrics.
            scenario_metrics = {s: {'per_query_rows': [], 'per_gap_agg': {}} for s in scenarios}

            for query_idx, (gap, rec) in enumerate(flat_records):
                recommender = rec.get('recommender_name', 'unknown')
                current = rec.get('current_results') or []
                future = rec.get('future_results') or []
                predicted = rec.get('recommended_results') or []

                # Use actual query position in the session so the x-axis of
                # accuracy_by_query_number reflects true temporal order and
                # can be averaged across gaps meaningfully.
                actual_query_id = rec.get('current_query_id', query_idx)

                for scenario in scenarios:
                    # Compute metrics (accepts lists directly, no DataFrame conversion)
                    metrics = self._compute_metrics_for_scenario(predicted, future, scenario, base_jaccard)

                    # Compute overlap metric
                    overlap = self._compute_overlap_for_scenario(current, future, predicted, scenario, base_jaccard)

                    row = {
                        'session_id': session_id,
                        'gap': gap,
                        'query_number': actual_query_id,
                        'recommender': recommender,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1'],
                        'overlap': overlap
                    }
                    scenario_metrics[scenario]['per_query_rows'].append(row)

                    # Aggregate per gap
                    gap_int = int(gap) if str(gap).isdigit() else -1
                    agg = scenario_metrics[scenario]['per_gap_agg']
                    if gap_int not in agg:
                        agg[gap_int] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'overlap': []}
                    agg[gap_int]['accuracy'].append(metrics['accuracy'])
                    agg[gap_int]['precision'].append(metrics['precision'])
                    agg[gap_int]['recall'].append(metrics['recall'])
                    agg[gap_int]['f1'].append(metrics['f1'])
                    agg[gap_int]['overlap'].append(overlap)

                # Free heavy data from this record to reduce peak memory usage
                rec['current_results'] = None
                rec['future_results'] = None
                rec['recommended_results'] = None

                if query_idx > 0 and query_idx % 50 == 0:
                    gc.collect()
                    logger.debug(f"Processed {query_idx}/{len(flat_records)} records")

            # Force garbage collection after processing all records
            gc.collect()
            logger.info(f"Completed metric computation for session {session_id}")

            # For each scenario, generate outputs (plots and CSVs)
            for scenario in scenarios:
                scenario_dir = session_base / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)

                per_query_rows = scenario_metrics[scenario]['per_query_rows']
                per_gap_agg = scenario_metrics[scenario]['per_gap_agg']

                # Create DataFrame from per-query rows
                pq_df = pd.DataFrame(per_query_rows)
                if pq_df.empty:
                    logger.warning(f"No metric rows for session {session_id} scenario {scenario}")
                    continue

                # Save per-query CSV
                pq_csv = scenario_dir / 'per_query_metrics.csv'
                pq_df.to_csv(pq_csv, index=False)

                # Determine unique recommenders
                unique_recommenders = pq_df['recommender'].unique()
                num_recommenders = len(unique_recommenders)
                show_legend = num_recommenders > 1

                # Plot accuracy by query number
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))

                    # Average accuracy across gaps at each (recommender, query_number)
                    # so the curve shows per-query-position performance rather than
                    # one raw point per (gap, query, recommender) combination.
                    pq_avg = (pq_df
                              .groupby(['recommender', 'query_number'], sort=True)['accuracy']
                              .mean()
                              .reset_index())

                    if num_recommenders == 1:
                        # Single recommender - simple line plot
                        ax.plot(pq_avg['query_number'], pq_avg['accuracy'], marker='o')
                    else:
                        # Multiple recommenders - plot each separately
                        colors = plt.cm.tab10(np.linspace(0, 1, num_recommenders))
                        for idx, recommender in enumerate(unique_recommenders):
                            rec_data = pq_avg[pq_avg['recommender'] == recommender].sort_values('query_number')
                            ax.plot(rec_data['query_number'], rec_data['accuracy'],
                                   marker='o', label=recommender, color=colors[idx])
                    
                    ax.set_title(f'Accuracy by Query Number ({scenario}) - Session {session_id}')
                    ax.set_xlabel('Query Number')
                    ax.set_ylabel('Accuracy')
                    ax.set_ylim(0, 1.05)
                    
                    if show_legend:
                        ax.legend()
                    
                    fig_path = scenario_dir / 'accuracy_by_query_number.png'
                    fig.tight_layout()
                    fig.savefig(fig_path, dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to plot accuracy by query number: {e}")

                # Plot distributions for accuracy/precision/recall/f1
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    
                    if num_recommenders == 1:
                        # Single recommender - simple histograms
                        sns.histplot(pq_df['accuracy'], kde=True, ax=axes[0,0]).set_title('Accuracy')
                        sns.histplot(pq_df['precision'], kde=True, ax=axes[0,1]).set_title('Precision')
                        sns.histplot(pq_df['recall'], kde=True, ax=axes[1,0]).set_title('Recall')
                        sns.histplot(pq_df['f1_score'], kde=True, ax=axes[1,1]).set_title('F1 Score')
                    else:
                        # Multiple recommenders - stacked histograms with hue
                        sns.histplot(data=pq_df, x='accuracy', hue='recommender', kde=True, ax=axes[0,0], legend=show_legend)
                        axes[0,0].set_title('Accuracy')
                        
                        sns.histplot(data=pq_df, x='precision', hue='recommender', kde=True, ax=axes[0,1], legend=show_legend)
                        axes[0,1].set_title('Precision')
                        
                        sns.histplot(data=pq_df, x='recall', hue='recommender', kde=True, ax=axes[1,0], legend=show_legend)
                        axes[1,0].set_title('Recall')
                        
                        sns.histplot(data=pq_df, x='f1_score', hue='recommender', kde=True, ax=axes[1,1], legend=show_legend)
                        axes[1,1].set_title('F1 Score')
                    
                    fig.tight_layout()
                    dist_path = scenario_dir / 'metric_distributions.png'
                    fig.savefig(dist_path, dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to plot metric distributions: {e}")

                # Gap-wise aggregated table and plots (now per recommender)
                gap_rows = []
                if num_recommenders == 1:
                    # Original behavior for single recommender
                    for gap_int, stats in sorted(per_gap_agg.items()):
                        if gap_int < 0:
                            continue
                        gap_row = {
                            'gap': gap_int,
                            'recommender': unique_recommenders[0],
                            'accuracy_mean': np.mean(stats['accuracy']) if stats['accuracy'] else 0.0,
                            'precision_mean': np.mean(stats['precision']) if stats['precision'] else 0.0,
                            'recall_mean': np.mean(stats['recall']) if stats['recall'] else 0.0,
                            'f1_mean': np.mean(stats['f1']) if stats['f1'] else 0.0,
                            'overlap_mean': np.mean(stats['overlap']) if stats['overlap'] else 0.0,
                        }
                        gap_rows.append(gap_row)
                else:
                    # Aggregate per gap AND per recommender
                    for recommender in unique_recommenders:
                        rec_data = pq_df[pq_df['recommender'] == recommender]
                        rec_gap_agg = {}
                        
                        for _, row in rec_data.iterrows():
                            gap_val = int(row['gap']) if str(row['gap']).isdigit() else -1
                            if gap_val < 0:
                                continue
                            if gap_val not in rec_gap_agg:
                                rec_gap_agg[gap_val] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'overlap': []}
                            rec_gap_agg[gap_val]['accuracy'].append(row['accuracy'])
                            rec_gap_agg[gap_val]['precision'].append(row['precision'])
                            rec_gap_agg[gap_val]['recall'].append(row['recall'])
                            rec_gap_agg[gap_val]['f1'].append(row['f1_score'])
                            rec_gap_agg[gap_val]['overlap'].append(row['overlap'])
                        
                        for gap_int, stats in sorted(rec_gap_agg.items()):
                            gap_row = {
                                'gap': gap_int,
                                'recommender': recommender,
                                'accuracy_mean': np.mean(stats['accuracy']) if stats['accuracy'] else 0.0,
                                'precision_mean': np.mean(stats['precision']) if stats['precision'] else 0.0,
                                'recall_mean': np.mean(stats['recall']) if stats['recall'] else 0.0,
                                'f1_mean': np.mean(stats['f1']) if stats['f1'] else 0.0,
                                'overlap_mean': np.mean(stats['overlap']) if stats['overlap'] else 0.0,
                            }
                            gap_rows.append(gap_row)

                gap_df = pd.DataFrame(gap_rows)
                gap_csv = scenario_dir / 'gap_aggregates.csv'
                gap_df.to_csv(gap_csv, index=False)

                # Plot gap metrics
                try:
                    if not gap_df.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        if num_recommenders == 1:
                            # Single recommender - simple line plot
                            ax.plot(gap_df['gap'], gap_df['accuracy_mean'], label='Accuracy', marker='o')
                            ax.plot(gap_df['gap'], gap_df['precision_mean'], label='Precision', marker='o')
                            ax.plot(gap_df['gap'], gap_df['recall_mean'], label='Recall', marker='o')
                            ax.plot(gap_df['gap'], gap_df['f1_mean'], label='F1', marker='o')
                            ax.legend()
                        else:
                            # Multiple recommenders - plot each recommender separately
                            colors = plt.cm.tab10(np.linspace(0, 1, num_recommenders))
                            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
                            
                            # Plot accuracy for each recommender
                            for idx, recommender in enumerate(unique_recommenders):
                                rec_data = gap_df[gap_df['recommender'] == recommender]
                                if not rec_data.empty:
                                    ax.plot(rec_data['gap'], rec_data['accuracy_mean'], 
                                           label=f'{recommender}', 
                                           marker=markers[idx % len(markers)], 
                                           color=colors[idx])
                            
                            if show_legend:
                                ax.legend()
                        
                        ax.set_xlabel('Gap')
                        ax.set_ylabel('Metric Mean')
                        ax.set_title(f'Gap Metrics - Accuracy ({scenario}) - Session {session_id}')
                        ax.set_ylim(0, 1.05)
                        fig.tight_layout()
                        gap_plot = scenario_dir / 'gap_metrics.png'
                        fig.savefig(gap_plot, dpi=150)
                        plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to plot gap metrics: {e}")
                
                # Plot overlap metric
                try:
                    if not pq_df.empty and 'overlap' in pq_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 4))

                        # Average overlap across gaps, same as accuracy plot
                        ov_avg = (pq_df
                                  .groupby(['recommender', 'query_number'], sort=True)['overlap']
                                  .mean()
                                  .reset_index())

                        if num_recommenders == 1:
                            # Single recommender - simple line plot
                            ax.plot(ov_avg['query_number'], ov_avg['overlap'], marker='o', color='purple')
                        else:
                            # Multiple recommenders - plot each separately
                            colors = plt.cm.tab10(np.linspace(0, 1, num_recommenders))
                            for idx, recommender in enumerate(unique_recommenders):
                                rec_data = ov_avg[ov_avg['recommender'] == recommender].sort_values('query_number')
                                ax.plot(rec_data['query_number'], rec_data['overlap'],
                                       marker='o', label=recommender, color=colors[idx])
                        
                        ax.set_title(f'Overlap by Query Number ({scenario}) - Session {session_id}')
                        ax.set_xlabel('Query Number')
                        ax.set_ylabel('Overlap')
                        ax.set_ylim(0, 1.05)
                        
                        if show_legend:
                            ax.legend()
                        
                        fig_path = scenario_dir / 'overlap_by_query_number.png'
                        fig.tight_layout()
                        fig.savefig(fig_path, dpi=150)
                        plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to plot overlap: {e}")

                output_dirs[scenario] = str(scenario_dir)

        # Generate cross-session summaries
        logger.info("Generating cross-session summaries...")
        self._generate_cross_session_summaries(scenarios)

        logger.info("Simple analysis complete")
        return {'output_dirs': output_dirs}
    
    def _generate_cross_session_summaries(self, scenarios: List[str]):
        """
        Generate summary statistics and plots across all sessions for each scenario.
        
        Args:
            scenarios: List of scenario names ('raw', 'close', 'similarity')
        """
        summary_base = self._get_analysis_base_dir() / 'summary'
        summary_base.mkdir(parents=True, exist_ok=True)
        
        for scenario in scenarios:
            logger.info(f"Creating cross-session summary for scenario: {scenario}")
            scenario_summary_dir = summary_base / scenario
            scenario_summary_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect all per-query metrics across sessions
            all_session_metrics = []
            
            for session_id in self.results_data.keys():
                session_dir = self._get_analysis_base_dir() / f"session_{session_id}" / scenario
                metrics_file = session_dir / 'per_query_metrics.csv'
                
                if metrics_file.exists():
                    try:
                        df = pd.read_csv(metrics_file)
                        all_session_metrics.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load metrics from {metrics_file}: {e}")
            
            if not all_session_metrics:
                logger.warning(f"No metrics found for scenario {scenario}")
                continue
            
            # Combine all metrics
            combined_df = pd.concat(all_session_metrics, ignore_index=True)
            
            # Save combined metrics
            combined_csv = scenario_summary_dir / 'all_sessions_metrics.csv'
            combined_df.to_csv(combined_csv, index=False)
            
            # Compute overall statistics
            overall_stats = {
                'scenario': scenario,
                'total_queries': len(combined_df),
                'total_sessions': len(combined_df['session_id'].unique()),
                'accuracy_mean': combined_df['accuracy'].mean(),
                'accuracy_std': combined_df['accuracy'].std(),
                'accuracy_median': combined_df['accuracy'].median(),
                'precision_mean': combined_df['precision'].mean(),
                'precision_std': combined_df['precision'].std(),
                'precision_median': combined_df['precision'].median(),
                'recall_mean': combined_df['recall'].mean(),
                'recall_std': combined_df['recall'].std(),
                'recall_median': combined_df['recall'].median(),
                'f1_mean': combined_df['f1_score'].mean(),
                'f1_std': combined_df['f1_score'].std(),
                'f1_median': combined_df['f1_score'].median(),
            }
            
            if 'overlap' in combined_df.columns:
                overall_stats['overlap_mean'] = combined_df['overlap'].mean()
                overall_stats['overlap_std'] = combined_df['overlap'].std()
                overall_stats['overlap_median'] = combined_df['overlap'].median()
            
            # Save overall statistics
            stats_df = pd.DataFrame([overall_stats])
            stats_csv = scenario_summary_dir / 'overall_statistics.csv'
            stats_df.to_csv(stats_csv, index=False)
            
            # Plot: Distribution of metrics across all sessions
            try:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
                if 'overlap' in combined_df.columns:
                    metrics_to_plot.append('overlap')
                
                for idx, metric in enumerate(metrics_to_plot):
                    if idx < len(axes):
                        ax = axes[idx]
                        sns.histplot(combined_df[metric], kde=True, ax=ax, bins=20)
                        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                        ax.set_xlabel(metric.replace("_", " ").title())
                        ax.set_ylabel('Count')
                        ax.axvline(combined_df[metric].mean(), color='red', linestyle='--', 
                                  label=f'Mean: {combined_df[metric].mean():.3f}')
                        ax.legend()
                
                # Hide unused subplots
                for idx in range(len(metrics_to_plot), len(axes)):
                    axes[idx].set_visible(False)
                
                fig.suptitle(f'Cross-Session Metric Distributions ({scenario})', fontsize=16)
                fig.tight_layout()
                dist_plot = scenario_summary_dir / 'cross_session_distributions.png'
                fig.savefig(dist_plot, dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to create cross-session distributions: {e}")
            
            # Plot: Boxplot by session
            try:
                if len(combined_df['session_id'].unique()) > 1:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    axes = axes.flatten()
                    
                    for idx, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
                        ax = axes[idx]
                        sns.boxplot(data=combined_df, x='session_id', y=metric, ax=ax)
                        ax.set_title(f'{metric.replace("_", " ").title()} by Session')
                        ax.set_xlabel('Session ID')
                        ax.set_ylabel(metric.replace("_", " ").title())
                        ax.tick_params(axis='x', rotation=45)
                    
                    fig.suptitle(f'Metrics by Session ({scenario})', fontsize=16)
                    fig.tight_layout()
                    session_box_plot = scenario_summary_dir / 'metrics_by_session.png'
                    fig.savefig(session_box_plot, dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to create session boxplots: {e}")
            
            # Aggregate by gap across all sessions
            if 'gap' in combined_df.columns:
                try:
                    gap_agg = combined_df.groupby('gap').agg({
                        'accuracy': ['mean', 'std', 'count'],
                        'precision': ['mean', 'std'],
                        'recall': ['mean', 'std'],
                        'f1_score': ['mean', 'std']
                    }).reset_index()
                    
                    gap_agg.columns = ['_'.join(col).strip('_') for col in gap_agg.columns.values]
                    gap_agg_csv = scenario_summary_dir / 'gap_aggregates_all_sessions.csv'
                    gap_agg.to_csv(gap_agg_csv, index=False)
                    
                    # Plot gap metrics across all sessions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.errorbar(gap_agg['gap'], gap_agg['accuracy_mean'], 
                               yerr=gap_agg['accuracy_std'], label='Accuracy', marker='o', capsize=5)
                    ax.errorbar(gap_agg['gap'], gap_agg['precision_mean'], 
                               yerr=gap_agg['precision_std'], label='Precision', marker='s', capsize=5)
                    ax.errorbar(gap_agg['gap'], gap_agg['recall_mean'], 
                               yerr=gap_agg['recall_std'], label='Recall', marker='^', capsize=5)
                    ax.errorbar(gap_agg['gap'], gap_agg['f1_score_mean'], 
                               yerr=gap_agg['f1_score_std'], label='F1', marker='d', capsize=5)
                    
                    ax.set_xlabel('Gap')
                    ax.set_ylabel('Metric Value')
                    ax.set_title(f'Gap Metrics Across All Sessions ({scenario})')
                    ax.set_ylim(0, 1.05)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    gap_plot = scenario_summary_dir / 'gap_metrics_all_sessions.png'
                    fig.savefig(gap_plot, dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to create gap aggregates: {e}")
            
            logger.info(f"Completed cross-session summary for {scenario}")
        
        logger.info("Cross-session summaries complete")

    def _records_match(self, rec_a: dict, rec_b: dict, scenario: str, base_jaccard: float) -> bool:
        """
        Check if two record dicts match under the given scenario.
        
        Args:
            rec_a: First record dict
            rec_b: Second record dict
            scenario: 'raw', 'close', or 'similarity'
            base_jaccard: Jaccard threshold for similarity scenario
        """
        if scenario == 'raw':
            return rec_a.keys() == rec_b.keys() and all(rec_a[k] == rec_b[k] for k in rec_a.keys())
        elif scenario == 'close':
            common_keys = set(rec_a.keys()) & set(rec_b.keys())
            if len(common_keys) < 5:
                return False
            matching = sum(1 for k in common_keys if rec_a.get(k) == rec_b.get(k))
            return matching >= 5
        elif scenario == 'similarity':
            set_a = set(f"{k}={rec_a[k]}" for k in rec_a.keys())
            set_b = set(f"{k}={rec_b[k]}" for k in rec_b.keys())
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            return (inter / union if union > 0 else 0.0) >= base_jaccard
        return False

    def _compute_metrics_for_scenario(self, predicted: list, actual: list, scenario: str, base_jaccard: float) -> Dict[str, float]:
        """
        Compute accuracy/precision/recall/f1 for a single prediction/actual pair under the given scenario.
        
        Args:
            predicted: List of dicts (predicted result rows)
            actual: List of dicts (actual/future result rows)
            scenario: 'raw', 'close', or 'similarity'
            base_jaccard: Jaccard threshold for similarity scenario
        """
        # Handle empty cases
        if not actual and not predicted:
            return {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        if not actual:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        if not predicted:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        true_positives = 0
        matched_pred_indices = set()

        for i, a in enumerate(actual):
            for j, p in enumerate(predicted):
                if j in matched_pred_indices:
                    continue

                if self._records_match(a, p, scenario, base_jaccard):
                    true_positives += 1
                    matched_pred_indices.add(j)
                    break

        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(actual) if actual else 0.0
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        # accuracy defined as proportion of actual tuples matched
        accuracy = true_positives / len(actual) if actual else 0.0

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def _compute_overlap_for_scenario(self, current: list, actual: list, 
                                      predicted: list, scenario: str, base_jaccard: float) -> float:
        """
        Compute overlap metric: proportion of predicted tuples that appear in both
        current and actual results (the overlap region).
        
        Optimized approach: Instead of computing the full inner join of
        current x actual — O(|current| * |actual|) which can be 10^10 — we check
        each predicted tuple (typically ~10) against both current and actual
        separately. This is O(|predicted| * (|current| + |actual|)).
        
        For exact matching (raw), hash sets give O(1) per lookup.
        
        Args:
            current: List of dicts (current query result rows)
            actual: List of dicts (future/actual query result rows)
            predicted: List of dicts (predicted result rows)
            scenario: Matching scenario ('raw', 'close', 'similarity')
            base_jaccard: Jaccard threshold for similarity scenario
        
        Returns:
            Overlap score between 0 and 1
        """
        # Handle empty cases
        if not current or not actual:
            return 0.0
        if not predicted:
            return 0.0
        
        if scenario == 'raw':
            # Build hash sets for O(1) membership testing
            def _row_key(rec):
                return frozenset((str(k), str(v)) for k, v in rec.items())
            
            current_set = set(_row_key(r) for r in current)
            actual_set = set(_row_key(r) for r in actual)
            
            matched_count = 0
            for p in predicted:
                p_key = _row_key(p)
                if p_key in current_set and p_key in actual_set:
                    matched_count += 1
        else:
            # For close/similarity: check each predicted against current AND actual.
            # O(|predicted| * (|current| + |actual|)) instead of O(|current| * |actual|)
            matched_count = 0
            for p_rec in predicted:
                in_current = any(
                    self._records_match(p_rec, c_rec, scenario, base_jaccard)
                    for c_rec in current
                )
                if in_current:
                    in_actual = any(
                        self._records_match(p_rec, a_rec, scenario, base_jaccard)
                        for a_rec in actual
                    )
                    if in_actual:
                        matched_count += 1
        
        return matched_count / len(predicted) if predicted else 0.0
    
    def _load_results(self):
        """Load all JSON result files from the results directory."""
        logger.info("Loading experiment results...")
        
        result_files = list(self.results_dir.glob("*.json"))
        if not result_files:
            logger.warning(f"No JSON result files found in: {self.results_dir}")
            return
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract session and gap from filename
                # Supported formats:
                #  - {session_id}__gap-{gap_value}.json (current)
                #  - {session_id}_gap-{gap_value}.json  (legacy)
                filename = file_path.stem
                session_id: Optional[str] = None
                gap: Optional[str] = None
                if '__gap-' in filename:
                    session_id, gap_part = filename.split('__gap-', 1)
                    gap = gap_part
                elif '_gap-' in filename:
                    session_id, gap_part = filename.split('_gap-', 1)
                    gap = gap_part
                else:
                    logger.warning(f"Unexpected filename format: {file_path.name}")
                
                if session_id is not None and gap is not None:
                    if session_id not in self.results_data:
                        self.results_data[session_id] = {}
                    self.results_data[session_id][gap] = data
                    logger.debug(f"Loaded {len(data)} records from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        logger.info(f"Loaded results for {len(self.results_data)} sessions")
    
    def _log_experiment_summary(self):
        """Log a summary of the experiment results."""
        logger.info("=== EXPERIMENT ANALYSIS SUMMARY ===")
        
        experiment_name = self.results_dir.name
        logger.info(f"Experiment: {experiment_name}")
        
        # Count sessions and gaps
        all_sessions = list(self.results_data.keys())
        all_gaps = set()
        all_recommenders = set()
        total_records = 0
        
        for session_id, session_data in self.results_data.items():
            gaps_for_session = list(session_data.keys())
            all_gaps.update(gaps_for_session)
            
            for gap, gap_data in session_data.items():
                total_records += len(gap_data)
                
                # Extract recommender names
                for record in gap_data:
                    if 'recommender_name' in record:
                        all_recommenders.add(record['recommender_name'])
        
        logger.info(f"Sessions analyzed: {sorted(all_sessions)}")
        logger.info(f"Gaps analyzed: {sorted(all_gaps)}")
        logger.info(f"Recommenders found: {sorted(all_recommenders)}")
        logger.info(f"Total prediction records: {total_records}")
        
        # Log per-session statistics
        for session_id in sorted(all_sessions):
            session_data = self.results_data[session_id]
            gaps_in_session = sorted(session_data.keys())
            records_in_session = sum(len(session_data[gap]) for gap in gaps_in_session)
            logger.info(f"  Session {session_id}: {len(gaps_in_session)} gaps, {records_in_session} records")
        
        # Log errors if any
        error_count = 0
        timeout_count = 0
        
        for session_id, session_data in self.results_data.items():
            for gap, gap_data in session_data.items():
                for record in gap_data:
                    if record.get('error_message'):
                        error_count += 1
                        if 'Timeout' in record['error_message']:
                            timeout_count += 1
        
        if error_count > 0:
            logger.warning(f"Found {error_count} errors in results ({timeout_count} timeouts)")
        
        logger.info("=== END EXPERIMENT ANALYSIS ===")
    
    def _get_analysis_base_dir(self) -> Path:
        """Return the base directory for analysis outputs (create if missing)."""
        analysis_dir = self.results_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        return analysis_dir
    
    def _generate_accuracy_plots(self) -> List[Path]:
        """Create per-session bar plots of average accuracy per recommender.
        Returns a list of paths to the created plot files.
        """
        plots: List[Path] = []
        if not self.results_data:
            return plots
        
        # Configure plotting style once
        sns.set_theme(style="whitegrid")
        
        # Build a long-form table of accuracies: one row per record
        eval_cfg = self.config.get('evaluation', {}) if isinstance(self.config, dict) else {}
        metrics = EvaluationMetrics(
            jaccard_threshold=eval_cfg.get('jaccard_threshold', 0.5),
            column_weights=eval_cfg.get('column_weights')
        )
        
        # Iterate sessions
        for session_id, session_data in self.results_data.items():
            records: List[Dict[str, Any]] = []
            # Flatten all records across all gaps for this session
            gaps_items = [(gap, gap_data) for gap, gap_data in session_data.items() if isinstance(gap_data, list)]
            for gap, gap_data in gaps_items:
                for rec in gap_data:
                    recommender = rec.get('recommender_name', 'unknown')
                    future = rec.get('future_results') or []
                    predicted = rec.get('recommended_results') or []
                    # Convert to DataFrames safely
                    future_df = pd.DataFrame(future) if isinstance(future, list) else pd.DataFrame()
                    pred_df = pd.DataFrame(predicted) if isinstance(predicted, list) else pd.DataFrame()
                    try:
                        acc = metrics.accuracy(pred_df, future_df)
                    except Exception as e:
                        logger.debug(f"Failed to compute accuracy for {session_id} gap {gap} {recommender}: {e}")
                        acc = 0.0
                    records.append({
                        'session_id': session_id,
                        'gap': str(gap),
                        'recommender': recommender,
                        'accuracy': acc,
                        'k': int(len(pred_df)) if pred_df is not None else 0,
                    })
            
            if not records:
                logger.info(f"No records to plot for session {session_id}")
                continue
            
            df = pd.DataFrame(records)
            # Average accuracy and average recommended size per recommender across the session (all gaps/pairs)
            summary = (
                df.groupby('recommender', as_index=False)
                  .agg(accuracy=('accuracy', 'mean'), mean_k=('k', 'mean'))
                  .sort_values('accuracy', ascending=False)
            )
            
            # Prepare output directory per session
            session_out_dir = self._get_analysis_base_dir() / f"session_{session_id}"
            session_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CSV summary (small, low-risk extra)
            try:
                summary_csv = session_out_dir / 'accuracy_summary.csv'
                summary.to_csv(summary_csv, index=False)
            except Exception as e:
                logger.debug(f"Failed to write CSV summary for session {session_id}: {e}")
            
            # Create bar plot
            plt.figure(figsize=(10, 5))
            ax = sns.barplot(data=summary, x='recommender', y='accuracy', palette='Blues_d')
            ax.set_title(f"Average Accuracy per Recommender (Session {session_id})")
            ax.set_xlabel("Recommender")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            # Annotate bars with accuracy and avg k values
            labels = [f"{row.accuracy:.2f}\n(n={int(round(row.mean_k))})" for row in summary.itertuples(index=False)]
            for patch, label in zip(ax.patches, labels):
                height = patch.get_height()
                ax.annotate(
                    label,
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=9,
                    xytext=(0, 3), textcoords='offset points'
                )
            
            # Save plot
            plot_path = session_out_dir / 'accuracy_per_recommender.png'
            try:
                plt.savefig(plot_path, dpi=150)
                plots.append(plot_path)
            except Exception as e:
                logger.error(f"Failed to save plot for session {session_id}: {e}")
            finally:
                plt.close()
        
        return plots
    
    def _generate_all_visualizations(self) -> List[Path]:
        """Generate all comprehensive visualizations per session."""
        all_plots: List[Path] = []
        if not self.results_data:
            return all_plots
        
        # Set publication style
        self._set_publication_style()
        
        # Build enriched dataframe for each session
        for session_id, session_data in self.results_data.items():
            logger.info(f"Generating visualizations for session {session_id}")
            
            # Prepare output directory per session
            session_out_dir = self._get_analysis_base_dir() / f"session_{session_id}"
            session_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Build DataFrame for this session
            session_df = self._build_session_dataframe(session_id, session_data)
            if session_df.empty:
                logger.warning(f"No data to plot for session {session_id}")
                continue
            
            # Generate all plot types
            plot_methods = [
                ("accuracy_comparison", self._create_accuracy_comparison),
                ("overlap_accuracy_comparison", self._create_overlap_accuracy_comparison),
                ("accuracy_by_query_number", self._create_accuracy_by_query_number),
                ("overlap_accuracy_by_query_number", self._create_overlap_accuracy_by_query_number),
                ("precision_comparison", self._create_precision_comparison),
                ("recall_comparison", self._create_recall_comparison),
                ("f1_comparison", self._create_f1_comparison),
                ("gap_analysis_accuracy", self._create_gap_analysis_accuracy),
                ("gap_analysis_precision", self._create_gap_analysis_precision),
                ("gap_analysis_recall", self._create_gap_analysis_recall),
                ("gap_analysis_f1", self._create_gap_analysis_f1),
                ("result_size_analysis", self._create_result_size_analysis),
                ("execution_time_analysis", self._create_execution_time_analysis),
                ("performance_heatmap", self._create_performance_heatmap),
                ("distribution_analysis", self._create_distribution_analysis),
                ("correlation_analysis", self._create_correlation_analysis),
                ("tuple_count_analysis", self._create_tuple_count_analysis),
                ("normalized_performance_comparison", self._create_normalized_performance_comparison),
                ("normalized_gap_analysis", self._create_normalized_gap_analysis)
            ]
            
            session_plots = []
            for plot_name, plot_method in plot_methods:
                try:
                    fig = plot_method(session_df, session_id)
                    if fig is not None:
                        plot_path = session_out_dir / f'{plot_name}.png'
                        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                        session_plots.append(plot_path)
                        plt.close(fig)
                        logger.debug(f"Created {plot_name} for session {session_id}")
                except Exception as e:
                    logger.error(f"Error creating {plot_name} for session {session_id}: {e}")
                    continue
            
            # Create combined PDF for the session
            if session_plots:
                try:
                    pdf_path = session_out_dir / f'all_plots_{session_id}.pdf'
                    with PdfPages(pdf_path) as pdf:
                        for plot_name, plot_method in plot_methods:
                            try:
                                fig = plot_method(session_df, session_id)
                                if fig is not None:
                                    pdf.savefig(fig, bbox_inches='tight')
                                    plt.close(fig)
                            except Exception as e:
                                logger.debug(f"Error adding {plot_name} to PDF: {e}")
                                continue
                    
                    session_plots.append(pdf_path)
                    logger.info(f"Created combined PDF with {len(plot_methods)} plots for session {session_id}")
                except Exception as e:
                    logger.error(f"Error creating combined PDF for session {session_id}: {e}")
            
            # Save CSV summary
            try:
                summary_csv = session_out_dir / 'session_data_summary.csv'
                session_df.to_csv(summary_csv, index=False)
                session_plots.append(summary_csv)
            except Exception as e:
                logger.debug(f"Failed to write CSV summary for session {session_id}: {e}")
            
            all_plots.extend(session_plots)
        
        return all_plots
    
    def _build_session_dataframe(self, session_id: str, session_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Build enriched DataFrame for a session."""
        eval_cfg = self.config.get('evaluation', {}) if isinstance(self.config, dict) else {}
        metrics = EvaluationMetrics(
            jaccard_threshold=eval_cfg.get('jaccard_threshold', 0.5),
            column_weights=eval_cfg.get('column_weights')
        )
        
        records: List[Dict[str, Any]] = []
        
        # Flatten all records across all gaps for this session
        for gap, gap_data in session_data.items():
            if not isinstance(gap_data, list):
                continue
                
            for rec in gap_data:
                recommender = rec.get('recommender_name', 'unknown')
                future = rec.get('future_results') or []
                predicted = rec.get('recommended_results') or []
                previous = rec.get('previous_results') or []  # Add previous results for overlap accuracy
                execution_time = rec.get('execution_time', 0.0)
                
                # Convert to DataFrames safely
                future_df = pd.DataFrame(future) if isinstance(future, list) else pd.DataFrame()
                pred_df = pd.DataFrame(predicted) if isinstance(predicted, list) else pd.DataFrame()
                previous_df = pd.DataFrame(previous) if isinstance(previous, list) else pd.DataFrame()
                
                # Calculate comprehensive metrics
                try:
                    accuracy = metrics.accuracy(pred_df, future_df)
                    precision = metrics.precision(pred_df, future_df)
                    recall = metrics.recall(pred_df, future_df)
                    f1_score = metrics.f1_score(pred_df, future_df)
                    jaccard = metrics.jaccard_similarity(pred_df, future_df)
                    
                    # Calculate overlap accuracy
                    overlap_accuracy = metrics.overlap_accuracy(previous_df, future_df, pred_df)
                    
                    # Count metrics for additional analysis
                    pred_count = len(pred_df)
                    actual_count = len(future_df)
                    intersection_count = len(pd.merge(pred_df, future_df, how='inner')) if not pred_df.empty and not future_df.empty else 0
                    union_count = len(pd.concat([pred_df, future_df]).drop_duplicates()) if not pred_df.empty or not future_df.empty else 0
                    
                    # Calculate sensitivity and specificity
                    sensitivity = recall  # Same as recall
                    # Specificity calculation (approximate for recommendation context)
                    total_possible = max(union_count, actual_count + pred_count - intersection_count)
                    true_negatives = max(0, total_possible - union_count)
                    false_positives = max(0, pred_count - intersection_count)
                    specificity = true_negatives / max(1, true_negatives + false_positives)
                    
                    # ROC-AUC approximation (simplified for this context)
                    # In a recommendation context, this is an approximation
                    roc_auc = (sensitivity + specificity) / 2 if (sensitivity + specificity) > 0 else 0.0
                    
                except Exception as e:
                    logger.debug(f"Failed to compute metrics for {session_id} gap {gap} {recommender}: {e}")
                    accuracy = precision = recall = f1_score = jaccard = overlap_accuracy = 0.0
                    pred_count = actual_count = intersection_count = union_count = 0
                    sensitivity = specificity = roc_auc = 0.0
                
                record = {
                    'session_id': session_id,
                    'gap': str(gap),
                    'recommender': recommender,
                    'query_number': rec.get('current_query_id', 0),  # Use actual query ID from record
                    'accuracy': accuracy,
                    'overlap_accuracy': overlap_accuracy,  # Add overlap accuracy
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'jaccard_similarity': jaccard,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'roc_auc': roc_auc,
                    'predicted_count': pred_count,
                    'actual_count': actual_count,
                    'intersection_count': intersection_count,
                    'union_count': union_count,
                    'execution_time': execution_time,
                    'error_message': rec.get('error_message', ''),
                    'has_error': bool(rec.get('error_message', '')),
                }
                
                records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Add derived columns
        df['gap_int'] = pd.to_numeric(df['gap'], errors='coerce')
        df['result_size_category'] = pd.cut(df['predicted_count'], bins=[0, 10, 50, 100, float('inf')], 
                                          labels=['Small (≤10)', 'Medium (11-50)', 'Large (51-100)', 'Very Large (>100)'])
        df['execution_time_category'] = pd.cut(df['execution_time'], bins=[0, 1, 5, 30, float('inf')], 
                                             labels=['Fast (≤1s)', 'Medium (1-5s)', 'Slow (5-30s)', 'Very Slow (>30s)'])
        
        return df
    
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
            'text.usetex': False,
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
    
    def _create_accuracy_comparison(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create accuracy comparison box plot."""
        if 'accuracy' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of accuracy by recommender
        sns.boxplot(data=df, x='recommender', y='accuracy', ax=ax)
        
        ax.set_title(f'Accuracy Distribution by Recommender (Session {session_id})')
        ax.set_xlabel('Recommender')
        ax.set_ylabel('Accuracy')
        # Dynamic y-axis scaling based on data
        max_val = df['accuracy'].max()
        min_val = df['accuracy'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        plt.xticks(rotation=45, ha='right')
        
        # Add mean points
        means = df.groupby('recommender')['accuracy'].mean()
        for i, (recommender, mean_val) in enumerate(means.items()):
            ax.plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
        
        if len(means) > 0:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_precision_comparison(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create precision comparison box plot."""
        if 'precision' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of precision by recommender
        sns.boxplot(data=df, x='recommender', y='precision', ax=ax)
        
        ax.set_title(f'Precision Distribution by Recommender (Session {session_id})')
        ax.set_xlabel('Recommender')
        ax.set_ylabel('Precision')
        # Dynamic y-axis scaling based on data
        max_val = df['precision'].max()
        min_val = df['precision'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        plt.xticks(rotation=45, ha='right')
        
        # Add mean points
        means = df.groupby('recommender')['precision'].mean()
        for i, (recommender, mean_val) in enumerate(means.items()):
            ax.plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
        
        if len(means) > 0:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_recall_comparison(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create recall comparison box plot."""
        if 'recall' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of recall by recommender
        sns.boxplot(data=df, x='recommender', y='recall', ax=ax)
        
        ax.set_title(f'Recall Distribution by Recommender (Session {session_id})')
        ax.set_xlabel('Recommender')
        ax.set_ylabel('Recall')
        # Dynamic y-axis scaling based on data
        max_val = df['recall'].max()
        min_val = df['recall'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        plt.xticks(rotation=45, ha='right')
        
        # Add mean points
        means = df.groupby('recommender')['recall'].mean()
        for i, (recommender, mean_val) in enumerate(means.items()):
            ax.plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
        
        if len(means) > 0:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_f1_comparison(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create F1 score comparison box plot."""
        if 'f1_score' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of F1 by recommender
        sns.boxplot(data=df, x='recommender', y='f1_score', ax=ax)
        
        ax.set_title(f'F1 Score Distribution by Recommender (Session {session_id})')
        ax.set_xlabel('Recommender')
        ax.set_ylabel('F1 Score')
        # Dynamic y-axis scaling based on data
        max_val = df['f1_score'].max()
        min_val = df['f1_score'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        plt.xticks(rotation=45, ha='right')
        
        # Add mean points
        means = df.groupby('recommender')['f1_score'].mean()
        for i, (recommender, mean_val) in enumerate(means.items()):
            ax.plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
        
        if len(means) > 0:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_overlap_accuracy_comparison(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create overlap accuracy comparison box plot."""
        if 'overlap_accuracy' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of overlap accuracy by recommender
        sns.boxplot(data=df, x='recommender', y='overlap_accuracy', ax=ax)
        
        ax.set_title(f'Overlap Accuracy Distribution by Recommender (Session {session_id})')
        ax.set_xlabel('Recommender')
        ax.set_ylabel('Overlap Accuracy')
        # Dynamic y-axis scaling based on data
        max_val = df['overlap_accuracy'].max()
        min_val = df['overlap_accuracy'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        plt.xticks(rotation=45, ha='right')
        
        # Add mean points
        means = df.groupby('recommender')['overlap_accuracy'].mean()
        for i, (recommender, mean_val) in enumerate(means.items()):
            ax.plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
        
        if len(means) > 0:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_accuracy_by_query_number(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create accuracy by query number line plot."""
        if 'accuracy' not in df.columns or 'query_number' not in df.columns or 'recommender' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Line plot of accuracy by query number for each recommender
        recommenders = df['recommender'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            rec_data = df[df['recommender'] == recommender].sort_values('query_number')
            if not rec_data.empty:
                ax.plot(rec_data['query_number'], rec_data['accuracy'], 
                       label=recommender, color=colors[i], marker='o', markersize=4, linewidth=2)
        
        ax.set_title(f'Accuracy by Query Number (Session {session_id})')
        ax.set_xlabel('Query Number')
        ax.set_ylabel('Accuracy')
        # Dynamic y-axis scaling
        max_val = df['accuracy'].max()
        min_val = df['accuracy'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_overlap_accuracy_by_query_number(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create overlap accuracy by query number line plot."""
        if 'overlap_accuracy' not in df.columns or 'query_number' not in df.columns or 'recommender' not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Line plot of overlap accuracy by query number for each recommender
        recommenders = df['recommender'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            rec_data = df[df['recommender'] == recommender].sort_values('query_number')
            if not rec_data.empty:
                ax.plot(rec_data['query_number'], rec_data['overlap_accuracy'], 
                       label=recommender, color=colors[i], marker='o', markersize=4, linewidth=2)
        
        ax.set_title(f'Overlap Accuracy by Query Number (Session {session_id})')
        ax.set_xlabel('Query Number')
        ax.set_ylabel('Overlap Accuracy')
        # Dynamic y-axis scaling
        max_val = df['overlap_accuracy'].max()
        min_val = df['overlap_accuracy'].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_gap_analysis_accuracy(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create accuracy vs gap analysis."""
        return self._create_gap_analysis_generic(df, session_id, 'accuracy', 'Accuracy', figsize)
    
    def _create_gap_analysis_precision(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create precision vs gap analysis."""
        return self._create_gap_analysis_generic(df, session_id, 'precision', 'Precision', figsize)
    
    def _create_gap_analysis_recall(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create recall vs gap analysis."""
        return self._create_gap_analysis_generic(df, session_id, 'recall', 'Recall', figsize)
    
    def _create_gap_analysis_f1(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create F1 vs gap analysis."""
        return self._create_gap_analysis_generic(df, session_id, 'f1_score', 'F1 Score', figsize)
    
    def _create_gap_analysis_generic(self, df: pd.DataFrame, session_id: str, metric_col: str, metric_name: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create generic gap analysis plot."""
        if metric_col not in df.columns or 'gap_int' not in df.columns:
            return None
        
        # Filter out invalid gaps
        valid_df = df.dropna(subset=['gap_int'])
        if valid_df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with trend lines
        recommenders = valid_df['recommender'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            rec_data = valid_df[valid_df['recommender'] == recommender]
            if rec_data.empty:
                continue
                
            # Scatter plot
            ax.scatter(rec_data['gap_int'], rec_data[metric_col], 
                      label=recommender, alpha=0.7, color=colors[i], s=50)
            
            # Trend line
            if len(rec_data) > 1:
                try:
                    z = np.polyfit(rec_data['gap_int'], rec_data[metric_col], 1)
                    p = np.poly1d(z)
                    gap_range = np.linspace(rec_data['gap_int'].min(), rec_data['gap_int'].max(), 100)
                    ax.plot(gap_range, p(gap_range), '--', color=colors[i], alpha=0.8)
                except:
                    pass
        
        ax.set_title(f'{metric_name} vs Gap Analysis (Session {session_id})')
        ax.set_xlabel('Gap')
        ax.set_ylabel(metric_name)
        # Dynamic y-axis scaling based on data
        max_val = valid_df[metric_col].max()
        min_val = valid_df[metric_col].min()
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def _create_result_size_analysis(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create result size analysis."""
        if 'result_size_category' not in df.columns or 'accuracy' not in df.columns:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Distribution of result sizes
        size_counts = df['result_size_category'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(size_counts)))
        wedges, texts, autotexts = ax1.pie(size_counts.values, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title(f'Distribution of Result Sizes (Session {session_id})')
        # Add legend in separate box
        ax1.legend(wedges, size_counts.index, title="Result Size Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Performance by result size
        sns.boxplot(data=df, x='result_size_category', y='accuracy', ax=ax2)
        ax2.set_title(f'Accuracy by Result Size (Session {session_id})')
        ax2.set_xlabel('Result Size Category')
        ax2.set_ylabel('Accuracy')
        # Dynamic y-axis scaling
        max_val = df['accuracy'].max()
        min_val = df['accuracy'].min()
        margin = (max_val - min_val) * 0.1
        ax2.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _create_execution_time_analysis(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create execution time analysis."""
        if 'execution_time' not in df.columns:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Distribution of execution times
        df['execution_time'].hist(bins=20, ax=ax1, alpha=0.7)
        ax1.set_title(f'Execution Time Distribution (Session {session_id})')
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_ylabel('Frequency')
        
        # Box plot by recommender
        if 'recommender' in df.columns:
            sns.boxplot(data=df, x='recommender', y='execution_time', ax=ax2)
            ax2.set_title('Execution Time by Recommender')
            ax2.set_xlabel('Recommender')
            ax2.set_ylabel('Execution Time (seconds)')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Performance vs execution time
        if 'accuracy' in df.columns:
            ax3.scatter(df['execution_time'], df['accuracy'], alpha=0.6)
            ax3.set_title('Accuracy vs Execution Time')
            ax3.set_xlabel('Execution Time (seconds)')
            ax3.set_ylabel('Accuracy')
            # Dynamic y-axis scaling
            max_val = df['accuracy'].max()
            min_val = df['accuracy'].min()
            margin = (max_val - min_val) * 0.1
            ax3.set_ylim(max(0, min_val - margin), min(1.05, max_val + margin))
        
        # Time by category
        if 'execution_time_category' in df.columns:
            time_counts = df['execution_time_category'].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(time_counts)))
            wedges, texts, autotexts = ax4.pie(time_counts.values, autopct='%1.1f%%', startangle=90, colors=colors)
            ax4.set_title('Execution Time Categories')
            # Add legend in separate box
            ax4.legend(wedges, time_counts.index, title="Time Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        return fig
    
    def _create_performance_heatmap(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create performance heatmap."""
        if not {'recommender', 'gap_int', 'accuracy'}.issubset(df.columns):
            return None
        
        # Filter out invalid gaps
        valid_df = df.dropna(subset=['gap_int'])
        if valid_df.empty:
            return None
            
        # Create pivot table for heatmap
        pivot_data = valid_df.pivot_table(values='accuracy', index='recommender', columns='gap_int', aggfunc='mean')
        
        if pivot_data.empty:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.2f', ax=ax, cbar_kws={'label': 'Accuracy'})
        
        ax.set_title(f'Performance Heatmap: Accuracy by Recommender and Gap (Session {session_id})')
        ax.set_xlabel('Gap')
        ax.set_ylabel('Recommender')
        
        plt.tight_layout()
        return fig
    
    def _create_distribution_analysis(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create distribution analysis of key metrics."""
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [col for col in metrics_cols if col in df.columns]
        
        if not available_metrics:
            return None
            
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Histogram and KDE
            df[metric].hist(bins=20, ax=ax, alpha=0.7, density=True, label='Histogram')
            
            # KDE overlay
            try:
                df[metric].plot.density(ax=ax, alpha=0.8, label='KDE')
            except:
                pass
            
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel('Density')
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Metric Distributions (Session {session_id})')
        plt.tight_layout()
        return fig
    
    def _create_correlation_analysis(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create correlation analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if col not in ['session_id'] and not col.endswith('_int')]
        
        if len(correlation_cols) < 2:
            return None
            
        corr_matrix = df[correlation_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create correlation heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', ax=ax, square=True, cbar_kws={'label': 'Correlation'})
        
        ax.set_title(f'Metric Correlation Matrix (Session {session_id})')
        
        plt.tight_layout()
        return fig
    
    def _create_tuple_count_analysis(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create tuple count analysis."""
        count_cols = ['predicted_count', 'actual_count', 'intersection_count', 'union_count']
        available_counts = [col for col in count_cols if col in df.columns]
        
        if not available_counts:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Predicted vs Actual counts
        if 'predicted_count' in df.columns and 'actual_count' in df.columns:
            ax1.scatter(df['actual_count'], df['predicted_count'], alpha=0.6)
            ax1.plot([0, df['actual_count'].max()], [0, df['actual_count'].max()], 'r--', alpha=0.8)
            ax1.set_xlabel('Actual Count')
            ax1.set_ylabel('Predicted Count')
            ax1.set_title('Predicted vs Actual Tuple Counts')
        
        # Distribution of predicted counts by recommender
        if 'predicted_count' in df.columns and 'recommender' in df.columns:
            sns.boxplot(data=df, x='recommender', y='predicted_count', ax=ax2)
            ax2.set_title('Predicted Count by Recommender')
            ax2.set_ylabel('Predicted Count')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Intersection vs Union
        if 'intersection_count' in df.columns and 'union_count' in df.columns:
            ax3.scatter(df['union_count'], df['intersection_count'], alpha=0.6)
            ax3.set_xlabel('Union Count')
            ax3.set_ylabel('Intersection Count')
            ax3.set_title('Intersection vs Union Counts')
        
        # Count distributions
        if available_counts:
            df[available_counts].boxplot(ax=ax4)
            ax4.set_title('Count Distributions')
            ax4.set_ylabel('Count')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(f'Tuple Count Analysis (Session {session_id})')
        plt.tight_layout()
        return fig
    
    def _create_normalized_performance_comparison(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create normalized performance comparison."""
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [col for col in metrics_cols if col in df.columns and 'recommender' in df.columns]
        
        if not available_metrics:
            return None
            
        # Normalize metrics per recommender (z-score normalization)
        normalized_data = []
        
        for recommender in df['recommender'].unique():
            rec_data = df[df['recommender'] == recommender]
            
            rec_normalized = {
                'recommender': recommender,
            }
            
            for metric in available_metrics:
                metric_values = rec_data[metric]
                if len(metric_values) > 1:
                    normalized = (metric_values - metric_values.mean()) / (metric_values.std() + 1e-8)
                    rec_normalized[f'{metric}_normalized'] = normalized.mean()
                else:
                    rec_normalized[f'{metric}_normalized'] = 0.0
            
            normalized_data.append(rec_normalized)
        
        if not normalized_data:
            return None
            
        norm_df = pd.DataFrame(normalized_data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create radar-like comparison
        x_pos = np.arange(len(norm_df))
        width = 0.2
        
        for i, metric in enumerate(available_metrics):
            normalized_col = f'{metric}_normalized'
            if normalized_col in norm_df.columns:
                ax.bar(x_pos + i * width, norm_df[normalized_col], width, 
                       label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Recommender')
        ax.set_ylabel('Normalized Performance (Z-Score)')
        ax.set_title(f'Normalized Performance Comparison (Session {session_id})')
        ax.set_xticks(x_pos + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(norm_df['recommender'], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_normalized_gap_analysis(self, df: pd.DataFrame, session_id: str, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create normalized gap analysis."""
        if not {'gap_int', 'recommender', 'accuracy'}.issubset(df.columns):
            return None
            
        # Filter out invalid gaps
        valid_df = df.dropna(subset=['gap_int'])
        if valid_df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize accuracy within each gap
        normalized_data = []
        
        for gap in valid_df['gap_int'].unique():
            gap_data = valid_df[valid_df['gap_int'] == gap]
            gap_mean = gap_data['accuracy'].mean()
            gap_std = gap_data['accuracy'].std()
            
            if gap_std > 0:
                gap_data = gap_data.copy()
                gap_data['accuracy_normalized'] = (gap_data['accuracy'] - gap_mean) / gap_std
                normalized_data.append(gap_data)
        
        if not normalized_data:
            return None
            
        norm_df = pd.concat(normalized_data, ignore_index=True)
        
        # Plot normalized performance by gap
        recommenders = norm_df['recommender'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            rec_data = norm_df[norm_df['recommender'] == recommender]
            if not rec_data.empty:
                ax.scatter(rec_data['gap_int'], rec_data['accuracy_normalized'], 
                          label=recommender, alpha=0.7, color=colors[i], s=50)
                
                # Trend line
                if len(rec_data) > 1:
                    try:
                        z = np.polyfit(rec_data['gap_int'], rec_data['accuracy_normalized'], 1)
                        p = np.poly1d(z)
                        gap_range = np.linspace(rec_data['gap_int'].min(), rec_data['gap_int'].max(), 100)
                        ax.plot(gap_range, p(gap_range), '--', color=colors[i], alpha=0.8)
                    except:
                        pass
        
        ax.set_title(f'Normalized Accuracy vs Gap Analysis (Session {session_id})')
        ax.set_xlabel('Gap')
        ax.set_ylabel('Normalized Accuracy (Z-Score)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a structured summary of the results data.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results_data:
            return {}
        
        all_sessions = list(self.results_data.keys())
        all_gaps = set()
        all_recommenders = set()
        total_records = 0
        error_count = 0
        
        for session_id, session_data in self.results_data.items():
            all_gaps.update(session_data.keys())
            
            for gap, gap_data in session_data.items():
                total_records += len(gap_data)
                
                for record in gap_data:
                    if 'recommender_name' in record:
                        all_recommenders.add(record['recommender_name'])
                    if record.get('error_message'):
                        error_count += 1
        
        # Available visualization types
        visualization_types = [
            'accuracy_comparison',
            'overlap_accuracy_comparison',
            'accuracy_by_query_number',
            'overlap_accuracy_by_query_number',
            'precision_comparison', 
            'recall_comparison',
            'f1_comparison',
            'gap_analysis_accuracy',
            'gap_analysis_precision',
            'gap_analysis_recall', 
            'gap_analysis_f1',
            'result_size_analysis',
            'execution_time_analysis',
            'performance_heatmap',
            'distribution_analysis',
            'correlation_analysis',
            'tuple_count_analysis',
            'normalized_performance_comparison',
            'normalized_gap_analysis'
        ]
        
        return {
            'experiment_name': self.results_dir.name,
            'sessions': sorted(all_sessions),
            'gaps': sorted(all_gaps),
            'recommenders': sorted(all_recommenders),
            'total_records': total_records,
            'error_count': error_count,
            'sessions_count': len(all_sessions),
            'gaps_count': len(all_gaps),
            'recommenders_count': len(all_recommenders),
            'analysis_capabilities': {
                'comprehensive_visualizations': True,
                'per_session_analysis': True,
                'multi_metric_support': True,
                'visualization_types': visualization_types,
                'output_formats': ['PNG', 'PDF', 'CSV'],
                'metrics_analyzed': [
                    'accuracy', 'overlap_accuracy', 'precision', 'recall', 'f1_score',
                    'jaccard_similarity', 'sensitivity', 'specificity', 'roc_auc'
                ]
            }
        }
    