"""
Analysis module for query prediction experiment results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

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
        
        # Generate session-based plots
        self._generate_session_plots()
        
        logger.info("Analysis completed")
    
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
                # Expected format: {session_id}__gap-{gap_value}.json
                filename = file_path.stem
                if '__gap-' in filename:
                    session_id, gap_part = filename.split('__gap-', 1)
                    gap = gap_part
                    
                    if session_id not in self.results_data:
                        self.results_data[session_id] = {}
                    
                    self.results_data[session_id][gap] = data
                    logger.debug(f"Loaded {len(data)} records from {file_path.name}")
                else:
                    logger.warning(f"Unexpected filename format: {file_path.name}")
                    
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
        
        return {
            'experiment_name': self.results_dir.name,
            'sessions': sorted(all_sessions),
            'gaps': sorted(all_gaps),
            'recommenders': sorted(all_recommenders),
            'total_records': total_records,
            'error_count': error_count,
            'sessions_count': len(all_sessions),
            'gaps_count': len(all_gaps),
            'recommenders_count': len(all_recommenders)
        }
    
    def _generate_session_plots(self):
        """Generate plots organized by session."""
        if not self.results_data:
            logger.warning("No results data available for plotting")
            return
            
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/numpy not available, skipping plot generation")
            return
        
        # Create analysis base directory
        analysis_dir = self.results_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating session-based plots in: {analysis_dir}")
        
        # Generate plots for each session
        for session_id in self.results_data.keys():
            self._generate_session_accuracy_plot(session_id, analysis_dir)
    
    def _generate_session_accuracy_plot(self, session_id: str, base_analysis_dir: Path):
        """
        Generate accuracy bar chart for a specific session showing all recommenders across gaps.
        
        Args:
            session_id: Session to generate plot for
            base_analysis_dir: Base analysis directory
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot generation")
            return
            
        # Create session subdirectory
        session_dir = base_analysis_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_data = self.results_data[session_id]
        
        # Extract accuracy data for this session
        accuracy_data = {}  # {recommender_name: {gap: (accuracy, num_tuples)}}
        
        for gap, gap_data in session_data.items():
            for record in gap_data:
                recommender_name = record.get('recommender_name')
                if not recommender_name:
                    continue
                
                # Calculate accuracy for this record
                accuracy = self._calculate_record_accuracy(record)
                
                # Get number of tuples recommended
                recommended_results = record.get('recommended_results', [])
                num_tuples = len(recommended_results) if recommended_results else 0
                
                if recommender_name not in accuracy_data:
                    accuracy_data[recommender_name] = {}
                
                accuracy_data[recommender_name][gap] = (accuracy, num_tuples)
        
        if not accuracy_data:
            logger.warning(f"No accuracy data found for session {session_id}")
            return
        
        # Prepare data for plotting
        recommenders = sorted(accuracy_data.keys())
        gaps = sorted(set(gap for rec_data in accuracy_data.values() for gap in rec_data.keys()))
        
        # Convert gaps to integers for proper sorting
        try:
            gaps = sorted([int(gap) for gap in gaps])
            gaps = [str(gap) for gap in gaps]  # Convert back to strings for consistency
        except ValueError:
            # If gaps aren't integers, keep them as strings
            gaps = sorted(gaps)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set up bar positions
        x = np.arange(len(gaps))
        width = 0.8 / len(recommenders) if recommenders else 0.8
        
        # Plot bars for each recommender
        colors = plt.cm.Set3(np.linspace(0, 1, len(recommenders)))
        
        for i, recommender in enumerate(recommenders):
            accuracies = []
            tuple_counts = []
            for gap in gaps:
                accuracy, num_tuples = accuracy_data[recommender].get(gap, (0.0, 0))
                accuracies.append(accuracy)
                tuple_counts.append(num_tuples)
            
            bars = ax.bar(x + i * width, accuracies, width, 
                         label=recommender, color=colors[i], alpha=0.8)
            
            # Add value labels on bars (accuracy and tuple count)
            for bar, acc, count in zip(bars, accuracies, tuple_counts):
                if acc > 0:
                    height = bar.get_height()
                    # Show accuracy and tuple count
                    ax.annotate(f'{acc:.3f}\n({count} tuples)',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=7)
        
        # Customize the plot
        ax.set_xlabel('Gap')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Recommender Accuracy by Gap - Session {session_id}')
        ax.set_xticks(x + width * (len(recommenders) - 1) / 2)
        ax.set_xticklabels(gaps)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Save the plot
        plot_path = session_dir / f'accuracy_by_gap_session_{session_id}.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated accuracy plot for session {session_id}: {plot_path}")
    
    def _calculate_record_accuracy(self, record: Dict[str, Any]) -> float:
        """
        Calculate accuracy for a single experiment record.
        
        Args:
            record: Single experiment record
            
        Returns:
            Accuracy value between 0 and 1
        """
        current_results = record.get('current_results', [])
        future_results = record.get('future_results', [])
        recommended_results = record.get('recommended_results', [])
        
        if not current_results or not future_results or not recommended_results:
            return 0.0
        
        try:
            # Convert to DataFrames for easier comparison
            current_df = pd.DataFrame(current_results)
            future_df = pd.DataFrame(future_results)
            recommended_df = pd.DataFrame(recommended_results)
            
            # Calculate overlap accuracy: intersection of (current ∩ future) with recommended
            # This measures how well the recommender predicted the overlap between current and future results
            
            # Find overlap between current and future results
            if current_df.empty or future_df.empty:
                return 0.0
            
            # Get common columns for comparison
            common_cols = list(set(current_df.columns).intersection(set(future_df.columns)))
            if not common_cols:
                return 0.0
            
            # Calculate overlap between current and future
            overlap = pd.merge(current_df[common_cols], future_df[common_cols], how='inner')
            
            if overlap.empty:
                return 0.0
            
            # Check how many recommended results match the overlap
            if recommended_df.empty:
                return 0.0
            
            # Get common columns with recommended results
            rec_common_cols = list(set(overlap.columns).intersection(set(recommended_df.columns)))
            if not rec_common_cols:
                return 0.0
            
            # Find matches between recommended and overlap
            matches = pd.merge(overlap[rec_common_cols], recommended_df[rec_common_cols], how='inner')
            
            # Calculate accuracy as matches / total_overlap
            accuracy = len(matches) / len(overlap) if len(overlap) > 0 else 0.0
            
            return min(1.0, max(0.0, accuracy))
            
        except Exception as e:
            logger.debug(f"Error calculating accuracy for record: {e}")
            return 0.0
