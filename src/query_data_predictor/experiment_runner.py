"""
Main experiment runner for the query results prediction framework.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.config_manager import ConfigManager
from query_data_predictor.metrics import EvaluationMetrics
from query_data_predictor.recommender.tuple_recommender import TupleRecommender


import logging

class ExperimentRunner:
    """
    Main class for running experiments and evaluating query predictions.
    """
    
    def __init__(self, data_path: str, config_path: Optional[str] = None):
        """
        Initialize the experiment runner.
        
        Args:
            data_path: Path to the dataset directory containing metadata.csv
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.logger.info("Loading configuration...")
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize data loaders
        self.logger.info(f"Initializing data loaders with path: {data_path}")
        self.data_loader = DataLoader(data_path)
        self.query_result_sequence = QueryResultSequence(self.data_loader)
        
        # Initialize metrics
        eval_config = self.config.get('evaluation', {})
        self.metrics = EvaluationMetrics(
            jaccard_threshold=eval_config.get('jaccard_threshold', 0.5),
            column_weights=eval_config.get('column_weights')
        )
        
        # Initialize recommender
        self.recommender = TupleRecommender(self.config)
        
        # Create output directory
        output_config = self.config.get('output', {})
        self.output_dir = Path(output_config.get('results_dir', 'experiment_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
        
    def run_experiment_for_session(self, session_id: str, gap: int = 1) -> Dict[str, Any]:
        """
        Run the experiment for a specific session with a configurable gap between queries.
        
        Args:
            session_id: The ID of the session to process
            gap: Number of queries to skip (default: 1, which means next query)
            
        Returns:
            Dictionary with experiment results
        """
        self.logger.info(f"Running experiment for session {session_id} with gap {gap}")
        
        try:
            # Get all pairs of queries with the specified gap
            query_pairs = list(self.query_result_sequence.iter_query_result_pairs(session_id, gap))
            
            if not query_pairs:
                self.logger.warning(f"No valid query pairs found for session {session_id} with gap {gap}")
                return {'session_id': session_id, 'error': 'No valid query pairs found', 'gap': gap}
            
            # Initialize results container
            experiment_results = {
                'session_id': session_id,
                'gap': gap,
                'query_predictions': [],
                'overall_metrics': {}
            }
            
            all_metrics = []
            
            # Process each query pair
            for start_query_id, target_query_id, start_results, target_results in query_pairs:
                self.logger.info(f"Processing query {start_query_id} to predict {target_query_id}")
                
                # Generate predictions
                try:
                    predicted_results = self.recommender.recommend_tuples(start_results)
                    
                    # Evaluate predictions against target results
                    query_metrics = self.metrics.standard_metrics(predicted_results, target_results)
                    
                    # Store results for this query pair
                    query_result = {
                        'start_query_id': start_query_id,
                        'target_query_id': target_query_id,
                        'metrics': query_metrics
                    }
                    experiment_results['query_predictions'].append(query_result)
                    all_metrics.append(query_metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error predicting from query {start_query_id} to {target_query_id}: {e}")
                    continue
            
            # Calculate overall metrics across all predictions
            if all_metrics:
                overall_metrics = {}
                for metric in all_metrics[0].keys():
                    values = [m[metric] for m in all_metrics]
                    overall_metrics[metric] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values)
                    }
                experiment_results['overall_metrics'] = overall_metrics
            
            # Save experiment results
            self._save_experiment_results(experiment_results, session_id, gap)
            
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"Error in experiment for session {session_id}: {e}")
            return {'session_id': session_id, 'gap': gap, 'error': str(e)}
    
    def run_experiment_for_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the experiment for all available sessions.
        
        Returns:
            Dictionary mapping session IDs to experiment results
        """
        # Get all session IDs
        sessions = self.data_loader.metadata['session_id'].unique()
        self.logger.info(f"Running experiment for {len(sessions)} sessions")
        
        results = {}
        
        for session_id in sessions:
            session_id = str(session_id)
            results[session_id] = self.run_experiment_for_session(session_id)
            
        # Calculate aggregate metrics across all sessions
        self._calculate_aggregate_metrics(results)
        
        return results
    
    def _calculate_aggregate_metrics(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Calculate aggregate metrics across all sessions and save to file.
        
        Args:
            all_results: Dictionary mapping session IDs to experiment results
        """
        self.logger.info("Calculating aggregate metrics across all sessions")
        
        # Collect metrics from all predictions
        all_metrics_by_type = {}
        total_predictions = 0
        
        # Gather metrics from all sessions
        for session_id, session_result in all_results.items():
            if 'error' in session_result:
                continue
                
            for query_prediction in session_result.get('query_predictions', []):
                metrics = query_prediction.get('metrics', {})
                
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics_by_type:
                        all_metrics_by_type[metric_name] = []
                    all_metrics_by_type[metric_name].append(value)
                    
                total_predictions += 1
        
        # Calculate aggregate statistics
        aggregate_metrics = {
            'total_sessions': len(all_results),
            'total_predictions': total_predictions,
            'metrics': {}
        }
        
        for metric_name, values in all_metrics_by_type.items():
            if values:
                aggregate_metrics['metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'std': float(np.std(values))
                }
        
        # Save aggregate metrics
        self._save_aggregate_metrics(aggregate_metrics)
        
    def _save_experiment_results(self, results: Dict[str, Any], session_id: str) -> None:
        """
        Save experiment results to file.
        
        Args:
            results: Dictionary with experiment results
            session_id: Session ID for naming the output file
        """
        output_config = self.config.get('output', {})
        save_format = output_config.get('save_format', 'pkl')
        
        file_path = self.output_dir / f"experiment_results_session_{session_id}.{save_format}"
        
        try:
            if save_format == 'pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f)
            elif save_format == 'json':
                # Convert numpy values to Python native types
                results_json = self._convert_to_json_serializable(results)
                with open(file_path, 'w') as f:
                    json.dump(results_json, f, indent=2)
            elif save_format == 'csv':
                # Flatten metrics for CSV format
                flat_results = self._flatten_results_for_csv(results)
                pd.DataFrame(flat_results).to_csv(file_path, index=False)
            else:
                self.logger.warning(f"Unsupported save format: {save_format}, defaulting to pickle")
                with open(file_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(results, f)
                    
            self.logger.info(f"Saved experiment results to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving experiment results: {e}")
    
    def _save_aggregate_metrics(self, aggregate_metrics: Dict[str, Any]) -> None:
        """
        Save aggregate metrics to file.
        
        Args:
            aggregate_metrics: Dictionary with aggregate metrics
        """
        output_config = self.config.get('output', {})
        save_format = output_config.get('save_format', 'pkl')
        
        file_path = self.output_dir / f"aggregate_metrics.{save_format}"
        
        try:
            if save_format == 'pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(aggregate_metrics, f)
            elif save_format == 'json':
                # Convert numpy values to Python native types
                metrics_json = self._convert_to_json_serializable(aggregate_metrics)
                with open(file_path, 'w') as f:
                    json.dump(metrics_json, f, indent=2)
            elif save_format == 'csv':
                # Create a flat representation of metrics
                metrics_flat = []
                for metric_name, stats in aggregate_metrics['metrics'].items():
                    row = {'metric': metric_name}
                    row.update(stats)
                    metrics_flat.append(row)
                
                # Add summary information
                summary = {
                    'metric': 'summary',
                    'total_sessions': aggregate_metrics['total_sessions'],
                    'total_predictions': aggregate_metrics['total_predictions']
                }
                metrics_flat.append(summary)
                
                pd.DataFrame(metrics_flat).to_csv(file_path, index=False)
            else:
                self.logger.warning(f"Unsupported save format: {save_format}, defaulting to pickle")
                with open(file_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(aggregate_metrics, f)
                    
            self.logger.info(f"Saved aggregate metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving aggregate metrics: {e}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy values to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _flatten_results_for_csv(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested results dictionary for CSV export.
        
        Args:
            results: Nested results dictionary
            
        Returns:
            List of flat dictionaries suitable for CSV export
        """
        flat_results = []
        
        # Get session ID
        session_id = results['session_id']
        
        # Process each query prediction
        for query_pred in results.get('query_predictions', []):
            flat_pred = {
                'session_id': session_id,
                'current_query_id': query_pred['current_query_id'],
                'next_query_id': query_pred['next_query_id']
            }
            
            # Add metrics
            for metric_name, value in query_pred.get('metrics', {}).items():
                flat_pred[f'metric_{metric_name}'] = value
                
            flat_results.append(flat_pred)
            
        return flat_results
