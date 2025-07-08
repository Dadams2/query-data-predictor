"""
Enhanced recommender experiments with structured data collection.

This integrates the new experimental data collection system with your existing
recommender evaluation framework.
"""

import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
import warnings
import signal
import time
from contextlib import contextmanager

# Import the enhanced collector
from experiment_collector import (
    ExperimentCollector, 
    QueryContext, 
    RecommendationResult, 
    EvaluationResult,
    collect_recommendation_experiment
)

from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.metrics import EvaluationMetrics
from query_data_predictor.recommenders import (
    DummyRecommender,
    RandomRecommender,
    ClusteringRecommender,
    InterestingnessRecommender
)

logger = logging.getLogger(__name__)


class RecommenderExperimentRunner:
    """
    Enhanced experiment runner with structured data collection and provenance tracking.
    """
    
    def __init__(self, 
                 config_path: str = "config.yaml", 
                 dataset_dir: str = "data/datasets",
                 output_dir: str = "experiment_data",
                 enable_full_tuple_storage: bool = True,
                 enable_state_tracking: bool = False):
        """
        Initialize the enhanced experiment runner.
        
        Args:
            config_path: Path to configuration file
            dataset_dir: Directory containing the dataset files
            output_dir: Directory for storing experimental results
            enable_full_tuple_storage: Whether to store complete tuple data
            enable_state_tracking: Whether to track recommender internal state
        """
        self.config = self._load_config(config_path)
        self.dataset_dir = Path(dataset_dir)
        self.dataloader = DataLoader(str(self.dataset_dir))
        self.query_result_sequence = QueryResultSequence(self.dataloader)
        self.evaluator = EvaluationMetrics()
        
        # Enhanced data collection
        self.collector = ExperimentCollector(
            base_output_dir=output_dir,
            collection_format="both",  # Store in both JSONL and structured format
            enable_tuple_storage=enable_full_tuple_storage,
            enable_state_tracking=enable_state_tracking
        )
        
        # Initialize recommenders
        self.recommender_config = self._create_recommender_config()
        self.recommenders = self._initialize_recommenders()
        
        # Session tracking
        self.current_session_id: Optional[str] = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'recommendation': {'mode': 'top_quartile'},
                'clustering': {'n_clusters': 3, 'random_state': 42},
                'random': {'random_seed': 42},
                'discretization': {'enabled': True, 'method': 'equal_width', 'bins': 5},
                'association_rules': {'min_support': 0.1, 'metric': 'confidence', 'min_threshold': 0.7}
            }
    
    def _create_recommender_config(self) -> Dict[str, Any]:
        """Create standardized configuration for all recommenders."""
        base_config = self.config.copy()
        base_config['recommendation'] = {'mode': 'top_quartile'}
        return base_config
    
    def _initialize_recommenders(self) -> Dict[str, Any]:
        """Initialize all recommender instances."""
        return {
            'dummy': DummyRecommender(self.recommender_config),
            'random': RandomRecommender(self.recommender_config),
            'clustering': ClusteringRecommender(self.recommender_config),
            'interestingness': InterestingnessRecommender(self.recommender_config)
        }
    
    def run_enhanced_experiment(self, 
                               session_id: str, 
                               max_gap: int = 5,
                               include_query_text: bool = False,
                               store_intermediate_states: bool = False) -> Dict[str, Any]:
        """
        Run enhanced experiments with full data collection.
        
        Args:
            session_id: ID of the session to experiment on
            max_gap: Maximum gap between queries to test
            include_query_text: Whether to extract and store actual query text
            store_intermediate_states: Whether to store recommender states
            
        Returns:
            Dictionary with experiment summary and collection info
        """
        
        # Start experimental session
        exp_session_id = self.collector.start_experiment_session(
            f"recommender_eval_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.current_session_id = exp_session_id
        
        logger.info(f"Starting enhanced experiment on session {session_id}")
        logger.info(f"Experiment session ID: {exp_session_id}")
        
        # Get all query IDs for this session
        query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)
        if len(query_ids) < 2:
            logger.warning(f"Session {session_id} has fewer than 2 queries, skipping")
            return {"error": "Insufficient queries", "session_id": session_id}
        
        logger.info(f"Session {session_id} has {len(query_ids)} queries")
        
        # Collect experiments
        experiment_ids = []
        total_experiments = 0
        successful_experiments = 0
        
        # Run experiments for different gaps
        for gap in range(1, min(max_gap + 1, len(query_ids))):
            logger.info(f"Testing gap {gap}")
            gap_results = self._run_enhanced_gap_experiment(
                session_id, gap, include_query_text, store_intermediate_states
            )
            experiment_ids.extend(gap_results['experiment_ids'])
            total_experiments += gap_results['total']
            successful_experiments += gap_results['successful']
        
        # Create summary
        summary = {
            "experiment_session_id": exp_session_id,
            "source_session_id": session_id,
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "error_rate": (total_experiments - successful_experiments) / total_experiments if total_experiments > 0 else 0,
            "experiment_ids": experiment_ids,
            "output_directory": str(self.collector.base_output_dir),
            "collection_complete": True
        }
        
        # Generate collection summary
        collection_summary = self.collector.create_analysis_summary(exp_session_id)
        summary.update(collection_summary)
        
        logger.info(f"Enhanced experiment completed: {successful_experiments}/{total_experiments} successful")
        return summary
    
    def _run_enhanced_gap_experiment(self, 
                                   session_id: str, 
                                   gap: int,
                                   include_query_text: bool,
                                   store_intermediate_states: bool) -> Dict[str, Any]:
        """Run enhanced experiments for a specific gap between queries."""
        
        experiment_ids = []
        total_count = 0
        successful_count = 0
        
        try:
            # Iterate through all valid query pairs with this gap
            for current_id, future_id, current_results, future_results in \
                self.query_result_sequence.iter_query_result_pairs(session_id, gap):
                
                # Skip if current results are empty
                if current_results.empty:
                    continue
                
                # Extract query information if needed
                current_query_text = self._get_query_text(session_id, current_id) if include_query_text else ""
                future_query_text = self._get_query_text(session_id, future_id) if include_query_text else ""
                
                # Test each recommender
                for recommender_name, recommender in self.recommenders.items():
                    total_count += 1
                    
                    exp_id = self._evaluate_enhanced_recommender(
                        session_id=session_id,
                        current_query_id=current_id,
                        future_query_id=future_id,
                        current_results=current_results,
                        future_results=future_results,
                        current_query_text=current_query_text,
                        future_query_text=future_query_text,
                        recommender_name=recommender_name,
                        recommender=recommender,
                        gap=gap,
                        store_states=store_intermediate_states
                    )
                    
                    if exp_id:
                        experiment_ids.append(exp_id)
                        successful_count += 1
                        
        except Exception as e:
            logger.error(f"Error in enhanced gap {gap} experiment for session {session_id}: {str(e)}", exc_info=True)
        
        return {
            "experiment_ids": experiment_ids,
            "total": total_count,
            "successful": successful_count
        }
    
    def _get_query_text(self, session_id: str, query_id: str) -> str:
        """Extract query text from dataset (placeholder - implement based on your data structure)."""
        # This would need to be implemented based on how your data stores query text
        # For now, return a placeholder
        return f"Query {query_id} from session {session_id}"
    
    def _evaluate_enhanced_recommender(self, 
                                     session_id: str,
                                     current_query_id: str, 
                                     future_query_id: str,
                                     current_results: pd.DataFrame,
                                     future_results: pd.DataFrame,
                                     current_query_text: str,
                                     future_query_text: str,
                                     recommender_name: str,
                                     recommender: Any,
                                     gap: int,
                                     store_states: bool = False) -> Optional[str]:
        """Evaluate a single recommender with enhanced data collection."""
        
        start_time = time.time()
        
        try:
            # Set timeout based on dataset size
            timeout_seconds = 30 if len(current_results) < 100 else 120
            
            with self._timeout(timeout_seconds):
                # Get recommendations
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    recommendations = recommender.recommend_tuples(current_results)
            
            execution_time = time.time() - start_time
            
            # Calculate comprehensive metrics
            overlap_accuracy = self.evaluator.overlap_accuracy(
                previous=current_results,
                actual=future_results,
                predicted=recommendations
            )
            
            jaccard_sim = self.evaluator.jaccard_similarity(recommendations, future_results)
            
            # Calculate additional metrics
            precision, recall, f1 = self._calculate_precision_recall_f1(
                recommendations, future_results, current_results
            )
            
            # Create contexts
            current_context = QueryContext(
                session_id=session_id,
                query_position=current_query_id,
                query_text=current_query_text,
                query_hash=self._hash_string(current_query_text),
                result_set_size=len(current_results)
            )
            
            future_context = QueryContext(
                session_id=session_id,
                query_position=future_query_id,
                query_text=future_query_text,
                query_hash=self._hash_string(future_query_text),
                result_set_size=len(future_results)
            )
            
            # Create recommendation result
            rec_result = RecommendationResult(
                experiment_id="",  # Will be generated
                predicted_tuples=recommendations,
                recommendation_metadata={
                    "recommender_type": recommender_name,
                    "input_size": len(current_results),
                    "output_size": len(recommendations)
                }
            )
            
            # Create evaluation result
            eval_result = EvaluationResult(
                experiment_id="",  # Will be generated
                overlap_accuracy=overlap_accuracy,
                jaccard_similarity=jaccard_sim,
                precision=precision,
                recall=recall,
                f1_score=f1,
                exact_matches=self._count_exact_matches(recommendations, future_results),
                predicted_count=len(recommendations),
                actual_count=len(future_results),
                intersection_count=len(pd.merge(recommendations, future_results, how='inner')),
                union_count=len(pd.concat([recommendations, future_results]).drop_duplicates())
            )
            
            # Collect the experiment
            experiment_id = self.collector.collect_experiment(
                session_id=self.current_session_id,
                current_query_position=current_context.query_position,
                target_query_position=future_context.query_position,
                recommender_name=recommender_name,
                current_query_context=current_context,
                target_query_context=future_context,
                recommendation_result=rec_result,
                actual_results=future_results,
                evaluation_result=eval_result,
                recommender_config=self.recommender_config,
                execution_time=execution_time
            )
            
            logger.debug(f"Enhanced evaluation completed for {recommender_name}, gap {gap}: "
                        f"accuracy={overlap_accuracy:.4f}, precision={precision:.4f}, "
                        f"recall={recall:.4f}, time={execution_time:.2f}s")
            
            return experiment_id
                        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            if "Timed out" in error_msg:
                logger.error(f"Timeout for {recommender_name} on gap {gap}: {error_msg}", exc_info=True)
            else:
                logger.error(f"Error evaluating {recommender_name} for gap {gap}: {error_msg}", exc_info=True)
            
            # Still collect the failed experiment for analysis
            try:
                current_context = QueryContext(
                    session_id=int(session_id),
                    query_position=int(str(current_query_id).split('_')[-1]) if '_' in str(current_query_id) else int(current_query_id),
                    query_text=current_query_text,
                    query_hash=self._hash_string(current_query_text),
                    result_set_size=len(current_results)
                )
                
                future_context = QueryContext(
                    session_id=int(session_id),
                    query_position=int(str(future_query_id).split('_')[-1]) if '_' in str(future_query_id) else int(future_query_id),
                    query_text=future_query_text,
                    query_hash=self._hash_string(future_query_text),
                    result_set_size=len(future_results)
                )
                
                # Create empty recommendation result for failed case
                rec_result = RecommendationResult(
                    experiment_id="",
                    predicted_tuples=pd.DataFrame(),
                    recommendation_metadata={"error": error_msg}
                )
                
                experiment_id = self.collector.collect_experiment(
                    session_id=self.current_session_id,
                    current_query_position=current_context.query_position,
                    target_query_position=future_context.query_position,
                    recommender_name=recommender_name,
                    current_query_context=current_context,
                    target_query_context=future_context,
                    recommendation_result=rec_result,
                    actual_results=future_results,
                    recommender_config=self.recommender_config,
                    execution_time=execution_time,
                    error_info=error_msg
                )
                
                return experiment_id
            except Exception as collect_error:
                logger.error(f"Failed to collect error case: {collect_error}", exc_info=True)
                return None
    
    def _calculate_precision_recall_f1(self, predicted: pd.DataFrame, 
                                     actual: pd.DataFrame, 
                                     baseline: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if predicted.empty:
            return 0.0, 0.0, 0.0
        
        # Use overlap with baseline as the relevant set
        overlap = pd.merge(baseline, actual, how='inner')
        if overlap.empty:
            return 0.0, 0.0, 0.0
        
        # True positives: predicted tuples that are in the overlap
        true_positives = len(pd.merge(predicted, overlap, how='inner'))
        
        # Precision: TP / (TP + FP) = TP / total_predicted
        precision = true_positives / len(predicted) if len(predicted) > 0 else 0.0
        
        # Recall: TP / (TP + FN) = TP / total_relevant
        recall = true_positives / len(overlap) if len(overlap) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _count_exact_matches(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> int:
        """Count exact tuple matches between predicted and actual."""
        return len(pd.merge(predicted, actual, how='inner'))
    
    def _hash_string(self, s: str) -> str:
        """Create hash of a string."""
        import hashlib
        return hashlib.md5(s.encode()).hexdigest()[:16]
    
    @contextmanager
    def _timeout(self, seconds):
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
    
    def load_experimental_results(self, 
                                 session_ids: Optional[List[str]] = None,
                                 recommender_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load experimental results for analysis."""
        return self.collector.load_experiment_results(
            session_ids=session_ids,
            recommender_names=recommender_names,
            include_tuples=False  # For summary analysis
        )
    
    def create_comparative_analysis(self) -> Dict[str, Any]:
        """Create comparative analysis across all collected experiments."""
        results_df = self.load_experimental_results()
        
        if results_df.empty:
            return {"error": "No experimental results found"}
        
        analysis = {
            "experiment_overview": {
                "total_experiments": len(results_df),
                "unique_sessions": results_df['meta_session_id'].nunique(),
                "recommenders_tested": results_df['meta_recommender_name'].unique().tolist(),
                "gaps_tested": sorted(results_df['meta_gap'].unique().tolist()),
                "success_rate": (results_df['meta_status'] == 'completed').mean()
            }
        }
        
        # Performance comparison
        if 'eval_overlap_accuracy' in results_df.columns:
            perf_comparison = results_df.groupby('meta_recommender_name').agg({
                'eval_overlap_accuracy': ['mean', 'std', 'count'],
                'eval_precision': ['mean', 'std'],
                'eval_recall': ['mean', 'std'],
                'eval_f1_score': ['mean', 'std'],
                'meta_execution_time_seconds': ['mean', 'std']
            }).round(4)
            
            analysis["performance_comparison"] = perf_comparison.to_dict()
        
        # Gap analysis
        gap_analysis = results_df.groupby(['meta_recommender_name', 'meta_gap'])['eval_overlap_accuracy'].mean().unstack(fill_value=0)
        analysis["accuracy_by_gap"] = gap_analysis.to_dict()
        
        return analysis


class TimeoutError(Exception):
    pass


def main():
    """Main function to run enhanced experiments."""
    # Setup enhanced experiment runner with timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"results/experiment/experiment_results_{timestamp}"
    
    runner = RecommenderExperimentRunner(
        output_dir=output_dir,
        enable_full_tuple_storage=True,
        enable_state_tracking=False
    )
    
    # Get first available session
    sessions = runner.dataloader.get_sessions()
    if not sessions:
        logger.error("No sessions found in dataset", exc_info=True)
        return
    
    session_id = sessions[0]
    logger.info(f"Using session {session_id}")
    
    # Run enhanced experiments
    experiment_summary = runner.run_enhanced_experiment(
        session_id=session_id, 
        max_gap=3,  # Start with smaller gap for testing
        include_query_text=True,
        store_intermediate_states=False
    )
    
    logger.info("Enhanced Experiment Summary:")
    for key, value in experiment_summary.items():
        if key != 'experiment_ids':  # Don't print long list
            logger.info(f"  {key}: {value}")
    
    # Create comparative analysis
    analysis = runner.create_comparative_analysis()
    logger.info("Comparative Analysis:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Enhanced experiment completed successfully!")
    logger.info(f"Results stored in: {runner.collector.base_output_dir}")


if __name__ == "__main__":
    main()
