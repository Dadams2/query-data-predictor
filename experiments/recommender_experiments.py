"""
Enhanced recommender experiments with structured data collection.

This integrates the new experimental data collection system with your existing
recommender evaluation framework.
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import hashlib
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
import warnings
import signal
import time
from contextlib import contextmanager

# Import the enhanced collector
try:
    from .experiment_collector import (
        ExperimentCollector, 
        QueryContext, 
        RecommendationResult, 
        EvaluationResult,
        collect_recommendation_experiment
    )
except ImportError:
    # Fallback for direct execution
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
from query_data_predictor.recommender import (
    DummyRecommender,
    RandomRecommender,
    ClusteringRecommender,
    InterestingnessRecommender,
    QueryExpansionRecommender,
    RandomTableRecommender,
    SimilarityRecommender,
    FrequencyRecommender,
    SamplingRecommender
)
from query_data_predictor.query_runner import QueryRunner

logger = logging.getLogger(__name__)


class MockQueryRunner:
    """Mock QueryRunner for testing QueryExpansionRecommender without a real database."""
    
    def __init__(self):
        self.executed_queries = []
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a mock query and return sample astronomical data."""
        self.executed_queries.append(query)
        
        # Generate mock SDSS-like data
        np.random.seed(42)  # For reproducibility
        n_rows = np.random.randint(10, 100)
        
        mock_data = {
            'objID': np.random.randint(1000000, 9999999, n_rows),
            'ra': np.random.uniform(0, 360, n_rows),
            'dec': np.random.uniform(-90, 90, n_rows),
            'z': np.random.uniform(0.1, 3.0, n_rows),
            'zConf': np.random.uniform(0.5, 1.0, n_rows),
            'modelMag_r': np.random.uniform(15, 25, n_rows),
            'SpecClass': np.random.choice(['GALAXY', 'QSO', 'STAR'], n_rows),
            'primTarget': np.random.randint(1, 1000, n_rows)
        }
        
        return pd.DataFrame(mock_data)
    
    def connect(self):
        """Mock connection."""
        pass
    
    def disconnect(self):
        """Mock disconnection."""
        pass


class RealDataQueryRunner:
    """QueryRunner that uses real data from the dataset for testing QueryExpansionRecommender."""
    
    def __init__(self, dataloader: DataLoader, session_id: int):
        """
        Initialize with access to real data.
        
        Args:
            dataloader: DataLoader instance with access to the dataset
            session_id: Session ID to use as the source of real data
        """
        self.dataloader = dataloader
        self.session_id = session_id
        self.executed_queries = []
        
        # Load the session data to have access to all query results
        try:
            self.session_data = dataloader.get_results_for_session(session_id)
            self.available_queries = sorted(self.session_data['query_position'].unique())
            logger.info(f"RealDataQueryRunner initialized with session {session_id}, "
                       f"{len(self.available_queries)} queries available")
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            # Fallback to mock data if real data fails
            self.session_data = None
            self.available_queries = []
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a 'query' by returning real data from the dataset.
        
        Since we don't have a real database, we simulate query execution by:
        1. Parsing the query to understand what kind of data is being requested
        2. Returning appropriate real data from our dataset
        """
        self.executed_queries.append(query)
        
        if self.session_data is None or not self.available_queries:
            # Fallback to mock data
            return self._generate_fallback_data()
        
        try:
            # Simple query interpretation - in a real system this would be much more sophisticated
            selected_query_position = self._select_appropriate_query(query)
            
            # Load the actual results for this query
            query_row = self.session_data[
                self.session_data['query_position'] == selected_query_position
            ].iloc[0]
            
            results_filepath = query_row['results_filepath']
            results_path = Path(results_filepath)
            
            if results_path.exists():
                with open(results_path, 'rb') as f:
                    real_results = pickle.load(f)
                
                # Add some variation to simulate different query results
                varied_results = self._add_query_variation(real_results, query)
                
                logger.debug(f"RealDataQueryRunner returned {len(varied_results)} rows "
                           f"from query position {selected_query_position}")
                
                return varied_results
            else:
                logger.warning(f"Results file not found: {results_filepath}")
                return self._generate_fallback_data()
                
        except Exception as e:
            logger.warning(f"Error processing query: {e}")
            return self._generate_fallback_data()
    
    def _select_appropriate_query(self, query: str) -> int:
        """
        Select an appropriate query position based on the query text.
        This is a simplified approach - in reality you'd parse SQL properly.
        """
        query_lower = query.lower()
        
        # Try to match query characteristics to available data
        if 'redshift' in query_lower or 'z ' in query_lower:
            # Prefer queries that might have redshift data
            for pos in [0, 5, 10, 15, 20]:
                if pos in self.available_queries:
                    return pos
        
        if 'count' in query_lower or 'limit' in query_lower:
            # For count queries, use smaller result sets
            for pos in [1, 3, 7, 11]:
                if pos in self.available_queries:
                    return pos
        
        if 'ra' in query_lower or 'dec' in query_lower or 'spatial' in query_lower:
            # For spatial queries, use any available query
            return np.random.choice(self.available_queries[:10])  # Use first 10 queries
        
        # Default: randomly select from available queries
        return np.random.choice(self.available_queries)
    
    def _add_query_variation(self, real_results: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Add some variation to the real results to simulate different queries.
        This makes the expansion queries return somewhat different but realistic data.
        """
        if real_results.empty:
            return real_results
        
        # Make a copy to avoid modifying the original
        varied_results = real_results.copy()
        
        query_lower = query.lower()
        
        # Simulate different query constraints by sampling/filtering the data
        if 'between' in query_lower or 'range' in query_lower:
            # For range queries, return a subset
            n_samples = min(len(varied_results), np.random.randint(10, 100))
            varied_results = varied_results.sample(n=n_samples, random_state=42).reset_index(drop=True)
        
        elif 'limit' in query_lower:
            # Extract limit number if possible, otherwise use a random limit
            try:
                limit_match = re.search(r'limit\s+(\d+)', query_lower)
                if limit_match:
                    limit_val = min(int(limit_match.group(1)), len(varied_results))
                    varied_results = varied_results.head(limit_val)
                else:
                    limit_val = min(50, len(varied_results))
                    varied_results = varied_results.head(limit_val)
            except:
                varied_results = varied_results.head(50)
        
        elif 'similar' in query_lower or 'nearby' in query_lower:
            # For similarity queries, add some noise to make results "similar but different"
            if 'ra' in varied_results.columns and 'dec' in varied_results.columns:
                # Add small random offsets to coordinates
                ra_noise = np.random.normal(0, 0.01, len(varied_results))
                dec_noise = np.random.normal(0, 0.01, len(varied_results))
                varied_results['ra'] = varied_results['ra'] + ra_noise
                varied_results['dec'] = varied_results['dec'] + dec_noise
        
        # Limit the number of results to avoid overwhelming the recommender
        max_results = 200  # Reasonable limit for expansion queries
        if len(varied_results) > max_results:
            varied_results = varied_results.sample(n=max_results, random_state=42).reset_index(drop=True)
        
        return varied_results
    
    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generate fallback mock data when real data is not available."""
        np.random.seed(42)
        n_rows = np.random.randint(10, 50)
        
        # Generate data that matches the structure we've seen in real data
        fallback_data = {
            'ra': np.random.uniform(0, 360, n_rows),
            'dec': np.random.uniform(-90, 90, n_rows),
            'type': np.random.choice([3, 6], n_rows),  # SDSS object types
            'modelmag_r': np.random.uniform(15, 25, n_rows),
            'z': np.random.uniform(0.01, 2.0, n_rows),
            'objid': np.random.randint(1000000000, 9999999999, n_rows)
        }
        
        return pd.DataFrame(fallback_data)
    
    def connect(self):
        """Mock connection."""
        pass
    
    def disconnect(self):
        """Mock disconnection."""
        pass


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
        # Create a real data QueryRunner for QueryExpansionRecommender and RandomTableRecommender
        # Use the first available session as the data source
        sessions = self.dataloader.get_sessions()
        if sessions:
            # Use the first session as the data source for expansion queries
            first_session = sessions[0]
            real_query_runner = RealDataQueryRunner(self.dataloader, first_session)
            logger.info(f"Using session {first_session} as data source for out-of-results recommenders")
        else:
            # Fallback to mock if no sessions available
            real_query_runner = MockQueryRunner()
            logger.warning("No sessions available, using mock data for out-of-results recommenders")
        
        return {
            # Out-of-results recommenders (need QueryRunner)
            'query_expansion': QueryExpansionRecommender(
                self.recommender_config, 
                query_runner=real_query_runner
            ),
            'random_table_baseline': RandomTableRecommender(
                self.recommender_config,
                query_runner=real_query_runner
            ),
            
            # In-results recommenders (work with current results only)
            'dummy': DummyRecommender(self.recommender_config),
            'random': RandomRecommender(self.recommender_config),
            'clustering': ClusteringRecommender(self.recommender_config),
            'interestingness': InterestingnessRecommender(self.recommender_config),
            'similarity': SimilarityRecommender(self.recommender_config),
            'frequency': FrequencyRecommender(self.recommender_config),
            'sampling': SamplingRecommender(self.recommender_config)
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
            
            # Calculate ROC-AUC
            roc_auc = self._calculate_roc_auc(recommendations, future_results, current_results)
            
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
                union_count=len(pd.concat([recommendations, future_results]).drop_duplicates()),
                roc_auc=roc_auc
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
                        f"recall={recall:.4f}, AUC={roc_auc:.4f}, time={execution_time:.2f}s")
            
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
        
        try:
            # Check if DataFrames have common columns for merge
            common_columns = list(set(baseline.columns).intersection(set(actual.columns)))
            
            if not common_columns:
                # Use tuple-based comparison if no common columns
                return self._calculate_precision_recall_f1_tuple_based(predicted, actual, baseline)
            
            # Use overlap with baseline as the relevant set
            overlap = pd.merge(baseline, actual, how='inner', on=common_columns)
            if overlap.empty:
                return 0.0, 0.0, 0.0
            
            # True positives: predicted tuples that are in the overlap
            pred_common_columns = list(set(predicted.columns).intersection(set(overlap.columns)))
            if not pred_common_columns:
                # Use tuple-based comparison if no common columns
                return self._calculate_precision_recall_f1_tuple_based(predicted, actual, baseline)
            
            true_positives = len(pd.merge(predicted, overlap, how='inner', on=pred_common_columns))
            
            # Precision: TP / (TP + FP) = TP / total_predicted
            precision = true_positives / len(predicted) if len(predicted) > 0 else 0.0
            
            # Recall: TP / (TP + FN) = TP / total_relevant
            recall = true_positives / len(overlap) if len(overlap) > 0 else 0.0
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return precision, recall, f1
            
        except Exception as e:
            # Fallback to tuple-based comparison if merge fails
            return self._calculate_precision_recall_f1_tuple_based(predicted, actual, baseline)
    
    def _calculate_precision_recall_f1_tuple_based(self, predicted: pd.DataFrame, 
                                                  actual: pd.DataFrame, 
                                                  baseline: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score using tuple-based comparison."""
        if predicted.empty:
            return 0.0, 0.0, 0.0
        
        # Convert DataFrames to sets of tuples
        baseline_tuples = self._dataframe_to_tuple_set(baseline)
        actual_tuples = self._dataframe_to_tuple_set(actual)
        predicted_tuples = self._dataframe_to_tuple_set(predicted)
        
        # Find overlap between baseline and actual
        overlap_tuples = baseline_tuples.intersection(actual_tuples)
        
        if not overlap_tuples:
            return 0.0, 0.0, 0.0
        
        # True positives: predicted tuples that are in the overlap
        true_positives = len(predicted_tuples.intersection(overlap_tuples))
        
        # Precision: TP / (TP + FP) = TP / total_predicted
        precision = true_positives / len(predicted_tuples) if len(predicted_tuples) > 0 else 0.0
        
        # Recall: TP / (TP + FN) = TP / total_relevant
        recall = true_positives / len(overlap_tuples) if len(overlap_tuples) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _calculate_roc_auc(self, predicted: pd.DataFrame, 
                         actual: pd.DataFrame, 
                         baseline: pd.DataFrame) -> float:
        """
        Calculate ROC-AUC score for binary classification.
        
        This treats the prediction as a binary classification task:
        - Positive class: tuples that appear in both baseline and actual (relevant tuples)
        - Negative class: tuples that appear in baseline but not actual (irrelevant tuples)
        - The recommender's job is to distinguish between these two classes
        """
        try:
            if predicted.empty or actual.empty or baseline.empty:
                return 0.5  # Random performance
            
            # Find the overlap (relevant tuples)
            common_columns = list(set(baseline.columns).intersection(set(actual.columns)))
            if not common_columns:
                return self._calculate_roc_auc_tuple_based(predicted, actual, baseline)
            
            overlap = pd.merge(baseline, actual, how='inner', on=common_columns)
            if overlap.empty:
                return 0.5  # No relevant tuples
            
            # Create ground truth labels for baseline tuples
            # 1 = relevant (in overlap), 0 = irrelevant (not in overlap)
            baseline_labels = []
            baseline_tuples = []
            
            for _, row in baseline.iterrows():
                baseline_tuples.append(row)
                # Check if this baseline tuple is in the overlap
                is_relevant = len(overlap.merge(pd.DataFrame([row]), how='inner', on=common_columns)) > 0
                baseline_labels.append(1 if is_relevant else 0)
            
            if len(set(baseline_labels)) < 2:
                return 0.5  # Need both positive and negative examples
            
            # Create prediction scores for baseline tuples
            # Higher score = more likely to be recommended
            pred_columns = list(set(predicted.columns).intersection(set(baseline.columns)))
            if not pred_columns:
                return self._calculate_roc_auc_tuple_based(predicted, actual, baseline)
            
            prediction_scores = []
            for _, baseline_row in baseline.iterrows():
                # Score = 1 if tuple was predicted, 0 otherwise
                # This is a simplified approach - in reality you might have confidence scores
                baseline_row_df = pd.DataFrame([baseline_row])
                is_predicted = len(predicted.merge(baseline_row_df, how='inner', on=pred_columns)) > 0
                prediction_scores.append(1.0 if is_predicted else 0.0)
            
            # Calculate ROC-AUC
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(baseline_labels, prediction_scores)
            
        except ImportError:
            # sklearn not available, use approximation
            return self._calculate_roc_auc_approximation(predicted, actual, baseline)
        except Exception as e:
            logger.warning(f"Error calculating ROC-AUC: {e}")
            return 0.5  # Default to random performance
    
    def _calculate_roc_auc_tuple_based(self, predicted: pd.DataFrame, 
                                     actual: pd.DataFrame, 
                                     baseline: pd.DataFrame) -> float:
        """Calculate ROC-AUC using tuple-based comparison."""
        try:
            # Convert to tuple sets
            baseline_tuples = self._dataframe_to_tuple_set(baseline)
            actual_tuples = self._dataframe_to_tuple_set(actual)
            predicted_tuples = self._dataframe_to_tuple_set(predicted)
            
            # Find relevant tuples (overlap between baseline and actual)
            relevant_tuples = baseline_tuples.intersection(actual_tuples)
            
            if not relevant_tuples or len(relevant_tuples) == len(baseline_tuples):
                return 0.5  # No discrimination possible
            
            # Create labels and scores
            labels = []
            scores = []
            
            for tuple_val in baseline_tuples:
                # Label: 1 if relevant, 0 if not
                labels.append(1 if tuple_val in relevant_tuples else 0)
                # Score: 1 if predicted, 0 if not
                scores.append(1.0 if tuple_val in predicted_tuples else 0.0)
            
            if len(set(labels)) < 2:
                return 0.5  # Need both classes
            
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(labels, scores)
            
        except ImportError:
            return self._calculate_roc_auc_approximation(predicted, actual, baseline)
        except Exception:
            return 0.5
    
    def _calculate_roc_auc_approximation(self, predicted: pd.DataFrame, 
                                       actual: pd.DataFrame, 
                                       baseline: pd.DataFrame) -> float:
        """
        Approximate ROC-AUC without sklearn.
        Uses the relationship between precision, recall, and AUC for balanced datasets.
        """
        try:
            precision, recall, _ = self._calculate_precision_recall_f1(predicted, actual, baseline)
            
            # For balanced binary classification, AUC ≈ (precision + recall) / 2
            # This is a rough approximation
            if precision == 0 and recall == 0:
                return 0.5
            
            # Ensure the result is between 0.5 and 1.0
            auc_approx = max(0.5, min(1.0, (precision + recall) / 2))
            return auc_approx
            
        except Exception:
            return 0.5
    
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
    
    def run_multi_session_experiments(self, 
                                     session_ids: List[str],
                                     max_gap: int = 5,
                                     include_query_text: bool = False,
                                     store_intermediate_states: bool = False,
                                     session_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run enhanced experiments across multiple sessions.
        
        Args:
            session_ids: List of session IDs to experiment on
            max_gap: Maximum gap between queries to test
            include_query_text: Whether to extract and store actual query text
            store_intermediate_states: Whether to store recommender states
            session_limit: Maximum number of sessions to process (for testing)
            
        Returns:
            Dictionary with comprehensive experiment summary
        """
        
        # Limit sessions if specified
        if session_limit:
            session_ids = session_ids[:session_limit]
        
        # Start multi-session experimental session
        exp_session_id = self.collector.start_experiment_session(
            f"multi_session_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.current_session_id = exp_session_id
        
        logger.info(f"Starting multi-session experiment on {len(session_ids)} sessions")
        logger.info(f"Experiment session ID: {exp_session_id}")
        
        # Track overall progress
        all_experiment_ids = []
        session_summaries = {}
        total_experiments = 0
        total_successful = 0
        failed_sessions = []
        
        for i, session_id in enumerate(session_ids):
            logger.info(f"Processing session {i+1}/{len(session_ids)}: {session_id}")
            
            try:
                # Run experiment for this session
                session_summary = self.run_enhanced_experiment(
                    session_id=session_id,
                    max_gap=max_gap,
                    include_query_text=include_query_text,
                    store_intermediate_states=store_intermediate_states
                )
                
                if 'error' in session_summary:
                    logger.warning(f"Session {session_id} failed: {session_summary['error']}")
                    failed_sessions.append(session_id)
                    continue
                
                # Accumulate results
                session_summaries[session_id] = session_summary
                all_experiment_ids.extend(session_summary.get('experiment_ids', []))
                total_experiments += session_summary.get('total_experiments', 0)
                total_successful += session_summary.get('successful_experiments', 0)
                
                logger.info(f"Session {session_id} completed: "
                           f"{session_summary.get('successful_experiments', 0)}/"
                           f"{session_summary.get('total_experiments', 0)} successful")
                
            except Exception as e:
                logger.error(f"Error processing session {session_id}: {str(e)}", exc_info=True)
                failed_sessions.append(session_id)
        
        # Create comprehensive summary
        multi_session_summary = {
            "experiment_session_id": exp_session_id,
            "sessions_processed": len(session_ids) - len(failed_sessions),
            "sessions_requested": len(session_ids),
            "failed_sessions": failed_sessions,
            "total_experiments": total_experiments,
            "successful_experiments": total_successful,
            "overall_success_rate": total_successful / total_experiments if total_experiments > 0 else 0,
            "experiment_ids": all_experiment_ids,
            "session_summaries": session_summaries,
            "output_directory": str(self.collector.base_output_dir),
            "collection_complete": True
        }
        
        logger.info(f"Multi-session experiment completed: {len(session_ids) - len(failed_sessions)}/"
                   f"{len(session_ids)} sessions successful")
        logger.info(f"Overall: {total_successful}/{total_experiments} experiments successful")
        
        return multi_session_summary
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about available sessions to help with selection."""
        sessions = self.dataloader.get_sessions()
        
        session_stats = {}
        for session_id in sessions:
            query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)
            session_stats[session_id] = {
                'query_count': len(query_ids),
                'max_possible_gap': len(query_ids) - 1 if len(query_ids) > 1 else 0
            }
        
        return {
            'total_sessions': len(sessions),
            'session_details': session_stats,
            'sessions_with_multiple_queries': len([s for s in session_stats.values() if s['query_count'] > 1]),
            'query_count_distribution': {
                'min': min(s['query_count'] for s in session_stats.values()) if session_stats else 0,
                'max': max(s['query_count'] for s in session_stats.values()) if session_stats else 0,
                'mean': np.mean([s['query_count'] for s in session_stats.values()]) if session_stats else 0
            }
        }
    
    def select_sessions_for_experiment(self, 
                                     min_queries: int = 3,
                                     max_sessions: Optional[int] = None,
                                     session_selection_strategy: str = 'random') -> List[str]:
        """
        Select sessions for experimentation based on criteria.
        
        Args:
            min_queries: Minimum number of queries per session
            max_sessions: Maximum number of sessions to select
            session_selection_strategy: Strategy for selection ('random', 'largest', 'smallest', 'diverse')
            
        Returns:
            List of selected session IDs
        """
        sessions = self.dataloader.get_sessions()
        
        # Filter sessions by minimum query count
        valid_sessions = []
        for session_id in sessions:
            query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)
            if len(query_ids) >= min_queries:
                valid_sessions.append((session_id, len(query_ids)))
        
        if not valid_sessions:
            logger.warning(f"No sessions found with at least {min_queries} queries")
            return []
        
        # Apply selection strategy
        if session_selection_strategy == 'random':
            np.random.shuffle(valid_sessions)
        elif session_selection_strategy == 'largest':
            valid_sessions.sort(key=lambda x: x[1], reverse=True)
        elif session_selection_strategy == 'smallest':
            valid_sessions.sort(key=lambda x: x[1])
        elif session_selection_strategy == 'diverse':
            # Select sessions with diverse query counts
            valid_sessions.sort(key=lambda x: x[1])
            # Take every nth session to get diversity
            step = max(1, len(valid_sessions) // (max_sessions or len(valid_sessions)))
            valid_sessions = valid_sessions[::step]
        
        # Limit number of sessions if specified
        if max_sessions:
            valid_sessions = valid_sessions[:max_sessions]
        
        selected_session_ids = [session_id for session_id, _ in valid_sessions]
        
        logger.info(f"Selected {len(selected_session_ids)} sessions using '{session_selection_strategy}' strategy")
        return selected_session_ids
    
    def create_comprehensive_analysis_and_visualization(self, 
                                                       output_subdir: str = "analysis") -> Dict[str, Any]:
        """
        Create comprehensive analysis and visualizations for multi-session experiments.
        
        Args:
            output_subdir: Subdirectory for analysis outputs
            
        Returns:
            Dictionary with analysis results and file paths
        """
        try:
            from .experiment_analyzer import ExperimentAnalyzer
        except ImportError:
            # Fallback for direct execution
            from experiment_analyzer import ExperimentAnalyzer
        
        # Initialize analyzer with the experiment data
        analyzer = ExperimentAnalyzer(
            experiment_data_dir=str(self.collector.base_output_dir),
            output_dir=str(Path(self.collector.base_output_dir) / output_subdir),
            include_tuple_analysis=False
        )
        
        # Load all results
        results_df = analyzer.load_all_results()
        
        if results_df.empty:
            logger.warning("No experimental results found for analysis")
            return {"error": "No data for analysis"}
        
        logger.info(f"Loaded {len(results_df)} experimental results for analysis")
        
        # Create all visualizations
        analysis_results = {
            "data_summary": {
                "total_experiments": len(results_df),
                "unique_sessions": results_df['meta_session_id'].nunique(),
                "recommenders": results_df['meta_recommender_name'].unique().tolist(),
                "gaps_tested": sorted(results_df['meta_gap'].unique().tolist()) if 'meta_gap' in results_df.columns else [],
                "success_rate": (results_df['meta_status'] == 'completed').mean() if 'meta_status' in results_df.columns else 0.0
            }
        }
        
        try:
            # Generate interactive dashboard
            dashboard_fig = analyzer.generate_performance_dashboard("multi_session_performance_dashboard.html")
            analysis_results["dashboard_created"] = True
            
            # Create statistical summary
            stats_summary = analyzer.create_statistical_summary()
            analysis_results["statistical_summary"] = stats_summary
            
            # Create detailed comparison report
            analyzer.create_detailed_comparison_report("multi_session_comparison_report.html")
            analysis_results["comparison_report_created"] = True
            
            # Create publication-ready visualizations
            viz_files = analyzer.create_publication_visualizations(
                subdir="publication_figures",
                save_pdf=True,
                save_individual=True
            )
            analysis_results["publication_visualizations"] = viz_files
            
            # Export data for further analysis
            analyzer.export_for_further_analysis("data_exports")
            analysis_results["data_exported"] = True
            
            analysis_results["analysis_output_dir"] = str(analyzer.output_dir)
            
        except Exception as e:
            logger.error(f"Error during analysis and visualization: {str(e)}", exc_info=True)
            analysis_results["error"] = str(e)
        
        return analysis_results
    
    def load_experiment_config(self, config_path: str = "experiments/multi_session_config.yaml") -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'experiment_config': {
                    'session_selection': {
                        'min_queries': 3,
                        'max_sessions': 10,
                        'selection_strategy': 'diverse'
                    },
                    'experiment_params': {
                        'max_gap': 5,
                        'include_query_text': True,
                        'store_intermediate_states': False
                    }
                }
            }
    
    def _dataframe_to_tuple_set(self, df: pd.DataFrame):
        """Convert DataFrame to set of tuples for comparison."""
        return {tuple(row) for row in df.itertuples(index=False, name=None)}
    
    def _count_exact_matches(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> int:
        """Count exact tuple matches between predicted and actual."""
        return len(pd.merge(predicted, actual, how='inner'))
    
    def _hash_string(self, s: str) -> str:
        """Create hash of a string."""
        import hashlib
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def run_comprehensive_recommender_gap_analysis(self, 
                                                   session_id: Optional[str] = None,
                                                   max_gap: int = 5,
                                                   include_all_recommenders: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive gap analysis experiment comparing all recommenders.
        
        Args:
            session_id: Specific session to analyze (uses first available if None)
            max_gap: Maximum gap between queries to test
            include_all_recommenders: If True, test all recommenders; if False, only out-of-results methods
            
        Returns:
            Dictionary with detailed analysis comparing all recommenders
        """
        # Select session
        if session_id is None:
            sessions = self.dataloader.get_sessions()
            if not sessions:
                return {"error": "No sessions available"}
            session_id = sessions[0]  # Use first session
        
        logger.info(f"Running comprehensive recommender gap analysis on session {session_id}")
        
        # Start experiment session
        exp_session_id = self.collector.start_experiment_session(
            f"comprehensive_gap_analysis_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.current_session_id = exp_session_id
        
        # Get query information
        query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)
        if len(query_ids) < 2:
            return {"error": "Insufficient queries in session", "session_id": session_id}
        
        logger.info(f"Session {session_id} has {len(query_ids)} queries, testing up to gap {max_gap}")
        
        # Initialize results tracking
        gap_analysis_results = {
            "session_id": session_id,
            "experiment_session_id": exp_session_id,
            "total_queries": len(query_ids),
            "max_gap_tested": min(max_gap, len(query_ids) - 1),
            "gap_results": {},
            "recommender_details": {},
            "performance_summary": {}
        }
        
        # Select recommenders to test
        if include_all_recommenders:
            test_recommenders = self.recommenders
            logger.info(f"Testing all {len(test_recommenders)} recommenders")
        else:
            # Only test out-of-results methods
            test_recommenders = {
                'query_expansion': self.recommenders['query_expansion'],
                'random_table_baseline': self.recommenders['random_table_baseline']
            }
            logger.info("Testing only out-of-results recommenders")
        
        # Log which recommenders we're testing
        recommender_list = list(test_recommenders.keys())
        logger.info(f"Recommenders to test: {', '.join(recommender_list)}")
        
        total_experiments = 0
        successful_experiments = 0
        
        # Test each gap
        for gap in range(1, min(max_gap + 1, len(query_ids))):
            logger.info(f"Testing gap {gap}")
            
            gap_results = {
                "gap": gap,
                "query_pairs": [],
                "recommender_results": {},
                "average_metrics": {}
            }
            
            # Test all query pairs with this gap
            query_pairs = list(self.query_result_sequence.iter_query_result_pairs(session_id, gap))
            
            for current_id, future_id, current_results, future_results in query_pairs:
                if current_results.empty:
                    continue
                
                pair_info = {
                    "current_query": current_id,
                    "future_query": future_id,
                    "current_size": len(current_results),
                    "future_size": len(future_results),
                    "results": {}
                }
                
                # Test each recommender
                for recommender_name, recommender in test_recommenders.items():
                    total_experiments += 1
                    
                    try:
                        # Get recommendations
                        start_time = time.time()
                        recommendations = recommender.recommend_tuples(current_results)
                        execution_time = time.time() - start_time
                        
                        # Calculate metrics
                        overlap_accuracy = self.evaluator.overlap_accuracy(
                            previous=current_results,
                            actual=future_results,
                            predicted=recommendations
                        )
                        
                        precision, recall, f1 = self._calculate_precision_recall_f1(
                            recommendations, future_results, current_results
                        )
                        
                        roc_auc = self._calculate_roc_auc(
                            recommendations, future_results, current_results
                        )
                        
                        # Capture recommender-specific details
                        recommender_info = {}
                        if recommender_name == 'query_expansion' and hasattr(recommender, 'query_runner'):
                            query_runner = recommender.query_runner
                            if hasattr(query_runner, 'executed_queries'):
                                recommender_info = {
                                    "queries_executed": len(query_runner.executed_queries),
                                    "executed_queries": query_runner.executed_queries.copy(),
                                    "recommendation_count": len(recommendations),
                                    "recommender_type": "out_of_results"
                                }
                                # Clear executed queries for next iteration
                                query_runner.executed_queries.clear()
                        elif recommender_name == 'random_table_baseline' and hasattr(recommender, 'query_runner'):
                            query_runner = recommender.query_runner
                            if hasattr(query_runner, 'executed_queries'):
                                recommender_info = {
                                    "queries_executed": len(query_runner.executed_queries),
                                    "executed_queries": query_runner.executed_queries.copy(),
                                    "recommendation_count": len(recommendations),
                                    "recommender_type": "out_of_results_baseline"
                                }
                                # Clear executed queries for next iteration
                                query_runner.executed_queries.clear()
                        else:
                            # In-results recommenders
                            recommender_info = {
                                "recommendation_count": len(recommendations),
                                "recommender_type": "in_results"
                            }
                        
                        # Create contexts for collector
                        current_context = QueryContext(
                            session_id=session_id,
                            query_position=current_id,
                            query_text=f"Query {current_id} from session {session_id}",
                            query_hash=self._hash_string(f"Query {current_id}"),
                            result_set_size=len(current_results)
                        )
                        
                        future_context = QueryContext(
                            session_id=session_id,
                            query_position=future_id,
                            query_text=f"Query {future_id} from session {session_id}",
                            query_hash=self._hash_string(f"Query {future_id}"),
                            result_set_size=len(future_results)
                        )
                        
                        # Create recommendation result
                        rec_result = RecommendationResult(
                            experiment_id="",  # Will be generated
                            predicted_tuples=recommendations,
                            recommendation_metadata={
                                "recommender_type": recommender_name,
                                "input_size": len(current_results),
                                "output_size": len(recommendations),
                                "recommender_info": recommender_info
                            }
                        )
                        
                        # Create evaluation result
                        eval_result = EvaluationResult(
                            experiment_id="",  # Will be generated
                            overlap_accuracy=overlap_accuracy,
                            jaccard_similarity=self.evaluator.jaccard_similarity(recommendations, future_results),
                            precision=precision,
                            recall=recall,
                            f1_score=f1,
                            exact_matches=self._count_exact_matches(recommendations, future_results),
                            predicted_count=len(recommendations),
                            actual_count=len(future_results),
                            intersection_count=len(pd.merge(recommendations, future_results, how='inner')),
                            union_count=len(pd.concat([recommendations, future_results]).drop_duplicates()),
                            roc_auc=roc_auc
                        )
                        
                        # Collect the experiment using the experiment collector
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
                        
                        result_info = {
                            "experiment_id": experiment_id,
                            "execution_time": execution_time,
                            "recommendation_count": len(recommendations),
                            "overlap_accuracy": overlap_accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "roc_auc": roc_auc,
                            "recommender_info": recommender_info
                        }
                        
                        pair_info["results"][recommender_name] = result_info
                        successful_experiments += 1
                        
                        logger.debug(f"Gap {gap}, {recommender_name}: accuracy={overlap_accuracy:.3f}, "
                                   f"precision={precision:.3f}, recall={recall:.3f}, time={execution_time:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Error testing {recommender_name} on gap {gap}: {e}")
                        pair_info["results"][recommender_name] = {"error": str(e)}
                
                gap_results["query_pairs"].append(pair_info)
            
            # Calculate average metrics for this gap
            for recommender_name in test_recommenders.keys():
                metrics = []
                for pair in gap_results["query_pairs"]:
                    if recommender_name in pair["results"] and "error" not in pair["results"][recommender_name]:
                        metrics.append(pair["results"][recommender_name])
                
                if metrics:
                    avg_metrics = {
                        "count": len(metrics),
                        "avg_accuracy": np.mean([m["overlap_accuracy"] for m in metrics]),
                        "avg_precision": np.mean([m["precision"] for m in metrics]),
                        "avg_recall": np.mean([m["recall"] for m in metrics]),
                        "avg_f1": np.mean([m["f1_score"] for m in metrics]),
                        "avg_execution_time": np.mean([m["execution_time"] for m in metrics]),
                        "total_recommendations": sum([m["recommendation_count"] for m in metrics])
                    }
                    
                    # Add recommender-specific metrics for out-of-results methods
                    if recommender_name in ['query_expansion', 'random_table_baseline']:
                        out_of_results_metrics = [m["recommender_info"] for m in metrics if m["recommender_info"] and "queries_executed" in m["recommender_info"]]
                        if out_of_results_metrics:
                            avg_metrics.update({
                                "avg_queries_executed": np.mean([e["queries_executed"] for e in out_of_results_metrics]),
                                "total_database_queries": sum([e["queries_executed"] for e in out_of_results_metrics]),
                                "recommender_category": "out_of_results"
                            })
                    else:
                        # In-results recommenders
                        avg_metrics["recommender_category"] = "in_results"
                    
                    gap_results["average_metrics"][recommender_name] = avg_metrics
            
            gap_analysis_results["gap_results"][gap] = gap_results
        
        # Create performance summary
        gap_analysis_results["performance_summary"] = {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0,
            "recommenders_tested": list(test_recommenders.keys())
        }
        
        # Add gap-wise performance trends
        gap_performance = {}
        for recommender_name in test_recommenders.keys():
            performance_by_gap = {}
            for gap, gap_result in gap_analysis_results["gap_results"].items():
                if recommender_name in gap_result["average_metrics"]:
                    metrics = gap_result["average_metrics"][recommender_name]
                    performance_by_gap[gap] = {
                        "accuracy": metrics["avg_accuracy"],
                        "precision": metrics["avg_precision"],
                        "recall": metrics["avg_recall"],
                        "f1": metrics["avg_f1"]
                    }
            gap_performance[recommender_name] = performance_by_gap
        
        gap_analysis_results["gap_performance_trends"] = gap_performance
        
        logger.info(f"Query Expansion gap analysis completed: {successful_experiments}/{total_experiments} successful")
        logger.info(f"Results stored in experiment session: {exp_session_id}")
        
        return gap_analysis_results


class TimeoutError(Exception):
    pass


def run_comprehensive_recommender_experiment():
    """Run a comprehensive experiment comparing QueryExpansionRecommender, RandomTableRecommender, and in-results recommenders."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"results/experiment/comprehensive_recommender_test_{timestamp}"
    
    runner = RecommenderExperimentRunner(
        output_dir=output_dir,
        enable_full_tuple_storage=True,
        enable_state_tracking=False
    )
    
    # Get session statistics
    session_stats = runner.get_session_statistics()
    logger.info("Session Statistics:")
    logger.info(f"  Total sessions: {session_stats['total_sessions']}")
    logger.info(f"  Sessions with multiple queries: {session_stats['sessions_with_multiple_queries']}")
    
    if session_stats['total_sessions'] == 0:
        logger.error("No sessions found in dataset")
        return
    
    # Use the first session for testing
    sessions = runner.dataloader.get_sessions()
    test_session = sessions[0]
    
    logger.info(f"Running comprehensive recommender experiment on session {test_session}")
    
    # Run the comprehensive gap analysis
    results = runner.run_comprehensive_recommender_gap_analysis(
        session_id=test_session,
        max_gap=5,  # Test gaps 1-5
        include_all_recommenders=True  # Test all recommenders
    )
    
    if "error" in results:
        logger.error(f"Experiment failed: {results['error']}")
        return
    
    # Print detailed results
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE RECOMMENDER COMPARISON - GAP ANALYSIS RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"Session ID: {results['session_id']}")
    logger.info(f"Total Queries: {results['total_queries']}")
    logger.info(f"Max Gap Tested: {results['max_gap_tested']}")
    logger.info(f"Success Rate: {results['performance_summary']['success_rate']:.1%}")
    logger.info(f"Recommenders Tested: {', '.join(results['performance_summary']['recommenders_tested'])}")
    
    # Performance by gap and recommender
    logger.info("\nPerformance by Gap and Recommender:")
    logger.info("-" * 60)
    
    # Organize results by recommender category
    out_of_results_recommenders = []
    in_results_recommenders = []
    
    for gap in sorted(results['gap_results'].keys()):
        logger.info(f"\nGap {gap} Results:")
        gap_data = results['gap_results'][gap]
        
        for recommender_name, metrics in gap_data['average_metrics'].items():
            category = metrics.get('recommender_category', 'unknown')
            
            if category == 'out_of_results':
                if recommender_name not in out_of_results_recommenders:
                    out_of_results_recommenders.append(recommender_name)
                marker = "🔍"  # Out-of-results methods
            else:
                if recommender_name not in in_results_recommenders:
                    in_results_recommenders.append(recommender_name)
                marker = "📊"  # In-results methods
            
            logger.info(f"  {marker} {recommender_name}: "
                       f"Accuracy={metrics['avg_accuracy']:.3f}, "
                       f"Precision={metrics['avg_precision']:.3f}, "
                       f"Recall={metrics['avg_recall']:.3f}, "
                       f"F1={metrics['avg_f1']:.3f}, "
                       f"Time={metrics['avg_execution_time']:.3f}s")
            
            # Additional details for out-of-results methods
            if 'avg_queries_executed' in metrics:
                logger.info(f"    Database Queries={metrics['avg_queries_executed']:.1f}, "
                           f"Recommendations={metrics['total_recommendations']}")
    
    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY COMPARISON")
    logger.info("=" * 80)
    
    logger.info(f"\nOut-of-Results Methods ({len(out_of_results_recommenders)}):")
    for rec in out_of_results_recommenders:
        logger.info(f"  🔍 {rec}")
    
    logger.info(f"\nIn-Results Methods ({len(in_results_recommenders)}):")
    for rec in in_results_recommenders:
        logger.info(f"  📊 {rec}")
    
    # Calculate overall performance averages
    logger.info("\nOverall Performance Averages:")
    logger.info("-" * 40)
    
    all_metrics = {}
    for gap_data in results['gap_results'].values():
        for recommender_name, metrics in gap_data['average_metrics'].items():
            if recommender_name not in all_metrics:
                all_metrics[recommender_name] = []
            all_metrics[recommender_name].append(metrics)
    
    # Sort by category and then by average accuracy
    recommender_performance = []
    for recommender_name, metrics_list in all_metrics.items():
        avg_accuracy = np.mean([m['avg_accuracy'] for m in metrics_list])
        avg_time = np.mean([m['avg_execution_time'] for m in metrics_list])
        category = metrics_list[0].get('recommender_category', 'unknown')
        
        recommender_performance.append({
            'name': recommender_name,
            'category': category,
            'avg_accuracy': avg_accuracy,
            'avg_time': avg_time
        })
    
    # Sort by category (out-of-results first) then by accuracy
    recommender_performance.sort(key=lambda x: (x['category'] != 'out_of_results', -x['avg_accuracy']))
    
    for perf in recommender_performance:
        marker = "🔍" if perf['category'] == 'out_of_results' else "📊"
        logger.info(f"  {marker} {perf['name']:<25} "
                   f"Accuracy: {perf['avg_accuracy']:.3f}  "
                   f"Time: {perf['avg_time']:.3f}s")
    
    # Best performers
    logger.info("\nBest Performers:")
    logger.info("-" * 40)
    
    best_overall = max(recommender_performance, key=lambda x: x['avg_accuracy'])
    fastest = min(recommender_performance, key=lambda x: x['avg_time'])
    
    best_out_of_results = max([p for p in recommender_performance if p['category'] == 'out_of_results'], 
                             key=lambda x: x['avg_accuracy'], default=None)
    best_in_results = max([p for p in recommender_performance if p['category'] == 'in_results'], 
                         key=lambda x: x['avg_accuracy'], default=None)
    
    logger.info(f"  Best Overall: {best_overall['name']} (Accuracy: {best_overall['avg_accuracy']:.3f})")
    logger.info(f"  Fastest: {fastest['name']} (Time: {fastest['avg_time']:.3f}s)")
    
    if best_out_of_results:
        logger.info(f"  Best Out-of-Results: {best_out_of_results['name']} (Accuracy: {best_out_of_results['avg_accuracy']:.3f})")
    if best_in_results:
        logger.info(f"  Best In-Results: {best_in_results['name']} (Accuracy: {best_in_results['avg_accuracy']:.3f})")
    
    logger.info(f"\nResults stored in: {runner.collector.base_output_dir}")
    logger.info("Comprehensive recommender experiment completed successfully!")


# Legacy function name for backward compatibility
def run_query_expansion_experiment():
    """Legacy function - now runs comprehensive experiment."""
    logger.info("Note: run_query_expansion_experiment now runs comprehensive comparison")
    run_comprehensive_recommender_experiment()


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
    
    # Get session statistics
    session_stats = runner.get_session_statistics()
    logger.info("Session Statistics:")
    logger.info(f"  Total sessions: {session_stats['total_sessions']}")
    logger.info(f"  Sessions with multiple queries: {session_stats['sessions_with_multiple_queries']}")
    logger.info(f"  Query count range: {session_stats['query_count_distribution']['min']}-{session_stats['query_count_distribution']['max']}")
    
    if session_stats['total_sessions'] == 0:
        logger.error("No sessions found in dataset", exc_info=True)
        return
    
    # Select sessions for experimentation
    selected_sessions = runner.select_sessions_for_experiment(
        min_queries=3,           # Minimum 3 queries per session
        max_sessions=10,         # Limit to 10 sessions for testing
        session_selection_strategy='diverse'  # Get diverse session sizes
    )
    
    if not selected_sessions:
        logger.error("No suitable sessions found for experimentation", exc_info=True)
        return
    
    logger.info(f"Selected {len(selected_sessions)} sessions for experimentation")
    
    # Run multi-session experiments
    experiment_summary = runner.run_multi_session_experiments(
        session_ids=selected_sessions,
        max_gap=5,  # Start with smaller gap for testing
        include_query_text=True,
        store_intermediate_states=False
    )
    
    logger.info("Multi-Session Experiment Summary:")
    for key, value in experiment_summary.items():
        if key not in ['experiment_ids', 'session_summaries']:  # Don't print long lists
            logger.info(f"  {key}: {value}")
    
    # Create comprehensive analysis and visualizations
    logger.info("Creating comprehensive analysis and visualizations...")
    analysis_results = runner.create_comprehensive_analysis_and_visualization()
    
    logger.info("Analysis Results:")
    for key, value in analysis_results.items():
        if key not in ['statistical_summary', 'publication_visualizations']:
            logger.info(f"  {key}: {value}")
    
    # Create comparative analysis
    analysis = runner.create_comparative_analysis()
    logger.info("Comparative Analysis Summary:")
    if 'experiment_overview' in analysis:
        overview = analysis['experiment_overview']
        logger.info(f"  Total experiments: {overview.get('total_experiments', 0)}")
        logger.info(f"  Unique sessions: {overview.get('unique_sessions', 0)}")
        logger.info(f"  Success rate: {overview.get('success_rate', 0):.2%}")
    
    logger.info("Multi-session experiment completed successfully!")
    logger.info(f"Results stored in: {runner.collector.base_output_dir}")
    logger.info(f"Analysis outputs in: {analysis_results.get('analysis_output_dir', 'Not available')}")


def run_single_session_test():
    """Quick single-session test for debugging and development."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"results/experiment/single_session_test_{timestamp}"
    
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
    
    session_id = sessions[2]
    logger.info(f"Running single-session test on session {session_id}")
    
    # Run enhanced experiments
    experiment_summary = runner.run_enhanced_experiment(
        session_id=session_id, 
        max_gap=3,  # Smaller gap for testing
        include_query_text=True,
        store_intermediate_states=False
    )
    
    logger.info("Single-Session Test Summary:")
    for key, value in experiment_summary.items():
        if key != 'experiment_ids':  # Don't print long list
            logger.info(f"  {key}: {value}")
    
    # Create basic analysis
    analysis = runner.create_comparative_analysis()
    logger.info("Basic Analysis:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Single-session test completed successfully!")
    logger.info(f"Results stored in: {runner.collector.base_output_dir}")


# Add command-line argument handling
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            run_single_session_test()
        elif sys.argv[1] == "comprehensive" or sys.argv[1] == "query_expansion":
            run_comprehensive_recommender_experiment()
        else:
            print("Usage: python recommender_experiments.py [single|comprehensive|query_expansion]")
            print("  single: Run single-session test")
            print("  comprehensive: Run comprehensive recommender comparison (includes query expansion, random table baseline, and in-results methods)")
            print("  query_expansion: Same as comprehensive (legacy alias)")
            print("  (no args): Run full multi-session experiments")
    else:
        main()
