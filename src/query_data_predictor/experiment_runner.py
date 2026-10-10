"""
Main experiment runner for the query results prediction framework.
"""

import os
import pandas as pd
import warnings
import json
import signal
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.metrics import EvaluationMetrics

from query_data_predictor.recommender import (
    BaseRecommender,
    DummyRecommender,
    RandomRecommender,
    ClusteringRecommender,
    InterestingnessRecommender,
    QueryExpansionRecommender,
    RandomTableRecommender,
    SimilarityRecommender,
    OutOfResultSimilarityRecommender,
    FrequencyRecommender,
    SamplingRecommender,
    MultiDimensionalInterestingnessRecommender,
    KernelDensityRecommender,
    TemporalInterestRecommender,
    ExploratoryInterestRecommender,
)
from query_data_predictor.query_runner import QueryRunner

from contextlib import contextmanager


logger = logging.getLogger(__name__) 

class ExperimentRunner:
    """
    Main class for running experiments and evaluating query predictions.
    """
    def __init__(self, output_dir: Path, dataset: str, sessions: List, gap: List, config: Dict[str, Any]):

        self.dataset = dataset
        self.sessions = sessions
        self.gap = gap
        self.config = config
        self.output_dir = output_dir
        # make actual output directory a timestamped directory with experiment name from config
        experiment_name = self.config.get('experiment', {}).get('name', 'experiment')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.query_runner = QueryRunner(
            **self._query_runner_params(),
            cache_dir=Path("data") / self.dataset,
        )
        self.query_result_sequence = QueryResultSequence(
            Path("queries") / self.dataset / "queries.csv",
            self.query_runner,
        )
        evaluation = config.get('evaluation', {})
        self.metrics = EvaluationMetrics(
            jaccard_threshold=evaluation.get('jaccard_threshold', 0.5),
            column_weights=evaluation.get('column_weights'),
            identity_columns=evaluation.get('identity_columns'),
        )
        self.recommenders = self._initialize_recommenders()
        logger.info(f"Initialized ExperimentRunner with config: {self.config}")

    def run_experiment(self):
        try:
            if len(self.sessions) == 0:
                self.sessions = self.query_result_sequence.get_sessions()
            for session in self.sessions:
                self.run_session_experiment(session)
        finally:
            self._write_query_errors()
            self.query_runner.disconnect()

    def run_session_experiment(self, session_id: str):

        logger.info(f"Running session experiment for session: {session_id}")
        query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)

        if len(query_ids) < 2:
            logger.warning(f"Session {session_id} has fewer than 2 queries, skipping")
            return {"error": "Insufficient queries", "session_id": session_id}

        if len(self.recommenders) == 0:
            logger.warning(f"Session {session_id} has no recommenders, skipping")
            return {"error": "No recommenders available", "session_id": session_id}

        logger.info(f"Session {session_id} has {len(query_ids)} queries")

        for gap in self.gap:

            logger.info(f"Running session {session_id} with gap {gap}")
            self.session_predict_with_gap(session_id, gap)

        return {"success": True, "session_id": session_id}

    def session_predict_with_gap(self, session_id: str, gap: int) -> Dict[str, Any]:
        store_intermediate_states = self.config.get('experiment', {}).get('store_intermediate_states', False)
        results = []
        try:
            self._reset_recommender_state_for_benchmark()

            # Iterate through all valid query pairs with this gap
            for current_id, future_id, current_results, future_results, current_query_text, future_query_text in \
                self.query_result_sequence.iter_query_result_pairs_with_text(session_id, gap):
                # Skip if current results are empty
                if current_results.empty:
                    continue
                # Get the target size (number of tuples in the future query)
                target_size = len(future_results)
                logger.debug(f"Testing with target size {target_size} for query pair {current_id}->{future_id}")
                # Test each recommender with the adaptive target size
                for recommender_name, recommender in self.recommenders.items():
                    logger.debug(f"Running {recommender_name} with gap {gap}")
                    result = self.get_results(
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
                    results.append(result)

            # Write all results for this session/gap to a single file
            filename = f"{session_id}__gap-{gap}.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Wrote all results for session {session_id} gap {gap} to {filepath}")
        except Exception as e:
            logger.error(f"Error in adaptive size gap {gap} experiment for session {session_id}: {str(e)}", exc_info=True)
        return

    def _reset_recommender_state_for_benchmark(self) -> None:
        """
        Reset temporal recommender state before each independent benchmark slice.

        This prevents recommenders with historical memory from leaking state across
        sessions or gap settings, which would otherwise distort results and grow
        per-query runtime over the full reproduction run.
        """
        for recommender_name, recommender in self.recommenders.items():
            clear_history = getattr(recommender, "clear_history", None)
            if callable(clear_history):
                clear_history()
                logger.debug(f"Reset historical state for recommender '{recommender_name}'")
    
    def get_results(self, 
            session_id: str,
            current_query_id: str, 
            future_query_id: str,
            current_results: pd.DataFrame,
            future_results: pd.DataFrame,
            current_query_text: str,
            future_query_text: str,
            recommender_name: str,
            recommender: BaseRecommender,
            gap: int,
            store_states: bool = False) -> Optional[str]:
        """Evaluate a single recommender and write results to disk in interpretable format."""
        start_time = time.time()

        result_record = {
            "session_id": session_id,
            "current_query_id": current_query_id,
            "future_query_id": future_query_id,
            "gap": gap,
            "recommender_name": recommender_name,
            "current_results": current_results.to_dict("records"),
            "future_results": future_results.to_dict("records"),
            "recommended_results": None,
            "current_query_text": current_query_text,
            "future_query_text": future_query_text,
            "execution_time": None,
            "timestamp": None,
            "error_message": None,
            "candidate_count": 0,
            "candidate_recall": 0.0,
            "novel_candidate_recall": 0.0,
            "novel_precision": 0.0,
            "novel_recall": 0.0,
            "novel_f1": 0.0,
            "novel_count": 0,
            "ndcg": 0.0,
        }
        # 'cheating' sizes the recommendation from the future result, which is not
        # available at prediction time; it is kept only to reproduce the legacy
        # protocol and is labelled in every record. Otherwise recommenders use the
        # configured budget.
        cheating = self.config.get('experiment', {}).get('mode', '') == 'cheating'
        top_k = len(future_results) if cheating else None
        result_record["budget_protocol"] = "legacy_future_size" if cheating else "configured"
        result_record["requested_k"] = top_k
        # TODO this should probably go somewhere else
        try:
            timeout_seconds = 30 if len(current_results) < 100 else 120
            with self._timeout(timeout_seconds):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    recommended_results = recommender.recommend_tuples(
                        current_results,
                        top_k=top_k,
                        current_query_text=current_query_text,
                        current_query=current_query_text,
                        prediction_gap=gap,
                    )
            execution_time = time.time() - start_time
            result_record["recommended_results"] = recommended_results.to_dict("records") if isinstance(recommended_results, pd.DataFrame) else recommended_results
            if isinstance(recommended_results, pd.DataFrame):
                candidates = getattr(recommender, "last_candidates", current_results)
                if not isinstance(candidates, pd.DataFrame):
                    candidates = current_results
                novel = self.metrics.novel_metrics(recommended_results, current_results, future_results)
                candidate_novel = self.metrics.novel_metrics(candidates, current_results, future_results)
                result_record.update({
                    "candidate_count": len(candidates.drop_duplicates()),
                    "candidate_recall": self.metrics.recall(candidates, future_results),
                    "novel_candidate_recall": candidate_novel["recall"],
                    "novel_precision": novel["precision"],
                    "novel_recall": novel["recall"],
                    "novel_f1": novel["f1"],
                    "novel_count": novel["count"],
                    "ndcg": self.metrics.ndcg_at_k(recommended_results, future_results, len(recommended_results)),
                })
            result_record["execution_time"] = execution_time
            result_record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            if "Timed out" in error_msg:
                logger.error(f"Timeout for {recommender_name} on gap {gap}: {error_msg}", exc_info=True)
                result_record["error_message"] = f"Timeout - {error_msg}"
            else:
                logger.error(f"Error evaluating {recommender_name} for gap {gap}: {error_msg}", exc_info=True)
                result_record["error_message"] = f"Error - {error_msg}"

        return result_record
     

    def _initialize_recommenders(self) -> Dict[str, BaseRecommender]:
        """Initialize only the recommenders specified in the experiment config."""
        
        # Define all available recommenders with their classes
        available_recommenders = {
            'dummy': DummyRecommender,
            'random': RandomRecommender,
            'clustering': ClusteringRecommender,
            'interestingness': InterestingnessRecommender,
            'similarity': SimilarityRecommender,
            'out_of_result_similarity': OutOfResultSimilarityRecommender,
            'frequency': FrequencyRecommender,
            'sampling': SamplingRecommender,
            'query_expansion': QueryExpansionRecommender,
            'random_table_baseline': RandomTableRecommender,
            'multidimensional_interestingness': MultiDimensionalInterestingnessRecommender,
            'kernel_density': KernelDensityRecommender,
            'temporal_interest': TemporalInterestRecommender,
            'exploratory_interest': ExploratoryInterestRecommender,
        }
        
        # Get the list of recommenders from config
        recommender_names = self.config.get('experiment', {}).get('recommenders', [])
        
        # Initialize only the specified recommenders
        initialized_recommenders = {}
        for name in recommender_names:
            if name in available_recommenders:
                try:
                    recommender_class = available_recommenders[name]
                    
                    # Handle recommenders that need special initialization (QueryRunner)
                    if name in ['query_expansion', 'random_table_baseline', 'kernel_density', 'exploratory_interest', 'out_of_result_similarity']:
                        recommender = recommender_class(self.config, query_runner=self.query_runner)
                    else:
                        recommender = recommender_class(self.config)
                    recommender.query_runner = self.query_runner
                    initialized_recommenders[name] = recommender
                    
                    logger.info(f"Initialized recommender: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize recommender {name}: {str(e)}")
            else:
                logger.warning(f"Unknown recommender '{name}' specified in config")
    
        logger.info(f"Initialized {len(initialized_recommenders)} recommenders: {list(initialized_recommenders.keys())}")
        return initialized_recommenders

    def _query_runner_params(self) -> Dict[str, Any]:
        load_dotenv()
        db_config_keys = {
            "dbname", "user", "password", "host", "port", "statement_timeout_seconds"
        }
        params = {
            "dbname": os.getenv("PG_DATA"),
            "user": os.getenv("PG_DATA_USER"),
            "password": os.getenv("PG_SESSION_PASSWORD"),
            "host": os.getenv("PG_HOST", "localhost"),
            "port": os.getenv("PG_PORT", "5432"),
        }
        params.update({
            k: v
            for k, v in self.config.get("query_runner", {}).items()
            if k in db_config_keys and v is not None
        })
        return {k: v for k, v in params.items() if v is not None}

    def _write_query_errors(self) -> None:
        errors = list(self.query_result_sequence.errors.values())
        if not errors:
            return
        with open(self.output_dir / "query_errors.json", "w") as file:
            json.dump(errors, file, indent=2, default=str)
        logger.warning("Recorded %s workload query errors", len(errors))

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
