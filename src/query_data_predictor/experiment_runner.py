"""
Main experiment runner for the query results prediction framework.
"""

import os
import pandas as pd
import numpy as np
import warnings
import pickle
import json
import signal
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv

from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.config_manager import ConfigManager
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
    FrequencyRecommender,
    SamplingRecommender,
    MultiDimensionalInterestingnessRecommender,
    KernelDensityRecommender,
)
from query_data_predictor.query_runner import QueryRunner

from contextlib import contextmanager

import logging


logger = logging.getLogger(__name__) 

class ExperimentRunner:
    """
    Main class for running experiments and evaluating query predictions.
    """
    def __init__(self, output_dir: Path, data_path: Path, sessions: List, gap: List, config: Dict[str, Any]):

        self.dataset_dir = data_path
        self.sessions = sessions
        self.gap = gap
        self.config = config
        self.output_dir = output_dir
        # make actual output directory a timestamped directory with experiment name from config
        experiment_name = self.config.get('experiment', {}).get('name', 'experiment')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataloader = DataLoader(str(self.dataset_dir))
        self.query_result_sequence = QueryResultSequence(self.dataloader)
        self.metrics = EvaluationMetrics(config['evaluation'])
        self.query_runner = None # only initalise if we need to
        self.recommenders = self._initialize_recommenders()
        logger.info(f"Initialized ExperimentRunner with config: {self.config}")

    def run_experiment(self):

        if len(self.sessions) == 0:
            # get all session IDs from the data loader
            self.sessions = self.dataloader.get_sessions()
        for session in self.sessions:
            self.run_session_experiment(session)
        return

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
        include_query_text = self.config.get('experiment', {}).get('include_query_text', False)
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
        }
        # if reccomendation mode is cheating set topk to be length of future results otherwise let reccomenders determine   
        top_k = len(future_results) if self.config.get('experiment', {}).get('mode', '') == 'cheating' else None
        # TODO this should probably go somewhere else
        try:
            timeout_seconds = 30 if len(current_results) < 100 else 120
            with self._timeout(timeout_seconds):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    recommended_results = recommender.recommend_tuples(current_results, top_k=top_k, current_query_text=current_query_text, future_query_text=future_query_text)
            execution_time = time.time() - start_time
            result_record["recommended_results"] = recommended_results.to_dict("records") if isinstance(recommended_results, pd.DataFrame) else recommended_results
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
            'frequency': FrequencyRecommender,
            'sampling': SamplingRecommender,
            'query_expansion': QueryExpansionRecommender,
            'random_table_baseline': RandomTableRecommender,
            'multidimensional_interestingness': MultiDimensionalInterestingnessRecommender,
            'kernel_density': KernelDensityRecommender,
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
                    if name in ['query_expansion', 'random_table_baseline', 'kernel_density']:
                        query_runner = self._get_query_runner()
                        initialized_recommenders[name] = recommender_class(self.config, query_runner=query_runner)
                        logger.warning(f"Skipping {name} - QueryRunner implementation needed")
                        continue
                    else:
                        initialized_recommenders[name] = recommender_class(self.config)
                    
                    logger.info(f"Initialized recommender: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize recommender {name}: {str(e)}")
            else:
                logger.warning(f"Unknown recommender '{name}' specified in config")
    
        logger.info(f"Initialized {len(initialized_recommenders)} recommenders: {list(initialized_recommenders.keys())}")
        return initialized_recommenders

    # TODO: add overrides for dotenv stuff elsewhere
    def _get_query_runner(self) -> QueryRunner:
        if self.query_runner == None: 
            # TODO have this actually mean something
            query_runner_config = self.config.get('query_runner', {})
            load_dotenv()
            DB_NAME = os.getenv("PG_DATA")
            DB_USER = os.getenv("PG_DATA_USER")
            DB_HOST = os.getenv("PG_HOST", "localhost")
            DB_PORT = os.getenv("PG_PORT", "5432")
            self.query_runner = QueryRunner(
                DB_NAME,
                DB_USER,
                host=DB_HOST,
                port=DB_PORT,
                **query_runner_config
            )
            self.query_runner.connect()
        return self.query_runner

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
