

import pandas as pd
import yaml
from pathlib import Path
import logging
from datetime import datetime

from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.metrics import EvaluationMetrics
from query_data_predictor.recommender import (
    DummyRecommender,
    RandomRecommender,
    ClusteringRecommender,
    InterestingnessRecommender
)

from query_data_predictor.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class RecencyEffectExperimentRunner:
    def __init__(self, config_path: str, dataset_dir: str):
        self.config = self._load_config(config_path)
        self.dataset_dir = Path(dataset_dir)
        self.dataloader = DataLoader(str(self.dataset_dir))
        self.query_sequence = QueryResultSequence(self.dataloader)
        self.evaluator = EvaluationMetrics()
        self.results = []
        self.recommenders = self._initialize_recommenders()

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_recommenders(self):
        return {
            'dummy': DummyRecommender(self.config),
            'random': RandomRecommender(self.config),
            'clustering': ClusteringRecommender(self.config),
            'interestingness': InterestingnessRecommender(self.config)
        }

    def run_experiment(self, session_id: str, max_gap: int = 5):
        logger.info(f"Starting recency effect experiment on session {session_id} with max_gap {max_gap}")
        query_ids = self.query_sequence.get_ordered_query_ids(session_id)
        if len(query_ids) < 2:
            logger.warning(f"Session {session_id} has fewer than 2 queries, skipping")
            return

        for gap in range(1, min(max_gap + 1, len(query_ids))):
            logger.info(f"Testing gap {gap}")
            for current_id, future_id, current_results, future_results in \
                self.query_sequence.iter_query_result_pairs(session_id, gap):
                if current_results.empty:
                    continue
                for recommender_name, recommender in self.recommenders.items():
                    self._evaluate_recommender(
                        session_id, current_id, future_id, current_results, future_results, recommender_name, recommender, gap
                    )

        results_df = pd.DataFrame(self.results)
        return results_df

    def _evaluate_recommender(self, session_id, current_query_id, future_query_id, current_results, future_results, recommender_name, recommender, gap):
        try:
            recommendations = recommender.recommend_tuples(current_results)
            overlap_accuracy = self.evaluator.overlap_accuracy(
                previous=current_results,
                actual=future_results,
                predicted=recommendations
            )
            self.results.append({
                'session_id': session_id,
                'gap': gap,
                'recommender': recommender_name,
                'overlap_accuracy': overlap_accuracy
            })
        except Exception as e:
            logger.error(f"Error evaluating {recommender_name} for gap {gap}: {e}")

def main():
    config_path = "experiments/configs/recency_effect_config.yaml"
    dataset_dir = "data/datasets"
    output_dir = Path("experiment_results/recency_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = RecencyEffectExperimentRunner(config_path, dataset_dir)
    sessions = runner.dataloader.get_sessions()
    if not sessions:
        logger.error("No sessions found")
        return

    session_id = sessions[0]
    results_df = runner.run_experiment(session_id, max_gap=5)

    if not results_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"recency_effect_results_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
