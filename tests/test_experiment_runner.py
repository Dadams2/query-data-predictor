from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from query_data_predictor.experiment_runner import ExperimentRunner


class _StatefulRecommender:
    def __init__(self):
        self.clear_history_calls = 0

    def clear_history(self):
        self.clear_history_calls += 1


def test_session_predict_with_gap_resets_stateful_recommenders(tmp_path):
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = {'experiment': {}}
    runner.output_dir = Path(tmp_path)
    runner.recommenders = {
        'stateful': _StatefulRecommender(),
        'stateless': object(),
    }
    runner.get_results = MagicMock(return_value={'ok': True})

    current_results = pd.DataFrame({'a': [1, 2]})
    future_results = pd.DataFrame({'a': [2]})

    query_result_sequence = MagicMock()
    query_result_sequence.iter_query_result_pairs_with_text.return_value = [
        (1, 2, current_results, future_results, 'select 1', 'select 2')
    ]
    runner.query_result_sequence = query_result_sequence

    runner.session_predict_with_gap('session-1', 10)

    assert runner.recommenders['stateful'].clear_history_calls == 1
    assert runner.get_results.call_count == 2
