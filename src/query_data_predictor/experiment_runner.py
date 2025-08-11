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


logger = logging.getLogger(__name__) 

class ExperimentRunner:
    """
    Main class for running experiments and evaluating query predictions.
    """
    def __init__(self, output_dir, data_path, sessions, gap, config: Dict[str, Any]):
        self.output_dir = output_dir
        self.data_path = data_path
        self.sessions = sessions
        self.gap = gap
        self.config = config
        self.data_loader = DataLoader(config['data_path'])
        self.query_sequence = QueryResultSequence(self.data_loader)
        self.recommender = TupleRecommender(config['recommender'])
        self.metrics = EvaluationMetrics(config['evaluation'])
