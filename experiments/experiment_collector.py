"""
Enhanced experimental data collection system for recommender evaluation.

This module provides a comprehensive framework for collecting, storing, and analyzing
recommender system experiments with detailed provenance and structured results.
"""

import pandas as pd
import numpy as np
import json
import pickle
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experimental run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RecommendationType(Enum):
    """Type of recommendation being evaluated."""
    NEXT_QUERY = "next_query"  # Predict next query results
    FUTURE_QUERY = "future_query"  # Predict future query results (with gap)
    SEQUENCE_COMPLETION = "sequence_completion"  # Predict remaining sequence


@dataclass
class ExperimentMetadata:
    """Metadata for a single experimental run."""
    experiment_id: str
    timestamp: datetime
    session_id: str
    current_query_position: int
    target_query_position: int
    gap: int  # Distance between current and target
    recommender_name: str
    recommender_version: str
    recommender_config_hash: str
    recommendation_type: RecommendationType
    status: ExperimentStatus
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable types."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['recommendation_type'] = self.recommendation_type.value
        data['status'] = self.status.value
        return data


@dataclass 
class QueryContext:
    """Context information about a query in the sequence."""
    session_id: int
    query_position: int
    query_text: str
    query_hash: str
    result_set_size: int
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RecommendationResult:
    """Results from a recommendation request."""
    experiment_id: str
    predicted_tuples: pd.DataFrame  # The recommended tuples
    confidence_scores: Optional[pd.Series] = None  # Optional confidence per tuple
    recommendation_metadata: Optional[Dict[str, Any]] = None  # Algorithm-specific metadata
    internal_state_hash: Optional[str] = None  # Hash of recommender internal state
    
    def get_tuple_count(self) -> int:
        """Get number of predicted tuples."""
        return len(self.predicted_tuples)
    
    def get_tuple_hashes(self) -> List[str]:
        """Get hash of each predicted tuple for exact matching."""
        return [
            hashlib.md5(str(tuple(row)).encode()).hexdigest()
            for _, row in self.predicted_tuples.iterrows()
        ]


@dataclass
class EvaluationResult:
    """Results from evaluating a recommendation against ground truth."""
    experiment_id: str
    
    # Core metrics
    overlap_accuracy: float
    jaccard_similarity: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float  # Added ROC-AUC metric
    
    # Set-based metrics
    exact_matches: int
    predicted_count: int
    actual_count: int
    intersection_count: int
    union_count: int
    
    # Ranking metrics (if applicable)
    ndcg_at_k: Optional[Dict[int, float]] = None
    map_score: Optional[float] = None
    
    # Confidence-based metrics (if confidence scores available)
    confidence_calibration: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentCollector:
    """
    Enhanced experimental data collector for recommender evaluation.
    
    This class provides structured collection, storage, and retrieval of experimental
    results with full provenance tracking and support for both batch and streaming analysis.
    """
    
    def __init__(self, 
                 base_output_dir: str = "experiment_data",
                 collection_format: str = "jsonl",
                 enable_tuple_storage: bool = True,
                 enable_state_tracking: bool = False):
        """
        Initialize the experiment collector.
        
        Args:
            base_output_dir: Base directory for storing experimental data
            collection_format: Format for storing results ("jsonl", "parquet", "both")
            enable_tuple_storage: Whether to store full tuple data (vs just hashes)
            enable_state_tracking: Whether to track recommender internal state
        """
        self.base_output_dir = Path(base_output_dir)
        self.collection_format = collection_format
        self.enable_tuple_storage = enable_tuple_storage
        self.enable_state_tracking = enable_state_tracking
        
        # Create directory structure
        self._setup_directories()
        
        # In-memory buffers for batch writing
        self._metadata_buffer: List[ExperimentMetadata] = []
        self._results_buffer: List[Dict[str, Any]] = []
        
        # Session for current experimental run
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        
    def _setup_directories(self):
        """Create the directory structure for experimental data."""
        directories = [
            self.base_output_dir,
            self.base_output_dir / "metadata",
            self.base_output_dir / "results", 
            self.base_output_dir / "tuples",
            self.base_output_dir / "context",
            self.base_output_dir / "analysis",
            self.base_output_dir / "state_snapshots"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def start_experiment_session(self, session_name: str = None) -> str:
        """
        Start a new experimental session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
        session_id = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.current_session_id = session_id
        self.session_start_time = datetime.now(timezone.utc)
        
        logger.info(f"Started experiment session: {session_id}")
        return session_id
    
    def collect_experiment(self,
                          # Core identifiers
                          session_id: str,
                          current_query_position: int,
                          target_query_position: int,
                          recommender_name: str,
                          
                          # Context data
                          current_query_context: QueryContext,
                          target_query_context: QueryContext,
                          
                          # Recommendation data
                          recommendation_result: RecommendationResult,
                          
                          # Ground truth
                          actual_results: pd.DataFrame,
                          
                          # Optional evaluation
                          evaluation_result: Optional[EvaluationResult] = None,
                          
                          # Additional metadata
                          recommender_config: Dict[str, Any] = None,
                          execution_time: float = None,
                          error_info: Optional[str] = None) -> str:
        """
        Collect a complete experimental result.
        
        Returns:
            Experiment ID for this specific run
        """
        
        # Generate experiment ID
        experiment_id = self._generate_experiment_id(
            session_id, current_query_position, target_query_position, recommender_name
        )
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            current_query_position=current_query_position,
            target_query_position=target_query_position,
            gap=target_query_position - current_query_position,
            recommender_name=recommender_name,
            recommender_version=self._get_recommender_version(recommender_name),
            recommender_config_hash=self._hash_config(recommender_config or {}),
            recommendation_type=RecommendationType.FUTURE_QUERY,
            status=ExperimentStatus.COMPLETED if error_info is None else ExperimentStatus.FAILED,
            execution_time_seconds=execution_time,
            error_message=error_info
        )
        
        # Collect the complete experimental record
        experiment_record = {
            'metadata': metadata.to_dict(),
            'current_query_context': current_query_context.to_dict(),
            'target_query_context': target_query_context.to_dict(),
            'recommendation_summary': {
                'predicted_count': recommendation_result.get_tuple_count(),
                'actual_count': len(actual_results),
                'tuple_hashes': recommendation_result.get_tuple_hashes() if not self.enable_tuple_storage else None,
                'confidence_available': recommendation_result.confidence_scores is not None,
                'metadata_available': recommendation_result.recommendation_metadata is not None
            }
        }
        
        # Add evaluation results if available
        if evaluation_result:
            experiment_record['evaluation'] = evaluation_result.to_dict()
        
        # Store the data
        self._store_experiment_data(experiment_id, experiment_record, 
                                  recommendation_result, actual_results, recommender_config)
        
        logger.debug(f"Collected experiment {experiment_id}")
        return experiment_id
    
    def _generate_experiment_id(self, session_id: str, current_pos: int, 
                               target_pos: int, recommender: str) -> str:
        """Generate a unique experiment ID."""
        components = [session_id, str(current_pos), str(target_pos), recommender]
        hash_input = "_".join(components) + "_" + datetime.now().isoformat()
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _get_recommender_version(self, recommender_name: str) -> str:
        """Get version information for a recommender (placeholder for now)."""
        return "1.0.0"  # In practice, this could inspect the actual recommender class
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create a hash of the configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _store_experiment_data(self, experiment_id: str, 
                              experiment_record: Dict[str, Any],
                              recommendation_result: RecommendationResult,
                              actual_results: pd.DataFrame,
                              recommender_config: Dict[str, Any]):
        """Store all components of an experimental result."""
        
        # 1. Store metadata and summary (always JSON for easy querying)
        metadata_file = self.base_output_dir / "metadata" / f"{experiment_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(experiment_record, f, indent=2, default=str)
        
        # 2. Store tuple data if enabled
        if self.enable_tuple_storage:
            # Predicted tuples
            pred_file = self.base_output_dir / "tuples" / f"{experiment_id}_predicted.parquet"
            recommendation_result.predicted_tuples.to_parquet(pred_file, index=False)
            
            # Actual tuples  
            actual_file = self.base_output_dir / "tuples" / f"{experiment_id}_actual.parquet"
            actual_results.to_parquet(actual_file, index=False)
            
            # Confidence scores if available
            if recommendation_result.confidence_scores is not None:
                conf_file = self.base_output_dir / "tuples" / f"{experiment_id}_confidence.parquet"
                recommendation_result.confidence_scores.to_frame('confidence').to_parquet(conf_file)
        
        # 3. Store configuration
        if recommender_config:
            config_file = self.base_output_dir / "context" / f"{experiment_id}_config.json"
            with open(config_file, 'w') as f:
                json.dump(recommender_config, f, indent=2, default=str)
        
        # 4. Store for streaming analysis
        if self.collection_format in ["jsonl", "both"]:
            self._append_to_jsonl(experiment_record)
    
    def _append_to_jsonl(self, record: Dict[str, Any]):
        """Append record to JSONL file for streaming analysis."""
        jsonl_file = self.base_output_dir / "results" / "experiments.jsonl"
        with open(jsonl_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    
    def flush_buffers(self):
        """Flush any buffered data to disk."""
        if self._metadata_buffer:
            # Could implement batch writing here
            self._metadata_buffer.clear()
        if self._results_buffer:
            self._results_buffer.clear()
    
    def load_experiment_results(self, 
                               session_ids: Optional[List[str]] = None,
                               recommender_names: Optional[List[str]] = None,
                               include_tuples: bool = False) -> pd.DataFrame:
        """
        Load experimental results for analysis.
        
        Args:
            session_ids: Filter by session IDs
            recommender_names: Filter by recommender names
            include_tuples: Whether to load actual tuple data
            
        Returns:
            DataFrame with experimental results
        """
        
        # Load from JSONL if available (most efficient for filtering)
        jsonl_file = self.base_output_dir / "results" / "experiments.jsonl"
        if jsonl_file.exists():
            records = []
            with open(jsonl_file, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    
                    # Apply filters
                    if session_ids and record['metadata']['session_id'] not in session_ids:
                        continue
                    if recommender_names and record['metadata']['recommender_name'] not in recommender_names:
                        continue
                    
                    # Flatten the record for easier analysis
                    flat_record = self._flatten_experiment_record(record)
                    records.append(flat_record)
            
            results_df = pd.DataFrame(records)
            
            # Load tuple data if requested
            if include_tuples and not results_df.empty:
                results_df = self._attach_tuple_data(results_df)
            
            return results_df
        
        else:
            # Fallback to loading individual files
            return self._load_from_individual_files(session_ids, recommender_names, include_tuples)
    
    def _flatten_experiment_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested experimental record for tabular analysis."""
        flat = {}
        
        # Metadata
        for key, value in record['metadata'].items():
            flat[f'meta_{key}'] = value
        
        # Context
        for key, value in record.get('current_query_context', {}).items():
            flat[f'current_{key}'] = value
        for key, value in record.get('target_query_context', {}).items():
            flat[f'target_{key}'] = value
        
        # Recommendation summary
        for key, value in record.get('recommendation_summary', {}).items():
            flat[f'rec_{key}'] = value
        
        # Evaluation
        for key, value in record.get('evaluation', {}).items():
            flat[f'eval_{key}'] = value
        
        return flat
    
    def _attach_tuple_data(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Attach tuple data to results DataFrame."""
        # This would load the actual tuple parquet files
        # Implementation depends on analysis needs
        return results_df
    
    def _load_from_individual_files(self, session_ids, recommender_names, include_tuples):
        """Load results from individual metadata files."""
        metadata_dir = self.base_output_dir / "metadata"
        records = []
        
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                record = json.load(f)
            
            # Apply filters
            if session_ids and record['metadata']['session_id'] not in session_ids:
                continue
            if recommender_names and record['metadata']['recommender_name'] not in recommender_names:
                continue
            
            flat_record = self._flatten_experiment_record(record)
            records.append(flat_record)
        
        results_df = pd.DataFrame(records)
        
        if include_tuples and not results_df.empty:
            results_df = self._attach_tuple_data(results_df)
        
        return results_df
    
    def create_analysis_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Create a summary of experimental results for analysis."""
        results_df = self.load_experiment_results(
            session_ids=[session_id] if session_id else None
        )
        
        if results_df.empty:
            return {"error": "No results found"}
        
        summary = {
            "total_experiments": len(results_df),
            "sessions": results_df['meta_session_id'].nunique(),
            "recommenders": results_df['meta_recommender_name'].unique().tolist(),
            "status_distribution": results_df['meta_status'].value_counts().to_dict(),
            "time_range": {
                "start": results_df['meta_timestamp'].min(),
                "end": results_df['meta_timestamp'].max()
            }
        }
        
        # Performance summary
        if 'eval_overlap_accuracy' in results_df.columns:
            summary["performance"] = {
                "mean_accuracy": results_df['eval_overlap_accuracy'].mean(),
                "std_accuracy": results_df['eval_overlap_accuracy'].std(),
                "accuracy_by_recommender": results_df.groupby('meta_recommender_name')['eval_overlap_accuracy'].agg(['mean', 'std', 'count']).to_dict()
            }
        
        return summary


# Convenience functions for integration with existing code

def collect_recommendation_experiment(collector: ExperimentCollector,
                                    session_id: str,
                                    current_query_id: str,
                                    future_query_id: str,
                                    current_results: pd.DataFrame,
                                    future_results: pd.DataFrame,
                                    recommender_name: str,
                                    recommendations: pd.DataFrame,
                                    overlap_accuracy: float,
                                    execution_time: float,
                                    error_info: str = None) -> str:
    """
    Convenience function to collect experiments in the format of the existing system.
    
    This bridges the gap between the current recommender_experiments.py format
    and the enhanced collection system.
    """
    
    # Extract position from query IDs (assuming they encode position)
    current_pos = int(current_query_id.split('_')[-1]) if '_' in current_query_id else 0
    future_pos = int(future_query_id.split('_')[-1]) if '_' in future_query_id else 1
    
    # Create contexts
    current_context = QueryContext(
        session_id=int(session_id),
        query_position=current_pos,
        query_text="",  # Would need to be provided
        query_hash=hashlib.md5(str(current_pos).encode()).hexdigest()[:16],
        result_set_size=len(current_results)
    )
    
    target_context = QueryContext(
        session_id=int(session_id),
        query_position=future_pos,
        query_text="",
        query_hash=hashlib.md5(str(future_pos).encode()).hexdigest()[:16],
        result_set_size=len(future_results)
    )
    
    # Create recommendation result
    rec_result = RecommendationResult(
        experiment_id="",  # Will be generated
        predicted_tuples=recommendations
    )
    
    # Create evaluation result
    eval_result = EvaluationResult(
        experiment_id="",  # Will be generated
        overlap_accuracy=overlap_accuracy,
        jaccard_similarity=0.0,  # Could compute if needed
        precision=0.0,  # Could compute if needed
        recall=0.0,  # Could compute if needed
        f1_score=0.0,  # Could compute if needed
        exact_matches=0,  # Could compute if needed
        predicted_count=len(recommendations),
        actual_count=len(future_results),
        intersection_count=0,  # Could compute if needed
        union_count=0  # Could compute if needed
    )
    
    return collector.collect_experiment(
        session_id=session_id,
        current_query_position=current_pos,
        target_query_position=future_pos,
        recommender_name=recommender_name,
        current_query_context=current_context,
        target_query_context=target_context,
        recommendation_result=rec_result,
        actual_results=future_results,
        evaluation_result=eval_result,
        execution_time=execution_time,
        error_info=error_info
    )
