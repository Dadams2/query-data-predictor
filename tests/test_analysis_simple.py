"""
Tests for the simple analysis functionality.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from query_data_predictor.analysis import ResultsAnalyzer


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary results directory with sample data."""
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    
    # Create sample result data
    sample_results = [
        {
            "session_id": "test_session",
            "current_query_id": 0,
            "future_query_id": 1,
            "gap": 1,
            "recommender_name": "test_recommender",
            "current_results": [
                {"col1": "A", "col2": "X", "col3": 1},
                {"col1": "B", "col2": "Y", "col3": 2},
                {"col1": "C", "col2": "Z", "col3": 3}
            ],
            "future_results": [
                {"col1": "A", "col2": "X", "col3": 1},
                {"col1": "D", "col2": "W", "col3": 4}
            ],
            "recommended_results": [
                {"col1": "A", "col2": "X", "col3": 1},
                {"col1": "B", "col2": "Y", "col3": 2}
            ],
            "execution_time": 1.0,
            "timestamp": "2025-01-01T00:00:00"
        },
        {
            "session_id": "test_session",
            "current_query_id": 1,
            "future_query_id": 2,
            "gap": 1,
            "recommender_name": "test_recommender",
            "current_results": [
                {"col1": "E", "col2": "V", "col3": 5}
            ],
            "future_results": [
                {"col1": "E", "col2": "V", "col3": 5},
                {"col1": "F", "col2": "U", "col3": 6}
            ],
            "recommended_results": [
                {"col1": "E", "col2": "V", "col3": 5}
            ],
            "execution_time": 0.5,
            "timestamp": "2025-01-01T00:00:01"
        }
    ]
    
    # Write results to JSON file
    results_file = results_dir / "test_session__gap-1.json"
    with open(results_file, 'w') as f:
        json.dump(sample_results, f)
    
    return results_dir


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return {
        'evaluation': {
            'jaccard_threshold': 0.5
        }
    }


def test_analyze_simple_runs(temp_results_dir, sample_config):
    """Test that analyze_simple runs without errors."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    # Run analyze_simple
    result = analyzer.analyze_simple()
    
    # Check that output_dirs were created
    assert 'output_dirs' in result
    assert len(result['output_dirs']) > 0


def test_analyze_simple_creates_scenario_dirs(temp_results_dir, sample_config):
    """Test that analyze_simple creates directories for each scenario."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    analyzer.analyze_simple()
    
    # Check that scenario directories exist
    analysis_dir = temp_results_dir / 'analysis'
    session_dir = analysis_dir / 'session_test_session'
    
    assert session_dir.exists()
    
    for scenario in ['raw', 'close', 'similarity']:
        scenario_dir = session_dir / scenario
        assert scenario_dir.exists(), f"Scenario directory {scenario} should exist"


def test_analyze_simple_creates_files(temp_results_dir, sample_config):
    """Test that analyze_simple creates expected output files."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    analyzer.analyze_simple()
    
    analysis_dir = temp_results_dir / 'analysis'
    session_dir = analysis_dir / 'session_test_session'
    
    for scenario in ['raw', 'close', 'similarity']:
        scenario_dir = session_dir / scenario
        
        # Check for expected files
        assert (scenario_dir / 'per_query_metrics.csv').exists()
        assert (scenario_dir / 'gap_aggregates.csv').exists()
        assert (scenario_dir / 'accuracy_by_query_number.png').exists()
        assert (scenario_dir / 'metric_distributions.png').exists()
        assert (scenario_dir / 'gap_metrics.png').exists()
        assert (scenario_dir / 'overlap_by_query_number.png').exists()


def test_analyze_simple_csv_content(temp_results_dir, sample_config):
    """Test that CSV files contain expected data."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    analyzer.analyze_simple()
    
    analysis_dir = temp_results_dir / 'analysis'
    session_dir = analysis_dir / 'session_test_session'
    raw_dir = session_dir / 'raw'
    
    # Check per_query_metrics.csv
    metrics_df = pd.read_csv(raw_dir / 'per_query_metrics.csv')
    
    assert 'session_id' in metrics_df.columns
    assert 'gap' in metrics_df.columns
    assert 'query_number' in metrics_df.columns
    assert 'accuracy' in metrics_df.columns
    assert 'precision' in metrics_df.columns
    assert 'recall' in metrics_df.columns
    assert 'f1_score' in metrics_df.columns
    assert 'overlap' in metrics_df.columns
    
    # Should have 2 queries
    assert len(metrics_df) == 2


def test_analyze_simple_gap_aggregates(temp_results_dir, sample_config):
    """Test that gap aggregates are computed correctly."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    analyzer.analyze_simple()
    
    analysis_dir = temp_results_dir / 'analysis'
    session_dir = analysis_dir / 'session_test_session'
    raw_dir = session_dir / 'raw'
    
    # Check gap_aggregates.csv
    gap_df = pd.read_csv(raw_dir / 'gap_aggregates.csv')
    
    assert 'gap' in gap_df.columns
    assert 'accuracy_mean' in gap_df.columns
    assert 'precision_mean' in gap_df.columns
    assert 'recall_mean' in gap_df.columns
    assert 'f1_mean' in gap_df.columns
    assert 'overlap_mean' in gap_df.columns
    
    # Should have 1 gap (gap=1)
    assert len(gap_df) == 1
    assert gap_df.iloc[0]['gap'] == 1


def test_compute_metrics_raw_scenario(temp_results_dir, sample_config):
    """Test raw scenario metric computation."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    # Test exact match (pass lists of dicts directly)
    predicted = [
        {"col1": "A", "col2": "X", "col3": 1},
        {"col1": "B", "col2": "Y", "col3": 2}
    ]
    actual = [
        {"col1": "A", "col2": "X", "col3": 1},
        {"col1": "C", "col2": "Z", "col3": 3}
    ]
    
    metrics = analyzer._compute_metrics_for_scenario(predicted, actual, 'raw', 0.5)
    
    # Should find 1 match out of 2 actual tuples
    assert metrics['accuracy'] == 0.5  # 1/2 actual matched
    assert metrics['precision'] == 0.5  # 1/2 predicted matched
    assert metrics['recall'] == 0.5  # Same as accuracy
    assert metrics['f1'] == 0.5


def test_compute_metrics_close_scenario(temp_results_dir, sample_config):
    """Test close scenario metric computation with enough columns for close matching."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    # Test with at least 5 matching columns (close requires >= 5)
    predicted = [
        {"c1": "A", "c2": "X", "c3": 1, "c4": "m", "c5": "n", "c6": 999},
        {"c1": "D", "c2": "W", "c3": 4, "c4": "q", "c5": "r", "c6": 0}
    ]
    actual = [
        {"c1": "A", "c2": "X", "c3": 1, "c4": "m", "c5": "n", "c6": 1},
        {"c1": "C", "c2": "Z", "c3": 3, "c4": "p", "c5": "s", "c6": 2}
    ]
    
    metrics = analyzer._compute_metrics_for_scenario(predicted, actual, 'close', 0.5)
    
    # First predicted matches first actual on 5 of 6 columns (c1-c5)
    assert metrics['accuracy'] == 0.5  # 1/2 actual matched
    assert metrics['precision'] == 0.5  # 1/2 predicted matched


def test_compute_overlap(temp_results_dir, sample_config):
    """Test overlap computation."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    current = [
        {"col1": "A", "col2": "X", "col3": 1},
        {"col1": "B", "col2": "Y", "col3": 2}
    ]
    actual = [
        {"col1": "A", "col2": "X", "col3": 1},  # In overlap
        {"col1": "C", "col2": "Z", "col3": 3}   # Not in overlap
    ]
    predicted = [
        {"col1": "A", "col2": "X", "col3": 1},  # Matches overlap
        {"col1": "B", "col2": "Y", "col3": 2}   # Not in overlap
    ]
    
    overlap = analyzer._compute_overlap_for_scenario(current, actual, predicted, 'raw', 0.5)
    
    # 1 out of 2 predicted tuples is in the overlap
    assert overlap == 0.5


def test_cross_session_summaries(temp_results_dir, sample_config):
    """Test that cross-session summaries are created."""
    analyzer = ResultsAnalyzer(
        results_dir=temp_results_dir,
        config=sample_config
    )
    
    analyzer.analyze_simple()
    
    analysis_dir = temp_results_dir / 'analysis'
    summary_dir = analysis_dir / 'summary'
    
    assert summary_dir.exists()
    
    for scenario in ['raw', 'close', 'similarity']:
        scenario_summary = summary_dir / scenario
        assert scenario_summary.exists()
        assert (scenario_summary / 'all_sessions_metrics.csv').exists()
        assert (scenario_summary / 'overall_statistics.csv').exists()


def test_empty_results_skipped(tmp_path, sample_config):
    """Test that empty results are skipped."""
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    
    # Create result with empty data
    sample_results = [
        {
            "session_id": "test_session",
            "current_query_id": 0,
            "future_query_id": 1,
            "gap": 1,
            "recommender_name": "test_recommender",
            "current_results": [],
            "future_results": [],
            "recommended_results": [],
            "execution_time": 1.0,
            "timestamp": "2025-01-01T00:00:00"
        }
    ]
    
    results_file = results_dir / "test_session__gap-1.json"
    with open(results_file, 'w') as f:
        json.dump(sample_results, f)
    
    analyzer = ResultsAnalyzer(results_dir=results_dir, config=sample_config)
    result = analyzer.analyze_simple()
    
    # Should complete without error
    assert 'output_dirs' in result
