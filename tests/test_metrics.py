"""
Tests for the result prediction experimentation framework.
"""

import pytest
import pandas as pd
from query_data_predictor.metrics import EvaluationMetrics


class TestMetrics:
    """Test suite for the evaluation metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Fixture for EvaluationMetrics instance."""
        return EvaluationMetrics(jaccard_threshold=0.5)
    
    @pytest.fixture
    def identical_dfs(self):
        """Fixture for identical DataFrames."""
        df1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        return df1, df1.copy()
    
    @pytest.fixture
    def different_dfs(self):
        """Fixture for completely different DataFrames."""
        df1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        df2 = pd.DataFrame({
            'A': [4, 5, 6],
            'B': ['d', 'e', 'f']
        })
        return df1, df2
    
    @pytest.fixture
    def partially_overlapping_dfs(self):
        """Fixture for partially overlapping DataFrames."""
        df1 = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['a', 'b', 'c', 'd']
        })
        df2 = pd.DataFrame({
            'A': [3, 4, 5, 6],
            'B': ['c', 'd', 'e', 'f']
        })
        return df1, df2
    
    def test_accuracy_identical(self, metrics, identical_dfs):
        """Test accuracy for identical DataFrames."""
        pred, actual = identical_dfs
        assert metrics.accuracy(pred, actual) == 1.0
    
    def test_accuracy_different(self, metrics, different_dfs):
        """Test accuracy for completely different DataFrames."""
        pred, actual = different_dfs
        assert metrics.accuracy(pred, actual) == 0.0
    
    def test_accuracy_partial(self, metrics, partially_overlapping_dfs):
        """Test accuracy for partially overlapping DataFrames."""
        pred, actual = partially_overlapping_dfs
        assert metrics.accuracy(pred, actual) == 0.5  # 2 out of 4 rows match
    
    def test_jaccard_similarity_identical(self, metrics, identical_dfs):
        """Test Jaccard similarity for identical DataFrames."""
        pred, actual = identical_dfs
        assert metrics.jaccard_similarity(pred, actual) == 1.0
    
    def test_jaccard_similarity_different(self, metrics, different_dfs):
        """Test Jaccard similarity for completely different DataFrames."""
        pred, actual = different_dfs
        assert metrics.jaccard_similarity(pred, actual) == 0.0
    
    def test_jaccard_similarity_partial(self, metrics, partially_overlapping_dfs):
        """Test Jaccard similarity for partially overlapping DataFrames."""
        pred, actual = partially_overlapping_dfs
        assert metrics.jaccard_similarity(pred, actual) == pytest.approx(0.33, abs=0.01)  # 2 shared out of 6 unique
    
    def test_standard_metrics(self, metrics, partially_overlapping_dfs):
        """Test that all_metrics returns all expected metrics."""
        pred, actual = partially_overlapping_dfs
        results = metrics.standard_metrics(pred, actual)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'jaccard_similarity',
            'precision', 'recall', 'f1_score',
            'jaccard_precision', 'jaccard_recall', 'jaccard_f1'
        ]
        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))
    
    def test_overlap_accuracy(self, metrics):
        """Test overlap accuracy metric with three DataFrames."""
        previous = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        actual = pd.DataFrame({
            'A': [2, 3, 4],
            'B': ['b', 'c', 'd']
        })
        predicted = pd.DataFrame({
            'A': [2, 3],
            'B': ['b', 'c']
        })

        # Expected overlap: rows with A=2, B='b' and A=3, B='c'
        # Predicted matches the overlap exactly
        assert metrics.overlap_accuracy(previous, actual, predicted) == 1.0

        predicted_partial = pd.DataFrame({
            'A': [2],
            'B': ['b']
        })

        # Predicted partially matches the overlap
        assert metrics.overlap_accuracy(previous, actual, predicted_partial) == 0.5

        predicted_none = pd.DataFrame({
            'A': [5],
            'B': ['e']
        })

        # Predicted does not match the overlap
        assert metrics.overlap_accuracy(previous, actual, predicted_none) == 0.0


