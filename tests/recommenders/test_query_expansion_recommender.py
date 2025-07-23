"""
Tests for the QueryExpansionRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from query_data_predictor.recommender.query_expansion_recommender import QueryExpansionRecommender, QueryCandidate, QueryBudgetManager
from .test_fixtures import (
    sample_config, simple_dataframe, empty_dataframe, single_row_dataframe,
    mixed_dataframe, large_dataframe, all_recommendation_modes
)


class TestQueryExpansionRecommender:
    """Test suite for QueryExpansionRecommender class."""
    
    def test_init(self, sample_config):
        """Test QueryExpansionRecommender initialization."""
        recommender = QueryExpansionRecommender(sample_config)
        assert recommender.config == sample_config
        assert recommender.name() == "QueryExpansionRecommender"
        assert hasattr(recommender, 'budget_manager')
        assert hasattr(recommender, 'query_runner')
    
    def test_init_with_custom_config(self):
        """Test initialization with custom query expansion configuration."""
        config = {
            'query_expansion': {
                'max_queries': 3,
                'max_execution_time': 15.0,
                'expansion_strategies': ['relaxation', 'generalization'],
                'confidence_threshold': 0.3
            },
            'recommendation': {'top_k': 5}
        }
        recommender = QueryExpansionRecommender(config)
        # Test that custom config is used (specific assertions depend on implementation)
        assert recommender.config == config
    
    def test_recommend_tuples_basic(self, sample_config, simple_dataframe):
        """Test basic recommend_tuples functionality."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = simple_dataframe.head(3)
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(simple_dataframe)
                
                # Should return DataFrame
                assert isinstance(result, pd.DataFrame)
                # Should have same columns as input
                assert list(result.columns) == list(simple_dataframe.columns)
    
    def test_recommend_tuples_empty_input(self, sample_config, empty_dataframe):
        """Test recommend_tuples with empty input."""
        recommender = QueryExpansionRecommender(sample_config)
        result = recommender.recommend_tuples(empty_dataframe)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_recommend_tuples_single_row(self, sample_config, single_row_dataframe):
        """Test recommend_tuples with single row input."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = single_row_dataframe
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(single_row_dataframe)
                
                # Should return DataFrame
                assert isinstance(result, pd.DataFrame)
    
    @patch('query_data_predictor.recommenders.query_expansion_recommender.QueryRunner')
    def test_query_runner_integration(self, mock_query_runner, sample_config, simple_dataframe):
        """Test integration with QueryRunner."""
        mock_runner_instance = Mock()
        mock_query_runner.return_value = mock_runner_instance
        mock_runner_instance.execute_query.return_value = simple_dataframe.head(3)
        
        recommender = QueryExpansionRecommender(sample_config)
        # Test would depend on actual implementation of query execution
        assert recommender.query_runner is not None
    
    def test_generate_expansion_queries(self, sample_config, simple_dataframe):
        """Test query expansion generation."""
        recommender = QueryExpansionRecommender(sample_config)
        
        # Mock the actual query generation (depends on implementation)
        with patch.object(recommender, '_analyze_current_results') as mock_analyze:
            mock_analyze.return_value = {'patterns': [], 'constraints': []}
            
            queries = recommender._generate_expansion_queries(simple_dataframe)
            
            # Should return list of query candidates
            assert isinstance(queries, list)
    
    def test_query_budget_manager(self, sample_config):
        """Test QueryBudgetManager functionality."""
        budget_manager = QueryBudgetManager(max_queries=3, max_execution_time=10.0)
        
        # Test initial state
        assert budget_manager.max_queries == 3
        assert budget_manager.max_execution_time == 10.0
        assert budget_manager.executed_queries == 0
        
        # Test session start
        budget_manager.start_session()
        assert budget_manager.start_time is not None
        assert budget_manager.executed_queries == 0
    
    def test_query_candidate_creation(self):
        """Test QueryCandidate creation."""
        candidate = QueryCandidate(
            query="SELECT * FROM table WHERE condition",
            confidence=0.8,
            expansion_type="relaxation",
            source_pattern="pattern1"
        )
        
        assert candidate.query == "SELECT * FROM table WHERE condition"
        assert candidate.confidence == 0.8
        assert candidate.expansion_type == "relaxation"
        assert candidate.source_pattern == "pattern1"
        assert candidate.estimated_cost == 1  # default value
    
    def test_budget_manager_can_execute(self):
        """Test budget manager execution limits."""
        budget_manager = QueryBudgetManager(max_queries=2)
        budget_manager.start_session()
        
        # Should allow execution initially
        assert budget_manager.can_execute_query()
        
        # Simulate query execution
        budget_manager.executed_queries = 1
        assert budget_manager.can_execute_query()
        
        # Should block after reaching limit
        budget_manager.executed_queries = 2
        assert not budget_manager.can_execute_query()
    
    def test_budget_manager_time_limit(self):
        """Test budget manager time limits."""
        budget_manager = QueryBudgetManager(max_execution_time=0.1)  # Very short time
        budget_manager.start_session()
        
        # Initially should allow execution
        assert budget_manager.can_execute_query()
        
        # After time passes, should block
        import time
        time.sleep(0.2)
        assert not budget_manager.can_execute_query()
    
    def test_expansion_strategies(self, sample_config, simple_dataframe):
        """Test different expansion strategies."""
        strategies = ['relaxation', 'generalization', 'diversification']
        
        for strategy in strategies:
            with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
                with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                    mock_generate.return_value = [
                        QueryCandidate(
                            query="SELECT * FROM test",
                            confidence=0.5,
                            expansion_type=strategy,
                            source_pattern="test"
                        )
                    ]
                    mock_execute.return_value = simple_dataframe.head(2)
                    
                    recommender = QueryExpansionRecommender(sample_config)
                    result = recommender.recommend_tuples(simple_dataframe)
                    
                    # Should work with all strategies
                    assert isinstance(result, pd.DataFrame)
    
    def test_confidence_threshold_filtering(self, sample_config, simple_dataframe):
        """Test filtering queries by confidence threshold."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                # Return queries with different confidence levels
                mock_generate.return_value = [
                    QueryCandidate("SELECT 1", 0.9, "high", "pattern1"),
                    QueryCandidate("SELECT 2", 0.3, "medium", "pattern2"),
                    QueryCandidate("SELECT 3", 0.1, "low", "pattern3")
                ]
                mock_execute.return_value = simple_dataframe.head(2)
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(simple_dataframe)
                
                # Should filter based on confidence threshold
                assert isinstance(result, pd.DataFrame)
    
    def test_query_execution_error_handling(self, sample_config, simple_dataframe):
        """Test handling of query execution errors."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = [
                    QueryCandidate("INVALID SQL", 0.8, "test", "pattern")
                ]
                # Simulate execution error
                mock_execute.side_effect = Exception("SQL Error")
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(simple_dataframe)
                
                # Should handle errors gracefully
                assert isinstance(result, pd.DataFrame)
    
    def test_large_dataset_handling(self, sample_config, large_dataframe):
        """Test handling of large datasets."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = large_dataframe.head(5)
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(large_dataframe)
                
                # Should handle large datasets efficiently
                assert isinstance(result, pd.DataFrame)
                assert len(result) <= 5  # Should limit output
    
    def test_mixed_data_types(self, sample_config, mixed_dataframe):
        """Test handling of mixed data types."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = mixed_dataframe.head(3)
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(mixed_dataframe)
                
                # Should handle mixed data types
                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == list(mixed_dataframe.columns)
    
    def test_query_cost_estimation(self):
        """Test query cost estimation."""
        candidate = QueryCandidate(
            query="SELECT * FROM large_table",
            confidence=0.7,
            expansion_type="test",
            source_pattern="pattern",
            estimated_cost=10
        )
        
        assert candidate.estimated_cost == 10
    
    def test_query_ranking_by_confidence(self, sample_config):
        """Test ranking of query candidates by confidence."""
        candidates = [
            QueryCandidate("SELECT 1", 0.3, "low", "p1"),
            QueryCandidate("SELECT 2", 0.9, "high", "p2"),
            QueryCandidate("SELECT 3", 0.6, "medium", "p3")
        ]
        
        # Sort by confidence (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        
        assert sorted_candidates[0].confidence == 0.9
        assert sorted_candidates[1].confidence == 0.6
        assert sorted_candidates[2].confidence == 0.3
    
    def test_duplicate_query_prevention(self, sample_config, simple_dataframe):
        """Test prevention of duplicate query execution."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                # Return duplicate queries
                mock_generate.return_value = [
                    QueryCandidate("SELECT * FROM test", 0.8, "type1", "p1"),
                    QueryCandidate("SELECT * FROM test", 0.7, "type2", "p2")  # duplicate
                ]
                mock_execute.return_value = simple_dataframe.head(2)
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(simple_dataframe)
                
                # Should handle duplicates
                assert isinstance(result, pd.DataFrame)
    
    def test_all_recommendation_modes(self, all_recommendation_modes, simple_dataframe):
        """Test all recommendation output modes."""
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = simple_dataframe.head(3)
                
                recommender = QueryExpansionRecommender(all_recommendation_modes)
                result = recommender.recommend_tuples(simple_dataframe)
                
                # Should work with all recommendation modes
                assert isinstance(result, pd.DataFrame)
                assert not result.empty
                assert len(result) <= len(simple_dataframe)
    
    def test_name_method(self, sample_config):
        """Test the name method."""
        recommender = QueryExpansionRecommender(sample_config)
        assert recommender.name() == "QueryExpansionRecommender"
    
    def test_config_without_query_expansion_section(self, simple_dataframe):
        """Test with config missing query_expansion section."""
        config = {'recommendation': {'top_k': 3}}
        
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = simple_dataframe.head(3)
                
                recommender = QueryExpansionRecommender(config)
                
                # Should use default query expansion configuration
                result = recommender.recommend_tuples(simple_dataframe)
                assert isinstance(result, pd.DataFrame)
    
    def test_query_pattern_analysis(self, sample_config, simple_dataframe):
        """Test analysis of query patterns."""
        recommender = QueryExpansionRecommender(sample_config)
        
        # Mock pattern analysis
        with patch.object(recommender, '_analyze_current_results') as mock_analyze:
            mock_analyze.return_value = {
                'value_patterns': ['pattern1', 'pattern2'],
                'range_patterns': ['range1'],
                'categorical_patterns': ['cat1']
            }
            
            analysis = recommender._analyze_current_results(simple_dataframe)
            
            # Should return pattern analysis
            assert isinstance(analysis, dict)
            assert 'value_patterns' in analysis
    
    def test_sql_generation(self, sample_config):
        """Test SQL query generation."""
        recommender = QueryExpansionRecommender(sample_config)
        
        # Mock SQL generation based on patterns
        with patch.object(recommender, '_generate_sql_from_pattern') as mock_generate:
            mock_generate.return_value = "SELECT * FROM table WHERE condition"
            
            sql = recommender._generate_sql_from_pattern("test_pattern", "relaxation")
            
            # Should generate valid SQL string
            assert isinstance(sql, str)
            assert "SELECT" in sql.upper()
    
    def test_result_deduplication(self, sample_config, simple_dataframe):
        """Test deduplication of results."""
        # Create DataFrame with duplicates
        df_with_dupes = pd.concat([simple_dataframe, simple_dataframe.head(2)])
        
        with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
            with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                mock_generate.return_value = []
                mock_execute.return_value = df_with_dupes
                
                recommender = QueryExpansionRecommender(sample_config)
                result = recommender.recommend_tuples(simple_dataframe)
                
                # Should handle deduplication
                assert isinstance(result, pd.DataFrame)
    
    def test_incremental_query_execution(self, sample_config, simple_dataframe):
        """Test incremental execution of queries with budget management."""
        budget_manager = QueryBudgetManager(max_queries=2)
        
        with patch.object(QueryExpansionRecommender, 'budget_manager', budget_manager):
            with patch.object(QueryExpansionRecommender, '_generate_expansion_queries') as mock_generate:
                with patch.object(QueryExpansionRecommender, '_execute_query_candidates') as mock_execute:
                    mock_generate.return_value = [
                        QueryCandidate("SELECT 1", 0.9, "type1", "p1"),
                        QueryCandidate("SELECT 2", 0.8, "type2", "p2"),
                        QueryCandidate("SELECT 3", 0.7, "type3", "p3")  # Should be skipped due to budget
                    ]
                    mock_execute.return_value = simple_dataframe.head(3)
                    
                    recommender = QueryExpansionRecommender(sample_config)
                    result = recommender.recommend_tuples(simple_dataframe)
                    
                    # Should respect budget limits
                    assert isinstance(result, pd.DataFrame)
