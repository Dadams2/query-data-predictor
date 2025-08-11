"""
Example integration of QueryExpansionRecommender with the existing experiment framework.

This shows how to use the QueryExpansionRecommender in your existing experiment setup
when you have access to a QueryRunner instance.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.dataloader import DataLoader
from query_data_predictor.query_result_sequence import QueryResultSequence
from query_data_predictor.metrics import EvaluationMetrics
from query_data_predictor.recommender import (
    DummyRecommender,
    RandomRecommender,
    ClusteringRecommender,
    InterestingnessRecommender,
    QueryExpansionRecommender
)

# Load environment variables
load_dotenv()


class EnhancedRecommenderExperimentRunner:
    """
    Enhanced experiment runner that includes QueryExpansionRecommender.
    
    This extends the existing experiment framework to support out-of-results
    recommenders that require database access.
    """
    
    def __init__(self, 
                 config_path: str = "config.yaml",
                 dataset_dir: str = "data/datasets",
                 enable_query_expansion: bool = True):
        """
        Initialize the enhanced experiment runner.
        
        Args:
            config_path: Path to configuration file
            dataset_dir: Directory containing the dataset files
            enable_query_expansion: Whether to enable query expansion recommender
        """
        self.config = self._load_config(config_path)
        self.dataset_dir = Path(dataset_dir)
        self.dataloader = DataLoader(str(self.dataset_dir))
        self.query_result_sequence = QueryResultSequence(self.dataloader)
        self.evaluator = EvaluationMetrics()
        
        # Database connection for query expansion
        self.query_runner = None
        self.enable_query_expansion = enable_query_expansion
        
        if self.enable_query_expansion:
            self.query_runner = self._setup_query_runner()
        
        # Initialize recommenders
        self.recommender_config = self._create_recommender_config()
        self.recommenders = self._initialize_recommenders()
    
    def _setup_query_runner(self) -> QueryRunner:
        """Setup database connection for query expansion."""
        # Get database connection parameters
        DB_NAME = os.getenv("PG_DATA")
        DB_USER = os.getenv("PG_DATA_USER")
        DB_HOST = os.getenv("PG_HOST", "localhost")
        DB_PORT = os.getenv("PG_PORT", "5432")
        
        if not DB_NAME or not DB_USER:
            print("Warning: Database connection parameters not found. QueryExpansionRecommender will be disabled.")
            self.enable_query_expansion = False
            return None
        
        query_runner = QueryRunner(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
        
        try:
            query_runner.connect()
            print(f"Connected to database: {DB_NAME}@{DB_HOST}")
            return query_runner
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            print("QueryExpansionRecommender will be disabled.")
            self.enable_query_expansion = False
            return None
    
    def _load_config(self, config_path: str) -> dict:
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
                'association_rules': {'min_support': 0.1, 'metric': 'confidence', 'min_threshold': 0.7},
                'query_expansion': {
                    'enable_range_expansion': True,
                    'enable_constraint_relaxation': True,
                    'budget': {'max_queries': 3, 'max_execution_time': 15.0}
                }
            }
    
    def _create_recommender_config(self) -> dict:
        """Create standardized configuration for all recommenders."""
        base_config = self.config.copy()
        base_config['recommendation'] = {'mode': 'top_quartile'}
        return base_config
    
    def _initialize_recommenders(self) -> dict:
        """Initialize all recommender instances."""
        recommenders = {
            'dummy': DummyRecommender(self.recommender_config),
            'random': RandomRecommender(self.recommender_config),
            'clustering': ClusteringRecommender(self.recommender_config),
            'interestingness': InterestingnessRecommender(self.recommender_config)
        }
        
        # Add QueryExpansionRecommender if database connection is available
        if self.enable_query_expansion and self.query_runner:
            recommenders['query_expansion'] = QueryExpansionRecommender(
                self.recommender_config, self.query_runner
            )
            print("QueryExpansionRecommender enabled")
        else:
            print("QueryExpansionRecommender disabled (no database connection)")
        
        return recommenders
    
    def run_single_recommendation_test(self, session_id: str, query_position: int, gap: int = 1):
        """
        Run a single recommendation test to demonstrate the query expansion recommender.
        
        Args:
            session_id: Session ID to test
            query_position: Query position in the session
            gap: Gap to the target query
            
        Returns:
            Dictionary with results from all recommenders
        """
        try:
            # Get current and future results
            current_results, future_results = self.query_result_sequence.get_query_pair_with_gap(
                session_id, query_position, gap
            )
            
            if current_results.empty or future_results.empty:
                print(f"Empty results for session {session_id}, query {query_position}")
                return {}
            
            print(f"Testing recommendations for session {session_id}, query {query_position}")
            print(f"Current results: {len(current_results)} rows")
            print(f"Future results: {len(future_results)} rows")
            
            # Test each recommender
            results = {}
            
            for name, recommender in self.recommenders.items():
                try:
                    print(f"\n--- Testing {name} recommender ---")
                    
                    # Prepare kwargs for query expansion recommender
                    kwargs = {}
                    if name == 'query_expansion':
                        # Get session data to extract original query
                        session_data = self.dataloader.get_results_for_session(session_id)
                        query_row = session_data[session_data['query_position'] == query_position]
                        if not query_row.empty:
                            kwargs['current_query'] = query_row.iloc[0]['current_query']
                            kwargs['session_id'] = session_id
                    
                    # Get recommendations
                    recommendations = recommender.recommend_tuples(current_results, **kwargs)
                    
                    # Evaluate recommendations
                    if not recommendations.empty:
                        evaluation = self.evaluator.evaluate_recommendation(
                            recommendations, future_results
                        )
                        
                        results[name] = {
                            'recommender': name,
                            'recommendations_count': len(recommendations),
                            'evaluation': evaluation,
                            'new_tuples_found': len(recommendations) - len(current_results) if len(recommendations) > len(current_results) else 0
                        }
                        
                        print(f"Recommendations: {len(recommendations)} rows")
                        print(f"Accuracy: {evaluation.get('overlap_accuracy', 'N/A'):.3f}")
                        if 'new_tuples_found' in results[name] and results[name]['new_tuples_found'] > 0:
                            print(f"New tuples found: {results[name]['new_tuples_found']}")
                    else:
                        print("No recommendations generated")
                        results[name] = {
                            'recommender': name,
                            'recommendations_count': 0,
                            'evaluation': {},
                            'new_tuples_found': 0
                        }
                        
                except Exception as e:
                    print(f"Error with {name} recommender: {e}")
                    results[name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            print(f"Error in recommendation test: {e}")
            return {}
    
    def compare_recommenders(self, session_id: str, max_queries: int = 5):
        """
        Compare all recommenders on multiple queries from a session.
        
        Args:
            session_id: Session ID to test
            max_queries: Maximum number of queries to test
            
        Returns:
            Comparison results
        """
        query_ids = self.query_result_sequence.get_ordered_query_ids(session_id)
        
        if len(query_ids) < 2:
            print(f"Session {session_id} has insufficient queries for testing")
            return {}
        
        # Limit number of queries to test
        test_queries = query_ids[:min(max_queries, len(query_ids) - 1)]
        
        all_results = {}
        
        for query_pos in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing Query Position: {query_pos}")
            print(f"{'='*50}")
            
            results = self.run_single_recommendation_test(session_id, query_pos)
            all_results[query_pos] = results
        
        # Summarize results
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        
        recommender_performance = {}
        
        for query_pos, results in all_results.items():
            for rec_name, rec_results in results.items():
                if rec_name not in recommender_performance:
                    recommender_performance[rec_name] = {
                        'total_tests': 0,
                        'total_accuracy': 0,
                        'total_new_tuples': 0,
                        'errors': 0
                    }
                
                if 'error' in rec_results:
                    recommender_performance[rec_name]['errors'] += 1
                else:
                    recommender_performance[rec_name]['total_tests'] += 1
                    acc = rec_results.get('evaluation', {}).get('overlap_accuracy', 0)
                    recommender_performance[rec_name]['total_accuracy'] += acc
                    recommender_performance[rec_name]['total_new_tuples'] += rec_results.get('new_tuples_found', 0)
        
        for rec_name, perf in recommender_performance.items():
            print(f"\n{rec_name}:")
            if perf['total_tests'] > 0:
                avg_accuracy = perf['total_accuracy'] / perf['total_tests']
                print(f"  Average Accuracy: {avg_accuracy:.3f}")
                print(f"  Total New Tuples Found: {perf['total_new_tuples']}")
                print(f"  Successful Tests: {perf['total_tests']}")
            if perf['errors'] > 0:
                print(f"  Errors: {perf['errors']}")
        
        return all_results
    
    def cleanup(self):
        """Clean up resources."""
        if self.query_runner:
            self.query_runner.disconnect()
            print("Database connection closed")


def main():
    """Main function to demonstrate the enhanced experiment runner."""
    print("Enhanced Recommender Experiment Runner")
    print("=" * 50)
    
    # Initialize the enhanced runner
    runner = EnhancedRecommenderExperimentRunner(
        config_path="config.yaml",
        dataset_dir="data/datasets",
        enable_query_expansion=True
    )
    
    try:
        # Get available sessions
        sessions = runner.dataloader.get_sessions()
        if not sessions:
            print("No sessions found in dataset")
            return
        
        print(f"Available sessions: {sessions}")
        
        # Test with the first session
        test_session = sessions[0]
        print(f"\nTesting with session: {test_session}")
        
        # Compare all recommenders
        results = runner.compare_recommenders(test_session, max_queries=3)
        
        # Display final results
        print(f"\nExperiment completed. Tested {len(results)} queries.")
        
    except Exception as e:
        print(f"Error in main experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
