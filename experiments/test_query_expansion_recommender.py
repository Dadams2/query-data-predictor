"""
Test and example usage for QueryExpansionRecommender.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.recommender.query_expansion_recommender import QueryExpansionRecommender

# Load environment variables
load_dotenv()

def test_query_expansion_recommender():
    """Test the QueryExpansionRecommender with a simple example."""
    
    # Configuration for the recommender
    config = {
        'recommendation': {
            'mode': 'top_k',
            'top_k': 10
        },
        'query_expansion': {
            'enable_range_expansion': True,
            'enable_constraint_relaxation': True,
            'enable_join_exploration': True,
            'enable_column_expansion': True,
            'enable_similarity_search': True,
            'budget': {
                'max_queries': 3,
                'max_execution_time': 15.0,
                'max_results_per_query': 500
            },
            'range_expansion_factor': 0.15,  # 15% range expansion
            'min_confidence_threshold': 0.4
        }
    }
    
    # Get database connection parameters
    DB_NAME = os.getenv("PG_DATA")
    DB_USER = os.getenv("PG_DATA_USER")
    DB_HOST = os.getenv("PG_HOST", "localhost")
    DB_PORT = os.getenv("PG_PORT", "5432")
    
    if not DB_NAME or not DB_USER:
        print("Database connection parameters not found in environment variables")
        return
    
    # Create QueryRunner
    query_runner = QueryRunner(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
    
    try:
        # Connect to database
        query_runner.connect()
        
        # Create recommender
        recommender = QueryExpansionRecommender(config, query_runner)
        
        # Create sample current results (simulating a small quasar query result)
        current_results = pd.DataFrame({
            'specObjID': [1234567890, 1234567891, 1234567892],
            'z': [1.96, 1.98, 1.97],
            'zConf': [0.99, 0.98, 0.99],
            'SpecClass': [3, 3, 4],
            'objID': [1111111111, 2222222222, 3333333333],
            'primTarget': [4, 4, 20]
        })
        
        print(f"Current results shape: {current_results.shape}")
        print("Current results:")
        print(current_results)
        print()
        
        # Get recommendations
        kwargs = {
            'session_id': 'test_session',
            'current_query': "SELECT specObjID,z,zConf,SpecClass,objID,primTarget FROM SpecObj WHERE (SpecClass=3 or SpecClass=4) and z between 1.95 and 2 and zConf>0.99"
        }
        
        print("Generating recommendations...")
        recommendations = recommender.recommend_tuples(current_results, **kwargs)
        
        print(f"Recommendations shape: {recommendations.shape}")
        print("Sample recommendations:")
        print(recommendations.head(10))
        
        if len(recommendations) > len(current_results):
            print(f"\nSuccess! Found {len(recommendations) - len(current_results)} new tuples through expansion queries")
        else:
            print("\nNo new tuples found, but expansion logic executed successfully")
            
    except Exception as e:
        print(f"Error testing QueryExpansionRecommender: {e}")
        import traceback
        traceback.print_exc()
    finally:
        query_runner.disconnect()


def create_config_example():
    """Create an example configuration file for the QueryExpansionRecommender."""
    
    config_example = {
        'recommendation': {
            'mode': 'top_k',
            'top_k': 15
        },
        'query_expansion': {
            # Enable/disable different expansion strategies
            'enable_range_expansion': True,
            'enable_constraint_relaxation': True,
            'enable_join_exploration': True,
            'enable_column_expansion': True,
            'enable_similarity_search': True,
            
            # Budget management to prevent excessive computation
            'budget': {
                'max_queries': 5,           # Maximum number of expansion queries to execute
                'max_execution_time': 30.0, # Maximum total time in seconds
                'max_results_per_query': 1000 # Maximum results per individual query
            },
            
            # Expansion parameters
            'range_expansion_factor': 0.2,  # How much to expand numeric ranges (20%)
            'min_confidence_threshold': 0.3, # Minimum confidence for candidate queries
            'max_query_complexity': 5        # Maximum complexity score for generated queries
        }
    }
    
    print("Example configuration for QueryExpansionRecommender:")
    print("=" * 50)
    import yaml
    print(yaml.dump(config_example, default_flow_style=False))


if __name__ == "__main__":
    print("QueryExpansionRecommender Test and Example")
    print("=" * 50)
    
    # Show configuration example
    create_config_example()
    print()
    
    # Run test
    test_query_expansion_recommender()
