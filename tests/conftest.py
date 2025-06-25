"""
Shared test fixtures for the query data predictor tests.
"""
import os
import pytest
import pandas as pd
import numpy as np
import pickle
import pathlib
import tempfile

@pytest.fixture
def sample_dataset_dir():
    """Create a temporary directory with sample data for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a metadata.csv file
        metadata_path = os.path.join(temp_dir, "metadata.csv")
        metadata_df = pd.DataFrame({
            "session_id": [1001, 1002, 1003],
            "path": [
                os.path.join(temp_dir, "query_prediction_session_1001.pkl"),
                os.path.join(temp_dir, "query_prediction_session_1002.pkl"), 
                os.path.join(temp_dir, "query_prediction_session_1003.pkl")
            ]
        })
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create query_results subdirectory for result files
        results_dir = os.path.join(temp_dir, "query_results")
        os.makedirs(results_dir)
        
        # Create sample result DataFrames
        result1_df = pd.DataFrame({
            "ra": [359.78, 357.68, 358.24],
            "dec": [36.23, 33.33, 35.23],
            "type": [3, 3, 3],
            "modelmag_g": [12.08, 11.76, 11.72],
            "objid": [758874373140775052, 758874297454690378, 758874298527646095]
        })
        result1_path = os.path.join(results_dir, "results_session_1001_query_0.pkl")
        with open(result1_path, "wb") as f:
            pickle.dump(result1_df, f)
            
        result2_df = pd.DataFrame({
            "ra": [360.12, 358.45],
            "dec": [37.45, 34.67], 
            "type": [3, 3],
            "modelmag_g": [13.21, 12.34],
            "objid": [758874373140775999, 758874297454691000]
        })
        result2_path = os.path.join(results_dir, "results_session_1001_query_1.pkl")
        with open(result2_path, "wb") as f:
            pickle.dump(result2_df, f)
            
        result3_df = pd.DataFrame({
            "ra": [361.45, 359.23, 357.89],
            "dec": [38.12, 35.78, 33.45],
            "type": [3, 3, 3], 
            "modelmag_g": [14.56, 13.78, 12.99],
            "objid": [758874373140776111, 758874297454691222, 758874298527647333]
        })
        result3_path = os.path.join(results_dir, "results_session_1001_query_3.pkl")
        with open(result3_path, "wb") as f:
            pickle.dump(result3_df, f)
            
        result4_df = pd.DataFrame({
            "ra": [362.11, 360.89, 359.56, 358.23],
            "dec": [39.45, 38.12, 36.78, 35.44],
            "type": [3, 3, 3, 3],
            "modelmag_g": [15.23, 14.67, 13.89, 13.12],
            "objid": [758874373140777000, 758874297454692000, 758874298527648000, 758874373140778000]
        })
        result4_path = os.path.join(results_dir, "results_session_1001_query_5.pkl")
        with open(result4_path, "wb") as f:
            pickle.dump(result4_df, f)
        
        # Session 1002 results
        result5_df = pd.DataFrame({
            "ra": [363.78, 362.45, 361.12],
            "dec": [40.23, 39.67, 38.89],
            "type": [3, 3, 3],
            "modelmag_g": [16.08, 15.76, 15.22],
            "objid": [758874373140779000, 758874297454693000, 758874298527649000]
        })
        result5_path = os.path.join(results_dir, "results_session_1002_query_0.pkl")
        with open(result5_path, "wb") as f:
            pickle.dump(result5_df, f)
            
        result6_df = pd.DataFrame({
            "ra": [364.12, 363.45],
            "dec": [41.45, 40.67], 
            "type": [3, 3],
            "modelmag_g": [17.21, 16.84],
            "objid": [758874373140780000, 758874297454694000]
        })
        result6_path = os.path.join(results_dir, "results_session_1002_query_2.pkl")
        with open(result6_path, "wb") as f:
            pickle.dump(result6_df, f)
            
        result7_df = pd.DataFrame({
            "ra": [365.45],
            "dec": [42.12],
            "type": [3], 
            "modelmag_g": [18.56],
            "objid": [758874373140781000]
        })
        result7_path = os.path.join(results_dir, "results_session_1002_query_4.pkl")
        with open(result7_path, "wb") as f:
            pickle.dump(result7_df, f)
        
        # Session 1003 single result
        result8_df = pd.DataFrame({
            "ra": [366.78],
            "dec": [43.23],
            "type": [3],
            "modelmag_g": [19.08],
            "objid": [758874373140782000]
        })
        result8_path = os.path.join(results_dir, "results_session_1003_query_0.pkl")
        with open(result8_path, "wb") as f:
            pickle.dump(result8_df, f)
        
        # Create session DataFrames that match the real structure
        session1_df = pd.DataFrame({
            "session_id": [1001, 1001, 1001, 1001],
            "query_position": [0, 1, 3, 5],  # Sequential with gaps
            "current_query": [
                "SELECT ra,dec,type FROM PhotoObj WHERE ra > 359", 
                "SELECT ra,dec,type FROM PhotoObj WHERE dec > 35",
                "SELECT ra,dec,type FROM PhotoObj WHERE ra > 361", 
                "SELECT ra,dec,type FROM PhotoObj WHERE dec > 39"
            ],
            "results_filepath": [result1_path, result2_path, result3_path, result4_path],
            "query_type": ["SELECT", "SELECT", "SELECT", "SELECT"],
            "query_length": [45, 46, 45, 46],
            "token_count": [8, 8, 8, 8],
            "has_join": [False, False, False, False],
            "has_where": [True, True, True, True],
            "result_column_count": [5, 5, 5, 5],
            "result_row_count": [3, 2, 3, 4]
        })
        with open(os.path.join(temp_dir, "query_prediction_session_1001.pkl"), "wb") as f:
            pickle.dump(session1_df, f)
        
        session2_df = pd.DataFrame({
            "session_id": [1002, 1002, 1002],
            "query_position": [0, 2, 4],  # Sequential with gaps
            "current_query": [
                "SELECT ra,dec,type FROM PhotoObj WHERE ra < 362",
                "SELECT ra,dec,type FROM PhotoObj WHERE dec < 41",
                "SELECT ra,dec,type FROM PhotoObj WHERE ra > 365"
            ],
            "results_filepath": [result5_path, result6_path, result7_path],
            "query_type": ["SELECT", "SELECT", "SELECT"],
            "query_length": [45, 45, 45],
            "token_count": [8, 8, 8],
            "has_join": [False, False, False],
            "has_where": [True, True, True],
            "result_column_count": [5, 5, 5],
            "result_row_count": [3, 2, 1]
        })
        with open(os.path.join(temp_dir, "query_prediction_session_1002.pkl"), "wb") as f:
            pickle.dump(session2_df, f)
            
        # Create a session with single query for edge case testing
        session3_df = pd.DataFrame({
            "session_id": [1003],
            "query_position": [0],
            "current_query": ["SELECT ra,dec,type FROM PhotoObj WHERE ra > 366"],
            "results_filepath": [result8_path],
            "query_type": ["SELECT"],
            "query_length": [45],
            "token_count": [8],
            "has_join": [False],
            "has_where": [True],
            "result_column_count": [5],
            "result_row_count": [1]
        })
        with open(os.path.join(temp_dir, "query_prediction_session_1003.pkl"), "wb") as f:
            pickle.dump(session3_df, f)
            
        yield temp_dir
