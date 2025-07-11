# this class is used to read and get the result data for queries. It is given a data path which should contain a metadata.csv file with the session_id and filepath columns for the queries
import pandas as pd
import numpy as np 
import pickle
import pathlib
import logging
from query_data_predictor.importer import DataImporter

logger = logging.getLogger(__name__)

class DataLoader():
    """
    Class to import data from a CSV file and convert it to a list of dictionaries.
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = pathlib.Path(dataset_dir)
        # read in metadata.csv 
        self.file_path = self.dataset_dir / "metadata.csv"
        if not self.file_path.exists():
            logger.error(f"CSV file not found: {self.file_path}")
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        self.metadata = pd.read_csv(self.file_path)
        self.memory_cache = {}
        logger.info(f"DataLoader initialized with {len(self.metadata)} sessions from {self.file_path}")
    
    # TODO LRU cache for results per session or something
    def get_results_for_query(self, session_id: int, query_id: int) -> np.ndarray:
        """
        Get the results for a specific query in a session.
        
        Args:
            session_id: The ID of the session
            query_id: The ID of the query
            
        Returns:
            The query results as a numpy array
        """
        # Find the specific query in the metadata

        if session_id not in self.memory_cache:
            self.get_results_for_session(session_id)
        
        data = self.memory_cache[session_id]

        query_rows = data[
            (data["session_id"] == session_id) & 
            (data["query_position"] == query_id)
        ]
        
        if len(query_rows) == 0:
            raise ValueError(f"Query ID {query_id} not found in session {session_id}")
        
        # Get the results file path
        if "results_filepath" not in data.columns:
            raise ValueError("Metadata does not contain a results_filepath column")
        
        results_file_path = query_rows["results_filepath"].values[0]
        results_path = self.dataset_dir / results_file_path
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        # Load and return the actual query results
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        
        return results


    def get_sessions(self):
        """
        Get all available sessions with their metadata.
        
        Returns:
            Dictionary mapping session IDs to session information
        """
        # sessions = {}
        # for session_id in self.metadata["session_id"].unique():
        #     # Get session data to populate session information
        #     try:
        #         session_data = self.get_results_for_session(session_id)
                
        #         # Create a session entry with basic information
        #         session_info = {
        #             'id': session_id,
        #             'queries': session_data if isinstance(session_data, dict) else 
        #                       {row['query_position']: row for _, row in session_data.iterrows()} 
        #                       if hasattr(session_data, 'iterrows') else {}
        #         }
                
        #         sessions[session_id] = session_info
        #     except (FileNotFoundError, ValueError) as e:
        #         # Skip sessions with missing files
        #         print(f"Warning: Could not load session {session_id}: {e}")
        
        # return sessions
        return self.metadata["session_id"].unique().tolist()
        
    def get_results_for_session(self, session_id: int):
        """
        Return all statement results for a given session ID.
        
        Args:
            session_id: The ID of the session to retrieve
            
        Returns:
            The data for the session, which could be a DataFrame or dictionary
        """
        # Find the session in the metadata
        session_rows = self.metadata[self.metadata["session_id"] == session_id]
        if len(session_rows) == 0:
            # check if session id is a string and add to error message 
            if isinstance(session_id, str):
                raise ValueError(f"Session ID '{session_id}' is a string not found in metadata")
            else:
                raise ValueError(f"Session ID {session_id} not found in metadata")
            
        # Get the file path from the metadata
        if "filepath" in self.metadata.columns:
            file_path_col = "filepath"
        elif "path" in self.metadata.columns:
            file_path_col = "path"
        else:
            raise ValueError("Metadata does not contain a filepath or path column")
            
        data_path = session_rows[file_path_col].values[0]
        data_path = self.dataset_dir / data_path
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dump file not found: {data_path}")
            
        # Read in the dump file
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        self.memory_cache[session_id] = data
        return data