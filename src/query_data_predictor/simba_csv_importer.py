import os
import pandas as pd
import numpy as np
from query_data_predictor.importer import DataImporter

QUERY_ID = "query_id"
SQL_QUERY = "sql_query"


class SimbaCSVImporter(DataImporter):
    """
    Class to import SIMBA CSV files and convert them to a list of dictionaries.
    Each CSV file represents a single session.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        # Extract session ID from filename (e.g., "CircActivity_testing_session1_simple.csv" -> "session1")
        self.session_id = self._extract_session_id(file_path)

    def _extract_session_id(self, file_path: str) -> str:
        """
        Extract session ID from the filename.
        Assumes format like: CircActivity_testing_session{id}_simple.csv
        """
        filename = os.path.basename(file_path)
        # Look for 'session' followed by a number
        import re
        match = re.search(r'session(\d+)', filename)
        if match:
            return f"{match.group(1)}"
        # Fallback: use the filename without extension
        return os.path.splitext(filename)[0]

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the CSV file and return it as a DataFrame.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.data = pd.read_csv(self.file_path)
        
        # Validate that required columns exist
        if QUERY_ID not in self.data.columns or SQL_QUERY not in self.data.columns:
            raise ValueError(
                f"CSV file must contain '{QUERY_ID}' and '{SQL_QUERY}' columns. "
                f"Found columns: {list(self.data.columns)}"
            )
        
        return self.data

    def get_sessions(self) -> np.ndarray:
        """
        Return all unique session IDs from the data as a numpy array.
        For SIMBA data, each file represents one session.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Return the session ID as a single-element array
        return np.array([self.session_id])

    def get_queries_for_session(self, session_id: str) -> np.ndarray:
        """
        Return all SQL queries for a given session ID as a numpy array.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Validate session ID matches this file's session
        if session_id != self.session_id:
            raise ValueError(
                f"Session ID mismatch. Expected '{self.session_id}', got '{session_id}'"
            )

        # Extract SQL queries in order of query_id
        queries = self.data.sort_values(QUERY_ID)[SQL_QUERY].to_numpy()
        return queries
