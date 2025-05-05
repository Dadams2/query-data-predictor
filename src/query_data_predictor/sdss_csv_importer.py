import os
import pandas as pd
import numpy as np

SESSION_ID = "session_id"
STATEMENT = "statement"


class SDSSCSVImporter:
    """
    Class to import SDSS CSV files and convert them to a list of dictionaries.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self) -> list:
        """
        Load data from the CSV file and return it as a list of dictionaries.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        # Read the CSV file into a DataFrame
        return pd.read_csv(self.file_path, sep="$")

    def get_sessions(self) -> np.ndarray:
        """
        Return all unique session IDs from the data as a numpy array.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please check the file path.")

        # Extract unique session IDs
        return self.data[SESSION_ID].unique()

    def get_queries_for_session(self, session_id: str) -> np.ndarray:
        """
        Return all statement IDs for a given session ID as a numpy array.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please check the file path.")

        # Filter data for the given session ID
        session_data = self.data[self.data[SESSION_ID] == session_id]

        # Extract statement IDs
        return session_data["statement"].to_numpy()
