from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from query_data_predictor.sdss_csv_importer import SDSSCSVImporter

# Path to the sample data file - you'll need to create this sample CSV file
SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "SQL_workload1.csv"


class TestSDSSCSVImporter:
    @pytest.fixture
    def csv_importer(self):
        """Fixture to create and cleanup a SDSSCSVImporter instance."""
        importer = SDSSCSVImporter(SAMPLE_DATA_PATH)
        importer.load_data()
        yield importer
        # No close method in the CSV importer, so no cleanup needed

    def test_init_with_valid_file(self):
        """Test initialization with a valid CSV file."""
        importer = SDSSCSVImporter(SAMPLE_DATA_PATH)
        importer.load_data()
        assert importer is not None

    def test_init_with_invalid_file(self):
        """Test initialization with an invalid CSV file path."""
        with pytest.raises(FileNotFoundError):
            SDSSCSVImporter("nonexistent_file.csv").load_data()

    def test_get_sessions(self, csv_importer):
        """Test retrieving session IDs."""
        sessions = csv_importer.get_sessions()
        assert isinstance(sessions, np.ndarray)
        assert len(sessions) > 0
        assert sessions[0] == 11015
        assert sessions[-1] == 22195

        # Sessions might be strings or other types depending on your data

    def test_get_queries_for_session(self, csv_importer):
        """Test retrieving statement IDs for a specific session."""
        # Get the first session ID to test with
        sessions = csv_importer.get_sessions()
        test_session = sessions[0]

        # Get statements for this session
        statements = csv_importer.get_queries_for_session(test_session)
        assert isinstance(statements, np.ndarray)
        assert len(statements) > 0
        assert (
            statements[0]
            == "SELECT bestobjID,z FROM SpecObj WHERE (SpecClass=3 or SpecClass=4) and z between 1.95 and 2 and zConf>0.99"
        )

    def test_get_queries_for_nonexistent_session(self, csv_importer):
        """Test retrieving statement IDs for a session that doesn't exist."""
        statements = csv_importer.get_queries_for_session("nonexistent_session")
        assert isinstance(statements, np.ndarray)
        assert len(statements) == 0

    def test_load_data_format(self, csv_importer):
        """Test that the loaded data has the expected format."""
        # Verify the data was loaded as a pandas DataFrame
        assert hasattr(csv_importer, "data")
        assert isinstance(csv_importer.data, pd.DataFrame)

        # Check for required columns
        required_columns = ["session_id", "statement"]
        for column in required_columns:
            assert column in csv_importer.data.columns
