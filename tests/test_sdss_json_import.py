from pathlib import Path
import pytest
import numpy as np
from query_data_predictor.sdss_json_importer import JsonDataImporter

# Path to the sample data file
SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sdss_joined_sample.json"


class TestDataLoader:
    @pytest.fixture
    def data_loader(self):
        """Fixture to create and cleanup a DataLoader instance."""
        loader = JsonDataImporter(SAMPLE_DATA_PATH)
        yield loader
        loader.close()

    def test_init_with_valid_file(self):
        """Test initialization with a valid JSON file."""
        loader = JsonDataImporter(SAMPLE_DATA_PATH)
        assert loader is not None
        loader.close()

    def test_init_with_invalid_file(self):
        """Test initialization with an invalid JSON file path."""
        with pytest.raises(FileNotFoundError):
            JsonDataImporter("nonexistent_file.json")

    def test_get_sessions(self, data_loader):
        """Test retrieving session IDs."""
        sessions = data_loader.get_sessions()
        assert isinstance(sessions, np.ndarray)
        assert len(sessions) > 0
        assert isinstance(sessions[0], np.int64)
        assert sessions[0] == 28

    def test_get_queries_for_session(self, data_loader):
        """Test retrieving statement IDs for a specific session."""
        # Get the first session ID to test with
        sessions = data_loader.get_sessions()
        test_session = sessions[0]

        # Get statements for this session
        statements = data_loader.get_queries_for_session(test_session)
        assert isinstance(statements, np.ndarray)
        assert len(statements) > 0
        assert isinstance(statements[0], str)
        assert (
            statements[0]
            == "SELECT * from dbo.fRegionsContainingPointEq(135.76428223, 49.32302094,'RUN', 0.025)"
        )

    def test_get_queries_for_nonexistent_session(self, data_loader):
        """Test retrieving statement IDs for a session that doesn't exist."""
        statements = data_loader.get_queries_for_session(99999999999)
        assert isinstance(statements, np.ndarray)
        assert len(statements) == 0
