from abc import ABC, abstractmethod

## abstract class for data importers
class DataImporter(ABC):
    """
    Abstract base class for data importers.
    """

    @abstractmethod
    def load_data(self):
        """
        Load data from the source.
        """
        pass

    @abstractmethod
    def get_sessions(self):
        """
        Get all unique session IDs.
        """
        pass

    @abstractmethod
    def get_queries_for_session(self, session_id):
        """
        Get all queries for a given session ID.
        """
        pass