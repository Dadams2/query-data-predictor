import psycopg2
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging
from .importer import DataImporter

logger = logging.getLogger(__name__)

# Constants
SESSION_ID_COLUMN = "sessionID"
STATEMENT_COLUMN = "statement"


class JsonDataImporter(DataImporter):
    def __init__(
        self, json_file_path: str, db_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the data loader with the path to the JSON file and optional database connection parameters.
        If db_params is not provided, parameters will be loaded from .env file.

        Args:
            json_file_path: Path to the JSON data file
            db_params: Optional dictionary with PostgreSQL connection parameters:
                       {'dbname': str, 'user': str, 'password': str, 'host': str, 'port': int}
        """
        self.json_file_path: str = json_file_path
        self.db_params = db_params if db_params else self._load_db_params_from_env()
        self.conn: Optional[psycopg2.extensions.connection] = None
        self._connect_to_db()

    def _load_db_params_from_env(self) -> Dict[str, Any]:
        """Load database parameters from environment variables (.env file)."""
        # Try to load from .env file (will not override existing env variables)
        load_dotenv()

        # Check if required environment variables exist
        required_params = [
            "PG_SESSION_DBNAME",
            "PG_SESSION_USER",
            "PG_SESSION_PASSWORD",
            "PG_HOST",
            "PG_PORT",
        ]
        missing_params = [param for param in required_params if not os.getenv(param)]

        if missing_params:
            raise ValueError(
                f"Missing required database parameters in .env file: {', '.join(missing_params)}\n"
                "Please create a .env file with the required parameters or provide them explicitly."
            )

        # Build and return the database parameters dictionary
        return {
            "dbname": os.getenv("PG_SESSION_DBNAME"),
            "user": os.getenv("PG_SESSION_USER"),
            "password": os.getenv("PG_SESSION_PASSWORD"),
            "host": os.getenv("PG_HOST"),
            "port": int(os.getenv("PG_PORT", "5432")),
        }

    def _create_engine(self):
        """Create a SQLAlchemy engine for the PostgreSQL database."""
        try:
            return create_engine(
                f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
            )
        except Exception as e:
            raise ConnectionError(f"Could not create SQLAlchemy engine: {e}")

    def _connect_to_db(self) -> None:
        """Establish connection to PostgreSQL database, creating it if it doesn't exist."""
        # First try to connect to the postgres database to check if our target DB exists
        temp_params = self.db_params.copy()
        target_dbname = temp_params["dbname"]
        temp_params["dbname"] = "postgres"  # Connect to default postgres database

        try:
            # Connect to postgres database
            temp_conn = psycopg2.connect(**temp_params)
            temp_conn.autocommit = True  # Needed for CREATE DATABASE

            with temp_conn.cursor() as cur:
                # Check if the target database exists
                cur.execute(
                    f"SELECT 1 FROM pg_database WHERE datname = '{target_dbname}'",
                )
                exists = cur.fetchone()

                if not exists:
                    # Create the database
                    cur.execute(f"CREATE DATABASE {target_dbname}")
                    logger.info(f"Created database: {target_dbname}")

            # Close the temporary connection
            temp_conn.close()

            # Now connect to the target database
            self.conn = psycopg2.connect(**self.db_params)

        except psycopg2.Error as e:
            raise ConnectionError(
                f"Could not connect to or create PostgreSQL database: {e}"
            )

    def load_data(self) -> None:
        """Load JSON data into PostgreSQL using pandas and psycopg2."""
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")

        try:
            # Read JSON data into a pandas DataFrame
            data = pd.read_json(self.json_file_path)

            # Check if the required columns exist
            if (
                SESSION_ID_COLUMN not in data.columns
                or STATEMENT_COLUMN not in data.columns
            ):
                raise ValueError(
                    f"JSON file must contain columns: {SESSION_ID_COLUMN}, {STATEMENT_COLUMN}"
                )

            # create sqlachemy engine for writing tables
            engine = self._create_engine()

            data.to_sql(
                "query_data_view",
                con=engine,
                if_exists="replace",
                index=False,
                method="multi",
            )

        except Exception as e:
            raise Exception(f"Error loading data into PostgreSQL: {e}")

    def get_sessions(self) -> List[int]:
        """Return all unique session IDs from the data as a numpy array."""
        if not self.conn:
            raise ConnectionError("Database connection not established")

        with self.conn.cursor() as cur:
            cur.execute(
                f'''
                SELECT DISTINCT "{SESSION_ID_COLUMN}"
                FROM query_data_view
                ORDER BY "{SESSION_ID_COLUMN}"
            '''
            )
            result = cur.fetchall()
        # Convert the result to a numpy array of session IDs
        return np.array([row[0] for row in result])

    def get_queries_for_session(self, session_id: str) -> np.ndarray:
        """Return all statement IDs for a given session ID as a numpy array."""
        if not self.conn:
            raise ConnectionError("Database connection not established")

        with self.conn.cursor() as cur:
            cur.execute(
                f'''
                SELECT "{STATEMENT_COLUMN}"
                FROM query_data_view
                WHERE "{SESSION_ID_COLUMN}" = {session_id} 
                ORDER BY "rankInSession"
            '''
            )
            result = cur.fetchall()

        # Convert the result to a numpy array of statement IDs
        return np.array([row[0] for row in result])

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
