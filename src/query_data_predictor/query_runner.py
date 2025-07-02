import psycopg2
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class QueryRunner:
    def __init__(self, dbname, user, host="localhost", port="5432"):
        self.db_params = {"dbname": dbname, "user": user, "host": host, "port": port}
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            logger.info(f"Connecting to database: {self.db_params['dbname']}@{self.db_params['host']}:{self.db_params['port']}")
            self.conn = psycopg2.connect(**self.db_params)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            logger.debug("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """Disconnect from the PostgreSQL database."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.debug("Database connection closed successfully")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")

    def execute_query(self, query):
        """Execute a single query and return the results as a pandas DataFrame."""
        try:
            logger.debug(f"Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(result, columns=columns)
            logger.debug(f"Query returned {len(df)} rows with columns: {columns}")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Failed query: {query}")
            raise

    def execute_queries(self, queries):
        """Execute multiple queries and return a list of pandas DataFrames."""
        logger.info(f"Executing batch of {len(queries)} queries")
        results = []
        for i, query in enumerate(queries):
            try:
                df = self.execute_query(query)
                results.append(df)
            except Exception as e:
                logger.error(f"Failed to execute query {i+1}/{len(queries)}: {e}")
                raise
        logger.info(f"Successfully executed {len(results)} queries")
        return results

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
