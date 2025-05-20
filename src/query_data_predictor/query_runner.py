import psycopg2
import pandas as pd


class QueryRunner:
    def __init__(self, dbname, user, host="localhost", port="5432"):
        self.db_params = {"dbname": dbname, "user": user, "host": host, "port": port}
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the PostgreSQL database."""
        self.conn = psycopg2.connect(**self.db_params)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def disconnect(self):
        """Disconnect from the PostgreSQL database."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def execute_query(self, query):
        """Execute a single query and return the results as a pandas DataFrame."""
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(result, columns=columns)
        return df

    def execute_queries(self, queries):
        """Execute multiple queries and return a list of pandas DataFrames."""
        results = []
        for query in queries:
            df = self.execute_query(query)
            results.append(df)
        return results

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
