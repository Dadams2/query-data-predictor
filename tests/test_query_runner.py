import os
import pytest
import polars as pl
from dotenv import load_dotenv
from query_data_predictor.query_runner import QueryRunner

# Load environment variables from .env file
load_dotenv()

TEST_QUERY = "SELECT bestobjID,z FROM SpecObj WHERE (SpecClass=3 or SpecClass=4) and z between 1.95 and 2 and zConf>0.99"

# Get database connection parameters from environment variables
DB_NAME = os.getenv("PG_DATA")
DB_USER = os.getenv("PG_DATA_USER")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")


@pytest.fixture
def query_runner():
    """Create and return a QueryRunner instance with connection params from env vars."""
    runner = QueryRunner(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT
    )
    return runner


def test_connection(query_runner):
    """Test that the connection can be established and closed."""
    try:
        query_runner.connect()
        assert query_runner.conn is not None
        assert query_runner.cursor is not None
    finally:
        query_runner.disconnect()
        assert query_runner.conn is None or query_runner.conn.closed == 1


def test_execute_query(query_runner):
    """Test executing a simple query."""
    try:
        query_runner.connect()
        # Simple test query that should work on any PostgreSQL database
        df = query_runner.execute_query("SELECT 1 as test_col")
        
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (1, 1)
        assert df.columns == ["test_col"]
        assert df[0, 0] == 1
    finally:
        query_runner.disconnect()
         

def test_sdss_quewry(query_runner):
    """Test executing a simple query."""
    try:
        query_runner.connect()
        # Simple test query that should work on any PostgreSQL database
        df = query_runner.execute_query(TEST_QUERY)
        
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (9, 2)
        assert df.columns == ["bestobjid", "z"]
    finally:
        query_runner.disconnect()


def test_execute_queries(query_runner):
    """Test executing multiple queries."""
    try:
        query_runner.connect()
        queries = [
            "SELECT 1 as test_col",
            "SELECT 2 as test_col",
        ]
        
        results = query_runner.execute_queries(queries)
        
        assert len(results) == 2
        assert all(isinstance(df, pl.DataFrame) for df in results)
        assert results[0][0, 0] == 1
        assert results[1][0, 0] == 2
    finally:
        query_runner.disconnect()


def test_context_manager():
    """Test using the QueryRunner as a context manager."""
    with QueryRunner(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT
    ) as runner:
        # Connection should be established
        assert runner.conn is not None
        assert runner.cursor is not None
        
        # Execute a simple query
        df = runner.execute_query("SELECT 1 as test_col")
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (1, 1)
    
    # Connection should be closed after exiting context
    assert runner.conn is None or runner.conn.closed == 1
