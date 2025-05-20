import os
import pytest
import polars as pl
import hashlib
from dotenv import load_dotenv
from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.dataset_creator import DatasetCreator
from pathlib import Path
from query_data_predictor.sdss_csv_importer import SDSSCSVImporter

# Load environment variables from .env file
load_dotenv()

TEST_QUERY = "SELECT bestobjID,z FROM SpecObj WHERE (SpecClass=3 or SpecClass=4) and z between 1.95 and 2 and zConf>0.99"
SAMPLE_CSV_PATH = Path(__file__).parent.parent / "data" / "SQL_workload1.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets"
QUERY_RESULTS_DIR = Path(__file__).parent.parent / "data" / "datasets" / "query_results"

# Get database connection parameters from environment variables
DB_NAME = os.getenv("PG_DATA")
DB_USER = os.getenv("PG_DATA_USER")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")

@pytest.fixture
def query_runner():
    """Create and return a QueryRunner instance with connection params from env vars."""
    runner = QueryRunner(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
    return runner



def test_full_build(query_runner):
    loader = SDSSCSVImporter(SAMPLE_CSV_PATH)
    creator = DatasetCreator(data_loader=loader, query_runner=query_runner, output_dir=OUTPUT_DIR, results_dir=QUERY_RESULTS_DIR)
    dataset_info = creator.build_dataset()
    # assert len(dataset_info) == 13 

    # check number of files created is equal to number of sessions
    assert len(dataset_info) == len(creator.data_loader.get_sessions())
    file = dataset_info[1]["file_path"]
    assert os.path.exists(file)
    md5 = hashlib.md5(open(file, "rb").read()).hexdigest()
    assert md5 == "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"