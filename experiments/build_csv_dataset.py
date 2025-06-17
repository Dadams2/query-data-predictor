import os
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

runner = QueryRunner(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)



loader = SDSSCSVImporter(SAMPLE_CSV_PATH)
creator = DatasetCreator(data_loader=loader, query_runner=runner, output_dir=OUTPUT_DIR, results_dir=QUERY_RESULTS_DIR)
dataset_info = creator.build_dataset()

print(len(dataset_info))
print(creator.data_loader.get_sessions())
file = dataset_info[1]["file_path"]
# print if file exists and its size
print(f"File exists: {os.path.exists(file)}")
print(f"File size: {os.path.getsize(file)} bytes")
# print md5 hash of the file
import hashlib
if not os.path.exists(file):
    raise FileNotFoundError(f"File {file} does not exist.") 
if os.path.getsize(file) == 0:
    raise ValueError(f"File {file} is empty.")  
with open(file, "rb") as f:
    file_hash = hashlib.md5(f.read()).hexdigest()
print(f"MD5 hash of the file: {file_hash}") 
