import os
import hashlib
from query_data_predictor.query_runner import QueryRunner
from query_data_predictor.dataset_creator import DatasetCreator
from pathlib import Path
from query_data_predictor.simba_csv_importer import SimbaCSVImporter
import logging

# Set up logging for this script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded database configuration for SIMBA
DB_NAME = "simba_sdss"
DB_USER = os.getenv("PG_DATA_USER", "postgres")  # fallback to 'postgres' if not set
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")

# Paths for SIMBA data
SAMPLE_CSV_PATH = Path(__file__).parent.parent / "data" / "datasets" / "simba_SDSS" / "SDSS_pattern_drilldown_session2.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets" / "simba_SDSS"
QUERY_RESULTS_DIR = Path(__file__).parent.parent / "data" / "datasets" / "simba_SDSS" / "query_results"


# SAMPLE_CSV_PATH = Path(__file__).parent.parent / "data" / "simba_test" / "CircActivity_testing_session1_simple.csv"
# OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets" / "simba_simple"
# QUERY_RESULTS_DIR = Path(__file__).parent.parent / "data" / "datasets" / "simba_simple" / "query_results"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize components
runner = QueryRunner(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
loader = SimbaCSVImporter(SAMPLE_CSV_PATH)
creator = DatasetCreator(data_loader=loader, query_runner=runner, output_dir=OUTPUT_DIR, results_dir=QUERY_RESULTS_DIR)

# Build the dataset
dataset_info = creator.build_dataset()

logger.info(f"Built dataset with {len(dataset_info)} sessions")
logger.info(f"Available sessions: {creator.data_loader.get_sessions()}")

# Validate the first session file
if len(dataset_info) > 0:
    file = dataset_info['filepath'][0]
    
    # Log if file exists and its size
    logger.info(f"File exists: {os.path.exists(file)}")
    logger.info(f"File size: {os.path.getsize(file)} bytes")
    
    # Log md5 hash of the file
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.") 
    if os.path.getsize(file) == 0:
        raise ValueError(f"File {file} is empty.")  
    
    with open(file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    logger.info(f"MD5 hash of the file: {file_hash}")
else:
    logger.warning("No dataset info generated!")
