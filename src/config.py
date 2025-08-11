import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tsfresh.feature_extraction import MinimalFCParameters

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parents[1]

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
QUERY_DIR = DATA_DIR / "sql"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
FEATURE_STORE_DIR = DATA_DIR / "feature-store"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_EDA_DIR = FIGURES_DIR / "EDA"
MLFLOW_DIR = PROJ_ROOT / "mlruns"
SPMF_JAR_FILE = PROJ_ROOT / "references" / "spmf.jar"

#####
# Database configuration
#####
DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
TRUSTED_CONNECTION = os.getenv("TRUSTED_CONNECTION")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
file_names = [f.stem for f in QUERY_DIR.glob("*.txt")]
QUERY_PATHS = {name: f"{QUERY_DIR}/{name}.txt" for name in file_names}


class SparkConfig:
    def __init__(
        self, driver_memory="40g", executor_memory="32g", cores=4, shuffle_partitions=16
    ):
        self.driver_memory = driver_memory
        self.executor_memory = executor_memory
        self.cores = cores
        self.shuffle_partitions = shuffle_partitions
