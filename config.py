import os
import logging
import threading
import hashlib
import logging.handlers
from decouple import config

# Base directory (remains on OneDrive, where scripts are located)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Local user data directory (on each user's computer)
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "Graham Stock Analysis")
try:
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR)
except (OSError, PermissionError) as e:
    logging.error(f"Failed to create USER_DATA_DIR {USER_DATA_DIR}: {str(e)}")
    raise

# Associated file paths (now local)
CACHE_DB = os.path.join(USER_DATA_DIR, "api_cache.db")
NYSE_LIST_FILE = os.path.join(USER_DATA_DIR, "otherlisted.txt")
NASDAQ_LIST_FILE = os.path.join(USER_DATA_DIR, "nasdaqlisted.txt")
FAVORITES_FILE = os.path.join(USER_DATA_DIR, "stock_favorites.json")
screening_log_file = os.path.join(USER_DATA_DIR, "screening.log")
analyze_log_file = os.path.join(USER_DATA_DIR, "analyze.log")

# API keys and constants
FMP_API_KEYS = config('FMP_API_KEYS', cast=lambda v: [s.strip() for s in v.split(',')])
FRED_API_KEY = config('FRED_API_KEY')
USE_FREE_API_KEY = config('USE_FREE_API_KEY', default=False, cast=bool)  # Added
CACHE_EXPIRY = 365 * 24 * 60 * 60
FAVORITES_LOCK = threading.Lock()
MAX_CALLS_PER_MINUTE_PAID = 300
MAX_CALLS_PER_MINUTE_FREE = 5

class FileHashError(Exception):
    pass

# Screening logger setup with size-based rotation
screening_handler = logging.handlers.RotatingFileHandler(
    screening_log_file, maxBytes=10*1024*1024, backupCount=5
)
screening_handler.setLevel(logging.INFO)
screening_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
screening_handler.setFormatter(screening_formatter)
screening_logger = logging.getLogger('screening')
screening_logger.addHandler(screening_handler)
screening_logger.setLevel(logging.DEBUG)
screening_logger.propagate = True

# Analyze logger setup with size-based rotation
analyze_handler = logging.handlers.RotatingFileHandler(
    analyze_log_file, maxBytes=10*1024*1024, backupCount=5
)
analyze_handler.setLevel(logging.INFO)
analyze_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
analyze_handler.setFormatter(analyze_formatter)
analyze_logger = logging.getLogger('analyze')
analyze_logger.addHandler(analyze_handler)
analyze_logger.setLevel(logging.DEBUG)
analyze_logger.propagate = True

# Console handler for development
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
screening_logger.addHandler(console)
analyze_logger.addHandler(console)

def get_file_hash(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        raise FileHashError(f"File not found: {file_path}")
    except Exception as e:
        raise FileHashError(f"Error computing hash for {file_path}: {str(e)}")