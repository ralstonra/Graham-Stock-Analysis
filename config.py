import asyncio
from collections import deque
import logging
import os
import time
import threading
import hashlib
import logging.handlers

# Base directory (remains on OneDrive, where scripts are located)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Local user data directory (on each user's computer)
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "Graham Stock Analysis")
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# Associated file paths (now local)
CACHE_DB = os.path.join(USER_DATA_DIR, "api_cache.db")
NYSE_LIST_FILE = os.path.join(USER_DATA_DIR, "otherlisted.txt")
NASDAQ_LIST_FILE = os.path.join(USER_DATA_DIR, "nasdaqlisted.txt")
FAVORITES_FILE = os.path.join(USER_DATA_DIR, "stock_favorites.json")
screening_log_file = os.path.join(USER_DATA_DIR, "screening.log")
analyze_log_file = os.path.join(USER_DATA_DIR, "analyze.log")

# API keys and constants (unchanged)
FMP_API_KEYS = ["PhsBX3X3LzaW8tUlOSJfMnX9GJUqXmH8", "cZhYea4KCtMopxUvDdeN1HQMMfHgat91"]
FRED_API_KEY = "d28d895f1f3837fc6b9415baa3ce6061"
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
screening_logger.propagate = False

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
analyze_logger.propagate = False

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

class AsyncRateLimiter:
    def __init__(self, max_calls, period, on_sleep=None):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = asyncio.Lock()
        self.on_sleep = on_sleep

    async def acquire(self):
        async with self.lock:
            current_time = time.time()
            while self.calls and current_time - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current_time - self.calls[0])
                if sleep_time > 0:
                    if self.on_sleep:
                        self.on_sleep(sleep_time)
                    logging.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
            self.calls.append(time.time())

paid_rate_limiter = AsyncRateLimiter(MAX_CALLS_PER_MINUTE_PAID, 60)
free_rate_limiter = AsyncRateLimiter(MAX_CALLS_PER_MINUTE_FREE, 60)