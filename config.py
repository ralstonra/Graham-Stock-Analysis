import asyncio
from collections import deque
import logging
import os
import time
import threading
import hashlib
import logging.handlers

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CACHE_DB = os.path.join(DATA_DIR, "api_cache.db")
NYSE_TICKERS_FILE = os.path.join(DATA_DIR, "nyse_tickers.pkl")
NASDAQ_TICKERS_FILE = os.path.join(DATA_DIR, "nasdaq_tickers.pkl")
FAVORITES_FILE = os.path.join(DATA_DIR, "stock_favorites.json")
NYSE_LIST_FILE = os.path.join(DATA_DIR, "otherlisted.txt")
NASDAQ_LIST_FILE = os.path.join(DATA_DIR, "nasdaqlisted.txt")
FMP_API_KEYS = ["PhsBX3X3LzaW8tUlOSJfMnX9GJUqXmH8", "cZhYea4KCtMopxUvDdeN1HQMMfHgat91"]
FRED_API_KEY = "d28d895f1f3837fc6b9415baa3ce6061"

CACHE_EXPIRY = 365 * 24 * 60 * 60

FAVORITES_LOCK = threading.Lock()

MAX_CALLS_PER_MINUTE_PAID = 300
MAX_CALLS_PER_MINUTE_FREE = 5

class FileHashError(Exception):
    pass

# Logging setup with daily rotation
log_file = os.path.join(DATA_DIR, 'nyse_graham_screen.log')
handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)  # Rotate daily, keep 7 backups
handler.setLevel(logging.DEBUG)  # Capture all levels to file
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logging.basicConfig(handlers=[handler], level=logging.INFO)  # Default to INFO for production
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)  # Console can show DEBUG for development
logging.getLogger().addHandler(console)

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