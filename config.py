# config.py: Configuration settings and utilities for the stock analysis application

import asyncio
from collections import deque
import logging
import os
import time
import threading

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")  # Define data subdirectory

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Logging setup
logging.basicConfig(filename=os.path.join(DATA_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

# Constants with absolute paths using DATA_DIR
CACHE_DB = os.path.join(DATA_DIR, "api_cache.db")
NYSE_TICKERS_FILE = os.path.join(DATA_DIR, "nyse_tickers.pkl")
NASDAQ_TICKERS_FILE = os.path.join(DATA_DIR, "nasdaq_tickers.pkl")
FAVORITES_FILE = os.path.join(DATA_DIR, "stock_favorites.json")
NYSE_LIST_FILE = os.path.join(DATA_DIR, "otherlisted.txt")
NASDAQ_LIST_FILE = os.path.join(DATA_DIR, "nasdaqlisted.txt")
FMP_API_KEYS = ["PhsBX3X3LzaW8tUlOSJfMnX9GJUqXmH8", "cZhYea4KCtMopxUvDdeN1HQMMfHgat91"]  # Replace with your actual FMP API keys

# Cache expiration time (365 days in seconds)
CACHE_EXPIRY = 365 * 24 * 60 * 60  # Historic Data rarely change, so cache for a year

# Lock for favorites file access
FAVORITES_LOCK = threading.Lock()

# Rate Limiter Class for managing API call frequency
class RateLimiter:
    def __init__(self, max_calls, period):
        """Initialize rate limiter with max calls and period in seconds."""
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make an API call, respecting rate limits."""
        async with self.lock:
            current_time = time.time()
            # Remove expired calls
            while self.calls and current_time - self.calls[0] >= self.period:
                self.calls.popleft()
            # Wait if limit reached
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.period - current_time
                logging.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                current_time = time.time()
            self.calls.append(current_time)

# Global rate limiter instance: 300 calls per 60 seconds
rate_limiter = RateLimiter(300, 60)