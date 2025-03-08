# config.py
import logging
import os
import json
import threading

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Logging setup
logging.basicConfig(filename=os.path.join(BASE_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

# Constants with absolute paths
CACHE_DB = os.path.join(BASE_DIR, "api_cache.db")
NYSE_TICKERS_FILE = os.path.join(BASE_DIR, "nyse_tickers.pkl")
NASDAQ_TICKERS_FILE = os.path.join(BASE_DIR, "nasdaq_tickers.pkl")
FAVORITES_FILE = os.path.join(BASE_DIR, "stock_favorites.json")
NYSE_LIST_FILE = os.path.join(BASE_DIR, "otherlisted.txt")
NASDAQ_LIST_FILE = os.path.join(BASE_DIR, "nasdaqlisted.txt")
FMP_API_KEYS = ["PhsBX3X3LzaW8tUlOSJfMnX9GJUqXmH8", "cZhYea4KCtMopxUvDdeN1HQMMfHgat91"]
ALPHA_VANTAGE_API_KEYS = ["PTSYEG3ID9GQNROF", "N1Q9W3HBJMZF4UI6"]
CACHE_EXPIRY = 365 * 24 * 60 * 60  # 365 days in seconds (31,536,000 seconds)

# Lock for favorites file access
FAVORITES_LOCK = threading.Lock()

def load_favorites():
    """Load favorite ticker lists with thread safety."""
    with FAVORITES_LOCK:
        try:
            if os.path.exists(FAVORITES_FILE):
                with open(FAVORITES_FILE, 'r') as f:
                    favorites = json.load(f)
                logging.info(f"Loaded {len(favorites)} favorite lists from {FAVORITES_FILE}")
                return favorites
            else:
                logging.warning(f"Favorites file {FAVORITES_FILE} not found. Starting with empty favorites.")
                return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in {FAVORITES_FILE}: {str(e)}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error loading {FAVORITES_FILE}: {str(e)}")
            return {}

def save_favorites(favorites):
    """Save favorite ticker lists with thread safety."""
    with FAVORITES_LOCK:
        try:
            logging.info(f"Saving {len(favorites)} favorite lists to {FAVORITES_FILE}")
            with open(FAVORITES_FILE, 'w') as f:
                json.dump(favorites, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving favorites to {FAVORITES_FILE}: {str(e)}")
            raise