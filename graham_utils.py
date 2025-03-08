import logging
import os
import pickle
import hashlib
from config import (CACHE_DB, NYSE_TICKERS_FILE, NASDAQ_TICKERS_FILE, FAVORITES_FILE,
                   FMP_API_KEYS, ALPHA_VANTAGE_API_KEYS, CACHE_EXPIRY, BASE_DIR)

# Logging setup (consistent with config.py)
logging.basicConfig(filename=os.path.join(BASE_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

def parse_tickers(input_str, exchange="Stock"):
    """Parse and validate ticker input for a specific exchange."""
    if not input_str or not isinstance(input_str, str):
        return []
    
    tickers = [t.strip().upper() for t in input_str.split(',') if t.strip() and t.strip().isalnum() and len(t.strip()) <= 5]
    seen = set()
    unique_tickers = [t for t in tickers if t not in seen and not seen.add(t)]
    logging.debug(f"Parsed {exchange} tickers: {unique_tickers}")
    return unique_tickers

def clear_cache():
    """Clear the API cache database and ticker pickle files."""
    try:
        if os.path.exists(CACHE_DB):
            os.remove(CACHE_DB)
            logging.info(f"Cleared API cache database: {CACHE_DB}")
        
        for file in [NYSE_TICKERS_FILE, NASDAQ_TICKERS_FILE]:
            if os.path.exists(file):
                os.remove(file)
                logging.info(f"Removed cached ticker file: {file}")
    except Exception as e:
        logging.error(f"Error clearing cache: {str(e)}")
        raise