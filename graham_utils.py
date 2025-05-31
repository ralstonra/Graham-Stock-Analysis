import logging
import os
import pickle
import hashlib
import aiohttp
import asyncio
from config import (CACHE_DB, NYSE_TICKERS_FILE, NASDAQ_TICKERS_FILE, FAVORITES_FILE,
                   FMP_API_KEYS, ALPHA_VANTAGE_API_KEYS, CACHE_EXPIRY, USER_DATA_DIR)

# Logging setup (consistent with config.py)
logging.basicConfig(filename=os.path.join(USER_DATA_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

async def validate_api_key(api_key: str, api_type: str = 'FMP') -> bool:
    """
    Validate an API key by making a test request.

    Args:
        api_key (str): The API key to validate.
        api_type (str): The type of API ('FMP' or 'AlphaVantage').

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    if api_type == 'FMP':
        url = f"https://financialmodelingprep.com/api/v3/profile/AAPL?apikey={api_key}"
    elif api_type == 'AlphaVantage':
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey={api_key}"
    else:
        logging.error(f"Unknown API type: {api_type}")
        return False
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and isinstance(data, list) and len(data) > 0 and 'symbol' in data[0]:
                        logging.info(f"API key validated for {api_type}")
                        return True
                    else:
                        logging.warning(f"API key validation failed for {api_type}: Unexpected response data")
                        return False
                else:
                    logging.warning(f"API key validation failed for {api_type}: Status {resp.status}")
                    return False
        except Exception as e:
            logging.error(f"Error validating API key for {api_type}: {str(e)}")
            return False

def parse_tickers(input_str, exchange="Stock"):
    """
    Parse and validate ticker input for a specific exchange.

    Args:
        input_str (str): Comma-separated string of tickers.
        exchange (str): The exchange type (default is 'Stock').

    Returns:
        list: List of unique, validated ticker symbols.
    """
    if not input_str or not isinstance(input_str, str):
        return []
    
    tickers = [t.strip().upper() for t in input_str.split(',') if t.strip() and t.strip().isalnum() and len(t.strip()) <= 5]
    seen = set()
    unique_tickers = [t for t in tickers if t not in seen and not seen.add(t)]
    logging.debug(f"Parsed {exchange} tickers: {unique_tickers}")
    return unique_tickers

def clear_cache():
    """
    Clear the API cache database, ticker pickle files, and in-memory caches if applicable.

    Raises:
        Exception: If an error occurs during cache clearing.
    """
    try:
        if os.path.exists(CACHE_DB):
            os.remove(CACHE_DB)
            logging.info(f"Cleared API cache database: {CACHE_DB}")
        
        for file in [NYSE_TICKERS_FILE, NASDAQ_TICKERS_FILE]:
            if os.path.exists(file):
                os.remove(file)
                logging.info(f"Removed cached ticker file: {file}")
        
        # Attempt to clear in-memory caches from graham_data.py if they exist in the global scope
        if 'FMP_DATA_CACHE' in globals():
            globals()['FMP_DATA_CACHE'].clear()
            logging.info("Cleared in-memory FMP_DATA_CACHE")
        if 'YAHOO_DATA_CACHE' in globals():
            globals()['YAHOO_DATA_CACHE'].clear()
            logging.info("Cleared in-memory YAHOO_DATA_CACHE")
        
    except Exception as e:
        logging.error(f"Error clearing cache: {str(e)}")
        raise