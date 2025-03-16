# graham_data.py: Data fetching, caching, and Graham score calculations for stock analysis

import sqlite3
import time
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime
from config import FMP_API_KEYS, rate_limiter, CACHE_DB, NYSE_LIST_FILE, NASDAQ_LIST_FILE, DATA_DIR
import os
import json
import pandas as pd
import threading
import hashlib
from config import FAVORITES_FILE
import numpy as np

# Constants
MAX_CONCURRENT_TICKERS = 15
MAX_CALLS_PER_MINUTE = 300
MAX_TICKERS_PER_MINUTE = 100
DELAY_BETWEEN_CALLS = 0.1
SECONDS_PER_MINUTE = 60
CACHE_EXPIRY = 365 * 24 * 60 * 60  # 365 days in seconds
BATCH_SIZE = 100  # Number of tickers per batch for IPO date fetching

# Thread-safe locks
FETCHED_TICKERS = set()
FETCHED_TICKERS_LOCK = threading.Lock()
FMP_DATA_CACHE = {}
FMP_CACHE_LOCK = threading.Lock()
YAHOO_DATA_CACHE = {}
YAHOO_CACHE_LOCK = threading.Lock()

# Rate limiters
FMP_LIMITER = asyncio.Semaphore(MAX_CALLS_PER_MINUTE)
YAHOO_LIMITER = asyncio.Semaphore(MAX_CALLS_PER_MINUTE)

def get_stocks_connection():
    """Establish a connection to the SQLite database and set up schema."""
    try:
        conn = sqlite3.connect(CACHE_DB, timeout=30)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [col[1] for col in cursor.fetchall()]
        expected_columns = ["ticker", "date", "price", "roe", "rotc", "eps", "dividend", "ticker_list_hash", "balance_data", "ipo_date", "timestamp"]

        if "timestamp" not in columns:
            logging.info("Adding 'timestamp' column to 'stocks' table.")
            cursor.execute("ALTER TABLE stocks ADD COLUMN timestamp REAL")
            conn.commit()

        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks
                         (ticker TEXT PRIMARY KEY, date TEXT, price REAL, roe TEXT, rotc TEXT, eps TEXT, 
                          dividend TEXT, ticker_list_hash TEXT, balance_data TEXT, ipo_date TEXT, timestamp REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS screening_progress
                         (exchange TEXT, ticker TEXT, timestamp TEXT, file_hash TEXT, status TEXT,
                          PRIMARY KEY (exchange, ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS graham_qualifiers
                         (ticker TEXT PRIMARY KEY, graham_score INTEGER, date TEXT, sector TEXT, exchange TEXT)''')
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize database connection: {str(e)}")
        raise

def get_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file for versioning."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error computing hash for {file_path}: {str(e)}")
        return ""

class TickerManager:
    def __init__(self, nyse_file: str, nasdaq_file: str):
        """Initialize ticker manager with NYSE and NASDAQ file paths."""
        self.nyse_tickers = set()
        self.nasdaq_tickers = set()
        self.nyse_file = nyse_file
        self.nasdaq_file = nasdaq_file

    async def initialize(self):
        """Load and filter tickers from NYSE and NASDAQ files."""
        filtered_nyse = await load_and_filter_tickers(self.nyse_file, exchange_filter='N', use_cache=True)
        filtered_nasdaq = await load_and_filter_tickers(self.nasdaq_file, exchange_filter='Q', use_cache=True)
        self.nyse_tickers = set(ticker["ticker"] for ticker in filtered_nyse)
        self.nasdaq_tickers = set(ticker["ticker"] for ticker in filtered_nasdaq)
        logging.info(f"Initialized NYSE common stock tickers: {len(self.nyse_tickers)}")
        logging.info(f"Initialized NASDAQ common stock tickers: {len(self.nasdaq_tickers)}")

    def get_tickers(self, exchange: str) -> set:
        """Retrieve tickers for a specific exchange."""
        if exchange == "NYSE":
            return self.nyse_tickers
        elif exchange == "NASDAQ":
            return self.nasdaq_tickers
        return set()

async def fetch_ipo_dates_batch(tickers: List[str], api_key: str) -> Dict[str, Optional[str]]:
    """Fetch IPO dates for a batch of tickers using the FMP API with rate limiting and timeout."""
    await rate_limiter.acquire()  # Respect the rate limit
    url = f"https://financialmodelingprep.com/api/v3/profile/{','.join(tickers)}?apikey={api_key}"
    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            logging.debug(f"Sending request to {url}")
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logging.debug(f"Received data for batch: {data[:2]}")  # Log first two items
                    return {item["symbol"]: item.get("ipoDate") for item in data if "symbol" in item}
                else:
                    logging.warning(f"Failed to fetch batch for {tickers}: Status {response.status}")
                    return {ticker: None for ticker in tickers}
        except asyncio.TimeoutError:
            logging.error(f"Timeout while fetching batch for {tickers}")
            return {ticker: None for ticker in tickers}
        except Exception as e:
            logging.error(f"Error fetching batch for {tickers}: {str(e)}")
            return {ticker: None for ticker in tickers}

async def load_and_filter_tickers(file_path: str, exchange_filter: Optional[str] = None, update_rate_limit=None, use_cache: bool = True) -> List[Dict]:
    """Load and filter tickers from a file, ensuring exchange filtering and age > 10 years."""
    file_hash = get_file_hash(file_path)
    logging.debug(f"Computed hash for {file_path}: {file_hash}")

    conn = None
    try:
        conn = sqlite3.connect(CACHE_DB, timeout=30)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, ipo_date, timestamp FROM stocks WHERE ticker_list_hash = ?", (file_hash,))
        cached_data = cursor.fetchall()
        cached_ipo_dates = {row[0]: (row[1], row[2]) for row in cached_data}
        logging.debug(f"Loaded {len(cached_ipo_dates)} IPO date entries from stocks table for hash {file_hash}")

        ticker_data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split('|')
            
            if 'otherlisted' in file_path:
                ticker_column = 'ACT Symbol'
            elif 'nasdaqlisted' in file_path:
                ticker_column = 'Symbol'
            else:
                raise ValueError(f"Unknown file format: {file_path}")
            
            logging.debug(f"Using ticker column '{ticker_column}' for {file_path}")
            
            try:
                symbol_idx = headers.index(ticker_column)
            except ValueError:
                raise ValueError(f"'{ticker_column}' not found in {file_path}")
            
            security_type_idx = headers.index('Security Type') if 'Security Type' in headers else None
            exchange_idx = headers.index('Exchange') if 'Exchange' in headers else None

            for line in lines[1:]:
                parts = line.strip().split('|')
                if len(parts) <= symbol_idx:
                    continue
                ticker = parts[symbol_idx].strip()
                if not ticker:
                    continue

                if security_type_idx is not None and len(parts) > security_type_idx:
                    security_type = parts[security_type_idx].strip()
                    if security_type != "Common Stock":
                        continue

                if exchange_filter and exchange_idx is not None and len(parts) > exchange_idx:
                    exchange = parts[exchange_idx].strip()
                    if exchange != exchange_filter:
                        continue

                if any(suffix in ticker for suffix in ['.U', '.W', '.PR', '$']):
                    continue

                ipo_date, timestamp = cached_ipo_dates.get(ticker, (None, None))
                current_time = time.time()
                is_fresh = timestamp and (current_time - timestamp < CACHE_EXPIRY)

                if ipo_date and is_fresh:
                    try:
                        ipo_year = int(ipo_date.split("-")[0])
                        current_year = datetime.now().year
                        available_data_years = min(10, current_year - ipo_year + 1)
                        has_complete_history = available_data_years >= 10
                        if available_data_years < 10:
                            continue
                    except (ValueError, AttributeError):
                        logging.warning(f"Invalid IPO date format for {ticker}: {ipo_date}, assuming 10 years")
                        available_data_years = 10
                        has_complete_history = False
                else:
                    available_data_years = 10
                    has_complete_history = False

                ticker_data.append({
                    "ticker": ticker,
                    "ipo_date": ipo_date,
                    "available_data_years": available_data_years,
                    "has_complete_history": has_complete_history
                })

        logging.info(f"Loaded {len(ticker_data)} tickers from {file_path} after filtering")

        uncached_tickers = [t["ticker"] for t in ticker_data if t["ipo_date"] is None or not is_fresh]
        if uncached_tickers and use_cache:
            logging.info(f"Fetching IPO dates for {len(uncached_tickers)} uncached/stale tickers using FMP in batches")
            total_batches = (len(uncached_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, len(uncached_tickers), BATCH_SIZE):
                batch = uncached_tickers[i:i + BATCH_SIZE]
                logging.info(f"Starting batch {i // BATCH_SIZE + 1} of {total_batches} with {len(batch)} tickers")
                ipo_dates = await fetch_ipo_dates_batch(batch, FMP_API_KEYS[0])
                logging.info(f"Completed batch {i // BATCH_SIZE + 1} of {total_batches}: fetched IPO dates for {len([d for d in ipo_dates.values() if d is not None])} out of {len(batch)} tickers")
                current_time = time.time()
                for ticker, ipo_date in ipo_dates.items():
                    try:
                        cursor.execute(
                            "INSERT OR REPLACE INTO stocks (ticker, date, price, roe, rotc, eps, dividend, ticker_list_hash, balance_data, ipo_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0.0, "", "", "", "", file_hash, json.dumps([]), ipo_date, current_time)
                        )
                        conn.commit()
                    except sqlite3.OperationalError as e:
                        logging.error(f"Database error while saving IPO date for {ticker}: {str(e)}")
                        continue

                    for t in ticker_data:
                        if t["ticker"] == ticker:
                            t["ipo_date"] = ipo_date
                            if ipo_date:
                                try:
                                    ipo_year = int(ipo_date.split("-")[0])
                                    current_year = datetime.now().year
                                    t["available_data_years"] = min(10, current_year - ipo_year + 1)
                                    t["has_complete_history"] = t["available_data_years"] >= 10
                                    if t["available_data_years"] < 10:
                                        ticker_data.remove(t)
                                except (ValueError, AttributeError):
                                    logging.warning(f"Invalid IPO date format for {ticker}: {ipo_date}, assuming 10 years")
                                    t["available_data_years"] = 10
                                    t["has_complete_history"] = False
                            break

        return ticker_data

    except sqlite3.Error as e:
        logging.error(f"Database error while loading tickers from {file_path}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error loading tickers from {file_path}: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()
            logging.debug("Database connection closed")

async def fetch_ipo_date(ticker: str, update_rate_limit=None, cancel_event=None) -> Optional[str]:
    """Fetch IPO date from FMP with rate limiting and error handling."""
    max_retries = 3
    backoff_base = 2
    keys = FMP_API_KEYS

    for api_key in keys:
        for attempt in range(max_retries):
            if cancel_event and cancel_event.is_set():
                logging.info(f"Cancelling IPO date fetch for {ticker} with key {api_key}")
                return None

            logging.debug(f"Attempt {attempt + 1}/{max_retries} for {ticker} with API key {api_key}")
            try:
                async with FMP_LIMITER:
                    await rate_limiter.acquire()
                    async with aiohttp.ClientSession() as session:
                        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
                        async with session.get(url) as response:
                            status = response.status
                            if status == 200:
                                data = await response.json()
                                if isinstance(data, list) and data and 'ipoDate' in data[0]:
                                    ipo_date = data[0].get("ipoDate")
                                    if ipo_date:
                                        logging.debug(f"Fetched IPO date for {ticker}: {ipo_date}")
                                        return ipo_date
                            elif status == 429:
                                if update_rate_limit:
                                    update_rate_limit(f"HTTP 429 for {api_key}, pausing (Attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(backoff_base ** attempt)
                            else:
                                logging.warning(f"Failed to fetch profile for {ticker} with key {api_key}: Status {status}")
            except Exception as e:
                logging.error(f"Error fetching IPO date for {ticker} with key {api_key}: {str(e)}")
                await asyncio.sleep(backoff_base ** attempt)
    logging.warning(f"Could not fetch IPO date for {ticker} after {max_retries} attempts with all keys")
    return None

async def fetch_with_multiple_keys_async(ticker, endpoint, api_keys, retries=3, backoff=2, update_rate_limit=None, session=None, cancel_event=None):
    """Fetch data from FMP with multiple API keys and error handling."""
    if not api_keys or all(not key for key in api_keys):
        logging.error(f"No valid API keys for endpoint {endpoint}")
        return None

    for attempt in range(retries):
        for api_key in api_keys:
            if cancel_event and cancel_event.is_set():
                logging.info(f"Cancelling fetch for {ticker} ({endpoint})")
                return None

            logging.debug(f"Attempt {attempt + 1}/{retries} for {ticker} ({endpoint}) with key {api_key}")
            try:
                async with FMP_LIMITER:
                    await rate_limiter.acquire()
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                    if session:
                        async with session.get(url) as response:
                            status = response.status
                            if status != 200:
                                if status == 429 and update_rate_limit:
                                    update_rate_limit(f"Rate limit hit for {api_key}, pausing")
                                    await asyncio.sleep(60)
                                continue
                            data = await response.json()
                            return data
                    else:
                        async with aiohttp.ClientSession() as temp_session:
                            async with temp_session.get(url) as response:
                                status = response.status
                                if status != 200:
                                    if status == 429 and update_rate_limit:
                                        update_rate_limit(f"Rate limit hit for {api_key}, pausing")
                                    await asyncio.sleep(60)
                                continue
                            data = await response.json()
                            return data
            except Exception as e:
                logging.error(f"Error fetching {endpoint} for {ticker} with key {api_key}: {str(e)}")
                await asyncio.sleep(backoff ** attempt)
    logging.error(f"All attempts to fetch {endpoint} for {ticker} failed")
    return None

async def fetch_fmp_data(ticker: str, keys: List[str], update_rate_limit=None, cancel_event=None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[Dict]]:
    """Fetch financial data from FMP with caching."""
    with FMP_CACHE_LOCK:
        if ticker in FMP_DATA_CACHE:
            logging.debug(f"Using cached FMP data for {ticker}")
            return FMP_DATA_CACHE[ticker]

    primary_key = keys[0]
    for attempt in range(2):
        if cancel_event and cancel_event.is_set():
            logging.info(f"Cancelling FMP data fetch for {ticker}")
            return None, None, None

        income_data, balance_data_fmp, _ = await _fetch_fmp_with_key(ticker, primary_key, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
        if income_data is not None:
            result = (income_data, balance_data_fmp, None)
            with FMP_CACHE_LOCK:
                FMP_DATA_CACHE[ticker] = result
            div_data = await _fetch_dividend_data(ticker, primary_key, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
            return (income_data, balance_data_fmp, div_data)
        else:
            logging.warning(f"Attempt {attempt + 1} failed for {ticker}. Retrying after pause.")
            if update_rate_limit:
                update_rate_limit(f"Pausing due to failure, retrying in 60s (Attempt {attempt + 1}/2)")
            await asyncio.sleep(60)

    logging.error(f"FMP fetch failed for {ticker} after 2 attempts")
    return None, None, None

async def _fetch_fmp_with_key(ticker: str, api_key: str, retries: int = 3, update_rate_limit=None, cancel_event=None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[Dict]]:
    """Helper function to fetch financial data with a single API key."""
    for attempt in range(retries):
        if cancel_event and cancel_event.is_set():
            return None, None, None

        try:
            async with FMP_LIMITER:
                await rate_limiter.acquire()
                async with aiohttp.ClientSession() as session:
                    url = f"https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{ticker}?period=annual&limit=15&apikey={api_key}"
                    async with session.get(url) as response:
                        if response.status == 429:
                            if update_rate_limit:
                                update_rate_limit(f"Rate limit hit, pausing for 60s")
                            await asyncio.sleep(60)
                            continue
                        financial_data = await response.json()
                        if not isinstance(financial_data, list):
                            logging.info(f"No financial data for {ticker}")
                            return None, None, None
                        years_fmp = [int(entry['documentfiscalyearfocus']) for entry in financial_data if 'documentfiscalyearfocus' in entry]
                        if years_fmp:
                            logging.info(f"Fetched {len(years_fmp)} years of data for {ticker}")
                            return financial_data, financial_data, None
                        return None, None, None
        except Exception as e:
            logging.error(f"Error fetching FMP data for {ticker}: {str(e)}")
            return None, None, None
    return None, None, None

async def _fetch_dividend_data(ticker: str, api_key: str, retries: int = 3, update_rate_limit=None, cancel_event=None) -> Optional[Dict]:
    """Fetch dividend data from FMP."""
    for attempt in range(retries):
        if cancel_event and cancel_event.is_set():
            return None

        try:
            async with FMP_LIMITER:
                await rate_limiter.acquire()
                async with aiohttp.ClientSession() as session:
                    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?apikey={api_key}"
                    async with session.get(url) as response:
                        if response.status == 429:
                            if update_rate_limit:
                                update_rate_limit(f"Rate limit hit, pausing for 60s")
                            await asyncio.sleep(60)
                            continue
                        div_data = await response.json()
                        if 'historical' in div_data and div_data['historical']:
                            logging.info(f"Fetched dividend data for {ticker}")
                            return div_data
                        return None
        except Exception as e:
            logging.error(f"Error fetching dividend data for {ticker}: {str(e)}")
            return None
    return None

async def fetch_historical_data(ticker: str, exchange="Stock", update_rate_limit=None, cancel_event=None) -> Tuple[List[float], List[float], List[float], List[float], List[int], Dict[str, float], List[Dict], Optional[str]]:
    """Fetch 10-year historical financial data for a ticker."""
    with FETCHED_TICKERS_LOCK:
        if ticker in FETCHED_TICKERS:
            logging.warning(f"Skipping duplicate fetch for {ticker} ({exchange})")
            return [0.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10, list(range(datetime.now().year - 9, datetime.now().year + 1)), {}, [], None
        FETCHED_TICKERS.add(ticker)

    target_years = list(range(datetime.now().year - 9, datetime.now().year + 1))
    roe_10y = [0.0] * 10
    rotc_10y = [0.0] * 10
    eps_10y = [0.0] * 10
    div_10y = [0.0] * 10
    years = target_years
    revenue = {}
    balance_data = []
    ipo_date = None

    if cancel_event and cancel_event.is_set():
        return roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date

    ipo_date = await fetch_ipo_date(ticker, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    if cancel_event and cancel_event.is_set():
        return roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date

    available_years = 10
    if ipo_date:
        try:
            ipo_year = int(ipo_date.split("-")[0])
            current_year = datetime.now().year
            available_years = min(10, current_year - ipo_year + 1)
            if available_years < 10:
                logging.info(f"{ticker} has only {available_years} years of data due to IPO {ipo_date}")
        except ValueError:
            logging.warning(f"Invalid IPO date for {ticker}: {ipo_date}, assuming 10 years")
            available_years = 10

    income_data, balance_data_fmp, div_data = await fetch_fmp_data(ticker, FMP_API_KEYS, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    if cancel_event and cancel_event.is_set():
        return roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date

    if income_data is None or balance_data_fmp is None:
        logging.warning(f"Failed to fetch FMP data for {ticker}")
        return roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date

    years_fmp = sorted([int(entry['documentfiscalyearfocus']) for entry in income_data if 'documentfiscalyearfocus' in entry], reverse=True)
    if not years_fmp:
        logging.warning(f"No historical data for {ticker}")
        return roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date

    years_to_process = min(available_years, len(years_fmp))
    years_fmp = years_fmp[:years_to_process]

    for i in range(years_to_process):
        year_str = str(years_fmp[i])
        year_entry = next((entry for entry in income_data if str(entry.get('documentfiscalyearfocus', '')) == year_str), {})
        balance_entry = next((entry for entry in balance_data_fmp if str(entry.get('documentfiscalyearfocus', '')) == year_str), {})

        net_income = float(year_entry.get('netincomeloss', 0.0))
        equity = float(balance_entry.get('stockholdersequity', 1.0))
        roe_10y[i] = (net_income / equity * 100) if equity != 0 else 0.0

        operating_cash = float(year_entry.get('netcashprovidedbyusedinoperatingactivities', 0.0))
        total_assets = float(balance_entry.get('assets', 1.0))
        rotc_10y[i] = (operating_cash / total_assets * 100) if total_assets != 0 else 0.0

        eps_10y[i] = float(year_entry.get('earningspersharebasic', 0.0))
        revenue[year_str] = float(year_entry.get('revenuefromcontractwithcustomerexcludingassessedtax', year_entry.get('salesrevenuenet', 0.0)))

    div_df = pd.DataFrame(div_data.get('historical', [])) if div_data else pd.DataFrame()
    if not div_df.empty:
        div_df['year'] = pd.to_datetime(div_df['date']).dt.year
        div_per_year = div_df.groupby('year')['dividend'].max().to_dict()
        for i in range(years_to_process):
            year = years_fmp[i]
            div_10y[i] = float(div_per_year.get(year, 0.0))
    else:
        for i in range(years_to_process):
            year = years_fmp[i]
            div_10y[i] = float(next((entry.get('commonstockdividendspersharedeclared', 0.0) for entry in income_data if str(entry.get('documentfiscalyearfocus', '')) == str(year)), 0.0))

    for entry in balance_data_fmp:
        if 'liabilitiescurrent' in entry and 'totalCurrentLiabilities' not in entry:
            entry['totalCurrentLiabilities'] = entry['liabilitiescurrent']
        if 'assetscurrent' in entry and 'totalCurrentAssets' not in entry:
            entry['totalCurrentAssets'] = entry['assetscurrent']

    balance_data = balance_data_fmp if balance_data_fmp else []
    return roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date

def calculate_cagr(start_value, end_value, years):
    """Calculate Compound Annual Growth Rate (CAGR)."""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1

def calculate_common_criteria(ticker: str, eps_10y: List[float], div_10y: List[float], revenue: Dict[str, float], balance_data: List[Dict], debt_to_equity: Optional[float], available_data_years: int, latest_revenue: float) -> int:
    """Calculate Graham score based on 6 common criteria, adjusted for available data."""
    score = 0
    logging.debug(f"Calculating common criteria for {ticker} (available_data_years={available_data_years})")

    # Criterion 1: Adequate Size (Revenue >= $500 million)
    if latest_revenue >= 500_000_000:
        score += 1
        logging.debug(f"{ticker} Criterion 1: Revenue >= $500M - Passed ({latest_revenue})")
    else:
        logging.debug(f"{ticker} Criterion 1: Revenue < $500M - Failed ({latest_revenue})")

    # Criterion 2: Strong Financial Condition (Current Ratio > 2)
    if balance_data and len(balance_data) > 0:
        latest_balance = balance_data[0]
        current_assets = float(latest_balance.get('totalCurrentAssets', 0))
        current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 0))
        if current_liabilities == 0:
            logging.warning(f"{ticker} Criterion 2: Zero current liabilities detected")
        else:
            current_ratio = current_assets / current_liabilities
            if current_ratio > 2:
                score += 1
                logging.debug(f"{ticker} Criterion 2: Current Ratio = {current_ratio:.2f} (> 2) - Passed")
            else:
                logging.debug(f"{ticker} Criterion 2: Current Ratio = {current_ratio:.2f} (<= 2) - Failed")
    else:
        logging.debug(f"{ticker} Criterion 2: No balance data - Failed")

    # Criterion 3: Earnings Stability (No more than 2 negative EPS years in last 10 years, scaled)
    if available_data_years >= 1:
        max_negative_years = min(2, available_data_years // 5)  # Scale: 0 for 1-4 years, 1 for 5-9 years, 2 for 10 years
        negative_eps_years = sum(1 for eps in eps_10y[-available_data_years:] if eps <= 0)
        if negative_eps_years <= max_negative_years:
            score += 1
            logging.debug(f"{ticker} Criterion 3: Earnings Stability - Passed ({negative_eps_years} negative years)")
        else:
            logging.debug(f"{ticker} Criterion 3: Earnings Stability - Failed ({negative_eps_years} negative years)")
    else:
        logging.debug(f"{ticker} Criterion 3: Insufficient data - Failed")

    # Criterion 4: Uninterrupted Dividend Payments (10 years or available years)
    if available_data_years >= 1:
        required_div_years = min(10, available_data_years)
        uninterrupted_div_years = sum(1 for div in div_10y[-required_div_years:] if div > 0)
        if uninterrupted_div_years == required_div_years:
            score += 1
            logging.debug(f"{ticker} Criterion 4: Uninterrupted Dividends - Passed ({uninterrupted_div_years}/{required_div_years} years)")
        else:
            logging.debug(f"{ticker} Criterion 4: Uninterrupted Dividends - Failed ({uninterrupted_div_years}/{required_div_years} years)")
    else:
        logging.debug(f"{ticker} Criterion 4: Insufficient data - Failed")

    # Criterion 5: Earnings Growth (EPS CAGR > 3%)
    if available_data_years >= 2:
        first_eps = eps_10y[-available_data_years]
        last_eps = eps_10y[-1]
        if first_eps > 0 and last_eps > 0:
            cagr = calculate_cagr(first_eps, last_eps, available_data_years - 1)
            if cagr > 0.03:
                score += 1
                logging.debug(f"{ticker} Criterion 5: EPS CAGR = {cagr:.2%} (> 3%) - Passed")
            else:
                logging.debug(f"{ticker} Criterion 5: EPS CAGR = {cagr:.2%} (<= 3%) - Failed")
        else:
            logging.debug(f"{ticker} Criterion 5: First or last EPS <= 0 - Failed")
    else:
        logging.debug(f"{ticker} Criterion 5: Insufficient data - Failed")

    # Criterion 6: Debt-to-Equity Ratio < 2
    if debt_to_equity is not None and isinstance(debt_to_equity, (int, float)):
        if debt_to_equity < 2:
            score += 1
            logging.debug(f"{ticker} Criterion 6: Debt-to-Equity = {debt_to_equity:.2f} (< 2) - Passed")
        else:
            logging.debug(f"{ticker} Criterion 6: Debt-to-Equity = {debt_to_equity:.2f} (>= 2) - Failed")
    else:
        logging.debug(f"{ticker} Criterion 6: Debt-to-Equity unavailable - Failed")

    logging.debug(f"Final Graham score for {ticker}: {score}/6")
    return score

def calculate_graham_score_8(ticker: str, price: float, pe_ratio: Optional[float], pb_ratio: Optional[float], debt_to_equity: Optional[float], eps_10y: List[float], div_10y: List[float], revenue: Dict[str, float], balance_data: List[Dict], available_data_years: int, latest_revenue: float) -> int:
    """Calculate Graham score based on 8 criteria, adjusted for available data."""
    score = 0
    logging.debug(f"Calculating Graham score (8 criteria) for {ticker} (price={price}, available_data_years={available_data_years})")

    # First 6 criteria same as calculate_common_criteria
    common_score = calculate_common_criteria(ticker, eps_10y, div_10y, revenue, balance_data, debt_to_equity, available_data_years, latest_revenue)
    score += common_score

    # Criterion 7: P/E Ratio <= 15
    if pe_ratio is not None and isinstance(pe_ratio, (int, float)):
        if pe_ratio <= 15:
            score += 1
            logging.debug(f"{ticker} Criterion 7: P/E = {pe_ratio:.2f} (<= 15) - Passed")
        else:
            logging.debug(f"{ticker} Criterion 7: P/E = {pe_ratio:.2f} (> 15) - Failed")
    else:
        logging.debug(f"{ticker} Criterion 7: P/E unavailable - Failed")

    # Criterion 8: P/B Ratio <= 1.5
    if pb_ratio is not None and isinstance(pb_ratio, (int, float)):
        if pb_ratio <= 1.5:
            score += 1
            logging.debug(f"{ticker} Criterion 8: P/B = {pb_ratio:.2f} (<= 1.5) - Passed")
        else:
            logging.debug(f"{ticker} Criterion 8: P/B = {pb_ratio:.2f} (> 1.5) - Failed")
    else:
        logging.debug(f"{ticker} Criterion 8: P/B unavailable - Failed")

    logging.debug(f"Final Graham score (8 criteria) for {ticker}: {score}/8")
    return score

def calculate_graham_value(earnings: Optional[float], expected_return: float, aaa_yield: float = 4.5, eps_10y: List[float] = None) -> float:
    """Calculate intrinsic value using Graham's formula."""
    if earnings is None or earnings <= 0:
        logging.debug("Cannot calculate Graham value: Earnings are None or <= 0")
        return "NQ"

    growth_factor = 1.0
    if eps_10y and len(eps_10y) >= 2:
        first_eps = eps_10y[0]
        last_eps = eps_10y[-1]
        if first_eps > 0 and last_eps > 0:
            years = len(eps_10y) - 1
            if years > 0:
                growth_rate = (last_eps / first_eps) ** (1 / years) - 1
                growth_factor = 1 + growth_rate
                logging.debug(f"Growth factor: {growth_factor} (rate={growth_rate:.2%})")
        else:
            logging.debug("First or last EPS <= 0, using growth_factor=1.0")

    value = (earnings * (8.5 + 2 * growth_factor) * aaa_yield) / (expected_return + 0.05)
    logging.debug(f"Graham value: {value:.2f} (earnings={earnings}, growth_factor={growth_factor})")
    return value

async def fetch_batch_data(tickers, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None):
    """Fetch data for a batch of tickers with fresh prices."""
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()

    results = []
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TICKERS)

    nyse_tickers = ticker_manager.get_tickers("NYSE")
    nasdaq_tickers = ticker_manager.get_tickers("NASDAQ")

    async def fetch_data(ticker):
        async with semaphore:
            async with YAHOO_LIMITER:
                try:
                    if cancel_event and cancel_event.is_set():
                        return {"ticker": ticker, "exchange": exchange, "error": "Cancelled by user"}

                    ticker_exchange = "Unknown"
                    if ticker in nyse_tickers and ticker in nasdaq_tickers:
                        ticker_exchange = "Dual-Listed (NYSE/NASDAQ)"
                    elif ticker in nyse_tickers:
                        ticker_exchange = "NYSE"
                    elif ticker in nasdaq_tickers:
                        ticker_exchange = "NASDAQ"

                    await asyncio.sleep(DELAY_BETWEEN_CALLS)
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    price = info.get('regularMarketPrice', info.get('previousClose', None))
                    if price is None:
                        logging.error(f"No price data for {ticker} from yfinance")
                        return {"ticker": ticker, "exchange": ticker_exchange, "error": "No price data"}

                    pe_ratio = info.get('trailingPE', None)
                    pb_ratio = info.get('priceToBook', None)
                    debt_to_equity = info.get('debtToEquity', None)
                    if debt_to_equity:
                        debt_to_equity = float(debt_to_equity) / 100 if isinstance(debt_to_equity, (int, float)) else debt_to_equity

                    if pe_ratio is None or pb_ratio is None or debt_to_equity is None:
                        async with aiohttp.ClientSession() as session:
                            fmp_data = await fetch_with_multiple_keys_async(ticker, "quote", FMP_API_KEYS, update_rate_limit=update_rate_limit, session=session, cancel_event=cancel_event)
                            if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
                                pe_ratio = pe_ratio or fmp_data[0].get('pe', None)
                                pb_ratio = pb_ratio or fmp_data[0].get('pb', None)
                                debt_to_equity = debt_to_equity or fmp_data[0].get('debtToEquity', None)

                    roe_10y, rotc_10y, eps_10y, div_10y, years, revenue, balance_data, ipo_date = await fetch_historical_data(ticker, ticker_exchange, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
                    if cancel_event and cancel_event.is_set():
                        return {"ticker": ticker, "exchange": ticker_exchange, "error": "Cancelled by user"}

                    available_data_years = 10
                    has_complete_history = True
                    if ipo_date:
                        ipo_year = int(ipo_date.split("-")[0])
                        current_year = datetime.now().year
                        available_data_years = min(10, current_year - ipo_year + 1)
                        has_complete_history = available_data_years >= 10

                    latest_revenue = max(revenue.values(), default=0)

                    earnings = eps_10y[-1] if eps_10y and eps_10y[-1] > 0 else None
                    intrinsic_value = calculate_graham_value(earnings, expected_return, aaa_yield=4.5, eps_10y=eps_10y) if earnings else "NQ"
                    buy_price = intrinsic_value * (1 - margin_of_safety) if isinstance(intrinsic_value, (int, float)) else "NQ"
                    sell_price = intrinsic_value * (1 + expected_return) if isinstance(intrinsic_value, (int, float)) else "NQ"
                    graham_score = calculate_graham_score_8(ticker, price, pe_ratio, pb_ratio, debt_to_equity, eps_10y, div_10y, revenue, balance_data, available_data_years, latest_revenue)

                    return {
                        "ticker": ticker,
                        "exchange": ticker_exchange,
                        "price": price,
                        "pe_ratio": pe_ratio,
                        "pb_ratio": pb_ratio,
                        "debt_to_equity": debt_to_equity,
                        "intrinsic_value": intrinsic_value,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "graham_score": graham_score,
                        "years": years,
                        "roe_10y": roe_10y,
                        "rotc_10y": rotc_10y,
                        "eps_10y": eps_10y,
                        "div_10y": div_10y,
                        "balance_data": balance_data,
                        "ipo_date": ipo_date,
                        "available_data_years": available_data_years,
                        "has_complete_history": has_complete_history,
                        "latest_revenue": latest_revenue
                    }
                except Exception as e:
                    logging.error(f"Error processing {ticker} ({exchange}): {str(e)}")
                    return {"ticker": ticker, "exchange": ticker_exchange, "error": str(e)}

    for ticker in tickers:
        tasks.append(fetch_data(ticker))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

async def fetch_stock_data(ticker, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None):
    """Fetch data for a single ticker."""
    results = await fetch_batch_data(
        [ticker],
        expected_return=expected_return,
        margin_of_safety=margin_of_safety,
        exchange=exchange,
        ticker_manager=ticker_manager,
        update_rate_limit=update_rate_limit,
        cancel_event=cancel_event
    )
    return results[0] if results else {"ticker": ticker, "exchange": exchange, "error": "Failed to fetch data"}

async def save_qualifying_stocks_to_favorites(qualifying_stocks, exchange):
    """Save qualifying stocks to favorites file."""
    try:
        list_name = f"{exchange}_Graham_Qualifiers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        favorites = {}
        if os.path.exists(FAVORITES_FILE):
            with open(FAVORITES_FILE, 'r') as f:
                favorites = json.load(f)

        favorites[list_name] = qualifying_stocks
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=4)
        logging.info(f"Saved qualifying stocks to favorites: {list_name}")
        return list_name
    except Exception as e:
        logging.error(f"Error saving qualifying stocks: {str(e)}")
        return None

def load_favorites():
    """Load favorites from file."""
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading favorites: {str(e)}")
    return {}

def get_stock_data_from_db(ticker, cursor):
    """Retrieve stock data from database."""
    try:
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if row:
            return {
                "ticker": row[0],
                "date": row[1],
                "price": row[2],
                "roe": [float(x) for x in row[3].split(",")] if row[3] else [],
                "rotc": [float(x) for x in row[4].split(",")] if row[4] else [],
                "eps": [float(x) for x in row[5].split(",")] if row[5] else [],
                "dividend": [float(x) for x in row[6].split(",")] if row[6] else [],
                "ticker_list_hash": row[7],
                "balance_data": json.loads(row[8]) if row[8] else [],
                "ipo_date": row[9],
                "timestamp": row[10]
            }
        return None
    except sqlite3.Error as e:
        logging.error(f"Database error fetching data for {ticker}: {str(e)}")
        return None

async def screen_nyse_graham_stocks(batch_size=25, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None):
    """Screen NYSE stocks using Graham criteria."""
    exchange = "NYSE"
    logging.info(f"Starting NYSE Graham screening")
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()
    file_path = NYSE_LIST_FILE
    current_file_hash = get_file_hash(file_path)

    conn = sqlite3.connect(CACHE_DB, timeout=30)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT file_hash FROM screening_progress WHERE exchange=?", (exchange,))
        stored_hash_row = cursor.fetchone()
        stored_hash = stored_hash_row[0] if stored_hash_row else None

        if stored_hash != current_file_hash:
            logging.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE ticker_list_hash != ?", (current_file_hash,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()
            use_cache = False
        else:
            use_cache = True

        cursor.execute("SELECT ticker, status FROM screening_progress WHERE exchange=?", (exchange,))
        progress = cursor.fetchall()
        completed_tickers = set(row[0] for row in progress if row[1] == "completed")
        if use_cache and len(completed_tickers) == len(ticker_manager.get_tickers(exchange)):
            cursor.execute("SELECT ticker, graham_score FROM graham_qualifiers WHERE exchange=?", (exchange,))
            qualifiers = cursor.fetchall()
            qualifying_stocks = [t for t, _ in qualifiers]
            graham_scores = [s for _, s in qualifiers]
            exchanges = [exchange] * len(qualifiers)
            if root and update_progress_animated:
                root.after(0, update_progress_animated, 100, list(ticker_manager.get_tickers(exchange)), len(qualifying_stocks))
            return qualifying_stocks, graham_scores, exchanges

        ticker_list = list(ticker_manager.get_tickers("NYSE"))
        filtered_ticker_data = [{"ticker": ticker, "ipo_date": None, "available_data_years": 10, "has_complete_history": False} for ticker in ticker_list]
        tickers = filtered_ticker_data if tickers is None else tickers
        ticker_list = [t["ticker"] for t in tickers]

        with FMP_CACHE_LOCK:
            FMP_DATA_CACHE.clear()
        with YAHOO_CACHE_LOCK:
            YAHOO_DATA_CACHE.clear()

        remaining_ticker_data = [t for t in tickers if t["ticker"] not in completed_tickers]
        if not remaining_ticker_data:
            cursor.execute("SELECT ticker, graham_score FROM graham_qualifiers WHERE exchange=?", (exchange,))
            qualifiers = cursor.fetchall()
            qualifying_stocks = [t for t, _ in qualifiers]
            graham_scores = [s for _, s in qualifiers]
            exchanges = [exchange] * len(qualifiers)
            if root and update_progress_animated:
                root.after(0, update_progress_animated, 100, ticker_list, len(qualifying_stocks))
            return qualifying_stocks, graham_scores, exchanges

        qualifying_stocks, graham_scores, exchanges = [], [], []
        total_tickers = len(remaining_ticker_data)
        processed_tickers = 0
        passed_tickers = 0

        dynamic_batch_size = min(batch_size, max(10, MAX_TICKERS_PER_MINUTE // 2))
        for i in range(0, len(remaining_ticker_data), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                break
            batch_data = remaining_ticker_data[i:i + dynamic_batch_size]
            batch = [t["ticker"] for t in batch_data]
            logging.info(f"Processing NYSE batch {i // dynamic_batch_size + 1} with tickers: {batch}")
            results = await fetch_batch_data(batch, exchange=exchange, update_rate_limit=update_rate_limit, ticker_manager=ticker_manager, cancel_event=cancel_event)
            for result, ticker_info in zip(results, batch_data):
                if cancel_event and cancel_event.is_set():
                    break
                if isinstance(result, dict):
                    ticker = result['ticker']
                    if 'error' in result:
                        cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                       (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "failed"))
                        conn.commit()
                        continue
                    score = result['graham_score']
                    cursor.execute(
                        "INSERT OR REPLACE INTO stocks (ticker, date, price, roe, rotc, eps, dividend, ticker_list_hash, balance_data, ipo_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['price'],
                         ",".join(map(str, result['roe_10y'])), ",".join(map(str, result['rotc_10y'])),
                         ",".join(map(str, result['eps_10y'])), ",".join(map(str, result['div_10y'])),
                         current_file_hash, json.dumps(result['balance_data']), ticker_info['ipo_date'], time.time())
                    )
                    conn.commit()
                    if score >= 5:
                        qualifying_stocks.append(ticker)
                        graham_scores.append(score)
                        exchanges.append(result['exchange'])
                        passed_tickers += 1
                        cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, graham_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                       (ticker, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown', result['exchange']))
                        conn.commit()
                    processed_tickers += 1
                    progress = (processed_tickers / total_tickers) * 100
                    if root and update_progress_animated:
                        root.after(0, update_progress_animated, progress, ticker_list, passed_tickers)
                    cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                   (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed"))
                    conn.commit()

        if not cancel_event or not cancel_event.is_set():
            if root and update_progress_animated:
                root.after(0, update_progress_animated, 100, ticker_list, passed_tickers)
            if qualifying_stocks:
                list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, exchange)
                if root and refresh_favorites_dropdown:
                    root.after(0, refresh_favorites_dropdown, list_name)

        return qualifying_stocks, graham_scores, exchanges

    except Exception as e:
        logging.error(f"Screening error: {str(e)}")
        raise
    finally:
        conn.close()

async def screen_nasdaq_graham_stocks(batch_size=25, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None):
    """Screen NASDAQ stocks using Graham criteria."""
    exchange = "NASDAQ"
    logging.info(f"Starting NASDAQ Graham screening")
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()
    file_path = NASDAQ_LIST_FILE
    current_file_hash = get_file_hash(file_path)

    conn = sqlite3.connect(CACHE_DB, timeout=30)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT file_hash FROM screening_progress WHERE exchange=?", (exchange,))
        stored_hash_row = cursor.fetchone()
        stored_hash = stored_hash_row[0] if stored_hash_row else None

        if stored_hash != current_file_hash:
            logging.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE ticker_list_hash != ?", (current_file_hash,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()
            use_cache = False
        else:
            use_cache = True

        cursor.execute("SELECT ticker, status FROM screening_progress WHERE exchange=?", (exchange,))
        progress = cursor.fetchall()
        completed_tickers = set(row[0] for row in progress if row[1] == "completed")
        if use_cache and len(completed_tickers) == len(ticker_manager.get_tickers(exchange)):
            cursor.execute("SELECT ticker, graham_score FROM graham_qualifiers WHERE exchange=?", (exchange,))
            qualifiers = cursor.fetchall()
            qualifying_stocks = [t for t, _ in qualifiers]
            graham_scores = [s for _, s in qualifiers]
            exchanges = [exchange] * len(qualifiers)
            if root and update_progress_animated:
                root.after(0, update_progress_animated, 100, list(ticker_manager.get_tickers(exchange)), len(qualifying_stocks))
            return qualifying_stocks, graham_scores, exchanges

        ticker_list = list(ticker_manager.get_tickers("NASDAQ"))
        filtered_ticker_data = [{"ticker": ticker, "ipo_date": None, "available_data_years": 10, "has_complete_history": False} for ticker in ticker_list]
        tickers = filtered_ticker_data if tickers is None else tickers
        ticker_list = [t["ticker"] for t in tickers]

        with FMP_CACHE_LOCK:
            FMP_DATA_CACHE.clear()
        with YAHOO_CACHE_LOCK:
            YAHOO_DATA_CACHE.clear()

        remaining_ticker_data = [t for t in tickers if t["ticker"] not in completed_tickers]
        if not remaining_ticker_data:
            cursor.execute("SELECT ticker, graham_score FROM graham_qualifiers WHERE exchange=?", (exchange,))
            qualifiers = cursor.fetchall()
            qualifying_stocks = [t for t, _ in qualifiers]
            graham_scores = [s for _, s in qualifiers]
            exchanges = [exchange] * len(qualifiers)
            if root and update_progress_animated:
                root.after(0, update_progress_animated, 100, ticker_list, len(qualifying_stocks))
            return qualifying_stocks, graham_scores, exchanges

        qualifying_stocks, graham_scores, exchanges = [], [], []
        total_tickers = len(remaining_ticker_data)
        processed_tickers = 0
        passed_tickers = 0

        dynamic_batch_size = min(batch_size, max(10, MAX_TICKERS_PER_MINUTE // 2))
        for i in range(0, len(remaining_ticker_data), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                break
            batch_data = remaining_ticker_data[i:i + dynamic_batch_size]
            batch = [t["ticker"] for t in batch_data]
            logging.info(f"Processing NASDAQ batch {i // dynamic_batch_size + 1} with tickers: {batch}")
            results = await fetch_batch_data(batch, exchange=exchange, update_rate_limit=update_rate_limit, ticker_manager=ticker_manager, cancel_event=cancel_event)
            for result, ticker_info in zip(results, batch_data):
                if cancel_event and cancel_event.is_set():
                    break
                if isinstance(result, dict):
                    ticker = result['ticker']
                    if 'error' in result:
                        cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                       (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "failed"))
                        conn.commit()
                        continue
                    score = result['graham_score']
                    cursor.execute(
                        "INSERT OR REPLACE INTO stocks (ticker, date, price, roe, rotc, eps, dividend, ticker_list_hash, balance_data, ipo_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['price'],
                         ",".join(map(str, result['roe_10y'])), ",".join(map(str, result['rotc_10y'])),
                         ",".join(map(str, result['eps_10y'])), ",".join(map(str, result['div_10y'])),
                         current_file_hash, json.dumps(result['balance_data']), ticker_info['ipo_date'], time.time())
                    )
                    conn.commit()
                    if score >= 5:
                        qualifying_stocks.append(ticker)
                        graham_scores.append(score)
                        exchanges.append(result['exchange'])
                        passed_tickers += 1
                        cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, graham_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                       (ticker, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown', result['exchange']))
                        conn.commit()
                    processed_tickers += 1
                    progress = (processed_tickers / total_tickers) * 100
                    if root and update_progress_animated:
                        root.after(0, update_progress_animated, progress, ticker_list, passed_tickers)
                    cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                   (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed"))
                    conn.commit()

        if not cancel_event or not cancel_event.is_set():
            if root and update_progress_animated:
                root.after(0, update_progress_animated, 100, ticker_list, passed_tickers)
            if qualifying_stocks:
                list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, exchange)
                if root and refresh_favorites_dropdown:
                    root.after(0, refresh_favorites_dropdown, list_name)

        return qualifying_stocks, graham_scores, exchanges

    except Exception as e:
        logging.error(f"Screening error: {str(e)}")
        raise
    finally:
        conn.close()

def clear_in_memory_caches():
    """Clear the in-memory caches FMP_DATA_CACHE and YAHOO_DATA_CACHE."""
    with FMP_CACHE_LOCK:
        FMP_DATA_CACHE.clear()
    with YAHOO_CACHE_LOCK:
        YAHOO_DATA_CACHE.clear()
    logging.info("Cleared in-memory FMP_DATA_CACHE and YAHOO_DATA_CACHE")