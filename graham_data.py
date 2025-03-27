import sqlite3
import time
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import threading
import hashlib
import concurrent.futures
import requests
from config import (FMP_API_KEYS, FRED_API_KEY, paid_rate_limiter, free_rate_limiter, CACHE_DB, NYSE_LIST_FILE, NASDAQ_LIST_FILE,
                    DATA_DIR, FAVORITES_LOCK, FileHashError, FAVORITES_FILE, CACHE_EXPIRY, MAX_CALLS_PER_MINUTE_PAID)

# Constants for batch processing and concurrency control
MAX_CONCURRENT_TICKERS = 10
MAX_TICKERS_PER_MINUTE = 250
DELAY_BETWEEN_CALLS = 0.1
BATCH_SIZE = 50

# Thread-safe set for fetched tickers
FETCHED_TICKERS = set()
FETCHED_TICKERS_LOCK = threading.Lock()

# Single-threaded executor for SQLite writes
db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Logging setup
logging.basicConfig(filename=os.path.join(DATA_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

# Cache variables for AAA yield
aaa_yield_cache = None
cache_timestamp = None
CACHE_DURATION = timedelta(days=1)

def get_aaa_yield(api_key, default_yield=0.045):
    """Fetch Moody's Seasoned AAA Corporate Bond Yield from FRED with caching."""
    global aaa_yield_cache, cache_timestamp
    current_time = datetime.now()

    if aaa_yield_cache is not None and cache_timestamp is not None:
        if current_time - cache_timestamp < CACHE_DURATION:
            return aaa_yield_cache

    if not api_key:
        logging.error("FRED_API_KEY not set. Using default yield: 4.5%")
        return default_yield

    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=AAA&api_key={api_key}&file_type=json&limit=1&sort_order=desc"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'observations' in data and len(data['observations']) > 0:
            yield_value = float(data['observations'][0]['value']) / 100
            aaa_yield_cache = yield_value
            cache_timestamp = current_time
            logging.info(f"Fetched AAA yield: {yield_value:.4f}")
            return yield_value
        else:
            raise ValueError("No observations found in FRED response")
    except requests.RequestException as e:
        logging.error(f"Error fetching AAA yield from FRED: {str(e)}")
        if aaa_yield_cache is not None:
            logging.info("Using cached AAA yield due to fetch error")
            return aaa_yield_cache
        else:
            logging.info(f"Using default AAA yield: {default_yield}")
            return default_yield

def get_stocks_connection():
    """Establish a connection to the SQLite database with increased timeout."""
    try:
        conn = sqlite3.connect(CACHE_DB, timeout=60)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT PRIMARY KEY,
            date TEXT,
            roe TEXT,
            rotc TEXT,
            eps TEXT,
            dividend TEXT,
            ticker_list_hash TEXT,
            balance_data TEXT,
            timestamp REAL,
            company_name TEXT,
            debt_to_equity REAL,
            eps_ttm REAL,
            book_value_per_share REAL,
            common_score INTEGER,
            latest_revenue REAL,
            available_data_years INTEGER
        )''')
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'available_data_years' not in columns:
            cursor.execute("ALTER TABLE stocks ADD COLUMN available_data_years INTEGER")
            logging.info("Added 'available_data_years' column to stocks table")
        cursor.execute('''CREATE TABLE IF NOT EXISTS graham_qualifiers (
            ticker TEXT PRIMARY KEY,
            common_score INTEGER,
            date TEXT,
            sector TEXT,
            exchange TEXT
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS screening_progress (
            exchange TEXT,
            ticker TEXT,
            timestamp TEXT,
            file_hash TEXT,
            status TEXT,
            PRIMARY KEY (exchange, ticker)
        )''')
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize database connection: {str(e)}")
        raise

def get_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for versioning."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        raise FileHashError(f"File not found: {file_path}")
    except Exception as e:
        raise FileHashError(f"Error computing hash for {file_path}: {str(e)}")

class TickerManager:
    def __init__(self, nyse_file: str, nasdaq_file: str):
        self.nyse_tickers = {}  # Changed to dict to store ticker: security_name
        self.nasdaq_tickers = {}  # Changed to dict to store ticker: security_name
        self.filtered_nyse = []
        self.filtered_nasdaq = []
        self.nyse_file = nyse_file
        self.nasdaq_file = nasdaq_file

    async def initialize(self):
        self.filtered_nyse = await load_and_filter_tickers(self.nyse_file, exchange_filter='N', use_cache=True)
        self.filtered_nasdaq = await load_and_filter_tickers(self.nasdaq_file, exchange_filter='Q', use_cache=True)
        self.nyse_tickers = {ticker["ticker"]: ticker["security_name"] for ticker in self.filtered_nyse}
        self.nasdaq_tickers = {ticker["ticker"]: ticker["security_name"] for ticker in self.filtered_nasdaq}
        logging.info(f"Initialized NYSE common stock tickers: {len(self.nyse_tickers)}")
        logging.info(f"Initialized NASDAQ common stock tickers: {len(self.nasdaq_tickers)}")

    def get_tickers(self, exchange: str) -> set:
        if exchange == "NYSE":
            return set(self.nyse_tickers.keys())
        elif exchange == "NASDAQ":
            return set(self.nasdaq_tickers.keys())
        return set()

    def get_security_name(self, ticker: str) -> str:
        """Retrieve the security name from the ticker files."""
        return self.nyse_tickers.get(ticker, self.nasdaq_tickers.get(ticker, 'Unknown'))

    def is_valid_ticker(self, ticker: str) -> bool:
        return ticker in self.nyse_tickers or ticker in self.nasdaq_tickers

async def load_and_filter_tickers(file_path: str, exchange_filter: Optional[str] = None, update_rate_limit=None, use_cache: bool = True) -> List[Dict]:
    try:
        file_hash = get_file_hash(file_path)
    except FileHashError as e:
        logging.error(str(e))
        return []

    logging.debug(f"Computed hash for {file_path}: {file_hash}")

    conn, cursor = get_stocks_connection()
    try:
        cursor.execute("SELECT ticker, timestamp, company_name FROM stocks WHERE ticker_list_hash = ?", (file_hash,))
        cached_data = cursor.fetchall()
        cached_entries = {row[0]: (row[1], row[2]) for row in cached_data}
        logging.debug(f"Loaded {len(cached_entries)} entries from stocks table for hash {file_hash}")

        ticker_data = []
        fresh_count = 0
        stale_count = 0
        missing_count = 0
        invalid_timestamp_count = 0

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
                security_name_idx = headers.index('Security Name')
            except ValueError as e:
                raise ValueError(f"Column not found in {file_path}: {str(e)}")

            security_type_idx = headers.index('Security Type') if 'Security Type' in headers else None
            exchange_idx = headers.index('Exchange') if 'Exchange' in headers else None

            for line in lines[1:]:
                parts = line.strip().split('|')
                if len(parts) <= symbol_idx:
                    continue
                ticker = parts[symbol_idx].strip()
                if not ticker or any(suffix in ticker for suffix in ['.', '$']):
                    continue

                # Extract Security Name
                if len(parts) > security_name_idx:
                    security_name = parts[security_name_idx].strip()
                else:
                    security_name = 'Unknown'

                if security_type_idx is not None and len(parts) > security_type_idx:
                    security_type = parts[security_type_idx].strip()
                    if security_type != "Common Stock":
                        continue

                if exchange_filter and exchange_idx is not None and len(parts) > exchange_idx:
                    exchange = parts[exchange_idx].strip()
                    if exchange != exchange_filter:
                        continue

                timestamp, company_name = cached_entries.get(ticker, (None, None))
                current_time = time.time()

                if timestamp is not None:
                    if isinstance(timestamp, (int, float)):
                        age = current_time - timestamp
                        if age < CACHE_EXPIRY:
                            logging.debug(f"{ticker} is fresh: age={age:.2f} seconds")
                            fresh_count += 1
                            is_fresh = True
                        else:
                            logging.info(f"{ticker} is stale: age={age:.2f} seconds")
                            stale_count += 1
                            is_fresh = False
                    else:
                        logging.info(f"{ticker} has invalid timestamp type: {type(timestamp)}")
                        invalid_timestamp_count += 1
                        is_fresh = False
                else:
                    logging.info(f"{ticker} not found in cache")
                    missing_count += 1
                    is_fresh = False

                ticker_data.append({
                    "ticker": ticker,
                    "is_fresh": is_fresh,
                    "company_name": company_name if is_fresh else None,
                    "security_name": security_name
                })

        logging.info(f"Loaded {len(ticker_data)} tickers from {file_path} after filtering")
        logging.info(f"Cache hit/miss summary: Hits={fresh_count}, Misses={len(ticker_data) - fresh_count} (Total={len(ticker_data)})")
        return ticker_data
    except sqlite3.Error as e:
        logging.error(f"Database error while loading tickers from {file_path}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error loading tickers from {file_path}: {str(e)}")
        return []
    finally:
        conn.close()

async def fetch_with_multiple_keys_async(ticker, endpoint, api_keys, retries=3, backoff=2, update_rate_limit=None, session=None, cancel_event=None):
    """
    Fetch data from FMP API with multiple API keys, handling retries and rate limits.
    Uses stable endpoints: /api/v3/{endpoint}/{ticker} with period=annual&limit=10 where applicable.
    """
    if not api_keys or all(not key for key in api_keys):
        logging.error(f"No valid API keys provided for endpoint {endpoint}")
        return None

    for api_key in api_keys:
        if cancel_event and cancel_event.is_set():
            logging.info(f"Cancelling fetch for {ticker} ({endpoint})")
            return None

        limiter = paid_rate_limiter if api_key == FMP_API_KEYS[0] else free_rate_limiter

        for attempt in range(retries):
            logging.debug(f"Attempt {attempt + 1}/{retries} for {ticker} ({endpoint}) with key ending in {api_key[-4:]}")
            try:
                await limiter.acquire()
                # Use stable endpoints with 10-year limit for financial statements
                if endpoint in ["income-statement", "balance-sheet-statement"]:
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=annual&limit=10&apikey={api_key}"
                else:
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                if session:
                    async with session.get(url) as response:
                        if response.status == 429:
                            if update_rate_limit:
                                update_rate_limit(f"Rate limit hit for key ending {api_key[-4:]}, pausing")
                            await asyncio.sleep(60)
                            continue
                        elif response.status != 200:
                            raise aiohttp.ClientError(f"API returned status {response.status}")
                        data = await response.json()
                        if not data:
                            raise ValueError("Empty response from API")
                        logging.info(f"Successfully fetched {endpoint} data for {ticker}")
                        return data
                else:
                    async with aiohttp.ClientSession() as temp_session:
                        async with temp_session.get(url) as response:
                            if response.status == 429:
                                if update_rate_limit:
                                    update_rate_limit(f"Rate limit hit for key ending {api_key[-4:]}, pausing")
                                await asyncio.sleep(60)
                            elif response.status != 200:
                                raise aiohttp.ClientError(f"API returned status {response.status}")
                            data = await response.json()
                            if not data:
                                raise ValueError("Empty response from API")
                            logging.info(f"Successfully fetched {endpoint} data for {ticker}")
                            return data
            except aiohttp.ClientError as e:
                logging.error(f"Network error for {ticker} ({endpoint}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(min(60 * (attempt + 1), 300))
                else:
                    break
            except json.JSONDecodeError:
                logging.error(f"JSON decoding error for {ticker} ({endpoint})")
                break
            except ValueError as e:
                logging.error(f"Data error for {ticker} ({endpoint}): {str(e)}")
                break
    logging.error(f"All attempts and keys exhausted for {ticker} ({endpoint})")
    return None

async def fetch_fmp_data(ticker: str, keys: List[str], update_rate_limit=None, cancel_event=None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """Fetch core financial data from FMP: income statement and balance sheet."""
    primary_key = keys[0]
    
    income_data = await fetch_with_multiple_keys_async(ticker, "income-statement", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    balance_data = await fetch_with_multiple_keys_async(ticker, "balance-sheet-statement", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    
    if not income_data or not balance_data:
        logging.error(f"FMP fetch failed for {ticker}: Incomplete data (Income: {bool(income_data)}, Balance: {bool(balance_data)})")
        return None, None
    
    logging.info(f"Successfully fetched FMP data for {ticker}")
    return income_data, balance_data

def fetch_historical_dividends(ticker: str, years: List[int]) -> Dict[int, float]:
    """
    Fetch historical dividends for the given ticker and years from YFinance.
    
    Args:
        ticker (str): The stock ticker symbol.
        years (List[int]): List of years to fetch dividends for.
    
    Returns:
        Dict[int, float]: A dictionary with years as keys and total DPS for that year as values.
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        div_dict = {}
        for year in years:
            year_div = dividends[dividends.index.year == year].sum()
            div_dict[year] = year_div if year_div > 0 else 0.0
        return div_dict
    except Exception as e:
        logging.error(f"Error fetching dividends for {ticker}: {str(e)}")
        return {year: 0.0 for year in years}

async def fetch_historical_data(ticker: str, exchange="Stock", update_rate_limit=None, cancel_event=None, income_data=None, balance_data=None) -> Tuple[List[float], List[float], List[float], List[float], List[int], Dict[str, float], List[Dict]]:
    """Process FMP data into historical financial metrics for Graham analysis using YFinance for dividends."""
    roe_list = []
    rotc_list = []
    eps_list = []
    div_list = []
    revenue = {}
    balance_data_list = []

    if not income_data or not balance_data:
        logging.error(f"No income or balance data provided for {ticker}")
        return [], [], [], [], [], {}, []

    # Extract years from income and balance data
    years_income = [int(entry['date'].split('-')[0]) for entry in income_data if 'date' in entry]
    years_balance = [int(entry['date'].split('-')[0]) for entry in balance_data if 'date' in entry]
    years_available = sorted(set(years_income) & set(years_balance), reverse=True)[:10]

    if not years_available:
        logging.error(f"No common years found for {ticker} between income and balance data")
        return [], [], [], [], [], {}, []

    # Fetch historical dividends from YFinance
    div_dict = await asyncio.to_thread(fetch_historical_dividends, ticker, years_available)

    for year in years_available:
        year_str = str(year)
        
        income_entry = next((entry for entry in income_data if entry['date'].startswith(year_str)), None)
        balance_entry = next((entry for entry in balance_data if entry['date'].startswith(year_str)), None)
        if not income_entry or not balance_entry:
            continue

        net_income = float(income_entry.get('netIncome', 0.0))
        equity = float(balance_entry.get('totalStockholdersEquity', 1.0))
        roe = (net_income / equity * 100) if equity != 0 else 0.0
        roe_list.append(roe)

        operating_income = float(income_entry.get('operatingIncome', 0.0))
        total_assets = float(balance_entry.get('totalAssets', 1.0))
        rotc = (operating_income / total_assets * 100) if total_assets != 0 else 0.0
        rotc_list.append(rotc)

        eps = float(income_entry.get('eps', 0.0))
        eps_list.append(eps)

        revenue[year_str] = float(income_entry.get('revenue', 0.0))

        div = div_dict.get(year, 0.0)
        div_list.append(div)

        balance_data_list.append(balance_entry)

    logging.debug(f"Fetched data for {ticker}: ROE={len(roe_list)}, ROTC={len(rotc_list)}, EPS={len(eps_list)}, Div={len(div_list)}, Years={years_available}")
    return roe_list, rotc_list, eps_list, div_list, years_available, revenue, balance_data_list

def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1

def calculate_common_criteria(ticker: str, eps_list: List[float], div_list: List[float], revenue: Dict[str, float], balance_data: List[Dict], debt_to_equity: Optional[float], available_data_years: int, latest_revenue: float) -> Optional[int]:
    """Calculate Graham score with data validation (6 common criteria)."""
    if available_data_years < 10:
        logging.warning(f"{ticker}: Insufficient data - {available_data_years} years")
        return None

    if not balance_data or 'totalCurrentAssets' not in balance_data[0] or 'totalCurrentLiabilities' not in balance_data[0]:
        logging.warning(f"Missing required balance sheet fields for {ticker}")
        return None

    score = 0

    revenue_passed = latest_revenue >= 500_000_000
    logging.debug(f"{ticker}: Revenue = ${latest_revenue / 1e6:.2f}M, Passed: {revenue_passed}")
    if revenue_passed:
        score += 1

    latest_balance = balance_data[0]
    current_assets = float(latest_balance.get('totalCurrentAssets', 0))
    current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 1))
    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
    current_passed = current_ratio > 2
    logging.debug(f"{ticker}: Current Ratio = {current_ratio:.2f}, Passed: {current_passed}")
    if current_passed:
        score += 1

    max_negative_years = min(2, available_data_years // 5)
    negative_eps_count = sum(1 for eps in eps_list if eps <= 0)
    stability_passed = negative_eps_count <= max_negative_years
    logging.debug(f"{ticker}: Negative EPS years = {negative_eps_count}, Allowed: {max_negative_years}, Passed: {stability_passed}")
    if stability_passed:
        score += 1

    dividend_passed = all(div > 0 for div in div_list)
    logging.debug(f"{ticker}: Uninterrupted Dividends: {dividend_passed}")
    if dividend_passed:
        score += 1

    if eps_list[-1] > 0 and eps_list[0] > 0:
        cagr = calculate_cagr(eps_list[-1], eps_list[0], available_data_years - 1)
        growth_passed = cagr > 0.03
        logging.debug(f"{ticker}: EPS CAGR = {cagr:.2%}, Passed: {growth_passed}")
        if growth_passed:
            score += 1
    else:
        logging.debug(f"{ticker}: EPS CAGR not calculated (invalid EPS)")

    debt_passed = debt_to_equity is not None and debt_to_equity < 2
    logging.debug(f"{ticker}: Debt-to-Equity = {debt_to_equity if debt_to_equity is not None else 'N/A'}, Passed: {debt_passed}")
    if debt_passed:
        score += 1

    logging.debug(f"{ticker}: Common Score = {score}/6 with {available_data_years} years of data")
    return score

def calculate_graham_score_8(ticker: str, price: float, pe_ratio: Optional[float], pb_ratio: Optional[float], debt_to_equity: Optional[float], eps_list: List[float], div_list: List[float], revenue: Dict[str, float], balance_data: List[Dict], available_data_years: int, latest_revenue: float) -> int:
    """Calculate full Graham score with 8 criteria."""
    score = 0
    common_score = calculate_common_criteria(ticker, eps_list, div_list, revenue, balance_data, debt_to_equity, available_data_years, latest_revenue)
    if common_score is not None:
        score += common_score
    if pe_ratio is not None and pe_ratio <= 15:
        score += 1
    if pb_ratio is not None and pb_ratio <= 1.5:
        score += 1
    logging.debug(f"{ticker}: Full Graham Score = {score}/8 with {available_data_years} years of data")
    return score

def calculate_graham_value(earnings: Optional[float], eps_list: List[float] = None) -> float:
    """Calculate Graham intrinsic value using Moody's AAA yield."""
    if not earnings or earnings <= 0:
        return float('nan')
    aaa_yield = get_aaa_yield(FRED_API_KEY)
    if aaa_yield <= 0:
        logging.error("Moody's AAA yield is zero or negative.")
        return float('nan')
    growth_rate = 0.0
    if eps_list and len(eps_list) >= 2 and eps_list[-1] > 0 and eps_list[0] > 0:
        growth_rate = calculate_cagr(eps_list[-1], eps_list[0], len(eps_list) - 1)
    g = growth_rate * 100
    normalization_factor = 4.4
    value = (earnings * (8.5 + 2 * g) * normalization_factor) / (100 * aaa_yield)
    return value

async def fetch_batch_data(tickers, screening_mode=True, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None):
    """Fetch data in batches, optimized for FMP API limits with robust error handling."""
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()

    results = []
    error_tickers = []
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TICKERS)
    nyse_tickers = ticker_manager.get_tickers("NYSE")
    nasdaq_tickers = ticker_manager.get_tickers("NASDAQ")

    # Fetch YFinance info once per ticker for price, shares outstanding, and name
    company_info = {}
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info is not None:
                company_info[ticker] = info
                price = info.get('regularMarketPrice', info.get('previousClose', None))
                if price is not None:
                    prices[ticker] = price
                else:
                    logging.error(f"No price data available for {ticker} from YFinance")
                    error_tickers.append(ticker)
            else:
                logging.error(f"No info available for {ticker} from YFinance")
                error_tickers.append(ticker)
        except Exception as e:
            logging.error(f"Error fetching info for {ticker}: {str(e)}")
            error_tickers.append(ticker)

    def save_to_db(result):
        future = db_executor.submit(_save_to_db, result)
        future.result()

    def _save_to_db(result):
        if 'timestamp' not in result or result['timestamp'] is None:
            result['timestamp'] = time.time()
        conn, cursor = get_stocks_connection()
        try:
            ticker_exchange = result['exchange']
            if ticker_exchange == "NYSE":
                ticker_list_hash = get_file_hash(NYSE_LIST_FILE)
            elif ticker_exchange == "NASDAQ":
                ticker_list_hash = get_file_hash(NASDAQ_LIST_FILE)
            else:
                ticker_list_hash = "Unknown"
            cursor.execute(
                """INSERT OR REPLACE INTO stocks 
                   (ticker, date, roe, rotc, eps, dividend, ticker_list_hash, balance_data, 
                    timestamp, company_name, debt_to_equity, eps_ttm, book_value_per_share, common_score, latest_revenue, available_data_years) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result['ticker'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 ",".join(map(str, result['roe_list'] if result['roe_list'] else [])),
                 ",".join(map(str, result['rotc_list'] if result['rotc_list'] else [])),
                 ",".join(map(str, result['eps_list'] if result['eps_list'] else [])),
                 ",".join(map(str, result['div_list'] if result['div_list'] else [])),
                 ticker_list_hash, json.dumps(result['balance_data']), result['timestamp'],
                 result['company_name'], result['debt_to_equity'], result['eps_ttm'], result['book_value_per_share'],
                 result['common_score'], result['latest_revenue'], result['available_data_years'])
            )
            conn.commit()
            logging.info(f"Saved {result['ticker']} to database with hash {ticker_list_hash}")
        except sqlite3.Error as e:
            logging.error(f"Database error saving {result['ticker']}: {str(e)}")
        finally:
            conn.close()

    async def fetch_data(ticker):
        async with semaphore:
            if cancel_event and cancel_event.is_set():
                return {"ticker": ticker, "exchange": exchange, "error": "Cancelled by user"}

            ticker_exchange = "NYSE" if ticker in nyse_tickers else "NASDAQ" if ticker in nasdaq_tickers else "Unknown"
            cached_data = get_stock_data_from_db(ticker)
            current_time = time.time()

            if cached_data and cached_data['timestamp'] and current_time - cached_data['timestamp'] < CACHE_EXPIRY:
                if screening_mode and cached_data['common_score'] is not None:
                    return {
                        "ticker": ticker,
                        "exchange": ticker_exchange,
                        "company_name": cached_data['company_name'],
                        "common_score": cached_data['common_score'],
                        "available_data_years": cached_data['available_data_years']
                    }
                elif not screening_mode:
                    price = prices.get(ticker, None)
                    if price is None:
                        logging.error(f"No price data for {ticker} from YFinance")
                        return {"ticker": ticker, "exchange": ticker_exchange, "error": "No price data from YFinance"}
                    pe_ratio = price / cached_data['eps_ttm'] if cached_data['eps_ttm'] > 0 else None
                    pb_ratio = price / cached_data['book_value_per_share'] if cached_data['book_value_per_share'] > 0 else None
                    intrinsic_value = calculate_graham_value(cached_data['eps_ttm'], cached_data['eps']) if cached_data['eps_ttm'] else float('nan')
                    buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                    sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                    graham_score = calculate_graham_score_8(ticker, price, pe_ratio, pb_ratio, cached_data['debt_to_equity'],
                                                            cached_data['eps'], cached_data['dividend'], {},
                                                            cached_data['balance_data'], cached_data['available_data_years'],
                                                            cached_data['latest_revenue'])
                    result = {
                        "ticker": ticker,
                        "exchange": ticker_exchange,
                        "company_name": cached_data['company_name'],
                        "price": price,
                        "common_score": cached_data['common_score'],
                        "roe_list": cached_data['roe'],
                        "rotc_list": cached_data['rotc'],
                        "eps_list": cached_data['eps'],
                        "div_list": cached_data['dividend'],
                        "years": cached_data.get('years', []),
                        "balance_data": cached_data['balance_data'],
                        "available_data_years": cached_data['available_data_years'],
                        "latest_revenue": cached_data['latest_revenue'],
                        "debt_to_equity": cached_data['debt_to_equity'],
                        "eps_ttm": cached_data['eps_ttm'],
                        "book_value_per_share": cached_data['book_value_per_share'],
                        "pe_ratio": pe_ratio,
                        "pb_ratio": pb_ratio,
                        "intrinsic_value": intrinsic_value,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "graham_score": graham_score
                    }
                    return result

            await asyncio.sleep(DELAY_BETWEEN_CALLS)
            price = prices.get(ticker, None) if not screening_mode else None
            security_name = ticker_manager.get_security_name(ticker)
            info = company_info.get(ticker, {})
            company_name = info.get('longName', security_name) if info else security_name
            shares_outstanding = info.get('sharesOutstanding', 1) if info else 1

            income_data, balance_data = await fetch_fmp_data(ticker, FMP_API_KEYS, update_rate_limit, cancel_event)
            if not income_data or not balance_data:
                missing = []
                if not income_data:
                    missing.append("income_data")
                if not balance_data:
                    missing.append("balance_data")
                logging.error(f"FMP data fetch failed for {ticker}: Missing {', '.join(missing)}")
                return {"ticker": ticker, "exchange": ticker_exchange, "error": f"FMP data fetch failed - Missing {', '.join(missing)}"}

            latest_income = income_data[0] if income_data else {}
            latest_balance = balance_data[0] if balance_data else {}
            net_income = float(latest_income.get('netIncome', 0))
            eps_ttm = net_income / shares_outstanding if shares_outstanding > 0 else 0
            shareholder_equity = float(latest_balance.get('totalStockholdersEquity', 1))
            book_value_per_share = shareholder_equity / shares_outstanding if shares_outstanding > 0 else 0
            long_term_debt = float(latest_balance.get('longTermDebt', 0))
            debt_to_equity = long_term_debt / shareholder_equity if shareholder_equity != 0 else None

            roe_list, rotc_list, eps_list, div_list, years_available, revenue, balance_data_list = await fetch_historical_data(
                ticker, ticker_exchange, update_rate_limit, cancel_event, income_data, balance_data
            )
            available_data_years = len(years_available)
            latest_revenue = revenue.get(str(years_available[0]) if years_available else '', 0)
            common_score = calculate_common_criteria(ticker, eps_list, div_list, revenue, balance_data_list, debt_to_equity, available_data_years, latest_revenue)

            full_result = {
                "ticker": ticker,
                "exchange": ticker_exchange,
                "company_name": company_name,
                "roe_list": roe_list,
                "rotc_list": rotc_list,
                "eps_list": eps_list,
                "div_list": div_list,
                "years": years_available,
                "balance_data": balance_data_list,
                "available_data_years": available_data_years,
                "latest_revenue": latest_revenue,
                "debt_to_equity": debt_to_equity,
                "eps_ttm": eps_ttm,
                "book_value_per_share": book_value_per_share,
                "timestamp": time.time(),
                "common_score": common_score
            }

            if not screening_mode and price is not None:
                pe_ratio = price / eps_ttm if eps_ttm > 0 else None
                pb_ratio = price / book_value_per_share if book_value_per_share > 0 else None
                intrinsic_value = calculate_graham_value(eps_ttm, eps_list) if eps_ttm else float('nan')
                buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                graham_score = calculate_graham_score_8(ticker, price, pe_ratio, pb_ratio, debt_to_equity, eps_list, div_list, revenue, balance_data_list, available_data_years, latest_revenue)
                full_result.update({
                    "price": price,
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "intrinsic_value": intrinsic_value,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "graham_score": graham_score
                })

            save_to_db(full_result)
            return full_result if not screening_mode else {
                "ticker": ticker,
                "exchange": ticker_exchange,
                "company_name": company_name,
                "common_score": common_score,
                "available_data_years": available_data_years
            }

    for ticker in tickers:
        tasks.append(fetch_data(ticker))

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=600)
    except asyncio.TimeoutError:
        logging.error("Timeout while fetching batch data")
        results = []

    # Filter out results with errors and compile error tickers
    error_tickers = [ticker for ticker, result in zip(tickers, results) if isinstance(result, dict) and 'error' in result]
    valid_results = [r for r in results if isinstance(r, dict) and 'error' not in r]

    if error_tickers:
        logging.info(f"Error tickers during batch fetch: {error_tickers}")

    return valid_results, error_tickers

async def fetch_stock_data(ticker, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None):
    results, error_tickers = await fetch_batch_data(
        [ticker], False, expected_return, margin_of_safety, exchange, ticker_manager, update_rate_limit, cancel_event
    )
    if not results or 'error' in results[0]:
        logging.error(f"Failed to fetch data for {ticker}")
        raise ValueError(f"Failed to fetch data for {ticker}")
    return results[0]

async def save_qualifying_stocks_to_favorites(qualifying_stocks, exchange):
    """Save qualifying stocks to favorites."""
    list_name = f"{exchange}_Qualifying_Stocks"
    with FAVORITES_LOCK:
        favorites = load_favorites()
        if not isinstance(favorites, dict):
            favorites = {}
        favorites[list_name] = qualifying_stocks
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=4)
    logging.info(f"Saved qualifying stocks to {list_name}")
    return list_name

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading favorites: {str(e)}")
    return {}

def get_stock_data_from_db(ticker):
    conn, cursor = get_stocks_connection()
    try:
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            stock_dict = dict(zip(columns, row))
            return {
                "ticker": stock_dict['ticker'],
                "date": stock_dict['date'],
                "roe": [float(x) if x.strip() else 0.0 for x in stock_dict['roe'].split(",")] if stock_dict['roe'] else [],
                "rotc": [float(x) if x.strip() else 0.0 for x in stock_dict['rotc'].split(",")] if stock_dict['rotc'] else [],
                "eps": [float(x) if x.strip() else 0.0 for x in stock_dict['eps'].split(",")] if stock_dict['eps'] else [],
                "dividend": [float(x) if x.strip() else 0.0 for x in stock_dict['dividend'].split(",")] if stock_dict['dividend'] else [],
                "ticker_list_hash": stock_dict['ticker_list_hash'],
                "balance_data": json.loads(stock_dict['balance_data']) if stock_dict['balance_data'] else [],
                "timestamp": stock_dict['timestamp'],
                "company_name": stock_dict['company_name'],
                "debt_to_equity": stock_dict['debt_to_equity'],
                "eps_ttm": stock_dict['eps_ttm'],
                "book_value_per_share": stock_dict['book_value_per_share'],
                "common_score": stock_dict['common_score'],
                "latest_revenue": stock_dict['latest_revenue'],
                "available_data_years": stock_dict['available_data_years']
            }
        return None
    except sqlite3.Error as e:
        logging.error(f"Database error fetching {ticker}: {str(e)}")
        return None
    finally:
        conn.close()

async def screen_nyse_graham_stocks(batch_size=25, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None):
    """Screen NYSE stocks with performance optimizations."""
    exchange = "NYSE"
    logging.info(f"Starting NYSE Graham screening")
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()
    file_path = NYSE_LIST_FILE
    current_file_hash = get_file_hash(file_path)

    conn, cursor = get_stocks_connection()
    try:
        cursor.execute("SELECT DISTINCT file_hash FROM screening_progress WHERE exchange=?", (exchange,))
        row = cursor.fetchone()
        stored_hash = row[0] if row else None
        if stored_hash != current_file_hash:
            logging.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE ticker_list_hash != ?", (current_file_hash,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()

        ticker_list = list(ticker_manager.get_tickers("NYSE"))
        filtered_ticker_data = [{"ticker": ticker} for ticker in ticker_list]
        tickers = filtered_ticker_data if tickers is None else tickers
        ticker_list = [t["ticker"] for t in tickers]

        valid_tickers = ticker_list
        logging.info(f"Using {len(valid_tickers)} tickers for NYSE screening")

        qualifying_stocks, common_scores, exchanges = [], [], []
        total_tickers = len(valid_tickers)
        processed_tickers = 0
        passed_tickers = 0
        error_tickers = []

        dynamic_batch_size = min(batch_size, max(10, MAX_CALLS_PER_MINUTE_PAID // 3))
        for i in range(0, len(valid_tickers), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                break
            batch_start = time.time()
            batch = valid_tickers[i:i + dynamic_batch_size]
            batch_results, batch_error_tickers = await fetch_batch_data(batch, True, exchange=exchange, ticker_manager=ticker_manager, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
            error_tickers.extend(batch_error_tickers)
            for result in batch_results:
                if 'error' in result:
                    logging.warning(f"Skipping {result['ticker']} due to error: {result['error']}")
                    continue
                ticker = result['ticker']
                common_score = result.get('common_score')
                available_data_years = result.get('available_data_years', 0)
                if available_data_years >= 10 and common_score is not None and common_score >= 5:
                    logging.debug(f"{ticker}: Qualified with score {common_score}/6")
                    qualifying_stocks.append(ticker)
                    common_scores.append(common_score)
                    exchanges.append(result['exchange'])
                    passed_tickers += 1
                    cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                   (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown', result['exchange']))
                else:
                    reason = "Insufficient data years" if available_data_years < 10 else "Low score" if common_score is not None else "No score calculated"
                    logging.debug(f"{ticker}: Disqualified - {reason}")
                processed_tickers += 1
                progress = (processed_tickers / total_tickers) * 100
                if root and update_progress_animated:
                    batch_time = time.time() - batch_start
                    remaining_batches = (len(valid_tickers) - i) / dynamic_batch_size
                    eta = remaining_batches * batch_time
                    root.after(0, lambda p=progress, t=valid_tickers, pt=passed_tickers, e=eta: update_progress_animated(p, t, pt, e))
                cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                               (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed"))
                conn.commit()

        if error_tickers:
            logging.info(f"Error tickers during NYSE screening: {error_tickers}")
        return qualifying_stocks, common_scores, exchanges, error_tickers
    finally:
        conn.close()

async def screen_nasdaq_graham_stocks(batch_size=25, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None):
    """Screen NASDAQ stocks with performance optimizations."""
    exchange = "NASDAQ"
    logging.info(f"Starting NASDAQ Graham screening")
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()
    file_path = NASDAQ_LIST_FILE
    current_file_hash = get_file_hash(file_path)

    conn, cursor = get_stocks_connection()
    try:
        cursor.execute("SELECT DISTINCT file_hash FROM screening_progress WHERE exchange=?", (exchange,))
        row = cursor.fetchone()
        stored_hash = row[0] if row else None
        if stored_hash != current_file_hash:
            logging.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE ticker_list_hash != ?", (current_file_hash,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()

        ticker_list = list(ticker_manager.get_tickers("NASDAQ"))
        filtered_ticker_data = [{"ticker": ticker} for ticker in ticker_list]
        tickers = filtered_ticker_data if tickers is None else tickers
        ticker_list = [t["ticker"] for t in tickers]

        valid_tickers = ticker_list
        logging.info(f"Using {len(valid_tickers)} tickers for NASDAQ screening")

        qualifying_stocks, common_scores, exchanges = [], [], []
        total_tickers = len(valid_tickers)
        processed_tickers = 0
        passed_tickers = 0
        error_tickers = []

        dynamic_batch_size = min(batch_size, max(10, MAX_CALLS_PER_MINUTE_PAID // 3))
        for i in range(0, len(valid_tickers), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                break
            batch_start = time.time()
            batch = valid_tickers[i:i + dynamic_batch_size]
            batch_results, batch_error_tickers = await fetch_batch_data(batch, True, exchange=exchange, ticker_manager=ticker_manager, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
            error_tickers.extend(batch_error_tickers)
            for result in batch_results:
                if 'error' in result:
                    logging.warning(f"Skipping {result['ticker']} due to error: {result['error']}")
                    continue
                ticker = result['ticker']
                common_score = result.get('common_score')
                available_data_years = result.get('available_data_years', 0)
                if available_data_years >= 10 and common_score is not None and common_score >= 5:
                    logging.debug(f"{ticker}: Qualified with score {common_score}/6")
                    qualifying_stocks.append(ticker)
                    common_scores.append(common_score)
                    exchanges.append(result['exchange'])
                    passed_tickers += 1
                    cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                   (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown', result['exchange']))
                else:
                    reason = "Insufficient data years" if available_data_years < 10 else "Low score" if common_score is not None else "No score calculated"
                    logging.debug(f"{ticker}: Disqualified - {reason}")
                processed_tickers += 1
                progress = (processed_tickers / total_tickers) * 100
                if root and update_progress_animated:
                    batch_time = time.time() - batch_start
                    remaining_batches = (len(valid_tickers) - i) / dynamic_batch_size
                    eta = remaining_batches * batch_time
                    root.after(0, lambda p=progress, t=valid_tickers, pt=passed_tickers, e=eta: update_progress_animated(p, t, pt, e))
                cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                               (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed"))
                conn.commit()

        if error_tickers:
            logging.info(f"Error tickers during NASDAQ screening: {error_tickers}")
        return qualifying_stocks, common_scores, exchanges, error_tickers
    finally:
        conn.close()

def clear_in_memory_caches():
    logging.info("No in-memory caches to clear (relying solely on SQLite database)")

if __name__ == "__main__":
    test_tickers = [{"ticker": "IBM"}, {"ticker": "JPM"}, {"ticker": "KO"}]
    asyncio.run(screen_nyse_graham_stocks(tickers=test_tickers))