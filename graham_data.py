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
import random
from ftplib import FTP
from config import (
    FMP_API_KEYS, FRED_API_KEY, paid_rate_limiter, free_rate_limiter, CACHE_DB, 
    NYSE_LIST_FILE, NASDAQ_LIST_FILE, USER_DATA_DIR, FAVORITES_LOCK, 
    FileHashError, FAVORITES_FILE, CACHE_EXPIRY, MAX_CALLS_PER_MINUTE_PAID, 
    screening_logger, analyze_logger
)

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

# Cache variables for AAA yield
aaa_yield_cache = None
cache_timestamp = None
CACHE_DURATION = timedelta(days=1)

# Hardcoded sector growth rates (no longer used for intrinsic value but kept for reference)
SECTOR_GROWTH_RATES = {
    "Energy": 4.0,
    "Materials": 4.0,
    "Industrials": 3.5,
    "Consumer Discretionary": 3.0,
    "Consumer Staples": 3.0,
    "Health Care": 5.5,
    "Financials": 4.5,
    "Information Technology": 6.0,
    "Communication Services": 5.0,
    "Utilities": 3.0,
    "Real Estate": 3.0,
    "Unknown": 4.0  # Default growth rate
}

def get_aaa_yield(api_key, default_yield=0.045):
    """Fetch Moody's Seasoned AAA Corporate Bond Yield from FRED with caching."""
    global aaa_yield_cache, cache_timestamp
    current_time = datetime.now()

    if aaa_yield_cache is not None and cache_timestamp is not None:
        if current_time - cache_timestamp < CACHE_DURATION:
            return aaa_yield_cache

    if not api_key:
        analyze_logger.error("FRED_API_KEY not set. Using default yield: 4.5%")
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
            analyze_logger.info(f"Fetched AAA yield: {yield_value:.4f}")
            return yield_value
        else:
            raise ValueError("No observations found in FRED response")
    except requests.RequestException as e:
        analyze_logger.error(f"Error fetching AAA yield from FRED: {str(e)}")
        if aaa_yield_cache is not None:
            analyze_logger.info("Using cached AAA yield due to fetch error")
            return aaa_yield_cache
        else:
            analyze_logger.info(f"Using default AAA yield: {default_yield}")
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
            cash_flow_data TEXT,
            key_metrics_data TEXT,
            timestamp REAL,
            company_name TEXT,
            debt_to_equity REAL,
            eps_ttm REAL,
            book_value_per_share REAL,
            common_score INTEGER,
            latest_revenue REAL,
            available_data_years INTEGER,
            sector TEXT,
            years TEXT,
            latest_total_assets REAL,
            latest_total_liabilities REAL,
            latest_shares_outstanding REAL,
            latest_long_term_debt REAL,
            latest_short_term_debt REAL,
            latest_current_assets REAL,
            latest_current_liabilities REAL,
            latest_book_value REAL,
            historic_pe_ratios TEXT,
            latest_net_income REAL,
            eps_cagr REAL
        )''')
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [col[1] for col in cursor.fetchall()]
        new_columns = [
            'latest_total_assets REAL',
            'latest_total_liabilities REAL',
            'latest_shares_outstanding REAL',
            'latest_long_term_debt REAL',
            'latest_short_term_debt REAL',
            'latest_current_assets REAL',
            'latest_current_liabilities REAL',
            'latest_book_value REAL',
            'historic_pe_ratios TEXT',
            'latest_net_income REAL',
            'eps_cagr REAL'
        ]
        for col in new_columns:
            col_name = col.split()[0]
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE stocks ADD COLUMN {col}")
                analyze_logger.info(f"Added '{col_name}' column to stocks table")
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
    except sqlite3.Error as e:
        analyze_logger.error(f"Failed to initialize database connection: {str(e)}")
        raise
    return conn, cursor

def get_sector_growth_rate(sector: str) -> float:
    """Fetch the growth rate for a given sector from the hardcoded dictionary."""
    growth_rate = SECTOR_GROWTH_RATES.get(sector, SECTOR_GROWTH_RATES["Unknown"])
    analyze_logger.debug(f"Retrieved growth rate for sector '{sector}': {growth_rate}%")
    return growth_rate

def get_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for versioning."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        raise FileHashError(f"File not found: {file_path}")
    except Exception as e:
        raise FileHashError(f"Error computing hash for {file_path}: {str(e)}")

def map_fmp_sector_to_app(sector: str) -> str:
    """Map FMP sector names to app's sector names."""
    mapping = {
        "Technology": "Information Technology",
        "Healthcare": "Health Care",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Defensive": "Consumer Staples",
        "Basic Materials": "Materials",
        "Communication Services": "Communication Services",
        "Financial Services": "Financials",
        "Utilities": "Utilities",
        "Industrials": "Industrials",
        "Energy": "Energy",
        "Real Estate": "Real Estate"
    }
    return mapping.get(sector, 'Unknown')

class TickerManager:
    def __init__(self, nyse_file: str, nasdaq_file: str, user_data_dir: str = USER_DATA_DIR):
        self.nyse_tickers = {}
        self.nasdaq_tickers = {}
        self.filtered_nyse = []
        self.filtered_nasdaq = []
        self.nyse_file = nyse_file
        self.nasdaq_file = nasdaq_file
        self.user_data_dir = user_data_dir

    async def initialize(self, force_update=False, callback=None):
        if force_update:
            await self.download_ticker_files()
        use_cache = not force_update
        self.filtered_nyse = await load_and_filter_tickers(self.nyse_file, exchange_filter='N', use_cache=use_cache)
        self.filtered_nasdaq = await load_and_filter_tickers(self.nasdaq_file, exchange_filter='Q', use_cache=use_cache)
        self.nyse_tickers = {ticker["ticker"]: ticker["security_name"] for ticker in self.filtered_nyse}
        self.nasdaq_tickers = {ticker["ticker"]: ticker["security_name"] for ticker in self.filtered_nasdaq}
        analyze_logger.info(f"Initialized NYSE common stock tickers: {len(self.nyse_tickers)}")
        analyze_logger.info(f"Initialized NASDAQ common stock tickers: {len(self.nasdaq_tickers)}")
        if callback:
            callback()

    async def download_ticker_files(self):
        """Download the latest ticker files from FTP and update the hash in the database."""
        try:
            ftp = FTP('ftp.nasdaqtrader.com')
            ftp.login()
            ftp.cwd('SymbolDirectory')
            files_to_download = ['nasdaqlisted.txt', 'otherlisted.txt']
            for file_name in files_to_download:
                local_path = os.path.join(self.user_data_dir, file_name)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {file_name}', f.write)
                # Compute and store the hash
                file_hash = get_file_hash(local_path)
                conn, cursor = get_stocks_connection()
                try:
                    exchange = 'NASDAQ' if 'nasdaq' in file_name.lower() else 'NYSE'
                    # Insert or update the hash with a dummy ticker to track file freshness
                    cursor.execute("""
                        INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status)
                        VALUES (?, ?, ?, ?, ?)
                    """, (exchange, 'FILE_HASH_TRACKER', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file_hash, 'completed'))
                    conn.commit()
                except sqlite3.Error as e:
                    analyze_logger.error(f"Error updating hash for {file_name}: {str(e)}")
                finally:
                    conn.close()
            ftp.quit()
            analyze_logger.info("Successfully downloaded and updated ticker files from FTP.")
        except Exception as e:
            analyze_logger.error(f"Failed to download ticker files: {str(e)}")
            raise

    def get_tickers(self, exchange: str) -> set:
        if exchange == "NYSE":
            return set(self.nyse_tickers.keys())
        elif exchange == "NASDAQ":
            return set(self.nasdaq_tickers.keys())
        return set()

    def get_security_name(self, ticker: str) -> str:
        return self.nyse_tickers.get(ticker, self.nasdaq_tickers.get(ticker, 'Unknown'))

    def is_valid_ticker(self, ticker: str) -> bool:
        return ticker in self.nyse_tickers or ticker in self.nasdaq_tickers

async def load_and_filter_tickers(file_path: str, exchange_filter: Optional[str] = None, update_rate_limit=None, use_cache: bool = True) -> List[Dict]:
    try:
        file_hash = get_file_hash(file_path)
    except FileHashError as e:
        analyze_logger.error(str(e))
        return []

    analyze_logger.debug(f"Computed hash for {file_path}: {file_hash}")

    conn, cursor = get_stocks_connection()
    try:
        if use_cache:
            cursor.execute("SELECT ticker, timestamp, company_name FROM stocks WHERE ticker_list_hash = ?", (file_hash,))
            cached_data = cursor.fetchall()
            cached_entries = {row[0]: (row[1], row[2]) for row in cached_data}
            analyze_logger.debug(f"Loaded {len(cached_entries)} entries from stocks table for hash {file_hash}")
        else:
            cached_entries = {}  # No cache, treat all as missing

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

            analyze_logger.debug(f"Using ticker column '{ticker_column}' for {file_path}")

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

                if use_cache:
                    timestamp, company_name = cached_entries.get(ticker, (None, None))
                else:
                    timestamp, company_name = None, None

                current_time = time.time()

                if timestamp is not None:
                    if isinstance(timestamp, (int, float)):
                        age = current_time - timestamp
                        if age < CACHE_EXPIRY:
                            analyze_logger.debug(f"{ticker} is fresh: age={age:.2f} seconds")
                            fresh_count += 1
                            is_fresh = True
                        else:
                            analyze_logger.info(f"{ticker} is stale: age={age:.2f} seconds")
                            stale_count += 1
                            is_fresh = False
                    else:
                        analyze_logger.info(f"{ticker} has invalid timestamp type: {type(timestamp)}")
                        invalid_timestamp_count += 1
                        is_fresh = False
                else:
                    analyze_logger.info(f"{ticker} not found in cache")
                    missing_count += 1
                    is_fresh = False

                ticker_data.append({
                    "ticker": ticker,
                    "is_fresh": is_fresh,
                    "company_name": company_name if is_fresh else None,
                    "security_name": security_name
                })

        analyze_logger.info(f"Loaded {len(ticker_data)} tickers from {file_path} after filtering")
        analyze_logger.info(f"Cache hit/miss summary: Hits={fresh_count}, Misses={len(ticker_data) - fresh_count} (Total={len(ticker_data)})")
        return ticker_data
    except sqlite3.Error as e:
        analyze_logger.error(f"Database error while loading tickers from {file_path}: {str(e)}")
        return []
    except Exception as e:
        analyze_logger.error(f"Error loading tickers from {file_path}: {str(e)}")
        return []
    finally:
        conn.close()

async def fetch_with_multiple_keys_async(ticker, endpoint, api_keys, retries=3, backoff=2, update_rate_limit=None, session=None, cancel_event=None):
    """
    Fetch data from FMP API with multiple API keys, handling retries and rate limits.
    Uses stable endpoints with appropriate parameters.
    """
    if not api_keys or all(not key for key in api_keys):
        analyze_logger.error(f"No valid API keys provided for endpoint {endpoint}")
        return None

    for api_key in api_keys:
        if cancel_event and cancel_event.is_set():
            analyze_logger.info(f"Cancelling fetch for {ticker} ({endpoint})")
            return None

        limiter = paid_rate_limiter if api_key == FMP_API_KEYS[0] else free_rate_limiter

        for attempt in range(retries):
            analyze_logger.debug(f"Attempt {attempt + 1}/{retries} for {ticker} ({endpoint}) with key ending in {api_key[-4:]}")
            try:
                await limiter.acquire()
                if endpoint in ["income-statement", "balance-sheet-statement", "cash-flow-statement", "key-metrics"]:
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=annual&limit=10&apikey={api_key}"
                elif endpoint == "historical-price-full/stock_dividend":
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                elif endpoint == "profile":
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                else:
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                if session:
                    async with session.get(url) as response:
                        if response.status == 429:
                            if update_rate_limit:
                                update_rate_limit(f"Rate limit hit for key ending {api_key[-4:]}")
                            await asyncio.sleep(60)
                            continue
                        elif response.status != 200:
                            raise aiohttp.ClientError(f"API returned status {response.status}")
                        data = await response.json()
                        if not data:
                            raise ValueError("Empty response from API")
                        analyze_logger.info(f"Successfully fetched {endpoint} data for {ticker}")
                        return data
                else:
                    async with aiohttp.ClientSession() as temp_session:
                        async with temp_session.get(url) as response:
                            if response.status == 429:
                                if update_rate_limit:
                                    update_rate_limit(f"Rate limit hit for key ending {api_key[-4:]}")
                                await asyncio.sleep(60)
                                continue
                            elif response.status != 200:
                                raise aiohttp.ClientError(f"API returned status {response.status}")
                            data = await response.json()
                            if not data:
                                raise ValueError("Empty response from API")
                            analyze_logger.info(f"Successfully fetched {endpoint} data for {ticker}")
                            return data
            except aiohttp.ClientError as e:
                analyze_logger.error(f"Network error for {ticker} ({endpoint}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(min(60 * (attempt + 1), 300))
                else:
                    break
            except json.JSONDecodeError:
                analyze_logger.error(f"JSON decoding error for {ticker} ({endpoint})")
                break
            except ValueError as e:
                analyze_logger.error(f"Data error for {ticker} ({endpoint}): {str(e)}")
                break
    analyze_logger.error(f"All attempts and keys exhausted for {ticker} ({endpoint})")
    return None

async def fetch_fmp_data(ticker: str, keys: List[str], update_rate_limit=None, cancel_event=None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[Dict], Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]]]:
    """Fetch core financial data from FMP: income statement, balance sheet, dividends, profile, cash flow statement, and key metrics."""
    primary_key = keys[0]
    
    income_data = await fetch_with_multiple_keys_async(ticker, "income-statement", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    balance_data = await fetch_with_multiple_keys_async(ticker, "balance-sheet-statement", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    dividend_data = await fetch_with_multiple_keys_async(ticker, "historical-price-full/stock_dividend", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    profile_data = await fetch_with_multiple_keys_async(ticker, "profile", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    cash_flow_data = await fetch_with_multiple_keys_async(ticker, "cash-flow-statement", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    key_metrics_data = await fetch_with_multiple_keys_async(ticker, "key-metrics", [primary_key], retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
    
    if not all([income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data]):
        analyze_logger.error(f"FMP fetch failed for {ticker}: Incomplete data (Income: {bool(income_data)}, Balance: {bool(balance_data)}, Dividends: {bool(dividend_data)}, Profile: {bool(profile_data)}, Cash Flow: {bool(cash_flow_data)}, Key Metrics: {bool(key_metrics_data)})")
        return None, None, None, None, None, None
    
    analyze_logger.info(f"Successfully fetched FMP data for {ticker}")
    return income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data

async def fetch_historical_data(ticker: str, exchange="Stock", update_rate_limit=None, cancel_event=None, income_data=None, balance_data=None, dividend_data=None) -> Tuple[List[float], List[float], List[float], List[float], List[int], Dict[str, float], List[Dict]]:
    """Process FMP data into historical financial metrics for Graham analysis using FMxP dividends."""
    roe_list = []
    rotc_list = []
    eps_list = []
    div_list = []
    revenue = {}
    balance_data_list = []

    if not income_data or not balance_data or not dividend_data:
        analyze_logger.error(f"No income, balance, or dividend data provided for {ticker}")
        return [], [], [], [], [], {}, []

    years_income = [int(entry['date'].split('-')[0]) for entry in income_data if 'date' in entry]
    years_balance = [int(entry['date'].split('-')[0]) for entry in balance_data if 'date' in entry]
    years_available = sorted(set(years_income) & set(years_balance))[:10]  # Ascending order (earliest to latest)

    if not years_available:
        analyze_logger.error(f"No common years found for {ticker} between income and balance data")
        return [], [], [], [], [], {}, []

    # Fetch dividend data correctly from 'historical' key
    dividend_history = dividend_data.get('historical', [])
    div_dict = {year: 0.0 for year in years_available}
    for div_entry in dividend_history:
        if 'date' in div_entry and 'adjDividend' in div_entry:
            try:
                div_year = int(div_entry['date'].split('-')[0])
                adj_dividend = float(div_entry['adjDividend'])
                if div_year in years_available:
                    div_dict[div_year] += adj_dividend
                    analyze_logger.debug(f"{ticker}: Added dividend {adj_dividend} for year {div_year}")
            except (ValueError, TypeError) as e:
                analyze_logger.warning(f"{ticker}: Invalid dividend entry {div_entry}: {str(e)}")

    analyze_logger.debug(f"{ticker}: Dividend dictionary after processing: {div_dict}")
    if not any(div_dict.values()):
        analyze_logger.warning(f"No dividend data found for {ticker} in the available years")

    # Process data for each year in ascending order
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

    analyze_logger.debug(f"Fetched data for {ticker}: ROE={len(roe_list)}, ROTC={len(rotc_list)}, EPS={len(eps_list)}, Div={len(div_list)}, Years={years_available}")
    analyze_logger.debug(f"{ticker}: EPS List: {eps_list}")
    analyze_logger.debug(f"{ticker}: Dividend List: {div_list}")
    return roe_list, rotc_list, eps_list, div_list, years_available, revenue, balance_data_list

def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1

def calculate_common_criteria(ticker: str, eps_list: List[float], div_list: List[float], revenue: Dict[str, float], balance_data: List[Dict], debt_to_equity: Optional[float], available_data_years: int, latest_revenue: float) -> Optional[int]:
    """Calculate Graham score with data validation (6 common criteria)."""
    if available_data_years < 10:
        analyze_logger.warning(f"{ticker}: Insufficient data - {available_data_years} years")
        return None

    if not balance_data or 'totalCurrentAssets' not in balance_data[0] or 'totalCurrentLiabilities' not in balance_data[0]:
        analyze_logger.warning(f"Missing required balance sheet fields for {ticker}")
        return None

    score = 0

    revenue_passed = latest_revenue >= 500_000_000
    analyze_logger.debug(f"{ticker}: Criterion 1 - Revenue >= $500M: {'Yes' if revenue_passed else 'No'} (${latest_revenue / 1e6:.2f}M)")
    if revenue_passed:
        score += 1

    latest_balance = balance_data[-1]  # Latest year (ascending order)
    current_assets = float(latest_balance.get('totalCurrentAssets', 0))
    current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 1))
    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
    current_passed = current_ratio > 2
    analyze_logger.debug(f"{ticker}: Criterion 2 - Current Ratio > 2: {'Yes' if current_passed else 'No'} ({current_ratio:.2f})")
    if current_passed:
        score += 1

    negative_eps_count = sum(1 for eps in eps_list if eps <= 0)
    stability_passed = negative_eps_count == 0
    analyze_logger.debug(f"{ticker}: Criterion 3 - All Positive EPS: {'Yes' if stability_passed else 'No'} (Negative EPS years: {negative_eps_count}, EPS List: {eps_list})")
    if stability_passed:
        score += 1

    dividend_passed = all(div > 0 for div in div_list)
    analyze_logger.debug(f"{ticker}: Criterion 4 - Uninterrupted Dividends: {'Yes' if dividend_passed else 'No'} (Div List: {div_list})")
    if dividend_passed:
        score += 1

    if len(eps_list) >= 2 and eps_list[0] > 0 and eps_list[-1] > 0:
        cagr = calculate_cagr(eps_list[0], eps_list[-1], available_data_years - 1)  # eps_list[0] is earliest, eps_list[-1] is latest
        growth_passed = cagr > 0.03
        analyze_logger.debug(f"{ticker}: Criterion 5 - EPS CAGR > 3%: {'Yes' if growth_passed else 'No'} ({cagr:.2%}, EPS List: {eps_list})")
        if growth_passed:
            score += 1
    else:
        analyze_logger.debug(f"{ticker}: Criterion 5 - EPS CAGR > 3%: No (invalid EPS), EPS List: {eps_list}")

    debt_passed = debt_to_equity is not None and debt_to_equity < 2
    analyze_logger.debug(f"{ticker}: Criterion 6 - Debt-to-Equity < 2: {'Yes' if debt_passed else 'No'} ({debt_to_equity if debt_to_equity is not None else 'N/A'})")
    if debt_passed:
        score += 1

    analyze_logger.debug(f"{ticker}: Common Score = {score}/6 with {available_data_years} years of data")
    return score

def calculate_graham_score_8(ticker: str, price: float, pe_ratio: Optional[float], pb_ratio: Optional[float], debt_to_equity: Optional[float], eps_list: List[float], div_list: List[float], revenue: Dict[str, float], balance_data: List[Dict], available_data_years: int, latest_revenue: float) -> int:
    """Calculate full Graham score with 8 criteria."""
    score = 0
    common_score = calculate_common_criteria(ticker, eps_list, div_list, revenue, balance_data, debt_to_equity, available_data_years, latest_revenue)
    if common_score is not None:
        score += common_score
        analyze_logger.debug(f"{ticker}: Common Score (part of Graham 8) = {common_score}/6")
    else:
        analyze_logger.warning(f"{ticker}: Common score is None, setting to 0")
    if pe_ratio is not None and pe_ratio <= 15:
        score += 1
        analyze_logger.debug(f"{ticker}: Criterion 7 - P/E Ratio <= 15: Passed (P/E = {pe_ratio})")
    else:
        analyze_logger.debug(f"{ticker}: Criterion 7 - P/E Ratio <= 15: Failed (P/E = {pe_ratio})")
    if pb_ratio is not None and pb_ratio <= 1.5:
        score += 1
        analyze_logger.debug(f"{ticker}: Criterion 8 - P/B Ratio <= 1.5: Passed (P/B = {pb_ratio})")
    else:
        analyze_logger.debug(f"{ticker}: Criterion 8 - P/B Ratio <= 1.5: Failed (P/B = {pb_ratio})")
    analyze_logger.debug(f"{ticker}: Full Graham Score = {score}/8 with {available_data_years} years of data")
    return score

async def calculate_graham_value(earnings: Optional[float], stock_data: dict) -> float:
    """
    Calculate Graham intrinsic value using Moody's AAA yield and stock-specific EPS CAGR.
    Caps the growth rate at zero to avoid negative intrinsic values when CAGR is negative.
    """
    if not earnings or earnings <= 0:
        analyze_logger.warning(f"Invalid earnings for {stock_data['ticker']}: {earnings}")
        return float('nan')
    aaa_yield = get_aaa_yield(FRED_API_KEY)
    if aaa_yield <= 0:
        analyze_logger.error("Moody's AAA yield is zero or negative.")
        return float('nan')
    # Get EPS CAGR from stock data, default to 0.0 if not present
    eps_cagr = stock_data.get('eps_cagr', 0.0)
    # Convert to percentage and cap at 0 to handle negative CAGR
    g = max(eps_cagr * 100, 0)
    # Calculate earnings multiplier, capped between 8.5 (no growth) and 20 (max growth)
    earnings_multiplier = min(8.5 + 2 * g, 20)
    normalization_factor = 4.4
    value = (earnings * earnings_multiplier * normalization_factor) / (100 * aaa_yield)
    analyze_logger.debug(f"Calculated Graham value for {stock_data['ticker']}: EPS={earnings}, eps_cagr={eps_cagr}, Earnings Multiplier={earnings_multiplier}, AAA Yield={aaa_yield}, Value={value}")
    return value

async def fetch_batch_data(tickers, screening_mode=True, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None):
    """Fetch data in batches: FMP during screening, YFinance price + cache during analysis."""
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()

    results = []
    error_tickers = []
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TICKERS)
    nyse_tickers = ticker_manager.get_tickers("NYSE")
    nasdaq_tickers = ticker_manager.get_tickers("NASDAQ")

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
                    cash_flow_data, key_metrics_data, timestamp, company_name, debt_to_equity, 
                    eps_ttm, book_value_per_share, common_score, latest_revenue, 
                    available_data_years, sector, years,
                    latest_total_assets, latest_total_liabilities, latest_shares_outstanding,
                    latest_long_term_debt, latest_short_term_debt, latest_current_assets,
                    latest_current_liabilities, latest_book_value, historic_pe_ratios,
                    latest_net_income, eps_cagr) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result['ticker'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 ",".join(map(str, result['roe_list'] if 'roe_list' in result else [])),
                 ",".join(map(str, result['rotc_list'] if 'rotc_list' in result else [])),
                 ",".join(map(str, result['eps_list'] if 'eps_list' in result else [])),
                 ",".join(map(str, result['div_list'] if 'div_list' in result else [])),
                 ticker_list_hash, 
                 json.dumps(result['balance_data'] if 'balance_data' in result else []),
                 json.dumps(result['cash_flow_data'] if 'cash_flow_data' in result else []),
                 json.dumps(result['key_metrics_data'] if 'key_metrics_data' in result else []),
                 result['timestamp'], result['company_name'], result['debt_to_equity'],
                 result['eps_ttm'], result['book_value_per_share'], result['common_score'],
                 result['latest_revenue'], result['available_data_years'], result['sector'],
                 ",".join(map(str, result['years'] if 'years' in result else [])),
                 result['latest_total_assets'], result['latest_total_liabilities'],
                 result['latest_shares_outstanding'], result['latest_long_term_debt'],
                 result['latest_short_term_debt'], result['latest_current_assets'],
                 result['latest_current_liabilities'], result['latest_book_value'],
                 result['historic_pe_ratios'], result['latest_net_income'], result['eps_cagr'])
            )
            conn.commit()
            analyze_logger.info(f"Saved {result['ticker']} to database with hash {ticker_list_hash}")
        except sqlite3.Error as e:
            analyze_logger.error(f"Database error saving {result['ticker']}: {str(e)}")
        finally:
            conn.close()

    async def fetch_data(ticker):
        try:
            async with semaphore:
                if cancel_event and cancel_event.is_set():
                    return {"ticker": ticker, "exchange": exchange, "error": "Cancelled by user"}

                ticker_exchange = "NYSE" if ticker in nyse_tickers else "NASDAQ" if ticker in nasdaq_tickers else "Unknown"
                cached_data = get_stock_data_from_db(ticker)
                current_time = time.time()

                # Modified cache check to include 'years'
                if cached_data and cached_data['timestamp'] and (current_time - cached_data['timestamp'] < CACHE_EXPIRY) and cached_data.get('years'):
                    analyze_logger.debug(f"Cache hit for {ticker}: Years={cached_data['years']}")
                    if screening_mode:
                        return {
                            "ticker": ticker,
                            "exchange": ticker_exchange,
                            "company_name": cached_data['company_name'],
                            "common_score": cached_data['common_score'],
                            "available_data_years": cached_data['available_data_years'],
                            "sector": cached_data['sector']
                        }
                    else:
                        # Fetch price from Yahoo Finance with retry logic
                        max_retries = 3
                        delay = 1  # Starting delay in seconds
                        for attempt in range(max_retries):
                            try:
                                stock = yf.Ticker(ticker)
                                history = stock.history(period="1d")
                                if not history.empty:
                                    price = history['Close'].iloc[-1]
                                else:
                                    raise ValueError(f"No historical price data for {ticker} from YFinance")
                                break
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 429 and attempt < max_retries - 1:
                                    analyze_logger.warning(f"Rate limit hit for {ticker}, retrying in {delay} seconds...")
                                    await asyncio.sleep(delay)
                                    delay *= 2
                                else:
                                    analyze_logger.error(f"Error fetching price for {ticker} from YFinance: {str(e)}")
                                    return {"ticker": ticker, "exchange": ticker_exchange, "error": f"YFinance fetch failed: {str(e)}"}
                            except Exception as e:
                                analyze_logger.error(f"Error fetching price for {ticker} from YFinance: {str(e)}")
                                return {"ticker": ticker, "exchange": ticker_exchange, "error": f"YFinance fetch failed: {str(e)}"}
                        else:
                            # All retries failed
                            analyze_logger.error(f"Failed to fetch price for {ticker} after {max_retries} attempts due to rate limiting")
                            return {"ticker": ticker, "exchange": ticker_exchange, "error": "Rate limit exceeded, please try again later"}

                        pe_ratio = price / cached_data['eps_ttm'] if cached_data['eps_ttm'] and cached_data['eps_ttm'] > 0 else None
                        pb_ratio = price / cached_data['book_value_per_share'] if cached_data['book_value_per_share'] and cached_data['book_value_per_share'] > 0 else None
                        intrinsic_value = await calculate_graham_value(cached_data['eps_ttm'], cached_data) if cached_data['eps_ttm'] and cached_data['eps_ttm'] > 0 else float('nan')
                        buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                        sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                        # Reconstruct revenue dictionary using years and latest_revenue
                        revenue = {}
                        if cached_data['years'] and cached_data['latest_revenue']:
                            revenue[str(cached_data['years'][-1])] = cached_data['latest_revenue']
                        # Ensure historical data lists are aligned with years (ascending order)
                        graham_score = calculate_graham_score_8(
                            ticker, price, pe_ratio, pb_ratio, cached_data['debt_to_equity'],
                            cached_data['eps_list'], cached_data['div_list'], revenue,
                            cached_data['balance_data'], cached_data['available_data_years'],
                            cached_data['latest_revenue']
                        )
                        result = {
                            "ticker": ticker,
                            "exchange": ticker_exchange,
                            "company_name": cached_data['company_name'],
                            "price": price,
                            "common_score": cached_data['common_score'],
                            "graham_score": graham_score,
                            "roe_list": cached_data['roe_list'],
                            "rotc_list": cached_data['rotc_list'],
                            "eps_list": cached_data['eps_list'],
                            "div_list": cached_data['div_list'],
                            "years": cached_data.get('years', []),
                            "balance_data": cached_data['balance_data'],
                            "cash_flow_data": cached_data['cash_flow_data'],
                            "key_metrics_data": cached_data['key_metrics_data'],
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
                            "sector": cached_data['sector'],
                            "latest_total_assets": cached_data['latest_total_assets'],
                            "latest_total_liabilities": cached_data['latest_total_liabilities'],
                            "latest_shares_outstanding": cached_data['latest_shares_outstanding'],
                            "latest_long_term_debt": cached_data['latest_long_term_debt'],
                            "latest_short_term_debt": cached_data['latest_short_term_debt'],
                            "latest_current_assets": cached_data['latest_current_assets'],
                            "latest_current_liabilities": cached_data['latest_current_liabilities'],
                            "latest_book_value": cached_data['latest_book_value'],
                            "historic_pe_ratios": cached_data['historic_pe_ratios'],
                            "latest_net_income": cached_data['latest_net_income'],
                            "eps_cagr": cached_data.get('eps_cagr', 0.0)
                        }
                        return result
                else:
                    if cached_data and not cached_data.get('years'):
                        analyze_logger.warning(f"Cache for {ticker} missing 'years', refetching data")

                # If cache miss or invalid, fetch from FMP (only during screening)
                if not screening_mode:
                    analyze_logger.error(f"No valid cached data for {ticker} during analysis. Screening required first.")
                    return {"ticker": ticker, "exchange": ticker_exchange, "error": "No cached data available for analysis"}

                await asyncio.sleep(DELAY_BETWEEN_CALLS)
                security_name = ticker_manager.get_security_name(ticker)
                
                income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data = await fetch_fmp_data(ticker, FMP_API_KEYS, update_rate_limit, cancel_event)
                if not all([income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data]):
                    missing = []
                    if not income_data:
                        missing.append("income_data")
                    if not balance_data:
                        missing.append("balance_data")
                    if not dividend_data:
                        missing.append("dividend_data")
                    if not profile_data:
                        missing.append("profile_data")
                    if not cash_flow_data:
                        missing.append("cash_flow_data")
                    if not key_metrics_data:
                        missing.append("key_metrics_data")
                    analyze_logger.error(f"FMP data fetch failed for {ticker}: Missing {', '.join(missing)}")
                    return {"ticker": ticker, "exchange": ticker_exchange, "error": f"FMP data fetch failed - Missing {', '.join(missing)}"}

                company_name = profile_data[0].get('companyName', security_name) if profile_data else security_name
                sector = map_fmp_sector_to_app(profile_data[0].get('sector', 'Unknown')) if profile_data else 'Unknown'

                latest_balance = balance_data[0] if balance_data else None
                shareholder_equity = float(latest_balance['totalStockholdersEquity']) if latest_balance and 'totalStockholdersEquity' in latest_balance else None
                long_term_debt = float(latest_balance['longTermDebt']) if latest_balance and 'longTermDebt' in latest_balance else None
                debt_to_equity = long_term_debt / shareholder_equity if shareholder_equity and shareholder_equity != 0 and long_term_debt is not None else None
                shares_outstanding = float(income_data[0]['weightedAverageShsOut']) if income_data and 'weightedAverageShsOut' in income_data[0] else None
                book_value_per_share = shareholder_equity / shares_outstanding if shares_outstanding and shares_outstanding > 0 and shareholder_equity is not None else None

                roe_list, rotc_list, eps_list, div_list, years_available, revenue, balance_data_list = await fetch_historical_data(
                    ticker, ticker_exchange, update_rate_limit, cancel_event, income_data, balance_data, dividend_data
                )
                latest_net_income = float(income_data[0]['netIncome']) if income_data and 'netIncome' in income_data[0] else None
                available_data_years = len(years_available)
                latest_revenue = revenue.get(str(years_available[-1]) if years_available else '', 0)  # Latest year
                eps_ttm = eps_list[-1] if eps_list else None  # Latest annual EPS
                common_score = calculate_common_criteria(ticker, eps_list, div_list, revenue, balance_data_list, debt_to_equity, available_data_years, latest_revenue)

                # Calculate eps_cagr
                if len(eps_list) >= 2 and eps_list[0] > 0 and eps_list[-1] > 0:
                    eps_cagr = calculate_cagr(eps_list[0], eps_list[-1], len(eps_list) - 1)
                else:
                    eps_cagr = 0.0

                # Extract new data points
                latest_total_assets = float(latest_balance['totalAssets']) if latest_balance and 'totalAssets' in latest_balance else None
                latest_total_liabilities = float(latest_balance['totalLiabilities']) if latest_balance and 'totalLiabilities' in latest_balance else None
                latest_shares_outstanding = shares_outstanding
                latest_long_term_debt = long_term_debt
                latest_short_term_debt = float(latest_balance['shortTermDebt']) if latest_balance and 'shortTermDebt' in latest_balance else None
                latest_current_assets = float(latest_balance['totalCurrentAssets']) if latest_balance and 'totalCurrentAssets' in latest_balance else None
                latest_current_liabilities = float(latest_balance['totalCurrentLiabilities']) if latest_balance and 'totalCurrentLiabilities' in latest_balance else None
                latest_book_value = shareholder_equity
                historic_pe_ratios = json.dumps([entry.get('peRatio', 0) for entry in key_metrics_data]) if key_metrics_data else '[]'

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
                    "cash_flow_data": cash_flow_data,
                    "key_metrics_data": key_metrics_data,
                    "available_data_years": available_data_years,
                    "latest_revenue": latest_revenue,
                    "debt_to_equity": debt_to_equity,
                    "eps_ttm": eps_ttm,
                    "book_value_per_share": book_value_per_share,
                    "timestamp": time.time(),
                    "common_score": common_score,
                    "sector": sector,
                    "latest_total_assets": latest_total_assets,
                    "latest_total_liabilities": latest_total_liabilities,
                    "latest_shares_outstanding": latest_shares_outstanding,
                    "latest_long_term_debt": latest_long_term_debt,
                    "latest_short_term_debt": latest_short_term_debt,
                    "latest_current_assets": latest_current_assets,
                    "latest_current_liabilities": latest_current_liabilities,
                    "latest_book_value": latest_book_value,
                    "historic_pe_ratios": historic_pe_ratios,
                    "latest_net_income": latest_net_income,
                    "eps_cagr": eps_cagr
                }

                save_to_db(full_result)
                return {
                    "ticker": ticker,
                    "exchange": ticker_exchange,
                    "company_name": company_name,
                    "common_score": common_score,
                    "available_data_years": available_data_years,
                    "sector": sector
                }
        except Exception as e:
            analyze_logger.error(f"Error processing {ticker}: {str(e)}")
            return {"ticker": ticker, "exchange": exchange, "error": f"Processing failed: {str(e)}"}

    for ticker in tickers:
        tasks.append(fetch_data(ticker))

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=600)
    except asyncio.TimeoutError:
        analyze_logger.error("Timeout while fetching batch data")
        results = []

    # Process results
    valid_results = []
    error_tickers = []
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            # Log any exception that occurred during fetch_data
            analyze_logger.error(f"Exception processing {ticker}: {str(result)}")
            error_tickers.append(ticker)
        elif isinstance(result, dict):
            if 'error' in result:
                # Explicit error dictionary from fetch_data
                error_tickers.append(ticker)
                analyze_logger.debug(f"Error for {ticker}: {result['error']}")
            else:
                # Valid result
                valid_results.append(result)
        else:
            # Unexpected result type
            analyze_logger.error(f"Unexpected result type for {ticker}: {type(result)}")
            error_tickers.append(ticker)

    analyze_logger.debug(f"Batch fetch complete: {len(valid_results)} valid, {len(error_tickers)} errors")
    analyze_logger.debug(f"Error tickers during batch fetch: {error_tickers}")
    return valid_results, error_tickers

async def fetch_stock_data(ticker, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None):
    results, error_tickers = await fetch_batch_data(
        [ticker], False, expected_return, margin_of_safety, exchange, ticker_manager, update_rate_limit, cancel_event
    )
    if not results or 'error' in results[0]:
        analyze_logger.error(f"Failed to fetch data for {ticker}")
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
    analyze_logger.info(f"Saved qualifying stocks to {list_name}")
    return list_name

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            analyze_logger.error(f"Error loading favorites: {str(e)}")
    return {}

def get_stock_data_from_db(ticker):
    conn, cursor = get_stocks_connection()
    try:
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            stock_dict = dict(zip(columns, row))
            years = [int(y) for y in stock_dict['years'].split(",")] if stock_dict.get('years') else []
            if not years:
                analyze_logger.warning(f"No years data found for {ticker} in database")
            roe_list = [float(x) if x.strip() else 0.0 for x in stock_dict['roe'].split(",")] if stock_dict['roe'] else []
            rotc_list = [float(x) if x.strip() else 0.0 for x in stock_dict['rotc'].split(",")] if stock_dict['rotc'] else []
            eps_list = [float(x) if x.strip() else 0.0 for x in stock_dict['eps'].split(",")] if stock_dict['eps'] else []
            div_list = [float(x) if x.strip() else 0.0 for x in stock_dict['dividend'].split(",")] if stock_dict['dividend'] else []
            # Ensure lists are aligned with years (ascending order)
            if years:
                data = list(zip(years, roe_list, rotc_list, eps_list, div_list))
                data.sort(key=lambda x: x[0])  # Sort by year (ascending)
                years, roe_list, rotc_list, eps_list, div_list = zip(*data)
                years = list(years)
                roe_list = list(roe_list)
                rotc_list = list(rotc_list)
                eps_list = list(eps_list)
                div_list = list(div_list)
            result = {
                "ticker": stock_dict['ticker'],
                "date": stock_dict['date'],
                "roe_list": roe_list,
                "rotc_list": rotc_list,
                "eps_list": eps_list,
                "div_list": div_list,
                "years": years,
                "balance_data": json.loads(stock_dict['balance_data']) if stock_dict['balance_data'] else [],
                "cash_flow_data": json.loads(stock_dict['cash_flow_data']) if stock_dict.get('cash_flow_data') else [],
                "key_metrics_data": json.loads(stock_dict['key_metrics_data']) if stock_dict.get('key_metrics_data') else [],
                "timestamp": stock_dict['timestamp'],
                "company_name": stock_dict['company_name'],
                "debt_to_equity": stock_dict.get('debt_to_equity'),
                "eps_ttm": stock_dict.get('eps_ttm'),
                "book_value_per_share": stock_dict.get('book_value_per_share'),
                "common_score": stock_dict.get('common_score', 0),  # Default to 0 if missing
                "latest_revenue": stock_dict.get('latest_revenue', 0.0),  # Default to 0.0 if missing
                "available_data_years": stock_dict['available_data_years'],
                "sector": stock_dict.get('sector', 'Unknown'),
                "latest_total_assets": stock_dict.get('latest_total_assets'),
                "latest_total_liabilities": stock_dict.get('latest_total_liabilities'),
                "latest_shares_outstanding": stock_dict.get('latest_shares_outstanding'),
                "latest_long_term_debt": stock_dict.get('latest_long_term_debt'),
                "latest_short_term_debt": stock_dict.get('latest_short_term_debt'),
                "latest_current_assets": stock_dict.get('latest_current_assets'),
                "latest_current_liabilities": stock_dict.get('latest_current_liabilities'),
                "latest_book_value": stock_dict.get('latest_book_value'),
                "historic_pe_ratios": json.loads(stock_dict.get('historic_pe_ratios', '[]')),
                "latest_net_income": stock_dict.get('latest_net_income'),
                "eps_cagr": stock_dict.get('eps_cagr', 0.0)
            }
            analyze_logger.debug(f"Retrieved from DB for {ticker}: Years={years}, Dividend={div_list}, EPS={eps_list}")
            return result
        return None
    except sqlite3.Error as e:
        analyze_logger.error(f"Database error fetching {ticker}: {str(e)}")
        return None
    finally:
        conn.close()

async def screen_nyse_graham_stocks(batch_size=18, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None):
    """Screen NYSE stocks with performance optimizations and selective logging."""
    exchange = "NYSE"
    screening_logger.info(f"Starting NYSE Graham screening with {len(tickers) if tickers else 'all'} tickers")
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
            screening_logger.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE ticker_list_hash != ?", (current_file_hash,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()

        ticker_list = list(ticker_manager.get_tickers("NYSE"))
        filtered_ticker_data = [{"ticker": ticker} for ticker in ticker_list]
        tickers = filtered_ticker_data if tickers is None else tickers
        ticker_list = [t["ticker"] for t in tickers]

        invalid_file = os.path.join(USER_DATA_DIR, "NYSE Invalid Tickers.txt")
        if os.path.exists(invalid_file):
            with open(invalid_file, 'r') as f:
                invalid_tickers = set(f.read().splitlines())
        else:
            invalid_tickers = set()
        valid_tickers = [t for t in ticker_list if t not in invalid_tickers]
        screening_logger.info(f"Excluding {len(ticker_list) - len(valid_tickers)} invalid NYSE tickers")

        qualifying_stocks, common_scores, exchanges = [], [], []
        total_tickers = len(valid_tickers)
        processed_tickers = 0
        passed_tickers = 0
        error_tickers = []

        sample_interval = max(10, total_tickers // 50)

        dynamic_batch_size = min(batch_size, max(10, MAX_CALLS_PER_MINUTE_PAID // 6))  # 6 calls/ticker now
        for i in range(0, len(valid_tickers), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                screening_logger.info("Screening cancelled by user")
                break
            batch_start = time.time()
            batch = valid_tickers[i:i + dynamic_batch_size]
            batch_results, batch_error_tickers = await fetch_batch_data(batch, True, exchange=exchange, ticker_manager=ticker_manager, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
            error_tickers.extend(batch_error_tickers)
            for result in batch_results:
                ticker = result['ticker']
                log_full = (processed_tickers < 10) or (processed_tickers >= total_tickers - 10) or (processed_tickers % sample_interval == 0)
                
                if 'error' in result:
                    if log_full:
                        screening_logger.warning(f"Skipping {ticker} due to error: {result['error']}")
                    continue
                
                common_score = result.get('common_score')
                available_data_years = result.get('available_data_years', 0)
                if available_data_years >= 10 and common_score is not None and common_score == 6:
                    if log_full:
                        screening_logger.info(f"{ticker}: Qualified with all 6 criteria met")
                    qualifying_stocks.append(ticker)
                    common_scores.append(common_score)
                    exchanges.append(result['exchange'])
                    passed_tickers += 1
                    cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['sector'], result['exchange']))
                else:
                    if log_full:
                        reason = "Insufficient data years" if available_data_years < 10 else "Did not meet all 6 criteria" if common_score is not None else "No score calculated"
                        screening_logger.info(f"{ticker}: Disqualified - {reason}")
                
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
            screening_logger.info(f"Batch {i // dynamic_batch_size + 1} took {time.time() - batch_start:.2f} seconds")

        if error_tickers:
            invalid_file = os.path.join(USER_DATA_DIR, "NYSE Invalid Tickers.txt")
            if os.path.exists(invalid_file):
                with open(invalid_file, 'r') as f:
                    existing_invalid = set(f.read().splitlines())
            else:
                existing_invalid = set()
            new_invalid = set(error_tickers) - existing_invalid
            if new_invalid:
                with open(invalid_file, 'a') as f:
                    for ticker in sorted(new_invalid):
                        f.write(ticker + '\n')
                screening_logger.info(f"Added {len(new_invalid)} new invalid tickers to {invalid_file}")

        screening_logger.info(f"Completed NYSE screening: {processed_tickers}/{total_tickers} processed, {passed_tickers} passed, {len(error_tickers)} errors")
        if error_tickers and total_tickers <= 20:
            screening_logger.info(f"Error tickers: {error_tickers}")
        elif error_tickers:
            sample_size = min(5, len(error_tickers))
            error_sample = random.sample(error_tickers, sample_size)
            screening_logger.info(f"Error tickers (random sample): {error_sample} (and {len(error_tickers) - sample_size} more)")

        return qualifying_stocks, common_scores, exchanges, error_tickers
    finally:
        conn.close()

async def screen_nasdaq_graham_stocks(batch_size=18, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None):
    """Screen NASDAQ stocks with performance optimizations and selective logging."""
    exchange = "NASDAQ"
    screening_logger.info(f"Starting NASDAQ Graham screening with {len(tickers) if tickers else 'all'} tickers")
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
            screening_logger.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE ticker_list_hash != ?", (current_file_hash,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()

        ticker_list = list(ticker_manager.get_tickers("NASDAQ"))
        filtered_ticker_data = [{"ticker": ticker} for ticker in ticker_list]
        tickers = filtered_ticker_data if tickers is None else tickers
        ticker_list = [t["ticker"] for t in tickers]

        invalid_file = os.path.join(USER_DATA_DIR, "NASDAQ Invalid Tickers.txt")
        if os.path.exists(invalid_file):
            with open(invalid_file, 'r') as f:
                invalid_tickers = set(f.read().splitlines())
        else:
            invalid_tickers = set()
        valid_tickers = [t for t in ticker_list if t not in invalid_tickers]
        screening_logger.info(f"Excluding {len(ticker_list) - len(valid_tickers)} invalid NASDAQ tickers")

        qualifying_stocks, common_scores, exchanges = [], [], []
        total_tickers = len(valid_tickers)
        processed_tickers = 0
        passed_tickers = 0
        error_tickers = []

        sample_interval = max(10, total_tickers // 50)

        dynamic_batch_size = min(batch_size, max(10, MAX_CALLS_PER_MINUTE_PAID // 6))  # 6 calls/ticker now
        for i in range(0, len(valid_tickers), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                screening_logger.info("Screening cancelled by user")
                break
            batch_start = time.time()
            batch = valid_tickers[i:i + dynamic_batch_size]
            batch_results, batch_error_tickers = await fetch_batch_data(batch, True, exchange=exchange, ticker_manager=ticker_manager, update_rate_limit=update_rate_limit, cancel_event=cancel_event)
            error_tickers.extend(batch_error_tickers)
            for result in batch_results:
                ticker = result['ticker']
                log_full = (processed_tickers < 10) or (processed_tickers >= total_tickers - 10) or (processed_tickers % sample_interval == 0)
                
                if 'error' in result:
                    if log_full:
                        screening_logger.warning(f"Skipping {ticker} due to error: {result['error']}")
                    continue
                
                common_score = result.get('common_score')
                available_data_years = result.get('available_data_years', 0)
                if available_data_years >= 10 and common_score is not None and common_score == 6:
                    if log_full:
                        screening_logger.info(f"{ticker}: Qualified with all 6 criteria met")
                    qualifying_stocks.append(ticker)
                    common_scores.append(common_score)
                    exchanges.append(result['exchange'])
                    passed_tickers += 1
                    cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['sector'], result['exchange']))
                else:
                    if log_full:
                        reason = "Insufficient data years" if available_data_years < 10 else "Did not meet all 6 criteria" if common_score is not None else "No score calculated"
                        screening_logger.info(f"{ticker}: Disqualified - {reason}")
                
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
            screening_logger.info(f"Batch {i // dynamic_batch_size + 1} took {time.time() - batch_start:.2f} seconds")

        if error_tickers:
            invalid_file = os.path.join(USER_DATA_DIR, "NASDAQ Invalid Tickers.txt")
            if os.path.exists(invalid_file):
                with open(invalid_file, 'r') as f:
                    existing_invalid = set(f.read().splitlines())
            else:
                existing_invalid = set()
            new_invalid = set(error_tickers) - existing_invalid
            if new_invalid:
                with open(invalid_file, 'a') as f:
                    for ticker in sorted(new_invalid):
                        f.write(ticker + '\n')
                screening_logger.info(f"Added {len(new_invalid)} new invalid tickers to {invalid_file}")

        screening_logger.info(f"Completed NASDAQ screening: {processed_tickers}/{total_tickers} processed, {passed_tickers} passed, {len(error_tickers)} errors")
        if error_tickers and total_tickers <= 20:
            screening_logger.info(f"Error tickers: {error_tickers}")
        elif error_tickers:
            sample_size = min(5, len(error_tickers))
            error_sample = random.sample(error_tickers, sample_size)
            screening_logger.info(f"Error tickers (random sample): {error_sample} (and {len(error_tickers) - sample_size} more)")

        return qualifying_stocks, common_scores, exchanges, error_tickers
    finally:
        conn.close()

def clear_in_memory_caches():
    analyze_logger.info("No in-memory caches to clear (relying solely on SQLite database)")

if __name__ == "__main__":
    test_tickers = [{"ticker": "IBM"}, {"ticker": "JPM"}, {"ticker": "KO"}]
    asyncio.run(screen_nyse_graham_stocks(tickers=test_tickers))